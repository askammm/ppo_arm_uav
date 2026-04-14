#!/usr/bin/env python3
"""
residual_ppo_node.py
ROS node that runs the trained residual PPO policy at 125 Hz and publishes
a wrench correction [Δτ_x, Δτ_y, Δτ_z, ΔT] to the interaction_controller.

Subscribes:
  mavros/odometry/in          (nav_msgs/Odometry)        - UAV state
  command/pose                (geometry_msgs/PoseStamped) - position setpoint
  command/trajectory          (trajectory_msgs/MultiDOFJointTrajectory) - trajectory setpoint
  debug/torque3D              (geometry_msgs/Vector3Stamped) - baseline torque
  /sarax_mani/joint_states    (sensor_msgs/JointState)   - arm current state
  /m4e_mani/trajectory        (sensor_msgs/JointState)   - arm next command ← feedforward

Publishes:
  /residual_ppo/wrench_correction  (std_msgs/Float32MultiArray) - [Δτx,Δτy,Δτz,ΔT]

Observation (26D):
  e_p(3) e_v(3) e_R(3) omega(3) tau_base(3) T_base(1) prev_action(4)
  q_arm_meas(2) dq_arm_meas(2) q_arm_next_meas(2)   ← arm state + feedforward

Usage:
  rosrun residual_ppo residual_ppo_node.py \
      --model $(find residual_ppo)/models/ppo_residual_sarax_final \
      [--vecnorm $(find residual_ppo)/models/ppo_residual_sarax_vecnorm.pkl] \
      [--alpha 0.3]
"""

import argparse
import os
import sys
import threading
from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import MultiDOFJointTrajectory

# Ensure envs/ is importable (for SaraxDynamics constants)
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from envs.sarax_dynamics import (
    SaraxDynamics,
    internal_to_sim_measured,
    quat_to_rot,
    sim_measured_to_internal,
    vee,
)


def _quat_from_msg(q_msg: Any) -> NDArray[np.float64]:
    return np.array([q_msg.x, q_msg.y, q_msg.z, q_msg.w], dtype=np.float64)


def _vec3_from_msg(v: Any) -> NDArray[np.float64]:
    return np.array([v.x, v.y, v.z], dtype=np.float64)


def _yaw_from_quat(q: NDArray[np.float64]) -> float:
    siny_cosp = 2.0 * (q[3] * q[2] + q[0] * q[1])
    cosy_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    return float(np.arctan2(siny_cosp, cosy_cosp))


class ResidualPPONode:
    def __init__(
        self,
        model_path: str,
        vecnorm_path: Optional[str] = None,
        alpha: float = 0.3,
        enabled: bool = True,
    ) -> None:
        # Node is initialized in __main__ before instantiation so get_param works.
        # Guard against double-init when running via rosrun without __main__ path.
        if not rospy.core.is_initialized():
            rospy.init_node("residual_ppo_node", anonymous=False)

        # --- Load model ---
        # 优先尝试 TorchScript actor.pt（无 numpy 版本依赖）
        # 其次尝试 SB3 .zip（要求与训练时相同的 numpy 版本）
        import torch
        model_dir   = os.path.dirname(os.path.abspath(model_path))
        pt_path     = os.path.join(model_dir, "actor.pt")
        zip_path    = (model_path if model_path.endswith(".zip")
                       else model_path + ".zip")
        obs_rms_path = os.path.join(model_dir, "obs_rms.npz")

        if os.path.exists(pt_path):
            # ── TorchScript 加载路径 ──────────────────────────────────
            self.model      = torch.jit.load(pt_path, map_location="cpu")
            self.model.eval()
            self._model_type = "torchscript"
            rospy.loginfo(f"[residual_ppo] Loaded TorchScript actor: {pt_path}")

            # 观测归一化：从 obs_rms.npz 加载（convert_model.py 生成）
            self.vecnorm = None
            if os.path.exists(obs_rms_path):
                _rms = np.load(obs_rms_path)
                self._obs_mean = _rms["mean"].astype(np.float32)
                self._obs_std  = np.sqrt(
                    _rms["var"].astype(np.float32) + 1e-8)
                self._use_obs_rms = True
                rospy.loginfo(f"[residual_ppo] Loaded obs_rms: {obs_rms_path}")
            else:
                self._use_obs_rms = False

        elif os.path.exists(zip_path):
            # ── SB3 加载路径（备用，要求 numpy 版本匹配）─────────────
            try:
                from stable_baselines3 import PPO
                from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            except ImportError:
                rospy.logfatal("[residual_ppo] stable-baselines3 not installed.")
                raise
            self.model      = PPO.load(model_path, device="cpu")
            self._model_type = "sb3"
            rospy.loginfo(f"[residual_ppo] Loaded SB3 model: {zip_path}")

            self.vecnorm = None
            self._use_obs_rms = False
            if vecnorm_path and os.path.exists(vecnorm_path):
                from envs.residual_env import ResidualPPOEnv
                dummy_env = DummyVecEnv([lambda: ResidualPPOEnv()])
                self.vecnorm = VecNormalize.load(vecnorm_path, dummy_env)
                self.vecnorm.training    = False
                self.vecnorm.norm_reward = False
                rospy.loginfo(f"[residual_ppo] Loaded VecNormalize: {vecnorm_path}")
        else:
            rospy.logfatal(
                f"[residual_ppo] 找不到模型文件。\n"
                f"  TorchScript: {pt_path}\n"
                f"  SB3 zip    : {zip_path}\n"
                f"请先运行 tools/convert_model.py 生成 actor.pt，"
                f"或确认 model_path 正确。")
            raise FileNotFoundError(pt_path)

        self.enabled: bool = enabled
        self.alpha: float = alpha

        # --- Physical limits for action scaling ---
        dyn = SaraxDynamics()
        self.tau_lim: NDArray[np.float64] = np.array([
            alpha * dyn.max_roll_torque,
            alpha * dyn.max_pitch_torque,
            alpha * dyn.max_yaw_torque,
        ])
        self.T_lim: float = alpha * dyn.max_thrust
        self.dyn: SaraxDynamics = dyn

        # --- State buffers (protected by lock) ---
        self._lock = threading.Lock()
        self._pos: NDArray[np.float64] = np.zeros(3)
        self._quat: NDArray[np.float64] = np.array([0.0, 0.0, 0.0, 1.0])
        self._vel_b: NDArray[np.float64] = np.zeros(3)
        self._omega: NDArray[np.float64] = np.zeros(3)
        self._pos_cmd: NDArray[np.float64] = np.zeros(3)
        self._yaw_cmd: float = 0.0
        self._tau_base: NDArray[np.float64] = np.zeros(3)
        self._T_base: NDArray[np.float64] = np.array([0.0])
        self._prev_action: NDArray[np.float64] = np.zeros(4)

        self._odom_received: bool = False
        self._cmd_received: bool = False

        # --- Arm state buffers ---
        self._q_arm: NDArray[np.float64] = np.zeros(3)   # internal arm angles
        self._dq_arm: NDArray[np.float64] = np.zeros(3)  # internal arm velocities
        self._q_arm_next: NDArray[np.float64] = np.zeros(3)   # next-step arm command
        self._arm_received: bool = False

        # Joint name → index mapping (populated on first JointState message)
        self._arm_state_topic: str = rospy.get_param(
            "~arm_joint_states_topic", "/sarax_mani/joint_states"
        )
        self._arm_joint_names: List[str] = rospy.get_param(
            "~arm_joint_names", ["mani_joint_1", "mani_joint_2"]
        )
        self._arm_idx: Optional[List[int]] = None

        # --- ROS interface ---
        self._pub = rospy.Publisher(
            "/residual_ppo/wrench_correction",
            Float32MultiArray,
            queue_size=1,
        )

        rospy.Subscriber("mavros/odometry/in", Odometry,
                         self._odom_cb, queue_size=1)
        rospy.Subscriber("command/pose", PoseStamped,
                         self._cmd_cb, queue_size=1)
        rospy.Subscriber("command/trajectory", MultiDOFJointTrajectory,
                         self._traj_cmd_cb, queue_size=1)
        rospy.Subscriber("debug/torque3D", Vector3Stamped,
                         self._torque_cb, queue_size=1)

        # Arm state: actual joint positions/velocities (150 Hz in sim)
        rospy.Subscriber(self._arm_state_topic, JointState,
                         self._arm_state_cb, queue_size=1)

        # Arm NEXT command (feedforward): the trajectory command sent to the arm
        # This gives the PPO advance notice of arm motion before it happens.
        rospy.Subscriber("/m4e_mani/trajectory", JointState,
                         self._arm_cmd_cb, queue_size=1)

        # 125 Hz timer (matches controller_node main loop)
        rospy.Timer(rospy.Duration(1.0 / 125.0), self._inference_cb)

        rospy.loginfo(
            "[residual_ppo] Node ready. enabled=%s obs_dim=26 arm_topic=%s",
            self.enabled,
            self._arm_state_topic,
        )

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Odometry) -> None:
        with self._lock:
            self._pos = _vec3_from_msg(msg.pose.pose.position)
            self._quat = _quat_from_msg(msg.pose.pose.orientation)
            self._vel_b = _vec3_from_msg(msg.twist.twist.linear)
            self._omega = _vec3_from_msg(msg.twist.twist.angular)
            self._odom_received = True

    def _cmd_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self._pos_cmd = _vec3_from_msg(msg.pose.position)
            self._yaw_cmd = _yaw_from_quat(_quat_from_msg(msg.pose.orientation))
            self._cmd_received = True

    def _traj_cmd_cb(self, msg: MultiDOFJointTrajectory) -> None:
        """
        Receive trajectory commands and extract the first target point.
        This mirrors the controller's use of mav_msgs::eigenTrajectoryPointFromTrajMsg,
        which reads the first point of the trajectory message.
        """
        if not msg.points:
            rospy.logwarn_throttle(
                5.0, "[residual_ppo] Received empty /command/trajectory message."
            )
            return

        point = msg.points[0]
        if not point.transforms:
            rospy.logwarn_throttle(
                5.0,
                "[residual_ppo] First /command/trajectory point has no transforms.",
            )
            return

        transform = point.transforms[0]
        with self._lock:
            self._pos_cmd = _vec3_from_msg(transform.translation)
            self._yaw_cmd = _yaw_from_quat(_quat_from_msg(transform.rotation))
            self._cmd_received = True

    def _torque_cb(self, msg: Vector3Stamped) -> None:
        with self._lock:
            self._tau_base = _vec3_from_msg(msg.vector)

    def _arm_state_cb(self, msg: JointState) -> None:
        """Receive current arm joint positions and velocities."""
        with self._lock:
            if self._arm_idx is None:
                # Build index mapping once
                try:
                    self._arm_idx = [list(msg.name).index(n)
                                     for n in self._arm_joint_names]
                    rospy.loginfo(
                        "[residual_ppo] Arm joint mapping on %s: %s -> %s",
                        self._arm_state_topic,
                        self._arm_joint_names,
                        self._arm_idx,
                    )
                except ValueError:
                    return   # joint names not yet known
            idx = self._arm_idx
            if len(msg.position) > max(idx):
                q_meas = np.array([msg.position[i] for i in idx], dtype=np.float64)
                q_internal, _ = sim_measured_to_internal(q_meas)
                self._q_arm = q_internal
            if len(msg.velocity) > max(idx):
                dq_meas = np.array([msg.velocity[i] for i in idx], dtype=np.float64)
                _, dq_internal = sim_measured_to_internal(np.zeros(2), dq_meas)
                self._dq_arm = dq_internal
            self._arm_received = True

    def _arm_cmd_cb(self, msg: JointState) -> None:
        """
        Receive the NEXT arm joint command (feedforward).
        Topic /m4e_mani/trajectory sends the target JointState that the
        arm impedance controller will track next — giving PPO advance notice.
        """
        with self._lock:
            if len(msg.position) >= 3:
                self._q_arm_next = np.array(msg.position[:3], dtype=np.float64)

    # ------------------------------------------------------------------
    # Inference (125 Hz)
    # ------------------------------------------------------------------

    def _build_obs(self) -> NDArray[np.float32]:
        """Build the 26D observation vector (matches residual_env.py)."""
        with self._lock:
            pos         = self._pos.copy()
            quat        = self._quat.copy()
            vel_b       = self._vel_b.copy()
            omega       = self._omega.copy()
            pos_cmd     = self._pos_cmd.copy()
            yaw_cmd     = self._yaw_cmd
            tau_base    = self._tau_base.copy()
            prev_action = self._prev_action.copy()
            q_arm_internal = self._q_arm.copy()
            dq_arm_internal = self._dq_arm.copy()
            q_arm_next_internal = self._q_arm_next.copy()

        R_bw  = quat_to_rot(quat)
        vel_w = R_bw @ vel_b

        e_p = pos - pos_cmd
        e_v = vel_w

        # Update dyn CoM/inertia so baseline wrench uses current arm config
        self.dyn.pos   = pos
        self.dyn.quat  = quat
        self.dyn.vel_b = vel_b
        self.dyn.omega = omega
        self.dyn.set_arm_state(q_arm_internal, dq_arm_internal)

        base_wrench, _, _, R_d_w = self.dyn.compute_baseline_wrench(
            pos_cmd, yaw_cmd)
        e_R_mat = 0.5 * (R_d_w.T @ R_bw - R_bw.T @ R_d_w)
        e_R     = vee(e_R_mat)
        T_base  = np.array([base_wrench[3]])

        q_arm_meas, dq_arm_meas = internal_to_sim_measured(
            q_arm_internal, dq_arm_internal
        )
        q_arm_next_meas, _ = internal_to_sim_measured(q_arm_next_internal)

        obs = np.concatenate([
            e_p, e_v, e_R, omega,          # 12
            tau_base, T_base,              #  4
            prev_action,                   #  4
            q_arm_meas, dq_arm_meas, q_arm_next_meas,  # 6
        ]).astype(np.float32)                          # = 26
        return obs

    def _inference_cb(self, event: rospy.timer.TimerEvent) -> None:
        if not self.enabled:
            return
        if not (self._odom_received and self._cmd_received):
            return
        # arm_received 可选：若机械臂话题未连接则用零值（向后兼容）
        if not self._arm_received:
            rospy.logwarn_throttle(5.0,
                "[residual_ppo] %s not yet received, using q_arm=0.",
                self._arm_state_topic,
            )

        try:
            obs = self._build_obs()

            # ── 观测归一化 ──────────────────────────────────────────
            if self._use_obs_rms:
                # TorchScript 路径：用 obs_rms.npz 的统计量手动归一化
                obs_norm = np.clip(
                    (obs - self._obs_mean) / self._obs_std, -10.0, 10.0
                ).astype(np.float32)
            elif self.vecnorm is not None:
                # SB3 路径：VecNormalize 归一化
                obs_norm = self.vecnorm.normalize_obs(
                    obs.reshape(1, -1))[0].astype(np.float32)
            else:
                obs_norm = obs

            # ── 模型推理 ────────────────────────────────────────────
            if self._model_type == "torchscript":
                import torch
                with torch.no_grad():
                    obs_t  = torch.as_tensor(obs_norm[None], dtype=torch.float32)
                    act_t  = self.model(obs_t)
                action = act_t.squeeze(0).numpy()
            else:
                # SB3
                action, _ = self.model.predict(obs_norm, deterministic=True)

            action = np.clip(action, -1.0, 1.0)

            delta_tau = action[:3] * self.tau_lim
            delta_T   = action[3]  * self.T_lim

            with self._lock:
                self._prev_action = action.copy()

            msg = Float32MultiArray()
            msg.data = [float(delta_tau[0]), float(delta_tau[1]),
                        float(delta_tau[2]), float(delta_T)]
            self._pub.publish(msg)

        except Exception as exc:
            rospy.logerr_throttle(1.0, f"[residual_ppo] inference error: {exc}")

    def spin(self) -> None:
        rospy.spin()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Support both CLI args (rosrun) and ROS params (roslaunch)."""
    # First init node so get_param works; anonymous=False so param ns is stable
    # Note: init_node is called inside ResidualPPONode.__init__, so here we
    # just parse CLI. ROS params are read after node init in __main__.
    parser = argparse.ArgumentParser(description="Residual PPO ROS node")
    parser.add_argument("--model",   default=None,
                        help="Path to trained model zip (without .zip extension)")
    parser.add_argument("--vecnorm", default=None,
                        help="Path to VecNormalize stats .pkl (optional)")
    parser.add_argument("--alpha",   type=float, default=None,
                        help="Residual action scale (fraction of max torque/thrust)")
    parser.add_argument("--enable",  action="store_true", default=None,
                        help="Enable residual correction (default: True)")
    # Filter out ROS remapping args
    import sys as _sys
    argv = rospy.myargv(_sys.argv[1:])
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    # Allow ROS params (set via <param> in launch file) to override CLI defaults
    rospy.init_node("residual_ppo_node", anonymous=False)
    model_path  = args.model   or rospy.get_param("~model",   None)
    vecnorm     = args.vecnorm or rospy.get_param("~vecnorm", "") or None
    alpha       = args.alpha   if args.alpha is not None else float(rospy.get_param("~alpha", 0.3))
    enabled     = args.enable  if args.enable is not None else bool(rospy.get_param("~enabled", True))

    if not model_path:
        rospy.logfatal("[residual_ppo] No model path provided. "
                       "Use --model <path> or <param name='model' value='...'/>")
        raise SystemExit(1)

    node = ResidualPPONode(
        model_path=model_path,
        vecnorm_path=vecnorm,
        alpha=alpha,
        enabled=enabled,
    )
    node.spin()
