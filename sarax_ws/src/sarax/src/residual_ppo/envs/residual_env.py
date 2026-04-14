"""
residual_env.py
Gymnasium environment for residual PPO training on sarax_plus + M4E arm.

Observation (26D):
  [0:3]   position error  e_p          (world frame)
  [3:6]   velocity error  e_v          (world frame)
  [6:9]   attitude error  e_R          (SO3 vee map)
  [9:12]  angular velocity omega        (body frame)
  [12:15] baseline torque  tau_base
  [15]    baseline thrust   T_base
  [16:20] previous residual action     (4D, Markov補償)
  [20:22] current sim joint angles     q_arm_meas
  [22:24] current sim joint velocities dq_arm_meas
  [24:26] NEXT sim joint command       q_arm_next_meas  ← feedforward

Action (4D, continuous, clipped to [-1,1]):
  [Δτ_x, Δτ_y, Δτ_z, ΔT]  scaled by alpha × max torque/thrust

Reward:
  r = -λ_p*||e_p||² - λ_v*||e_v||² - λ_R*||e_R||²
      - λ_w*||omega||² - λ_u*||action||²
      + r_survive
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.sarax_dynamics import SaraxDynamics, internal_to_sim_measured, vee


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_ENV_CONFIG = {
    # Reward weights
    "lambda_p": 2.0,
    "lambda_v": 0.5,
    "lambda_R": 0.3,
    "lambda_w": 0.1,
    "lambda_u": 0.05,
    "r_survive": 0.5,
    # Episode params
    "dt": 0.008,           # 125 Hz
    "max_steps": 1000,     # 8 s
    "pos_reset_std": 0.5,  # m
    # Residual action scale
    "alpha": 0.3,
    # Wind curriculum
    "wind_force_max": 0.0,
    # Safety
    "max_pos_error": 5.0,  # m
    # Arm trajectory curriculum:
    #   "static"  — arm held at random fixed pose each episode
    #   "slow"    — arm moves slowly (ω < 0.3 rad/s)
    #   "dynamic" — arm moves at full speed
    "arm_curriculum": "static",
}


class ResidualPPOEnv(gym.Env):
    """
    Gymnasium environment wrapping SaraxDynamics (UAV + M4E arm).
    Observation includes the simulated 2-DOF arm feedback interface and the
    next-step arm command in the same interface so the policy can anticipate
    arm-induced disturbances.
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        cfg = {**DEFAULT_ENV_CONFIG, **(config or {})}
        self.cfg = cfg

        self.dyn   = SaraxDynamics()
        alpha      = cfg["alpha"]

        # Action limits
        self.tau_lim = np.array([
            alpha * self.dyn.max_roll_torque,
            alpha * self.dyn.max_pitch_torque,
            alpha * self.dyn.max_yaw_torque,
        ])
        self.T_lim = alpha * self.dyn.max_thrust

        # Joint limits shorthand
        self._q_min = self.dyn.p["joint_pos_min"]
        self._q_max = self.dyn.p["joint_pos_max"]
        self._dq_max = self.dyn.p["joint_vel_max"]

        # Observation: 26D
        obs_high = np.full(26, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Action: 4D in [-1,1]
        self.action_space = spaces.Box(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count  = 0
        self._pos_cmd     = np.zeros(3)
        self._yaw_cmd     = 0.0
        self._prev_action = np.zeros(4)
        self._wind_max    = cfg["wind_force_max"]

        # Arm trajectory state
        self._q_arm_target   = np.zeros(3)  # current episode target
        self._q_arm_next_cmd = np.zeros(3)  # command that will be applied NEXT step

    # ------------------------------------------------------------------
    # Curriculum setters
    # ------------------------------------------------------------------

    def set_wind_curriculum(self, wind_force_max):
        self._wind_max = float(wind_force_max)

    def set_arm_curriculum(self, mode: str):
        assert mode in ("static", "slow", "dynamic")
        self.cfg["arm_curriculum"] = mode

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        # Goal (random hover point)
        self._pos_cmd    = rng.uniform(-1.5, 1.5, size=3).astype(np.float64)
        self._pos_cmd[2] = rng.uniform(0.5, 2.5)
        self._yaw_cmd    = rng.uniform(-np.pi, np.pi)

        # Random arm configuration for this episode.
        # In simulation the first joint is fixed and only the last two DOFs
        # are observable through /sarax_mani/joint_states.
        q_arm_init = rng.uniform(self._q_min, self._q_max).astype(np.float64)
        q_arm_init[0] = 0.0

        # For static curriculum: arm target = initial pose (no movement)
        # For dynamic: target changes over episode
        self._q_arm_target   = rng.uniform(self._q_min, self._q_max)
        self._q_arm_target[0] = 0.0
        self._q_arm_next_cmd = q_arm_init.copy()

        # UAV initial state near goal
        init_pos    = self._pos_cmd + rng.normal(0.0, self.cfg["pos_reset_std"], 3)
        init_pos[2] = max(0.1, init_pos[2])

        self.dyn.reset(pos=init_pos, q_arm=q_arm_init)

        self._step_count  = 0
        self._prev_action = np.zeros(4)

        obs = self._get_obs()
        return obs.astype(np.float32), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Scale to physical units
        delta_tau    = action[:3] * self.tau_lim
        delta_T      = action[3]  * self.T_lim
        delta_wrench = np.array([delta_tau[0], delta_tau[1],
                                  delta_tau[2], delta_T])

        # Determine next arm command BEFORE stepping
        # (this is q_arm_next saved in obs of this step)
        q_arm_next_actual = self._compute_arm_command()

        # Baseline UAV wrench
        base_wrench, e_p, e_v, R_d_w = self.dyn.compute_baseline_wrench(
            self._pos_cmd, self._yaw_cmd)
        total_wrench = base_wrench + delta_wrench

        # Step: apply arm command as feedforward
        wind = self._sample_wind()
        self.dyn.step(total_wrench, self.cfg["dt"],
                      wind_force=wind, q_arm_cmd=q_arm_next_actual)

        # Update next command for next obs
        self._q_arm_next_cmd = self._compute_arm_command()

        self._step_count += 1
        obs      = self._get_obs()
        e_p_new  = obs[0:3]
        e_v_new  = obs[3:6]
        e_R_new  = obs[6:9]
        omega    = obs[9:12]

        reward = (
            - self.cfg["lambda_p"] * float(np.dot(e_p_new, e_p_new))
            - self.cfg["lambda_v"] * float(np.dot(e_v_new, e_v_new))
            - self.cfg["lambda_R"] * float(np.dot(e_R_new, e_R_new))
            - self.cfg["lambda_w"] * float(np.dot(omega, omega))
            - self.cfg["lambda_u"] * float(np.dot(action, action))
            + self.cfg["r_survive"]
        )

        self._prev_action = action.copy()

        truncated  = self._step_count >= self.cfg["max_steps"]
        terminated = np.linalg.norm(e_p_new) > self.cfg["max_pos_error"]

        return obs.astype(np.float32), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_arm_command(self):
        """
        Generate next-step arm joint command according to curriculum.
        Returns q_arm_cmd (3,).
        """
        mode = self.cfg["arm_curriculum"]
        if mode == "static":
            # Hold current position
            q_cmd = self.dyn.q_arm.copy()
            q_cmd[0] = 0.0
            return q_cmd
        elif mode == "slow":
            # Move toward target at ≤ 0.3 rad/s
            dt   = self.cfg["dt"]
            diff = self._q_arm_target - self.dyn.q_arm
            diff[0] = 0.0
            max_step = 0.3 * dt
            step = np.clip(diff, -max_step, max_step)
            q_cmd = np.clip(self.dyn.q_arm + step, self._q_min, self._q_max)
            q_cmd[0] = 0.0
            return q_cmd
        else:  # dynamic
            # Move at full speed toward target; randomize target when reached
            dt   = self.cfg["dt"]
            diff = self._q_arm_target - self.dyn.q_arm
            if np.linalg.norm(diff) < 0.05:
                self._q_arm_target = self.np_random.uniform(
                    self._q_min, self._q_max)
                self._q_arm_target[0] = 0.0
                diff = self._q_arm_target - self.dyn.q_arm
            diff[0] = 0.0
            step = np.clip(diff, -self._dq_max * dt, self._dq_max * dt)
            q_cmd = np.clip(self.dyn.q_arm + step, self._q_min, self._q_max)
            q_cmd[0] = 0.0
            return q_cmd

    def _get_obs(self):
        from envs.sarax_dynamics import quat_to_rot
        state  = self.dyn.get_state()
        R_bw   = quat_to_rot(state["quat"])
        vel_w  = R_bw @ state["vel_b"]

        e_p    = state["pos"] - self._pos_cmd
        e_v    = vel_w

        base_wrench, _, _, R_d_w = self.dyn.compute_baseline_wrench(
            self._pos_cmd, self._yaw_cmd)
        e_R_mat = 0.5 * (R_d_w.T @ R_bw - R_bw.T @ R_d_w)
        e_R     = vee(e_R_mat)

        omega    = state["omega"]
        tau_base = base_wrench[:3]
        T_base   = np.array([base_wrench[3]])

        q_arm_meas, dq_arm_meas = internal_to_sim_measured(
            state["q_arm"], state["dq_arm"]
        )
        q_next_meas, _ = internal_to_sim_measured(self._q_arm_next_cmd)

        obs = np.concatenate([
            e_p, e_v, e_R, omega,          # 12
            tau_base, T_base,              #  4
            self._prev_action,             #  4
            q_arm_meas, dq_arm_meas, q_next_meas,  # 6
        ])                                         # = 26 total
        return obs.astype(np.float32)

    def _sample_wind(self):
        if self._wind_max <= 0:
            return None
        return self.np_random.uniform(-self._wind_max, self._wind_max, size=3)
