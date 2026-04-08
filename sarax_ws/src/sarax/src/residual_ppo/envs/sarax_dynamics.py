"""
sarax_dynamics.py
6-DOF quadrotor dynamics simulation matching the sarax_plus SITL configuration,
extended with a 3-DOF manipulator (M4E) whose state dynamically updates the
UAV's inertia matrix and centre-of-mass.

Physical parameters:
  UAV (sarax_plus):  mass_uav=9.535 kg, J_uav=diag([0.2,0.2,0.3]) kg.m²
  Arm  (M4E):        mass_arm=1.685 kg (4 links)
  Combined hover mass ≈ 11.22 kg  (matches sarax_plus_sitl_param.yaml)

Arm model (simplified 3-DOF planar chain, mounted below body):
  joint1 (yaw  around body-z, continuous)
  joint2 (pitch around body-y, ±π/2)
  joint3 (pitch around body-y, -1.9…2.6 rad)

  Link lengths (along joint axis direction):
    L1 = 0.075 m  (joint1→joint2 offset)
    L2 = 0.400 m  (joint2→joint3 offset)
    L3 = 0.200 m  (joint3→EE centre-of-mass)

  Link masses: [m1=0.141, m2=1.066, m3=0.363, m4=0.114] kg

Mount offset of arm base from UAV CoM (body frame):
  r_mount = [0, 0, -0.10] m  (directly below body centre)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Default UAV parameters (sarax_plus SITL)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "mass_uav": 9.535,         # kg  (UAV without arm)
    "mass_arm": 1.685,         # kg  (arm total)
    "arm_length": 0.45,        # m   (rotor arm length)
    "num_of_arms": 44,
    "inertia_uav": np.array([0.2, 0.2, 0.3]),   # kg.m²  (UAV only)
    "moment_constant": 0.0202,
    "thrust_constant": 9.68e-05,
    "max_rotor_speed": 730.0,  # rad/s
    "gravity": 9.81,
    "coaxial_thrust_scale": 0.8,
    # M4E manipulator geometry
    "arm_mount_offset": np.array([0.0, 0.0, -0.10]),   # body frame
    "link_masses":   np.array([0.141, 1.066, 0.363, 0.114]),
    "link_lengths":  np.array([0.075, 0.400, 0.200]),   # L1,L2,L3
    # Joint limits  [j1, j2, j3]
    "joint_pos_min": np.array([-np.pi,    -np.pi/2, -1.90]),
    "joint_pos_max": np.array([ np.pi,     np.pi/2,  2.60]),
    "joint_vel_max": np.array([1.0, 1.0, 1.0]),         # rad/s
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def skew(v):
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ],
    ])


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def quat_to_rot(q):
    """Quaternion [x,y,z,w] → 3×3 rotation matrix (body→world)."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ])


def rot_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s; y = 0.25 * s
        x = (R[0, 1] + R[1, 0]) / s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s; z = 0.25 * s
        x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s
    return np.array([x, y, z, w])


def normalize_quat(q):
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Arm forward kinematics (body frame, simplified planar chain)
# ---------------------------------------------------------------------------

def arm_link_positions(q_arm, p):
    """
    Compute CoM position of each arm link in body frame.
    q_arm = [j1, j2, j3]  (joint angles)
    Returns list of 4 CoM positions (body frame).
    """
    j1, j2, j3 = q_arm
    r0 = p["arm_mount_offset"].copy()     # arm base in body frame
    L1, L2, L3 = p["link_lengths"]

    # Rotation from base to link frames (simplified planar model):
    # joint1 rotates around body-z (yaw)
    # joint2,3 rotate around the resulting lateral axis (pitch chain)
    c1, s1 = np.cos(j1), np.sin(j1)
    c2, s2 = np.cos(j2), np.sin(j2)
    c23 = np.cos(j2 + j3)
    s23 = np.sin(j2 + j3)

    # Unit vector along arm segment after j1 rotation (in body XY plane)
    u_lat = np.array([c1, s1, 0.0])          # lateral direction

    # CoM of link1 (base motor, small, approx at mount point)
    p1 = r0.copy()

    # Joint2 position from base
    pos_j2 = r0 + L1 * np.array([0, 0, -1.0])  # j1 offset roughly downward
    # CoM of link2: along pitch direction
    p2 = pos_j2 + (L2 / 2) * (c2 * u_lat + np.array([0, 0, -s2]))

    # Joint3 position
    pos_j3 = pos_j2 + L2 * (c2 * u_lat + np.array([0, 0, -s2]))
    # CoM of link3
    p3 = pos_j3 + (L2 / 2) * (c23 * u_lat + np.array([0, 0, -s23]))

    # EE (link4) CoM
    pos_ee_base = pos_j3 + L2 * (c23 * u_lat + np.array([0, 0, -s23]))
    p4 = pos_ee_base + (L3 / 2) * (c23 * u_lat + np.array([0, 0, -s23]))

    return [p1, p2, p3, p4]


def compute_arm_inertia_contribution(q_arm, p):
    """
    Compute the arm's contribution to the composite system using
    the parallel-axis theorem.
    Returns:
        delta_J : (3,3)  additional inertia from arm (body frame)
        r_com_arm : (3,)  arm CoM in body frame
        mass_arm : float
    """
    masses = p["link_masses"]
    positions = arm_link_positions(q_arm, p)
    mass_arm = float(np.sum(masses))

    # Arm CoM in body frame (mass-weighted average)
    r_com_arm = sum(m * r for m, r in zip(masses, positions)) / mass_arm

    # Parallel-axis theorem: I_total = sum_i [ m_i * (|r_i|^2 I - r_i r_i^T) ]
    # (point-mass approximation for each link's contribution)
    delta_J = np.zeros((3, 3))
    for m, r in zip(masses, positions):
        delta_J += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

    return delta_J, r_com_arm, mass_arm


# ---------------------------------------------------------------------------
# Main dynamics class
# ---------------------------------------------------------------------------

class SaraxDynamics:
    """
    6-DOF UAV + 3-DOF manipulator dynamics.

    State:
      pos    (3)  world frame
      quat   (4)  [x,y,z,w]  body→world
      vel_b  (3)  body frame
      omega  (3)  body frame
      q_arm  (3)  joint angles [j1,j2,j3]
      dq_arm (3)  joint velocities
    """

    def __init__(self, params=None):
        p = dict(DEFAULT_PARAMS) if params is None else dict(params)
        # copy mutable arrays
        for k in ("inertia_uav", "arm_mount_offset", "link_masses",
                  "link_lengths", "joint_pos_min", "joint_pos_max",
                  "joint_vel_max"):
            p[k] = np.array(p[k], dtype=float)
        self.p = p

        self.g       = p["gravity"]
        self.J_uav   = np.diag(p["inertia_uav"])
        self.arm_len = p["arm_length"]
        self.ct      = p["thrust_constant"]
        self.cm      = p["moment_constant"]
        self.max_omega = p["max_rotor_speed"]

        # Rotor thrust limits
        max_rotor_thrust = self.ct * self.max_omega**2
        self.max_thrust       = 8 * max_rotor_thrust * p["coaxial_thrust_scale"]
        cos45 = np.cos(np.radians(45))
        self.max_roll_torque  = 2 * cos45 * self.arm_len * max_rotor_thrust
        self.max_pitch_torque = 2 * cos45 * self.arm_len * max_rotor_thrust
        self.max_yaw_torque   = 8 * self.cm * max_rotor_thrust

        # Controller gains (Lee + impedance)
        self.K_o = np.diag([3.0, 3.0, 1.5])
        self.G_o = 0.5 * np.trace(self.K_o) * np.eye(3) - self.K_o
        self.D_o = np.diag([0.3, 0.3, 0.15])
        self.Kp  = np.diag([4.0, 4.0, 5.0])
        self.Kv  = np.diag([3.2, 3.2, 4.0])

        self.reset()

    # ------------------------------------------------------------------
    # Internal: update composite mass/inertia from current arm state
    # ------------------------------------------------------------------

    def _update_composite(self):
        """Recompute total mass, CoM offset, and inertia from q_arm."""
        delta_J, r_com_arm, m_arm = compute_arm_inertia_contribution(
            self.q_arm, self.p)
        m_uav = self.p["mass_uav"]
        self.mass = m_uav + m_arm

        # Composite CoM offset from original UAV CoM (body frame)
        # UAV CoM at origin; arm CoM at r_com_arm
        self.r_com_offset = m_arm * r_com_arm / self.mass  # CoM shift

        # Composite inertia at new CoM (Steiner transfer for UAV body too)
        r_uav = -self.r_com_offset    # UAV CoM relative to composite CoM
        J_uav_shifted = (self.J_uav
                         + m_uav * (np.dot(r_uav, r_uav) * np.eye(3)
                                    - np.outer(r_uav, r_uav)))
        self.J     = J_uav_shifted + delta_J
        self.J_inv = np.linalg.inv(self.J)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self, pos=None, yaw=0.0, q_arm=None):
        self.pos   = np.zeros(3) if pos is None else np.array(pos, float)
        R0 = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw),  np.cos(yaw), 0],
                       [0,            0,           1]])
        self.quat  = normalize_quat(rot_to_quat(R0))
        self.vel_b = np.zeros(3)
        self.omega = np.zeros(3)
        # Default arm: folded (all joints ≈ 0)
        self.q_arm  = np.zeros(3) if q_arm is None else np.array(q_arm, float)
        self.dq_arm = np.zeros(3)
        self.t = 0.0
        self._update_composite()

    def set_arm_state(self, q_arm, dq_arm=None):
        self.q_arm  = np.clip(q_arm,  self.p["joint_pos_min"],
                                       self.p["joint_pos_max"])
        self.dq_arm = np.zeros(3) if dq_arm is None else np.array(dq_arm, float)
        self._update_composite()

    def get_state(self):
        return {
            "pos":    self.pos.copy(),
            "quat":   self.quat.copy(),
            "vel_b":  self.vel_b.copy(),
            "omega":  self.omega.copy(),
            "q_arm":  self.q_arm.copy(),
            "dq_arm": self.dq_arm.copy(),
            "mass":   self.mass,
            "t":      self.t,
        }

    # ------------------------------------------------------------------
    # Baseline controller (mirrors interaction_controller)
    # ------------------------------------------------------------------

    def compute_baseline_wrench(self, pos_cmd, yaw_cmd=0.0,
                                 vel_cmd=None, acc_cmd=None):
        """
        Lee trajectory tracker + impedance attitude controller.
        Uses current composite mass and inertia (arm-dependent).
        """
        if vel_cmd is None: vel_cmd = np.zeros(3)
        if acc_cmd is None: acc_cmd = np.zeros(3)

        R_bw   = quat_to_rot(self.quat)
        vel_w  = R_bw @ self.vel_b

        e_p = self.pos - pos_cmd
        e_v = vel_w - vel_cmd

        # Desired total force direction
        I_a_d = (-self.Kp @ e_p - self.Kv @ e_v
                 + self.mass * self.g * np.array([0, 0, 1]) + acc_cmd)

        thrust = I_a_d.dot(R_bw[:, 2])

        Bz_d  = I_a_d / (np.linalg.norm(I_a_d) + 1e-9)
        Bx_d  = np.array([np.cos(yaw_cmd), np.sin(yaw_cmd), 0.0])
        By_d  = np.cross(Bz_d, Bx_d)
        n_By  = np.linalg.norm(By_d)
        By_d  = By_d / n_By if n_By > 1e-6 else np.array([0, 1, 0])
        R_d_w = np.column_stack([np.cross(By_d, Bz_d), By_d, Bz_d])

        # Impedance attitude
        R_bd        = R_d_w.T @ R_bw
        spring_tilde = -2.0 * (self.G_o @ R_bd - (self.G_o @ R_bd).T)
        tau         = vee(spring_tilde) - self.D_o @ self.omega

        return np.array([tau[0], tau[1], tau[2], thrust]), e_p, e_v, R_d_w

    # ------------------------------------------------------------------
    # Dynamics step
    # ------------------------------------------------------------------

    def step(self, wrench, dt, wind_force=None, q_arm_cmd=None):
        """
        Integrate one step.
        wrench      : [tau_x, tau_y, tau_z, T]  UAV body control
        wind_force  : optional (3,) world-frame disturbance [N]
        q_arm_cmd   : optional (3,) next arm joint position target (first-order)
        """
        # ── arm dynamics: first-order tracking of command ──────────
        if q_arm_cmd is not None:
            q_cmd = np.clip(q_arm_cmd,
                            self.p["joint_pos_min"], self.p["joint_pos_max"])
            arm_tau = 5.0  # time constant [s]^-1
            self.dq_arm = (q_cmd - self.q_arm) * arm_tau
            self.dq_arm = np.clip(self.dq_arm,
                                  -self.p["joint_vel_max"],
                                   self.p["joint_vel_max"])
            self.q_arm  = np.clip(self.q_arm + self.dq_arm * dt,
                                  self.p["joint_pos_min"],
                                  self.p["joint_pos_max"])
            self._update_composite()   # recompute mass/J after arm move

        tau    = wrench[:3]
        thrust = wrench[3]
        R_bw   = quat_to_rot(self.quat)

        # Translational (world frame)
        thrust_w = R_bw[:, 2] * thrust
        grav_w   = np.array([0, 0, -self.mass * self.g])
        ext_f    = np.array(wind_force) if wind_force is not None else np.zeros(3)
        acc_w    = (thrust_w + grav_w + ext_f) / self.mass

        vel_w        = R_bw @ self.vel_b
        self.pos    += vel_w * dt + 0.5 * acc_w * dt**2
        vel_w_new    = vel_w + acc_w * dt
        self.vel_b   = R_bw.T @ vel_w_new

        # Rotational (body frame) — use composite J
        alpha      = self.J_inv @ (tau - np.cross(self.omega, self.J @ self.omega))
        self.omega += alpha * dt

        # Quaternion kinematics
        wx, wy, wz = self.omega
        Omega = np.array([
            [ 0,   wz, -wy,  wx],
            [-wz,  0,   wx,  wy],
            [ wy, -wx,  0,   wz],
            [-wx, -wy, -wz,  0 ],
        ])
        self.quat += 0.5 * Omega @ self.quat * dt
        self.quat  = normalize_quat(self.quat)
        self.t    += dt

    # ------------------------------------------------------------------
    # Rollout helper
    # ------------------------------------------------------------------

    def rollout(self, pos_cmd, yaw_cmd=0.0, duration=5.0, dt=0.008,
                residual_fn=None, wind_fn=None, arm_traj_fn=None):
        """
        arm_traj_fn(t) -> q_arm_cmd (3,)  — optional arm trajectory
        residual_fn(state, base_wrench) -> delta_wrench (4,)
        """
        history = []
        self.reset(pos=np.zeros(3))
        for _ in range(int(duration / dt)):
            state = self.get_state()
            base_wrench, e_p, _, _ = self.compute_baseline_wrench(pos_cmd, yaw_cmd)
            delta  = residual_fn(state, base_wrench) if residual_fn else np.zeros(4)
            total  = base_wrench + delta
            wind   = wind_fn(self.t) if wind_fn else None
            q_cmd  = arm_traj_fn(self.t) if arm_traj_fn else None
            self.step(total, dt, wind_force=wind, q_arm_cmd=q_cmd)
            history.append((self.t, state, base_wrench.copy(),
                            total.copy(), e_p.copy()))
        return history
