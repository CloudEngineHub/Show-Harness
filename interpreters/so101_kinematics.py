"""SO-101 forward kinematics and TCP inverse kinematics in numpy.

The chain is the Workshop USD's own UsdPhysics joints (extracted by
scripts/trajectory/real2sim/so101_workshop/dump_kinematics.py into configs/so101_chain.json),
so sim and the real arm use one model. numpy only: no simulator is needed to import this.
Everything is expressed in the robot BASE frame (the `base` link), which is what Show-Harness
calls the robot/world frame for MV_* tokens.

Joint vector order and units are the Workshop's: rad, ordered
Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw.

IK solves TCP position (3) + tool pitch in the arm plane (1) + optionally the yaw of the
jaw's closing axis (1, through Wrist_Roll). The arm is 5-DoF: the hand can only tilt about
the arm-plane normal, so this is everything that can be held. With the tool pointing
straight down it is the FULL orientation (pitch fixes the axis, yaw fixes the rest).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
ALL_JOINTS = ARM_JOINTS + ("Jaw",)
IK_JOINTS = (0, 1, 2, 3)  # indices into ALL_JOINTS that the IK moves

DEFAULT_CHAIN = Path(__file__).resolve().parents[1] / "configs" / "so101_chain.json"
# lerobot's so101_new_calib.urdf `gripper_frame_joint`: fingertip centre in the gripper link.
DEFAULT_TCP_OFFSET = (-0.0079, -0.000218, -0.0981)


def quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q if q[0] >= 0 else -q


def _tf(pos, quat_wxyz) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = quat_wxyz_to_mat(quat_wxyz)
    t[:3, 3] = pos
    return t


def _rot_axis(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    t = np.eye(4)
    if axis == "X":
        t[1:3, 1:3] = [[c, -s], [s, c]]
    elif axis == "Y":
        t[0, 0], t[0, 2], t[2, 0], t[2, 2] = c, s, -s, c
    else:
        t[0:2, 0:2] = [[c, -s], [s, c]]
    return t


@dataclass
class _Joint:
    name: str
    parent: str
    child: str
    pre: np.ndarray   # parent body -> joint frame
    post: np.ndarray  # joint frame -> child body (inverse of child-side local frame)
    axis: str


class So101Kinematics:
    def __init__(self, chain_path: str | Path = DEFAULT_CHAIN,
                 tcp_offset=DEFAULT_TCP_OFFSET) -> None:
        chain = json.loads(Path(chain_path).read_text())
        by_name = {j["name"]: j for j in chain["joints"]}
        self.joints: list[_Joint] = []
        for name in ALL_JOINTS:
            j = by_name[name]
            self.joints.append(_Joint(
                name=name,
                parent=j["body0"].rsplit("/", 1)[-1],
                child=j["body1"].rsplit("/", 1)[-1],
                pre=_tf(j["localPos0"], j["localRot0_wxyz"]),
                post=np.linalg.inv(_tf(j["localPos1"], j["localRot1_wxyz"])),
                axis=j["axis"],
            ))
        lim = np.asarray(chain["soft_joint_pos_limits"], dtype=np.float64)
        self.lower, self.upper = lim[:, 0], lim[:, 1]
        self.tcp_offset = np.asarray(tcp_offset, dtype=np.float64)
        # Orient the in-plane "outward" direction once, at the default pose, so tool_pitch is
        # continuous everywhere afterwards.
        self._radial_sign = 1.0
        q0 = np.asarray(chain["default_joint_pos"], dtype=np.float64)
        p0 = self.tcp(q0)[0]
        if self._radial(q0) @ np.array([p0[0], p0[1], 0.0]) < 0:
            self._radial_sign = -1.0

    # ---------------------------------------------------------------- FK
    def link_poses(self, q) -> dict[str, np.ndarray]:
        """4x4 pose of every link in the base frame."""
        q = np.asarray(q, dtype=np.float64)
        poses = {"base": np.eye(4)}
        for i, j in enumerate(self.joints):
            poses[j.child] = poses[j.parent] @ j.pre @ _rot_axis(j.axis, q[i]) @ j.post
        return poses

    def tcp(self, q) -> tuple[np.ndarray, np.ndarray]:
        """(position (3,), rotation (3,3)) of the TCP in the base frame."""
        g = self.link_poses(q)["gripper"]
        return g[:3, :3] @ self.tcp_offset + g[:3, 3], g[:3, :3]

    def tool_axis(self, q) -> np.ndarray:
        """Unit vector from the gripper link origin toward the fingertips."""
        _, r = self.tcp(q)
        v = r @ self.tcp_offset
        return v / np.linalg.norm(v)

    def closing_axis(self, q) -> np.ndarray:
        """Unit vector along which the jaw opens/closes (gripper link +x)."""
        return self.tcp(q)[1][:, 0]

    def arm_plane_normal(self, q) -> np.ndarray:
        """Axis of the Pitch joint in the base frame: normal of the arm's vertical plane.

        Pitch, Elbow and Wrist_Pitch are parallel to it, so it is the only axis the arm can
        tilt the hand about.
        """
        j = self.joints[ALL_JOINTS.index("Pitch")]
        return (self.link_poses(q)[j.parent] @ j.pre)[:3, 2]

    def _radial(self, q) -> np.ndarray:
        n = self.arm_plane_normal(q)
        r = np.cross(n, [0.0, 0.0, 1.0])
        return self._radial_sign * r / np.linalg.norm(r)

    def tool_pitch(self, q) -> float:
        """Tool-axis angle in the arm plane, measured from horizontal-outward toward down (rad).

        0 = pointing straight out from the base, +pi/2 = straight down, >pi/2 = tipped back.
        Continuous through vertical (unlike an elevation angle), so pitch tokens can cross it.
        """
        a = self.tool_axis(q)
        return float(np.arctan2(-a[2], a @ self._radial(q)))

    def tool_yaw(self, q) -> tuple[float, float]:
        """(azimuth of the closing axis' horizontal projection in the base frame, its length).

        The length says how meaningful the yaw is: 1 when the closing axis is horizontal,
        0 when it is vertical. The jaw axis is symmetric, so yaw is taken modulo pi.
        """
        c = self.closing_axis(q)
        return float(np.arctan2(c[1], c[0])), float(np.hypot(c[0], c[1]))

    # ---------------------------------------------------------------- IK
    def _task(self, q, with_yaw: bool) -> np.ndarray:
        p, _ = self.tcp(q)
        t = [*p, self.tool_pitch(q)]
        if with_yaw:
            t.append(self.tool_yaw(q)[0])
        return np.asarray(t)

    def jacobian(self, q, joints, with_yaw: bool, eps: float = 1e-6) -> np.ndarray:
        """Numerical Jacobian of [tcp_xyz, tool_pitch(, tool_yaw)] w.r.t. ``joints``."""
        q = np.asarray(q, dtype=np.float64)
        base = self._task(q, with_yaw)
        jac = np.zeros((len(base), len(joints)))
        for c, i in enumerate(joints):
            dq = q.copy()
            dq[i] += eps
            d = self._task(dq, with_yaw) - base
            if with_yaw:
                d[4] = _wrap_pi(d[4])
            jac[:, c] = d / eps
        return jac

    def task_error(self, q, target_pos, target_pitch, target_yaw=None) -> np.ndarray:
        err = [*(np.asarray(target_pos) - self.tcp(q)[0]), _wrap(target_pitch - self.tool_pitch(q))]
        if target_yaw is not None:
            err.append(_wrap_pi(target_yaw - self.tool_yaw(q)[0]))
        return np.asarray(err)

    def solve(self, q_init, target_pos, target_pitch: float | None = None,
              target_yaw: float | None = None, iters: int = 50, damping: float = 1e-3,
              pitch_weight: float = 0.05, yaw_weight: float = 0.05,
              tol_m: float = 1e-4, tol_rad: float = 2e-3) -> tuple[np.ndarray, np.ndarray]:
        """Damped least squares from ``q_init``. Returns (q, task error at q).

        Task = TCP position (+ tool pitch in the arm plane) (+ closing-axis yaw). With a yaw
        target Wrist_Roll joins the solve (5 joints, 5 rows: square); without it the roll is
        left alone. Weights are metres per radian: 0.05 means a 1 rad orientation error
        costs as much as 5 cm of position error, so position wins any conflict. The returned
        error vector is [dx, dy, dz, dpitch(, dyaw)] in metres / radians.
        """
        q = np.asarray(q_init, dtype=np.float64).copy()
        target_pos = np.asarray(target_pos, dtype=np.float64)
        if target_pitch is None:
            target_pitch = self.tool_pitch(q)
        with_yaw = target_yaw is not None
        idx = np.asarray(IK_JOINTS + ((4,) if with_yaw else ()))
        w = np.diag([1.0, 1.0, 1.0, pitch_weight] + ([yaw_weight] if with_yaw else []))
        for _ in range(iters):
            err = self.task_error(q, target_pos, target_pitch, target_yaw)
            if np.linalg.norm(err[:3]) < tol_m and np.all(np.abs(err[3:]) < tol_rad):
                break
            jac = w @ self.jacobian(q, idx, with_yaw)
            e = w @ err
            # Active set: a joint sitting on a limit and asked to push past it is dropped and
            # the step re-solved, so the remaining joints carry the position error. Plain
            # clipping instead throws that share of the step away and the solve stalls
            # (reset pose: Wrist_Pitch is 0.14 rad from its limit, MV_UP stalled 3 mm short).
            free = np.ones(len(idx), dtype=bool)
            rows = len(e)
            for _ in range(len(idx)):
                jf = jac[:, free]
                dq = np.zeros(len(idx))
                dq[free] = jf.T @ np.linalg.solve(jf @ jf.T + damping ** 2 * np.eye(rows), e)
                nxt = q[idx] + dq
                blocked = free & (((nxt <= self.lower[idx]) & (dq < 0)) | ((nxt >= self.upper[idx]) & (dq > 0)))
                at_limit = blocked & ((q[idx] <= self.lower[idx] + 1e-9) | (q[idx] >= self.upper[idx] - 1e-9))
                if not at_limit.any():
                    break
                free &= ~at_limit
            q[idx] = np.clip(q[idx] + dq, self.lower[idx], self.upper[idx])
        return q, self.task_error(q, target_pos, target_pitch, target_yaw)

    def joint_limit_margin(self, q) -> np.ndarray:
        """Distance (rad) of each joint to its nearer soft limit."""
        q = np.asarray(q, dtype=np.float64)
        return np.minimum(q - self.lower, self.upper - q)


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _wrap_pi(a: float) -> float:
    """Wrap to [-pi/2, pi/2): the closing axis is a line, not a direction."""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)
