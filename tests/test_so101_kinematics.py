"""SO-101 kinematics (interpreters/so101_kinematics.py): FK must reproduce the simulator's
own link poses; IK must reach the targets it is given and SAY when it cannot.

tests/fixtures/so101_fk_samples.json holds PhysX link poses of the Workshop SO-101 at random
joint vectors, so this runs without Isaac Sim.
"""
import json
from pathlib import Path

import numpy as np

from interpreters.so101_kinematics import So101Kinematics, quat_wxyz_to_mat

SAMPLES = Path(__file__).resolve().parent / "fixtures" / "so101_fk_samples.json"


def _pose(p, q_xyzw):
    t = np.eye(4)
    t[:3, :3] = quat_wxyz_to_mat([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    t[:3, 3] = p
    return t


def test_fk_matches_physx():
    kin = So101Kinematics()
    data = json.loads(SAMPLES.read_text())
    names = data["link_names"]
    worst_pos, worst_rot = 0.0, 0.0
    for s in data["samples"]:
        tfs = [_pose(t[:3], t[3:]) for t in s["link_tf_world_xyzw"]]
        base_inv = np.linalg.inv(tfs[names.index("base")])
        fk = kin.link_poses(s["q"])
        for name in ("gripper", "jaw", "wrist", "lower_arm"):
            ref = base_inv @ tfs[names.index(name)]
            worst_pos = max(worst_pos, float(np.linalg.norm(fk[name][:3, 3] - ref[:3, 3])))
            cos = np.clip((np.trace(fk[name][:3, :3].T @ ref[:3, :3]) - 1) / 2, -1, 1)
            worst_rot = max(worst_rot, float(np.arccos(cos)))
    assert worst_pos < 1e-4, f"FK position error {worst_pos * 1000:.3f} mm"
    assert worst_rot < 1e-3, f"FK rotation error {np.degrees(worst_rot):.3f} deg"


Q_RESET = np.array([-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465])


def test_ik_reaches_2cm_steps_pitch_only():
    kin = So101Kinematics()
    q = Q_RESET
    p0, _ = kin.tcp(q)
    pitch0 = kin.tool_pitch(q)
    for d in np.eye(3) * 0.02:
        for sign in (1, -1):
            qs, err = kin.solve(q, p0 + sign * d, pitch0)
            assert np.linalg.norm(err[:3]) < 5e-4, (d * sign, err)
            assert abs(qs[4] - q[4]) < 1e-12 and abs(qs[5] - q[5]) < 1e-12  # roll / jaw untouched


def test_tool_pitch_is_continuous_through_vertical():
    kin = So101Kinematics()
    q = Q_RESET.copy()
    prev = kin.tool_pitch(q)
    for _ in range(60):
        q[3] -= 0.02  # sweep Wrist_Pitch
        cur = kin.tool_pitch(q)
        assert abs(cur - prev) < 0.05
        prev = cur


def test_orientation_hold_is_exact_where_reachable():
    """Pitch 60 deg + yaw held: 2 cm moves keep pitch and closing-axis yaw exactly.

    The third rotation, the HEADING of a tilted tool, follows the base rotation: a 5-DoF arm
    cannot hold it unless the tool points straight down. Lateral tokens turn it by a few
    degrees; that is physics, and the bound below documents how much.
    """
    kin = So101Kinematics()
    p_start = np.array([0.04, -0.26, 0.12])  # base frame (task (0.26, -0.04, 0.12))
    q, err = kin.solve(Q_RESET, p_start, np.radians(60), 0.0, iters=300)
    assert np.linalg.norm(err[:3]) < 1e-4 and np.all(np.abs(err[3:]) < 1e-3), err
    r0 = kin.tcp(q)[1]
    for d in np.vstack([np.eye(3), -np.eye(3)]) * 0.02:
        qs, err = kin.solve(q, kin.tcp(q)[0] + d, np.radians(60), 0.0)
        assert np.linalg.norm(err[:3]) < 1e-4, (d, err)
        assert np.degrees(abs(kin.tool_pitch(qs) - np.radians(60))) < 0.1, d
        yaw = (kin.tool_yaw(qs)[0] + np.pi / 2) % np.pi - np.pi / 2  # closing axis is a line
        assert np.degrees(abs(yaw)) < 0.1, d
        cos = np.clip((np.trace(kin.tcp(qs)[1].T @ r0) - 1) / 2, -1, 1)
        assert np.degrees(np.arccos(cos)) < 5.0, d  # heading change, lateral moves only


def test_unreachable_orientation_is_reported_not_hidden():
    """Top-down at the wrist limit cannot be held: the solver must SAY so (feasible() relies on it)."""
    kin = So101Kinematics()
    q, _ = kin.solve(Q_RESET, np.array([0.0, -0.25, 0.10]), np.pi / 2, 0.0, iters=300)
    _, err = kin.solve(q, kin.tcp(q)[0] + [0, 0, 0.02], np.pi / 2, 0.0)
    assert np.degrees(np.max(np.abs(err[3:]))) > 1.0  # pitch had to give: reported


def test_constant_roll_gives_horizontal_closing_axis_everywhere():
    """Pitch/Elbow/Wrist_Pitch are parallel: at Wrist_Roll = -pi/2 the jaw closes along the
    arm-plane normal in EVERY pose (the controller's default roll relies on this)."""
    kin = So101Kinematics()
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.uniform(kin.lower, kin.upper)
        q[4] = -np.pi / 2
        assert abs(kin.closing_axis(q) @ kin.arm_plane_normal(q)) > 1 - 1e-9
        assert abs(kin.closing_axis(q)[2]) < 1e-5  # USD quaternions are rounded (0.70711)
