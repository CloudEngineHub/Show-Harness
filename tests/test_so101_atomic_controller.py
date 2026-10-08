"""SO-101 interpreter (interpreters/so101_atomic_controller.py) on a physics-free arm: exact
tokens, whole-token refusals, RT_* on 5 DoF, closure, contact handling.

The stand-in arm moves each tick a fraction of the way to (command - sag): a first-order lag
plus a constant gravity sag on the arm joints, the two effects the real controller has to
cope with. Its base is yawed +90 deg in the world, as in the Workshop scene.
"""
import numpy as np
import yaml

from core.action_units import RT_ATOMS
from interpreters.so101_atomic_controller import (
    DEFAULT_PRIMITIVES,
    ROLL_CLOSING_HORIZONTAL,
    UNSUPPORTED,
    UNSUPPORTED_REASON,
    So101ArmController,
    So101AtomicExec,
)
from scripts.trajectory.real2sim.atomic_tokenizer import MOVE_DIRS, OPPOSITE

Q_RESET = np.array([-0.2736, -0.6109, -0.0745, 1.5148, -1.6034, -0.1465])
R_WB = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class KinematicArm(So101ArmController):
    def __init__(self, q0=Q_RESET, alpha: float = 0.5, sag=(0.0, 0.008, 0.012, 0.006, 0.0, 0.0),
                 wall_y_max=None, **kw):
        super().__init__(**kw)
        # Optional obstacle: the TCP cannot pass task-frame y = wall_y_max (the joints simply
        # stop, like a hand pressed against the rack).
        self.wall_y_max = wall_y_max
        self.q = np.asarray(q0, dtype=np.float64).copy()
        self.alpha = float(alpha)
        self.sag = np.asarray(sag, dtype=np.float64)

    def _read_q(self):
        return self.q.copy()

    def _send(self, q_cmd):
        q_next = self.q + self.alpha * (np.asarray(q_cmd) - self.sag - self.q)
        if self.wall_y_max is not None:
            y = (R_WB @ self.kin.tcp(q_next)[0])[1]
            if y > self.wall_y_max:
                return  # blocked: the arm does not move this tick
        self.q = q_next

    def _r_world_base(self):
        return R_WB

    def success(self):
        return False

    def grab_frames(self):
        raise NotImplementedError("no cameras on the kinematic arm")

    def reset(self, seed: int = 0):
        self._begin_episode()



HOME_TASK = np.array([0.26, -0.04, 0.12])  # world (0.21, -0.04, 0.12): above the vials


def make(pitch_deg=60.0, yaw=0.0):
    arm = KinematicArm()
    arm.reset()
    ex = So101AtomicExec(arm, step_m=0.02)
    assert ex.goto_pose(HOME_TASK, np.radians(pitch_deg), yaw, budget=400)
    ex.quiesce()
    return arm, ex


def orient(arm):
    q = arm.q_obs()
    return arm.kin.tool_pitch(q), arm.kin.tool_yaw(q)[0]


def test_every_token_is_20mm_on_axis_and_keeps_orientation():
    arm, ex = make()
    for tok in MOVE_DIRS:
        for t in (tok, OPPOSITE[tok]):
            p0, o0 = arm.tcp_pos(), orient(arm)
            ex.move(t)
            assert ex.last_blocked is None, (t, ex.last_blocked)
            d = arm.tcp_pos() - p0
            along = d @ MOVE_DIRS[t]
            assert abs(along - 0.02) < 1e-3, (t, along)
            assert np.linalg.norm(d - along * MOVE_DIRS[t]) < 1e-3, t
            o1 = orient(arm)
            assert abs(np.degrees(o1[0] - o0[0])) < 0.5, t
            assert abs(np.degrees((o1[1] - o0[1] + np.pi / 2) % np.pi - np.pi / 2)) < 0.5, t


def test_gravity_sag_is_learned():
    arm, ex = make()
    ex.hold(20)
    assert np.allclose(arm.q_bias[1:4], arm.sag[1:4], atol=1e-3)


def test_infeasible_token_is_refused_without_moving():
    arm, ex = make()
    arm.workspace = {"min": [0.0, -1.0, arm.tcp_pos()[2] - 0.01], "max": [1.0, 1.0, 1.0]}
    p0, c0 = arm.tcp_pos(), arm.ctrl_step_count
    assert ex.move("MV_DOWN") == 0.0
    assert ex.last_blocked.startswith("workspace(z)")
    assert arm.ctrl_step_count == c0 and np.allclose(arm.tcp_pos(), p0)
    ex.move("MV_UP")
    assert ex.last_blocked is None


def test_rotation_tokens_turn_10deg_and_hold_the_tcp():
    arm, ex = make()
    for tok, sign_p, sign_y in (("RT_PITCH_FWD", -1, 0), ("RT_PITCH_BACK", 1, 0),
                                ("RT_YAW_CCW", 0, 1), ("RT_YAW_CW", 0, -1)):
        p0, o0 = arm.tcp_pos(), orient(arm)
        ex.rotate(tok)
        assert ex.last_blocked is None, (tok, ex.last_blocked)
        o1 = orient(arm)
        dp = np.degrees(o1[0] - o0[0])
        dy = np.degrees((o1[1] - o0[1] + np.pi / 2) % np.pi - np.pi / 2)
        assert abs(dp - 10 * sign_p) < 0.5 and abs(dy - 10 * sign_y) < 0.5, (tok, dp, dy)
        assert np.linalg.norm(arm.tcp_pos() - p0) < 1e-3, tok


def test_roll_is_refused():
    arm, ex = make()
    assert ex.rotate("RT_ROLL_LEFT") == 0.0 and ex.last_blocked == "unsupported_on_5dof"


def test_random_walk_closes():
    arm, ex = make()
    rng = np.random.default_rng(0)
    ex.lattice_origin = arm.tcp_pos().copy()
    p0 = arm.tcp_pos().copy()
    seq = []
    while len(seq) < 40:
        t = str(rng.choice(list(MOVE_DIRS)))
        ex.move(t)
        if not ex.last_blocked:
            seq.append(t)
    for t in reversed(seq):
        ex.move(OPPOSITE[t])
    ex.quiesce()
    assert np.linalg.norm(arm.tcp_pos() - p0) < 1e-3


def test_tokens_end_at_rest_without_overshoot():
    arm, ex = make()
    trace = []
    arm.step_hooks.append(lambda rec: trace.append(np.asarray(rec["tcp_obs"][:3])))
    for t in ("MV_UP", "MV_DOWN", "MV_LEFT", "MV_RIGHT"):
        trace.clear()
        p0 = arm.tcp_pos()
        ex.move(t)
        assert ex.last_converged, t
        prog = [(p - p0) @ MOVE_DIRS[t] for p in trace]
        assert max(prog) - 0.02 < 1e-3, (t, max(prog))     # no overshoot past 20 mm
        assert np.linalg.norm(trace[-1] - trace[-3]) < 1e-3  # still at the end: at rest


def test_contact_aborts_the_token_and_never_runs_away():
    """Hand pressed against an obstacle: stop, say 'contact', do not learn the push as sag."""
    arm, ex = make()
    arm.wall_y_max = arm.tcp_pos()[1] + 0.008  # wall 8 mm to the right
    p0 = arm.tcp_pos()
    ex.move("MV_RIGHT")
    # no force sensor on this arm: the wall shows up as a stall (at rest short of the target)
    # or as a joint gap without progress; either way the token must be aborted
    assert ex.last_blocked in ("contact", "stall"), ex.last_blocked
    assert np.all(np.abs(arm.q_bias) <= arm.bias_max_rad + 1e-12)
    ex.hold(60)  # sitting against the wall must not drift anywhere
    assert np.linalg.norm(arm.tcp_pos() - p0) < 0.012
    ex.move("MV_LEFT")  # and the arm is still usable afterwards
    assert ex.last_blocked is None and ex.last_converged


def make_roll(pitch_deg=47.0):
    arm = KinematicArm()
    arm.reset()
    ex = So101AtomicExec(arm, step_m=0.02)
    assert ex.goto_pose(HOME_TASK, np.radians(pitch_deg), roll=ROLL_CLOSING_HORIZONTAL, budget=600)
    return arm, ex


def test_roll_mode_keeps_the_jaw_closing_horizontally():
    """Default mode: Wrist_Roll at -pi/2 -> closing axis horizontal after every token."""
    arm, ex = make_roll()
    for tok in MOVE_DIRS:
        for t in (tok, OPPOSITE[tok]):
            ex.move(t)
            assert ex.last_blocked is None, (t, ex.last_blocked)
            c = arm.tcp_rotmat()[:, 0]
            assert abs(np.degrees(np.arcsin(c[2]))) < 0.3, (t, c)


def test_roll_mode_yaw_tokens_turn_the_hand_10deg():
    arm, ex = make_roll()
    for tok, sign in (("RT_YAW_CCW", 1), ("RT_YAW_CW", -1)):
        r0, p0 = arm.q_obs()[4], arm.tcp_pos()
        ex.rotate(tok)
        assert ex.last_blocked is None, (tok, ex.last_blocked)
        assert abs(np.degrees(arm.q_obs()[4] - r0) - 10 * sign) < 0.3, tok
        assert np.linalg.norm(arm.tcp_pos() - p0) < 1e-3, tok


def test_hand_floor_uses_the_whole_gripper():
    """The floor is checked on the collision hull, so a token that keeps the TCP above the
    floor but drives the fingertips below it is refused."""
    arm, ex = make_roll()
    q = arm.q_obs()
    tcp_z = arm.tcp_pos()[2]
    low = arm.hand_lowest_z(q)
    assert low < tcp_z  # fingertip region hangs below the TCP
    # floor between (lowest point - 20 mm) and the TCP after MV_DOWN: the TCP alone would pass
    arm.hand_floor_task_z = low - 0.019
    p0 = arm.tcp_pos()
    ex.move("MV_DOWN")
    assert ex.last_blocked and ex.last_blocked.startswith("hand_floor"), ex.last_blocked
    assert np.allclose(arm.tcp_pos(), p0, atol=1e-4)


def test_integral_closes_the_gap_when_sag_exceeds_the_bias_cap():
    """Sag beyond bias_max_rad (unmodelled at full reach): the servo's integral term still
    brings every token to within tolerance instead of stalling 1-2 mm short."""
    arm, ex = make_roll()
    arm.sag = np.array([0.0, 0.05, 0.03, 0.02, 0.0, 0.0])  # > bias_max_rad on Pitch
    for t in ("MV_FWD", "MV_UP", "MV_BACK", "MV_DOWN"):
        p0 = arm.tcp_pos()
        ex.move(t)
        assert ex.last_blocked is None, (t, ex.last_blocked, ex.last_abort)
        assert abs(np.linalg.norm(arm.tcp_pos() - p0) - 0.02) < 1e-3, t


def test_primitives_match_the_shared_token_contract():
    """configs/primitives_so101.yaml must give MV_* the same directions as the tokenizer, so
    sim data from this interpreter means what every other embodiment's data means."""
    prim = yaml.safe_load(DEFAULT_PRIMITIVES.read_text())["atomic_primitives"]
    for tok, vec in MOVE_DIRS.items():
        assert np.allclose(prim[tok], vec), tok


def test_unsupported_rotation_fails_deterministically():
    """RT_ROLL_* is part of the shared vocabulary but not realisable on 5 DoF: check() and
    rotate() must agree, give the same reason every time, and never move the arm."""
    arm, ex = make_roll()
    assert set(UNSUPPORTED) <= set(RT_ATOMS)
    for tok in UNSUPPORTED:
        for _ in range(3):
            p0, q0, c0 = arm.tcp_pos(), arm.q_obs(), arm.ctrl_step_count
            assert ex.check(tok) == (False, UNSUPPORTED_REASON)
            assert ex.run(tok) == 0.0 and ex.last_blocked == UNSUPPORTED_REASON
            assert arm.ctrl_step_count == c0 and np.allclose(arm.q_obs(), q0)
            assert np.allclose(arm.tcp_pos(), p0)


def test_check_agrees_with_execution():
    """check() is what generators call BEFORE recording a frame: whatever it accepts must run
    unblocked, whatever it refuses must be refused by run() without moving."""
    arm, ex = make_roll()
    arm.workspace = {"min": [0.0, -1.0, arm.tcp_pos()[2] - 0.01], "max": [1.0, 1.0, 1.0]}
    for tok in [*MOVE_DIRS, "RT_PITCH_FWD", "RT_PITCH_BACK", "RT_YAW_CW", "RT_YAW_CCW"]:
        ok, why = ex.check(tok)
        p0 = arm.tcp_pos()
        ex.run(tok)
        if ok:
            assert ex.last_blocked is None, (tok, ex.last_blocked)
            ex.run(OPPOSITE[tok])
        else:
            assert ex.last_blocked == why and np.allclose(arm.tcp_pos(), p0, atol=1e-4), tok
