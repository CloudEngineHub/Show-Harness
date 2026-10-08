"""Quantitative motion check of the SO-101 interpreter on the Workshop sim.

Numbers go to motion_report.json; pass criteria are the CRITERIA constants below, fixed
before looking at results. The values recorded in configs/primitives_so101.yaml come from
this script. Sections:

  scene     measure table / vial / rack geometry; set the Z floor from it
  grid      from several start poses: each MV_* token and its opposite. Travel, off-axis,
            orientation change, overshoot, control steps, refusals.
  walk      N random feasible tokens, then the exact inverse sequence: closure error
  rotate    RT_PITCH / RT_YAW tokens: achieved angle, TCP drift while turning

    WORKSHOP_ROOT=... bash scripts/trajectory/real2sim/so101_workshop/run_in_docker.sh \
        scripts/trajectory/real2sim/so101_workshop/verify_motion.py --out /workspace/out/verify_motion
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack-DR")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pitch_deg", type=float, default=47.0, help="home tool pitch (90 = straight down)")
parser.add_argument("--orient", choices=["roll", "yaw"], default="roll",
                    help="second orientation reference: Wrist_Roll angle (default) or closing-axis yaw")
parser.add_argument("--roll_deg", type=float, default=-90.0,
                    help="Wrist_Roll in roll mode (-90 = jaw closes horizontally in every pose)")
parser.add_argument("--yaw_deg", type=float, default=0.0,
                    help="home closing-axis yaw in the BASE frame (0 = across the vials)")
parser.add_argument("--home_world", default="0.22,-0.06,0.14", help="home TCP, world frame")
parser.add_argument("--grid_world", default="0.18,0.24|-0.12,0.0|0.10,0.16",
                    help="x values|y values|z values of the grid start points, world frame")
parser.add_argument("--walk_tokens", type=int, default=80)
parser.add_argument("--sections", default="scene,grid,walk,rotate")
parser.add_argument("--out", default="/workspace/out/verify_motion")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import sys  # noqa: E402

import numpy as np  # noqa: E402

from interpreters.so101_atomic_controller import So101AtomicExec  # noqa: E402
from scripts.trajectory.real2sim.atomic_tokenizer import MOVE_DIRS, OPPOSITE, to_np  # noqa: E402
from scripts.trajectory.real2sim.backends.so101_workshop import So101WorkshopBackend  # noqa: E402
from scripts.trajectory.real2sim.so101_workshop.scene import VialsRackScene  # noqa: E402

# Pass criteria. Set BEFORE looking at results; change only with a written reason.
CRITERIA = {
    "travel_mm": [18.0, 22.0],        # one MV_* token = 20 mm +-10 %
    "off_axis_mm_max": 2.0,           # sideways error of one token
    "overshoot_mm_max": 2.0,          # peak travel past the end point during a token
    "orientation_deg_max": 1.0,       # hand pitch / second-reference change during a translation
    "walk_closure_mm_max": 2.0,       # N random tokens then the inverse sequence
    "rot_step_deg": [9.0, 11.0],      # one RT_* token = 10 deg
    "rot_tcp_drift_mm_max": 3.0,      # TCP excursion while rotating
    "z_floor_above_table_m": 0.008,   # the TCP may not go lower than this above the mat
    "obstacle_margin_m": 0.03,        # grid start points closer than this to the rack are skipped
    "hand_floor_above_table_m": 0.003,  # lowest point of the gripper/jaw collision hulls above the mat
}


class Trace:
    """Per-env-step TCP trace, for overshoot."""

    def __init__(self):
        self.on = False
        self.pts = []

    def __call__(self, rec):
        if self.on:
            self.pts.append(np.asarray(rec["tcp_obs"][:3]))


def world_to_task(be, p_w):
    return np.asarray(p_w) - to_np(be.robot.data.root_pos_w[0])


def scene_geometry(be):
    from pxr import Usd, UsdGeom

    stage = be.u.sim.stage
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
    out = {}
    for name, path in (("vial_1", "/World/envs/env_0/Vial_1"), ("vial_2", "/World/envs/env_0/Vial_2"),
                       ("vial_3", "/World/envs/env_0/Vial_3"), ("rack", "/World/envs/env_0/Rack_Left")):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        out[name] = {"min": list(r.GetMin()), "max": list(r.GetMax())}
    for k in ("vial_1", "vial_2", "vial_3"):
        obj = be.u.scene[k]
        out[k]["pos_w"] = to_np(obj.data.root_pos_w[0]).tolist()
        out[k]["quat_w"] = to_np(obj.data.root_quat_w[0]).tolist()
    vmins = [out[k]["min"][2] for k in ("vial_1", "vial_2", "vial_3") if k in out]
    out["table_top_z_w"] = float(min(vmins))  # lying vials rest on the mat
    return out


def orientation(be):
    """(tool pitch, the HELD second reference): Wrist_Roll in roll mode, closing yaw in yaw mode."""
    q = be.q_obs()
    held = be.kin.tool_yaw(q)[0] if be.yaw_ref is not None else q[4]
    return be.kin.tool_pitch(q), held


def closing_tilt_deg(be):
    return float(np.degrees(np.arcsin(np.clip(be.tcp_rotmat()[2, 0], -1, 1))))


def ang(a):
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def main():
    os.makedirs(args.out, exist_ok=True)
    sections = args.sections.split(",")
    crit = CRITERIA
    be = So101WorkshopBackend.make(args.task, seed=args.seed)
    trace = Trace()
    be.step_hooks.append(trace)
    ex = So101AtomicExec(be)
    pitch, yaw = np.radians(args.pitch_deg), np.radians(args.yaw_deg)
    roll = np.radians(args.roll_deg)

    def goto(pos_task, budget=400):
        if args.orient == "yaw":
            return ex.goto_pose(pos_task, pitch, yaw, budget=budget)
        return ex.goto_pose(pos_task, pitch, roll=roll, budget=budget)
    report = {"args": vars(args)}

    be.reset(args.seed)
    geo = scene_geometry(be)
    scene = VialsRackScene(be)  # obstacles from the CURRENT poses (USD world bounds are stale)
    report["scene"] = geo
    floor_task = geo["table_top_z_w"] - to_np(be.robot.data.root_pos_w[0])[2] + crit["z_floor_above_table_m"]
    table_task = geo["table_top_z_w"] - to_np(be.robot.data.root_pos_w[0])[2]
    # the WHOLE hand (collision hulls) must stay above the table + margin; the TCP box floor
    # only keeps the TCP itself off the table
    be.hand_floor_task_z = table_task + crit["hand_floor_above_table_m"]
    be.workspace = {"min": [0.05, -0.30, min(floor_task, table_task + 0.002)], "max": [0.45, 0.25, 0.30]}
    report["workspace_task"] = be.workspace

    def go_home(tag):
        ok = goto(world_to_task(be, [float(x) for x in args.home_world.split(",")]))
        ex.quiesce()
        return ok

    # ---------------------------------------------------------------- grid
    if "grid" in sections:
        xs, ys, zs = ([float(v) for v in part.split(",")] for part in args.grid_world.split("|"))
        starts = [(x, y, z) for x in xs for y in ys for z in zs]
        rack_lo, rack_hi = scene.rack_box(crit["obstacle_margin_m"])

        def in_obstacle(pw):
            pt = world_to_task(be, pw)
            return bool(np.all(pt >= rack_lo) and np.all(pt <= rack_hi))
        rows = []
        for sw in starts:
            if in_obstacle(sw):
                rows.append({"start_world": sw, "reached": None, "skipped": "inside rack (+margin)", "tokens": {}})
                continue
            reached = goto(world_to_task(be, sw))
            ex.quiesce()
            entry = {"start_world": sw, "reached": reached, "tokens": {}}
            if not reached:
                rows.append(entry)
                continue
            for tok in MOVE_DIRS:
                for t in (tok, OPPOSITE[tok]):
                    p0 = be.tcp_pos()
                    o0 = orientation(be)
                    c0 = be.ctrl_step_count
                    trace.pts, trace.on = [], True
                    travel = ex.move(t)
                    trace.on = False
                    d = be.tcp_pos() - p0
                    axis = MOVE_DIRS[t]
                    along = float(d @ axis)
                    prog = [float((p - p0) @ axis) for p in trace.pts] or [0.0]
                    o1 = orientation(be)
                    entry["tokens"].setdefault(tok, []).append({
                        "token": t,
                        "blocked": ex.last_blocked,
                        "along_mm": along * 1000,
                        "off_axis_mm": float(np.linalg.norm(d - along * axis) * 1000),
                        "overshoot_mm": float(max(0.0, max(prog) - along) * 1000),
                        "dpitch_deg": float(np.degrees(o1[0] - o0[0])),
                        "dyaw_deg": float(np.degrees(ang(o1[1] - o0[1]))),
                        "closing_tilt_deg": closing_tilt_deg(be),
                        "ctrl_steps": be.ctrl_step_count - c0,
                        "travel_m": travel,
                    })
            rows.append(entry)
        report["grid"] = rows

    # ---------------------------------------------------------------- walk
    if "walk" in sections:
        go_home("walk")
        rng = np.random.default_rng(args.seed)
        ex.lattice_origin = be.tcp_pos().copy()
        p_start, o_start = be.tcp_pos().copy(), orientation(be)
        seq, refused, reasons, avoided = [], 0, {}, 0
        margin = crit["obstacle_margin_m"]
        boxes = [scene.rack_box(margin)]
        for vl in scene.vials():  # cylinder -> AABB of its two end caps, + radius + margin
            ends = np.array([vl.pos + vl.axis * scene.vial_length / 2, vl.pos - vl.axis * scene.vial_length / 2])
            pad = scene.vial_diameter / 2 + margin
            boxes.append((ends.min(0) - pad, ends.max(0) + pad))

        def hits_obstacle(p_task):
            return any(bool(np.all(p_task >= lo) and np.all(p_task <= hi)) for lo, hi in boxes)

        # bounded: a pose where every token is refused must end the walk, not hang it.
        # Tokens that would END inside an obstacle (+margin) are not attempted: the walk tests
        # tracking in free space; contact handling is tested separately.
        while len(seq) < args.walk_tokens and refused + avoided < 20 * args.walk_tokens:
            t = str(rng.choice(list(MOVE_DIRS)))
            if hits_obstacle(ex.lattice_node(be.tcp_pos()) + MOVE_DIRS[t] * ex.step_m):
                avoided += 1
                continue
            ex.move(t)
            if ex.last_blocked:
                refused += 1
                reasons[ex.last_blocked] = reasons.get(ex.last_blocked, 0) + 1
                continue
            seq.append(t)
        mid = be.tcp_pos().copy()
        for t in reversed(seq):
            ex.move(OPPOSITE[t])
        ex.quiesce()
        o_end = orientation(be)
        ex.lattice_origin = None
        report["walk"] = {
            "tokens": len(seq), "refused_attempts": refused, "refusal_reasons": reasons,
            "avoided_obstacle_tokens": avoided,
            "max_excursion_mm": float(np.linalg.norm(mid - p_start) * 1000),
            "closure_error_mm": float(np.linalg.norm(be.tcp_pos() - p_start) * 1000),
            "closure_dpitch_deg": float(np.degrees(o_end[0] - o_start[0])),
            "closure_dyaw_deg": float(np.degrees(ang(o_end[1] - o_start[1]))),
        }

    # ---------------------------------------------------------------- rotate
    if "rotate" in sections:
        go_home("rotate")
        rot = []
        for tok, back in (("RT_PITCH_FWD", "RT_PITCH_BACK"), ("RT_PITCH_BACK", "RT_PITCH_FWD"),
                          ("RT_YAW_CCW", "RT_YAW_CW"), ("RT_YAW_CW", "RT_YAW_CCW")):
            for t in [tok] * 3 + [back] * 3:
                p0, o0 = be.tcp_pos(), orientation(be)
                trace.pts, trace.on = [], True
                ex.rotate(t)
                trace.on = False
                o1 = orientation(be)
                drift = max((np.linalg.norm(p - p0) for p in trace.pts), default=0.0)
                rot.append({"token": t, "blocked": ex.last_blocked,
                            "dpitch_deg": float(np.degrees(o1[0] - o0[0])),
                            "dyaw_deg": float(np.degrees(ang(o1[1] - o0[1]))),
                            "tcp_drift_max_mm": float(drift * 1000),
                            "tcp_end_err_mm": float(np.linalg.norm(be.tcp_pos() - p0) * 1000)})
        report["rotate"] = rot

    report["verdict"] = evaluate(report, crit)
    with open(f"{args.out}/motion_report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps(report["verdict"], indent=1, default=float), flush=True)
    ok = all(v.get("PASS", False) for v in report["verdict"].values())
    print(f"[verify_motion] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def evaluate(r, c):
    out = {}
    if "grid" in r:
        toks = [m for e in r["grid"] for ms in e["tokens"].values() for m in ms if not m["blocked"]]
        travel = np.array([m["along_mm"] for m in toks])
        out["grid"] = {
            "starts_reached": f"{sum(bool(e['reached']) for e in r['grid'])}/{sum(1 for e in r['grid'] if not e.get('skipped'))}",
            "executed": len(toks),
            "blocked": sum(1 for e in r["grid"] for ms in e["tokens"].values() for m in ms if m["blocked"]),
            "contacts_in_free_space": sum(1 for e in r["grid"] for ms in e["tokens"].values() for m in ms
                                          if m["blocked"] in ("contact", "not_converged")),
            "skipped_starts": sum(1 for e in r["grid"] if e.get("skipped")),
            "travel_mm_min_max": [float(travel.min()), float(travel.max())] if len(toks) else None,
            "off_axis_mm_max": max((m["off_axis_mm"] for m in toks), default=None),
            "overshoot_mm_max": max((m["overshoot_mm"] for m in toks), default=None),
            "dpitch_deg_absmax": max((abs(m["dpitch_deg"]) for m in toks), default=None),
            "dyaw_deg_absmax": max((abs(m["dyaw_deg"]) for m in toks), default=None),
            "closing_tilt_deg_absmax": max((abs(m.get("closing_tilt_deg", 0.0)) for m in toks), default=None),
        }
        g = out["grid"]
        g["PASS"] = bool(len(toks) and g["contacts_in_free_space"] == 0 and travel.min() >= c["travel_mm"][0] and travel.max() <= c["travel_mm"][1]
                         and g["off_axis_mm_max"] <= c["off_axis_mm_max"]
                         and g["overshoot_mm_max"] <= c["overshoot_mm_max"]
                         and g["dpitch_deg_absmax"] <= c["orientation_deg_max"]
                         and g["dyaw_deg_absmax"] <= c["orientation_deg_max"])
    if "walk" in r:
        w = r["walk"]
        out["walk"] = {"PASS": bool(w["tokens"] > 0 and w["closure_error_mm"] <= c["walk_closure_mm_max"]
                                    and abs(w["closure_dpitch_deg"]) <= c["orientation_deg_max"]
                                    and abs(w["closure_dyaw_deg"]) <= c["orientation_deg_max"]), **w}
    if "rotate" in r:
        ex_ = [m for m in r["rotate"] if not m["blocked"]]
        angs = [abs(m["dpitch_deg"]) + abs(m["dyaw_deg"]) for m in ex_]
        out["rotate"] = {
            "executed": len(ex_), "blocked": len(r["rotate"]) - len(ex_),
            "angle_deg_min_max": [min(angs), max(angs)] if angs else None,
            "tcp_drift_mm_max": max((m["tcp_drift_max_mm"] for m in ex_), default=None),
        }
        out["rotate"]["PASS"] = bool(angs and min(angs) >= c["rot_step_deg"][0] and max(angs) <= c["rot_step_deg"][1]
                                     and out["rotate"]["tcp_drift_mm_max"] <= c["rot_tcp_drift_mm_max"])
    return out


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        sys.stdout.flush()
        os._exit(rc)  # Kit's shutdown can hang inside the container; the report is on disk
