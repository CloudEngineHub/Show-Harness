"""Measure the SO-101 Workshop camera contract against the shared prompt geometry.

The shared prompts read directions off the images (prompts/controller.txt): on the agentview,
image left/right -> MV_LEFT/MV_RIGHT, bottom -> MV_FWD, top -> MV_BACK; on the wrist view the
fingertips are at the TOP and an object to the gripper's left -> MV_LEFT, near the bottom ->
MV_FWD. This script checks that configs/robot_so101_workshop.yaml's rotation/flip produces
exactly that, by projection rather than by eye:

* agentview: each MV_* moves the TCP; project start and end into the raw camera.
* wrist: measured on the RENDERED frames, not by projection: each MV_* is executed for real
  and the scene shift between the wrist frames before and after is found by phase
  correlation; the camera rides on the hand, so the hand heads toward minus that shift.
* fingertips: the fingers move with the wrist camera, so they are the pixels that stay
  unchanged through all the moves; their centroid must land in the top half.

Every raw image point is then pushed through ``core.record.images.rotate_and_flip`` itself
(a one-pixel image, so the check cannot disagree with the transform the data goes through).
Exit status 1 on any mismatch. The transformed views are saved for a look by eye as well.

    WORKSHOP_ROOT=... bash scripts/trajectory/real2sim/so101_workshop/run_in_docker.sh \\
        scripts/trajectory/real2sim/so101_workshop/check_views.py --out /workspace/out/check_views
"""
import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--robot-config", default="configs/robot_so101_workshop.yaml")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pitch-deg", type=float, default=47.0, help="hand tilt for the check (grasp-like)")
parser.add_argument("--home", type=float, nargs=3, default=(0.20, -0.02, 0.17),
                    help="TCP start in the WORLD frame (m): above the vials")
parser.add_argument("--out", default="/workspace/out/check_views")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from core.config import camera_contract  # noqa: E402
from core.record.images import rotate_and_flip  # noqa: E402
from interpreters.so101_atomic_controller import ROLL_CLOSING_HORIZONTAL, So101AtomicExec  # noqa: E402
from interpreters.so101_kinematics import quat_wxyz_to_mat  # noqa: E402
from scripts.trajectory.real2sim.atomic_tokenizer import OPPOSITE  # noqa: E402
from scripts.trajectory.real2sim.backends.so101_workshop import So101WorkshopBackend  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# Expected image direction of each token AFTER the contract (u right, v down).
EXPECT_AGENTVIEW = {"MV_FWD": (0, 1), "MV_BACK": (0, -1), "MV_LEFT": (-1, 0), "MV_RIGHT": (1, 0)}
EXPECT_WRIST = {"MV_FWD": (0, 1), "MV_BACK": (0, -1), "MV_LEFT": (-1, 0), "MV_RIGHT": (1, 0)}


def camera(be, name):
    cam = be.u.scene[f"camera_{name}"]
    k = cam.data.intrinsic_matrices[0].cpu().numpy()
    c = cam.data.pos_w[0].cpu().numpy() - be.robot.data.root_pos_w[0].cpu().numpy()  # task frame
    r = quat_wxyz_to_mat(cam.data.quat_w_ros[0].cpu().numpy())
    return k, c, r


def project(k, c, r, p):
    pc = r.T @ (np.asarray(p, dtype=np.float64) - c)
    if pc[2] <= 1e-6:
        raise ValueError(f"point behind the camera (depth {pc[2]:.3f} m)")
    uv = k @ pc
    return uv[:2] / uv[2]


def transform_point(uv, shape, degrees, flip):
    """Where raw point ``uv`` (u right, v down) lands after rotate_and_flip, as a formula so
    points outside the frame keep their direction. :func:`self_check` pins it to the real
    function on in-frame pixels."""
    h, w = shape
    u, v = float(uv[0]), float(uv[1])
    for _ in range((int(degrees) % 360) // 90):  # np.rot90 k=1: counter-clockwise
        u, v, h, w = v, (w - 1) - u, w, h
    mode = (flip or "none").lower()
    if mode in ("vertical", "both"):
        v = (h - 1) - v
    if mode in ("horizontal", "both"):
        u = (w - 1) - u
    return np.array([u, v]), (h, w)


def self_check(shape, degrees, flip):
    h, w = shape
    for u, v in ((3, 5), (w - 7, 2), (w // 2, h - 4)):
        img = np.zeros((h, w, 1), dtype=np.uint8)
        img[v, u] = 255
        out = rotate_and_flip(img, degrees, flip)
        vv, uu = np.unravel_index(int(np.argmax(out[..., 0])), out.shape[:2])
        got, _ = transform_point((u, v), shape, degrees, flip)
        assert (uu, vv) == tuple(int(x) for x in got), ((u, v), (uu, vv), got)


def scene_shift(a, b, ignore, pool=4, max_px=60):
    """Translation (du, dv) that best moves image ``a`` onto ``b``, ignoring the ``ignore``
    pixels (the fingers, which ride with the camera). Brute-force masked mean squared
    difference on ``pool``-x down-sampled images: dark uniform regions and render noise
    defeated phase correlation (it locked on zero shift)."""
    def down(x):
        h, w = (x.shape[0] // pool) * pool, (x.shape[1] // pool) * pool
        return x[:h, :w].reshape(h // pool, pool, w // pool, pool).mean(axis=(1, 3))
    a, b, m = down(a), down(b), down(~ignore) > 0.5
    r = max_px // pool
    best, arg = np.inf, (0, 0)
    for dv in range(-r, r + 1):
        for du in range(-r, r + 1):
            sa = a[max(0, -dv):a.shape[0] - max(0, dv), max(0, -du):a.shape[1] - max(0, du)]
            sb = b[max(0, dv):b.shape[0] - max(0, -dv), max(0, du):b.shape[1] - max(0, -du)]
            sm = m[max(0, -dv):m.shape[0] - max(0, dv), max(0, -du):m.shape[1] - max(0, du)]
            if sm.sum() < 0.3 * m.size:
                continue
            err = float(((sa - sb) ** 2)[sm].mean())
            if err < best:
                best, arg = err, (du, dv)
    return np.array(arg, dtype=np.float64) * pool


def agrees(vec, expect):
    """Expected component has the right sign AND dominates the other one."""
    ax = 0 if expect[0] else 1
    other = 1 - ax
    return bool(np.sign(vec[ax]) == np.sign(expect[ax]) and abs(vec[ax]) > abs(vec[other]))


def main():
    cfg = yaml.safe_load(open(os.path.join(REPO, args.robot_config)))
    contract = camera_contract(cfg)
    be = So101WorkshopBackend.make(cfg["task"], seed=args.seed, fixed_look=cfg.get("fixed_look", False),
                                   preplaced_vial=cfg.get("preplaced_vial", True), **contract)
    be.reset(args.seed)
    ex = So101AtomicExec(be)
    home_task = np.asarray(args.home) - be.robot.data.root_pos_w[0].cpu().numpy()
    ok = ex.goto_pose(home_task, np.radians(args.pitch_deg), roll=ROLL_CLOSING_HORIZONTAL)
    ex.quiesce()
    if not ok:
        print("[check_views] could not reach the start pose; pick another --home", flush=True)
        return 1
    raw_av, raw_wr = be.raw_frames()
    self_check(raw_av.shape[:2], be.agentview_rotation_degrees, be.agentview_flip)
    self_check(raw_wr.shape[:2], be.wrist_rotation_degrees, be.wrist_flip)
    av_name, wr_name = be.agentview_camera, be.wrist_camera
    ka, ca, ra = camera(be, av_name)
    tcp = be.tcp_pos()
    report, failures = {"contract": contract, "tokens": {}}, []
    os.makedirs(args.out, exist_ok=True)
    gray = lambda img: img[..., :3].astype(np.float64).mean(axis=2)  # noqa: E731
    pairs, av_dirs = {}, {}
    for tok, d in ex.move_dirs.items():
        a0, _ = transform_point(project(ka, ca, ra, tcp), raw_av.shape[:2], be.agentview_rotation_degrees, be.agentview_flip)
        a1, _ = transform_point(project(ka, ca, ra, tcp + d * 0.05), raw_av.shape[:2], be.agentview_rotation_degrees, be.agentview_flip)
        av_dirs[tok] = a1 - a0
        before = gray(be.raw_frames()[1])
        ex.move(tok)
        if ex.last_blocked:
            failures.append(f"{tok} refused at the check pose ({ex.last_blocked})")
            continue
        ex.quiesce()
        pairs[tok] = (before, gray(be.raw_frames()[1]))
        ex.move(OPPOSITE[tok])
        ex.quiesce()
    # The fingers ride with the camera: unchanged through every move, and textured (a uniform
    # patch of mat can also be unchanged).
    static = np.logical_and.reduce([np.abs(b_ - a_) < 10.0 for a_, b_ in pairs.values()])
    ref = next(iter(pairs.values()))[0]
    gv, gu = np.gradient(ref)
    fingers = static & (np.hypot(gu, gv) > 4.0)
    # grow the mask over the finger body (its flat interior has no gradient)
    for _ in range(6):
        fingers = fingers | np.roll(fingers, 1, 0) | np.roll(fingers, -1, 0) | np.roll(fingers, 1, 1) | np.roll(fingers, -1, 1)
    fingers &= static
    for tok, (before, after) in pairs.items():
        Image.fromarray(np.concatenate([before, after], axis=1).astype(np.uint8)).save(
            os.path.join(args.out, f"wrist_pair_{tok}.png"))
        shift = scene_shift(before, after, fingers)
        c = np.array([raw_wr.shape[1] / 2, raw_wr.shape[0] / 2])
        w0, _ = transform_point(c, raw_wr.shape[:2], be.wrist_rotation_degrees, be.wrist_flip)
        w1, _ = transform_point(c - shift, raw_wr.shape[:2], be.wrist_rotation_degrees, be.wrist_flip)
        av_dir, wr_dir = av_dirs[tok], w1 - w0
        row = {"agentview_dir_uv": np.round(av_dir, 1).tolist(), "wrist_heading_uv": np.round(wr_dir, 1).tolist(),
               "wrist_raw_scene_shift_uv": np.round(shift, 1).tolist()}
        if tok in EXPECT_AGENTVIEW:
            row["agentview_ok"] = agrees(av_dir, EXPECT_AGENTVIEW[tok])
            row["wrist_ok"] = agrees(wr_dir, EXPECT_WRIST[tok])
            failures += [f"{tok} agentview"] * (not row["agentview_ok"]) + [f"{tok} wrist"] * (not row["wrist_ok"])
        report["tokens"][tok] = row
    vs, us = np.nonzero(fingers)
    if len(vs) < 200:
        failures.append(f"wrist: fingers not found (only {len(vs)} static textured pixels)")
    else:
        tip, (h, _) = transform_point((us.mean(), vs.mean()), raw_wr.shape[:2], be.wrist_rotation_degrees, be.wrist_flip)
        report["wrist_finger_pixels"] = int(len(vs))
        report["wrist_fingers_centroid_raw_uv"] = [round(float(us.mean()), 1), round(float(vs.mean()), 1)]
        report["wrist_fingers_centroid_uv"] = np.round(tip, 1).tolist()
        report["fingertips_at_top"] = bool(tip[1] < h / 2)
        if not report["fingertips_at_top"]:
            failures.append("wrist fingertips not at the top")
    Image.fromarray((fingers * 255).astype(np.uint8)).save(os.path.join(args.out, "wrist_fingers_mask_raw.png"))
    report["failures"] = failures
    os.makedirs(args.out, exist_ok=True)
    av, wr = be.grab_frames()
    Image.fromarray(np.concatenate([av, wr], axis=1)).save(os.path.join(args.out, "views_contract.png"))
    Image.fromarray(np.concatenate([raw_av, raw_wr], axis=1)).save(os.path.join(args.out, "views_raw.png"))
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)
    print(f"[check_views] {'PASS' if not failures else 'FAIL: ' + ', '.join(failures)}", flush=True)
    be.close()
    return 0 if not failures else 1


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)  # Kit's shutdown can hang inside the container; the results are on disk
