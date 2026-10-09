"""Extract the SO-101 kinematic chain from the Workshop USD and sample FK ground truth.

Runs inside the Workshop image (run_in_docker.sh). Writes to --out:
  so101_chain.json  - every UsdPhysics revolute/fixed joint of the robot: body0/body1,
                      localPos0/localRot0/localPos1/localRot1, axis, limits
  fk_samples.json   - random joint configurations (within limits) and the resulting body
                      poses in the robot ROOT frame, read back from PhysX
  gripper_geom.json - local-frame bounding boxes of the gripper / jaw bodies (to place the TCP)

The numpy FK in interpreters/so101_kinematics.py is built from so101_chain.json (copied to
configs/) and checked against fk_samples.json (every 5th sample is tests/fixtures/
so101_fk_samples.json), so the IK used in sim and on the real arm is the same model.
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack")
parser.add_argument("--samples", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="/workspace/out/kinematics")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import sim_to_real_so101.tasks  # noqa: E402,F401


def vec(v):
    return [float(x) for x in v]


def quat_wxyz(q):
    return [float(q.GetReal())] + [float(x) for x in q.GetImaginary()]


def main():
    os.makedirs(args.out, exist_ok=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    stage = u.sim.stage
    robot_root = "/World/envs/env_0/Robot"

    # ---- chain from USD
    joints = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_root)):
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.FixedJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            j = UsdPhysics.Joint(prim)
            b0 = j.GetBody0Rel().GetTargets()
            b1 = j.GetBody1Rel().GetTargets()
            rec = {
                "path": str(prim.GetPath()),
                "name": prim.GetName(),
                "type": prim.GetTypeName(),
                "body0": str(b0[0]) if b0 else None,
                "body1": str(b1[0]) if b1 else None,
                "localPos0": vec(j.GetLocalPos0Attr().Get()),
                "localRot0_wxyz": quat_wxyz(j.GetLocalRot0Attr().Get()),
                "localPos1": vec(j.GetLocalPos1Attr().Get()),
                "localRot1_wxyz": quat_wxyz(j.GetLocalRot1Attr().Get()),
            }
            if prim.IsA(UsdPhysics.RevoluteJoint):
                r = UsdPhysics.RevoluteJoint(prim)
                rec["axis"] = str(r.GetAxisAttr().Get())
                rec["lower_deg"] = r.GetLowerLimitAttr().Get()
                rec["upper_deg"] = r.GetUpperLimitAttr().Get()
            joints.append(rec)

    # body world scale check: FK assumes unit scale on every link
    xcache = UsdGeom.XformCache()
    bodies = {}
    for name in robot.body_names:
        path = f"{robot_root}/{name}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        m = xcache.GetLocalToWorldTransform(prim)
        scale = [float(np.linalg.norm([m[i][0], m[i][1], m[i][2]])) for i in range(3)]
        bodies[name] = {"path": path, "world_scale": scale}

    chain = {
        "source": ("UsdPhysics joints of the SO-101 USD in NVIDIA Sim-to-Real-SO-101-Workshop v1.0 "
                   "(Apache-2.0, tag v1.0 = d327be4), extracted by "
                   "scripts/trajectory/real2sim/so101_workshop/dump_kinematics.py"),
        "robot_root": robot_root,
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "joints": joints,
        "bodies": bodies,
        "soft_joint_pos_limits": robot.data.soft_joint_pos_limits[0].tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
    }
    with open(os.path.join(args.out, "so101_chain.json"), "w") as f:
        json.dump(chain, f, indent=2)

    # ---- gripper geometry (local bbox of each link's subtree, in that link's frame)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
    geom = {}
    for name in robot.body_names:
        if not any(k in name.lower() for k in ("gripper", "jaw", "wrist")):
            continue
        prim = stage.GetPrimAtPath(f"{robot_root}/{name}")
        if not prim.IsValid():
            continue
        box = bbox_cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        geom[name] = {"min": vec(box.GetMin()), "max": vec(box.GetMax())}
    with open(os.path.join(args.out, "gripper_geom.json"), "w") as f:
        json.dump(geom, f, indent=2)

    # ---- FK samples straight from PhysX (no stepping: write state, forward, read)
    env.reset()
    rng = np.random.default_rng(args.seed)
    lim = robot.data.soft_joint_pos_limits[0].cpu().numpy()
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_quat = robot.data.root_quat_w[0].cpu().numpy()  # wxyz
    samples = []
    for _ in range(args.samples):
        q = rng.uniform(lim[:, 0], lim[:, 1]).astype(np.float32)
        qt = torch.tensor(q, device=u.device).unsqueeze(0)
        robot.write_joint_state_to_sim(qt, torch.zeros_like(qt))
        u.sim.forward()
        tf = robot.root_physx_view.get_link_transforms()[0].cpu().numpy()  # (n_links, 7) pos + quat xyzw
        samples.append({"q": q.tolist(), "link_tf_world_xyzw": tf.tolist()})
    with open(os.path.join(args.out, "fk_samples.json"), "w") as f:
        json.dump({
            "root_pos_w": root_pos.tolist(),
            "root_quat_w_wxyz": root_quat.tolist(),
            "link_names": list(robot.root_physx_view.shared_metatype.link_names),
            "samples": samples,
        }, f)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        os._exit(0)  # Kit's shutdown can hang inside the container; the files are on disk
