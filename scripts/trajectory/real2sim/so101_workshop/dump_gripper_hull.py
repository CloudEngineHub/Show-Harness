"""Extract the gripper and moving-jaw COLLISION shapes from the Workshop USD, as convex-hull
vertices in each link's frame.

The controller's floor check uses them to keep the WHOLE hand above the table, not just the
TCP. Writes --out/so101_gripper_hull.json in the format of configs/so101_gripper_hull.json
({"source", "gripper": [[x, y, z], ...], "jaw": [...]}) and a report next to it.
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Lerobot-So101-Teleop-Vials-To-Rack")
parser.add_argument("--out", default="/workspace/out/kinematics")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

from scripts.trajectory.real2sim.backends.so101_workshop import make_env  # noqa: E402


def main():
    os.makedirs(args.out, exist_ok=True)
    env = make_env(args.task)
    stage = env.unwrapped.sim.stage
    root = "/World/envs/env_0/Robot"
    xc = UsdGeom.XformCache()
    out, report = {}, {}
    for link in ("gripper", "jaw"):
        link_prim = stage.GetPrimAtPath(f"{root}/{link}")
        to_link = np.array(xc.GetLocalToWorldTransform(link_prim)).T  # USD is row-vector
        to_link = np.linalg.inv(to_link)
        pts, n_meshes = [], 0
        # collision prims are usually instanced: traverse instance proxies too
        colliders = [p for p in Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies())
                     if p.HasAPI(UsdPhysics.CollisionAPI)]
        report[f"{link}_colliders"] = [f"{p.GetPath()} ({p.GetTypeName()})" for p in colliders][:20]
        # CollisionAPI sits on a `collisions` Xform; the meshes are its descendants
        meshes = [m for c in colliders for m in Usd.PrimRange(c, Usd.TraverseInstanceProxies())
                  if m.IsA(UsdGeom.Mesh)]
        for prim in meshes:
            mesh = UsdGeom.Mesh(prim)
            p = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            if p.size == 0:
                continue
            m = np.array(xc.GetLocalToWorldTransform(prim)).T
            pw = (m @ np.c_[p, np.ones(len(p))].T).T
            pts.append((to_link @ pw.T).T[:, :3])
            n_meshes += 1
        if not pts:
            report[link] = "no collision mesh"
            continue
        allp = np.vstack(pts)
        hull = ConvexHull(allp)
        out[link] = allp[hull.vertices].round(5).tolist()
        report[link] = {"meshes": n_meshes, "points": int(len(allp)), "hull_vertices": int(len(hull.vertices)),
                        "min": allp.min(0).round(4).tolist(), "max": allp.max(0).round(4).tolist()}
    source = ("Convex-hull vertices (m, link frame) of the gripper/jaw collision meshes of the SO-101 USD "
              "in NVIDIA Sim-to-Real-SO-101-Workshop v1.0 (Apache-2.0), extracted by "
              "scripts/trajectory/real2sim/so101_workshop/dump_gripper_hull.py")
    with open(os.path.join(args.out, "so101_gripper_hull.json"), "w") as f:
        json.dump({"source": source, **out}, f, separators=(",", ":"))
    with open(os.path.join(args.out, "so101_gripper_hull_report.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    try:
        main()
    finally:
        os._exit(0)  # Kit's shutdown can hang inside the container; the files are on disk
