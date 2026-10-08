"""Privileged geometry of the Workshop vials -> rack scene, for scripted generators and the
motion check ONLY (never given to a policy).

Everything is returned in the TASK frame (world orientation, origin at the robot base). The
rack holes (``Rack_Left/Body1/Mesh/top_*``) are read once from USD relative to the rack, then
composed with the rack's CURRENT pose: with Fabric on, USD child prims are not updated, so
reading them directly returns stale poses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from interpreters.so101_kinematics import quat_wxyz_to_mat
from scripts.trajectory.real2sim.atomic_tokenizer import to_np

VIALS = ("vial_1", "vial_2", "vial_3")
RACK = "rack_left"
RACK_PRIM = "/World/envs/env_0/Rack_Left"


def _tf(pos, rot) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


@dataclass
class Vial:
    name: str
    pos: np.ndarray      # geometric centre (on the axis); NOT the root, which is ~17 mm from the cone end
    rot: np.ndarray      # 3x3
    axis: np.ndarray     # long-axis unit vector (local z, + toward the cap)
    root: np.ndarray     # root origin (near the cone tip)

    @property
    def up_z(self) -> float:
        """|vertical component of the axis|: 1 = standing, 0 = lying."""
        return float(abs(self.axis[2]))


class VialsRackScene:
    def __init__(self, backend) -> None:
        from pxr import Usd, UsdGeom

        self.be = backend
        self.u = backend.u
        stage = self.u.sim.stage
        xc = UsdGeom.XformCache()
        rack_prim = stage.GetPrimAtPath(RACK_PRIM)
        rack_w = np.array(xc.GetLocalToWorldTransform(rack_prim)).T
        self.slot_local = []
        for p in Usd.PrimRange(rack_prim):
            if p.GetName().startswith("top_"):
                slot_w = np.array(xc.GetLocalToWorldTransform(p)).T
                self.slot_local.append(np.linalg.inv(rack_w) @ slot_w)
        if not self.slot_local:
            raise RuntimeError("rack holes (top_*) not found")
        # Table top = the mat's top face. The mat is static and DR only yaws it, so its USD
        # world bound is right (estimating from the vials is a few mm off).
        mb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False) \
            .ComputeWorldBound(stage.GetPrimAtPath("/World/envs/env_0/Mat")).ComputeAlignedRange()
        self.table_top = float(mb.GetMax()[2]) - float(self._base_w()[2])
        rb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False) \
            .ComputeUntransformedBound(rack_prim).ComputeAlignedRange()
        self.rack_local_box = np.array([list(rb.GetMin()), list(rb.GetMax())])
        # Size and geometric centre come from the LOCAL bound: with Fabric on, USD does not
        # follow physics, so world bounds keep the pre-randomisation pose (measured: all three
        # vials reported the same x range).
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False)
        r = cache.ComputeUntransformedBound(stage.GetPrimAtPath("/World/envs/env_0/Vial_1")).ComputeAlignedRange()
        lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
        ext = hi - lo
        self.vial_center_local = (lo + hi) / 2           # root -> geometric centre (local)
        self.vial_diameter = float(np.sort(ext)[0])
        self.vial_length = float(np.sort(ext)[-1])

    def _base_w(self) -> np.ndarray:
        return to_np(self.be.robot.data.root_pos_w[0])

    def vials(self) -> list[Vial]:
        out = []
        for name in VIALS:
            obj = self.u.scene[name]
            rot = quat_wxyz_to_mat(to_np(obj.data.root_quat_w[0]))
            root = to_np(obj.data.root_pos_w[0]) - self._base_w()
            out.append(Vial(name, root + rot @ self.vial_center_local, rot, rot[:, 2], root))
        return out

    def vial(self, name: str) -> Vial:
        return next(v for v in self.vials() if v.name == name)

    def rack_box(self, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """The rack's current bound (task-frame AABB). USD world bounds are stale; not used."""
        rack = self.u.scene[RACK]
        rot = quat_wxyz_to_mat(to_np(rack.data.root_quat_w[0]))
        pos = to_np(rack.data.root_pos_w[0]) - self._base_w()
        corners = np.array([[x, y, z] for x in self.rack_local_box[:, 0] for y in self.rack_local_box[:, 1]
                            for z in self.rack_local_box[:, 2]])
        w = (rot @ corners.T).T + pos
        return w.min(0) - margin, w.max(0) + margin

    def rack_gap(self, pts: np.ndarray) -> float:
        """Exact distance from points (task frame) to the rack as an ORIENTED box.

        The world AABB of a yawed rack (+-0.5 rad randomisation) is much larger than the rack
        and flagged clear grasps as collisions; testing in the rack's own frame is exact.
        """
        rack = self.u.scene[RACK]
        rot = quat_wxyz_to_mat(to_np(rack.data.root_quat_w[0]))
        pos = to_np(rack.data.root_pos_w[0]) - self._base_w()
        local = (np.asarray(pts) - pos) @ rot  # = rot.T @ (p - pos) per row
        lo, hi = self.rack_local_box
        return float(np.linalg.norm(np.maximum(np.maximum(lo - local, local - hi), 0.0), axis=1).min())

    def slots(self) -> list[np.ndarray]:
        """Centre of each hole's opening (task frame)."""
        rack = self.u.scene[RACK]
        rack_t = _tf(to_np(rack.data.root_pos_w[0]), quat_wxyz_to_mat(to_np(rack.data.root_quat_w[0])))
        return [(rack_t @ loc)[:3, 3] - self._base_w() for loc in self.slot_local]

    def slot_occupied(self, slot: np.ndarray, vials: list[Vial] | None = None) -> bool:
        for v in vials or self.vials():
            if v.up_z > 0.7 and np.linalg.norm((v.pos - slot)[:2]) < 0.02:
                return True
        return False

    def lying_vials(self) -> list[Vial]:
        vs = self.vials()
        slots = self.slots()
        return [v for v in vs if v.up_z < 0.5
                and all(np.linalg.norm((v.pos - s)[:2]) > 0.04 for s in slots)]
