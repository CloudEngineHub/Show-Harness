"""SO-101 on the NVIDIA Sim-to-Real SO-101 Workshop v1.0 (Isaac Sim 5.1 / Isaac Lab).

https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop, tag ``v1.0`` (Apache-2.0). The env
and its dependency stack live in the Workshop's own Docker image; nothing here is installed
into this repo's environments. Run generators inside that image with
``scripts/trajectory/real2sim/so101_workshop/run_in_docker.sh`` (docs/simulators.md).

Only I/O is here. The motion core is the embodiment interpreter
(``interpreters/so101_atomic_controller.py``), shared with the future real-arm backend, so
what is verified in sim is what runs on hardware. One tick = one ``env.step`` (60 Hz, joint
position targets); one control step = 2 ticks (30 Hz).

Workshop specifics absorbed here:

* **Base frame.** The robot base link is yawed +90 deg in the world. The controller works in
  the TASK frame (world orientation, origin at the base), so MV_* keep the shared meaning.
* **Contact is measured**, on the distal links, with a per-object breakdown (vials, rack). A
  joint-gap heuristic false-fires at full reach where gravity sag and lag are large.
* **Rack contact filter is the rack's rigid body** (``Rack_Left/Body1``): ``Rack_Left`` is
  only its Xform, and filtering on it put every rack contact under "other".
* **Success is the task's own predicate and it is lenient** (see :meth:`success`).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from interpreters.so101_atomic_controller import So101ArmController
from interpreters.so101_kinematics import quat_wxyz_to_mat
from scripts.trajectory.real2sim.atomic_tokenizer import AtomicSimEnv, prepared_pair, to_np

DEFAULT_TASK = "Lerobot-So101-Teleop-Vials-To-Rack-DR"


# Links whose contacts mean "the hand/forearm touched something". The base sits on the table
# and the shoulder/upper arm never come near the scene in this task.
CONTACT_LINKS = ("lower_arm", "wrist", "gripper", "jaw")
# Filters must name the RIGID BODY prim: for the rack that is Rack_Left/Body1 (Rack_Left is
# only its Xform). With "Rack_Left" every rack contact landed in "other" (measured 2-3 N).
CONTACT_FILTERS = ("Vial_1", "Vial_2", "Vial_3", "Rack_Left/Body1")

ROBOT_PRIM = "/World/envs/env_0/Robot"
# The jaw parts (what grips). The fixed finger is one mesh with the gripper body, so the body
# turns black too. Bindings sit on each mesh's PARENT Xform with strongerThanDescendants, so the
# override goes there (bound on the mesh itself, the parent still won -- measured).
JAW_MESHES = ("gripper/visuals/wrist_roll_follower_so101_v1", "jaw/visuals/moving_jaw_so101_v1")
ARM_WHITE = (0.95, 0.95, 0.95)  # the Workshop's ROBOT_COLORS["white"] / ["black"]
JAW_BLACK = (0.08, 0.08, 0.08)


def apply_fixed_look(env, env_ids=None) -> None:
    """Arm white, jaws black (make_env removes the DR colour swap on every reset).

    Must run BEFORE the sim starts (an EventTerm with mode="prestartup"): USD edits made after
    start never reached the renderer for the instanced parts (measured: the arm stayed the
    USD's default yellow).

    All printed parts share one material (material_a_3d_printed), so the jaws get a copy of it,
    turned black. Their meshes are instanced and cannot be edited, so only those links'
    visuals are de-instanced (looks only; collision shapes and physics are untouched).
    """
    from pxr import Gf, Sdf, UsdShade

    stage = env.sim.stage
    src_mat = stage.GetPrimAtPath(f"{ROBOT_PRIM}/Looks/material_a_3d_printed")
    src = src_mat.GetChild("Shader")
    src.GetAttribute("inputs:diffuse_color_constant").Set(Gf.Vec3f(*ARM_WHITE))
    # copy the MDL source and authored inputs (a spec in a referenced layer cannot be copied)
    mat = UsdShade.Material.Define(stage, f"{ROBOT_PRIM}/Looks/material_jaw_black")
    sh = UsdShade.Shader.Define(stage, mat.GetPath().AppendChild("Shader"))
    for a in src.GetAttributes():
        if a.HasAuthoredValue():
            sh.GetPrim().CreateAttribute(a.GetName(), a.GetTypeName(), custom=False).Set(a.Get())
    sh.GetPrim().GetAttribute("inputs:diffuse_color_constant").Set(Gf.Vec3f(*JAW_BLACK))
    for out in ("surface", "displacement", "volume"):  # the MDL material's outputs:mdl:*
        mat.CreateOutput(f"mdl:{out}", Sdf.ValueTypeNames.Token).ConnectToSource(sh.ConnectableAPI(), "out")
    with Sdf.ChangeBlock():
        for m in JAW_MESHES:
            stage.GetPrimAtPath(f"{ROBOT_PRIM}/{m.split('/')[0]}/visuals").SetInstanceable(False)
    for m in JAW_MESHES:
        prim = stage.GetPrimAtPath(f"{ROBOT_PRIM}/{m}")
        if not prim.IsValid():
            raise RuntimeError(f"jaw part not found: {m}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def make_env(task: str = DEFAULT_TASK, seed: int = 0, device: str = "cuda:0",
             episode_length_s: float = 3600.0, hand_contact: bool = True,
             fixed_look: bool = False, preplaced_vial: bool = True):
    """Create the Workshop env for token collection: no time-out, no auto-reset.

    The non-Eval task variants already have ``terminations = None``; the long episode length
    only keeps ``episode_length_buf`` from ever mattering. Success is read from the
    ``subtask_terms`` observations instead of a termination.

    fixed_look: arm white, jaws black, every episode (drops the DR colour swap; lighting, mat
    rotation and the other DR terms stay).
    preplaced_vial: False stops the reset from pre-inserting one vial in the rack (1 in 3
    episodes by default), so three vials always lie on the mat.
    """
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import sim_to_real_so101.tasks  # noqa: F401  (registers the gym ids)

    cfg = parse_env_cfg(task, device=device, num_envs=1)
    cfg.seed = seed
    if hand_contact:
        # Measured contact on the arm's distal links, with per-object breakdown, so contact is
        # OBSERVED instead of inferred from the joint gap (which false-fires at full reach).
        from isaaclab.sensors import ContactSensorCfg

        cfg.scene.contact_hand = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(CONTACT_LINKS) + ")",
            update_period=0.0, history_length=1, debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/" + n for n in CONTACT_FILTERS],
        )
    cfg.episode_length_s = episode_length_s
    if getattr(cfg, "terminations", None) is not None:
        raise ValueError(f"{task} has terminations (Eval variant?); use a non-Eval task id")
    if fixed_look:
        from isaaclab.managers import EventTermCfg

        if hasattr(cfg.events, "reset_set_robot_visual_material"):
            cfg.events.reset_set_robot_visual_material = None
        cfg.events.fixed_look = EventTermCfg(func=apply_fixed_look, mode="prestartup")
        cfg.scene.replicate_physics = False  # required by prestartup events; 1 env, no effect
    if not preplaced_vial:
        cfg.events.reset_vials_setup.params["rack_placement_prob"] = 0.0
    return gym.make(task, cfg=cfg)


class So101WorkshopBackend(So101ArmController, AtomicSimEnv):
    """``AtomicSimEnv`` for the Workshop SO-101. Motion, feasibility and refusal come from
    :class:`~interpreters.so101_atomic_controller.So101ArmController`; this class adds only the
    simulator I/O (joints in, joint targets out, cameras, contact sensor, success).

    Execute tokens with :class:`~interpreters.so101_atomic_controller.So101AtomicExec`, not the
    generic ``AtomicExec``: ``supports_rotation`` is False because RT_* on this arm is not an
    arbitrary world-axis rotation (see configs/primitives_so101.yaml).
    """

    # Fallbacks ONLY: the camera contract a dataset is built with comes from
    # configs/robot_so101_workshop.yaml via core.config.camera_contract (see that file).
    agentview_camera = "external_D455"
    wrist_camera = "ego"
    agentview_rotation_degrees = 90
    agentview_flip = "none"
    agentview_square_size = 256
    wrist_rotation_degrees = 180
    wrist_flip = "none"
    wrist_square_size = 256

    def __init__(self, env, **kw) -> None:
        for key in ("agentview_camera", "wrist_camera", "agentview_rotation_degrees", "agentview_flip", "agentview_crop_aspect",
                    "agentview_square_size", "wrist_rotation_degrees", "wrist_flip",
                    "wrist_crop_aspect", "wrist_square_size"):
            if key in kw:
                setattr(self, key, kw.pop(key))
        self.env = env
        self.u = env.unwrapped
        self.robot = self.u.scene["robot"]
        kw.setdefault("tick_dt", float(self.u.step_dt))
        super().__init__(**kw)
        self._obs: Any = None
        self.contact_is_measured = "contact_hand" in self.u.scene.keys()

    # ------------------------------------------------------------ I/O
    def _read_q(self) -> np.ndarray:
        return to_np(self._obs["policy"]["joint_pos_obs"][0])

    def _send(self, q_cmd: np.ndarray) -> None:
        import torch

        action = torch.tensor(q_cmd, dtype=torch.float32, device=self.u.device).unsqueeze(0)
        self._obs, *_ = self.env.step(action)

    def _r_world_base(self) -> np.ndarray:
        return quat_wxyz_to_mat(to_np(self.robot.data.root_quat_w[0]))

    # ------------------------------------------------------------ measured contact
    def contact_report(self, min_force_n: float = 0.5) -> list[dict]:
        """Links touching something, with the net force and what it is pressing on."""
        if "contact_hand" not in self.u.scene.keys():
            return []
        sensor = self.u.scene["contact_hand"]
        net = to_np(sensor.data.net_forces_w[0])            # (links, 3)
        mat = sensor.data.force_matrix_w
        mat = None if mat is None else to_np(mat[0])        # (links, filters, 3)
        out = []
        for i, link in enumerate(sensor.body_names):
            f = float(np.linalg.norm(net[i]))
            if f < min_force_n:
                continue
            with_ = {}
            explained = np.zeros(3)
            if mat is not None:
                for j, name in enumerate(CONTACT_FILTERS):
                    fj = float(np.linalg.norm(mat[i, j]))
                    explained += mat[i, j]
                    if fj >= min_force_n:
                        with_[name] = round(fj, 2)
            # unexplained = the VECTOR residual (|net| - sum|f_j| is not a force: it read 2-3 N
            # of phantom "other" contact whenever the grip forces were not collinear)
            other = float(np.linalg.norm(net[i] - explained))
            if other >= min_force_n:
                with_["other"] = round(other, 2)
            out.append({"link": link, "force_n": round(f, 2), "with": with_})
        return out

    def in_contact(self, threshold_n: float = 2.0) -> bool:
        """Measured: any distal link pressed with more than ``threshold_n`` against anything
        except a vial held between closed fingers (that force is the grasp itself)."""
        if "contact_hand" not in self.u.scene.keys():
            return super().in_contact()
        closed = self.q_des[5] < (self.jaw_open_rad + self.jaw_close_rad) / 2
        for c in self.contact_report(threshold_n):
            w = dict(c["with"])
            if closed and c["link"] in ("gripper", "jaw"):
                for name in ("Vial_1", "Vial_2", "Vial_3"):
                    w.pop(name, None)
            if any(f >= threshold_n for f in w.values()):
                self.last_contact = c
                return True
        return False

    # ------------------------------------------------------------ task / cameras
    def success(self) -> bool:
        """The task's own ``vial_placed`` term. It is LENIENT and latched: it fires at the frame
        the fingers open if ANY vial that was just held is within 45 deg of vertical and inside
        the rack's footprint, and stays set. It does not check that the vial went into a hole,
        that it stays upright, or which vial it was -- review the videos before trusting it."""
        return bool(to_np(self._obs["subtask_terms"]["vial_placed"]).reshape(-1)[0] > 0.5)

    def grasped(self) -> bool:
        return bool(to_np(self._obs["subtask_terms"]["vial_grasped"]).reshape(-1)[0] > 0.5)

    def raw_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """(agentview, wrist) uint8 HWC 480x640, untransformed (the Workshop's ``rgb_<camera>``)."""
        vis = self._obs["visual"]
        return (to_np(vis[f"rgb_{self.agentview_camera}"][0])[..., :3],
                to_np(vis[f"rgb_{self.wrist_camera}"][0])[..., :3])

    def grab_frames(self) -> tuple[np.ndarray, np.ndarray]:
        return prepared_pair(self, *self.raw_frames())

    # ------------------------------------------------------------ lifecycle
    @classmethod
    def make(cls, task: str = DEFAULT_TASK, seed: int = 0, fixed_look: bool = False,
             preplaced_vial: bool = True, **kw) -> "So101WorkshopBackend":
        """Build the env and the backend. Isaac Sim must already be running (``AppLauncher``
        with ``enable_cameras``), as for the RoboLab backend."""
        return cls(make_env(task, seed=seed, fixed_look=fixed_look, preplaced_vial=preplaced_vial), **kw)

    def reset(self, seed: int, settle_ctrl_steps: int = 30) -> None:
        self._obs, _ = self.env.reset(seed=seed)
        self._begin_episode(settle_ctrl_steps)

    def close(self) -> None:
        self.env.close()
