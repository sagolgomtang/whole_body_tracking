"""Script to play a checkpoint if an RL agent from RSL-RL.

Usage examples:

# (A) W&B run path or file (기존 기능 유지)
python scripts/rsl_rl/play.py \
    --task Tracking-Flat-G1-v0 \
    --num_envs 2 \
    --wandb_path joony/tracking_g1/0pms9g8m/model_best.pt \
    --motion_file /home/user/rl_ws/src/whole_body_tracking/artifacts/aiming2_subject2/motion.npz

# (B) LOCAL checkpoint path + export only (권장)
python scripts/rsl_rl/play.py \
    --task Tracking-Flat-G1-v0 \
    --num_envs 1 \
    --wandb_path /home/user/rl_ws/src/whole_body_tracking/logs/rsl_rl/g1_flat/2026-01-01_06-58-10_walk2_subject4/model_best.pt \
    --motion_file /home/user/rl_ws/src/whole_body_tracking/artifacts/walk2_subject4/motion.npz \
    --headless --export_only
"""

import argparse
import os
import pathlib
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O ops.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument(
    "--follow_camera",
    action="store_true",
    default=False,
    help="Follow the robot with the viewer camera (default: free camera).",
)
parser.add_argument(
    "--replay_motion_file",
    type=str,
    default=None,
    help="Path to a motion file to replay as markers in the same sim.",
)
parser.add_argument(
    "--replay_registry_name",
    type=str,
    default=None,
    help="W&B registry name for a motion artifact (e.g. myorg/motions:latest).",
)
parser.add_argument("--replay_env_id", type=int, default=0, help="Env index for replay visualization.")
parser.add_argument("--replay_stride", type=int, default=1, help="Stride for replay motion frames.")
parser.add_argument(
    "--replay_anchor_index",
    type=int,
    default=0,
    help="Body index in motion file used as alignment anchor (default: 0).",
)
parser.add_argument(
    "--replay_motion_is_world",
    action="store_true",
    default=False,
    help="Motion positions are already in world frame (no env_origin offset).",
)
parser.add_argument(
    "--replay_no_yaw_align",
    action="store_true",
    default=False,
    help="Disable yaw alignment when overlaying replay markers.",
)
parser.add_argument(
    "--replay_use_command_time",
    action="store_true",
    default=False,
    help="Use MotionCommand time_steps for replay timing (sync with policy target).",
)
parser.add_argument("--export_only", action="store_true", default=False, help="Only export ONNX and exit.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# =============================================================================
# Debug helper (optional)
# =============================================================================
def dump_stage_robot_debug(prefix="/World/envs"):
    """Dump where robot prims & rigid bodies actually exist (must run inside Isaac Sim python)."""
    from omni.usd import get_context
    from pxr import UsdPhysics

    stage = get_context().get_stage()
    if stage is None:
        print("[DBG] stage is None")
        return

    envs = stage.GetPrimAtPath("/World/envs")
    print("[DBG] /World/envs valid:", envs.IsValid())
    if envs.IsValid():
        for c in envs.GetChildren():
            print("[DBG] /World/envs child:", c.GetPath())

    env0 = stage.GetPrimAtPath("/World/envs/env_0")
    print("[DBG] /World/envs/env_0 valid:", env0.IsValid())
    if env0.IsValid():
        for c in env0.GetChildren():
            print("[DBG] env_0 child:", c.GetPath())

    robot = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    print("[DBG] /World/envs/env_0/Robot valid:", robot.IsValid())

    rb = []
    prefix2 = "/World/envs/env_0"
    for p in stage.Traverse():
        ps = p.GetPath().pathString
        if ps.startswith(prefix2) and p.HasAPI(UsdPhysics.RigidBodyAPI):
            rb.append(ps)

    print("[DBG] RigidBody count under", prefix2, "=", len(rb))
    for s in rb[:40]:
        print("   [DBG] RB:", s)


# =============================================================================
# [G1 ONLY PATCH] Robust activate_contact_sensors for instanced prims + API mismatch
# =============================================================================
def _should_enable_g1_patch(task_name: str | None) -> bool:
    if not task_name:
        return False
    return "g1" in task_name.lower()


ENABLE_G1_PATCH = _should_enable_g1_patch(args_cli.task)
print(f"[G1 PATCH] task='{args_cli.task}', ENABLE={ENABLE_G1_PATCH}")

if ENABLE_G1_PATCH:
    try:
        import omni.usd
        from pxr import Usd, UsdPhysics, PhysxSchema
        from isaaclab.sim.schemas import schemas as _schemas

        _orig_activate_contact_sensors = _schemas.activate_contact_sensors

        def _has_rigidbody_api(prim: Usd.Prim) -> bool:
            try:
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    return True
            except Exception:
                pass
            try:
                if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    return True
            except Exception:
                pass
            try:
                if UsdPhysics.RigidBodyAPI(prim):
                    return True
            except Exception:
                pass
            try:
                if PhysxSchema.PhysxRigidBodyAPI(prim):
                    return True
            except Exception:
                pass
            return False

        def _is_rigid_body_any_api(prim: Usd.Prim) -> bool:
            if _has_rigidbody_api(prim):
                return True
            try:
                if prim.IsInstanceProxy():
                    proto_prim = prim.GetPrimInPrototype()
                    if proto_prim and proto_prim.IsValid():
                        return _has_rigidbody_api(proto_prim)
            except Exception:
                pass
            return False

        def _iter_prims_instance_safe(root: Usd.Prim):
            try:
                return Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            except Exception:
                return Usd.PrimRange(root)

        def _activate_contact_sensors_instance_safe(root_prim_path: str):
            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath(root_prim_path)
            if not root or not root.IsValid():
                return _orig_activate_contact_sensors(root_prim_path)

            root_to_author = root
            try:
                if root.IsInstance():
                    proto = root.GetPrototype()
                    if proto and proto.IsValid():
                        root_to_author = proto
            except Exception:
                pass

            rigid_prims = []
            for prim in _iter_prims_instance_safe(root_to_author):
                if _is_rigid_body_any_api(prim):
                    rigid_prims.append(prim)

            print(
                f"[G1 PATCH] activate_contact_sensors root='{root_prim_path}' "
                f"author_on='{root_to_author.GetPath()}'"
            )
            print(f"[G1 PATCH] rigid bodies found: {len(rigid_prims)}")

            applied = 0
            for prim in rigid_prims:
                prim_to_author = prim
                try:
                    if prim.IsInstanceProxy():
                        prim_to_author = prim.GetPrimInPrototype()
                except Exception:
                    pass

                try:
                    api = PhysxSchema.PhysxContactReportAPI(prim_to_author)
                    if not api:
                        api = PhysxSchema.PhysxContactReportAPI.Apply(prim_to_author)

                    try:
                        attr = api.GetEnabledAttr()
                        if attr:
                            attr.Set(True)
                        else:
                            api.CreateEnabledAttr(True)
                    except Exception:
                        pass

                    try:
                        attr = api.GetThresholdAttr()
                        if attr:
                            attr.Set(0.0)
                        else:
                            api.CreateThresholdAttr(0.0)
                    except Exception:
                        pass

                    applied += 1
                except Exception:
                    pass

            print(f"[G1 PATCH] contact report applied on {applied}/{len(rigid_prims)} rigid bodies.")
            if applied == 0:
                return _orig_activate_contact_sensors(root_prim_path)
            return None

        _schemas.activate_contact_sensors = _activate_contact_sensors_instance_safe
        print("[G1 PATCH] Enabled (schemas.activate_contact_sensors patched).")

    except Exception as e:
        print(f"[G1 PATCH] Skipped (patch failed to install): {e}")
else:
    print("[G1 PATCH] Disabled (non-g1 task).")


"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

# =============================================================================
# [PLAY HOTFIX] PPO signature mismatch
# =============================================================================
import inspect

try:
    import rsl_rl.algorithms.ppo as _ppo_mod

    _sig = inspect.signature(_ppo_mod.PPO.__init__)
    if "lcp_cfg" not in _sig.parameters:
        _orig_ppo_init = _ppo_mod.PPO.__init__

        def _patched_ppo_init(self, *args, lcp_cfg=None, bound_loss_cfg=None, **kwargs):
            return _orig_ppo_init(self, *args, **kwargs)

        _ppo_mod.PPO.__init__ = _patched_ppo_init
        print("[PLAY HOTFIX] Patched PPO.__init__ to ignore lcp_cfg/bound_loss_cfg")
    else:
        print("[PLAY HOTFIX] PPO already supports lcp_cfg")

except Exception as _e:
    print(f"[PLAY HOTFIX] PPO patch skipped: {_e}")


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


def _resolve_replay_motion_file() -> str | None:
    if args_cli.replay_registry_name:
        import pathlib
        import wandb

        registry_name = args_cli.replay_registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_path = pathlib.Path(artifact.download()) / "motion.npz"
        return str(motion_path)

    if args_cli.replay_motion_file:
        return args_cli.replay_motion_file

    return None


def _setup_replay_visualizers(env, motion_file: str):
    import numpy as np
    import torch
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.markers.config import FRAME_MARKER_CFG

    data = np.load(motion_file)
    body_count = int(data["body_pos_w"].shape[1])
    body_indexes = torch.arange(body_count, device=env.device, dtype=torch.long)
    motion = MotionLoader(motion_file, body_indexes, device=env.device)

    replay_body_visualizers = []
    for i in range(body_count):
        marker_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
            prim_path=f"/Visuals/Replay/body_{i}"
        )
        marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        replay_body_visualizers.append(VisualizationMarkers(marker_cfg))

    replay_state = {
        "motion": motion,
        "time_step": 0,
        "body_visualizers": replay_body_visualizers,
        "aligned": False,
        "delta_pos": None,
        "delta_ori": None,
    }
    return replay_state


def _align_replay_to_robot(base_env, replay_state):
    import torch
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat

    motion = replay_state["motion"]
    time_step = 0

    robot = base_env.scene["robot"]
    robot_anchor_body_index = 0
    command = None
    try:
        command = base_env.command_manager.get_term("motion")
        robot_anchor_body_index = int(command.robot_anchor_body_index)
    except Exception:
        pass

    robot_anchor_pos = robot.data.body_pos_w[args_cli.replay_env_id, robot_anchor_body_index]
    robot_anchor_quat = robot.data.body_quat_w[args_cli.replay_env_id, robot_anchor_body_index]

    if args_cli.replay_use_command_time and command is not None:
        time_step = int(command.time_steps[args_cli.replay_env_id].item())

    anchor_index = int(args_cli.replay_anchor_index)
    env_origin = base_env.scene.env_origins[args_cli.replay_env_id]
    anchor_pos = motion.body_pos_w[time_step, anchor_index]
    anchor_quat = motion.body_quat_w[time_step, anchor_index]

    if args_cli.replay_no_yaw_align:
        delta_ori = torch.tensor([1.0, 0.0, 0.0, 0.0], device=robot_anchor_quat.device)
    else:
        delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))

    if args_cli.replay_motion_is_world:
        env_origin = torch.zeros_like(env_origin)

    delta_pos = robot_anchor_pos - (quat_apply(delta_ori, anchor_pos) + env_origin)

    replay_state["time_step"] = time_step
    replay_state["delta_pos"] = delta_pos
    replay_state["delta_ori"] = delta_ori
    replay_state["aligned"] = True


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if not args_cli.follow_camera:
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.asset_name = ""

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = None

    # -----------------------------
    # checkpoint loading
    # -----------------------------
    if args_cli.wandb_path:
        # (1) LOCAL checkpoint path
        if os.path.isfile(args_cli.wandb_path):
            resume_path = args_cli.wandb_path
            print(f"[INFO] Local checkpoint: {resume_path}")

            if args_cli.motion_file is None:
                raise RuntimeError("Local checkpoint mode requires --motion_file.")
            env_cfg.commands.motion.motion_file = args_cli.motion_file
            print(f"[INFO] Using motion file: {env_cfg.commands.motion.motion_file}")

        # (2) W&B path (기존 동작)
        else:
            import wandb

            run_path = args_cli.wandb_path
            api = wandb.Api()

            if "model" in args_cli.wandb_path:
                run_path = "/".join(args_cli.wandb_path.split("/")[:-1])

            wandb_run = api.run(run_path)
            files = [f.name for f in wandb_run.files() if "model" in f.name]

            if "model" in args_cli.wandb_path:
                file = args_cli.wandb_path.split("/")[-1]
            else:
                if "model_best.pt" in files:
                    file = "model_best.pt"
                else:
                    numeric = []
                    for x in files:
                        if x.startswith("model_") and x.endswith(".pt"):
                            token = x.split("_", 1)[1].split(".", 1)[0]
                            if token.isdigit():
                                numeric.append((int(token), x))
                    if not numeric:
                        raise RuntimeError(f"No numeric model_####.pt found in run files: {files}")
                    file = max(numeric, key=lambda t: t[0])[1]

            wandb_run.file(str(file)).download("./logs/rsl_rl/temp", replace=True)
            resume_path = f"./logs/rsl_rl/temp/{file}"
            print(f"[INFO] Loading W&B checkpoint from: {run_path}/{file}")

            if args_cli.motion_file is not None:
                env_cfg.commands.motion.motion_file = args_cli.motion_file
                print(f"[INFO] Using motion file from CLI: {env_cfg.commands.motion.motion_file}")
            else:
                art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
                if art is None:
                    art = next((a for a in wandb_run.logged_artifacts() if a.type == "motions"), None)
                if art is None:
                    raise RuntimeError("No motions artifact found in the run. You must pass --motion_file.")
                motion_path = pathlib.Path(art.download()) / "motion.npz"
                env_cfg.commands.motion.motion_file = str(motion_path)
                print(f"[INFO] Using motion file from W&B artifact: {env_cfg.commands.motion.motion_file}")

    # (3) local experiment directory mode (기존)
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        if args_cli.motion_file is not None:
            env_cfg.commands.motion.motion_file = args_cli.motion_file
            print(f"[INFO] Using motion file from CLI: {env_cfg.commands.motion.motion_file}")

    if resume_path is None:
        raise RuntimeError("resume_path is None.")

    # -----------------------------
    # create environment
    # -----------------------------
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    except Exception as e:
        print("[DBG] gym.make failed:", repr(e))
        dump_stage_robot_debug()
        raise

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent if required
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    cfg = agent_cfg.to_dict()
    cfg.setdefault("obs_groups", {})

    ppo_runner = OnPolicyRunner(env, cfg, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    base_env = env.unwrapped
    replay_state = None
    replay_motion_file = _resolve_replay_motion_file()
    if replay_motion_file is not None:
        if args_cli.replay_env_id >= base_env.scene.num_envs:
            print(
                f"[REPLAY] replay_env_id={args_cli.replay_env_id} out of range "
                f"(num_envs={base_env.scene.num_envs}); skipping replay."
            )
        else:
            print(f"[REPLAY] Loading motion: {replay_motion_file}")
            replay_state = _setup_replay_visualizers(base_env, replay_motion_file)

    # -----------------------------
    # export (NO PLAY)
    # -----------------------------
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    # 가장 중요한 부분: actor_obs_normalizer를 우선 사용
    normalizer = getattr(ppo_runner.alg.policy, "actor_obs_normalizer", None)
    if normalizer is None:
        normalizer = getattr(ppo_runner, "obs_normalizer", None)
    if normalizer is None:
        normalizer = getattr(getattr(ppo_runner, "alg", None), "obs_normalizer", None)

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=normalizer,
        path=export_model_dir,
        filename="policy_best.onnx",
    )
    attach_onnx_metadata(
        env.unwrapped,
        args_cli.wandb_path if args_cli.wandb_path else "none",
        export_model_dir,
        filename="policy_best.onnx",
    )
    print(f"[INFO] Exported ONNX to: {os.path.join(export_model_dir, 'policy.onnx')}")

    if args_cli.export_only:
        print("[INFO] export_only enabled. Exiting without stepping.")
        env.close()
        simulation_app.close()
        return

    # -----------------------------
    # play (optional)
    # -----------------------------
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    obs, _ = env.get_observations()
    if replay_state is not None and not replay_state["aligned"]:
        _align_replay_to_robot(base_env, replay_state)
    timestep = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        try:
            command = base_env.command_manager.get_term("motion")
            err = command.anchor_pos_w - command.robot_anchor_pos_w
            err_vec = err[args_cli.replay_env_id].detach().cpu().numpy().tolist()
            print(f"[ANCHOR_ERR] env={args_cli.replay_env_id} vec={err_vec}")
        except Exception as exc:
            print(f"[ANCHOR_ERR] failed to read motion command: {exc}")

        if replay_state is not None:
            from isaaclab.utils.math import quat_apply, quat_mul

            if not replay_state["aligned"]:
                _align_replay_to_robot(base_env, replay_state)

            motion = replay_state["motion"]
            command = None
            if args_cli.replay_use_command_time:
                try:
                    command = base_env.command_manager.get_term("motion")
                except Exception:
                    command = None

            if args_cli.replay_use_command_time and command is not None:
                t = int(command.time_steps[args_cli.replay_env_id].item())
                replay_state["time_step"] = t
            else:
                t = replay_state["time_step"]
            delta_pos = replay_state["delta_pos"]
            delta_ori = replay_state["delta_ori"]
            env_origin = base_env.scene.env_origins[args_cli.replay_env_id]
            if args_cli.replay_motion_is_world:
                env_origin = torch.zeros_like(env_origin)
            pos = motion.body_pos_w[t]
            delta_ori_b = delta_ori.unsqueeze(0).expand_as(motion.body_quat_w[t])
            pos = delta_pos + quat_apply(delta_ori_b, pos) + env_origin
            quat = quat_mul(delta_ori_b, motion.body_quat_w[t])
            for i, vis in enumerate(replay_state["body_visualizers"]):
                vis.visualize(pos[i].unsqueeze(0), quat[i].unsqueeze(0))
            if not args_cli.replay_use_command_time:
                replay_state["time_step"] = (
                    replay_state["time_step"] + max(int(args_cli.replay_stride), 1)
                ) % motion.time_step_total

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
