"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""
"""
excution example:

python scripts/rsl_rl/play.py \
    --task Tracking-Flat-G1-v0 \
    --num_envs 2 \
    --wandb_path joony/tracking_g1/0pms9g8m/model_best.pt \
    --motion_file /home/user/rl_ws/src/whole_body_tracking/artifacts/aiming2_subject2/motion.npz

"""


import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
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
#   - Enabled when task name contains "g1" (case-insensitive)
#   - Fixes: ValueError "No rigid bodies are present under this prim"
# =============================================================================
def _should_enable_g1_patch(task_name: str | None) -> bool:
    if not task_name:
        return False
    t = task_name.lower()
    return "g1" in t


ENABLE_G1_PATCH = _should_enable_g1_patch(args_cli.task)
print(f"[G1 PATCH] task='{args_cli.task}', ENABLE={ENABLE_G1_PATCH}")

if ENABLE_G1_PATCH:
    try:
        import omni.usd
        from pxr import Usd, UsdPhysics, PhysxSchema
        from isaaclab.sim.schemas import schemas as _schemas

        _orig_activate_contact_sensors = _schemas.activate_contact_sensors

        def _has_rigidbody_api(prim: Usd.Prim) -> bool:
            # UsdPhysics / PhysxSchema 둘 다 검사
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
            # wrapper 객체 방식도 한번 더 (버전별/에셋별 방어)
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
            # instance proxy면 prototype prim도 함께 검사 (G1에서 자주 걸림)
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
            # instance proxies 포함 traversal
            try:
                return Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            except Exception:
                return Usd.PrimRange(root)

        def _activate_contact_sensors_instance_safe(root_prim_path: str):
            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath(root_prim_path)
            if not root or not root.IsValid():
                return _orig_activate_contact_sensors(root_prim_path)

            # root가 instance면 prototype에 author
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

                    # enabled attr
                    try:
                        attr = api.GetEnabledAttr()
                        if attr:
                            attr.Set(True)
                        else:
                            api.CreateEnabledAttr(True)
                    except Exception:
                        pass

                    # threshold attr (0.0 -> report all)
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
                # 폴백 → 기존 에러/동작 유지 (디버깅에 필요)
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
import os
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

# =============================================================================
# [PLAY HOTFIX] PPO signature mismatch (runner passes lcp_cfg/bound_loss_cfg)
#   - Some runner versions always pass lcp_cfg/bound_loss_cfg to PPO(...)
#   - Some PPO implementations don't accept those kwargs -> TypeError
#   - Patch PPO.__init__ to ignore them (play-only).
# =============================================================================
import inspect

try:
    import rsl_rl.algorithms.ppo as _ppo_mod

    _sig = inspect.signature(_ppo_mod.PPO.__init__)
    if "lcp_cfg" not in _sig.parameters:
        _orig_ppo_init = _ppo_mod.PPO.__init__

        def _patched_ppo_init(self, *args, lcp_cfg=None, bound_loss_cfg=None, **kwargs):
            # Ignore lcp_cfg / bound_loss_cfg for compatibility
            return _orig_ppo_init(self, *args, **kwargs)

        _ppo_mod.PPO.__init__ = _patched_ppo_init
        print("[PLAY HOTFIX] Patched rsl_rl.algorithms.ppo.PPO.__init__ to ignore lcp_cfg/bound_loss_cfg")
    else:
        print("[PLAY HOTFIX] PPO already supports lcp_cfg (no patch needed).")

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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    resume_path = None

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])

        wandb_run = api.run(run_path)

        # loop over files in the run
        files = [f.name for f in wandb_run.files() if "model" in f.name]

        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            # 1) prefer model_best.pt if exists
            if "model_best.pt" in files:
                file = "model_best.pt"
            else:
                # 2) otherwise pick largest model_####.pt
                numeric = []
                for x in files:
                    if x.startswith("model_") and x.endswith(".pt"):
                        token = x.split("_", 1)[1].split(".", 1)[0]  # part after 'model_'
                        if token.isdigit():
                            numeric.append((int(token), x))
                if not numeric:
                    raise RuntimeError(f"No numeric model_####.pt found in run files: {files}")
                file = max(numeric, key=lambda t: t[0])[1]

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        # (1) Highest priority: CLI motion_file
        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file
        else:
            # (2) Try to load from W&B artifact (used or logged)
            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            if art is None:
                art = next((a for a in wandb_run.logged_artifacts() if a.type == "motions"), None)

            if art is None:
                print("[WARN] No motions artifact found in the run. You must pass --motion_file.")
            else:
                # Expect motion.npz inside artifact
                motion_path = pathlib.Path(art.download()) / "motion.npz"
                env_cfg.commands.motion.motion_file = str(motion_path)
                print(f"[INFO]: Using motion file from W&B artifact: {env_cfg.commands.motion.motion_file}")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

        # if user provided motion_file, apply it
        if args_cli.motion_file is not None:
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

    if resume_path is None:
        raise RuntimeError("resume_path is None. Provide --wandb_path or configure local checkpoint loading.")

    # create isaac environment
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

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    cfg = agent_cfg.to_dict()

    # rsl_rl 최신쪽은 obs_groups 키가 없으면 KeyError로 죽음
    # => 없으면 기본값으로 넣어준다.
    cfg.setdefault("obs_groups", {})   # 또는 None을 원하면 cfg.setdefault("obs_groups", None)

    ppo_runner = OnPolicyRunner(env, cfg, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)

# --- normalizer may not exist depending on rsl_rl version ---
    normalizer = getattr(ppo_runner, "obs_normalizer", None)

    # some versions keep it inside the algorithm object
    if normalizer is None:
        normalizer = getattr(getattr(ppo_runner, "alg", None), "obs_normalizer", None)

    # if still None, export without normalizer (exporter should handle None)
    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )

    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
