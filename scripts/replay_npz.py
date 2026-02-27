"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python scripts/replay_npz.py --motion_file /home/user/rl_ws/src/whole_body_tracking/artifacts/run1_subject2/motion_origin.npz
    python scripts/replay_npz.py --motion_file /home/user/rl_ws/src/whole_body_tracking/artifacts/run1_subject2/motion.npz \\
        --xy_scale 0.5 --zero_origin --anchor_index 0
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

"""

"""

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb registry.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to a local motion.npz file.")
parser.add_argument("--xy_scale", type=float, default=1.0, help="Scale XY positions (and lin vel) by this factor.")
parser.add_argument("--zero_origin", action="store_true", default=False, help="Subtract initial anchor XY to zero.")
parser.add_argument("--anchor_index", type=int, default=0, help="Anchor body index for zero_origin (default: 0).")
parser.add_argument("--pos_scale", type=float, default=1.0, help="Scale positions (and lin vel) by this factor.")
parser.add_argument(
    "--pos_scale_xy_only",
    action="store_true",
    default=False,
    help="If set, only scale XY of positions (and lin vel).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=G1_CYLINDER_CFG.spawn.replace(activate_contact_sensors=False),
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    if args_cli.motion_file:
        motion_file = args_cli.motion_file
    else:
        if args_cli.registry_name is None:
            raise RuntimeError("Either --motion_file or --registry_name must be provided.")
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib
        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    anchor_xy0 = None
    if args_cli.zero_origin:
        if not (0 <= args_cli.anchor_index < motion.body_pos_w.shape[1]):
            raise IndexError(
                f"anchor_index {args_cli.anchor_index} out of range for body_pos_w shape {motion.body_pos_w.shape}"
            )
        anchor_xy0 = motion.body_pos_w[0, args_cli.anchor_index, :2].clone()
        if args_cli.xy_scale != 1.0:
            anchor_xy0 = anchor_xy0 * args_cli.xy_scale
        if args_cli.pos_scale != 1.0:
            if args_cli.pos_scale_xy_only:
                anchor_xy0 = anchor_xy0 * args_cli.pos_scale
            else:
                anchor_xy0 = anchor_xy0 * args_cli.pos_scale

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        body_pos = motion.body_pos_w[time_steps].clone()
        body_lin_vel = motion.body_lin_vel_w[time_steps].clone()
        if args_cli.xy_scale != 1.0:
            body_pos[..., :2] = body_pos[..., :2] * args_cli.xy_scale
            body_lin_vel[..., :2] = body_lin_vel[..., :2] * args_cli.xy_scale
        if args_cli.pos_scale != 1.0:
            if args_cli.pos_scale_xy_only:
                body_pos[..., :2] = body_pos[..., :2] * args_cli.pos_scale
                body_lin_vel[..., :2] = body_lin_vel[..., :2] * args_cli.pos_scale
            else:
                body_pos = body_pos * args_cli.pos_scale
                body_lin_vel = body_lin_vel * args_cli.pos_scale
        if anchor_xy0 is not None:
            body_pos[..., :2] = body_pos[..., :2] - anchor_xy0

        root_states[:, :3] = body_pos[:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = body_lin_vel[:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / 30.0
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
