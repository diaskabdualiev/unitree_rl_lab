"""Universal motion replay for G1 robots (23dof or 29dof).

Automatically detects robot type from NPZ file.

Usage:
    # G1-23dof Bata Dias
    python scripts/mimic/replay_motion.py \
        -f source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_23dof/bata_dias/bata_dias_full_60hz.npz

    # G1-29dof Bata Dias
    python scripts/mimic/replay_motion.py \
        -f source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/bata_dias/bata_dias_full_60hz.npz

    # With speed control
    python scripts/mimic/replay_motion.py -f motion.npz --speed 0.5
"""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay motion for G1 robot (auto-detect 23/29 dof).")
parser.add_argument("--file", "-f", type=str, required=True, help="Path to NPZ motion file")
parser.add_argument("--speed", "-s", type=float, default=1.0, help="Playback speed (0.5=slow, 2.0=fast)")
parser.add_argument("--loop", "-l", action="store_true", help="Loop playback")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Detect DOF from NPZ before launching app
npz_data = np.load(args_cli.file)
n_joints = npz_data['joint_pos'].shape[-1]
npz_data.close()

if n_joints == 23:
    robot_type = "23dof"
elif n_joints == 29:
    robot_type = "29dof"
else:
    raise ValueError(f"Unknown joint count: {n_joints}. Expected 23 or 29.")

print(f"\n>>> Detected G1-{robot_type} motion ({n_joints} joints)")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import correct robot config
if robot_type == "23dof":
    from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG as ROBOT_CFG
else:
    from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG

from unitree_rl_lab.tasks.mimic.mdp import MotionLoader


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

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    motion = MotionLoader(
        args_cli.file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )

    duration = motion.time_step_total * sim_dt / args_cli.speed
    print(f"\n{'='*50}")
    print(f"  Motion Replay - G1-{robot_type}")
    print(f"{'='*50}")
    print(f"  File: {args_cli.file.split('/')[-1]}")
    print(f"  Frames: {motion.time_step_total}")
    print(f"  Duration: {duration:.2f}s @ {args_cli.speed}x speed")
    print(f"  Loop: {'Yes' if args_cli.loop else 'No'}")
    print(f"{'='*50}")
    print(f"\n  Close window to exit\n")

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    time_acc = 0.0

    while simulation_app.is_running():
        time_acc += args_cli.speed
        if time_acc >= 1.0:
            time_steps += int(time_acc)
            time_acc -= int(time_acc)

        reset_ids = time_steps >= motion.time_step_total
        if reset_ids.any():
            if args_cli.loop:
                time_steps[reset_ids] = 0
            else:
                time_steps = torch.clamp(time_steps, max=motion.time_step_total - 1)

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(motion.joint_pos[time_steps], motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
