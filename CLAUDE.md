# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Links

- **[MOTION_CAPTURE_PIPELINE.md](./MOTION_CAPTURE_PIPELINE.md)** — How to create CSV motion files (Video→SMPL→Robot, Mixamo, LAFAN1, PHC)
- **[ROKOKO_PIPELINE.md](./ROKOKO_PIPELINE.md)** — Rokoko FBX → CSV pipeline for mimic training
- **[G1_23DOF_GUIDE.md](./G1_23DOF_GUIDE.md)** — Mimic training for G1-23dof robot (CSV conversion, NPZ creation, training)
- **[XR_TELEOPERATE_GUIDE.md](./XR_TELEOPERATE_GUIDE.md)** — Подробное руководство по запуску xr_teleoperate + unitree_sim_isaaclab
- **[4D_HUMANS_GUIDE.md](./4D_HUMANS_GUIDE.md)** — 4D-Humans: извлечение 3D поз из видео (Video→SMPL→PKL)
- **[SETUP_LOG.md](./SETUP_LOG.md)** — Журнал установки unitree_rl_lab

## Projects Overview

This workspace contains 5 interconnected Unitree robotics projects:

| Project | Purpose | Stack |
|---------|---------|-------|
| **unitree_rl_lab** | RL training for locomotion (walking, running) | Python, Isaac Lab 2.3.0, Isaac Sim 5.1.0 |
| **unitree_sim_isaaclab** | Manipulation simulation & data collection | Python, Isaac Lab, Isaac Sim 4.5/5.0 |
| **unitree_mujoco** | Lightweight MuJoCo simulation + policy deploy | C++/Python, MuJoCo 3.2.6 |
| **unitree_sdk2** | Low-level C++ robot control SDK | C++, CMake |
| **xr_teleoperate** | XR teleoperation & data recording | Python, WebRTC, Pinocchio |

## Architecture

```
xr_teleoperate (data collection)
       ↓
unitree_sim_isaaclab (simulation/replay)
       ↓
unitree_rl_lab (RL training)
       ↓
unitree_sdk2 (deploy to real robot)
```

All projects share DDS communication protocol for sim-to-real compatibility.

## Common Commands

### unitree_rl_lab

```bash
conda activate env_isaaclab

# Install
./unitree_rl_lab.sh -i

# List available tasks
./unitree_rl_lab.sh -l

# Train (headless)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity

# Inference (with visualization)
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity

# Direct script usage
python scripts/rsl_rl/train.py --headless --task Unitree-G1-29dof-Velocity
python scripts/rsl_rl/play.py --task Unitree-G1-29dof-Velocity
```

**Configuration**: Set `UNITREE_MODEL_DIR` or `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`

### unitree_sim_isaaclab

```bash
conda activate unitree_sim_env

# Download assets
. fetch_assets.sh

# Run simulation with teleoperation
python sim_main.py --device cpu --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds --robot_type g129

# Data replay
python sim_main.py ... --replay --file_path "/path/to/data"

# Data generation with augmentation
python sim_main.py ... --generate_data --generate_data_dir "./data" \
  --modify_light --modify_camera

# Headless mode (no GUI)
python sim_main.py ... --headless
```

**Tasks**: `Isaac-PickPlace-Cylinder-*`, `Isaac-PickPlace-RedBlock-*`, `Isaac-Stack-RgyBlock-*`, `Isaac-Move-*-Wholebody`

### unitree_sdk2

```bash
cd /home/dias/Documents/unitree/unitree_sdk2

# Install dependencies
sudo apt install cmake g++ build-essential libyaml-cpp-dev \
  libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev

# Build
rm -rf build && mkdir build && cd build
cmake .. && make -j$(nproc)

# Install to system
sudo make install
```

**Installed to:**
- Library: `/usr/local/lib/libunitree_sdk2.a`
- Headers: `/usr/local/include/unitree/`
- CMake: `/usr/local/lib/cmake/unitree_sdk2/`

**Built examples** (in `build/bin/`):

| Example | Description |
|---------|-------------|
| `g1_loco_client` | High-level locomotion |
| `g1_dual_arm_example` | Dual arm low-level control |
| `g1_dex3_example` | Dex3 hand control |
| `g1_arm5_sdk_dds_example` | Arm control (5 DOF) |
| `g1_arm7_sdk_dds_example` | Arm control (7 DOF) |
| `test_publisher` / `test_subscriber` | DDS test |

**Test DDS:**
```bash
export LD_LIBRARY_PATH=/home/dias/Documents/unitree/unitree_sdk2/thirdparty/lib/x86_64:$LD_LIBRARY_PATH
./build/bin/test_publisher &
./build/bin/test_subscriber
```

**Use in CMake project:**
```cmake
find_package(unitree_sdk2 REQUIRED)
target_link_libraries(your_app unitree_sdk2)
```

### unitree_mujoco

```bash
cd /home/dias/Documents/unitree/unitree_mujoco

# Install dependencies
sudo apt install libglfw3-dev libyaml-cpp-dev

# Download MuJoCo (if not exists)
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://github.com/google-deepmind/mujoco/releases/download/3.2.6/mujoco-3.2.6-linux-x86_64.tar.gz
tar -xzf mujoco-3.2.6-linux-x86_64.tar.gz

# Create symlink
cd /home/dias/Documents/unitree/unitree_mujoco/simulate
ln -sf ~/.mujoco/mujoco-3.2.6 mujoco

# Build
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/usr/local/lib/cmake"
make -j$(nproc)
```

**Run simulator:**
```bash
cd /home/dias/Documents/unitree/unitree_mujoco/simulate/build

# G1-29dof
./unitree_mujoco -r g1 -s scene_29dof.xml -i 1 -n lo

# Go2
./unitree_mujoco -r go2 -s scene.xml -i 1 -n lo

# H1
./unitree_mujoco -r h1 -s scene.xml -i 1 -n lo
```

**Parameters:**
- `-r` — robot type (g1, go2, h1, h1_2, b2, b2w, go2w)
- `-s` — scene file
- `-i` — DDS domain ID (1 for simulation, 0 for real robot)
- `-n` — network interface (lo for localhost)

**Available robots:**
| Robot | Scene Files |
|-------|-------------|
| g1 | scene.xml, scene_23dof.xml, scene_29dof.xml |
| go2 | scene.xml, scene_terrain.xml |
| h1 | scene.xml |
| h1_2 | scene.xml |

### xr_teleoperate

```bash
conda activate tv

# Run teleoperation (simulation)
python teleop/teleop_hand_and_arm.py --ee=dex3 --sim --record

# Controls: r=start teleop, s=start/stop recording, q=quit
```

## Deploy to Physical Robot (Sim2Real)

```bash
# Build controller (requires unitree_sdk2 installed)
cd unitree_rl_lab/deploy/robots/g1_29dof
mkdir build && cd build
cmake .. && make

# Run on robot
./g1_ctrl --network eth0
```

## Key Paths

- **Robot configs**: `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`
- **RL tasks**: `unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/`
- **Sim tasks**: `unitree_sim_isaaclab/tasks/g1_tasks/`, `tasks/h1-2_tasks/`
- **Deploy code**: `unitree_rl_lab/deploy/robots/`
- **DDS module**: `unitree_sim_isaaclab/dds/`

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Ubuntu 20.04 / 22.04 | **Ubuntu 22.04** |
| **NVIDIA Driver** | 535.129+ | 550.x or 580.x |
| **GPU** | RTX 2070 | RTX 3080+ / RTX 4080 |
| **VRAM** | 8 GB | 16 GB+ |
| **RAM** | 32 GB | 64 GB |
| **Disk** | 50 GB free | 100 GB free |
| **GLIBC** | 2.35+ (for Isaac Sim 5.x) | 2.35+ |

## Ubuntu Version Compatibility

**Ubuntu 22.04 is recommended** — all projects work with simple pip install.

### Full Compatibility Matrix

| Project | Ubuntu 20.04 | Ubuntu 22.04 | Ubuntu 24.04 | Notes |
|---------|--------------|--------------|--------------|-------|
| **unitree_rl_lab** | ❌ | ✅ | ✅ | Requires Isaac Sim 5.1.0 (GLIBC 2.35+) |
| **unitree_sim_isaaclab** | ⚠️ Binary only | ✅ pip install | ✅ pip install | Isaac Sim 5.0.0 |
| **unitree_sdk2** | ✅ | ✅ | ✅ | C++ - just rebuild |
| **unitree_mujoco** | ✅ | ✅ | ✅ | Very portable, minimal deps |
| **xr_teleoperate** | ✅ | ✅ | ✅ | Python only |
| **MuJoCo (engine)** | ✅ | ✅ | ✅ | Works everywhere (Linux/Win/Mac) |

### Simulator Comparison

| Simulator | Ubuntu 20.04 | Ubuntu 22.04 | Ubuntu 24.04 | Install |
|-----------|--------------|--------------|--------------|---------|
| **MuJoCo** | ✅ | ✅ | ✅ | `pip install mujoco` |
| **Isaac Sim 4.x** | ✅ | ✅ | ❓ | Binary download |
| **Isaac Sim 5.x** | ❌ | ✅ | ✅ | `pip install isaacsim` |

### Why Ubuntu 20.04 has limitations

Isaac Sim 5.x requires **GLIBC 2.35+**, but Ubuntu 20.04 has GLIBC 2.31:
```bash
# Check your GLIBC version
ldd --version
```

GLIBC is a core system library that **cannot be upgraded** without upgrading Ubuntu itself.

### Upgrade Ubuntu 20.04 → 22.04

```bash
# 1. Update current system
sudo apt update && sudo apt upgrade -y

# 2. Install update manager
sudo apt install update-manager-core

# 3. Run upgrade (will upgrade to 22.04 LTS, NOT 24.04)
sudo do-release-upgrade
```

**Notes:**
- `do-release-upgrade` upgrades to next LTS only: 20.04 → 22.04 → 24.04
- Process takes 30-60 minutes
- Backup important data before upgrading
- Avoid `-d` flag (development/unstable releases)

### After Upgrade Checklist

After upgrading to Ubuntu 22.04:

```bash
# 1. Verify GLIBC version (should be 2.35+)
ldd --version

# 2. Reinstall NVIDIA drivers if needed
sudo apt install nvidia-driver-535

# 3. Rebuild unitree_sdk2
cd /home/dias/Documents/unitree/unitree_sdk2
rm -rf build && mkdir build && cd build
cmake .. && make

# 4. Reinstall conda environments (clean install recommended)
conda env remove -n env_isaaclab
conda env remove -n unitree_sim_env
# Then follow installation steps below
```

### Recommendation

**Upgrade to Ubuntu 22.04** for full compatibility with all projects:
- ✅ unitree_sdk2 works (just rebuild with `cmake .. && make`)
- ✅ unitree_sim_isaaclab works (pip install)
- ✅ unitree_rl_lab works (pip install)
- ✅ unitree_mujoco works (pip install)
- ✅ All other projects work unchanged

## Environment Setup

Two separate conda environments are required due to version incompatibility.

### Ubuntu 22.04+ (pip install) — Recommended

```bash
# For unitree_rl_lab (Isaac Sim 5.1.0 + IsaacLab 2.3.0)
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# Install Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.0
./isaaclab.sh --install

# For unitree_sim_isaaclab (Isaac Sim 5.0.0 + IsaacLab 2.2.0)
conda create -n unitree_sim_env python=3.11
conda activate unitree_sim_env
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
```

### Ubuntu 20.04 (limited support)

Only **unitree_mujoco** + pre-trained policies work fully. For Isaac Sim:

```bash
# Option 1: Use Isaac Sim 4.5.0 binary (limited unitree_rl_lab support)
# Download from: https://developer.nvidia.com/isaac-sim

# Option 2: Use Docker with Ubuntu 22.04 base
# See unitree_sim_isaaclab/Dockerfile

# Option 3: Upgrade to Ubuntu 22.04 (recommended)
```

### xr_teleoperate (any Ubuntu)

```bash
conda create -n tv python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
```

## Supported Robots

| Robot | unitree_rl_lab | unitree_sim_isaaclab |
|-------|----------------|---------------------|
| Go2 | ✓ | - |
| H1 | ✓ | - |
| H1-2 (27dof) | - | ✓ |
| G1-29dof | ✓ | ✓ |
| G1-23dof | ✓ | - |

**End effectors** (unitree_sim_isaaclab): DEX1 (gripper), DEX3 (3-finger), Inspire

## Simulation Architecture

### Real Robot vs Simulation

On real Unitree robots, the onboard PC1 runs a **proprietary locomotion controller** (walking, balance). In simulation, this controller is **not included** — you need trained RL policies instead.

```
REAL ROBOT:                          SIMULATION:
┌─────────────┐                      ┌─────────────┐
│ Your Code   │                      │ Your Code   │
│ (sdk2)      │                      │ (sdk2)      │
└──────┬──────┘                      └──────┬──────┘
       │ high-level                         │ high-level
       ▼                                    ▼
┌─────────────┐                      ┌─────────────┐
│ Unitree     │                      │ policy.onnx │ ← From unitree_rl_lab
│ Locomotion  │                      │ (RL policy) │
│ Controller  │                      └──────┬──────┘
│ (on PC1)    │                             │ low-level
└──────┬──────┘                             ▼
       │ low-level                   ┌─────────────┐
       ▼                             │ unitree_    │
┌─────────────┐                      │ mujoco      │
│ Motors      │                      └─────────────┘
└─────────────┘
```

### Simulation Modes in unitree_sim_isaaclab

| Mode | USD File | Description |
|------|----------|-------------|
| **BASE_FIX** | `*_base_fix.usd` | Robot body fixed (for arm manipulation only) |
| **WHOLEBODY** | `*_wholebody_*.usd` | Free body (requires balance controller) |

### Pre-trained Policies (Ready to Use)

Located in `unitree_rl_lab/deploy/robots/g1_29dof/config/policy/`:

| Policy | Path | Description |
|--------|------|-------------|
| **Walking** | `velocity/v0/exported/policy.onnx` | Velocity control (vx, vy, ωz) |
| **Gangnam Style** | `mimic/gangnam_style/exported/policy.onnx` | Dance motion |
| **Dance 102** | `mimic/dance_102/exported/policy.onnx` | Dance motion |
| **Bata Dias** | `mimic/bata_dias/exported/policy.onnx` | Custom Rokoko motion |

**Note**: These `.onnx` files are for MuJoCo/real robot deployment only. For Isaac Lab `play.py`, you need `.pt` checkpoints from training.

## Training in Isaac Lab

### Available Training Tasks

| Task | Robot | Type | Pre-trained |
|------|-------|------|-------------|
| `Unitree-G1-29dof-Velocity` | G1 (29 joints) | Walking | ✅ Yes |
| `Unitree-Go2-Velocity` | Go2 (quadruped) | Walking | ❌ No |
| `Unitree-H1-Velocity` | H1 (humanoid) | Walking | ❌ No |
| `Unitree-G1-29dof-Mimic-Gangnanm-Style` | G1 | Dance | ✅ Yes |
| `Unitree-G1-29dof-Mimic-Dance-102` | G1 | Dance | ✅ Yes |
| `Unitree-G1-29dof-Mimic-Bata-Dias` | G1 | Custom | ✅ Yes |
| `Unitree-G1-23dof-Mimic-Bata-Dias` | G1-23dof | Custom | ❌ Training |
| `Unitree-G1-23dof-Velocity` | G1-23dof | Walking | ❌ Training |

## G1-23DOF Conversion (from 29DOF)

### Difference Between 29DOF and 23DOF

G1-23dof is a simplified version without wrist pitch/yaw and waist roll/pitch joints.

**Removed joints (6 total):**

| Joint | 29DOF Index | Description |
|-------|-------------|-------------|
| `waist_roll_joint` | 13 | Waist roll |
| `waist_pitch_joint` | 14 | Waist pitch |
| `left_wrist_pitch_joint` | 20 | Left wrist pitch |
| `left_wrist_yaw_joint` | 21 | Left wrist yaw |
| `right_wrist_pitch_joint` | 27 | Right wrist pitch |
| `right_wrist_yaw_joint` | 28 | Right wrist yaw |

**Result:** 29 - 6 = 23 joints

### Step 1: Convert CSV (29DOF → 23DOF)

```bash
cd /home/dias/Documents/unitree/unitree_rl_lab

python -c "
import numpy as np

# Input: 29dof CSV (36 columns = 7 pose + 29 joints)
input_csv = 'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/YOUR_MOTION/motion.csv'
output_csv = 'source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_23dof/YOUR_MOTION/motion.csv'

data = np.loadtxt(input_csv, delimiter=',')
print(f'Input: {data.shape}')

# Remove columns: 7+13, 7+14, 7+20, 7+21, 7+27, 7+28
COLUMNS_TO_REMOVE = [20, 21, 27, 28, 34, 35]
columns_to_keep = [i for i in range(36) if i not in COLUMNS_TO_REMOVE]
data_23dof = data[:, columns_to_keep]

print(f'Output: {data_23dof.shape}')  # Should be (N, 30)
np.savetxt(output_csv, data_23dof, delimiter=',', fmt='%.6f')
"
```

Or use existing script:
```bash
python source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_23dof/dance_102/convert_29dof_to_23dof.py
```

### Step 2: Create NPZ from CSV

```bash
conda activate env_isaaclab

python scripts/mimic/csv_to_npz.py \
    -f source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_23dof/YOUR_MOTION/motion.csv \
    --input_fps 60 \
    --robot g1_23dof \
    --headless
```

**Important:** Use `--robot g1_23dof` flag!

### Step 3: Create Task Configuration

Create directory structure:
```
tasks/mimic/robots/g1_23dof/YOUR_MOTION/
├── __init__.py           # Task registration
├── tracking_env_cfg.py   # Environment config
├── motion.csv            # Converted CSV
└── motion.npz            # Generated NPZ
```

Copy and modify from existing task (e.g., `bata_dias`):
```bash
cp -r tasks/mimic/robots/g1_23dof/bata_dias tasks/mimic/robots/g1_23dof/YOUR_MOTION
```

Edit `__init__.py`:
```python
import gymnasium as gym

gym.register(
    id="Unitree-G1-23dof-Mimic-YOUR-MOTION",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
```

Register in parent `__init__.py`:
```bash
echo "from .YOUR_MOTION import *" >> tasks/mimic/robots/g1_23dof/__init__.py
```

### Step 4: Train

```bash
# Mimic (dance/motion)
./unitree_rl_lab.sh -t --task Unitree-G1-23dof-Mimic-Bata-Dias --headless --num_envs 4096

# Velocity (walking)
./unitree_rl_lab.sh -t --task Unitree-G1-23dof-Velocity --headless --num_envs 4096
```

### Step 5: Preview Motion (without training)

```bash
python scripts/mimic/replay_motion.py \
    -f source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_23dof/bata_dias/bata_dias_full_60hz.npz \
    --loop
```

### Multi-GPU Training (Parallel)

```bash
# Terminal 1: GPU 0
CUDA_VISIBLE_DEVICES=0 ./unitree_rl_lab.sh -t --task Unitree-G1-23dof-Mimic-Bata-Dias --headless --num_envs 4096

# Terminal 2: GPU 1
CUDA_VISIBLE_DEVICES=1 ./unitree_rl_lab.sh -t --task Unitree-G1-23dof-Velocity --headless --num_envs 4096
```

### G1-23DOF Task Files

| Task | Location |
|------|----------|
| Velocity | `tasks/locomotion/robots/g1/23dof/` |
| Bata Dias | `tasks/mimic/robots/g1_23dof/bata_dias/` |
| Dance 102 | `tasks/mimic/robots/g1_23dof/dance_102/` |

### Two Types of Training

#### Velocity (Locomotion) — No data needed

Robot learns through trial & error with reward functions:

```
Reward = forward_velocity + stability - energy - falling
```

```bash
# Train walking from scratch
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --headless --num_envs 4096
```

#### Mimic (Imitation) — Requires motion data

Robot learns to replicate recorded human movements:

```
Motion Capture (CSV) → Convert to NPZ → Train to imitate
```

### Motion Data Conversion

Before training Mimic tasks, convert CSV to NPZ:

```bash
cd /home/dias/Documents/unitree/unitree_rl_lab
conda activate env_isaaclab

# Gangnam Style
python scripts/mimic/csv_to_npz.py \
  -f source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/gangnanm_style/G1_gangnam_style_V01.bvh_60hz.csv \
  --input_fps 60

# Dance 102
python scripts/mimic/csv_to_npz.py \
  -f source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/G1_Take_102.bvh_60hz.csv \
  --input_fps 60
```

### Training Commands

```bash
# With visualization (slower, for debugging)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --num_envs 64

# Headless (faster, for actual training)
./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --headless --num_envs 4096

# Monitor with TensorBoard
tensorboard --logdir logs/rsl_rl/
# Open http://localhost:6006
```

### Training Output

After training, models are saved to:
```
logs/rsl_rl/<task_name>/
├── model_*.pt         ← Checkpoints for play.py
└── exported/
    └── policy.onnx    ← For MuJoCo/real robot
```

### Multi-GPU Usage

Isaac Lab uses **1 GPU** by default. Multi-GPU doesn't provide significant speedup for RL (bottleneck is physics simulation, not neural network).

**Best strategies:**

```bash
# Select specific GPU
CUDA_VISIBLE_DEVICES=0 ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --headless

# Run 2 experiments in parallel (2x experiments, not 2x speed)
# Terminal 1:
CUDA_VISIBLE_DEVICES=0 ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Velocity --headless
# Terminal 2:
CUDA_VISIBLE_DEVICES=1 ./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity --headless
```

**Speed optimization**: Increase `--num_envs` (e.g., 4096) instead of adding GPUs.

### Running Simulation with Walking

**Option 1: Isaac Lab (GPU required)**
```bash
conda activate env_isaaclab
cd unitree_rl_lab
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Velocity
```

**Option 2: MuJoCo + policy.onnx (lighter)**
```bash
# Terminal 1: Start MuJoCo simulator
cd unitree_mujoco/simulate/build
./unitree_mujoco

# Terminal 2: Run controller with policy
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl --network lo
```

### MuJoCo Gamepad/Keyboard Control

MuJoCo supports Xbox 360, PlayStation (DualShock 4), Switch controllers, and keyboard.

**Config file**: `unitree_mujoco/simulate/config.yaml`

```yaml
robot: "g1"
robot_scene: "scene_29dof.xml"
domain_id: 1
interface: "lo"

use_joystick: 1
joystick_type: "xbox"      # "xbox", "switch", or "keyboard"
joystick_device: "/dev/input/js1"
joystick_bits: 16

enable_elastic_band: 1     # Virtual spring to prevent falling
```

**Check connected gamepads:**
```bash
ls /dev/input/js*
jstest /dev/input/js0
```

**Gamepad Controls (Xbox 360 / DualShock 4):**

| Combo | Action |
|-------|--------|
| LT + D-pad ↑ | FixStand (stand up) |
| RB + X | Velocity (walking mode) |
| LT + D-pad ↓ | Dance 102 |
| LT + D-pad ← | Gangnam Style |
| LT + D-pad → | My Dance |
| LT + A | Bata Dias |
| LT + B | Passive (stop) |
| Left Stick | Movement (forward/back/strafe) |
| Right Stick | Rotation |

**Keyboard Controls (when `joystick_type: "keyboard"`):**

| Key | Action |
|-----|--------|
| 1 | FixStand |
| 2 | Velocity |
| 3 | Dance 102 |
| 4 | Gangnam Style |
| 0 | Passive |
| W/S/A/D | Movement |
| Q/E | Rotation |

**Elastic Band (prevents falling during tests):**

| Key | Action |
|-----|--------|
| 9 | Toggle elastic band on/off |
| 7 or ↑ | Shorten band (lift robot) |
| 8 or ↓ | Lengthen band (lower robot) |

### Running Dance Policies in MuJoCo

```bash
# Terminal 1: MuJoCo with gamepad
cd /home/dias/Documents/unitree/unitree_mujoco/simulate/build
./unitree_mujoco

# Terminal 2: Controller with dance policies
cd /home/dias/Documents/unitree/unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl --network lo
```

**Sequence:**
1. LT + ↑ → Robot stands up (FixStand)
2. LT + ↓ → Dance 102 **OR** LT + ← → Gangnam Style

### Training Dance Policies

**Available dance motions:**

| Dance | Task ID | Motion File |
|-------|---------|-------------|
| Gangnam Style | `Unitree-G1-29dof-Mimic-Gangnanm-Style` | G1_gangnam_style_V01.bvh_60hz.csv |
| Dance 102 | `Unitree-G1-29dof-Mimic-Dance-102` | G1_Take_102.bvh_60hz.csv |
| Bata Dias | `Unitree-G1-29dof-Mimic-Bata-Dias` | bata_dias_full_60hz.csv |

**See [ROKOKO_PIPELINE.md](./ROKOKO_PIPELINE.md)** for creating custom motions from Rokoko.

**Train on multiple GPUs in parallel:**
```bash
# Terminal 1: GPU 0
CUDA_VISIBLE_DEVICES=0 ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Mimic-Gangnanm-Style --headless --num_envs 4096

# Terminal 2: GPU 1
CUDA_VISIBLE_DEVICES=1 ./unitree_rl_lab.sh -t --task Unitree-G1-29dof-Mimic-Dance-102 --headless --num_envs 4096
```

**After training, deploy to MuJoCo:**
```bash
# Copy trained ONNX to deploy
cp logs/rsl_rl/unitree_g1_29dof_mimic_gangnanm_style/*/exported/policy.onnx \
   deploy/robots/g1_29dof/config/policy/mimic/gangnam_style/exported/

cp logs/rsl_rl/unitree_g1_29dof_mimic_dance_102/*/exported/policy.onnx \
   deploy/robots/g1_29dof/config/policy/mimic/dance_102/exported/
```

**View specific training run:**
```bash
./unitree_rl_lab.sh -p --task Unitree-G1-29dof-Mimic-Gangnanm-Style --load_run 2025-12-25_18-19-07
```

## DDS Communication

All simulators use same DDS protocol as real robots:

| Topic | Direction | Description |
|-------|-----------|-------------|
| `rt/lowcmd` | → Robot | Joint commands |
| `rt/lowstate` | ← Robot | Joint states, IMU |
| `rt/sportmodestate` | ← Robot | High-level state |

**Domain ID**: Simulation uses `domain_id=1`, real robot uses `domain_id=0`

## Troubleshooting

### NVIDIA Driver Issues

**Problem**: Isaac Sim shows wrong driver version or RTX renderer fails
```
[Error] The currently installed NVIDIA graphics driver is unsupported
Installed driver: 535.18
The unsupported driver range: [0.0, 535.129)
```

**Solution**: Update NVIDIA driver
```bash
sudo apt update
sudo apt install nvidia-driver-550  # or nvidia-driver-570/580
sudo reboot
```

### Missing Motion Data (Mimic tasks)

**Problem**: Training mimic task fails with missing `.npz` file
```
AssertionError: Invalid file path: .../G1_gangnam_style_V01.bvh_60hz.npz
```

**Solution**: Convert CSV to NPZ first (see "Motion Data Conversion" section above)

### No logs directory for play.py

**Problem**: `play.py` fails because no trained model exists
```
FileNotFoundError: .../logs/rsl_rl/unitree_g1_29dof_velocity
```

**Solution**: Train the model first with `train.py`, or use MuJoCo with pre-trained `.onnx` policies

### PCIe Bandwidth Warning

**Problem**: Multi-GPU setup shows PCIe warnings
```
[Warning] PCIe link width current (4) and maximum (16) don't match
```

**Impact**: Reduced GPU-to-GPU transfer speed. Not critical for single-GPU training.

**Solution**: Move GPU to x16 PCIe slot if available, or ignore for single-GPU workloads

### CPU Powersave Mode

**Problem**: Training is slower than expected
```
[Warning] CPU performance profile is set to powersave
```

**Solution**:
```bash
sudo cpupower frequency-set -g performance
```

### cyclonedds Build Error (unitree_sdk2_python)

**Problem**: Installing unitree_sdk2_python fails
```
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

**Solution**: Build cyclonedds from source first
```bash
cd /home/dias/Documents/unitree
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install

# Then install with CYCLONEDDS_HOME set
cd /home/dias/Documents/unitree/unitree_sdk2_python
export CYCLONEDDS_HOME="/home/dias/Documents/unitree/cyclonedds/install"
pip install -e .
```

### numpy Version Conflicts

**Problem**: unitree_sdk2_python installs numpy 2.x but Isaac Sim requires 1.26.0
```
isaacsim-kernel requires numpy==1.26.0, but you have numpy 2.2.6
```

**Solution**: Downgrade numpy after installing unitree_sdk2_python
```bash
pip install numpy==1.26.0 packaging==23.0 sentry-sdk==1.43.0
```

### teleimager Missing Dependencies

**Problem**: teleimager requires aiortc
```
teleimager 1.0.0 requires aiortc, which is not installed
```

**Solution**:
```bash
pip install aiortc
```

## xr_teleoperate — VR Teleoperation

### Purpose

Control Unitree robots in real-time using **VR devices** (Apple Vision Pro, Meta Quest 3, Pico 4). The system captures hand/controller movements and translates them to robot arm and hand commands.

### Supported Hardware

| Component | Options |
|-----------|---------|
| **VR Headsets** | Apple Vision Pro, Meta Quest 3/3S, Pico 4 Ultra |
| **Robots** | G1-29dof, G1-23dof, H1, H1-2 |
| **End Effectors** | Dex1 (gripper), Dex3 (3-finger), Inspire, BrainCo |
| **Camera** | Realsense D435i (robot head) |

### Directory Structure

```
xr_teleoperate/
├── teleop/
│   ├── teleop_hand_and_arm.py   ← Main entry point
│   ├── robot_control/            # IK, hand control
│   ├── televuer/                 # WebRTC VR interface
│   └── utils/                    # Data recording
├── assets/                       # Robot URDF models
└── requirements.txt
```

### Environment Setup

```bash
# 1. Create conda environment
conda create -n tv python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge -y
conda activate tv

# 2. Initialize submodules
cd /home/dias/Documents/unitree/xr_teleoperate
git submodule update --init --depth 1

# 3. Install teleimager
cd teleop/teleimager
pip install -e . --no-deps

# 4. Install televuer
cd ../televuer
pip install -e .

# 5. Install dependencies
cd ../..
pip install -r requirements.txt
pip install aiortc

# 6. Install unitree_sdk2_python
cd /home/dias/Documents/unitree/unitree_sdk2_python
pip install -e .
```

### SSL Certificates Setup

Required for VR device connection:

```bash
cd /home/dias/Documents/unitree/xr_teleoperate/teleop/televuer

# Generate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem -subj "/CN=localhost"

# Copy to config directory
mkdir -p ~/.config/xr_teleoperate/
cp cert.pem key.pem ~/.config/xr_teleoperate/

# Open firewall port
sudo ufw allow 8012
```

### Commands

```bash
conda activate tv
cd xr_teleoperate

# Teleoperation in simulation mode
python teleop/teleop_hand_and_arm.py --arm=g1_29 --ee=dex3 --sim

# Teleoperation with real robot + recording
python teleop/teleop_hand_and_arm.py --arm=g1_29 --ee=dex3 --record

# Different display modes
python teleop/teleop_hand_and_arm.py --arm=g1_29 --ee=dex3 --display-mode=immersive
python teleop/teleop_hand_and_arm.py --arm=g1_29 --ee=dex3 --display-mode=ego
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `r` | Start teleoperation |
| `s` | Start/stop recording |
| `q` | Quit |

### CLI Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--arm` | g1_29, g1_23, h1, h1_2 | Robot type |
| `--ee` | dex1, dex3, inspire | End effector |
| `--input-mode` | hand, controller | Input method |
| `--display-mode` | immersive, ego, pass-through | VR display mode |
| `--sim` | flag | Simulation mode |
| `--record` | flag | Enable recording |
| `--frequency` | int (default: 30) | Control frequency |

### Data Output

Recorded data is saved for imitation learning:
```
recordings/
├── episode_001/
│   ├── joint_positions.npy
│   ├── joint_velocities.npy
│   ├── hand_positions.npy
│   └── camera_images/
```

## unitree_sim_isaaclab — Manipulation Simulation

### Purpose

Isaac Lab-based simulation for **manipulation tasks** (pick-place, stacking). Integrates with xr_teleoperate for dataset collection and augmentation.

### Available Tasks

**G1-29dof Tasks (15 total):**

| Task Type | End Effectors | Description |
|-----------|---------------|-------------|
| `Pick-Place-Cylinder` | Dex1, Dex3, Inspire | Pick up cylinder, place at target |
| `Pick-Place-RedBlock` | Dex1, Dex3, Inspire | Pick up red block |
| `Stack-RgyBlock` | Dex1, Dex3, Inspire | Stack colored blocks |
| `Move-Cylinder-Wholebody` | Dex1, Dex3, Inspire | Move while walking |
| `Pick-RedBlock-into-Drawer` | Dex1, Dex3 | Place block in drawer |

**H1-2 Tasks (6 total):** Pick-Place and Stack with Inspire hand

### Directory Structure

```
unitree_sim_isaaclab/
├── sim_main.py              ← Main entry point
├── tasks/
│   ├── g1_tasks/            # 15 G1 task definitions
│   ├── h1-2_tasks/          # 6 H1-2 task definitions
│   ├── common_config/       # Robot/camera configs
│   ├── common_observations/ # State observations
│   ├── common_rewards/      # Reward functions
│   └── common_termination/  # Task completion conditions
├── dds/                     # DDS communication layer
├── action_provider/         # Action sources (DDS, replay)
├── image_server/            # Camera streaming
└── requirements.txt
```

### Environment Setup

```bash
# 1. Create conda environment
conda create -n unitree_sim_env python=3.11 -y
conda activate unitree_sim_env

# 2. Install PyTorch with CUDA
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

# 3. Install Isaac Sim 5.0.0
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

# 4. Install Isaac Lab 2.2.0 (separate from unitree_rl_lab's v2.3.0!)
cd /home/dias/Documents/unitree
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab_v2.2
cd IsaacLab_v2.2
git checkout v2.2.0
./isaaclab.sh --install

# 5. Build cyclonedds (required for unitree_sdk2_python)
cd /home/dias/Documents/unitree
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install

# 6. Install unitree_sdk2_python
cd /home/dias/Documents/unitree/unitree_sdk2_python
export CYCLONEDDS_HOME="/home/dias/Documents/unitree/cyclonedds/install"
pip install -e .

# 7. Fix dependency conflicts
pip install numpy==1.26.0 packaging==23.0 sentry-sdk==1.43.0

# 8. Install project requirements and download assets
cd /home/dias/Documents/unitree/unitree_sim_isaaclab
pip install -r requirements.txt
. fetch_assets.sh
```

### Isaac Lab Versions

**Important**: Two separate Isaac Lab installations are required:

| Project | Isaac Lab | Path |
|---------|-----------|------|
| unitree_rl_lab | v2.3.0 | `/home/dias/Documents/unitree/IsaacLab/` |
| unitree_sim_isaaclab | v2.2.0 | `/home/dias/Documents/unitree/IsaacLab_v2.2/` |

### Commands

```bash
conda activate unitree_sim_env
cd unitree_sim_isaaclab

# Run simulation with DDS teleoperation
python sim_main.py --device gpu --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --enable_dex3_dds --robot_type g129

# Replay recorded data
python sim_main.py --device gpu \
  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --replay --file_path "/path/to/recorded_data"

# Generate augmented dataset
python sim_main.py --device gpu \
  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --generate_data --generate_data_dir "./augmented_data" \
  --modify_light --modify_camera

# Headless mode (no GUI, faster)
python sim_main.py --headless ...

# Keyboard control for testing
python send_commands_keyboard.py
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--device cpu/gpu` | Compute device |
| `--task <name>` | Task name (see task list) |
| `--enable_dex1_dds` | Enable 2-finger gripper DDS |
| `--enable_dex3_dds` | Enable 3-finger hand DDS |
| `--enable_inspire_dds` | Enable Inspire hand DDS |
| `--robot_type g129/h1_2` | Robot type |
| `--enable_cameras` | Enable camera observations |
| `--headless` | No GUI mode |
| `--replay` | Replay mode |
| `--file_path` | Path to replay data |
| `--generate_data` | Data generation mode |
| `--generate_data_dir` | Output directory |
| `--modify_light` | Augment lighting |
| `--modify_camera` | Augment camera angles |
| `--step_hz` | Control frequency (default: 100) |

### Task Naming Convention

```
Isaac-<TaskType>-<Object>-<Robot>-<EndEffector>-<ControlMode>

Examples:
- Isaac-PickPlace-Cylinder-G129-Dex3-Joint
- Isaac-Stack-RgyBlock-G129-Inspire-Joint
- Isaac-Move-Cylinder-G129-Dex1-Wholebody
```

### DDS Integration

Same DDS protocol as real robots:

| Topic | Direction | Description |
|-------|-----------|-------------|
| `rt/lowcmd` | → Sim | Joint commands |
| `rt/lowstate` | ← Sim | Joint states |
| Hand-specific topics | ↔ | Dex3, Gripper, Inspire commands |

**Domain ID**: Simulation uses `domain_id=1`

## Project Comparison

| Aspect | unitree_rl_lab | unitree_sim_isaaclab | xr_teleoperate |
|--------|----------------|---------------------|----------------|
| **Purpose** | RL training (locomotion) | Manipulation simulation | VR teleoperation |
| **Control** | RL policy | DDS commands | VR controllers |
| **Tasks** | Walking, dancing | Pick-place, stacking | Data collection |
| **Isaac Sim** | 5.1.0 | 5.0.0 | Not needed |
| **Output** | policy.onnx | Dataset | Dataset |
| **Robot Focus** | Legs + whole body | Arms + hands | Arms + hands |

## Complete Data Pipeline

```
┌─────────────────────┐
│   xr_teleoperate    │  ← Operator controls robot via VR
│  (data collection)  │
└──────────┬──────────┘
           │ recorded trajectories
           ▼
┌─────────────────────┐
│ unitree_sim_isaaclab│  ← Replay + augmentation
│ (simulation/augment)│     (lighting, camera angles)
└──────────┬──────────┘
           │ augmented dataset
           ▼
┌─────────────────────┐
│   unitree_rl_lab    │  ← Train policy (imitation/RL)
│     (training)      │
└──────────┬──────────┘
           │ policy.onnx
           ▼
┌─────────────────────┐
│  Real Robot/MuJoCo  │  ← Deploy trained policy
│    (deployment)     │
└─────────────────────┘
```

### When to Use Each Project

| Goal | Project |
|------|---------|
| Train walking/running | unitree_rl_lab |
| Train dancing (mimic) | unitree_rl_lab |
| Collect manipulation data | xr_teleoperate |
| Replay/augment data | unitree_sim_isaaclab |
| Train manipulation policy | unitree_sim_isaaclab + RL |
| Deploy to real robot | unitree_sdk2 + policy.onnx |
