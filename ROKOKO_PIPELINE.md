# Rokoko FBX → Robot CSV Pipeline

Полный пайплайн конвертации motion capture данных из Rokoko в формат для обучения Unitree G1.

## Overview

```
┌─────────────────┐
│  Video файл     │
│  (любой формат) │
└────────┬────────┘
         │ Rokoko Vision (web service)
         ▼
┌─────────────────┐
│  motion.fbx     │  ← Rokoko FBX (30 FPS)
│  (Rokoko export)│
└────────┬────────┘
         │ Blender (fbx_to_bvh.py)
         ▼
┌─────────────────┐
│  motion.bvh     │  ← BVH с Rokoko joint names
└────────┬────────┘
         │ sed (rename joints)
         ▼
┌─────────────────────┐
│ motion_renamed.bvh  │  ← BVH с LAFAN1 joint names
└────────┬────────────┘
         │ GMR bvh_to_robot.py --format nokov
         ▼
┌─────────────────┐
│  motion.pkl     │  ← Robot motion (30 FPS)
└────────┬────────┘
         │ interpolate 30→60 Hz
         ▼
┌─────────────────┐
│ motion_60hz.csv │  ← Ready for training
└────────┬────────┘
         │ csv_to_npz.py
         ▼
┌─────────────────┐
│ motion_60hz.npz │  ← Isaac Lab training format
└─────────────────┘
```

## Requirements

### Software
- **Blender** 3.0+ (for FBX→BVH conversion)
- **GMR** (General Motion Retargeting)
- **Isaac Lab** (for training)

### Conda Environments
```bash
# GMR environment
conda activate gmr

# Isaac Lab environment
conda activate env_isaaclab
```

## Step 1: Rokoko Video → FBX

1. Go to [Rokoko Vision](https://vision.rokoko.com/)
2. Upload video file
3. Process and download FBX

Output: `motion.fbx` (376 frames @ 30 FPS example)

## Step 2: FBX → BVH (Blender)

```bash
cd /home/dias/Documents/unitree/motion_capture

blender --background --python scripts/fbx_to_bvh.py -- \
  data/fbx/motion.fbx \
  data/bvh/motion.bvh
```

**Script location**: `/home/dias/Documents/unitree/motion_capture/scripts/fbx_to_bvh.py`

## Step 3: Rename Joints

Rokoko uses different joint names than LAFAN1/GMR expects:

| Rokoko | LAFAN1 (GMR) |
|--------|--------------|
| LeftThigh | LeftUpLeg |
| RightThigh | RightUpLeg |
| LeftShin | LeftLeg |
| RightShin | RightLeg |
| LeftToe | LeftToeBase |
| RightToe | RightToeBase |

```bash
cd /home/dias/Documents/unitree/motion_capture/data/bvh

sed -e 's/LeftThigh/LeftUpLeg/g' \
    -e 's/RightThigh/RightUpLeg/g' \
    -e 's/LeftShin/LeftLeg/g' \
    -e 's/RightShin/RightLeg/g' \
    -e 's/LeftToe/LeftToeBase/g' \
    -e 's/RightToe/RightToeBase/g' \
    motion.bvh > motion_renamed.bvh
```

## Step 4: BVH → Robot PKL (GMR)

```bash
cd /home/dias/Documents/unitree/motion_capture/GMR
conda activate gmr

# With visualization
python scripts/bvh_to_robot.py \
  --bvh_file ../data/bvh/motion_renamed.bvh \
  --robot unitree_g1 \
  --format nokov \
  --save_path ../data/output/motion.pkl \
  --loop

# Headless (faster)
python scripts/bvh_to_robot.py \
  --bvh_file ../data/bvh/motion_renamed.bvh \
  --robot unitree_g1 \
  --format nokov \
  --save_path ../data/output/motion.pkl \
  --headless
```

**Important**: Use `--format nokov` for Rokoko data (expects `LeftToeBase` instead of `LeftToe`).

## Step 5: PKL → CSV (60Hz)

```python
import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp

# Load PKL
with open('motion.pkl', 'rb') as f:
    data = pickle.load(f)

# Interpolate 30Hz → 60Hz
n_frames = len(data['root_pos'])
t_in = np.linspace(0, 1, n_frames)
t_out = np.linspace(0, 1, n_frames * 2 - 1)

# Position interpolation (linear)
root_pos_interp = interp1d(t_in, data['root_pos'], axis=0)(t_out)

# Rotation interpolation (slerp)
rotations = R.from_quat(data['root_rot'])
root_rot_interp = Slerp(t_in, rotations)(t_out).as_quat()

# Joint interpolation (linear)
dof_interp = interp1d(t_in, data['dof_pos'], axis=0)(t_out)

# Save CSV (no headers)
output = np.hstack([root_pos_interp, root_rot_interp, dof_interp])
np.savetxt('motion_60hz.csv', output, delimiter=',', fmt='%.6f')
```

## Step 6: Create Training Task

### Directory Structure

```
unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/
└── your_motion/
    ├── __init__.py
    ├── tracking_env_cfg.py
    ├── motion_60hz.csv
    └── motion_60hz.npz
```

### Create NPZ

```bash
cd /home/dias/Documents/unitree/unitree_rl_lab
conda activate env_isaaclab

python scripts/mimic/csv_to_npz.py \
  -f source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/your_motion/motion_60hz.csv \
  --input_fps 60 \
  --headless
```

### Register Task

**`__init__.py`**:
```python
import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Mimic-Your-Motion",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
```

**`tracking_env_cfg.py`**: Copy from `gangnanm_style/tracking_env_cfg.py` and update `motion_file` path.

## Step 7: Train

```bash
cd /home/dias/Documents/unitree/unitree_rl_lab
conda activate env_isaaclab

# With visualization
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-Your-Motion --num_envs 64

# Headless (faster)
python scripts/rsl_rl/train.py --task Unitree-G1-29dof-Mimic-Your-Motion --headless --num_envs 4096
```

Training takes ~9 hours for 30K iterations on RTX 4080.

## Step 8: Export to ONNX

```python
import torch
import torch.nn as nn

# Load checkpoint
checkpoint = torch.load('logs/rsl_rl/.../model_29999.pt', map_location='cpu')
state_dict = checkpoint['model_state_dict']

# Create actor network (same as training)
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(154, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 29),
        )
    def forward(self, x):
        return self.actor(x)

model = Actor()
model.load_state_dict({k: v for k, v in state_dict.items() if k.startswith('actor.')})
model.eval()

# Export ONNX (FIXED batch size = 1, not dynamic!)
torch.onnx.export(
    model,
    torch.randn(1, 154),
    'policy.onnx',
    input_names=['obs'],
    output_names=['actions'],
    opset_version=11
)
```

**Important**: Do NOT use `dynamic_axes` - MuJoCo controller expects fixed batch size `[1, 154]`.

## Step 9: Deploy to MuJoCo

### Copy files

```bash
mkdir -p deploy/robots/g1_29dof/config/policy/mimic/your_motion/{exported,params}

cp policy.onnx deploy/robots/g1_29dof/config/policy/mimic/your_motion/exported/
cp motion_60hz.csv deploy/robots/g1_29dof/config/policy/mimic/your_motion/params/
cp gangnam_style/params/deploy.yaml deploy/robots/g1_29dof/config/policy/mimic/your_motion/params/
```

### Update config.yaml

Add to `deploy/robots/g1_29dof/config/config.yaml`:

```yaml
FSM:
  _:
    # ... existing entries ...
    Mimic_Your_Motion:
      id: 105
      type: Mimic

  FixStand:
    transitions:
      # ... existing ...
      Mimic_Your_Motion: LT + A.on_pressed  # or other button

  Mimic_Your_Motion:
    transitions:
      Passive: LT + B.on_pressed
      Velocity: RB + X.on_pressed
    fps: 60
    motion_file: config/policy/mimic/your_motion/params/motion_60hz.csv
    policy_dir: config/policy/mimic/your_motion/
    time_start: 0.0
    time_end: 12.5  # duration in seconds
```

### Run

```bash
# Terminal 1: MuJoCo
cd /home/dias/Documents/unitree/unitree_mujoco/simulate/build
./unitree_mujoco -r g1 -s scene_29dof.xml -i 1 -n lo

# Terminal 2: Controller
cd /home/dias/Documents/unitree/unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl --network lo
```

Controls:
- `LT + ↑` — FixStand (stand up)
- `LT + A` — Your Motion
- `LT + B` — Passive (stop)

## G1-23dof Conversion

To create 23dof version from 29dof CSV:

```python
import numpy as np

# Load 29dof CSV (36 columns: 7 pose + 29 joints)
data = np.loadtxt('motion_60hz.csv', delimiter=',')

# Remove 6 joints: waist_roll, waist_pitch, wrist_pitch/yaw (both arms)
COLUMNS_TO_REMOVE = [20, 21, 27, 28, 34, 35]  # 0-indexed
columns_to_keep = [i for i in range(36) if i not in COLUMNS_TO_REMOVE]
data_23dof = data[:, columns_to_keep]  # Now 30 columns: 7 pose + 23 joints

np.savetxt('motion_23dof_60hz.csv', data_23dof, delimiter=',', fmt='%.6f')
```

Then create NPZ with `--robot g1_23dof`:

```bash
python scripts/mimic/csv_to_npz.py \
  -f .../motion_23dof_60hz.csv \
  --input_fps 60 \
  --robot g1_23dof \
  --headless
```

## Troubleshooting

### "tried creating tensor with negative value in shape"

ONNX exported with dynamic batch size. Re-export without `dynamic_axes`:

```python
torch.onnx.export(model, dummy_input, 'policy.onnx',
    input_names=['obs'], output_names=['actions'],
    opset_version=11)  # NO dynamic_axes!
```

### BVH file too large (24GB+)

sed command corrupted the file. Use single-line sed:

```bash
sed -e 's/A/B/g' -e 's/C/D/g' input.bvh > output.bvh
```

### KeyError: 'LeftToe'

Use `--format nokov` instead of `--format lafan1`:

```bash
python scripts/bvh_to_robot.py --format nokov ...
```

### Missing frames in BVH

Blender exported wrong frame range. Check `fbx_to_bvh.py` uses action frame range:

```python
if armature.animation_data and armature.animation_data.action:
    action = armature.animation_data.action
    frame_start = int(action.frame_range[0])
    frame_end = int(action.frame_range[1])
```

## File Locations

| File | Path |
|------|------|
| FBX input | `/home/dias/Documents/unitree/motion_capture/data/fbx/` |
| BVH output | `/home/dias/Documents/unitree/motion_capture/data/bvh/` |
| PKL output | `/home/dias/Documents/unitree/motion_capture/data/output/` |
| GMR scripts | `/home/dias/Documents/unitree/motion_capture/GMR/scripts/` |
| Blender script | `/home/dias/Documents/unitree/motion_capture/scripts/fbx_to_bvh.py` |
| Training tasks | `unitree_rl_lab/source/.../tasks/mimic/robots/g1_29dof/` |
| Deploy policies | `unitree_rl_lab/deploy/robots/g1_29dof/config/policy/mimic/` |
