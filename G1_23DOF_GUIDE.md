# G1-23dof Mimic Training Guide

This guide explains how to train motion imitation (mimic) tasks for the G1-23dof robot.

## Overview

### G1-29dof vs G1-23dof

| Feature | G1-29dof | G1-23dof |
|---------|----------|----------|
| **Waist joints** | 3 (yaw, roll, pitch) | 1 (yaw only) |
| **Arm joints** | 7 per arm | 5 per arm |
| **Total joints** | 29 | 23 |
| **Wrist** | roll, pitch, yaw | roll only |
| **Use case** | Full expressiveness | Cost-effective |

### Removed joints in 23dof

```
Waist:
├── waist_roll_joint   (removed)
└── waist_pitch_joint  (removed)

Left arm:
├── left_wrist_pitch_joint  (removed)
└── left_wrist_yaw_joint    (removed)

Right arm:
├── right_wrist_pitch_joint (removed)
└── right_wrist_yaw_joint   (removed)
```

## Why Examples Are Only for 29dof

1. **Flagship model**: G1-29dof is the full version with more expressiveness
2. **Motion capture**: Unitree records dances on real G1-29dof robots
3. **Official data**: All official CSV files contain 29 joints
4. **Conversion**: Easier to convert 29dof → 23dof than create separate recordings

## CSV File Format

### 29dof CSV (36 columns)
```
x, y, z, qx, qy, qz, qw, j1, j2, ..., j29
0-2     3-6              7-35 (29 joints)
```

### 23dof CSV (30 columns)
```
x, y, z, qx, qy, qz, qw, j1, j2, ..., j23
0-2     3-6              7-29 (23 joints)
```

**Important**: CSV must match the robot's joint count. You cannot use 29dof CSV directly with 23dof robot.

## Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Start with 29dof CSV from Unitree                       │
│    G1_Take_102.bvh_60hz.csv (36 columns)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Convert to 23dof CSV                                    │
│    Removes columns: 20, 21, 27, 28, 34, 35                 │
│    → G1_Take_102_23dof_60hz.csv (30 columns)               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Create NPZ with body positions via simulation           │
│    Runs Isaac Lab to compute forward kinematics            │
│    → G1_Take_102_23dof_60hz.npz                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Train policy                                            │
│    → logs/rsl_rl/.../model_*.pt                            │
│    → logs/rsl_rl/.../exported/policy.onnx                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Deploy to MuJoCo                                        │
│    Copy policy.onnx + CSV to deploy/robots/g1_23dof/       │
└─────────────────────────────────────────────────────────────┘
```

## Step-by-Step Commands

### Step 1: Convert 29dof CSV to 23dof

```bash
cd /home/dias/Documents/unitree/unitree_rl_lab
conda activate env_isaaclab

# Use existing conversion script
python source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_23dof/dance_102/convert_29dof_to_23dof.py
```

Or create your own conversion:

```python
import numpy as np

# Load 29dof CSV
data = np.loadtxt("G1_Take_102.bvh_60hz.csv", delimiter=",")

# Columns to remove (0-indexed, after 7 root columns):
# waist_roll (13), waist_pitch (14),
# left_wrist_pitch (20), left_wrist_yaw (21),
# right_wrist_pitch (27), right_wrist_yaw (28)
# In CSV: add 7 to get actual column indices
cols_to_remove = [20, 21, 27, 28, 34, 35]  # 7+13, 7+14, 7+20, 7+21, 7+27, 7+28

data_23dof = np.delete(data, cols_to_remove, axis=1)
np.savetxt("G1_Take_102_23dof_60hz.csv", data_23dof, delimiter=",")
```

### Step 2: Create NPZ with Forward Kinematics

```bash
cd /home/dias/Documents/unitree/unitree_rl_lab
conda activate env_isaaclab

python scripts/mimic/csv_to_npz.py \
  -f source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_23dof/dance_102/G1_Take_102_23dof_60hz.csv \
  --input_fps 60 \
  --robot g1_23dof
```

This will:
- Load the 23dof CSV
- Spawn G1-23dof robot in Isaac Lab
- Replay the motion frame by frame
- Record `body_pos_w`, `body_quat_w` from forward kinematics
- Save to NPZ file

### Step 3: Train the Policy

```bash
# With visualization (slower)
python scripts/rsl_rl/train.py --task Unitree-G1-23dof-Mimic-Dance-102

# Headless (faster, recommended)
python scripts/rsl_rl/train.py --task Unitree-G1-23dof-Mimic-Dance-102 --headless --num_envs 4096
```

### Step 4: Test the Trained Policy

```bash
python scripts/rsl_rl/play.py --task Unitree-G1-23dof-Mimic-Dance-102
```

### Step 5: Deploy to MuJoCo

```bash
# Create policy directory
mkdir -p deploy/robots/g1_23dof/config/policy/mimic/dance_102/exported
mkdir -p deploy/robots/g1_23dof/config/policy/mimic/dance_102/params

# Copy trained policy
cp logs/rsl_rl/unitree_g1_23dof_mimic_dance_102/*/exported/policy.onnx \
   deploy/robots/g1_23dof/config/policy/mimic/dance_102/exported/

# Copy motion CSV for reference
cp source/.../G1_Take_102_23dof_60hz.csv \
   deploy/robots/g1_23dof/config/policy/mimic/dance_102/params/
```

## Robot Configurations

### UNITREE_G1_23DOF_CFG vs UNITREE_G1_23DOF_MIMIC_CFG

Located in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`:

| Config | Purpose | joint_sdk_names |
|--------|---------|-----------------|
| `UNITREE_G1_23DOF_CFG` | SDK communication | Has empty strings for index alignment |
| `UNITREE_G1_23DOF_MIMIC_CFG` | Mimic training | Clean 23 joint names |

**For mimic tasks, always use `UNITREE_G1_23DOF_MIMIC_CFG`**

### Joint Order (23dof)

```python
joint_sdk_names = [
    # Left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (1)
    "waist_yaw_joint",
    # Left arm (5)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    # Right arm (5)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]
```

### Body Names for Tracking

```python
body_names = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_rubber_hand",  # Note: rubber_hand, not link
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_rubber_hand",
]
```

## File Locations

```
unitree_rl_lab/
├── scripts/mimic/
│   └── csv_to_npz.py                    # Supports --robot g1_23dof
│
├── source/unitree_rl_lab/unitree_rl_lab/
│   ├── assets/robots/unitree.py         # Robot configs
│   │   ├── UNITREE_G1_23DOF_CFG
│   │   ├── UNITREE_G1_23DOF_MIMIC_CFG
│   │   └── UNITREE_G1_23DOF_MIMIC_ACTION_SCALE
│   │
│   └── tasks/mimic/robots/g1_23dof/
│       ├── __init__.py
│       └── dance_102/
│           ├── __init__.py              # Registers task
│           ├── tracking_env_cfg.py      # Environment config
│           ├── convert_29dof_to_23dof.py
│           ├── create_npz_23dof.py      # Manual NPZ (deprecated)
│           ├── G1_Take_102_23dof_60hz.csv
│           └── G1_Take_102_23dof_60hz.npz
│
└── deploy/robots/g1_23dof/
    ├── main.cpp                         # mode_machine = 4
    ├── config/config.yaml
    └── config/policy/                   # Deploy trained policies here
```

## Registered Tasks

| Task ID | Robot | Motion |
|---------|-------|--------|
| `Unitree-G1-23dof-Mimic-Dance-102` | G1-23dof | Dance 102 |

## Adding New Dances for 23dof

To add Gangnam Style or other dances:

1. Convert the 29dof CSV:
```bash
# Create directory
mkdir -p source/.../tasks/mimic/robots/g1_23dof/gangnanm_style/

# Convert CSV (modify script for new file)
python convert_29dof_to_23dof.py \
  --input G1_gangnam_style_V01.bvh_60hz.csv \
  --output G1_gangnam_style_23dof_60hz.csv
```

2. Create NPZ:
```bash
python scripts/mimic/csv_to_npz.py \
  -f .../G1_gangnam_style_23dof_60hz.csv \
  --input_fps 60 \
  --robot g1_23dof
```

3. Create `tracking_env_cfg.py` (copy from dance_102, update paths)

4. Create `__init__.py` to register the task

5. Update `g1_23dof/__init__.py` to import new dance

## Troubleshooting

### Error: Empty joint names
```
ValueError: Not all regular expressions are matched!
        : []
```
**Cause**: Using `UNITREE_G1_23DOF_CFG` instead of `UNITREE_G1_23DOF_MIMIC_CFG`
**Fix**: Use MIMIC version in csv_to_npz.py

### Error: Body name not found
```
ValueError: ... left_wrist_roll_link ...
```
**Cause**: 23dof uses `left_wrist_roll_rubber_hand` not `left_wrist_roll_link`
**Fix**: Update body_names in tracking_env_cfg.py

### Error: CUDA device-side assert
```
CUDA error: device-side assert triggered
```
**Cause**: NPZ has zero body_pos_w (created without simulation)
**Fix**: Recreate NPZ using `csv_to_npz.py --robot g1_23dof`

### Error: Column count mismatch
```
ValueError: could not broadcast input array
```
**Cause**: Using 29dof CSV with 23dof robot
**Fix**: Convert CSV first with `convert_29dof_to_23dof.py`
