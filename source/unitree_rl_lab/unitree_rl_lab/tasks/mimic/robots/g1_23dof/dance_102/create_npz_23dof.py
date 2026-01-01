"""
Create NPZ file for G1 23dof from CSV.
This is a simplified version that creates motion data without full simulation.
"""

import numpy as np
import os
from scipy.spatial.transform import Rotation

# G1 23dof joint order (matching joint_sdk_names in unitree.py)
JOINT_NAMES_23DOF = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]

# Body names for motion tracking
BODY_NAMES = [
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
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
]

INPUT_FPS = 60
OUTPUT_FPS = 50


def interpolate_motion(data, input_fps, output_fps):
    """Interpolate motion data to target fps."""
    n_frames = data.shape[0]
    duration = (n_frames - 1) / input_fps
    n_output_frames = int(duration * output_fps) + 1

    input_times = np.linspace(0, duration, n_frames)
    output_times = np.linspace(0, duration, n_output_frames)

    output_data = np.zeros((n_output_frames, data.shape[1]))
    for i in range(data.shape[1]):
        output_data[:, i] = np.interp(output_times, input_times, data[:, i])

    return output_data


def create_npz(csv_path: str, output_path: str):
    """Create NPZ from 23dof CSV."""

    # Load CSV
    data = np.loadtxt(csv_path, delimiter=',')
    print(f"Loaded CSV: {data.shape} (frames x columns)")

    n_frames = data.shape[0]
    n_joints = 23

    # Parse CSV columns
    # 0-2: x, y, z (root position)
    # 3-6: qx, qy, qz, qw (root orientation as quaternion)
    # 7-29: joint positions (23 joints)

    root_pos = data[:, 0:3]  # (n_frames, 3)
    root_quat = data[:, 3:7]  # (n_frames, 4) - qx, qy, qz, qw
    joint_pos = data[:, 7:30]  # (n_frames, 23)

    print(f"Root position range: {root_pos.min(axis=0)} to {root_pos.max(axis=0)}")
    print(f"Joint positions shape: {joint_pos.shape}")

    # Interpolate to output fps
    duration = (n_frames - 1) / INPUT_FPS
    n_output = int(duration * OUTPUT_FPS) + 1

    root_pos_interp = interpolate_motion(root_pos, INPUT_FPS, OUTPUT_FPS)
    root_quat_interp = interpolate_motion(root_quat, INPUT_FPS, OUTPUT_FPS)
    joint_pos_interp = interpolate_motion(joint_pos, INPUT_FPS, OUTPUT_FPS)

    # Normalize quaternions
    root_quat_interp = root_quat_interp / np.linalg.norm(root_quat_interp, axis=1, keepdims=True)

    # Compute velocities (finite differences)
    dt = 1.0 / OUTPUT_FPS

    # Root linear velocity
    root_lin_vel = np.zeros_like(root_pos_interp)
    root_lin_vel[1:] = (root_pos_interp[1:] - root_pos_interp[:-1]) / dt
    root_lin_vel[0] = root_lin_vel[1]

    # Root angular velocity (from quaternion differences)
    root_ang_vel = np.zeros((n_output, 3))
    for i in range(1, n_output):
        q1 = root_quat_interp[i - 1]
        q2 = root_quat_interp[i]
        # Convert to rotation objects
        r1 = Rotation.from_quat(q1)  # scipy uses [x,y,z,w]
        r2 = Rotation.from_quat(q2)
        # Relative rotation
        r_diff = r2 * r1.inv()
        # Convert to axis-angle and get angular velocity
        rotvec = r_diff.as_rotvec()
        root_ang_vel[i] = rotvec / dt
    root_ang_vel[0] = root_ang_vel[1]

    # Joint velocity
    joint_vel = np.zeros_like(joint_pos_interp)
    joint_vel[1:] = (joint_pos_interp[1:] - joint_pos_interp[:-1]) / dt
    joint_vel[0] = joint_vel[1]

    # Create placeholder body positions/orientations
    # In a full implementation, these would come from forward kinematics
    n_bodies = len(BODY_NAMES)
    body_pos = np.zeros((n_output, n_bodies, 3))
    body_quat = np.zeros((n_output, n_bodies, 4))
    body_quat[:, :, 3] = 1.0  # w=1 for identity quaternion

    body_lin_vel = np.zeros((n_output, n_bodies, 3))
    body_ang_vel = np.zeros((n_output, n_bodies, 3))

    # Set root body (pelvis) from root motion
    body_pos[:, 0, :] = root_pos_interp
    body_quat[:, 0, :] = root_quat_interp
    body_lin_vel[:, 0, :] = root_lin_vel
    body_ang_vel[:, 0, :] = root_ang_vel

    # Save NPZ with correct keys for MotionLoader
    np.savez(
        output_path,
        fps=OUTPUT_FPS,
        joint_names=JOINT_NAMES_23DOF,
        body_names=BODY_NAMES,
        # Keys expected by MotionLoader
        joint_pos=joint_pos_interp.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
    )

    print(f"\nSaved NPZ to: {output_path}")
    print(f"  Duration: {duration:.2f} sec")
    print(f"  Frames: {n_output} @ {OUTPUT_FPS} fps")
    print(f"  Joints: {n_joints}")
    print(f"  Bodies: {n_bodies}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "G1_Take_102_23dof_60hz.csv")
    npz_path = os.path.join(script_dir, "G1_Take_102_23dof_60hz.npz")

    create_npz(csv_path, npz_path)
