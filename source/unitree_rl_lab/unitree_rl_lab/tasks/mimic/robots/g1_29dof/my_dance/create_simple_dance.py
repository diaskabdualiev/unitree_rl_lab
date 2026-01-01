#!/usr/bin/env python3
"""
Create a simple dance motion for G1-29dof robot.
Output: my_dance.csv (60 FPS, 10 seconds)
"""

import numpy as np
import os

# Dance parameters
FPS = 60
DURATION = 10  # seconds
NUM_FRAMES = FPS * DURATION

# G1-29dof joint indices
JOINTS = {
    # Left leg
    "left_hip_pitch": 0, "left_hip_roll": 1, "left_hip_yaw": 2,
    "left_knee": 3, "left_ankle_pitch": 4, "left_ankle_roll": 5,
    # Right leg
    "right_hip_pitch": 6, "right_hip_roll": 7, "right_hip_yaw": 8,
    "right_knee": 9, "right_ankle_pitch": 10, "right_ankle_roll": 11,
    # Torso
    "waist_yaw": 12, "waist_roll": 13, "waist_pitch": 14,
    # Left arm
    "left_shoulder_pitch": 15, "left_shoulder_roll": 16, "left_shoulder_yaw": 17,
    "left_elbow": 18, "left_wrist_roll": 19, "left_wrist_pitch": 20, "left_wrist_yaw": 21,
    # Right arm
    "right_shoulder_pitch": 22, "right_shoulder_roll": 23, "right_shoulder_yaw": 24,
    "right_elbow": 25, "right_wrist_roll": 26, "right_wrist_pitch": 27, "right_wrist_yaw": 28,
}

# Default standing pose (from FixStand)
DEFAULT_POSE = np.array([
    # Left leg
    -0.1, 0, 0, 0.3, -0.2, 0,
    # Right leg
    -0.1, 0, 0, 0.3, -0.2, 0,
    # Torso
    0, 0, 0,
    # Left arm
    0, 0.25, 0, 0.97, 0.15, 0, 0,
    # Right arm
    0, -0.25, 0, 0.97, -0.15, 0, 0,
])

def create_dance():
    """Create a simple arm wave dance."""
    frames = []

    for i in range(NUM_FRAMES):
        t = i / FPS  # time in seconds

        # Base position (slight up-down motion)
        base_x = 0.0
        base_y = 0.0
        base_z = 0.79 + 0.02 * np.sin(2 * np.pi * t * 2)  # 2 Hz bounce

        # Base orientation (quaternion xyzw, slight rotation)
        angle = 0.1 * np.sin(2 * np.pi * t * 0.5)  # slow sway
        qx, qy, qz, qw = 0, 0, np.sin(angle/2), np.cos(angle/2)

        # Joint positions
        joints = DEFAULT_POSE.copy()

        # Arm wave animation
        wave_phase = 2 * np.pi * t * 1.5  # 1.5 Hz wave

        # Left arm wave
        joints[JOINTS["left_shoulder_pitch"]] = 0.5 + 0.5 * np.sin(wave_phase)
        joints[JOINTS["left_shoulder_roll"]] = 0.3 + 0.2 * np.sin(wave_phase + np.pi/4)
        joints[JOINTS["left_elbow"]] = 0.5 + 0.3 * np.sin(wave_phase + np.pi/2)

        # Right arm wave (opposite phase)
        joints[JOINTS["right_shoulder_pitch"]] = 0.5 + 0.5 * np.sin(wave_phase + np.pi)
        joints[JOINTS["right_shoulder_roll"]] = -0.3 - 0.2 * np.sin(wave_phase + np.pi + np.pi/4)
        joints[JOINTS["right_elbow"]] = 0.5 + 0.3 * np.sin(wave_phase + np.pi + np.pi/2)

        # Torso twist
        joints[JOINTS["waist_yaw"]] = 0.15 * np.sin(wave_phase)

        # Knee bounce
        bounce = 0.05 * np.sin(2 * np.pi * t * 2)
        joints[JOINTS["left_knee"]] = 0.3 + bounce
        joints[JOINTS["right_knee"]] = 0.3 + bounce

        # Combine into frame
        frame = np.concatenate([
            [base_x, base_y, base_z],  # position
            [qx, qy, qz, qw],          # quaternion
            joints                      # 29 joints
        ])
        frames.append(frame)

    return np.array(frames)


if __name__ == "__main__":
    print("Creating simple dance motion...")
    motion = create_dance()

    # Save CSV
    output_file = os.path.join(os.path.dirname(__file__), "my_dance_60hz.csv")
    np.savetxt(output_file, motion, delimiter=",", fmt="%.6f")

    print(f"Saved: {output_file}")
    print(f"Frames: {motion.shape[0]}, Columns: {motion.shape[1]}")
    print(f"Duration: {motion.shape[0] / FPS:.1f} seconds at {FPS} FPS")
