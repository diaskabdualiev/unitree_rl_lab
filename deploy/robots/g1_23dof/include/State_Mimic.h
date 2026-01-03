#pragma once

#include <atomic>
#include "FSM/State_RLBase.h"

class State_Mimic : public FSMState
{
public:
    State_Mimic(int state_mode, std::string state_string);

    void enter();

    void run();

    void exit()
    {
        spdlog::info("[State_Mimic] exit() called, stopping thread...");
        policy_thread_running.store(false);
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
        spdlog::info("[State_Mimic] exit() done, thread joined");
    }

    class MotionLoader_;

    static std::shared_ptr<MotionLoader_> motion; // for obs computation
private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::shared_ptr<MotionLoader_> motion_; // for saving

    std::thread policy_thread;
    std::atomic<bool> policy_thread_running{false};
    std::array<float, 2> time_range_;
};

/**
 * Motion loader for G1-23DOF mimic policy deployment.
 *
 * IMPORTANT: CSV file MUST be in Isaac Lab joint order (not SDK order)!
 * Use scripts/mimic/convert_29dof_to_23dof_isaac_order.py to convert.
 *
 * CSV format: 30 columns = 7 pose (x,y,z,qx,qy,qz,qw) + 23 joints (Isaac Lab order)
 * Isaac Lab order uses joint_ids_map: [0,6,12,1,7,15,22,2,8,16,23,3,9,17,24,4,10,18,25,5,11,19,26]
 */
class State_Mimic::MotionLoader_
{
public:
    MotionLoader_(std::string motion_file, float fps)
    : dt(1.0f / fps)
    {
        auto data = isaaclab::load_csv(motion_file);

        num_frames = data.size();
        duration = num_frames * dt;

        // CSV columns 7+ contain joint positions in Isaac Lab order (NOT SDK order!)
        // No remapping needed here - the CSV should already be in correct order
        for(int i(0); i < num_frames; ++i)
        {
            root_positions.push_back(Eigen::VectorXf::Map(data[i].data(), 3));
            root_quaternions.push_back(Eigen::Quaternionf(data[i][6],data[i][3], data[i][4], data[i][5]));
            dof_positions.push_back(Eigen::VectorXf::Map(data[i].data() + 7, data[i].size() - 7));
        }
        dof_velocities = _comupte_raw_derivative(dof_positions);

        update(0.0f);
    }

    void update(float time) 
    {
        float phase = std::clamp(time / duration, 0.0f, 1.0f);
        index_0_ = std::round(phase * (num_frames - 1));
        index_1_ = std::min(index_0_ + 1, num_frames - 1);
        blend_ = std::round((time - index_0_ * dt) / dt * 1e5f) / 1e5f;
    }

    void reset(const isaaclab::ArticulationData & data, float t = 0.0f)
    {
        update(t);
        auto init_to_anchor = isaaclab::yawQuaternion(this->root_quaternion()).toRotationMatrix();
        auto world_to_anchor = isaaclab::yawQuaternion(data.root_quat_w).toRotationMatrix();
        world_to_init_ = world_to_anchor * init_to_anchor.transpose();
    }

    Eigen::VectorXf joint_pos() {
        return dof_positions[index_0_] * (1 - blend_) + dof_positions[index_1_] * blend_;
    }

    Eigen::VectorXf root_position() {
        return root_positions[index_0_] * (1 - blend_) + root_positions[index_1_] * blend_;
    }

    Eigen::VectorXf joint_vel() {
        return dof_velocities[index_0_] * (1 - blend_) + dof_velocities[index_1_] * blend_;
    }

    Eigen::Quaternionf root_quaternion() {
        return root_quaternions[index_0_].slerp(blend_, root_quaternions[index_1_]);
    }

    float dt;
    int num_frames;
    float duration;

    std::vector<Eigen::VectorXf> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;

    Eigen::Matrix3f world_to_init_;
private:
    int index_0_;
    int index_1_;
    float blend_;

    std::vector<Eigen::VectorXf> _comupte_raw_derivative(const std::vector<Eigen::VectorXf>& data)
    {
        std::vector<Eigen::VectorXf> derivative;
        for(int i = 0; i < data.size() - 1; ++i) {
            derivative.push_back((data[i + 1] - data[i]) / dt);
        }
        derivative.push_back(derivative.back());
        return derivative;
    }
};

REGISTER_FSM(State_Mimic)