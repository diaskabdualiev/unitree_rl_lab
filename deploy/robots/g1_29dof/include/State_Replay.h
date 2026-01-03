#pragma once

#include <atomic>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include "FSM/FSMState.h"

class State_Replay : public FSMState
{
public:
    State_Replay(int state_mode, std::string state_string);

    void enter() override;
    void run() override;
    void exit() override
    {
        spdlog::info("[State_Replay] exit() called, stopping thread...");
        replay_thread_running.store(false);
        if (replay_thread.joinable()) {
            replay_thread.join();
        }
        spdlog::info("[State_Replay] exit() done");
    }

    // Simple motion loader for CSV replay
    class MotionLoader
    {
    public:
        MotionLoader(const std::string& csv_path, float fps);

        // Get interpolated joint positions at time t
        Eigen::VectorXf get_joint_pos(float time);

        float duration() const { return duration_; }
        int num_joints() const { return num_joints_; }

    private:
        std::vector<Eigen::VectorXf> joint_positions_;
        float dt_;
        float duration_;
        int num_joints_;
        int num_frames_;
    };

private:
    std::unique_ptr<MotionLoader> motion_loader_;

    std::thread replay_thread;
    std::atomic<bool> replay_thread_running{false};

    float time_start_;
    float time_end_;
    float control_dt_;
    bool loop_;

    std::vector<float> kp_;
    std::vector<float> kd_;

    // Blend from current pose to first frame
    float blend_duration_;
    Eigen::VectorXf initial_pose_;
};

REGISTER_FSM(State_Replay)
