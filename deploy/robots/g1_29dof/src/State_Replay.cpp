#include "State_Replay.h"
#include "param.h"

// ============== MotionLoader Implementation ==============

State_Replay::MotionLoader::MotionLoader(const std::string& csv_path, float fps)
    : dt_(1.0f / fps)
{
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + csv_path);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stof(cell));
        }

        // CSV format: root_pos(3) + root_quat(4) + joint_pos(29) = 36 columns
        // We only need joint positions (columns 7-35)
        if (row.size() >= 36) {
            Eigen::VectorXf joints(29);
            for (int i = 0; i < 29; ++i) {
                joints[i] = row[7 + i];
            }
            joint_positions_.push_back(joints);
        }
    }

    num_frames_ = joint_positions_.size();
    num_joints_ = 29;
    duration_ = num_frames_ * dt_;

    spdlog::info("[MotionLoader] Loaded {} frames, duration: {:.2f}s, {} joints",
                 num_frames_, duration_, num_joints_);
}

Eigen::VectorXf State_Replay::MotionLoader::get_joint_pos(float time)
{
    if (joint_positions_.empty()) {
        return Eigen::VectorXf::Zero(num_joints_);
    }

    float phase = std::clamp(time / duration_, 0.0f, 1.0f);
    int idx0 = static_cast<int>(phase * (num_frames_ - 1));
    int idx1 = std::min(idx0 + 1, num_frames_ - 1);
    float blend = (time - idx0 * dt_) / dt_;
    blend = std::clamp(blend, 0.0f, 1.0f);

    return joint_positions_[idx0] * (1.0f - blend) + joint_positions_[idx1] * blend;
}

// ============== State_Replay Implementation ==============

State_Replay::State_Replay(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];

    // Load motion file
    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if (!motion_file.is_absolute()) {
        motion_file = param::proj_dir / motion_file;
    }

    float fps = cfg["fps"] ? cfg["fps"].as<float>() : 60.0f;
    motion_loader_ = std::make_unique<MotionLoader>(motion_file.string(), fps);

    spdlog::info("[State_Replay] Loaded motion '{}' duration={:.2f}s",
                 motion_file.stem().string(), motion_loader_->duration());

    // Time range
    time_start_ = cfg["time_start"] ? cfg["time_start"].as<float>() : 0.0f;
    time_end_ = cfg["time_end"] ? cfg["time_end"].as<float>() : motion_loader_->duration();
    time_start_ = std::clamp(time_start_, 0.0f, motion_loader_->duration());
    time_end_ = std::clamp(time_end_, time_start_, motion_loader_->duration());

    // Control parameters
    control_dt_ = cfg["control_dt"] ? cfg["control_dt"].as<float>() : 0.02f; // 50Hz default
    loop_ = cfg["loop"] ? cfg["loop"].as<bool>() : false;
    blend_duration_ = cfg["blend_duration"] ? cfg["blend_duration"].as<float>() : 1.0f;

    // PD gains (29 joints for G1-29dof)
    if (cfg["kp"]) {
        kp_ = cfg["kp"].as<std::vector<float>>();
    } else {
        kp_ = std::vector<float>(29, 50.0f); // default kp
    }

    if (cfg["kd"]) {
        kd_ = cfg["kd"].as<std::vector<float>>();
    } else {
        kd_ = std::vector<float>(29, 3.0f); // default kd
    }

    spdlog::info("[State_Replay] time_range=[{:.2f}, {:.2f}], loop={}, blend={:.2f}s",
                 time_start_, time_end_, loop_, blend_duration_);
}

void State_Replay::enter()
{
    spdlog::info("[State_Replay] enter()");

    // Set PD gains
    for (int i = 0; i < 29; ++i) {
        lowcmd->msg_.motor_cmd()[i].kp() = kp_[i];
        lowcmd->msg_.motor_cmd()[i].kd() = kd_[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    // Capture initial pose for blending
    initial_pose_.resize(29);
    for (int i = 0; i < 29; ++i) {
        initial_pose_[i] = lowstate->msg_.motor_state()[i].q();
    }

    // Start replay thread
    replay_thread_running.store(true);
    replay_thread = std::thread([this]() {
        spdlog::info("[State_Replay] replay_thread started");

        using clock = std::chrono::high_resolution_clock;
        const auto dt = std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(control_dt_));

        auto start_time = clock::now();
        auto next_tick = start_time + dt;

        float motion_time = time_start_;
        float elapsed = 0.0f;

        while (replay_thread_running.load()) {
            elapsed = std::chrono::duration<float>(clock::now() - start_time).count();

            // Blending phase
            if (elapsed < blend_duration_) {
                float blend_alpha = elapsed / blend_duration_;
                // Smooth blend (ease-in-out)
                blend_alpha = blend_alpha * blend_alpha * (3.0f - 2.0f * blend_alpha);

                Eigen::VectorXf target = motion_loader_->get_joint_pos(time_start_);
                Eigen::VectorXf blended = initial_pose_ * (1.0f - blend_alpha) + target * blend_alpha;

                for (int i = 0; i < 29; ++i) {
                    lowcmd->msg_.motor_cmd()[i].q() = blended[i];
                }
            } else {
                // Motion playback phase
                motion_time = time_start_ + (elapsed - blend_duration_);

                if (motion_time > time_end_) {
                    if (loop_) {
                        // Reset for looping
                        start_time = clock::now() - std::chrono::duration_cast<clock::duration>(
                            std::chrono::duration<double>(blend_duration_));
                        motion_time = time_start_;
                    } else {
                        // Stay at end position
                        motion_time = time_end_;
                    }
                }

                Eigen::VectorXf target = motion_loader_->get_joint_pos(motion_time);
                for (int i = 0; i < 29; ++i) {
                    lowcmd->msg_.motor_cmd()[i].q() = target[i];
                }
            }

            std::this_thread::sleep_until(next_tick);
            next_tick += dt;
        }

        spdlog::info("[State_Replay] replay_thread exiting");
    });
}

void State_Replay::run()
{
    // Commands are set in replay_thread
    // This function is called by FSM at its own rate
}
