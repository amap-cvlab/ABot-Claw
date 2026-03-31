# 🦞 ABot-Claw

**AMAP CV Lab**

**ABot-Claw** is a next-generation embodied AI framework built on OpenClaw that unifies VLN, VLA, and WAM models through the **VLAC (Vision-Language-Action-Critic)** loop, enabling real-time task monitoring and self-adaptation. By sharing a **multimodal memory**—centered on vision, semantics, and geometry—across devices and leveraging centralized control, it breaks down the silos of single-task robots, delivering a robust, collaborative, and elastically scalable **multi-agent system**.

---

## ✨ Core Features

### 🧠 1. VLAC: Task Progress Feedback Mechanism

Introduces the **VLAC (Vision-Language-Action-Critic) closed-loop feedback mechanism** for dynamic task control.  

- **Real-time Assessment:** Agents assess task completion and execution progress.  
- **Adaptive Adjustment:** VLAC provides feedback and triggers strategy updates when deviations occur, enhancing success rates and robustness in long-horizon tasks.

### 🔗 2. End-to-End Closed-Loop Interaction

Supports **VLA (Vision-Language-Action)** and **WAM (World Action Model)** for full-cycle autonomy.  

- **Seamless Integration:** Tightly aligns perception with action to accurately follow natural language instructions.  
- **High Autonomy:** Executes complex multi-step tasks end-to-end without human intervention.

### 👥 3. Multi-Robot Collaboration & Elastic Architecture

Enables decentralized collaboration via a shared "Brain."  

- **Unified Decision Making:** All robots share one Agent Runtime for synchronized state and joint reasoning.  
- **Hot-Swappable Support:** Robots can join or replace others anytime without disrupting task flow.

### 📸 4. Vision-Centric Memory Mechanism

Features a unified **Memory System** for persistent knowledge storage.  

- **Integrated Structure:** Combines geometric maps (for localization) and semantic maps / image-feature-GPS (for understanding).  
- **Long-Context Understanding:** Retains and retrieves key visual history to overcome occlusion and delayed feedback.

---

## 🏗️ System Architecture

ABot-Claw employs a layered microservices architecture to ensure high cohesion and low coupling:

1. **Infrastructure Layer:**
  - **GPU Server:** Hosts high-performance computing models including Yolo, Depth, VLA, Grasp Anything, VLN, and WAM.
  - **Robots & Cameras:** Physical terminals including Dog, G1, PiPer, and various camera devices.
2. **Runtime Core (Agent Runtime):**
  - **Gateway:** Handles message routing for CLI/Web UI and Channels (Telegram, DingTalk, Feishu).
  - **Agent Loop:** The core intelligence cycle containing Context, Tools, and Skills.
  - **Scheduler & Device:** Manages task scheduling (Heartbeat/Cron) and local device interaction (File System, Shell, Browser).
3. **Memory & Knowledge:**
  - **Vision-centric Memory:** A central repository at the top layer storing geometric maps, semantic maps, and image feature indices for global access.

---

## 🚀 Quick Start

### Server Side

Pleace refer to the installation instructions in each service.

### Robot Side

1. Install Python dependencies:

```bash
cd robot_client/arm_piper/agent_server
pip3 install -r requirements.txt
```

2. Build ROS driver in `robot_driver_ros`: (Download the corresponding ROS driver for your robot.)

```bash
cd robot_driver_ros/src
git clone https://github.com/agilexrobotics/piper_ros.git
git clone https://github.com/realsenseai/realsense-ros.git -b ros1-legacy
# Install the dependencies required for the corresponding driver.

cd ../..
catkin_make
```

3. Update robot topics and source path in `robot_client/arm_piper/agent_server/robot_sdk/config.yaml`:

```yaml
ros:
  image_topic: "/your_camera/color/image_raw"
  depth_topic: "/your_camera/aligned_depth_to_color/image_raw"
  camera_info_topic: "/your_camera/color/camera_info"
  joint_state_topic: "/your_joint_states_topic"
  end_pose_topic: "/your_end_pose_topic"

piper:
  setup_bash: "/absolute_path_to/robot_client/arm_piper/agent_server/robot_driver_ros/devel/setup.bash"
```

> Please modify the ROS topics according to your robot setup, and make sure `setup_bash` points to your local `devel/setup.bash`.

4. Launch components in order:
  - Arm driver
    ```bash
    cd robot_client/arm_piper/agent_server/robot_driver_ros/src/piper_ros
    ./can_activate.sh

    cd ../..
    source devel/setup.bash

    roslaunch piper start_single_piper.launch can_port:=can0 auto_enable:=true
    ```
  - Camera driver
    ```bash
    cd robot_client/arm_piper/agent_server/robot_driver_ros
    source devel/setup.bash
    roslaunch realsense2_camera rs_rgbd.launch
    ```
  - Arm MoveIt
    ```bash
    cd robot_client/arm_piper/agent_server/robot_driver_ros
    source devel/setup.bash
    roslaunch piper_with_gripper_moveit demo.launch use_rviz:=false
    ```
5. After all three components are running, start the robot agent server:

```bash
cd robot_client/arm_piper/agent_server
python3 server.py --port 8888
```

---

## 🙏 Acknowledgement

This project builds upon the following open-source projects. We thank these teams for their contributions:

- [Tidybot-Universe](https://github.com/TidyBot-Services/Tidybot-Universe.git)

---

