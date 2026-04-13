# ROS Driver 安装说明

本目录用于放置机器人相关 ROS 驱动工作空间代码。  
请前往对应机器人/相机厂商的官方仓库拉取驱动，并严格按照官方文档完成依赖安装与配置。

## 推荐流程

1. 进入 `src` 目录拉取驱动代码：

```bash
cd robot_client/arm_piper/agent_server/robot_driver_ros/src

# 示例（请按你的设备替换为实际仓库）
git clone <ROBOT_DRIVER_REPO_URL>
git clone <CAMERA_DRIVER_REPO_URL>
```

2. 回到工作空间根目录编译：

```bash
cd ..
catkin_make
```

3. 编译完成后，按各驱动仓库文档分别启动：

- 机械臂驱动
- 相机驱动
- MoveIt（如该驱动提供）

## 注意事项

- 不同型号机器人、相机、ROS 版本（如 Noetic/Melodic）对应的驱动分支不同，请使用官方推荐分支。
- 若官方仓库要求额外依赖（`rosdep`、系统库、udev、CAN 配置等），请先完成再编译。

## 参考
- piper机械臂驱动仓库：`git clone https://github.com/agilexrobotics/piper_ros`
- realsense相机驱动仓库：`git clone https://github.com/realsenseai/realsense-ros.git -b ros1-legacy`
