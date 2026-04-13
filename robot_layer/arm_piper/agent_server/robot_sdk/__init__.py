"""Robot SDK — Piper arm control + perception.

所有 SDK 共享同一份 config.yaml 配置文件。
切换到不同机器人时，只需修改 config.yaml 中的话题 / 坐标系 / 服务地址。
也可通过环境变量 ROBOT_SDK_CONFIG 指定配置文件路径。

In submitted code (via /code/execute), the wrapper pre-creates:
  - ``env``   — PiperRobotEnv   (MoveIt arm/gripper control + cameras)
  - ``yolo``  — YoloSDK         (YOLOv5 object detection + 3D localization)
  - ``grasp`` — GraspSDK        (AnyGrasp 6DoF grasp pose generation)

Architecture:
  - env runs in the main process under /usr/bin/python3 (ROS/MoveIt environment)
  - yolo calls a remote YOLO HTTP detection service for pixel bboxes,
    then performs 3D projection locally using ROS depth/camera_info/TF.
  - grasp calls a remote Grasp HTTP service for YOLO detection + AnyGrasp
    6DoF grasp pose generation.
  - Both yolo and grasp run directly in the main process via HTTP,
    no subprocess or anygraspenv needed.

Example usage in submitted code:

    # ---- 机器人控制 (env) ----
    state = env.get_robot_state()
    env.move_joints([0.0, 0.08, -0.32, -0.02, 1.06, -0.034])
    env.set_gripper(0.04)
    env.move_to_pose([0.2, -0.05, 0.1, 0.0, 1.0, 0.0, 0.0])
    env.reset()
    images, ts = env.read_cameras()
    pose = env.get_robot_end_pose()

    # ---- 目标检测 (yolo, HTTP API + ROS) ----
    labels = yolo.detect_env()
    detections = yolo.segment_3d("bottle")

    # ---- 抓取位姿 (grasp, HTTP API + ROS) ----
    results = grasp.get_grasp_pose("bottle", top_k=5)
    best = results[0]["grasps"][0]
    endpose = best["translation_base_retreat"] + best["quaternion_base"]
    env.move_to_pose(endpose)
"""
