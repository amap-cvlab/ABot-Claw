#!/usr/bin/env python
"""
Piper 机械臂 SDK —— 基于 MoveIt 服务的控制接口

暴露接口:
    - move_joints()           关节控制
    - move_to_pose()          末端位姿控制 (欧拉角 / 四元数)
    - set_gripper()           夹爪开合
    - reset()                 复位
    - read_cameras()          拍照
    - get_robot_state()       获取关节电机角度
    - get_robot_end_pose()    获取末端位姿
"""

import math
import os
import sys
import shlex
import time
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2
import yaml

# 在 ROS 环境 source 前就读取配置（此时 yaml 一定可用）
_ROBOT_SDK_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_piper_config():
    cfg_path = os.environ.get(
        "ROBOT_SDK_CONFIG",
        os.path.join(_ROBOT_SDK_DIR, "config.yaml"),
    )
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


# 默认：本仓库 agent_server/robot_driver_ros/devel/setup.bash（相对于 robot_sdk 目录）
_DEFAULT_SETUP_BASH = os.path.normpath(
    os.path.join(_ROBOT_SDK_DIR, "..", "robot_driver_ros", "devel", "setup.bash")
)

_boot_cfg = _load_piper_config()
_setup_raw = (_boot_cfg.get("piper") or {}).get("setup_bash")
if _setup_raw:
    SETUP_BASH = os.path.normpath(
        _setup_raw if os.path.isabs(_setup_raw) else os.path.join(_ROBOT_SDK_DIR, _setup_raw)
    )
else:
    SETUP_BASH = _DEFAULT_SETUP_BASH

if os.environ.get("PIPER_ENV_SOURCED") != "1":
    try:
        import moveit_ctrl.srv  # noqa: F401
    except Exception:
        os.environ["PIPER_ENV_SOURCED"] = "1"
        caller_script = os.path.abspath(sys.argv[0])
        args = " ".join(shlex.quote(a) for a in sys.argv[1:])
        cmd = (
            f"source {shlex.quote(SETUP_BASH)} >/dev/null 2>&1 && "
            f"exec {shlex.quote(sys.executable)} {shlex.quote(caller_script)} {args}"
        )
        os.execvp("bash", ["bash", "-lc", cmd])

import rospy
from geometry_msgs.msg import PoseStamped
from moveit_ctrl.srv import JointMoveitCtrl, JointMoveitCtrlRequest
from tf.transformations import quaternion_from_euler

from config import get_config
from piper_image_sdk import ImageRecorder, Recorder


@dataclass
class CameraFrame:
    device_id: str
    stream_type: str
    frame: np.ndarray
    timestamp: float
    width: int
    height: int
    depth_scale: Optional[float] = None


class PiperRobotEnv:
    """基于 MoveIt 的 Piper 机械臂统一控制环境

    所有参数均可省略，默认从 config.yaml 读取。
    """

    def __init__(
        self,
        max_velocity=None,
        max_acceleration=None,
        init_ros_node=True,
    ):
        cfg = get_config()
        piper_cfg = cfg.get("piper", {})
        ros_cfg = cfg.get("ros", {})

        self.RESET_JOINTS = piper_cfg.get("reset_joints", [0.0, 0.08, -0.32, -0.02, 1.06, -0.034])
        self.RESET_GRIPPER = piper_cfg.get("reset_gripper", 0.059)
        self.NUM_JOINTS = piper_cfg.get("num_joints", 6)
        self.GRIPPER_MIN = piper_cfg.get("gripper_min", 0.0)
        self.GRIPPER_MAX = piper_cfg.get("gripper_max", 0.06)

        self._joint_state_topic = ros_cfg.get("joint_state_topic", "/joint_states_single")
        self._end_pose_topic = ros_cfg.get("end_pose_topic", "/end_pose")

        if init_ros_node and not rospy.get_node_uri():
            rospy.init_node("piper_robot_env", anonymous=True)
            print("ROS node initialized: piper_robot_env")

        self.max_velocity = max_velocity if max_velocity is not None else piper_cfg.get("max_velocity", 0.5)
        self.max_acceleration = max_acceleration if max_acceleration is not None else piper_cfg.get("max_acceleration", 0.5)

        self._latest_camera_frames = {}
        self._camera_lock = threading.Lock()

        cameras_cfg = cfg.get("cameras", {})
        self._cam_alias = {
            name: info.get("alias", name)
            for name, info in cameras_cfg.items()
        }
        self._camera_meta = [
            {"device_id": self._cam_alias[name], "name": self._cam_alias[name]}
            for name, info in cameras_cfg.items()
            if info.get("enabled", False)
        ]

        self._init_robot_interface()
        self._init_cameras()

        print("PiperRobotEnv (MoveIt) initialized")

    # ------------------------------------------------------------------ #
    #                        初始化
    # ------------------------------------------------------------------ #

    def _init_robot_interface(self):
        self.joint_recorder = Recorder(
            "left", init_node=False,
            joint_state_topic=self._joint_state_topic,
        )

        self._end_pose = None
        rospy.Subscriber(self._end_pose_topic, PoseStamped, self._end_pose_cb)

        rospy.sleep(0.5)
        print("Robot ROS interface initialized")

    def _end_pose_cb(self, msg):
        self._end_pose = msg

    def _init_cameras(self):
        self.image_recorder = ImageRecorder(init_node=False)
        rospy.sleep(1.0)
        print("Cameras initialized")

    # ------------------------------------------------------------------ #
    #                   MoveIt 服务调用（内部）
    # ------------------------------------------------------------------ #

    def _call_moveit_arm(self, joint_states, max_velocity=None, max_acceleration=None):
        """通过 joint_moveit_ctrl_arm 服务控制 6 个关节"""
        vel = max_velocity or self.max_velocity
        acc = max_acceleration or self.max_acceleration

        rospy.wait_for_service("joint_moveit_ctrl_arm", timeout=5.0)
        srv = rospy.ServiceProxy("joint_moveit_ctrl_arm", JointMoveitCtrl)
        req = JointMoveitCtrlRequest()
        req.joint_states = list(joint_states)
        req.gripper = 0.0
        req.max_velocity = vel
        req.max_acceleration = acc

        resp = srv(req)
        if not resp.status:
            rospy.logwarn(f"moveit_ctrl_arm failed, error_code: {resp.error_code}")
        return resp.status

    def _call_moveit_gripper(self, gripper, max_velocity=None, max_acceleration=None):
        """通过 joint_moveit_ctrl_gripper 服务控制夹爪"""
        vel = max_velocity or self.max_velocity
        acc = max_acceleration or self.max_acceleration

        rospy.wait_for_service("joint_moveit_ctrl_gripper", timeout=5.0)
        srv = rospy.ServiceProxy("joint_moveit_ctrl_gripper", JointMoveitCtrl)
        req = JointMoveitCtrlRequest()
        req.joint_states = [0.0] * self.NUM_JOINTS
        req.gripper = float(np.clip(gripper, self.GRIPPER_MIN, self.GRIPPER_MAX))
        req.max_velocity = vel
        req.max_acceleration = acc

        resp = srv(req)
        if not resp.status:
            rospy.logwarn(f"moveit_ctrl_gripper failed, error_code: {resp.error_code}")
        return resp.status

    def _call_moveit_piper(self, joint_states, gripper, max_velocity=None, max_acceleration=None):
        """通过 joint_moveit_ctrl_piper 服务同时控制关节 + 夹爪"""
        vel = max_velocity or self.max_velocity
        acc = max_acceleration or self.max_acceleration

        rospy.wait_for_service("joint_moveit_ctrl_piper", timeout=5.0)
        srv = rospy.ServiceProxy("joint_moveit_ctrl_piper", JointMoveitCtrl)
        req = JointMoveitCtrlRequest()
        req.joint_states = list(joint_states)
        req.gripper = float(np.clip(gripper, self.GRIPPER_MIN, self.GRIPPER_MAX))
        req.max_velocity = vel
        req.max_acceleration = acc

        resp = srv(req)
        if not resp.status:
            rospy.logwarn(f"moveit_ctrl_piper failed, error_code: {resp.error_code}")
        return resp.status

    def _call_moveit_endpose(self, endpose, max_velocity=None, max_acceleration=None):
        """
        通过 joint_moveit_ctrl_endpose 服务控制末端位姿
        endpose: 6 元素 [x,y,z,roll,pitch,yaw] 或 7 元素 [x,y,z,qx,qy,qz,qw]
        """
        vel = max_velocity or self.max_velocity
        acc = max_acceleration or self.max_acceleration

        rospy.wait_for_service("joint_moveit_ctrl_endpose", timeout=5.0)
        srv = rospy.ServiceProxy("joint_moveit_ctrl_endpose", JointMoveitCtrl)
        req = JointMoveitCtrlRequest()
        req.joint_states = [0.0] * self.NUM_JOINTS
        req.gripper = 0.0
        req.max_velocity = vel
        req.max_acceleration = acc
        req.joint_endpose = self._convert_endpose(endpose)

        resp = srv(req)
        if not resp.status:
            rospy.logwarn(f"moveit_ctrl_endpose failed, error_code: {resp.error_code}")
        return resp.status

    @staticmethod
    def _convert_endpose(endpose):
        if len(endpose) == 6:
            x, y, z, roll, pitch, yaw = endpose
            qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
            return [x, y, z, qx, qy, qz, qw]
        elif len(endpose) == 7:
            return list(endpose)
        else:
            raise ValueError("endpose must be 6 (Euler) or 7 (Quaternion) elements")

    # ------------------------------------------------------------------ #
    #                    状态读取（内部）
    # ------------------------------------------------------------------ #

    def _get_robot_state(self):
        qpos = self.joint_recorder.qpos
        qvel = getattr(self.joint_recorder, "qvel", None)

        if qpos is None:
            rospy.logwarn("No joint state received yet, returning reset defaults")
            qpos = list(self.RESET_JOINTS) + [self.RESET_GRIPPER]
            qvel = [0.0] * (self.NUM_JOINTS + 1)

        joint_positions = np.array(qpos[: self.NUM_JOINTS])
        gripper_position = np.array(
            [qpos[self.NUM_JOINTS]] if len(qpos) > self.NUM_JOINTS else [0.0]
        )
        joint_velocities = (
            np.array(qvel[: self.NUM_JOINTS]) if qvel is not None else np.zeros(self.NUM_JOINTS)
        )

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "gripper_position": gripper_position,
        }

    def _get_images(self):
        images_raw = self.image_recorder.get_images()
        image_dict = {}
        for cfg_name, alias in self._cam_alias.items():
            if cfg_name in images_raw and images_raw[cfg_name] is not None:
                image_dict[alias] = images_raw[cfg_name]
        return image_dict

    # ================================================================== #
    #                       公开 API
    # ================================================================== #

    # ---- 1. 关节控制 ---- #

    def move_joints(self, joint_states, gripper=None, max_velocity=None, max_acceleration=None):
        """
        关节空间控制

        Args:
            joint_states: list[float], 6 个关节角度 (弧度)
            gripper: float | None, 夹爪开合度 (米, 0.0~0.06);
                     为 None 时只控制关节不动夹爪
            max_velocity: float, MoveIt 最大速度比例 (0~1)
            max_acceleration: float, MoveIt 最大加速度比例 (0~1)

        Returns:
            dict: {"success": bool}
        """
        assert len(joint_states) == self.NUM_JOINTS, (
            f"joint_states must have {self.NUM_JOINTS} elements, got {len(joint_states)}"
        )

        if gripper is not None:
            ok = self._call_moveit_piper(joint_states, gripper, max_velocity, max_acceleration)
        else:
            ok = self._call_moveit_arm(joint_states, max_velocity, max_acceleration)
        return {"success": ok}

    # ---- 2. 末端位姿控制 ---- #

    def move_to_pose(self, endpose, max_velocity=None, max_acceleration=None):
        """
        末端位姿控制 (MoveIt IK)

        Args:
            endpose: list[float]
                - 6 元素: [x, y, z, roll, pitch, yaw]  (米 / 弧度)
                - 7 元素: [x, y, z, qx, qy, qz, qw]   (米 / 四元数)
            max_velocity: float
            max_acceleration: float

        Returns:
            dict: {"success": bool}
        """
        ok = self._call_moveit_endpose(endpose, max_velocity, max_acceleration)
        return {"success": ok}

    # ---- 3. 夹爪控制 ---- #

    def set_gripper(self, position, max_velocity=None, max_acceleration=None):
        """
        独立控制夹爪开合

        Args:
            position: float, 夹爪开合度 (米, 0.0 = 闭合, 0.06 = 全开)
            max_velocity: float
            max_acceleration: float

        Returns:
            dict: {"success": bool}
        """
        ok = self._call_moveit_gripper(position, max_velocity, max_acceleration)
        return {"success": ok}

    # ---- 4. 复位 ---- #

    def reset(self, max_velocity=None, max_acceleration=None):
        """
        复位: 关节回到默认位置，夹爪回到默认开合度

        Returns:
            dict: {"success": bool}
        """
        print("Resetting robot ...")
        ok = self._call_moveit_piper(
            self.RESET_JOINTS, self.RESET_GRIPPER, max_velocity, max_acceleration
        )
        if ok:
            print("Robot reset complete")
        else:
            rospy.logwarn("Robot reset may have failed")
        return {"success": ok}

    # ---- 5. 拍照 ---- #

    def read_cameras(self):
        """
        读取当前相机图像

        Returns:
            images: dict[str, np.ndarray]
                {
                    'left_camera_0_left':  cam_high 图像,
                    'wrist_camera_0_left': cam_low 图像,
                }
            timestamps: dict[str, float]
        """
        camera_obs = self._get_images()
        timestamps = {name: rospy.Time.now().to_sec() for name in camera_obs}

        with self._camera_lock:
            for name, img in camera_obs.items():
                if img is None:
                    continue
                self._latest_camera_frames[(name, "color")] = CameraFrame(
                    device_id=name,
                    stream_type="color",
                    frame=img,
                    timestamp=timestamps.get(name, time.time()),
                    width=int(img.shape[1]),
                    height=int(img.shape[0]),
                    depth_scale=None,
                )
        return camera_obs, timestamps

    # ---- 6. 获取关节电机角度 ---- #

    def get_robot_state(self):
        """
        获取当前机器人关节状态

        Returns:
            dict:
                - joint_positions:  np.ndarray (6,)  关节角度 (弧度)
                - joint_velocities: np.ndarray (6,)  关节速度 (弧度/秒)
                - gripper_position: np.ndarray (1,)  夹爪开合 (米)
        """
        return self._get_robot_state()

    # ---- 7. 获取末端位姿 ---- #

    def get_robot_end_pose(self):
        """
        获取当前末端位姿 (从 /end_pose 话题)

        Returns:
            dict | None:
                - position:          [x, y, z]  (米)
                - orientation_quat:  [qx, qy, qz, qw]
                - orientation_euler: [roll, pitch, yaw] (弧度)
                - timestamp:         float (秒)
        """
        if self._end_pose is None:
            rospy.logwarn("No end pose received yet from /end_pose")
            return None

        p = self._end_pose.pose.position
        q = self._end_pose.pose.orientation
        t = self._end_pose.header.stamp.to_sec()

        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return {
            "position": [p.x, p.y, p.z],
            "orientation_quat": [q.x, q.y, q.z, q.w],
            "orientation_euler": [roll, pitch, yaw],
            "timestamp": t,
        }

    # ---- 兼容旧接口 ---- #

    def get_observation(self):
        obs = {"timestamp": {}}
        obs["robot_state"] = self._get_robot_state()
        obs["timestamp"]["robot_state"] = rospy.Time.now().to_sec()
        obs["image"] = self._get_images()
        obs["timestamp"]["cameras"] = rospy.Time.now().to_sec()
        return obs

    def get_latest_decoded_frame(self, stream_type: str, device_id: str | None = None):
        if stream_type != "color":
            return None

        with self._camera_lock:
            has_cache = bool(self._latest_camera_frames)
        if not has_cache:
            try:
                self.read_cameras()
            except Exception:
                return None

        with self._camera_lock:
            if device_id is not None:
                return self._latest_camera_frames.get((device_id, stream_type))
            for (dev, st), frame in self._latest_camera_frames.items():
                if st == stream_type:
                    return frame
        return None

    def get_all_frames(self):
        images, _ = self.read_cameras()
        out = {}
        for device_id, img in images.items():
            if img is None:
                continue
            ok, buf = cv2.imencode(".jpg", img)
            if ok:
                out[device_id] = buf.tobytes()
        return out

    def get_cameras(self):
        return list(self._camera_meta)

    def get_state(self):
        robot_state = self.get_robot_state()
        end_pose = self.get_robot_end_pose()

        with self._camera_lock:
            cameras = [
                {
                    "device_id": device_id,
                    "stream_type": stream_type,
                    "timestamp": frame.timestamp,
                    "width": frame.width,
                    "height": frame.height,
                }
                for (device_id, stream_type), frame in self._latest_camera_frames.items()
            ]

        return {
            "robot_state": robot_state,
            "end_pose": end_pose,
            "cameras": cameras,
            "timestamp": time.time(),
        }


if __name__ == "__main__":

    env = PiperRobotEnv()

    # 1. 获取关节状态
    print("\n[1] get_robot_state()")
    state = env.get_robot_state()
    print(f"  joint_positions:  {state['joint_positions']}")
    print(f"  joint_velocities: {state['joint_velocities']}")
    print(f"  gripper_position: {state['gripper_position']}")

    # 2. 获取末端位姿
    print("\n[2] get_robot_end_pose()")
    pose = env.get_robot_end_pose()
    if pose:
        print(f"  position:    {pose['position']}")
        print(f"  euler:       {pose['orientation_euler']}")
        print(f"  quaternion:  {pose['orientation_quat']}")

    # 3. 拍照
    print("\n[3] read_cameras()")
    images, ts = env.read_cameras()
    for cam, img in images.items():
        if img is not None:
            fname = f"test_{cam}.png"
            cv2.imwrite(fname, img)
            print(f"  {cam}: shape={img.shape}, saved {fname}")

    # 4. 复位 (取消注释以实际执行)
    # print("\n[4] reset()")
    env.reset()

    # 5. 关节控制 (取消注释以实际执行)
    # print("\n[5] move_joints()")
    # env.move_joints([0.0, 0.2, -0.4, 0.0, 0.8, 0.0])

    # 6. 带夹爪的关节控制 (取消注释以实际执行)
    # print("\n[6] move_joints() with gripper")
    # env.move_joints([0.0, 0.2, -0.2, 0.0, 0.8, 0.0], gripper=0.0)

    # 7. 末端位姿控制 - 欧拉角 (取消注释以实际执行)
    # print("\n[7] move_to_pose() euler")
    # env.move_to_pose([0.2056, -0.0451, 0.0667, 0.7021, 0.6670, -0.1744])

    # 8. 末端位姿控制 - 四元数 (取消注释以实际执行)
    # print("\n[8] move_to_pose() quaternion")
    # env.move_to_pose([0.1283, -0.0352, 0.1015, 0.2030, 0.8992, -0.0189, 0.3872])

    # 9. 夹爪控制 (取消注释以实际执行)
    # print("\n[9] set_gripper()")
    # env.set_gripper(0.06)
    # import time
    # time.sleep(1)
    # env.set_gripper(0.0)

    print("\nAll tests done.")
