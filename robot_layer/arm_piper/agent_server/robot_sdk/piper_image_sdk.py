import time
from collections import deque

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState

try:
    from robot_sdk.config import get_config
except ImportError:
    from config import get_config

_ALL_CAMERA_NAMES = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]


class ImageRecorder:
    """订阅多路 ROS 相机话题，缓存最新 BGR 帧。

    哪些相机启用 / 各相机话题地址均由 config.yaml 中 ``cameras`` 节决定。
    """

    def __init__(self, init_node=True, is_debug=False):
        self.is_debug = is_debug
        self.bridge = CvBridge()

        cfg = get_config()
        cam_cfg = cfg.get("cameras", {})

        self.camera_names = list(_ALL_CAMERA_NAMES)
        self.valid_camera_names = [
            name for name in self.camera_names
            if cam_cfg.get(name, {}).get("enabled", False)
        ]
        if not self.valid_camera_names:
            self.valid_camera_names = ["cam_low"]

        self._camera_topics = {
            name: cam_cfg.get(name, {}).get("topic", f"/{name}")
            for name in self.camera_names
        }

        if init_node:
            rospy.init_node("image_recorder", anonymous=True)

        for cam_name in self.camera_names:
            setattr(self, f"{cam_name}_rgb_image", None)
            setattr(self, f"{cam_name}_timestamp", 0.0)

            topic = self._camera_topics[cam_name]
            cb = lambda data, _name=cam_name: self._image_cb(_name, data)
            rospy.Subscriber(topic, Image, cb, queue_size=1)
            print(f"Subscribed to {topic} ({cam_name})")

            if self.is_debug:
                setattr(self, f"{cam_name}_timestamps", deque(maxlen=50))

        self.cam_last_timestamps = {n: 0.0 for n in self.camera_names}
        time.sleep(0.5)

    def _image_cb(self, cam_name: str, data: Image) -> None:
        setattr(
            self,
            f"{cam_name}_rgb_image",
            self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8"),
        )
        setattr(
            self,
            f"{cam_name}_timestamp",
            data.header.stamp.secs + data.header.stamp.nsecs * 1e-9,
        )
        if self.is_debug:
            getattr(self, f"{cam_name}_timestamps").append(time.time())

    def get_images(self) -> dict:
        """返回 {cam_name: np.ndarray | zeros} 字典。

        启用的相机返回最新帧，未启用的填零。
        """
        image_dict = {}
        for cam_name in self.camera_names:
            if cam_name in self.valid_camera_names:
                rgb_image = getattr(self, f"{cam_name}_rgb_image")
                self.cam_last_timestamps[cam_name] = getattr(self, f"{cam_name}_timestamp")
                image_dict[cam_name] = rgb_image
            else:
                ref = getattr(self, f"{self.valid_camera_names[0]}_rgb_image")
                image_dict[cam_name] = np.zeros_like(ref) if ref is not None else None
        return image_dict


class Recorder:
    """订阅关节状态话题，缓存 qpos / qvel / effort。"""

    def __init__(self, side="left", init_node=True, is_debug=False, joint_state_topic=None):
        self.qpos = None
        self.qvel = None
        self.effort = None
        self.is_debug = is_debug

        if joint_state_topic is None:
            cfg = get_config()
            joint_state_topic = cfg.get("ros", {}).get(
                "joint_state_topic", "/joint_states_single"
            )

        if init_node:
            rospy.init_node("recorder", anonymous=True)

        rospy.Subscriber(joint_state_topic, JointState, self._joint_cb, queue_size=1)

        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def _joint_cb(self, data: JointState) -> None:
        self.qpos = data.position
        self.qvel = data.velocity
        self.effort = data.effort
        if self.is_debug:
            self.joint_timestamps.append(time.time())
