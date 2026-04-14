"""
YOLOv5 目标检测 SDK —— robot_sdk 封装层 (HTTP API 版)

通过远程 YOLO HTTP 检测服务获取像素坐标 bbox，
在本地使用 ROS 深度图 + 相机内参 + TF 完成 3D 投影。

无需 anygraspenv 或本地 torch 模型，直接在主进程 (/usr/bin/python3) 中运行。

配置优先级: 构造函数参数 > config.yaml > 内置默认值

暴露接口:
    - detect_env()            -> List[str]        当前画面中所有物体类别
    - segment_3d(object_name) -> List[dict]       指定物体在 base_link 下的 3D 坐标
    - start() / stop()                            生命周期管理
"""

from __future__ import annotations

import base64
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import rospy
import tf2_ros
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image as RosImage

try:
    from robot_sdk.config import get_config
except ImportError:
    from config import get_config


class YoloSDK:
    """
    YOLOv5 目标检测服务封装 (HTTP API 版)

    内部通过 HTTP 调用远程 YOLO 检测服务获取像素 bbox，
    同时订阅 ROS 深度图 / 相机内参 / TF 完成 3D 坐标转换。
    直接在主进程中运行，不再需要子进程。

    所有参数均可省略，默认从 config.yaml 读取；
    也可在构造时显式传入以覆盖配置文件。

    用法:
        yolo = YoloSDK()
        yolo.start()

        labels = yolo.detect_env()
        print(labels)  # ['bottle', 'cup', 'keyboard']

        detections = yolo.segment_3d("bottle")
        for d in detections:
            print(d["position_base"], d["depth_m"])

        yolo.stop()
    """

    def __init__(
        self,
        yolo_url: Optional[str] = None,
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
        image_topic: Optional[str] = None,
        depth_topic: Optional[str] = None,
        camera_info_topic: Optional[str] = None,
        camera_frame_id: Optional[str] = None,
        base_frame_id: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        cfg = get_config()
        ros_cfg = cfg.get("ros", {})
        yolo_cfg = cfg.get("yolo", {})

        self._yolo_url = yolo_url or yolo_cfg.get("url", "http://localhost:8017/detect")
        self._conf_thres = conf_thres if conf_thres is not None else yolo_cfg.get("conf_thres", 0.5)
        self._iou_thres = iou_thres if iou_thres is not None else yolo_cfg.get("iou_thres", 0.45)

        self._image_topic = image_topic or ros_cfg.get("image_topic", "/wrist_camera/color/image_raw")
        self._depth_topic = depth_topic or ros_cfg.get("depth_topic", "/wrist_camera/aligned_depth_to_color/image_raw")
        self._camera_info_topic = camera_info_topic or ros_cfg.get("camera_info_topic", "/wrist_camera/color/camera_info")
        self._camera_frame_id = camera_frame_id or ros_cfg.get("camera_frame_id", "wrist_camera_color_optical_frame")
        self._base_frame_id = base_frame_id or ros_cfg.get("base_frame_id", "base_link")
        self._timeout = timeout if timeout is not None else yolo_cfg.get("start_timeout", 30.0)
        self._request_timeout = request_timeout if request_timeout is not None else yolo_cfg.get("request_timeout", 10.0)

        self._bridge = CvBridge()
        self._lock = threading.Lock()

        self._last_color_bgr: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._camera_K: Optional[np.ndarray] = None

        self._tf_buffer: Optional[tf2_ros.Buffer] = None
        self._tf_listener: Optional[tf2_ros.TransformListener] = None

        self._started = False

    # ================================================================== #
    #                       生命周期
    # ================================================================== #

    def start(self) -> "YoloSDK":
        """启动 ROS 订阅 (图像、深度、相机内参、TF)"""
        if self._started:
            return self

        if not rospy.core.is_initialized():
            rospy.init_node("yolo_sdk", anonymous=True, disable_signals=True)

        self._sub_color = rospy.Subscriber(
            self._image_topic, RosImage, self._on_color, queue_size=1,
        )
        self._sub_depth = rospy.Subscriber(
            self._depth_topic, RosImage, self._on_depth, queue_size=1,
        )
        self._sub_info = rospy.Subscriber(
            self._camera_info_topic, CameraInfo, self._on_info, queue_size=1,
        )
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        deadline = time.time() + self._timeout
        while time.time() < deadline and not rospy.is_shutdown():
            with self._lock:
                if self._last_color_bgr is not None:
                    break
            time.sleep(0.05)

        with self._lock:
            if self._last_color_bgr is None:
                raise RuntimeError(
                    f"等待 ROS 图像超时 ({self._timeout}s): {self._image_topic}"
                )

        self._started = True
        print("YoloSDK: started (HTTP API + ROS)")
        return self

    def stop(self) -> None:
        """停止 ROS 订阅"""
        if not self._started:
            return
        for attr in ("_sub_color", "_sub_depth", "_sub_info"):
            sub = getattr(self, attr, None)
            if sub is not None:
                sub.unregister()
        self._started = False
        print("YoloSDK: stopped")

    def __enter__(self) -> "YoloSDK":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    # ================================================================== #
    #                       ROS 回调
    # ================================================================== #

    def _on_color(self, msg: RosImage) -> None:
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            img = self._bridge.imgmsg_to_cv2(msg)
        with self._lock:
            self._last_color_bgr = img

    def _on_depth(self, msg: RosImage) -> None:
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            depth = self._bridge.imgmsg_to_cv2(msg)
        with self._lock:
            self._last_depth = np.asarray(depth, dtype=np.float32)

    def _on_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._camera_K = np.array(msg.K, dtype=np.float64).reshape(3, 3)

    # ================================================================== #
    #                       内部工具
    # ================================================================== #

    def _read_color(self) -> np.ndarray:
        if not self._started:
            self.start()
        with self._lock:
            img = None if self._last_color_bgr is None else self._last_color_bgr.copy()
        if img is None:
            raise RuntimeError(f"尚未收到 ROS 图像: {self._image_topic}")
        return img

    def _encode_image(self, img_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            raise RuntimeError("cv2.imencode 失败")
        return base64.b64encode(buf).decode("utf-8")

    def _call_yolo_api(self, img_b64: str) -> List[Dict]:
        """调用远程 YOLO HTTP 检测服务，返回 detections 列表"""
        payload = {
            "image": img_b64,
            "conf_thres": self._conf_thres,
            "iou_thres": self._iou_thres,
        }
        resp = requests.post(
            self._yolo_url, json=payload, timeout=self._request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("detections", [])

    @staticmethod
    def _sample_depth(
        depth_map: np.ndarray,
        xyxy: Tuple[int, int, int, int],
        num_samples: int = 24,
    ) -> Optional[float]:
        """在 bbox 中心区域随机采样深度，取截断均值 (米)"""
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half_range = min(abs(x2 - x1), abs(y2 - y1)) // 4
        h, w = depth_map.shape[:2]
        valid: List[float] = []
        for _ in range(num_samples):
            px = cx + random.randint(-half_range, half_range)
            py = cy + random.randint(-half_range, half_range)
            if 0 <= px < w and 0 <= py < h:
                d = float(depth_map[py, px])
                if d > 0:
                    valid.append(d)
        if not valid:
            return None
        valid.sort()
        n = len(valid)
        trimmed = valid[n // 4 : n // 4 + n // 2] or valid
        return float(np.mean(trimmed)) / 1000.0

    @staticmethod
    def _deproject(
        u: float, v: float, depth_m: float, K: np.ndarray,
    ) -> Tuple[float, float, float]:
        """像素坐标 + 深度 → 相机坐标系 3D 点"""
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        return ((u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m)

    @staticmethod
    def _quat_to_rot(q) -> np.ndarray:
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def _cam_to_base(
        self, x: float, y: float, z: float,
    ) -> Tuple[float, float, float]:
        """相机坐标系 → base_link 坐标系"""
        trans = self._tf_buffer.lookup_transform(
            self._base_frame_id,
            self._camera_frame_id,
            rospy.Time(0),
            rospy.Duration(1.0),
        )
        t = trans.transform.translation
        R = self._quat_to_rot(trans.transform.rotation)
        p = R @ np.array([x, y, z]) + np.array([t.x, t.y, t.z])
        return (float(p[0]), float(p[1]), float(p[2]))

    # ================================================================== #
    #                       公开 API
    # ================================================================== #

    def detect_env(self) -> List[str]:
        """
        检测当前画面中的所有物体类别 (去重)

        Returns:
            List[str]: 物体类别名列表, 如 ["bottle", "cup"]
        """
        img = self._read_color()
        img_b64 = self._encode_image(img)
        detections = self._call_yolo_api(img_b64)
        names = list({d["class_name"] for d in detections})
        return sorted(names)

    def segment_3d(self, object_name: str) -> List[Dict]:
        """
        检测指定物体并返回 base_link 坐标系下的 3D 坐标

        Args:
            object_name: COCO 类别名, 如 "cup", "bottle", "keyboard"

        Returns:
            List[dict]: 每个检测到的实例包含:
                - label: str            类别名
                - confidence: float     置信度
                - xyxy: [x1,y1,x2,y2]  像素坐标框
                - position_camera: [x,y,z]  相机坐标系下位置 (米)
                - position_base: [x,y,z]    base_link 坐标系下位置 (米)
                - depth_m: float        深度 (米)
        """
        if not self._started:
            self.start()

        img = self._read_color()
        img_b64 = self._encode_image(img)
        detections = self._call_yolo_api(img_b64)

        matched = [d for d in detections if d["class_name"] == object_name]
        if not matched:
            return []

        with self._lock:
            depth_map = None if self._last_depth is None else self._last_depth.copy()
            K = None if self._camera_K is None else self._camera_K.copy()
        if depth_map is None:
            raise RuntimeError(f"尚未收到深度图: {self._depth_topic}")
        if K is None:
            raise RuntimeError(f"尚未收到相机内参: {self._camera_info_topic}")

        out: List[Dict] = []
        for det in matched:
            xyxy = (
                int(det["x1"]),
                int(det["y1"]),
                int(det["x2"]),
                int(det["y2"]),
            )
            depth_m = self._sample_depth(depth_map, xyxy)
            if depth_m is None or depth_m <= 0:
                continue
            cx_px = (xyxy[0] + xyxy[2]) / 2.0
            cy_px = (xyxy[1] + xyxy[3]) / 2.0
            pos_cam = self._deproject(cx_px, cy_px, depth_m, K)
            try:
                pos_base = self._cam_to_base(*pos_cam)
            except Exception as e:
                rospy.logwarn("TF 转换失败: %s", e)
                continue
            out.append({
                "label": det["class_name"],
                "confidence": float(det["confidence"]),
                "xyxy": list(xyxy),
                "position_camera": list(pos_cam),
                "position_base": list(pos_base),
                "depth_m": depth_m,
            })
        return out


if __name__ == "__main__":
    yolo = YoloSDK()
    with yolo:
        for _ in range(5):
            labels = yolo.detect_env()
            print(f"环境物体: {labels}")

            for label in labels[:3]:
                dets = yolo.segment_3d(label)
                for d in dets:
                    bx, by, bz = d["position_base"]
                    print(
                        f"  {d['label']}: base({bx:.3f}, {by:.3f}, {bz:.3f}) "
                        f"depth={d['depth_m']:.3f}m"
                    )
            time.sleep(0.5)
