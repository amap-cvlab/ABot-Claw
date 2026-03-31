"""
YOLOv5 目标检测服务 —— 精简封装。

暴露接口：
  - detect_env()            -> List[str]          当前画面中检测到的所有物体类别
  - segment_3d(object_name) -> List[Detection3D]   指定物体在 base_link 下的 3D 坐标
"""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rospy
import tf2_ros
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image as RosImage


@dataclass(frozen=True)
class Detection3D:
    """单个物体的 3D 检测结果 (base_link 坐标系)。"""
    label: str
    confidence: float
    xyxy: Tuple[int, int, int, int]
    position_camera: Tuple[float, float, float]
    position_base: Tuple[float, float, float]
    depth_m: float


class YoloDetector:
    """
    轻量 YOLOv5 检测服务，内部管理 ROS 订阅 + TF。

    接口：
      detect_env()            -> 当前画面所有物体类别 (去重)
      segment_3d(object_name) -> 指定物体在 base_link 坐标系下的 3D 位置列表
    """

    def __init__(
        self,
        model_name: str = "yolov5l6",
        conf: float = 0.5,
        device: Optional[str] = None,
        image_topic: str = "/wrist_camera/color/image_raw",
        depth_topic: str = "/wrist_camera/aligned_depth_to_color/image_raw",
        camera_info_topic: str = "/wrist_camera/color/camera_info",
        camera_frame_id: str = "wrist_camera_color_optical_frame",
        base_frame_id: str = "base_link",
        node_name: str = "yolo_detector",
        anonymous: bool = True,
        wait_first_image_s: float = 5.0,
    ) -> None:
        os.environ.setdefault("YOLOv5_AUTOINSTALL", "0")
        os.environ.setdefault("YOLO_AUTOINSTALL", "0")

        self._model = torch.hub.load("ultralytics/yolov5", model_name)
        self._model.conf = conf
        if device is not None:
            self._model.to(device)

        self._image_topic = image_topic
        self._depth_topic = depth_topic
        self._camera_info_topic = camera_info_topic
        self._camera_frame_id = camera_frame_id
        self._base_frame_id = base_frame_id
        self._node_name = node_name
        self._anonymous = anonymous
        self._wait_first_image_s = float(wait_first_image_s)

        self._started = False
        self._bridge = CvBridge()
        self._lock = threading.Lock()

        self._last_color_bgr: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._camera_K: Optional[np.ndarray] = None

        self._tf_buffer: Optional[tf2_ros.Buffer] = None
        self._tf_listener: Optional[tf2_ros.TransformListener] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> "YoloDetector":
        if self._started:
            return self
        if not rospy.core.is_initialized():
            rospy.init_node(self._node_name, anonymous=self._anonymous, disable_signals=True)

        self._sub_color = rospy.Subscriber(self._image_topic, RosImage, self._on_color, queue_size=1)
        self._sub_depth = rospy.Subscriber(self._depth_topic, RosImage, self._on_depth, queue_size=1)
        self._sub_info = rospy.Subscriber(self._camera_info_topic, CameraInfo, self._on_info, queue_size=1)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self._started = True

        deadline = time.time() + self._wait_first_image_s
        while time.time() < deadline and not rospy.is_shutdown():
            with self._lock:
                if self._last_color_bgr is not None:
                    break
            time.sleep(0.01)
        with self._lock:
            if self._last_color_bgr is None:
                raise RuntimeError(f"等待 ROS 图像超时 ({self._wait_first_image_s}s): {self._image_topic}")
        return self

    def stop(self) -> None:
        if not self._started:
            return
        for attr in ("_sub_color", "_sub_depth", "_sub_info"):
            sub = getattr(self, attr, None)
            if sub is not None:
                sub.unregister()
        self._started = False

    def __enter__(self) -> "YoloDetector":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # ROS 回调
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _read_color(self) -> np.ndarray:
        if not self._started:
            self.start()
        with self._lock:
            img = None if self._last_color_bgr is None else self._last_color_bgr.copy()
        if img is None:
            raise RuntimeError(f"尚未收到 ROS 图像: {self._image_topic}")
        return img

    @staticmethod
    def _sample_depth(
        depth_map: np.ndarray,
        xyxy: Tuple[int, int, int, int],
        num_samples: int = 24,
    ) -> Optional[float]:
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
        trimmed = valid[n // 4: n // 4 + n // 2] or valid
        return float(np.mean(trimmed)) / 1000.0

    @staticmethod
    def _deproject(u: float, v: float, depth_m: float, K: np.ndarray) -> Tuple[float, float, float]:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        return ((u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m)

    @staticmethod
    def _quat_to_rot(q) -> np.ndarray:
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
        ])

    def _cam_to_base(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        trans = self._tf_buffer.lookup_transform(
            self._base_frame_id, self._camera_frame_id,
            rospy.Time(0), rospy.Duration(1.0),
        )
        t = trans.transform.translation
        R = self._quat_to_rot(trans.transform.rotation)
        p = R @ np.array([x, y, z]) + np.array([t.x, t.y, t.z])
        return (float(p[0]), float(p[1]), float(p[2]))

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def detect_env(self) -> List[str]:
        """返回当前画面中检测到的所有物体类别名 (去重)。"""
        img = self._read_color()
        results = self._model(img)
        df = results.pandas().xyxy[0]
        if df is None or len(df) == 0:
            return []
        return list(df["name"].unique())

    def segment_3d(self, object_name: str) -> List[Detection3D]:
        """
        检测指定物体并返回 base_link 坐标系下的 3D 坐标。

        参数:
          object_name: COCO 类别名 (如 "cup", "bottle", "keyboard")

        返回:
          每个检测到的实例对应一个 Detection3D
        """
        if not self._started:
            self.start()

        img = self._read_color()
        results = self._model(img)
        df = results.pandas().xyxy[0]
        if df is None or len(df) == 0:
            return []
        df = df[df["name"] == object_name]
        if len(df) == 0:
            return []

        with self._lock:
            depth_map = None if self._last_depth is None else self._last_depth.copy()
            K = None if self._camera_K is None else self._camera_K.copy()
        if depth_map is None:
            raise RuntimeError(f"尚未收到深度图: {self._depth_topic}")
        if K is None:
            raise RuntimeError(f"尚未收到相机内参: {self._camera_info_topic}")

        out: List[Detection3D] = []
        for row in df.itertuples(index=False):
            xyxy = (int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax))
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
            out.append(Detection3D(
                label=str(row.name),
                confidence=float(row.confidence),
                xyxy=xyxy,
                position_camera=pos_cam,
                position_base=pos_base,
                depth_m=depth_m,
            ))
        return out


# ======================================================================
# 快速测试
# ======================================================================
if __name__ == "__main__":
    detector = YoloDetector(model_name="yolov5l6", conf=0.5)
    with detector:
        while not rospy.is_shutdown():
            print("环境物体:", detector.detect_env())
            dets = detector.segment_3d("cup")
            for d in dets:
                bx, by, bz = d.position_base
                print(f"  {d.label}: base({bx:.3f}, {by:.3f}, {bz:.3f})  depth={d.depth_m:.3f}m")
            time.sleep(0.5)
