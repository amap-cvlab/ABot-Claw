"""
AnyGrasp 抓取位姿 SDK
"""

from __future__ import annotations

import base64
import threading
import time
import math
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import rospy
from cv_bridge import CvBridge
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image as RosImage

from config import get_config


class GraspSDK:
    """
    AnyGrasp 6DoF 抓取位姿服务封装 (HTTP API 版)

    通过 HTTP 调用远程 Grasp 服务（`POST /grasp/detect`），并订阅 ROS：
      - 颜色图像
      - 深度图像
      - CameraInfo（相机内参）

    服务端返回 camera 坐标系下的抓取候选位姿；本 SDK 使用 TF 把它们转换到 base_link。

    所有参数均可省略，默认从 config.yaml 读取；
    也可在构造时显式传入以覆盖配置文件。

    用法:
        grasp = GraspSDK()
        grasp.start()

        results = grasp.get_grasp_pose("bottle", top_k=5)
        for res in results:
            if res["grasps"]:
                best = res["grasps"][0]
                exec_pose = best["translation_base_retreat"] + best["quaternion_base"]
                print(exec_pose)

        grasp.stop()
    """

    def __init__(
        self,
        grasp_url: Optional[str] = None,
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
        g = cfg.get("grasp", {})
        y = cfg.get("yolo", {})

        # --- HTTP urls ---
        # 兼容历史配置：有些环境只填了 "...:8018/detect"（应为 "...:8018/grasp/detect"）
        raw_grasp_url = grasp_url or g.get("url", "http://localhost:8015/grasp/detect")
        self._grasp_url = self._normalize_grasp_url(raw_grasp_url)

        self._yolo_url = y.get("url", "http://localhost:8017/detect")

        # --- thresholds (仅 detect_env 使用；grasp/detect 本身的阈值在 README 中未要求) ---
        self._conf_thres = conf_thres if conf_thres is not None else y.get("conf_thres", 0.5)
        self._iou_thres = iou_thres if iou_thres is not None else y.get("iou_thres", 0.45)

        # --- ROS topics / frames ---
        self._image_topic = image_topic or ros_cfg.get("image_topic", "/wrist_camera/color/image_raw")
        self._depth_topic = depth_topic or ros_cfg.get("depth_topic", "/wrist_camera/aligned_depth_to_color/image_raw")
        self._camera_info_topic = camera_info_topic or ros_cfg.get(
            "camera_info_topic", "/wrist_camera/color/camera_info"
        )
        self._camera_frame_id = camera_frame_id or ros_cfg.get("camera_frame_id", "wrist_camera_color_optical_frame")
        self._base_frame_id = base_frame_id or ros_cfg.get("base_frame_id", "base_link")

        self._timeout = timeout if timeout is not None else g.get("start_timeout", 30.0)
        self._request_timeout = request_timeout if request_timeout is not None else g.get("request_timeout", 10.0)

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._last_color_bgr: Optional[np.ndarray] = None
        self._last_depth_u16: Optional[np.ndarray] = None
        self._camera_K: Optional[np.ndarray] = None

        self._tf_buffer: Optional[tf2_ros.Buffer] = None
        self._tf_listener: Optional[tf2_ros.TransformListener] = None
        self._started = False

    @staticmethod
    def _normalize_grasp_url(url: str) -> str:
        """
        将历史/不完整的 grasp.url 归一到 ".../grasp/detect"。
        """
        u = url.strip()
        if not u:
            return u

        # Already normalized
        if u.rstrip("/").endswith("/grasp/detect"):
            return u

        # Common wrong form: ".../detect" (missing "/grasp")
        if u.rstrip("/").endswith("/detect") and "/grasp/" not in u:
            return u.rstrip("/")[: -len("/detect")] + "/grasp/detect"

        # Fallback: if it points to host/port without path, append default
        parsed = urlparse(u)
        if not parsed.path or parsed.path == "/":
            return u.rstrip("/") + "/grasp/detect"

        return u

    # ================================================================== #
    #                       生命周期
    # ================================================================== #

    def start(self) -> "GraspSDK":
        """启动 ROS 图像/深度/相机内参订阅 + TF"""
        if self._started:
            return self

        if not rospy.core.is_initialized():
            rospy.init_node("grasp_sdk", anonymous=True, disable_signals=True)

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
                if (
                    self._last_color_bgr is not None
                    and self._last_depth_u16 is not None
                    and self._camera_K is not None
                ):
                    break
            time.sleep(0.05)

        with self._lock:
            if self._last_color_bgr is None or self._last_depth_u16 is None or self._camera_K is None:
                raise RuntimeError(
                    "等待 ROS 传感器数据超时 "
                    f"({self._timeout}s): color={self._image_topic}, "
                    f"depth={self._depth_topic}, info={self._camera_info_topic}"
                )

        self._started = True
        print("GraspSDK: started (HTTP API + ROS)")
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
        print("GraspSDK: stopped")

    def __enter__(self) -> "GraspSDK":
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
        """
        保存 16-bit 深度图（尽量保留毫米语义）。
        """
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            depth = self._bridge.imgmsg_to_cv2(msg)

        arr = np.asarray(depth)
        if arr.dtype == np.uint16:
            depth_u16 = arr
        else:
            # 兼容 float 深度（单位可能是 m），尽量转成 mm 的 uint16
            depth_mm = arr.astype(np.float32) * (1000.0 if arr.dtype.kind == "f" else 1.0)
            depth_mm = np.clip(depth_mm, 0, 65535)
            depth_u16 = depth_mm.astype(np.uint16)

        with self._lock:
            self._last_depth_u16 = depth_u16

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

    def _read_depth_u16(self) -> np.ndarray:
        if not self._started:
            self.start()
        with self._lock:
            d = None if self._last_depth_u16 is None else self._last_depth_u16.copy()
        if d is None:
            raise RuntimeError(f"尚未收到 ROS 深度图: {self._depth_topic}")
        return d

    def _read_camera_K(self) -> np.ndarray:
        if not self._started:
            self.start()
        with self._lock:
            K = None if self._camera_K is None else self._camera_K.copy()
        if K is None:
            raise RuntimeError(f"尚未收到相机内参: {self._camera_info_topic}")
        return K

    def _encode_color_image_rgb(self, img_bgr: np.ndarray) -> str:
        """
        编码为服务端要求的 RGB PNG（base64）。
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".png", img_rgb)
        if not ok:
            raise RuntimeError("cv2.imencode 失败")
        return base64.b64encode(buf).decode("utf-8")

    def _encode_image_bgr_legacy(self, img_bgr: np.ndarray) -> str:
        """
        编码为 PNG(base64)，保持与 `YoloSDK` 的历史实现一致（不做 BGR->RGB 转换）。
        """
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            raise RuntimeError("cv2.imencode 失败")
        return base64.b64encode(buf).decode("utf-8")

    def _encode_depth_image_png_u16(self, depth_u16: np.ndarray) -> str:
        """
        编码为 16-bit PNG（uint16）的 base64，满足 README 的深度输入规范。
        """
        d = np.asarray(depth_u16)
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[:, :, 0]
        if d.dtype != np.uint16:
            d = np.clip(d.astype(np.float32), 0, 65535).astype(np.uint16)

        ok, buf = cv2.imencode(".png", d)
        if not ok:
            raise RuntimeError("cv2.imencode depth 失败")
        return base64.b64encode(buf).decode("utf-8")

    def _call_yolo_detect_api(self, img_rgb_b64: str) -> List[Dict]:
        """
        用 YOLO HTTP 服务做 detect_env（返回 detections 列表）。
        """
        payload = {"image": img_rgb_b64, "conf_thres": self._conf_thres, "iou_thres": self._iou_thres}
        resp = requests.post(self._yolo_url, json=payload, timeout=self._request_timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("detections", [])

    @staticmethod
    def _quat_to_rot_xyzw(qxyzw: List[float] | Tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
        q = np.asarray(qxyzw, dtype=np.float64).reshape(4,)
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    @staticmethod
    def _rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
        """
        旋转矩阵 -> 四元数 (qx,qy,qz,qw)
        """
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
        m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
        m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
        tr = m00 + m11 + m22

        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2.0
            qw = 0.25 * S
            qx = (m21 - m12) / S
            qy = (m02 - m20) / S
            qz = (m10 - m01) / S
        elif m00 > m11 and m00 > m22:
            S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            qw = (m21 - m12) / S
            qx = 0.25 * S
            qy = (m01 + m10) / S
            qz = (m02 + m20) / S
        elif m11 > m22:
            S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            qw = (m02 - m20) / S
            qx = (m01 + m10) / S
            qy = 0.25 * S
            qz = (m12 + m21) / S
        else:
            S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            qw = (m10 - m01) / S
            qx = (m02 + m20) / S
            qy = (m12 + m21) / S
            qz = 0.25 * S

        q = np.array([qx, qy, qz, qw], dtype=np.float64)
        # Normalize for safety
        n = np.linalg.norm(q)
        if n > 0:
            q /= n
        return q

    def _cam_to_base_point(self, p_cam: List[float] | Tuple[float, float, float] | np.ndarray) -> np.ndarray:
        """
        相机坐标系点 -> base_link 坐标系点
        """
        if self._tf_buffer is None:
            raise RuntimeError("TF buffer not initialized (call start() first)")

        trans = self._tf_buffer.lookup_transform(
            self._base_frame_id,
            self._camera_frame_id,
            rospy.Time(0),
            rospy.Duration(1.0),
        )
        t = trans.transform.translation
        q = trans.transform.rotation
        R_tf = self._quat_to_rot_xyzw([q.x, q.y, q.z, q.w])
        p = np.asarray(p_cam, dtype=np.float64).reshape(3,)
        p_base = R_tf @ p + np.array([t.x, t.y, t.z], dtype=np.float64)
        return p_base

    def _cam_to_base_rot(self, R_cam: np.ndarray) -> np.ndarray:
        """
        相机坐标系旋转矩阵 -> base_link 坐标系旋转矩阵
        """
        if self._tf_buffer is None:
            raise RuntimeError("TF buffer not initialized (call start() first)")

        trans = self._tf_buffer.lookup_transform(
            self._base_frame_id,
            self._camera_frame_id,
            rospy.Time(0),
            rospy.Duration(1.0),
        )
        q = trans.transform.rotation
        R_tf = self._quat_to_rot_xyzw([q.x, q.y, q.z, q.w])
        R_base = R_tf @ np.asarray(R_cam, dtype=np.float64).reshape(3, 3)
        return R_base

    # ================================================================== #
    #                       公开 API
    # ================================================================== #

    def detect_env(self) -> List[str]:
        """
        检测当前画面中的所有物体类别 (去重)

        Returns:
            List[str]: 物体类别名列表
        """
        img = self._read_color()
        img_b64 = self._encode_image_bgr_legacy(img)
        detections = self._call_yolo_detect_api(img_b64)
        names = list({d["class_name"] for d in detections})
        return sorted(names)

    def get_grasp_pose(self, object_name: str, top_k: int = 5) -> List[Dict]:
        """
        对指定物体执行 YOLO 检测 + AnyGrasp 6DoF 抓取位姿生成

        Args:
            object_name: COCO 类别名, 如 "cup", "bottle"
            top_k: 每个物体实例最多返回的抓取数量

        Returns:
            List[dict]: 每个检测到的物体实例一个, 内含:
                - label: str              类别名
                - confidence: float       YOLO 置信度
                - xyxy: [x1,y1,x2,y2]    像素坐标框
                - grasps: List[dict]      最多 top_k 个 6DoF 抓取位姿
                    每个 grasp dict 包含:
                      - score: float                      AnyGrasp 置信度
                      - width: float                      抓取宽度 (米)
                      - translation_base: [x,y,z]         AnyGrasp 几何抓取中心 (base_link)，仅作参考
                      - rotation_base: [[3x3]]            base_link 下旋转矩阵
                      - translation_base_retreat: [x,y,z] **机械臂应到达的最终末端位置** (与 quaternion_base 组成 move_to_pose)
                      - quaternion_base: [qx,qy,qz,qw]   四元数
                    重要: 执行抓取时只移动到 translation_base_retreat + quaternion_base，
                    不要再移动到 translation_base。
        """
        img_bgr = self._read_color()
        depth_u16 = self._read_depth_u16()
        K = self._read_camera_K()

        color_image_b64 = self._encode_color_image_rgb(img_bgr)
        depth_image_b64 = self._encode_depth_image_png_u16(depth_u16)

        payload = {
            "color_image": color_image_b64,
            "depth_image": depth_image_b64,
            "camera_intrinsics": K.tolist(),
            "object_name": object_name,
            "top_k": top_k,
        }

        resp = requests.post(self._grasp_url, json=payload, timeout=self._request_timeout)
        resp.raise_for_status()
        data = resp.json()
        results_in = data.get("results", []) or []

        out: List[Dict] = []
        for inst in results_in:
            label = inst.get("label", "")
            if label and label != object_name:
                continue

            xyxy = inst.get("xyxy") or []
            grasps_in = inst.get("grasps", []) or []
            grasps_out: List[Dict[str, Any]] = []

            for g in grasps_in:
                # 如果服务端已经直接返回 base_link 可执行位姿，则直接透传
                if "translation_base_retreat" in g and "quaternion_base" in g:
                    grasps_out.append(g)
                    continue

                translation_camera_retreat = g.get("translation_camera_retreat")
                if translation_camera_retreat is None:
                    translation_camera_retreat = g.get("translation_camera")
                if translation_camera_retreat is None:
                    continue

                R_cam = g.get("rotation_camera")
                if R_cam is None:
                    q_cam = g.get("quaternion_camera_xyzw")
                    if q_cam is not None:
                        R_cam = self._quat_to_rot_xyzw(q_cam)

                if R_cam is None:
                    # 没有旋转信息无法转换到 base_link
                    continue

                p_base = self._cam_to_base_point(translation_camera_retreat)
                R_base = self._cam_to_base_rot(R_cam)
                q_base = self._rot_to_quat_xyzw(R_base)

                R_cam_out = R_cam.tolist() if isinstance(R_cam, np.ndarray) else R_cam
                t_cam_ret_out = (
                    [float(x) for x in translation_camera_retreat]
                    if isinstance(translation_camera_retreat, (list, tuple, np.ndarray))
                    else translation_camera_retreat
                )

                grasps_out.append({
                    "score": float(g.get("score", 0.0)),
                    "width": float(g.get("width", 0.0)),
                    # 保留相机坐标系信息，便于调试
                    "translation_camera_retreat": t_cam_ret_out,
                    "rotation_camera": R_cam_out,
                    # 给执行用的 base_link 位姿
                    "translation_base_retreat": [float(x) for x in p_base.tolist()],
                    "quaternion_base": [float(x) for x in q_base.tolist()],
                })

            out.append({
                "label": label,
                "confidence": float(inst.get("confidence", 0.0)),
                "xyxy": [int(x) for x in xyxy] if isinstance(xyxy, list) else [],
                "grasps": grasps_out,
            })

        return out


if __name__ == "__main__":
    grasp = GraspSDK()
    with grasp:
        for _ in range(3):
            results = grasp.get_grasp_pose("bottle", top_k=3)
            for res in results:
                if not res["grasps"]:
                    print(f"[{res['label']}] 无抓取位姿")
                    continue
                best = res["grasps"][0]
                rx, ry, rz = best["translation_base_retreat"]
                qx, qy, qz, qw = best["quaternion_base"]
                print(
                    f"[{res['label']}] score={best['score']:.3f}\n"
                    f"  exec (retreat): ({rx:.4f}, {ry:.4f}, {rz:.4f})\n"
                    f"  quat: ({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})"
                )
            time.sleep(0.5)
