"""
AnyGrasp 抓取位姿服务 

基于 YoloDetector 进行目标检测，再用 AnyGrasp 生成 6DoF 抓取位姿。
不做任何可达性过滤，直接按 AnyGrasp 分数排序返回。

暴露接口：
  - get_grasp_pose(object_name, top_k) -> List[GraspResult]
    返回指定物体在 base_link 下的 6DoF 抓取位姿 (纯 AnyGrasp 分数排序)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import rospy

from yolo_service import YoloDetector

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "grasp_detection"))
from gsnet import AnyGrasp


def _rot_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """3x3 旋转矩阵 -> 四元数 (x, y, z, w)。"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    return (x / norm, y / norm, z / norm, w / norm)


@dataclass(frozen=True)
class Grasp6D:
    """单个 6DoF 抓取位姿 (camera 坐标系)。"""
    score: float
    width: float
    translation_camera: Tuple[float, float, float]
    rotation_camera: np.ndarray                      # 3x3
    translation_camera_retreat: Tuple[float, float, float]
    approach_retreat: float

    def quaternion_camera(self) -> Tuple[float, float, float, float]:
        """camera 坐标系下的四元数 (x, y, z, w)。"""
        return _rot_to_quat(self.rotation_camera)


@dataclass(frozen=True)
class GraspResult:
    """一个物体的 YOLO 识别 + AnyGrasp 抓取结果。"""
    label: str
    confidence: float
    xyxy: Tuple[int, int, int, int]
    grasps: List[Grasp6D]


# ======================================================================
# 主服务类 (无过滤)
# ======================================================================

class GraspDetectorNoFilter:
    """
    YOLO + AnyGrasp 抓取位姿服务（无机械臂约束）。

    接口：
      get_grasp_pose(object_name, top_k) -> List[GraspResult]

    直接按 AnyGrasp 分数排序返回，不做任何可达性过滤。
    """

    def __init__(
        self,
        checkpoint_path: str,
        max_gripper_width: float = 0.1,
        gripper_height: float = 0.03,
        top_down_grasp: bool = True,
        approach_retreat: float = 0.10,
        depth_scale: float = 1000.0,
        z_range: Tuple[float, float] = (0.01, 1.0),
        bbox_padding: int = 10,
        **yolo_kwargs,
    ) -> None:
        self._yolo = YoloDetector(**yolo_kwargs)

        self._approach_retreat = approach_retreat
        self._depth_scale = depth_scale
        self._z_min, self._z_max = z_range
        self._bbox_padding = bbox_padding

        cfg = SimpleNamespace(
            checkpoint_path=checkpoint_path,
            max_gripper_width=max(0, min(0.1, max_gripper_width)),
            gripper_height=gripper_height,
            top_down_grasp=top_down_grasp,
            debug=False,
        )
        self._anygrasp = AnyGrasp(cfg)
        self._anygrasp.load_net()
        rospy.loginfo("AnyGrasp 模型加载完成: %s", checkpoint_path)

    @property
    def yolo(self) -> YoloDetector:
        return self._yolo

    def start(self) -> "GraspDetectorNoFilter":
        self._yolo.start()
        return self

    def stop(self) -> None:
        self._yolo.stop()

    def __enter__(self) -> "GraspDetectorNoFilter":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # 内部：构建物体点云
    # ------------------------------------------------------------------

    def _build_object_pointcloud(
        self,
        color_bgr: np.ndarray,
        depth_map: np.ndarray,
        K: np.ndarray,
        xyxy: Tuple[int, int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = depth_map.shape[:2]
        pad = self._bbox_padding
        x1, y1 = max(0, xyxy[0] - pad), max(0, xyxy[1] - pad)
        x2, y2 = min(w, xyxy[2] + pad), min(h, xyxy[3] + pad)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        depth_roi = depth_map[y1:y2, x1:x2].astype(np.float32)
        color_rgb = color_bgr[y1:y2, x1:x2, ::-1].astype(np.float32) / 255.0

        xmap, ymap = np.meshgrid(
            np.arange(x1, x2, dtype=np.float32),
            np.arange(y1, y2, dtype=np.float32),
        )

        pz = depth_roi / self._depth_scale
        px = (xmap - cx) / fx * pz
        py = (ymap - cy) / fy * pz

        mask = (pz > self._z_min) & (pz < self._z_max)
        points = np.stack([px, py, pz], axis=-1)[mask].astype(np.float32)
        colors = color_rgb[mask].astype(np.float32)
        return points, colors

    def _as_dict(self, results: List[GraspResult], object_name: str, top_k: int) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "frame_id": self._yolo._camera_frame_id,
            "target": object_name,
            "top_k": top_k,
            "count": len(results),
            "results": [],
        }
        for res in results:
            item = {
                "label": res.label,
                "confidence": res.confidence,
                "xyxy": list(res.xyxy),
                "grasps": [],
            }
            for g in res.grasps:
                qx, qy, qz, qw = g.quaternion_camera()
                item["grasps"].append({
                    "score": g.score,
                    "width": g.width,
                    "translation_camera": list(g.translation_camera),
                    "translation_camera_retreat": list(g.translation_camera_retreat),
                    "quaternion_camera_xyzw": [qx, qy, qz, qw],
                    "rotation_camera": g.rotation_camera.tolist(),
                })
            data["results"].append(item)
        return data

    def get_grasp_pose_from_frame(
        self,
        color_bgr: np.ndarray,
        depth_map: np.ndarray,
        K: np.ndarray,
        object_name: str,
        top_k: int = 5,
    ) -> List[GraspResult]:
        """
        输入单帧 RGB-D 与目标类别，输出相机坐标系抓取位姿。
        这是无 service 的直接调用接口。
        """
        det_results = self._yolo._model(color_bgr)
        df = det_results.pandas().xyxy[0]
        if df is None or len(df) == 0:
            return []
        df = df[df["name"] == object_name]
        if len(df) == 0:
            return []

        results: List[GraspResult] = []
        for row in df.itertuples(index=False):
            xyxy = (int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax))
            points, colors = self._build_object_pointcloud(color_bgr, depth_map, K, xyxy)
            if len(points) < 50:
                rospy.logwarn("'%s' 点云过稀 (%d 点)，跳过", object_name, len(points))
                continue

            p_min, p_max = points.min(axis=0), points.max(axis=0)
            margin = 0.02
            lims = [
                float(p_min[0] - margin), float(p_max[0] + margin),
                float(p_min[1] - margin), float(p_max[1] + margin),
                float(p_min[2] - margin), float(p_max[2] + margin),
            ]
            try:
                gg, _ = self._anygrasp.get_grasp(
                    points, colors, lims=lims,
                    apply_object_mask=True, dense_grasp=False, collision_detection=True,
                )
            except Exception as e:
                rospy.logwarn("AnyGrasp 检测失败: %s", e)
                continue

            if len(gg) == 0:
                results.append(GraspResult(
                    label=str(row.name), confidence=float(row.confidence),
                    xyxy=xyxy, grasps=[],
                ))
                continue

            gg = gg.nms().sort_by_score()
            candidates = gg[0:min(len(gg), top_k)]
            grasp_list: List[Grasp6D] = []
            for g in candidates:
                t_cam = (float(g.translation[0]), float(g.translation[1]), float(g.translation[2]))
                R_cam = np.array(g.rotation_matrix, dtype=np.float64)
                approach = R_cam[:, 2]
                t_retreat = (
                    t_cam[0] - approach[0] * self._approach_retreat,
                    t_cam[1] - approach[1] * self._approach_retreat,
                    t_cam[2] - approach[2] * self._approach_retreat,
                )
                grasp_list.append(Grasp6D(
                    score=float(g.score),
                    width=float(g.width),
                    translation_camera=t_cam,
                    rotation_camera=R_cam,
                    translation_camera_retreat=t_retreat,
                    approach_retreat=self._approach_retreat,
                ))

            results.append(GraspResult(
                label=str(row.name),
                confidence=float(row.confidence),
                xyxy=xyxy,
                grasps=grasp_list,
            ))
        return results

    def get_grasp_pose(
        self,
        object_name: str,
        top_k: int = 5,
    ) -> List[GraspResult]:
        """
        对指定物体执行 YOLO 检测 + AnyGrasp 6DoF 抓取位姿生成。

        直接按 AnyGrasp 分数降序排列，不做任何机械臂可达性过滤。

        参数:
          object_name: COCO 类别名 (如 "cup", "bottle", "orange")
          top_k: 每个物体最多返回的抓取数量

        返回:
          List[GraspResult]，每个检测到的物体实例一个，
          内含最多 top_k 个 camera 下的 6DoF 位姿。
        """
        if not self._yolo._started:
            self.start()

        with self._yolo._lock:
            color_bgr = None if self._yolo._last_color_bgr is None else self._yolo._last_color_bgr.copy()
            depth_map = None if self._yolo._last_depth is None else self._yolo._last_depth.copy()
            K = None if self._yolo._camera_K is None else self._yolo._camera_K.copy()

        if color_bgr is None:
            raise RuntimeError(f"尚未收到彩色图像: {self._yolo._image_topic}")
        if depth_map is None:
            raise RuntimeError(f"尚未收到深度图: {self._yolo._depth_topic}")
        if K is None:
            raise RuntimeError(f"尚未收到相机内参: {self._yolo._camera_info_topic}")
        return self.get_grasp_pose_from_frame(color_bgr, depth_map, K, object_name, top_k)


# ======================================================================
# 快速测试
# ======================================================================
if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(description="AnyGrasp 抓取位姿检测 (无约束, 相机坐标系输出)")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--target", default="orange")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    grasp = GraspDetectorNoFilter(
        checkpoint_path=args.checkpoint_path,
        model_name="yolov5l6",
        conf=0.5,
    )

    with grasp:
        results = grasp.get_grasp_pose(args.target, top_k=args.top_k)
        print(json.dumps(grasp._as_dict(results, args.target, args.top_k), ensure_ascii=False, indent=2))
