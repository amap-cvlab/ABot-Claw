"""
空间记忆 SDK —— robot_sdk 封装层 (HTTP API 版)

通过 Spatial Memory Hub HTTP 服务完成物体记忆的写入与查询。
无需 ROS 依赖，直接在主进程中运行。

配置优先级: 构造函数参数 > 环境变量 SPATIAL_MEMORY_HUB_URL > config.yaml > 内置默认值

暴露接口:
    - upsert_object(...)   -> dict   写入/更新物体记忆，返回 {"ok": True, "id": "..."}
    - query_object(...)    -> list    按名称查询物体，返回匹配结果列表
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from config import get_config


@dataclass
class Pose:
    """6-DoF 位姿"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    frame_id: str = "map"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "frame_id": self.frame_id,
        }


class MemorySDK:
    """
    Spatial Memory Hub 物体记忆封装

    提供物体记忆的写入 (upsert) 与查询 (query) 能力。
    所有参数均可省略，默认从 config.yaml / 环境变量读取。

    用法:
        mem = MemorySDK()

        # 写入物体
        result = mem.upsert_object(
            object_name="red_cup",
            robot_id="humanoid_001",
            robot_type="humanoid",
            robot_pose=Pose(x=1.0, y=1.0),
            object_pose=Pose(x=1.2, y=1.1, z=0.8),
            detect_confidence=0.92,
            image_b64="...",
        )
        print(result)  # {"ok": True, "id": "abc123"}

        # 查询物体
        results = mem.query_object("red_cup", n_results=5)
        for r in results:
            print(r["name"], r["pose"])
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ):
        cfg = get_config()
        mem_cfg = cfg.get("spatial_memory", {})

        self._base_url = (
            base_url
            or os.environ.get("SPATIAL_MEMORY_HUB_URL")
            or mem_cfg.get("url", "http://30.79.84.82:8012")
        ).rstrip("/")
        self._timeout = (
            request_timeout
            if request_timeout is not None
            else mem_cfg.get("request_timeout", 10.0)
        )

    # ================================================================== #
    #                       内部工具
    # ================================================================== #

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(
            f"{self._base_url}{path}",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ================================================================== #
    #                       公开 API
    # ================================================================== #

    def upsert_object(
        self,
        object_name: str,
        robot_id: str,
        robot_type: str,
        robot_pose: Pose | Dict[str, Any],
        object_pose: Pose | Dict[str, Any],
        detect_confidence: float = 0.0,
        image_b64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        写入或更新一个物体记忆

        Args:
            object_name:        物体名称，如 "red_cup"
            robot_id:           机器人 ID，如 "humanoid_001"
            robot_type:         机器人类型，如 "humanoid"
            robot_pose:         观测时机器人位姿 (Pose 或 dict)
            object_pose:        物体位姿 (Pose 或 dict)
            detect_confidence:  检测置信度 (0~1)
            image_b64:          可选，base64 编码的物体图像

        Returns:
            dict: {"ok": True, "id": "..."}

        Raises:
            requests.HTTPError: 服务端返回非 2xx
            RuntimeError:       响应中 ok != True
        """
        payload: Dict[str, Any] = {
            "object_name": object_name,
            "robot_id": robot_id,
            "robot_type": robot_type,
            "robot_pose": robot_pose.to_dict() if isinstance(robot_pose, Pose) else robot_pose,
            "object_pose": object_pose.to_dict() if isinstance(object_pose, Pose) else object_pose,
            "detect_confidence": detect_confidence,
        }
        if image_b64 is not None:
            payload["image"] = image_b64

        data = self._post("/memory/object/upsert", payload)
        if not data.get("ok"):
            raise RuntimeError(f"upsert_object 失败: {data}")
        return data

    def insert_object(
        self,
        object_name: str,
        robot_id: str,
        robot_type: str,
        robot_pose: Pose | Dict[str, Any],
        object_pose: Pose | Dict[str, Any],
        detect_confidence: float = 0.0,
        image_b64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """插入（别名 upsert）：写入/更新一个物体记忆"""
        return self.upsert_object(
            object_name=object_name,
            robot_id=robot_id,
            robot_type=robot_type,
            robot_pose=robot_pose,
            object_pose=object_pose,
            detect_confidence=detect_confidence,
            image_b64=image_b64,
        )

    def query_object(
        self,
        name: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        按名称查询物体记忆

        Args:
            name:       要查询的物体名称（支持模糊匹配）
            n_results:  最多返回条数

        Returns:
            list[dict]: 匹配的物体记录列表，每条包含 name / pose / confidence 等字段；
                        无匹配时返回空列表
        """
        data = self._post("/query/object", {"name": name, "n_results": n_results})
        return data.get("results", [])


if __name__ == "__main__":
    import base64
    import os

    image_path = "test.png"

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    with open(image_path, "rb") as f:
        demo_b64 = base64.b64encode(f.read()).decode("utf-8")

    mem = MemorySDK(base_url="http://30.79.84.82:8012")

    print("=== upsert_object ===")
    result = mem.upsert_object(
        object_name="red_cup_demo",
        robot_id="humanoid_001",
        robot_type="humanoid",
        robot_pose=Pose(x=1.0, y=1.0),
        object_pose=Pose(x=1.2, y=1.1, z=0.8),
        detect_confidence=0.92,
        image_b64=demo_b64,
    )
    print(result)

    print("\n=== query_object ===")
    results = mem.query_object("red_cup_demo")
    for r in results:
        print(r)
