"""Piper 机器人状态聚合器 — 从 PiperRobotEnv 采集并维护快照。"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StateAggregator:
    """周期轮询 PiperRobotEnv，维护最新的机器人状态快照。

    状态结构:
      {
        "timestamp": float,
        "arm": {
          "joint_positions":  [6 floats]  关节角度 (rad),
          "joint_velocities": [6 floats]  关节速度 (rad/s),
          "end_pose": {
            "position":          [x, y, z],
            "orientation_quat":  [qx, qy, qz, qw],
            "orientation_euler": [roll, pitch, yaw],
          } | None,
        },
        "gripper": {
          "position": float,  开合度 (m), 0.0=关闭, 0.06=全开
        },
        "cameras": {
          "left_camera_0_left":  bool,  是否有最新帧
          "wrist_camera_0_left": bool,
        },
      }
    """

    def __init__(self, env=None, poll_hz: float = 10.0) -> None:
        """
        Args:
            env: PiperRobotEnv 实例；为 None 时状态全为空（dry-run / 测试）
            poll_hz: 轮询频率
        """
        self._env = env
        self._poll_hz = poll_hz
        self._state: dict[str, Any] = self._empty_state()
        self._task: Optional[asyncio.Task] = None
        self._last_moved_at: float = 0.0
        self._prev_joint_positions: list[float] = []

    # ------------------------------------------------------------------ #
    #  Public helpers
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    def last_moved_at(self) -> float:
        """返回最近一次检测到机器人运动的时间戳。"""
        return self._last_moved_at

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "timestamp": 0.0,
            "arm": {
                "joint_positions": [],
                "joint_velocities": [],
                "end_pose": None,
            },
            "gripper": {
                "position": 0.0,
            },
            "cameras": {},
        }

    def _update_movement(self, joint_positions: list) -> None:
        if joint_positions and self._prev_joint_positions:
            if any(
                abs(a - b) > 0.005
                for a, b in zip(joint_positions, self._prev_joint_positions)
            ):
                self._last_moved_at = time.time()
        if joint_positions:
            self._prev_joint_positions = list(joint_positions)

    async def _poll_loop(self) -> None:
        interval = 1.0 / self._poll_hz
        while True:
            try:
                await self._poll_once()
            except Exception:
                logger.exception("State poll error")
            await asyncio.sleep(interval)

    async def _poll_once(self) -> None:
        if self._env is None:
            self._state = self._empty_state()
            self._state["timestamp"] = time.time()
            return

        loop = asyncio.get_event_loop()

        # 关节状态
        robot_state: dict = {}
        try:
            robot_state = await loop.run_in_executor(None, self._env.get_robot_state)
        except Exception as e:
            logger.debug("get_robot_state failed: %s", e)

        joint_positions = list(robot_state.get("joint_positions", []))
        joint_velocities = list(robot_state.get("joint_velocities", []))
        gripper_position = float(
            (robot_state.get("gripper_position") or [0.0])[0]
            if robot_state.get("gripper_position") is not None
            else 0.0
        )

        # 末端位姿
        end_pose = None
        try:
            end_pose = await loop.run_in_executor(None, self._env.get_robot_end_pose)
        except Exception as e:
            logger.debug("get_robot_end_pose failed: %s", e)

        # 相机可用性（只检查帧是否为 None，不解码）
        cameras: dict[str, bool] = {}
        try:
            images, _ = await loop.run_in_executor(None, self._env.read_cameras)
            cameras = {name: img is not None for name, img in images.items()}
        except Exception as e:
            logger.debug("read_cameras failed: %s", e)

        self._update_movement(joint_positions)

        self._state = {
            "timestamp": time.time(),
            "arm": {
                "joint_positions": joint_positions,
                "joint_velocities": joint_velocities,
                "end_pose": end_pose,
            },
            "gripper": {
                "position": gripper_position,
            },
            "cameras": cameras,
        }
