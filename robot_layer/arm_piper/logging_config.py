from __future__ import annotations

import logging
import os
from collections import deque
from logging.handlers import RotatingFileHandler


_LOG_BUFFER = None  # type: deque[str] | None


class _InMemoryLogBuffer(logging.Handler):
    """简单的内存日志缓冲，用于 /state/logs 接口读取。"""

    def __init__(self, maxlen: int = 1000) -> None:
        super().__init__()
        self._buffer: deque[str] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._buffer.append(msg)

    def get_logs(self, limit: int) -> list[str]:
        if limit <= 0:
            return []
        return list(self._buffer)[-limit:]


def get_log_buffer() -> _InMemoryLogBuffer | None:
    """提供给 /state/logs 使用的全局日志缓冲。"""
    global _LOG_BUFFER
    return _LOG_BUFFER


def setup_logging(name: str = "agent_server") -> logging.Logger:
    """
    简单的日志初始化函数，返回配置好的 logger。
    - 输出到控制台
    - 输出到内存缓冲（用于前端拉取日志）
    - 可选输出到项目根目录下的 logs/<name>.log（滚动日志）
    """
    global _LOG_BUFFER

    logger = logging.getLogger(name)

    # 避免重复添加 handler（多次调用或热重载时）
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 内存日志缓冲（供 /state/logs 读取）
    _LOG_BUFFER = _InMemoryLogBuffer(maxlen=2000)
    _LOG_BUFFER.setFormatter(formatter)
    logger.addHandler(_LOG_BUFFER)

    # 可选：文件日志（如果有写权限就启用）
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"{name}.log")

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        # 如果文件日志初始化失败，不影响控制台日志
        pass

    return logger

