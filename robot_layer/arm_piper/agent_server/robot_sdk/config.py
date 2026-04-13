"""
统一配置加载器

从 config.yaml 读取机器人相关配置，供各 SDK 模块使用。
支持通过环境变量 ROBOT_SDK_CONFIG 指定配置文件路径。

用法:
    from config import get_config
    cfg = get_config()
    print(cfg["ros"]["image_topic"])
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.yaml"
)

_cached_config: Dict[str, Any] | None = None


def get_config(config_path: str | None = None) -> Dict[str, Any]:
    """加载并缓存配置。

    优先级: 参数 > 环境变量 ROBOT_SDK_CONFIG > 同目录下 config.yaml
    """
    global _cached_config

    if config_path is None and _cached_config is not None:
        return _cached_config

    path = config_path or os.environ.get("ROBOT_SDK_CONFIG", _DEFAULT_CONFIG_PATH)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if config_path is None:
        _cached_config = cfg
    return cfg


def reload_config(config_path: str | None = None) -> Dict[str, Any]:
    """强制重新加载配置 (清除缓存)"""
    global _cached_config
    _cached_config = None
    return get_config(config_path)
