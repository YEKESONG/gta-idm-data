"""检测器基类、结果结构与注册表。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# 我们想丢弃的内容类别。motion 检测器用 "static"/"moving"，
# 其余检测器命中坏内容时给出对应标签。
DISCARD_LABELS = {"death", "menu", "loading", "non_gameplay"}


@dataclass
class DetectorResult:
    """单个检测器对单帧的判断。"""

    name: str  # 检测器名字
    label: str  # 该检测器给这帧打的标签，如 "death" / "gameplay" / "static"
    score: float  # 置信度 / 强度，含义由具体检测器定义（0~1 或幅度值）
    info: dict[str, Any] = field(default_factory=dict)  # 调试用附加信息

    @property
    def is_discard(self) -> bool:
        return self.label in DISCARD_LABELS


class FrameDetector(ABC):
    """逐帧检测器基类。

    检测器可以是**有状态**的（如运动检测需要上一帧），
    因此每处理完一个片段，pipeline 会调用 reset() 清状态。
    """

    name: str = "base"

    def reset(self) -> None:
        """切换到新片段时调用，清除帧间状态。默认无状态。"""

    @abstractmethod
    def __call__(self, frame: np.ndarray) -> DetectorResult:
        """frame 为 BGR (H, W, 3) uint8。"""


# ---- 注册表：把检测器名字映射到类，便于按配置动态构建 ----
REGISTRY: dict[str, type[FrameDetector]] = {}


def register(cls: type[FrameDetector]) -> type[FrameDetector]:
    """类装饰器：把检测器登记进 REGISTRY。"""
    REGISTRY[cls.name] = cls
    return cls


def build_detectors(names: list[str], **kwargs: Any) -> list[FrameDetector]:
    """按名字列表实例化检测器。

    kwargs 会透传给检测器构造函数（用于传阈值、模型路径等）；
    每个检测器只挑自己 __init__ 认识的参数。
    """
    import inspect

    detectors: list[FrameDetector] = []
    for name in names:
        if name not in REGISTRY:
            raise KeyError(f"未知检测器 '{name}'，已注册：{sorted(REGISTRY)}")
        cls = REGISTRY[name]
        sig = inspect.signature(cls.__init__)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        det = cls(**accepted)
        # 分类器在没有权重时会自我禁用，跳过它以免空跑。
        if getattr(det, "disabled", False):
            continue
        detectors.append(det)
    return detectors
