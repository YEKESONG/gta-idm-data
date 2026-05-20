"""帧级检测器集合。

设计原则：每个检测器只回答"这一帧像不像某类要丢弃的内容"，
彼此独立、可单独开关。把多个检测器的逐帧结果汇总成"整片保留/丢弃"
的决策交给上层 policy 处理（关注点分离）。

扩展方式：
- 加一个新启发式检测器：在本目录新建文件，继承 FrameDetector，
  用 @register 注册，然后在配置的 enabled_detectors 里写上它的 name。
- 接入训练好的 ML 分类器：实现 classifier.FrameClassifier 协议，
  框架会用 ClassifierDetector 把它包成一个普通检测器，和启发式无缝混用。
"""

from .base import DetectorResult, FrameDetector, REGISTRY, register, build_detectors

# 导入各检测器以触发 @register 注册（副作用导入，必须保留）。
from . import death_screen, menu_shop, loading_black, motion, classifier  # noqa: F401

__all__ = [
    "DetectorResult",
    "FrameDetector",
    "REGISTRY",
    "register",
    "build_detectors",
]
