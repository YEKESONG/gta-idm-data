"""读条 / 黑屏 / 转场画面检测。

加载和转场常表现为：整屏接近全黑，或整屏接近单一颜色（纯色过渡）。
这两种都没有动作信息，需丢弃。

启发式：
- 平均亮度极低 → 黑屏；
- 或全图亮度方差极低（几乎纯色）→ 转场/读条底色。
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import DetectorResult, FrameDetector, register


@register
class LoadingBlackDetector(FrameDetector):
    name = "loading_black"

    def __init__(
        self,
        brightness_threshold: float = 18.0,  # 平均亮度低于此值视为黑屏
        variance_threshold: float = 8.0,  # 亮度标准差低于此值视为纯色
    ) -> None:
        self.brightness_threshold = brightness_threshold
        self.variance_threshold = variance_threshold

    def __call__(self, frame: np.ndarray) -> DetectorResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_b = float(gray.mean())
        std_b = float(gray.std())

        is_black = mean_b < self.brightness_threshold
        is_uniform = std_b < self.variance_threshold

        score = 1.0 if (is_black or is_uniform) else 0.0
        label = "loading" if score > 0 else "gameplay"
        info = {"mean_brightness": round(mean_b, 2), "std": round(std_b, 2)}
        return DetectorResult(self.name, label, score, info)
