"""运动检测（光流幅度）。

我们要的是"移动 + 开火"数据，长时间站着不动的片段价值低。
本检测器估计相邻采样帧之间的整体运动强度：
- 用 Farneback 稠密光流算每个像素的位移，取平均幅度作为运动强度；
- 低于阈值判为 "static"（静止），否则 "moving"。

注意：static 本身不在 DISCARD_LABELS 里——是否因为"静止帧太多"丢整片，
由上层 policy 按比例决定（见 policy.static_frame_ratio_to_drop）。
这是有状态检测器：需要上一帧，故实现了 reset()。
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import DetectorResult, FrameDetector, register


@register
class MotionDetector(FrameDetector):
    name = "motion"

    def __init__(self, min_motion_magnitude: float = 0.6, resize_to: int = 160) -> None:
        self.min_motion_magnitude = min_motion_magnitude
        self.resize_to = resize_to  # 缩小后算光流，速度快很多且足够判断
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def __call__(self, frame: np.ndarray) -> DetectorResult:
        gray = self._prep(frame)
        if self._prev is None:
            self._prev = gray
            # 片段首帧无从比较，按"运动"放行，交给后续帧判断。
            return DetectorResult(self.name, "moving", 0.0, {"first_frame": True})

        flow = cv2.calcOpticalFlowFarneback(
            self._prev, gray, None,
            pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
        )
        mag = float(np.linalg.norm(flow, axis=2).mean())
        self._prev = gray

        label = "static" if mag < self.min_motion_magnitude else "moving"
        return DetectorResult(self.name, label, mag, {"motion_mag": round(mag, 3)})

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = self.resize_to / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
