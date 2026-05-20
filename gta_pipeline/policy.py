"""把逐帧检测结果汇总成"整片保留 / 丢弃"的决策。

为什么单独一层：检测器只管单帧打分，"多少坏帧才丢整片"是策略问题，
分开后既能复用同一批检测器试不同阈值，也方便讲清楚每片为何被丢。

规则：
- 坏帧：任一检测器把该帧判为 death/menu/loading/non_gameplay；
- 静止帧：运动检测器判为 static；
- 坏帧占比 > bad_frame_ratio_to_drop  → 丢（含死亡/菜单/读条太多）；
- 静止帧占比 > static_frame_ratio_to_drop → 丢（缺少移动/开火动作）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import FilterConfig
from .filters.base import DetectorResult


@dataclass
class ClipDecision:
    keep: bool
    reason: str
    n_frames: int
    bad_frame_ratio: float
    static_frame_ratio: float
    # 各坏标签计数，便于分析数据集里到底过滤掉了什么。
    label_counts: dict[str, int] = field(default_factory=dict)


def decide_clip(
    per_frame_results: list[list[DetectorResult]], cfg: FilterConfig
) -> ClipDecision:
    """per_frame_results[i] 是第 i 个采样帧上所有检测器的结果列表。"""
    n = len(per_frame_results)
    if n == 0:
        return ClipDecision(False, "no_frames", 0, 0.0, 0.0)

    bad_frames = 0
    static_frames = 0
    label_counts: dict[str, int] = {}

    for frame_results in per_frame_results:
        frame_is_bad = False
        for r in frame_results:
            if r.is_discard:
                frame_is_bad = True
                label_counts[r.label] = label_counts.get(r.label, 0) + 1
            if r.name == "motion" and r.label == "static":
                static_frames += 1
        if frame_is_bad:
            bad_frames += 1

    bad_ratio = bad_frames / n
    static_ratio = static_frames / n

    if bad_ratio > cfg.bad_frame_ratio_to_drop:
        return ClipDecision(False, "too_many_bad_frames", n, bad_ratio, static_ratio, label_counts)
    if static_ratio > cfg.static_frame_ratio_to_drop:
        return ClipDecision(False, "too_static", n, bad_ratio, static_ratio, label_counts)

    return ClipDecision(True, "ok", n, bad_ratio, static_ratio, label_counts)
