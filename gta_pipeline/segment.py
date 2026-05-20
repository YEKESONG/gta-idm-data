"""第 3 阶段：切片 + 抽帧。

把一段长视频切成若干 Clip（片段）。两种模式：
- "scene"：用 PySceneDetect 按镜头变化切。GTA 实况里"进商店/死亡/读条"
  通常伴随明显画面切换，按镜头切能让这些坏段落自然落在独立片段里，
  方便后续整片丢弃。
- "window"：固定时长窗口，简单稳健，适合连续 free-roam 录像。

关键设计：这一步**不真正剪切视频文件**，只产出 (start, end) 时间区间元数据。
真正的物理裁剪留到"通过过滤之后"再做（见 pipeline 的 clean 阶段），
避免给最终要丢弃的片段白白写盘。

抽帧：过滤阶段不需要逐帧检测，按 sample_fps（如每秒 2 帧）采样即可，
既快又足以判断一个片段属于哪类内容。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import SegmentConfig
from .download import DownloadedVideo


@dataclass
class Clip:
    source_path: Path
    index: int  # 在源视频中的序号
    start_s: float
    end_s: float
    fps: float

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s

    @property
    def clip_id(self) -> str:
        return f"{self.source_path.stem}_clip{self.index:04d}"


def segment_video(dl: DownloadedVideo, cfg: SegmentConfig) -> list[Clip]:
    if cfg.mode == "scene":
        spans = _scene_spans(dl.path, cfg.scene_threshold)
    else:
        spans = _window_spans(dl.path, cfg.window_seconds)

    # 统一做时长约束：丢弃过短片段，把过长片段再切成多段。
    clips: list[Clip] = []
    idx = 0
    for start, end in spans:
        for s, e in _enforce_length(start, end, cfg.min_clip_seconds, cfg.max_clip_seconds):
            clips.append(Clip(dl.path, idx, s, e, dl.fps))
            idx += 1
    return clips


def _video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return frames / fps if fps else 0.0


def _window_spans(path: Path, window: float) -> list[tuple[float, float]]:
    dur = _video_duration(path)
    spans = []
    t = 0.0
    while t < dur:
        spans.append((t, min(t + window, dur)))
        t += window
    return spans


def _scene_spans(path: Path, threshold: float) -> list[tuple[float, float]]:
    """用 PySceneDetect 检测镜头边界。失败时退回固定窗口。"""
    try:
        from scenedetect import detect, ContentDetector
    except Exception:
        print("[segment] 未安装 scenedetect，回退到固定窗口切片。")
        return _window_spans(path, window=10.0)

    try:
        scenes = detect(str(path), ContentDetector(threshold=threshold))
    except Exception as e:  # noqa: BLE001
        print(f"[segment] 场景检测失败 ({e})，回退固定窗口。")
        return _window_spans(path, window=10.0)

    return [(s.get_seconds(), e.get_seconds()) for s, e in scenes]


def _enforce_length(
    start: float, end: float, min_s: float, max_s: float
) -> list[tuple[float, float]]:
    dur = end - start
    if dur < min_s:
        return []  # 太短，直接丢
    if dur <= max_s:
        return [(start, end)]
    # 太长：均匀再切成 <= max_s 的小段。
    out = []
    t = start
    while t < end:
        out.append((t, min(t + max_s, end)))
        t += max_s
    return out


def sample_frames(clip: Clip, sample_fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """按 sample_fps 在片段时间区间内采样帧，产出 (相对时间戳, BGR 图像)。

    用 OpenCV 的时间定位（CAP_PROP_POS_MSEC）seek，避免把整段读进内存。
    """
    cap = cv2.VideoCapture(str(clip.source_path))
    try:
        step = 1.0 / max(sample_fps, 1e-6)
        t = clip.start_s
        while t < clip.end_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            yield t - clip.start_s, frame
            t += step
    finally:
        cap.release()
