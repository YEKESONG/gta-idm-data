"""数据清单（manifest）写入。

用 JSONL（每行一个 JSON）记录每个保留片段的元数据：来源、时间区间、
过滤统计等。JSONL 适合流式追加、易 grep、可被 PyTorch Dataset 直接读。
后续 IDM / world model 训练时按这个清单加载片段。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .policy import ClipDecision
from .segment import Clip


@dataclass
class ManifestEntry:
    clip_id: str
    source_video: str
    platform: str
    start_s: float
    end_s: float
    fps: float
    exported_path: str
    decision_reason: str
    bad_frame_ratio: float
    static_frame_ratio: float
    label_counts: dict


class ManifestWriter:
    """逐行追加的清单写入器（上下文管理器）。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = None

    def __enter__(self) -> "ManifestWriter":
        self._f = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        if self._f:
            self._f.close()

    def write(self, clip: Clip, platform: str, exported_path: Path, decision: ClipDecision) -> None:
        entry = ManifestEntry(
            clip_id=clip.clip_id,
            source_video=clip.source_path.name,
            platform=platform,
            start_s=round(clip.start_s, 3),
            end_s=round(clip.end_s, 3),
            fps=clip.fps,
            exported_path=str(exported_path),
            decision_reason=decision.reason,
            bad_frame_ratio=round(decision.bad_frame_ratio, 4),
            static_frame_ratio=round(decision.static_frame_ratio, 4),
            label_counts=decision.label_counts,
        )
        assert self._f is not None
        self._f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        self._f.flush()
