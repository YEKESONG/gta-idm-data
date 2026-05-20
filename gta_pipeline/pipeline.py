"""把四个阶段串成一条端到端流水线。

    discover -> download -> (per video) segment -> (per clip) filter -> export + manifest

设计成"每个阶段函数都能单独调用"，pipeline.run() 只是把它们按顺序粘起来，
方便你在 notebook 里单独调试某一步。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import PipelineConfig
from .discovery import discover
from .download import DownloadedVideo, download_many
from .filters.base import build_detectors
from .manifest import ManifestWriter
from .policy import decide_clip
from .segment import Clip, sample_frames, segment_video


def run(cfg: PipelineConfig) -> None:
    """跑完整条流水线。"""
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    # 1) 发现候选视频（只取元数据）。
    refs = discover(
        platforms=cfg.discovery.platforms,
        queries=cfg.discovery.queries,
        seed_urls=cfg.discovery.seed_urls,
        max_results_per_query=cfg.discovery.max_results_per_query,
    )
    print(f"[pipeline] 发现 {len(refs)} 个候选视频")

    # 2) 下载。
    downloaded = download_many(refs, cfg.download, cfg.raw_dir)
    print(f"[pipeline] 成功下载 {len(downloaded)} 个视频")

    # 3+4) 逐视频切片、逐片过滤、导出、写清单。
    with ManifestWriter(cfg.manifest_path) as manifest:
        kept = 0
        for dl in downloaded:
            kept += _process_video(dl, cfg, manifest)
    print(f"[pipeline] 完成。保留干净片段 {kept} 个 -> {cfg.clean_dir}")


def _process_video(dl: DownloadedVideo, cfg: PipelineConfig, manifest: ManifestWriter) -> int:
    clips = segment_video(dl, cfg.segment)
    print(f"[pipeline] {dl.path.name}: 切出 {len(clips)} 个片段")

    # 每个视频构建一套新的检测器实例（含可插拔分类器）。
    detectors = build_detectors(
        cfg.filter.enabled_detectors,
        min_motion_magnitude=cfg.filter.min_motion_magnitude,
        classifier_ckpt=cfg.filter.classifier_ckpt,
        classifier_threshold=cfg.filter.classifier_threshold,
    )

    kept = 0
    for clip in clips:
        for d in detectors:
            d.reset()  # 清掉上一个片段留下的帧间状态（如光流的 prev）

        # 对该片段采样帧，逐帧跑所有检测器。
        per_frame_results = []
        for _ts, frame in sample_frames(clip, cfg.segment.sample_fps):
            per_frame_results.append([d(frame) for d in detectors])

        decision = decide_clip(per_frame_results, cfg.filter)
        if not decision.keep:
            continue

        exported = export_clip(clip, cfg.clean_dir)
        manifest.write(clip, dl.ref.platform, exported, decision)
        kept += 1
    return kept


def export_clip(clip: Clip, out_dir: Path) -> Path:
    """用 ffmpeg 把通过过滤的时间区间裁成独立 mp4。

    没有 ffmpeg 时不裁剪，清单里记录源视频 + 时间区间（训练时按区间读取即可）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip.clip_id}.mp4"

    if shutil.which("ffmpeg") is None:
        return clip.source_path  # 退化：保留源路径，区间信息已在 manifest

    # -ss 放在 -i 前是快速 seek；用 -t（时长）而非 -to，避免两者混用时的歧义。
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{clip.start_s:.3f}",
        "-t", f"{clip.duration:.3f}",
        "-i", str(clip.source_path),
        "-r", str(int(clip.fps)),  # 固定帧率，保证按帧对齐动作标签
        "-an",  # 去音轨（动作建模用不到）
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        return out_path
    except subprocess.CalledProcessError as e:
        print(f"[export] ffmpeg 裁剪失败 {clip.clip_id}: {e}")
        return clip.source_path
