"""第 2 阶段：下载。

把 VideoRef 下载成本地 mp4，并统一规格（分辨率上限、固定帧率）。
固定帧率很关键：后续要按帧给动作打标签（IDM），帧率不一致会让
"第 t 帧"对应的真实时间错位。

依赖 ffmpeg：yt-dlp 用它做转码/重采样。若系统没有 ffmpeg，
下载能完成但无法按 fps 重采样——会在 download_one 里给出明确提示。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import DownloadConfig
from .discovery import VideoRef


@dataclass
class DownloadedVideo:
    ref: VideoRef
    path: Path
    fps: int
    height: int


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def download_one(
    ref: VideoRef, cfg: DownloadConfig, out_dir: Path
) -> DownloadedVideo | None:
    """下载单个视频，按 cfg 规格输出。失败返回 None（不抛异常，便于批处理跳过）。"""
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    # 文件名用 平台_视频id，保证跨平台唯一且可追溯到来源。
    out_tmpl = str(out_dir / f"{ref.platform}_{ref.video_id}.%(ext)s")

    # 只下不超过 max_height 的画质。关键：优先 H.264(avc1)编码——
    # OpenCV 自带的 ffmpeg 解不了 AV1（会刷 "Failed to get pixel format"），
    # 而 YouTube 在 <=720p 几乎总提供 avc1。最后兜底 best 防止彻底下不到。
    h = cfg.max_height
    fmt = (
        f"bestvideo[height<={h}][vcodec^=avc1]+bestaudio/"
        f"best[height<={h}][vcodec^=avc1]/"
        f"bestvideo[height<={h}]+bestaudio/"
        f"best[height<={h}]"
    )

    postprocessors = []
    if _has_ffmpeg():
        # 用 ffmpeg 把帧率重采样到固定 fps。
        postprocessors.append(
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        )

    opts = {
        "format": fmt,
        "outtmpl": out_tmpl,
        "quiet": True,
        "ignoreerrors": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
        # 启用 EJS solver 远程组件，破解 YouTube n-challenge 拿全清晰度。
        **({"remote_components": cfg.remote_components} if cfg.remote_components else {}),
        "postprocessors": postprocessors,
        # ffmpeg 重采样帧率（-r）。放 postprocessor_args 里透传给转码步骤。
        "postprocessor_args": {"videoconvertor": ["-r", str(cfg.fps)]} if _has_ffmpeg() else {},
        **cfg.extra_ydl_opts,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(ref.url, download=True)
    except Exception as e:  # noqa: BLE001 — 批处理里单个失败不应中断全局
        print(f"[download] 失败 {ref.url}: {e}")
        return None

    if not info:
        return None

    # 时长超限的直接判废（在拿到完整 info 后才知道准确时长）。
    duration = info.get("duration") or 0
    if cfg.max_duration_s and duration > cfg.max_duration_s:
        print(f"[download] 跳过超长视频 {ref.video_id} ({duration}s)")
        return None

    # 找到实际落盘的文件。
    produced = list(out_dir.glob(f"{ref.platform}_{ref.video_id}.*"))
    video_files = [p for p in produced if p.suffix.lower() in {".mp4", ".mkv", ".webm"}]
    if not video_files:
        return None

    return DownloadedVideo(
        ref=ref,
        path=video_files[0],
        fps=cfg.fps,
        height=min(cfg.max_height, info.get("height") or cfg.max_height),
    )


def download_many(
    refs: list[VideoRef], cfg: DownloadConfig, out_dir: Path
) -> list[DownloadedVideo]:
    if not _has_ffmpeg():
        print(
            "[download] 警告：未检测到 ffmpeg，无法重采样帧率/统一容器。"
            "请先安装 ffmpeg（如 `sudo apt install ffmpeg`）以保证后续按帧对齐。"
        )
    out: list[DownloadedVideo] = []
    for ref in refs:
        dl = download_one(ref, cfg, out_dir)
        if dl:
            out.append(dl)
    return out
