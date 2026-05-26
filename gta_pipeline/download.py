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
from typing import Any

from .config import DownloadConfig
from .discovery import VideoRef


@dataclass
class DownloadedVideo:
    ref: VideoRef
    path: Path
    fps: int
    height: int
    duration_s: float = 0.0  # 源视频时长，用于按累计小时数收集数据


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_progress_hook():
    """生成 yt-dlp 下载进度回调；每推进约 10% 打印一行（避免刷屏）。

    即便 opts 里 quiet=True，progress_hooks 仍会被调用——这是静默下载时
    仍能看到百分比的标准做法。每个视频用独立闭包，进度状态互不干扰。
    """
    state = {"last_pct": -10}

    def hook(d: dict) -> None:
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            got = d.get("downloaded_bytes") or 0
            if total:
                pct = int(got * 100 / total)
                if pct >= state["last_pct"] + 10:
                    state["last_pct"] = pct
                    speed = (d.get("speed") or 0) / 1e6
                    print(f"[download]   {pct:3d}%  {got / 1e6:.0f}/{total / 1e6:.0f} MB  {speed:.1f} MB/s")
        elif status == "finished":
            print("[download]   分流下载完成，正在合并/转码 ...")

    return hook


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
        # 仅做容器归一化为 mp4，方便 OpenCV 读取。
        # 注意：不在这里重采样帧率——下载流若已是 mp4，转换器会跳过，-r 不生效。
        # 帧率统一改到切片导出（export_clip 的 -r）那一步，对源文件保留原生帧率即可。
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
        "progress_hooks": [_make_progress_hook()],  # quiet 下仍打印下载百分比
        "ignoreerrors": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
        # 启用 EJS solver 远程组件，破解 YouTube n-challenge 拿全清晰度。
        **({"remote_components": cfg.remote_components} if cfg.remote_components else {}),
        # 批量抓取礼貌性：每次下载前随机等待，降低被站点限流的风险。
        **(
            {"sleep_interval": cfg.sleep_interval_s, "max_sleep_interval": cfg.sleep_interval_s * 2}
            if cfg.sleep_interval_s > 0
            else {}
        ),
        "postprocessors": postprocessors,
        **cfg.extra_ydl_opts,
    }

    # 本地已有完整文件（非 .part）则豁免时长上限：已经下好的视频（含之前下的超长录像）
    # 直接复用、处理掉，不浪费已花的下载。时长上限只用于拦截「尚未下载的」新超长视频，
    # 避免再白下几个 G。这样 max_duration_s 可常设，已下的超长仍会被消化。
    have_local = any(
        p.suffix.lower() in {".mp4", ".mkv", ".webm"}
        for p in out_dir.glob(f"{ref.platform}_{ref.video_id}.*")
    )

    # ---- 下载前先按时长粗筛：超长直接跳过，绝不把整段下到本地再丢弃 ----
    # 优先用发现阶段已拿到的时长（discover 的 flat 模式通常已带 duration）；
    # 拿不到时再做一次轻量探测（只解析元数据、不下载），代价远小于白下几个 G。
    duration = ref.duration_s
    if duration is None and not have_local:
        duration = _probe_duration(ref.url, opts)
    if not have_local and cfg.max_duration_s and duration and duration > cfg.max_duration_s:
        print(
            f"[download] 跳过超长视频 {ref.video_id} "
            f"({duration:.0f}s > 上限 {cfg.max_duration_s}s)，未下载"
        )
        return None

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(ref.url, download=True)
    except Exception as e:  # noqa: BLE001 — 批处理里单个失败不应中断全局
        print(f"[download] 失败 {ref.url}: {e}")
        _cleanup(out_dir, ref)  # 清掉可能残留的 .part / 半成品，避免堆积占盘
        return None

    if not info:
        _cleanup(out_dir, ref)
        return None

    # 兜底：万一发现阶段没给时长、探测也失败（duration 仍为 None），
    # 这里用下载后拿到的准确 info 再校验一次时长。
    duration = info.get("duration") or duration or 0
    if not have_local and cfg.max_duration_s and duration > cfg.max_duration_s:
        print(f"[download] 跳过超长视频 {ref.video_id} ({duration:.0f}s)")
        _cleanup(out_dir, ref)
        return None

    # 找到实际落盘的文件。
    produced = list(out_dir.glob(f"{ref.platform}_{ref.video_id}.*"))
    video_files = [p for p in produced if p.suffix.lower() in {".mp4", ".mkv", ".webm"}]
    if not video_files:
        _cleanup(out_dir, ref)
        return None

    return DownloadedVideo(
        ref=ref,
        path=video_files[0],
        fps=cfg.fps,
        height=min(cfg.max_height, info.get("height") or cfg.max_height),
        duration_s=float(duration),
    )


def _probe_duration(url: str, base_opts: dict[str, Any]) -> float | None:
    """只解析元数据、不下载，拿到视频时长（秒）；失败返回 None。

    用于下载前的时长粗筛：避免把超长视频整段下到本地后才发现超限。
    只透传必要选项，保持探测轻量。
    """
    import yt_dlp

    opts: dict[str, Any] = {"quiet": True, "skip_download": True, "ignoreerrors": True}
    if base_opts.get("remote_components"):
        opts["remote_components"] = base_opts["remote_components"]
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:  # noqa: BLE001 — 探测失败就当未知时长，交给下载后兜底
        return None
    if not info:
        return None
    dur = info.get("duration")
    return float(dur) if dur else None


def _cleanup(out_dir: Path, ref: VideoRef) -> None:
    """删掉某视频在 out_dir 下的所有残留文件（含 .part / 分流的 .webm 等）。

    下载失败或判废时调用，防止半成品堆积、占满磁盘。
    """
    for p in out_dir.glob(f"{ref.platform}_{ref.video_id}.*"):
        p.unlink(missing_ok=True)


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
