"""把四个阶段串成一条端到端流水线。

    discover -> download -> (per video) segment -> (per clip) filter -> export + manifest

设计成"每个阶段函数都能单独调用"，pipeline.run() 只是把它们按顺序粘起来，
方便你在 notebook 里单独调试某一步。
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import PipelineConfig
from .discovery import VideoRef, discover
from .download import DownloadedVideo, download_one
from .filters.base import build_detectors
from .manifest import ManifestWriter
from .policy import decide_clip
from .segment import Clip, sample_frames, segment_video


def run(cfg: PipelineConfig, refresh_discovery: bool = False) -> None:
    """跑完整条流水线（批量、可断点续跑）。

    采用**流式 + 下载预取**：后台线程并行下载视频放入有界队列，主线程从队列取出
    逐个「切片 → 过滤 → 导出」。这样网络 I/O（下载）与 CPU 计算（切片/过滤）重叠进行，
    而有界队列把磁盘占用限制在「正在下的 + 缓冲的」少数几个视频，适合大批量抓取。
    """
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    # 1) 发现候选视频（只取元数据）。命中缓存则不重新搜索，避免重复触发限流。
    refs = discover(
        platforms=cfg.discovery.platforms,
        queries=cfg.discovery.queries,
        seed_urls=cfg.discovery.seed_urls,
        max_results_per_query=cfg.discovery.max_results_per_query,
        require_title_keywords=cfg.discovery.require_title_keywords,
        cache_path=cfg.discovery_cache_path,
        refresh=refresh_discovery,
    )
    print(f"[pipeline] 发现 {len(refs)} 个候选视频")

    # 断点续跑：跳过此前已处理过的视频（来自 processed.txt，含各视频时长）。
    processed = _load_processed(cfg.processed_log)
    todo = [r for r in refs if _key(r) not in processed]
    done_s = sum(processed.values())  # 此前累计已处理的源视频秒数（跨重跑累加）
    target_s = cfg.target_hours * 3600  # 0 = 不限
    print(
        f"[pipeline] 其中 {len(refs) - len(todo)} 个已处理过"
        f"（累计 {done_s / 3600:.1f}h），本次待处理 {len(todo)} 个"
    )
    if target_s:
        print(f"[pipeline] 目标累计时长 {cfg.target_hours}h")
    if not todo:
        print("[pipeline] 没有待处理视频，结束。")
        return

    # ---- 下载预取流水线：后台下载线程（生产者）+ 主线程处理（消费者）----
    concurrency = max(1, cfg.download_concurrency)
    prefetch = max(1, cfg.download_prefetch)
    workers = _resolve_workers(cfg.num_workers)
    print(
        f"[pipeline] 并行配置：下载线程 {concurrency}，预取缓冲 {prefetch} 个视频，"
        f"过滤/导出进程 {workers}"
    )

    # ref_q：待下载任务（预填全部 ref + 每个下载线程一个结束标记）。
    ref_q: queue.Queue = queue.Queue()
    for ref in todo:
        ref_q.put(ref)
    for _ in range(concurrency):
        ref_q.put(_DL_SENTINEL)
    # out_q：已下载、待处理的视频（有界 = 背压：满了下载线程自动阻塞等消费）。
    out_q: queue.Queue = queue.Queue(maxsize=prefetch)
    stop_event = threading.Event()  # 达到目标时通知下载线程停下

    def _download_worker() -> None:
        """生产者：不断从 ref_q 取任务下载，结果塞进 out_q。"""
        while not stop_event.is_set():
            ref = ref_q.get()
            if ref is _DL_SENTINEL:
                break
            try:
                dl = download_one(ref, cfg.download, cfg.raw_dir)
            except Exception as e:  # noqa: BLE001 — 单个下载失败不应拖垮整批
                print(f"[pipeline] 下载 {_key(ref)} 出错：{e}")
                dl = None
            # 队列满则阻塞重试（背压）；期间若已 stop 就放弃投递并清掉白下的文件。
            delivered = False
            while not stop_event.is_set():
                try:
                    out_q.put((ref, dl), timeout=1.0)
                    delivered = True
                    break
                except queue.Full:
                    continue
            if not delivered and dl is not None and not cfg.keep_raw:
                dl.path.unlink(missing_ok=True)  # 已停止：丢弃未处理的下载，省磁盘
        out_q.put(_DONE_SENTINEL)  # 通知主线程：本下载线程已退出

    threads = [threading.Thread(target=_download_worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()

    # 2+3+4) 主线程消费：常驻进程池整条 pipeline 共用，检测器只构建一次。
    # manifest 用追加模式且只在主线程写，续跑不丢记录、也不会多线程写冲突。
    kept_total = 0
    processed_count = 0
    active = concurrency  # 还在运行的下载线程数；收到 _DONE_SENTINEL 递减
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_filter_worker,
        initargs=(cfg.filter, cfg.segment.sample_fps),
    ) as pool, ManifestWriter(cfg.manifest_path) as manifest:
        while active > 0:
            item = out_q.get()
            if item is _DONE_SENTINEL:
                active -= 1
                continue
            ref, dl = item
            # 已决定停止（达标）：把队列里残留的已下载视频丢弃，只等下载线程收尾。
            if stop_event.is_set():
                if dl is not None and not cfg.keep_raw:
                    dl.path.unlink(missing_ok=True)
                continue
            # 已达目标累计时长则停止（达成需求即止，不必跑完所有候选）。
            if target_s and done_s >= target_s:
                print(f"[pipeline] 已达目标 {cfg.target_hours}h（累计 {done_s / 3600:.1f}h），停止。")
                stop_event.set()
                if dl is not None and not cfg.keep_raw:
                    dl.path.unlink(missing_ok=True)
                continue
            if dl is None:
                continue  # 下载失败/被跳过：不记 processed，下次可重试
            processed_count += 1
            print(
                f"[pipeline] ({processed_count}/{len(todo)}, 累计 {done_s / 3600:.1f}h) "
                f"处理 {_key(ref)} ..."
            )
            try:
                kept_total += _process_video(dl, cfg, manifest, pool, workers)
            except Exception as e:  # noqa: BLE001 — 单个视频出错不应中断整批
                print(f"[pipeline] 处理 {_key(ref)} 出错，跳过：{e}")
                continue
            # 成功处理：记入 processed（带时长），累加小时数，按需删原始片省磁盘。
            _mark_processed(cfg.processed_log, _key(ref), dl.duration_s)
            done_s += dl.duration_s
            if not cfg.keep_raw:
                dl.path.unlink(missing_ok=True)

    print(
        f"[pipeline] 完成。累计源视频 {done_s / 3600:.1f}h，"
        f"本次新增干净片段 {kept_total} 个 -> {cfg.clean_dir}"
    )


# 下载预取队列用的两种哨兵：投喂给下载线程的"无更多任务"标记，
# 和下载线程回报主线程的"我已退出"标记。用唯一对象做哨兵，避免与真实数据混淆。
_DL_SENTINEL = object()
_DONE_SENTINEL = object()


def _key(ref: VideoRef) -> str:
    return f"{ref.platform}_{ref.video_id}"


def _load_processed(path: Path) -> dict[str, float]:
    """读 processed.txt，返回 {视频key: 源时长秒}。兼容旧格式（无时长的纯 key 行）。"""
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            key, dur = line.split("\t", 1)
            out[key] = float(dur) if dur else 0.0
        else:
            out[line] = 0.0  # 旧格式：只有 key，时长按 0 计
    return out


def _mark_processed(path: Path, key: str, duration_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{key}\t{duration_s:.1f}\n")


# ---- 片段级并行：每个 worker 进程构建一次检测器，处理多个片段时复用 ----
_W_DETECTORS = None
_W_FILTER_CFG = None
_W_SAMPLE_FPS = None


def _init_filter_worker(filter_cfg, sample_fps) -> None:
    """worker 进程初始化：限制 OpenCV 内部线程并构建检测器。

    进程级并行已经在吃满 CPU 核，必须把 OpenCV 自身的多线程关掉（setNumThreads(1)），
    否则「进程数 × 每进程线程数」会超额订阅 CPU，反而更慢。
    """
    import cv2

    cv2.setNumThreads(1)
    global _W_DETECTORS, _W_FILTER_CFG, _W_SAMPLE_FPS
    _W_FILTER_CFG = filter_cfg
    _W_SAMPLE_FPS = sample_fps
    _W_DETECTORS = build_detectors(
        filter_cfg.enabled_detectors,
        min_motion_magnitude=filter_cfg.min_motion_magnitude,
        classifier_ckpt=filter_cfg.classifier_ckpt,
        classifier_threshold=filter_cfg.classifier_threshold,
    )


def _filter_export_clip(task):
    """worker：处理单个片段。通过过滤则导出，返回 (clip, exported_path, decision)；否则 None。"""
    clip, clean_dir = task
    for d in _W_DETECTORS:
        d.reset()  # 清掉上一个片段的帧间状态（如光流的 prev）
    per_frame_results = []
    for _ts, frame in sample_frames(clip, _W_SAMPLE_FPS):
        per_frame_results.append([d(frame) for d in _W_DETECTORS])
    decision = decide_clip(per_frame_results, _W_FILTER_CFG)
    if not decision.keep:
        return None
    exported = export_clip(clip, clean_dir)
    return (clip, exported, decision)


def _resolve_workers(num_workers: int) -> int:
    """并行进程数：>0 用指定值；0 则按 CPU 核数自动留余量（留 2 核给 IO/系统，上限 12）。"""
    if num_workers and num_workers > 0:
        return num_workers
    cpu = os.cpu_count() or 4
    return max(1, min(cpu - 2, 12))


def _process_video(
    dl: DownloadedVideo,
    cfg: PipelineConfig,
    manifest: ManifestWriter,
    pool: ProcessPoolExecutor,
    workers: int,
) -> int:
    """切片 + 过滤 + 导出单个视频。复用传入的常驻进程池（不再每个视频重建池）。"""
    clips = segment_video(dl, cfg.segment)
    total = len(clips)
    print(f"[pipeline] {dl.path.name}: 切出 {total} 个片段")
    if total == 0:
        return 0

    tasks = [(clip, cfg.clean_dir) for clip in clips]
    kept = 0
    done = 0
    # 片段彼此独立 -> 分给进程池并行做「采样 + 过滤 + 导出」；
    # manifest 只在主进程串行写，避免多进程并发写冲突。
    for result in pool.map(_filter_export_clip, tasks, chunksize=8):
        done += 1
        if result is not None:
            clip, exported, decision = result
            manifest.write(clip, dl.ref.platform, exported, decision)
            kept += 1
        if done % 100 == 0 or done == total:
            print(f"[pipeline]   过滤/导出进度 {done}/{total}，已保留 {kept}（{workers} 进程并行）")
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
