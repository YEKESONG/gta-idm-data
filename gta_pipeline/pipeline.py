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

    采用**三级流水线**，三种资源同时干活、互不空等：

        下载线程(网络IO) ─dl_q→ 切片线程(1核,场景检测) ─seg_q→ 主线程过滤(进程池,多核)

    这样「视频 N+1 的下载/切片」与「视频 N 的过滤」重叠进行，消除了原来「切片时
    进程池空转、过滤时主线程干等」的浪费。队列都有界（背压），把磁盘占用限制在
    少数几个在途视频，适合大批量抓取。manifest / processed.txt 只在主线程串行写。
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

    # 断点续跑：跳过此前已处理过的视频（来自 processed.txt）。已处理视频不会被重复选中。
    processed = _load_processed(cfg.processed_log)
    todo = [r for r in refs if _key(r) not in processed]
    # 按平台优先级排序：列在前面的平台先处理（如先 B 站 seed、再 YouTube）。
    todo = _prioritize(todo, cfg.platform_priority)
    # target_hours 现在按「有效（干净片段）时长」计，而非源视频时长——源视频经切分+
    # 过滤后产出率只约 40%，按源时长收会严重不足。续跑基线从已有 manifest 的干净片段
    # 时长算起（而非 processed.txt 的源时长），新增量只统计本次实际保留下来的片段。
    done_s = _existing_clean_seconds(cfg.manifest_path)
    target_s = cfg.target_hours * 3600  # 0 = 不限
    print(
        f"[pipeline] 其中 {len(refs) - len(todo)} 个已处理过"
        f"（已有有效时长 {done_s / 3600:.1f}h），本次待处理 {len(todo)} 个"
    )
    if target_s:
        print(f"[pipeline] 目标有效（干净片段）时长 {cfg.target_hours}h")
    if not todo:
        print("[pipeline] 没有待处理视频，结束。")
        return

    # ---- 三级流水线的并行参数 ----
    concurrency = max(1, cfg.download_concurrency)
    dl_prefetch = max(1, cfg.download_prefetch)
    seg_prefetch = max(1, cfg.segment_prefetch)
    workers = _resolve_workers(cfg.num_workers)
    print(
        f"[pipeline] 并行配置：下载线程 {concurrency}（缓冲 {dl_prefetch}）→ "
        f"切片线程 1（缓冲 {seg_prefetch}）→ 过滤/导出进程 {workers}"
    )

    # 切片线程现在与过滤进程池并行跑，必须把主进程 OpenCV 的内部多线程关掉，
    # 否则切片解码会和 12 个过滤进程抢核（超额订阅），整体反而更慢。
    # 让场景检测稳定占约 1 个核，正好用上 _resolve_workers 留出的余量。
    import cv2

    cv2.setNumThreads(1)

    stop_event = threading.Event()  # 达标时通知下载/切片线程停下

    def _discard_raw(dl: DownloadedVideo | None) -> None:
        """停止时清掉「下了但来不及处理」的原始文件，省磁盘。"""
        if dl is not None and not cfg.keep_raw:
            dl.path.unlink(missing_ok=True)

    # --- 第 1 级：下载线程（生产者）。ref_q 预填全部 ref + 每线程一个结束标记 ---
    ref_q: queue.Queue = queue.Queue()
    for ref in todo:
        ref_q.put(ref)
    for _ in range(concurrency):
        ref_q.put(_DL_SENTINEL)
    dl_q: queue.Queue = queue.Queue(maxsize=dl_prefetch)  # 已下载 → 待切片（有界=背压）

    def _download_worker() -> None:
        while not stop_event.is_set():
            ref = ref_q.get()
            if ref is _DL_SENTINEL:
                break
            try:
                dl = download_one(ref, cfg.download, cfg.raw_dir)
            except Exception as e:  # noqa: BLE001 — 单个下载失败不应拖垮整批
                print(f"[pipeline] 下载 {_key(ref)} 出错：{e}")
                dl = None
            if not _put_until_stop(dl_q, (ref, dl), stop_event):
                _discard_raw(dl)  # 已停止、没投递成功：丢弃白下的文件
        dl_q.put(_DONE_SENTINEL)  # 本下载线程退出

    # --- 第 2 级：切片线程。消费 dl_q，跑场景检测，产出 (ref, dl, clips) 到 seg_q ---
    seg_q: queue.Queue = queue.Queue(maxsize=seg_prefetch)  # 已切片 → 待过滤（有界=背压）

    def _segment_worker() -> None:
        remaining_downloaders = concurrency  # 需收齐这么多个下载线程的结束标记
        while remaining_downloaders > 0:
            item = dl_q.get()
            if item is _DONE_SENTINEL:
                remaining_downloaders -= 1
                continue
            ref, dl = item
            if stop_event.is_set() or dl is None:
                _discard_raw(dl)  # 已停止 / 下载失败：不切片
                continue
            try:
                clips = segment_video(dl, cfg.segment)
            except Exception as e:  # noqa: BLE001 — 单个视频切片失败不应中断整批
                print(f"[pipeline] 切片 {_key(ref)} 出错，跳过：{e}")
                _discard_raw(dl)
                continue
            print(f"[pipeline] {dl.path.name}: 切出 {len(clips)} 个片段")
            if not _put_until_stop(seg_q, (ref, dl, clips), stop_event):
                _discard_raw(dl)  # 已停止、没投递成功：丢弃
        seg_q.put(_DONE_SENTINEL)  # 切片线程退出（仅 1 个）

    threads = [threading.Thread(target=_download_worker, daemon=True) for _ in range(concurrency)]
    threads.append(threading.Thread(target=_segment_worker, daemon=True))
    for t in threads:
        t.start()

    # --- 第 3 级：主线程过滤/导出。常驻进程池整条 pipeline 共用，检测器只构建一次。 ---
    kept_total = 0
    processed_count = 0
    seg_active = 1  # 切片线程数；收到它的 _DONE_SENTINEL 即结束
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_filter_worker,
        initargs=(cfg.filter, cfg.segment.sample_fps),
    ) as pool, ManifestWriter(cfg.manifest_path) as manifest:
        while seg_active > 0:
            item = seg_q.get()
            if item is _DONE_SENTINEL:
                seg_active -= 1
                continue
            ref, dl, clips = item
            # 已决定停止（达标）：丢弃残留的已切片视频，只等线程收尾。
            if stop_event.is_set():
                _discard_raw(dl)
                continue
            # 已达目标有效时长则停止（达成需求即止，不必跑完所有候选）。
            if target_s and done_s >= target_s:
                print(f"[pipeline] 已达目标 {cfg.target_hours}h（有效时长 {done_s / 3600:.1f}h），停止。")
                stop_event.set()
                _discard_raw(dl)
                continue
            processed_count += 1
            print(
                f"[pipeline] ({processed_count}/{len(todo)}, 有效时长 {done_s / 3600:.1f}h) "
                f"过滤 {_key(ref)} ..."
            )
            try:
                kept_clips, kept_seconds = _filter_video(dl, clips, cfg, manifest, pool, workers)
                kept_total += kept_clips
            except Exception as e:  # noqa: BLE001 — 单个视频出错不应中断整批
                print(f"[pipeline] 处理 {_key(ref)} 出错，跳过：{e}")
                continue
            # 成功处理：记入 processed（带源时长，仅作记录/去重用），按需删原始片省磁盘。
            # done_s 累加的是「本视频实际保留下来的干净片段时长」，对齐 target 的有效时长语义。
            _mark_processed(cfg.processed_log, _key(ref), dl.duration_s)
            done_s += kept_seconds
            if not cfg.keep_raw:
                dl.path.unlink(missing_ok=True)

    print(
        f"[pipeline] 完成。累计有效时长 {done_s / 3600:.1f}h，"
        f"本次新增干净片段 {kept_total} 个 -> {cfg.clean_dir}"
    )


# 下载预取队列用的两种哨兵：投喂给下载线程的"无更多任务"标记，
# 和下载线程回报主线程的"我已退出"标记。用唯一对象做哨兵，避免与真实数据混淆。
_DL_SENTINEL = object()
_DONE_SENTINEL = object()


def _put_until_stop(q: queue.Queue, item, stop_event: threading.Event) -> bool:
    """把 item 放进有界队列，队列满则阻塞重试（背压）。

    返回是否投递成功：若投递前/期间 stop_event 被置位则放弃（返回 False），
    让上游能及时清理白做的产物并退出，而不是死等下游消费。
    """
    while not stop_event.is_set():
        try:
            q.put(item, timeout=1.0)
            return True
        except queue.Full:
            continue
    return False


def _key(ref: VideoRef) -> str:
    return f"{ref.platform}_{ref.video_id}"


def _prioritize(refs: list[VideoRef], priority: list[str]) -> list[VideoRef]:
    """按平台优先级稳定排序：priority 里靠前的平台排前面，未列出的排最后。

    下载线程按列表顺序 FIFO 取，所以靠前的平台会被先下载/处理。稳定排序保证
    同一平台内部仍保持原发现顺序。priority 为空则原样返回（不改变顺序）。
    """
    if not priority:
        return refs
    rank = {p: i for i, p in enumerate(priority)}
    last = len(priority)  # 未在 priority 中列出的平台统一排到最后
    return sorted(refs, key=lambda r: rank.get(r.platform, last))


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


def _existing_clean_seconds(manifest_path: Path) -> float:
    """累加已有 manifest 里所有干净片段的时长（秒），作为续跑的「有效时长」基线。

    target_hours 按有效（干净片段）时长计，所以续跑要从已经导出的 clean 片段算起，
    而不是 processed.txt 里记的源视频时长。manifest 是已保留片段的权威记录。
    """
    if not manifest_path.exists():
        return 0.0
    import json

    total = 0.0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
            total += float(o["end_s"]) - float(o["start_s"])
        except (ValueError, KeyError):
            continue  # 跳过损坏/不完整的行，不让基线统计中断
    return total


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


def _filter_video(
    dl: DownloadedVideo,
    clips: list[Clip],
    cfg: PipelineConfig,
    manifest: ManifestWriter,
    pool: ProcessPoolExecutor,
    workers: int,
) -> tuple[int, float]:
    """过滤 + 导出单个视频的片段（切片已由切片线程预先完成）。

    复用传入的常驻进程池（不再每个视频重建池）。片段彼此独立，分给进程池并行做
    「采样 + 过滤 + 导出」；manifest 只在主进程串行写，避免多进程并发写冲突。

    返回 (保留片段数, 保留片段总时长秒)。时长用于按「有效时长」累计判停。
    """
    total = len(clips)
    if total == 0:
        return 0, 0.0

    tasks = [(clip, cfg.clean_dir) for clip in clips]
    kept = 0
    kept_seconds = 0.0
    done = 0
    for result in pool.map(_filter_export_clip, tasks, chunksize=8):
        done += 1
        if result is not None:
            clip, exported, decision = result
            manifest.write(clip, dl.ref.platform, exported, decision)
            kept += 1
            kept_seconds += clip.duration
        if done % 100 == 0 or done == total:
            print(f"[pipeline]   过滤/导出进度 {done}/{total}，已保留 {kept}（{workers} 进程并行）")
    return kept, kept_seconds


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
