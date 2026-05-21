"""第 1 阶段：发现候选视频。

目标：把"关键词 / 种子 URL"变成一批 VideoRef（视频引用），
但**不下载**视频本身——只取元数据（URL、时长、标题等），
这样可以先按时长等廉价信号粗筛，再决定下载哪些，省带宽。

实现思路：
- 统一用 yt-dlp 的 extract_flat 模式（只解析列表、不下完整信息），
  它原生支持 ytsearch:（YouTube）和 bilisearch:（Bilibili）搜索前缀；
- Twitch 没有公开搜索接口，因此走 seed_urls：给定频道/VOD 列表 URL，
  yt-dlp 会把它展开成一个个 VOD。
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# --- 限流防护参数 ---
# 发现阶段每个网络请求（搜索/展开）之间的随机间隔区间（秒）。随机化让请求节奏不机械，
# 比固定间隔更不易被判定为 bot。旧版零间隔会被限流、候选骤减到几十个。
_DISCOVER_SLEEP_MIN_S = 2.0
_DISCOVER_SLEEP_MAX_S = 4.0
# 万一仍撞上 429（限流），指数退避重试而非静默丢弃：首次等待秒数 + 最大重试次数。
_DISCOVER_BACKOFF_S = 5.0
_DISCOVER_MAX_RETRIES = 5


def _discover_sleep() -> None:
    """两次发现请求之间的随机等待，降低被限流的概率。"""
    time.sleep(random.uniform(_DISCOVER_SLEEP_MIN_S, _DISCOVER_SLEEP_MAX_S))

# 各平台 -> yt-dlp 搜索前缀。None 表示该平台不支持关键词搜索，
# 只能通过 seed_urls（频道/播放列表 URL）发现视频。
SEARCH_PREFIX: dict[str, str | None] = {
    "youtube": "ytsearch",
    "bilibili": "bilisearch",
    "twitch": None,
}


@dataclass
class VideoRef:
    """一个候选视频的轻量引用（下载前）。"""

    platform: str
    video_id: str
    url: str
    title: str = ""
    duration_s: float | None = None  # 某些平台 flat 模式拿不到，可能为 None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ydl_flat(query_or_url: str) -> list[dict[str, Any]]:
    """用 yt-dlp 在 flat 模式下解析一个搜索表达式或 URL，返回条目列表。

    遇到 429（限流）会指数退避重试，而不是静默返回空——否则那一批候选会无声消失，
    正是上次"假装跑完"的根因。非限流错误则按原样跳过（返回空列表）。
    """
    import yt_dlp  # 延迟导入：仅在真正发现时才需要

    opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",  # 只展开列表，不深挖每个视频
        # 关掉 ignoreerrors，让顶层 429 抛成异常好被下面捕获重试。
        # flat 模式只解析列表、不深挖单条视频，单条失败一般不会触发整体异常。
        "ignoreerrors": False,
    }
    delay = _DISCOVER_BACKOFF_S
    for attempt in range(1, _DISCOVER_MAX_RETRIES + 1):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(query_or_url, download=False)
        except Exception as e:  # noqa: BLE001 — 区分限流(可重试)与其它错误(跳过)
            msg = str(e)
            is_rate_limited = "429" in msg or "Too Many Requests" in msg
            if is_rate_limited and attempt < _DISCOVER_MAX_RETRIES:
                print(
                    f"[discover] 触发限流(429)，退避 {delay:.0f}s 后重试 "
                    f"({attempt}/{_DISCOVER_MAX_RETRIES})：{query_or_url[:50]}"
                )
                time.sleep(delay)
                delay *= 2  # 指数退避
                continue
            print(f"[discover] 解析失败，跳过：{query_or_url[:50]}（{e}）")
            return []
        if not info:
            return []
        # 搜索/播放列表结果在 "entries" 里；单视频则 info 本身就是条目。
        entries = info.get("entries") if isinstance(info, dict) else None
        return list(entries) if entries else [info]
    return []


def _entry_to_ref(entry: dict[str, Any], platform: str) -> VideoRef | None:
    if not entry:
        return None
    url = entry.get("url") or entry.get("webpage_url")
    vid = entry.get("id")
    if not url or not vid:
        return None
    # flat 模式给的 url 有时是相对 id，补全成可下载的完整 URL。
    if platform == "youtube" and not str(url).startswith("http"):
        url = f"https://www.youtube.com/watch?v={url}"
    return VideoRef(
        platform=platform,
        video_id=str(vid),
        url=str(url),
        title=entry.get("title") or "",
        duration_s=entry.get("duration"),
    )


def discover(
    platforms: list[str],
    queries: list[str],
    seed_urls: list[str],
    max_results_per_query: int,
    require_title_keywords: list[str] | None = None,
    cache_path: Path | None = None,
    refresh: bool = False,
) -> list[VideoRef]:
    """返回去重后的候选视频列表。

    Args:
        platforms: 要搜索的平台名（见 SEARCH_PREFIX）。
        queries: 搜索关键词。
        seed_urls: 直接给定的 URL（频道/播放列表/单视频），不依赖搜索。
        max_results_per_query: 每个 (平台, 关键词) 取多少条。
        require_title_keywords: 若非空，仅对「关键词搜索」结果按标题过滤（保留标题
            含任一关键词的，不区分大小写）；seed_urls 视为可信来源，不受此过滤影响。
        cache_path: 候选列表缓存文件。命中且非空时直接复用、不发任何网络请求
            （重跑/断点续跑避免重复触发限流）。为 None 则不读写缓存。
        refresh: True 时忽略缓存、强制重新搜索（并覆盖写回缓存）。
    """
    # 缓存命中：直接复用，一个网络请求都不发——这是断点续跑时最有效的限流防护。
    if cache_path is not None and not refresh:
        cached = _load_cache(cache_path)
        if cached:
            print(
                f"[discover] 复用候选缓存 {len(cached)} 个：{cache_path} "
                f"（要重新搜索就删掉它，或加 --refresh-discovery）"
            )
            return cached

    # 搜索结果与 seed_urls 结果分开收集：标题白名单只用于「关键词搜索」结果
    # （给搜索兜底质量）；seed_urls 是人工精选的可信来源，全部保留、不做标题过滤。
    search_refs: list[VideoRef] = []
    seed_refs: list[VideoRef] = []

    # --- 关键词搜索 ---
    for platform in platforms:
        prefix = SEARCH_PREFIX.get(platform)
        if prefix is None:
            # 该平台不支持搜索（如 Twitch），跳过，靠 seed_urls 补。
            continue
        for q in queries:
            expr = f"{prefix}{max_results_per_query}:{q}"
            for entry in _ydl_flat(expr):
                ref = _entry_to_ref(entry, platform)
                if ref:
                    search_refs.append(ref)
            _discover_sleep()  # 限流保护：每个搜索请求后随机稍歇

    # --- 种子 URL 展开 ---
    for url in seed_urls:
        platform = _guess_platform(url)
        for entry in _ydl_flat(url):
            ref = _entry_to_ref(entry, platform)
            if ref:
                seed_refs.append(ref)
        _discover_sleep()  # 限流保护：每个展开请求后随机稍歇

    # 标题白名单只过滤搜索结果；seed_urls 直接全保留。
    search_refs = _filter_by_title(search_refs, require_title_keywords)
    result = _dedup(search_refs + seed_refs)

    # 写回缓存：下次重跑直接复用，不再发起整轮搜索。
    if cache_path is not None and result:
        _save_cache(cache_path, result)
        print(f"[discover] 候选已缓存到 {cache_path}（{len(result)} 个），重跑将直接复用。")
    return result


def _save_cache(path: Path, refs: list[VideoRef]) -> None:
    """把候选列表写成 jsonl 缓存（每行一个 VideoRef）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in refs:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def _load_cache(path: Path) -> list[VideoRef]:
    """读 jsonl 候选缓存还原成 VideoRef 列表；不存在或损坏则返回空（触发重新搜索）。"""
    if not path.exists():
        return []
    out: list[VideoRef] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(VideoRef(**json.loads(line)))
        except Exception:  # noqa: BLE001 — 缓存损坏就当未命中，重新搜索
            return []
    return out


def _filter_by_title(
    refs: list[VideoRef], keywords: list[str] | None
) -> list[VideoRef]:
    """仅保留标题含任一关键词的视频（不区分大小写）。keywords 为空则不过滤。

    注意：标题缺失的条目会被排除（无法确认是否含关键词）。
    """
    if not keywords:
        return refs
    lowered = [k.lower() for k in keywords]
    out = []
    for r in refs:
        title = (r.title or "").lower()
        if any(k in title for k in lowered):
            out.append(r)
    return out


def _guess_platform(url: str) -> str:
    u = url.lower()
    if "bilibili" in u:
        return "bilibili"
    if "twitch" in u:
        return "twitch"
    return "youtube"


def _dedup(refs: list[VideoRef]) -> list[VideoRef]:
    seen: set[tuple[str, str]] = set()
    out: list[VideoRef] = []
    for r in refs:
        key = (r.platform, r.video_id)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
