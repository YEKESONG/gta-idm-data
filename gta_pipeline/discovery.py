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

from dataclasses import dataclass, asdict
from typing import Any

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
    """用 yt-dlp 在 flat 模式下解析一个搜索表达式或 URL，返回条目列表。"""
    import yt_dlp  # 延迟导入：仅在真正发现时才需要

    opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",  # 只展开列表，不深挖每个视频
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(query_or_url, download=False)
    if not info:
        return []
    # 搜索/播放列表结果在 "entries" 里；单视频则 info 本身就是条目。
    entries = info.get("entries") if isinstance(info, dict) else None
    return list(entries) if entries else [info]


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
) -> list[VideoRef]:
    """返回去重后的候选视频列表。

    Args:
        platforms: 要搜索的平台名（见 SEARCH_PREFIX）。
        queries: 搜索关键词。
        seed_urls: 直接给定的 URL（频道/播放列表/单视频），不依赖搜索。
        max_results_per_query: 每个 (平台, 关键词) 取多少条。
    """
    refs: list[VideoRef] = []

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
                    refs.append(ref)

    # --- 种子 URL 展开 ---
    for url in seed_urls:
        platform = _guess_platform(url)
        for entry in _ydl_flat(url):
            ref = _entry_to_ref(entry, platform)
            if ref:
                refs.append(ref)

    return _dedup(refs)


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
