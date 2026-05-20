"""集中式配置。

整条 pipeline 的所有可调参数都收敛到这里，用 dataclass 描述，
既能在代码里直接 `PipelineConfig()` 拿到一份合理默认值，
也能从 YAML 文件加载覆盖（见 configs/default.yaml）。

为什么用 dataclass 而不是裸 dict：
- 字段有默认值、有类型提示，IDE 能补全，改参数时不容易拼错 key；
- 嵌套结构清晰，每个阶段（下载/切片/过滤）的参数各归各的。
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class DiscoveryConfig:
    """第 1 阶段：发现候选视频。"""

    # 每个平台的搜索关键词。GTA V 的纯 gameplay 关键词最好带上
    # "no commentary"（无解说，画面更干净）这类限定词。
    queries: list[str] = field(
        default_factory=lambda: [
            "GTA V gameplay no commentary",
            "GTA 5 free roam shooting",
        ]
    )
    # 想抓取的平台。值要和 discovery.PLATFORMS 里注册的名字对应。
    platforms: list[str] = field(default_factory=lambda: ["youtube"])
    # 每个 (平台, 关键词) 组合最多取多少个候选视频。
    max_results_per_query: int = 20
    # 直接给定的视频/频道/播放列表 URL（绕过搜索，适合 Bilibili/Twitch）。
    seed_urls: list[str] = field(default_factory=list)


@dataclass
class DownloadConfig:
    """第 2 阶段：下载。"""

    # 目标分辨率上限。480p 对 world model 训练足够（通常还会再 resize 到 ≤256），
    # 且大幅省带宽/磁盘。需要更高清细节（如 HUD 小字）再调回 720。
    max_height: int = 480
    # 目标输出帧率：在切片导出（export_clip 的 -r）时应用，保证最终干净片段为该帧率。
    # 原始下载文件保留源帧率（不重采样），它只是中间产物。
    fps: int = 20
    # 单个视频时长上限（秒），过滤掉超长合集。0 表示不限制。
    max_duration_s: int = 1800
    # YouTube 的 n-challenge 现在需要 EJS solver 脚本（从远程拉取）才能拿到全部清晰度。
    # 默认开启 ejs:github（yt-dlp 官方推荐）。注意：这会下载并执行远程解算脚本，
    # 不想用远程组件就设为 []（但 YouTube 下载可能只剩低清晰度甚至失败）。
    remote_components: list[str] = field(default_factory=lambda: ["ejs:github"])
    # yt-dlp 额外参数透传（高级用法）。
    extra_ydl_opts: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentConfig:
    """第 3 阶段：切片 + 抽帧。"""

    # 切片方式："scene"=按镜头检测切，"window"=固定时长窗口切。
    mode: str = "scene"
    scene_threshold: float = 27.0  # PySceneDetect content detector 阈值
    window_seconds: float = 10.0  # window 模式下每片时长
    min_clip_seconds: float = 2.0  # 太短的片丢弃
    max_clip_seconds: float = 30.0  # 太长的片再切
    # 过滤判定时的抽帧采样率（每秒采几帧做检测，不必逐帧）。
    sample_fps: float = 2.0


@dataclass
class FilterConfig:
    """第 4 阶段：过滤。"""

    # 每个检测器的开关与参数。键名对应 filters/ 下注册的检测器名字。
    enabled_detectors: list[str] = field(
        default_factory=lambda: ["death_screen", "menu_shop", "loading_black", "motion"]
    )
    # 一个片段中"坏帧"占比超过该阈值就丢弃整片。
    bad_frame_ratio_to_drop: float = 0.25
    # 运动检测：光流幅度低于该阈值视为"静止"（站着不动，价值低）。
    min_motion_magnitude: float = 0.6
    # 静止帧占比超过该阈值也丢弃（我们要的是移动/开火数据）。
    static_frame_ratio_to_drop: float = 0.6
    # 可选 ML 分类器权重路径；为空则只用启发式规则。
    classifier_ckpt: str | None = None
    # 分类器把某帧判为非 gameplay 的置信度阈值。
    classifier_threshold: float = 0.5


@dataclass
class PipelineConfig:
    work_dir: str = "data"  # 所有产物的根目录
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)

    # ---- 派生路径：统一在这里定义，避免各模块各拼各的 ----
    @property
    def raw_dir(self) -> Path:
        return Path(self.work_dir) / "raw"  # 下载的原始视频

    @property
    def clips_dir(self) -> Path:
        return Path(self.work_dir) / "clips"  # 切片后、过滤前

    @property
    def clean_dir(self) -> Path:
        return Path(self.work_dir) / "clean"  # 通过过滤的干净片段

    @property
    def manifest_path(self) -> Path:
        return Path(self.work_dir) / "manifest.jsonl"  # 元数据清单

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """从 YAML 加载，未指定的字段沿用 dataclass 默认值。"""
        import yaml  # 延迟导入：不写 YAML 的人无需装 PyYAML

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return _from_dict(cls, raw)


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """递归地把嵌套 dict 还原成（嵌套的）dataclass 实例。"""
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    # 注意：开了 `from __future__ import annotations` 后，f.type 是字符串，
    # 需用 get_type_hints 解析回真正的类型，否则识别不出嵌套 dataclass。
    import typing

    type_by_name = typing.get_type_hints(cls)
    for key, value in data.items():
        if key not in type_by_name:
            continue  # 忽略 YAML 里多余的键，保持向前兼容
        ftype = type_by_name[key]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _from_dict(ftype, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)
