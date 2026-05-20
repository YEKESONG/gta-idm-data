"""命令行入口。

示例：
    # 用配置文件跑完整流水线
    python -m scripts.run_pipeline --config configs/default.yaml

    # 不给配置则用全套默认值（仅 YouTube 搜索）
    python -m scripts.run_pipeline

    # 只跑发现阶段，先看看会抓到哪些视频（不下载）
    python -m scripts.run_pipeline --config configs/default.yaml --discover-only
"""

from __future__ import annotations

import argparse

from gta_pipeline import PipelineConfig
from gta_pipeline.discovery import discover
from gta_pipeline.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="GTA gameplay 数据爬取 + 自动过滤 pipeline")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="只跑发现阶段并打印候选视频，不下载/不过滤",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()

    if args.discover_only:
        refs = discover(
            platforms=cfg.discovery.platforms,
            queries=cfg.discovery.queries,
            seed_urls=cfg.discovery.seed_urls,
            max_results_per_query=cfg.discovery.max_results_per_query,
        )
        print(f"发现 {len(refs)} 个候选：")
        for r in refs:
            dur = f"{r.duration_s:.0f}s" if r.duration_s else "?"
            print(f"  [{r.platform}] {dur:>6}  {r.title[:60]}  {r.url}")
        return

    run(cfg)


if __name__ == "__main__":
    main()
