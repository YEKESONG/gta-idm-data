"""GTA gameplay 视频爬取 + 自动过滤 pipeline（用于 ActSWM 的 IDM 数据准备）。

整条流水线分四个阶段：
    discovery  ->  download  ->  segment  ->  filter  ->  clean clips + manifest

每个阶段都是独立模块，可单独调用，也可由 pipeline.run() 串起来。
"""

from .config import PipelineConfig

__all__ = ["PipelineConfig"]
__version__ = "0.1.0"
