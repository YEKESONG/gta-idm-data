# gta-idm-data

GTA gameplay 视频**爬取 + 自动过滤** pipeline，为 ActSWM 的 IDM / world model
准备"纯移动 + 开火"的视频数据（剔除购买装备、死亡画面、菜单、读条等无效片段）。

## 流水线

```
discover ──► download ──► segment ──► filter ──► clean clips + manifest.jsonl
 (找候选)     (yt-dlp)     (切片)      (过滤)
```

四个阶段都是独立模块（`gta_pipeline/` 下），可单独调用，也可由
`gta_pipeline.pipeline.run()` 串成端到端。

## 安装

```bash
pip install -r requirements.txt
# 还需系统级 ffmpeg（下载重采样 + 片段裁剪）：
sudo apt install ffmpeg          # Debian/Ubuntu
```

## 快速开始

```bash
# 1) 先只看发现阶段会抓到哪些视频（不下载，便于检查关键词）
python -m scripts.run_pipeline --config configs/default.yaml --discover-only

# 2) 跑完整流水线
python -m scripts.run_pipeline --config configs/default.yaml
```

产物：
- `data/raw/`    下载的原始视频（`keep_raw: false` 时处理完即删）
- `data/clean/`  通过过滤的干净片段（mp4）
- `data/manifest.jsonl`  每个保留片段的元数据（来源、时间区间、过滤统计）
- `data/processed.txt`  已处理视频清单（断点续跑用）

## 批量自动抓取

`configs/batch.yaml` 是搜索驱动的批量配置：填关键词 + 平台，自动发现→下载→切片→过滤。

```bash
python -m scripts.run_pipeline --config configs/batch.yaml --discover-only  # 先看会搜到啥
python -m scripts.run_pipeline --config configs/batch.yaml                  # 正式批量跑
```

批量特性（均在 `pipeline.run()` 中）：
- **流式处理**：逐个视频「下载→处理」，磁盘占用只与单个视频相关，不会一次性下满。
- **断点续跑**：随时 `Ctrl+C`；重跑时按 `processed.txt` 跳过已处理视频，manifest 追加不重复。
- **按总时长收集**：`target_hours` 累计满 N 小时源视频后自动停止（跨重跑累加，记录在 `processed.txt`）。`0` = 不限。
- **省磁盘**：`keep_raw: false` 时处理完即删原始片（干净片段已存 `clean/`）。
- **抓取礼貌性**：`download.sleep_interval_s` 控制视频间随机等待，降低被限流风险。
- **容错**：单个视频报错只跳过该视频，不中断整批。

### 大规模收集（如仅 YouTube 凑 ~300 小时）

`configs/youtube_300h.yaml`：仅 YouTube、`target_hours: 300`、`keep_raw: false`，并用
`require_title_keywords` 强制候选标题必须含 `GTA V`/`GTA5`。
```bash
python -m scripts.run_pipeline --config configs/youtube_300h.yaml --discover-only  # 先验候选
python -m scripts.run_pipeline --config configs/youtube_300h.yaml                  # 跑到满 300h 自动停
```

> `discovery.require_title_keywords`：非空时只保留标题（不区分大小写）含其中任一关键词的候选；标题缺失的会被排除。

## 过滤策略

启发式规则 + 可插拔分类器，逐帧打分后由 `policy.py` 汇总成整片保留/丢弃：

| 检测器 | 丢弃目标 | 原理 |
|--------|----------|------|
| `death_screen` | WASTED/BUSTED/任务结算 | 全屏去饱和 + 中央大字（可选 OCR / 模板） |
| `menu_shop`    | 菜单/商店/暂停/地图 | 平坦区域占比高（缺少自然纹理） |
| `loading_black`| 黑屏/读条/转场 | 亮度极低或近纯色 |
| `motion`       | 长时间站立不动 | Farneback 光流幅度低 |

各阈值见 `configs/default.yaml`。

### 提升精度的两个钩子

1. **模板匹配**：把死亡画面截图放 `assets/templates/death/`、菜单/商店截图放
   `assets/templates/menu/`（`.png`/`.jpg`），对应检测器会自动加载并启用。
2. **ML 分类器兜底**：训练一个 ResNet18 帧分类器（类别建议
   `[gameplay, death, menu, loading]`），把权重路径填到 `filter.classifier_ckpt`，
   并在 `enabled_detectors` 里加上 `classifier`。接口见
   `gta_pipeline/filters/classifier.py` 的 `FrameClassifier` 协议。

## 扩展

- 新增启发式检测器：在 `gta_pipeline/filters/` 新建文件，继承 `FrameDetector`、
  用 `@register` 注册，再在配置 `enabled_detectors` 写上其 `name`。
- Twitch 没有公开搜索接口：把频道/VOD 列表 URL 填进 `discovery.seed_urls`。

## 合规提示

仅用于学术研究的数据准备。爬取前请确认目标平台的服务条款与版权要求，
控制抓取频率，避免对站点造成压力。
