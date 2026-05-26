# gta-idm-data

GTA V gameplay 视频的**自动采集 + 清洗** pipeline：从 YouTube / Bilibili 抓取实况录像，
自动切片并剔除「死亡画面、菜单商店、读条黑屏、长时间站立不动」等无效片段，最终产出一批
**「持续移动 + 开火」的干净短片**，为 ActSWM 的 IDM（逆动力学模型）/ world model 训练备料。

一条命令即可从「一堆关键词 / 合集 URL」滚动产出几百小时清洗后的训练数据，支持断点续跑、
按有效时长自动收口、多平台优先级、三级并行流水线。

---

## 流水线总览

```
discover ──► download ──► segment ──► filter ──► clean/*.mp4 + manifest.jsonl
 找候选       yt-dlp       切成片段     逐帧过滤     干净片段 + 元数据清单
(只取元数据)  (≤480p)     (只产区间)   (4 检测器)   (通过过滤才物理裁剪)
```

四个阶段都是 `gta_pipeline/` 下的独立模块，可单独调用调试，也由
[`gta_pipeline/pipeline.py`](gta_pipeline/pipeline.py) 的 `run()` 串成端到端、并行执行。

---

## 安装

```bash
# 推荐用项目自带的 conda 环境（含 yt-dlp / opencv / scenedetect）
conda activate gta_idm_data
pip install -r requirements.txt

# 还需系统级 ffmpeg（下载合流 + 片段裁剪都依赖它）
sudo apt install ffmpeg          # Debian/Ubuntu
```

> ⚠️ 跑 pipeline 务必用 `gta_idm_data` 环境，默认的 `ml` 环境没有 `yt_dlp`。

---

## 快速开始

主力配置是 [`configs/youtube_300h.yaml`](configs/youtube_300h.yaml)（B 站合集优先 + YouTube 关键词搜索，凑满 250h 有效时长）：

```bash
# 1) 先「干跑」发现阶段：只列候选、不下载，确认数量和排序对不对（几十秒）
python -m scripts.run_pipeline --config configs/youtube_300h.yaml --discover-only

# 2) 正式跑：随时可 Ctrl+C，重跑自动断点续跑、累计有效时长跨重跑累加
python -m scripts.run_pipeline --config configs/youtube_300h.yaml
```

其它现成配置：
- `configs/test_single.yaml` — 只处理 1 个指定视频，第一遍跑通用。
- `configs/batch.yaml` — 关键词驱动的通用批量模板。
- `configs/default.yaml` — 全套默认值。

命令行开关：

| 开关 | 作用 |
|------|------|
| `--config <yaml>` | 指定配置文件（不给则用全套默认值） |
| `--discover-only` | 只跑发现阶段并打印候选（含平台分布 + 优先级排序），不下载/不过滤 |
| `--refresh-discovery` | 忽略候选缓存，强制重新搜索（一般不需要，见下文「增量种子」） |

---

## 产物

| 路径 | 内容 |
|------|------|
| `data/clean/*.mp4` | 通过过滤的干净片段（固定帧率、无音轨） |
| `data/manifest.jsonl` | 每个保留片段的元数据：来源、时间区间、帧率、过滤统计、丢弃标签计数 |
| `data/processed.txt` | 已处理完成的视频清单（断点续跑跳过用，每行 `平台_id\t源时长`） |
| `data/discovery_cache.jsonl` | 候选列表缓存（重跑直接复用，零网络请求） |
| `data/discovery_cache.jsonl.seeds` | 已展开过的种子 URL 记录（增量采集用） |
| `data/raw/` | 下载的原始视频（`keep_raw: false` 时处理完即删，省磁盘） |

`manifest.jsonl` 每行一个 JSON，训练时可被 PyTorch `Dataset` 直接按行读、按时间区间取帧。

---

## 配置要点（以 `youtube_300h.yaml` 为例）

```yaml
work_dir: data
target_hours: 250          # 累计「有效（干净片段）时长」满 250h 即停（不是源视频时长！）
keep_raw: false            # 处理完即删原始片，省磁盘
platform_priority: [bilibili, youtube]   # B 站合集先采，再 YouTube
num_workers: 0             # 过滤进程数，0=自动（按 CPU 核数留余量，上限 12）

discovery:
  platforms: [youtube]     # 只对 YouTube 做关键词搜索（B 站搜索质量差，只用 seed_urls）
  queries: [ "GTA V gameplay no commentary", ... ]
  max_results_per_query: 100
  require_title_keywords: ["GTA V", "GTA 5"]   # 标题白名单，只对搜索结果生效
  seed_urls:                                    # 精选 URL，不受标题过滤，全部保留
    - "https://www.bilibili.com/video/BV1indnBREvD"   # B 站多 P 合集会展开成全部分 P
    - "https://www.youtube.com/playlist?list=..."

download:
  max_height: 480          # 480p 对 world model 足够，省带宽/磁盘
  fps: 20                  # 最终干净片段统一帧率（保证按帧对齐动作标签）
  max_duration_s: 10800    # 单视频时长上限 3h，超过的在下载前跳过（防超长视频拖死流水线）

segment:
  mode: scene              # 按镜头切（scene）或固定窗口（window）
  scene_threshold: 27.0    # PySceneDetect 内容检测阈值
  min_clip_seconds: 2.0    # 太短的片段丢弃
  max_clip_seconds: 30.0   # 太长的片段再切
  sample_fps: 2.0          # 过滤时的抽帧采样率（每秒采 2 帧做检测）

filter:
  enabled_detectors: [death_screen, menu_shop, loading_black, motion]
  bad_frame_ratio_to_drop: 0.25      # 坏帧占比超过 25% 丢整片
  static_frame_ratio_to_drop: 0.6    # 静止帧占比超过 60% 丢整片
  min_motion_magnitude: 0.6          # 光流幅度低于此视为「静止」
```

---

## 设计思路

### 一、如何筛选（发现候选 + 内容过滤）

筛选分两层：**发现阶段先按廉价信号粗筛**（标题、平台、时长），**过滤阶段再按画面内容精筛**。

**发现层**（[`discovery.py`](gta_pipeline/discovery.py)）：
- **来源**：① 关键词搜索（YouTube 用 `ytsearch`、B 站用 `bilisearch`，统一走 yt-dlp 的 flat 模式，只取元数据不下载）；② `seed_urls` 精选频道/播放列表/合集 URL，yt-dlp 自动展开成全部视频。
- **标题白名单**：`require_title_keywords` 只对「关键词搜索」结果做子串过滤（兜底搜索质量），`seed_urls` 视为可信来源不过滤。
- **平台优先级**：`platform_priority` 让某平台候选排到队首先处理（如 B 站精选合集优先于 YouTube 搜索）。
- **缓存 + 增量种子**：候选写入 `discovery_cache.jsonl`，重跑零网络请求复用；新加的 `seed_urls` 会被**增量展开**并入缓存（用 `.seeds` sidecar 记录已展开的，不重复、也不重搜关键词）——所以随时往配置贴新 URL，直接重跑即可采集。
- **B 站多 P 视频的坑**：同一个 BV 号的几十个分 P 共用相同 id、仅靠 `?p=N` 区分。代码从 URL 解析「BV 号 + 分 P 号」作唯一 `video_id`，避免被去重塌缩成一集、只下到第一集。

**过滤层**（[`filters/`](gta_pipeline/filters/) + [`policy.py`](gta_pipeline/policy.py)）：采用「逐帧检测器打分 → 整片决策」两步，职责分离便于复用阈值、也讲得清每片为何被丢。

| 检测器 | 丢弃目标 | 原理 |
|--------|----------|------|
| `death_screen` | WASTED/BUSTED/任务结算 | 全屏去饱和 + 中央大字（可选 OCR / 模板匹配） |
| `menu_shop`    | 菜单/商店/暂停/地图 | 平坦区域占比高（缺少自然纹理） |
| `loading_black`| 黑屏/读条/转场 | 亮度极低或近纯色 |
| `motion`       | 长时间站立不动 | Farneback 光流幅度低于阈值视为静止 |

每个采样帧过一遍所有检测器，[`decide_clip`](gta_pipeline/policy.py) 汇总成整片决策：
- 任一检测器把某帧判为 `death/menu/loading/non_gameplay` → 该帧记为**坏帧**；
- `motion` 判为 `static` → 记为**静止帧**；
- **坏帧占比 > 25%** 或 **静止帧占比 > 60%** → 丢弃整片，否则保留。

> 注意：过滤只在 `sample_fps`（默认 2 帧/秒）的采样帧上做，既快又足以判断片段属于哪类内容。

### 二、如何切片裁剪（segment + export）

关键设计：**切片分两步，先产「时间区间」、过滤通过后才物理裁剪**，避免给最终要丢弃的片段白白写盘。

1. **找镜头边界**（[`segment.py`](gta_pipeline/segment.py)）：`scene` 模式用 PySceneDetect 的 `ContentDetector` 按画面内容突变切镜头——GTA 实况里「进商店 / 死亡 / 读条」通常伴随明显画面切换，按镜头切能让这些坏段落自然落在独立片段里、方便整片丢弃。也支持 `window` 固定时长窗口模式。
2. **时长约束**：丢弃 < `min_clip_seconds` 的碎片，把 > `max_clip_seconds` 的长片均匀再切。这一步**只产出 `(start_s, end_s)` 区间元数据，不碰视频文件**。
3. **物理裁剪**（`export_clip`，仅对过滤通过的片段）：用 ffmpeg `-ss`（放在 `-i` 前做快速 seek）+ `-t`（时长）裁出独立 mp4，`-r` 固定帧率（保证按帧对齐动作标签）、`-an` 去音轨（动作建模用不到）。裁剪带 **300s 超时**，个别片段卡住也只是跳过、不会阻塞。

### 三、如何并行（三级流水线）

三种资源（网络 / 单核场景检测 / 多核过滤）同时干活、互不空等：

```
下载线程 ×N (网络IO) ─dl_q→ 切片线程 ×1 (单核场景检测) ─seg_q→ 过滤进程池 ×M (多核)
   download_concurrency        PySceneDetect              num_workers
```

- **重叠执行**：「视频 N+1 的下载 / 切片」与「视频 N 的过滤」并行进行，消除了原来「切片时进程池空转、过滤时主线程干等」的浪费。
- **有界队列背压**：`dl_q` / `seg_q` 容量受 `download_prefetch` / `segment_prefetch` 限制，下载快于处理时自动阻塞，把磁盘占用限制在少数几个在途视频，适合大批量抓取。
- **避免超额订阅**：过滤是进程级多核并行，所以把 OpenCV 自身的多线程关掉（`cv2.setNumThreads(1)`），否则「进程数 × 每进程线程数」会抢核反而更慢；场景检测刚好稳定占约 1 个核。
- **进程池用 `spawn` 而非 `fork`**（关键）：主进程已经起了下载/切片线程，此时 `fork` 出的 worker 会继承「别的线程正持有、却永不释放」的底层锁（malloc/OpenCV/ffmpeg 全局锁），导致 worker 全部死锁、整条流水线挂起。`spawn` 启动全新解释器、不继承父进程线程与锁状态，从根上消除这个死锁。
- **串行写**：`manifest.jsonl` / `processed.txt` 只在主线程写，避免多进程并发写冲突。

### 四、稳健性与收口

- **断点续跑**：每处理完一个视频追加进 `processed.txt`，重跑时按它跳过已完成视频（不重复选中、不重复下载）。已下好的本地文件 yt-dlp 会跳过下载直接复用。
- **按有效时长收口**：`target_hours` 统计的是**过滤后保留下来的干净片段时长**（不是源视频时长——源经切分+过滤后产出率只约 40%）。续跑基线从已有 `manifest.jsonl` 累加，累计达标即停。
- **超长视频防护**：`max_duration_s` 在下载前按时长粗筛跳过十几小时的录像（它们会切出几千片段、极易拖垮流水线）；但**本地已下好的完整文件豁免该上限并优先处理**，不浪费已花的下载。
- **容错**：单个视频下载/切片/过滤报错只跳过该视频，不中断整批。
- **抓取礼貌性**：发现与下载阶段都有随机间隔（`sleep_interval_s` / 内置退避），降低被站点限流（429）的风险。

---

## 提升过滤精度的两个钩子

1. **模板匹配**：把死亡画面截图放 `assets/templates/death/`、菜单/商店截图放
   `assets/templates/menu/`（`.png`/`.jpg`），对应检测器会自动加载并启用。
2. **ML 分类器兜底**：训练一个帧分类器（类别建议 `[gameplay, death, menu, loading]`），
   把权重路径填到 `filter.classifier_ckpt`，并在 `enabled_detectors` 里加上 `classifier`。
   接口见 [`gta_pipeline/filters/classifier.py`](gta_pipeline/filters/classifier.py)。

---

## 扩展

- **新增检测器**：在 `gta_pipeline/filters/` 新建文件，继承 `FrameDetector`、用 `@register`
  注册，再在配置 `enabled_detectors` 写上其 `name` 即可（构造参数由注册表按需透传）。
- **新增平台 / 来源**：关键词搜索见 `discovery.SEARCH_PREFIX`；没有公开搜索接口的平台
  （如 Twitch）把频道/VOD URL 填进 `discovery.seed_urls`。

---

## 合规提示

仅用于学术研究的数据准备。爬取前请确认目标平台的服务条款与版权要求，控制抓取频率，
避免对站点造成压力。
