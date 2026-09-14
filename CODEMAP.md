# 仓库代码全景地图（CODEMAP）

> 生成方式：遍历仓库全部源码（`src/`、`scripts/`、`tests/`、`configs/`、`wuzhe/`）逐文件整理。
> 覆盖：职责 / 接口 / 依赖 / 输入输出 / 处理过程。生成日期：2026 年（基于当前工作区快照）。
> 环境与产物目录（`.pixi/`、`.worktrees/`、`artifacts/`、`results/`、`outputs/`、`checkpoints/`、`Data/`、`cache/`）不在文档范围内，仅作为输入/输出被引用。

## 目录

- 1. 数据层 src/data
- 2. 模型核心 src/model（根模块）
- 3. 损失项 src/model/loss_terms（基础与病灶类）
- 4. 损失项 src/model/loss_terms（频谱路由与感知类）
- 5. 模型子包 conditioning/noise/priors/frequency
- 6. 机制验证库 src/mechanism_validation
- 7. 训练/评估入口与核心实验脚本
- 8. 机制验证脚本（H1-H6 假设检验）
- 9. H4-v2 与 V2 冻结/生产管线脚本（上：H4-v2 交叉验证）
- 9. H4-v2 与 V2 冻结/生产管线脚本（下：V2 冻结与生产管线）
- 10. H3 固定调度与先验锚定路由实验脚本
- 11. 评估与行为分析脚本
- 12. 诊断/云协作/预训练杂项脚本
- 13. wuzhe/ 内嵌工程（对比实验与论文/专利脚本）
- 14. 测试套件 tests/（上：数据/频域/机制与管线测试）
- 14. 测试套件 tests/（下：路由/评估/特征与云协作测试）
- 15. 配置文件与根配置（上：根配置与消融计划）
- 15. 配置文件与根配置（下：生产实验配置与机制管线契约）

---

## 0. 项目定位与总体架构

### 0.1 项目是什么

**pet-ct-diffusion**（pixi workspace 名）：面向子宫内膜癌的 **CT→PET 图像合成 + 病灶小目标分割/保留** 的研究型扩散模型仓库。核心模型为 **SLMF-BBDM**：基于条件均值（conditional mean）残差的 **Brownian Bridge 扩散模型**，叠加层级频带（L2/L1 × LH/HL/HH）的 **spectral-evidence router**（频谱证据路由）、多种解剖/代谢先验（Gabor、Hotspot、Organ、Semantic）与约 25 种损失项。

当前模型准确表述（引自 docs/current_model_status.md）：
> 基于条件均值 residual Brownian bridge 的时间步条件化层级频带 spectral-evidence router（含 CT 支持底板），配合 τ 门控的病灶保留 + 归一化摄取代理（uptake proxy）监督。

### 0.2 顶层目录结构

| 目录 | 角色 | 详述章节 |
|---|---|---|
| `src/data/` | 数据层：DICOM/PNG 清单、张量缓存、split 清单、器官/语义预处理、血统(lineage)追踪 | 第 1 章 |
| `src/model/` | 模型核心：SLMF-BBDM、UNet 骨干（bbdm/wavelet）、trainer、registry、EMA、均值预测器/预训练、segmenter、config 解析 | 第 2 章 |
| `src/model/loss_terms/` | ~25 个损失项 + 损失聚合 trainer（LossTerm 体系） | 第 3、4 章 |
| `src/model/conditioning/ noise/ priors/ frequency/` | 四个策略子包：条件适配/beta 调度、噪声调度、先验生成、频域分析与路由 | 第 5 章 |
| `src/mechanism_validation/` | 机制验证库：H1-H6 假设检验、H4-v2 内部 CV、特征溯源/因果/涌现、放大一致性审计 | 第 6 章 |
| `src/scripts/` + `scripts/` | 训练/评估入口、消融 runner、H1-H6 验证脚本、V2 冻结管线、H3/先验锚定实验、评估分析、诊断、云协作、预训练 | 第 7-12 章 |
| `wuzhe/` | 内嵌的历史快照工程：与根 src 近重复，独有 comparison_experiments（pix2pix/cyclegan/reggan/district_gan/cpdm 五个对比 baseline）与论文/专利文档生成脚本 | 第 13 章 |
| `tests/` | 49 个 pytest 文件：单元/管线/runner 测试 | 第 14 章 |
| `configs/` | 实验配置（yaml 消融计划 + json 机制管线契约）与根配置 pyproject/pixi | 第 15 章 |
| `docs/` | 设计文档、机制冻结清单、研究雷达（不在代码地图范围内，作为背景引用） | — |
| `Data/ main_data/ wuzhe/main_data wuzhe/wuzhe_data` | 原始与中间数据（DICOM/PNG/标签/五折划分 CSV） | — |

### 0.3 核心数据流（端到端）

```
DICOM(Data/) ──data_manifest──▶ manifest.csv ──dataset──▶ .npz 张量缓存 ─┐
                                                                          ├─▶ split_manifest ─▶ organ_preprocess ─▶ semantic_builder
PNG(main_data/) ──png_cache──▶ PNG 张量缓存（当前生产数据路径）──────────┘
                                                                    │
configs/experiments/*.yaml ──train_v2.py──▶ src/model/trainer.py ◀────┘
     │                                        │
     │                              src/model/slmf_bbdm.py（BBDM 前向/反向）
     │                              ├─ wavelet_unet / bbdm_unet（骨干）
     │                              ├─ frequency/spectral_router（层级频带路由）
     │                              ├─ priors/*（gabor/hotspot/organ/semantic）
     │                              ├─ noise/*、conditioning/*（调度与条件注入）
     │                              └─ loss_terms/*（病灶保留+频域+一致性损失）
     ▼
checkpoints/（ckpt + EMA + resolved_config + metrics jsonl）
     │
     ├─▶ scripts/evaluate.py（测试集指标）/ src/scripts/evaluate.py
     ├─▶ scripts/run_frequency_ablations / run_spectral_router_v5（消融筛选）
     └─▶ scripts/validate_h1..h6 + src/mechanism_validation（机制验证）
              └─▶ results/mechanism_validation*/（decision.json 门禁）
```

### 0.4 常用入口命令（pixi tasks 摘录）

| 任务 | 命令要点 |
|---|---|
| 数据清单 | `pixi run data-manifest`（Data/ → cache/manifest.csv） |
| PNG 缓存 | `pixi run data-cache-png-main`（main_data/ → cache/tensors_main，当前生产数据路径） |
| 划分/器官/语义 | `pixi run data-split` / `data-organ` / `data-semantic` |
| 主训练 | `pixi run train`（slmf_full.yaml）或 `pixi run train-png-baseline`（推荐稳定入口） |
| 冒烟训练 | `pixi run train-smoke` / `train-png-smoke`（假数据 5 epoch） |
| 评估 | `pixi run eval-png-baseline`（test split） |
| 频域消融 | `pixi run train-frequency-ablations[-v2]` |
| 路由消融 V5 | `pixi run train-spectral-router-v5` |
| 机制验证 | `pixi run mechanism-validate-local|all|plan`（H2-H6 fail-closed 门禁） |
| H4-v2 内部 CV | `pixi run mechanism-v2-freeze|internal|plan` |
| V2 云端/推理审计 | `pixi run v2-cloud-stage03` / `v2-inference-audit` |
| 测试 | `pixi run test`（pytest，pyproject 中排除 wuzhe 等目录） |
| 工具 | `pixi run inspect`（参数量/模块状态）、`gpu-info` |

### 0.5 技术栈

Python ≥3.11（win-64 pixi 环境）；PyTorch ≥2.8 (cu128) + torchvision、accelerate、einops、ema-pytorch、diffusers、transformers、timm；医学影像：pydicom、nibabel、TotalSegmentator、scikit-image、opencv；评估：pytorch-fid、lpips；可视化/日志：matplotlib、tensorboard、wandb；测试：pytest（--strict-markers）。PyPI 走清华镜像，conda 走清华 conda-forge/bioconda 镜像。

### 0.6 实验演进脉络（配置主线）

slmf_baseline/slmf_full（DICOM+SUV 全家桶）→ **PNG-only baseline**（去 DICOM/SUV/organ 依赖，稳定无条纹）→ 频域消融 v1/v2（residual-frequency、boundary-reliable v3/v4）→ **spectral-router v5**（当前生产：时间步条件化层级频带路由）→ v6/v7/v8（路由修复/冷偏置/最终校准筛选）→ **prior-anchored router**（先验锚定，探索性，含 100e/300e/identifiable 变体）→ **perceptual_x0 / fullstack-p2**（病灶感知感知 x0 损失，最新方向）。并行主线：机制验证管线 v1/v2（H1-H6 假设的门禁化验证）。

### 0.7 新人阅读路线

1. 第 15 章（配置）+ 第 7 章（train_v2 入口）理解"一条命令发生了什么"；
2. 第 1 章（数据层）→ 第 2 章（模型核心，重点 slmf_bbdm.py / trainer.py）；
3. 第 5 章（四个策略子包）→ 第 3、4 章（损失体系）；
4. 第 6、8-10 章（机制验证：这是本仓库的方法论特色——每个机制主张都要过 H 门禁）；
5. 第 13 章（wuzhe 对比实验，写论文 baseline 用）；第 14 章（测试即契约）。

---

## 1. 数据层 src/data

### 模块概述

本组文件构成 SLMF-BBDM（子宫内膜癌 CT→PET 合成 / 小病灶分割扩散模型）的完整离线数据管线与在线训练数据入口。
设计模式为典型的**两阶段（离线缓存 + 在线读取）**：Stage 0 由 data_manifest.py 对原始 PNG/DICOM 做配对审计产出 manifest.csv；随后两条互斥路线把源数据转成 .npz 张量缓存——DICOM 路线（dataset.CacheBuilder，保留 HU/SUV 物理标度）与 PNG 路线（png_cache.py，仅强度归一化、无物理意义）。
缓存建好后由 organ_preprocess.py（TotalSegmentator 器官先验）与 semantic_builder.py（冻结骨干语义 token）向同一 .npz **增量回写**可选键；split_manifest.py 生成患者级划分清单防泄漏。
数据流：原始数据 → manifest 审计 → .npz 缓存（可选键增强）→ split_manifest → dataset.CachedDataset/build_dataloaders → src/model/trainer.py 消费；lineage.py 以 SHA-256 封印链机制贯穿缓存与 checkpoint。
所有模块仅通过包子模块路径直接 import（`src/data/__init__.py` 只有 docstring，不做 re-export）。

---

### `src/__init__.py`

**职责**：包根标识文件，仅一行 docstring `SLMF-BBDM: Small-Lesion Metabolic-Fidelity Brownian Bridge Diffusion Model.`，无任何代码逻辑。

**接口**：无（无 import、无定义）。

**依赖**：
- 内部： 无
- 外部： 无

**输入**：无。
**输出**：无。
**处理过程**：不适用（空标记文件，使 src 成为 Python 包）。

---

### `src/data/__init__.py`

**职责**：数据子包标识文件，仅一行 docstring "SLMF-BBDM data pipeline: datasets, preprocessing, and manifest tools"，不做任何符号 re-export。

**接口**：无。
**依赖**：
- 内部： 无
- 外部： 无
**输入**：无。
**输出**：无。
**处理过程**：不适用；外部消费者必须写全路径如 `from src.data.dataset import build_dataloaders`。

---

### `src/data/data_manifest.py`

**职责**：Stage 0 数据审计与配对工具——扫描 CT/PET 图像与标注的 PNG/DICOM 目录树，按 sample_id（`患者3位+切片3位` 六位数字）配对，做患者级 train/val/test 划分，输出 manifest.csv 与审计报告。它是 DICOM 路线缓存（dataset.CacheBuilder）的上游输入。

**接口**（全部为模块级函数/数据类，含 CLI）：
- `@dataclass DicomSlice`：path/sop_instance_uid/series_uid/instance_number/z。
- `@dataclass SampleRecord`：sample_id、patient_id、slice_id、split + 四路（ct/pet/ct_label/pet_label）PNG/DICOM 路径与存在标志、DICOM UID/Z 元数据、mapping_status/mapping_warning。
- `parse_sample_id(sample_id, pattern=DEFAULT_SAMPLE_ID_REGEX) -> Tuple[str, int]`：按正则 `^(?P<patient_id>\d{3})(?P<slice_id>\d{3})$` 解析，失败抛 ValueError。
- `scan_png_dir(root) -> Dict[sample_id, abs_path]`：重复 id 抛错。
- `scan_dicom_series(root, order) -> Tuple[Dict[patient_id, List[DicomSlice]], errors]`：按 z_desc/z_asc/instance_asc/filename 排序，坏文件跳过并记入 errors。
- `build_sample_records(raw_root, dicom_root, test_png_root, sample_id_regex, dicom_slice_order, dicom_index_offset) -> List[SampleRecord]`：核心配对。
- `assign_splits(records, test_sample_ids=None, val_ratio=0.0, seed=42) -> None`：就地写入 split，末尾调用 `_verify_split_no_overlap` 断言三集合患者无交集。
- `compute_dataset_fingerprint(records) -> str`：sha256 前 12 位；路径降为 `basename:filesize` 保证跨机器可移植。
- `write_csv / write_jsonl / write_split_csvs`：序列化。
- `write_report(records, out_root, dicom_errors=None)`：写 report.json + report.md。
- CLI：`python -m src.data.data_manifest --raw-root Data --dicom-root <dir> --output cache/manifest.csv --val-ratio 0.15 --seed 42 [--splits-dir DIR] [--test-png-root DIR] [--dicom-slice-order z_desc] [--dicom-index-offset 0] [--test-sample-ids ...] [--print-stats]`

**依赖**：
- 内部： 无（纯独立工具，产物被 src/data/dataset.py 的 CacheBuilder 消费）
- 外部： pydicom（懒加载于 `_read_dicom_slice_meta`）、csv/json/hashlib/re/dataclasses/datetime

**输入**：目录树 `<root>/part_{CT,PET}/{train,test}_data/{ImageSet,LabelSet}/{PNG,DICOM}`；sample_id 文件名约定；CLI 参数（默认 raw-root=Data、output=cache/manifest.csv）。
**输出**：cache/manifest.csv（39 列）、可选 jsonl 与 train/val/test 分包 csv、manifest_report/report.json + report.md（缺失清单、患者重叠检查、指纹）。
**处理过程**：
1. 扫描 8 个 PNG 目录（CT/PET × 图像/标注 × train/test），构建 sample_id → 路径映射，重复 id 抛错；关键目录缺失抛 FileNotFoundError。
2. 扫描对应 8 个 DICOM 目录，逐文件读元数据（stop_before_pixels），坏文件跳过；train/test 患者重叠直接抛 ValueError。
3. 按指定顺序对每位患者的 DICOM 切片排序，并为**每种模态独立**构建 `slice_id → 本模态局部序号` 索引（防止 CT-only/PET-only 切片互相错位）。
4. 对全部 sample_id 并集逐条生成 SampleRecord：填 PNG 路径/来源子集，再用 局部序号+offset 在 DICOM 列表中挂接路径与 UID/Z；找不到则置 mapping_status 码（missing_ct_dicom 等）。
5. `assign_splits`：显式 test_sample_ids 或按 `*_source_subset=="test"` 标记 test 患者，其余患者 seed 洗牌后按 val_ratio 切 train/val，末尾断言无患者重叠。
6. 写 manifest.csv/jsonl/split csv，并生成审计报告（配对计数、缺失清单、指纹、DICOM 错误）。

---

### `src/data/dataset.py`

**职责**：**全仓库训练数据入口**。Phase 1 `CacheBuilder` 把 data_manifest 产出的 manifest.csv 中的 DICOM 离线转成归一化 .npz 缓存（保留 HU/SUV 物理标度）；Phase 2 `CachedDataset/FakeDataset + build_dataloaders` 在线读缓存构建 DataLoader，供 scripts/train_v2.py 与 src/model/trainer.py 训练。

**接口**：
- `compute_suv(ds, activity_conc) -> Tuple[np.ndarray, bool, dict]`：从 PET DICOM 计算 SUV（BQML→体重/衰减校正剂量），返回 (suv, suv_ok, meta)。
- `@dataclass PreprocessConfig`：ct_hu_min=-200/ct_hu_max=500、pet_suv_max=50、image_size=192、strict_suv=True。
- `class CacheBuilder`：`__init__(config)`；`build(manifest_csv, output_dir, *, validate_spatial=True, z_tolerance_mm=2.0) -> Dict stats`（total_rows/built/skipped/errors）。
- `@dataclass SampleEntry`：sample_id/patient_id/slice_id/split/cache_path/has_mask。
- `class CachedDataset(Dataset)`：`__init__(cache_dir, split="train", *, augment=False, split_manifest=None, required_keys=["ct","pet"], optional_keys=[], augmentation_seed=0)`；`__len__`；`__getitem__(idx) -> Dict[str, torch.Tensor]`（idx 支持 (epoch, i) 二元组）。
- `class FakeDataset(Dataset)`：`__init__(num_samples=32, image_size=32, seed=0)`，同构 sample 供冒烟测试。
- `class EpochShuffleSampler(Sampler)`：`__init__(data_source, *, generator)`、`set_epoch(epoch)`、`__iter__` 产出 (epoch, index) 戳记索引。
- `build_dataloaders(data_cfg, run_cfg) -> Tuple[DataLoader, Optional[DataLoader]]`：配置驱动工厂。
- CLI：`python -m src.data.dataset --manifest cache/manifest.csv --out-dir cache/tensors [--ct-hu-min -150] [--ct-hu-max 250] [--pet-suv-max 20] [--image-size 192] [--no-spatial-check] [--z-tolerance-mm 2.0] [--allow-non-suv]`

**依赖**：
- 内部： `src.data.split_manifest.SplitManifest`（函数内 import，做患者级划分权威判定）
- 外部： numpy、torch/torch.nn.functional、PIL、pydicom（懒加载）、csv/json（懒加载）

**输入**：
- CacheBuilder：manifest CSV 行（sample_id/split/ct_dicom_path/pet_dicom_path/ct_dicom_z/pet_dicom_z/pet_label_png_path/ct_label_png_path）。
- CachedDataset：cache_dir 下 `<sid>.npz` + `<sid>_meta.json`；可选 split_manifest.csv。
- build_dataloaders 配置键：`data.cache_dir / data.val_cache_dir / data.split_manifest / data.image_size(192) / data.batch_size(4) / data.val_batch_size / data.augment / data.use_fake_data / data.required_keys / data.optional_keys`、`runtime.num_workers(4) / pin_memory / persistent_workers / prefetch_factor / dataloader_seed`。

**输出**：
- .npz 载荷（CacheBuilder，均 float32，H=W=image_size 默认 192）：`ct [1,H,W]`（HU 窗宽窗位→[0,1]→线性到 [-1,1]）、`ct_hu [1,H,W]`（物理 HU）、`pet [1,H,W]`（SUV/活度 clip→[0,1]→[-1,1]）、`mu_map [1,H,W]`（0.096·HU/1000+0.096）、可选 `pet_suv [1,H,W]` 或 `pet_activity [1,H,W]`（二选一）、可选 `mask [1,H,W]`（PNG 标注>0.5 二值）、`scale_meta_json`（uint8 JSON 字节：pet_suv_max/suv_ok/pet_raw_min·max/ct_hu_min·max/pet_physical_key/patient_id/slice_id/uptake_min/weight_kg/age_years/thickness_mm/z_mm 等）；另写 `<sid>_meta.json`（split/has_label/source="dicom"）。organ/semantic 键由 organ_preprocess.py / semantic_builder.py 后续回写。
- `CachedDataset.__getitem__` 返回的 **sample 字典（trainer 消费的张量结构）**：`ct [1,H,W]`、`pet [1,H,W]`（合成目标）、`mask [1,H,W]`、`organ_mask [6,H,W]` 与 `organ_distance [6,H,W]`（缺失补零，6 类=子宫盆腔/膀胱/直肠肠管/骨/脂肪/肌肉其它）、`mu_map [1,H,W]`、`ct_hu [1,H,W]`、`pet_suv [1,H,W]`、`pet_activity [1,H,W]`（均 float32）、`meta`（scale_meta dict，非张量）、可选 `semantic_tokens [num_tokens=4, token_dim=64]`。
- build_dataloaders：训练 loader（EpochShuffleSampler + drop_last + 按需 pin_memory/persistent_workers/prefetch）与验证 loader（shuffle=False，无 val 样本时返回 None）。

**处理过程**：
1. CacheBuilder.build 逐行校验 manifest：split 合法、DICOM 存在、可选 z 坐标空间校验（|ct_z−pet_z|≤容差）。
2. 读 CT DICOM→HU（RescaleSlope/Intercept），派生 mu_map 与窗宽窗位归一化 ct；读 PET DICOM→compute_suv，strict 模式 SUV 失败即抛错，非 strict 回退活度 min-max。
3. 标注 PNG→最近邻插值→二值 mask；抽取 DICOM 条件元数据（年龄/层厚/z/注射后时间/体重）组装 scale_meta，JSON 字节存入 npz，并写 `<sid>_meta.json`。
4. CachedDataset 初始化：显式传入但缺失的 split_manifest 是硬错误（拒绝静默回退 _meta.json 造成患者泄漏）；manifest 为权威（patient_id/split 优先取 manifest 行），否则回退 _meta.json 的 split 注记；空 split 抛 ValueError。
5. 1/20 抽样扫描 .npz 键存在性，required 键缺失打 WARNING（下游依赖该键的 loss 可能静默退化）。
6. `__getitem__`：np.load→torch 张量，可选键零回退，解析 scale_meta_json；augment 开启时以 `(seed, epoch, idx)` 派生确定性种子，50% 概率对 19 通道堆叠张量做水平翻转（保证 persistent_workers 下断点续训完全可复现）。
7. build_dataloaders：**先判 use_fake_data 再判 cache_dir**（防冒烟测试误成真训练）；构建 train/val Dataset 与三个独立 seed 的 Generator（sampler/worker/val）。
8. CLI 入口把参数装进 PreprocessConfig 后调用 CacheBuilder.build 并打印 stats JSON。

**被 trainer 消费的方式**：scripts/train_v2.py:247 调用 `build_dataloaders(data_cfg, run_cfg)` 后把 (train_loader, val_loader) 传入 `Trainer(model, config, train_loader, val_loader)`（src/model/trainer.py:574-580）。Trainer 训练循环 `for batch in self.train_loader`，用 `_to_device`（trainer.py:416/938，键值对张量 .to(device, non_blocking) 而 dict 型 meta 原样保留）搬到 GPU；每 epoch 从 `train_loader.sampler` 取样器推进 epoch（trainer.py:910）以驱动 EpochShuffleSampler 的可复现增强；`batch["pet"]` 作为合成/评估目标（trainer.py:1519），`batch["ct"]`/`batch["mask"]` 用于可视化（trainer.py:2140/2147）。整个 batch dict 进一步传给 SLMFBBDM：`batch["ct"]/batch["pet"]` 进扩散主干（slmf_bbdm.py 多处），`batch.get("meta")` 经 `_meta_to_tensor` 变条件张量（slmf_bbdm.py:1639/2040），先验模块 `src/model/priors/semantic.py` 读 `batch["semantic_tokens"]`（semantic.py:60-61）、`src/model/priors/organ.py` 读 `batch["organ_mask"/"organ_distance"/"mu_map"]`（organ.py:50-52）、`hotspot.py/gabor.py` 读 `batch["ct"]`。

---

### `src/data/lineage.py`

**职责**：严格缓存血统（lineage）机制——当配置 `data.require_cache_lineage: true` 时，训练前 fail-closed 校验封印缓存元数据（cache_lineage.json）自洽且与锁定的数据集契约一致，并把血统嵌入 checkpoint、在加载时逐字段比对。

**接口**：
- 常量 `REQUIRED_CHECKPOINT_LINEAGE_FIELDS`：6 个 sha256 字段（manifest_semantic/raw_png_combined/preprocessing_config/dataset_contract/cache_payload/cache_metadata）。
- `class CacheLineageError(ValueError)`：所有血统失败统一异常。
- `load_checkpoint_data_lineage(config, *, root=None) -> Optional[dict]`：读 config.data（require_cache_lineage/use_fake_data/cache_dir/cache_lineage/dataset_contract），非严格且无 lineage 文件返回 None，严格模式任何不一致抛错。
- `attach_data_lineage(checkpoint, data_lineage) -> dict`：在 checkpoint 顶层稳定键 `data_lineage` 注入血统。
- `validate_checkpoint_data_lineage(checkpoint, expected_lineage, *, required, context="checkpoint") -> Optional[dict]`：校验输入 checkpoint 内嵌血统（自哈希 + 6 字段格式 + lineage_type=="verified_png_tensor_cache"）并与当前缓存逐字段比对。

**依赖**：
- 内部： 无（被 src/model/trainer.py:30、src/model/loss_terms/trainer.py:23、src/model/mean_pretraining.py:15、src/model/slmf_bbdm.py:445/2282、src/mechanism_validation/model_experiments.py:25、scripts/train_v2.py:23、scripts/evaluate.py:45 及大量 validate_* 脚本消费）
- 外部： hashlib/json/os/re/pathlib（纯标准库）

**输入**：运行配置键 `data.require_cache_lineage / data.use_fake_data / data.cache_dir / data.cache_lineage / data.dataset_contract`；缓存目录内 cache_lineage.json；数据集契约 JSON（含 contract_sha256 自哈希）；checkpoint dict（含/缺 data_lineage 键）。
**输出**：返回血统 dict（或 None）；异常消息 JSON 化列出 mismatch 字段；不写文件。
**处理过程**：
1. 解析 config.data：use_fake_data 且严格→抛错；非严格且无缓存配置→返回 None。
2. 定位 lineage 文件：显式 data.cache_lineage 优先，否则 `<cache_dir>/cache_lineage.json`；缺失时严格抛错、宽松返回 None。
3. 校验 lineage 自哈希（剔除 cache_metadata_sha256 后 canonical-JSON sha256 比对）、6 个字段均为 64 位 hex、lineage_type 必须为 "verified_png_tensor_cache"、文件必须位于 cache_dir 内。
4. 若配置 dataset_contract：读契约、验其自哈希，再逐字段比对 manifest/raw_png/preprocessing_config/contract 四个 sha256；严格模式下契约缺失即抛错。
5. attach_data_lineage 把通过校验的血统写入 checkpoint["data_lineage"]；validate_checkpoint_data_lineage 供加载方按 required 决定缺 data_lineage/与当前缓存不一致时是否抛错。

---

### `src/data/organ_preprocess.py`

**职责**：器官先验离线增强——按患者重建 CT HU 体数据，跑 TotalSegmentator 3D 分割，把 117 个 TS 标签映射到固定 6 器官类，逐切片生成 one-hot organ_mask 与带符号距离变换 organ_distance 回写 .npz。

**接口**：
- 常量 `TOTALSEGMENTATOR_V2_MAPPING: Dict[int, int]`（TS 标签→0..5）、`NUM_ORGAN_CLASSES = 6`、`ORGAN_CLASS_NAMES`（uterus_pelvic/bladder/rectum_bowel/bone/fat/muscle_other）。
- `_distance_transform_2d(mask) -> np.ndarray`：内正外负、tanh 软归一到约 [-1,1]。
- `_build_ct_volume_from_cache(cache_dir, patient_id)`：从缓存重建 `[D,H,W]` HU 体 + 尽力 affine + 排序 npz 路径表（供切片对应）；无 nibabel 或无切片返回 None。
- `process_cache_with_organ_prior(cache_dir, total_segmentator_binary=None, mapping=None, gpu=True, image_size=192, stats=None) -> Dict stats`（total_npz/updated/skipped/errors/organ_coverage_*）。
- `generate_organ_report(stats) -> str`：Markdown 质检报告（逐器官平均覆盖率、零掩码百分比>50% 告警）。
- `_write_zeros_organ(npz_path)` / `_update_npz_with_organ(npz_path, organ_mask, organ_distance)`：向既有 .npz 增量写键。
- CLI：`python -m src.data.organ_preprocess --cache-dir cache/tensors [--image-size 192] [--gpu|--cpu] [--organ-map mapping.json] [--output-report report.md] [--dry-run]`

**依赖**：
- 内部： 无（操作 dataset.py/png_cache.py 产出的 .npz 缓存；产物 organ_mask/organ_distance 键被 src/data/dataset.py 的 CachedDataset 读取并最终喂给 src/model/priors/organ.py）
- 外部： numpy、torch（最近邻重采样）；懒加载 scipy.ndimage（距离变换）、totalsegmentator.python_api、nibabel、tempfile

**输入**：既有 .npz 缓存目录（需含 ct_hu 或 ct + scale_meta_json 中的 HU 窗）；sample_id 前 3 位 = 患者号；可选 JSON 映射覆盖默认 TS→6 类映射。
**输出**：每个 .npz 回写 `organ_mask [6,H,W] float32` 与 `organ_distance [6,H,W] float32`（TS 不可用时写全零）；stdout/文件 Markdown 报告；返回 stats dict。
**处理过程**：
1. 按文件名前 3 位把 .npz 分组到患者。
2. 每位患者从缓存重建 CT HU 体（优先 ct_hu 键；legacy 回退由 ct + scale_meta HU 窗反推），构造假设 1mm 面内/元数据层厚的 affine，保存排序路径表以建立 体轴↔npz 精确对应。
3. 体数据写临时 .nii.gz，调 TotalSegmentator python_api（task="total", fast=True, ml=True, device 按 --gpu/--cpu）得 [D,H,W] 多标签分割；失败记 error 并对该患者全部切片写零。
4. 逐切片取 2D 标签图（索引越界取最近邻切片），最近邻重采样到 image_size，按映射逐 TS 标签聚合 max 生成每类二值掩码。
5. 每类掩码做带符号距离变换并 tanh 归一，与掩码一起回写 .npz。
6. 累计 organ_coverage，汇总均值与零掩码百分比，生成报告；--dry-run 只统计文件/患者数并探测 TS/nibabel 可用性。

---

### `src/data/png_cache.py`

**职责**：PNG 路线缓存构建器——把按 split 组织的 PNG 切片（main_data/split.csv + train/ct、train/pet_peizhuan|pet、train/label 等）转换为与 CachedDataset 兼容的 .npz。明确 PNG 强度不具物理 SUV/HU 意义，scale_meta 中关闭物理键，SUV 相关损失应禁用。

**接口**：
- `build_png_cache(*, png_root, split_csv=None, out_dir, split_manifest=None, image_size=192, ct_subdir="ct", pet_subdirs=("pet_peizhuan","pet"), label_subdir="label", normalization="auto", mask_threshold=0.0, pet_invert=False, allow_missing_mask=False, overwrite=True) -> Dict stats`（total_rows/built/skipped/pet_source_counts/errors/cache_dir/split_manifest）。
- 辅助：`_load_png_array`（mask 用最近邻、图像用双线性重采样，兼容 8/16 位模式）、`_normalise_unit`（auto：按值域判 1/255/65535 除数，否则 minmax）、`_normalise_image`（[0,1]→[-1,1]）、`_normalise_mask`（阈值二值化）、`_write_split_manifest`。
- CLI：`python -m src.data.png_cache --png-root main_data --out-dir cache/tensors [--split-csv ...] [--split-manifest ...] [--image-size 192] [--ct-subdir ct] [--pet-subdirs pet_peizhuan pet] [--normalization auto|minmax] [--mask-threshold 0] [--pet-invert] [--allow-missing-mask] [--no-overwrite] [--allow-skips]`；有 skip 且未加 --allow-skips 时退出码 2。

**依赖**：
- 内部： 无（产物布局直接匹配 src.data.dataset.CachedDataset；同时生成的 split_manifest.csv 可供 CachedDataset 的 split_manifest 参数与 src.data.split_manifest.SplitManifest 使用）
- 外部： numpy、PIL、csv/json

**输入**：`png_root/split.csv`（列 file_name/sample_id、patient_id、split）；`png_root/<split>/ct|pet_peizhuan|pet|label/<sample_id>.png`；pet_invert 针对白底反色 PET（背景 255、热灶最暗，先 255-x 再归一）。
**输出**：每样本 `<sid>.npz`（ct/pet 各 [1,H,W]∈[-1,1]、mask [1,H,W]、mu_map 全零占位、scale_meta_json：suv_ok=False、pet_physical_kind="png_intensity"、pet/ct_raw_min·max、patient_id/slice_id）+ `<sid>_meta.json`（source="png"、pet_source、PNG 路径）；同目录级 split_manifest.csv（5 列标准布局）与同名 .json stats。
**处理过程**：
1. 校验 split.csv/png_root 存在、pet_subdirs 非空，读 CSV 行。
2. 逐行解析 sample_id（去扩展名）、split 合法性、patient_id（CSV 列优先，回退文件名前 3 位）。
3. 在对应 split 子目录按 pet_subdirs 优先级查找 CT/PET PNG（缺 CT/PET 或缺 label 且未豁免则 skip 计数并记前 50 条错误）。
4. 加载重采样 PNG；pet_invert 时先反色；auto/minmax 归一到 [0,1] 再映射 [-1,1]；mask 按阈值二值化（无 label 时全零）。
5. 组装 scale_meta（显式声明非物理标度）与 payload，np.savez_compressed 写出，并写 _meta.json 与 manifest 行。
6. 结束写 split_manifest.csv + stats JSON 并返回；PET 来源计数写入 pet_source_counts。

---

### `src/data/semantic_builder.py`

**职责**：语义 token 离线构建器——CT HU→三窗宽伪 RGB（软组织/脂软/骨窗）→冻结骨干（RadImageNet ResNet50 / DINOv2 ViT-S/14 / 随机投影基线）→逐切片 token [4,64] 回写 .npz，训练期由 SemanticPrior 先验经 Cross-Attention 注入。

**接口**：
- `ct_hu_to_3win_rgb(ct_hu, windows=None) -> np.ndarray [H,W,3]∈[0,1]`：默认窗 `soft_tissue W350/L40、fat_soft W400/L-50、bone W1500/L500`。
- `ct_norm_to_3win_rgb(ct_norm, ct_hu_min=-150, ct_hu_max=250)`：[-1,1] 归一 CT 反推 HU 后走上函数（legacy 缓存回退）。
- `class _BackboneWrapper(nn.Module)`（抽象）：`forward(rgb [B,3,H,W]) -> [B,num_tokens,token_dim]`；实现类 `_RadImageNetBackbone`（timm resnet50.radimagenet_irh，2048 维池化特征→Linear 展开；timm 缺失时回退统计特征补零）、`_DINOv2Backbone`（torch.hub dinov2_vits14，384 维 CLS；文档注明**未验证**）、`_RandomBackbone`（seed=42 固定随机投影基线）。
- `class SemanticTokenBuilder`：`__init__(backend="radimagenet", token_dim=64, num_tokens=4, device="cpu", image_size=192)`；`process_cache(cache_dir, batch_size=16, meta=None) -> Dict stats`（total/updated/skipped/errors），冻结参数、eval 模式、no_grad。
- CLI：`python -m src.data.semantic_builder --cache-dir cache/tensors [--backend radimagenet|dinov2|random] [--token-dim 64] [--num-tokens 4] [--batch-size 16] [--gpu] [--dry-run]`

**依赖**：
- 内部： 无（读 dataset.py/png_cache.py 产出的 .npz；回写的 semantic_tokens 由 src/data/dataset.py 的 CachedDataset 透传，训练时被 src/model/priors/semantic.py:60-61 的 `batch["semantic_tokens"]` 分支消费）
- 外部： numpy、torch；懒加载 timm（RadImageNet）、torch.hub（DINOv2）

**输入**：.npz 缓存目录；每文件需 ct_hu（优先）或 ct + scale_meta_json 的 ct_hu_min/max；RGB 统一重采样到 224×224（骨干输入约定）。
**输出**：每个 .npz 回写 `semantic_tokens [num_tokens, token_dim] float32`（默认 [4,64]）；stats dict。
**处理过程**：
1. 按 batch_size 分批遍历排序后的 .npz。
2. 每文件读 scale_meta 取 HU 窗；有 ct_hu 直接三窗 RGB，否则用归一 CT 反推 HU（legacy 回退）；形状异常记 error 跳过。
3. RGB → [3,H,W] → 双线性 224×224。
4. 批量前向冻结骨干，token_proj 展开为 [B,num_tokens,token_dim]。
5. 按 valid_indices 对齐写回各 .npz 的 semantic_tokens 键（np.savez_compressed 整体重写）。
6. 打印进度与统计；--dry-run 仅检查文件数与后端可加载性。

---

### `src/data/split_manifest.py`

**职责**：患者级划分清单——从 .npz 缓存按 patient_id 分组生成 split_manifest.csv（+同名 .json 统计），保证 train/val/test 无患者泄漏；提供 SplitManifest 运行时读取类供 CachedDataset 强制执行划分。

**接口**：
- `_parse_sample_id(sid) -> Tuple[pid, slice_id]`：前 3 位患者号、第 4-6 位切片号。
- `_compute_fingerprint(rows, seed) -> str`：sha256 前 12 位。
- `generate_split_manifest(cache_dir, output_csv, val_ratio=0.15, seed=42, test_patient_ids=None, no_test=False) -> Dict stats`：总/各 split 患者数与样本数、fingerprint、三类患者 id 清单。
- `load_split_manifest(manifest_path) -> Dict[sample_id, row]`。
- `class SplitManifest`：`__init__(manifest_path)`（加载 CSV + 同名 .json stats）、`get_split(sample_id)`、`get_entry(sample_id) -> Optional[row]`、`get_patient_split(patient_id)`、`get_samples_for_split(split)`、`validate_no_overlap() -> bool`（多 split 患者打 WARNING 返回 False）、`__len__`、`__repr__`。
- CLI：`python -m src.data.split_manifest --cache-dir cache/tensors --output cache/split_manifest.csv [--val-ratio 0.15] [--seed 42] [--test-patients PID ...] [--no-test] [--print-stats]`

**依赖**：
- 内部： `src.data.split_manifest.SplitManifest` 被 src/data/dataset.py 的 CachedDataset 消费（患者级划分权威来源）；也被 scripts/pretrain_pet_encoder.py、tests/test_smoke.py、tests/test_perceptual_x0_loss.py 使用
- 外部： csv/hashlib/json/random/numpy

**输入**：含 `*.npz` 的缓存目录（文件名前 3 位=患者号）；val_ratio/seed；可选显式 test 患者集合或 no_test。
**输出**：split_manifest.csv（列 sample_id/patient_id/slice_id/split/cache_path 绝对路径）+ split_manifest.json（统计 + fingerprint + 各 split 患者 id 清单）；返回 stats dict。
**处理过程**：
1. 扫描缓存 .npz，按患者分组。
2. 确定 test 患者：显式集合 >（非 no_test 时）默认取排序后最后 10% 患者且至少 1 人。
3. 余下患者用 seed 洗牌，按 val_ratio 切 val/train（≥2 患者且 ratio>0 时保证至少 1 个 val 患者）。
4. 逐样本展开写 CSV（cache_path 记绝对路径），写 JSON 统计与指纹。
5. 末尾三断言 train/val/test 患者两两无交集；SplitManifest.validate_no_overlap 供运行时复查。

---

### 模块依赖小结

**本组内部依赖**（组内唯一一条边）：`src/data/dataset.py` → `src.data.split_manifest.SplitManifest`（CachedDataset 构造时函数内 import，manifest 行为 patient_id/split 的权威）。其余文件彼此独立，仅通过**磁盘产物**耦合：data_manifest.py 的 manifest.csv 喂 dataset.CacheBuilder；png_cache.py 直接产出 .npz + split_manifest.csv；organ_preprocess.py 与 semantic_builder.py 向任一路线的 .npz 增量回写 organ_*/semantic_tokens 键；lineage.py 校验的 cache_lineage.json/cache 目录契约由 png 缓存封印流程产出。

**被内部模块消费**（grep 查证，排除 artifacts/ 历史快照）：
- `src.data.dataset.build_dataloaders`：scripts/train_v2.py:23/247（构建 loader 后传入 src/model/trainer.py 的 Trainer）；tests/test_smoke.py、tests/test_trainer_monitoring.py 等大量测试。
- `src.data.lineage`：src/model/trainer.py:30、src/model/loss_terms/trainer.py:23、src/model/mean_pretraining.py:15、src/model/slmf_bbdm.py:445/2282、src/mechanism_validation/model_experiments.py:25、scripts/train_v2.py、scripts/evaluate.py、scripts/audit_checkpoint_lineage.py、scripts/pretrain_conditional_mean.py、scripts/calibrate_h3_v2_full_timestep_native_null.py、scripts/validate_h*.py、scripts/run_prior_anchored_router_100e.py 等（attach/load/validate 三函数组合）。
- `src.data.split_manifest`：src/data/dataset.py、scripts/pretrain_pet_encoder.py、tests/test_smoke.py、tests/test_perceptual_x0_loss.py。
- `src.data.data_manifest / png_cache / semantic_builder / organ_preprocess`：作为 `python -m` CLI 由运维/审计流程调用，并被 src/mechanism_validation/feature_provenance.py、src/model/priors/semantic.py（文档引用）、src/model/loss_terms/false_hotspot.py（概念引用）、scripts/run_mechanism_stage0_data_audit.py、tests/test_cloud_stage0c_gate.py、tests/test_smoke.py 引用。
- 数据张量的最终去向：`batch` 字典 → src/model/trainer.py（pet 为目标、ct/mask 可视化、_to_device 搬运、sampler.set_epoch 驱动复现增强）→ src/model/slmf_bbdm.py（ct/pet 进主干、meta→_meta_to_tensor）→ src/model/priors/*（semantic_tokens/organ_mask/organ_distance/mu_map/ct）。


---

## 2. 模型核心 src/model（根模块）

### 模块概述

本组文件构成 SLMF-BBDM（Small-Lesion Metabolic-Fidelity Brownian Bridge diffusion Model，面向子宫内膜癌 CT→PET 合成与小病灶分割）的核心模型层：`slmf_bbdm.py` 是总装类，按『注册表 + 接口约定』的可插拔设计将先验模块（Gabor/器官/热点/语义）、条件适配器、噪声调度（DDPM/布朗桥/尺度自适应）、UNet 骨干与可插拔损失栈拼装为一个 `nn.Module`；`trainer.py` 提供完整的训练/验证/恢复/导出循环。`interfaces.py` 定义四大插件基类（PriorModule/ConditionAdapter/NoiseSchedule/LossTerm）与 ConditionBundle/LossContext 数据契约，`registry.py` 提供按名实例化的轻量注册表，二者共同保证『每个模块都有 enabled 开关，禁用时走 NoOp 路径，训练主循环零条件分支』的消融原则。`bbdm_unet.py`/`wavelet_unet.py` 是去噪骨干（后者用 Haar DWT/IWT 替代普通下/上采样），`mean_predictor.py`+`mean_pretraining.py` 实现条件均值（PET 低频分量）预训练与残差桥，`segmenter.py` 是冻结的微型 PET 病灶分割器，`ema.py` 提供推理权重指数滑动平均，`config_utils.py` 负责 YAML 配置加载/点路径覆盖/消融解析/PNG 基线校验。数据流：YAML 配置 → SLMFBBDM.from_config → batch(ct,pet,mask,organ,meta…) → priors 产 ConditionBundle → adapter 拼通道 → noise_schedule 加噪 → UNet 预测 x0 → LossTerm 栈汇总 → Trainer 反传/EMA/检查点。

---

### `src/model/__init__.py`

**职责**：模型包入口，声明包级 docstring 并 re-export 最常用的四个公共符号，使外部可直接 `from src.model import SLMFBBDM`。

**接口**：
- `from .slmf_bbdm import SLMFBBDM` — 主模型类
- `from .bbdm_unet import BBDMUNet` — 去噪 UNet 骨干
- `from .interfaces import ConditionBundle, LossContext` — 条件/损失数据契约
- `from .config_utils import load_full_config, save_resolved_config, resolve_runtime_profile` — 配置三入口

**依赖**：
- 内部: src.model.slmf_bbdm、src.model.bbdm_unet、src.model.interfaces、src.model.config_utils
- 外部: 无

**输入**：无运行时输入（纯导出层）。

**输出**：无；仅定义包公共 API 面。

**处理过程**：
1. 声明模块 docstring（SLMF-BBDM 全称）。
2. 依次导入主模型、UNet、接口 dataclass、配置工具并绑定到包命名空间。

---

### `src/model/bbdm_unet.py`

**职责**：BBDM 去噪骨干：残差 U-Net + 瓶颈交叉注意力。编码器 4 级、解码器对称 4 级，输出层在 heteroscedastic 模式下输出双通道（均值 + 对数方差）。

**接口**：
- `class _TimeEmbedding(dim)` — 正弦时间嵌入 + 2 层 MLP，`forward(t:[B]) -> [B,dim]`
- `class _ResBlock(in_ch, out_ch, time_dim, dropout)` — FiLM 风格时间条件残差块，`forward(x:[B,C,H,W], t_emb:[B,dim]) -> [B,out_ch,H,W]`
- `class _CrossAttention(query_dim, kv_dim, num_heads=4)` — 基于 `F.scaled_dot_product_attention` 的 Flash 交叉注意力，残差相加
- `def _zero_conv(in_ch, out_ch)` — 零初始化 1×1 卷积（供外部 ControlNet 式注入复用）
- `class BBDMUNet(in_channels=2, base_channels=64, channel_mult=(1,2,4,4), num_res_blocks=2, time_dim=256, dropout=0, enable_heteroscedastic=True, ca_kv_dim=64, ca_num_heads=4, meta_dim=0)`
  - `forward(x:[B,C,H,W], timesteps:[B], context_tokens=None:[B,N,kv], skip_injections=None:List[Tensor], meta=None:[B,meta_dim], ca_beta=None:[B,1]) -> [B,1|2,H,W]`

**依赖**：
- 内部: 无（被 src/model/slmf_bbdm.py、src/model/wavelet_unet.py 复用）
- 外部: torch、torch.nn、torch.nn.functional、math

**输入**：noisy_x 与条件拼接后的张量 `x`（通道数 = 2 或 3，自条件开启时 +1）、扩散时间步、可选语义 token 上下文、ControlNet 跳跃注入列表、可选 FiLM 元数据向量。

**输出**：形状 [B,1,H,W] 的 ε/x0 预测；heteroscedastic 时 [B,2,H,W]（第 2 通道为 logvar）。

**处理过程**：
1. `_TimeEmbedding` 把时间步编为 sin/cos 嵌入并过 MLP；有 meta 时叠加 `meta_proj(meta)`。
2. head 3×3 卷积把输入升到 base 通道。
3. 编码器逐级：num_res_blocks 个 _ResBlock → 记入 skips → stride-2 卷积下采样（最后一级不下采样）。
4. 瓶颈：2× _ResBlock 后，若给 context_tokens 则过 _CrossAttention；ca_beta 给出时按 `h + β·(h_ca − h)` 门控混合。
5. 解码器逐级：bilinear 2× 上采样 → 3×3 卷积调通道 → pop skip（可选 zero-conv 注入先加到 skip 上，尺寸不符时先插值）→ concat → _ResBlock 序列。
6. 输出 GroupNorm → SiLU → 1×1 卷积，通道数 1（或 heteroscedastic 为 2）。

---

### `src/model/config_utils.py`

**职责**：配置加载/点路径（dotlist）覆盖/消融预设解析/PNG 基线启动校验，是所有训练与评估脚本读取 YAML 的统一入口。

**接口**：
- `_load_yaml(path) -> dict`；`_normalize_scalar_strings(value) -> Any`（递归把 "1e-4" 等数字串转标量）
- `apply_dotlist_overrides(config, overrides) -> dict` — 按 `modules.gabor.enabled` 点路径写嵌套键，深拷贝返回
- `resolve_ablation_overrides(ablation_config_path, ablation_name) -> dict` — 从 ablations.yaml 取 `ablations.<name>.overrides`
- `load_full_config(config_path, ablation=None, ablation_config_path='configs/experiments/ablations.yaml', overrides=None) -> dict` — 主入口：YAML → 消融覆盖 → CLI key=value 覆盖
- `save_resolved_config(cfg, out_dir)` — 写 `resolved_config.yaml`
- `resolve_runtime_profile(cfg) -> dict` — 无 CUDA 时把 image_size 压到 ≤32、batch=1、eval 步数=2
- `class PNGBaselineConfigError(ValueError)`；`validate_png_baseline_config(config) -> dict`（data.mode=png 时硬性禁用 roi_suv/organ_prior/organ_consistency/metadata/无 ckpt 的 segmenter，返回启动状态字典）；`log_startup_status(status)` — 美化打印状态

**依赖**：
- 内部: 无（被 src/model/__init__.py 导出；训练与评估脚本消费）
- 外部: yaml、os、re、copy、pathlib、typing；resolve_runtime_profile 内惰性 import torch

**输入**：YAML 配置路径（如 configs/experiments/slmf_full.yaml）、消融名、`key=value` 覆盖列表、包含 data/modules/losses/model/segmenter/evaluation 节的配置字典。

**输出**：解析后的完整配置 dict；写出的 `resolved_config.yaml`；PNG 校验的状态 dict 或 PNGBaselineConfigError。

**处理过程**：
1. `_load_yaml` 读 YAML 并递归规范化数字字符串。
2. 若指定 ablation，从 ablations.yaml 读取 overrides 并以点路径合并。
3. 逐条解析 CLI `key=value`（value 先 yaml.safe_load 再规范化）合并。
4. resolve_runtime_profile 在 CPU 环境套用烟雾测试档参数。
5. validate_png_baseline_config 汇总 data_mode/organ_prior/roi_suv/metadata/segmenter/gabor 路由等状态；data.mode=png 时收集违规项并抛 PNGBaselineConfigError。
6. log_startup_status 打印 60 字符宽的启动横幅供日志留档。

---

### `src/model/ema.py`

**职责**：模型可训练参数的指数滑动平均（EMA），为推理提供更稳定的权重；支持按优化步间隔更新与 apply/restore 权重换入换出。

**接口**：
- `class EMA(model, decay=0.999, update_every=10)`
  - `update()` — 计步，每 update_every 步才真正滑动
  - `apply()` / `restore()` — 把 shadow 权重拷入模型 / 恢复训练权重
  - `state_dict() -> dict` / `load_state_dict(state)` — 序列化 {shadow, step_count, decay}

**依赖**：
- 内部: 无（由 src/model/trainer.py 消费）
- 外部: torch、torch.nn

**输入**：任意 nn.Module（构造时克隆所有 requires_grad 参数为 shadow）。

**输出**：无返回值；副作用是修改模型参数数据与内部 shadow 字典。

**处理过程**：
1. 构造时 `_register` 克隆全部可训练参数为 shadow。
2. `update()` 每 update_every 个优化步执行 `shadow = decay·shadow + (1−decay)·param`。
3. `apply()` 备份当前训练权重并把 shadow 拷入模型（供验证/采样）。
4. `restore()` 从备份恢复训练权重继续训练。
5. state_dict/load_state_dict 随检查点保存与恢复 EMA 状态。

---

### `src/model/interfaces.py`

**职责**：全部可插拔模块的统一接口约定：ConditionBundle/LossContext 两个数据容器 + PriorModule/ConditionAdapter/NoiseSchedule/LossTerm 四个抽象基类。禁用模块返回 NoOp 等价输出，使主训练循环无消融分支。

**接口**：
- `@dataclass ConditionBundle` — 三类条件统一容器：`maps`（空间图 → ControlNet/Zero-Conv 跳层注入）、`tokens`（1D 语义 token → 瓶颈交叉注意力）、`scalars`（全局标量 → FiLM/AdaGN），另含 `logs`；方法 `get_map/get_token/merge/copy`、`empty()`
- `@dataclass LossContext` — LossTerm 计算所需全部上下文：model_pred、loss_target、target_pet、pred_x0、timesteps、tau、batch、condition、可选 pred_logvar/pred_residual/target_residual/mean_pet
- `class PriorModule(enabled=True)` — `forward(batch, timesteps, partial_bundle=None) -> ConditionBundle`；`partial_bundle` 允许先运行的先验（如 Gabor）向后续先验（如 Hotspot）传递特征
- `class ConditionAdapter(enabled=True)` — `forward(noisy_x, raw_condition, condition, timesteps) -> Tensor`（UNet 输入通道）
- `class NoiseSchedule(enabled=True)` — `add_noise(x0, noise, timesteps, condition) -> Tensor`
- `class LossTerm(enabled=True, weight=1.0)` — `name/weight` 类属性；`forward(ctx) -> (scalar_loss, log_dict)`

**依赖**：
- 内部: 无（被 src/model/slmf_bbdm.py 及 src/model/conditioning、priors、noise、losses 子包广泛实现/消费）
- 外部: torch、torch.nn、dataclasses、typing

**输入**：实现类的具体输入由子类决定；契约层面为 batch 张量字典、[B] 时间步、ConditionBundle。

**输出**：ConditionBundle / UNet 输入张量 / 加噪张量 / (损失标量, 日志字典)。

**处理过程**：
1. ConditionBundle 用三个 dict 分通道存放条件，merge 支持跨先验增量合并，copy 提供浅拷贝防串改。
2. LossContext 一次性打包损失项可能用到的全部张量（含残差桥与均值分支的可选项）。
3. 四个基类均带 enabled 标志，子类在 enabled=False 时应返回空/恒等输出，保证消融无分支。
4. 基类 forward 均 raise NotImplementedError，作为纯接口契约。

---

### `src/model/mean_predictor.py`

**职责**：CT 条件的低频 PET 均值预测器：小卷积编码器只预测 PET 的 Haar 二级低频系数 LL2，再以零细节带重构出 mean_pet；架构上禁止该分支直接产出后续交给残差桥的高频细节。

**接口**：
- `class LowFrequencyPETPredictor(in_channels=1, base_channels=32, levels=2)`（levels 固定为 2，base_channels≥4）
  - `forward(ct:[B,C,H,W]) -> {'ll2':[B,1,H/4,W/4], 'mean_pet':[B,1,H,W]}`

**依赖**：
- 内部: src.model.frequency.haar.reconstruct_lowpass
- 外部: torch、torch.nn

**输入**：CT 图像张量 [B,1,H,W]，H/W 必须被 4 整除，否则 ValueError。

**输出**：字典：LL2 系数（训练时配 Charbonnier 损失）与全分辨率低频 PET 端点（扩散桥的 x_source 均值端）。

**处理过程**：
1. 校验 levels==2 与 base_channels≥4。
2. 编码器：3×3 卷积 → 两次 stride-2 卷积逐级降采样 → 1×1 卷积压到单通道得到 LL2。
3. 校验输入空间尺寸被 4 整除。
4. 用 haar.reconstruct_lowpass(ll2, levels=2) 上采样两级（细节带置零）得 mean_pet。
5. 返回 {'ll2', 'mean_pet'} 供 SLMFBBDM 条件均值分支与 MeanPretrainer 共用。

---

### `src/model/mean_pretraining.py`

**职责**：LowFrequencyPETPredictor 的独立预训练工具：Charbonnier LL2 损失（可选病灶区域排除）、带数据血缘校验的检查点保存与防篡改指纹 sidecar。

**接口**：
- `_CHECKPOINT_LINEAGE_FINGERPRINT_FIELDS` — 6 个血缘指纹字段名常量
- `write_checkpoint_fingerprint(checkpoint_path, checkpoint, data_lineage) -> dict` — 写 `.fingerprint.json` sidecar（含 .pt 自身 SHA-256、病灶排除策略、cache 指纹）
- `mean_target_ll2(pet:[B,1,H,W]) -> [B,1,H/4,W/4]` — 两级 Haar DWT 取 LL2 作为目标
- `mean_charbonnier_loss(pred_ll2, target_ll2, epsilon=1e-3) -> Tensor`
- `pathology_excluded_mean_loss(pred_ll2, target_ll2, mask, *, epsilon, guard_radius_px) -> Tensor` — mask 4×max-pool 到 LL2 分辨率并按半径膨胀后，仅背景像素平均（与 scripts/validate_h2_pathology_excluded_residual.py 对齐的 H2 排除）
- `class MeanPretrainer(predictor, train_loader, val_loader, config, device=None)` — val_loader 必填；方法 `train_epoch() -> float`、`validate() -> float`、`run(epochs, output_dir) -> list[dict]`

**依赖**：
- 内部: src.data.lineage（attach_data_lineage/load_checkpoint_data_lineage）、src.mechanism_validation.common.canonical_json_sha256、src.model.frequency.haar.haar_dwt2、src.model.mean_predictor.LowFrequencyPETPredictor
- 外部: torch、torch.nn.functional、torch.utils.data.DataLoader、hashlib、json、math、pathlib、typing

**输入**：batch 字典 {'ct','pet'(,'mask')}；配置键 modules.conditional_mean.*（charbonnier_eps/pathology_exclusion.enabled/guard_radius_px）、training.*（learning_rate/lr_min/weight_decay）、runtime.*（grad_clip_norm/amp）。

**输出**：`mean_last.pt`、`mean_best.pt`（含 format_version 1|2、model、mean_config、epoch、val_loss、data_lineage）、`mean_best.pt.fingerprint.json`、`history.json`（epoch/train_loss/val_loss/learning_rate）。

**处理过程**：
1. 构造时强制要求验证加载器，加载数据血缘、AdamW 优化器、AMP（bf16 优先）与 GradScaler。
2. `_batch_loss`：预测器前向取 ll2，对 pet 做两级 DWT 得目标 ll2；默认 Charbonnier，启用排除时校验 batch 有 mask 并走病理排除损失。
3. `train_epoch`：zero_grad → autocast 前向 → scaler 反传 → 梯度裁剪 → step/update，返回平均损失。
4. `validate`：no_grad 评估，非有限值直接 FloatingPointError。
5. `run`：CosineAnnealingLR 按 epoch 循环训练+验证，逐 epoch 覆写 mean_last.pt；刷新最优时保存 mean_best.pt 并写指纹 sidecar；每轮写 history.json 并打印进度行。

---

### `src/model/registry.py`

**职责**：零外部依赖的轻量注册表：每类可插拔模块（Prior/Adapter/NoiseSchedule/LossTerm）各一个 Registry 实例，按 name 注册、按含 `name` 键的配置字典实例化。

**接口**：
- `class RegistryError(Exception)`
- `class Registry(kind='module')`：
  - `register(name, cls)` — 重名抛 RegistryError
  - `build(cfg) -> Any` — 弹出 cfg['name'] 查表，其余键值作 kwargs 构造
  - `names() -> list`；`__contains__(name)`；`__repr__()`
- 全局实例：`prior_registry`、`adapter_registry`、`noise_registry`、`loss_registry`

**依赖**：
- 内部: 无（供 src/model/conditioning、priors、noise、losses 等子包注册条目）
- 外部: typing（Any/Callable/Dict）

**输入**：注册名 + 工厂（类/可调用对象）；build 时含 name 键的配置 dict。

**输出**：build 返回按配置实例化的模块对象；未知名抛出含可用名列表的 RegistryError。

**处理过程**：
1. register 查重重名后写入内部字典。
2. build 复制配置、弹出 name、查表，其余键透传构造器。
3. names/__contains__ 支持枚举与存在性检查，便于报错信息与调试。

---
### `src/model/slmf_bbdm.py`

**职责**：SLMF-BBDM 主模型（约 2300 行）：总装 CT 编码器、四类先验、Zero-Conv 适配器、噪声调度、UNet 骨干、条件均值/残差桥/残差频率注入、损失栈与元数据 FiLM，提供训练 forward、DDIM 式采样与 MC 不确定性采样。

**接口**：
- 模块级工具：`_meta_to_tensor(meta_batch, B, device, dtype) -> [B,5]`（5 个 DICOM 元数据键归一化到 [-1,1]）；`resolve_noise_config(modules_cfg) -> dict`；`_add_noise(schedule, x0, noise, timesteps, condition, x_source=None)`（按签名自适应）；`_get_alpha_cumprod(schedule, timesteps)`（DDPM/ScaleAdaptive 兼容）；`_bbdm_ddim_step(schedule, x_t, pred_x0, x_source, timesteps, next_timesteps)`（保留推断噪声轨迹的确定性布朗桥步）；`_image_gradient_l1_per_sample(pred, target) -> [B]`
- `class _CTEncoder(in_channels=1, base_ch=64, out_channels=(64,128,256,256))` — 3/7 双卷积多尺度 stem + 3 级下采样，输出 [c1,c2,c3,c4] 四尺度特征
- `class SLMFBBDM(image_size=192, objective='pred_x0', initialization_seed=None, enable_heteroscedastic=True, heteroscedastic_logvar_min=-6, heteroscedastic_logvar_max=2, ct_encoder_config, prior_configs, adapter_config, noise_config, loss_configs, condition_dropout_config, base_loss_config, self_conditioning_config, conditional_mean_config, residual_bridge_config, residual_frequency_config, wavelet_unet_config, meta_config, segmenter_config, expected_data_lineage, require_checkpoint_lineage, sample_scheduler='ddim', eval_sampling_steps=20)`
  - `forward(batch, timesteps=None) -> (total_loss, logs)` — 训练一步
  - `sample(batch, num_steps=None, progress=False, cfg_scale=1.0, initial_noise=None) -> {'synthetic_pet', ('mean_pet','pred_residual'), ('logvar'), ('hotspot_prior')}` — DDIM 式采样 + 弱 CFG（建议 ≤1.5 防幻觉病灶）
  - `sample_mc(batch, n_samples=20, num_steps=None, progress=False)` — 返回 synthetic_pet/epistemic_var/aleatoric_logvar/confidence_map/samples
  - `from_config(config) -> SLMFBBDM`（classmethod）；`set_training_epoch(epoch_index)`；`get_trainable_params()/get_total_params()`；`train(mode)`（冻结模块保持 eval）
  - 内部构建器：`_build_prior(name, cfg)`（gabor/organ_prior/hotspot_prior/semantic_prior→NoOp）、`_build_noise(cfg)`（ddpm/bbdm_bridge/scale_adaptive）、`_build_loss(name, cfg)`（约 20 种损失项）、`build_condition_bundle(batch, timesteps)`、`_build_skip_injections(...)`、`_build_adapter_injections(...)`、`_build_frequency_injections(...)`、`_base_reconstruction_loss(...)`、`_min_snr_weight(...)`、`_model_input(...)`、`_loss_epoch_scale(name)`

**依赖**：
- 内部: src.model.interfaces（ConditionBundle/LossContext/PriorModule 等类型）、src.model.bbdm_unet.BBDMUNet、src.model.conditioning.adapter（ZeroConvAdapter/RawConcatAdapter）、src.model.conditioning.beta_schedule.multi_level_betas、src.model.conditioning.dropout.ConditionDropout；惰性导入 src.model.priors.{noop,gabor,organ,hotspot,semantic}、src.model.noise.{base,scale_adaptive}、src.model.loss_terms.*（topk/normalized_lesion_peak/frequency/residual_frequency/spectral_router/boundary_frequency/frequency_gate_tv/roi_suv/false_hotspot/lesion_roi/nonlesion_body_lowpass/route_utility_supervision/heteroscedastic/hotspot/organ_consistency/segmenter_consistency/patch_nce/perceptual_x0/base.DisabledLossTerm）、src.model.mean_predictor、src.model.segmenter.TinySegmenter、src.model.wavelet_unet.WaveletBBDMUNet、src.model.frequency.{haar,residual_preconditioner,boundary_reliable,spectral_router}、src.data.lineage（validate/load_checkpoint_data_lineage）
- 外部: torch、torch.nn、pathlib、typing、inspect（签名探测）、tqdm（可选进度条）、warnings

**输入**：训练 batch 字典 {'ct':[B,1,H,W], 'pet':[B,1,H,W], 'mask'?, 'organ_mask'?, 'semantic_tokens'?, 'meta'?, 'router_confidence'?}；顶层 YAML 节 model./modules./losses./data./runtime.（from_config 读取 image_size、objective、modules.conditional_mean.checkpoint(format)、modules.residual_bridge、modules.residual_frequency.mode∈{legacy,boundary_reliable,spectral_evidence_router}、losses.<name>.epoch_warmup/epoch_window 等）。

**输出**：forward 返回 (total_loss 标量, logs 字典——含 loss/*、module/* 开关量、frequency/* 路由诊断)；sample 返回 synthetic_pet 等张量字典；无直接文件写出。

**处理过程**：
1. **构造期**：解析 base_loss（mse/l1/gradient 权重、min_snr 开关）、自条件、Gabor 五条独立路由开关（enabled/inject_adapter/use_for_noise/use_for_hotspot/use_for_loss）；按 prior_configs 逐个构建先验（禁用→NoOpPrior）；适配器 enabled→ZeroConvAdapter 否则 RawConcatAdapter；构建噪声调度；校验条件均值/残差桥/残差频率的启依赖链（residual_bridge 需 conditional_mean+bbdm_bridge；residual_frequency 需 residual_bridge；路由监督损失需 spatial_destination 等）；加载均值检查点（mean/full_model 两种格式 + 血缘校验）并按需冻结；按 mode 构建残差频率预条件器（legacy ResidualFrequencyPreconditioner / boundary_reliable BoundaryReliableFrequencyInjector / spectral_evidence_router SpectralEvidenceFrequencyRouter，后者携带 H3 调度、warmup/ramp、空间路由等大量旋钮）；可选冻结段器（加载 ckpt 后 requires_grad=False+eval）；initialization_seed 非空时用 fork_rng 隔离 RNG 构建 UNet 保证消融变体初始化配对；最后构建损失栈（解析 epoch_warmup/epoch_window 课程窗口）与 ConditionDropout。
2. **条件组装**：build_condition_bundle 先放 ct 与 ct_feat_0..3，再按序运行各先验并以 partial_bundle 串联（前序先验特征可供后续先验使用），merge 成单一 ConditionBundle。
3. **残差桥改写目标**：residual_bridge 开启时模型目标改为 x0−mean_pet、桥源端点置零（detach_bridge 可控），采样时再加回 mean_pet。
4. **加噪与丢弃**：_add_noise 按调度签名兼容传 x_source；ConditionDropout 在适配器读取条件**之前**施加（训练期条件鲁棒性）。
5. **注入与 UNet 前向**：skip_injections = Zero-Conv 适配器注入（CT/organ/gabor/hotspot 特征 + 时变 tau）+ 残差频率注入（三种 mode 各自调用，逐级相加，级数不匹配报错）；legacy 无小波注入时先用 modulate_residual 调制去噪状态；meta→FiLM 向量；ca_beta 用 multi_level_betas 取瓶颈列做时变交叉注意力门；自条件以概率 p 先做一次 no_grad UNet 前向取 pred 作第三输入通道。
6. **损失组装**：heteroscedastic 拆分 pred/logvar 并 clamp；残差桥下 pred_x0 = mean_pet + pred_model；Gabor use_for_loss 时把 pred/target 的 Gabor 描述子写回 condition.maps 供 GaborConsistencyLoss；LossContext 打包全部上下文；总损失 = 基础重构损失（τ 分段加权 MSE/L1/梯度 + min-SNR 归一化）+ 均值低频 Charbonnier（残差桥时）+ Σ epoch_scale·term.weight·loss（epoch 课程为 0 时跳过但仍记 enabled 日志）；最后附 module/* 与 frequency/* 诊断日志。
7. **采样**：x_T 初始化按调度分三路（BBDM：x_T=m_T·source+(1−m_T)·0+σ_T·ε；自定义反向 step_from_prediction；标准 x_T=噪声）；DDIM 时间步 linspace T−1→0，逐步重建条件、注入、UNet 前向；cfg_scale>1 时用全零条件做弱 CFG 混合；反向步按调度三选一（step_from_prediction / _bbdm_ddim_step 保留推断 ε / 标准 DDIM）；末步直接返回 pred_x0（而非仍含噪的 x_t）；残差桥时 synthetic_pet = mean_pet + 最终残差，并回传 logvar（与 pred_x0 同源）。
8. **MC 采样**：sample_mc 跑 n 次独立 sample，聚合均值、跨样本认知方差、平均 aleatoric logvar，合成总方差归一化得 confidence_map。

---
### `src/model/trainer.py`

**职责**：SLMF-BBDM 性能优化训练循环（约 2200 行）：AMP/torch.compile/channels_last/TF32/fused AdamW 加速，梯度累积、EMA、谱路由分阶段冻结训练、固定追踪样本验证、多标签最优检查点与早停、防篡改的 epoch 级 metrics JSONL 账本与精确 RNG 断点续训。

**接口**：
- 模块级函数：`_strict_repo_relative_path(value, *, label) -> str`（仓库相对 POSIX 路径硬校验）；`_validate_prior_anchored_resume_identity(*, current_config, checkpoint_config, checkpoint_path)`（prior-anchored 100E 管线恢复身份 fail-closed 校验）；`resolve_checkpoint_dir(config) -> str`（training.checkpoint_dir 或 checkpoints/<experiment.name>）；`resolve_sample_dir(config) -> str`；`_atomic_torch_save(payload, path)`（tmp+os.replace 原子写）；`_stripe_score(pred_np) -> float`（8 方向 Sobel 梯度各向异性，检测条带伪影）；`_to_unit_interval(array)`；`_compute_body_background_sample_metrics(pred, target, ct, lesion_mask, *, ...)`（非病灶身体区低通 MAE/偏差/高通能量比/分位数偏差）；`_compute_pet_sample_metrics(pred, target, mask, topk_percent, min_k, max_k, area_quantiles)`（病灶峰值/topK 峰值/质心距离/内外峰值比/failure 等指标）；`_small_lesion_variants(...)`；`_stratified_indices(total, count)`；`_to_device/_chunk_batch/_cat_sampled/_sampled_output_to_cpu`（分块采样工具）；`_detect_dtype()`；`_fused_adamw(params, lr, wd)`；`_spectral_parameter_group(name) -> {'base','projection','router','descriptor'}`；`_build_optimizer(model, training_cfg) -> AdamW`（可选分组学习率 + no-decay 切分）
- `class Trainer(model: SLMFBBDM, config: dict, train_loader: DataLoader, val_loader=None, device=None)`：
  - `run(num_epochs=None)` — 主循环；`train_epoch() -> Dict[str,float]`；`train_step(batch)`；`eval_step(batch)`；`ema_scope()`（上下文管理器）；`save_checkpoint(tag=None)`；`load_checkpoint(path)`；`_sample_with_eval_seed(batch)`；`_compute_val_sample_metrics(batch, synth=None)`；`_save_best_checkpoints(metrics) -> bool`；`_check_early_stopping(improved) -> bool`；`_save_sample_grid(batch, synth_pet)`；`_append_epoch_metrics(...)`；`_truncate_metrics_jsonl_to_epoch()`；`_apply_spectral_training_phase()`；`_set_spectral_router_epoch()`；`_rng_state_dict()/_load_rng_state(checkpoint)`；`_monitoring_state_dict()/_load_monitoring_state(checkpoint)`

**依赖**：
- 内部: src.data.lineage（attach_data_lineage/load_checkpoint_data_lineage/validate_checkpoint_data_lineage）、src.model.slmf_bbdm.SLMFBBDM、src.model.ema.EMA
- 外部: torch、torch.utils.data.DataLoader、numpy、scipy.ndimage（条带/身体背景指标）、skimage.metrics（可选 SSIM）、matplotlib（可选样本网格）、json/math/os/random/re/time/copy/contextlib/pathlib/typing

**输入**：SLMFBBDM 实例、完整配置 dict（training.{num_epochs,learning_rate,lr_min,weight_decay,checkpoint_dir,resume_from,ema,optimizer_groups,spectral_training_phases}、runtime.{amp,channels_last,torch_compile,gradient_accumulate_every,grad_clip_norm,log_interval,eval_interval,eval_num_samples,eval_seed,eval_sample_batch_size,sample_interval,save_interval,best_checkpoint,early_stopping,tracked_sample_ids,metrics_jsonl,sample_dir,gradient_diagnostics}、experiment.{name,seed}、losses.nonlesion_body_lowpass.{sigma,body_threshold,lesion_exclusion_radius}）、训练/验证 DataLoader。

**输出**：编号检查点 `ckpt_epochNNNN.pt`（model/optimizer/scheduler/ema/epoch/step/config/monitoring/rng + 数据血缘）；最优检查点 `ckpt_best_{lesion,image,combined,background}.pt`（EMA 权重物化、可 lightweight）；每 epoch 追加的 `training_metrics.jsonl`（schema_version/epoch/step/phase/train/eval/validation，原子重写全文件）；`outputs/samples/<exp>/epoch_NNNN.png`（CT/Target/Pred/Mask 网格）；控制台训练报表。

**处理过程**：
1. **构造**：读取 runtime 加速项（AMP dtype 自动 bf16>fp16>fp32、channels_last、reduce-overhead compile、TF32）；`_build_optimizer` 按参数名把谱路由分支分成 projection/router/descriptor/base 四组（可选差异化 LR 与 bias/offset no-decay），只纳入 requires_grad 参数；解析 spectral_training_phases（阶段需连续无缝覆盖到 num_epochs）；建 EMA 与 CosineAnnealingLR；配置 best_checkpoint 四标签、background 约束与早停参数；确定 metrics_jsonl 路径。
2. **每 epoch**：`_apply_spectral_training_phase` 冻结/解冻分组并清梯度 → set_training_epoch 同步损失课程与路由器 epoch → sampler.set_epoch → 遍历 train_step（autocast 前向、loss/grad_accum 反传、凑满即梯度裁剪+step+EMA 更新）；epoch 末补一次 optimizer step、scheduler.step，聚合日志并附 perf/*（epoch 秒数、lr、GPU 峰值内存）与 phase/train_* 标志。
3. **验证与模型选择**：eval_interval 到点时在 ema_scope（EMA 权重）+ _eval_rng_scope（fork_rng+固定 eval_seed，不扰动训练 RNG）下跑 eval_step 与固定追踪批采样；`_select_tracked_batch` 惰性扫描 val_loader 按病灶面积分层（或 tracked_sample_ids 指定）选固定样本；`_compute_val_sample_metrics` 在 CPU 上算 MAE/MSE/PSNR/SSIM/条带分/病灶峰值族/背景族指标并以批次内 Q33/Q67 分层小病灶低估率；`_save_best_checkpoints` 按三类分数（lesion=−(topq+0.5peak+0.02centroid+0.5fail)、image=ssim−mae−0.1stripe 超额、combined=α 混合−stripe 罚）与背景约束刷新四类 best 检查点（分数更新先于任何写盘，保证快照一致）。
4. **采样留档**：sample_interval 到点用同一 EMA+固定种子对追踪批采样（`_chunk_batch` 按 eval_sample_batch_size 分块控显存，逐块 CPU 化再合并），`_save_sample_grid` 写 png（matplotlib 缺失则跳过）；此可选产物先于账本与检查点完成，避免"已完成运行被可选失败拖成 FAILED"。
5. **账本提交**：`_append_epoch_metrics` 严格校验现有 JSONL 是 1..epoch−1 的精确前缀（空行/坏 JSON/重复/乱序/超界均报错），tmp+fsync+os.replace 原子提交恰一条记录；顺序上先账本后检查点，使检查点失败时恢复可安全截断"有效未来尾部"。
6. **断点续训**：`load_checkpoint` CPU 加载（weights_only）→ prior-anchored 身份校验（管线 ID、resume_from 必须指向 checkpoint_dir 内 ckpt_epochNNNN.pt 且与实际加载路径一致、剔除 resume_from 后配置深比较须相等）→ 数据血缘校验 → 恢复 model/optimizer/scheduler/ema（shadow 按 live 参数设备重映射）→ epoch/step/monitoring → 在 RNG 恢复**之前**重建固定追踪批（扫描确定性）→ `_load_rng_state` 恢复 Python/NumPy(以张量编码避免 pickle 限制)/Torch CPU/CUDA/DataLoader 与 sampler 生成器（缺失即 fail-closed）→ 截断 metrics JSONL 至精确前缀。
7. **早停**：combined 分数连续 patience 个 epoch 无改善（且过 min_epochs）即停止主循环。

---
### `src/model/wavelet_unet.py`

**职责**：BBDM 去噪骨干的小波变体：用 Haar DWT（下采样）与 Haar IWT（上采样）+ 可学习跨子带混合替换 BBDMUNet 的 stride-2 卷积与双线性上采样，把尺度变换显式对齐到频率子带，配合残差频率注入模块服务小目标合成。

**接口**：
- `_group_count(channels, maximum=32) -> int` — 找能整除通道数的最大 GroupNorm 组数
- `_validate_transition(in_channels, out_channels, kernel_size)` — 正通道数 + 奇数核校验
- `class WaveletDownsample(in_channels, out_channels, mix_kernel_size=3)` — `forward(x) -> Tensor`：haar_dwt2 得 LL+3 细节带 → concat 4C 通道 → 1×1 Conv + GN + SiLU + k×k Conv 混合
- `class WaveletUpsample(in_channels, out_channels, mix_kernel_size=3)` — `forward(x) -> Tensor`：卷积扩到 4C 通道 → chunk 成 LL/LH/HL/HH → haar_idwt2 重构细尺度
- `class WaveletBBDMUNet(in_channels=2, base_channels=64, channel_mult=(1,2,4,4), num_res_blocks=2, time_dim=256, dropout=0, enable_heteroscedastic=True, ca_kv_dim=64, ca_num_heads=4, meta_dim=0, mix_kernel_size=3)`
  - `forward(x, timesteps, context_tokens=None, skip_injections=None, meta=None, ca_beta=None) -> [B,1|2,H,W]` — 签名与 BBDMUNet 完全一致，可直接替换

**依赖**：
- 内部: src.model.bbdm_unet（_CrossAttention/_ResBlock/_TimeEmbedding 复用）、src.model.frequency.haar（haar_dwt2/haar_idwt2）
- 外部: torch、torch.nn

**输入**：与 BBDMUNet 相同的模型输入张量（通道 2/3）、时间步、语义 token、skip 注入列表（长度必须等于解码器级数、且逐级形状必须与 skip 完全一致否则报错）、meta 向量、ca_beta；H/W 必须被 2^(级数−1)=8 整除。

**输出**：[B,1,H,W]（heteroscedastic 时 [B,2,H,W]）预测。

**处理过程**：
1. 构造校验：channel_mult 至少两级、num_res_blocks≥1、各级通道为正。
2. 编码器每级 _ResBlock×num_res_blocks 后以 WaveletDownsample 过渡到下一级（Haar 分解 + 学习混合）。
3. 瓶颈 2× _ResBlock + 可选 _CrossAttention（ca_beta 门控同 BBDMUNet）。
4. 解码器每级先 WaveletUpsample（预测四子带再 IWT 重构），pop skip 后严格校验注入形状并相加，concat 进 _ResBlock 序列。
5. GroupNorm → SiLU → 1×1 输出卷积（heteroscedastic 双通道）。

---
### 模块依赖小结

**本组向内依赖**（grep 查证）：根模块大量消费 src/model 的四个子包——conditioning（adapter/beta_schedule/dropout）、priors（gabor/organ/hotspot/semantic/noop）、noise（base/scale_adaptive）、loss_terms（约 20 个损失实现）与 frequency（haar/residual_preconditioner/boundary_reliable/spectral_router），均为 slmf_bbdm.py 构造期惰性导入；mean_predictor/mean_pretraining/wavelet_unet 依赖 src.model.frequency.haar；mean_pretraining 与 slmf_bbdm、trainer 还依赖 **src.data.lineage**（数据血缘附加/校验）与 **src.mechanism_validation.common.canonical_json_sha256**——模型层与数据治理层的强耦合是本仓库的重要特征。

**本组向外暴露**：`from src.model import SLMFBBDM/BBDMUNet/ConditionBundle/LossContext/load_full_config`；Trainer 经 `src.model.trainer` 路径被 scripts/train_v2.py（主训练入口）、scripts/evaluate.py、scripts/eval_action_axis_screen.py、scripts/eval_router_background_suite.py 及 src/mechanism_validation/feature_causality.py 消费；SLMFBBDM 与 config_utils 被 src/scripts/inspect_model.py、src/mechanism_validation/model_experiments.py 消费；mean_pretraining 被 scripts/pretrain_conditional_mean.py、scripts/validate_h2_pathology_excluded_residual.py、scripts/retrain_excluded_mean.ps1 消费；scripts/diag_router_grad.py、run_prior_anchored_router_100e.py、summarize_prior_anchored_run.py 使用 Trainer/router 诊断；segmenter 的 TinySegmenter 在 slmf_bbdm.py 内实例化（损失 segmenter_consistency 引用）。

**跨模块发现**：registry.py 的四个全局注册表（prior/adapter/noise/loss）目前**仅定义未接线**——SLMFBBDM 实际用 _build_prior/_build_noise/_build_loss 的 if/elif 分派构建模块，注册表机制是预留的迁移目标；各子包是否已 register 未在本组文件中体现，接线路径需查 src/model/conditioning、priors、noise、losses 子包确认。


---

## 3. 损失项 src/model/loss_terms（基础与病灶类）

### 模块概述

本组文件实现 SLMF-BBDM 训练所需的全部辅助损失项（loss terms）及其统一调度器。所有损失项均继承自 `src/model/interfaces.py` 中的 `LossTerm(nn.Module)` 基类（属性 `enabled`/`weight`，接口 `forward(ctx: LossContext) -> (scalar_loss, log_dict)`），禁用时返回稳定零值日志（NoOp 模式），使主训练循环无需条件分支即可做消融。`LossContext` 数据类统一携带 `model_pred`/`loss_target`/`target_pet`/`pred_x0`/`timesteps`/`tau`(=t/T)/`batch`/`condition`(ConditionBundle) 等字段，各损失项从 `ctx.batch` 取解剖掩码、从 `ctx.condition` 取先验模块产出的条件图。设计模式为"策略 + 注册表"：`__init__.py` 汇总导出，`trainer.py`（LossTrainer 聚合器）按配置实例化并加权求和。数据流：UNet 前向 → 主 trainer 组装 LossContext → 各 LossTerm 各自计算（多数带 τ 时间门控，即"去噪早/中/晚期分段激活"）→ LossTrainer 聚合为总损失并合并日志。

### `src/model/loss_terms/__init__.py`

**职责**：损失项包的注册门面（registry facade），把所有损失项类与 base 工具统一 re-export，供 trainer 与配置侧以 `src.model.loss_terms.XXXLoss` 形式导入。
**接口**：无顶层 class/def，纯 import 聚合。导出符号：`DisabledLossTerm`、`tau_gate`（来自 .base）；`TopKLesionLoss`；`FocalFrequencyLoss`；`ROISUVLoss`；`FalseHotspotLoss`；`HeteroscedasticNLLLoss`；`HotspotPriorLoss`；`PatchNCELoss`；`LesionROIL1Loss`、`OutsidePeakRankingLoss`；`BoundaryFrequencyLoss`；`FrequencyGateTVLoss`；`NormalizedLesionPeakLoss`；`SpectralRouterRegularizationLoss`；`LesionAwarePerceptualX0Loss`、`PETFeatureEncoder`。
**依赖**：
- 内部: src.model.loss_terms.base / .topk / .frequency / .roi_suv / .false_hotspot / .heteroscedastic / .hotspot / .patch_nce / .lesion_roi / .boundary_frequency / .frequency_gate_tv / .normalized_lesion_peak / .spectral_router / .perceptual_x0
- 外部: 无（纯包内聚合）

**输入**：无（模块导入期执行）。
**输出**：包级命名空间符号，无文件产出。
**处理过程**：
1. 从 `.base` 导入禁用态占位 `DisabledLossTerm` 与硬门控 `tau_gate`；
2. 依次导入 13 个具体损失项模块的全部公开类；
3. 注意：还导出了 `normalized_lesion_peak`、`spectral_router`、`perceptual_x0` 三个模块的类，这三个文件不在本部件清单内（属扩展组，另行成文）。

### `src/model/loss_terms/base.py`

**职责**：提供损失项公共工具——NoOp 占位损失 `DisabledLossTerm`（配置关闭某损失时返回稳定零日志）与 τ 时间门控函数 `tau_gate`（硬阈值）/ `smooth_tau_gate`（sigmoid 平滑）。
**接口**：
- `class DisabledLossTerm(LossTerm)`：`__init__(name="disabled", weight=1.0)`（基类以 `enabled=False` 初始化）；`forward(ctx) -> (0.0 标量, {name/enabled: 0, name/loss: 0})`，零张量建在 `ctx.target_pet.device` 上。
- `def tau_gate(tau: Tensor, max_tau=1.0, min_tau=0.0) -> Tensor`：元素级硬门控，`min_tau ≤ τ ≤ max_tau` 时为 1，否则 0。
- `def smooth_tau_gate(tau: Tensor, max_tau=0.25, sharpness=30.0) -> Tensor`：`sigmoid(sharpness*(max_tau-τ))`，τ 小（去噪晚期、细节阶段）→ 门趋近 1。

**依赖**：
- 内部: src.model.interfaces（LossContext、LossTerm）
- 外部: torch、torch.nn、typing

**输入**：`tau` 张量（τ = t/T ∈ [0,1]，形状同 timestep 广播，通常 [B] 或 [B,1,1,1]）。
**输出**：与输入同形状、取值 [0,1] 的门控张量；DisabledLossTerm 返回 (标量零, 日志字典)。
**处理过程**（smooth_tau_gate 被几乎所有时间门控损失复用）：
1. 调用方传入当前扩散时间比例 τ 与激活上限 `max_tau`；
2. 计算 `sigmoid(sharpness * (max_tau - tau))`，sharpness 默认 30 使过渡陡峭近似阶跃；
3. 返回门控张量，损失项用它逐样本加权后取 mean；
4. `DisabledLossTerm` 则作为统一占位符，保证被禁用损失在日志里仍有固定键名。
### `src/model/loss_terms/boundary_frequency.py`

**职责**：边界聚焦的 PET 频域监督——在病灶/器官边界环带与 CT-PET 边缘共识区上，对预测 x0 的梯度幅值与 Haar 小波高频细节施加加权 Charbonnier 损失，强化小病灶边缘保真。
**接口**：
- `class BoundaryFrequencyLoss(LossTerm)`（name="boundary_frequency"）：`__init__(lesion_weight=1.0, anatomy_weight=0.5, organ_weight=0.5, wavelet_weight=0.25, boundary_radius=2, epsilon=1e-3, active_tau_max=0.7, enabled=True, weight=0.05)`；`forward(ctx) -> (标量, 日志字典)`。
- 模块级辅助函数：`_gradient_magnitude(image, epsilon)`、`_edge_weight(image)`、`_boundary_ring(mask, radius)`、`_available_boundary(mask, reference, radius, collapse_channels)`、`_weighted_charbonnier(prediction, target, spatial_weight, epsilon)`。

**依赖**：
- 内部: src.model.frequency.haar（haar_dwt2）、src.model.interfaces（LossContext、LossTerm）、src.model.loss_terms.base（smooth_tau_gate）
- 外部: torch、torch.nn.functional、typing

**输入**：`ctx.pred_x0`（[B,C,H,W]，为 None 时整项返回零）；`ctx.target_pet`；`ctx.batch["ct"]`、`ctx.batch["mask"]`（病灶掩码）、`ctx.batch["organ_mask"]`（[B,6,H,W]，均可缺省）；`ctx.condition.scalars["frequency_noise_reliability"]`（可选，替代 τ 门控）；`ctx.tau`。
**输出**：加权总损失标量；日志键：`boundary_frequency/{enabled, available, lesion_available, organ_available, lesion, anatomy, organ, wavelet, gate_mean, loss}`。
**处理过程**：
1. `pred_x0` 缺失或未启用时返回零损失 + 零日志（available=0）；
2. 由掩码形态学（max_pool 膨胀 − 腐蚀）提取病灶边界环带 `lesion_ring` 与多通道合并的器官环带 `organ_ring`，掩码缺失时环带为零图并记录 available 标量；
3. 计算 CT 与 target PET 各自归一化梯度图之积作为解剖边缘共识权重 `anatomy_consensus`（无 ε，均匀区不造伪边）；
4. 对 pred 与 target 分别求梯度幅值，在三组空间权重下计算加权 Charbonnier 误差，得 lesion/anatomy/organ 三个分量；
5. 对 pred/target 各做 `haar_dwt2`，拼接三方向高频细节，边界并集区（max of 三权重）下采样后做加权 Charbonnier 得 wavelet 分量；
6. 按 4 个分量权重线性组合成 raw；门控取 `frequency_noise_reliability` 标量均值，缺省则 `smooth_tau_gate(tau, max_tau=active_tau_max)`（τ<0.7 晚期激活）；
7. 返回 `gated * self.weight`，并输出各分量与门控均值的 detach 日志。

### `src/model/loss_terms/false_hotspot.py`

**职责**：假热点抑制损失——在"应当冷"的组织（骨/脂肪/肌肉及非器官背景）内惩罚高预测 PET 值，同时放行膀胱/直肠的生理性摄取与子宫旁病灶区，减少分割前的假阳性热点。
**接口**：
- `class FalseHotspotLoss(LossTerm)`（name="false_hotspot"）：`__init__(active_tau_max=0.3, threshold_percentile=0.95, enabled=True, weight=0.05)`；`forward(ctx) -> (标量, 日志)`。
- 私有方法：`_build_cold_mask(ctx) -> Tensor`（构造冷区掩码）、`_region_masks(ctx) -> Dict[str, Tensor]`（分区域日志掩码）。

**依赖**：
- 内部: src.model.interfaces（LossContext、LossTerm）、src.model.loss_terms.base（smooth_tau_gate）
- 外部: torch、torch.nn、typing

**输入**：`ctx.pred_x0`（缺省回退 `ctx.model_pred`）；`ctx.target_pet`（仅取形状/设备）；`ctx.batch["mask"]`（病灶掩码，可选）；`ctx.batch["organ_mask"]`（[B,6,H,W]，类语义 0 子宫/盆腔、1 膀胱、2 直肠肠管、3 骨、4 脂肪、5 肌肉，可选）；`ctx.tau`。
**输出**：门控加权标量损失；日志键：`false_hotspot/{loss, enabled, bone_mean, fat_mean, muscle_cold_mean}`（无 organ_mask 时为 `other_background_mean`）。
**处理过程**：
1. 未启用返回 (0, {enabled:0})；计算 `smooth_tau_gate(tau, max_tau=0.3)`，仅中晚期（τ<0.3）激活；
2. `_build_cold_mask`：全 1 出发，减去病灶掩码，再减去 organ_mask 的 0/1/2 类（子宫、膀胱、直肠），剩余即"冷区"（骨 3/脂肪 4/肌肉 5/背景）；
3. 取 `pred * cold_mask` 在每样本内按 `threshold_percentile=0.95` 分位数定阈值；
4. `high_mask = (pred > q) 且 cold`，损失 = high 区 |pred| 均值（分母 clamp_min(1)）；
5. 乘以门控均值再乘 `self.weight`；
6. `_region_masks` 输出 bone/fat/muscle_cold（或 other_background）各区域平均预测强度用于监控。

### `src/model/loss_terms/frequency.py`

**职责**：Focal Frequency Loss（ICCV 2021）——在傅里叶域做聚焦加权的 L1，误差大的频率分量权重更高，直接对抗 PET 合成的模糊问题、显式激励病灶高频边缘。
**接口**：`class FocalFrequencyLoss(LossTerm)`（name="focal_frequency"）：`__init__(alpha=1.0, active_tau_max=0.4, enabled=True, weight=0.1)`；`forward(ctx) -> (标量, 日志)`。
**依赖**：
- 内部: src.model.interfaces（LossContext、LossTerm）、src.model.loss_terms.base（smooth_tau_gate）
- 外部: torch、torch.nn、typing

**输入**：`ctx.pred_x0`（缺省回退 `ctx.model_pred`）；`ctx.target_pet`（同形状 [B,C,H,W]，支持 bf16 输入、内部转 float32）；`ctx.tau`。
**输出**：加权标量；日志键：`focal_frequency/{loss, gate_mean, enabled}`。
**处理过程**：
1. 未启用返回零；计算 `smooth_tau_gate(tau, max_tau=0.4)` 晚期门控；
2. pred/target 转 float32 后各做 `torch.fft.rfft2(norm="ortho")`；
3. 频域误差 = 复谱差的模 `|F_pred - F_target|`（实部+虚部合并）；
4. 聚焦权重矩阵 = `freq_error.detach().pow(alpha)`（α=1 时即误差自身，梯度不回传到权重）；
5. 损失 = `(weight_matrix * freq_error).mean()`，乘门控均值与 `self.weight` 返回。
### `src/model/loss_terms/frequency_gate_tv.py`

**职责**：空间频率门控平滑正则——对条件模块写出的频率门控图总变差（TV）做弱约束，防止门控图空间上抖动；本类不自行计算 TV，只消费模型侧预算好的标量。
**接口**：`class FrequencyGateTVLoss(LossTerm)`（name="frequency_gate_tv"）：`__init__(enabled=True, weight=0.001)`；`forward(ctx) -> (标量, 日志)`。
**依赖**：
- 内部: src.model.interfaces（LossContext、LossTerm）
- 外部: torch、typing、__future__.annotations

**输入**：`ctx.condition.scalars["frequency_gate_tv"]`（条件/适配器模块预先计算的门控 TV 张量；缺失或未启用时返回零）；`ctx.target_pet`（仅用于建零张量设备/dtype）。
**输出**：`raw.mean() * weight` 标量；日志键：`frequency_gate_tv/{enabled, available, loss}`。
**处理过程**：
1. 在 target_pet 上建零标量；
2. 从 `condition.scalars` 取 `frequency_gate_tv`，键不存在或未启用 → 返回零 + available=0 日志；
3. 取 mean 得 raw，返回 `raw * self.weight` 与 detach 日志（available=1）。

### `src/model/loss_terms/heteroscedastic.py`

**职责**：异方差 NLL 损失——当模型同时输出均值与 log 方差时，用高斯负对数似然让模型自学"哪些区域本质上不确定"，从而在校准意义上自动下调不确定区域的 L1/L2 权重。
**接口**：`class HeteroscedasticNLLLoss(LossTerm)`（name="heteroscedastic_nll"）：`__init__(logvar_min=-6.0, logvar_max=2.0, enabled=True, weight=1.0)`；`forward(ctx) -> (标量, 日志)`。
**依赖**：
- 内部: src.model.interfaces（LossContext、LossTerm）
- 外部: torch、torch.nn、typing

**输入**：`ctx.pred_logvar`（[B,C,H,W]，与 pred 同形状；为 None 时整项静默为零）；`ctx.pred_x0`（缺省回退 `ctx.model_pred`）；`ctx.target_pet`。
**输出**：加权标量；日志键：`heteroscedastic_nll/{loss, logvar_mean, enabled}`。
**处理过程**：
1. 未启用或 `pred_logvar is None` → 返回 (0, {enabled: 相应 0/1})；
2. `logvar = ctx.pred_logvar.clamp(logvar_min, logvar_max)` 防止数值爆炸（-6..2）；
3. 精度 `precision = exp(-logvar)`，`mse = (pred - target)²`；
4. `loss = 0.5 * (precision * mse + logvar).mean()`（标准高斯 NLL）；
5. 乘 `self.weight` 返回，日志附 logvar 均值监控不确定度水平。

### `src/model/loss_terms/hotspot.py`

**职责**：热点先验监督损失——直接监督 CT→热点 tiny prior 模块输出的 `hotspot_prior` 概率图（ConditionBundle maps 中的条件图，而非 PET 预测本身），用 Dice + Focal + 可选距离图 L1 三项联合，且提供三种靶区构造模式避免把远处生理摄取拉进靶区。
**接口**：
- `class HotspotPriorLoss(LossTerm)`（name="hotspot_prior"）：`__init__(active_tau_min=0.25, active_tau_max=0.75, focal_gamma=2.0, dice_weight=1.0, focal_weight=1.0, distance_weight=0.25, pet_threshold_quantile=0.95, target_mode="mask_only", local_uptake_radius=8, enabled=True, weight=0.1)`；`forward(ctx) -> (标量, 日志)`；私有 `_target(ctx) -> Tensor`。
- 模块级函数：`_safe_prob`（clamp 1e-5..1-1e-5）、`_normalise_map`（min-max 归一化）、`_dilate_mask`（max_pool 膨胀）、`_get_distance_target`（按优先级找距离图）。

**依赖**：
- 内部: src.model.interfaces（LossContext、LossTerm）、src.model.loss_terms.base（smooth_tau_gate）
- 外部: torch、torch.nn.functional、typing

**输入**：`ctx.condition.get_map("hotspot_prior")`（[B,1,H,W] 概率图，缺失返回零）；`ctx.batch["mask"]`（病灶掩码）；`ctx.batch` 中可选键 `hotspot_distance`/`lesion_distance`/`distance_map`/`organ_distance`（[B,1,H,W]）；`ctx.target_pet`；`ctx.tau`。配置键 `target_mode` ∈ {mask_only, mask_plus_local_uptake, legacy}。
**输出**：门控加权标量；日志键：`hotspot_prior/{loss, dice, focal, distance, gate_mean, enabled}`。
**处理过程**：
1. 取 `hotspot_prior` 条件图，未启用或缺失 → 返回零；
2. 门控为"中段窗"：`gate = smooth_tau_gate(tau, max_tau=0.75) * (1 - smooth_tau_gate(tau, max_tau=0.25))`，只在 τ∈[0.25,0.75] 的中期激活；
3. `_target` 按模式构造监督靶：`mask_only` 仅病灶掩码（PNG 基线推荐）；`mask_plus_local_uptake` 在膨胀 8 px 的病灶 ROI 内取 PET top-5% 加入靶区（禁止全图分位数）；`legacy` 全图 top-q%（仅向后兼容）；
4. Dice：`1 - mean((2·∩+1)/(∪+1))`；
5. Focal：在 `torch.amp.autocast(enabled=False)` 下手算 BCE 与 pt（规避 CUDA autocast 下 BCE 不安全），`((1-pt)^γ · bce).mean()`，γ=2；
6. 若 batch 提供距离图（归一化后），追加 `F.l1_loss(pred, distance_target)`；
7. 三项按权重求和 → 乘门控均值 → 乘 `self.weight` 返回。
### `src/model/loss_terms/lesion_roi.py`

**职责**：密集病灶 ROI 损失对——`LesionROIL1Loss` 在膨胀病灶 ROI 内做逐像素重建监督；`OutsidePeakRankingLoss` 用排序 hinge 强制"病灶内最亮峰值高于病灶外峰值 + margin"，直击"热点长在病灶外、病灶内偏冷"的失败模式。
**接口**：
- `class LesionROIL1Loss(LossTerm)`（name="lesion_roi_l1"）：`__init__(dilate_radius=3, beta=0.05, active_tau_max=0.7, enabled=True, weight=1.0)`；`forward(ctx) -> (标量, 日志)`。
- `class OutsidePeakRankingLoss(LossTerm)`（name="outside_peak_ranking"）：`__init__(margin=0.05, inside_radius=3, outside_radius=8, topk_percent=0.01, active_tau_max=0.65, enabled=True, weight=0.1)`；私有 `_topk_mean(pred, mask) -> (values, valid)`；`forward(ctx) -> (标量, 日志)`。
- 模块级 `_dilate_mask(mask, radius)`：二值化后 max_pool 膨胀，保持 [B,1,H,W]。

**依赖**：
- 内部: src.model.interfaces（LossContext、LossTerm）、src.model.loss_terms.base（smooth_tau_gate）
- 外部: torch、torch.nn.functional、typing、__future__.annotations

**输入**（两者共用）：`ctx.batch["mask"]`（病灶掩码；为 None 或全零时返回零损失但仍记 enabled=1）；`ctx.pred_x0`（缺省回退 `ctx.model_pred`）；`ctx.target_pet`（仅 L1 用）；`ctx.tau`。
**输出**：加权标量。日志键：`lesion_roi_l1/{loss, gate_mean, roi_pixels, enabled}`；`outside_peak_ranking/{loss, inside_peak, outside_peak, gate_mean, enabled}`。
**处理过程**：
- LesionROIL1Loss：
  1. 掩码膨胀 `dilate_radius=3` 得 ROI；
  2. β>0 用 `F.smooth_l1_loss(beta=β)` 否则纯 L1，得逐像素 diff；
  3. 逐样本 ROI 均值 `(diff*roi).sum/roi.sum`（分母 clamp_min(1)）；
  4. 乘 `smooth_tau_gate(tau, max_tau=0.7)` 逐样本门控后 mean，再乘 weight。
- OutsidePeakRankingLoss：
  1. `inside = dilate(mask, 3)`，`outside = 1 - dilate(mask, 8)`；
  2. `_topk_mean` 逐样本在掩码内取前 1% 像素（至少 1 个）的均值作为峰值，空掩码记 invalid；
  3. hinge：`relu(outside_peak - inside_peak + margin)`；
  4. 仅对 inside/outside 均有效的样本求均值，乘 τ<0.65 门控与 weight。

### `src/model/loss_terms/organ_consistency.py`

**职责**：器官一致性损失——按器官类别对预测 PET 施加"值合理性"约束：在冷器官（骨/脂肪/肌肉）整区域上惩罚正向预测均值，对子宫区 >0.8 的极端值轻惩罚，膀胱/直肠完全放行；与 FalseHotspotLoss 的区别是连续均值级约束而非分位数离群检测，且在去噪早中期激活。
**接口**：`class OrganConsistencyLoss(LossTerm)`（name="organ_consistency"）：`__init__(active_tau_min=0.25, cold_weight=0.02, enabled=True, weight=0.1)`；`forward(ctx) -> (标量, 日志)`。
**依赖**：
- 内部: src.model.interfaces（LossContext、LossTerm）
- 外部: torch、typing

**输入**：`ctx.batch["organ_mask"]`（[B,6,H,W]；缺失时返回零损失）；`ctx.pred_x0`（缺省回退 `ctx.model_pred`，PET 归一化域 [-1,1]）；`ctx.tau`。
**输出**：加权标量；日志键：`organ_consistency/{loss, gate_mean, enabled, bone_loss, fat_loss, muscle_loss, uterus_loss}`。
**处理过程**：
1. 未启用或无 organ_mask → 返回零；
2. 早段门控 `gate = sigmoid(10*(τ - 0.25))`（τ>0.25 早中期激活——器官级分布先于细节成形）；
3. 对冷类 {骨:3, 脂:4, 肌:5}：`relu(pred)` 在该器官掩码内的均值作为区域损失（只罚正值，负值不动），权重 bone/fat=1.0、muscle=0.5，统一乘 `cold_weight`；
4. 子宫（类 0）：`relu(pred - 0.8)` 区域均值乘 0.01 轻约束（邻近病灶，允许高摄取）；
5. 膀胱（1）/直肠（2）不进入损失；
6. 总和乘门控均值与 `self.weight`，输出分器官日志。
### `src/model/loss_terms/patch_nce.py`

**职责**：CT-PET 空间绑定的 2D PatchNCE 对比损失。模块 docstring 明确警告：原始图像级 PatchNCE 会奖励模型把 CT 高频纹理复制进 PET（"CT 解剖纹理泄漏"失败模式），PNG 基线必须保持 `losses.patch_nce.enabled: false`；如需对比绑定应改用独立冻结编码器的高级特征（不在本文件实现范围）。
**接口**：`class PatchNCELoss(LossTerm)`（name="patch_nce"）：`__init__(patch_size=3, num_patches=256, temperature=0.07, active_tau_max=0.6, enabled=True, weight=0.1)`；私有 `_patches(x) -> Tensor`（unfold 展开补丁并转置为 [B,N,K]）；`forward(ctx) -> (标量, 日志)`。
**依赖**：
- 内部: src.model.interfaces（LossContext、LossTerm）、src.model.loss_terms.base（smooth_tau_gate）
- 外部: torch、torch.nn.functional、typing

**输入**：`ctx.batch["ct"]`（必需键，缺失会 KeyError）；`ctx.pred_x0`（缺省回退 `ctx.model_pred`）；`ctx.tau`。
**输出**：加权标量；日志键：`patch_nce/{loss, gate_mean, enabled}`。
**处理过程**：
1. 未启用返回零；`smooth_tau_gate(tau, max_tau=0.6)` 中晚期门控；
2. CT 与 pred-PET 各按 3×3 补丁 unfold，L2 归一化得 [B,N,9] 描述子；
3. N 超过 `num_patches=256` 时随机下采样同一组补丁位置（保持正负样本对应）；
4. 逐样本构造 logits = `ct_patches[b] @ pet_patches[b]^T / temperature`，对角线为正样本；
5. `F.cross_entropy(logits, arange(N))` 逐样本求均值后跨 batch 平均，乘门控均值与 `self.weight`。

### `src/model/loss_terms/roi_suv.py`

**职责**：临床定量精度损失——把预测/靶 PET 从 [-1,1] 归一化域反解到物理 SUV 域，在病灶 ROI 内约束 SUVmax 与 SUVmean 绝对误差（TBR 仅计算用于日志、不进损失），无有效 SUV 标定（suv_ok=False）的样本被跳过而非近似。
**接口**：
- `class ROISUVLoss(LossTerm)`（name="roi_suv"）：`__init__(active_tau_max=0.25, enabled=True, weight=0.2)`；`forward(ctx) -> (标量, 日志)`。
- 模块级函数：`_as_bool/_as_float`（张量/标量安全转换）、`_suv_valid_mask(meta_list, B, device)`（有效样本掩码）、`_denormalise_pet(pet_norm, meta_list)`（按 `pet_suv_max` 反归一化）、`_de_collate_meta(meta, batch_size)`（把 DataLoader collate 后的 dict-of-lists 还原为逐样本 dict 列表）、`_extract_meta_list(ctx)`。

**依赖**：
- 内部: src.model.interfaces（LossContext、LossTerm）、src.model.loss_terms.base（smooth_tau_gate）
- 外部: torch、torch.nn、typing

**输入**：`ctx.batch["meta"]`（逐样本元数据，含 `pet_suv_max`/`suv_ok`；需反 collate）；`ctx.batch["mask"]`（病灶掩码，缺失返回零）；`ctx.batch["pet_suv"]`（可选：CachedDataset 直接提供的未截断物理 SUV 靶）；`ctx.batch["organ_mask"]`（可选，用于背景定义）；`ctx.pred_x0`（缺省回退 `model_pred`）；`ctx.target_pet`；`ctx.tau`。
**输出**：加权标量；日志键：`roi_suv/{loss, suv_max_error, suv_mean_error, tbr_error, pred_suv_mean, target_suv_mean, valid_fraction, enabled}`。
**处理过程**：
1. τ<0.25 最晚期门控（广播到 [B]）；无病灶掩码直接返回零；
2. `_extract_meta_list` 反 collate meta，`_suv_valid_mask` 标出 suv_ok 且有 pet_suv_max 的样本；全部无效则返回全零日志；
3. `_denormalise_pet`：`suv = (pet_norm+1)/2 * pet_suv_max`（无效样本置零并被掩掉）；靶优先用 batch["pet_suv"] 原生 SUV 张量；
4. 病灶 ROI 内算 SUVmax / SUVmean 的逐样本绝对误差；
5. 背景 = 非病灶 ∧ 非任何器官区，算 TBR = SUVmean(病灶)/SUVmean(背景) 的误差——仅记录日志（源码注释：TBR 分母 clamp 到 1e-6 时会发散、曾占 ~99% 总损失迫使模型输出全零，故从损失中剔除）；
6. 逐样本损失 = suv_max + suv_mean，按 `suv_valid * gate` 加权、除以有效样本数，再乘 `self.weight`。

### `src/model/loss_terms/segmenter_consistency.py`

**职责**：分割器一致性损失（L_seg）——以冻结 TinySegmenter 为"质量判官"，对合成 PET 与真实 PET 各自过分割器，惩罚两者松弛距离变换输出（Stage 2，非硬掩码，提供远边界稠密梯度）的 L1 差，让"看起来像真病灶"的信号可反传到扩散模型。设计约束：仅当冻结分割器在验证集病灶召回 ≥0.70 时才启用（配置 `enabled` 人工控制）。
**接口**：`class SegmenterConsistencyLoss(LossTerm)`（name="segmenter_consistency"）：`__init__(segmenter=None, active_tau_max=0.3, enabled=True, weight=0.1)`；`set_segmenter(segmenter: nn.Module)`（延迟注入冻结分割器）；`forward(ctx) -> (标量, 日志)`。
**依赖**：
- 内部: src.model.interfaces（LossContext、LossTerm）、src.model.loss_terms.base（smooth_tau_gate）
- 外部: torch、torch.nn、torch.nn.functional、typing

**输入**：构造时或 `set_segmenter` 注入的冻结分割器 nn.Module（缺失返回零）；`ctx.pred_x0`（缺省回退 `model_pred`）；`ctx.target_pet`；`ctx.tau`。
**输出**：加权标量；日志键：`segmenter_consistency/{loss, gate_mean, enabled}`。
**处理过程**：
1. 未启用或无分割器 → 返回零；
2. `smooth_tau_gate(tau, max_tau=0.3)` 最晚期门控（分割器对粗糙中间态无意义）；
3. `torch.no_grad()` 下对真实 PET 前向得 seg_target（判官侧不建梯度）；
4. 对合成 PET 前向得 seg_pred（梯度穿过分割器反传到扩散模型）；
5. 逐样本 L1 均值，乘门控后 batch 平均，乘 `self.weight` 返回。

### `src/model/loss_terms/topk.py`

**职责**：Top-K Focal 病灶损失——对仅占 0.1% 像素的小病灶是"生存必需"项：逐样本在病灶掩码内从**靶** PET 选最亮 k 个像素（用靶 Top-K 而非 pred/target 各自独立 Top-K，以保持空间对应），对模型在同位置的预测施加焦点加权绝对误差。
**接口**：`class TopKLesionLoss(LossTerm)`（name="topk_lesion"）：`__init__(topk_percent=0.01, focal_gamma=2.0, active_tau_max=0.25, enabled=True, weight=0.2)`；`forward(ctx) -> (标量, 日志)`。
**依赖**：
- 内部: src.model.interfaces（LossContext、LossTerm）、src.model.loss_terms.base（smooth_tau_gate）
- 外部: torch、torch.nn、math、typing、__future__.annotations

**输入**：`ctx.pred_x0`（缺省回退 `model_pred`）；`ctx.target_pet`；`ctx.batch["mask"]`（病灶掩码；None 时返回零但仍记 enabled）；`ctx.tau`。
**输出**：加权标量；日志键：`topk_lesion/{loss, weighted_loss, gate_mean, k_mean, enabled}`。
**处理过程**：
1. τ<0.25 最晚期门控（reshape 成 [B] 逐样本）；
2. 逐样本取掩码内像素；空掩码样本跳过；全部为空则返回零（enabled=1）；
3. k = clamp(ceil(count×topk_percent), 3..16, count)——下限 3 保证极小病灶也有监督点；
4. `torch.topk(target_values, k)` 定位最亮位置，取同位置 pred 靶误差 |Δ|；
5. focal 权重 = `(1 - exp(-error))^γ`，误差越大权重越高；
6. 逐样本 `(focal_weight * error).mean() * gate[i]`，跨样本平均后乘 `self.weight`，日志附实际 k 均值。
### `src/model/loss_terms/trainer.py`

**职责**：SLMF-BBDM 的性能优化训练循环（Trainer：bf16 AMP、torch.compile、梯度累积、EMA、CosineAnnealing、批量日志、checkpoint 数据血缘）。**重要更正：它并不是损失聚合器**——损失聚合实际发生在 `src/model/slmf_bbdm.py` 的 `SLMFBBDM.forward` 内（见下）；本文件通过 `self.model(batch)` 直接拿聚合好的 `(loss, logs)`。此外本文件是 `src/model/trainer.py` 的**陈旧孤儿副本**：其相对导入 `from .slmf_bbdm import SLMFBBDM`、`from .ema import EMA` 在 loss_terms/ 包内无法解析（该目录下无这两个文件），import 即 ModuleNotFoundError；全仓库无任何模块引用它（0 处 import）。
**接口**：
- `class Trainer`：`__init__(model: SLMFBBDM, config: Dict, train_loader, val_loader=None, device=None)`；`train_step(batch) -> Optional[Dict[str, float]]`；`eval_step(batch) -> Dict[str, float]`（@no_grad）；`train_epoch() -> Dict[str, float]`；`run(num_epochs=None)`；`save_checkpoint(tag=None)`；`load_checkpoint(path)`；`ema_scope()`（上下文管理器，临时换 EMA 权重）；`set_segmenter` 无；私有 `_apply_optimizer_step`、`_save_sample_grid(batch, synth_pet)`。
- 模块级：`_detect_dtype() -> torch.dtype`（bf16>fp16>fp32）、`_fused_adamw(params, lr, wd)`。

**依赖**：
- 内部: src.data.lineage（attach_data_lineage / load_checkpoint_data_lineage / validate_checkpoint_data_lineage）；`.slmf_bbdm`（SLMFBBDM）、`.ema`（EMA）——**这两个相对导入在本包内断链**（真实文件在 src/model/ 下），故本文件实际不可运行
- 外部: torch、torch.utils.data.DataLoader、os、time、contextlib、typing、matplotlib（_save_sample_grid 内延迟导入）

**输入**：config 键 `runtime.{amp, channels_last, torch_compile, gradient_accumulate_every, grad_clip_norm, log_interval, eval_interval, sample_interval, save_interval}`、`training.{learning_rate, weight_decay, ema.{decay, update_every}, num_epochs, lr_min}`、`experiment.name`、`data.require_cache_lineage`；batch 字典（ct/pet/mask 等，由 DataLoader 提供）。
**输出**：checkpoint 写入 `checkpoints/<experiment.name>/ckpt_epochNNNN.pt`（含 model/optimizer/scheduler/ema/epoch/step/config + 数据血缘）；采样网格 PNG 写入 `outputs/samples/<experiment.name>/epoch_NNNN.png`（CT / Target PET / Pred PET / Lesion mask 四列，最多 4 行）；控制台 epoch/eval/采样日志。
**处理过程**：
1. 构造：设备选择 + TF32 + channels_last + 可选 torch.compile(reduce-overhead)，fused AdamW，EMA(decay=0.999)，CosineAnnealingLR；
2. `train_step`：batch 非阻塞搬设备 → CUDA autocast(bf16) 下 `loss, logs = self.model(batch)`（**模型内部完成 LossContext 组装 + 全部 loss_terms 加权求和**）→ `loss/grad_accum` 反传 → 累计满梯度即裁剪（norm=1.0）+ step + EMA.update；
3. 日志只在 `log_interval` 的整数倍步做 `.item()` 转浮点（含 `loss/total` 还原乘回 grad_accum）；
4. `train_epoch`：遍历 loader、冲刷残余梯度、scheduler.step()、汇总均值日志（perf/epoch_seconds、perf/lr、GPU 峰值内存）；
5. `run`：逐 epoch 训练；按 eval_interval 在 `ema_scope()` 下评估 val；按 save_interval 存 checkpoint（带数据血缘校验）；按 sample_interval 用 EMA 权重 `model.sample` 出合成 PET 并保存网格图；
6. 启动横幅打印启用的 priors 与 `loss_terms`（`{n for n, loss in self.model.loss_terms.items() if loss.enabled}`——这是本文件与损失项体系的唯一接触点）。
- **真实调度机制（补充说明，供对照）**：`src/model/slmf_bbdm.py` 构造时按 config `losses.*` 经 `_build_loss` 填充 `self.loss_terms = nn.ModuleDict()`（含 epoch_warmup 线性爬坡系数）；`forward` 第 1700-1758 行组装 `LossContext(model_pred, loss_target, target_pet=x0, pred_x0, timesteps, tau, batch, condition, pred_logvar, ...)$ 后：先加恒开的 base 扩散重建损失，再 `for name, term in self.loss_terms.items(): loss_val, loss_logs = term(ctx); total_loss += epoch_scale * loss_val`，日志统一冠以 `loss/` 前缀并附 `loss/{name}/epoch_scale`。各损失项返回的 `(标量, 日志字典)$ 即在此处被统一加权聚合。

### 模块依赖小结

**本组 import 的内部模块**：
- `src.model.interfaces`（LossTerm、LossContext、ConditionBundle）——全部 14 个损失文件的共同基座；
- `src.model.loss_terms.base`（smooth_tau_gate / DisabledLossTerm / tau_gate）——被 topk、frequency、roi_suv、false_hotspot、hotspot、patch_nce、lesion_roi、boundary_frequency、heteroscedastic(否)、segmenter_consistency、organ_consistency(否) 等复用（heteroscedastic 与 organ_consistency 未用门控函数，前者无 τ 门控、后者内联 sigmoid）；
- `src.model.frequency.haar`（haar_dwt2）——仅 boundary_frequency.py；
- `src.data.lineage` + `.slmf_bbdm` + `.ema`——仅 trainer.py（后两者在本包内断链，文件为孤儿副本）。

**被哪些内部模块/脚本消费**：
- **真正的调度方是 `src/model/slmf_bbdm.py`**：其 `_build_loss` 按配置 `losses.<name>.*` 直接从本包各子模块 import 具体类并实例化进 `nn.ModuleDict`（含 topk、frequency、roi_suv、false_hotspot、lesion_roi(L1+Ranking)、heteroscedastic、hotspot、organ_consistency、segmenter_consistency、patch_nce、boundary_frequency、frequency_gate_tv，以及 `base.DisabledLossTerm` 作为未知/禁用名的兜底）；本组清单外的同目录扩展项 residual_frequency、spectral_router、normalized_lesion_peak、nonlesion_body_lowpass、route_utility_supervision、perceptual_x0 也走同一通道（另文覆盖）。
- `src/model/trainer.py`（正式训练循环，非本组孤儿副本）通过 `self.model.loss_terms` 打印启用清单并消费聚合后的总损失；
- `src/model/loss_terms/__init__.py` 门面被 tests（test_smoke.py、test_png_baseline.py、test_boundary_frequency.py、test_perceptual_x0_loss.py 等）与 scripts（train_v2.py、evaluate.py、inspect_model.py、diag_router_grad.py、pretrain_pet_encoder.py）间接/直接使用；
- `src/model/loss_terms/trainer.py` 本身：全仓库 0 引用（孤儿文件，且相对导入断链不可运行）——建议后续清理时由维护者决策删除或恢复引用（本地图仅记录，不改码）。


---

## 4. 损失项 src/model/loss_terms（频谱路由与感知类）
### 模块概述

`src/model/loss_terms` 是 SLMF-BBDM 的可插拔损失项集合，本部件覆盖其中 6 个"频谱路由与感知类"损失项。所有项均继承 `src/model/interfaces.py` 的 `LossTerm`，遵守统一协议 `forward(ctx: LossContext) -> (标量损失, 日志字典)`，由 `src/model/slmf_bbdm.py` 的 `_build_loss` 按配置名实例化并注册进 `nn.ModuleDict self.loss_terms`，训练循环按 `name` 加权求和。设计上这些项都是"非图像域"的正则/监督：通过 `LossContext` 读取 `pred_x0`/`pred_residual`/`target_residual`/`batch['mask'|'ct']` 以及 `condition.maps`/`condition.scalars`（路由器写入的诊断量），除 perceptual_x0（均匀时间加权）与 spectral_router（无 τ 门控）外，多数项用 `base.smooth_tau_gate` 做桥时间 τ 的软门控——τ→0（去噪末期、细节成形阶段）门控开到 1。数据流：trainer → slmf_bbdm 前向（遍历 loss_terms）→ 各损失项 forward(ctx) → 返回 (loss×weight, 监控指标)。

### `src/model/loss_terms/normalized_lesion_peak.py`

- **职责**：面向 PNG 归一化 PET 目标的鲁棒病灶峰值监督——逐样本在病灶区内匹配 Top-Q 峰值，并用非对称"冷惩罚"（低估重罚、小高估容忍）抑制小病灶摄取被系统性低估。
- **接口**：`NormalizedLesionPeakLoss(topk_percent=0.10, min_k=3, max_k=16, beta=0.02, active_tau_max=0.25, cold_weight=1.0, cold_tolerance=0.02, enabled=True, weight=0.05)`，类属性 `name="normalized_lesion_peak"`；`forward(ctx: LossContext) -> (loss, logs)`，logs 含 `loss/pred_peak/target_peak/signed_bias/cold_penalty/gate_mean/valid_count/k_mean/enabled`。构造期对全部超参做范围校验（ValueError）。
- **依赖**：内部：`..interfaces`（LossContext/LossTerm）、`.base.smooth_tau_gate`；外部：`torch`、`torch.nn.functional`、`math`。
- **输入**：`ctx.pred_x0`（缺失时回退 `ctx.model_pred`）、`ctx.target_pet`、`ctx.batch['mask']`（病灶掩码，缺失按全零处理）、`ctx.tau`（每样本桥时间）。
- **输出**：标量损失（已乘 `weight`）+ 上述监控字典（全部 detach）。
- **处理过程**：
  1. 选定预测张量（优先 pred_x0），未启用时返回稳定零损失与零日志。
  2. 用 `smooth_tau_gate(tau, max_tau=active_tau_max)` 计算逐样本门控（默认 0.25，即只在去噪末期激活），并校验 τ 数量与 batch 一致。
  3. 逐样本取 `mask>0.5` 的像素集，为空则跳过该样本。
  4. 计算 k = clamp(ceil(像素数×topk_percent), min_k, max_k, 像素数)，对预测与目标各取 top-k 均值得到 pred_peak / target_peak。
  5. 对称项：`F.smooth_l1_loss(pred_peak, target_peak, beta)` 做常规峰值匹配。
  6. 非对称冷惩罚：`relu(target_peak - pred_peak - cold_tolerance)`，仅低估超过容差时生效，乘 cold_weight。
  7. 样本损失乘以 τ 门控后批内平均，再乘 `weight`；全部样本无效时返回零损失及完整零日志。
### `src/model/loss_terms/nonlesion_body_lowpass.py`

- **职责**：CT 身体内、病灶外区域的低频生理摄取场监督——针对"验证中位数已接近但高摄取分位数整体偏冷"的失效模式，在低通域做全区域重建 + 目标分位数以上软加权的误差惩罚；掩码/目标仅用于监督，不进入推理。
- **接口**：`NonLesionBodyLowpassLoss(sigma=4.0, body_threshold=0.03, body_closing_radius=2, lesion_exclusion_radius=8, tail_quantile=0.75, tail_weight=2.0, tail_temperature=0.02, charbonnier_eps=1e-3, active_tau_max=0.70, enabled=True, weight=0.2)`，`name="nonlesion_body_lowpass"`；高斯核以非持久 buffer `_gaussian_kernel` 注册（不影响 checkpoint 兼容）。`forward` 返回 loss 与 `lowpass_mae/tail_lowpass_mae/signed_bias/target_tail_threshold/gate_mean/body_nonlesion_pixels/valid_samples` 等日志。模块级辅助 `_dilate`（max_pool 膨胀）、`_binary_closing`（膨胀+腐蚀闭运算）。
- **依赖**：内部：`..interfaces`、`.base.smooth_tau_gate`；外部：`torch`、`torch.nn.functional`、`math`。
- **输入**：`ctx.target_pet`、`ctx.pred_x0`（或 model_pred）、`ctx.batch['ct']`（必需，缺失抛 ValueError）、`ctx.batch['mask']`（可选，用于病灶排除）、`ctx.tau`。输入约定为模型域 [-1,1]。
- **输出**：标量损失（×weight）+ 低频 MAE / 尾部 MAE / 符号偏差等监控字典。
- **处理过程**：
  1. 三者变换到单位区间 [0,1]：目标与 CT 做 clamp，预测不 clamp（越界预测仍能收到指向目标的梯度）。
  2. 用预置可分离高斯核（replicate padding、分组卷积）对预测与目标做模糊，得到低频场 pred_low / target_low。
  3. no_grad 下构造监督区域：CT 单位值 > body_threshold 做二值闭运算得身体掩码，病灶掩码按 lesion_exclusion_radius 膨胀后剔除，region = body×(1-excluded)。
  4. 逐像素 Charbonnier 误差（eps 平滑的 L1）。
  5. 逐样本计算目标低频场的 tail_quantile 分位数阈值，sigmoid 软尾部权重 (1 + tail_weight·tail_soft) 强调高摄取区。
  6. 加权误差求平均为逐样本损失；区域为空的样本记为无效并跳过；同时统计低频 MAE、硬尾部 MAE、符号偏差等日志量。
  7. 逐样本损失乘 τ 门控（max_tau=active_tau_max）后按有效样本数归一化，再乘 weight 返回。
### `src/model/loss_terms/perceptual_x0.py`（重点）

- **职责**：PFM（Perceptual Flow Matching）启发的 x0 感知损失——在**冻结**的 PET 病灶编码器多尺度特征空间里监督恢复的干净 PET `pred_x0`；目标分支 no_grad、预测分支保留梯度。fail-closed 策略：除 P4_FEAT_RANDOM 负对照外必须提供真实编码器 checkpoint，且可选 `_lineage` 血统校验（sha256、患者 split、召回率门槛），防止过期/跨患者编码器被静默使用。明确声明这不是 PFM 的忠实复刻，BBDM 前向/逆向方程不变。
- **接口**：
  - `PETFeatureEncoder(nn.Module)`：轻量多尺度编码器，`forward_features(x)` 返回 `{"full","half","quarter"}` 三尺度特征图，`forward` 返回分割 logits；`get_total_params()` 统计参数量。
  - `build_encoder(checkpoint=None, checkpoint_dir="checkpoints", encoder_kind="pretrained", in_channels=1, base_channels=16, feature_layers=...) -> (encoder, meta)`：加载冻结权重（含 sha256 计算与 sidecar lineage 读取 `_load_sidecar_lineage`）。
  - `build_feature_layer(...)`：构造特征层模块。
  - `LesionAwarePerceptualX0Loss(*, checkpoint=None, checkpoint_dir="checkpoints", encoder_kind="pretrained", feature_layers=("full","half","quarter"), layer_weights=None→(1.0,0.5,0.25), distance="charbonnier"|"l1", charbonnier_eps=1e-3, region_mode="global"|"lesion_balanced", lesion_weight=4.0, background_weight=1.0, dilate_radius=3, timestep_weighting="uniform", require_checkpoint_lineage=False, enabled=True, weight=1.0)`，`name="perceptual_x0"`。
  - 模块常量：`FEATURE_LAYER_NAMES`、`MIN_LESION_RECALL=0.70`、`MIN_SMALL_LESION_RECALL=0.70`。
- **依赖**：内部：`..interfaces`、`.base.smooth_tau_gate`（uniform 模式下实际未用到）；外部：`torch`、`torch.nn`、`hashlib`、`json`、`pathlib`。
- **输入**：`ctx.pred_x0`（形状必须等于 target_pet，否则 ValueError；缺失回退 model_pred）、`ctx.target_pet`、`ctx.batch['mask']`（lesion_balanced 模式用）、编码器 checkpoint 文件。
- **输出**：标量损失（×weight）+ 日志：`layer_{full|half|quarter}` 逐层损失、`region_mode`、`encoder_kind`、`encoder_total_params`/`encoder_trainable_params`（应为 0 证明冻结）等。
- **处理过程**：
  1. 构造期由 build_encoder 创建编码器并置于本损失模块名下（参数 requires_grad=False，优化器不可见）；`require_checkpoint_lineage=True` 时执行血统校验：checkpoint_sha256 与实际文件一致、train_patient_split 只含 train/calibration 患者、lesion_recall 与 small_lesion_recall 均 ≥0.70，任一缺失即 ValueError。
  2. forward 中目标分支在 no_grad 下、预测分支带梯度地分别过 `encoder.forward_features`。
  3. 每尺度特征先 `_stable_channel_normalize`（稳定通道归一化），再按 distance 计算 Charbonnier（eps 平滑）或 L1 距离图。
  4. global 模式（或无 mask）直接全图逐样本平均；lesion_balanced 模式把 [B,1,H,W] 掩码下采样（avg_pool+最近邻插值）到特征分辨率，膨胀半径按 `dilate_radius/scale_factor` 随尺度缩放，病灶区与背景区分别 `_region_mean`（空区域返回 0 防 NaN）后按 lesion_weight / background_weight 合成。
  5. 三尺度逐层损失乘 layer_weights 求和得逐样本损失。
  6. timestep_weighting="uniform"（v1 唯一支持）时直接批平均——契约规定每个时间步等权，不使用 τ 门控。
  7. 非有限损失直接抛 RuntimeError（fail-closed，不用零掩盖训练故障），最后乘 weight 返回。
### `src/model/loss_terms/residual_frequency.py`

- **职责**：定义在布朗桥残差与重建 PET 上的频域损失，包含两个独立损失项：`ResidualWaveletLoss`（两级 Haar 小波低/中/高频带匹配）与 `GaborConsistencyLoss`（Gabor 幅值 + 方向分布一致性），两者均做病灶区加权以强化小目标。
- **接口**：
  - 模块级 `_lesion_weight_map(mask, reference, lesion_weight)`（生成 1+lesion_weight·mask 权重图，mask 缺失时全 1）与 `_weighted_charbonnier(pred, target, mask, lesion_weight, epsilon)`（加权 Charbonnier 平均）。
  - `ResidualWaveletLoss(lesion_weight=4.0, band_weights=(0.5,1.0,1.5), epsilon=1e-3, active_tau_max=0.7, enabled=True, weight=0.05)`，`name="residual_wavelet"`；返回 low/mid/high/gate_mean/loss 日志。
  - `GaborConsistencyLoss(lesion_weight=3.0, orientation_weight=0.1, epsilon=1e-6, active_tau_max=0.7, enabled=True, weight=0.02)`，`name="gabor_consistency"`；依赖 `condition.maps` 中 gabor_pred_feat/gabor_target_feat/gabor_pred_orientation/gabor_target_orientation 四键，返回 amplitude/orientation_js 日志。
- **依赖**：内部：`..frequency.haar.haar_dwt2`、`..interfaces`、`.base.smooth_tau_gate`；外部：`torch`、`torch.nn.functional`。
- **输入**：ResidualWavelet 用 `ctx.pred_residual`/`ctx.target_residual`（任一缺失→available=0 零损失）、`ctx.batch['mask']`、`ctx.tau`；Gabor 用 `ctx.condition.maps` 的四个 gabor 特征图、`ctx.batch['mask']`、`ctx.tau`。
- **输出**：各自标量损失（×weight）+ 分频带/分项监控字典。
- **处理过程**：
  1. ResidualWavelet：可用性检查（残差缺失时优雅降级为 available=0 的零损失）。
  2. 对预测与目标残差各做两级 Haar DWT：LL2 为低频带，第二级细节拼接为中频带，第一级细节拼接为高频带。
  3. 三个频带分别计算病灶加权 Charbonnier（mask 分辨率不匹配时自适应 max_pool 下采样）。
  4. 按 band_weights=(低,中,高) 加权求和，乘 τ 门控（max_tau=0.7）与 weight。
  5. GaborConsistency：检查 maps 四键齐全，缺任一降级为零损失。
  6. 幅值项：pred/target Gabor 特征的加权 Charbonnier；方向项：两组方向响应 clamp+归一化为分布后计算逐像素 JS 散度图，再病灶加权平均。
  7. amplitude + orientation_weight×orientation_js，乘 τ 门控与 weight 返回。
### `src/model/loss_terms/route_utility_supervision.py`

- **职责**：层级频谱路由的"目的地效用"显式监督——端到端图像损失无法在双分支与 U-Net 共适应时辨别 native-vs-shallow 内部决策，本项给路由目的地赋予稳定语义：病灶集中的目标残差细节可上移一层（shallow），非病灶细节留在 native 层；全局路由按病灶/背景能量对比给目标。病灶掩码与目标 PET 仅监督用，不进推理。
- **接口**：`RouteUtilitySupervisionLoss(lesion_dilate_radius=3, spatial_weight=1.0, global_weight=0.25, spatial_tv_weight=1e-3, positive_weight=4.0, background_target=0.02, lesion_target=0.90, global_target_min=0.05, global_target_max=0.55, active_tau_max=0.70, enabled=True, weight=0.05)`，`name="route_utility_supervision"`；私有辅助 `_resize_lesion`（膨胀+最近邻下采样掩码到路由图分辨率）、`_targets`（构造空间/全局软目标）。
- **依赖**：内部：`..frequency.haar.haar_dwt2`、`..interfaces`、`.base.smooth_tau_gate`；外部：`torch`、`torch.nn.functional`。
- **输入**：必需 `ctx.batch['mask']`（缺失 ValueError）；必需条件量 `ctx.condition.maps['spectral_route_spatial_shallow_l2'/'spectral_route_spatial_shallow_l1']`（两级空间浅路由概率图）与 `ctx.condition.scalars['spectral_route_conditional_shallow']`（全局条件浅路由，缺失即 ValueError——本项不优雅降级）；`ctx.target_residual`（缺失回退 target_pet）、`ctx.tau`。
- **输出**：标量损失（×weight）+ `spatial/global/spatial_tv/route_spatial_std/target_spatial_std/global_target/gate_mean` 等日志。
- **处理过程**：
  1. 从 ConditionBundle 取两级空间浅路由图与全局浅路由概率，任一缺失即抛错（该损失要求路由器必须输出诊断量）。
  2. 对目标残差做两级 Haar 分解，第二级/第一级细节带分别与 L2/L1 空间路由图对应。
  3. 病灶掩码膨胀后最近邻插值到每级路由图分辨率并广播到 3 个方向子带。
  4. `_targets`（no_grad）：细节能量按均值归一并 tanh，空间目标 = background_target + (lesion_target-background_target)×lesion×(0.5+0.5·能量)；全局目标按病灶/背景平均能量的 contrast（有病灶时）在 [global_target_min, global_target_max] 内插值。
  5. 空间监督：路由概率先 clamp 再取 logit，用 `binary_cross_entropy_with_logits`（AMP 安全；注释明确拒绝概率空间 BCE），正类（病灶）像素按 positive_weight 加权。
  6. 全局监督：global_route[:, level] 同样经 logit-BCE 对全局目标回归。
  7. 空间路由图的一阶总变分（TV）正则抑制路由抖动。
  8. 逐样本空间+全局加权合成，乘 τ 门控后按门控归一平均，加 spatial_tv_weight×TV，最后乘 weight 返回。
### `src/model/loss_terms/spectral_router.py`（重点）

- **职责**：频谱路由的**非图像域正则项**——只消费 `ConditionBundle.scalars` 中由路由器前向写出的诊断标量/路由张量（时间平滑、DCT/Gabor 参数偏移、active 质量、先验锚定偏差、单调性、曲率、预算、浅路由概率等），施加可选加权正则；每个子项在其所需诊断缺失时自动 no-op，因此对 legacy 固定/学习策略同样可用。刻意不回指路由器模块本身。
- **接口**：`SpectralRouterRegularizationLoss(enabled=True, weight=1.0, temporal_weight=1e-4, dct_weight=1e-4, gabor_weight=1e-4, active_mass_weight=0.0, active_mass_floor=0.25, prior_anchor_weight=0.0, monotonic_weight=0.0, curvature_weight=0.0, budget_weight=0.0, shallow_weight=0.0)`，`name="spectral_router_regularization"`；`forward(ctx)` 在 fp32 中执行并返回 (加权标量, 20+ 项日志：每子项的原始值与 effective 值、is_prior_anchored、各 phase 门/anchor_scale 等)。辅助静态/类方法：`_finite_float`（设备搬运 + nan_to_num 清洗）、`_get`（scalars 取键）、`_mean_or_zero`、`_route_per_sample`（[B,6]/[B,2,3] 路由张量逐样本均值）、`_batch_gate`（标量或 [B] 门向量，clamp 到 [0,1]）、`_masked_mean`、`_route_pair`（成对取键并校验形状一致）。
- **依赖**：内部：仅 `..interfaces`（LossContext/LossTerm）——不 import frequency 侧路由器、不用 smooth_tau_gate；外部：`torch`。
- **输入**：`ctx.condition.scalars` 中约 18 个键：`spectral_route_temporal_smoothness`、`spectral_dct_weight_offset`、`spectral_gabor_parameter_offset`、`spectral_route_active_mass`、`spectral_route_is_learned`、`spectral_route_is_prior_anchored`、`spectral_route_active_phase`、`spectral_route_destination_phase`、`spectral_route_anchor_scale`、`spectral_route_active_delta`、`spectral_route_active`/`active_next`、`spectral_route_delta_prev`/`delta_next`、`spectral_route_has_prev`/`has_next`、`spectral_route_prior_active`、`spectral_route_shallow_probability`；`ctx.target_pet` 仅用于设备/批大小参考。
- **输出**：fp32 标量 `weight×Σ(子权重×子项)` + 全子项 detach 日志。
- **处理过程**：
  1. 以 fp32 零张量为参考（AMP 下仍强制 fp32，因诊断是微小的全局量）；未启用返回零。
  2. 逐键 `_get` 读取并 nan_to_num 清洗，空/缺键经 `_mean_or_zero` 归零。
  3. active-mass 地板罚：`relu(floor-active_mass)²×is_learned×(1-is_prior_anchored)`——prior_anchored 路由已有显式预算/锚定项保护，不再与其低可用区先验冲突。
  4. 先验锚定项：active_delta² 逐样本均值 × prior_gate×anchor_scale（effective 值）；仅当 delta 键存在。
  5. 单调性项：`relu(active_next-active)` 逐样本均值，按 has_next 掩码 `_masked_mean`，effective 再乘 prior_gate×active_phase。
  6. 曲率项：`|Δnext - 2Δ + Δprev|` 按 has_prev×has_next 掩码；三键缺一则 no-op。
  7. 预算项：`(mean(active)-mean(prior_active))²` 逐样本；浅路由项：shallow_probability 逐样本均值乘 prior_gate×destination_phase。
  8. 全部子项按构造权重加权求和，整体 nan_to_num 后乘 weight 返回（每项同时记录原始与 effective 两个量便于监控）。
- **与 `src/model/frequency/spectral_router.py` 的分工**：frequency 侧的 `SpectralEvidenceFrequencyRouter`（继承 `BoundaryReliableFrequencyInjector`，约 2500 行）是**前向网络模块**——在线从 Haar 细节/DCT/Gabor/CT 支撑场等证据计算 native/shallow/null 三路路由（native_only / fixed_prior / h3_native_null / prior_anchored_learned / learned / learned_no_null / legacy_off 七种策略），并把 `spectral_route_*` 诊断量写入 ConditionBundle；本文件是**纯损失项**——不参与前向、不 import 路由器，只读取这些诊断标量做正则约束。即"路由器生产诊断、损失项消费诊断"，两者通过 ConditionBundle 解耦，损失项因此兼容任何能产出同名键的路由策略（含 legacy）。

### 模块依赖小结

- **内部依赖**（grep 查证）：`..interfaces`（LossContext / LossTerm / ConditionBundle，6 个文件全部使用）；`.base`（smooth_tau_gate：normalized_lesion_peak / nonlesion_body_lowpass / residual_frequency / route_utility_supervision；perceptual_x0 仅 uniform 之外的保留分支引用）；`..frequency.haar.haar_dwt2`（residual_frequency、route_utility_supervision）。本目录 6 个文件均**不** import `frequency/spectral_router.py`（分工见上条目）。
- **消费方**（grep 查证）：`src/model/slmf_bbdm.py` 的 `_build_loss` 按配置名分别 import——normalized_lesion_peak（L1031）、ResidualWaveletLoss（L1051）、GaborConsistencyLoss（L1061）、SpectralRouterRegularizationLoss（L1071）、NonLesionBodyLowpassLoss（L1128）、RouteUtilitySupervisionLoss（L1147）、LesionAwarePerceptualX0Loss（L1225），注册进 `self.loss_terms`（L834-839）并在训练前向遍历（L1742）；`src/model/loss_terms/__init__.py` 统一再导出 NormalizedLesionPeakLoss、SpectralRouterRegularizationLoss、LesionAwarePerceptualX0Loss、PETFeatureEncoder（等）；`src/model/trainer.py`（L1277）、`src/model/loss_terms/trainer.py`（L238）、`src/scripts/inspect_model.py`（L13）打印启用损失清单。perceptual_x0 的编码器 checkpoint 由外部训练脚本产出（本部件内仅约定 lineage 格式）。


---

## 5. 模型子包 conditioning/noise/priors/frequency

### 模块概述

本组是 SLMF-BBDM 的四个策略子包，全部围绕 `src/model/interfaces.py` 中的抽象基类（`PriorModule` / `ConditionAdapter` / `NoiseSchedule`）实现"策略模式"：conditioning 负责把条件注入 UNet 跳连（Zero-Conv 适配器 + 时变 β 调度 + 条件 dropout），noise 负责前向加噪公式（DDPM / BBDM 布朗桥 / 尺度自适应），priors 负责从 batch 张量生成 `ConditionBundle`（Gabor / 器官 / 热点 / 语义 / NoOp），frequency 负责残差桥变体的频域分析（Haar 变换 / DCT 描述子 / 可靠性门控 / 谱证据路由 / H3 冻结调度加载）。数据流为：batch → priors 产出 ConditionBundle →（可选）ConditionDropout → noise.add_noise 构造 x_t → frequency 注入器与 conditioning.adapter 各自产出 4 级 skip injections → `slmf_bbdm.py` 的 `_build_skip_injections` 汇入 `BBDMUNet`。四个子包彼此几乎不互相 import（仅 adapter 依赖 beta_schedule、spectral_router 依赖 frequency 其余模块），耦合点集中在 `slmf_bbdm.py` 的工厂方法 `_build_prior`/`_build_noise` 与 `residual_frequency_mode` 三分支，以及 `trainer.py` 的 `_set_spectral_router_epoch` / 参数组分类。

---

### `src/model/conditioning/__init__.py`

**职责**：conditioning 子包的公共出口，汇聚适配器、β 调度与条件 dropout 三个模块的顶层符号。

**接口**：无独立逻辑，仅 re-export：`ZeroConvAdapter`、`RawConcatAdapter`（来自 adapter）、`local_beta`、`beta_schedules`（来自 beta_schedule）、`ConditionDropout`（来自 dropout）。

**依赖**：
- 内部: 无（仅包内相对导入 `.adapter` / `.beta_schedule` / `.dropout`）
- 外部: 无

**输入**：无（纯导出层）。
**输出**：模块级命名空间 `src.model.conditioning` 下可 import 的符号。
**处理过程**：
1. 声明子包 docstring；
2. 从三个子模块 re-export 顶层类与函数，供 `slmf_bbdm.py` 与测试按包路径引用。

---

### `src/model/conditioning/adapter.py`

**职责**：实现条件注入适配器——轻量 Zero-Conv 方案（替代完整 ControlNet，每级仅一个 1×1 零初始化卷积）与原始拼接兜底方案。核心注入公式：`h_l = h_l + β_l(t) · ZeroConv_l(cond_l)`。

**接口**：
- `class ZeroConvAdapter(ConditionAdapter)`：`__init__(ct_channels=(64,128,256,256), organ_channels=(16,32,64), gabor_channels=32, hotspot_channels=1, enabled=True)`；`forward(noisy_x, raw_condition, condition, timesteps, ct_feats=None, hw_list=None) -> Tensor`（返回拼接张量 [B, 2, H, W]）；`get_zero_conv_outputs(ct_feats, organ_feats, gabor_feat, hotspot_prior, hw_list, timesteps=None, tau=None) -> List[Tensor]`（4 级零卷积输出，可被 β 调制）；`_build_cond_per_level(...) -> List[Tensor]`；模块级 `_zero_module(module)` 将参数全部置零。
- `class RawConcatAdapter(ConditionAdapter)`：`forward` 仅做通道维拼接；`get_zero_conv_outputs(...)` 恒返回 `[]`。

**依赖**：
- 内部: `src.model.interfaces`（`ConditionBundle`、`ConditionAdapter`）、`src.model.conditioning.beta_schedule`（`multi_level_betas`、`global_beta`——后者导入未用）
- 外部: torch、torch.nn

**输入**：`ct_feats` 为 4 级 CT 特征列表（L0 192² 到 L3 24²，通道 64/128/256/256）；`organ_feats` 为 3 级器官特征（16/32/64 通道）；`gabor_feat` [B, 32, H, W]；`hotspot_prior` [B, 1, H, W]；`hw_list` 为各级分辨率；`tau` [B] 归一化时间。
**输出**：`get_zero_conv_outputs` 返回按 [L3, L2, L1, L0] 顺序（调用方 `slmf_bbdm.py` 再 reverse 成解码器深→浅序）的每级注入张量，通道数等于该级 CT 特征通道。
**处理过程**：
1. `__init__` 为 4 个分辨率级各建一个 1×1 零初始化卷积（L0 输入 97ch = CT64+Gabor32+hotspot1；L1–L3 输入 = CT + organ）；
2. `_build_cond_per_level` 把各条件双线性插值到目标分辨率，缺失的条件用全零张量占位并在通道维拼接；
3. `get_zero_conv_outputs` 逐级过零卷积得到注入增量；
4. 若提供 `tau`，调用 `multi_level_betas(tau, levels=5)` 取 5 级 β 矩阵，丢弃 bottleneck 列后翻转为 [L0→L3]，对每级输出做逐样本乘法调制（早期 τ→1 结构主导、晚期 τ→0 细节主导）；
5. `forward` 当前为最简实现：直接 `cat([noisy_x, raw_condition])`（模型输入构造），真正的注入走 `get_zero_conv_outputs`。

**挂接位置**：`slmf_bbdm.py` L23 导入；L320–324 按配置 `adapter_cfg.enabled` 构造 `self.adapter`；L1526 在 `_build_adapter_injections` 中调用 `get_zero_conv_outputs`（gabor_feat 仅在 gabor 路由 `inject_adapter=True` 时传入），产出汇入 UNet `skip_injections`。

---

### `src/model/conditioning/beta_schedule.py`

**职责**：定义条件注入强度的时变 β 调度族，控制不同条件类型在各时间步的影响力（早期全局结构、晚期细节纹理）。

**接口**：
- `local_beta(tau: Tensor, gamma=1.5) -> Tensor`：`(1-τ)^γ`，后期上升，适合 Gabor/细节；
- `global_beta(tau, floor=0.3, slope=0.7) -> Tensor`：`floor + slope·τ`，前期强，适合全局语义；
- `spatial_beta(tau, floor=0.6, slope=0.4) -> Tensor`：近似常数、后期略升，适合器官/CT 空间条件；
- `semantic_beta(tau) -> Tensor`：`sin(π·τ)`，中期峰值；
- `beta_schedules: Dict[str, Callable]`：`{"local","global","spatial","semantic","identity"}` 注册表；
- `multi_level_betas(tau: Tensor, levels=5) -> Tensor`：返回 [B, levels] 的层级×时间锚点插值 β 矩阵（锚点 τ ∈ {1.0, 0.6, 0.25, 0.0}，行为 bottleneck→L0，向量化 `searchsorted` + 线性插值）。

**依赖**：
- 内部: 无
- 外部: math、torch

**输入**：`tau` 为 [B] 张量（`timesteps / num_train_timesteps`）。
**输出**：与输入同形状的权重张量（`multi_level_betas` 为 [B, levels]）。
**处理过程**：
1. 提供 4 个解析调度函数 + identity；
2. 注册进 `beta_schedules` 字典供按名查找；
3. `multi_level_betas` 构造 5×4 锚点值表（如 bottleneck 在 τ=1 时 1.0、τ=0 时 0；L0 相反）；
4. 对每个 τ 用 `searchsorted` 在降序锚点上定位左右区间；
5. 按线性比例混合两列锚点值得到该样本每级的 β。

**挂接位置**：`ZeroConvAdapter.get_zero_conv_outputs` 内部调用 `multi_level_betas`；`slmf_bbdm.py` L24 导入、L1643 与 L2117 在训练/采样中取 β 矩阵 bottleneck 列作为 Cross-Attention 的 `ca_beta`。

---

### `src/model/conditioning/dropout.py`

**职责**：训练期条件 dropout——以独立概率把各类条件置零，提升缺失条件鲁棒性并支持推理时的弱 CFG。

**接口**：`class ConditionDropout`：`__init__(p_organ=0.1, p_hotspot=0.1, p_semantic=0.1, p_meta=0.1, p_gabor=0.1, enabled=True)`；`apply(condition: ConditionBundle, training=True) -> ConditionBundle`。

**依赖**：
- 内部: `src.model.interfaces`（`ConditionBundle`）
- 外部: random、torch

**输入**：一个已构建的 `ConditionBundle`（含 `maps` / `tokens` / `scalars` 字典）与训练标志。
**输出**：条件可能被逐类置零的新 `ConditionBundle`（先 `copy()` 再改，绝不原地修改原 bundle）。
**处理过程**：
1. 非训练态或未启用时原样返回；
2. `copy()` 避免影响其他消费者；
3. 依次以 `p_organ`/`p_gabor`/`p_hotspot`/`p_semantic`/`p_meta` 掷骰，命中即把对应键（名含 "organ"/"gabor"/"hotspot_prior"/"semantic" 的 map/token 及全部 scalars）替换为 `torch.zeros_like`。

**挂接位置**：`slmf_bbdm.py` L25 导入、L874–875 由 `condition_dropout_config` 构造 `self.condition_dropout`；L1614 在 `training_step` 中加噪之后、适配器读取条件之前调用 `apply`（注释明确要求 dropout 必须先于 adapter）。

---

### `src/model/noise/__init__.py`

**职责**：noise 子包出口，导出 DDPM、BBDM 桥与尺度自适应三种噪声调度。

**接口**：re-export `DDPMNoiseSchedule`、`BBDMBridgeSchedule`（来自 base）、`ScaleAdaptiveNoise`（来自 scale_adaptive）。

**依赖**：
- 内部: 无（包内相对导入）
- 外部: 无

**输入**：无。
**输出**：`src.model.noise` 命名空间符号。
**处理过程**：声明 docstring 后 re-export 三个调度类。

---

### `src/model/noise/base.py`

**职责**：实现两种基础噪声调度——标准 DDPM 前向过程与 BBDM 布朗桥（CT→PET 端点约束桥）。基类 `NoiseSchedule` 定义于 `src.model.interfaces`（核心方法 `add_noise`）。

**接口**：
- 模块级 `_cosine_beta_schedule(timesteps, s=0.008) -> Tensor`、`_linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02) -> Tensor`；
- `class DDPMNoiseSchedule(NoiseSchedule)`（name="ddpm"）：`__init__(num_train_timesteps=1000, beta_schedule="cosine", enabled=True)`，注册非持久 buffer `betas/alphas/alphas_cumprod/sqrt_alphas_cumprod/sqrt_one_minus_alphas_cumprod`；`add_noise(x0, noise, timesteps, condition) -> Tensor`（`√ᾱ_t·x0 + √(1-ᾱ_t)·ε`，[B] 系数广播到 [B,1,1,1]）；`get_tau(timesteps) -> Tensor`；
- `class BBDMBridgeSchedule(NoiseSchedule)`（name="bbdm_bridge"）：`__init__(num_train_timesteps=1000, m_schedule="linear", sigma_scale=1.0, enabled=True)`，预计算 buffer `m_t`（linear 或 cosine）与 `sigma_t = σ·√(2m(1-m))`；`add_noise(x0=PET, noise, timesteps, condition, x_source=CT)`：`x_t = m_t·CT + (1-m_t)·PET + σ_t·ε`（x_source 缺省从 `condition.get_map("ct")` 取，两者皆无则报错）；`get_tau`；`get_m(timesteps) -> Tensor`；`predict_x0_from_xt(x_t, x_source, pred_noise, timesteps) -> Tensor`（从桥方程反解 PET）。

**依赖**：
- 内部: `src.model.interfaces`（`ConditionBundle`、`NoiseSchedule`）
- 外部: math、torch、torch.nn

**输入**：`x0` [B,1,H,W]（PET 目标）、`noise` 同形、`timesteps` [B] 整数、`x_source` [B,1,H,W]（仅 BBDM）。
**输出**：加噪状态 `x_t` [B,1,H,W]；`predict_x0_from_xt` 返回预测的干净 PET。
**处理过程（BBDM）**：
1. 构造时按 `m_schedule` 计算 m_t 序列与桥噪声 σ_t；
2. `add_noise` 查表取 m、σ 并广播；
3. 均值项 `m·CT + (1-m)·PET` 加上 `σ·ε`——t=0 纯 PET、t=T≈CT+噪声，反向采样从 CT 出发走向 PET；
4. `predict_x0_from_xt` 用 `(x_t - m·CT - σ·ε) / (1-m)` 反解 PET（分母 clamp 防除零）。

**挂接位置**：`slmf_bbdm.py` `_build_noise`（L973–1010）按配置 `name="ddpm"/"bbdm_bridge"` 构造；训练加噪经 L93–105 的 `_add_noise` 包装（用 inspect 缓存签名判断是否传 `x_source`）；采样循环检测 `schedule.m_t` 属性走 BBDM 反向步。

---

### `src/model/noise/scale_adaptive.py`

**职责**：尺度自适应噪声调度——用拉普拉斯金字塔把图像按频带分解，对各频带施加不同噪声强度，高频带再被 Gabor 能量调制（纹理锐利区域少加噪），以保护小病灶高频边缘；同时保留 DDPM 式信号衰减保证与反向过程兼容。

**接口**：`class ScaleAdaptiveNoise(NoiseSchedule)`（name="scale_adaptive_noise"）：`__init__(num_train_timesteps=1000, num_scales=3, low_sigma_mult=1.0, mid_sigma_mult=0.75, high_sigma_mult=0.45, use_gabor_energy=True, gabor_weight=1.0, bridge_mode=False, enabled=True)`；`add_noise(x0, noise, timesteps, condition, x_source=None) -> Tensor`；`step_from_prediction(x_t, pred_x0, timesteps, next_timesteps, condition, x_source=None) -> Tensor`（DDIM 式反向步）；`get_tau(timesteps)`；内部 `_laplacian_pyramid(x, levels) -> List[Tensor]`、`_reconstruct_from_pyramid(pyramid) -> Tensor`、`_get_scale_multipliers(timesteps, gabor_energy, shape)`、`_get_band_sigmas(timesteps, condition, pyramid)`。bridge_mode=True 时注册 `m_t`/`sigma_t` buffer，使采样循环识别 `is_bbdm=True` 走桥反向步。

**依赖**：
- 内部: `src.model.interfaces`（`ConditionBundle`、`NoiseSchedule`）；读取 `condition.get_map("gabor_energy")`（由 GaborPrior 产出）
- 外部: torch、torch.nn.functional

**输入**：`x0`/`noise` [B,1,H,W]，`timesteps` [B]，`condition`（取 gabor_energy [B,1,H,W]），可选 `x_source`（CT，桥模式必填）。
**输出**：重建后的加噪状态 [B,1,H,W]（或反向一步后的状态）。
**处理过程**：
1. 以线性 β 预计算 `base_sigma`/`alphas_cumprod` 等 DDPM 基准 buffer（桥模式另算 m_t）；
2. `add_noise` 先对 noise 与 x0 各做 3 级拉普拉斯金字塔分解（avg_pool 下采样 + 上采样残差）；
3. `_get_band_sigmas` 对最高频带取 `high_sigma_mult`（并被 8×8 池化后的 Gabor 能量按 `1 - gabor_weight·g` 压低）、中间带取 mid、最低带取 low；
4. bridge_mode 且有 x_source：每带按 `m·CT_band + (1-m)·PET_band + σ_band·ε_band`；否则按 DDPM `√ᾱ·x0_band + σ_band·ε_band`；
5. `_reconstruct_from_pyramid` 逐级上采样加残差重建 x_t；
6. `step_from_prediction` 对 x_t 与 pred_x0 分别分解，按对应公式反推 ε 后再合成下一时间步状态（与加噪严格对称）。

**挂接位置**：`slmf_bbdm.py` `_build_noise` L989–1008 分支 name="scale_adaptive"（gabor 能量仅在 gabor 路由 `use_for_noise=True` 时启用）；被训练/采样经统一 `_add_noise`/反向步接口消费。

---

### `src/model/priors/__init__.py`

**职责**：priors 子包出口，导出五个先验模块（含 NoOp 空实现）。

**接口**：re-export `NoOpPrior`、`GaborPrior`、`OrganPrior`、`HotspotPrior`、`SemanticPrior`。

**依赖**：
- 内部: 无（包内相对导入）
- 外部: 无

**输入**：无。
**输出**：`src.model.priors` 命名空间符号。
**处理过程**：docstring 声明"每个先验产出 ConditionBundle 或 NoOp 空 bundle"后 re-export 五个类。

---

### `src/model/priors/gabor.py`

**职责**：相位稳定的复数 Gabor 描述子——以 scales×orientations 网格滤波器组提取局部方向统计，用正交相位（cos/sin）幅值响应避免原始载波相位在解码特征中产生条纹伪影。

**接口**：`class GaborPrior(PriorModule)`（name="gabor"）：`__init__(filters=None, kernel_size=15, enabled=True, scales=4, orientations=8, min_frequency=0.06, max_frequency=0.28, parameter_delta=0.25)`；`describe(image: Tensor, detach_parameters=False) -> Dict[str, Tensor]`（返回 gabor_feat 幅值 [B,F,H,W]、gabor_orientation [B,O,H,W]、gabor_energy [B,1,H,W]、gabor_anisotropy [B,1,H,W]）；`parameter_offset_energy() -> Tensor`；`forward(batch, timesteps, partial_bundle=None) -> ConditionBundle`；内部 `_bounded_parameters`、`_build_quadrature_kernels`、`_build_kernels`（兼容旧测试的实核访问器）。可学习参数 `log_frequency/theta_raw/log_sigma/gamma_raw` 为围绕网格锚点的有界偏差（tanh 限幅，频率/σ 变化率 ≤ log1p(δ)，角度限制在网格单元 1/4 内防止相邻滤波器交叉），`phase` 仅作冻结的 state-dict 兼容键。

**依赖**：
- 内部: `src.model.interfaces`（`ConditionBundle`、`PriorModule`）
- 外部: math、torch、torch.nn、torch.nn.functional

**输入**：`batch["ct"]` [B,1,H,W]。
**输出**：`ConditionBundle(maps={gabor_feat, gabor_orientation, gabor_energy, gabor_anisotropy})`——被 ScaleAdaptiveNoise（energy）、HotspotPrior（energy）、spectral_router（feat/orientation/anisotropy）、ZeroConvAdapter（feat）四路消费。
**处理过程**：
1. 构造 logspace 频率 × 均匀角度的笛卡尔网格锚点 buffer（非持久，兼容旧 checkpoint）；
2. `_bounded_parameters` 把四个 raw 参数经 tanh 有界映射为实际 frequency/theta/sigma/gamma；
3. `_build_quadrature_kernels` 由网格坐标旋转生成高斯包络×cos/sin 载波的核，去均值后 L2 归一化为 [F,1,K,K]；
4. `describe` 对图像做同尺寸卷积，合成幅值 `√(re²+im²)` 并按 RMS 归一化；
5. 按尺度平均得 orientation、再平均得 energy；
6. 用 2θ 角度的 cos/sin 矢量合成计算各向异性 anisotropy∈[0,1]。

**挂接位置**：`slmf_bbdm.py` `_build_prior` L927–936（name="gabor"）；L301–307 的 `gabor_routes` 决定其输出流向（inject_adapter/use_for_noise/use_for_hotspot/use_for_loss）。

---

### `src/model/priors/hotspot.py`

**职责**：微型热点先验网络——轻量 CT→病灶候选 U-Net，输出软注意力图引导扩散过程聚焦病灶区域（参数预算 <0.5M）。

**接口**：
- `class _TinyUNet(nn.Module)`：`__init__(in_ch=1, base=16)`、`forward(x) -> Tensor`（sigmoid 输出 [B,1,H,W]），两级下采样瓶颈 + 跳连解码；
- `class HotspotPrior(PriorModule)`（name="hotspot_prior"）：`__init__(base_channels=16, params_max=500_000, use_gabor_energy=True, enabled=True)`（超预算打印警告）；`_build_input(batch, partial_bundle) -> Tensor`；`forward(batch, timesteps, partial_bundle=None) -> ConditionBundle`。

**依赖**：
- 内部: `src.model.interfaces`（`ConditionBundle`、`PriorModule`）；经 `partial_bundle.get_map("gabor_energy")` 消费 GaborPrior 输出
- 外部: torch、torch.nn、torch.nn.functional

**输入**：`batch["ct"]` [B,1,H,W]；可选 `partial_bundle` 中的 gabor_energy（形状不符时双线性插值对齐，缺失时用全零通道占位保持 2 通道输入）。
**输出**：`ConditionBundle(maps={"hotspot_prior": [B,1,H,W]})`，被 ZeroConvAdapter L0 级拼接消费。
**处理过程**：
1. `_build_input` 拼接 CT 与（插值后的）Gabor 能量两通道；
2. `_TinyUNet` 编码（16→32→64 通道，SiLU 激活）到 1/4 分辨率瓶颈；
3. 双线性上采样 + 跳连解码回原分辨率；
4. 1×1 卷积 + sigmoid 输出热点概率图，包成 ConditionBundle。

**挂接位置**：`slmf_bbdm.py` `_build_prior` L945–961（gabor 能量输入仅在 gabor 路由 `use_for_hotspot=True` 时开启，配置无法绕过关闭的路由）。

---

### `src/model/priors/noop.py`

**职责**：空操作先验——模块禁用或名称未识别时返回空 ConditionBundle，保证 `self.priors` 字典结构稳定。

**接口**：`class NoOpPrior(PriorModule)`（name="noop"）：`__init__(**kwargs)`（恒 `enabled=False`）；`forward(batch, timesteps, partial_bundle=None) -> ConditionBundle`。

**依赖**：
- 内部: `src.model.interfaces`（`ConditionBundle`、`PriorModule`）
- 外部: torch

**输入**：任意 batch 字典与 timesteps（被忽略）。
**输出**：仅含 `logs={"noop/enabled": False}` 的空 bundle。
**处理过程**：
1. 构造即 `enabled=False`；
2. `forward` 直接返回带禁用日志的空 ConditionBundle（不产生任何 map/token）。

**挂接位置**：`slmf_bbdm.py` L313–316（配置禁用的先验以 NoOp 占位并保留原 name）、`_build_prior` L921/925（显式禁用或未知名称时返回）。

---

### `src/model/priors/organ.py`

**职责**：器官先验编码器——把冻结的解剖分割掩码、（可选）符号距离变换与 μ 图编码为 3 个尺度的空间特征，供 Zero-Conv 适配器 L1–L3 注入，用于降低膀胱/直肠/骨区假阳性。

**接口**：`class OrganPrior(PriorModule)`（name="organ_prior"）：`__init__(organ_channels=6, distance_channels=6, mu_map_channels=1, out_channels=(16,32,64), enabled=True)`（三级 Conv+SiLU，stage2/3 步长 2 下采样）；`_build_input(batch) -> Tensor`（13 通道拼接，缺失项全零替代）；`forward(batch, timesteps, partial_bundle=None) -> ConditionBundle`。

**依赖**：
- 内部: `src.model.interfaces`（`ConditionBundle`、`PriorModule`）
- 外部: torch、torch.nn

**输入**：`batch["organ_mask"]` [B,6,H,W]（TotalSegmentator one-hot，冻结）、可选 `batch["organ_distance"]` [B,6,H,W]、`batch["mu_map"]` [B,1,H,W]；以 `batch["ct"]` 的形状/设备为参考。
**输出**：`ConditionBundle(maps={"organ_feat_1": [B,16,H,W], "organ_feat_2": [B,32,H/2,W/2], "organ_feat_3": [B,64,H/4,W/4]})`。
**处理过程**：
1. `_build_input` 从 batch 取三类输入，任一缺失则以同形零张量补齐；
2. 通道维拼接成 13 通道输入；
3. 三级卷积编码（H → H/2 → H/4）；
4. 三个尺度特征写入 ConditionBundle maps。

**挂接位置**：`slmf_bbdm.py` `_build_prior` L937–944；输出在 `_build_adapter_injections` L1507–1511 被读取。

---

### `src/model/priors/semantic.py`

**职责**：语义先验——把离线预计算（RadImageNet ResNet50 等骨干）并缓存于 .npz 的冻结视觉特征作为全局上下文 token 注入，供 UNet 瓶颈 Cross-Attention 使用；缺缓存时退化为确定性 null token。docstring 记录了 DINOv2 路径未验证、以及备选方案（可学习位置嵌入 / BiomedCLIP / 直接删除）的取舍。

**接口**：`class SemanticPrior(PriorModule)`（name="semantic_prior"）：`__init__(mode="cached_tokens", token_dim=64, num_tokens=4, enabled=True)`（cached_tokens 模式建 `nn.Linear(token_dim, token_dim)` 投影与 `null_tokens` 零参数）；`forward(batch, timesteps, partial_bundle=None) -> ConditionBundle`。

**依赖**：
- 内部: `src.model.interfaces`（`ConditionBundle`、`PriorModule`）
- 外部: torch、torch.nn

**输入**：`batch["semantic_tokens"]` [B, N, token_dim]（离线缓存，dataset 提供）；缺省时用 null_tokens 展开。
**输出**：`ConditionBundle(tokens={"semantic": [B, N, D]})`——被 `slmf_bbdm.py` L1645 取出作为 UNet `context_tokens`。
**处理过程**：
1. 禁用时返回空 bundle；
2. cached_tokens 模式且 batch 含 semantic_tokens：线性投影缓存特征；
3. 否则 null_tokens 扩展到 batch 维后投影（保持设备一致）；
4. 非 cached_tokens 模式直接输出 null token。

**挂接位置**：`slmf_bbdm.py` `_build_prior` L962–969；token 流入 UNet 瓶颈 Cross-Attention（受 `multi_level_betas` bottleneck 列 β 调制）。

---

### `src/model/frequency/__init__.py`

**职责**：frequency 子包出口，导出残差 BBDM 变体所需的频域构件。

**接口**：re-export `SelectedDCTDescriptor`、`haar_dwt2`、`haar_idwt2`、`reconstruct_lowpass`、`ResidualFrequencyPreconditioner`、`SpectralEvidenceFrequencyRouter`；`__all__` 显式列出。

**依赖**：
- 内部: 无（包内相对导入；注意 `src/model/loss_terms/frequency.py` 是另一个同名无关模块）
- 外部: 无

**输入**：无。
**输出**：`src.model.frequency` 命名空间符号。
**处理过程**：re-export 六个符号；`boundary_reliable` 与两个 schedule 加载器不在此导出，由 `slmf_bbdm.py` / `spectral_router.py` 按模块路径直接导入。

---

### `src/model/frequency/haar.py`

**职责**：零依赖的正交归一化二维 Haar 变换，是整个 frequency 子包（及多个 loss、UNet、均值预测器）的底层数学基元。

**接口**：
- `haar_dwt2(x: Tensor) -> (LL, (LH, HL, HH))`：一级正交 Haar 分解（系数含 ×0.5 因子保证 2D 正交归一，能量/系数损失可比）；
- `haar_idwt2(ll, details) -> Tensor`：精确逆变换（要求 LL 与三个 detail 形状一致）；
- `reconstruct_lowpass(ll, levels=2) -> Tensor`：细节全零时从 LL 重建图像；
- 类型别名 `HaarDetails = Tuple[Tensor, Tensor, Tensor]`；内部 `_validate_image` 强制 [B,C,H,W] 且空间维为偶数。

**依赖**：
- 内部: 无
- 外部: torch

**输入**：4 维张量 [B,C,H,W]（dwt 要求 H、W 为偶数；idwt 输入为半分辨率系数）。
**输出**：`haar_dwt2` 返回 LL 与 LH/HL/HH 四个半分辨率张量；`haar_idwt2` 返回全分辨率重建。
**处理过程**：
1. 校验 ndim=4 且空间维偶数；
2. 用步长 2 切片取四个 2×2 子块 a/b/c/d；
3. 按 Haar 基（求和/行差/列差/对角差）各乘 0.5 计算 LL/LH/HL/HH；
4. 逆变换由线性组合还原 a/b/c/d 后散布回 stride-2 网格；
5. `reconstruct_lowpass` 用零细节做 levels 次 idwt。

**挂接位置**：被 `boundary_reliable.py`、`residual_preconditioner.py`、`slmf_bbdm.py`（L1728 均值低频损失）、`wavelet_unet.py`、`mean_predictor.py`（reconstruct_lowpass）、`mean_pretraining.py`、`loss_terms/{residual_frequency,boundary_frequency,route_utility_supervision}.py` 消费。

---

### `src/model/frequency/dct_descriptor.py`

**职责**：小型选定频率 DCT 描述子——把 Haar 三细节带能量图池化到 8×8 后投影到 12 个选定低频 DCT 基，产出固定 12 维/带的谱证据向量（V5 版本）。

**接口**：`class SelectedDCTDescriptor(nn.Module)`：`__init__(pooled_size=8, selected_frequencies=12)`（pooled_size 必须为 8，频率数 ∈[1,12]，注册基矩阵 `basis` [K,8,8] 与先验权重 `prior_weights` buffer、可学习 `weight_offsets` 参数）；`frequency_weights() -> Tensor`（先验 logit + 偏移的 softmax）；`forward(details: Tensor) -> (descriptor, diagnostics)`（输入强制 [B,3,H,W] Haar 细节；descriptor [B,3,K]，diagnostics 含 frequency_weights 与 weight_offset_energy）。模块级 `_SELECTED_8X8` 为 12 个 (u,v) 频率对常量、`_dct_basis(size, frequencies)` 生成归一化 DCT 基。

**依赖**：
- 内部: 无（不依赖 haar，输入须已是 Haar 细节堆叠）
- 外部: math、torch、torch.nn、torch.nn.functional

**输入**：Haar 细节堆叠张量 [B,3,H,W]。
**输出**：descriptor [B,3,12]（行归一化系数 × 频率权重）与诊断字典。
**处理过程**：
1. `log1p(|details|)` 压缩能量动态范围后自适应平均池化到 8×8；
2. 减去每图均值（去直流）并记录尺度用于数值下限；
3. `einsum` 与 DCT 基内积取绝对值得 [B,3,K] 系数；
4. 用 max(系数和, pooled_scale·√eps) 做分母归一化防除零；
5. 乘以 `frequency_weights()` 的 softmax 权重输出，同时回传权重与偏移能量正则项。

**挂接位置**：`spectral_router.py` L16 导入、L537–541 实例化为 `self.dct_descriptor`（`_dct_evidence` 调用），构成 48 维证据向量的前 36 维。

---

### `src/model/frequency/residual_preconditioner.py`

**职责**："legacy" 模式的残差频域预条件器——把噪声残差经 Haar 分解为低/中/高/高-L0 四带，按桥调度感知的带门控调制后经零初始化投影头输出与解码器层级匹配的注入。

**接口**：`class ResidualFrequencyPreconditioner(nn.Module)`：`__init__(output_channels=(256,256,128,64), band_scales=(1.0,0.5,0.25), inject_wavelet=True, use_gabor_gate=False, gabor_orientations=8, gate_strength=0.1, state_modulation=True)`；`forward(noisy_residual, timesteps, schedule, gabor_orientation=None) -> (injections: List[Tensor], diagnostics)`（injections 为 [L3,L2,L1,L0] 四张量）；`modulate_residual(residual, gabor_orientation) -> Tensor`（仅门控 level-1 细节、保留 LL）；`gabor_factor(orientation, size)`；`_gate_features(timesteps, schedule)`（要求 schedule 具备 m_t/sigma_t，即布朗桥调度）。内部 `_ZeroProjection`（末层零初始化保证初始为恒等）与 `_IndependentBandGates`（输入 5 维 [τ, progress, 3×log_SNR] 的三带 sigmoid 门，初始偏置 [-0.4,0,0.4]）。

**依赖**：
- 内部: `src.model.frequency.haar`（`haar_dwt2`、`haar_idwt2`）
- 外部: math、torch、torch.nn、torch.nn.functional

**输入**：`noisy_residual` [B,1,H,W]（桥噪声状态/残差）、`timesteps` [B]、桥调度对象（提供 `get_tau`/`m_t`/`sigma_t`）、可选 gabor_orientation [B,8,H,W]。
**输出**：4 级注入列表（L3 来自 LL/2、L2 来自 2 级细节、L1 来自 1 级细节、L0 为 2× 上采样的高频）与诊断（band_gates、band_log_snr）；`inject_wavelet=False` 时返回空列表（仅用于状态调制模式）。
**处理过程**：
1. 两级 Haar 分解得 LL2、细节2（中带）、细节1（高带），LL2 再下采样、高频再 2× 上采样的 high_l0；
2. 各带除以 band_scales 归一化；
3. `_gate_features` 由调度算每带 log-SNR（信号功率 = (1-m)·scale²，噪声功率 = σ²），与 τ、progress 拼接；
4. `_IndependentBandGates` 输出三带门控；
5. 高频带可选乘 Gabor 因子（`1 + gate_strength·tanh(conv)`）；
6. 四带分别乘对应门列后过四个 `_ZeroProjection` 头得到注入。

**挂接位置**：`slmf_bbdm.py` L490–501（`residual_frequency_mode="legacy"` 时构造为 `self.residual_preconditioner`）；`modulate_residual` 在 L1623/L2106 作为状态调制分支使用。

---

### `src/model/frequency/boundary_reliable.py`

**职责**：边界可靠频带注入器（decoder-only）——读取桥状态、CT、时间步与可选 Gabor 方向能量，但从不改写扩散状态；仅把经多重可靠性门控的 Haar 细节残差投影进解码器跳连。它是 `SpectralEvidenceFrequencyRouter` 的直接基类。

**接口**：`class BoundaryReliableFrequencyInjector(nn.Module)`（类属性 `inject_wavelet=True`）：
- `__init__(output_channels=(256,256,128,64), band_scales=(0.5,0.25), use_noise_release=True, use_ct_reliability=True, use_content_reliability=True, use_subband_gates=True, use_directional_reliability=False, gabor_orientations=8, gate_max=0.25, snr_center=0.0, snr_temperature=2.0, cross_temperature=1.0, content_hidden_channels=16, ct_reliability_floors=(0.0,0.0), use_gabor_agreement=False, gabor_agreement_alpha=0.10, gabor_agreement_l2=False, gabor_agreement_l1=True, detach_gabor_descriptor=True)`；
- `decompose(image) -> (LL1, details1, details2)`（静态，两级 Haar）；`reconstruct_l0(details1)`（静态，IDWT(0, LH1, HL1, HH1) 的纯高频重建）；`noise_reliability(timesteps, schedule) -> [B,2]`（解析桥 log-SNR 的 sigmoid，要求调度有 m_t/sigma_t）；`cross_modal_reliability(residual_details, ct_details, level_index)`（归一化局部能量距离的指数衰减，带下限）；`directional_reliability(orientation_energy, size)`（Gabor 方向能量到 LH/HL/HH 的映射）；`gabor_agreement(q_g, q_h)`（静态，三带分布的 Bhattacharyya 一致度）；`gabor_agreement_reliability(...)`；`apply_reliability_floor(reliability, floor)`（静态，[0,1]→[floor,1] 不反转序）；`forward(noisy_state, timesteps, schedule, ct, gabor_orientation=None, gabor_anisotropy=None) -> (injections: List[Tensor], diagnostics)`。

内部构件：`_ZeroProjection`（末层零初始化）、`_ContentGate`（7 维全局统计→三带 sigmoid 门）、`_stack_details`/`_split_details`/`_total_variation` 工具函数。

**依赖**：
- 内部: `src.model.frequency.haar`（`HaarDetails`、`haar_dwt2`、`haar_idwt2`）
- 外部: math、torch、torch.nn、torch.nn.functional

**输入**：`noisy_state` 与 `ct` 均为 [B,1,H,W] 且形状相同、空间维可被 8 整除；桥调度对象；可选 gabor_orientation [B,8,H,W]、gabor_anisotropy [B,1,H,W]。
**输出**：按解码器深→浅序 [L3, L2, L1, L0] 的注入列表（L3 恒为零张量，L0 为门控 L1 细节的纯高频重建）与诊断（gates_l2/gates_l1、noise_reliability、gate_tv 总变差正则）。
**处理过程**：
1. decompose 桥状态与 CT 得两级细节；
2. `noise_reliability` 由桥调度 m/σ 与 band_scales 算每级 log-SNR → sigmoid；
3. `_level_gates` 级联五种可靠性：cross（对 CT 能量分布的贴近度，带 floor）× content（7 维统计经 _ContentGate 的三带门）× directional（Gabor 方向带能量归一化映射）× gabor_agreement（Bhattacharyya 一致度乘 anisotropy 置信度）；
4. 总门 = gate_max · noise · cross · content · direction · agreement，clamp 到 [0, gate_max]；
5. 细节除以 band_scales 后 tanh 限幅再乘门；
6. L2/L1 过 _ZeroProjection 头，L1 细节经 reconstruct_l0 得 L0 注入，L3 全零，拼成 4 级列表返回。

**挂接位置**：`slmf_bbdm.py` L502–544（`mode="boundary_reliable"` 构造）；L1466–1485 调用 forward（use_gabor_agreement 开启时附传 gabor_anisotropy）。

---

### `src/model/frequency/spectral_router.py`

**职责**：谱证据频率路由器（2559 行，frequency 子包核心）——继承 `BoundaryReliableFrequencyInjector`，用固定 48 维谱证据（residual DCT + CT DCT + 差值 + 标量）对每个 Haar 细节包做 native/shallow/null 三路路由，支持 7 种路由策略、H3 冻结先验锚定、分阶段训练进度、以及仅推理期的三类机制审计干预。

**接口**（顶层）：
- 路由头：`BoundedAmplitudeHead(input_features, hidden_channels, minimum=-0.05, maximum=0.10)`（有界信任幅度，初始精确为零）；`ConservativeRouteHead(..., initial_null_probability=0.90)`（三路 softmax，零初始化 + 先验偏置）；`NoNullRouteHead(..., initial_native_probability=0.5)`（双路无 null）；`BoundedActiveLogitDeltaHead(..., maximum_absolute_delta=2.0)`（有界 log-odds 修正，初始零）；`SpatialDestinationDeltaHead(hidden_channels=16, maximum_absolute_delta=3.0)`（6 通道卷积、零均值空间 logit 修正）；
- `UncertaintyAwareRouteSelector(confidence_threshold)`：H4-v2 接口——置信度低于阈值时逐元素弃权回退到冻结路由（默认不接入生产）；
- `BiasFreeZeroProjection(in_channels, out_channels, init_scale=0.0)`：无偏置投影保证 P(0)=0；`init_scale>0` 用于 shallow 路径的冷启动解锁（实验 A，2026-07-27）；
- `class SpectralEvidenceFrequencyRouter(BoundaryReliableFrequencyInjector)`：`__init__` 约 50 个参数（output_channels/band_scales/ct_reliability_floors、dct_enabled/gabor_enabled/cross_level_enabled/hard_all_null、route_policy ∈ {native_only, fixed_prior, h3_native_null, prior_anchored_learned, learned, learned_no_null, legacy_off}、fixed_prior、native_warmup_epochs/routing_ramp_epochs、ct_support_* 、uncertainty_aware_*、h3_schedule_path/sha256/source/repository_root/allow_unverified_preview_lineage/num_train_timesteps、prior_warmup/active_ramp/destination_warmup/destination_ramp/anchor_decay_end/final_scale、active_logit_delta_max、initial_destination_native_probability、shallow_projection_init_scale、destination_bootstrap/floor/ceiling、spatial_destination_*、low_frequency_*）；关键方法：`forward(current_residual, timestep, schedule, ct, gabor_orientation=None, gabor_anisotropy=None, gabor_feat=None, lesion_score=None, topq_mask=None, router_confidence=None) -> (injections, diagnostics)`；`set_training_epoch(epoch)` / `phase_label(epoch)`（prior_frozen→active_ramp→active_only_hold→destination_ramp→full_adaptive）；`set_routing_progress(progress)`；`set_inference_destination_intervention(mode, fixed_l2, fixed_l1, shuffle_seed)` / `set_inference_route_action_intervention(actions_l2, actions_l1)` / `set_inference_frequency_intervention(mode ∈ {full, detail_off, ll_off, all_frequency_off})` 及对应查询属性；`_acquire_routes`（按策略分发）、`_level_evidence`（48 维证据构造）、`_prior_anchored_route_at`（H3 先验 + 有界修正 + 目的地分解）、`_ct_support_field`（H1 鲁棒中值/Sigma 归一化 CT 支撑场）、`_spatial_destination_map`（零均值有界空间目的地）、`_route_diagnostics`（逐级 active_mass/entropy/null 分位）、`_load_from_state_dict`（拒绝 checkpoint 携带的 H3 表覆盖已验证调度）。

**依赖**：
- 内部: `src.model.frequency.boundary_reliable`（基类与 _split_details/_stack_details/_total_variation）、`src.model.frequency.dct_descriptor`、`src.model.frequency.h3_native_null_schedule`（`load_h3_native_null_schedule`、`native_null_routes`）、`src.model.frequency.prior_anchor_schedule`（`load_prior_anchor_schedule`）
- 外部: math、torch、torch.nn.functional、torch.nn

**输入**：`current_residual` 与 `ct` [B,1,H,W]（同形、空间维被 8 整除）、`timestep` [B]、桥调度（m_t/sigma_t/num_train_timesteps）、可选 gabor_feat/orientation/anisotropy（来自 GaborPrior）、`router_confidence` [B,2,3]（H4-v2 启用时必填）；H3 策略另需 schedule JSON 路径 + SHA-256。
**输出**：4 级注入 [L3(零), L2(native), L1(native_l1 + shallow_l2 + 可选低频), L0(shallow_l1)] 与大型诊断字典（gates/routes/noise_reliability/gate_tv/route_temporal_smoothness/DCT 权重/Gabor 一致度/injection RMS/逐级 active_mass/entropy/null 分位/先验锚定进度等）。
**处理过程**：
1. 校验输入与时间步；`hard_all_null` 直接返回全零注入与全 null 诊断；
2. decompose 残差与 CT；算噪声可靠性与两级基础门控；可选 CT 支撑场（鲁棒 sigmoid）在证据构造之后才乘入门控（CT 只做空间支撑、不能决定 PET 带幅值）；
3. `_level_evidence` 拼装 [B,3,48] 证据：residual DCT 12 + CT DCT 12 + |diff| 12 + 标量 12（带分布×2、Haar 一致度、归一化时间步、log-SNR、噪声释放、噪声校准证据、Gabor 全局/方向/各向异性能量、Gabor-Haar/DCT 一致度）；ct_support_only 模式下 CT 侧证据置零；
4. 幅度头输出有界 δ，`amplitude = clamp(gate·(1+δ), 0, gate_max)`（H3 类策略结构性旁路幅度头）；
5. `_acquire_routes` 按策略得路由 [B,3,3]：fixed 类查表、h3_native_null 查冻结表 [a,0,1-a]、prior_anchored_learned 由 H3 先验 active 经有界 logit 修正 + 目的地头分解 native/shallow（含 bootstrap 与相位 ramp）、learned 由 softmax 头输出并经 `_blend_with_native` 从纯 native 渐进混合；均计算 t+1（先验锚定还含 t-1）时间步路由的时间平滑度；
6. 可选 H4-v2 弃权选择器替换低置信 (level,band) 路由为冻结回退；
7. 可选空间目的地头把全局目的地 logit 加零均值有界 δ 再 sigmoid，得逐像素 native/shallow 分配；
8. 细节 tanh 限幅 × 幅度 × 路由得 native/shallow 分支；native L2/L1 过 projection_heads，shallow L2 经 reconstruct_l0 后由 `l2_to_l1_projection` 送入 L1，shallow L1 送 L0；可选低频背景分支（tanh(LL1/CT-LL1)×噪声门）；最后按推理频率干预置零各分支，拼装 [l3,l2,l1,l0] 与全部诊断。

**挂接位置**：`slmf_bbdm.py` L545 起（`mode="spectral_evidence_router"` 构造为 `self.residual_preconditioner`）；L1384–1465 调用 forward 并把路由诊断写入 `condition.scalars`（供 route_utility_supervision 等 loss 消费）、把空间路由图写回 `condition.maps`；L759 透传推理目的地干预；`trainer.py` L854–860 `_set_spectral_router_epoch` 每 epoch 调用 `set_training_epoch`，L497–520/638 把 projection_heads/l2_to_l1_projection/low_frequency_projection（projection 组）、route/amplitude/prior_active/prior_destination/spatial_destination heads（router 组）、dct_descriptor（descriptor 组）划入差异化学习率，L1037–1053 `_router_phase_label` 把 phase 标记写进 JSONL 监控记录，L1296 打印先验锚定摘要。

---

### `src/model/frequency/h3_native_null_schedule.py`

**职责**：H3-v2 全时间步 native/null 调度的 fail-closed（凡错即拒）加载器——每个 (level, band, timestep) 一个冻结 active 质量 a，映射为 [native, shallow, null] = [a, 0, 1-a]；绝不插值、绝不看图、绝不发明 shallow 概率。

**接口**：
- 常量：`H3_V2_PIPELINE_ID = "H3_V2_FULL_TIMESTEP_NATIVE_NULL_CALIBRATION_V1"`、`H3_V2_ROUTE_ORDER = ("native","shallow","null")`、`H3_V2_BAND_ORDER = ("l2_lh","l2_hl","l2_hh","l1_lh","l1_hl","l1_hh")`；
- `load_h3_native_null_schedule(path, *, expected_file_sha256: str, expected_num_train_timesteps: int) -> (active_mass: Tensor[2,3,T], metadata: dict)`：文件 SHA-256 比对（64 位 hex）→ JSON 解析 → schema_version/pipeline_id/decision=PASS/inference_schedule_allowed=True/production_activation_allowed=False/自哈希一致 → route/band 顺序精确匹配 → 时间步数一致 → lookup_policy 禁止插值、shallow 结构为零、声明的禁用运行时输入（target_pet/lesion_mask 等）齐全 → 每带数值有限、∈[0,1]、随时间步非增（容差 1e-7）；
- `native_null_routes(active_mass, timestep, *, level_index, reference) -> Tensor`：整数索引查表返回精确 [B, 3 bands, 3 routes] 行（shallow 列恒零）；
- 内部 `_file_sha256`、`_canonical_sha256`（排序紧凑 JSON 规范化）、`_self_hash`（剔除 schedule_sha256 字段后自哈希）。

**依赖**：
- 内部: 无（被 `spectral_router.py` 与 `prior_anchor_schedule.py` 消费）
- 外部: hashlib、json、pathlib、torch

**输入**：H3-v2 调度 JSON 文件路径 + 期望文件 SHA-256 + 期望训练时间步数。
**输出**：shape [2,3,T] 的 float32 active 质量张量（[L2,L1]×[LH,HL,HH]×t）与元数据字典（绝对路径、双哈希、pipeline_id、步数、顺序表）。
**处理过程**：
1. 校验文件存在与 64 位十六进制期望哈希，逐块计算实际 SHA-256 比对；
2. 解析 JSON（utf-8-sig），逐项校验身份/许可/自哈希字段；
3. 校验 route/band 顺序、时间步数、查找策略与禁用输入声明；
4. 逐带提取 native_active_mass 数值，检查有限性、[0,1] 范围与时间步单调非增；
5. reshape 为 [2,3,T] 返回；任何一步不符即抛 ValueError/FileNotFoundError（fail-closed）。

**挂接位置**：`spectral_router.py` L17–20 导入；h3_native_null 与 prior_anchored_learned 策略构造时加载（L657–682，buffer 特意 non-persistent 防止旧 checkpoint 覆盖已验证调度）；`native_null_routes` 在 `_acquire_routes` 中按时间步查表。

---

### `src/model/frequency/prior_anchor_schedule.py`

**职责**：先验锚定调度的窄分发器——正式 H3-v2 契约仍归 `h3_native_null_schedule` 所有；本模块额外提供一个单独标记的"直连 PNG 预览"加载路径，其 checkpoint/缓存血缘对当前运行未验证，消费必须显式 opt-in。

**接口**：
- 常量：`FORMAL_H3_V2 = "formal_h3_v2"`、`DIRECT_PNG_PREVIEW = "direct_png_preview"`、`PRIOR_ANCHOR_SCHEDULE_SOURCES`、`DIRECT_PNG_PREVIEW_PIPELINE_ID = "H3_DIRECT_PNG_PRIOR_PREVIEW_V1"`、`DIRECT_PNG_PREVIEW_PATH_POLICY`（POSIX 分隔的仓库相对路径策略）；
- `load_prior_anchor_schedule(path, *, schedule_source, expected_file_sha256, expected_num_train_timesteps, repository_root=None, allow_unverified_preview_lineage=False) -> (Tensor, dict)`：顶层分发器——formal 分支禁止携带 allow 标志并直接委托 `load_h3_native_null_schedule`；direct_png_preview 分支要求 allow=True 并走 `_load_direct_png_preview`；其他来源抛错；
- `_load_direct_png_preview(...)`：HMAC 恒时哈希比对 → `_validate_preview_flags`（schema/pipeline/decision=PREVIEW_ONLY/四布尔标志）→ `_validate_preview_paths`（paths/outputs 必填键、POSIX 相对路径规范化、预览文件必须位于 output_dir）→ `_validate_preview_lineage_shape`（input_contract：raw_png_direct、192 分辨率、manifest/分区/原始 PNG 指纹哈希；dataset_contract 全检查为真；checkpoint 血缘显式未验证、均值 checkpoint 必须病理排除）→ `_validate_preview_analysis`（m_schedule=linear、σ=1.0、等患者加权、clip 变换、逐带等渗非增、充分统计量法、[a,0,1-a] 公式）→ `_preview_active_mass`（六带数值校验同 H3）；
- 工具函数：`_resolve_preview_path`（解析为仓库内绝对路径 + POSIX 相对串）、`_portable_path`（拒绝反斜杠/URI/绝对路径/.. 逃逸）、`_sha256`/`_file_sha256`/`_mapping`/`_positive_int`。

**依赖**：
- 内部: `src.model.frequency.h3_native_null_schedule`（`H3_V2_BAND_ORDER`、`H3_V2_ROUTE_ORDER`、`load_h3_native_null_schedule`）
- 外部: hashlib、hmac、json、re、pathlib、torch

**输入**：调度 JSON 路径、来源标签、期望 SHA-256、期望时间步数；预览分支另需 repository_root 与显式的 `allow_unverified_preview_lineage=True`。
**输出**：与 H3 相同的 [2,3,T] active 质量张量；预览分支 metadata 标注 `preview_only=True`、`lineage_verified=False`、`inference_schedule_allowed=False`、`production_activation_allowed=False`。
**处理过程**：
1. 按 schedule_source 二分派（formal 不允许 allow 标志）；
2. 预览路径先做恒时哈希比对与仓库内路径解析；
3. 依序做标志、路径、血缘形状、分析参数四组校验（任何字段不符即抛错）；
4. 提取并校验六带 active 质量（有限、[0,1]、非增）；
5. 返回张量与完整溯源元数据。

**挂接位置**：`spectral_router.py` L21 导入；prior_anchored_learned 策略构造时按 `h3_schedule_source` 调用（L664–673）。

---

### 模块依赖小结

**本组 import 的内部模块**（按子包）：
- conditioning：`src.model.interfaces`（adapter/dropout）、`src.model.conditioning.beta_schedule`（adapter→multi_level_betas）；
- noise：`src.model.interfaces`（base/scale_adaptive）；scale_adaptive 运行时读取 GaborPrior 写入 bundle 的 `gabor_energy`（数据耦合非 import）；
- priors：`src.model.interfaces`（全部五个文件）；hotspot 运行时消费 `partial_bundle` 中的 gabor_energy；
- frequency：`haar` 被 boundary_reliable / residual_preconditioner 消费；`dct_descriptor`、`h3_native_null_schedule`、`prior_anchor_schedule`、`boundary_reliable` 被 spectral_router 消费；prior_anchor_schedule 复用 h3_native_null_schedule 的常量与加载器。外部依赖仅 torch（+math/hashlib/hmac/json/re/pathlib 标准库），无任何第三方库。

**被哪些内部模块/脚本消费**（grep 查证，不含 artifacts 快照与 tests）：
- conditioning：`src/model/slmf_bbdm.py`（L23–25 导入；adapter 构造 L320、dropout L874、β 调度 L1643/L2117 与 adapter 内部）；`src/data/dataset.py` L342 附近仅注释提及（"DICOM metadata for model conditioning"，实际是元数据采集而非 import）；
- noise：`src/model/slmf_bbdm.py`（L977/983/990 工厂构建；`_add_noise` 签名适配 L93）；`scripts/train_v2.py` L82–83（配置构造）；多个校准/评估脚本（calibrate_h3_v2、validate_h3_logsnr_recoverability、validate_h4_*、estimate_h3_prior_from_png、eval_action_axis_screen、diagnose_gabor_adapter）直接 import 桥调度；
- priors：`src/model/slmf_bbdm.py`（L314 NoOp 占位、L921–969 `_build_prior` 工厂）；`src/model/trainer.py` L1276（打印启用的先验）；`src/model/loss_terms/trainer.py` L237（同样打印）；`scripts/train_v2.py` L76/235；`scripts/diagnose_stripes.py` L280；
- frequency：`src/model/slmf_bbdm.py`（L491/503/546 三模式构造 residual_preconditioner、L1728 haar 用于均值低频损失）；`src/model/wavelet_unet.py`（haar_dwt2/haar_idwt2）；`src/model/mean_predictor.py`（reconstruct_lowpass）；`src/model/mean_pretraining.py`（haar_dwt2）；`src/model/loss_terms/residual_frequency.py`、`boundary_frequency.py`、`route_utility_supervision.py`（均 haar_dwt2，后者还消费 router 写入 condition.scalars 的路由诊断）；`src/model/trainer.py`（路由器参数分组 L497–520、`_set_spectral_router_epoch` L854–860、`_router_phase_label` L1037–1053）；脚本侧 `scripts/calibrate_h3_v2_full_timestep_native_null.py`、`estimate_h3_prior_from_png.py`、`build_h3_v2_experiment_configs.py`、`run_prior_anchored_router_100e.py`、`validate_h2/h3/h4_*` 系列、`eval_action_axis_screen.py`。

**跨模块要点**：四个子包到 `slmf_bbdm.py` 的接线集中在三处工厂/分支（`_build_prior`、`_build_noise`、`residual_frequency_mode` 三分支），gabor 输出经 `gabor_routes` 路由表决定流向（adapter/noise/hotspot/loss），是子包间唯一的运行时数据耦合；frequency 的路由器与 trainer 存在双向协作（trainer 推 epoch、router 产出 phase 标签与监控标量）。


---

## 6. 机制验证库 src/mechanism_validation

### 模块概述

本包是整个 SLMF-BBDM 项目"正式机制验证 (formal mechanism validation)"阶段的可复用核心库：围绕假设 H1–H6 提供冻结 (frozen) 的患者级划分、统计推断（bootstrap / sign-flip / 置换检验）、病灶特征溯源与因果干预审计、病灶放大机制审计以及锁定终点的对照训练实验编排。设计模式上贯穿三条原则：(1) fail-closed——任何上游 gate 非 PASS、ID 缺失、患者泄漏都直接抛异常而非静默降级；(2) 纯函数分层——`magnification.py` 完全无文件系统/模型依赖，`magnification_audit.py`、`feature_*.py` 在其上组合应用逻辑；(3) fit/apply 分离——`h4_v2.py` 的统计量只在 development 患者上拟合、外部队列只走 apply 路径。数据流为：CSV manifest → 患者划分/血缘哈希 → 模型前向（采样或特征提取）→ 逐样本指标 → 患者级聚合 → bootstrap CI + 预注册 gate → JSON/CSV 工件。本包自身不含任何 argparse CLI；全部由 `scripts/validate_h*`、`scripts/eval_feature_*.py`、`run_all_mechanism_validations.py` 等脚本驱动（该 runner 由 `configs/mechanism_validation_pipeline_v1.json` 编排，结果落在 `results/mechanism_validation/99_full_pipeline`）。

---

### `src/mechanism_validation/__init__.py`

**职责**：包门面 (facade)。将 `common`、`magnification`、`magnification_audit` 三个子模块的公开符号统一 re-export 到 `src.mechanism_validation` 命名空间，供下游脚本 `from src.mechanism_validation import ...` 一次性导入。

**接口**：无 class/def 定义；仅 import 与 `__all__` 列表。导出常量 `BOOTSTRAP_REPLICATES / CALIBRATION_FRACTION / PARTITION_SEED`；导出函数 `decision_guardrails, load_json, paired_patient_bootstrap, patient_partition, read_manifest, sign_flip_p, write_json, write_mechanism_manifest, backproject_crop, bootstrap_patient_balanced_slope, build_primary_gate, compute_gate_statistics, couple_noise_to_crop, crop_resize, lesion_crop_box, patient_balanced_slope, run_frozen_magnification_audit, validate_cohort, write_audit_artifacts`；导出数据类 `AuditTables, CohortArrays, CropBox, MagnificationView, PatientBootstrapResult`。注意 `feature_*`、`h4_v2`、`internal_cv`、`model_experiments` 不在门面内，必须按子模块路径导入。

**依赖**：
- 内部: `src.mechanism_validation.common`、`src.mechanism_validation.magnification`、`src.mechanism_validation.magnification_audit`
- 外部: 无

**输入**：无运行时输入（纯导入期模块）。

**输出**：无文件输出；建立 `__all__` 导出面。

**处理过程**：
1. 从 `.common` 导入统计与 manifest 工具；
2. 从 `.magnification` 导入几何/噪声/斜率原语与三个 dataclass；
3. 从 `.magnification_audit` 导入审计应用层 API；
4. 以 `__all__`（26 个符号）固定公开契约，避免隐式导出。

---

### `src/mechanism_validation/common.py`

**职责**：全库共享的 fail-closed 基础设施：原子 JSON/CSV 写入、SHA256 血缘哈希、H1 锁定患者划分的精确复现、患者级 bootstrap 与 sign-flip 检验、决策护栏 (guardrails) 与上游 PASS 强校验。

**接口**（顶层 def）：
- 常量 `PARTITION_SEED=42`、`CALIBRATION_FRACTION=0.20`、`BOOTSTRAP_REPLICATES=10_000`、`MANIFEST_COLUMNS=("sample_id","patient_id","slice_id","split","cache_path")`
- `canonical_json_sha256(payload) -> str`：排序键 + 紧凑分隔符的规范 JSON 哈希
- `file_sha256(path, block_size=1MB) -> str`
- `write_json(path, payload) -> None`（临时文件 + `os.replace` 原子替换）；`load_json(path) -> dict`（拒绝非对象根）
- `write_csv(path, rows) -> None`；`read_manifest(path) -> list[dict]`（校验必需列、sample_id 唯一、患者不跨 split）
- `patient_partition(rows, *, seed, calibration_fraction) -> dict[patient_id, role]`：精确复现锁定 H1 99/25/31 划分，train → {mechanism_train, calibration}，val → validation
- `partition_sha256 / partition_counts / write_mechanism_manifest(source_rows, partition, output) -> dict`：写出派生 manifest（train/val/test 重映射）并返回 {path, sha256, partition_sha256, counts, mapping}
- `aggregate_patient_rows(rows, metric_names) -> list[dict]`：按 (partition, patient_id) 分组求均值
- `bootstrap_mean(values, *, seed, replicates=10_000) -> dict`（estimate + ci95_low/high）
- `paired_patient_bootstrap(patient_rows, *, partition, left, right, seed, replicates) -> dict`（含 contrast 与 sign_flip_p）
- `sign_flip_p(effects, *, seed, replicates, two_sided=False) -> float`（+1 修正的单/双侧 p）
- `decision_guardrails(*, checkpoint_lineage="PASS") -> dict`
- `require_upstream_pass(path, expected_stage=None) -> dict`：上游 decision JSON 非 PASS 即 `RuntimeError`

**依赖**：
- 内部: 无（被本包其余全部模块及 `src.model.mean_pretraining` 消费）
- 外部: 标准库 `csv/hashlib/json/math/os/collections/pathlib/typing`、`numpy`

**输入**：split-manifest CSV（列须含 `MANIFEST_COLUMNS`）；患者级指标行列表；任意可 JSON 化 payload。

**输出**：写出/读回 JSON 与 CSV 文件；返回划分字典、bootstrap 统计字典、guardrail 字典。

**处理过程**：
1. `read_manifest` 逐项 fail-closed 校验（列缺失/空表/重复 sample_id/患者跨 split 泄漏均抛错）；
2. `patient_partition` 用 `np.random.default_rng(seed)` 对排序后的 train 患者洗牌，前 `round(0.2*N)` 名为 calibration；
3. `write_mechanism_manifest` 将 role→split 映射回写 CSV 并记录文件 SHA256 与计数；
4. `bootstrap_mean` 过滤非有限值后整批重采样（replicates×N 索引矩阵）取 2.5%/97.5% 分位；
5. `sign_flip_p` 以随机 ±1 符号构造零分布，返回 `(exceed+1)/(replicates+1)`；
6. `decision_guardrails` 固化"验证/测试不驱动训练"等声明字段；
7. `require_upstream_pass` 在进入任何阶段前强制校验上游 gate。

**主要消费脚本**：`scripts/validate_h1_local_spectral_asymmetry.py`、`validate_h2_pathology_excluded_residual.py`、`validate_h3_logsnr_recoverability.py`、`validate_h4_noise_band_calibration.py`、`validate_mechanism_curriculum.py`、`validate_ct_support_head.py`、`validate_artifact_safety.py`、`develop_h4_v2_uncertainty_aware.py`、`export_v2_production_calibration_bundle.py`、`audit_v2_freeze_integrity.py`、`audit_h3_fixed_schedule_inference.py`、`estimate_h3_prior_from_png.py`、`build_h3_v2_experiment_configs.py`、`analyze_h3_v2_experiment.py`、`calibrate_h3_v2_full_timestep_native_null.py`、`run_h4_v2_pipeline.py`、`run_h4_v2_internal_exploratory_pipeline.py`、`pretrain_pet_encoder.py`（用 `patient_partition`/`partition_sha256`）、`pretrain_tiny_segmenter.py`、`validate_perceptual_x0_few_step.py`、`sync_cloud_worktrees.py`、`run_dual_experiment_queue.py`；`src/model/mean_pretraining.py` 用 `canonical_json_sha256`；测试 `tests/test_h2_mechanism.py` 等。

---

### `src/mechanism_validation/feature_causality.py`

**职责**：A2 阶段因果干预审计（RQ-E3："模型是否真的使用 c2 层病灶信息"）。通过对 `model.build_condition_bundle` 做 monkey-patch 上下文管理器，在每一步 DDIM 去噪中修改 `ct_feat_1` 特征图，跑 6 臂干预矩阵并输出患者级对照统计与 M2 gate。

**接口**：
- 常量 `C2_KEY="ct_feat_1"`、`_BACKGROUND_NONINFERIOR_MARGIN=0.05`
- `class patch_build_condition_bundle(model, intervention_fn)`：上下文管理器，`__enter__` 时以包装函数替换实例属性 `build_condition_bundle`（原方法执行后再施加干预），`__exit__` 删除实例属性恢复类级绑定方法；可安全嵌套
- 干预函数（签名 `(bundle, batch) -> bundle`）：`c2_lesion_zero`（病灶区置零，必要性检验）、`c2_background_replace(bundle, batch, ring_width=3)`（环均值填充）、`c2_nonlesion_samearea(bundle, batch, seed=0)`（同面积非病灶区置零，空间特异性对照，优先 `organ_mask`>`mu_map` 做全组织覆盖约束，无有效放置则 no-op 并记录覆盖统计）、`c2_shifted_mask(bundle, batch, shift_fraction=0.25)`（网格相对平移的假手术对照）、`ct_inpaint`（bundle 级 CT 病灶置零，注意不触及 `x_source` 与重提取特征）
- `ct_inpaint_batch(sample) -> sample`：批级干预，修改 `sample["ct"]` 使编码器与 UNet 原始通道同时看到无病灶 CT（完整 CT 病灶依赖消融的正确实现）
- 字典 `INTERVENTIONS`（baseline + 5 个 bundle 级）与 `BATCH_INTERVENTIONS`（仅 ct_inpaint）
- `compute_sample_metrics(sample, target, mask) -> dict[str, float]`：包装 `src.model.trainer._compute_pet_sample_metrics` 并追加 `nonlesion_mae`、`nonlesion_topq_peak_error_norm`（假热点代理）
- `run_causality_audit(model, loader, device, output_dir, *, seeds=(0,1,2,3), interventions=None, max_samples=None, num_sampling_steps=None) -> dict`：完整干预矩阵 → `causal_metrics.csv` / `causal_patient_summary.csv` / `causal_decision.json`

**依赖**：
- 内部: `src.mechanism_validation.common`（bootstrap_mean/sign_flip_p/write_csv/write_json）、`src.mechanism_validation.feature_emergence`（`_patient_keys`、`_dilate`）；延迟导入 `src.model.trainer._compute_pet_sample_metrics`、`src.model.trainer._to_unit_interval`
- 外部: `hashlib/pathlib/typing`、`numpy`、`torch`、`torch.nn.functional`

**输入**：已训练模型（须有 `build_condition_bundle`、`sample()`、bundle 含 `maps["ct_feat_1"]`）；产出 batch 的 loader（batch 含 `ct/pet/mask/meta.patient_id`，可选 `organ_mask/mu_map`）；`output_dir`；seed 序列。

**输出**：`causal_metrics.csv`（逐样本×seed×干预指标）、`causal_patient_summary.csv`（患者×干预均值）、`causal_decision.json`（主终点 `lesion_topq_peak_error_norm` 的 delta bootstrap CI、sign-flip p、M2 gate 合取判定：c2 置零须恶化且两个对照更弱且背景非劣效 ≤0.05）。

**处理过程**：
1. 校验干预名 ⊆ INTERVENTIONS ∪ BATCH_INTERVENTIONS，`model.eval()`；
2. 逐 batch、逐样本取出单样本 dict，`_hash_seed(patient, sample_idx, seed)` 由 SHA256 派生稳定种子；
3. 每个 (sample, seed) 内先 `torch.manual_seed(base_seed)` 再循环全部干预——保证各干预共享同一采样噪声；
4. 批级干预先变换 `sample["ct"]`，bundle 级干预经 `patch_build_condition_bundle` 上下文施加后调用 `model.sample`；
5. `compute_sample_metrics` 计算病灶 TopQ/峰值/质心与背景非病灶指标；
6. 写 CSV 后按 (intervention, patient) 聚合均值，再对每干预构造 vs baseline 的逐患者差值；
7. `bootstrap_mean` + `sign_flip_p` 得 delta CI 与 p；M2 gate 为合取：`c2_worsens ∧ sham_weaker ∧ nonlesion_weaker ∧ background_noninferior`；
8. 写出 decision JSON 并返回。

**主要消费脚本**：`scripts/eval_feature_causality.py`（CLI 入口，另复用 `common.file_sha256/write_json/canonical_json_sha256` 与 `feature_emergence._patient_keys`）；测试 `tests/test_feature_causality.py`（覆盖全部干预与 runner）。

---

### `src/mechanism_validation/feature_emergence.py`

**职责**：A1+A2 阶段的表征分析库：RQ-E1 用共享权重探针的逐层线性 CKA（配对 vs 患者置换 vs 随机初始化基线）检验浅层 CT/PET 特征是否为共享跨模态解剖；RQ-E2 用通道无关激活图检验病灶信息是否在 CT 中层 (c2) 显现。统计单位恒为患者。

**接口**：
- 常量 `FEATURE_KEYS=("ct_feat_0".."ct_feat_3")`
- `compute_linear_cka(X[N,D], Y[N,D]) -> float`：双侧中心化 Gram 矩阵的 HSIC/范数比，N<2 或退化分母返回 NaN
- `extract_ct_encoder_features(model, batch) -> dict`：同一训练好的 `ct_encoder` 分别跑 CT 与 PET（明确声明为 shared-weight representation probe，非原生双模态编码器）
- `_patient_keys(batch) -> list[str]`：三种 collate 形态下提取 `meta.patient_id`；缺失时拒绝位置 ID 回退并抛错
- `cross_modal_similarity_analysis(model, loader, device, output_dir, *, seed=42, num_layers=None, n_permutations=200) -> dict`：输出 `layer_metrics.csv`、`patient_summary.csv`、`cross_modal_similarity.json`（含 S1 gate：c1 层 delta CKA 的 CI 下界 >0 ∧ 置换 p<0.05 ∧ 配对 CKA > 随机初始化基线）
- `_random_init_encoder(model) -> Module`：deepcopy 后 Kaiming 重置 Conv2d 的随机初始化镜像编码器
- `channel_agnostic_activation(feat[N,C,H,W]) -> [N,1,H,W]`：逐通道 z-score 后 RMS 聚合
- `lesion_ring_contrast(activation, mask, ring_width=3) -> float`：log(病灶均值/环均值)
- `_dilate(mask2d, iterations)`：max_pool2d 迭代膨胀
- `lesion_discriminability(activation, mask, body_mask=None, ring_width=6) -> {auroc, auprc, lesion_pixels, background_pixels}`：负类限制在体内环带，避免 padding 造成平凡分离
- `lesion_centroid_hit(activation, mask) -> {centroid_hit}`：激活质心落入病灶掩码的指示
- `lesion_emergence_analysis(model, loader, device, output_dir, *, fixed_t=0, lesion_quantiles=(0.25,0.5,0.75), ring_width=3, frozen_quartiles=None) -> dict`：输出 `cohort.csv`、`lesion_emergence_metrics.csv`、`lesion_emergence.json`（含 M1 gate 合取：c2 对比 CI>0 ∧ c2−c1 配对 delta CI>0 ∧ Q1 最小病灶分层一致）

**依赖**：
- 内部: `src.mechanism_validation.common`（bootstrap_mean/sign_flip_p/write_csv/write_json）；模型侧仅依赖传入对象的 `ct_encoder` / `build_condition_bundle` 属性契约（指向 `src.model.slmf_bbdm` 的模型）
- 外部: `numpy`、`torch`、`torch.nn.functional`、`sklearn.metrics`（延迟导入 roc_curve/precision_recall_curve/auc）、`copy`

**输入**：模型（含 `ct_encoder`、`build_condition_bundle`）；loader 的 batch（`ct/pet/mask/meta.patient_id`）；病灶面积分位数（默认从当前 loader 冻结，held-out 评估应传入 `frozen_quartiles`）；固定时间步 `fixed_t`。

**输出**：三组 CSV + 两个决策 JSON（cross_modal_similarity.json 的 S1 gate、lesion_emergence.json 的 M1 gate），返回同构 dict。

**处理过程**（`lesion_emergence_analysis` 为例）：
1. 冻结病灶面积四分位（provided 或 loader_split，显式记录 frozen_on）；
2. 逐 batch 构造 `build_condition_bundle(batch, t_batch)` 取 `ct_feat_0..3`；
3. 激活图上采样到全分辨率（避免小病灶被深层下采样抹掉），同时计算掩码"生存率"（往返下/上采样后仍覆盖的原始病灶像素比例）；
4. 逐层计算 ring contrast / AUROC / AUPRC / 质心命中，按面积四分位打标；
5. 写 cohort 与 metric CSV；
6. 按 (layer, quartile) 患者级聚合 ring contrast 并 bootstrap；
7. M1 gate：c2 vs c1 配对差值 bootstrap CI 与 sign-flip p，加 Q1 方向一致性合取判定；
8. 写 JSON 并返回。

**主要消费脚本**：`scripts/eval_feature_emergence.py`（CLI 入口，同时调用 `feature_provenance.audit_feature_provenance`）；`scripts/eval_feature_causality.py`（复用 `_patient_keys`）；测试 `tests/test_feature_emergence.py`、`tests/test_feature_cli_integration.py`。

---

### `src/mechanism_validation/feature_provenance.py`

**职责**：A0 阶段静态特征溯源审计：在任何特征数字被解读之前，钉死特征来源（哪个编码器、权重是否共享、checkpoint SHA256、输入归一化指针、逐层张量名/形状、时间步策略），并强制措辞规则——同一 CT 编码器跑 PET 只能称"shared-weight representation probe"，禁止称"原生双模态编码器"。纯静态、无前向传播，可在 CPU 上廉价运行。

**接口**：
- `audit_feature_provenance(model, config, checkpoint_path, output_dir) -> dict`：主入口；校验 checkpoint 存在后返回溯源 payload，并写 `output_dir/feature_provenance.json`
- `sha256_bytes(data: bytes) -> str`：通用字节哈希
- 私有：`_layer_shapes(model)`（从 `ct_encoder` 的 `stem_fuse/down1..down3` 推断 c1..c4 层通道数与 `image_size//2^i` 空间尺寸）、`_encoder_parameters(model)`（类名/参数量/可训练量）、`_input_normalisation(config)`（data.mode/image_size/cache_dir/split_manifest/augment/cache_lineage/dataset_contract；像素归一化在 cache 构建期即 `src/data/png_cache.py --pet-invert`，由 lineage 记录）

**依赖**：
- 内部: `src.mechanism_validation.common`（canonical_json_sha256/file_sha256/write_json）；仅反射访问模型的 `ct_encoder/image_size` 属性（`_CTEncoder` 结构契约）
- 外部: `hashlib/pathlib/typing`、`torch`

**输入**：实例化模型（含 `ct_encoder`）；完整 config dict（`data.*` 键）；checkpoint 文件路径；输出目录。

**输出**：`feature_provenance.json`，schema `lesion_feature_emergence_v1`：`feature_source`（encoder 参数、shared_weight=True、probe_kind、逐层 layers 表、image_size）、`checkpoint`（path/sha256/weights_are_ema 留空由 CLI 回填）、`config`（sha256/input_normalisation）、`timestep_policy`（说明 ct_feat 与 t 无关、随每步重算；fixed_t_for_extraction 留空）、`decision`（dual_modal_encoder_claim=False）。

**处理过程**：
1. 校验 checkpoint 是文件并计算其 SHA256，config 计算规范 JSON 哈希；
2. `_layer_shapes` 反射枚举编码器四块，取声明 `out_channels` 或 Sequential 末层 Conv2d 通道；
3. 组装层级表（name=c{i+1}、tensor_key=ct_feat_{i}、channels、spatial、downsample_scale=2^i）；
4. `_input_normalisation` 摘取 data config 指针并注明像素归一化由 cache lineage 承载；
5. 填入 timestep_policy 说明与硬编码 decision 声明；
6. `write_json` 原子写出并返回 payload。

**主要消费脚本**：`scripts/eval_feature_emergence.py`（A0 步骤，先于 A1/A2 运行）；测试 `tests/test_feature_provenance.py`（含与 `common.file_sha256` 的交叉校验）。

---

### `src/mechanism_validation/h4_v2.py`

**职责**：H4-v2 不确定性感知证据链的 fit/apply 双路径库：在 mechanism_train 患者上拟合总体统计与 Ridge 系数、在 development calibration 上冻结置信度与亚组阈值，四个预注册对照 (`no_route / h3_fixed_schedule / original_evidence / uncertainty_aware`) 的 MSE 比较与探针 (probe) 自哈希封印。

**接口**：
- 常量 `BANDS=("l2_lh","l2_hl","l2_hh","l1_lh","l1_hl","l1_hh")`、`MODEL_ORDER`（4 对照顺序）、`RIDGE_ALPHA=1.0`、`NEIGHBOR_RADIUS=1`、`PATIENT_SHRINKAGE_K=60.0`、`CONFIDENCE_QUANTILE=0.25`、`MIN_CONFIRMATION_PATIENTS=40`、`MIN_SUBGROUP_PATIENTS=10`、`MIN_ACTIVE_COVERAGE=0.50`、`SUBGROUP_MARGIN_FRACTION=0.05`
- `normalize_ids(frame) -> DataFrame`：patient_id 补零至 3 位、sample_id 至 6 位并派生 slice_id
- `add_context_features(rows) -> DataFrame`：加相邻切片 (±1) 与同取向跨尺度 peer 的证据上下文 → `context_evidence/context_dispersion/context_count/neighbor_count/scale_peer_present`
- `fit_population_stats(context, *, partition="mechanism_train") -> dict["band|t", {median, scale}]`（MAD 稳健尺度）；`apply_hierarchical_calibration(context, population_stats, *, shrinkage_k=60) -> DataFrame`（总体 z + 患者带内收缩偏移 + 置信度加权证据列）
- `fit_original_standardization(frame, *, partition) -> {mean, scale}`
- `fit_comparators(calibrated, original_standardization, *, partition, alpha) -> dict`：两个冻结查找表对照 + 两个 Ridge（对 H3 固定表的加性校正，特征为证据/证据×log_snr/证据×band one-hot）
- `apply_comparators(calibrated, *, models, original_standardization, confidence_threshold) -> DataFrame`：低于置信阈值时 uncertainty_aware 回退到 h3_fixed_schedule（router_active/abstained 列）
- `patient_errors(predictions) -> DataFrame`：逐患者 MSE（mse_{model}）
- `fit_patient_attributes(h1_sample_rows) -> DataFrame`；`freeze_subgroup_spec(attributes, patient_error_frame) -> dict`（few_slices/small_lesion/low_contrast 三亚组阈值取 train 分位 0.25）；`assign_subgroups(attributes, subgroup_spec) -> DataFrame`
- `effect_statistics(patient_error_frame, *, partition, left, right, seed, replicates=10_000) -> dict`（MSE 差 bootstrap + sign-flip）；`development_diagnostics(..., partition="validation", seed=20260730) -> dict`（4 组预注册成对比较）
- `seal_probe(payload) -> dict` / `validate_probe(payload) -> None`（自哈希 + schema_version==2 + band/model 顺序校验）；`patient_set_sha256(patient_ids) -> str`

**依赖**：
- 内部: `src.mechanism_validation.common`（bootstrap_mean/canonical_json_sha256/sign_flip_p）；被 `internal_cv.py` 反向依赖 `normalize_ids`
- 外部: `hashlib/json/math/collections/typing`、`numpy`、`pandas`、`sklearn.linear_model.Ridge`

**输入**：H4-v2 长表行（必需列 patient_id/sample_id/band/timestep/log_snr/noise_calibrated_evidence/recoverability/partition/slice_id）；H1 样本行（partition/patient_id/sample_id/mask_area/pet_ll2_lesion_ring_log_ratio）。

**输出**：返回 DataFrame 与可 JSON 化的冻结模型/spec dict；探针含 `probe_sha256` 自哈希字段。

**处理过程**（fit 路径）：
1. `normalize_ids` 统一 ID 格式后 `add_context_features` 构建上下文中位数与离散度；
2. `fit_population_stats` 在 mechanism_train 上按 (band, timestep) 拟合稳健 median/MAD；
3. `apply_hierarchical_calibration` 计算 population_z、患者带内收缩偏移与几何均值型 evidence_confidence；
4. `fit_comparators` 拟合 4 对照（Ridge 目标为 recoverability 减去 H3 固定表预测的残差）；
5. `apply_comparators` 打分并按置信度门控路由回退；
6. `patient_errors` → `effect_statistics`/`development_diagnostics` 做患者级 MSE 成对推断；
7. `freeze_subgroup_spec` + `assign_subgroups` 冻结并应用亚组定义；
8. `seal_probe`/`validate_probe` 封印与校验探针一致性。

**主要消费脚本**：`scripts/develop_h4_v2_uncertainty_aware.py`、`scripts/export_v2_production_calibration_bundle.py`、`scripts/freeze_h4_v2_internal_cv_plan.py`（用 BANDS/normalize_ids）、`scripts/freeze_v2_main_integration_contract.py`、`scripts/validate_h4_v2_external_confirmation.py`、`scripts/validate_h4_v2_internal_cv.py`；测试 `tests/test_h4_v2_mechanism.py`。

---

### `src/mechanism_validation/internal_cv.py`

**职责**：H4-v2 内部探索性分析的冻结患者级嵌套交叉验证划分：只用外层评估前可得的患者属性（切片数、平均病灶面积、平均 PET 对比度 + manifest split/原 partition 两个类别变量）做均衡分组搜索，显式排除可恢复性目标与对照误差，防信息泄漏。

**接口**：
- 常量 `BALANCE_CONTINUOUS_FIELDS=("slice_count","mean_mask_area","mean_pet_ll2_log_contrast")`、`BALANCE_CATEGORICAL_FIELDS=("manifest_split","original_partition")`
- `seal_mapping(payload, *, hash_field) -> dict` / `validate_sealed_mapping(payload, *, hash_field) -> str`：任意映射的自哈希封印/校验
- `patient_balance_attributes(h1_rows) -> DataFrame`：按患者聚合三个预注册均衡变量，校验患者不跨 split/partition 且连续量全部有限
- `balance_feature_matrix(attributes, *, quantile_bins) -> (matrix, feature_names)`：连续量的百分位 rank + 分位箱指示 + 类别 one-hot，均标准化
- `balanced_partition_search(attributes, *, folds, seed, candidates, quantile_bins) -> (assignments, diagnostics)`：在固定大小随机候选中选折均值平方目标最小者，返回 patient_id→fold 与诊断（目标值、最大绝对标准化特征均值、特征名、折患者数）
- `fold_balance_table(attributes, assignments, *, fold_column="fold") -> DataFrame`：每折均值/中位数/类别计数表
- `partition_fingerprint(outer_assignments, nested_roles) -> str`：整个嵌套方案的规范 JSON 哈希
- `build_nested_roles(attributes, outer_assignments, *, outer_folds, inner_folds, inner_seed, inner_candidates, quantile_bins) -> (DataFrame, list[diagnostics])`：每外层折独立冻结内层 mechanism_train/calibration 角色（calibration 折 = outer_fold % inner_folds）
- `assert_patient_partition_integrity(outer_assignments, nested_roles, *, patient_ids, outer_folds) -> None`：覆盖完整性、无重复、三角色互斥、外层留出集一致

**依赖**：
- 内部: `src.mechanism_validation.common.canonical_json_sha256`、`src.mechanism_validation.h4_v2.normalize_ids`
- 外部: `typing`、`numpy`、`pandas`

**输入**：H1 长表行（须含 patient_id/sample_id/manifest_split/partition/mask_area/pet_ll2_lesion_ring_log_ratio）；外层折数/内层折数/种子/候选数/分位箱数。

**输出**：划分 DataFrame（patient_id, fold / outer_fold, role, inner_fold）、诊断 dict 列表、方案指纹字符串；纯内存返回，不写文件（落盘由调用脚本负责）。

**处理过程**：
1. `patient_balance_attributes` 聚合并 fail-closed 校验（患者跨 split、非有限值均抛错）；
2. `balance_feature_matrix` 构造 rank/quantile/category 三类均衡特征；
3. `balanced_partition_search` 枚举 candidates 个种子排列，目标 = 各折特征均值平方的平均，取最小；
4. `build_nested_roles` 对每外层折在其余患者上重复搜索，指派 calibration/mechanism_train/outer_evaluation 三角色；
5. 校验角色表行数 = 患者数 × 外层折数且无重复；
6. `assert_patient_partition_integrity` 终检三角色互斥与外层留出集一致；
7. `partition_fingerprint` 输出可封印的方案哈希。

**主要消费脚本**：`scripts/freeze_h4_v2_internal_cv_plan.py`、`scripts/validate_h4_v2_internal_cv.py`；测试 `tests/test_h4_v2_mechanism.py`。

---

### `src/mechanism_validation/magnification.py`

**职责**：病灶放大 (magnification) 审计的纯几何/噪声/统计原语层：刻意零文件系统、零 checkpoint、零模型、零 CLI 依赖，使同一原语可先被冻结机制审计复用、审计通过后再被训练管线复用。

**接口**：
- `@dataclass(frozen=True) CropBox(top, left, size)`（含 bottom/right 属性与负值/非正校验）
- `@dataclass(frozen=True) MagnificationView(box, output_size)`（factor = output_size/box.size）
- `@dataclass(frozen=True) PatientBootstrapResult`（patients/requested_replicates/valid_replicates/intercept/slope/slope_ci95_low/high/x_reference/reference_prediction/reference_ci95_low/high）
- `lesion_crop_box(mask, crop_size) -> CropBox`：包含完整非空病灶的方形裁剪（病灶外接框超限或掩码为空抛错，越界由 min/max 夹紧并二次断言不截断病灶）
- `crop_resize(tensor, box, *, output_size, mode) -> Tensor`：声明式裁剪 + 双线性/最近邻 resize，接受 [H,W]/[C,H,W]/[B,C,H,W]
- `backproject_crop(zoom_tensor, box, *, canvas_size, mode, base=None) -> Tensor`：放大结果缩回裁剪尺寸并贴回画布（可叠加在 base 全图预测上）
- `couple_noise_to_crop(full_noise, box, *, output_size, epsilon=1e-6) -> Tensor`：把全图噪声裁剪/缩放后按样本标准化的"解剖学索引"放大噪声
- `patient_balanced_slope(patient_ids, x, y) -> float`：每患者总权重为 1 的加权最小二乘斜率（≥2 患者、≥2 有限观测，方差退化抛错）
- `bootstrap_patient_balanced_slope(patient_ids, x, y, *, seed, replicates, x_reference=None) -> PatientBootstrapResult`：患者簇 bootstrap，x_reference 缺省为患者均值的均值；逐 draw 拟合失败则跳过，全部退化抛错

**依赖**：
- 内部: 无（是被 `magnification_audit.py` 消费的最底层）
- 外部: `dataclasses/typing`、`numpy`、`torch`、`torch.nn.functional`

**输入**：掩码张量（末两维为空间维）；任意 2/3/4 维图像张量；(patient_ids, x, y) 平行序列。

**输出**：CropBox / 张量 / 斜率 float / PatientBootstrapResult，全部纯内存。

**处理过程**（`bootstrap_patient_balanced_slope`）：
1. `_linear_inputs` 过滤非有限点并断言 ≥2 患者；
2. 全量数据做患者平衡加权拟合得到点估计 intercept/slope；
3. 确定参考 x（患者内均值再平均）；
4. 循环 replicates 次：有放回整患者重采样，拼成新伪患者单元；
5. 对每个 draw 重新平衡拟合，退化 draw 跳过并计数；
6. 对斜率与参考点预测取 2.5%/97.5% 分位得 CI；
7. 组装不可变 result dataclass 返回。

**主要消费脚本**：`src/mechanism_validation/magnification_audit.py`（核心消费者）；`scripts/audit_magnification_consistency.py`（用 `lesion_crop_box` + audit API）；测试 `tests/test_magnification_mechanism.py`、`tests/test_magnification_audit.py`。

---

### `src/mechanism_validation/magnification_audit.py`

**职责**：冻结病灶放大机制审计的应用层：把 `magnification` 的几何/统计原语组合成配对"全图 vs 放大"推理表、只插值往返对照表与预声明 gate 判定；自身不加载 checkpoint 与配置（由调用方注入模型与 cohort）。

**接口**：
- `@dataclass(frozen=True) CohortArrays(cohort: DataFrame, ct, target, mask: np.ndarray, sample_ids)`
- `@dataclass(frozen=True) AuditTables(sample_metrics, patient_metrics, roundtrip_metrics: DataFrame)`
- `validate_cohort(data: CohortArrays) -> None`：形状 [N,1,H,W] 一致、sample_id 对齐且唯一、全有限、每样本病灶非空、`lesion_area` 与掩码像素数严格一致
- `run_frozen_magnification_audit(*, model, cohort, device, seeds, crop_sizes, input_size, num_steps, batch_size, amp=False) -> AuditTables`：配对全图/放大推理 + 插值往返对照（模型须实现 `sample({"ct":...}, num_steps=..., initial_noise=...) -> {"synthetic_pet": tensor}`）
- `compute_gate_statistics(tables, *, crop_sizes, bootstrap_seed, bootstrap_replicates) -> dict[crop_size, statistics]`：相对/绝对 TopQ 改进对 log2 病灶面积的平衡斜率（x_reference=Q25 面积）、上下文假热点恶化率、往返占比
- `build_primary_gate(statistics_by_crop, *, primary_crop_size) -> dict`：五项预注册检查（斜率 CI 上界<0、Q25 相对改进≥3% 且 CI 下界>0、上下文假热点恶化≤5%、往返占比≤25%），主裁剪尺寸 FAIL 则整体 FAIL，敏感性分析不能拯救
- `write_audit_artifacts(output, *, tables, manifest, gate) -> None`：向新建目录（`mkdir(exist_ok=False)` 防覆盖）写 sample/patient/roundtrip CSV、`audit_manifest.json`、`gate.json`（NaN→null，allow_nan=False）

**依赖**：
- 内部: `src.mechanism_validation.magnification`（六个原语）；`scripts.evaluate` 的 `compute_normalized_lesion_metrics`、`compute_target_relative_false_hotspots`（src 包反向依赖 scripts 目录，需仓库根在 sys.path）
- 外部: `dataclasses/json/pathlib/typing`、`numpy`、`pandas`、`torch`

**输入**：CohortArrays（identity 表 + [N,1,input_size,input_size] 的 ct/target/mask）；唯一 seed 序列与 crop_sizes 序列（1..input_size）；DDIM 步数、批大小、amp 开关。

**输出**：AuditTables 三表；gate dict（decision "PASS"/"FAIL"、阈值、敏感性结果）；四个审计工件文件。

**处理过程**：
1. `validate_cohort` 全量 fail-closed 校验后进入 `torch.inference_mode()`；
2. 每批先为每个 crop_size 计算 `lesion_crop_box`，并对 target 做"裁剪→放大→回投"往返，记录插值下限对照指标；
3. 每个 seed 用 `(seed + 1_000_003*sample_index)` 派生确定性全图噪声，采样全图预测与逐样本病灶指标；
4. 每 crop_size 构造 zoom_ct 与 `couple_noise_to_crop` 的配对 zoom_noise，采样放大预测；
5. 放大结果 `backproject_crop` 回贴到全图预测上，与同一 target/mask 计算病灶指标；
6. 在裁剪框内单独计算全图/放大的上下文假热点密度；
7. 汇总 sample_metrics（full/zoom/绝对与相对改进/上下文热点），`_patient_table` 聚合患者层；
8. `compute_gate_statistics` → `build_primary_gate` → `write_audit_artifacts` 完成判定与落盘。

**主要消费脚本**：`scripts/audit_magnification_consistency.py`；测试 `tests/test_magnification_audit.py`。

---

### `src/mechanism_validation/model_experiments.py`

**职责**：H5/H6 锁定固定终点 (fixed-endpoint) 模型实验编排：为 5 个预声明变体生成自含正式配置、子进程调用 `scripts/train_v2.py` 训练与 `scripts/evaluate.py` 全量评估、校验 checkpoint 血缘/epoch/内嵌 config/变体标记，并提供患者配对效应统计。明确禁止 best-checkpoint 选择、早停、验证驱动改道。

**接口**：
- 常量 `FORMAL_SEED=4242`、`EVALUATION_SEED=42`、`DEFAULT_EPOCHS=50`、`DEFAULT_MC_STEPS=20`、`VARIANTS=("null_reference","full","no_ct_support","no_recoverability","no_artifact_safety")`
- `resolve(root, value) -> Path`、`load_yaml(path) -> dict`、`write_yaml(path, payload) -> None`（临时文件原子替换）、`read_csv_rows(path) -> list[dict]`
- `prepare_variant_config(*, root, base_config, cache_dir, cache_lineage, contract, mechanism_manifest, mean_checkpoint, curriculum, variant, epochs, seed=FORMAL_SEED) -> dict`：深拷贝基配置后以点路径覆盖（experiment/data/modules.conditional_mean 冻结均值模型/training 固定 50 epoch/runtime 关闭 best/早停/评估间隔>epoch）+ `_variant_overrides` 的路由器差异 + `formal_mechanism` 血缘块
- `ensure_variant(*, root, work_dir, base_config, cache_dir, cache_lineage, contract, mechanism_manifest, mean_checkpoint, curriculum, variant, epochs=50, mc_steps=20, force=False) -> dict`：主入口；强制 CUDA（CPU 冒烟 profile 改变图像尺寸，正式 gate 禁用）；训练（或复用已有 `ckpt_epoch{E:04d}.pt`）→ `_validate_checkpoint` → 对 calibration(val)/validation(test) 两个 split 评估（带 provenance JSON 的可复用判定）→ 写 `work_dir/runs/{variant}.json`
- `load_evaluation(record, role) -> dict`；`paired_patient_effects(left_eval, right_eval, *, metric, lower_is_better) -> list[row]`（匹配患者交集，效应统一编码为"正值=左边更好"）；`effect_statistics(effects, *, seed, replicates=10_000) -> dict`；`calibration_absolute_margin(effects, *, quantile=0.95) -> float`（≥5 名 calibration 患者否则 RuntimeError）
- `load_formal_context(*, root, contract_path, h2_decision_path, curriculum_decision_path, cache_dir, cache_lineage, base_config) -> dict`：交叉校验 H2 与 curriculum 决策均 PASS 且 contract SHA 一致、mechanism manifest/下游均值 checkpoint/冻结 curriculum 的 SHA256 逐一比对，返回共享不可变输入

**依赖**：
- 内部: `src.mechanism_validation.common`（bootstrap_mean/canonical_json_sha256/file_sha256/sign_flip_p/write_json）、`src.data.lineage`（load/validate_checkpoint_data_lineage）、`src.model.config_utils`（load_full_config/resolve_runtime_profile）；子进程调用 `scripts/train_v2.py`、`scripts/evaluate.py`
- 外部: `copy/csv/hashlib/json/os/subprocess/sys/pathlib/typing`、`numpy`、`torch`、`yaml`

**输入**：基配置 YAML、cache 目录与 lineage、dataset contract JSON、H2 决策目录（mechanism_split_manifest.csv + analysis_spec.json）、冻结均值 checkpoint、curriculum JSON（router_epoch_schedule/loss_active_tau_max/ct_support_head）、变体名。

**输出**：`work_dir/configs/{variant}.yaml`、`work_dir/logs/*.log`（流式训练/评估日志）、`checkpoints/{experiment}/ckpt_epoch{E:04d}.pt`、`work_dir/evaluations/{variant}/{role}.json` + `{role}_provenance.json`、`work_dir/runs/{variant}.json` 记录（config/checkpoint/evaluations 的路径与 SHA256）；函数返回 record/效应统计 dict。

**处理过程**（`ensure_variant`）：
1. 断言 CUDA 可用，否则拒绝（CPU smoke profile 改尺寸，正式 gate 禁用）；
2. `prepare_variant_config` 深拷贝基配置并施加公共 + 变体覆盖，写入 formal_mechanism 血缘块；
3. 写出 YAML，`load_checkpoint_data_lineage` 解析 sealed lineage，失败即拒绝；
4. checkpoint 缺失或 force 时以 `_run_streaming` 子进程跑 `scripts/train_v2.py`（逐行转发 stdout 到控制台与日志）；
5. `_validate_checkpoint`：血缘校验、epoch 精确等于 epochs、内嵌 config 规范哈希 == resolve_runtime_profile 后的期望哈希、formal_mechanism.variant 匹配；
6. 对 val/test 两个 split：构造期望 provenance，若已有评估的 provenance 完全一致且未 force 则复用，否则子进程跑 `scripts/evaluate.py`（--weights ema --seed 42 --mc-steps）并写 provenance；
7. 校验评估 JSON 的 num_patients>0，汇总路径与 SHA256 写 `runs/{variant}.json`；
8. 下游用 `paired_patient_effects` + `effect_statistics` + `calibration_absolute_margin` 完成 H5/H6 的患者级配对推断。

**主要消费脚本**：`scripts/validate_h5_router_role_separation.py`、`scripts/validate_h6_final_integration.py`、`scripts/validate_artifact_safety.py`、`scripts/validate_perceptual_x0_few_step.py`（均导入本模块并依赖 `common`）。

---

### 模块依赖小结

**包内依赖图**（grep 查证 `from src.mechanism_validation` / 相对导入）：
- `common.py` 为零依赖底座，被 `h4_v2`、`internal_cv`、`magnification_audit`（间接）、`model_experiments`、`feature_causality`、`feature_emergence`、`feature_provenance` 全部消费；
- `h4_v2 → common`；`internal_cv → common + h4_v2.normalize_ids`；`magnification_audit → magnification`；`model_experiments → common`；`feature_causality → common + feature_emergence(_patient_keys/_dilate)`；`feature_emergence → common`；`feature_provenance → common`；`__init__ → common + magnification + magnification_audit`。

**对外跨层引用（值得注意的反向依赖）**：
- `magnification_audit.py` 从 `scripts.evaluate` 导入两个指标函数——src 包反向依赖 scripts 目录，要求仓库根位于 sys.path；
- `feature_causality.py` 延迟导入 `src.model.trainer` 的私有函数 `_compute_pet_sample_metrics`、`_to_unit_interval`，保证与训练评估指标口径一致；
- `model_experiments.py` 引用 `src.data.lineage` 与 `src.model.config_utils`，并以子进程驱动 `scripts/train_v2.py`、`scripts/evaluate.py`；
- `src.model.mean_pretraining.py` 反向消费 `common.canonical_json_sha256`（机制库不仅服务验证，也进入预训练血缘）。

**外部消费方（按模块，均经 grep 验证）**：
- `common`：约 25+ 个 `scripts/validate_h1~h6`、`audit_*`、`develop/export/freeze_*`、`run_h4_v2_*`、`pretrain_pet_encoder`、`pretrain_tiny_segmenter`、`validate_perceptual_x0_few_step`、`run_dual_experiment_queue`、`sync_cloud_worktrees` 等脚本及 `src.model.mean_pretraining`；
- `magnification`/`magnification_audit`：`scripts/audit_magnification_consistency.py`、`tests/test_magnification_mechanism.py`、`tests/test_magnification_audit.py`；
- `feature_emergence`/`feature_causality`/`feature_provenance`：`scripts/eval_feature_emergence.py`、`scripts/eval_feature_causality.py` 与对应 tests；
- `h4_v2`/`internal_cv`：`scripts/develop_h4_v2_uncertainty_aware.py`、`export_v2_production_calibration_bundle.py`、`freeze_h4_v2_internal_cv_plan.py`、`freeze_v2_main_integration_contract.py`、`validate_h4_v2_external_confirmation.py`、`validate_h4_v2_internal_cv.py`、`tests/test_h4_v2_mechanism.py`；
- `model_experiments`：`scripts/validate_h5_router_role_separation.py`、`validate_h6_final_integration.py`、`validate_artifact_safety.py`、`validate_perceptual_x0_few_step.py`。
- `scripts/run_all_mechanism_validations.py` 本身不直接 import 本包，而是经 `configs/mechanism_validation_pipeline_v1.json` 编排上述脚本，汇总结果至 `results/mechanism_validation/99_full_pipeline`。另注意 `artifacts/pfm_simple_cloud_*` 目录下存在本包与脚本的冻结副本（云运行同步产物），非活跃源码。


---

## 7. 训练/评估入口与核心实验脚本

### 模块概述

本组文件构成 SLMF-BBDM 仓库的"入口层"与"实验编排层"：`scripts/train_v2.py` 是唯一的正式训练 CLI（配置解析 → 模型构建 → DataLoader → Trainer），`scripts/evaluate.py` 是独立评估 CLI（图像质量 / 病灶保真 / SUV 临床 / 不确定性 / 失败检测全量表）；`src/scripts/` 下的同名文件是对前者的薄再导出模块包装（哈希不同即源于此），另含三个一次性诊断小工具与一个数据划分重同步脚本。`scripts/` 下其余文件是"计划驱动（plan-driven）"的消融流水线编排器：读取 YAML 计划 → 生成 train/eval 子进程命令清单（manifest）→ 按 stage 顺序执行 `train_v2.py`/`evaluate.py` 子进程 → 用硬门控（hard gates）+ 病灶加权复合分筛选晋级变体 → 产出患者配对 bootstrap 对比报告。设计模式上：入口脚本采用"配置对象直传 + 延迟导入"的薄入口模式；编排器采用"命令清单 + 幂等跳过 + dry-run"的子进程编排模式；V4→V5 编排器之间通过复用 `run_frequency_ablations` 的底层函数（`_run_entries`、`passes_hard_gates` 等）形成分层继承。数据流：YAML 计划/实验配置 → 子进程训练产出 `checkpoints/<experiment>/ckpt_*.pt` + `resolved_config.yaml` + `run_metadata.json` → 子进程评估产出 `results/**/*.json` → 门控/配对统计产出 `promotion_decision.json`、`leaderboard.csv`、`paired_comparison.json`。

---

### `src/scripts/check_ts.py`

（`scripts/check_ts.py` 与本文件 SHA256 完全相同：151F5659…，二者为同一文件的副本，此处仅详述一份。）

**职责**：一次性环境诊断脚本，内省 TotalSegmentator v2 的 `python_api` 模块，打印其公开名称、函数签名与类清单，用于确认依赖 API 是否可用。

**接口**：无顶层 class/def，纯顶层脚本。用法：`python src/scripts/check_ts.py`（或 `python scripts/check_ts.py`），无 CLI 参数。

**依赖**：
- 内部: 无
- 外部: totalsegmentator.python_api、inspect

**输入**：无文件输入，仅反射读取已安装的 totalsegmentator 包。

**输出**：stdout 三段清单——公开名称列表、函数及其 `inspect.signature` 签名、类列表。

**处理过程**：
1. `import totalsegmentator.python_api as api`；
2. 过滤下划线开头的私有名得到 `public`；
3. 逐个打印公开名称；
4. 对可调用且非类的对象尝试打印签名（失败则退化为 `name(...)`）；
5. 打印所有 `inspect.isclass` 的类名。

---

### `src/scripts/evaluate.py`

**职责**：对 `scripts/evaluate.py`（正式评估 CLI）的模块包装：再导出其全部符号，使评估逻辑可按 `src.scripts.evaluate` 路径被 import，并支持 `python src/scripts/evaluate.py` 直接运行。SHA256 与 `scripts/evaluate.py` 不同（669A…/DC52… 差异即"包装 vs 实现"）。

**接口**：`from scripts.evaluate import *` 与 `from scripts.evaluate import main`；`__main__` 分支调用 `raise SystemExit(main())`。CLI 用法与 `scripts/evaluate.py` 完全一致（见下一条目）。

**依赖**：
- 内部: scripts.evaluate（canonical 实现）
- 外部: 无（间接依赖 canonical 模块的外部库）

**输入**：同 `scripts/evaluate.py`（--config/--checkpoint 等 CLI 参数）。

**输出**：同 `scripts/evaluate.py`（评估 JSON 报告与 stdout 报表）。

**处理过程**：
1. 通配再导出 `scripts.evaluate` 命名空间；
2. 显式再导出 `main`（避免 `*` 漏导）；
3. 作为脚本运行时直接委托 `main()`。

---

### `src/scripts/gpu_info.py`

（`scripts/gpu_info.py` 与本文件 SHA256 完全相同：03C426F8…，仅详述一份。）

**职责**：GPU 环境探针，打印 CUDA 可用性、设备数、设备名、bf16 支持与显存总量，用于训练前快速确认运行时。

**接口**：无 class/def，纯顶层脚本。用法：`python src/scripts/gpu_info.py`，无参数。

**依赖**：
- 内部: 无
- 外部: torch

**输入**：无文件输入；读取本机 CUDA 驱动状态。

**输出**：stdout 五行：CUDA available / Device count / Device name / bf16 support / Memory (GB)；无 CUDA 时打印 `Device: CPU`。

**处理过程**：
1. 检查 `torch.cuda.is_available()`；
2. 为真时依次打印 device_count、get_device_name(0)、is_bf16_supported、`device_properties(0).total_memory/1024**3`；
3. 否则打印 CPU 提示。

---

### `src/scripts/inspect_model.py`

（`scripts/inspect_model.py` 与本文件 SHA256 完全相同：702A48DC…，仅详述一份。）

**职责**：模型干跑（dry-run）检查：按正式实验配置实例化 SLMFBBDM，打印可训练/总参数量、各先验模块与损失项的启用状态，不启动训练。

**接口**：无 class/def，纯顶层脚本。用法：`python src/scripts/inspect_model.py`（工作目录须为仓库根；配置路径硬编码为 `configs/experiments/slmf_full.yaml`）。

**依赖**：
- 内部: src.model.slmf_bbdm.SLMFBBDM、src.model.config_utils.load_full_config
- 外部: os、sys

**输入**：硬编码配置文件 `configs/experiments/slmf_full.yaml`。

**输出**：stdout 三行——`Trainable / Total` 参数量、`model.priors` 各项 enabled、`model.loss_terms` 各项 enabled。

**处理过程**：
1. `sys.path.insert` 把仓库根加入路径；
2. `load_full_config` 加载完整配置；
3. `SLMFBBDM.from_config(cfg)` 构建模型；
4. 调用 `get_trainable_params()/get_total_params()` 打印参数量；
5. 以列表推导打印 priors 与 loss_terms 的启用位。

---

### `src/scripts/resync_split_to_newdata.py`

**职责**：以新数据集 `Data/data/patient_split_summary.csv` 的患者级 train/val 划分为权威映射，仅重写老数据集 `cache/split_manifest.csv` 的 `split` 列（其余列原样保留），并生成统计 JSON；操作可逆（原文件备份为 `.bak`）。

**接口**：
- `load_patient_split(path: Path) -> dict[str, str]`：读 CSV 返回 `patient_id -> split` 映射；
- `main() -> int`：主流程，返回退出码。用法：`python src/scripts/resync_split_to_newdata.py`，无 CLI 参数（路径全部由 `ROOT = Path(__file__).resolve().parents[1]` 推导）。

**依赖**：
- 内部: 无（仅操作数据文件）
- 外部: csv、hashlib、json、shutil、sys、collections.defaultdict、pathlib.Path

**输入**：`Data/data/patient_split_summary.csv`（权威患者→split 映射，须含 `patient_id`、`split` 列）；`cache/split_manifest.csv`（老清单，含 `sample_id`、`patient_id`、`split` 等）。常量 `FALLBACK_SPLIT="train"`。

**输出**：`cache/split_manifest.csv`（重写 split 列）、`cache/split_manifest.csv.bak`（备份）、`cache/split_manifest.json`（统计：总样本/患者数、各 split 患者与样本数、患者 ID 列表、12 位十六进制 fingerprint、 defaulted 患者列表）；stdout 打印新旧分布对比。

**处理过程**：
1. 校验权威映射与老清单两个文件存在，缺失即返回 1；
2. `load_patient_split` 读入映射并统计新数据集 train/val 患者数；
3. `shutil.copy2` 备份老清单为 `.csv.bak`；
4. 逐行按 patient_id 查映射更新 `split` 列；映射缺失的患者计入 `unknown` 并默认归 train（打印 WARN）；
5. `assert` 校验同一患者不得跨 split；
6. 写回 CSV（保留原 fieldnames 顺序）；
7. 计算 `(sample_id, patient_id, split)` 排序后 SHA256 前 12 位作为 fingerprint，连同统计写入 `.json`；
8. 打印旧/新分布与患者数摘要，返回 0。

---

### `src/scripts/train_v2.py`

**职责**：对 `scripts/train_v2.py`（正式训练 CLI）的模块包装：再导出其符号（含下划线私有名 `_save_run_metadata`、`_set_seed`、`main`），使训练入口可按 `src.scripts.train_v2` 被 import/执行。SHA256 与 `scripts/train_v2.py` 不同（差异即"包装 vs 实现"）。

**接口**：`from scripts.train_v2 import *` 及显式再导出 `_save_run_metadata, _set_seed, main`；`__main__` 分支 `raise SystemExit(main())`。CLI 用法与 `scripts/train_v2.py` 完全一致（见下一条目）。

**依赖**：
- 内部: scripts.train_v2（canonical 实现）
- 外部: 无（间接依赖 canonical 模块的外部库）

**输入**：同 `scripts/train_v2.py`（--config/--ablation/--ablation-config/--override）。

**输出**：同 `scripts/train_v2.py`（checkpoint、resolved_config.yaml、run_metadata.json）。

**处理过程**：
1. 通配再导出 canonical 模块命名空间；
2. 显式补导三个下划线符号（`*` 不导下划线名）；
3. 直接运行时委托 `main()`。

---

### `scripts/train_v2.py`（主训练入口）

**职责**：SLMF-BBDM 的正式训练 CLI：加载带消融/覆盖的完整配置，校验运行时与数据血缘，构建模型与 DataLoader，处理 init_from/resume_from 两种续训路径，最终交给 `Trainer.run()`；并在训练前落盘 `run_metadata.json` 记录模块/损失启用态以复现实验。

**接口**（顶层函数）：
- `_set_seed(seed: int) -> None`：同时播种 random/numpy/torch(±CUDA)；
- `_select_initial_model_state(checkpoint: dict, weights: str = "raw") -> dict`：从 checkpoint 选 raw 或 EMA 初始权重（支持 `ema_model`/`model_ema`/`ema.shadow` 三种存放形式，EMA shadow 以 raw 为底叠加）；
- `_save_run_metadata(model, config, ckpt_dir, ablation, data_lineage=None) -> None`：写 `run_metadata.json`；
- `main()`：CLI 主函数。

CLI 用法：
```
python scripts/train_v2.py --config configs/experiments/slmf_full.yaml
python scripts/train_v2.py --config ... --ablation no_gabor
python scripts/train_v2.py --config ... --ablation no_gabor --ablation-config configs/experiments/ablations.yaml
python scripts/train_v2.py --config ... --override modules.gabor.enabled=false --override training.num_epochs=50
```
参数：`--config`（必填 YAML 路径）、`--ablation`（消融预设名，默认 None）、`--ablation-config`（默认 `configs/experiments/ablations.yaml`）、`--override`（可重复的 `key=value`）。

配置解析链：`load_full_config(config, ablation, ablation_config_path, overrides)`（src.model.config_utils）完成 YAML 合并 → 检查 `runtime.require_cuda`（为 true 且无 CUDA 时直接抛错，禁止静默降级 CPU）→ 取 `experiment.seed`（默认 42）播种 → `resolve_runtime_profile`（无 CUDA 时降级 CPU profile）→ `validate_png_baseline_config`+`log_startup_status`（PNG 基线模式下若残留 DICOM/SUV/organ 路径则 fail-fast）→ `load_checkpoint_data_lineage`（校验缓存血缘 sha256）→ `resolve_checkpoint_dir`+`save_resolved_config`（确定并保存 resolved 配置）。

**依赖**：
- 内部: src.data.lineage（load_checkpoint_data_lineage、validate_checkpoint_data_lineage）、src.model.config_utils（load_full_config、save_resolved_config、resolve_runtime_profile、validate_png_baseline_config、log_startup_status）、src.model.slmf_bbdm.SLMFBBDM、src.model.trainer（Trainer、resolve_checkpoint_dir）、src.data.dataset.build_dataloaders（函数内延迟导入）
- 外部: argparse、json、os、random、sys、numpy、torch

**输入**：实验 YAML 配置（experiment/data/runtime/training/modules 等节）；可选 `training.init_from`（权重初始化 checkpoint 路径）、`training.init_weights`（raw|ema）、`training.resume_from`（完整恢复 checkpoint，与 init_from 互斥）；数据缓存目录与 split manifest 由配置指定。

**输出**：`checkpoints/<experiment>/` 下的 checkpoint（.pt，由 Trainer 落盘）、`resolved_config.yaml`、`run_metadata.json`（含 enabled_modules/enabled_losses/lesion 后训练信息/参数量/data_lineage）；stdout 各阶段日志。

**处理过程**：
1. argparse 解析四个参数，`load_full_config` 合并基础配置 + 消融预设 + override 链；
2. `require_cuda` 守卫与 `_set_seed`（先于模型/加载器初始化）；
3. `resolve_runtime_profile` 解析运行档位，PNG 基线校验并打印启动状态，校验缓存血缘；
4. `resolve_checkpoint_dir` 得 ckpt_dir 并 `save_resolved_config`；
5. `SLMFBBDM.from_config(config)` 建模，随即 `_save_run_metadata` 落盘模块/损失启用态；
6. init_from 路径：`torch.load(..., weights_only=True)` → `validate_checkpoint_data_lineage` → `_select_initial_model_state`（raw/ema）→ `load_state_dict`；优化器/调度器/EMA 一律全新（避免旧动量拖拽新损失面）；init_from 与 resume_from 同时设置则抛错；
7. 打印 stage、启用的 priors/losses、可训练/总参数量；延迟导入并调用 `build_dataloaders(data_cfg, run_cfg)` 得 (train_loader, val_loader)；
8. `Trainer(model, config, train_loader, val_loader)`；resume_from 时 `trainer.load_checkpoint`；`trainer.run()` 开始训练。

---

### `scripts/evaluate.py`

**职责**：正式评估 CLI：在指定 split 上对（可选 checkpoint 加载的）SLMFBBDM 做采样推理，逐样本计算图像质量、归一化病灶保真、物理 SUV 临床、不确定性、失败检测、方向性伪影、边界保真、假热点等十余族指标，聚合成全局/患者级汇总并输出 JSON 报告与终端报表。

**接口**（主要顶层函数）：
- 指标族：`compute_stripe_metrics(pred, target) -> Dict`；`compute_directional_spectrum_error(pred, target, orientations=8) -> float`；`compute_boundary_metrics(pred, target, ct, lesion_mask, organ_mask, boundary_radius=2) -> Dict`；`compute_mae/compute_mse(pred, target, mask=None) -> float`；`compute_psnr(pred, target, max_val=2.0)`；`compute_ssim(pred, target, data_range=2.0)`（纯 numpy 11×11 高斯窗实现）；`compute_calibration_metrics(pred_values, target_values, prefix="suv_calib") -> Dict`（斜率/截距/R²/Bland-Altman LoA）；`compute_uncertainty_metrics(uncertainty, lesion_mask, confidence=None) -> Dict`；`compute_failure_detection_metrics(pred, target, lesion_mask, uncertainty_metrics=None, outside_margin=0.05, lesion_min_ratio=0.6, uncertainty_ratio_threshold=2.0, model_space=False) -> Dict`；`_denormalise_pet_np(pet_norm, meta)`（[-1,1]→SUV）；`compute_suv_metrics(pred, target, lesion_mask, organ_mask, meta, target_suv=None) -> Dict`（SUVmax/SUVmean/TBR 误差）；`compute_false_hotspot_count(pred, organ_mask, threshold_percentile=0.95) -> Dict`；`compute_target_relative_false_hotspots(pred, target, lesion_mask, *, excess_margin=0.05, target_quantile=0.99) -> Dict`；`compute_normalized_lesion_metrics(pred, target, lesion_mask, organ_mask, topk_percent=0.10, min_k=3, max_k=16) -> Dict`（PNG 基线主病灶指标，含 core/ring top-q、peak-to-boundary 距离等）；
- 工具：`_select_checkpoint_state(checkpoint, weights="ema") -> Tuple[state, source]`（支持 `ema_materialized_as_model`/`ema_model`/`model_ema`/`ema.shadow`）；`_build_stratified_subset(dataset, count)`（按病灶面积分位数确定性抽样）；`_annotate_small_lesion_metrics(all_metrics, quantile=0.25, underestimate_tolerance=0.05) -> Dict`（仅用 GT 面积标注小病灶层，防泄漏）；
- 主循环：`evaluate(model, dataloader, device="cuda", amp=True, save_samples=None, mc_samples=1, mc_steps=None, failure_thresholds=None, small_lesion_quantile=0.25, small_lesion_underestimate_tolerance=0.05) -> Dict`；`print_report(summary)`；`main()`。

CLI 用法：
```
python scripts/evaluate.py --config configs/experiments/slmf_full.yaml \
    --checkpoint checkpoints/slmf_bbdm_full/ckpt_epoch0100.pt --split test \
    --output results/eval_report.json
# 可复现 PNG 基线：--weights ema --split val --max-samples 16 --seed 42
# CPU 冒烟：--fake-data
```
其余参数：`--weights {ema,raw}`（默认 ema）、`--device`、`--no-amp`、`--mc-samples`、`--mc-steps`、`--failure-outside-margin`、`--failure-lesion-ratio`、`--failure-uncertainty-ratio`、`--small-lesion-quantile`、`--small-lesion-underestimate-tolerance`、`--allow-train-fallback`（split 缺失时退回 train，仅调试用）。CLI 未给的值依次回退到 config 的 `evaluation.*`/`runtime.eval_seed`/`experiment.seed`。

**依赖**：
- 内部: src.data.lineage（load/validate_checkpoint_data_lineage）、src.model.config_utils（load_full_config、resolve_runtime_profile）、src.model.slmf_bbdm.SLMFBBDM、src.model.loss_terms.roi_suv._de_collate_meta、src.model.trainer（_compute_pet_sample_metrics、_stripe_score、_stratified_indices、_to_unit_interval）、src.data.dataset（CachedDataset、FakeDataset、build_dataloaders；延迟导入）
- 外部: argparse、json、math、os、sys、collections、pathlib、typing、numpy、torch、torch.nn.functional、torch.utils.data（DataLoader、Subset）、scipy.ndimage（binary_dilation、binary_erosion、correlate）

**输入**：实验 YAML（data.cache_dir、data.split_manifest、data.val_batch_size、evaluation.* 等）；checkpoint .pt（含 model 与可选 EMA 态）；batch 张量：ct/pet/mask 形如 [B,1,H,W]（[-1,1]），organ_mask [B,6,H,W]，可选 pet_suv 与逐样本 meta（patient_id、suv_ok、pet_suv_max 等）。

**输出**：`--output` 指定路径的 JSON 汇总（每指标 mean/std/median/min/max、`num_samples`、`num_patients`、`physical_suv_available`、小病灶层信息、`per_patient` 患者级聚合、SUV 校准汇总）；stdout 分节报表（Image Quality / Normalized-Intensity Lesion / Small-Lesion Stratum / Clinical SUV / SUV Calibration / Uncertainty / Failure Detection / Directional Artifacts / Boundary Fidelity / False Hotspots）。

**处理过程**：
1. `main()`：加载配置并 `resolve_runtime_profile`，`--fake-data` 时置 `data.use_fake_data`；解析 eval seed/MC/失败阈值/小病灶参数（CLI > config > 默认）；
2. `SLMFBBDM.from_config` 建模；有 checkpoint 则 `torch.load` → 血缘校验 → `_select_checkpoint_state`（默认 EMA，与 trainer 的模型选择一致）→ `load_state_dict`；
3. 构建数据集：fake → FakeDataset(32, image_size)；否则 `CachedDataset(cache_dir, split, augment=False, split_manifest)`，split 为空时默认拒绝（除非 --allow-train-fallback）；`--max-samples` 时按病灶面积分层子集；
4. `evaluate()`：`model.eval()`，逐 batch 在 `torch.amp.autocast`（bf16 可用则 bf16）下 `model.sample` 或 `model.sample_mc`（mc_samples>1）采样；取 `synthetic_pet`、`total_var/epistemic_var`、`confidence_map`；
5. 逐样本计算：MAE/MSE/PSNR/SSIM + 条纹 + 边界五项；有病灶 mask 时无条件输出归一化病灶指标；`suv_ok` 时才输出 SUV 临床指标（并用真实 pet_suv 作 target）；有不确定性图时输出 uncertainty 三项；
6. 失败检测：优先在物理 SUV 空间判定（outside_peak 超过 inside、病灶过冷、不确定性比值过高三类失败及 failure_reason）；随后计算假热点（背景 95 分位法 + target-relative 法）；
7. 聚合：先按 GT 面积标注小病灶层（阈值只依赖 GT，跨模型可比），再全局 mean/std/median/min/max 与患者级聚合，最后追加 SUV 校准（pred vs target 的 polyfit 斜率/R²/LoA）；
8. `print_report` 打印报表，`--output` 时写 JSON（`default=str` 容错），返回 0。

---

### `scripts/run_frequency_ablations.py`

**职责**：残差频率模块消融的总编排器（也是 V4/V5 编排器共享的底层库）：按 YAML 计划生成 mean 预训练 / screen 筛查 / promote 晋升三阶段的 train+eval 子进程命令，执行后用硬门控剔除伪影劣化者、用病灶加权复合分排序，选出 top-k 晋级到完整预算训练，并产出排行榜与配对对比。

**接口**（顶层函数，共 20 个，关键的）：
- `passes_hard_gates(metrics, reference, gates) -> tuple[bool, List[str]]`：按 `max_value/max_delta/max_ratio`（lower）或 `min_value/min_delta/min_ratio`（higher）对候选 vs 参考做门控，返回通过位与失败原因；
- `composite_score(metrics) -> float`：病灶加权分 = 0.70×lesion（−topq_peak −0.50·peak −0.02·centroid −0.50·failure −0.20·lesion_boundary）+ 0.30×image（ssim − mae −0.20·stripe −100·hotspot −0.10·anatomy_boundary −0.10·direction_error）；
- `select_promotions(records, reference_id, gates, top_k) -> tuple[List[str], List[Dict]]`：门控过滤 + 得分排序，取前 top_k；
- `build_train_command(*, python, config, ablation_config, preset, experiment, epochs, seed, eval_interval, save_interval=None, sample_interval=None, early_stopping_enabled=False, extra_overrides=None) -> List[str]`：拼 `python scripts/train_v2.py --config ... --ablation ... --override ...`（固定注入 resume_from=null、init_from=null）；
- `build_mean_pretrain_command(...) -> List[str]`：拼 `python scripts/pretrain_conditional_mean.py ...`；
- `build_eval_command(*, python, config, checkpoint, output, split, max_samples, seed, mc_steps) -> List[str]`：拼 `python scripts/evaluate.py --weights ema ...`；
- `_run_manifest_entry(plan, variant, settings, python, phase) -> Dict`：单个变体的完整 entry（id/preset/experiment/checkpoint/result/train_command/eval_command/completion_checkpoint）；
- `build_execution_manifest(plan, *, python, stage, promoted_ids=None) -> Dict`：按 stage 汇总 mean_run/screen_runs/promotion_runs；
- `_run_entries(entries, force, checkpoint_validator=None) -> List[Dict]`：幂等执行（checkpoint/结果已存在且非 force 则跳过），校验必需 checkpoint 存在，评估后读回 JSON metrics 记录；
- `checkpoint_epoch(path) -> int`：torch.load 读 checkpoint 的整数 epoch 元数据；
- `_write_rankings`/`_write_paired_results`/`_write_final_gate_results`：落盘 promotion_decision.json、leaderboard.csv、`{phase}_paired_comparison.json`、final_gate_decision.json；`main()`。

CLI 用法：
```
python scripts/run_frequency_ablations.py --plan configs/experiments/frequency_ablation_plan.yaml \
    [--stage {mean,screen,promote,all}] [--dry-run] [--force]
```

**依赖**：
- 内部: scripts.compare_v4_results.compare_all_results（统计对比）；经子进程消费 scripts.train_v2、scripts.evaluate、scripts.pretrain_conditional_mean
- 外部: argparse、csv、json、math、subprocess、sys、pathlib、typing、yaml、torch（checkpoint_epoch 内延迟导入）

**输入**：计划 YAML（`configs/experiments/frequency_ablation_plan.yaml`：base_config、ablation_config、variants[]（id/preset/overrides）、mean_pretrain、screen/promote（epochs/seed/eval_interval/max_samples/mc_steps/split）、hard_gates、reference_id（默认 R0）、top_k、paired_comparison、common_train_overrides、output_dir）。

**输出**：`<output_dir>/dry_run_manifest.json`（dry-run）；`leaderboard.csv` + `promotion_decision.json`（screen）；`screen_paired_comparison.json`；`promotion_results.json`、`promotion_paired_comparison.json`、`final_gate_decision.json`、`promotion/leaderboard.csv`（promote）；以及每个变体的 checkpoints/<experiment>/ 与 results JSON。

**处理过程**：
1. `main()` 解析参数并 yaml.safe_load 计划，创建 output_dir；`--dry-run` 时生成并打印命令清单后返回；
2. stage 含 mean 时：`_run_mean` 幂等执行条件均值预训练（checkpoint 已存在则跳过），并校验产物存在；
3. stage 含 screen 时：`_require_planned_mean_checkpoint` 先校验冻结均值 checkpoint 在盘；`_run_entries` 逐变体"训练→校验 checkpoint→评估→读回 metrics"；
4. `select_promotions` 以参考变体（默认 R0）为基线过硬门控，按 composite_score 排序取 top_k；
5. `_write_rankings` + `_write_paired_results` 落盘决策与配对 bootstrap 对比；
6. stage=promote 时从 promotion_decision.json 读回晋级名单（可按 `include_reference_in_promotion` 强制并入参考）；
7. promote 阶段重新生成完整预算的 train/eval 命令并执行，写 promotion_results.json 与 promotion 配对对比；
8. `_write_final_gate_results` 用 `promotion_hard_gates`（若有）做最终门控判定，写 final_gate_decision.json 与 promotion/ 排行榜。

---

### `scripts/run_boundary_reliable_v4.py`

**职责**：V4"边界可靠（BR-D）"两段式消融编排器：Stage A 筛查 peak/CT 类变体（D 系列）选出最优 preset，Stage B 把每条 Gabor 路由挂接到该胜者 preset 上再筛，最后把前 2 名推 300-epoch 完整训练；复用 `run_frequency_ablations` 的全部执行/门控/排序底层。

**接口**：
- `build_stage_b_variants(plan, selected_d_id) -> list[Dict]`：Stage B 变体 = 计划模板 overrides + 胜者 D 的 preset，附 `inherited_d` 溯源；
- `_stage_a_variant(plan, variant_id) -> Mapping`：按 id 取 Stage A 变体（缺失抛 KeyError）；
- `_build_entries(plan, variants, settings, python, phase) -> list[Dict]`：批量生成 entry（委托 `_run_manifest_entry`）；
- `build_v4_dry_run_manifest(plan, *, python) -> Dict`：用 `dry_run_selected_d`（默认 D3）拼三阶段命令清单；
- `_rank_candidates(records, *, reference_id, gates, top_k)`：门控+复合分排序（与 select_promotions 同型但保留全部行）；
- `_load_decision(output_dir, phase) -> Mapping`：读回某阶段的 promotion_decision.json；
- `_run_stage(plan, variants, settings, *, phase, python, force)`：执行 + 排序 + 落盘排行榜（top_k 上限 2）；
- `_write_promotion_outputs(plan, promotion_records, stage_b_decision)`：最终门控判定 + promotion_results + 配对对比；
- `main()`。

CLI 用法：
```
python scripts/run_boundary_reliable_v4.py --plan configs/experiments/boundary_reliable_ablation_plan_v4.yaml \
    [--stage {stage-a,stage-b,promote,all}] [--dry-run] [--force]
```

**依赖**：
- 内部: scripts.compare_v4_results.compare_all_results、scripts.run_frequency_ablations（_json_safe、_require_planned_mean_checkpoint、_run_entries、_run_manifest_entry、_write_rankings、composite_score、passes_hard_gates）
- 外部: argparse、json、sys、pathlib、typing、yaml

**输入**：计划 YAML（stage_a/stage_b/promote 各含 variants、epochs、seed、reference_id、hard_gates、top_k 等；output_dir；paired_comparison）。

**输出**：`<output_dir>/dry_run_manifest.json`、`stage_a/leaderboard.csv`+`promotion_decision.json`、`stage_b/...`、`final_gate_decision.json`、`promotion_results.json`、`paired_comparison.json`，及各实验 checkpoint/评估 JSON。

**处理过程**：
1. 解析计划并建 output_dir；dry-run 时打印三阶段命令清单返回；
2. `_require_planned_mean_checkpoint` 校验冻结均值 checkpoint；
3. Stage A：`_run_stage` 执行 D 系列训练+评估，门控排序后取第一名 D（stage-a 单独运行到此为止；后续 stage 从决策文件读回）；
4. 无 D 通过则终止；否则 `build_stage_b_variants` 把 Gabor 路由嫁接到胜者 preset；
5. Stage B：同流程筛 Gabor 路由，top_k 上限 2；
6. 无路由通过则不启动 300-epoch；否则取前 2 条进入 promote 阶段完整训练+评估（`_run_entries`）；
7. `_write_promotion_outputs`：以 Stage B 参考为基线过 stage_b.hard_gates 生成 final_gate_decision（accepted 名单）与 promotion_results；
8. 用 promote 结果路径生成患者配对 bootstrap 对比写 paired_comparison.json。

---

### `scripts/compare_v4_results.py`

**职责**：评估 JSON 的确定性患者配对比较器：读取多份含 `per_patient` 映射的评估结果，对指定指标在共享患者集合上做逐对差值统计（win/tie/loss、中位差）与 bootstrap 95% CI，是所有消融编排器的统计后端。

**接口**：
- `_load_result(path) -> Mapping`：加载 JSON 并校验含 per_patient；
- `compare_result_pair(left_path, right_path, metric_directions, *, seed=42, resamples=10_000) -> Dict`：单对比较；metric_directions 为 `{metric: "lower"|"higher"}`；
- `compare_all_results(result_paths, metric_directions, *, seed=42, resamples=10_000) -> Dict`：对排序后的 ID 两两组合（itertools.combinations）批量比较；
- `main()`。

CLI 用法：
```
python scripts/compare_v4_results.py --result A=pathA.json --result B=pathB.json \
    --metrics '{"lesion_topq_peak_error_norm_mean": "lower", "ssim_mean": "higher"}' \
    --output results/paired.json [--seed 42] [--resamples 10000]
```

**依赖**：
- 内部: 无（纯统计工具，被 4 个编排器消费）
- 外部: argparse、itertools、json、math、pathlib、typing、numpy

**输入**：评估 JSON 文件（须含 `per_patient`：患者 ID → 指标映射）；--metrics 为 JSON 字符串的方向映射；--result 为可重复的 `ID=path`。

**输出**：--output JSON 报告：`{seed, resamples, comparisons: [{left_id, right_id, left, right, shared_patients, metrics: {name: {direction, paired_patients, wins, ties, losses, median_difference, bootstrap_95_ci}}}]}`。

**处理过程**：
1. 解析 --result 为 ID→Path 映射（格式非法即抛 ValueError），json.loads 解析方向映射；
2. `compare_all_results` 对 ID 排序后两两组合；
3. 每对取 per_patient 交集患者（排序），按指标收集双方均有限的数值对；
4. 差值 = left − right；按方向（lower 时差<0 为 win，higher 反之）统计 wins/ties/losses 与中位差；
5. 样本 ≥2 时以 `default_rng(seed+metric_index)` 有放回重采样 resamples 次取中位差的 2.5/97.5 百分位为 95% CI；
6. 汇总写输出 JSON（自动建父目录）。

---

### `scripts/run_v5_t0_c1_300.py`（已弃用）

**职责**：V5 等预算 T_legacy/C1 300-epoch"抢救"实验的窄口径编排器；已被公平对照重构（T_native/T_fixed/C1/C_no_null，由新版 `run_spectral_router_v5.py` 编排）取代，仅保留向后诊断兼容，原始结果存于 `results/spectral_router_ablations_v5/promote/evidence-s3-legacy/`。

**接口**：
- `load_plan(path) -> dict`：yaml.safe_load 并校验为 mapping；
- `build_rescue_entries(plan, *, python=sys.executable, evidence_id="S3") -> list[dict]`：从 V5 计划的 Stage B 变体中取 `RESCUE_ROUTE_IDS=("T_legacy","C1")` 两条，用 `_build_entries`（V5 版）生成 300-epoch promote entry；`plan.promote.epochs` 非 300 即抛错；
- `run_rescue(plan, *, force=False, dry_run=False, python=..., evidence_id="S3") -> Path`：执行并产出配对比较 JSON；
- `_print_manifest`/`_comparison_output` 辅助；`parse_args()`/`main()`。

CLI 用法：
```
python scripts/run_v5_t0_c1_300.py [--plan configs/experiments/spectral_router_ablation_plan_v5.yaml] \
    [--evidence-id S3] [--force] [--dry-run]
```

**依赖**：
- 内部: scripts.compare_v4_results.compare_all_results、scripts.run_frequency_ablations（_require_planned_mean_checkpoint、_run_entries）、scripts.run_spectral_router_v5（_build_entries、_validate_promotion_checkpoint、build_stage_b_variants）
- 外部: argparse、json、subprocess、sys、pathlib、typing、yaml

**输入**：V5 计划 YAML（promote 节须 epochs=300；paired_comparison 节提供 metrics/seed/resamples）。

**输出**：promote 结果目录下 `paired_t0_c1_epoch300.json`（compare_all_results 报告 + selected_evidence_id/checkpoint_epoch/comparison 附加字段）；dry-run 时打印 train/eval 命令清单。

**处理过程**：
1. `load_plan` 读计划；`build_rescue_entries` 取 T_legacy/C1 两条 promote entry 并强校验 300 epoch；
2. dry-run 仅打印命令与输出路径；
3. `_require_planned_mean_checkpoint` 校验均值 checkpoint；`_run_entries(..., checkpoint_validator=_validate_promotion_checkpoint)` 确保评估的是精确 epoch-300 checkpoint；
4. 以两条结果路径调 `compare_all_results` 做患者配对 bootstrap 比较；
5. 附加溯源字段后写 `paired_t0_c1_epoch300.json` 并打印路径。

---

### `scripts/run_spectral_router_v5.py`

**职责**：V5 谱证据路由（spectral evidence router）编排器：Stage A 筛"证据变体"（S 系列）选定证据 preset，Stage B 在其上公平对照六条路由（N0/T_legacy/T_native/T_fixed/C1/C_no_null，固定注入 `modules.residual_frequency.mode=spectral_evidence_router`），promote 阶段强制包含参考路由 T_native 做等预算 300-epoch 对比，且校验 checkpoint 精确 epoch 与证据溯源一致性。

**接口**：
- `build_stage_b_variants(plan, selected_evidence_id) -> list[Dict]`：按 `expected_ids=("N0","T_legacy","T_native","T_fixed","C1","C_no_null")` 顺序实例化路由，逐路由写死 cross_level_router 的 enabled/hard_all_null/policy/fixed_prior 覆盖（如 T_fixed 默认 prior [0.05,0.05,0.90]，C_no_null 默认 [0.5,0.5]），附 `inherited_evidence`；
- `validate_checkpoint_epoch(path, *, expected_epoch=300) -> int`：`checkpoint_epoch` 读元数据并强校验等于期望 epoch；
- `_validate_promotion_checkpoint(checkpoint, entry) -> int`：entry 的 `required_checkpoint_epoch`（默认 300）包装；
- `_build_entries(plan, variants, settings, python, phase, *, evidence_id=None) -> list[Dict]`：带证据隔离（experiment_prefix 加 `evidence-<id>` 后缀、phase 加子目录）；phase=promote 时经 `_exact_final_checkpoint` 把评估 checkpoint 切到 `completion_checkpoint`（精确末 epoch）；
- `build_v5_dry_run_manifest(plan, *, python, stage="all", selected_evidence_id=None) -> Dict`；
- `_select_eligible_stage_b_routes`：从共享排名剔除参考、诊断专用（`_NON_PROMOTABLE_IDS={"N0","T_legacy"}`）与未过门控者，上限 4；
- `_promotion_route_ids`：参考 + 至多 top_k 候选集合；
- `_record_evidence_provenance`/`_load_selected_evidence`/`_load_stage_b_decision`：把 selected_evidence_id 写进/读回决策文件并校验 preset、experiment token 一致（防跨证据混用产物）；
- `_run_v5_stage`：包装共享 `_run_stage`，Stage B 后重算 eligible 路由并记溯源；
- `_write_promotion_outputs`：以 300-epoch T_native 为参考（缺失即抛错，保证公平），可用 promote 专属 hard_gates 做终审，写 final_gate_decision/promotion_results/paired_comparison；`main()`。

CLI 用法：
```
python scripts/run_spectral_router_v5.py --plan configs/experiments/spectral_router_ablation_plan_v5.yaml \
    [--stage {stage-a,stage-b,promote,all}] [--dry-run] [--force]
```

**依赖**：
- 内部: scripts.compare_v4_results.compare_all_results、scripts.run_boundary_reliable_v4（_load_decision、_run_stage as _shared_run_stage）、scripts.run_frequency_ablations（_json_safe、_require_planned_mean_checkpoint、_run_entries、_run_manifest_entry、_write_rankings、checkpoint_epoch、passes_hard_gates）
- 外部: argparse、json、sys、pathlib、typing、yaml

**输入**：V5 计划 YAML（stage_a/stage_b/promote、stage_b.reference_id（=T_native）、top_k 默认 3、output_dir、paired_comparison）。

**输出**：`<output_dir>/dry_run_manifest.json`；`stage_a/`、`stage_b/evidence-<id>/`、`promote/evidence-<id>/` 各阶段 leaderboard/promotion_decision/评估 JSON；顶层 `final_gate_decision.json`、`promotion_results.json`、`paired_comparison.json`（仅 300-epoch 等预算条目参与）。

**处理过程**：
1. 解析计划建目录；dry-run 时按已有决策文件恢复 selected_evidence，打印三阶段命令；
2. 校验均值 checkpoint；Stage A 执行证据变体筛查（复用共享 _run_stage）；
3. `_load_selected_evidence` 从 stage_a 决策读回首选证据 ID，并校验其 ranked 行的 preset 与计划一致（新鲜运行结果须与决策文件吻合，否则抛错）；
4. `build_stage_b_variants` 实例化公平对照路由（全部锁定 spectral_evidence_router 模式）；
5. Stage B 以证据隔离路径运行；`_select_eligible_stage_b_routes` 剔除 N0/T_legacy/参考/未过门控者得晋级名单，`_record_evidence_provenance` 写入溯源；
6. 无候选通过时仍单独 promote T_native 以保留等预算基线；
7. promote 批 = 参考 + 至多 top_k 候选；`_run_entries` 以 `_validate_promotion_checkpoint` 强校验每个 checkpoint 元数据 epoch=300（非 300 抛错）；
8. `_write_promotion_outputs`：以 T_native-300 为参考做终审门控与患者配对 bootstrap 对比，落盘三份决策/结果 JSON。

---

### `scripts/evaluate_v5_checkpoint_trajectory.py`

**职责**：对一个 V5 实验目录的 checkpoint 轨迹（默认 epoch 50/100/150/200/250/300 + best_lesion/best_combined）逐一调 `scripts.evaluate` 子进程评估，汇总成单实验轨迹 JSON 并打印关键指标摘要表，用于观察训练动态与选点。

**接口**：
- `_find_checkpoints(experiment_dir) -> Dict[str, Path]`：rglob 匹配 `**/ckpt_epoch*.pt` 与 `**/ckpt_best_*.pt`，返回 `{stem: path}`；
- `evaluate_checkpoint(config_path, checkpoint_path, *, split="val", max_samples=64, seed=42, mc_steps=20, weights="ema", device=None, python=sys.executable) -> Dict`：以 `python -m scripts.evaluate` 子进程评估单个 checkpoint（临时文件承接输出 JSON，finally 删除）；
- `collect_trajectory(experiment_dir, config_path, *, ..., whitelist_epochs=None) -> Dict`：白名单 epoch 缺 checkpoint 时 fail-fast（FileNotFoundError），逐个评估并容错记录 {`error`: str}；
- `_extract_summary_metrics(trajectory) -> Dict[str, Dict[str, Optional[float]]]`：抽取 12 个关键标量（lesion_topq_peak_error_norm_mean、ssim_mean、failure_any_mean、small_lesion_* 等）；`main()`。

CLI 用法：
```
python scripts/evaluate_v5_checkpoint_trajectory.py \
    --experiment-dir checkpoints/sr_v5_full_evidence-s3_t_native \
    [--config <yaml>] [--output-dir results/spectral_router_ablations_v5/trajectory] \
    [--split val] [--max-samples 64] [--seed 42] [--mc-steps 20] [--weights ema] \
    [--device cuda] [--epochs 50 100 150 200 250 300]
```
`--config` 缺省时自动探测 `{experiment-dir}/resolved_config.yaml`，探测失败则以错误退出。

**依赖**：
- 内部: scripts.evaluate（以 `python -m scripts.evaluate` 子进程方式消费，非 import）
- 外部: argparse、json、subprocess（函数内导入）、sys、tempfile（函数内导入）、pathlib、typing

**输入**：实验目录（含 ckpt_epoch*.pt / ckpt_best_*.pt 与 resolved_config.yaml）；白名单 epoch 列表。

**输出**：`<output-dir>/<experiment_name>_trajectory.json`（experiment_dir/config/results 三层，results 按 checkpoint label 索引完整评估 JSON 或 error）；stdout 摘要表（行=checkpoint，列=关键指标）。

**处理过程**：
1. 解析参数，自动探测或要求 --config；
2. `_find_checkpoints` 递归发现全部 checkpoint；
3. 白名单非空时先校验所有 `ckpt_epoch{e:04d}` 均在盘，缺失即抛 FileNotFoundError；
4. 逐个 epoch checkpoint 调 `evaluate_checkpoint`（子进程 `python -m scripts.evaluate`，临时 JSON 回读），异常降级为 {`error`} 记录；
5. 再评估 `ckpt_best_lesion`/`ckpt_best_combined`（存在才评）；
6. 轨迹 JSON 写入输出目录（`default=str` 容错）；
7. `_extract_summary_metrics` 抽取关键指标并打印对齐的摘要表。

---

### `scripts/visualize_eval.py`

**职责**：定性可视化小脚本：加载固定 checkpoint（`checkpoints/slmf_bbdm_full/ckpt_epoch0100.pt`），在 val split 中筛含病灶 mask 的样本取前 8 个，逐例输出 CT / 目标 PET / 预测 PET / 病灶 mask 四联 PNG。

**接口**：无 class/def，纯顶层脚本。用法：`python scripts/visualize_eval.py`（仓库根运行；CKPT/CONFIG/OUT 为硬编码常量，无 CLI）。

**依赖**：
- 内部: src.model.config_utils（load_full_config、resolve_runtime_profile）、src.model.slmf_bbdm.SLMFBBDM、src.data.dataset.CachedDataset
- 外部: os、sys、pathlib、torch、numpy、matplotlib（Agg 后端）、torch.utils.data（DataLoader、Subset）

**输入**：硬编码配置 `configs/experiments/slmf_full.yaml` 与 checkpoint `checkpoints/slmf_bbdm_full/ckpt_epoch0100.pt`；CachedDataset 的 val split（cache_dir 与 split_manifest 取自配置）；batch: ct/pet/mask [1,1,H,W]。

**输出**：`results/vis_epoch100/sample_{i:02d}_pid{pid}.png` 四联图（dpi=120）；stdout 每例 pred_max/tgt_max 与保存路径。

**处理过程**：
1. sys.path 注入 scripts 目录；matplotlib 切 Agg 后端；
2. 加载配置（resolve_runtime_profile）→ `SLMFBBDM.from_config` → .to("cuda").eval() → 载入 checkpoint 的 model 态；
3. 构建 val CachedDataset（augment=False），以 `ds.entries[i].has_mask` 筛出含病灶样本，取前 8 组 DataLoader(batch_size=1)；
4. 逐例 no_grad 采样 `model.sample(batch)["synthetic_pet"]`，取 [0,0] 转 numpy；
5. 1×4 子图渲染（CT gray / PET hot，vmin=-1,vmax=1 / mask gray），标题含 pid 与 pred/tgt 数值范围；
6. 保存 PNG 并打印进度。

---

### 模块依赖小结

**本组 import 的内部模块**（排除 artifacts/ 历史快照）：
- `scripts/train_v2.py` → src.data.lineage、src.model.config_utils、src.model.slmf_bbdm、src.model.trainer、src.data.dataset（延迟）；
- `scripts/evaluate.py` → src.data.lineage、src.model.config_utils、src.model.slmf_bbdm、src.model.loss_terms.roi_suv（_de_collate_meta）、src.model.trainer（_compute_pet_sample_metrics/_stripe_score/_stratified_indices/_to_unit_interval）、src.data.dataset（延迟）；
- `src/scripts/train_v2.py`、`src/scripts/evaluate.py` → 仅 re-export scripts.train_v2 / scripts.evaluate；`src/scripts/inspect_model.py`、`scripts/visualize_eval.py` → src.model.slmf_bbdm、src.model.config_utils（后者另用 src.data.dataset.CachedDataset）；
- 编排器链：scripts.compare_v4_results（无内部依赖）← scripts.run_frequency_ablations ← scripts.run_boundary_reliable_v4 ← scripts.run_spectral_router_v5 ← scripts.run_v5_t0_c1_300（同时直接依赖中间两层）；scripts.evaluate_v5_checkpoint_trajectory 以 `python -m scripts.evaluate` 子进程消费评估 CLI；`check_ts.py`、`gpu_info.py`、`resync_split_to_newdata.py` 无内部依赖。

**被哪些内部模块/脚本消费**（grep 查证，均排除 artifacts/）：
- scripts.train_v2：src/scripts/train_v2.py、tests/test_smoke.py，及全部编排器子进程（train 命令）；
- scripts.evaluate：src/scripts/evaluate.py、scripts/evaluate_v5_checkpoint_trajectory.py（-m 子进程）、scripts/audit_magnification_consistency.py、scripts/eval_action_axis_screen.py、scripts/eval_router_background_suite.py、src/mechanism_validation/magnification_audit.py、tests（test_boundary_frequency、test_formal_mechanism_pipeline、test_peak_diagnostics、test_trainer_monitoring、test_smoke），及全部编排器子进程（eval 命令）；
- scripts.compare_v4_results：run_frequency_ablations、run_boundary_reliable_v4、run_spectral_router_v5、run_v5_t0_c1_300、tests/test_frequency_ablation_runner.py；
- scripts.run_frequency_ablations：run_boundary_reliable_v4、run_spectral_router_v5、run_v5_t0_c1_300、tests/test_frequency_ablation_runner.py；
- scripts.run_boundary_reliable_v4：run_spectral_router_v5、tests/test_frequency_ablation_runner.py；
- scripts.run_spectral_router_v5：run_v5_t0_c1_300、tests/test_frequency_ablation_runner.py；
- scripts.evaluate_v5_checkpoint_trajectory：tests/test_frequency_ablation_runner.py；
- src.scripts.* 三件套（check_ts/gpu_info/inspect_model）：无内部消费者（手动运行）。

**重要跨模块发现**：① 训练/评估的真实实现只在 scripts/，src/scripts/ 同名文件是 re-export 薄包装（哈希差异原因），tests 与 `python -m scripts.evaluate` 均以 scripts.* 为 canonical；② evaluate.py 反向 import trainer 的四个下划线工具函数，评估指标与训练期监控共享同一套 stripe/topq/stratified 语义；③ 编排器体系呈四层继承（frequency→V4→V5→t0_c1 救援），V4/V5 只复用底层执行与门控、不重复实现统计；④ V5 的公平对照约束（T_native 必进 300-epoch 批、精确 epoch 校验、证据溯源写回决策文件）是跨 run_spectral_router_v5 与 run_v5_t0_c1_300 的关键契约。


---

## 8. 机制验证脚本（H1-H6 假设检验）
### 模块概述

本目录脚本实现 SLMF-BBDM 机制的"先证伪后实现"验证流水线（由 `configs/mechanism_validation_pipeline_v1.json` 编排）：H1 谱不对称 → H2 残差富集 → H3 log-SNR 可恢复性 → H4 噪声带校准 → CT 支持头/课程/伪影安全前置门 → H5 路由角色分离 → H6 最终集成收益。设计上强制：患者为统计单元、阈值只来自 mechanism_train/calibration 划分、验证集只评估一次、患者级 bootstrap 95% CI、失败门阻断下游阶段、每阶段产出 decision.json。数据流为：LOCKED 数据集契约 + split_manifest + 原始 PNG/缓存 → 各脚本统计检验 → `results/mechanism_validation/<stage>/` 下的 decision.json / report.md / 明细 CSV，供下游阶段 `require_upstream_pass` 链式消费。

### `scripts/validate_h1_local_spectral_asymmetry.py`（H1，942 行）

**职责**：Stage 0B / H1 局部谱不对称验证。两个可独立证伪的子命题：(1) CT 多尺度 Haar 带响应能定位标注 PET 病灶支持；(2) 给定病灶几何后，CT 带幅值对 PET 病灶带幅值不再提供可恢复的增量预测（为 H2"残差富集"铺垫）。纯本地数据检验，不用缓存/模型/训练。

**接口**：
- `run(root, output, *, raw_root_override=None) -> dict`：主流程，返回并写 decision.json。
- `main(argv=None) -> int`：CLI，`--root`（默认仓库根）、`--output`（默认 `results/mechanism_validation/00B_h1_spectral_asymmetry`）、`--raw-root`（PNG 物理根覆盖，字节指纹必须匹配 LOCKED 契约）。决策 PASS 返回 0，否则 2。
- 关键常量：`BANDS`（ll2 + l1/l2 的 lh/hl/hh 共 7 个带）、`RING_RADIUS=12`、`BOOTSTRAP/PERMUTATION_REPLICATES=10000`、`RIDGE_ALPHAS`、`PARTITION_SEED=42`、`CALIBRATION_FRACTION=0.20`。

**依赖**：内部: Stage-0A 的 `results/mechanism_validation/00_data_audit/dataset_contract.json`（须 LOCKED 且自哈希一致）、split_manifest。外部: numpy、pandas、Pillow、scipy(ndimage/stats)、scikit-learn(Ridge, roc_auc_score)。

**输入**：LOCKED 数据集契约、`split_manifest.csv`、`{raw_root}/{split}/{ct|pet|label}/*.png`（192×192）。

**输出**：`patient_partition.csv`、样本级带能量明细 CSV、`decision.json`（含 H1_support_localization / H1_pet_band_amplitude 两个门 + 尺寸阈值 small/medium）、`report.md`、`execution_metadata.json`。

**处理过程**：
1. `_load_contract` 校验契约自哈希、LOCKED 状态与 dataset_hypothesis_allowed 标志。
2. 索引三模态 PNG，与 manifest 逐一对齐，重算 combined_sha256 与契约比对，防止数据漂移。
3. `_patient_partition`：train 患者按 seed=42 洗牌切 20% 为 calibration、其余 mechanism_train；val 患者整体作 validation。
4. 逐样本计算两级 Haar 带能量图（细节子带 3×3 均值滤波），构造病灶环（膨胀 12px 减 mask），按带统计病灶/环/全图均值。
5. **定位门**：calibration 上选带与方向（directional AUC = 病灶 vs 环），validation 一次性报告患者级 AUC + bootstrap 95% CI + 置换 p，并做 BH-FDR 校正。
6. **振幅门**：Ridge 回归（α 网格选择）在给定几何基线特征上增量预测 PET 带幅值，报告 incremental skill 及其 bootstrap CI，与 calibration 置换零分布 q95 比较。
7. 汇总 H1 决策（两子门均 PASS 才 PASS），写 decision.json/report.md，声明"仅关联/预测证据，非因果或互信息主张"、阈值在验证前冻结等 guardrails。

**流水线阶段**：`00B_h1_spectral_asymmetry`（hypothesis=H1，requires `00A_data_audit`）。**统计方法**：directional AUC + 患者级 bootstrap 95% CI（10k）、符号翻转/置换检验（10k）、BH-FDR、Ridge 增量 skill + 置换零分布。
### `scripts/validate_h2_pathology_excluded_residual.py`（H2，859 行）

**职责**：H2 配对固定 epoch 条件均值 + 残差富集门。训练两个仅接收 CT 输入、同初始化/同患者划分/同批序/同优化器/同 epoch 预算的 `LowFrequencyPETPredictor`：`included` 用全部 LL2 像素损失，`excluded` 从 LL2 损失中剔除膨胀后的病理支持区。唯一干预是训练损失；推理时两模型都不看 mask。验证"病理支持区外仍能在病灶处富集 PET 残差，且掩膜边界不劣化"。

**接口**：
- `run(args) -> dict`：加载契约/H1 决策/缓存 lineage → 复用或训练两个变体 → 评估 → 决策。
- `main(argv=None) -> int`：异常时写 FAIL decision.json 并返回 2；PASS 返回 0。
- `IndexedDataset(base: CachedDataset, indices)`：按患者角色筛选的稳定数据集视图（被 H3/H4 复用）。
- `pathology_excluded_mean_loss(pred_ll2, target_ll2, mask, *, epsilon, guard_radius_px)` / `pathology_included_mean_loss(...)`：Charbonnier LL2 损失（病灶支持区经 4× max-pool + 半径膨胀后置零）。
- CLI：`--root --config(默认 configs/experiments/slmf_png_residual_frequency.yaml) --contract --h1-decision --manifest --cache-dir --cache-lineage --output --epochs(30) --batch-size(8) --guard-radius-px(8) --seed(4242) --num-workers --device --force`。

**依赖**：内部: `src.data.dataset.CachedDataset`、`src.data.lineage`（checkpoint 血缘）、`src.mechanism_validation.common`（患者划分/配对 bootstrap/决策 guardrails/manifest 写出）、`src.model.frequency.haar.reconstruct_lowpass`、`src.model.mean_predictor.LowFrequencyPETPredictor`、`src.model.mean_pretraining.mean_target_ll2`。外部: numpy、torch、yaml、sklearn。

**输入**：LOCKED 契约、H1 decision.json（`require_upstream_pass`）、`split_manifest.csv`、`cache/tensors_main` 缓存与 `cache_lineage.json`、实验配置 YAML。

**输出**：`results/mechanism_validation/01_h2_residual_enrichment/` 下 decision.json（含 residual_enrichment 与 mask_boundary_safety 双门、阈值来源声明）、患者级明细 CSV、mechanism manifest、两个变体 checkpoint。

**处理过程**：
1. 校验上游：契约 LOCKED、H1 决策 PASS、缓存 lineage 一致，重建 mechanism_train/calibration/validation 患者划分。
2. `_validate_reusable_checkpoint`：若已有血缘完备的同配置 checkpoint 则复用，否则同种子初始化。
3. `_train_variant` 分别训练 included/excluded：AdamW + CosineAnnealingLR + AMP + 梯度裁剪，唯一差异是损失函数（固定 epoch，不做模型选择）。
4. `_evaluate_variants`：在 calibration/validation 上计算每患者 residual enrichment（病灶区 PET−mean_pet 残差富集）与边界梯度误差（`_spatial_regions` 构造病灶/环区）。
5. `_decision_from_metrics`：主门 = validation 上 excluded−included 配对患者 bootstrap 95% CI 下界 >0 且 sign-flip p<0.05；边界安全门 = CI 上界 ≤ calibration 配对绝对差 q95 非劣性边际。
6. 写 decision.json（决策、证据、阈值、guardrails、stop_rule），异常路径也落盘 FAIL。

**流水线阶段**：`01_h2_residual_enrichment`（hypothesis=H2，requires `00C_cloud_data_gate`；decision_profile=model_mechanism）。**统计方法**：配对患者 bootstrap 95% CI（10k）、符号翻转置换检验、calibration q95 非劣性边际。
### `scripts/validate_h3_logsnr_recoverability.py`（H3，712 行）

**职责**：H3 患者级“可恢复性交叉点”检验。用解析 Brownian 桥噪声律（`BBDMBridgeSchedule` 的 m_t/σ_t）对 H2 病灶剔除残差加噪，在固定时间网格上测量 4 个预声明任务的可恢复性随 log-SNR 的衰减：coarse（LL2 余弦对齐）、shape（病灶支持 AUC）、intensity（病灶残差强度恢复）、local_frequency（病灶局部 L1-HH 余弦）。校准期冻结各任务阈值并选出分离最大的任务对，验证期一次性做配对门检验。

**接口**：
- `run(args) -> dict`：校验 H2 PASS 与同契约 → 加载 H2 downstream_mean_checkpoint → 生成可恢复性行 → 校准/验证 → 决策。
- `_recoverability_rows(model, datasets, schedule, timesteps, intensity_scale, ...)`：按划分×批×时间步生成样本级 (task, timestep, log_snr, recoverability) 行；噪声种子确定性派生（`_seeded_noise_like`）。
- `freeze_recoverability_thresholds(...)` / `crossing_log_snr(...)`：阈值冻结与“曲线跌破阈值的首个 log-SNR”计算；`select_calibration_pair(...)` 选分离最大的任务对；`_h3_decision(...)` 做验证期配对推断；`_intensity_scale(...)` 以校准患者病灶残差中位幅值定标 intensity 任务。
- `main(argv=None) -> int`：PASS→0，FAIL/异常→2（异常时写 FAIL decision.json）。CLI（流水线用法）：`--root --contract --h2-decision --manifest --cache-dir --cache-lineage --output`，另有 `--config` 与时间步网格参数（`DEFAULT_TIMESTEPS` 10 个点 0..950）。

**依赖**：内部: 复用 H2 脚本的 `IndexedDataset/_load_predictor/_loader/_make_partition_datasets/_spatial_regions`；`src.data.lineage`、`src.mechanism_validation.common`（bootstrap_mean/sign_flip_p/partition 等）、`src.model.frequency.haar.haar_dwt2`、`src.model.noise.base.BBDMBridgeSchedule`。外部: numpy、torch、yaml、scikit-learn(roc_auc_score)。

**输入**：LOCKED 契约、H2 decision.json（须 PASS 且 next_stage_allowed、契约哈希一致）、H2 均值预测 checkpoint、split_manifest、张量缓存 + cache_lineage、实验配置 YAML。

**输出**：`results/mechanism_validation/02_h3_recoverability_curves/` 下 decision.json（H3 门 + 选定任务对）、样本/患者级曲线 CSV、execution_metadata.json。

**处理过程**：
1. 前置校验：H2 决策 PASS、契约哈希一致、均值 checkpoint 存在，重建患者划分。
2. `_intensity_scale` 在校准划分上确定 intensity 任务定标。
3. `_recoverability_rows`：对每个划分、每个固定时间步，以确定性种子生成桥噪声，构造 noisy=(1−m)·residual+σ·noise，计算四任务指标与 log_snr。
4. `_patient_curves` 聚合到患者级任务×时间步曲线。
5. calibration 上 `freeze_recoverability_thresholds` 冻结阈值、`select_calibration_pair` 选最分离任务对（验证集不参与任何阈值/网格/课程选择）。
6. `_patient_crossings` 计算每位患者每任务的 crossing log-SNR；`_h3_decision` 在 validation 上做配对差 bootstrap 95% CI + sign-flip p。
7. 判定 PASS 需：可识别患者比例 ≥0.80 且 CI 下界 >0 且 p<0.05；写 decision.json。

**流水线阶段**：`02_h3_recoverability_curves`（hypothesis=H3，requires `01_h2_residual_enrichment`）。**统计方法**：患者级 crossing log-SNR 配对 bootstrap 95% CI + 符号翻转检验，可识别比例门槛 0.80。

### `scripts/validate_h4_noise_band_calibration.py`（H4，750 行）

**职责**：H4 纯噪声校准的带证据检验：对每患者×Haar 带×H3 时间步，计算 evidence = log(加噪残差带能量 / 匹配纯噪声带能量)，检验该证据在 log-SNR 与带身份之上对真实可恢复性（掩膜内余弦对齐）有增量预测力。

**接口**：
- `run(args) -> dict`：校验 H3 PASS → 加载均值 checkpoint → 采样带证据行 → 拟合校准探针 → 验证 → 决策。
- `_sample_band_rows(...)`：确定性种子生成 noisy 与 pure_noise（σ·noise），逐带计算掩膜内能量与对齐度。
- `fit_calibrated_probes(...)`（内含 `choose_alpha`）：mechanism_train 上分别拟合 baseline(log_snr+带 one-hot) 与 full(+evidence) 的 Ridge；`calibration_permutation_null(...)` 生成 calibration 患者置换零分布 q95；`validation_probe_statistics(...)` 输出 validation 增量 skill 患者级 bootstrap + sign-flip。
- `_feature_matrices(...)` 特征标准化 + 带 one-hot；`_mse_skill(...)` 相对基线 MSE 的 skill。
- `main(argv=None) -> int`：PASS→0，异常→2 并写 FAIL decision.json。CLI：`--root --config --contract --h3-decision --manifest --cache-dir --cache-lineage --output --batch-size(8) --num-workers(2) --device --null-replicates(2000) --bootstrap-replicates(10000)`。

**依赖**：内部: 复用 H2 脚本 `IndexedDataset/_load_predictor/_loader/_make_partition_datasets` 与 H3 脚本 `_seeded_noise_like`；`src.data.lineage`、`src.mechanism_validation.common`、`src.model.frequency.haar.haar_dwt2`、`src.model.noise.base.BBDMBridgeSchedule`。外部: numpy、torch、yaml、scikit-learn(Ridge)。

**输入**：LOCKED 契约、H3 decision.json（须 PASS）、H2 均值 checkpoint（经 H3 决策）、split_manifest、张量缓存 + lineage、配置 YAML。

**输出**：`results/mechanism_validation/03_h4_noise_calibration/` 下 `sample_band_evidence.csv`、`patient_band_evidence.csv`、`validation_patient_errors.csv`、`calibration.json`（α、置换零分布 q95、特征均值/方差）、`analysis_spec.json`、decision.json、execution_metadata.json。

**处理过程**：
1. 前置校验 H3 决策与契约/血缘一致，重建划分，加载均值预测器。
2. `_sample_band_rows`：固定时间步上构造 noisy 与 matched pure-noise，计算 6 个细节带（l1/l2 的 lh/hl/hh）的 evidence 与 recoverability。
3. `_patient_band_rows` 聚合患者×带×时间步均值。
4. `fit_calibrated_probes` 在 mechanism_train 上按 α 网格（0.01–100）拟合 baseline/full 两个 Ridge 探针。
5. `calibration_permutation_null` 用 calibration 患者置换（默认 2000 次）冻结增量 skill 零分布 q95 阈值。
6. `validation_probe_statistics` 在 validation 一次性评估：增量 skill 患者级 bootstrap 95% CI + sign-flip p。
7. PASS 需 CI 下界 > 零分布 q95 且 p<0.05；写 decision.json（明确“仅预测证据，非因果/互信息主张”）。

**流水线阶段**：`03_h4_noise_calibration`（hypothesis=H4，requires `02_h3_recoverability_curves`）。**统计方法**：Ridge 增量 skill + calibration 患者置换零分布 q95 + 患者级 bootstrap 95% CI 与符号翻转检验。
### `scripts/validate_artifact_safety.py`（H5 前置门，310 行）

**职责**：伪影安全门：在完全相同的数据、条件均值、初始化种子、优化器、epoch 预算与最终 EMA 端点下，对比“完整学习路由器（full）”与“硬零参考（null_reference，hard-all-null 谱注入）”，验证 full 不引入假热点、方向条纹、掩膜边界伪影或全局误差劣化。属 H5 路由角色检验的前置安全门。

**接口**：
- `run(args) -> dict`：加载正式上下文 → 复用/训练两个变体 → 评估 5 个预声明安全端点 → 交集门决策。
- `SAFETY_ENDPOINTS`：target_relative_false_hotspot_density（假热点）、stripe_abs_excess（方向条纹）、lesion_boundary_gradient_mae_norm（掩膜边界保真）、failure_any（复合失败）、mae（全图误差），全部 lower_is_better，效应定义为“正值 = full 更安全”。
- `main(argv=None) -> int`：PASS→0，异常→2 并写 FAIL decision.json（stop_rule：停止 H5/H6 形式化模型主张）。
- CLI：`--root --contract --h2-decision --curriculum-decision --base-config(默认 configs/experiments/slmf_png_spectral_router_v5.yaml) --cache-dir --cache-lineage --formal-run-dir(默认 results/mechanism_validation/formal_model_runs_v1) --output --epochs --mc-steps --force`。

**依赖**：内部: `src.mechanism_validation.common`（guardrails/写盘）；`src.mechanism_validation.model_experiments`（`load_formal_context / ensure_variant / load_evaluation / paired_patient_effects / calibration_absolute_margin / effect_statistics / DEFAULT_EPOCHS / DEFAULT_MC_STEPS`）。外部: 仅标准库（argparse/json/sys/datetime/pathlib），计算由内部模块承担。

**输入**：LOCKED 契约、H2 与课程（05_curriculum）decision.json（须 PASS）、张量缓存 + lineage、路由器基础配置 YAML；正式训练产物放在 formal_run_dir（可复用）。

**输出**：`results/mechanism_validation/06_artifact_safety/` 下 `patient_safety_effects.csv`、`calibration_margins.json`、`formal_runs.json`、decision.json、execution_metadata.json。

**处理过程**：
1. `load_formal_context` 校验契约/H2/课程决策、缓存血缘与均值 checkpoint。
2. `ensure_variant` 对 null_reference 与 full 各自复用或训练固定 epoch 的正式模型（EMA 端点，无模型选择）。
3. 对 5 个端点逐一：calibration 上 `paired_patient_effects` 计算配对患者效应，`calibration_absolute_margin` 取其绝对差 q95 作为非劣性边际。
4. validation 上 `effect_statistics`（种子派生）计算患者级 bootstrap 95% CI；单端点 PASS 需 `ci95_low >= -margin`。
5. 患者对数不足 5 时报错；全部端点 PASS 才整体 PASS（交集门，不得用验证集放宽边际）。
6. 写出明细 CSV、边际、formal_runs 与 decision.json（含 stop_rule：PASS 才允许路由消融与最终集成）。

**流水线阶段**：`06_artifact_safety`（hypothesis=H5_PREREQUISITE，requires `05_curriculum`）。**统计方法**：calibration 配对患者绝对差 q95 非劣性边际 + validation 患者级 bootstrap 95% CI 交集门。
### `scripts/validate_h5_router_role_separation.py`（H5，404 行）

**职责**：H5 路由角色分离检验。把固定端点的完整路由器（full）与三个单因子消融变体（no_ct_support / no_recoverability / no_artifact_safety）在完全相同的正式训练设置下对比。每个角色拥有一个预声明结局族，calibration 只能在族内选一个主端点（标准化配对效应最大者），验证集不得更改；H5 仅当三个角色的验证效应都倾向 full（患者 bootstrap CI 下界 >0 且单侧 sign-flip p<0.05）才 PASS。

**接口**：
- `run(args) -> dict`：校验伪影门 PASS → `load_formal_context` → `ensure_variant` 训练/复用 4 个变体 → 逐角色选端点与检验 → 交集门。
- `ROLE_SPECS`：ct_spatial_support（候选 lesion_centroid_distance、lesion_peak_to_boundary_distance）、residual_recoverability（directional_spectrum_error_norm、lesion_ring_core_ratio_error）、artifact_safety（假热点/条纹/边界/复合失败 4 端点），并附角色解读。
- `select_calibration_metric(full, ablation, candidates) -> (metric, lower_is_better, rows, audit)`：仅用 calibration，选标准化配对效应最大的端点（同分按预声明顺序）。
- `main(argv=None) -> int`：PASS→0，异常→2 并写 FAIL decision.json。CLI（流水线用法）：`--root --contract --h2-decision --curriculum-decision --artifact-decision --cache-dir --cache-lineage --output --epochs --mc-steps [--force]`（另有 `--base-config/--formal-run-dir`）。

**依赖**：内部: `src.mechanism_validation.common`；`src.mechanism_validation.model_experiments`（ensure_variant/load_evaluation/load_formal_context/paired_patient_effects/effect_statistics/DEFAULT_EPOCHS/DEFAULT_MC_STEPS/resolve）。外部: numpy 与标准库。

**输入**：LOCKED 契约、H2 与课程 decision.json（须 PASS）、伪影门（06_artifact_safety）decision.json（须 PASS 且契约一致）、张量缓存 + lineage、路由器基础配置 YAML、formal_run_dir。

**输出**：`results/mechanism_validation/07_h5_router_roles/` 下 `patient_role_effects.csv`、`frozen_role_metrics.json`（含候选审计）、`formal_runs.json`、decision.json、execution_metadata.json。

**处理过程**：
1. 校验伪影门 PASS 且契约哈希一致，加载正式上下文（均值 checkpoint、课程、血缘）。
2. `ensure_variant` 复用/训练 full 与三个消融变体（同种子、同 epoch、EMA 端点）。
3. 逐角色在 calibration 上 `select_calibration_metric` 冻结主端点（效应 = full 优于消融）。
4. validation 上 `paired_patient_effects` + `effect_statistics` 计算患者级 bootstrap 95% CI 与 sign-flip p。
5. 单角色 PASS 需 CI 下界 >0 且 p<0.05；三角色全 PASS 才整体 PASS（交集门）。
6. 写明细 CSV、冻结端点、formal_runs、decision.json（声明 mask 不参与路由、仅配对消融证据）。

**流水线阶段**：`07_h5_router_roles`（hypothesis=H5，requires `06_artifact_safety`）。**统计方法**：calibration 内标准化配对效应选端点 + validation 患者级 bootstrap 95% CI + 单侧符号翻转检验（三角色交集门）。

### `scripts/validate_h6_final_integration.py`（H6，369 行）

**职责**：H6 最终集成检验：复用固定端点正式训练的 full 路由器与硬零参考，要求小病灶主端点显著获益（优于参考），同时小病灶低估次端点与全部伪影/全局安全端点非劣。锁定数据集无 test 队列，故不作外部测试泛化主张，主张范围限定为锁定 PNG 队列的 held-out 患者验证。

**接口**：
- `run(args) -> dict`：校验 H5 PASS → 复用 null_reference/full 正式模型 → 主端点优效 + 次端点与 6 个安全端点非劣 → 交集门。
- `PRIMARY_SMALL_LESION = (small_lesion_topq_peak_error_norm, lower_is_better=True)`；`SECONDARY_SMALL_LESION = (small_lesion_underestimate, True)`；`SAFETY_ENDPOINTS`：假热点/条纹/边界梯度/复合失败/mae（lower 更好）+ ssim（higher 更好）。
- `main(argv=None) -> int`：PASS→0，异常→2 并写 FAIL decision.json。CLI（流水线用法）：`--root --contract --h2-decision --curriculum-decision --h5-decision --cache-dir --cache-lineage --output --epochs --mc-steps [--force]`（另有 `--base-config/--formal-run-dir`）。

**依赖**：内部: `src.mechanism_validation.common`；`src.mechanism_validation.model_experiments`（ensure_variant/load_evaluation/load_formal_context/paired_patient_effects/calibration_absolute_margin/effect_statistics/resolve）。外部: 标准库（无重型三方依赖）。

**输入**：LOCKED 契约、H2/课程/H5 decision.json（须 PASS 且契约一致）、张量缓存 + lineage、路由器基础配置、formal_run_dir 中可复用的正式模型。

**输出**：`results/mechanism_validation/08_h6_final_integration/` 下 `patient_final_effects.csv`、`calibration_margins.json`、`formal_runs.json`、decision.json（含 `formal_final_performance_claims_allowed`）、execution_metadata.json。

**处理过程**：
1. 校验 H5 决策 PASS 与契约一致，加载正式上下文，复用/重建 null_reference 与 full。
2. 主门：validation 上 full−null 的小病灶峰值误差配对效应，PASS 需 bootstrap CI 下界 >0 且 sign-flip p<0.05（优效）。
3. 次门：small_lesion_underestimate 用 calibration 配对绝对差 q95 作非劣性边际，validation CI 下界 ≥ −边际。
4. 安全门：6 个安全/全局端点逐一用同样的 q95 边际做非劣性检验，全部 PASS 才通过。
5. 三门交集决定 H6；写患者明细、边际、formal_runs 与 decision.json。
6. decision 显式声明：不作物理 SUV 主张、不作外部 test 泛化主张；PASS 才允许最终形式化性能主张。

**流水线阶段**：`08_h6_final_integration`（hypothesis=H6，requires `07_h5_router_roles`）。**统计方法**：主端点患者级 bootstrap 95% CI + 符号翻转优效检验；次/安全端点 calibration q95 非劣性边际交集门。
### `scripts/validate_ct_support_head.py`（H5 前置门，208 行）

**职责**：H1/H4 通过后冻结“CT 支持头”。不训练任何模型、不把 H1 重新解释为幅值预测，只登记下游允许的唯一 CT 角色：在 H1 选中的 Haar 带与方向上，输出有界的空间支持/可靠性场（解析式：两级 Haar 响应 → 冻结方向 → 逐 slice robust median/MAD 归一化 → sigmoid 到 [0,1]）。禁止角色：PET 病灶带幅值目标或替代品；mask 不作为输入、不可训练。

**接口**：
- `run(args) -> dict`：校验 H1 PASS + H4 PASS/next_stage_allowed + 三者契约哈希一致 → 从 H1 decision 提取选中带/方向/CI → 判定 eligible → 写出头配置与决策。
- eligible 条件：H1 定位门 PASS、选中带非空、方向 ∈ {+,-}、CI 两端齐全且下界 >0.5、置换 p<0.05、选中带−raw CT 的 CI 下界 >0 且 p<0.05、幅值门状态为 `PASS_NO_RECOVERABLE_INCREMENT`。
- `main(argv=None) -> int`：PASS→0，异常→2 并写 FAIL decision.json。CLI：`--root --contract --h1-decision --h4-decision --output`。

**依赖**：内部: 仅 `src.mechanism_validation.common`（load_json/write_json/guardrails/file_sha256）。外部: 标准库（argparse/json/sys/datetime/pathlib）。

**输入**：LOCKED 契约、H1 decision.json（00B）、H4 decision.json（03）。

**输出**：`results/mechanism_validation/04_ct_support_head/` 下 `ct_support_head.json`（头定义：head_type/带/方向/操作序列/允许与禁止的路由角色）、decision.json、execution_metadata.json。

**处理过程**：
1. 加载契约与 H1/H4 决策，逐项校验 PASS 与契约一致性。
2. 从 H1 决策读取 selected_band、direction、validation CI、选中带−raw 对比与幅值门状态。
3. 按 eligible 条件判定（含幅值增量不可恢复的措辞确认）。
4. 生成解析式 CT 支持头配置（冻结、无拟合阈值、无 mask 输入）。
5. 写 ct_support_head.json 与 decision.json；FAIL 时 stop_rule 要求停止 CT 头及依赖路由阶段、不得回改 H1 阈值。

**流水线阶段**：`04_ct_support_head`（hypothesis=H5_PREREQUISITE，requires `03_h4_noise_calibration`）。**统计方法**：无新统计检验，仅消费 H1 的 AUC/CI/置换证据做资格判定。

### `scripts/validate_mechanism_curriculum.py`（H5 前置门，275 行）

**职责**：无需验证反馈地冻结 H3 衍生的“时间/任务释放”课程。把 calibration 患者的各任务 crossing log-SNR 中位数映射到最近的预声明 H3 时间步，得到各损失的 `active_tau_max`；epoch 预热/爬坡按预声明分数由正式 epoch 数决定。验证指标与 checkpoint 性能都不得修改课程。

**接口**：
- `run(args) -> dict`：校验 H3/H4/CT 头均 PASS 且契约一致 → 读 H3 `patient_crossings.csv / patient_recoverability_curves.csv / analysis_spec.json` → 映射释放时间步 → 冻结课程。
- `TASK_TO_LOSSES`：coarse→lesion_roi_l1；shape→outside_peak_ranking；intensity→topk_lesion、normalized_lesion_peak；local_frequency→boundary_frequency、residual_wavelet。
- `_nearest_timestep(target_log_snr, grid)`：在 calibration 时间网格上取最近 log-SNR 的时间步（平手取更小 t）。
- `main(argv=None) -> int`：PASS→0，异常→2 并写 FAIL decision.json。CLI：`--root --contract --h3-decision --h4-decision --ct-head-decision --output --formal-epochs(50) --warmup-fraction(0.10) --ramp-fraction(0.10)`。

**依赖**：内部: `src.mechanism_validation.common`。外部: 标准库（csv/json/math/statistics 等），无训练。

**输入**：LOCKED 契约、H3/H4/CT 头 decision.json（均须 PASS）、H3 输出目录的 patient_crossings.csv、patient_recoverability_curves.csv、analysis_spec.json（取 num_train_timesteps）。

**输出**：`results/mechanism_validation/05_curriculum/` 下 `frozen_curriculum.json`（task_releases、loss_active_tau_max、router_epoch_schedule、ct_support_head 节选）、decision.json、execution_metadata.json。

**处理过程**：
1. 校验 H3/H4/CT 头决策与契约一致性；calibration 患者数 ≥5。
2. 从 H3 曲线构建 calibration 时间步×log-SNR 网格。
3. 逐任务收集 calibration 患者的 crossing log-SNR（可识别数 ≥ max(5, 半数)），取中位数并映射到最近预声明时间步，tau = t/(T−1)。
4. 按 `TASK_TO_LOSSES` 把 tau 写入各损失的 active_tau_max。
5. 由 formal_epochs×预声明分数计算 warmup/ramp epoch（须早于终点），连同 CT 支持头配置（选中带/方向、support-only、无 mask）写入 frozen_curriculum.json。
6. 写 decision.json（声明 training_performed=False、validation_used_to_choose_course=False）。

**流水线阶段**：`05_curriculum`（hypothesis=H5_PREREQUISITE，requires `04_ct_support_head`；流水线以 `--formal-epochs 50` 调用）。**统计方法**：calibration 患者中位数映射（无推断检验），课程冻结式设计。
### `scripts/run_mechanism_stage0_data_audit.py`（Stage 0A，1502 行）

**职责**：Stage 0A 本地数据血缘审计（假设编号 DATA，非 H1-H6）。纯诊断脚本：不建缓存、不改划分、不加载模型推理、不改训练代码。权威划分是 `main_data/split_manifest.csv`；原始 PNG 库可为本地 `Data/data` 或云端 `main_data`，身份由锁定的内容指纹而非物理目录名决定。所有图像比较先患者内聚合再按患者 bootstrap（10k 次）。

**接口**：
- `run(root, output, *, audit_checkpoint_inventory, raw_root_override, manifest_path_override, locked_contract_path) -> dict`：主审计流程，返回含 `data_gate` 的决策。
- `main(argv=None) -> int`：`data_gate==PASS` 返回 0 否则 2。CLI：`--root --output(默认 results/mechanism_validation/00_data_audit) --skip-checkpoint-inventory --raw-root(默认 Data/data 布局，云端传 main_data) --manifest(审计新/外部确认队列时的权威清单覆盖) --locked-contract(可选不可变契约，提供时审计内容必须逐项匹配)`。
- 关键函数：`_load_locked_contract`（契约自哈希+LOCKED 校验）、`_locked_contract_mismatches`（manifest 语义哈希/原始 PNG 组合哈希/文件数/模态数/划分样本与患者数/预处理配置哈希逐项比对）、`_manifest_fingerprint`、`_index_pngs`（重复检测）、`_patient_aggregate/_bootstrap_mean/_bootstrap_table`、`_audit_checkpoints`（遗留 checkpoint 内嵌元数据清点）。

**依赖**：内部: 无 src 导入，自包含（契约由本脚本产出/校验）。外部: numpy、Pillow、scipy.ndimage、yaml、tomllib 及标准库。

**输入**：split_manifest.csv、三模态 PNG 库（train/val/test × ct/pet/label）、可选 LOCKED 契约与清单覆盖。

**输出**：`results/mechanism_validation/00_data_audit/` 下 LOCKED `dataset_contract.json`（H1 直接读取；流水线命令另引用 `configs/dataset_contract_stage0a_v1.json` 作契约路径）、decision.json（data_gate）、患者级审计表 CSV/JSON。

**处理过程**：
1. 读取并指纹 manifest（canonical JSON 的 semantic_sha256），校验列结构与 sample_id 规则。
2. 索引三模态 PNG，检测重复/缺失，计算逐文件与组合 SHA-256。
3. 图像级诊断统计（尺寸/模式/病灶环/边界均值/CT-PET NCC 等），先患者内聚合。
4. 患者级 bootstrap 95% CI（AUDIT_SEED、10k 次）生成审计表。
5. 与 LOCKED 契约逐项比对（划分计数、模态计数、预处理配置哈希）；不一致即 FAIL。
6. 可选清点遗留 checkpoint 内嵌元数据（默认不阻断本地数据门）。
7. 汇总 data_gate 决策并落盘。

**流水线阶段**：`00A_data_audit`（hypothesis=DATA，scopes=local/all，requires 空，是整条链的根）。**统计方法**：患者内聚合 + 按患者 bootstrap 95% CI 的描述性审计（无假设检验门）。

### `scripts/run_cloud_stage0c_gate.py`（Stage 0C，1515 行）

**职责**：Stage 0C 云端数据/缓存/checkpoint 血缘门（假设编号 DATA）。只做检测不做训练：在允许训练前，验证云端构建的 PNG 缓存是对 LOCKED Stage 0A 契约的完整、数值忠实的实现。检查穷尽：manifest 语义与患者泄漏、全部原始 PNG 内容指纹、每样本恰一个 NPZ+JSON sidecar、缓存张量与契约预处理逐元素比对（atol 1e-6）、PET 极性/mask 阈值/dtype/形状、PASS 后才原子封装内容寻址的 `cache_lineage.json`、可选 checkpoint 必须内嵌全部锁定指纹。

**接口**：
- `run(args) -> dict`：主门流程，返回含 `cloud_training_gate` 的决策。
- `main(argv=None) -> int`：门 PASS 返回 0，硬门失败 2；未提供 checkpoint 时血缘为 DEFERRED，不阻断训练前缓存门。CLI：`--root --contract --manifest --raw-root --cache-dir(默认 cache/tensors_main) --output(默认 results/mechanism_validation/00C_cloud_data_gate) --checkpoint(可重复) --seal-cache(PASS 后原子写/刷新 <cache-dir>/cache_lineage.json，无已封装元数据时 cloud_training_gate PASS 所需)`。raw-root 探测顺序：契约声明路径 → 常见仓库位置 → 环境变量 `CT_PET_RAW_ROOT`。
- 关键函数：`_validate_contract`（自哈希+预处理哈希+元数据一致性+受支持预处理规则：192×192、PIL BILINEAR/NEAREST、uint8/127.5−1、PET 反相、mask>127）、`_validate_manifest`（sample_id 六位正则、患者前缀/切片后缀、患者跨 split 泄漏、划分计数）、`_validate_cache`（逐样本 NPZ 张量 vs 由原始 PNG 按契约重算的期望数组）、`_seal_or_validate_cache_lineage`、`_validate_checkpoints`。

**依赖**：内部: 无 src 导入，自包含检测器；消费 LOCKED 契约与 Stage 0A 审计产物。外部: numpy、Pillow 及标准库。

**输入**：LOCKED 契约（configs/dataset_contract_stage0a_v1.json）、split_manifest、原始 PNG 库、云端 NPZ 缓存目录 + sidecar、可选 checkpoint（.pt/.pth/.ckpt）。

**输出**：`results/mechanism_validation/00C_cloud_data_gate/` 下 decision.json（`cloud_training_gate`、checkpoint_lineage、training_allowed 等）与审计表；（--seal-cache 时）`cache/tensors_main/cache_lineage.json`。

**处理过程**：
1. `_validate_contract`：契约/预处理自哈希与受支持规则白名单校验。
2. `_validate_manifest`：结构、split 合法性、患者跨 split 泄漏、与契约划分计数一致、semantic_sha256 匹配。
3. `_validate_and_fingerprint_raw`：全部原始 PNG 内容指纹与契约组合哈希比对。
4. `_load_contract_expected_arrays` + `_validate_cache`：逐样本把缓存张量与按契约预处理重算的期望数组比对（ENGINEERING_ATOL=1e-6），校验 dtype/形状/元数据与 NPZ/JSON 一一对应。
5. `_seal_or_validate_cache_lineage`：全部内容检查 PASS 后原子封装/校验缓存血缘。
6. `_validate_checkpoints`：可选 checkpoint 须内嵌锁定数据与缓存指纹；缺省为 DEFERRED。
7. 汇总 `cloud_training_gate` 决策并落盘；异常路径写 FAIL decision.json。

**流水线阶段**：`00C_cloud_data_gate`（hypothesis=DATA，scopes=all，requires `00A_data_audit + 00B_h1_spectral_asymmetry`，流水线以 `--seal-cache` 调用）。**统计方法**：确定性指纹与逐元素数值比对（1e-6），无抽样推断。

### `scripts/run_all_mechanism_validations.py`（编排器，813 行）

**职责**：CT→PET 机制门的严格单命令编排器。按 `configs/mechanism_validation_pipeline_v1.json` 锁定顺序执行各阶段，只信任各阶段自己的 decision.json；命令非零、决策缺失/无效、契约不匹配、门失败或阶段未实现都会阻断全部下游阶段。绝不提升遗留 checkpoint、绝不把缺失证据变成 PASS（fail-closed）。典型用法：`pixi run mechanism-validate-local`（本地数据前缀）/`pixi run mechanism-validate-all`（完整流水线）。

**接口**：
- `class PipelineSpecError(ValueError)`：流水线规范内部不一致时抛出。
- `_validate_pipeline_spec(spec) -> list[stage]`：校验 schema_version、唯一 stage id、scopes⊆{local,all}、requires 只能引用更早阶段、必需字段、implemented 阶段须有字符串命令列表；合并 decision_profiles 与阶段级 required_assertions。
- `_selected_stages(stages, scope, from_stage, through_stage)`：按 scope 与起止阶段筛选。
- `_format_command(stage, ...)`：用 {python}{root}{contract}{raw_root}{cache_dir}{output} 上下文渲染命令占位符。
- `_decision_passes(stage, payload, expected_contract_sha256)`：校验 pass_path/pass_value、契约哈希与全部 required_assertions（嵌套点路径取值）。
- `_run_command(command, cwd, log_path)`：subprocess 执行并把日志写文件；`_verified_external_prerequisites`（DFS）核验已存在决策；`_mirror_stage_decision` 把外部决策镜像进 run 目录。
- `_overall_decision(...)`：聚合 H1-H6 的 hypothesis_status；仅当 scope==all 且六个假设全 PASS 才 `all_hypotheses_validated=model_mechanism_claims_allowed=true`；最终性能主张还须 H6 PASS。
- `main(argv=None) -> int`：决策 PASS/DRY_RUN 返回 0，否则 3；规范/运行错误返回 2 并写 ERROR decision.json；KeyboardInterrupt 130。CLI：`--root --pipeline --contract --raw-root --cache-dir(cache/tensors_main) --summary-dir(results/mechanism_validation/99_full_pipeline) --scope(local|all，默认 all) --from-stage --through-stage --run-id --allow-partial(实现前缀跑到首个缺失验证器) --dry-run`。

**依赖**：内部: 不导入 src，纯编排；消费 `configs/mechanism_validation_pipeline_v1.json` 与 `configs/dataset_contract_stage0a_v1.json`，并以子进程调用上述各 validate/audit/gate 脚本。外部: 标准库（subprocess/hashlib/json 等）。

**输入**：流水线规范 JSON、LOCKED 契约、原始 PNG 根（自动探测契约路径/main_data/Data/data 或显式 --raw-root）、缓存目录、各上游阶段 decision.json（外部前提核验）。

**输出**：`results/mechanism_validation/99_full_pipeline/<run-id>/` 下各阶段决策镜像、命令日志、汇总 decision.json（hypothesis_status、stage_status、claims_allowed 标志、stop_rule）。

**处理过程**：
1. 载入并 `_validate_pipeline_spec` 规范化阶段（含 profile 断言合并）。
2. `_select_raw_root` 按目录形状探测/校验原始 PNG 根。
3. `_verified_external_prerequisites`：对已声称完成的外部阶段决策做 DFS 核验（存在、PASS、契约一致），防伪造。
4. 预检所选范围内所有阶段 implemented（缺 `--allow-partial` 则失败）。
5. 依拓扑顺序执行：渲染命令 → `_run_command`（日志落盘）→ 读该阶段 decision.json → `_decision_passes` 校验门值/契约/断言；任一失败即阻断依赖阶段（fail-closed）。
6. `_overall_decision` 聚合假设与阶段状态，判定是否允许模型机制/最终性能主张。
7. `--dry-run` 只输出将要执行的命令与决策 DRY_RUN。

**流水线阶段**：自身不属于任何假设阶段，是 `mechanism_validation_pipeline_v1` 的执行入口（编排 00A→00B→00C→01→02→03→04→05→06→07→08 全链）。
### 模块依赖小结

**依赖的内部模块**（grep 查证）：
- `src.mechanism_validation.common`：被 H2/H3/H4/CT 头/课程/伪影/H5/H6 全部消费（患者划分、配对 bootstrap、sign-flip、guardrails、读写工具）。
- `src.mechanism_validation.model_experiments`：被伪影安全、H5、H6 消费（`load_formal_context/ensure_variant/load_evaluation/paired_patient_effects/calibration_absolute_margin/effect_statistics`）。
- `src.data.dataset.CachedDataset`、`src.data.lineage`：被 H2/H3/H4 消费（张量缓存与 checkpoint 血缘）。
- `src.model.frequency.haar`、`src.model.mean_predictor`、`src.model.mean_pretraining`、`src.model.noise.base.BBDMBridgeSchedule`：分别被 H2/H3/H4 消费（Haar 变换、低频均值预测器、LL2 目标、桥噪声调度）。
- 脚本间复用：H3 复用 H2 的 `IndexedDataset/_load_predictor/_loader/_make_partition_datasets/_spatial_regions`；H4 复用 H2 同套工具与 H3 的 `_seeded_noise_like`。Stage 0A/0C 与编排器自包含，不导入 src。

**消费方**（grep 查证）：
- `configs/mechanism_validation_pipeline_v1.json`：以命令形式调用全部 12 个脚本（各阶段 command 字段）。
- `pixi.toml`：`mechanism-validate-local` / `mechanism-validate-all` 任务别名指向 `scripts/run_all_mechanism_validations.py`。
- `tests/test_cloud_stage0c_gate.py`、`tests/test_mechanism_validation_runner.py`：单测直接导入 0C 门与编排器内部函数。
- `scripts/validate_h4_v2_external_confirmation.py`：导入 0C 门的 `_manifest_semantic_sha256`；`scripts/run_h4_v2_pipeline.py` 复用 Stage 0A 审计；`scripts/audit_checkpoint_lineage.py` 在文档中与 0C 门对照（不重哈希原始 PNG）。
- `configs/cloud_worktree_sync.yaml`：云同步清单包含编排器脚本。未查到其他直接导入本组脚本的仓库内消费方（artifacts/ 下的副本除外）。

**阶段链速览**：00A(DATA) → 00B(H1) → 00C(DATA) → 01(H2) → 02(H3) → 03(H4) → 04(H5 前置) → 05(H5 前置) → 06(H5 前置) → 07(H5) → 08(H6)；除 H1 纯本地外，模型阶段均需云端缓存血缘；任一阶段 FAIL 阻断全部下游。


---

## 9. H4-v2 与 V2 冻结/生产管线脚本（上：H4-v2 交叉验证）

### 模块概述

本组脚本构成 H4-v2"不确定性感知证据路由"假设的完整证据生命周期：`develop_h4_v2_uncertainty_aware.py` 在锁定的 99/25/31 患者分片上开发并密封冻结探针（仅开发、永不自证 PASS）；`run_h4_v2_pipeline.py` 一键编排"开发冻结 →（可选）外部新队列 Stage 0A/0C 门 → 零重叠外部确认"。`validate_h4_v2_external_confirmation.py` 是外部确认的执行者，对 9 项预检与主优越性/活跃覆盖/亚组非劣效门全部 fail-closed，PASS 才解锁 CT 支持头。在缺乏新队列期间，`freeze_h4_v2_internal_cv_plan.py` 依据密封配置 `configs/h4_v2_internal_exploratory_nested_cv_v1.json` 在查看任何结局前预冻结 5×5 患者级嵌套 CV 划分；`run_h4_v2_internal_exploratory_pipeline.py` 编排"冻结划分 → 外层评估"两步；`validate_h4_v2_internal_cv.py` 执行 155 人严格折外评估并给出中文探索性结论。所有脚本共享 fail-closed 决策模式（任何异常都落盘 FAIL decision）、哈希密封与谱系校验，机制核心实现在 src/mechanism_validation/h4_v2.py 与 internal_cv.py 中复用。

### `scripts/validate_h4_v2_external_confirmation.py` — H4-v2 冻结探针的零重叠外部确认（fail-closed）

**职责**：在真正全新的患者队列上确认已冻结的 H4-v2 探针（证据路由机制）。对"患者重叠、数据/预处理谱系、探针被改动、队列过小、路由活跃覆盖不足、主比较失败、任一预注册鲁棒性亚组失败"全部 fail-closed，并产出权威 decision.json 决定后续阶段（CT 支持头）是否解锁。

**接口**：`python scripts/validate_h4_v2_external_confirmation.py --external-contract <json> --external-manifest <json> --external-cache-dir <dir> --external-cache-lineage <json> --confirmation-cohort-id <id> [--probe results/mechanism_validation_v2/00_h4_v2_development/frozen_probe.json] [--confirmation-split test] [--mean-checkpoint .../mean_excluded_fixed.pt] [--output results/mechanism_validation_v2/01_h4_v2_external_confirmation] [--num-workers 2] [--device auto]`。返回码：0=PASS，2=FAIL/PREFLIGHT 失败/异常。核心函数：`_validate_contract`（LOCKED 自哈希校验）、`_preflight_failure`、`_external_attributes`（ll2 病灶/外环对数比属性）、`_subgroup_statistics`（非劣效亚组检验）、`_run`/`main`（异常时写 FAIL decision）。

**依赖**：内部：`src/mechanism_validation/h4_v2`（MODEL_ORDER、validate_probe、apply_hierarchical_calibration、apply_comparators、assign_subgroups、effect_statistics、patient_errors、patient_set_sha256、fit_patient_attributes、normalize_ids、add_context_features）、`scripts/validate_h1_local_spectral_asymmetry`（RING_RADIUS、_band_energy_maps）、`scripts/validate_h2_pathology_excluded_residual`（IndexedDataset、_load_predictor）、`scripts/validate_h4_noise_band_calibration`（_sample_band_rows）、`scripts/run_cloud_stage0c_gate`（_manifest_semantic_sha256）、`src/data/dataset.CachedDataset`、`src/data/lineage`（load/validate_checkpoint_data_lineage）、`src/mechanism_validation/common`（load_json/read_manifest/write_json/file_sha256/canonical_json_sha256）、`src/model/noise/base.BBDMBridgeSchedule`。外部：numpy、pandas、torch、scipy.ndimage、argparse、json。

**输入**：开发期冻结的 frozen_probe.json（含 probe_sha256、development_dataset、formal_confirmation_gate、mechanism 各组件、robustness_subgroups、wording_boundary）；外部数据集契约（LOCKED + 自哈希）、外部 manifest、外部缓存目录与缓存谱系文件；病理剔除均值检查点 mean_excluded_fixed.pt（sha256 须与探针一致且内嵌开发期谱系）。

**输出**：`<output>/decision.json`（schema_version=2、PASS/FAIL、preflight_checks、四大比较效应量、H4_v2 状态块、guardrails、next_stage_allowed）、`preflight_checks.json`、`execution_metadata.json`、`external_sample_predictions.csv`、`external_patient_errors.csv`、`external_patient_attributes.csv`。

**处理过程**：
1. 校验冻结探针（validate_probe）并核对 4 个实现文件（h4_v2.py、本脚本、H4-v1 脚本、H1 属性脚本）的 sha256 与探针冻结值完全一致，防探针漂移。
2. 校验外部契约 LOCKED 且自哈希匹配；计算 manifest 语义哈希。
3. 执行 9 项预检：探针冻结态、队列身份≠开发队列、数据集契约不同、预处理配置相同、manifest-契约一致、原始字节为新数据、manifest 内容为新、患者零重叠、最少患者数/分片非空；任一失败即写 PREFLIGHT FAIL 并退出码 2。
4. 校验外部缓存谱系与检查点双重谱系（检查点内嵌开发期 dataset_contract/preprocessing 哈希须与探针一致）。
5. 用 CachedDataset + IndexedDataset 装载外部 split，校验缓存样本/患者集合与权威 manifest 完全一致。
6. 以探针冻结的 bridge schedule、timesteps、batch_size 调用 _sample_band_rows 生成各证据通路频带残差行，逐一执行 normalize_ids → add_context_features → apply_hierarchical_calibration（冻结总体统计+收缩系数）→ apply_comparators（冻结模型/标准化/置信度阈值）→ patient_errors。
7. 计算 4 组患者级 bootstrap 95%CI 效应（h3_vs_no_route、original_vs_h3、uncertainty_vs_h3、uncertainty_vs_original）；主判据 = uncertainty_vs_h3 与 uncertainty_vs_original 的 ci95_low>0 且 sign_flip_p<0.05；同时检查路由活跃比例≥门槛及三个鲁棒性亚组（few_slices/small_lesion/low_contrast，非劣效界+最小患者数）。
8. 写 decision.json 与全部 CSV/元数据；PASS 才置 next_stage="CT_support_head"，否则 CT 头/课程化/H5/H6 保持封锁；main() 捕获一切异常同样写 FAIL decision（EXCEPTION_FAIL_CLOSED）返回 2。

### `scripts/run_h4_v2_pipeline.py` — 一键 H4-v2 开发冻结 + 外部确认流水线（编排器）

**职责**：以单条命令串起 H4-v2 的完整生命周期：本地开发/冻结 →（可选）外部新队列 Stage 0A 数据审计 → Stage 0C 缓存/谱系门 → 冻结探针外部确认。任何子阶段失败即整体 FAIL，并决定是否解锁下一阶段（CT 支持头）。

**接口**：`python scripts/run_h4_v2_pipeline.py [--root .] [--output-root results/mechanism_validation_v2] [--external-raw-root DIR] [--external-manifest JSON] [--external-cache-dir DIR] [--confirmation-cohort-id ID] [--confirmation-split test] [--num-workers 2] [--device auto] [--plan]`。返回码：0=PASS/DEFERRED_NEW_COHORT，2=FAIL。`--plan` 仅打印 configs/mechanism_validation_pipeline_v2.json 的冻结流水线定义后退出。核心函数：`_run`（子进程执行并回传码）、`_finish`（汇总 stages 写 decision.json）、`build_parser`、`main`。

**依赖**：内部：`scripts/develop_h4_v2_uncertainty_aware.py`（阶段0 开发冻结）、`scripts/run_mechanism_stage0_data_audit.py`（阶段1 外部 Stage 0A）、`scripts/run_cloud_stage0c_gate.py`（阶段2 外部 Stage 0C 封缓存）、`scripts/validate_h4_v2_external_confirmation.py`（阶段3 确认）、`src/mechanism_validation/common`（load_json/write_json）、`configs/mechanism_validation_pipeline_v2.json`（计划文本）。外部：argparse、subprocess、json。

**输入**：无外部输入时仅运行本地开发/冻结；若提供任一外部参数（external-raw-root/manifest/cache-dir/confirmation-cohort-id）则要求四项齐全，且必须已存在 00_h4_v2_development/frozen_probe.json（禁止云端重冻结探针）。

**输出**：`<output-root>/99_pipeline/decision.json`（schema_version=2、各阶段 stages 状态、decision=PASS/FAIL/DEFERRED_NEW_COHORT、next_stage）；子阶段各自落盘于 00_h4_v2_development、01_external_stage0a、02_external_stage0c、03_h4_v2_confirmation。

**处理过程**：
1. 解析参数并创建 99_pipeline 目录；`--plan` 时直接打印流水线配置文件内容返回 0。
2. 防重冻结守卫：若给了外部参数但 frozen_probe.json 不存在 → FAIL（"cloud-side refreezing is forbidden"）。
3. 运行 develop_h4_v2_uncertainty_aware.py 完成本地开发/冻结，非零退出即 FAIL；成功则记录 probe_sha256。
4. 未提供任何外部参数 → 写 DEFERRED_NEW_COHORT（探针已冻结，等待零重叠新队列）并返回 0。
5. 外部四参数不齐全 → FAIL（"External confirmation inputs are incomplete"）。
6. 依次执行外部 Stage 0A 数据审计（--skip-checkpoint-inventory，产出 dataset_contract.json）与 Stage 0C 门（--seal-cache 封缓存并生成 cache_lineage.json），任一非 PASS 即 FAIL。
7. 调用 validate_h4_v2_external_confirmation.py，把冻结探针、外部契约/manifest/缓存谱系逐一透传；确认非 PASS 即 FAIL 且下游机制保持封锁。
8. 全部通过 → 写 PASS decision、next_stage="CT_support_head"，返回 0。

### `scripts/develop_h4_v2_uncertainty_aware.py` — H4-v2 不确定性感知证据协议的开发与冻结（仅开发，永不自证）

**职责**：基于 H4-v1 失败后的锁定分片（99/25/31 患者）开发并"密封"（seal）H4-v2 不确定性感知证据路由探针，把全部阈值/模型/亚组规格/确认门写进 frozen_probe.json；明确 31 个验证患者已被 H4-v1 暴露，其结果仅为探索性诊断，本命令永远不产生 PASS，正式确认必须交给外部确认命令。

**接口**：`python scripts/develop_h4_v2_uncertainty_aware.py [--root .] [--contract configs/dataset_contract_stage0a_v1.json] [--h3-decision results/mechanism_validation/02_h3_recoverability_curves/decision.json] [--h4-v1-decision results/mechanism_validation/03_h4_noise_calibration/decision.json] [--h4-v1-samples .../sample_band_evidence.csv] [--h1-samples .../00B_h1_spectral_asymmetry/sample_band_metrics.csv] [--output results/mechanism_validation_v2/00_h4_v2_development] [--allow-refreeze]`。返回码：0=冻结成功，2=异常 FAIL。核心函数：`_validate_contract`、`_check_sources`（前置证据谱系与 99/25/31 分片校验）、`_subgroup_diagnostics`、`_run`/`main`（异常 fail-closed 写 decision）。

**依赖**：内部：`src/mechanism_validation/h4_v2`（BANDS、CONFIDENCE_QUANTILE、MIN_ACTIVE_COVERAGE、MIN_CONFIRMATION_PATIENTS、MIN_SUBGROUP_PATIENTS、MODEL_ORDER、NEIGHBOR_RADIUS、PATIENT_SHRINKAGE_K、RIDGE_ALPHA 及 fit_population_stats/fit_comparators/fit_original_standardization/freeze_subgroup_spec/seal_probe/validate_probe/development_diagnostics 等全套）、`src/mechanism_validation/common`（canonical_json_sha256/file_sha256/load_json/write_json）。外部：numpy、pandas、argparse、json。

**输入**：Stage 0A 数据集契约（自哈希校验）；H3 decision.json + analysis_spec.json（须 PASS 且 timestep 网格一致）；H4-v1 decision.json（须 FAIL，即 v2 是失败后的新协议）与其 sample_band_evidence.csv；H1 sample_band_metrics.csv（患者属性源）。

**输出**：`frozen_probe.json`（探针 sha256 自封，含 mechanism 全参数、robustness_subgroups、formal_confirmation_gate、downstream_unlock_policy、5 个实现文件的 sha256）、`decision.json`（DEVELOPMENT_ONLY + FROZEN + DEFERRED_NEW_COHORT）、`development_sample_predictions.csv`、`development_patient_errors.csv`、`development_patient_attributes.csv`、`development_diagnostics.json`、`calibration_diagnostics.json`、`router_coverage.json`、`development_subgroup_diagnostics.json`、`execution_metadata.json`；重冻结时归档 `superseded_probe_<sha>.json`。

**处理过程**：
1. 校验契约自哈希；`_check_sources` 强制 H3=PASS、H4-v1=FAIL、两者契约谱系一致，H1/H4 样本身份一致，患者分片严格等于锁定的 99/25/31，多时间步网格完整（样本数×BANDS×timesteps），并与冻结 H3 网格一致。
2. 仅用 mechanism_train+calibration 拟合：add_context_features → fit_population_stats → apply_hierarchical_calibration → fit_original_standardization → fit_comparators（岭回归，固定 RIDGE_ALPHA 不扫参）。
3. 置信度阈值取 calibration 分区证据置信度的 CONFIDENCE_QUANTILE 分位（threshold_source=calibration_q25），弃权时回退 h3_fixed_schedule。
4. 用 fit 分区患者属性 freeze_subgroup_spec 冻结三个鲁棒性亚组（few_slices/small_lesion/low_contrast）规格。
5. seal_probe 密封探针：写入 claim、wording_boundary、开发数据集指纹（患者集合哈希、暴露验证患者）、源工件 sha256（含 h4_v2.py 与 4 个脚本的实现哈希）、bridge_schedule、置信度公式、formal_confirmation_gate（主优越性 ci95_low>0 且 sign_flip_p<0.05、亚组非劣效、最小患者数/活跃覆盖）与 downstream_unlock_policy（PASS 前封锁 CT 头/课程化/H5/H6）。
6. 落盘冲突守卫：已存在不同哈希的探针时，除非 --allow-refreeze（归档旧探针后替换），否则拒绝覆盖；相同哈希则直接复用。
7. 密封后才对已暴露的 validation 患者计算诊断：apply_comparators → patient_errors → development_diagnostics（全分区与 calibration 专属 seed）、router_coverage（各分区活跃/弃权比例）、_subgroup_diagnostics（验证分区亚组效应，标记 exploratory_only）。
8. 写 DEVELOPMENT_ONLY decision（model_mechanism_claims_allowed=False、next_stage_allowed=False）；main() 异常时写 EXCEPTION_FAIL_CLOSED FAIL 返回 2，冻结未完成前禁止外部确认与下游阶段。

### `scripts/freeze_h4_v2_internal_cv_plan.py` — H4-v2 内部探索性嵌套 CV 计划的预冻结（外层评估前）

**职责**：在查看任何 H4-v2 可恢复性结果之前，按 `configs/h4_v2_internal_exploratory_nested_cv_v1.json` 冻结患者级 5×5 嵌套交叉验证划分（外层折 + 每折内层 train/calibration 角色），保证折平衡只依赖 H1 患者属性而非结局变量，产出密封的 frozen_plan.json 作为后续内部探索管线的唯一划分依据。

**接口**：`python scripts/freeze_h4_v2_internal_cv_plan.py [--root .] [--config configs/h4_v2_internal_exploratory_nested_cv_v1.json] [--h4-v1-decision .../03_h4_noise_calibration/decision.json] [--h4-v1-samples .../sample_band_evidence.csv] [--h1-samples .../00B_h1_spectral_asymmetry/sample_band_metrics.csv] [--output results/mechanism_validation_v2/02_h4_v2_internal_exploratory_nested_cv/00_frozen_plan]`。返回码：0=FROZEN/已存在且一致，2=异常 FAIL。核心函数：`_validate_existing_plan`（幂等复跑校验）、`_run`/`main`（PLAN_FREEZE_EXCEPTION_FAIL_CLOSED）。

**依赖**：内部：`src/mechanism_validation/internal_cv`（balanced_partition_search、build_nested_roles、assert_patient_partition_integrity、partition_fingerprint、fold_balance_table、patient_balance_attributes、seal_mapping、validate_sealed_mapping）、`src/mechanism_validation/h4_v2`（BANDS、normalize_ids）、`src/mechanism_validation.common`（file_sha256/load_json/write_json）。外部：pandas、argparse、json。

**输入**：密封的 CV 配置（config_sha256=891a30e…，155 患者/1191 样本约束、5×5 折、seed=20260811/20260911、4096/2048 平衡候选、quantile_bins=4、三连续+两类别平衡字段、difficult 患者 044/153 必须在场）；H4-v1 decision（必须保持 FAIL）；H4-v1 sample_band_evidence.csv（仅用于身份核验）与 H1 sample_band_metrics.csv（平衡属性来源）。

**输出**：`00_frozen_plan/frozen_plan.json`（plan_sha256 密封、partition.fingerprint_sha256、frozen_files 三文件哈希、freeze_guardrails）、`outer_patient_assignments.csv`、`nested_patient_roles.csv`、`outer_fold_balance.csv`、`decision.json`（FROZEN + outer_evaluation_allowed=true）、`execution_metadata.json`。

**处理过程**：
1. 校验配置自身密封哈希（validate_sealed_mapping, hash_field=config_sha256）。
2. 幂等守卫：若 frozen_plan.json 已存在，则校验其密封哈希、config_sha256、source_artifacts 与当前输入一致且状态为 FROZEN_BEFORE_OUTER_EVALUATION，一致则直接重放旧 decision 返回 0（防重划分）。
3. 核验 H4-v1 决策必须仍为 FAIL；H1 与 H4-v1 的样本/患者身份完全一致，H4-v1 频带集合与冻结 BANDS 一致。
4. 按配置 dataset_constraints 硬校验：患者=155、样本=1191、原始分区计数 99/25/31、困难患者 044/153 在场。
5. patient_balance_attributes 生成平衡属性后，balanced_partition_search 以 outer_seed 在 4096 个候选中搜索最平衡的 5 折外层划分；build_nested_roles 再以 inner_seed 在每个外层折内构建 5 折内层（内层仅取 mechanism_train/calibration 两角色）。
6. assert_patient_partition_integrity 断言患者不跨折、折数完整；partition_fingerprint 计算划分指纹；fold_balance_table 输出外层折平衡表。
7. 落盘三个划分 CSV，写 frozen_plan.json（balance_uses_h4_v2_outcomes=False、slice_level_random_split=False 等冻结守卫）并用 seal_mapping 密封 plan_sha256；写 FROZEN decision（outer_evaluation_allowed=True）与执行元数据；异常时 fail-closed 写 FAIL 并置 outer_evaluation_allowed=False 返回 2。

与 `configs/h4_v2_internal_exploratory_nested_cv_v1.json` 的联动：本脚本是该配置的唯一执行入口——配置中的 partition 段决定折结构/种子/平衡候选，dataset_constraints 决定硬校验，gates 段（主比较、≥4 个正向外层折、活跃覆盖≥0.5、亚组非劣效）则供下游 run_h4_v2_internal_exploratory_pipeline.py / validate_h4_v2_internal_cv.py 消费；frozen_plan.json 的 plan_sha256 与 config_sha256 会被下游脚本复验。

### `scripts/run_h4_v2_internal_exploratory_pipeline.py` — 一键冻结划分 + 内部探索嵌套 CV 外层评估（编排器）

**职责**：把内部探索流程封装成单命令：先调用 freeze_h4_v2_internal_cv_plan.py 冻结患者级嵌套划分（幂等），再调用 validate_h4_v2_internal_cv.py 执行外层评估，最后把两个子阶段的结论汇总为管线级 decision.json（含中文结论 wording 与"是否推荐后续内部使用"标志）。

**接口**：`python scripts/run_h4_v2_internal_exploratory_pipeline.py [--root .] [--config configs/h4_v2_internal_exploratory_nested_cv_v1.json] [--plan]`。返回码：透传子阶段退出码（冻结失败返回 freeze_code，评估失败返回 evaluation_code）。`--plan` 仅打印 pipeline_id/scope/两条将执行的命令/外层评估是否已跑过，返回 0。核心函数：`_run_command`（子进程回传码）、`main`。

**依赖**：内部：`scripts/freeze_h4_v2_internal_cv_plan.py`（子阶段0 冻结划分）、`scripts/validate_h4_v2_internal_cv.py`（子阶段1 外层评估）、`src/mechanism_validation.common`（load_json/write_json）；固定目录 `results/mechanism_validation_v2/02_h4_v2_internal_exploratory_nested_cv` 下的 00_frozen_plan/01_outer_evaluation/99_pipeline。外部：argparse、subprocess、json。

**输入**：密封的嵌套 CV 配置（透传给两个子脚本）；无数据文件直接输入。

**输出**：`99_pipeline/decision.json`（stage=99_H4_V2_INTERNAL_EXPLORATORY_PIPELINE，含 plan_sha256、partition_fingerprint、conclusion、completed_outer_evaluation、H4_v1_preserved、recommended_for_subsequent_internal_use）；子阶段产物分别落在 00_frozen_plan/ 与 01_outer_evaluation/。

**处理过程**：
1. 解析配置路径与三个固定输出子目录；组装 freeze 与 evaluation 两条子命令（均透传 --config，评估命令额外透传 --plan frozen_plan.json 与 --output）。
2. `--plan` 模式：只打印管线定义（pipeline_id、scope、命令清单、外层评估是否已存在）后返回 0。
3. 先跑计划冻结；非零退出 → 写 failure_phase=PLAN_FREEZE 的 FAIL decision（recommended_for_subsequent_internal_use=False）并透传退出码。
4. 再跑外层评估子命令，读取 01_outer_evaluation/decision.json。
5. 以评估 decision 是否含 conclusion 且 decision∈{PASS,FAIL} 判定 completed_outer_evaluation；把 plan_sha256/partition_fingerprint/conclusion/H4_v1_preserved/recommended_for_subsequent_internal_use 原样上提为管线级 decision。
6. 写 99_pipeline/decision.json 并透传评估退出码结束。

与配置/子脚本的联动：本脚本是 `configs/h4_v2_internal_exploratory_nested_cv_v1.json` 的编排入口——它保证"划分冻结一定先于外层评估"的时序，且 frozen_plan.json 由冻结子阶段生成后以路径参数交给评估脚本复验，避免评估阶段擅自重建划分。

### `scripts/validate_h4_v2_internal_cv.py` — 冻结划分下的 155 患者严格折外 H4-v2 评估

**职责**：对全部 155 名内部患者执行"每人恰好折外一次"的嵌套 CV 外层评估：每个外层折内仅用该折内层（mechanism_train/calibration）患者重拟合总体统计/标准化/岭回归比较器与置信度阈值，再对折外患者预测，最终按冻结 gates 判定 PASS/FAIL 并给出配置预设的中文结论（"v2 获得现有数据集内部探索性支持"或"未获得支持"）。

**接口**：`python scripts/validate_h4_v2_internal_cv.py [--root .] [--config configs/h4_v2_internal_exploratory_nested_cv_v1.json] [--plan .../00_frozen_plan/frozen_plan.json] [--h4-v1-decision .../03_h4_noise_calibration/decision.json] [--h4-v1-samples .../sample_band_evidence.csv] [--h1-samples .../00B_h1_spectral_asymmetry/sample_band_metrics.csv] [--output .../02_h4_v2_internal_exploratory_nested_cv/01_outer_evaluation]`。返回码：0=PASS（或幂等重放已完成 PASS），2=FAIL/异常。核心函数：`_verify_frozen_inputs`（冻结输入五重校验）、`_comparison`/`_all_comparisons`（四组比较+逐折分解）、`_subgroup_statistics`（折余量调整非劣效）、`_improvement_summary`、`_run`/`main`（OUTER_EVALUATION_EXCEPTION_FAIL_CLOSED，技术失败归档仅允许重试一次）。

**依赖**：内部：`src/mechanism_validation/h4_v2`（add_context_features、fit_population_stats、apply_hierarchical_calibration、fit_original_standardization、fit_comparators、apply_comparators、patient_errors、fit_patient_attributes、freeze_subgroup_spec、assign_subgroups、normalize_ids、BANDS）、`src/mechanism_validation/internal_cv`（validate_sealed_mapping、partition_fingerprint、assert_patient_partition_integrity）、`src/mechanism_validation.common`（bootstrap_mean、sign_flip_p、file_sha256、load_json、write_json）。外部：numpy、pandas、argparse、json。

**输入**：密封 config（config_sha256）与密封 frozen_plan.json（plan_sha256、partition fingerprint、frozen_files 哈希）；H4-v1 decision（必须保持 FAIL）与 sample_band_evidence.csv；H1 sample_band_metrics.csv；划分 CSV（outer_patient_assignments.csv、nested_patient_roles.csv）。

**输出**：`01_outer_evaluation/` 下：`decision.json`（PASS/FAIL + 中文 conclusion、四门 gate_status、comparisons、guardrails、recommended_for_subsequent_internal_use、validation_exposure_disclosure）、`out_of_fold_sample_predictions.csv`、`out_of_fold_patient_errors.csv`、`out_of_fold_patient_report.csv`（含逐患者效应与 improved 标志）、`difficult_patients_044_153.csv/.json`、`fold_fitted_components.json`（每折拟合组件全量）、`outer_fold_results.json`、`comparison_statistics.json`、`subgroup_statistics.json`、`original_partition_statistics.json`（含 31 名先前暴露验证患者的披露）、`execution_metadata.json`；技术失败时归档 `technical_failure_before_outer_evaluation.json`。

**处理过程**：
1. 幂等与防覆盖守卫：已有完整 decision（同 stage/pipeline/plan_sha256 且含 conclusion）则直接重放返回对应码；仅有技术失败归档且无 report.csv 时允许一次重试（第二次拒绝）；其余情况拒绝覆盖。
2. `_verify_frozen_inputs`：校验 config 与 plan 双密封哈希、pipeline_id/config_sha256 一致、source_artifacts 四哈希未变、H4-v1 仍 FAIL、两个划分 CSV 哈希与指纹未变、患者划分完整性断言。
3. 核验 H1/H4-v1 样本身份、频带集合与冻结患者集合完全一致（155 人全员、无排除）。
4. 逐外层折（5 折）：按 nested roles 重打 partition 标签 → fit_population_stats（患者剔除的 band×timestep 稳健统计）→ apply_hierarchical_calibration → fit_original_standardization → fit_comparators（ridge_alpha 取配置）→ 置信度阈值=该折 calibration 患者置信度 q0.25 分位 → apply_comparators → patient_errors；再用该折 H1 属性 freeze_subgroup_spec 冻结亚组规格并记录折级非劣效余量。
5. 抽取该折 outer_evaluation 角色患者为折外预测/误差/属性（合并 manifest_split、original_partition 元数据），断言外层患者未参与本折拟合或阈值选择。
6. 合并 5 折折外结果：断言无重复患者、覆盖全部 155 人、数量与配置一致；构造逐患者报告（effect_uncertainty_vs_h3/original、improved 标志）。
7. 判门：主比较 uncertainty_vs_h3 与 uncertainty_vs_original 需 ci95_low>0、sign_flip_p<0.05 且正向外层折数≥4；路由活跃患者均值≥0.5；三个鲁棒性亚组（原始效应≥0 且折余量调整后 ci95_low≥0、≥10 人）；困难患者 044/153 全在场且全员覆盖。
8. 产出逐折与原始分区（mechanism_train/calibration/validation，披露 31 人先前暴露）分层统计、全部 CSV/JSON；按 allowed_conclusions 给出中文结论，PASS 才置 recommended_for_subsequent_internal_use=true；main() 异常 fail-closed 返回 2。

### 模块依赖小结

- **机制核心 `src/mechanism_validation/h4_v2.py`**：全部 6 个脚本共享的证据管线原语——add_context_features → fit_population_stats → apply_hierarchical_calibration → fit_original_standardization → fit_comparators → apply_comparators → patient_errors，以及 fit_patient_attributes / freeze_subgroup_spec / assign_subgroups / effect_statistics / seal_probe / validate_probe / patient_set_sha256。开发脚本用它"拟合+密封"；外部确认脚本只用它的 apply_*（冻结参数重放）；内部 CV 脚本每折重新调用 fit_*（折内重拟合）。
- **划分核心 `src/mechanism_validation/internal_cv.py`**：仅服务内部探索线——balanced_partition_search / build_nested_roles（冻结脚本用其生成划分）、partition_fingerprint / assert_patient_partition_integrity / validate_sealed_mapping（评估脚本用其复验划分未被篡改）、seal_mapping（计划密封）。
- **密封配置 `configs/h4_v2_internal_exploratory_nested_cv_v1.json`**（config_sha256=891a30e…）：内部探索线的唯一真源——dataset_constraints 硬校验 155 患者/1191 样本/99-25-31 分区/困难患者 044、153；partition 定义 5×5 折与种子；mechanism 固定 ridge_alpha=1.0、收缩 k=60、置信度分位 0.25；gates 定义主比较/≥4 正向折/活跃覆盖≥0.5/亚组非劣效；allowed_conclusions 预注册中文结论措辞。冻结脚本执行其 partition/constraints 段并生成 frozen_plan.json；编排器保证冻结先于评估；评估脚本复验其哈希后按 gates 判定。
- **证据上游（只读）**：H3 decision+analysis_spec（H4-v2 的前置 PASS 条件与 bridge schedule/timestep 网格来源）、H4-v1 decision（必须保持 FAIL 的历史约束）与 sample_band_evidence.csv（频带证据行）、H1 sample_band_metrics.csv（患者属性源）、Stage 0A 数据集契约与病理剔除均值检查点（外部线双重谱系校验）。
- **外部线的编排依赖**：run_h4_v2_pipeline.py 串联 develop → run_mechanism_stage0_data_audit → run_cloud_stage0c_gate → validate_h4_v2_external_confirmation，禁止云端重冻结探针；内部线 run_h4_v2_internal_exploratory_pipeline.py 串联 freeze_plan → validate_internal_cv。
- **外部库**：numpy/pandas（数据与统计）、torch+scipy（仅外部确认线：加载检查点、环状膨胀）、argparse/subprocess/json（CLI 与编排）。


---

## 9. H4-v2 与 V2 冻结/生产管线脚本（下：V2 冻结与生产管线）

### 模块概述

本组脚本构成 SLMF-BBDM 仓库 **V2 版本冻结与生产校准导出流程**（H4-v2 之后的下游链）：两个 freeze 脚本分别锁定 V2 主模型集成契约（Stage V2-04）与五个下游阶段门协议（06A/06B/07/08/09）；audit_v2_freeze_integrity 在任何主模型改动前只读复核冻结工件（H4-v2 自洽 + H4-v1 仍 FAIL）；export 脚本把已锁机制在原始 99/25/31 分区上重拟合、密封为生产校准包；run_v2_main_pipeline 是云端一键编排器（freeze→V2-03 重训→V2-05A 审计，未建阶段诚实报 DEFERRED）；audit_v2_inference_admissibility 判定冻结证据能否在"仅 CT、无目标 PET/mask"的生产推理中复算，fail-closed 地阻断或放行路由激活。链条挂在 H4-v2 PASS 上而非 H4-v1 FAIL，任何脚本不得把未运行阶段改标 PASS。
### `scripts/freeze_v2_main_integration_contract.py` — V2 主模型集成契约冻结（Stage V2-04）

**职责**：在外层评估开始之前，锁定 V2 主模型集成的"集成契约"——四个对比器（comparators）、不确定性感知证据配方、无泄漏上下文获取规则与固定回退行为。所有冻结值均取自已锁定的 `src/mechanism_validation/h4_v2.py` 常量与已冻结的 H4-v2 嵌套 CV 计划，本脚本不发明新科学。产出带自哈希的 JSON 契约与 decision.json，状态为 FROZEN_BEFORE_EVALUATION，并武装（armed）一个 FAIL 条件：若无法构建无泄漏的上下文感知推理路径，集成以 FAIL 终止，禁止改名为降级机制。

**接口**：无参数 CLI 脚本，须在仓库根目录运行：
```bash
python scripts/freeze_v2_main_integration_contract.py
```
返回码 0 表示冻结成功；stdout 打印输出路径与 contract_sha256。

**依赖**：
- 内部：`src/mechanism_validation/common`（`canonical_json_sha256`、`load_json`）；`src/mechanism_validation/h4_v2`（BANDS、CONFIDENCE_QUANTILE、MIN_ACTIVE_COVERAGE、MIN_CONFIRMATION_PATIENTS、MIN_SUBGROUP_PATIENTS、MODEL_ORDER、NEIGHBOR_RADIUS、PATIENT_SHRINKAGE_K、RIDGE_ALPHA、SUBGROUP_MARGIN_FRACTION 等常量）；上一步 Stage V2-02 产出的 `00_frozen_plan/frozen_plan.json`。
- 外部：标准库 json/os/sys/datetime/pathlib。

**输入**：`results/mechanism_validation_v2/02_h4_v2_internal_exploratory_nested_cv/00_frozen_plan/frozen_plan.json`（读取其 pipeline_id、config_sha256、plan_sha256、partition 指纹与 H4_v1_outcome）。

**输出**：`results/mechanism_validation_v2/04_main_integration_freeze/resolved_integration_contract.json`（schema_version=2，含 contract_sha256 自哈希）与同目录 `decision.json`（decision=FROZEN_BEFORE_EVALUATION，记录 armed_fail_condition 与 current_context_path_status=NOT_YET_IMPLEMENTED）。

**处理过程**：
1. 将仓库根加入 sys.path，从 V2-02 目录读取已冻结的 H4-v2 nested-CV 计划。
2. 组装契约字典：四对比器定义（no_route / h3_fixed_schedule / original_evidence / uncertainty_aware，按 MODEL_ORDER 排序）。
3. 写入 v2_evidence_recipe：上下文来源（显式 patient_id+slice_id 索引的相邻切片与跨尺度同伴 band）、总体归一化（per (band,timestep) 的 median+1.4826*MAD）、患者-band 收缩（count/(count+K)）、证据置信度公式、Ridge 参数（alpha、无截距）与弃权规则（低于置信分位数阈值时逐元素回退到 H3 固定计划）。
4. 写入 frozen_grid（band 顺序、log-SNR 映射与裁剪、邻域半径、各类最小覆盖/患者数阈值、亚组非劣效边际比例）。
5. 写入 context_acquisition_rule：必须使用显式 patient_id/slice_id 与可复现的确定性上下文索引；禁止把目标 PET、recoverability 标签、病灶 mask、随机批邻接作为推理输入；附 fail-closed 条款（无法实现无泄漏路径即 FAIL，不得改名单切片代理仍称 H4-v2）。
6. 写入 fixed_fallback_behavior 与 h4_v2_reference（回指计划哈希/分区指纹）。
7. 以 canonical_json_sha256 计算契约自哈希并写回契约文件。
8. 生成并写出 decision.json，打印相对输出路径与哈希后返回 0。
### `scripts/freeze_v2_stage_protocols.py` — V2 各阶段门禁协议预注册冻结（Stage 06A/06B/07/08/09）

**职责**：在外层评估执行前预注册（pre-register）V2 主管线全部下游阶段门的协议：对比器、指标、阈值来源、停止规则与上游链。所有 decision 一律写为 `DEFERRED_NOT_RUN`——患者级评估需要云端 Win 缓存 + GPU，尚未在本工作副本上执行；任何阶段未经真实患者级运行不得改标 PASS。关键设计：V2 链挂在 H4-v2 的 PASS 上而非 H4-v1 的 FAIL，不复用旧的 validate_ct_support_head.py / validate_mechanism_curriculum.py 链。

**接口**：无参数 CLI 脚本，须在仓库根目录运行：
```bash
python scripts/freeze_v2_stage_protocols.py
```
返回码 0；逐阶段打印 decision 与 protocol_sha256 前 12 位。

**依赖**：
- 内部：`src/mechanism_validation/common`（`canonical_json_sha256`、`load_json`）；读取 Stage V2-02 的 `00_frozen_plan/frozen_plan.json` 与 `99_pipeline/decision.json`（H4-v2 PASS）。
- 外部：标准库 json/os/sys/datetime/pathlib。

**输入**：H4-v2 冻结计划（pipeline_id、plan_sha256、partition 指纹）与 H4-v2 pipeline decision（decision、H4_v1_preserved）。

**输出**：`results/mechanism_validation_v2/` 下五个阶段目录，各含 `decision.json`（含 protocol_sha256 自哈希）与 `execution_metadata.json`（run_executed=False，注明云端-only 原因）：
- `06A_ct_support_gate`（CT 支持门：仅允许有界空间支持/可靠性，禁预测 PET 幅值；不达标则保持 CT OFF，不得阻塞基础 v2）
- `06B_curriculum_gate`（课程门：冻结 intensity→coarse→shape→local-frequency 顺序，仅源自 H3；无内部支持则用固定训练计划）
- `07_artifact_safety`（伪影安全门：预注册 false_hotspot_density/stripe_or_directional_excess 等 5 项伪影指标，分数通过才实现 eSafe）
- `08_h5_v2_router_ablations`（角色消融：4 个基础对比器 + 8 项消融/负控制，逐外层折与患者级报告，含困难患者 044/153 与追加报告患者 002/022/080）
- `09_h6_v2_final_integration`（最终集成：5 对比器，必须满足小病灶主指标改善、伪影/全图误差非劣效、≥4/5 外层折方向一致、保留患者 044/153、禁止用外层结果重调）

**处理过程**：
1. 定义公共守则 COMMON_GUARDRAILS：患者为随机单元、禁止切片级随机划分、阈值只能取自对应外层折的内部机制训练/校准集、患者级 bootstrap 95% CI、mask 永不作为路由输入、验证指标不驱动训练课程、checkpoint 血缘必需、结论仅限探索性。
2. `_h4_v2_ref()` 读入 H4-v2 冻结计划与 pipeline decision，构造 upstream 引用（含 chains_through_h4_v1_FAIL=False）。
3. 依次构造五个阶段的协议 payload（stage_06a/06b/07/08/09 函数），各含对比器、阈值来源、停止规则、cloud_command（指向 `pixi run python scripts/run_v2_main_pipeline.py --stage <name>`）与 guardrails。
4. 对每个阶段调用 `_write()`：剥离旧哈希后以 canonical_json_sha256 计算 protocol_sha256 写入 decision.json，并写 execution_metadata.json 标记未运行。
5. 打印每个阶段的决策与哈希摘要，返回 0。
### `scripts/audit_v2_freeze_integrity.py` — V2 冻结完整性只读审计

**职责**：在任何 V2 主模型集成改动之前运行。不做任何重新冻结，只重推导已冻结 H4-v2 工件是否仍自洽、H4-v1 是否仍为 FAIL。任一检查失败即写 FAIL 并以非零码退出，使云端一键管线在触碰主模型前停止。审计是只读的：绝不改写冻结工件；若冻结文件已变化，正确动作是停下上报而非重冻结。脚本内硬编码"规格锁定"参考值（pipeline_id、config/plan/partition 哈希、155 名患者、99/25/31 分区、困难患者 044/153），盘上工件须同时与规格值和计划内记录值一致。

**接口**：CLI（argparse），在仓库根目录运行：
```bash
python scripts/audit_v2_freeze_integrity.py [--output <decision.json 路径>]
```
`--output` 默认 `results/mechanism_validation_v2/99_v2_model_pipeline/freeze_integrity/decision.json`。全部通过返回 0，否则返回 1（含必需冻结工件缺失时直接 return 1 并逐条打印 MISSING）。

**依赖**：
- 内部：`src/mechanism_validation/common`（`canonical_json_sha256`、`file_sha256`、`load_json`）；被审计的冻结工件（V2-02 冻结计划目录、V2-02 99_pipeline/decision.json、H4-v1 03_h4_noise_calibration/decision.json）。
- 外部：argparse/json/os/sys/datetime/pathlib、csv（校验外层患者分配表）、subprocess+git（工作树清洁检查）。

**输入**：`results/mechanism_validation_v2/02_h4_v2_internal_exploratory_nested_cv/00_frozen_plan/` 下的 frozen_plan.json、outer_patient_assignments.csv、nested_patient_roles.csv、outer_fold_balance.csv；`99_pipeline/decision.json`；H4-v1 的 `03_h4_noise_calibration/decision.json`。启动即检查这六个文件存在，缺失则列出清单（results/ 被 gitignore，需从冻结机器按字节原样拷贝）。

**输出**：审计 decision.json（schema_version=2，stage=00_v2_freeze_integrity_audit，含 8 项 checks 明细、spec_reference 与 audit_sha256 自哈希，refreeze_performed=False）；stdout 逐项打印 OK/XX 与决策。

**处理过程**（8 项检查）：
1. `_check_h4_v1_fail`：H4-v1 decision 仍为 FAIL。
2. `_check_plan_self_hash`：frozen_plan 的 plan_sha256 声明值==重算值==规格值。
3. `_check_config_sha`：计划内 config_sha256 与规格一致。
4. `_check_partition_fingerprint`：分区指纹哈希与规格一致。
5. `_check_frozen_files`：计划 frozen_files 表内每个分区文件存在且逐字节 file_sha256 一致。
6. `_check_patient_integrity`：155 名患者、0 排除、分区 99/25/31、困难患者 044/153 在位；并独立解析 outer_patient_assignments.csv（zfill(3) 规范化 patient_id）复核患者集合，不单信计划摘要。
7. `_check_h4_v2_pipeline_pass`：H4-v2 pipeline decision 为 PASS 且 H4_v1_preserved=FAIL 且 pipeline_id 匹配。
8. `_check_working_tree_clean_on_frozen`：`git status --porcelain` 确认冻结目录在 git 工作树中未被修改。
最后汇总 overall=all(passed)，写决策文件（含自哈希），按结果返回 0/1。
### `scripts/export_v2_production_calibration_bundle.py` — V2 生产校准包导出（Stage V2-04 使能件）

**职责**：把已锁定的 H4-v2 机制在**原始开发分区**（99 mechanism_train + 25 calibration，非嵌套 CV 折）上重拟合，封装成单一密封校准包，供下游代码：(a) 设定路由器置信度阈值；(b) 计算无泄漏的上下文感知置信度（总体统计 + 患者/band 收缩 + ridge）；(c) 提供未来 recoverability→route 映射所需的恢复度计划。不引入新科学——完全复用 `src/mechanism_validation/h4_v2.py` 中与冻结 H4-v2 嵌套 CV 相同的拟合函数；来源样本证据 CSV 经 SHA-256 对冻结计划核验。校准包是新的冻结工件（独立 pipeline_id + 自哈希），显式声明"不激活选择器、不通过任何门"。

**接口**：CLI（argparse，CPU-only，本地与云端均可运行），在仓库根目录：
```bash
python scripts/export_v2_production_calibration_bundle.py [--samples <csv>] [--plan <json>] [--output-dir <dir>]
```
默认 samples=`results/mechanism_validation/03_h4_noise_calibration/sample_band_evidence.csv`、plan=V2-02 冻结计划、output-dir=`results/mechanism_validation_v2/04_main_integration_freeze`。成功返回 0；来源 SHA 不匹配或分区计数异常抛 ValueError。

**依赖**：
- 内部：`src/mechanism_validation/common`（`canonical_json_sha256`、`file_sha256`、`load_json`）；`src/mechanism_validation/h4_v2`（BANDS/CONFIDENCE_QUANTILE/MODEL_ORDER/NEIGHBOR_RADIUS/PATIENT_SHRINKAGE_K/RIDGE_ALPHA 常量，及 add_context_features、apply_hierarchical_calibration、fit_comparators、fit_original_standardization、fit_population_stats 拟合函数）。
- 外部：argparse/json/os/sys/datetime/pathlib、numpy、pandas。

**输入**：H4-v1 的 sample_band_evidence.csv（原始 99/25/31 分区，patient_id 按字符串读取）与 H4-v2 冻结计划 frozen_plan.json（取 source_artifacts.h4_v1_sample_evidence_sha256 等核验字段）。

**输出**：`production_calibration_bundle.json`（schema_version=1，pipeline_id=V2_PRODUCTION_CALIBRATION_BUNDLE_V1，含 confidence_threshold、population_stats、original_standardization、按 MODEL_ORDER 排列的 models、fit_recipe、frozen_grid、source 溯源与 usage_contract、explicit_non_claim、bundle_sha256 自哈希）与 `calibration_bundle_decision.json`（decision=FROZEN，记录源核验结果与阈值）；stdout 打印包路径、哈希、阈值。

**处理过程**：
1. 解析参数并加载冻结计划；对样本 CSV 计算 file_sha256，与计划内冻结哈希比对，不一致即拒绝导出。
2. 读入 CSV，校验 band 集合与冻结 BANDS 一致、按 partition 统计患者数恰为 99/25/31。
3. 按与 H4-v2 完全相同的配方在原始分区上重拟合：add_context_features 构建上下文特征 → fit_population_stats（mechanism_train）→ apply_hierarchical_calibration 层级校准 → fit_original_standardization → fit_comparators（ridge alpha=RIDGE_ALPHA）。
4. 取 calibration 分区（25 人）的 evidence_confidence，按 CONFIDENCE_QUANTILE 分位数计算 confidence_threshold。
5. 计算原始分区患者→角色指纹（`_partition_fingerprint`），组装 bundle（fit_recipe 写明各步骤所用分区、source 溯源链、usage_contract 三条消费契约）。
6. canonical_json_sha256 计算 bundle_sha256 并落盘 bundle。
7. 写 calibration_bundle_decision.json，打印摘要，返回 0。
### `scripts/run_v2_main_pipeline.py` — V2 主管线云端编排器（入口）

**职责**：V2 主管线的云端 Windows 入口（免 pwsh），经 pixi 调用。诚实契约：有真实实现的阶段（freeze、excluded_mean、inference_audit）真正执行；仍缺 (a) 无泄漏上下文感知推理路径与 (b) 云端 GPU 逐对比器 runner 的患者级评估阶段，一律报告 DEFERRED_NOT_RUN 并注明确切缺失前置条件，绝不被本脚本改标 PASS。子脚本用同一解释器（sys.executable）拉起，因此无论 shell 是否已激活 pixi 环境命令均可用。

**接口**：CLI（argparse，必选 --stage）：
```bash
pixi run python scripts/run_v2_main_pipeline.py --stage {freeze|excluded_mean|inference_audit|ct_support|curriculum|artifact_safety|h5_v2|h6_v2|all} [--epochs 30] [--seed 42] [--guard-px 8] [--output-dir checkpoints/freq_mean_excluded_v1]
```
--stage all 依次执行 freeze→excluded_mean→inference_audit，随后对五个 DEFERRED 阶段逐个 report_deferred。任何阶段非零退出即短路返回该码。

**依赖**：
- 内部（被编排）：`scripts/audit_v2_freeze_integrity.py`（freeze 阶段）、`scripts/pretrain_conditional_mean.py`（excluded_mean 阶段，配置 configs/experiments/slmf_png_residual_frequency.yaml + configs/dataset_contract_stage0a_v1.json）、`scripts/audit_v2_inference_admissibility.py`（inference_audit 阶段）；引用 `results/mechanism_validation_v2/<stage_dir>/decision.json` 协议文件。
- 外部：argparse/subprocess/sys/pathlib。

**输入**：命令行参数（epochs/seed/guard-px/output-dir）；间接依赖冻结工件（经 audit_v2_freeze_integrity 校验）与训练数据缓存 lineage（cache/tensors_main/cache_lineage.json）。

**输出**：无直接文件输出——产出由被编排子脚本生成：审计 decision.json（freeze/inference_audit）、`<output-dir>/mean_best.pt` 与指纹旁车 `mean_best.pt.fingerprint.json`（excluded_mean；生产 v5 checkpoint 路径有意不变更）；DEFERRED 阶段仅打印协议路径与缺失前置条件。链路状态打印（如 "V2-03 PASS; V2-05A FAIL… fail-closed"）。

**处理过程**：
1. `_run_python` 辅助函数：以 sys.executable + 仓库根 cwd 调子脚本并回显命令行。
2. run_freeze：先跑 audit_v2_freeze_integrity.py；失败打印"主模型改动前停止"。
3. run_excluded_mean（Stage V2-03）：先过 freeze 门，再用 --override 强制 fail-closed 血缘重训病理排除均值（experiment.name=freq_mean_excluded_v1、pathology_exclusion.enabled=true、guard_radius_px、require_cache_lineage=true、use_fake_data=false），训练后检查指纹旁车存在并提示切换前先审计。
4. run_inference_audit（Stage V2-05A）：先过 freeze 门，再跑 audit_v2_inference_admissibility.py；失败则不确定性感知路由激活、H5-v2 及依赖 H4-v2 的 H6 全部受阻。
5. report_deferred：对 ct_support/curriculum/artifact_safety/h5_v2/h6_v2 打印 DEFERRED_NOT_RUN、协议 decision.json 路径与缺失前置（上下文路径 + 云端 runner），明确"不标 PASS、不要改标"。
6. main 按 --stage 分派；单独请求 DEFERRED 阶段也先过 freeze 门再报告。
### `scripts/audit_v2_inference_admissibility.py` — V2 冻结证据的生产推理可接纳性 fail-closed 审计（Stage V2-05A）

**职责**：把两个问题刻意分开：(1) 内部探索性 H4-v2 分析是否用了密封源工件（可以仍为真）；(2) 同一证据能否在 CT-only PET 合成中、无目标 PET 图像与病灶 mask 的条件下计算。本审计失败绝不改写或改标 H4-v1/H4-v2 结果，只阻断其"生产路由用途"——FAIL 时不确定性感知路由激活、H5-v2 及 H4-v2 依赖的 H6 被列入 blocked_stages。判定核心是对 H4-v1 证据生成器 `_sample_band_rows` 做 AST 静态分析：若其读取 batch["pet"]/batch["mask"]、residual 由目标 PET 构造、且调用 mask 能量/对齐函数，而契约又禁止这些输入，则冲突坐实 → 冻结证据生产不可接纳。

**接口**：CLI（argparse），在仓库根目录运行：
```bash
python scripts/audit_v2_inference_admissibility.py [--bundle <json>] [--contract <json>] [--generator <py>] [--output-dir <dir>] [--allow-scientific-fail-exit-zero]
```
默认 bundle=`04_main_integration_freeze/production_calibration_bundle.json`、contract=`resolved_integration_contract.json`、generator=`scripts/validate_h4_noise_band_calibration.py`、输出目录 `05A_inference_admissibility_audit`。PASS 返回 0；科学性 FAIL 返回 2（除非 --allow-scientific-fail-exit-zero 仅供报告聚合时返回 0）；密封工件缺失时写 MISSING_FROZEN_ARTIFACT_FAIL_CLOSED 的 FAIL 决策（不改写、需从开发机原样拷贝后重跑）。

**依赖**：
- 内部：`src/mechanism_validation/common`（`canonical_json_sha256`、`file_sha256`）；上游产物 production_calibration_bundle.json、resolved_integration_contract.json、sample_band_evidence.csv；被静态审查的 `scripts/validate_h4_noise_band_calibration.py`。
- 外部：argparse/ast/json/sys/datetime/pathlib/typing（纯标准库，AST 解析无第三方依赖）。

**输入**：V2-04 导出的校准包与集成契约（均验自哈希 bundle_sha256/contract_sha256）、H4-v1 样本证据 CSV（验 SHA 与包内记录一致）、H4-v1 生成器源码。

**输出**：`results/mechanism_validation_v2/05A_inference_admissibility_audit/decision.json`（schema_version=2，含 lineage_checks、generator_semantics（batch_reads/reads_target_pet/reads_lesion_mask/residual_uses_target_pet 等）、contract_conflict、conflict_proven、production_router_activation_allowed/next_stage_allowed、blocked_stages、unblocked_independent_work（H3 固定计划、独立 H1 契约下 CT 支持门、不主张 H4-v2 激活的伪影安全工具）、stop_rule 与 decision_sha256 自哈希）；stdout 完整打印决策 JSON。

**处理过程**：
1. 解析参数并检查四个必需工件存在；缺失走 build_missing_artifact_decision 分支（FAIL、scientific_gate_evaluated=False、给出"从开发仓库原样拷贝、勿在云端重新生成"的修复指引）。
2. `_self_hash_matches` 校验 bundle 与契约的自哈希完整；file_sha256 校验源 CSV 与包内记录一致；确认包内 h4_v1_outcome 仍为 FAIL（lineage_checks 五项）。
3. `inspect_h4_generator` 用 ast.parse 定位 `_sample_band_rows`：收集 `batch[...]` 下标读取键名、调用函数名、以及 "residual 赋值右侧是否用到 pet"，输出 reads_target_pet / reads_lesion_mask / residual_uses_target_pet / uses_masked_band_energy 等语义标记。
4. 从契约 context_acquisition_rule.forbidden_as_inference_input 提取禁用项，确认契约禁止 target PET 与 mask。
5. conflict_proven = 生成器读 PET 且 residual 用 PET 且读 mask 且用 masked_band 函数且契约两禁均在；inference_admissible = lineage 全过 且 冲突未坐实。
6. 组装决策 payload（含 blocked_stages 三项、stop_rule"不得实现单切片 CT 代理冒称 H4-v2；新观测证据=新机制，需新 pipeline_id + 内部开发 + 冻结门"），计算 decision_sha256 后写出。
7. 按 PASS/FAIL 与 --allow-scientific-fail-exit-zero 决定退出码（0/2）。

### 模块依赖小结

- **对 `src/mechanism_validation` 的依赖**：六个脚本全部依赖 `common.py` 的 `canonical_json_sha256`/`file_sha256`/`load_json` 做规范化哈希与只读加载；`freeze_v2_main_integration_contract.py` 与 `export_v2_production_calibration_bundle.py` 还分别从 `h4_v2.py` 取冻结常量与完整拟合函数（add_context_features/fit_population_stats/apply_hierarchical_calibration/fit_original_standardization/fit_comparators），保证冻结契约与校准包"零新科学"。
- **冻结工件链（上游→下游）**：Stage V2-02 产物（`00_frozen_plan/frozen_plan.json`、outer_patient_assignments.csv 等 + `99_pipeline/decision.json` H4-v2 PASS）是所有脚本的共同输入 → `freeze_v2_main_integration_contract.py` 与 `export_v2_production_calibration_bundle.py` 产出 V2-04 目录下的 `resolved_integration_contract.json` + decision.json 与 `production_calibration_bundle.json` + `calibration_bundle_decision.json`（校准包的源 CSV 经 SHA 对 V2-02 计划核验）→ `audit_v2_inference_admissibility.py` 消费契约+校准包+源 CSV+H4-v1 生成器源码，产出 `05A_inference_admissibility_audit/decision.json`；H4-v1 的 `03_h4_noise_calibration/decision.json`（FAIL）与 sample_band_evidence.csv 贯穿被引用。
- **编排与调用关系**：`run_v2_main_pipeline.py` 是唯一入口，按 freeze（=`audit_v2_freeze_integrity.py`）→ excluded_mean（=`pretrain_conditional_mean.py`，fail-closed 血缘重训）→ inference_audit（=`audit_v2_inference_admissibility.py`）串联，全部用 sys.executable 拉起；`freeze_v2_stage_protocols.py` 写出的五个协议 decision.json 中的 cloud_command 反向指向本编排器的对应 --stage，形成"协议预注册 ↔ 执行入口"的闭环；请求任一 DEFERRED 阶段也先过 freeze 门。
- **外部依赖极轻**：全部脚本仅标准库；唯一例外是 `export_v2_production_calibration_bundle.py` 需要 numpy+pandas（读 CSV 与分位数计算），`audit_v2_freeze_integrity.py` 需要 git 子进程做工作树清洁检查——整套流程除重训阶段外 CPU-only、可在本地与云端一致运行。


---

## 10. H3 固定调度与先验锚定路由实验脚本

### 模块概述

本模块是 SLMF-BBDM 机制验证流水线中 H3 假设（Haar 频带"可恢复性→路由先验"）的两代实验脚本集合：V1 固定调度（05B，审计其十点网格/标量映射两个 fail-closed 失败并无人值守复核）与 V2 全时域 native/null（05C，标定 `[native,shallow,null]=[a,0,1-a]` 后做匹配双臂模型实验），以及后续"先验锚定 100-epoch 路由"云端实验（运行器/汇总器/只读审计器）。设计上全部 fail-closed：每个脚本都用 SHA-256 钉死输入工件、以分区指纹锁定数据角色、把科学 FAIL 与基础设施失败区分开，且任何产物都不授权生产训练/激活或 H5/H6。数据流为：`calibrate`→(decision+全时域调度)→`build_configs`→(双臂 YAML+experiment_plan)→训练/评估→`analyze`；PNG 预览先验(`estimate_h3_prior_from_png`)→`run_prior_anchored_router_100e`(云端重估+训练)→`summarize`/`audit`。调度最终由 `src/model/frequency/spectral_router.py` 按 policy 分派 `load_h3_native_null_schedule`/`load_prior_anchor_schedule` 加载，路由正则项由 `src/model/loss_terms/spectral_router.py` 消费。

### `scripts/run_h3_fixed_schedule_overnight.py`

**职责**：H3 固定调度推理审计（V2-05B）的可靠无人值守运行器。编排"冻结前审计 → H3 门 → 定向测试 → 全量测试 → 源卫生检查 → 冻结后审计"六阶段流水线；刻意不含生产/模型训练阶段，永不启动 H5/H6。H3 门的科学 FAIL 是预期且成功审计的结果，而基础设施/完整性/测试失败仍以非零退出码停止运行。

**接口**：
- `class OvernightRunner(args, run_dir, keep_awake)`：核心编排器；`run() -> int` 顺序执行各阶段，`run_command_stage(stage)` 流式运行子进程，`validate_h3()/validate_freeze()/validate_final_integrity()` 校验决策语义与哈希不变，`_resume_record()` 支持按阶段记录断点续跑。
- `class ExclusiveRunLock(path, run_dir)`：跨进程独占锁，`acquire(resume=... )/`release()`，含同 run 陈旧锁恢复。
- `class KeepWindowsAwake`：上下文管理器，Windows 下阻止睡眠；`class RunLogger(path, append)`：原子追加运行日志。
- `build_plan(args, run_dir) -> dict`：构造六阶段计划（00_environment/01_freeze_pre/02_h3_gate[期望 FAIL 且 next_stage_allowed=False]/03_targeted_tests/04_full_tests/05_git_diff_check/06_freeze_post）。
- `parse_args(argv)` / `main(argv) -> int`。CLI 用法：`python scripts/run_h3_fixed_schedule_overnight.py [--run-dir DIR] [--resume] [--skip-full-tests] [--stage-timeout-seconds N] [--heartbeat-seconds N] [--min-free-disk-gb N] [--pytest-temp-root DIR] [--dry-run]`；推荐入口为 `scripts/run_h3_fixed_schedule_overnight.ps1`。

**依赖**：内部：`scripts/audit_h3_fixed_schedule_inference.py`（H3 门）、`scripts/audit_v2_freeze_integrity.py`（冻结前后审计）、`configs/h3_fixed_schedule_inference_v1.json`、`results/mechanism_validation_v2/04_main_integration_freeze/production_calibration_bundle.json`、`resolved_integration_contract.json`、`results/mechanism_validation_v2/05A_inference_admissibility_audit/decision.json`、`tests/test_h3_fixed_schedule_inference.py`、`tests/test_v2_inference_admissibility.py`；外部：argparse/hashlib/json/subprocess/threading/ctypes（Windows 保活）/tempfile。

**输入**：CLI 参数；仓库内受保护基线文件（05A 决策、H3 配置、标定包、集成契约、H4 样本频带证据 CSV）；pytest 测试套件。

**输出**：`results/mechanism_validation_v2/05B_h3_fixed_schedule_inference/overnight_runs/<run-id>/` 下的 `state.json`、`run.log`、`heartbeat.json`、`summary.json`，以及 `artifacts/{freeze_pre,freeze_post,h3_gate}/decision.json` 与各阶段 record JSON；进程退出码。

**处理过程**：
1. `parse_args` 校验参数（`--resume` 必须配 `--run-dir`，与 `--dry-run` 互斥）。
2. `resolve_run_dir` 定位/新建运行目录，`ExclusiveRunLock.acquire` 获取独占锁（检测并恢复同 run 的陈旧锁）。
3. 进入 `KeepWindowsAwake` 上下文，创建 `OvernightRunner`，启动心跳线程周期性原子更新 `heartbeat.json`。
4. `prepare_context` 记录环境清单与全部输入文件 SHA-256（`_input_hashes`、`_protected_baseline_hashes`）。
5. `build_plan` 生成阶段计划后逐阶段执行：内部清点 → 子进程命令（流式读取 stdout/stderr、超时控制、进程树终止）→ 写阶段 record。
6. H3 门按 `expected_decision="FAIL"` 校验语义；冻结前/后与受保护基线文件哈希必须逐字节一致，否则 fail-closed。
7. 全部阶段完成后 `validate_final_integrity` 复核并写 `summary.json`；任何异常走 `fail()` 记录 traceback 并以非零码退出。
8. `finally` 中停止心跳、释放锁，返回退出码（KeyboardInterrupt → 130）。

### `scripts/audit_h3_fixed_schedule_inference.py`

**职责**：V2-05B 阶段的 fail-closed 审计脚本。证明冻结 H3 表的两个独立科学失败：(1) 它仅支持在十个探针时刻点上按 band/timestep/常数精确查找，而生产训练/采样使用 0..999 全时域且未冻结任何网格外解析器；(2) 表中只含一个标量而频谱路由器需要 `[native, shallow, null]` 三元组。审计在模型评估之前即停止。

**接口**：
- `build_decision(*, config_path=DEFAULT_CONFIG, bundle_path=None, contract_path=None, model_source_path=None, router_source_path=None, production_config_path=None) -> dict`：产出完整决策文档（各 gate + failure_phase）。
- 检查器族：`inspect_config(config, load_error)`（config SHA-256 对 SPEC 钉死值）、`inspect_frozen_artifact(path, payload, load_error, expected_file_sha256, self_hash_field, expected_self_sha256)`、`inspect_runtime_file(path, expected_file_sha256, parse_error)`、`inspect_production_code_state(model_source, model_tree, router_source, router_tree, production_config)`（AST 级源码状态检查）、`inspect_h3_schedule(bundle, config)`（冻结网格查找）、`inspect_production_timestep_consumption(...)`（证明全时域消费未定义）、`inspect_mapping_identifiability(bundle, contract, config)`（证明三元映射不可辨识）。
- `main(argv) -> int`。CLI：`python scripts/audit_h3_fixed_schedule_inference.py [--config] [--bundle] [--contract] [--model-source] [--router-source] [--production-config] [--output-dir] [--allow-scientific-fail-exit-zero]`；仅当"完整评估的科学 FAIL"（完整性 PASS + 网格 PASS + 两个不可定义证明 + failure_phase=INDEPENDENT_TIMESTEP_AND_MAPPING_GATES_FAIL_CLOSED）且带 `--allow-scientific-fail-exit-zero` 时返回 0，否则 2。

**依赖**：内部：`src/mechanism_validation/common`（canonical_json_sha256/file_sha256）、`src/model/slmf_bbdm.py` 与 `src/model/frequency/spectral_router.py`（仅做 SHA-256 钉死校验与 AST 检查——即与 `loss_terms/spectral_router.py` 路由实现的联动点）、`configs/h3_fixed_schedule_inference_v1.json`、`configs/experiments/slmf_png_spectral_router_v5.yaml`、`results/.../production_calibration_bundle.json`、`resolved_integration_contract.json`；外部：argparse/ast/json/math/os/sys/datetime。

**输入**：冻结 H3 配置、生产标定包、已解析集成契约、模型源码（slmf_bbdm.py）、频谱路由器源码（frequency/spectral_router.py）、生产 v5 实验配置。

**输出**：`results/mechanism_validation_v2/05B_h3_fixed_schedule_inference/decision.json`（原子写入）+ stdout 完整决策 JSON；退出码。

**处理过程**：
1. 用 `_safe_load/_safe_source/_safe_yaml` 容错加载六类输入（错误记入 gate 而非抛异常）。
2. `inspect_config` 与 `inspect_frozen_artifact` 将 config/bundle/contract 的 file/self SHA-256 与模块常量 SPEC_* 比对。
3. `inspect_runtime_file` + `inspect_production_code_state` 对 `src/model/slmf_bbdm.py`、`src/model/frequency/spectral_router.py`、v5 生产配置做哈希与 AST 双重校验（确认生产代码当前未定义 off-grid 消费）。
4. 三向核对 H3 源分区与 H4-v2 嵌套分区指纹（config ↔ bundle ↔ contract ↔ SPEC）。
5. `inspect_h3_schedule` 验证冻结表仅在 SPEC_TIMESTEPS 十点上可精确查找。
6. `inspect_production_timestep_consumption` 证明生产 0..999 全时域消费无定义；`inspect_mapping_identifiability` 证明标量→三元路由映射不可辨识。
7. 汇总为 gates（frozen_input_integrity / frozen_grid_lookup / production_timestep_consumption / mapping_identifiability），核心完整性失败时后续 gate 标记 NOT_EVALUATED。
8. 原子写出 decision.json 并按退出码规则返回。

### `scripts/calibrate_h3_v2_full_timestep_native_null.py`

**职责**：离线机制开发 worker，标定独立的 H3-v2 全时域 native/null 调度。只允许读 99 例 mechanism-train 与 25 例 calibration 角色的 PET/病灶掩膜（绝不读暴露的 validation 角色）。运行时工件把"高于偶然的可恢复性自由度"映射为 `[native, shallow, null] = [a, 0, 1-a]` 并直接定义 0..999 每个时刻；它不恢复 H4-v2、不授权生产/H5/H6。

**接口**：
- `run(args) -> dict`：主流程，产出完整 decision；`_existing_complete_decision(output_dir, config_sha256)` 幂等短路（同 config 已完整则直接返回）。
- `_validate_protocol_config(root, config_path) -> (config, source_hashes)`；`_validate_mean_checkpoint(root, config, lineage)`；`_mechanism_datasets(cache_dir, manifest_path, partition)`。
- `_prepare_inputs(...)`（`@torch.no_grad()`，预计算并缓存模型输入，identity 钉死 config/checkpoint/seed/noise）；`_compute_shard(...)`（`@torch.no_grad()`，按 timestep 分片计算可恢复性并缓存 .npz）；`_fit_and_gate(...)`（保序回归拟合 + 门控 + bootstrap 对比）；`_bootstrap_contrast(...)`。
- `_parse_args(argv)` / `main(argv) -> int`。CLI：`python scripts/calibrate_h3_v2_full_timestep_native_null.py --output-dir DIR [--root ROOT] [--config configs/h3_v2_full_timestep_native_null_v1.json] [--device auto] [--require-cuda] [--batch-size N] [--timestep-chunk-size N] [--num-workers N] [--shard-size 25] [--allow-scientific-fail-exit-zero]`；PASS 返回 0，科学 FAIL 带 flag 返回 0 否则 3，异常写 technical_failure.json 返回 2。

**依赖**：内部：`scripts/validate_h2_pathology_excluded_residual`（IndexedDataset、_load_predictor、_loader）、`scripts/validate_h4_noise_band_calibration`（BANDS、_band_tensors、_masked_band_alignment）、`src/data/dataset.CachedDataset`、`src/data/lineage`（密封缓存 lineage 校验）、`src/mechanism_validation/common`（bootstrap_mean/partition/read_manifest/write_json 等）、`src/model/noise/base.BBDMBridgeSchedule`；外部：numpy、torch、torch.nn.functional、sklearn.isotonic.IsotonicRegression、argparse/hashlib/json。

**输入**：协议配置 `configs/h3_v2_full_timestep_native_null_v1.json`（dataset_contract/calibration 冻结参数）、密封缓存 lineage、权威 manifest、均值检查点、mechanism_train+calibration 角色的 PET 与病灶掩膜缓存。

**输出**：`--output-dir` 下 `decision.json`（或 technical_failure.json）、`patient_recoverability.npz`、按时刻分片的 .npz 缓存、`full_timestep_schedule_summary.csv`、`calibration_patient_errors.csv`；PASS 时附全时域调度工件（`native_active_mass`、`mapping="[active_mass,0,1-active_mass]"`、`lookup_policy=exact_integer_timestep_only`、`inference_schedule_allowed=true`、`production_activation_allowed=false`、`h5_h6_allowed=false`）。

**处理过程**：
1. `_validate_protocol_config` 校验配置与源文件哈希；`_existing_complete_decision` 命中已完成决策则幂等返回。
2. 加载密封缓存 lineage 并逐字段比对；校验 manifest 文件 SHA-256、患者分区指纹与角色计数（99/188 样本、25/…、237 validation），并断言暴露验证患者（044/080/153）从未被读取。
3. `_validate_mean_checkpoint` 验证均值检查点；batch_size 与 timestep_chunk_size 由协议冻结不可覆盖，`--shard-size` 必须整除 1000；`--require-cuda` 时强制 CUDA。
4. `_mechanism_datasets` 构建数据集并核对两角色患者数，`_load_predictor` 加载预测器。
5. `_prepare_inputs` 在 no_grad 下批量预计算输入并缓存（identity 含 config/checkpoint 哈希、analysis_seed、noise_identity）。
6. 以 `BBDMBridgeSchedule`（num_train_timesteps=1000）为桥，按 `shard_size` 分片调用 `_compute_shard` 计算每患者×频带×时刻的可恢复性并落盘缓存。
7. 拼接为 `patient_recoverability.npz`（患者×BANDS×1000），`_fit_and_gate` 用保序回归拟合每频带 native 活性质量 a，bootstrap 对比 calibration 角色与 band 常数误差并给出 PASS/FAIL 门控。
8. 写出 summary/误差 CSV；PASS 时生成全时域调度工件与 decision（含 self SHA-256），失败时写 technical_failure.json。

### `scripts/run_h3_v2_overnight.py`

**职责**：隔离的 H3-v2 标定 + 匹配模型实验一夜运行器。fail-closed 设计：强制真实 CUDA（拒绝 CPU 调试档）；密封并复核全部注册运行时源与历史证据；标定只读 mechanism-train/calibration 患者；标定 FAIL 是"已完成的科学停止"而非基础设施失败；只有标定 PASS 后才启动模型训练，且所有可变输出只写在本 run 目录之下；任何结果都不授权生产训练/激活、H5、H6。中断后 `--resume` 可续跑已验证的标定分片；训练开始后的中断则新建独立尝试（attempt），不覆盖部分尝试、不伪造优化器/RNG 连续性。

**接口**：
- `class OvernightRunner(args, run_dir, keep_awake)`：核心编排器；`run() -> int`、`run_command_stage/run_internal_stage`、`_validate_calibration_output`、`_model_smoke`（GPU 冒烟）、`_validate_training_stage/_validate_evaluation_stage/_validate_analysis_output`、`_seal_evaluation_provenance`、`_final_integrity`、`_historical_failure_checks`、`_prepare_resume_experiment_attempt`。
- `class ExclusiveFileLock(path, run_dir, purpose)`：流水线锁与开发 GPU 锁双锁；`class KeepWindowsAwake`、`class RunLogger`、`resolve_run_dir`、`validate_protocol`。
- `build_plan(args, run_dir) -> dict`、`parse_args(argv)`、`main(argv) -> int`。CLI：`python scripts/run_h3_v2_overnight.py [--run-dir] [--resume] [--dry-run] [--skip-full-tests] [--epochs 50] [--device-index 0] [--shard-size 25] [--heartbeat-seconds 30] [--min-free-disk-gb 15] [--min-free-gpu-gb 8] [--test/calibration/training/evaluation-timeout-seconds]`；推荐入口 `scripts/run_h3_v2_overnight.ps1`。

**依赖**：内部：`scripts/calibrate_h3_v2_full_timestep_native_null.py`（阶段 04）、`scripts/build_h3_v2_experiment_configs.py`（阶段 05）、`scripts/train_v2.py`（阶段 07/08）、`scripts/evaluate.py`（阶段 09/10）、`scripts/analyze_h3_v2_experiment.py`（阶段 11）、`scripts/audit_v2_freeze_integrity.py`、`configs/h3_v2_full_timestep_native_null_v1.json`、`tests/test_h3_v2_*.py`；外部：argparse/hashlib/subprocess/threading/ctypes/uuid。

**输入**：协议配置；GPU/磁盘资源预检所需环境；历史证据（05B 决策等受保护根）；CLI 参数。

**输出**：`results/mechanism_validation_v2/05C_h3_v2_full_timestep_native_null/overnight_runs/<run-id>/` 下 state.json、run.log、heartbeat.json、summary.json、`artifacts/calibration/`（标定决策）、`artifacts/model_experiment/attempts/attempt_NN/`（两个变体的配置、训练检查点、评估输出与分析结果）。

**处理过程**：
1. `parse_args` 严格校验（仅支持 device-index 0、shard-size 整除 1000、resume/dry-run 互斥等）。
2. `resolve_run_dir` 建目录；先取流水线锁再取开发 GPU 锁（`ExclusiveFileLock`，含死属主恢复），进入 `KeepWindowsAwake`。
3. 阶段 00 CUDA/磁盘预检（`_gpu_preflight`、`_resource_preflight_payload`，GPU 显存 ≥ 阈值、磁盘 ≥ 阈值）。
4. 阶段 01 冻结完整性前置审计 + 阶段 02/03 定向与全量测试（可 `--skip-full-tests`）。
5. 阶段 04 调用标定脚本（分片可续跑）；`_validate_calibration_output` 校验决策——FAIL 则直接跳到阶段 12 以"成功科学停止"收尾。
6. 标定 PASS 后：阶段 05 构建实验配置 → 阶段 06 GPU 模型冒烟（`_model_smoke`）→ 阶段 07/08 分别训练 no_route 与 h3_v2_native_null 两变体（`--epochs` 控制，检查点写在本 run 目录）。
7. 阶段 09/10 分别评估两变体并 `_seal_evaluation_provenance` 密封评估来源；阶段 11 调用分析脚本产出配对统计。
8. 阶段 12 `_final_integrity` 复核受保护根未被写入；写 summary.json，释放双锁并返回退出码。

### `scripts/build_h3_v2_experiment_configs.py`

**职责**：构建隔离、匹配的 H3-v2 模型实验配置。刻意"非执行"：只验证预注册协议与已完成的标定决策，导出锁定的 99/25/31 manifest，写出两个 run 拥有的 YAML 变体配置（no_route 与 h3_v2_native_null）加一份实验计划（experiment_plan.json）；不启动训练、不评估模型、不写入遗留 checkpoint/证据目录。

**接口**：
- `build(args) -> Path`：主流程，返回 experiment_plan.json 路径。关键校验器：`_validate_protocol(root, protocol_path)`、`_validate_calibration_decision(path, protocol, runtime_source_hashes)`、`_validate_schedule(calibration_decision_path, decision, protocol)`、`_validate_source_data(root, protocol)`、`_validate_mean_checkpoint`、`_load_base_config`（基配置哈希钉死）、`_prepare_variant_config`（派生单变体 YAML）、`_materialize_manifest`（物化 99/25/31 manifest）、`_commit_new_or_identical`（只允许新建或字节一致复用）。
- `_parse_args(argv)` / `main(argv) -> int`（失败返回 14）。CLI：`python scripts/build_h3_v2_experiment_configs.py --calibration-decision PATH --output-dir PATH --run-id ID [--root] [--protocol-config configs/h3_v2_full_timestep_native_null_v1.json] [--epochs N] [--attempt attempt_01]`。

**依赖**：内部：`src/model/frequency/h3_native_null_schedule`（`load_h3_native_null_schedule`、`H3_V2_PIPELINE_ID`——**与调度加载器的直接联动点**：本脚本用它在`_validate_schedule`中加载并核对标定产出的全时域调度工件）、`src/data/lineage`、`src/mechananism_validation/common`（实际为 `src/mechanism_validation/common`：manifest/分区/哈希工具）、`configs/experiments/slmf_png_spectral_router_v5.yaml`（基配置，SHA-256 钉死）；外部：yaml、argparse/copy/json/re/tempfile。

**输入**：预注册协议配置、标定 decision.json（须 PASS 且 pipeline_id 匹配）、权威 manifest 与密封缓存 lineage、均值检查点、v5 基配置。

**输出**：`<output-dir>/attempts/<attempt>/` 下 `experiment_plan.json`（含 plan_sha256 自哈希）、`configs/{no_route,h3_v2_native_null}.yaml`、派生 manifest；stdout 打印计划路径与哈希。

**处理过程**：
1. 校验 run_id/attempt 为安全标识符，output_dir 必须在仓库内且 run 拥有（`_require_run_owned_output`）。
2. `_validate_protocol` 校验协议配置与注册运行时源哈希；`_validate_calibration_decision` 要求标定决策 PASS 且哈希链一致。
3. `_validate_schedule` 经 `load_h3_native_null_schedule` 加载调度工件并核对 schedule_sha256/num_train_timesteps/route_order/band_order 与 mapping。
4. `_validate_source_data` 复算患者分区指纹与计数（mechanism_train 99 / calibration 25 / validation 31），与标定决策交叉一致。
5. `_materialize_manifest` 物化锁定 manifest 并回验其分区哈希与计数。
6. 对每个变体 `_prepare_variant_config`：以 v5 基配置为底，注入同种子（same_initialization_seed）、同 epoch 预算、EMA 评估、test 全集评估等字段；h3_v2_native_null 额外注入 `modules.residual_frequency.cross_level_router.h3_schedule_path/h3_schedule_sha256` 与 `policy=h3_native_null`，no_route 则 `hard_all_null=true`（这是写入 spectral_router 消费字段的联动点）。
7. 匹配性 fail-closed 审计：抹平身份/输出/路由干预字段后两臂 deep-equal，否则报错。
8. 组装 experiment_plan.json（预算、stop_and_claim_rules、source_identity），写盘并复验自哈希。

### `scripts/analyze_h3_v2_experiment.py`

**职责**：分析隔离的 H3-v2 "no_route vs native/null" 匹配模型实验。分析单位是患者。先验证预注册协议、标定 PASS、冻结调度、实验计划、派生机制 manifest、两个最终检查点与两份评估工件，然后才计算任何对比；暴露的 31 例 validation 队列只使用一次且必须两臂全覆盖。SUPPORT/NO_SUPPORT 均为内部探索性决策，不授权生产/H5/H6，也不改变历史 05B 固定调度 FAIL。

**接口**：
- `run(args) -> (decision_dict, decision_name)`：主流程。校验器族：`_validate_config`、`_validate_calibration`、`_validate_plan`、`_validate_schedule`、`_validate_manifests`、`_validate_variants`（含两臂检查点/配置/日志哈希）、`_validate_evaluation_provenance`、`_validate_eval`（患者集完整性）。统计族：`_analyze_metrics`（主门 + 安全门）、`_bootstrap_indices/_ci95`、`_difference_statistic/_ratio_statistic`、`_paired_values`。
- `_parse_args(argv)` / `main(argv) -> int`。CLI：`python scripts/analyze_h3_v2_experiment.py --experiment-plan PATH --output-dir DIR [--root] [--protocol-config configs/h3_v2_full_timestep_native_null_v1.json] [--calibration-decision PATH] [--no-route-eval PATH] [--h3-v2-eval PATH]`（可选覆盖参数必须与密封计划一致）。

**依赖**：内部：`src/mechanism_validation/common`（manifest/分区/哈希）、实验计划与调度工件（由 build_h3_v2_experiment_configs 与 calibrate 脚本产出）、两臂评估 validation.json 与 validation_provenance.json；外部：numpy、yaml、argparse/csv/hashlib/json/math。

**输入**：协议配置、experiment_plan.json、标定 decision.json、冻结调度 JSON、派生 manifest、两变体最终 EMA 检查点与评估工件。

**输出**：`--output-dir` 下 `decision.json`（SUPPORT/NO_SUPPORT + primary_gate/safety_gates/difficult_patients/input_artifacts 全链哈希）与 `patient_differences.csv`（按患者配对差异）。

**处理过程**：
1. `_validate_config` 校验协议与运行时源哈希；定位实验计划并由其密封引用解析标定决策路径（CLI 覆盖必须与计划一致）。
2. `_validate_calibration` + `_validate_plan` 验证标定 PASS、计划 plan_sha256 及其引用文件哈希。
3. `_validate_schedule` 校验冻结调度 file/self 哈希与 num_train_timesteps、mapping。
4. `_validate_manifests` 复算患者分区，锁定暴露 validation 31 例患者与样本数；`_validate_variants` 验证两臂配置/检查点（epoch 精确匹配）/评估工件齐全。
5. `_validate_eval` 要求两臂评估患者集合与锁定集合完全一致且互相相等。
6. `_analyze_metrics` 以患者为单位做配对 bootstrap（95% CI）计算主门指标与安全门指标的差异/比值统计。
7. 主门 + 全部安全门通过 → SUPPORT，否则 NO_SUPPORT；措辞取协议中预注册的 allowed_positive/negative_wording。
8. 对困难患者（002/022/044/080/153）显式报告（validation 角色给出两臂指标，非该角色注明排除原因）；原子写出 decision.json 与 patient_differences.csv。

**与调度的联动**：第 3 步 `_validate_schedule` 与 `calibrate_h3_v2_full_timestep_native_null.py` 产出的全时域调度工件（`[native,shallow,null]=[a,0,1-a]`）哈希绑定；实验两臂的差异唯一来源即 build 脚本注入的 `cross_level_router.h3_schedule_path`（h3_v2_native_null 臂）vs `hard_all_null`（no_route 臂），对应 `src/model/frequency/h3_native_null_schedule.py` 的加载与路由消费。

### `scripts/estimate_h3_prior_from_png.py`

**职责**：直接从配对 PNG 文件估计探索性六频带 H3 先验（preview）。刻意绕过所有张量缓存，用于本地代码验证与缓存布局不同时的可移植云端重算。输出恒标 `preview_only`，不能被 fail-closed 的正式 H3-v2 调度加载器消费。对每个样本构造冻结条件均值残差 `residual = PET - mean_model(CT)`，跨全部 Brownian-bridge 时刻复用同一确定性高斯噪声图；利用 Haar 线性性，用每样本每频带的三个充分统计量精确评估 0..T-1 全时刻掩膜余弦曲线，避免 `样本×时刻×图像` 的内存展开。

**接口**：
- `@dataclass(frozen=True) class PngPriorEntry`：PNG 样本条目；`class DirectPngPriorDataset(Dataset)`：直接读 PNG 三元组（CT/PET/掩膜）的数据集。
- `select_png_root(root, explicit, dataset_contract_path)`：按 数据集契约→main_data→Data/data 顺序定位 PNG 根；`build_png_entries(...)`：按 manifest 与角色索引 PNG 并做患者均衡子集；`read_png_triplet`/`_read_grayscale`：读取与归一化。
- `load_mean_predictor(checkpoint_path, device)`：加载条件均值预测器；`band_tensors(image)`（haar_dwt2 六频带）、`band_alignment_statistics(...)`、`recoverability_from_statistics(...)`；`estimate_patient_curves(...)`（`@torch.no_grad()`，充分统计量 A/B/C 精确全时刻曲线）；`fit_isotonic_prior(patient_curves, patient_roles)`（按频带保序单调不增约束）。
- `run(args) -> dict`、`_parse_args(argv)`、`main(argv) -> int`。CLI：`python scripts/estimate_h3_prior_from_png.py --mean-checkpoint PATH [--root] [--png-root] [--manifest main_data/split_manifest.csv] [--dataset-contract configs/dataset_contract_stage0a_v1.json] [--output-dir results/h3_png_prior_preview] [--pet-subdirectories pet pet_peizhuan] [--image-size 192] [--num-train-timesteps 1000] [--m-schedule linear] [--sigma-scale 1.0] [--analysis-seed] [--batch-size 16] [--num-workers] [--device auto] [--max-patients-per-role N] [--no-plot]`。

**依赖**：内部：`src/model/frequency/haar.haar_dwt2`、`src/model/mean_predictor.LowFrequencyPETPredictor`、`src/model/noise/base.BBDMBridgeSchedule`、`src/mechanism_validation/common`；外部：numpy、torch、PIL、sklearn.isotonic.IsotonicRegression、torch.utils.data。

**输入**：原始 PNG 三元组（CT/PET/病灶掩膜，pet 与 pet_peizhuan 子目录）、split manifest、数据集契约、病理排除的条件均值检查点。

**输出**：`--output-dir` 下 `h3_prior_preview.json`（decision=PREVIEW_ONLY、`preview_native_active_mass`、`route_mapping_preview`=`[a,0,1-a]`、crossings 汇总）、`h3_prior_curves.csv`、`h3_prior_patient_curves.npz`、`h3_prior_curves.png`（可 --no-plot）。

**处理过程**：
1. `_validate_h3_preview_parameters` 校验参数；`select_png_root` 定位 PNG 根，`build_png_entries` 按 manifest 角色索引样本（可选确定性患者均衡子集）。
2. `_validate_dataset_contract` 核对数据集契约与 manifest/清单一致；`load_mean_predictor` 加载均值模型并强制其为病理排除检查点且分区指纹匹配。
3. `DirectPngPriorDataset` 按 192×192 与固定归一化读取三元组；构造 `BBDMBridgeSchedule`（1000 步）。
4. `estimate_patient_curves`（no_grad）：每样本固定噪声（seed=sha256(analysis_seed|role|sample_id)），对每频带累计充分统计量 A=<x,x>、B=<e,e>、C=<x,e>，精确算出每时刻病灶掩膜余弦对齐可恢复性并缩放到 [0,1]。
5. `fit_isotonic_prior` 在"等患者权重 + 每频带随时刻保序单调不增"约束下拟合 native 活性质量 a；`_crossing_summary` 统计 a 越过 0.5 等阈值的时刻。
6. 写出曲线 CSV、患者曲线 NPZ 与预览图（matplotlib 经 `_write_plot`）。
7. 组装 `h3_prior_preview.json`：preview_only、路径以仓库相对 POSIX 序列化、input_contract（cache_read=False）、analysis 假设与 route_mapping_preview。
8. 返回摘要 dict（样本数/患者数/设备/输出路径），main 打印 JSON 并返回 0。

**与调度的联动**：本脚本与 `calibrate_h3_v2_full_timestep_native_null.py` 共享同一曲线方法论（同一 BBDMBridgeSchedule、同样的 `[a,0,1-a]` 映射与保序约束），但其产物被显式标记为不可被 `src/model/frequency/h3_native_null_schedule.py` 的 fail-closed 加载器消费（formal_h3_v2_claim_allowed=False）；`run_prior_anchored_router_100e.py` 的 `estimate_cloud_prior` 在云端用它的方法论重算先验。

### `scripts/summarize_prior_anchored_run.py`

**职责**：在不改变训练状态的前提下，汇总一次"先验锚定 100-epoch 路由实验"运行目录的完整性：校验运行配置身份、检查点序列、逐 epoch metrics.jsonl、runner state.json、先验工件契约与五阶段（prior_frozen→active_ramp→active_only_hold→destination_ramp→full_adaptive）观察点，产出 COMPLETE/INCOMPLETE 摘要（显式声明不构成科学声明）。

**接口**：
- `phase_for_epoch(epoch) -> str`：1-10 prior_frozen、11-20 active_ramp、21-30 active_only_hold、31-40 destination_ramp、41-100 full_adaptive、>100 out_of_contract。
- `summarize_run(run_dir_value, *, output=None, root=ROOT, write=True) -> dict`：主流程；辅助：`read_metrics_jsonl(path, expected_total_epochs)`（递归扫描 train/eval/validation 三段的 router/先验/病灶观测指标）、`inspect_checkpoints(checkpoint_dir, config, root)`、`_prior_summary(config, root)`（复用 `validate_prior_artifact`）、`_runner_state_summary(...)`、`_interesting_metrics`/`_metric_section_contract_errors`。
- `parse_args(argv)` / `main(argv) -> int`（错误返回 2）。CLI：`python scripts/summarize_prior_anchored_run.py --run-dir DIR [--output PATH] [--no-write]`。

**依赖**：内部：大量复用 `scripts/run_prior_anchored_router_100e`（CHECKPOINT_PATTERN、PHASE_OBSERVATION_EPOCHS、PIPELINE_ID、TOTAL_EPOCHS、RunnerError、inspect_resume_checkpoint、validate_prior_artifact、路径/哈希/原子写工具）；外部：argparse/json/math/collections。

**输入**：运行目录下的 `resolved_config.yaml`、`state.json`、`training_metrics.jsonl`、检查点目录、先验工件。

**输出**：`<run-dir>/summary.json`（或 --output 指定路径，--no-write 只打印）：decision=COMPLETE/INCOMPLETE、completion_checks 七项、runner_state、prior、checkpoints、metrics 摘要、phase_contract 表。

**处理过程**：
1. 解析 run-dir 为仓库相对路径，读取 `resolved_config.yaml` 并验证 `prior_anchored_run.pipeline_id` 匹配。
2. `inspect_checkpoints` 检查检查点目录（命名、epoch 覆盖、最终检查点有效性）。
3. `read_metrics_jsonl` 读取逐 epoch 记录，递归 train/eval/validation 三段提取路由质量（frequency/route_native|shallow|null_mass 等规范指标）、先验与病灶观测指标，并校验 1..100 epoch 完整。
4. 读取 state.json（缺失/损坏记录错误），`_runner_state_summary` 校验 runner 状态契约。
5. `_prior_summary` 经 `validate_prior_artifact` 校验先验工件（prior 锚定调度的输入）。
6. 计算 completion_checks：配置 100 epoch、runner 契约、先验契约、最终检查点、metrics 恰为 1..100、最终 step 与检查点一致、五个阶段观察点齐全且阶段名匹配。
7. 组装 summary（含 phase_contract 逐观察点表），`write_json_atomic` 原子写出。
8. main 打印 JSON；RunnerError/ValueError/FileNotFoundError → stderr + 退出码 2。

### `scripts/run_prior_anchored_router_100e.py`

**职责**：云端 100-epoch 先验锚定路由实验运行器。本地安全入口是 `--preflight-only`（静态契约校验，不需要 CUDA/云端张量缓存/原始 PNG/均值检查点）。真实启动流程：选择病理排除的均值检查点 → 在 run 目录内重新估计 direct-PNG 先验 → 写 run 拥有的 resolved YAML → 调用 `scripts/train_v2.py` 训练 → 校验最终检查点与精确 1..100 的 metrics 账本。所有持久化项目路径均为仓库相对 POSIX 路径。支持断点续跑（--resume/--resume-latest）与已完成检测（ALREADY_COMPLETE）。

**接口**：
- `run_cloud(args, *, root=ROOT) -> dict`：主流程（真实运行）；`build_static_preflight(config_path, runs_root, png_root, manifest, dataset_contract, mean_checkpoint)`：--preflight-only 的静态契约校验。
- 关键辅助：`validate_base_config(base_config, root)`；`select_mean_checkpoint(root, explicit)`（默认候选 freq_mean_excluded_v1/mean_best.pt 等，须病理排除）；`estimate_cloud_prior(run_dir, mean_checkpoint, manifest, dataset_contract, png_root, device, root)`（子进程调用 `scripts/estimate_h3_prior_from_png.py` 在 run 目录内重估先验）；`validate_prior_artifact`（校验 h3_prior_preview.json 契约）；`materialize_run_config(...)`（把先验路径/哈希/均值检查点注入基配置生成 resolved_config.yaml）；`invoke_training(run_dir, config_path, resume_checkpoint, root)`（构造并执行 train_v2.py 命令）；`find_latest_resume_checkpoint`/`inspect_resume_checkpoint`（检查点命名/RNG/lineage 验证）；`validate_complete_metrics_jsonl`（1..100 epoch 账本）；`update_state`（原子 state.json 状态机）；`select_latest_resumable_run`/`prepare_fresh_run`；`_require_cuda`。
- `class RunnerError(RuntimeError)`；常量：`TOTAL_EPOCHS=100`、`PHASE_OBSERVATION_EPOCHS=(10,20,30,40,100)`、`CHECKPOINT_PATTERN`。CLI：`python scripts/run_prior_anchored_router_100e.py [--config configs/experiments/slmf_png_prior_anchored_router_100e.yaml] [--runs-root results/prior_anchored_router_100e/runs] [--run-dir] [--resume | --resume-latest] [--preflight-only] [--mean-checkpoint] [--png-root] [--manifest] [--dataset-contract] [--device cuda]`。

**依赖**：内部：`scripts/estimate_h3_prior_from_png.py`（子进程重估先验）、`scripts/train_v2.py`（训练）、`configs/experiments/slmf_png_prior_anchored_router_100e.yaml`（基配置）、均值检查点；外部：yaml、argparse/subprocess/hashlib/re。

**输入**：基配置、病理排除均值检查点、原始 PNG 根/manifest/数据集契约（云端重估先验用）、CUDA 设备；续跑时已有的 run 目录（state.json、resolved_config.yaml、检查点、metrics.jsonl）。

**输出**：`results/prior_anchored_router_100e/runs/<run-id>/` 下 `state.json`（RUNNING/INTERRUPTED/FAILED/COMPLETE 状态机）、`resolved_config.yaml`、`h3_prior_preview.json`（run 内先验）、`logs/`（train/estimate 日志）、检查点目录与 metrics；stdout JSON 决策（COMPLETE/ALREADY_COMPLETE）。

**处理过程**：
1. `_require_cuda` 强制 CUDA；解析/归一化全部路径为仓库相对 POSIX。
2. 续跑模式：`select_latest_resumable_run` 或显式 --run-dir；`_resume_context` 读取配置与状态，`find_latest_resume_checkpoint` 找到最新合法检查点（校验 RNG 状态与 lineage）。若已达 epoch 100 → `validate_complete_metrics_jsonl` 后标记 ALREADY_COMPLETE，不再重启。
3. 新跑：`prepare_fresh_run` 创建 run 目录并置 state=RUNNING；`validate_base_config` 校验基配置。
4. `select_mean_checkpoint` 选病理排除均值检查点并哈希；合并 CLI 覆盖的 manifest/契约后 `_verify_run_inputs` 云端输入预检。
5. `estimate_cloud_prior` 子进程调用 estimate_h3_prior_from_png.py 在 run 目录内重估 direct-PNG 先验；`validate_prior_artifact` 校验产物契约并取哈希。
6. `materialize_run_config` 将先验路径/哈希、均值检查点、路径覆盖注入基配置，原子写出 resolved_config.yaml 并再验输入。
7. `invoke_training` 执行 train_v2.py（续跑传 resume_checkpoint）；完成后校验 ckpt_epoch0100.pt 存在且元数据合法、metrics 恰为 1..100 且最终 step 与检查点一致。
8. 更新 state=COMPLETE 并返回决策 dict；任何异常（含 KeyboardInterrupt）都落盘 INTERRUPTED/FAILED 后抛出，main 返回 2。

**与调度的联动**：resolved 配置把 `h3_prior_preview.json` 交给 `src/model/frequency/prior_anchor_schedule.py` 消费（先验锚定的五阶段路由调度：prior_frozen→active_ramp→active_only_hold→destination_ramp→full_adaptive），路由质量由 `src/model/loss_terms/spectral_router.py` 按时刻输出 native/shallow/null 质量；`summarize_prior_anchored_run.py` 与 `audit_prior_anchored_run.py` 均复用本模块常量与校验器对账。

### `scripts/audit_prior_anchored_run.py`

**职责**：对一次先验锚定 100-epoch 云端运行做只读审计。绝不修改训练代码、resolved_config.yaml、先验工件、检查点或 training_metrics.jsonl；唯一写入是 run 自己 `observations/` 目录下的单个 JSON 报告，其余全部打印到 stdout 以便从云端拷回。边界规则：云端 run 目录是唯一事实源（不假设本地 PNG/缓存/检查点与云端一致）；所有报告路径为仓库相对；不刷新任何 Formal H3 运行时 SHA、不绕过 fail-closed 门；探索性运行——报告不得解读为临床疗效/生产就绪/因果路由收益，A/B/C/D 尚非有效配对比较。

**接口**：
- `audit(run_dir: Path, output: Path | None) -> dict`：主流程，返回完整报告（同时写 observations/audit.json）。
- 辅助：`read_metrics(path)`（逐 epoch 展平记录、查重/断档/NaN）、`inspect_final_checkpoint(path)`（weights-only 可加载性、必需键、epoch/step/监控字段）、`verify_phase_behavior(records, router_cfg)`（按配置的 warmup/ramp/decay 参数核对每 epoch 路由阶段行为）、`detect_anomalies(records_by_epoch)`（loss/NaN/突变类异常）、`combined_score/best_validation_epoch(records, alpha, stripe_penalty)`（最优验证 epoch 与 final-vs-best 对比）、`table_for/render_table`（train/route/val 三张定表）、`print_report`。
- `parse_args(argv)` / `main(argv) -> int`。CLI：`pixi run python scripts/audit_prior_anchored_run.py --run-dir results/prior_anchored_router_100e/runs/<run-id> [--output PATH]`（默认输出 `<run-dir>/observations/audit.json`）。

**依赖**：内部：无 Python 级导入（与 runner/summarizer 仅共享契约常量：TOTAL_EPOCHS=100、PHASE_OBSERVATION_EPOCHS=(10,20,30,40,100)、phase_for_epoch 阶段表、CHECKPOINT_RE 命名）；读取 run 目录的 state.json、resolved_config.yaml（`modules.residual_frequency.cross_level_router` 的 policy/prior_warmup_epochs/prior_active_ramp_epochs/prior_destination_warmup_epochs/prior_destination_ramp_epochs/prior_anchor_decay_end_epoch/prior_anchor_final_scale/h3_schedule_source/h3_allow_unverified_preview_lineage）、h3_prior_preview.json、检查点与 metrics；外部：yaml（必需，缺失退出 2）、argparse/hashlib/json/math/re。

**输入**：云端 run 目录全套工件（state.json、resolved_config.yaml、training_metrics.jsonl、检查点目录、先验工件、logs/）。

**输出**：`<run-dir>/observations/audit.json`（唯一写盘）+ stdout 报告：sha_checks（config/先验/metrics 三向哈希核对）、train/route/val 三张表、phase_behavior、anomalies、best_validation、completeness_gate（12 项）、verdict（RUN_COMPLETE_NUMERICALLY_STABLE / RUN_COMPLETE_WITH_ANOMALIES_TO_INVESTIGATE / INCOMPLETE_REVIEW_GATE / RESULT_UNUSABLE_RERUN）。

**处理过程**：
1. 读取 state.json 与 resolved_config.yaml（容错记录读取错误），抽取 training/runtime/router/prior_anchored_run/best_checkpoint 配置。
2. 优先按 state.json 声明的仓库相对路径定位先验/metrics/最终检查点（state 为其作担保），回退到配置再到 run 目录默认。
3. sha_checks：对 resolved_config、prior_artifact、training_metrics 做期望 SHA-256 与实际文件哈希交叉核对。
4. `read_metrics` 展平逐 epoch 记录（train/eval/validation 三段），检查重复/断档/NaN 与 step 严格递增、phase 逐 epoch 匹配。
5. `inspect_final_checkpoint` 以 weights-only 方式检查 ckpt_epoch0100.pt：可加载、必需键、epoch=100、step、监控字段。
6. `verify_phase_behavior` 按 router 配置的阶段参数核对路由行为（先验冻结→活性爬坡→仅活性保持→目的地爬坡→全自适应）；`detect_anomalies` 扫描数值异常。
7. `best_validation_epoch` 以 combined_alpha/stripe_penalty 组合分数求最优验证 epoch，判断"最终即最优"。
8. 汇总 12 项 completeness_gate 全过且无异常 → RUN_COMPLETE_NUMERICALLY_STABLE；否则按规则降级；写出 observations/audit.json 并返回报告。

**与调度的联动**：`router_config` 小节直接复核 `src/model/frequency/prior_anchor_schedule.py` 的阶段参数（warmup/ramp/decay/scale 与 h3_schedule_source 指向的先验工件来源、是否允许未验证 preview lineage）；route_table 中的 native/shallow/null 质量来自 `src/model/loss_terms/spectral_router.py` 在训练期写出的逐 epoch 观测。

### 模块依赖小结

**依赖的内部模块**（grep 查证）：
- `src/model/frequency/h3_native_null_schedule.py`（`load_h3_native_null_schedule`、`H3_V2_PIPELINE_ID`）：被 `build_h3_v2_experiment_configs.py` 导入校验调度工件；正式 H3-v2 契约的属主，被 `prior_anchor_schedule.py` 与 `frequency/spectral_router.py` 复用。
- `src/model/frequency/prior_anchor_schedule.py`（`load_prior_anchor_schedule`）：formal_h3_v2 与 direct_png_preview 双来源的 fail-closed 分派加载器；消费 `estimate_h3_prior_from_png.py` 产出的 `h3_prior_preview.json`（preview 消费需显式 opt-in `h3_allow_unverified_preview_lineage`）。被 `frequency/spectral_router.py`（policy=prior_anchored_learned 分支，line 664）调用。
- `src/model/frequency/spectral_router.py`：路由器实现，按 `route_policy∈{h3_native_null, prior_anchored_learned}` 在构造时分派上述两个加载器并把 active_mass 注册为非持久 buffer；`audit_h3_fixed_schedule_inference.py` 对其源码做 SHA-256+AST 钉死检查（`DEFAULT_ROUTER_SOURCE`）。
- `src/model/loss_terms/spectral_router.py`（`SpectralRouterRegularizationLoss`）：训练期路由正则项，含 `prior_anchor_weight` 先验锚定惩罚与 `spectral_route_is_prior_anchored` 门控；其逐 epoch 写出的 native/shallow/null 质量是 summarize/audit 两脚本的 route_table 数据源。
- `src/model/noise/base.BBDMBridgeSchedule`（calibrate/estimate）、`src/model/mean_predictor`、`src/model/frequency/haar.haar_dwt2`（estimate）、`src/data/dataset.CachedDataset` 与 `src/data/lineage`（calibrate/build）、`src/mechanism_validation/common`（哈希/分区/manifest 工具，被 6 个脚本共享）。
- 脚本间复用：`run_h3_v2_overnight.py` 子进程调度 calibrate/build/train_v2/evaluate/analyze；`summarize_prior_anchored_run.py` 直接 import `run_prior_anchored_router_100e` 的常量与校验器；`audit_prior_anchored_run.py` 不导入 runner 而共享同一契约常量（100 epochs、(10,20,30,40,100) 观察点、ckpt 命名）。

**消费方**（grep 查证）：`src/` 下未发现任何对本批脚本的导入（它们是叶子 worker，由 overnight runner 以子进程方式编排或由人直接运行）；`tests/test_h3_png_prior_preview.py`、`tests/test_prior_anchored_router_100e_runner.py`、`tests/test_summarize_prior_anchored_run.py` 直接导入测试；`run_h3_fixed_schedule_overnight.py` 与 `run_h3_v2_overnight.py` 分别编排 05B/05C 全链路；`audit_prior_anchored_run.py` 的报告由人读取（云端 observations/audit.json）。`artifacts/pfm_simple_cloud_*` 与 `.worktrees/*` 下存在这三个 src 文件的快照副本（非本模块依赖）。


---

## 11. 评估与行为分析脚本

### 模块概述

本组是 SLMF-BBDM（CT→PET 合成 / 病灶小目标分割扩散模型）的实验后评估与机制行为分析脚本层，全部位于 `scripts/` 下、均以 CLI 方式独立运行，遵循"只读模型、零训练"的审计范式：加载已训练 checkpoint（EMA/raw 权重 + 数据 lineage 校验），通过 `residual_preconditioner` 路由器暴露的推理干预 API（destination / frequency / route_action / gain / window）做因果性干预实验，或在既有评估工件上做纯离线统计。

数据流大体分三层：(1) 逐样本采样评估内核（`scripts/eval_router_background_suite.py` 的 `evaluate_variant`，被 oracle 脚本直接复用）；(2) 干预矩阵筛查（B0-B8 行为轴、六槽位 oracle 枚举、梯度死锁诊断）；(3) 下游零训练分析（medoid 轨迹统计、JSONL 训练指标对比、`src/mechanism_validation` 特征涌现/因果审计的 CLI 封装）。公共模式包括：fail-closed 输出目录（拒绝复用非空目录以防陈旧 COMPLETE.json）、SHA-256 身份指纹与断点续跑 manifest、患者级 cluster bootstrap 统计、B0 字节级复现门，以及 `cloud_run_manifest.json` + `COMPLETE.json` 的云端同步完成协议。

---

### `scripts/diag_router_grad.py`

**职责**：诊断 prior-anchored 谱路由器是否存在"梯度死锁"（experiment A）。冻结全模型、仅放开两个路由头（`prior_active_heads` + `prior_destination_heads`），在单个固定 batch 上测量 reg_on 基线、A0（关闭全部谱路由正则后的数据损失梯度下限）与 A1（router-only AdamW 高学习率过拟合），按预声明阈值输出 PASS / FAIL / INCONCLUSIVE 判决，区分"正则是限制因素"与"幅度契约（零初始化投影）是根因"。对模型代码只读：正则权重仅在内存置零、永不持久化。

**接口**：可执行脚本，顶层 `main() -> None`，辅助函数均为模块级私有：`_resolve_run_config(template, checkpoint) -> (config, source)`、`_zero_router_regularizers(model) -> List[str]`（置零 9 项正则权重）、`_freeze_all_but_router_heads(model) -> (trainable, frozen)`、`_head_final_rms_grad(heads) -> float`、`_route_snapshot(model, logs) -> Dict`、`_set_router_epoch(model, epoch)`。CLI 用法：

```bash
python -u scripts/diag_router_grad.py \
    --config configs/experiments/slmf_png_prior_anchored_router_100e.yaml \
    --checkpoint results/prior_anchored_router_100e/runs/<run>/checkpoints/ckpt_epoch0100.pt \
    --steps 50 --lr 1e-2 --device cuda --out results/router_diag
```

（`--checkpoint` 可省略 → fresh-init，用于验证 experiment D 的去零投影；此时 `--config` 须传 resolved_config。）

**依赖**：
- 内部: `src.data.dataset.build_dataloaders`；`src.model.slmf_bbdm.SLMFBBDM`（`from_config` / `loss_terms.spectral_router_regularization` / `residual_preconditioner`）。
- 外部: torch、yaml、argparse、json、hashlib、os、sys、time。

**输入**：实验模板 YAML（`h3_schedule_path=null`，需由 run 的 `resolved_config.yaml` 或 `prior/h3_prior_preview.json` 解析补全）、checkpoint `.pt`（`weights_only=True`，`strict=False`）、CLI 参数 `--steps/--lr/--seed/--device/--out`、单个训练 batch（复用项目 loader 保证键名与归一化一致）。

**输出**：`results/router_diag/diag_router_grad_<UTC时间戳>.json`（`meta` 含阈值与参数量统计、`reg_on_baseline`、`A0_reg_off_grad_floor`、`A1_per_step` 逐步记录、`verdict`）+ stdout 摘要表与一行判决。不写任何 checkpoint。

**处理过程**：
1. 解析参数；有 checkpoint 时优先采用其 run 目录的 `resolved_config.yaml`，否则把 run 自带的 prior artifact 路径与 sha256 注入模板 config。
2. `SLMFBBDM.from_config` 构建模型、加载权重、`train()` 模式，`set_training_epoch(99)` 使 active/destination progress = 1。
3. `build_dataloaders` 取第一个 batch 送入 device；冻结全部参数后仅重新放开两个路由头。
4. reg_on 基线：完整前向 + backward，测两 head `.final` 子模块的 RMS 梯度（应复现审计的 ~1e-6）。
5. A0：内存置零 `spectral_router_regularization` 的全部权重后重测——唯一梯度源变成 base diffusion + lesion 损失。
6. A1：router-only AdamW（`--lr`）在同 batch 上过拟合 `--steps` 步，逐步记录 `active_delta_abs_mean`、`prior_active_mae`、`route_shallow_mass`、参数更新/范数比。
7. 依阈值（`GRAD_FLOOR_PASS=1e-4`、`GRAD_FLOOR_FAIL=1e-7`、`DELTA_PASS=1e-2`、`DELTA_FAIL=1e-3`、`SHALLOW_PASS=1e-2`）给出判决：PASS=正则是限制因素；FAIL=幅度契约是根因（建议进入 experiment D）；否则 INCONCLUSIVE。
8. 写 JSON 工件并打印判决。

---

### `scripts/eval_run_metrics.py`

**职责**：离线评估一个 prior-anchored router 训练 run（300e，含 experiment D 修复）对照 100e no-op 基线。读取 `training_metrics.jsonl`，按 lesion peak error 选最佳 epoch，打印逐评估 epoch 的病灶指标与 shallow-mass 尾部（D 修复是否让 shallow 分支保持存活），并对最佳 epoch 与硬编码基线做头对头对比与判决。

**接口**：可执行脚本，顶层 `main() -> None`；辅助：`_find_metrics(name) -> Optional[str]`（默认 glob `results/prior_anchored_router_300e/**/training_metrics.jsonl`）、`_val(row) -> dict`、`_find_value(node, substr)`（递归搜索嵌套 dict 中首个含子串的数值键，用于在 train 块深处的 `route_shallow_mass`）、`_read(path) -> List[dict]`。CLI 用法：

```bash
pixi run python -u -m scripts.eval_run_metrics \
    --metrics results/prior_anchored_router_300e/checkpoints/training_metrics.jsonl
```

**依赖**：
- 内部: 无（纯离线 JSONL 分析，不 import 任何 src 模块）。
- 外部: argparse、glob、json、os（仅标准库）。

**输入**：`training_metrics.jsonl`，记录结构为 `{"epoch":..., "train":{...}, "eval":{...}, "validation":{mae, lesion_peak_error_norm, ...}}`——验证指标在 `validation` 块（键无 `val/` 前缀），路由诊断（如 `route_shallow_mass`）在 `train` 块。

**输出**：仅 stdout：(1) 逐 epoch 指标表（ep/peak/topq/roi/fail/smund/o-i/psnr/stripe/shallow）；(2) 最佳 epoch 与基线 `BASELINE`（100e no-op ep70：peak 0.1330、topq 0.1810、shallow 9.10e-7 等）逐指标 delta/% 与箭头；(3) 判决 `D HELPED`（peak 改善≥5% 且 shallow 尾部≥1e-3）/ `D HURT`（peak 恶化≥5%）/ `D NEUTRAL`。

**处理过程**：
1. 定位 metrics 文件（显式路径或默认 glob）。
2. 逐行 `json.loads` 读入全部行（容忍坏行）。
3. 过滤出 `validation.lesion_peak_error_norm` 非空的 eval 行并按 epoch 排序。
4. 打印逐 epoch 指标行，shallow 值经 `_find_value` 递归查找。
5. 取 `lesion_peak_error_norm` 最小的 best epoch，映射 `VAL_KEYS` 后与 `BASELINE` 逐指标对比。
6. 对全部行的最后 30% epoch 求 `route_shallow_mass` 均值（D-alive 阈值 1e-3）。
7. 打印 peak/topq 相对变化与三选一判决。

---

### `scripts/eval_oracle_destination.py`

**职责**：同 checkpoint 的精确 oracle 上界分析。六个路由动作槽位（L2-LH/HL/HH、L1-LH/HL/HH）枚举全部完整动作映射（默认 `native,shallow,null` → 3^6 = 729 种；`--choices native,shallow` → 2^6 = 64 种 destination-only 搜索），每种映射下重置评估 RNG、同一有序数据集采样并记录逐样本指标，再按预声明目标指标为每个样本独立选择最优动作映射。目标 PET 仅在推理后用于选路——是不可部署的上界分析；`null` 同时改变可用性与目的地。

**接口**：可执行脚本。公开函数：`enumerate_action_maps(choices) -> list[tuple[str,...]]`（`itertools.product`）、`action_map_code(actions) -> str`（如 `\"NNN-SS0\"`）、`action_map_payload(actions) -> dict[slot, action]`、`select_oracle_rows(baseline_rows, variant_rows, action_maps, *, objective, lower_is_better, learned_fallback) -> (forced_rows, fallback_rows)`（逐样本选最优映射；fallback 版本在 oracle 不优于 learned 时回落，保证非劣上包络）；入口 `_main(args) -> int`、`build_parser()`、`main() -> int`。CLI 用法：

```bash
pixi run python -u -m scripts.eval_oracle_destination \
    --config configs/experiments/slmf_png_prior_anchored_router_identifiable_full_100e.yaml \
    --checkpoint results/.../ckpt_best_combined.pt \
    --split val --max-samples 64 --steps 50 \
    --objective lesion_topq_peak_error_norm --choices native,shallow,null \
    --output-dir results/oracle_destination/identifiable_full_e100
```

**依赖**：
- 内部: `scripts.eval_router_background_suite`（复用 `_json_safe`、`_load_dataset`、`_numeric_summary`、`_paired_effect`、`_seed_evaluation`、`_select_checkpoint_state`、`_sha256`、`_write_csv`、`_write_json`、`evaluate_variant`）；`src.data.lineage.load_checkpoint_data_lineage / validate_checkpoint_data_lineage`；`src.model.config_utils.load_full_config / resolve_runtime_profile`；`src.model.slmf_bbdm.SLMFBBDM`。
- 外部: torch、argparse、itertools、json、math、os、sys、time、pathlib、collections.Counter。

**输入**：resolved config YAML、checkpoint `.pt`（`weights_only`）、`--choices` 动作集、`--objective` 逐样本数值指标（默认 `lesion_topq_peak_error_norm`）、`--maximize-objective`、`--frequency-mode full|ll_off`、`--plan-only`（只打印枚举计划）、`--resume/--max-combinations 729`、`--weights ema|raw`，以及透传给 `evaluate_variant` 的几何参数（`--lowpass-sigma 4.0`、`--body-threshold 0.03`、`--lesion-exclusion-radius 8`、`--hotspot-hit-radius 3`、`--recovery-ratio 0.6`、`--small-lesion-quantile 0.25` 等）。

**输出**：`--output-dir` 下：`run_manifest.json`（身份指纹 + oracle 语义与警告）、`learned_baseline.jsonl`、`variants/<code>.jsonl`（每动作映射一份，断点续跑缓存）、`oracle_forced_samples.{jsonl,csv}`、`oracle_with_learned_fallback_samples.{jsonl,csv}`、`variant_summary.csv`、`oracle_summary.json`（基线/forced/fallback 数值摘要、配对效应、`_selection_summary` 的组合与槽位选择分布 `slot_fractions`）。

**处理过程**：
1. 解析 choices 并枚举动作映射，超过 `--max-combinations` 即拒绝；`--plan-only` 只打印编号/代码/payload。
2. 写 `run_manifest.json` 身份指纹（config/ckpt sha256、split、max_samples、batch_size、steps、seed、choices、objective、frequency_mode）；`--resume` 时 `_validate_resume_manifest` 校验身份一致否则报错。
3. 加载 config 与 checkpoint，校验数据 lineage；`_resolve_conditional_mean_checkpoint` 将配置的 conditional-mean checkpoint 解析为绝对路径或自举到被评估 checkpoint（`mean_predictor.*` 张量）。
4. 构建 `SLMFBBDM`、加载 EMA/raw 权重；要求 router 具备 `set_inference_route_action_intervention` API，并设置 `set_inference_destination_intervention("learned")` 与 frequency 干预。
5. `_load_dataset` + 固定顺序 DataLoader；先评估 learned baseline（`_load_or_evaluate` 支持 resume 缓存）。
6. 逐动作映射调用 `router.set_inference_route_action_intervention(actions_l2=..., actions_l1=...)` → `evaluate_variant`（grid_samples=0）→ 落盘 `variants/<code>.jsonl`，打印进度与 ETA。
7. `select_oracle_rows` 逐样本按 objective 选最优映射（forced），并生成带 learned 回落的 fallback 版本（oracle 不优于 learned 时标记 `oracle_used_learned_fallback=1`）。
8. `_paired_effect` 对 forced/fallback vs learned 计算患者级配对 bootstrap 效应（seed+7001/7002），写入 `oracle_summary.json`；最后清除路由干预。

---

### `scripts/eval_action_axis_screen.py`

**职责**：固定 router checkpoint 的免训练 B0-B8 行为轴筛查。B0 = learned 基线；B1-B8 依次为频率全关 / 细节关 / LL 关、全 native / 全 shallow 路由动作、细节增益 0.5 / 1.5、仅晚期时间窗（`logsnr_min=0`）。脚本整体 fail-closed：checkpoint SHA-256 必须与用户提供的期望值一致、验证队列确定且患者平衡、每次干预复用相同 RNG 种子与样本顺序、筛查开始前 B0 必须逐字节复现，全脚本无 optimizer / backward / checkpoint 写入。

**接口**：公开函数：`build_patient_balanced_subset(dataset, *, count, max_per_patient, allow_positive_only_cohort=False, all_validation=False) -> (Subset, records)`（按病灶面积四分位分层 small/large/middle/background + 患者上限轮转挑选，断言层覆盖与患者上限）；`evaluate_run(*, model, loader, cohort, device, seed, steps, amp, body_threshold, lesion_exclusion_radius, lowpass_sigma, return_frequency_trace=True) -> (metrics_rows, predictions[N,1,H,W], static_arrays, q_rows)`（`@torch.no_grad()`，采样 + 六主指标 + 逐时间步路由 q 轨迹）；`_paired_summary(rows, *, seeds, bootstrap_replicates, bootstrap_seed)`（variant−B0 配对效应，患者聚类 bootstrap CI95 + MCSE 判定）；`_aggregation_rows(...)`（pixel mean / median / 全图 medoid 三种聚合的指标）；`_snr_rows(*, residual, cohort, schedule, steps, noise_replicates, seed, device) -> (sample_rows, patient_rows)`（残差 Haar 各 level/band SNR）；`_region_effect_rows`、`_sampling_variation_rows`、`_medoid(stack)`；入口 `_main(args)`、`build_parser()`、`main(argv)`。CLI 用法：

```bash
python scripts/eval_action_axis_screen.py \
    --config configs/experiments/<exp>.yaml --checkpoint results/.../ckpt.pt \
    --expected-checkpoint-sha256 <hex> --output results/<screen_dir> \
    [--variants B0,B1,B4] [--seeds 42,43,44,45] [--samples 64 --max-per-patient 3] \
    [--all-validation] [--allow-positive-only-cohort] [--skip-q-trace] [--skip-band-snr]
```

**依赖**：
- 内部: `scripts.eval_router_background_suite`（`_body_nonlesion_mask`、`compute_body_background_metrics`）；`scripts.evaluate`（`_seed_evaluation`、`_select_checkpoint_state`、`_to_numpy`、`compute_normalized_lesion_metrics`、`compute_ssim`、`compute_target_relative_false_hotspots`）；`src.data.dataset.CachedDataset`（`_main` 内延迟 import）；`src.data.lineage`；`src.model.config_utils`；`src.model.frequency.haar.haar_dwt2`；`src.model.slmf_bbdm.SLMFBBDM`；`src.model.trainer._to_unit_interval`。
- 外部: torch、numpy、argparse、csv、itertools、json、hashlib、subprocess、matplotlib 未用（无绘图）。

**输入**：config YAML（`data.cache_dir / split_manifest / required_keys`，`--split` 仅允许 `val`）、checkpoint + 强制 `--expected-checkpoint-sha256`、`--weights raw|ema`（默认 raw）、队列参数（`--samples 64`、`--max-per-patient 3`、`--all-validation`、`--allow-positive-only-cohort`）、`--seeds \"42,43,44,45\"`、`--steps 50`、几何参数（body_threshold / lesion_exclusion_radius / lowpass_sigma）、统计参数（`--bootstrap-replicates 5000`、`--analysis-seed`、`--noise-replicates 16`）。

**输出**：`--output` 下：`cohort.csv`、`cohort_audit.json`（分层计数与数据局限声明）、`run_manifest.json`（scope 标注 screen 或 full_validation_confirmation、git 状态、全部参数）、`reproducibility.json`、`predictions/<variant>_seed<seed>.npz`、`cohort_tensors.npz`（target/ct/mask/mean_pet 静态张量，供下游分析）、`metrics.csv`、`q_trace.csv`（逐时间步 level×band 的 q 与病灶/背景均值）、`action_output_effects.csv`、`sampling_variation.csv`、`paired_summary.csv`、`aggregation_metrics.csv`、`band_snr_sample.csv` / `band_snr_patient.csv`、`interventions/<variant>_seed<seed>.json`、`COMPLETE.json`。

**处理过程**：
1. 校验文件存在、checkpoint sha256 匹配、输出目录为空（或 `--resume-empty`）；启用确定性算法（`torch.use_deterministic_algorithms(True)`）。
2. 加载 config/checkpoint、lineage 校验、strict 载入权重；检查 router 具备 4 个推理干预方法（gain/window/intervention/route_action）。
3. `build_patient_balanced_subset` 构建分层患者平衡队列（小/大病灶层必选，背景层视可用性；`--all-validation` 直接全量）。
4. 复现门：同 seed 连跑两次 B0 `evaluate_run`，要求预测 `array_equal` 且指标行全等，否则 `RuntimeError` 中止。
5. 双重循环 seed × variant：`_configure_variant` 设置 destination=frequency=route/gain/window 干预并记录实际生效值 → `evaluate_run` 采样（可选 frequency_trace）→ 保存预测 npz；非 B0 变体对同 seed B0 计算 `_region_effect_rows`（病灶/背景/全局输出 L1 差）。
6. B0 各 seed 间做 `_sampling_variation_rows`（纯采样噪声对照）。
7. `_paired_summary` 计算各 variant × 六主指标的配对效应（正 = 变体优于 B0），患者级 bootstrap CI95 与 `effect_gt_2_mcse`；`_aggregation_rows` 评估三种多 seed 聚合策略；`_snr_rows` 用 `haar_dwt2` 与 `model.noise_schedule` 的 `(1-m_t)^2·E_signal / (sigma_t^2·E_noise)` 计算各 level/band SNR。
8. 写出全部 CSV/JSON 与 `COMPLETE.json`（声明 `training_or_optimizer_used=false`，这是下游 medoid 分析的验收条件）。

---

### `scripts/analyze_medoid_zero_training.py`

**职责**：四轨迹 medoid run 的零训练后续分析，纯粹下游：只读已保存的 cohort 张量、四条 B0 预测轨迹（`eval_action_axis_screen.py` 的输出工件）与已算好的指标表，绝不加载 optimizer 或更新参数。三项预指定检查：(1) 患者聚合病灶分歧度 vs medoid TopQ 误差（刻画 medoid 不确定性）；(2) 连通域面积 vs baseline 输出梯度质量；(3) 面积 vs target-residual Haar 高频占比——后两项是"是否启动 instance-balanced、scale-matched 输出损失训练"的两道机制门。

**接口**：公开函数：`medoid_indices(stack[seed,sample,...]) -> np.ndarray`（逐样本取最低索引的全图 L1 medoid）；`trajectory_dispersion(stack, masks) -> pd.DataFrame`（全图/病灶内 pixel std 与成对 L1）；`baseline_output_gradient_maps(stack, target, *, batch_size=16) -> np.ndarray`（tau=0 推理端 \`pred_x0\` 输出梯度代理：`w_mse*MSE + L1 + 0.1*w_grad*梯度L1` 对预测求 abs 梯度后在四轨迹上平均）；`component_mechanism_rows(*, masks, gradient_maps, target_residual, cohort)`（连通域 × 两级 Haar 能量分解行）；统计工具 `patient_cluster_bootstrap_spearman`、`patient_aggregate_spearman`、`within_patient_slope`（患者固定效应斜率）、`_holm_adjust`；入口 `run_analysis(args) -> int`、`build_parser()`、`main(argv)`。CLI 用法：

```bash
python scripts/analyze_medoid_zero_training.py \
    --run results/<action_axis_run_dir> --output results/<analysis_dir> \
    [--bootstrap-replicates 10000] [--analysis-seed 20260729] \
    [--gradient-batch-size 16] [--verify-reference results/<prior_analysis_dir>]
```

**依赖**：
- 内部: 无任何 `src.` 导入（纯下游分析；torch 仅用于构建输出梯度代理的计算图）。
- 外部: numpy、pandas、matplotlib（Agg 后端）、scipy（`ndimage`、`stats`）、torch、argparse、csv、json、hashlib。

**输入**：`--run` 目录必须包含 `run_manifest.json`、`COMPLETE.json`（status=COMPLETE 且声明 `training_or_optimizer_used=false`）、`cohort.csv`、`cohort_tensors.npz`、`metrics.csv`、`aggregation_metrics.csv` 与 `predictions/B0_seed<seed>.npz`×4（manifest 恰好 4 个 seed）；逐文件校验样本顺序一致，预测形状须与 target 匹配。

**输出**：`--output` 下：`analysis_summary.json`（预注册检查、机制门、训练门决策、11 类谬误扫描、局限声明）、`validation_report.md`（Material Passport + 主统计表 + 训练门 + 谬误表 + 复现性）、`slice_analysis.csv`、`uncertainty_error_associations.csv`、`uncertainty_sensitivity.csv`、`patient_benefit_failure.csv`、`patient_profile_comparison.csv`、`continuous_size_effects.csv`、`component_mechanism_audit.csv`、`mechanism_gate_summary.csv`、`mechanism_secondary_associations.csv`、`uncertainty_vs_error.png`、`patient_benefit_waterfall.png`、`continuous_size_effect.png`、`mechanism_audits.png`、`artifact_manifest.json`；`--verify-reference` 时额外产出 `reproducibility.json`（14 个确定性工件 sha256 逐一比对，判定 REPRODUCIBLE）。

**处理过程**：
1. `_load_run` 严格校验源 run 完整性、零训练声明、恰好四轨迹、样本顺序一致后加载 cohort/metrics/aggregation/静态张量/预测栈（含各 npz 的 sha256）。
2. `medoid_indices` 选 medoid，`trajectory_dispersion` 计算分歧度；`_metric_improvement_frame` 合成 baseline（seed 均值）vs medoid 的逐切片绝对/相对改善与 `log2_lesion_area`。
3. `_uncertainty_rows`：4 个分歧度预测子 × 4 个误差结果的 patient-aggregate Spearman；`_uncertainty_sensitivity_rows` 做 mean/median/min-3-slices/leave-one-patient-out 多规格敏感性。
4. `_patient_rows` / `_profile_comparison`：患者级 benefit/failure 分组（TopQ 相对改善 >0 与否）、medoid seed 熵、Mann-Whitney 探索性对比。
5. `baseline_output_gradient_maps` 在四条轨迹上求输出梯度代理；`component_mechanism_rows` 对每个病灶连通域记录面积、梯度质量/密度/份额与 target-residual 的高/中/低频能量占比。
6. `_continuous_size_rows`：`within_patient_slope` 估计"病灶面积每翻倍的改善量"（患者固定效应）。
7. `_mechanism_summary` 两道门：面积↔log 梯度质量期望正相关、面积↔高频占比期望负相关，患者聚类 bootstrap CI 全部偏向预期方向才算 established；与不确定性检查一起做 Holm 校正；两门全过 → `ELIGIBLE_FOR_INSTANCE_BALANCED_SCALE_MATCHED_TRAINING`，否则 `DO_NOT_START_NEW_OUTPUT_LOSS_TRAINING`。
8. `_write_report` 写 markdown 报告；`--verify-reference` 时逐工件 sha256 比对并重写报告/摘要；最后写 `artifact_manifest.json` 并打印训练门决策。

---

### `scripts/eval_router_background_suite.py`

**职责**：Stage-2/3 路由器与 PET 背景评估套件，同时是本组的公共评估内核。固定一个 checkpoint、EMA 权重、数据集顺序与噪声种子，交叉 destination 干预（`learned | native_only | fixed | shuffled_destination`）× 频率干预（`full | detail_off | ll_off | all_frequency_off`）× DDIM 步数（如 20,100）；对 `mean_pet + pred_residual` 分解报告病灶恢复、全局热点、非病灶身体低/高通能量、分位偏置、条纹、failure、假热点等全套指标。fixed 目的地值取自 checkpoint epoch 的 train 块（`training_metrics.jsonl`），不从验证输出估计。

**接口**：公开函数（被其它脚本复用）：`evaluate_variant(model, loader, *, device, amp, steps, seed, lowpass_sigma, body_threshold, lesion_exclusion_radius, hotspot_hit_radius, recovery_ratio, small_lesion_quantile, small_lesion_underestimate_tolerance, grid_samples, verbose=True) -> (rows, grid)`（`@torch.no_grad()`，逐样本采样 + 全套指标行 + 网格图记录）；`compute_body_background_metrics(pred, target, mean_pet, pred_residual, ct, lesion_mask, *, ...) -> dict`；`compute_global_hotspot_metrics(pred, lesion_mask, *, hit_radius)`；`compute_false_hotspot_components(pred, target, lesion_mask, *, excess_margin, target_quantile)`；`_body_nonlesion_mask(ct, lesion_mask, *, body_threshold, exclusion_radius) -> (body, region)`；`_load_dataset(config, *, split, max_samples, allow_train_fallback) -> (dataset, selected_indices)`；`_load_fixed_destination(metrics_path, epoch) -> dict[level, list[float]]`；`_paired_effect(candidate, baseline, *, metric, lower_is_better, seed, bootstrap_samples=5000)`；`_numeric_summary(rows)`；`_save_grid(path, records, *, title, lowpass_sigma)`；入口 `_main(args)`、`build_parser()`、`main()`。CLI 用法：

```bash
pixi run python -u -m scripts.eval_router_background_suite \
    --config configs/experiments/slmf_png_prior_anchored_router_300e.yaml \
    --checkpoint results/.../ckpt_epoch0100.pt \
    --metrics results/.../training_metrics.jsonl \
    --split val --max-samples 64 --steps 20,100 \
    --output-dir results/router_background_suite/epoch0100
```

**依赖**：
- 内部: `scripts.evaluate`（`_annotate_small_lesion_metrics`、`_build_stratified_subset`、`_de_collate_meta`、`_seed_evaluation`、`_select_checkpoint_state`、`_to_numpy`、`compute_boundary_metrics`、`compute_failure_detection_metrics`、`compute_false_hotspot_count`、`compute_mae/mse/psnr/ssim`、`compute_normalized_lesion_metrics`、`compute_stripe_metrics`、`compute_target_relative_false_hotspots`）；`src.data.dataset.CachedDataset / FakeDataset`（`_load_dataset` 内延迟 import）；`src.data.lineage`；`src.model.config_utils`；`src.model.slmf_bbdm.SLMFBBDM`；`src.model.trainer._to_unit_interval`。
- 外部: torch、numpy、scipy.ndimage（`binary_closing/dilation/fill_holes`、`distance_transform_edt`、`gaussian_filter`、`label`）、matplotlib（`_save_grid` 内 Agg）、argparse、csv、json、hashlib。

**输入**：config YAML、checkpoint `.pt`（epoch 用于 fixed 回读）、`--metrics` JSONL（fixed 模式读键 `frequency/prior_anchor_{l2|l1}_{lh|hl|hh}_conditional_shallow` 共 6 个，缺一即报错）、`--modes` / `--frequency-modes` / `--steps` 逗号列表、`--fixed-epoch`、几何参数（`--lowpass-sigma 4.0`、`--body-threshold 0.03`、`--lesion-exclusion-radius 8`、`--hotspot-hit-radius 3`、`--recovery-ratio 0.6`、`--small-lesion-quantile 0.25`）、`--grid-samples 4`、`--shuffle-seed`（shuffled_destination 模式要求 batch≥2）、`--seconds-per-epoch`（训练时间当量换算）。

**输出**：`--output-dir` 下：`run_manifest.json`（schema v2，含 config/ckpt/metrics sha256 与 fixed 目的地值）、每个组合子目录 \`<mode>[_<freq>]_steps<NNN>/\` 内 `samples.jsonl`（逐样本全部指标）、`summary.json`（均值/中位数/标准差摘要 + 实际干预状态）、`decomposition_grid.png`（CT/Target/Mean/Residual/Pred/低通对比/掩膜八列网格）；根级 `suite_summary.json`（含 vs learned 的配对比较与 interpretation）、`run_summary.csv`、`paired_comparisons_vs_learned{,_full}.csv`。

**处理过程**：
1. 解析并校验 modes / frequency_modes / steps 组合合法性。
2. 构建模型、加载 checkpoint、lineage 校验、EMA/raw 权重选择；要求 router 具备 destination 与 frequency 两个干预 API。
3. fixed 模式从 `training_metrics.jsonl` 对应 epoch 的 train 块读六个 prior_anchor 条件 shallow 概率（checkpoint-epoch 训练诊断，非验证拟合）。
4. `_load_dataset`（Fake/CachedDataset，可选分层子集与 train 回退）→ 固定顺序 DataLoader。
5. 三重循环 mode × frequency_mode × step_count：设置 router destination/frequency 干预 → `evaluate_variant` 采样；逐样本先算 mae/mse/psnr/ssim/stripe/boundary，再算身体背景低/高通与分位指标，病灶内算 TopQ/峰值/ROI 恢复率与全局热点，最后 failure 检测与两级假热点指标。
6. `_annotate_small_lesion_metrics` 按面积分位标注小病灶行并派生小病灶恢复率字段。
7. 每组合落盘 samples.jsonl / summary.json / 网格 PNG 并打印 ETA。
8. 全部候选组合 vs `(learned, full, 同步数)` 基线做 `_paired_effect` 患者级配对 bootstrap（正效应 = 候选更好），汇总进 `suite_summary.json` 与 CSV。

---

### `scripts/eval_feature_emergence.py`

**职责**：Stage-A 零训练表征审计 CLI（A0 + A1 + A2），是 `src/mechanism_validation/feature_*.py` 的命令行封装：A0 特征来源审计（`feature_provenance.audit_feature_provenance`，静态）、A1 跨模态共享权重探针线性 CKA（`feature_emergence.cross_modal_similarity_analysis`）、A2 通道无关病灶涌现分析（`feature_emergence.lesion_emergence_analysis`）。只需 CT encoder 前向与 bundle 构建、无需扩散采样，可在 CPU 运行。

**接口**：可执行脚本，`main() -> int`（退出码 0/1）；辅助：`_select_checkpoint_state(checkpoint, weights=\"ema\")`（复刻 `scripts/evaluate` 的 EMA 选择：`ema_materialized_as_model` / `ema_model` / `model_ema` / `ema.shadow` 合并）、`_load_frozen_quartiles(args, config, loader)`（fail-closed：val/test split 必须由 `--frozen-quartiles-json` 提供训练集冻结的四分位数，从评估 split 现算属验证泄漏、直接拒绝；train/fake 可返回 None 由分析函数自行冻结）、`_patient_set_hash`、`_split_manifest_sha256`、`_git_sha`。CLI 用法：

```bash
python scripts/eval_feature_emergence.py \
    --config configs/experiments/slmf_png_xxx.yaml \
    --checkpoint results/.../ckpt_best_lesion.pt \
    --split val --output results/lesion_feature_emergence_v1/c1_audit/ \
    [--fake-data] [--max-samples N] [--device cpu|cuda:1] [--fixed-t 0] \
    [--skip-cka] [--skip-emergence] [--frozen-quartiles-json q.json]
```

参数透传关系：A0 ← `audit_feature_provenance(model, config, args.checkpoint, out_dir)`（`--fixed-t` 记入 `provenance["timestep_policy"]`）；A1 ← `cross_modal_similarity_analysis(model, loader, device, out_dir, seed=args.seed)`；A2 ← `lesion_emergence_analysis(model, loader, device, out_dir, fixed_t=args.fixed_t, frozen_quartiles=_load_frozen_quartiles(...))`；`--max-samples` 经 Subset 截断 loader；`--fake-data` 改写 config（FakeDataset、image_size=64、关 lineage 与 conditional_mean/mean_predictor/residual_bridge/residual_frequency 模块）并放宽 `load_state_dict(strict=False)`；`--weights ema|raw` 决定 checkpoint 状态选择。

**依赖**：
- 内部: `src.data.dataset.CachedDataset / FakeDataset`；`src.data.lineage.load_checkpoint_data_lineage / validate_checkpoint_data_lineage`；`src.mechanism_validation.common.file_sha256 / write_json / load_json / canonical_json_sha256`；`src.mechanism_validation.feature_emergence.cross_modal_similarity_analysis / lesion_emergence_analysis / _patient_keys`；`src.mechanism_validation.feature_provenance.audit_feature_provenance`；`src.model.config_utils.load_full_config / resolve_runtime_profile`；`src.model.slmf_bbdm.SLMFBBDM`。
- 外部: torch、numpy、argparse、datetime、platform、socket、subprocess（git sha）。

**输入**：resolved 实验 YAML、完整模型 checkpoint（`.pt`，`weights_only=True` + lineage 校验，required 由 `data.require_cache_lineage` 决定）、`--split train|val|test`、数据 `cache_dir/split_manifest` 或 `--fake-data`；val/test 强制要求 `--frozen-quartiles-json`（含非降序 q1/q2/q3）。

**输出**：`--output`（必须为空目录，防陈旧 COMPLETE.json 被同步误判）下：`feature_provenance.json`（A0）、`audit_summary.json`（A1 `s1_gate`/层指标 + A2 `m1_gate`/层摘要）、`cloud_run_manifest.json`（hostname/gpu/git_sha/checkpoint 与 lineage sha、patient_set_hash、eligibility=`exploratory|restricted_no_train_lineage`、command、起止时间、exit_code、status）、`COMPLETE.json`（全部成功时）或 `error_report.json`（任一阶段失败时，fail-closed：失败阶段不得产出正式决策摘要）。

**处理过程**：
1. 拒绝复用非空输出目录；`load_full_config` + `resolve_runtime_profile`；`--fake-data` 时降级 config（关四个依赖云上 checkpoint 的模块）。
2. `SLMFBBDM.from_config` 构建模型，`torch.load(weights_only=True)` + lineage 校验 + `_select_checkpoint_state`（EMA shadow 合并进 model 态）→ strict（或 fake 宽松）加载 → `.eval()`。
3. A0：`audit_feature_provenance` 生成来源审计 JSON（补记 EMA 标记、训练 git sha、fixed_t 策略）。
4. 构建 loader：fake → `FakeDataset(16, 64)`；否则 `CachedDataset(cache_dir, split, augment=False)` + 可选 Subset 与 `val_batch_size`。
5. A1（`--skip-cka` 可跳过）与 A2（`--skip-emergence` 可跳过）分别 try/except：成功记入 results，失败记 error 并继续（不中断另一阶段）。
6. 汇总 `cloud_run_manifest`：数据模式、seed、完整命令行、checkpoint/config/lineage/split_manifest/patient_set 哈希与 eligibility。
7. 无错 → `audit_summary.json` + `COMPLETE.json`，返回 0；有错 → `error_report.json`，返回 1。

---

### `scripts/eval_feature_causality.py`

**职责**：Stage-A2 因果干预审计 CLI（c2 病灶干预矩阵），是 `src/mechanism_validation/feature_causality.py` 的命令行封装。对现有 checkpoint 运行 baseline、c2 lesion zero、background replace、nonlesion same-area、shifted-mask sham、CT inpainting 等干预，跨干预复用相同 per-seed 采样噪声；需要扩散采样（GPU），不得与主线训练共用 GPU（排队或用 `CUDA_VISIBLE_DEVICES`）。

**接口**：可执行脚本，`main() -> int`（退出码 0/1）；辅助 `_select_checkpoint_state`（与 emergence 版逻辑一致）、`_split_manifest_sha256`、`_patient_set_hash`、`_git_sha`。CLI 用法：

```bash
python scripts/eval_feature_causality.py \
    --config configs/experiments/slmf_png_xxx.yaml \
    --checkpoint results/.../ckpt_best_lesion.pt \
    --split val --output results/lesion_feature_emergence_v1/c2_causal/ \
    [--fake-data] [--max-samples N] [--num-seeds 4] [--num-steps N] \
    [--interventions baseline lesion_zero ...] [--device cuda:1]
```

参数透传关系：核心调用为 `run_causality_audit(model, loader, device, out_dir, seeds=tuple(range(args.num_seeds)), interventions=args.interventions, max_samples=args.max_samples, num_sampling_steps=args.num_steps)`；源模块的 `INTERVENTIONS`（干预名 → 实现映射）同时充当 `--interventions` 的 argparse `choices`（缺省 None = 全部干预）；`--fake-data` 的 config 降级逻辑与 `eval_feature_emergence.py` 完全相同（同样放宽 strict 加载）。

**依赖**：
- 内部: `src.data.dataset.CachedDataset / FakeDataset`；`src.data.lineage.load_checkpoint_data_lineage / validate_checkpoint_data_lineage`；`src.mechanism_validation.common.file_sha256 / write_json / canonical_json_sha256`；`src.mechanism_validation.feature_causality.INTERVENTIONS / run_causality_audit`；`src.mechanism_validation.feature_emergence._patient_keys`；`src.model.config_utils.load_full_config / resolve_runtime_profile`；`src.model.slmf_bbdm.SLMFBBDM`。
- 外部: torch、numpy、argparse、datetime、platform、socket、subprocess（git sha）。

**输入**：与 emergence 相同的 config/checkpoint/split/weights/fake-data 语义，另加：`--num-seeds 4`（采样种子取 0..n-1）、`--num-steps`（缺省沿用 config 的采样步数）、`--interventions`（`INTERVENTIONS.keys()` 的子集）、`--max-samples`。

**输出**：`--output`（必须为空）下：成功时 `causal_decision.json`（`run_causality_audit` 返回的决策结构）+ `cloud_run_manifest.json`（与 emergence 同构：哈希指纹、seeds、eligibility、status、errors）+ `COMPLETE.json`；失败时 `cloud_run_manifest.json` + `error_report.json`，退出码 1。

**处理过程**：
1. 拒绝非空输出目录；加载 config 并按 `--fake-data` 降级（关 lineage 与四个模块）。
2. 构建模型 + checkpoint lineage 校验 + EMA/raw 权重选择加载，`.eval()`。
3. 构建 loader（Fake 或 `CachedDataset(augment=False)`，`--max-samples` 截断为 Subset）。
4. 以 `seeds=(0..num_seeds-1)` 调用 `run_causality_audit`：同一 seed 下各干预使用相同采样噪声，产出因果决策结构；异常捕获进 errors。
5. 写 `cloud_run_manifest`（含 checkpoint/config/patient_set 哈希与 eligibility）。
6. 成功 → `causal_decision.json` + `COMPLETE.json`、exit 0；失败 → `error_report.json`、exit 1。

---

### 模块依赖小结

**本组 import 的内部模块**（grep 查证）：
- `src.data.dataset`：`build_dataloaders`（diag_router_grad）、`CachedDataset / FakeDataset`（suite、action axis、两个 feature CLI）。
- `src.data.lineage`：`load_checkpoint_data_lineage / validate_checkpoint_data_lineage`（除 eval_run_metrics、analyze_medoid 外的 6 个脚本统一做 checkpoint-数据血缘校验）。
- `src.model.config_utils`（`load_full_config / resolve_runtime_profile`）、`src.model.slmf_bbdm.SLMFBBDM`（5 个需建模型的脚本共用）。
- `src.model.frequency.haar.haar_dwt2`（action axis 的 SNR 审计）、`src.model.trainer._to_unit_interval`（suite 与 action axis 的归一化）。
- `src.mechanism_validation.common / feature_emergence / feature_provenance / feature_causality`：仅被两个 `eval_feature_*` CLI 封装调用（CLI 负责参数透传、lineage/EMA 选择、fail-closed manifest；分析本体在 src 侧，便于测试复用）。

**脚本间依赖（跨文件复用）**：
- `scripts/eval_router_background_suite.py` 是评估内核：`scripts/eval_oracle_destination.py` 从它 import 11 个符号（含 `evaluate_variant / _load_dataset / _paired_effect`）；`scripts/eval_action_axis_screen.py` 复用 `_body_nonlesion_mask / compute_body_background_metrics`；`scripts/validate_h2_pathology_excluded_residual.py`（不在本组）也复用其 `_load_dataset / evaluate_variant`。
- 三者再共同依赖 `scripts/evaluate.py` 的指标函数族（`compute_normalized_lesion_metrics`、`_select_checkpoint_state`、`_seed_evaluation` 等）——它是整个评估层的度量库。
- `scripts/analyze_medoid_zero_training.py` 不 import 任何 `src.` 模块，纯下游消费 `eval_action_axis_screen.py` 的工件（4 条 `predictions/B0_seed*.npz` + `cohort_tensors.npz` + `metrics.csv` + `aggregation_metrics.csv`，并验收其 `COMPLETE.json` 的零训练声明）。
- `scripts/eval_run_metrics.py`（纯 stdlib 离线 JSONL 对比）与 `scripts/diag_router_grad.py`（唯一直接调 `build_dataloaders` 并做 backward 的脚本）相互独立。

**被消费方**：`tests/test_eval_router_background_suite.py`、`tests/test_prior_anchored_router_smoke.py`、`tests/test_oracle_destination.py`、`tests/test_action_axis_screen.py`、`tests/test_medoid_zero_training.py`、`tests/test_feature_cli_integration.py`（同时测两个 feature CLI）、`tests/test_feature_causality.py`、`tests/test_feature_emergence.py`、`tests/test_feature_provenance.py` 直接测试本组脚本与底层 src 模块；`configs/cloud_worktree_sync.yaml` 将全部 8 个脚本列入云端 worktree 同步清单；`configs/experiments/slmf_png_prior_anchored_router_{100e,300e}.yaml` 的注释引用 diag_router_grad 用法。`artifacts/` 下的若干快照目录是这些脚本的历史副本。

**重要跨模块发现**：路由器干预契约 `set_inference_destination_intervention / set_inference_frequency_intervention / set_inference_route_action_intervention / set_inference_frequency_gain / set_inference_frequency_window`（挂在 `model.residual_preconditioner` 上）是 suite、oracle、action axis 三个脚本的共同前提——脚本启动即检查 API 存在并 fail-closed；训练指标键 `frequency/prior_anchor_{level}_{band}_conditional_shallow`（training_metrics.jsonl 的 train 块）被 suite 用于 fixed 目的地回读、被 eval_run_metrics 用于 shallow 存活审计，是训练侧与评估侧的隐式键名契约。


---

## 12. 诊断/云协作/预训练杂项脚本

### 模块概述

本组 17 个脚本（全部存在，无缺失）是 SLMF-BBDM 仓库的"外围工具层"，可分为四类：① 模型诊断（gabor/adapter 注入可视化、条纹溯源、损失项量级、PNG 缓存配准校验）；② 数据机理分析（CT 频带→病灶可预测性 probe、CT/PET 频率-几何关联统计）；③ 预训练权重生产者（M0 条件均值 mean_best.pt、PET 编码器 encoder_best.pt、TinySegmenter segmenter.pt，均带 lineage sidecar）；④ 云协作流水线（本地/云端 worktree 分发同步、双 worktree 实验队列、run 汇总排行榜、ToDesk 部署 zip、perceptual-x0 五臂消融运行器与少步验证器）。设计模式上高度一致：fail-closed（宁可 BLOCKED 也不伪造可运行状态）、SHA256 溯源 + git provenance 记录、dry-run 默认 / --apply 显式写入、只读诊断绝不改训练产物。数据流为：本地主 checkout（代码唯一正本）→ allowlist 分发到云端两个 worktree（.worktrees/spectral-router-v5 与 .worktrees/lesion-feature-emergence）→ 云端 CUDA 训练产出 run 目录（state.json / training_metrics.jsonl / checkpoint）→ 只读扫描器汇总成 leaderboard 供本地决策。预训练权重的消费链已用 grep 在 configs/ 下逐一查证（见各条目与模块依赖小结）。

---

### `scripts/diagnose_gabor_adapter.py`

**职责**：只读诊断脚本。加载训练好的 SLMF-BBDM checkpoint，对若干带掩膜验证样本采样，检查 Gabor / Hotspot / ZeroConv Adapter 通路是否携带条纹状伪影。

**接口**：无顶层类；函数：`_load_checkpoint(path, map_location="cpu")`、`_select_state_dict(ckpt, weights)`、`_move_batch`、`_to_numpy`、`_show`、`_text_cell`、`_resize_norm_map`、`_adapter_outputs_by_tau(model, condition, tau_values)`、`_sample_indices(ds, limit, sample_ids)`、`main() -> int`。CLI 用法：`python scripts/diagnose_gabor_adapter.py --config configs/experiments/slmf_full.yaml --checkpoint <ckpt.pt> --weights raw|ema --split val --out-dir results/diag_gabor_adapter --num-samples 4 --sample-id <id> --num-steps N --tau 0 --tau 500 --tau 900 --device auto`。

**依赖**：
- 内部：`src.data.dataset.CachedDataset`；`src.model.config_utils.load_full_config / resolve_runtime_profile`；`src.model.slmf_bbdm.SLMFBBDM`
- 外部：`torch`、`matplotlib`(Agg)、`csv`

**输入**：实验 YAML（默认 slmf_full.yaml）、checkpoint 路径（`ckpt["model"]` 原始权重或 ema 键）、τ 值列表、样本 id 或前 N 个带 mask 的 val 样本（batch 张量 ct/pet/mask，1×1×H×W）。

**输出**：`<out-dir>/diag_XX_<sample_id>_pid<pid>.png`（4×5 面板：CT/Target/Pred/Mask/HotspotPrior、Gabor energy、各 τ 的 L0-L3 Adapter 注入范数图）+ `diagnostics_summary.csv`（pred/target/gabor/hotspot 统计与 adapter_L{level}_T{tau}_mean/max 列）。

**处理过程**：
1. 解析配置并 resolve_runtime_profile，按 --device 选 cuda/cpu；
2. `SLMFBBDM.from_config` 建模，`load_state_dict(state, strict=False)` 加载 raw/ema 权重并告警缺失键；
3. `CachedDataset` 取 val split 中带 mask 的样本（或指定 sample-id），DataLoader batch=1；
4. `model.build_condition_bundle` 构建条件束，`model.sample` 生成 synthetic_pet；
5. `_adapter_outputs_by_tau` 对每个 τ 调 `model.adapter.get_zero_conv_outputs`（依赖 ct_feat_0..3 / organ_feat / gabor_feat / hotspot_prior）；
6. 渲染 matplotlib 面板并逐样本存 png；
7. 汇总每样本统计行写 diagnostics_summary.csv。

---

### `scripts/validate_png_alignment.py`

**职责**：训练前置的 PNG→NPZ 缓存内部一致性/配准体检：逐样本核对形状、mask 非空、mask 内外 PET 亮度关系、PET 峰值-质心距离、CT-PET 左右/上下翻转配准异常。

**接口**：`_ncc(a, b) -> float`（归一化互相关）、`_flip_anomaly(ct, pet) -> Dict`（base/lr/ud NCC 与 delta）、`_save_overlay(out_path, ct, pet, mask, sid)`、`check_one(npz_path) -> Optional[Dict]`、`run(config_path, output_dir, max_samples, seed, overlay_n) -> Dict`、`main() -> int`。CLI：`python scripts/validate_png_alignment.py --config configs/experiments/slmf_png_baseline.yaml --output-dir results/alignment --max-samples 100 --seed 42 --overlay-n 12`。

**依赖**：
- 内部：`src.model.config_utils.load_full_config / resolve_runtime_profile`
- 外部：`numpy`、`matplotlib`、`csv`、`json`、`random`

**输入**：实验 YAML 中的 `data.cache_dir`（.npz 缓存目录，每个文件含 ct/pet/mask 数组，[1,H,W] 或 [H,W]，float，值域约 [-1,1]）。

**输出**：`alignment_report.csv`（逐样本行）、`alignment_summary.json`（聚合比例 + risks 列表 + ok 标志）、`overlays/<sample_id>.png`（CT/PET/mask/CT+mask 四联图）。

**处理过程**：
1. 读配置定位 cache_dir，随机抽样 max_samples 个 npz；
2. `check_one` 逐样本：校验三数组形状一致、mask>0.5 面积；
3. 计算 mask 内/外 PET mean/max、`in_gt_out_mean/max` 布尔、PET 峰值到 mask 质心的欧氏距离；
4. `_flip_anomaly` 用 NCC 比较 CT 原图 vs 左右/上下翻转与 PET 的相关性，delta>0.1 视为异常；
5. 按抽样索引渲染 overlay png；
6. 聚合风险规则（>50% 空 mask、in>out 比例<0.7、翻转异常>30%、峰值距离中位数>15px）写 summary 并打印。

---

### `scripts/diagnose_stripes.py`

**职责**：条纹溯源实验：同一 checkpoint / 同一批样本 / 同一种子下，按 5 种 Gabor 路由预设（A_gabor_off 参照、B adapter_only、C noise_only、D hotspot_only、E current）分别采样，量化哪种路由是条纹伪影的主导来源。

**接口**：常量 `ROUTING_PRESETS: Dict[str, Dict[str, bool]]`；`_stripe_score(pred) -> float`（Sobel 8 方向能量各向异性）、`_high_freq_ratio(pred)`（FFT 顶部 25% 频段能量占比）、`_lesion_metrics(pred, target, mask) -> Dict`、`_corr`、`_set_seed`、`build_model_with_routes(base_config, preset_name) -> Tuple[SLMFBBDM, Dict]`、`load_weights(model, ckpt_path, device) -> str`、`_save_panel`、`diagnose(...) -> Dict`、`main() -> int`。CLI：`python scripts/diagnose_stripes.py --config <yaml> --checkpoint <ckpt> --split val --sample-ids ID... --seed 42 --output-dir results/diag_stripes --num-steps N --device cuda --fake-data`。

**依赖**：
- 内部：`src.model.config_utils.load_full_config / resolve_runtime_profile / apply_dotlist_overrides`；`src.model.slmf_bbdm.SLMFBBDM`；`src.data.dataset.CachedDataset / FakeDataset`（延迟导入）
- 外部：`numpy`、`torch`、`scipy.ndimage.correlate`、`matplotlib`

**输入**：基础实验 YAML、checkpoint（strict=False 容错加载）、样本 id 列表（默认取前 4 个）、种子、可选 fake-data 冒烟模式。

**输出**：`<output-dir>/<preset>/<preset>_s<i>.png` 六联面板（CT/Target/Pred/Mask/Gabor energy/Hotspot）、`stripe_metrics.csv`（每 preset×样本的 stripe_score/high_freq_ratio/病灶误差/路由开关）、`gabor_filters.json`（E_current 下 Gabor 滤波器 frequency/theta/sigma/gamma/phase 参数转储）、`stripe_summary.json`（各 preset 平均 stripe 分与 dominant_route 结论，阈值 Δ>0.02）。

**处理过程**：
1. 加载并 resolve 配置（fake_data 时切 FakeDataset）；
2. 遍历 DataLoader 挑选指定 sample-id（或默认 4 个）样本并缓存；
3. 逐 preset：`apply_dotlist_overrides` 改写 modules.gabor.*（及 scale_adaptive_noise / hotspot_prior 联动）后 `SLMFBBDM.from_config` 重建模型；
4. strict=False 加载同一 checkpoint，eval 模式；
5. 每样本重置种子后 `build_condition_bundle` + `model.sample`，取 gabor_energy/gabor_feat/hotspot_prior；
6. 计算 stripe_score、high_freq_ratio、病灶指标、gabor-pred 相关性，写行并渲染面板；
7. 以 A_gabor_off 为参照计算 Δ，输出 dominant_route 结论 JSON。

---

### `scripts/diagnose_loss.py`

**职责**：33 行的临时诊断片段：在真实验证样本（带 mask）上打印各损失分项量级，对比"随机 τ（真实训练分布）"与"τ=0（最终去噪步，病灶 loss 最激活）"两种情形。

**接口**：仅 `run(label, **kw)`（调用 `model(batch, **kw)` 返回 loss/logs 并打印标量项）。无可执行 CLI 参数——`CKPT = "checkpoints/slmf_bbdm_full/ckpt_epoch0600.pt"`、`CONFIG = "configs/experiments/slmf_full.yaml"`、样本数 8、batch 4 均硬编码，需改文件后 `python scripts/diagnose_loss.py`。

**依赖**：
- 内部：`src.model.config_utils.load_full_config / resolve_runtime_profile`；`src.model.slmf_bbdm.SLMFBBDM`；`src.data.dataset.CachedDataset`
- 外部：`torch`、`torch.utils.data.DataLoader / Subset`

**输入**：硬编码的 checkpoint 与实验 YAML；CachedDataset val split 中 `entry.has_mask` 的前 8 条（batch dict 含 ct/pet/mask 等 tensor）。

**输出**：仅 stdout——两组各约数十行 `(loss_key, value)` 与 total 对比；无文件产物。

**处理过程**：
1. 加载配置与 checkpoint（weights_only），model.eval()；
2. 取前 8 个带 mask 的 val 样本组成 batch=4（实际只取一个 batch）；
3. `run("随机τ")`：固定种子后前向，打印 total 与全部标量 log 项；
4. `run("τ=0")`：显式传 `timesteps=torch.zeros(4)` 再前向打印，观察病灶相关损失项在最终去噪步的放大。

---

### `scripts/analyze_ct_frequency_lesion_predictability.py`

**职责**：患者级轻量 probe：检验"CT 频带特征能否定位标注的 PET 病灶"。不对验证患者调参；训练患者中确定性划出 20% 做阈值校准，main_data/split_manifest.csv 为权威 split（Data/data 的 train/val 物理目录仅当文件仓库用）。

**接口**：`@dataclass Sample(sample_id, patient_id, split, ct, mask, body)`；`_index_pngs`、`_read_ct/_read_mask/_read_pet_uptake`（PET 反相：白底黑摄取）、`_largest_body`（最大连通体+填洞+膨胀）、`_haar2`（二级 Haar 小波）、`_spectral_maps(ct) -> dict`（raw/ll2/l1·l2 的 lh/hl/hh 及 energy）、`_coords`、`_feature_array`、`_safe_auc/_safe_ap`、`_ring`（mask 外 12px 环）、`_sample_training_indices`（正像素≤160 + 环内/远端负样本 3:1）、`_fit_probe`（HistGradientBoostingClassifier，类权重平衡）、`_predict_probe`、`_calibrate_threshold`（校准集上按 Dice 搜分位阈值）、`_evaluate`（全图/环内 AUC·AP、Dice、small/medium/large 分层）、`_band_energy_diagnostics`、`main() -> int`。CLI：`python scripts/analyze_ct_frequency_lesion_predictability.py --root <repo> --size 192 --seed 42 --output results/ct_frequency_lesion_probe`。

**依赖**：
- 内部：无（不 import src.*，纯独立分析）
- 外部：`numpy`、`PIL`、`scipy.ndimage`、`sklearn`(HistGradientBoostingClassifier, roc_auc_score, average_precision_score)、`csv`

**输入**：`Data/data/{train,val,test}/{ct,pet,label}/*.png` 与 `main_data/split_manifest.csv`；图像统一 resize 到 --size。

**输出**：`results/ct_frequency_lesion_probe/probe_metrics.csv`（atlas_no_ct 基线 + coords/raw_ct/local_patch/ll2/l2_details/l1_details/all_frequency/coords_all_frequency 各 probe 的 AP/ring_AUC/Dice）、`band_energy_metrics.csv`、`probe_metrics_by_size.csv`（病灶大小三分层）、`summary.json`（含 manifest、image_size、病灶面积分位等）。

**处理过程**：
1. 索引三种模态 PNG，校验 manifest 覆盖无缺；
2. 逐行读 CT/mask，构建 body 前景，val 行额外读 PET 计算环内 PET-AUC；
3. 训练患者随机（定种子）划 20% 为校准患者，其余为拟合集；
4. 拟合 no-CT 解剖图谱基线（拟合集 mask 均值 + 高斯平滑）并校准/评估；
5. 逐特征种类拟合 HistGradientBoosting 像素级 probe（采样正负像素）；
6. 校准集搜最优 Dice 阈值，val 集评估并按 mask 面积三分位分层；
7. 频带能量诊断（病灶/环内能量比等）；
8. 写四份输出文件并打印摘要。

---

### `scripts/analyze_ct_pet_frequency_geometry.py`

**职责**：对成对 PNG 数据集做 CT/PET 频率-几何关联统计：以 CT 体部前景与 PET 高摄取连通域作为"器官代理"（明确不命名器官），量化 Haar 频带能量、径向 FFT 频段占比、摄取组件几何及其患者级相关（含 BH-FDR 校正）。

**接口**：常量 `BANDS`（ll2 + 两级 lh/hl/hh 共 7 频带）；`_index_pngs`、`_read_image`(可反相)、`_read_mask`、`_largest_body`、`_haar2`、`_frequency_energy_maps`、`_centroid/_distance/_weighted_spread/_ring/_safe_mean`、`_safe_corr`(pearson/spearman)、`_spatial_correlations`、`_radial_fft(image, body, bins=18)`、`_fft_band_proportions`(low/mid/high)、`_cosine/_jensen_shannon_distance`、`_uptake_components`(Otsu 式阈值+连通域 top5)、`_pairwise_centroid_stats`、`_balanced_sample`(每患者≤2 切片)、`_bh_fdr`、`_association_rows`(预声明相关对)、`main() -> int`。CLI：`python scripts/analyze_ct_pet_frequency_geometry.py --root <repo> --size 192 --slices-per-patient 2 --seed 42 --output results/ct_pet_frequency_geometry`。

**依赖**：
- 内部：无（不 import src.*）
- 外部：`numpy`、`pandas`、`PIL`、`scipy.ndimage`、`scipy.stats`(pearsonr/spearmanr)

**输入**：同上一脚本的 PNG 三模态目录 + split_manifest.csv；患者平衡抽样。

**输出**：`results/ct_pet_frequency_geometry/` 下 `sample_metrics.csv`（每切片宽表：体部面积/病灶等效直径/摄取面积/FFT cosine·JS/各频带能量等）、`component_metrics.csv`（CT 体部与 PET 摄取组件几何）、`patient_metrics.csv`、`band_summary.csv`（患者级各 Haar 频带汇总）、`association_summary.csv`（预声明变量对 + spearman_fdr_bh）、`summary.json`。

**处理过程**：
1. 索引 PNG + 读 manifest，患者平衡抽样；
2. 每切片：读 CT/PET（PET 反相）/mask，构建 body 与病灶环；
3. 计算 CT/PET 各自 Haar 频带能量图与径向 FFT 剖面（18 bins），得出 low/mid/high 占比与 CT-PET 剖面 cosine/JS 距离；
4. `_uptake_components` 提取 PET 高摄取连通域 top5 与 CT 体部组件，记录几何；
5. 汇总到切片/患者两级 DataFrame；
6. `_association_rows` 对约 50 个预声明变量对计算 Pearson+Spearman 并做 BH-FDR；
7. 写 6 份输出并打印 summary.json。

---

### `scripts/pretrain_conditional_mean.py`

**职责**：M0 预训练：训练 CT 条件化的 LL2 低频 PET 均值预测器（`LowFrequencyPETPredictor`），供残差 BBDM 变体使用；由 `MeanPretrainer` 完成训练循环并按验证最优产出 `mean_best.pt`。

**接口**：`_set_seed(seed)`、`main()`。CLI：`python scripts/pretrain_conditional_mean.py --config <base.yaml> --output-dir <M0目录> --epochs 30 --seed N --override key=value（可重复）`。

**依赖**：
- 内部：`src.data.dataset.build_dataloaders`；`src.data.lineage.load_checkpoint_data_lineage`；`src.model.config_utils`（load_full_config / resolve_runtime_profile / validate_png_baseline_config / log_startup_status / save_resolved_config）；`src.model.mean_predictor.LowFrequencyPETPredictor`；`src.model.mean_pretraining.MeanPretrainer`
- 外部：`numpy`、`torch`

**输入**：基础实验 YAML（要求 `modules.conditional_mean.enabled=true`，否则 ValueError；结构参数 in_channels/base_channels/levels 取自该节）；数据配置驱动 train/val loader（val 必须存在，用于 mean_best 选择）。

**输出**：`<output-dir>/mean_best.pt`（最优条件均值 checkpoint）+ `save_resolved_config` 写出的解析后配置。**消费方（grep 已验证）**：主训练配置通过 `modules.conditional_mean.checkpoint` 加载——如 `configs/experiments/slmf_png_spectral_router_v5.yaml`、`slmf_png_boundary_reliable_v4.yaml` 及 spectral_router_ablation_plan_v5 / cold_bias_v7 / final_calibration_v8 / boundary_reliable_ablation_plan_v3·v4 / frequency_ablation_plan_v2 等 plan YAML 均指向 `checkpoints/freq_mean_pretrain_v2/mean_best.pt`；`slmf_png_prior_anchored_router_100e/300e.yaml`、`slmf_png_fullstack_perceptual_p2_300e.yaml`、`slmf_png_background_tail_repair_e100.yaml` 指向病理排除版 `checkpoints/freq_mean_excluded_v1/mean_best.pt`。

**处理过程**：
1. 加载并 resolve 配置，--override 逐条覆盖，确定种子；
2. `validate_png_baseline_config` + `log_startup_status` 启动体检；
3. `load_checkpoint_data_lineage` 校验缓存血缘（cache_metadata_sha256）；
4. 校验 `modules.conditional_mean.enabled` 并实例化 `LowFrequencyPETPredictor`；
5. `build_dataloaders` 构建训练/验证 loader（无 val 则报错）；
6. `save_resolved_config` 落盘解析配置，`MeanPretrainer(...).run(epochs, output_dir)` 训练；
7. 打印最优 checkpoint 路径 `mean_best.pt`。

---

### `scripts/audit_checkpoint_lineage.py`

**职责**：对已封印（sealed）的 Stage 0C 缓存做"快速 checkpoint 血脉盘点"：不重哈希原始 PNG、不逐张比较缓存张量，仅验证封印血缘自哈希与数据契约，再用假张量（metadata_only）检查 checkpoint 元数据。原始数据/缓存可能变动时应改跑完整 Stage 0C gate（`scripts/run_cloud_stage0c_gate.py`）。

**接口**：`_write_json`、`_write_csv`、`_resolve`、`run(args) -> dict`、`_parse_args`、`main(argv) -> int`（PASS→0，FAIL→2，Ctrl-C→130）。CLI：`python scripts/audit_checkpoint_lineage.py --root <repo> --cache-dir cache/tensors_main --contract configs/dataset_contract_stage0a_v1.json --checkpoint <path>（可重复，必填）--output results/mechanism_validation/00C_checkpoint_inventory_legacy --hash-checkpoints`。

**依赖**：
- 内部：`scripts.run_cloud_stage0c_gate._validate_checkpoints`；`src.data.lineage`（load_checkpoint_data_lineage, CacheLineageError）
- 外部：`argparse`、`csv`、`json`、`platform`

**输入**：封印缓存目录（内含 cache_lineage.json）、数据集契约 JSON、一个或多个 checkpoint 文件/目录。

**输出**：`<output>/decision.json`（stage=00C_checkpoint_lineage_inventory、decision PASS/FAIL、六项血缘指纹 manifest_semantic/raw_png_combined/preprocessing_config/dataset_contract/cache_payload/cache_metadata 的 sha256、stop_rule）、`checkpoint_lineage.csv`、`execution_metadata.json`；decision 同步打印到 stdout。

**处理过程**：
1. 组装最小 lineage 配置并调 `load_checkpoint_data_lineage`；抛 CacheLineageError 时直接写 FAIL decision 并返回（要求改跑完整 gate）；
2. 打印封印缓存元数据 sha256；
3. `_validate_checkpoints(..., metadata_only=True, hash_checkpoints=args.hash_checkpoints)` 用假张量检查各 checkpoint 元数据（可选对文件做 SHA-256）；
4. 写 checkpoint_lineage.csv；
5. 组装 decision（training_allowed=True、formal_checkpoint_claims_allowed=PASS 时为真、model_mechanism_claims_allowed 恒 False）；
6. 写 decision.json 与 execution_metadata.json 并返回退出码。

---

### `scripts/summarize_cloud_runs.py`

**职责**：云端 results/ 下跨 run 的只读扫描器：递归发现 run 目录（含 state.json 或 training_metrics.jsonl），通用解析指标（total_epochs 由 state/config/实测推断，不硬编码），逐 run 评级并生成排行榜 + 晋升候选。设计边界：只在云端真实 run 上执行（本地 results/ 只有 FAILED 桩）；不构成完整性契约或深度审计（分别另用 summarize_prior_anchored_run.py / audit_prior_anchored_run.py）。

**接口**：`discover_runs(runs_root, *, filter_glob) -> list[Path]`、`read_metrics_generic`、`_infer_total_epochs`、`_infer_experiment`、`_final_val_metrics`、`_router_state`、`_alpha_stripe_from_config`、`classify_run(run_dir, *, root, with_checkpoint, with_sha) -> dict`（永不抛异常）、`_inspect_checkpoint`、`grade(summary) -> (label, reasons)`（COMPLETE_STABLE / COMPLETE_ANOMALY / ROUTER_NOOP / INCOMPLETE / FAILED / EMPTY）、`sort_summaries`、`write_leaderboard_csv`、`build_promoted`、`render_terminal_table`、`grade_histogram`、`parse_args`、`main(argv) -> int`、`_render_readme`。CLI：`pixi run python scripts/summarize_cloud_runs.py --runs-root results --filter "prior_anchored*" --sort-by best_combined --promote-top 5 --top N --out results/cloud_runs_summary --with-checkpoint --with-sha --no-write`。

**依赖**：
- 内部：`scripts.audit_prior_anchored_run` 的纯函数（best_validation_epoch / combined_score / detect_anomalies / flatten_record / fmt / is_number / read_json / read_yaml / repo_relative / sha256_file / utc_now）
- 外部：`yaml`（必需）、`csv`、`argparse`

**输入**：`results/**` 下的 run 目录（state.json / resolved_config.yaml / training_metrics.jsonl，checkpoint 可选检查）；FINAL_VAL_KEYS（val/mae、ssim、psnr、stripe_score、lesion_* 等）与 ROUTER_KEYS（frequency/route_* 与 prior_anchor_*）。

**输出**：`<out>/runs.json`（pipeline_id=CLOUD_RUNS_SUMMARY_V1 全量 summary）、`promoted.json`（top-N 晋升候选 + complete_but_router_noop 名单 + "非因果授权"声明）、`leaderboard.csv`、`README.md`（自描述），以及终端直方图/排行榜；`--no-write` 时仅打印。

**处理过程**：
1. `discover_runs` 按 marker 文件递归发现 run 目录，filter 对 repo 相对 POSIX 路径 fnmatch；
2. `classify_run` 读 state/config/metrics（metrics 路径先取 state/config 声明再回退 run 本地）；
3. 推断 total_epochs 后重读指标，提取末轮 val、router 状态（shallow_mass / active_delta，双阈值 1e-3 判 no-op）、best epoch、异常计数；
4. 可选检查 checkpoint（torch.load weights_only）与 sha256；
5. `grade`：空/失败 → EMPTY/FAILED；完整性闸（无坏行/非有限值/重复 epoch、step 严格递增、epoch 齐全）不过 → INCOMPLETE（叠加 router no-op 改写）；完整且无异常 → COMPLETE_STABLE，有异常 → COMPLETE_ANOMALY；
6. 排序 + 直方图，`build_promoted` 只在"完整且非 no-op"里取 top-N；
7. 写 runs.json / promoted.json / leaderboard.csv / README.md 并打印建议（no-op run 不得进入配对消融）。

---

### `scripts/sync_cloud_worktrees.py`

**职责**：单次上传、manifest 驱动的源码分发：把唯一正本源码按显式 allowlist 分发进云端两个 worktree（"物理共存、逻辑隔离"——数据/缓存/checkpoint 只读共享，代码与输出严格隔离）。默认 dry-run，仅 `--apply` 写入；路径加固（防 ../ 与符号链接逃逸）、覆盖前备份、SHA256 前后校验、冲突检测、原子写、活动实验锁阻塞。

**接口**：`SyncError`；`SyncConfig`（roots: source/router/feature/manifests/backups；groups: common/router_only/feature_only；exclude 正则；`target_root(target)`）；`load_config`；`FileEntry(rel, group)`；`discover_files`、`_walk_source`、`_entry_targets`、`find_active_locks`、`_lock_summary`、`_atomic_copy`、`_classify`、`_check_drift`；`SyncResult`（copied/skipped/conflicts/errors + ok）；`sync_once(config, target, *, apply, ...) -> (SyncResult, entries)`；`load_state/save_state`；`main(argv) -> int`；`_run_verify`。CLI：`python scripts/sync_cloud_worktrees.py --config configs/cloud_worktree_sync.yaml --source <源根> --target router|feature|all (--dry-run | --apply | --verify) --run-id <id>`。

**依赖**：
- 内部：`src.mechanism_validation.common.file_sha256 / write_json`
- 外部：`yaml`、`shutil`、`subprocess`(git 状态)、`re`

**输入**：`configs/cloud_worktree_sync.yaml`。**本地↔云端目录对应关系（grep 已验证）**：`roots.source: "."` = 本地主 checkout（云端则是上传载荷解包处）→ 分发目标 `roots.router: ".worktrees/spectral-router-v5"`（谱路由主线训练）与 `roots.feature: ".worktrees/lesion-feature-emergence"`（病灶特征可解释性分支）；manifests 写 `cloud_sync/manifests/<id>.json` + 状态 `last_sync_state.json`，备份写 `cloud_sync/backups/<id>/<target>/`。分组：common（src/data、src/model 主体、通用 scripts、实验 YAML、docs）双发；router_only（src/model/frequency/**、run_spectral_router_v5.py、diagnose_stripes.py 等）；feature_only（src/mechanism_validation/feature_*.py、eval_feature_*.py）。exclude 正则硬性保护 .git/data/cache/checkpoints/results/cloud_runs/cloud_sync 等树。

**输出**：--apply 时目标 worktree 内的被覆盖文件 + `cloud_sync/manifests/<run_id>.json` 与更新后的 state；--dry-run 仅报告将要发生的拷贝；--verify 只读校验不一致时退出非 0。

**处理过程**：
1. 载入 YAML 配置并校验 roots/groups 完整性；
2. `discover_files` 遍历源根，按 allowlist（字面量+glob）与 exclude 正则筛出 FileEntry，并标注目标（common→双发，router_only/feature_only→单发）；
3. `sync_once` 逐文件：`_resolve_within` 词法+真实双重解析防逃逸，符号链接检查，源 SHA256；
4. `_check_drift` 对比上次同步 state：目标被改动且≠源 → conflict 拒绝覆盖；
5. `_classify` 判断 identical→skip / 需拷贝；
6. apply 时：旧文件备份 → `_atomic_copy`（临时文件 + os.replace）→ 复验 SHA256 记录；
7. 写 per-run manifest JSON 与 state；`--apply` 前检查所有 run.lock，存在 running 锁则拒绝（退出码 2）。

---

### `scripts/run_dual_experiment_queue.py`

**职责**：单 GPU 串行双 worktree 实验队列：对同一冻结 checkpoint 依序执行 A. router 主线作业（--router-command）→ B. 特征涌现分析（A0+A1+A2）→ C. 特征因果分析（M2，仅当 B 的 A1 s1 / A2 m1 两个 gate 都通过）。启动器级 fail-closed：checkpoint 稳定性采样、输出目录防重用、RunLock、假数据降级为 exploratory、五条正式资格前置检查。

**接口**：`QueueError`；`_git_sha/_git_state`；`_split_command`、`_has_flag`、`_interventions_in_command`；`_snapshot_checkpoint`、`checkpoint_stable(path, *, interval, samples)`（size+mtime+sha256 三点采样）；`verify_analysis_code(feature_worktree) -> dict`（只读检查 feature_causality 源码含真实 margin 常量与 body-mask 控制代码路径）；`_controls_present_in_command`；`compute_eligibility(cmd, *, split, has_frozen_quartiles, analysis_checks)`；`RunLock`（acquire/release/is_running，写 <worktree>/run.lock）；`_env_summary`；`ExperimentSpec(name, command, worktree_root, run_dir, device)`；`run_experiment(spec, *, checkpoint_sha256, formal) -> dict`；`read_emergence_gates`（读 cross_modal_similarity.json 的 s1_gate 与 lesion_emergence.json 的 m1_gate，缺失=None fail-closed）；`emergence_passed`；`_substitute`（{checkpoint}/{config}/{device}/{router_out}/{feature_out}/{emergence_out}/{causality_out}/{frozen} 占位符）；`_build_emergence_command/_build_causality_command`（自动拼 scripts/eval_feature_emergence.py / eval_feature_causality.py）；`main(argv) -> int`；`_write_queue_manifest`。CLI：`python scripts/run_dual_experiment_queue.py --router-worktree .worktrees/spectral-router-v5 --feature-worktree .worktrees/lesion-feature-emergence --checkpoint <frozen.pt> --config <yaml> --split val --frozen-quartiles-json <json> --output-root cloud_runs --run-id <id> --device cuda --device-feature cuda --router-command "..." --feature-emergence-command "..." --feature-causality-command "..." --skip-router --skip-causality --dry-run --parallel --check-interval 2 --check-samples 3`。

**依赖**：
- 内部：`src.mechanism_validation.common.file_sha256 / write_json`
- 外部：`shlex`、`subprocess`、`socket`、`uuid`

**输入**：冻结 checkpoint 路径（拒绝 last.pt——训练中产物）；两个 worktree 根；显式命令模板或 --config 自动构建；分位数冻结 JSON（正式资格必需）。**目录对应**：输出写 `<output-root>/router/<run_id>` 与 `<output-root>/interpretability/<run_id>`（默认 cloud_runs/，与 sync 脚本的 exclude 保护树一致）。

**输出**：每实验 `<name>.stdout.log / .stderr.log / .manifest.json`（命令、returncode、checkpoint_sha256、formal_eligibility、环境摘要）；队列级 `queue_manifest.json` 与 `COMPLETE.json`（run_id/status/data_mode/eligibility/checkpoint_sha256/git_sha_feature）。

**处理过程**：
1. 预检：checkpoint 存在且非 last.pt、输出目录为空、--parallel 在同设备时直接拒绝（串行为唯一实现）；
2. 解析/构建三个命令并做占位符替换；
3. `verify_analysis_code` 只读检查 feature worktree 的分析代码结构；`compute_eligibility` 判定 formal/exploratory；
4. `checkpoint_stable` 多点采样确认冻结（sha 记录入 summary）；
5. router 作业持锁执行，失败即阻断下游；成功后复检 checkpoint 稳定性；
6. emergence 作业持锁执行，读 gate JSON；
7. gate 双通过才执行 causality，否则记 skipped（INCOMPLETE）；
8. 汇总写 queue_manifest.json + COMPLETE.json，退出码反映状态。

---

### `scripts/audit_magnification_consistency.py`

**职责**：小病灶表征稀释机理的冻结 full/zoom 审计 CLI：消费已完成 medoid 确认 run 导出的不可变队列（cohort），绝不训练或更新 checkpoint。被 `tests/test_magnification_audit_cli.py` 消费。

**接口**：`_parse_int_csv`（argparse 类型：逗号分隔正整数且去重）、`_sha256`、`_read_json`、`load_source_cohort(source_run) -> (CohortArrays, manifest)`（保留零填充 patient 身份）、`_validate_crop_plan`、`_validate_frozen_input_hashes`、`_git_provenance`、`_resolve_device`、`_source_weights`、`_load_frozen_model`、`_manifest`、`_write_dry_run`、`build_parser`、`main(argv) -> int`。CLI：`python scripts/audit_magnification_consistency.py --config <yaml> --checkpoint <pt> --source-run <medoid run 目录> --output <目录> (--dry-run | --execute) --seeds 42,43,44,45 --crop-sizes 96,48 --primary-crop-size 48 --input-size 192 --ddim-steps 50 --batch-size 4 --bootstrap-seed 20260729 --bootstrap-replicates 10000 --weights source|model|raw|ema --device auto --amp`。

**依赖**：
- 内部：`scripts.evaluate._select_checkpoint_state`；`src.mechanism_validation.magnification.lesion_crop_box`；`src.mechanism_validation.magnification_audit`（CohortArrays, build_primary_gate, compute_gate_statistics, run_frozen_magnification_audit, validate_cohort, write_audit_artifacts）；`src.model.config_utils.load_full_config / resolve_runtime_profile`；`src.model.slmf_bbdm.SLMFBBDM`
- 外部：`numpy`、`pandas`、`torch`、`subprocess`(git)

**输入**：实验 YAML、冻结 checkpoint、source run 目录（内含 run_manifest.json / cohort.csv / cohort_tensors.npz）；审计超参（种子×裁剪尺寸网格、DDIM 步数、bootstrap 重采样）。

**输出**：--dry-run 时校验输入并只写 provenance manifest；--execute 时 `run_frozen_magnification_audit` 的完整审计产物（由 `write_audit_artifacts` 写出，含 gate 统计与 bootstrap 置信区间）；输出目录已存在则报错（防覆盖）。

**处理过程**：
1. 解析并解析绝对路径，校验 config/checkpoint 存在、output 不存在、primary_crop_size ∈ crop_sizes、数值参数为正；
2. `load_source_cohort` 载入冻结队列；
3. `_validate_crop_plan` 确认裁剪尺寸相对输入/病灶可行；
4. `_validate_frozen_input_hashes` 复核 config/checkpoint/manifest/cohort 的 sha256 与源 manifest 一致；
5. 记录 git provenance 与设备/权重来源；
6. dry-run 分支写 manifest 即返回；execute 分支 `_load_frozen_model` 装载冻结 SLMFBBDM 后执行多种子 × 多裁剪尺寸的 full/zoom 审计并写产物。

---

### `scripts/pretrain_tiny_segmenter.py`

**职责**：预训练 P1_SEG_OUT 臂消费的冻结 Stage-2 `TinySegmenter`：仅在 mechanism_train 患者上两阶段训练、在锁定 calibration 患者上校准召回 gate，从未见过 held-out 验证患者；双 gate（总体与小病灶召回 ≥0.70）都通过才写正典 `segmenter.pt`（纯 state_dict，因 SLMFBBDM 用 `TinySegmenter.load_state_dict` 装载）。

**接口**：`SegmenterPretrainError(PretrainError)`；`sha256_file`；`encoder_roles_from_partition(partition) -> dict`（仅保留 mechanism_train→train 与 calibration 两角色）；`prepare_records(samples, *, image_size, include_stage2_target)`（一次性载入 NPZ 并预计算昂贵的 Stage-2 目标）；`_batches`；`train_stage(model, optimizer, records, *, stage, epochs, batch_size, device)`；`recall_metrics(model, records, *, device, small_quantile) -> dict`（像素 recall/precision/dice + 切片检出率 + 小病灶四分位层）；`run_training(...) -> dict`；`main(argv) -> int`。CLI：`python scripts/pretrain_tiny_segmenter.py --manifest main_data/split_manifest.csv --cache-dir cache/tensors --output checkpoints/tiny_segmenter_v1 --image-size 192 --base-channels 16 --batch-size 4 --stage1-epochs 5 --stage2-epochs 20 --learning-rate 1e-3 --seed 42 --small-lesion-quantile 0.25 --lesion-recall-min 0.70 --small-lesion-recall-min 0.70`。

**依赖**：
- 内部：`scripts.pretrain_pet_encoder`（复用 PretrainError/_load_sample/load_manifest/load_pet_masks/patient_partition）；`src.mechanism_validation.common.partition_sha256`；`src.model.segmenter.TinySegmenter / segmenter_loss`
- 外部：`numpy`、`torch`（要求 CUDA）

**输入**：split manifest（经 `patient_partition` 得 mechanism_train/calibration/validation 三角色）、NPZ 缓存（pet+mask，缺一即拒绝部分运行）、超参。

**输出**：`<output>/segmenter.pt`（TinySegmenter state_dict，已置 stage=2）+ `segmenter.pt.lineage.json`（checkpoint_sha256、架构参数、train_patient_split、partition_sha256、eval_metrics、gates、种子与两阶段 epoch 数）；正典文件已存在则拒绝覆写。**消费方（grep 已验证）**：`configs/experiments/perceptual_x0_ablation_plan_v1.yaml` 的 P1_SEG_OUT 臂（segmenter_consistency checkpoint 占位 `checkpoints/tiny_segmenter_v1/segmenter.pt`）。

**处理过程**：
1. 校验 CUDA、两阶段 epoch≥1、batch≥1；正典 checkpoint 存在即抛错；
2. 定种子；载 manifest→partition→samples（缺失计数>0 抛错），剔除 validation 患者；
3. `prepare_records` 预载训练/校准记录（训练集含预计算 Stage-2 目标）；
4. Stage-1：高斯 blob 目标（gaussian_sigma=6.0）训练 5 epoch；Stage-2：EDT 式边界目标训练 20 epoch，逐 epoch 打印供云端监控；
5. calibration 上 `recall_metrics`（阈值 0.0 二值化）计算总体与小病灶四分位层指标；
6. 双 gate 通过才保存 state_dict + lineage sidecar，否则仅打印 GATE_FAIL 报告并退出码 1。

---

### `scripts/build_cloud_runner_zip.py`

**职责**：为 perceptual-x0 消融构建可部署云端运行器 zip：恒打包 `scripts/cloud_run_perceptual_x0_ablation.ps1`（重编码为 UTF-8 BOM + CRLF，保证 Windows PowerShell 5.1 正确解析中文注释与反引号续行），`--add` 附加文件保持仓库相对路径；打印 zip 路径、条目、SHA256 与云端一行粘贴命令。

**接口**：`_ps1_bytes(text) -> bytes`（统一 CRLF + utf-8-sig）；`main(argv) -> int`。CLI：`python scripts/build_cloud_runner_zip.py --out artifacts/pfm_cloud_runner_<date>.zip --add scripts/pretrain_tiny_segmenter.py --add configs/experiments/perceptual_x0_ablation_plan_v1.yaml --console-ascii`。

**依赖**：
- 内部：无 import（仅引用仓库内文件路径常量）
- 外部：`zipfile`、`hashlib`、`argparse`

**输入**：`scripts/cloud_run_perceptual_x0_ablation.ps1`（必须存在）与 `--add` 的仓库相对附加文件。

**输出**：`artifacts/pfm_cloud_runner_<YYYYMMDD>.zip`（.ps1 条目一律 BOM+CRLF 重编码，其余原样）；stdout 打印条目清单、sha256、云端上传路径与一行命令。**本地↔云端目录对应**：本地 zip 位于 `artifacts/` → 经 ToDesk 上传到云端**父目录** `D:\ECPC-IDS-SEVEN-Work3`（ToDesk 传输面板只显示该层）→ 实际云端 worktree 为其下一级 `D:\ECPC-IDS-SEVEN-Work3\pfm_simple_20260803_retry`；生成的命令自动 `Set-Location` 到 worktree、按前缀 `pfm_cloud_runner*.zip` 找父目录中最新 zip、`Expand-Archive` 解压进 worktree 并以 Bypass 策略运行 ps1。--console-ascii 将命令降为纯 ASCII（路径含非 ASCII 时提示风险）。

**处理过程**：
1. 校验 runner 与全部 --add 路径存在；
2. 组装 (相对路径, bytes) 条目（.ps1 统一转码）；
3. 写 DEFLATED zip 并计算整体 sha256；
4. 拼装云端一行命令（cd → 找 zip → 解压 → 运行）；
5. 打印上传说明、zip 条目、sha256 与命令。

---

### `scripts/pretrain_pet_encoder.py`

**职责**：P2/P3 臂消费的冻结 `PETFeatureEncoder` checkpoint 的唯一生产者：编码器 + 临时小分割头（_SegHead，训练脚手架、训后丢弃）在二值病灶 mask 上联合训练，仅用 mechanism_train + calibration 患者（验证患者绝不入训），训练**完成后**才冻结编码器（修复了先冻结导致 checkpoint 里是随机初始化编码器的设计错误），并写出损失项 fail-closed 血脉校验所需的 lineage sidecar。

**接口**：`PretrainError(RuntimeError)`；`sha256_file`；`load_manifest(path) -> SplitManifest`；`patient_partition(manifest) -> dict`（patient_id→mechanism_train/calibration/validation，无验证患者即拒绝）；`load_pet_masks(manifest, cache_dir) -> (samples, missing)`；`_as_single_image_bchw`（把 [H,W]/[1,H,W]/[1,1,H,W] 规范为单样本 BCHW，多通道/批量即 fail-closed，且拒绝 NaN/Inf）；`_load_sample(sample, image_size) -> (pet, mask)`；`class _SegHead(nn.Module)`（encoder.forward_features 的 quarter/half 特征经 ConvTranspose 融合回全分辨率 logits）；`_recall_gate_metrics(pairs, *, small_quartile) -> dict`（IoU>0 且阈值 0.5 的检出率 + 小病灶四分位层）；`run_training(...) -> dict`；`dry_run(...) -> dict`（预检报告，状态恒 BLOCKED：真实训练是云端步骤）；`main(argv) -> int`。CLI：`python scripts/pretrain_pet_encoder.py --manifest main_data/split_manifest.csv --cache-dir cache/tensors --output checkpoints/pet_feature_encoder_v1 --base-channels 16 --image-size 192 --epochs 20 --learning-rate 1e-3 --seed 42 --dry-run --small-lesion-quantile 0.25`。

**依赖**：
- 内部：`src.data.split_manifest.SplitManifest`；`src.model.loss_terms.perceptual_x0`（PETFeatureEncoder, MIN_LESION_RECALL, MIN_SMALL_LESION_RECALL）；`src.mechanism_validation.common`（patient_partition, partition_sha256，延迟导入）
- 外部：`numpy`、`torch`（真实训练要求 CUDA）、`json`、`hashlib`

**输入**：split manifest、NPZ 缓存（pet/mask 键必须齐全，缓存不完整即拒绝）、训练超参。

**输出**：`<output>/encoder_best.pt`（state_dict + base_channels/epochs/seed/partition_sha256）+ `encoder_best.pt.lineage.json`（schema_version=1、checkpoint_sha256、train_patient_split、eval_metrics{lesion_recall, small_lesion_recall, dataset, small_lesion_quartile/threshold, base_channels}）；stdout JSON 报告（gates.pass、status=COMPLETE/GATE_FAIL）。**消费方（grep 已验证）**：`configs/experiments/perceptual_x0_ablation_plan_v1.yaml` P2_FEAT_GLOBAL / P3_FEAT_LESION_BALANCED 臂（`checkpoints/pet_feature_encoder_v1/encoder_best.pt`），以及 P2 胜出后的 300e 主训练配置 `configs/experiments/slmf_png_fullstack_perceptual_p2_300e.yaml`（losses.perceptual_x0.checkpoint）。

**处理过程**：
1. --dry-run：解析 manifest/partition/缓存完备性，输出 BLOCKED 预检 JSON；
2. 真实训练：断言 CUDA；载 manifest→partition→samples，剔除 validation 患者，缺缓存即抛错；
3. 实例化 encoder + _SegHead，显式 requires_grad=True 联合训练（AdamW，按 param id 去重避免双优化组）；
4. 逐样本 BCEWithLogits 训练 epochs 轮，周期打印 loss；
5. 训练完成后冻结编码器（requires_grad=False + eval）；
6. calibration 患者上以 detach logits 算 `_recall_gate_metrics`；
7. 保存 checkpoint + sha256 + lineage sidecar；
8. 对照 MIN_LESION_RECALL / MIN_SMALL_LESION_RECALL 出 gate 结论并退出。

---

### `scripts/run_perceptual_x0_ablation.py`

**职责**：perceptual-x0 五臂消融（Stage A）运行器：臂名 P0_PIXEL / P1_SEG_OUT / P2_FEAT_GLOBAL / P3_FEAT_LESION_BALANCED / P4_FEAT_RANDOM。本地安全入口是 --dry-run（审计五臂公平契约、报告各变体解析命令、缺真实编码器 checkpoint 时标 BLOCKED）；真实启动需冻结编码器 checkpoint（P4 显式用随机编码器，不需要）+ CUDA。

**接口**：常量 `ARM_NAMES`、`DEFAULT_CONFIG=perceptual_x0_ablation_plan_v1.yaml`、`BASE_CONFIG=slmf_png_baseline.yaml`、`DEFAULT_OUTPUT=results/perceptual_x0_few_step_v1`；`RunnerError`；`torch_cuda_available`、`utc_now`、`sha256_file`、`read_yaml`、`write_json_atomic`、`git_info`；`_arm_loss_overrides(arm) -> dict`（把臂内 losses.* 展开为点路径覆盖）；`resolve_variant_config(plan, *, arm_name, base_config, seed, root) -> dict`；`_encoder_checkpoint_for(arm, root)`；`audit_arm(plan, *, arm_name, base_config, root) -> dict`（检查：仅 segmenter_consistency/perceptual_x0 两个监督损失可变、基线损失不被改、base 与 plan 的 split_manifest 解析为同一物理文件且存在、plan 声明的 cache_dir 存在且编码器与生成模型读同一数据版本、编码器 checkpoint 存在）；`run_dry_run`；`launch_training(*, plan, plan_path, base_config, variant, seed, output_dir, root, resume) -> dict`；`main(argv) -> int`。CLI：`python scripts/run_perceptual_x0_ablation.py --config configs/experiments/perceptual_x0_ablation_plan_v1.yaml --base-config configs/experiments/slmf_png_baseline.yaml --dry-run --variant P3_FEAT_LESION_BALANCED --seed 42 --output results/perceptual_x0_few_step_v1 --resume`。

**依赖**：
- 内部：`src.model.config_utils.apply_dotlist_overrides / load_full_config`
- 外部：`yaml`、`subprocess`（调 scripts/train_v2.py）、`hashlib`

**输入**：五臂 plan YAML（arms.* 的 losses 覆盖、fairness.split_manifest/cache_dir）；基础配置 slmf_png_baseline.yaml；--variant 指定臂。

**输出**：`<output>/configs/<variant>.yaml`（解析后的该臂完整配置）、`<output>/logs/<variant>.log`（实时流式日志）、`resolved_runs.json`（每变体配置+命令+执行状态）、`execution_metadata.json`（git commit/branch/dirty）。

**处理过程**：
1. 读 plan YAML；dry-run 分支逐臂 `audit_arm` + 解析命令，汇总 resolved_runs.json（状态 READY/BLOCKED）；
2. 真实分支：audit BLOCKED 即抛 RunnerError；
3. 无 CUDA 即拒绝（本地只许 dry-run）；
4. `resolve_variant_config`：基线配置 + 臂的点路径监督损失覆盖 + 公平性修正（manifest/cache 统一）；
5. --resume 时把 resume 语义映射到 training.resume_from（train_v2.py 无 --resume 参数，训练器自动找最新 checkpoint）；
6. 写该臂 YAML，`subprocess.Popen` 运行 `python scripts/train_v2.py --config ...`，stdout 逐行转发并落日志；
7. 非零退出抛错；成功写 resolved_runs.json。

---

### `scripts/validate_perceptual_x0_few_step.py`

**职责**：perceptual-x0 消融的患者级少步验证：固定采样种子在 NFE ∈ {20,8,4,2} 网格上对一对臂运行现有评估能力，聚合为配对患者效应并按 H6 风格配对统计（paired_patient_effects / effect_statistics / calibration_absolute_margin）裁决主终点与安全性 gate。被 `tests/test_perceptual_x0_validator.py` 消费。

**接口**：常量 `EVAL_SEED=42`、`NFE_GRID=(20,8,4,2)`；`ValidatorError`；`utc_now`、`load_json`、`git_info`；`run_evaluate(*, config, checkpoint, split, nfe, output, root, seed, log_path)`（调 scripts/evaluate.py --weights ema --num-steps NFE）；`_resolve_path`；`load_arm_checkpoints(plan_path, root) -> dict[str, Path]`（从 plan 的 resolved_runs.json 找各臂 checkpoint）；`load_plan`；`collect_effects`；`freeze_calibration_margins`（在 calibration split 冻结非劣性 margin）；`_verify_endpoints_present`；`_gate_primary`、`_gate_non_inferiority`、`_gate_learned_representation`、`_resolve_arm`、`_safety_delta_gate`、`_endpoint_direction`；`derive_mechanism_partition` / `_filter_eval_to_partition`（按机制 partition 过滤评估行防泄漏）；`run_validation(...) -> decision`；`main(argv) -> int`。CLI：`python scripts/validate_perceptual_x0_few_step.py --plan configs/experiments/perceptual_x0_ablation_plan_v1.yaml --manifest main_data/split_manifest.csv --checkpoint ARM=PATH（可重复） --config ARM=PATH（可重复） --output results/perceptual_x0_few_step_v1 --force`。

**依赖**：
- 内部：`src.mechanism_validation.common`（canonical_json_sha256, file_sha256, write_csv, write_json）；`src.mechanism_validation.model_experiments`（calibration_absolute_margin, effect_statistics, paired_patient_effects）
- 外部：`subprocess`、`csv`、`json`

**输入**：plan YAML（含评估网格与主/安全终点定义）、split manifest、每臂 checkpoint（ARM=PATH 显式给或从 plan 解析；缺文件即拒绝）、每臂已解析模型配置（拒绝把实验 plan YAML 当模型配置）。

**输出**：`<output>/patient_effects.csv`（每比较的逐患者效应行）、`calibration_margins.json`（冻结的非劣性 margin）、`resolved_runs.json`（每臂×NFE 的评估溯源：checkpoint sha、seed）、`decision.json`（主终点 + 安全 gate 判定）、`execution_metadata.json`。

**处理过程**：
1. 解析 --checkpoint/--config 的 ARM=PATH 对；无显式 checkpoint 时从 plan 的 resolved_runs.json 推断；每臂配置缺省取 configs/experiments/perceptual_x0_<ARM>.yaml 且不存在即抛错；
2. 校验全部臂 checkpoint 文件存在；
3. 对每臂 × 每 NFE 调 `run_evaluate`（固定 seed=42、ema 权重、split、--num-steps）；
4. `collect_effects` 汇聚评估输出为逐患者指标，`derive_mechanism_partition` + 过滤确保只用合法 partition 患者；
5. 配对患者数 < --min-paired-patients（默认 5）或任一必需终点缺失 → FAIL（fail-closed）；
6. `freeze_calibration_margins` 在 calibration 上冻结 margin，验证 split 只跑冻结胜者；
7. 依序评估主终点 gate、非劣性 gate、习得表征 gate 与安全 delta gate，写 decision.json。

---

### 模块依赖小结

**本组 import 的内部模块**（grep 查证）：
- `src.model.config_utils`（load_full_config / resolve_runtime_profile / apply_dotlist_overrides / validate_png_baseline_config / save_resolved_config / log_startup_status）——诊断、对齐校验、均值预训练、消融运行器、放大审计共 7 个脚本使用；
- `src.model.slmf_bbdm.SLMFBBDM`（diagnose_gabor_adapter / diagnose_stripes / diagnose_loss / audit_magnification_consistency）；
- `src.model.mean_predictor.LowFrequencyPETPredictor` + `src.model.mean_pretraining.MeanPretrainer`（仅 pretrain_conditional_mean）；
- `src.model.loss_terms.perceptual_x0`（PETFeatureEncoder, MIN_LESION_RECALL, MIN_SMALL_LESION_RECALL——仅 pretrain_pet_encoder）；
- `src.model.segmenter`（TinySegmenter, segmenter_loss——仅 pretrain_tiny_segmenter）；
- `src.data.dataset`（CachedDataset / FakeDataset / build_dataloaders）、`src.data.lineage`（load_checkpoint_data_lineage, CacheLineageError）、`src.data.split_manifest.SplitManifest`；
- `src.mechanism_validation.common`（file_sha256 / write_json / write_csv / canonical_json_sha256 / patient_partition / partition_sha256——同步、队列、汇总、验证器与两个预训练器共用）、`src.mechanism_validation.model_experiments`（配对统计）、`src.mechanism_validation.magnification(.audit)`；
- 脚本互相 import：`audit_checkpoint_lineage`←`scripts.run_cloud_stage0c_gate._validate_checkpoints`；`summarize_cloud_runs`←`scripts.audit_prior_anchored_run` 纯函数；`pretrain_tiny_segmenter`←`scripts.pretrain_pet_encoder` 数据工具；`audit_magnification_consistency`←`scripts.evaluate._select_checkpoint_state`。两个频谱分析脚本完全独立（零内部依赖）。

**被谁消费**：测试套件导入 `scripts.audit_magnification_consistency`（tests/test_magnification_audit_cli.py）、`scripts.validate_perceptual_x0_few_step`（tests/test_perceptual_x0_validator.py）、`scripts.run_perceptual_x0_ablation`（tests/test_perceptual_x0_loss.py、test_pretrain_tiny_segmenter.py）、`scripts.pretrain_pet_encoder`/`scripts.pretrain_tiny_segmenter`（tests/test_perceptual_x0_loss.py、test_pretrain_tiny_segmenter.py）；`configs/cloud_worktree_sync.yaml` 的 allowlist 显式收录本组多个脚本分发上云；云端部署链为 build_cloud_runner_zip（zip+ps1）→ cloud_run_perceptual_x0_ablation.ps1 → run_perceptual_x0_ablation → train_v2.py → validate_perceptual_x0_few_step；artifacts/pfm_simple_cloud_* 目录是云端测试树快照（非源码）。

**预训练权重 → 主配置加载点**（重要跨模块发现）：
1. `pretrain_conditional_mean.py` → `mean_best.pt` → 主训练读 `modules.conditional_mean.checkpoint`：freq_mean_pretrain_v2 被 slmf_png_spectral_router_v5 / boundary_reliable_v4 及 spectral_router_ablation_plan_v5 / cold_bias_v7 / final_calibration_v8 / boundary_reliable_ablation_plan_v3·v4 / frequency_ablation_plan_v2 引用；freq_mean_excluded_v1（病理排除版）被 prior_anchored_router_100e/300e、fullstack_perceptual_p2_300e、background_tail_repair_e100 引用；
2. `pretrain_pet_encoder.py` → `encoder_best.pt` + lineage sidecar → perceptual_x0_ablation_plan_v1.yaml 的 P2/P3 臂（losses.perceptual_x0.checkpoint）与 P2 胜出后的 slmf_png_fullstack_perceptual_p2_300e.yaml；
3. `pretrain_tiny_segmenter.py` → `segmenter.pt` → perceptual_x0_ablation_plan_v1.yaml 的 P1_SEG_OUT 臂（segmenter_consistency checkpoint）。三个生产者全部带 sha256 lineage sidecar，且严格排除 validation 患者（机制 partition 约束），是感知损失 fail-closed 校验链的上游。


---

## 13. wuzhe/ 内嵌工程（对比实验与论文/专利脚本）

### 模块概述

- 本组是嵌在主仓库内部的"子工程" wuzhe/：自带一份历史分叉的 src 快照（与根 src 非同一份，见文末哈希比对）、两个数据目录（main_data 8:2 划分、wuzhe_data 五折交叉验证）以及对比实验包 comparison_experiments。
- comparison_experiments 用"注册表 + 工厂"（MODEL_REGISTRY/build_comparison_model）与"适配器"（ComparisonBase 统一 loss, logs = model(batch) / model.sample(batch)['synthetic_pet'] 接口对齐 SLMF-BBDM）实现 5 个 baseline；OriginalProtocolTrainer 按模型类型分发到各自的官方训练协议（生成器/判别器分离优化器、ImagePool、AdamW 等）。
- 数据流：data/origin_data → split_5fold_three_area_balanced.py → wuzhe_data/fold_1..5（叠加 main_data 8:2）→ LightweightPNGSliceDataset（ct/pet_peizhuan/label PNG）→ train_comparison.py 训练 → evaluate_comparison.py 打指标 → output/comparison_metrics 汇总。
- 专利文档链为纯文档处理流水线：build_patent_disclosure_docx.py 生成初稿 → revise_patent_avoid_prior_art.py → integrate_scheme_into_innovations.py → integrate_losses_into_two_innovations.py → tighten_allowability_revision.py，逐版改写 wuzhe/output/patent_disclosure/ 下的 docx。
- 两个 split 质检脚本只用 Python 标准库（五折划分脚本甚至用 zlib/struct 手写 PNG 解码），产出 HTML 可视化报告。

### wuzhe/comparison_experiments/__init__.py

**职责**：包标识文件，仅一行 docstring（"Comparison baselines aligned to the SLMF-BBDM training interface."），把 comparison_experiments 标记为可导入包，无任何代码。
**接口**：无顶层 class/def。
**依赖**：内部: 无；外部: 无。
**输入**：无。**输出**：无。
**处理过程**：仅声明包身份，供 train_comparison.py 等以 comparison_experiments.xxx 形式导入。

### wuzhe/comparison_experiments/models.py

**职责**：实现 5 个 CT→PET 对比 baseline（Pix2Pix/CycleGAN/RegGAN/CPDM/District-specific GAN），全部对齐 SLMF-BBDM 的 batch 接口（batch["ct"]/batch["pet"] 均为 [B,1,H,W]、归一化 [-1,1]）。
**接口**：
- 公共件：_requires_grad(modules, flag)；_image_gradient_l1(pred, target)；_default_reconstruction_loss(pred, target, *, mse_weight, l1_weight, gradient_weight) -> (loss, logs)；ConvBlock/UpBlock/ResnetBlock；UNetGenerator(in,out,base_channels)；ResnetGenerator(in,out,base,num_blocks)；PatchDiscriminator(in_channels, base_channels, num_layers=3)（70×70 PatchGAN）；LSGANLoss.forward(pred, target_is_real)；SimpleUNet(in,out,base,final_activation)；RegistrationNet(in_channels=2, base=32, max_flow=0.15).forward(moving, fixed)->flow；warp_image(image, flow)；smoothness_loss(flow)；VectorQuantizerEMA(num_embeddings, embedding_dim)；VQGANFirstStage(in,latent,base,num_embeddings,use_vq).encode/decode/forward；SpatialRescalerCondition(in,out,n_stages).forward(x, size)。
- @dataclass LossWeights（lambda_l1/cycle/identity/registration/smooth/attention/district/vq/first_stage/latent/patch_overlap、mse/l1/gradient_weight）。
- ComparisonBase(nn.Module)：model_name；get_trainable_params()/get_total_params()；sample(batch)->{"synthetic_pet"}；generate(ct) 抽象。
- Pix2PixComparison(base_channels=64, lambda_l1=100)；CycleGANComparison(base_channels, num_res_blocks=6, lambda_cycle=10, lambda_identity=5)；RegGANComparison(base_channels, num_res_blocks, lambda_registration=10, lambda_smooth=0.1)；CPDMComparison(base_channels, first_stage_channels, latent_channels=3, vq_num_embeddings=512, use_vq, num_train_timesteps=1000, ...) 含 _attention_map/_attenuation_map/_condition_latent/_add_bridge_noise/_sample_impl；DistrictSpecificGANComparison(base_channels, lambda_district=100, patch_size=96, patch_overlap=24, sliding_window_inference=True) 含 _district_masks/random_patch_batch/_sliding_window_generate。
- MODEL_REGISTRY dict + build_comparison_model(config) -> ComparisonBase（读 config["comparison_model"]（回退 "model"），pop "name" 后 **实例化）。
**依赖**：内部: 无（自包含）；外部: torch、torch.nn、torch.nn.functional。
**输入**：batch 字典（ct [B,1,H,W]、pet [B,1,H,W]，可选 mask/mu_map/ct_hu/district_mask）；config dict（comparison_model 块）。
**输出**：forward 返回 (total_loss, logs_dict)；sample 返回 {"synthetic_pet": [B,1,H,W]}（district_gan 额外带 district_masks）。
**处理过程**：
1. 组装公共积木：ConvBlock/UpBlock（InstanceNorm+转置卷积）、ResnetBlock（反射填充残差）、PatchGAN 判别器、LSGAN 损失。
2. Pix2Pix：U-Net 生成器（4 下采样+瓶颈+3 上采样跳跃拼接，Tanh 输出）+ 2 通道条件 PatchGAN；G 损失 = LSGAN(fake→real) + 100·L1，D 损失 = 0.5·(LSGAN(real)+LSGAN(fake.detach))。
3. CycleGAN：g_ct_to_pet/g_pet_to_ct 双 ResnetGenerator（6 残差块）+ d_pet/d_ct 单通道判别器；损失 = 双向对抗 + 10·cycle L1 + 5·identity L1。
4. RegGAN：ResnetGenerator + RegistrationNet（Tanh×max_flow 的小形变场网）+ 条件判别器；损失 = 对抗 + 10·L1(fake, warp(pet, flow)) + 0.1·flow 平滑（训练校正配准误差下的监督目标）。
5. CPDM：VQGANFirstStage（EMA 码本 512×3、2× 下采样 latent）→ 布朗桥加噪 x_t = m_t·ct_z + (1-m_t)·pet_z + σ_t·ε（m_t 从 1 线性到 0，σ_t = 0.25·sin）→ SimpleUNet 输入 9 通道（x_t+ct_z+cond_z）输出预测 grad = ct_z-pet_z（objective=grad）→ 解码回像素；条件为 CT+attention map（mask 或边缘能量回退）+attenuation map（mu_map/ct_hu/1000 回退）经 SpatialRescalerCondition；损失 = 重建(MSE+L1+梯度) + attention 加权 L1 + latent 梯度 MSE + latent L1 + 一阶段重建 + VQ 码本。
6. DistrictGAN：head/trunk/arms/legs 四套 U-Net 生成器与 PatchGAN；district_mask 缺省时按轴向带 [0,H/5,H/2,4H/5,H] 四段硬 mask 经 avg_pool(9) 软化并归一化求和为 1；训练时 random_patch_batch 裁 96×96 随机 patch，逐区计算对抗与 100·区域 L1，软 mask 重叠>1.01 处罚 patch overlap 损失；推理用 96/24 滑窗重叠平均。
7. build_comparison_model 按名称查 MODEL_REGISTRY 构造，未知名称抛 ValueError。

### wuzhe/comparison_experiments/png_dataset.py

**职责**：对比实验专用的轻量 PNG 切片数据集（刻意不依赖 torchvision，只用 PIL+torch），扫描 root/{split}/{ct,pet_peizhuan,label} 目录按文件名配对，输出与主模型一致的 batch 字典。
**接口**：_parse_sample_id(sample_id)->(patient_id, slice_id)（前 6 位数字拆前 3 后 3）；_read_png_unit(path)->[0,1] float32；_unit_to_range；_resize_to_tensor(arr, size, mode)；@dataclass PNGEntry；LightweightPNGSliceDataset(root, split="train", *, image_size=192, ct_dir="ct", pet_dir="pet_peizhuan", label_dir="label", augment=False, require_label=True)（__len__/__getitem__/_scan）；build_png_dataloaders(data_cfg, run_cfg) -> (train_loader, val_loader)。
**依赖**：内部: 无；外部: numpy、torch、torch.nn.functional、PIL.Image、torch.utils.data。
**输入**：数据根目录（如 wuzhe/main_data 或 wuzhe/wuzhe_data/fold_N），子目录 ct/pet_peizhuan/label 下的 PNG；配置键 data.png_root/image_size/batch_size/val_batch_size/png_ct_dir/png_pet_dir/png_label_dir/png_require_label/augment 与 runtime.num_workers/pin_memory/persistent_workers/prefetch_factor。
**输出**：__getitem__ 返回 dict：ct/pet [1,S,S]∈[-1,1]、mask [1,S,S]（nearest 插值>0.5 二值）、organ_mask [6,S,S] 与 organ_distance/mu_map/ct_hu/pet_suv 全零占位、pet_activity [1,S,S]∈[0,1]、meta（sample_id/patient_id/slice_id/原始尺寸/suv_ok=False 等）。
**处理过程**：
1. 构造时检查三个模态目录存在，扫描文件名（stem 去重）取 ct∩pet(∩label) 交集并排序。
2. 每样本解析 sample_id 前缀得到患者/切片号，建 PNGEntry 列表并打印统计。
3. __getitem__ 逐张读 PNG 归一化到 [0,1]，再线性映射到 [-1,1] 并 F.interpolate 到 image_size（label 用 nearest）。
4. augment 时 50% 概率水平翻转 ct/pet/mask。
5. 填充 organ/mu_map 等占位零张量与 meta 字典，保持与主模型 batch 键完全兼容。
6. build_png_dataloaders 分别构建 train（shuffle、drop_last）与 val（不 shuffle）DataLoader。

### wuzhe/comparison_experiments/trainers.py

**职责**：为 5 个 baseline 提供"原论文协议"训练器——与 models.py 内联合更新不同，这里按官方实现使用生成器/判别器分离的优化器、ImagePool 缓冲、分步梯度更新与早停/检查点管理。
**接口**：ImagePool(pool_size=50).query(image)；_detect_dtype()；_move_batch(batch, device)；_adam(params, lr, weight_decay, betas=(0.5,0.999))；OriginalProtocolTrainer(model, config, train_loader, val_loader=None, device=None)，方法 _step_pix2pix/_step_cyclegan/_step_reggan/_step_district_gan/_step_diffusion_or_generic（均返回 Dict[str,float]）、train_step(batch)、eval_step(batch)、evaluate_loader()、train_epoch()、run(num_epochs=None)、save_checkpoint(tag=None)。
**依赖**：内部: comparison_experiments.models（五个模型类、_requires_grad、smoothness_loss、warp_image）；外部: torch、torch.nn.functional、torch.utils.data.DataLoader、os/random/time/collections.OrderedDict。
**输入**：已构建模型与 DataLoader；config 键 runtime.amp/log_interval/eval_interval/sample_interval/save_interval/grad_clip_norm/num_workers、training.num_epochs/learning_rate/weight_decay/lr_min/early_stopping.{enabled,patience,min_delta,warmup_epochs}、comparison_model.pool_size。
**输出**：checkpoints/<experiment.name>/ckpt_epochNNNN.pt 与 ckpt_best.pt（含 model/epoch/step/config/protocol/best_val_loss/early_stopping/优化器状态）；stdout 训练日志。
**处理过程**：
1. __init__ 按 isinstance 分派 protocol：pix2pix_official / cyclegan_official（含双 ImagePool）/ reggan_public_code_aligned（生成器+注册网共用 opt_g）/ district_gan_paper_reproduction / cpdm_bbdm_aligned 与 generic（单 AdamW）。
2. 每个优化器配 CosineAnnealingLR(T_max=num_epochs, eta_min=lr_min)。
3. GAN 协议分步更新：以 pix2pix 为例先 D（detach fake）后 G；cyclegan 先 G 后 D 且 D 用 ImagePool 回放历史假图；district 先随机 patch 化 batch 再先 D（no_grad 合成）后 G。
4. cpdm/generic 走 model(batch) 联合损失单步 backward，autocast AMP（bf16 优先）仅对扩散协议启用。
5. train_epoch 聚合各 log_interval 处的标量，scheduler.step()，输出 perf/epoch_seconds 与 perf/lr。
6. run() 主循环：每 eval_interval 在验证集评估，val loss 改善则存 ckpt_best.pt，否则累计 no_improve_epochs，超过 warmup+patience 触发早停；每 save_interval 存周期检查点，每 sample_interval 打印样例 PET 数值范围。
7. save_checkpoint 把 model/优化器/协议/早停状态一并 torch.save。

### wuzhe/comparison_experiments/train_comparison.py

**职责**：对比实验训练入口 CLI——读 YAML 配置构建模型与 PNG 数据加载器，套用 OriginalProtocolTrainer 跑完整训练；不改动主模型 SLMF-BBDM 代码。
**接口**：_set_seed(seed)；_parse_value(raw)（bool/None/int/float/str）；_load_config(path, overrides)->Dict（配合 apply_dotlist_overrides）；_save_model_metadata(model, config, ckpt_dir)（写 comparison_metadata.json）；main()。
CLI：python comparison_experiments/train_comparison.py --config comparison_experiments/configs/pix2pix.yaml [--override training.num_epochs=50 ...]（在 wuzhe/ 目录下运行）。
**依赖**：内部: comparison_experiments.models.build_comparison_model、comparison_experiments.png_dataset.build_png_dataloaders、comparison_experiments.trainers.OriginalProtocolTrainer、src.model.config_utils（apply_dotlist_overrides/resolve_runtime_profile/save_resolved_config，经 sys.path.insert(wuzhe/) 实际解析到 wuzhe/src）；外部: argparse/json/os/random/sys/pathlib、numpy、torch、yaml。
**输入**：--config YAML（experiment/data/runtime/training/comparison_model 五块）；--override 可重复的 key=value 点路径覆盖。
**输出**：checkpoints/<experiment.name>/ 下 resolved 配置、comparison_metadata.json（参数量、接口说明）、ckpt_epoch*.pt/ckpt_best.pt。
**处理过程**：
1. 解析 CLI，加载 YAML 并应用 dotlist 覆盖。
2. resolve_runtime_profile 解析运行档位，按 experiment.seed 固定 random/numpy/torch 种子。
3. build_comparison_model 构建模型，打印可训练/总参数量，写元数据。
4. build_png_dataloaders 构建训练/验证加载器；按 runtime.device（auto→cuda）选设备。
5. 实例化 OriginalProtocolTrainer 并 run() 完整训练。

### wuzhe/comparison_experiments/evaluate_comparison.py

**职责**：在 PNG train/val 划分上评估对比模型最优检查点：按 ckpt 内 config 重建模型，逐样本计算 MAE/MSE/RMSE/PSNR/SSIM 及 ROI（病灶 mask 内）指标并汇总落盘。
**接口**：_torch_load(path, device)（weights_only 兼容回退）；_de_collate_meta(meta, batch_size)；_ssim_torch(pred, target, data_range=2.0)（11×11 均值池化实现）；_sample_metrics(pred, target, mask)->Dict；_aggregate(rows)->Dict（mean/std/median/min/max）；evaluate(checkpoint, png_root, *, split="val", pet_dir="pet_peizhuan", image_size=192, batch_size=4, output_dir="output/comparison_metrics", device=None)->Dict；main()。
CLI：python comparison_experiments/evaluate_comparison.py --checkpoint ... --png-root ... [--split val] [--pet-dir pet_peizhuan] [--image-size 192] [--batch-size 4] --output-dir ... [--device cuda]。
**依赖**：内部: comparison_experiments.models.build_comparison_model、comparison_experiments.png_dataset.LightweightPNGSliceDataset；外部: argparse/csv/json/math/os/sys、numpy、torch、torch.nn.functional、torch.utils.data.DataLoader。
**输入**：训练产出的 .pt 检查点（须含 config）、PNG 数据根、split/pet_dir/image_size 等。
**输出**：<output-dir>/metrics_summary.json（含均值统计与 checkpoint/png_root 等溯源字段）与 metrics_per_sample.csv（每样本一行），并打印 JSON 摘要；返回 summary。
**处理过程**：
1. torch.load 检查点，取出内嵌 config，用 build_comparison_model 重建模型并 load_state_dict，切 eval。
2. 构建指定 split 的 LightweightPNGSliceDataset 与 DataLoader（不 shuffle、workers=0）。
3. 逐 batch model.sample 取 synthetic_pet，与 batch["pet"]/batch["mask"] 在 CPU 上逐样本算 _sample_metrics（PSNR 按 data_range=2，mask 非空时加 roi_mae/roi_mse/roi_rmse）。
4. 每 10 个 batch 打印进度；_aggregate 汇总忽略 NaN。
5. 写 metrics_summary.json 与 metrics_per_sample.csv，stdout 打印并返回 summary。

### wuzhe/comparison_experiments/run_all_png_experiments.py

**职责**：对比实验总调度器：先在 main_data 上跑全部模型的主实验，再依次跑 wuzhe_data/fold_1..5 五折交叉验证，每个实验"训练→选 ckpt_best.pt→验证集评估"，最终汇总全部指标。
**接口**：_config_for(model)->Path；_run(cmd, dry_run=False)；_load_summary(path)；_latest_checkpoint(exp_name)（优先 ckpt_best.pt，否则最大 epoch）；_train_one(model, exp_name, png_root, args)；_evaluate_one(exp_name, png_root, args)->Dict；_run_train_eval(model, exp_name, root, phase, fold, args)->Dict；_write_aggregate(rows, output_root)；main()。
CLI：python comparison_experiments/run_all_png_experiments.py [--main-root main_data] [--cv-root wuzhe_data] [--models pix2pix,cyclegan,reggan,cpdm,district_gan] [--epochs 1000] [--image-size 192] [--batch-size 4] [--pet-dir pet_peizhuan] [--early-stop-patience 80] [--early-stop-min-delta 1e-4] [--early-stop-warmup 50] [--output-root output/comparison_metrics] [--device ...] [--no-amp] [--dry-run] [--continue-on-error]。
**依赖**：内部: 无直接 import（以 subprocess 调 train_comparison.py/evaluate_comparison.py，cwd=wuzhe/）；外部: argparse/csv/json/subprocess/sys/pathlib。
**输入**：CLI 参数（见上）；configs/<model>.yaml 基础配置。
**输出**：每个实验 checkpoints/<exp_name>/ 与 output/comparison_metrics/<exp_name>/；聚合 output/comparison_metrics/all_metrics_summary.json 与 .csv（列含 phase/fold/model）；失败清单 failures.json（--continue-on-error 时）。
**处理过程**：
1. 解析参数，解析 main_root/cv_root/output_root 为 wuzhe/ 下绝对路径。
2. 主实验阶段：对每个模型 guarded 调 _run_train_eval（exp_name 形如 main_<model>，phase="main"，fold="main_8_2"）。
3. _train_one 拼 ~20 条 --override（png_root、目录名、epochs、early stopping 等）调用训练脚本。
4. _evaluate_one 取最新检查点调用评估脚本并读回 metrics_summary.json。
5. 五折阶段：fold_1..fold_5 逐折 × 逐模型（exp_name 形如 cv_fold_1_<model>）。
6. guarded 捕获异常：默认立即抛出终止，--continue-on-error 时记 failures.json 继续下一个实验。
7. 每完成一个实验即增量 _write_aggregate，最后统一写汇总并在有失败时 SystemExit。

### wuzhe/comparison_experiments/run_all_comparisons.ps1

**职责**：PowerShell 批量入口，按固定顺序依次用五个 YAML 配置训练全部对比模型（不含评估与五折）。
**接口**（CLI）：powershell -ExecutionPolicy Bypass -File comparison_experiments/run_all_comparisons.ps1 [-Epochs 1000] [-ExtraOverride ""]。
**依赖**：内部: 调用 comparison_experiments/train_comparison.py 与 configs/*.yaml；外部: PowerShell、python。
**输入**：-Epochs 覆盖 training.num_epochs；-ExtraOverride 追加一条 key=value 覆盖。
**输出**：各模型 checkpoints/<cmp_*>/ 检查点与训练日志。
**处理过程**：遍历 pix2pix/cyclegan/reggan/cpdm/district_gan 五个名字，拼 comparison_experiments/configs/<name>.yaml 逐个执行 python 训练命令，带 --override training.num_epochs 与可选 ExtraOverride。

### wuzhe/comparison_experiments/run_main_then_fivefold.ps1

**职责**：一键总流程的 PowerShell 封装：主实验（main_data）+ 五折交叉验证（wuzhe_data/fold_1..5），转发参数给 run_all_png_experiments.py。
**接口**（CLI）：powershell -File comparison_experiments/run_main_then_fivefold.ps1 [-Epochs 1000] [-ImageSize 192] [-BatchSize 4] [-ValBatchSize 4] [-EarlyStopPatience 80] [-EarlyStopMinDelta 0.0001] [-EarlyStopWarmup 50] [-PetDir pet_peizhuan] [-Models "pix2pix,cyclegan,reggan,cpdm,district_gan"]。
**依赖**：内部: 调用 comparison_experiments/run_all_png_experiments.py（--main-root main_data --cv-root wuzhe_data）；外部: PowerShell、python。
**输入**：上述参数（PetDir 默认 pet_peizhuan 即配准后 PET）。
**输出**：与 run_all_png_experiments.py 相同——checkpoints/ 与 output/comparison_metrics/ 汇总。
**处理过程**：把 PowerShell 参数逐一映射为 python CLI 参数（含早停三参）后单次调用 run_all_png_experiments.py。

### wuzhe/comparison_experiments/README.md

**职责/要点**：本目录使用说明：(1) 统一接口 loss, logs = model(batch)、model.sample(batch)["synthetic_pet"]，输入 batch["ct"]/batch["pet"] [B,1,H,W]，可选 mask/mu_map/ct_hu/district_mask；(2) 五个模型的年份、复现依据（官方代码/论文）与损失策略对照表；(3) 主实验（main_data）+五折（wuzhe_data）一键流程、默认早停参数（warmup 50/patience 80/min_delta 1e-4）、指标输出 output/comparison_metrics/、--dry-run 与冒烟测试命令。
**依赖**：内部: 引用本目录各脚本；外部: 无。

### wuzhe/comparison_experiments/REPRODUCTION_STATUS.md

**职责/要点**：复现度自评表：Pix2Pix 与 CycleGAN 85–90%（官方结构+分离优化+ImagePool），RegGAN 75–85%，CPDM 70–80%（VQGAN-like 一阶段+9→3 UNet+objective=grad），District-specific GAN 65–75%。遗留差距两条：CPDM 需官方 VQGAN checkpoint 才能进一步逼近；District GAN 原文是 whole-body 3D patch，当前数据接口只有 2D 切片，真实 head/trunk/arms/legs 区域仅做接口兼容与 2D 轴向带近似。
**依赖**：内部: 无；外部: 无。

### wuzhe/comparison_experiments/configs/pix2pix.yaml

**职责**：Pix2Pix baseline 配置：experiment.name=cmp_pix2pix（seed 42）、data 块（image_size 192、batch_size 4、augment）、runtime 块（amp/channels_last、num_workers 4、eval/save/sample_interval 50）、training 块（1000 epoch、lr 2e-4、lr_min 1e-6、ema 0.999）。
**接口**：YAML 配置文件，被 train_comparison.py --config 消费。
**依赖**：内部: 无；外部: 无（由 yaml.safe_load 解析）。
**输入**：文件本身。**输出**：解析后的 config dict。
**处理过程**：comparison_model 块仅 3 键：name: pix2pix、base_channels: 64、lambda_l1: 100.0。

### wuzhe/comparison_experiments/configs/cyclegan.yaml

**职责**：CycleGAN baseline 配置，骨架同 pix2pix.yaml，差异点：experiment.name=cmp_cyclegan、batch_size/val_batch_size=2（双生成器显存更大）、comparison_model 指定 name: cyclegan、base_channels 64、num_res_blocks 6、pool_size 50、lambda_cycle 10.0、lambda_identity 5.0。
**接口**：YAML 配置文件。**依赖**：内部: 无；外部: 无。**输入**：文件本身。**输出**：config dict。**处理过程**：pool_size 由 OriginalProtocolTrainer 读取用于构建双 ImagePool。

### wuzhe/comparison_experiments/configs/reggan.yaml

**职责**：RegGAN baseline 配置，骨架同上，batch_size=2；comparison_model：name: reggan、base_channels 64、num_res_blocks 6、lambda_registration 10.0、lambda_smooth 0.1。
**接口**：YAML 配置文件。**依赖**：内部: 无；外部: 无。**输入**：文件本身。**输出**：config dict。**处理过程**：lambda_registration/lambda_smooth 分别加权注册校正 L1 与形变场平滑损失。

### wuzhe/comparison_experiments/configs/cpdm.yaml

**职责**：CPDM（条件布朗桥扩散）baseline 配置，是五个配置中最详细的一个。
**接口**：YAML 配置文件。**依赖**：内部: 无；外部: 无。**输入**：文件本身。**输出**：config dict。
**处理过程**：与其它配置的差异：training.learning_rate=1e-4 且 weight_decay=0.01（其它为 2e-4/0.0）；data.optional_keys 增补 [mu_map, ct_hu]；comparison_model 15 键：name: cpdm、base_channels 64、first_stage_channels 32、latent_channels 3、vq_num_embeddings 512、use_vq true、num_train_timesteps 1000、lambda_attention/lambda_vq/lambda_first_stage/lambda_latent 1.0、mse_weight 1.0、l1_weight 1.0、gradient_weight 0.1。

### wuzhe/comparison_experiments/configs/district_gan.yaml

**职责**：District-specific GAN baseline 配置，batch_size=2；comparison_model：name: district_gan、base_channels 64、lambda_district 100.0、lambda_patch_overlap 0.1、patch_size 96、patch_overlap 24、sliding_window_inference: true。
**接口**：YAML 配置文件。**依赖**：内部: 无；外部: 无。**输入**：文件本身。**输出**：config dict。
**处理过程**：patch_size/patch_overlap 决定随机 patch 训练与滑窗推理（stride = 96-24 = 72）的几何参数。

### wuzhe/build_model_report_pdf.py

**职责**：用 reportlab 以编程方式绘制 SLMF-BBDM 模型汇报 PDF（横版 A4 共 7 页：封面/概览/架构/模块/创新点/训练/状态），含中英文混排、流程图节点与卡片版式。
**接口**：register_fonts()（注册 C:\Windows\Fonts\msyh.ttc 微软雅黑）；fit_lines(text, font, size, max_width)->list[str]（按字形宽度混排换行，保持 pred_x0、Gabor、CT→PET 等英文 token 完整）；draw_text/draw_bullets/round_rect/card/page_header/section_title/arrow/node 等绘图原语；cover/page_overview/page_architecture/page_modules/innovation_card/page_innovations/page_training/page_status 各页绘制函数；build() -> Path（无 CLI 参数，直接 python wuzhe/build_model_report_pdf.py）。
**依赖**：内部: 无（文本内容硬编码自项目现状描述）；外部: reportlab（pdfgen.canvas、pdfbase.ttfonts、lib.colors、lib.pagesizes）、math/re/pathlib。
**输入**：无命令行输入；依赖 Windows 系统字体路径。
**输出**：wuzhe/output/pdf/SLMF_BBDM_model_report.pdf（OUT 常量基于脚本所在目录）。
**处理过程**：
1. 注册中文字体并定义配色/页面常量。
2. fit_lines 实现中英文混排测量换行。
3. 依次绘制 7 页：封面（标题+要点）、概览（任务/数据/指标卡片）、架构（节点+箭头流水线图）、模块（先验模块卡片）、创新点（两张 innovation_card）、训练（损失与阶段策略）、状态（结论：下一步是数据预处理、正式训练与消融验证而非继续堆模块）。
4. c.save() 落盘并返回路径。

### wuzhe/build_patent_disclosure_docx.py

**职责**：从零生成 SLMF-BBDM 专利技术交底书 docx 初稿（封面、摘要、技术领域、背景、方案、模块、损失、权利要求布局、应用场景、结论等 16 节），中文排版（宋体、首行缩进、表格底纹、图注）。
**接口**：set_run_font(run, size/bold/color/name="SimSun")；set_cell_shading(cell, fill)；set_cell_text；add_heading(doc, text, level=1)；add_para(doc, text, bold_prefix, first_line=True)；add_bullet；add_formula；add_caption；add_callout(doc, title, body)（灰底提示框）；add_table(doc, rows, widths)；add_figure(doc, path, caption, width=6.2)；build_doc() -> OUT_DOCX（无 CLI 参数）。
**依赖**：内部: 无代码依赖，但读取运行目录下 output/figures 与 output/midterm_report/figures 的四张 PNG 插图；外部: python-docx（Document、OxmlElement、qn、Inches/Pt/RGBColor、枚举）。
**输入**：FIGURES 列出的 4 个图片路径（architecture/pipeline/innovations/registration_qc）。
**输出**：<cwd>/output/patent_disclosure/SLMF_BBDM_子宫内膜癌CT_to_PET合成专利交底书.docx（相对路径，应在 wuzhe/ 下运行）。
**处理过程**：
1. 初始化文档样式：Normal 宋体 11pt、行距 1.25、页眉页脚。
2. 绘制封面：标题、副标题、信息表格（发明名称/技术方向/核心模型/交底重点/适用对象）与"交底说明"提示框。
3. 依 16 节顺序写正文：摘要、背景、SLMF-BBDM 方案（布朗桥、Gabor 频率保护、Zero-Conv 先验注入、SUV 阶段化损失）、模块清单、损失公式、权利要求 1–9、部署/质控/产业场景、结论。
4. 中途用 add_figure 插四张图并配图注，doc.save(OUT_DOCX) 返回路径。

### wuzhe/integrate_scheme_into_innovations.py

**职责**：专利交底书第一轮结构改写：把独立的"技术方案"章节内容合并进两个创新点（数据配准/HU/SUV 归一化并入创新点一，多尺度编码/方向频率/器官先验并入创新点二），移动图 4 位置并重排图号，删除孤立章节。
**接口**：set_east_asian_font(run, font="SimSun")；insert_paragraph_after(paragraph, text, style_name="Normal")；replace_paragraph_text(paragraph, text)；remove_paragraph(paragraph)；main()（无 CLI 参数）。
**依赖**：内部: 无代码依赖；外部: python-docx（Document、OxmlElement、qn）、copy.deepcopy、pathlib。
**输入**：output/patent_disclosure/slmf_bbdm_patent_prior_art_optimized.docx（SRC，相对 cwd）。
**输出**：output/patent_disclosure/slmf_bbdm_patent_innovation_integrated.docx。
**处理过程**：
1. 打开源 docx，按硬编码段落索引（29/37）在两个创新点末尾分别插入 3/4 段补充文字（HU/PET 归一化公式、器官分区通道、浅深层条件分工、C_0/C_l/Δh_l 公式等）。
2. 按图注文本定位"图4 配准质控图"，将其图与注移动到创新点一新增段落之后。
3. 删除"技术方案"标题到"7.损失函数"之间的独立章节，并把后者的标题去掉编号。
4. 用映射表统一改写四处图注与图说明（图4→图1 顺移）。
5. doc.save(OUT) 并打印路径。

### wuzhe/integrate_losses_into_two_innovations.py

**职责**：专利交底书第二轮改写：把独立"损失函数与公式化描述"章节拆解并入两个创新点标题与正文（创新点一加小病灶高摄取-频域联合损失，创新点二加 ROI-SUV/器官规则/冷区假热点/异方差损失），并同步改写有益效果、权利要求 1/6 与权利布局文字。
**接口**：set_east_asian_font；replace_paragraph_text；insert_after(paragraph, text, style_name)；remove_paragraph；find_para(doc, exact)（按精确文本找段落，找不到抛 ValueError）；main()。
**依赖**：内部: 无代码依赖；外部: python-docx、pathlib。
**输入**：output/patent_disclosure/slmf_bbdm_patent_innovation_integrated.docx（上一轮产物）。
**输出**：output/patent_disclosure/slmf_bbdm_patent_two_innovations_with_losses.docx。
**处理过程**：
1. 改写两个创新点标题，把损失机制写入标题本身。
2. 用 find_para 定位两处锚段，在其后各插入 4 段损失公式文字（L_base/L_top（TopK+focal）/L_freq/L_NCE；L_SUV/L_organ/L_false/L_NLL 与 L_total 门控式）。
3. 删除"损失函数与公式化描述"整章（至"有益效果"前）。
4. 按前缀匹配批量替换 6 处下游文字（有益效果、图4 说明、权利要求 1 与 6、权利布局、总结段），使全文与"两创新点+绑定损失"的新框架一致。
5. 保存新 docx。

### wuzhe/revise_patent_avoid_prior_art.py

**职责**：专利交底书"规避现有技术"改写轮：按段落索引 REPLACEMENTS 字典整段替换约 40 处文字，把可保护点从"公开模块本身"改写为"组合限定关系"（频带桥式噪声 ρ_b(p)、器官代谢规则、时间门控零初始化适配、SUV 阶段化误差重分配），并重写权利要求 1–9。
**接口**：find_source_docx() -> Path（在 output/patent_disclosure/ 下按 mtime 倒序找首段以"初步名称"开头的 docx，找不到抛 FileNotFoundError）；replace_paragraph_text(paragraph, text)（保留首 run 样式并强制东亚字体 SimSun）；main()。
**依赖**：内部: 无代码依赖；外部: python-docx（Document、qn）、copy.deepcopy、pathlib。
**输入**：自动发现的当前交底书 docx（SRC，模块加载时即解析）。
**输出**：output/patent_disclosure/slmf_bbdm_patent_prior_art_optimized.docx。
**处理过程**：
1. find_source_docx 按修改时间倒序尝试打开候选 docx，首段以"初步名称"开头者判定为当前版本。
2. 定义 {段索引: 新文本} 的 REPLACEMENTS（覆盖标题、技术领域、背景、步骤 3–6、两个创新点全部公式段、模块说明、损失、权利要求 1–9、扩展实施方式与验证/消融方案）。
3. 逐条 replace_paragraph_text 按索引整段替换。
4. 保存到 OUT。

### wuzhe/output/patent_disclosure/tighten_allowability_revision.py

**职责**：专利交底书"可授权性收紧"终轮：按段落索引替换约 30 处文字，使描述与代码实现严格一致——前向噪声只由频带倍率与方向频率能量调制，热点候选/器官分区/SUV 移入条件注入与阶段化损失，不再声称进入加噪公式。
**接口**：set_paragraph_text(paragraph, text)（clear+add_run，不保留原 run 样式）；main()（索引越界抛 IndexError）。
**依赖**：内部: 无代码依赖；外部: python-docx、pathlib。
**输入**：output/patent_disclosure/slmf_bbdm_patent_two_innovations_with_losses.docx（SRC，相对 cwd）。
**输出**：output/patent_disclosure/slmf_bbdm_patent_allowability_checked.docx。
**处理过程**：
1. 定义 {段索引: 新文本} 的 REPLACEMENTS，逐段改写技术领域/背景/步骤 4/噪声设计/损失门控/权利要求等，措辞与前一轮的差别在于"热点候选与器官分区不直接改写前向噪声"。
2. Document(SRC) 打开，逐条 set_paragraph_text 替换（索引超界即报错以防错位）。
3. 保存 OUT 并打印路径。

### wuzhe/main_data/visualize_split_quality.py

**职责**：分析并可视化 main_data 患者级 8:2 划分质量——患者数/切片数/三类肿瘤面积层的验证集占比是否接近 20%、患者是否泄漏跨 split、各模态文件是否齐备，产出 HTML 报告与两个 CSV。
**接口**：read_csv/write_csv/to_int/safe_pct/pct_text/esc 工具；infer_area_classes(split_rows)；derive_patient_rows(split_rows, area_classes)；summarize(patient_rows, area_classes)；patient_leakage(split_rows)->List[str]；split_file_names；modality_file_report(data_root, split_rows)；slice_count_distribution；patient_slice_strata(patient_rows, bins=5)；bar_svg/ratio_svg/table/badge（SVG+HTML 生成）；build_analysis(data_root, expected_val_ratio, low_slice_threshold)->Dict；build_report(...)->str；parse_args()；main()。
CLI：python main_data/visualize_split_quality.py [--data-root <默认脚本目录>] [--output ...] [--expected-val-ratio 0.2] [--low-slice-threshold 2]。
**依赖**：内部: 无；外部: 仅标准库（argparse/csv/html/collections/pathlib/typing）。
**输入**：data-root 下 split.csv、patient_summary.csv、split_summary.csv，以及 train/<modality>/val/<modality> 目录（如已物化）。
**输出**：split_quality_report.html、split_quality_summary.csv（criterion/train/val/total/val_ratio）、split_quality_checks.csv（check/value/target/absolute_deviation/status）。
**处理过程**：
1. 读入三个 CSV，推导每患者的面积类（small/medium/large）与切片归属。
2. summarize 统计 train/val 的患者数、切片数与各类切片数，计算 val 占比。
3. patient_leakage 检查同一患者是否出现在两个 split；modality_file_report 核对 ct/pet/label 文件名集合一致性。
4. 构造占比检查行（目标 20%、偏差与 ok/warn/badge 状态）。
5. bar_svg/ratio_svg 生成内联 SVG 图，build_report 拼完整 HTML（含配色常量）。
6. 写报告与两个 CSV，打印输出路径。

### wuzhe/wuzhe_data/split_5fold_three_area_balanced.py

**职责**：患者级三面积分层五折划分器：读取 data/origin_data/label 下掩膜 PNG 统计白区面积并三等分位分类，用"贪心构造+成对交换"最小化代价函数，使每折在患者数、切片数、三个面积类、切片数分箱与低切片患者数上都接近 20%，并物化 fold 目录与清单 CSV。
**接口**：@dataclass SliceRecord/PatientRecord（n_slices/class_counts/total_area/mean_area 属性）；read_mask_area(mask_path, threshold)（优先 PIL，缺省走 read_png_area_stdlib——用 zlib/struct 手写 PNG 解码 + unfilter_row + paeth_predictor）；extract_patient_id(file_name, regex)；assign_area_classes(items)（等频分位）；build_patient_records；squared_relative_error(value, target)；make_slice_count_bins(patients)；fold_cost(folds, all_patients, slice_count_balance_weight)（权重 3/5/8×相对误差平方 + 切片分箱与低切片项）；make_folds(patients, seed, repeats, swap_rounds, ...)（多次随机重启取最优）；improve_by_swapping（成对患者交换下降法）；infer_modalities；write_csv；patient_fold_map；write_outputs（slice_manifest/patient_summary/fold_summary/fold_N.csv）；link_or_copy(src, dst, mode)；materialize_files；print_summary；default_paths()；parse_args()；main()。
CLI：python wuzhe_data/split_5fold_three_area_balanced.py [--data-root data/origin_data] [--label-dir label] [--output-root wuzhe_data] [--patient-regex "^(\d{3})"] [--threshold 127] [--seed 42] [--repeats 30] [--swap-rounds 300] [--slice-count-balance-weight 1.0] [--modalities ...] [--materialize copy|hardlink|none]。
**依赖**：内部: 无；外部: 标准库（argparse/csv/math/os/random/re/shutil/struct/zlib/dataclasses/pathlib/typing；PIL 可选）。
**输入**：origin_data 的 label 目录掩膜 PNG 与各模态目录（ct/pet 等）。
**输出**：wuzhe_data/origin_wuzhe_data/fold_1..5（原始验证折）与 wuzhe_data/fold_1..5（每折内 train/val 的组合数据集），以及 slice_manifest.csv、patient_summary.csv、fold_summary.csv、fold_N.csv 清单。
**处理过程**：
1. 扫描 label 文件，用患者正则提取患者 ID，read_mask_area 统计阈值以上白区像素面积。
2. assign_area_classes 按面积等频分三档 small/medium/large，聚合出 PatientRecord。
3. make_folds：repeats 次重启——按切片数/最大类计数排序后逐患者贪心放入代价最低折，再 improve_by_swapping 做成对交换局部搜索，保留总 fold_cost 最低方案。
4. write_outputs 写各清单 CSV（含患者→折映射）。
5. infer_modalities 找出包含全部 label 文件名的子目录，materialize_files 按 copy/hardlink/none 把各折 train/val 文件物化到 fold_N/train|val/<modality>/。
6. print_summary 打印每折患者/切片/类分布，输出根路径。

### wuzhe/wuzhe_data/visualize_5fold_split_quality.py

**职责**：分析并可视化五折划分质量：逐折检查患者数/切片数/三个面积类的验证占比是否接近 20%、患者是否跨折重复、各折模态文件是否齐备，产出 HTML 报告与两个 CSV（与 main_data 版报告同风格）。
**接口**：read_csv/write_csv/to_int/esc/pct/ratio；derive_fold_summary(data_root)（从 fold_N.csv 汇总）；fold_rows_by_split(summary_rows)；patient_fold_assignments(data_root)->Dict[patient, set[fold]]（泄漏检测）；modality_file_report(data_root)；bar_svg(labels, values, title, target)；grouped_bar_svg(labels, series, title)；table/badge；analyze(data_root, expected_val_ratio)->Dict；build_report(...)->str；parse_args()；main()。
CLI：python wuzhe_data/visualize_5fold_split_quality.py [--data-root <默认脚本目录>] [--output ...] [--expected-val-ratio 0.2]。
**依赖**：内部: 无；外部: 仅标准库（argparse/csv/html/collections/pathlib/typing）。
**输入**：data-root 下 slice_manifest.csv、patient_summary.csv、fold_summary.csv、fold_1.csv..fold_5.csv，及 fold_N/train|val、origin_wuzhe_data/fold_N 目录。
**输出**：fivefold_quality_report.html、fivefold_quality_summary.csv（fold/patients/patient_ratio/slices/slice_ratio/class_0..2_slices 及 ratio/patients_le_2_slices 等 13 列）、fivefold_quality_checks.csv。
**处理过程**：
1. 读 fold_summary.csv 等清单，按折聚合 train/val 的患者与各类切片数。
2. patient_fold_assignments 检查同一患者是否被分到多个折（患者级泄漏）。
3. modality_file_report 核对每折各模态文件集合与 label 的一致性。
4. analyze 生成占比行与检查行（目标 20% + 偏差 + ok/warn/bad 状态）。
5. bar_svg/grouped_bar_svg 生成内联 SVG（含目标线），build_report 拼 HTML。
6. 写报告与两个 CSV，打印输出路径。

### 模块依赖小结

**本组 import 的内部模块**：
- 唯一的 src 引用是 wuzhe/comparison_experiments/train_comparison.py 的 from src.model.config_utils import apply_dotlist_overrides, resolve_runtime_profile, save_resolved_config。注意其 PROJECT_ROOT = Path(__file__).resolve().parents[1] 即 wuzhe/，sys.path 插入的是 wuzhe/ 而非仓库根，因此该 import 实际解析到 wuzhe/src/model/config_utils.py（该文件存在），不是根 src/model/config_utils.py。
- 其余全部文件零内部 src 依赖：comparison_experiments 内部仅互相引用（models ← trainers/train_comparison/evaluate_comparison，png_dataset ← 两个入口）；run_all_png_experiments.py 以 subprocess 调用两个入口脚本；专利脚本链只读写 docx 中间产物；三个数据脚本纯标准库。
- grep 查证：根 src/ 下没有任何文件引用 comparison_experiments 或 wuzhe（两条 grep 均为空），即本组是自包含子工程，不被主工程反向消费；其消费者只有自身的 CLI/ps1 入口与专利脚本链（build → revise → integrate_scheme → integrate_losses → tighten 串联消费 docx 版本）。

**wuzhe/src 与根 src 的关系（Get-FileHash SHA256 抽样比对，仅三对，不逐文件展开）**：
- src/model/trainer.py（93,039 B）vs wuzhe/src/model/trainer.py（11,957 B）：哈希不同；wuzhe 版是仅约 1/8 体量的"performance-optimised"精简变体，逐行 diff 约 1,951 行，实质是两个不同实现。
- src/model/slmf_bbdm.py（108,725 B）vs wuzhe/src/model/slmf_bbdm.py（41,572 B）：哈希不同；wuzhe 版约为根版 38% 体量的早期快照，diff 约 1,431 行。
- src/data/dataset.py（35,020 B）vs wuzhe/src/data/dataset.py（38,512 B）：哈希不同；体量相近（wuzhe 版反而略大），diff 约 398 行，属小规模分叉。
- 结论：wuzhe/src 并非根 src 的逐字节历史拷贝，而是已显著分叉的旧版/精简版快照（模型与训练器大幅缩水、dataset 略有差异）。在本子工程内运行时实际生效的是 wuzhe/src 这一份。


---

## 14. 测试套件 tests/（上：数据/频域/机制与管线测试）
### 模块概述

`tests/` 采用 pytest 约定式布局（各文件自行 `sys.path.insert` 指向仓库根），以小型合成张量（32×32 方块/随机图）做"契约式"单元测试：不依赖真实数据，而是固化模块的数学行为、路由开关语义与管线脚本的编排逻辑。本部件覆盖上半部分：PNG 基线与网络结构、频域损失族、H2-H4 机制验证、V2 云端/推理管线及先验锚定实验 runner。

### 基线模型与结构契约

### `tests/test_png_baseline.py`

- **被测对象**：`src/model/slmf_bbdm.py`（SLMFBBDM）与 `src/model/config_utils.py` 的 `validate_png_baseline_config`。
- **关键用例**：`TestGaborRouting.test_routes_parsed_from_config`/`test_routes_default_to_false`/`test_inject_adapter_false_passes_none_to_adapter` 验证 Gabor 四条路由（inject_adapter/use_for_noise/use_for_hotspot/use_for_loss）被真实解析并独立生效；docstring 另列出 sample() 返回终端 pred_x0、HotspotPriorLoss mask_only 排除掩码外热点、PNG 基线无 OrganPrior 可训练参数、配置校验拒绝 SUV/organ/metadata 激活、全路由关闭时 forward+sample 干净运行。
- **揭示的契约**：PNG-only 基线必须"纯净"——Gabor 路由默认全 False 且关闭时物理上不注入特征（gabor_feat=None 直达 adapter）；配置层强制 PNG 模式与 SUV/器官/元数据输入互斥。

### `tests/test_wavelet_unet.py`

- **被测对象**：`src/model/wavelet_unet.py`（WaveletDownsample/WaveletUpsample/WaveletBBDMUNet）。
- **关键用例**：`test_wavelet_downsample_mixes_all_subbands_and_backpropagates`、`test_wavelet_upsample_expands_to_four_subbands_and_backpropagates` 验证 Haar 子带混合层的形状与梯度有限性；`test_complete_wavelet_unet_preserves_public_shapes_and_skip_contract` 固化 4 级 skip_injections 形状契约（3 个 Down + 3 个 Up）；`test_complete_wavelet_unet_scale_changes_do_not_call_interpolate` 用 monkeypatch 禁用 `F.interpolate` 证明尺度变化全由小波变换完成。
- **揭示的契约**：尺度升降频只允许走小波路径（禁止 interpolate），公开接口形状与 skip 注入通道数不可漂移。

### 边界/频域损失契约

### `tests/test_boundary_frequency.py`

- **被测对象**：`src/model/loss_terms/boundary_frequency.py`（BoundaryFrequencyLoss）及 `src/model/interfaces.py` 的 ConditionBundle/LossContext。
- **关键用例**：`test_shifted_lesion_boundary_costs_more_than_aligned_prediction`（偏移预测比对齐预测代价更高）、`test_boundary_loss_uses_predicted_x0_not_noisy_model_tensor`（损失只依赖 pred_x0，对 model_pred 噪声不变）、`test_empty_organ_is_unavailable_but_nonempty_organ_activates_boundary`、`test_ct_only_edge_is_not_treated_as_pet_anatomy_consensus`（PET 全黑时 CT 边缘不计入解剖一致性）；后半段还有 `test_bridge_reliability_suppresses_boundary_supervision_at_high_noise`（高噪声桥可靠度抑制监督）、`test_frequency_gate_tv_loss_consumes_differentiable_router_scalar`（门控 TV 损失消费可微路由标量）、`test_boundary_metrics_reward_aligned_lesion_and_anatomy_edges`、`test_directional_spectrum_compares_prediction_to_target_not_isotropy`（方向谱对齐目标而非各向同性）。
- **揭示的契约**：边界频率监督以掩码可用性门控（lesion/organ_available 日志位），解剖一致性需 PET 与 CT 双边证据，且输入必须是去噪后的 pred_x0 而非带噪预测。
### `tests/test_normalized_lesion_peak.py`

- **被测对象**：`src/model/loss_terms/normalized_lesion_peak.py`（NormalizedLesionPeakLoss）。
- **关键用例**：`test_peak_loss_selects_prediction_and_target_topq_independently`（pred/target 各自独立取 top-q 峰值）、`test_peak_loss_clamps_k_between_minimum_and_maximum`（k 截断在 [min_k,max_k]）、`test_empty_masks_are_excluded_without_diluting_valid_sample`（空掩码样本剔除不稀释均值）、`test_tau_gate_is_applied_per_sample_before_valid_mean`（tau≤active_tau_max 逐样本门控）、`test_peak_loss_uses_pred_x0_and_backpropagates`、`test_peak_loss_rejects_invalid_configuration`。
- **揭示的契约**：峰值监督基于 pred_x0 可微反传，仅在低噪声（小 tau）时间步激活，k 值按病灶大小自适应钳位，无效样本按 valid_count 排除。

### `tests/test_reliable_frequency.py`

- **被测对象**：`src/model/frequency/boundary_reliable.py`（BoundaryReliableFrequencyInjector）及 `src/model/frequency/haar.py`。
- **关键用例**：`test_uses_exactly_two_haar_levels_and_decoder_native_shapes`（恰好两级 Haar、注入形状对齐解码器、最深层注入为全零）、`test_l0_is_subband_aligned_inverse_reconstruction_with_zero_lowpass`、`test_gates_are_finite_bounded_and_noise_reliability_tracks_bridge_snr`（门限幅 [0,gate_max] 且可靠度随桥 SNR 单调）、`test_ct_soft_reliability_uses_energy_not_signed_coefficients`（符号翻转不变）、`test_ct_reliability_floor_is_monotonic_and_preserves_exact_endpoints`、`test_gabor_quadrature_energy_only_changes_directional_reliability`、`test_zero_initialized_skip_residuals_are_exact_noops_but_learnable` 等。
- **揭示的契约**：频域注入严格两级 Haar（禁止虚构第三级）、L0 用零低通逆变换重建；门控有界且由桥 SNR/CT 能量可靠度驱动；零初始化 skip 残差是"精确 no-op 但可学习"的恒等初始化约定。

### `tests/test_peak_diagnostics.py`

- **被测对象**：`scripts/evaluate.py` 的 `compute_normalized_lesion_metrics`。
- **关键用例**：`test_normalized_peak_metrics_report_signed_and_topq_errors`（签名偏差/TopQ 误差的数值精确断言）、`test_core_ring_metrics_locate_center_peak_and_preserve_target_relation`（核心-环带峰值梯度）、`test_single_pixel_lesion_uses_core_fallback`、`test_normalized_peak_metrics_report_positive_cold_bias_magnitude`（冷偏差取幅值）、`test_empty_lesion_returns_nan_for_new_peak_diagnostics`、`test_peak_diagnostics_reject_invalid_topq_configuration`。
- **揭示的契约**：评估指标在 [-1,1] 模型空间计算，空病灶返回 NaN 而非 0，单像素病灶走 core_fallback 路径，topq 配置非法即抛错。
### 频域消融与云端数据门控

### `tests/test_frequency_ablation_runner.py`

- **被测对象**：`scripts/run_frequency_ablations.py`（消融编排 runner：硬门控/复合分/计划清单/晋升命令）及各频域 preset 的模型构建。
- **关键用例**：`test_metric_value_accepts_flat_evaluator_keys_and_rejects_missing`、`test_hard_gates_compare_candidates_to_r0_reference`（相对 R0 参考的门控）、`test_hard_gates_support_absolute_limits_without_reference_metric`、`test_composite_score_prefers_lesion_fidelity_without_ignoring_image_quality`、`test_v4_composite_score_uses_robust_topq_peak_when_available`；随后是 v1→v8 演进的清单契约：`test_dry_run_manifest_contains_all_screen_blocks_and_no_promotions_yet`、`test_v2_manifest_runs_mean_first_and_applies_common_train_overrides`、`test_v3_plan_reuses_frozen_mean_and_pins_fixed_promotions`、`test_v4_plan_has_two_fixed_64_sample_stages_and_exact_promotions`、`test_v5_plan_pins_two_stages_gates_and_exact_final_promotion` / `test_v5_dry_run_manifest_is_exact_deterministic_and_v5_only`（含证据命名空间、精确 epoch 校验、300-epoch 参考）、`test_v6_separates_mechanism_screen_from_endpoint_gates`、`test_v7_cold_bias_plan_is_a_common_checkpoint_causal_screen`、`test_v8_final_calibration_uses_v7_ema_and_independent_confirmation_seed`；另有 `test_every_boundary_reliable_preset_completes_a_real_forward`、`test_every_frequency_preset_completes_a_real_model_forward`、`test_zero_initialized_br_b_is_initially_equivalent_to_paired_r2` 等真实前向冒烟。
- **揭示的契约**：消融晋升采用"产物门控"两段式——先硬门（安全指标不劣化）再复合分（病灶保真优先）；每个版本的计划清单必须确定性、精确固定晋升与 epoch，dry-run 产物即实验协议的可审计快照。

### `tests/test_cloud_stage0c_gate.py`

- **被测对象**：`scripts/run_cloud_stage0c_gate.py`（Stage-0C 缓存校验门）与 `src/data/lineage.py`（attach/load 数据血缘）。
- **关键用例**：`test_cache_seal_and_checkpoint_lineage`（构造 8×8 PNG 原始数据→192×192 缓存 npz+sidecar，验证双射/内容校验与 lineage 封印，`pet_matches_noninverted` 必须为假）、`test_raw_root_safe_autodiscovery`、`test_main_png_cache_task_is_locked_to_contract_preprocessing`（pixi 任务锁定契约预处理）、`test_strict_training_lineage_is_embedded_and_auditable`。
- **揭示的契约**：云端训练只允许消费通过 Stage-0C 门（缓存与 raw PNG 双射 + 内容一致 + 契约 sha256）的数据；PET 反转、预处理尺寸等均写入可审计血缘并在 checkpoint 内封印。

### 机制假设验证（H2/H3/H4）

### `tests/test_h2_mechanism.py`

- **被测对象**：`scripts/validate_h2_pathology_excluded_residual.py`（含/不含病灶掩码的残差富集损失与决策门）与 `src/mechanism_validation/common.py`（split_manifest 读取与患者级划分）。
- **关键用例**：`test_formal_partition_reproduces_h1_patient_counts`（99/25/31 患者计数复现 H1 划分）、`test_excluded_loss_ignores_dilated_pathology_support`（膨胀病理支撑区被排除后损失≈0）、`test_zero_mask_makes_included_and_excluded_losses_equal`（空掩码退化为 included）、`test_h2_gate_uses_patient_paired_validation_and_calibration_margin`（calibration 定阈值→validation 成对判 PASS 与非劣性边际）。
- **揭示的契约**：H2 假设（频域收益来自病理区而非全局）以"排除病理支撑区后富集消失 + 边界梯度非劣"为判据，划分必须患者级且与 H1 一致。
### `tests/test_h3_h4_mechanisms.py`

- **被测对象**：`scripts/validate_h3_logsnr_recoverability.py`（log-SNR 可恢复性交叉点）与 `scripts/validate_h4_noise_band_calibration.py`（噪声分段证据探针）。
- **关键用例**：`test_crossing_uses_fixed_monotone_envelope`（单调包络上取首个上穿点，未达阈值返回 NaN）、`test_calibration_selects_pair_and_direction_without_validation`（仅用 calibration 患者选 (task,direction) 对）、`test_h4_evidence_probe_improves_over_logsnr_baseline`（合成数据上校准技能>0.5、增量 MSE 技能 CI 下界>0、置换 null 有限）。
- **揭示的契约**：H3 假设以 log-SNR 阈值交叉刻画可恢复性；H4 假设要求噪声校准证据在统计上优于纯 log-SNR 基线——两者均坚持"校准集选参、验证集报告"的患者级防泄漏纪律。

### `tests/test_formal_mechanism_pipeline.py`

- **被测对象**：`src/model/frequency/spectral_router.py`（SpectralEvidenceFrequencyRouter，H1 机制载体）、`scripts/evaluate.py` 的 `compute_target_relative_false_hotspots`、机制管线规格完整性。
- **关键用例**：`test_h1_ct_support_is_bounded_and_shared_across_pet_bands`（CT 支撑场有界 [0,1] 且跨 PET 子带共享）、`test_support_only_mode_removes_ct_from_learned_evidence`（support-only 模式下证据对 CT 幅度不变）、`test_no_recoverability_zeroes_h4_evidence_and_logsnr`（use_noise_release=False 时证据与 log-SNR 全零）、`test_target_relative_hotspots_use_target_not_prediction_percentile`、`test_formal_pipeline_has_no_unimplemented_stage`。
- **揭示的契约**：正式机制管线中 CT 的角色被限制为"支撑场/门控"而非可学习证据；假热点以目标（而非预测）分位定义；管线规格必须全部 implemented。

### `tests/test_mechanism_validation_runner.py`

- **被测对象**：`scripts/run_all_mechanism_validations.py`（机制验证总编排）与 `scripts/run_mechanism_stage0_data_audit.py`。
- **关键用例**：`test_repository_pipeline_spec_is_valid`（configs/mechanism_validation_pipeline_v1.json 覆盖 H1-H6 且顺序固定）、`test_decision_requires_exact_locked_contract`（decision 内 dataset_contract_sha256 必须与锁定契约完全一致）、`test_decision_profile_assertions_are_enforced`、`test_raw_root_selection_is_portable_between_local_and_cloud`、`test_incomplete_all_pipeline_fails_before_running_commands`（scope=all 有未实现阶段则在执行前失败）、`test_failed_gate_blocks_downstream_stage`、`test_from_stage_requires_and_accepts_verified_canonical_prerequisite`。
- **揭示的契约**：机制验证管线是"契约锁定 + 阶段门"的有向无环编排——上游 gate 失败或契约不匹配即阻断下游，跨本地/云端物理路径可移植。
### `tests/test_h4_v2_mechanism.py`

- **被测对象**：`src/mechanism_validation/h4_v2.py`（H4 二代统计：分层校准/比较器/封印探针/内部 CV）与 `src/data/dataset.py` 的 CachedDataset。
- **关键用例**：`test_h4_v2_context_and_abstention_fallback_are_deterministic`（置信度低于阈值时回退到 H3 固定调度且 router_abstained=1）、`test_h4_v2_frozen_probe_detects_mutation`（seal_probe 自哈希，篡改即 ValueError）、`test_cached_dataset_uses_authoritative_manifest_patient_and_slice`、`test_internal_cv_partition_is_patient_only_complete_and_deterministic`、`test_internal_cv_sealed_mapping_detects_mutation`。
- **揭示的契约**：H4-v2 采用"不确定性感知 + 弃权回退"的路由决策，统计产物（探针/CV 映射）必须自封印防篡改；患者/切片归属以权威 manifest 为准而非缓存文件名。

### 均值预训练与残差频域主线

### `tests/test_mean_pretraining.py`

- **被测对象**：`src/model/mean_pretraining.py`（MeanPretrainer/mean_target_ll2/mean_charbonnier_loss/病理排除损失/检查点指纹）、`src/model/mean_predictor.py` 与 CLI 脚本。
- **关键用例**：`test_mean_target_is_exact_pet_ll2`（目标=两级 Haar 低通 LL2 的精确恒等）、`test_mean_charbonnier_loss_is_finite_and_zero_error_equals_epsilon`、`test_mean_pretrainer_updates_predictor_and_writes_versioned_checkpoints`（mean_best/last.pt 含 format_version=1 与 history.json）、`test_mean_pretrainer_requires_validation_loader`、`test_mean_pretraining_cli_runs_fake_data_and_writes_all_artifacts`、`test_pathology_excluded_mean_loss_ignores_error_inside_support`、`test_mean_pretrainer_writes_fingerprint_sidecar_with_sha256_and_policy`、`test_write_checkpoint_fingerprint_is_fail_closed_for_missing_file`。
- **揭示的契约**：低频均值预训练是独立可复现的阶段——Charbonnier 损失、版本化 checkpoint、sha256 指纹 sidecar 与病理排除策略必须完整落盘；指纹缺失文件时 fail-closed。

### `tests/test_residual_frequency.py`

- **被测对象**：`src/model/slmf_bbdm.py`（`_bbdm_ddim_step`、V5 残差桥配置路由、冻结均值加载）、`src/model/frequency/haar.py`、`src/model/mean_predictor.py`、`src/model/frequency/gabor.py`、残差频域前置器与损失项。
- **关键用例**：`test_bbdm_ddim_step_preserves_inferred_noise_trajectory`/`test_bbdm_ddim_step_does_not_drop_nonzero_noise`（DDIM 步保持推断噪声轨迹）；`TestHaarTransform`（两级往返精确、低通重建细节为零、奇数尺寸拒绝）；`TestLowFrequencyPETPredictor`（只输出 LL2 低通）；`TestComplexGaborDescriptor`（正交幅度对相位旋转不变、旧 checkpoint 键严格加载）；`TestResidualFrequencyPreconditioner`（零初始化注入匹配解码器形状、门独立非 softmax、gabor 因子从恒等出发有界）；`TestResidualBBDMIntegration`（V5 拒绝非规范 DCT 配置、boundary_reliable 模式保留 organ/ROI SUV 接口、冻结均值严格加载且训练模式不更新、非法模块组合构造期失败、`test_residual_bridge_forward_and_sampling_reconstruct_pet`）；`TestResidualFrequencyLosses`（谱路由 active-mass 惩罚、残差小波损失病灶加权且无上下文时 no-op、Gabor 一致性匹配目标方向而非各向同性）。
- **揭示的契约**：残差频域主线 = 冻结的条件均值低频 + 仅作用于残差的频域监督；桥采样数学、恒等初始化 no-op、旧权重兼容与"损失必须挂在真实路由上"是被固化的核心行为。
### V2 云端管线与推理准入审计

### `tests/test_v2_cloud_stage03_runner.py`

- **被测对象**：`scripts/run_cloud_v2_stage03.ps1`（云端 Stage-0.3 训练编排脚本，做文本断言）。
- **关键用例**：`test_cloud_stage03_runner_is_fail_closed_and_stage_limited`（脚本必须包含冻结完整性审计、缓存血缘、数据集契约、均值重训、checkpoint 血缘、病理排除开关、sha256，以及 `next_stage_allowed = $false`/`production_cutover_allowed = $false` 的 fail-closed 旗标；且不得包含 ct_support/curriculum/artifact_safety/h5_v2/h6_v2 等未授权 Stage）、`test_cloud_stage03_runner_refuses_checkpoint_overwrite_by_default`（默认拒绝覆盖已有 checkpoint，仅 `-AuditExisting` 开关放行）。
- **揭示的契约**：云端一键脚本被限制为单一阶段的可审计执行器——产物覆盖与阶段越权都被默认禁止。

### `tests/test_v2_inference_admissibility.py`

- **被测对象**：`scripts/audit_v2_inference_admissibility.py`（生产推理准入审计）及 `scripts/validate_h4_noise_band_calibration.py` 的生成器语义。
- **关键用例**：`test_h4_generator_semantics_expose_forbidden_inputs`（静态检查确认 H4 生成器读取 target PET/病灶掩码——属于生产推理不可用输入）、`test_frozen_bundle_fails_production_inference_admissibility`（冻结校准 bundle 因禁用输入被判定 FAIL 且 conflict_proven）、`test_admissibility_decision_is_new_and_does_not_overwrite_v1`（新决策写 05A_ 前缀且不碰 v1 产物）、`test_missing_frozen_artifact_fails_closed_without_scientific_claim`（缺产物时 FAIL 且 scientific_gate_evaluated=False）。
- **揭示的契约**：机制证据若依赖推理期不可得的输入（目标图/掩码），则禁止激活到生产路由；审计必须 fail-closed 并区分"科学门未评"与"科学门失败"。

### `tests/test_h3_fixed_schedule_inference.py`

- **被测对象**：`scripts/audit_h3_fixed_schedule_inference.py`（H3 固定调度推理审计），基于 sha256 封印的 decision.json 快照做回归。
- **关键用例**：`test_formal_gate_splits_grid_pass_from_two_independent_failures`（冻结输入完整性 PASS 与网格查找 PASS，但生产时间步消费未定义、映射不可辨识两路独立 FAIL）、`test_nineteen_of_twenty_production_eval_steps_are_off_grid`（20 步评估中 19 步落在冻结网格之外）、`test_h3_source_and_h4_nested_partition_lineage_are_distinct`、`test_unproven_fail_is_not_fully_evaluated_or_exit_zero`、`test_fresh_audit_fails_closed_after_runtime_source_advance`、`test_tampered_bundle_and_contract_fail_closed`、`test_all_downstream_actions_remain_disabled`、`test_cli_does_not_reclassify_runtime_integrity_failure`。
- **揭示的契约**：H3 固定调度进入生产前必须证明"评估时间步全部在已验证网格内且消费关系可定义"；封印决策的任何输入（bundle/契约/运行时源）被篡改或推进后重审都须 fail-closed，并禁用一切下游动作。
### `tests/test_h3_v2_full_timestep_native_null.py`

- **被测对象**：`scripts/calibrate_h3_v2_full_timestep_native_null.py` 与 `src/model/frequency/h3_native_null_schedule.py`（冻结全时间步"原生零假设"调度）及 SpectralEvidenceFrequencyRouter 的 h3_native_null 路由策略。
- **关键用例**：`test_schedule_loader_and_exact_native_null_lookup`（精确整数时间步查找）、`test_schedule_loader_rejects_file_hash_mismatch`/`_rejects_canonical_self_hash_mismatch`/`_rejects_timestep_count_mismatch`/`_rejects_nonmonotonic_band`/`_rejects_nonzero_shallow_policy`（五类非法调度拒绝）、`test_h3_router_requires_both_schedule_path_and_hash`、`test_h3_router_uses_exact_schedule_and_freezes_evidence_heads`、`test_h3_router_rejects_uncertainty_aware_h4_selector`、`test_synthetic_calibration_passes_and_never_allocates_shallow`/`_fails_when_band_constant_is_better`（合成校准的门判定）、`test_protocol_config_self_hash_and_runtime_source_hashes`。
- **揭示的契约**：H3-v2 调度是双重 sha256（文件+规范化自哈希）封印的只读产物；路由只允许精确时间步查表、浅层路由必须结构性为零、禁止挂接 H4 不确定性选择器；production_activation_allowed 恒为 False。

### `tests/test_h3_v2_overnight_runner.py`

- **被测对象**：`scripts/run_h3_v2_overnight.py`（过夜训练编排 OvernightRunner/KeepWindowsAwake）。
- **关键用例**：`test_overnight_plan_is_isolated_and_conditional`（hard_guards 全集：CUDA 必需、校准失败先停、所有产物限制在 run_dir 内、不评估 destination/artifact_safety/curriculum、禁止生产激活、不覆盖 05B；阶段顺序 04 校准→条件式 07/08 训练）、`test_skip_full_tests_is_explicit_in_plan`、`test_resume_requires_exact_run_directory`、`test_frozen_v1_accepts_only_visible_cuda_device_zero`、`test_scientific_controls_do_not_have_short_or_invalid_values`（epochs/shard/heartbeat 参数边界）、`test_runner_contract_freezes_cuda_selection_environment`、`test_resume_resource_preflight_preserves_initial_stage_artifact`、`test_targeted_tests_exclude_retired_h3_fixed_schedule_suite`。
- **揭示的契约**：过夜 runner 是带硬护栏的隔离实验执行器——写权限圈定在 run_dir、恢复必须指向原目录、CUDA 环境被冻结进 runner contract 以保证可复现。

### `tests/test_h3_png_prior_preview.py`

- **被测对象**：`scripts/estimate_h3_prior_from_png.py`（从 PNG 数据估计 H3 先验预览）。
- **关键用例**：`test_sufficient_statistics_match_explicit_noisy_haar_alignment`（充分统计量实现的掩码余弦对齐与显式逐带计算逐点一致）、`test_fixed_noise_identity_is_independent_of_batch_order`（固定噪声由 (sample_id,role) 哈希派生，与批序无关）、`test_direct_png_inventory_uses_manifest_roles_not_physical_folders`、`test_png_preprocessing_matches_locked_contract`、`test_png_root_auto_detection_falls_back_from_local_to_cloud_layout`、`test_isotonic_prior_uses_train_patients_and_is_non_increasing`（保序回归只用训练患者且非增）、`test_portable_path_never_serializes_machine_absolute_path`、`test_preview_payload_cannot_pass_formal_schedule_loader`。
- **揭示的契约**：PNG 先验只是"预览"产物——噪声注入按样本身份可复现、划分取自 manifest 而非目录结构、序列化路径必须可移植，且预览 payload 结构上无法冒充正式封印调度。
### 先验锚定路由（prior anchor）实验族

### `tests/test_prior_anchor_schedule.py`

- **被测对象**：`src/model/frequency/prior_anchor_schedule.py`（加载 DIRECT_PNG_PREVIEW / FORMAL_H3_V2 两种来源的先验锚定调度）。
- **关键用例**：`test_direct_png_preview_loads_only_with_explicit_lineage_opt_in`（预览调度必须显式 h3_allow_unverified_preview_lineage 才可加载）、`test_formal_source_delegates_to_existing_strict_h3_loader`（formal 来源委托给严格 H3 加载器且拒绝预览血缘覆盖）、`test_direct_png_preview_rejects_file_hash_mismatch`/`_rejects_schema_or_flag_mismatch`/`_rejects_invalid_six_by_t_curve`/`_rejects_nonportable_payload_paths`/`_file_must_remain_within_repository_root`/`_rejects_timestep_and_lineage_shape_mismatch`、`test_prior_anchor_schedule_rejects_unknown_source`。
- **揭示的契约**：预览调度与正式调度是严格隔离的两级信任体系——预览自带 preview_only/inference_schedule_allowed=False 等旗标、路径必须可移植且留在仓库内、六带×T 曲线与哈希全部校验，未验证血缘需显式 opt-in。

### `tests/test_prior_anchored_paired_ablation.py`

- **被测对象**：`configs/experiments/prior_anchored_paired_ablation_v1.yaml` 与 `slmf_png_prior_anchored_router_100e.yaml`（A/B/C/D 路由消融计划，用 apply_dotlist_overrides 物化）。
- **关键用例**：`test_four_variants_are_declared_and_policy_is_unique`（A_no_route=native_only、B_fixed_h3_native_null、C_h3_prior_anchored_adaptive、D_unconstrained_learned 四种策略一一对应）、`test_declared_overrides_are_scoped_to_the_router_block`（去掉路由块后四变体配置完全一致）、`test_pairing_contract_fields_match_base`（seed/manifest/cache/契约/epoch 等共享契约与基础配置对齐）、`test_variant_C_matches_the_cloud_base_config_exactly`（C 即 100e 云端主实验）、`test_variant_B_is_blocked_until_a_formal_schedule_exists`、`test_plan_marks_itself_exploratory_and_non_production`、`test_plan_names_all_known_pairing_blockers`。
- **揭示的契约**：消融计划 fail-closed——差异必须只来自路由块；在架构/初始化/数据顺序配对被证明之前，计划自标 exploratory、B 被封堵、禁止因果归因与生产激活。

### `tests/test_prior_anchored_router_100e_runner.py`

- **被测对象**：`scripts/run_prior_anchored_router_100e.py`（100-epoch 先验锚定路由训练 runner：预检/恢复/均值检查点/指标校验）。
- **关键用例**：`test_preflight_is_static_and_does_not_require_cloud_files`、`test_project_paths_reject_absolute_and_escaping_values`、`test_run_owned_config_uses_only_relative_paths`、`test_mean_checkpoint_selection_skips_non_excluded_candidate`、`test_resume_latest_selects_newest_incomplete_run`、`test_fresh_launch_never_reuses_directory_with_checkpoint`、`test_resume_does_not_silently_skip_corrupt_latest_checkpoint`、`test_resume_rejects_rng_tensor_that_cannot_be_restored`/`_rejects_full_config_identity_mismatch`/`_rejects_absolute_checkpoint_owned_resume_path`/`_rejects_lineage_that_differs_from_current_cache_metadata`、`test_resume_fails_closed_when_no_numbered_checkpoint_exists`、`test_complete_metrics_requires_exact_ordered_1_to_100` 等指标完备性校验、`test_fresh_setup_failure_is_persisted_in_runner_state`、`test_prior_artifact_requires_six_dense_curves`。
- **揭示的契约**：长跑 runner 的"可恢复性"被逐项固化——恢复要求 RNG 态、配置身份、缓存血缘三者与当前一致且全部可还原；完成判定要求 1..100 有序完整指标流与 checkpoint 对齐；先验产物必须含六条稠密曲线，任何缺口均 fail-closed。


---

## 14. 测试套件 tests/（下：路由/评估/特征与云协作测试）

### 模块概述

`tests/` 采用 pytest 约定式布局（各文件自行 `sys.path.insert` 指向仓库根），以小型合成张量与 tmp_path 产物做"契约式"单元/冒烟测试，不依赖真实数据、CUDA 或正式云端契约。本部件覆盖下半部分：先验锚定/频谱自适应路由的损失与路由器契约、路由修复与评估、训练机制（oracle destination、trainer 监控、medoid、感知 x0）、放大审计机制、特征涌现/溯源/因果分析，以及云协作与双实验编排。

### 路由（Prior-Anchored / Spectral Router）契约

### `tests/test_prior_anchored_router_loss.py`

- **被测对象**：`src/model/loss_terms/spectral_router.py`（SpectralRouterRegularizationLoss），经 `src/model/interfaces.py` 的 LossContext 驱动。
- **关键用例**：`test_legacy_terms_are_unchanged_when_new_inputs_are_absent`（legacy 项数值不受新输入影响）、`test_prior_anchored_terms_apply_phase_scales_and_boundary_masks`（anchor/monotonic/curvature/budget/shallow 五项的相位缩放与 t=0 边界掩码精确数值）、`test_missing_raw_tensors_make_each_optional_term_a_noop`、`test_active_mass_floor_is_disabled_for_prior_anchored_mode`、`test_boundary_masks_can_disable_neighbour_regularization`、`test_nonfinite_router_diagnostics_produce_finite_float32_loss_and_logs`、`test_new_weights_must_be_nonnegative`、`test_route_tensor_shape_contract_is_checked`、`test_masked_mean_preserves_fractional_weight_semantics`。
- **揭示的契约**：路由正则在 legacy 与 prior-anchored 双模式下数值精确可预测；缺张量=该项 no-op、非有限诊断 fail-safe 输出有限 float32；curvature 在无 prev 样本处按掩码剔除；诊断日志键 `{term.name}/xxx` 稳定。

### `tests/test_spectral_router.py`

- **被测对象**：`src/model/priors/gabor.py`（GaborPrior）、`src/model/frequency/dct_descriptor.py`（SelectedDCTDescriptor）与 `src/model/frequency/spectral_router.py`（BoundedAmplitudeHead/ConservativeRouteHead/不确定性感知选择器与路由器/SpectralEvidenceFrequencyRouter）。
- **关键用例**：Gabor 偏移能量零初始且可训练；DCT 描述子有限/频率权重归一/常数输入为零/不放大近常数输入/初始等于声明先验/非法配置抛 ValueError；`test_bounded_amplitude_head_starts_neutral_and_respects_limits`、`test_conservative_route_head_is_null_biased_and_sums_to_one`；不确定性选择器 `test_uncertainty_aware_selector_abstains_exactly_to_h3_schedule` 及形状校验/fail-closed/`test_uncertainty_aware_router_full_abstain_equals_fallback_exactly`/逐元素混合选择/同置信度确定性/`test_uncertainty_aware_router_gradient_only_through_active_routes`/与训练掩码无关；路由器侧 `test_router_first_forward_is_exact_noop_and_routes_are_conservative`、`test_native_warmup_and_ramp_gradually_release_learned_routes`、`test_hard_all_null_short_circuits_before_descriptors_and_projection`、半精度有限性三连（参考 dtype 保持/桥端点/零路由熵）、`test_router_normalizes_optional_gabor_evidence_to_reference_dtype`、`test_all_route_policies_emit_common_route_mass_diagnostics`、`test_router_receives_gradients_after_zero_projection_learns`。
- **揭示的契约**：所有自适应头"零初始、保守默认、null 偏置"；弃权语义=精确回退 H3 原生路由；dtype 跟随参考（CT）张量而非证据；零初始化投影不阻断梯度；路由质量诊断键跨策略统一。

### `tests/test_prior_anchored_spectral_router.py`

- **被测对象**：`SpectralEvidenceFrequencyRouter` 的 `route_policy="prior_anchored_learned"` 模式（monkeypatch `load_prior_anchor_schedule` 注入合成先验表）。
- **关键用例**：`test_prior_anchored_phase_schedule_matches_100_epoch_contract`（epoch 0-9 全禁、10-19 active 线性爬升、30-39 destination 释放、anchor 终值 0.10）、`test_warmup_is_exact_h3_native_null_and_bypasses_adaptive_heads`、`test_active_only_then_destination_release_are_separated`、`test_prior_endpoints_remain_exact_and_diagnostics_are_finite`、`test_route_masses_partition_one_and_match_phase_invariants`、`test_prior_phase_buffers_survive_state_dict_round_trip`、`test_verified_prior_is_not_checkpoint_owned_and_legacy_key_is_ignored`、`test_prior_forward_does_not_read_progress_buffers_with_item`、`test_shallow_projection_dezero_breaks_cold_start_deadlock` 与 `test_prior_anchored_router_dezeros_only_shallow_projections`。
- **揭示的契约**：训练相位由 `set_training_epoch` 驱动、端点精确不变；先验调度表不随 checkpoint 序列化（靠 sha256+lineage 外部校验）；warmup 期输出逐位等于 H3 原生 null 路由；冷启动死锁仅靠"浅层投影去零"化解且不触碰深层。

### `tests/test_prior_anchored_router_smoke.py`

- **被测对象**：先验锚定自适应路由的端到端 CPU 链路（config 载入→direct-PNG prior 载入→建模→一步训练→优化步→epoch 相位更新→模型 state 往返→四 epoch Trainer→metrics JSONL）。
- **关键用例**：`test_prior_anchored_router_end_to_end_on_cpu`（全链路合成张量冒烟，只写 tmp_path，不触碰正式 100-epoch 云契约）、`test_background_repair_freezes_router_and_forces_native_destination`、`test_smoke_prior_load_is_fail_closed_against_hash_tamper`（prior JSON 哈希被篡改必须拒绝）、`test_prior_destination_inference_interventions_preserve_availability`、`test_router_background_suite_runs_all_interventions_on_cpu`。
- **揭示的契约**：preview prior 的 JSON schema（schema_version/pipeline_id/decision=PREVIEW_ONLY/多层 sha256 指纹）是 fail-closed 契约——任何哈希或血缘不匹配即拒绝加载；探索链可在纯 CPU 合成数据上完整自证。

### `tests/test_summarize_prior_anchored_run.py`

- **被测对象**：`scripts/summarize_prior_anchored_run.py`（summarize_run/phase_for_epoch/read_metrics_jsonl）与 `scripts/run_prior_anchored_router_100e.py` 的 100-epoch 运行契约。
- **关键用例**：`test_phase_boundaries_are_unambiguous`、`test_partial_run_summary_reports_latest_and_missing_boundaries`、`test_epoch_100_checkpoint_alone_does_not_mark_run_complete`、`test_complete_requires_every_run_contract`、`test_invalid_runner_state_never_completes`、`test_complete_state_requires_metrics_path_and_hash`、`test_metrics_jsonl_rejects_non_finite_json_constants`、`test_metrics_steps_must_increase_strictly`、`test_metric_sections_allow_only_flat_finite_scalars_or_null`、`test_each_tenth_epoch_requires_eval_and_validation`、`test_phase_observations_require_canonical_route_masses`、`test_final_metrics_step_must_match_final_checkpoint`、`test_invalid_preview_contract_never_completes`、`test_configured_budget_must_be_exactly_100`、`test_metrics_jsonl_surfaces_trainer_nested_sections`、`test_summary_write_is_atomic_and_relative`、`test_summary_rejects_absolute_output_path`。
- **揭示的契约**："运行完成"是产物级谓词：metrics JSONL（仅有限标量、步严格递增、每 10 epoch 有 eval+validation、相位观测点含规范路由质量）+ 最终 checkpoint 步对齐 + metrics 哈希 + 预览契约 + 预算恰为 100 全部就位才判定完成；摘要写入原子且禁止绝对路径。

### `tests/test_eval_router_background_suite.py`

- **被测对象**：`scripts/eval_router_background_suite.py`（_load_fixed_destination/compute_body_background_metrics/compute_global_hotspot_metrics）。
- **关键用例**：`test_fixed_destination_is_read_from_requested_epoch_train_block`（从指定 epoch 的 train 块读取固定 destination，缺 epoch 即 RuntimeError）、`test_body_background_metrics_separate_low_and_high_frequency_error`（体部非病灶区的低通 MAE 与高通能量比分离度量）、`test_global_hotspot_metric_uses_global_not_mask_conditioned_peak`（全局热点命中不用病灶掩码条件化的峰值）。
- **揭示的契约**：背景评估锚定训练日志里记录的先验 destination 分布而非重新估计；体部背景按频段拆分误差；热点指标取全局峰以防"掩码内过拟合"式假阳性。

### 路由修复与 Oracle 干预

### `tests/test_identifiable_router_repair.py`

- **被测对象**：`src/model/slmf_bbdm.py`（SLMFBBDM 全模型）经 `_repair_config` 构建的"全可辨识分层路由修复"配置（destination bootstrap 概率/天花板、空间 destination 头、DCT 描述子、低频背景门、route_utility_supervision 损失），复用 smoke 测试的 `_smoke_config`/`_write_preview` 夹具。
- **关键用例**：`test_full_repair_has_bootstrap_spatial_utility_and_low_frequency`（损失有限、shallow 质量>0、L3 注入恒为零、空间 destination 精确 0.25 初始、utility 梯度抵达空间头）、`test_frequency_branch_interventions_are_exact_and_state_free`、`test_phase_schedule_prevents_router_projection_coadaptation`、`test_conditional_mean_can_be_recovered_from_full_model_checkpoint`、`test_route_utility_is_safe_under_bfloat16_autocast`。
- **揭示的契约**：路由修复必须在真实全模型（而非孤立模块）中自洽：频带干预精确且无状态泄漏；相位日程阻断路由与投影的共适应；条件均值可从 full_model checkpoint 恢复；bfloat16 autocast 下数值安全。

### `tests/test_oracle_destination.py`

- **被测对象**：`scripts/eval_oracle_destination.py`（enumerate_action_maps/action_map_code/select_oracle_rows/_resolve_conditional_mean_checkpoint）。
- **关键用例**：`test_action_map_enumeration_is_exact_and_stable`（2^6/3^6 动作映射枚举精确且编码稳定）、`test_missing_external_mean_self_bootstraps_from_evaluated_checkpoint`（外部 mean 缺失时回退到被评估 checkpoint 自举）、`test_oracle_selection_uses_complete_per_sample_action_maps`（逐样本按完整动作映射选 oracle，失败映射回退 LEARNED）、`test_per_band_route_action_intervention_is_exact_and_state_free`。
- **揭示的契约**：oracle 上限分析以"每样本完整动作映射"为原子而非全局均值；路由动作干预必须精确、可重复且不改动模型状态；条件均值解析存在自举回退路径。

### 训练机制与损失契约

### `tests/test_nonlesion_body_lowpass.py`

- **被测对象**：`src/model/loss_terms/nonlesion_body_lowpass.py`（NonLesionBodyLowpassLoss）与 `src/model/trainer.py` 的 `_compute_body_background_sample_metrics`。
- **关键用例**：`test_nonlesion_lowpass_loss_is_zero_for_exact_prediction`、`test_nonlesion_lowpass_loss_excludes_dilated_lesion`（膨胀病灶邻域整体剔除）、`test_upper_uptake_tail_receives_more_weight_and_gradient`（高摄取上尾加权且梯度有限）、`test_sampled_background_metric_matches_identical_pet`（采样监控指标对相同 PET 输出一致）、`test_background_repair_config_is_optimizer_fresh_and_transfer_bounded`。
- **揭示的契约**：背景尾部修复只监督"非病灶体区"的低频误差，病灶区按膨胀半径排除；上尾（高摄取方向）更强监督且对 pred 可微；训练器采样监控与损失口径保持一致。

### `tests/test_trainer_monitoring.py`

- **被测对象**：`src/model/trainer.py`（Trainer、_build_optimizer、_compute_pet_sample_metrics、_stratified_indices）的确定性监控、checkpoint 选择与断点续训。
- **关键用例**：`test_spectral_optimizer_groups_assign_learning_rates_and_no_decay`（projection/router/descriptor 分组学习率、bias/offset 免 weight_decay、参数全覆盖不重复）；签名空间 PET 指标三连（`test_signed_pet_metrics_do_not_select_zeroed_mask_background`、空掩码跳过、robust topq peak）；`test_checkpoint_selection_penalizes_worse_topq_peak_error`、`test_checkpoint_selection_returns_one_shared_combined_improvement`；eval 采样 RNG 隔离/可重复/分块不超限、初始 tracked batch 扫描不推进训练 RNG；`test_all_best_checkpoints_capture_one_atomic_monitoring_snapshot`、`test_sample_grid_failure_prevents_epoch_ledger_and_checkpoint_commit`；early stopping 以 epoch 计耐心且改进重置原点；EMA shadow overlay 与轻量 EMA checkpoint 识别；固定分层病灶子集与底部分位小病灶标注、小病灶冷偏差子容差、failure 指标显式模型空间转换、stripe 三分数；resume 系列（截断陈旧 metrics、保留不可解析行、拒绝缺失/越界 epoch、拒绝 full config 不匹配、拒绝绝对/跨 run 指针、`test_checkpoint_rng_roundtrip_is_weights_only_safe_and_exact`、persistent worker 下 epoch 戳数据精确、validation 指标整批留 CPU）。
- **揭示的契约**：训练监控全链路确定性：RNG 状态与 checkpoint 精确往返（weights_only 安全），eval RNG 与训练 RNG 严格隔离，最优 checkpoint 基于单一原子监控快照与共享组合改进分；resume 是 fail-closed 的账本校验（metrics/config/指针指纹不符即拒绝提交）。

### `tests/test_action_axis_screen.py`

- **被测对象**：`scripts/eval_action_axis_screen.py`（VARIANTS、_configure_variant、_medoid、build_patient_balanced_subset、种子/变体解析），以 `_Router` 假路由验证推理干预轴。
- **关键用例**：`test_patient_balanced_subset_has_cap_and_required_strata`（每患者采样上限+必需病灶分层）、`test_positive_only_cohort_requires_explicit_override`/`test_positive_only_cohort_preserves_lesion_strata_and_cap`（全阳性队列需显式覆盖）、`test_all_validation_ignores_sampling_cap_and_preserves_order`、`test_all_nine_variants_reset_and_apply_exact_axes`（九个变体精确重置并施加 destination/路由动作/频率模式/增益/logSNR 窗口干预）、`test_medoid_uses_generated_image_distances_only`、`test_metric_row_exact_comparison_treats_nan_as_equal`、`test_seed_parser_requires_four_distinct_values`、`test_variant_parser_supports_b0_only_and_rejects_missing_baseline`。
- **揭示的契约**：动作轴筛选的评估面是"患者平衡+病灶分层"子集；每个变体运行间必须精确重置全部干预轴（防状态泄漏）；基线 B0 必选；NaN==NaN 的逐位精确指标对比语义。

### `tests/test_medoid_zero_training.py`

- **被测对象**：`scripts/analyze_medoid_zero_training.py`（medoid_indices、trajectory_dispersion、within_patient_slope、_haar_dwt2、component_mechanism_rows）。
- **关键用例**：`test_medoid_indices_selects_existing_central_candidate_and_breaks_ties`（选取真实中心候选、平局取首个）、`test_trajectory_dispersion_separates_lesion_and_full_image`、`test_within_patient_slope_recovers_patient_intercept_controlled_effect`（患者内斜率精确恢复 2.0 且 CI 收缩到点）、`test_numpy_haar_preserves_energy`（Haar 变换能量守恒）、`test_component_mechanism_rows_reports_each_component`（按连通域输出面积/梯度质量/高频残差占比）。
- **揭示的契约**：零训练 medoid 分析的统计原语具有合成数据上的精确解析性质（确定性 tie-break、小波能量守恒、患者内去混杂斜率），组件级机制行按病灶连通域展开且量有界。

### 感知 x0 与预训练契约

### `tests/test_perceptual_x0_validator.py`

- **被测对象**：`scripts/validate_perceptual_x0_few_step.py`（_endpoint_direction、derive_mechanism_partition、_gate_non_inferiority）。
- **关键用例**：`test_endpoint_direction_ssim_is_higher_is_better`（锁定审计发现的 SSIM 方向反转修复）、`test_partition_derives_calibration_from_train_and_validation_from_val`（校准患者只从 train 抽取、与 validation 患者不相交）、`test_partition_rejects_missing_manifest`、`test_ssim_non_inferiority_uses_fail_closed_lower_ci_bound`（P1 审计发现：SSIM 曾用乐观 CI 上界，非劣性必须按 CI 下界 fail-closed）、`test_cached_dataset_fails_closed_on_missing_manifest`。
- **揭示的契约**：few-step 感知 x0 验证的非劣性判定采用保守 CI 下界且指标方向显式声明；机制划分保证 calibration/validation 患者不相交；manifest 缺失一律 fail-closed。

### `tests/test_perceptual_x0_loss.py`

- **被测对象**：`src/model/loss_terms/perceptual_x0.py`（LesionAwarePerceptualX0Loss、PETFeatureEncoder）及其预训练/消融 runner 的配置审计。
- **关键用例**：基础契约组 `test_loss_is_finite_scalar`、`test_pred_x0_grad_flows`、`test_encoder_params_grad_all_none`（编码器冻结）、`test_target_branch_is_detached`、`test_global_vs_lesion_balanced_distinguishable`、空/单像素/多尺度掩码无 NaN、`test_random_control_requires_explicit_kind`、`test_missing_checkpoint_fails_closed`；lineage fail-closed 三连（`test_lineage_fake_hash_fails_closed`/`test_lineage_low_recall_fails_closed`/`test_lineage_test_patient_fails_closed`）；`test_quarter_scale_is_truly_one_quarter`、`test_uniform_timestep_weighting_ignores_tau`、`test_optimizer_excludes_frozen_encoder_params`、`test_config_roundtrip`；runner 侧 `test_dry_run_audit_detects_arm_differences`、`test_resolved_config_disables_validation_driven_selection`、`test_runner_records_base_commit_in_resolved_config`/`test_runner_requires_base_commit`、pretrain 采样归一化缓存形状/拒绝歧义通道、`test_encoder_partition_excludes_validation_patients`、`test_pretrain_encoder_is_jointly_trained_then_frozen`。
- **揭示的契约**：感知 x0 损失梯度只流向 pred_x0（编码器冻结且不进 optimizer）；预训练 checkpoint 必须携带 lineage sidecar（sha256 匹配、召回门槛达标、无测试患者）否则 fail-closed；消融公平性由强制 base commit 与禁用验证集选择保证。

### `tests/test_pretrain_tiny_segmenter.py`

- **被测对象**：`scripts/pretrain_tiny_segmenter.py`（encoder_roles_from_partition、recall_metrics）与 `scripts/run_perceptual_x0_ablation.py` 的 audit_arm。
- **关键用例**：`test_encoder_roles_exclude_validation`（validation 患者不进入编码器角色表）、`test_recall_metrics_reports_overall_and_small_lesions`（整体/小病灶召回、precision、dice 与 small_samples 计数）、`test_p1_audit_requires_passing_segmenter_lineage`（P1_SEG_OUT 臂要求 segmenter checkpoint 通过 lineage 审计，否则被拦截）。
- **揭示的契约**：小目标分割器预训练与评估互斥 validation 患者；P1 消融臂仅在 segmenter lineage（require_checkpoint_lineage）通过审计后方可运行。

### 全模型冒烟

### `tests/test_smoke.py`

- **被测对象**：以 `src/model/slmf_bbdm.py`（SLMFBBDM）为核心的整库 CPU 冒烟：配置载入、模块注册表、各先验（Gabor/organ/hotspot/semantic/NoOp）、条件适配器、噪声调度（DDPM/BBDM 桥/scale-adaptive）、损失项、全模型前向、DDIM/MC 采样、Trainer 步与数据集。
- **关键用例**：TestConfig（dotlist 覆盖、CPU 运行时 profile、科学计数法归一化）；临床评估辅助（SUV 校准拟合、冷病灶/外部峰 failure 检测、不确定性指标、finetune 从 EMA 初始化）；注册表重复注册抛错；先验形状与禁用语义；zero_conv 适配器调制 L3、condition dropout 不改输入；scale-adaptive 噪声的频带/广播/信号衰减契约；各损失禁用返零、lesion_roi_l1 忽略远外误差、outside_peak_ranking 惩罚外部峰；TinySegmenter 形状/参数预算/目标热图；模型集成（baseline/full 前向、`test_model_rejects_configured_base_diffusion_loss`、CFG（cfg=1 为 no-op）、self-conditioning、metadata FiLM、`test_ablation_profiles` 开关切换改变输出且非先验模块状态日志正确）；Trainer 一步/梯度累积 flush/resume 后停在配置总 epoch；数据集（npz 关闭、organ_distance 与物理图、水平翻转各场对齐、PNG 缓存从 split CSV 构建 npz+manifest）；ROI SUV 损失消费 meta、FalseHotspotV2 冷掩码排除温器官、split manifest 生成/加载、semantic builder。
- **揭示的契约**：整库在 32×32 合成批上端到端可运行且每个开关产生可观测差异；disabled 损失、NoOp 先验、cfg=1 必须是严格 no-op；数据增强必须保持 CT/PET/mask/器官图对齐。

### 放大机制（Magnification）审计

### `tests/test_magnification_audit.py`

- **被测对象**：`src/mechanism_validation/magnification_audit.py`（validate_cohort、run_frozen_magnification_audit、build_primary_gate、write_audit_artifacts）与 `magnification.PatientBootstrapResult`。
- **关键用例**：`test_validate_cohort_rejects_identity_or_shape_mismatch`（sample_ids 身份与张量形状校验）、`test_frozen_audit_emits_full_zoom_and_roundtrip_rows`（恒等模型下样本×种子×crop 的 full/zoom/roundtrip 行数精确）、`test_primary_gate_cannot_be_rescued_by_sensitivity_crop`（主门不可被敏感性 crop 补救）、`test_artifact_writer_refuses_overwrite_and_emits_schema`。
- **揭示的契约**：放大一致性审计是"冻结"评估（模型仅推理、seeds/crops 固定）；主门按主 crop 的患者级 bootstrap CI 判定且敏感性分析只做解释不做补救；产物写入拒绝覆盖并携带 schema。

### `tests/test_magnification_audit_cli.py`

- **被测对象**：`scripts/audit_magnification_consistency.py`（load_source_cohort 与 main CLI）。
- **关键用例**：`test_source_loader_preserves_zero_padded_identities`（保留零填充的 patient/sample ID 不被数值化）、`test_dry_run_validates_contract_and_refuses_overwrite`（dry-run 校验 config/checkpoint 哈希契约、写 audit_manifest 与 DRY_RUN.json、不落 sample_metrics.csv、重复运行 FileExistsError）、`test_dry_run_fails_closed_on_source_hash_mismatch`（源 run 哈希不匹配即失败）。
- **揭示的契约**：CLI 只消费带 run_manifest（config/checkpoint sha256）与 COMPLETE.json 的已完成 run；dry-run 不产生正式指标产物；身份字符串保留前导零、契约校验 fail-closed。

### `tests/test_magnification_mechanism.py`

- **被测对象**：`src/mechanism_validation/magnification.py`（CropBox、lesion_crop_box、crop_resize、backproject_crop、couple_noise_to_crop、patient_balanced_slope、bootstrap_patient_balanced_slope）及 SLMFBBDM.sample 的 initial_noise 契约。
- **关键用例**：显式 initial_noise 使采样逐位可复现且绕过内部随机抽取（monkeypatch randn_like 证明）、拒绝 shape/dtype/device 不兼容张量、省略时保留内部随机行为；`test_lesion_crop_box_is_centered_and_deterministic`、`test_lesion_crop_box_shifts_at_boundary_without_truncation`（边界平移不截断病灶）、`test_lesion_crop_box_rejects_empty_or_oversized_lesion`；`test_crop_resize_and_backprojection_preserve_constant_region`、`test_nearest_mask_resize_remains_binary`、`test_coupled_noise_is_standardized_and_anatomically_indexed`、`test_patient_balanced_slope_is_invariant_to_slice_duplication`、`test_patient_bootstrap_is_deterministic_and_keeps_linear_effect`。
- **揭示的契约**：放大机制分析建立在"可复现推理"（显式噪声注入）与无损裁剪几何（裁剪框不截断病灶、crop-往返保常数、掩码二值）之上；统计推断按患者平衡、对切片重复不变。

### 特征分析（Feature Provenance/Emergence/Causality）

### `tests/test_feature_provenance.py`

- **被测对象**：`src/mechanism_validation/feature_provenance.py`（audit_feature_provenance）——A0 特征来源溯源审计。
- **关键用例**：`test_provenance_audit_writes_feature_provenance_json`（decision 断言 dual_modal_encoder_claim=False、representation_is_shared_weight_probe=True，含 checkpoint/config sha256 与 4 层特征源）、`test_provenance_layer_shapes_follow_halving`（c1-c4 空间 64/32/16/8 逐级减半、tensor_key 命名）、`test_provenance_checkpoint_sha256_changes_with_content`（权重变化即哈希变化）、`test_provenance_json_is_canonical_parseable`（合法 JSON 且含 shared-weight representation probe 声明）。
- **揭示的契约**：A0 审计明确声明特征来源是"共享权重的 CT 编码器探针"而非双模态编码器；溯源产物绑定 checkpoint 与 config 的内容哈希，层形状遵循减半金字塔契约。

### `tests/test_feature_emergence.py`

- **被测对象**：`src/mechanism_validation/feature_emergence.py`（compute_linear_cka、channel_agnostic_activation、lesion_ring_contrast、lesion_discriminability 等）——A1 跨模态 CKA 与 A2 病灶涌现分析。
- **关键用例**：`test_compute_linear_cka_is_bounded_and_identity`（CKA 有界于 [0,1] 且自相似>0.9）、`test_channel_agnostic_activation_is_channel_invariant`（通道聚合后形状/有限性）、`test_lesion_ring_contrast_positive_when_lesion_hotter`（病灶热于环带时对比度为正）、`test_lesion_discriminability_auroc_high_for_separated_maps`、`test_cross_modal_similarity_paired_beats_shuffled`（配对 CT/PET 相似度高于打乱基线）、`test_lesion_emergence_writes_cohort_and_gate`（落盘队列级产物与判定门）。
- **揭示的契约**：涌现度量在受控合成数据上方向正确；跨模态相似度必须以"患者配对 vs 打乱"为对照基线；分析输出队列级表与 gate，而非仅样本散点。

### `tests/test_feature_causality.py`

- **被测对象**：`src/mechanism_validation/feature_causality.py`（patch_build_condition_bundle、c2_lesion_zero、同面积非病灶/移位掩码对照干预、run_causality_audit）——A2 因果干预审计（c2 病灶特征必要性）。
- **关键用例**：`test_patch_bundle_context_restores_original`（上下文退出后实例属性清除、还原类方法）、`test_c2_lesion_zero_modifies_c2_region`、`test_interventions_are_non_destructive`（干预不改动原 bundle）、`test_nonlesion_samearea_is_exact_same_area_and_nonempty`（对照区域面积精确相等且非空）、`test_nonlesion_samearea_respects_body_mask`（对照限制在体部掩码内）、`test_shifted_mask_is_nonempty_on_small_grid`、`test_run_causality_audit_smoke`。
- **揭示的契约**：因果干预通过临时 patch `build_condition_bundle` 实现，必须可完全还原且非破坏性；对照干预保持面积/体部约束，保证消融差异只可归因于病灶特征本身。

### `tests/test_feature_cli_integration.py`

- **被测对象**：特征涌现/因果审计 CLI（`scripts/eval_feature_emergence.py` 的 _load_frozen_quartiles）与 `src/mechanism_validation/feature_emergence.py` 的 _patient_keys。
- **关键用例**：_patient_keys 系列（接受 dict-of-lists 与 list-of-dicts 两种默认 collate 形态；拒绝缺失 meta、空 patient_id、None patient_id）；frozen quartiles 系列（`test_load_frozen_quartiles_requires_q1_q2_q3`：val/test 缺 frozen-quartiles-json 即 RuntimeError、JSON 提供 q1/q2/q3、train 分支允许无 JSON）。
- **揭示的契约**：正式验证依赖的 fail-closed 行为——患者身份缺失一律拒绝（不回退假 ID 防统计污染）；A2 在 val/test 上必须使用 train 冻结分位数，杜绝选择泄漏。

### 云协作与实验编排

### `tests/test_cloud_worktree_sync.py`

- **被测对象**：`scripts/sync_cloud_worktrees.py`（SyncConfig、sync_once、discover_files、find_active_locks、_resolve_within、_check_drift）——云端双工作树同步工具。
- **关键用例**：docstring 列出的 12 条 fail-closed 行为逐一落实——`test_dry_run_modifies_nothing`、`test_apply_is_byte_identical_and_idempotent`（SHA256 一致 + 二次 apply 全跳过）、`test_manual_target_edit_is_conflict`（手工改动判冲突不覆盖）、`test_traversal_entry_rejected`/`test_junction_escape_rejected`/`test_resolve_within_rejects_traversal`、`test_forbidden_assets_never_copied`（checkpoint/results/.git 永不复制）、`test_target_extra_files_never_deleted`、`test_active_lock_blocks_apply`（活动实验锁阻塞写入）、`test_atomic_copy_failure_preserves_original`、`test_groups_are_segregated`（router_only 不达 feature、feature_only 不达 router）、`test_load_config_real`、`test_check_drift_reports_conflict_when_manually_edited`。
- **揭示的契约**：分布式双实验（路由/特征）同步是 fail-closed 的：字节级一致性、幂等重放、冲突显式暴露而非覆盖、危险路径与产物目录硬隔离、持锁实验期间禁止任何写入。

### `tests/test_dual_experiment_queue.py`

- **被测对象**：`scripts/run_dual_experiment_queue.py`（checkpoint_stable、compute_eligibility、read_emergence_gates、verify_analysis_code、main）——顺序执行的双工作树实验队列。
- **关键用例**：docstring 列出的第 13-17 条 fail-closed 行为——`test_nonempty_output_dir_rejected`（非空输出目录拒绝）、`test_same_gpu_parallel_rejected`（同 GPU 并行请求返回 rc=2 REFUSING）、`test_checkpoint_changing_refuses_launch`（两次快照间 checkpoint 变化即拒绝启动）、`test_last_pt_checkpoint_refused`（训练中 last.pt 拒绝）；emergence 门三连（文件缺失 fail-closed、`test_emergence_gates_require_both_true`、JSON 读取）；`test_full_queue_blocks_causality_when_emergence_gates_fail`（上游失败/门未过则因果分析不运行）；`test_smoke_fake_never_formal`、`test_formal_requires_all_checks`、`test_missing_module_is_fail_closed`、`test_no_frozen_quartiles_on_val_is_exploratory`、`test_missing_spatial_controls_block_formal`、`test_verify_analysis_code_real_module`/`test_verify_analysis_code_missing_module`。
- **揭示的契约**：双实验队列是级联 fail-closed 编排——emergence 门通过是 causality 阶段的前置条件；"正式 COMPLETE"只能由全检查通过 + 真实分析模块校验 + 冻结分位数 + 空间对照齐全的运行产生；checkpoint 必须静态稳定且不得是训练中间产物。



---

## 15. 配置文件与根配置（上：根配置与消融计划）

### 模块概述

本部分覆盖仓库两大根配置（pyproject.toml 仅承载 pytest 设置，pixi.toml 才是项目元数据/依赖/任务中心）与 configs/experiments/ 下的主实验配置（slmf_baseline / slmf_full / ablations）及 v1–v8 共 8 份消融计划 YAML。pixi tasks 串起数据预处理→训练→评估→测试全管线；消融计划 YAML 由 scripts/run_frequency_ablations.py、run_spectral_router_v5.py 等驱动脚本消费，按版本演进（频率筛选→谱证据路由器→冷启动偏差→最终标定）。这些文件是理解仓库实验组织的入口。
### `pyproject.toml`

- **职责**：仅为 pytest 提供配置（无 [project] 元数据段；包元数据与依赖全部在 pixi.toml）。圈定测试范围、排除数据/产物目录、启用严格 marker 校验。
- **接口**：无自身 CLI；被 pixi run test / test-smoke（内部调 python -m pytest）隐式消费。用法：pixi run test、pytest tests/test_smoke.py -k sample。
- **依赖**：内部：无（纯配置）。外部：pytest >= 7.0。
- **输入**：tests/ 目录下的测试文件。
- **输出**：pytest 终端报告（-p no:cacheprovider 禁写 .pytest_cache）。
- **处理过程**：
  1. testpaths=["tests"] 把收集范围锚定在 tests/；
  2. norecursedirs 排除 .pixi/.git/cache/checkpoints/Data/notebook/outputs/results/samples/vis_epoch1000/wuzhe/wuzhe_data；
  3. addopts 注入 --strict-markers（未注册 marker 报错）与 -p no:cacheprovider；
  4. minversion="7.0" 拒绝过旧 pytest。### `pixi.toml`

- **职责**：仓库唯一的任务运行中心与依赖清单。定义 workspace（name=pet-ct-diffusion，平台 win-64，conda 渠道走清华镜像），约 40 个 tasks 覆盖数据预处理、机制门禁、训练、消融、评估、测试、工具，并以 conda + PyPI 双轨声明全部依赖（torch 走 cu128 官方索引）。
- **接口**（全部经 pixi run <task> 调用）：
  - 数据：data-manifest → data-cache（.npz）/ data-cache-png-main（PNG 锁定缓存）→ data-split → data-organ（TotalSegmentator，需 GPU）→ data-semantic；一键 data-setup（&& 四步串联）；validate-png-alignment 对齐检查。
  - 机制门禁：mechanism-validate-local|all|plan、mechanism-v2-freeze|internal|plan、v2-cloud-stage03（ps1 上云）、v2-inference-audit；all 范围在任一 H2–H6 验证器未实现时 fail-closed。
  - 训练：train（slmf_full）、train-smoke（假数据 5 epoch）、train-no-gabor|no-hotspot|no-organ|base-only（--ablation/--override 消融）、train-wandb、train-png-baseline / train-png-smoke（推荐稳定入口 slmf_png_baseline.yaml，启动时跑 validate_png_baseline_config）、train-resume（$CKPT）。
  - 频率/路由消融：train-frequency-ablations（v1 计划 --stage all）、train-frequency-ablations-v2（共享 PET 均值预训练+冻结再筛选晋级）、train-spectral-router-v5 / dry-run-spectral-router-v5、train-spectral-router-v5-t0-c1 / dry-run（精确 300 epoch 晋级）。
  - 评估/诊断：eval-full、eval-val、eval-png-baseline、sample-one（$SAMPLE_ID）、diagnose-stripes（A–E 五种 Gabor 路由对比）。
  - 测试与工具：test / test-smoke / test-sample、inspect（参数量 dry-run）、gpu-info。
- **依赖**：内部：src.data.*（data_manifest/dataset/png_cache/split_manifest/organ_preprocess/semantic_builder）、scripts/（train_v2、evaluate、run_frequency_ablations、run_spectral_router_v5、run_v5_t0_c1_300、diagnose_stripes、validate_png_alignment、run_all_mechanism_validations、freeze_h4_v2_internal_cv_plan、run_h4_v2_internal_exploratory_pipeline、audit_v2_inference_admissibility、run_cloud_v2_stage03.ps1、inspect_model、gpu_info）、configs/experiments/*.yaml。外部：conda（python >=3.11,<3.14、scikit-image 0.25.x）；PyPI 清华镜像 + cu128 索引（torch>=2.8.0、torchvision、accelerate、einops、ema-pytorch>=0.4.2、numpy、pillow、pytorch-fid、scipy、tqdm、nibabel、TotalSegmentator >=2.13,<3、lpips、timm、transformers、diffusers、opencv-python、matplotlib、tensorboard、wandb、pytest、pydicom >=3.0.2,<4）。
- **输入**：Data/ 原始 DICOM、main_data PNG、cache/ 中间产物、configs/ 配置、环境变量 $CKPT / $SAMPLE_ID。
- **输出**：cache/manifest*、cache/tensors*、cache/split_manifest.csv、checkpoints/、outputs/samples、results/diag_stripes|alignment、WandB/TensorBoard 日志。
- **处理过程**：
  1. 解析 [workspace] 建立 win-64 环境，混合解析 conda 与 pypi 依赖（PyPI 走清华 index-url，torch 锁 cu128 源）；
  2. [tasks] 按注释分组，数据管线按 1→5 编号串联；
  3. 训练类 task 以 configs/experiments/slmf_full.yaml 为默认基座，用 --override key=value 或 --ablation 实现消融，避免复制配置；
  4. 消融类 task 把「计划 YAML + --stage」参数化，交给专用驱动脚本做多轮筛选/晋级；
  5. 机制验证独立成组，local/all/plan 三档范围，all 档失败关闭；
  6. 评估/诊断复用 evaluate.py 与 diagnose_stripes.py，经环境变量注入 checkpoint 与样本 ID。
### `configs/experiments/slmf_baseline.yaml`

- **职责**：SLMF-BBDM 的对照基线配置——全部模块与辅助损失关闭，只保留纯 BBDM + CT 拼接（pred_x0 / DDIM 采样），用于与 slmf_full 逐项对比归因。
- **接口**：无 CLI 本身；由 scripts/train_v2.py --config 加载，配合 --override 使用：python scripts/train_v2.py --config configs/experiments/slmf_baseline.yaml；亦被 scripts/evaluate.py 等消费。
- **依赖**：内部：scripts/train_v2.py、scripts/evaluate.py、src/ 配置解析（experiment/data/runtime/training/model/modules/losses/metadata 段）。外部：无（纯 YAML）。
- **输入**：cache/tensors（.npz 缓存）与 cache/split_manifest.csv 患者级划分；required_keys=ct/pet/mask。
- **输出**：训练/评估产物（checkpoints、样本），实验名 slmf_bbdm_baseline、seed 42。
- **处理过程**：
  1. experiment 段固定实验名与随机种子；
  2. data 段指向 .npz 缓存（image_size 192、batch 4、增广开）；
  3. runtime 段配置 AMP/channels_last/梯度累积 2/梯度裁剪 1.0 与 eval/save/sample 间隔（均 50）；
  4. training 段：1000 epoch、lr 1e-4、cosine 至 lr_min 1e-6、EMA 0.999（每 10 步更新）；
  5. model 段：pred_x0 + DDIM 20 步采样，base_loss=mse 1.0+l1 1.0+gradient 0.1+minSNR(γ=5)；
  6. modules/losses/metadata 全部 enabled:false——这是「基线」语义的实质。
### `configs/experiments/slmf_full.yaml`

- **职责**：SLMF-BBDM 全量配置（除 semantic_prior/metadata/segmenter 外所有模块与损失开启），是 pixi train/train-smoke/各消融 task 的默认基座；注释中保留了权重调参痕迹（↑↓ 标注相对旧值的改动）。
- **接口**：python scripts/train_v2.py --config configs/experiments/slmf_full.yaml [--override key=value ... | --ablation <name>]；python scripts/evaluate.py --config ... --split test|val [--sample-id ID]。pixi.toml 的 train/train-wandb/train-resume 直接引用它。
- **依赖**：内部：scripts/train_v2.py、scripts/evaluate.py、src/ 各模块（gabor/organ_prior/hotspot_prior/zero_adapter/scale_adaptive_noise 等）。外部：无（纯 YAML；模块本体依赖 TotalSegmentator、timm 等）。
- **输入**：cache/tensors + split_manifest.csv；optional_keys 扩展 organ_mask/organ_distance/mu_map/ct_hu/pet_suv/semantic_tokens/meta。
- **输出**：实验 slmf_bbdm_full 的 checkpoints/样本/日志；training.stage=lesion_posttrain（ROI 稠密重建 + 病灶外峰值排序的小病灶 PET 保真阶段）。
- **处理过程**：
  1. data 段 batch 8、optional_keys 声明可选先验键；
  2. model 段开启 heteroscedastic（logvar 裁剪 [-6,2]）、self-conditioning(p=0.5)、diffusers UNet；
  3. modules 段：gabor(32 滤波器+噪声/损失双用途)、organ_prior(6 器官通道)、hotspot_prior(≤0.5M 参数)、zero_adapter、condition_dropout（按先验分概率丢弃，gabor 0.15 最高——疑似条纹源）、scale_adaptive_noise（bridge_mode，按低/中/高 σ 1.0/0.75/0.45 系数+Gabor 能量）；
  4. losses 段启用 topk_lesion、lesion_roi_l1（膨胀半径 3 的 ROI 内 L1）、outside_peak_ranking（ROI 内外峰值 margin 排序）、focal_frequency、patch_nce、roi_suv、false_hotspot、organ_consistency、hotspot_prior、heteroscedastic_nll，均带 weight 与 active_tau_max/min 时间窗调度；segmenter_consistency 预留（召回 ≥0.70 后开）；
  5. evaluation 段：MC 8 采样 + 失败检测阈值（outside_margin/lesion_min_ratio/uncertainty_ratio_threshold）；
  6. metadata 与 segmenter 段默认关闭并附三阶段训练协议注释（heatmap→distance→mask）。
### `configs/experiments/ablations.yaml`

- **职责**：消融预设库（503 行，约 30 个命名预设），每项为作用于 slmf_full.yaml 之上的 override 键值集合，按研究阶段分组：频率块筛选（freq_*）→ 边界可靠 V3（br_r2/f2/f4/b/c/d）→ V4 峰值保持（br_v4_d0/d1/d3/d4）→ br_e/br_f → 总基线 baseline → 单模块消融 → 组合消融 → V5 谱证据路由器（sr_v5_s0–s3）。
- **接口**：python scripts/train_v2.py --config configs/experiments/slmf_full.yaml --ablation <name>（文件头注释即用法；pixi task train-no-gabor 用 --ablation no_gabor）。预设名即 CLI 参数值。
- **依赖**：内部：scripts/train_v2.py（--ablation 解析器）；被 frequency/boundary/spectral 各计划 YAML 间接引用（计划里的 variant 往往对应此处预设语义）。外部：无。
- **输入**：slmf_full.yaml 基座配置。
- **输出**：合并后的最终配置（override 深覆盖 modules.*/losses.*/model.* 键）。
- **处理过程**：
  1. 顶层唯一键 ablations:，下挂预设名 → {description, overrides} 两字段结构；
  2. freq_r0/r2/f1–f4 逐步叠加 residual_frequency 能力（conditional_mean→residual_bridge→wavelet 注入→Gabor 门控→损失），形成 R0 参照到 F4 全开阶梯；
  3. br_* 系列固定 wavelet_unet.enabled=false，以 residual_frequency.mode=legacy|boundary_reliable 切换，逐个打开 noise_release/ct_reliability/content_reliability/subband_gates/directional_reliability 开关；
  4. br_v4_d* 在 BR-D 上叠加 normalized_lesion_peak 损失与 ct_reliability_floor_l2/l1 软下限的四种组合；
  5. 单模块消融（no_gabor/no_organ/no_hotspot/no_zero_adapter/isotropic_noise/ddpm_not_bbdm/no_time_varying_loss/no_heteroscedastic/no_self_conditioning/no_condition_dropout/no_patch_nce）与组合（loss_only_baseline/prior_only_baseline/baseline）支撑模块归因；
  6. sr_v5_s0–s3 切换 residual_frequency.mode=spectral_evidence_router，按 DCT/Gabor 证据描述子与 cross_level_router 的开/关做因子筛选。
### `configs/experiments/frequency_ablation_plan.yaml`

- **职责**：频率残差块消融计划 v1——在 PNG-only 基座上筛选 R0/R2/F1–F4 六个频率变体（50 epoch 筛选），top_k=2 晋级 300 epoch 全量训练，并用硬门限卡伪影。
- **接口**：python scripts/run_frequency_ablations.py --plan configs/experiments/frequency_ablation_plan.yaml --stage all（pixi task train-frequency-ablations）；--dry-run 可预演。
- **依赖**：内部：scripts/run_frequency_ablations.py、基座 configs/experiments/slmf_png_residual_frequency.yaml + ablations.yaml（preset freq_r0/r2/f1–f4）；tests/test_frequency_ablation_runner.py 引用。外部：无。
- **输入/输出**：输入为基座配置与 .npz/PNG 缓存；输出 results/frequency_ablations（筛选与晋级产物、门限报告）。
- **处理过程**：base_config+ablation_config+reference_id=R0+top_k=2 定义骨架；variants 用 preset 名映射 ablations.yaml；screen 段（50 epoch/val/16 样本/MC20）跑全部变体；hard_gates 以 failure_any/false_hotspot_density/stripe_excess（lower）与 ssim（higher, max_delta 0.03）相对 R0 卡门；promote 段（300 epoch/64 样本/early_stopping）训练通过门限的 top-2。
### `configs/experiments/frequency_ablation_plan_v2.yaml`

- **职责**：v2 迭代——先预训练一个共享 PET 条件均值网络并冻结，再对同组 R0/R2/F1–F4 变体筛选+晋级，消除 v1 中各变体独立学均值带来的混杂。
- **接口**：python scripts/run_frequency_ablations.py --plan configs/experiments/frequency_ablation_plan_v2.yaml --stage all（pixi task train-frequency-ablations-v2）。
- **依赖**：内部：scripts/run_frequency_ablations.py（支持 mean_pretrain 段）、slmf_png_residual_frequency.yaml、ablations.yaml、checkpoints/freq_mean_pretrain_v2/mean_best.pt。外部：无。
- **输入/输出**：输入同 v1；输出 results/frequency_ablations_v2 与 checkpoints/freq_mean_pretrain_v2。
- **处理过程**：在 v1 结构上新增 mean_pretrain 段（enabled:true、30 epoch、产物 mean_best.pt）；common_train_overrides 给所有变体统一注入 initialization_seed=4242、conditional_mean.checkpoint/freeze=true/loss_weight=0；variants/screen/promote/hard_gates 与 v1 完全一致（prefix 改 freq_v2_*），保证与 v1 结果可比。
### `configs/experiments/boundary_reliable_ablation_plan_v3.yaml`

- **职责**：边界可靠（boundary-reliable）残差频率消融 v3——以 R2 为参照，筛选 br_r2/f2/f4/b/c/d/e/f 八个预设（BR-B 只噪声释放，逐级加 CT 可靠性、内容门控、方向可靠性、边界损失）。
- **接口**：python scripts/run_frequency_ablations.py --plan configs/experiments/boundary_reliable_ablation_plan_v3.yaml --stage all（与 v1/v2 同一 runner，设计文档 docs/superpowers/plans/2026-07-16-reliable-wavelet-ablation-v3.md 确认）。
- **依赖**：内部：基座 configs/experiments/slmf_png_boundary_reliable.yaml、ablations.yaml（br_* 预设）、复用 v2 的冻结均值 checkpoint（mean_pretrain.enabled=false 但 common_train_overrides 仍指向它）。外部：无。
- **输入/输出**：输入为基座+预设+冻结均值权重；输出 results/boundary_reliable_ablations_v3。
- **处理过程**：reference_id=R2、top_k=2；screen 50 epoch（early_stopping=false）跑 8 变体；沿用 v1/v2 同一组硬门限（failure_any/false_hotspot_density/stripe_excess lower、ssim higher）对 R2 比较；promote 300 epoch（eval_interval 20、early_stopping=false）训练通过门限的前 2 名。
### `configs/experiments/boundary_reliable_ablation_plan_v4.yaml`

- **职责**：v4 双阶段计划——stage_a 在 BR-D 上筛 D0/D1/D3/D4（Top-Q 峰值损失 × CT 可靠性软下限组合），stage_b 以胜者 D 为基座筛 G0/G1/G2（方向可靠性 / Gabor 一致性），最后 300 epoch 晋级 + 10000 次重采样的配对比较。
- **接口**：python scripts/run_boundary_reliable_v4.py --plan configs/experiments/boundary_reliable_ablation_plan_v4.yaml --stage all（该文件即 runner 的 --plan 默认值，见 scripts/run_boundary_reliable_v4.py:207）。
- **依赖**：内部：基座 configs/experiments/slmf_png_boundary_reliable_v4.yaml、ablations.yaml（br_v4_d* 预设）、stage_b 变体直接写 overrides（use_directional_reliability/use_gabor_agreement/gabor_agreement_alpha 等）、v2 冻结均值 checkpoint。外部：无。
- **输入/输出**：输入为基座配置与均值权重；输出 results/boundary_reliable_ablations_v4（stage_a/stage_b/promote/paired_comparison 报告）。
- **处理过程**：stage_a（top_k=1，变体走 preset）→ 硬门限在旧四项上新增 lesion_topq_peak_error_norm/lesion_peak_error_norm/stripe_excess 的 max_value 与 ssim min_value 0.85 绝对阈值；stage_b（top_k=2，dry_run_selected_d=D3 供 dry-run）变体改用内联 overrides 并支持 detach_gabor_descriptor；promote 段 require_final_checkpoint=true；paired_comparison 段列 12 项 lower/higher 指标做 bootstrap 显著性检验。
### `configs/experiments/spectral_router_ablation_plan_v5.yaml`

- **职责**：V5 谱证据路由器两阶段计划——stage_a 以 S0 为参照筛 S0–S3（sr_v5_s0–s3 预设，DCT/Gabor 证据因子），stage_b 以 T_native 为参照筛 6 种跨层路由策略（N0/T_legacy/T_native/T_fixed[0.05,0.05,0.90]/C1 learned/C_no_null），胜者 300 epoch 精确晋级 + 10000 次重采样配对比较（含 small_lesion_* 分层指标）。
- **关键键**：base_config=slmf_png_spectral_router_v5.yaml；stage_a.{reference_id,top_k,dry_run_selected_evidence:S3}；stage_b 变体全用内联 overrides（cross_level_router.policy/hard_all_null/fixed_prior）；promote.{trajectory_checkpoints:[50..300],require_final_checkpoint}；hard_gates 分层：stage_b 以 50-epoch 匹配参照卡相对退化（lesion_topq max_value 0.080、mae max_delta 0.002 等）。
- **消费脚本**：scripts/run_spectral_router_v5.py:504（--plan 默认值即此文件）；scripts/run_v5_t0_c1_300.py:39（T0+C1 精确 300 epoch 复训）；pixi tasks train-spectral-router-v5 / dry-run-spectral-router-v5；tests/test_frequency_ablation_runner.py 多处断言；configs/cloud_worktree_sync.yaml:113 云端同步清单。
### `configs/experiments/spectral_router_repair_plan_v6.yaml`

- **职责**：V6 修复计划——保留 V5 证据前端，修复采样器与学习式路由优化：以 50% 初始 null 概率+native 预热 10 epoch 缓解 90%-null 梯度饥饿，加 temporal/active_mass 正则防全-null 局部最优，EMA 加快到 0.995、分组学习率（projection/router 5e-4、descriptor 2e-4）；stage_a 仅重录 S3 基线（hard_gates 空=溯源锁存），stage_b 最小因果筛 N0/T_native/C1/C_no_null（legacy 与 fixed 路线退役），promote 100 epoch 确认。
- **关键键**：common_train_overrides 大幅扩容（cross_level_router.initial_null_probability/native_warmup_epochs/routing_ramp_epochs、losses.spectral_router_regularization.*、training.optimizer_groups.*、runtime.gradient_diagnostics）；stage_b/promote 各自带 hard_gates（promote 用严格端点门：lesion_topq max_value 0.12 且 max_ratio 0.90）。
- **消费脚本**：文件头注明经 scripts/run_spectral_router_v5.py --plan <本文件> 运行（V5 结果保持原样不动）；tests/test_frequency_ablation_runner.py:1142 断言其内容。
### `configs/experiments/spectral_router_cold_bias_plan_v7.yaml`

- **职责**：V7 冷偏差因果隔离——所有候选从同一修复版 C1-100 权重（training.init_from=sr_v6_confirm_evidence-s3_c1/ckpt_epoch0100.pt）出发、全新优化器/EMA，30 epoch 精确端点（禁止挑 best checkpoint）定位小病灶欠估瓶颈：CTRL（再练 30 epoch 的纯对照）、COLD（非对称峰值监督 weight 0.20/cold_weight 2.0）、CT_RELAX（放宽 CT 可靠性下限）、COLD_ALIGN（+topk 定位）、COLD_RANK（+outside_peak_ranking 防false hotspot）。
- **关键键**：include_reference_in_promotion:true、reference_id=CTRL、screen.{require_final_checkpoint,evaluate_final_checkpoint}、双门限体系（hard_gates 筛选用相对比率 + promotion_hard_gates 晋级用绝对值含 small_lesion_* 分层不许劣化）、evaluation.small_lesion_quantile=0.25。
- **消费脚本**：scripts/run_spectral_router_v5.py --plan（同 schema）；tests/test_frequency_ablation_runner.py:1239；configs/cloud_worktree_sync.yaml:114 云端同步。
### `configs/experiments/spectral_router_final_calibration_plan_v8.yaml`

- **职责**：V8 终标定——非重设计：所有候选从 V7 最安全端点（sr_v7_confirm_cold_rank/ckpt_epoch0050.pt 的 EMA 权重 init_weights=ema）低学习率（2e-5，分组 5e-5/5e-5/2e-5）微调 12 epoch，检验能否闭合最后 0.0086 峰值误差缺口：CAL_CTRL（纯低 LR 续训对照）、PEAK25/PEAK_WIDE（加强非对称峰值校正及扩展噪声窗）、BALANCED（同步加强安全排序）、TOPK18（弱定位增量，避开 V7 有害值 0.25）。
- **关键键**：top_k=1、promote.seed=43（换数据序独立复训而非续训 screen 状态）、双门限（筛选相对 ratio 0.98 + promotion_hard_gates 绝对值 lesion_topq/lesion_peak ≤0.12 且 small_lesion_* 不劣化）、evaluation.small_lesion_underestimate_tolerance=0.05。
- **消费脚本**：scripts/run_spectral_router_v5.py --plan；tests/test_frequency_ablation_runner.py:1340；configs/cloud_worktree_sync.yaml:115。
### 模块依赖小结

- **根配置双轨**：`pyproject.toml` 只管 pytest，`pixi.toml` 承担元数据、conda+PyPI 依赖（torch cu128、TotalSegmentator、lpips 等）与约 40 个 task；所有实验入口都是 `pixi run <task>`。
- **配置三层结构**：主实验配置（slmf_baseline / slmf_full，experiment/data/runtime/training/model/modules/losses/evaluation 段）→ 消融预设库（ablations.yaml，~30 个命名 override 组）→ 消融计划（v1–v8 YAML，引用 base_config + ablation_config + preset/inline overrides + screen/promote/hard_gates）。
- **计划演进链**：frequency v1→v2（引入冻结共享均值）；boundary_reliable v3→v4（BR-B..F 因子筛 → D/G 双阶段）；spectral_router v5→v6→v7→v8（证据筛选 → 采样器/路由修复 → 冷偏差因果隔离 → 低 LR 终标定），后者通过 checkpoints/init_from 逐版继承前者产物（freq_mean_pretrain_v2 → sr_v6_confirm → sr_v7_confirm_cold_rank）。
- **消费脚本**：scripts/run_frequency_ablations.py（v1/v2/v3）、scripts/run_boundary_reliable_v4.py（v4）、scripts/run_spectral_router_v5.py（v5–v8 通用 --plan）、scripts/run_v5_t0_c1_300.py、scripts/train_v2.py 与 scripts/evaluate.py（主配置）；tests/test_frequency_ablation_runner.py 为全部计划的内容回归测试；configs/cloud_worktree_sync.yaml 把 v1/v3–v8 列入云端同步清单。
- **外部依赖面**：配置本身为纯 YAML；其描述的模块依赖 PyTorch/torchvision(cu128)、TotalSegmentator（organ_prior）、timm/transformers（semantic）、lpips/pytorch-fid（评估）、wandb/tensorboard（日志）。


---

## 15. 配置文件与根配置（下：生产实验配置与机制管线契约）
### 模块概述

本部件覆盖 `configs/` 下两层契约：`configs/experiments/` 的 12 个训练/消融 YAML（统一由 `scripts/train_v2.py` 加载、`src/model/config_utils.py` 解析），以及根级 6 个 JSON 机制管线契约 + `cloud_worktree_sync.yaml` 云端分发清单。
实验 YAML 共享同一 schema（experiment/data/runtime/training/model/modules/losses/evaluation），演进主线是 `modules.residual_frequency` 的路由策略：boundary_reliable → spectral_evidence_router → prior_anchored_learned，最终叠加 perceptual_x0 感知损失。
JSON 侧是"先证伪后实现"的冻结契约：dataset_contract 锁定数据指纹，mechanism_validation_pipeline v1/v2 与 h3/h4 各阶段配置预注册决策门与 claim 边界，全部内嵌 sha256 自校验。
`cloud_worktree_sync.yaml` 则以白名单把源码按 common / router_only / feature_only 三组分发到云端双 worktree。

### `configs/experiments/slmf_png_boundary_reliable.yaml`

**用途**：PNG-only「边界可靠」残差 BBDM 的 BR-F 默认底座（300e 全量训练）：条件均值联合训练 + `residual_frequency.mode: boundary_reliable`（CT/内容/子带/方向四重可靠性门控），Wavelet U-Net 原型保持关闭；是 boundary_reliable_ablation_plan_v3 消融的解析基。

**关键键**：
- `training.num_epochs: 300`、lr `1e-4`、EMA 0.999/10；`data`：`cache/tensors_main` + `../main_data/split_manifest.csv`（早期带 `../` 的路径）、192×192、batch 4、梯度累积 2
- `modules.residual_frequency`：`use_ct_reliability`/`use_content_reliability`/`use_subband_gates`/`use_directional_reliability` 全开，`gate_max: 0.25`、`band_scales [0.5, 0.25]`
- 损失组合：`lesion_roi_l1(1.0)`、`topk_lesion(0.4)`、`outside_peak_ranking(0.1)`、`boundary_frequency(0.05)`、`frequency_gate_tv(0.001)`；`residual_wavelet`/`gabor_consistency` 关闭
- `conditional_mean`：levels 2 / loss_weight 1.0（与主干联合训练，未冻结）

**消费方**：`tests/test_frequency_ablation_runner.py`（L1732/L1765/L1789）；`configs/experiments/boundary_reliable_ablation_plan_v3.yaml`（base 引用）；`configs/cloud_worktree_sync.yaml`（common 白名单 L106）；`docs/superpowers/plans/2026-07-16-reliable-wavelet-ablation-v3.md`。

### `configs/experiments/slmf_png_boundary_reliable_v4.yaml`

**用途**：BR-D V4 50-epoch 消融底座：冻结的预训练条件均值（`freq_mean_pretrain_v2`）+ boundary_reliable 频率模块；峰值损失与 CT/Gabor 路由开关由 `boundary_reliable_ablation_plan_v4.yaml` 的 preset 在此基础上选择。

**关键键**：
- `conditional_mean`：`freeze: true` + `checkpoint: checkpoints/freq_mean_pretrain_v2/mean_best.pt` + `loss_weight: 0.0`（对比 v3 的联合训练）；`use_directional_reliability: false`
- `training.num_epochs: 50`、`eval_num_samples: 64`、eval/save/sample interval 全部 10（快筛节奏）
- 损失精简为 `lesion_roi_l1(1.0)` + `topk_lesion(0.4)` + `outside_peak_ranking(0.1)`；新增探针键 `normalized_lesion_peak`（默认关，0.05/topk 0.10）与 `gabor_agreement_*`（默认关）

**消费方**：`tests/test_frequency_ablation_runner.py`（L461）；`configs/experiments/boundary_reliable_ablation_plan_v4.yaml`；`configs/cloud_worktree_sync.yaml`（router_only 组）；`docs/superpowers/plans/2026-07-16-br-d-peak-gabor-v4.md`、`2026-07-16-spectral-evidence-conservative-router-v5.md`。

### `configs/experiments/slmf_png_residual_frequency.yaml`

**用途**：官方推荐配方 F4「PNG-only 条件均值 + 残差频率 BBDM」：条件均值联合训练、小波注入 + Gabor 门控的频率残差模块；`frequency_ablation_plan(v1/v2)` 的 R0/R2/F1-F4 全部由此文件解析。

**关键键**：
- `modules.residual_frequency`：`inject_wavelet: true`、`use_gabor_gate: true`、`band_scales [1.0, 0.5, 0.25]`、`gate_strength 0.1`；`gabor.use_for_loss: true`
- 损失：`lesion_roi_l1(1.0)` + `topk_lesion(0.4)` + `outside_peak_ranking(0.1)` + `residual_wavelet(0.05, lesion×4)` + `gabor_consistency(0.02)`
- `num_epochs: 300` + `early_stopping(patience 40, min 50)`；`split_manifest: main_data/...`（已去掉 `../` 前缀）

**消费方**：`tests/test_frequency_ablation_runner.py`、`tests/test_mean_pretraining.py`；`configs/experiments/frequency_ablation_plan.yaml`/`_v2.yaml`（base）；`scripts/retrain_excluded_mean.ps1`、`scripts/run_v2_main_pipeline.py`、`scripts/validate_h2_pathology_excluded_residual.py`、`validate_h3_logsnr_recoverability.py`、`validate_h4_noise_band_calibration.py`；`configs/cloud_worktree_sync.yaml`。

### `configs/experiments/slmf_png_spectral_router_v5.yaml`

**用途**：谱证据路由器 V5 底座（`spectral_evidence_router` 模式，50e 快筛）：冻结预训练均值 + DCT/Gabor 双描述子 + 学习型跨层三路路由；是 v6 修复、v7 冷偏置、v8 终校准计划及 H3 推理契约的锚点生产配置。

**关键键**：
- `residual_frequency.mode: spectral_evidence_router`；`ct_reliability_floor_l2/l1: 0.25/0.50`；`dct_descriptor`（8×8 池化/12 频点）与 `gabor_descriptor` 启用
- `cross_level_router`：`policy: learned`、`initial_null_probability 0.90`、`fixed_prior [0.05, 0.05, 0.90]`
- 损失切换：`topk_lesion(0.15, topk 0.10)` + `normalized_lesion_peak(0.10, cold_weight 1.0)` + `spectral_router_regularization(1.0, dct/gabor 1e-4)`；`boundary_frequency`/`outside_peak_ranking` 关闭；`evaluation` 增加小病灶分位指标

**消费方**：`tests/test_frequency_ablation_runner.py`（L529/1436/1493）；`spectral_router_ablation_plan_v5.yaml`/`repair_plan_v6`/`cold_bias_plan_v7`/`final_calibration_plan_v8`；`configs/h3_fixed_schedule_inference_v1.json` 与 `h3_v2_full_timestep_native_null_v1.json`（runtime_sources 按 sha256 锁定本文件）；`configs/cloud_worktree_sync.yaml`。

### `configs/experiments/slmf_png_prior_anchored_router_100e.yaml`

**用途**：H3 先验锚定自适应路由的 100-epoch 探索模板（云端 runner 会写出 run-owned resolved YAML）：把 direct-PNG H3 先验预览注入 `prior_anchored_learned` 路由策略，并包含 Experiment D 浅层投影去零初始化。

**关键键**：
- `data`：`dataset_contract: configs/dataset_contract_stage0a_v1.json` + `require_cache_lineage: true`；`runtime`：`device: cuda`/`require_cuda`、`gradient_diagnostics: true`
- `training.optimizer_groups`：projection/router lr `5e-4`、descriptor `2e-4`；EMA 0.995/每步
- `cross_level_router`：`h3_schedule_source: direct_png_preview`、warmup/ramp 10/10/30/10、`prior_anchor_decay_end_epoch 100`、`shallow_projection_init_scale 0.01`（修 ~3.6e-11 梯度冷启动死锁）
- `spectral_router_regularization` 新增 `prior_anchor 1e-2 / monotonic 2e-3 / curvature 2e-4 / budget 2e-3 / shallow_cost 0`；`artifact_safety_weight` 为 INERT DEAD KEY（runner 强制 ==0）

**消费方**：`scripts/run_prior_anchored_router_100e.py`（专用 runner，锁定 100e）；`scripts/diag_router_grad.py`；`tests/test_prior_anchored_paired_ablation.py`；`configs/experiments/prior_anchored_paired_ablation_v1.yaml`（base_config）与 `slmf_png_prior_anchored_router_300e.yaml`（头部引用）；`docs/current_model_status.md`；`configs/cloud_worktree_sync.yaml`。

### `configs/experiments/slmf_png_prior_anchored_router_300e.yaml`

**用途**：prior-anchored 路由 + Experiment D 的 300-epoch 交付训练：绕过锁定 100e 的专用 runner、直接 `train_v2.py` 启动；复用 100e run 已估计的 direct-PNG H3 先验（sha256 锁定）。

**关键键**：
- `num_epochs: 300`、`checkpoint_dir: results/prior_anchored_router_300e/checkpoints`（新目录防覆盖 100e 产物）；`init_from: null`（全新训练而非续训）
- `h3_schedule_path` 指向 100e run 的 `prior/h3_prior_preview.json` + `h3_schedule_sha256: dfe9a562…`；`prior_anchor_decay_end_epoch: 300`（随训练拉长）
- `phase_observation_epochs [10, 40, 100, 200, 300]`；`torch_compile: false`（Windows 云机无 triton）；其余（seed/切分/优化器组/均值 ckpt `freq_mean_excluded_v1`/eval 设置）与 100e 逐字节一致

**消费方**：`scripts/eval_router_background_suite.py`（背景评估套件）；`configs/experiments/slmf_png_fullstack_perceptual_p2_300e.yaml`（逐字节基座）；`docs/cloud_dual_experiment_workflow.md`；`configs/cloud_worktree_sync.yaml`。

### `configs/experiments/slmf_png_prior_anchored_router_identifiable_full_100e.yaml`

**用途**：可辨识全量修复 run（100e）：把路由学习拆成三阶段（分支自举→路由辨识→冻结路由稳定化），叠加空间路由、路由效用监督与非病灶低频监督，使各机制可独立开关做有序退化归因。

**关键键**：
- `spectral_training_phases`：branch_bootstrap(1-25, base+projection) / route_identification(26-40, router+descriptor) / frozen_route_stabilization(41-100, base+projection)
- 路由：`destination_mode: learned` + `destination_bootstrap_probability 0.25`（floor 0.02/ceiling 0.75）、`spatial_destination_enabled: true`、H3 先验 sha `3c3b518b…`；`conditional_mean.checkpoint` 改用 300e 的 `ckpt_best_lesion.pt`（不再冻结）
- 新损失：`route_utility_supervision`（lesion_target 0.90/background 0.02，仅 26-40 窗口）、`nonlesion_body_lowpass(0.05, tail×1.5, 41-60 预热)`、`residual_wavelet(0.02)`；`early_stopping(patience 20/min 60)` + `background_constraints` 非劣包络与 4 个 save_tags

**消费方**：`scripts/eval_oracle_destination.py`（默认 --config）；`configs/cloud_worktree_sync.yaml`（common 组 L111）。

### `configs/experiments/slmf_png_background_tail_repair_e100.yaml`

**用途**：背景尾部修复微调（等效 Epoch 101-150）：从 300e 平衡 Epoch-100 EMA 权重出发，冻结谱路由并强制 `native_only`，用 `nonlesion_body_lowpass` 监督 CT 体部非病灶低频 PET 场、软加权生理摄取上四分位。

**关键键**：
- `training`：`init_from: results/prior_anchored_router_300e/checkpoints/ckpt_epoch0100.pt` + `init_weights: ema`、`lr 2e-5`、`num_epochs 50`、`stage: background_tail_repair`
- `cross_level_router`：`destination_mode: native_only`、`prior_warmup_epochs 100`/`prior_destination_warmup_epochs 110`（整个修复窗保持可用性封闭）
- `nonlesion_body_lowpass`：weight `0.20`、`tail_quantile 0.75`、`tail_weight 2.0`；`spectral_router_regularization` 关闭（路由已冻结，正则项变常数）
- `save_interval: 999` + lightweight 检查点只存 `best_background`，且须满足 `background_constraints`（failure_rate≤0.265625、lesion peak≤0.1365、topq≤0.1720）

**消费方**：`tests/test_nonlesion_body_lowpass.py`（L149，校验损失参数与本配置一致）；`configs/cloud_worktree_sync.yaml`（router_only 组）。

### `configs/experiments/prior_anchored_paired_ablation_v1.yaml`

**用途**：prior-anchored 路由 A/B/C/D 配对消融计划，当前 `status: BLOCKED`——四变体参数结构/幅度契约/初始化 RNG 不可配对，B 变体还缺 decision=PASS 的正式 H3-v2 schedule；文件本身即 fail-closed 声明，禁止执行与因果归因。

**关键键**：
- `base_config: slmf_png_prior_anchored_router_100e.yaml`；`paired_execution_allowed`/`causal_attribution_allowed`/`scientific_claim_allowed`/`production_activation_allowed` 全 false
- `shared_contract` 锁 seed=42、manifest/cache/dataset_contract、pathology-excluded 均值 ckpt、100 epochs、eval/dataloader 种子；`declared_override_scope` 仅 `cross_level_router`
- `variants`：A_no_route（native_only）/ B_fixed_h3_native_null（阻塞）/ C_h3_prior_anchored_adaptive（=100e 基配置）/ D_unconstrained_learned；`primary_comparisons` 15 项（val 指标 + route_{native,shallow,null}_mass）

**消费方**：`tests/test_prior_anchored_paired_ablation.py`（PLAN_PATH）；`scripts/summarize_cloud_runs.py`（L903）；`docs/current_model_status.md`、`docs/freeze_gap_checklist.md`；`configs/cloud_worktree_sync.yaml`。

### `configs/experiments/perceptual_x0_ablation_plan_v1.yaml`

**用途**：PFM 启发的冻结特征感知监督 Stage-A 五臂消融契约（P0 像素参考 / P1 分割输出一致 / P2 全局特征 / P3 病灶均衡特征主候选 / P4 随机编码器阴性对照）：所有臂共享 baseline 数据与批序、仅监督项不同，权重只在 calibration 分区选择。

**关键键**：
- `base_config: slmf_png_baseline.yaml` + `base_commit_sha256 6a90921e…`（不可变 git 基座随每次运行记录）
- P3 主候选：`region_mode: lesion_balanced`、`lesion_weight 4.0`、charbonnier、`feature_layers [full, half, quarter]`×`[1.0, 0.5, 0.25]`、encoder `pet_feature_encoder_v1`；P4 `encoder_kind: random`（无 checkpoint 阴性对照）
- `fairness`：shared_epochs 300、`weight_selection_partition: calibration`、验证分区零参与
- `evaluation`：`nfe_grid [20, 8, 4, 2]`、主终点 `small_lesion_topq_peak_error_norm`、主对比 P3@8 vs P0@8、表征对照 P3@8 vs P4@8

**消费方**：`scripts/run_perceptual_x0_ablation.py`（runner + 公平性 fail-closed 审计）、`scripts/validate_perceptual_x0_few_step.py`、`scripts/cloud_run_perceptual_x0_ablation.ps1`、`scripts/build_cloud_runner_zip.py`；`docs/experiments/pfm_lesion_aware_perceptual_x0_plan.md`。

### `configs/experiments/slmf_png_baseline.yaml`

**用途**：一切 PNG-only 实验的最初底座：普通 BBDM bridge + raw CT concat + 基础重建/病灶损失；Gabor 滤波器保留但四条路由全关（切断 Gabor→条纹传播），启动时 `validate_png_baseline_config` 强校验 PNG 纯净性（开 SUV/organ/metadata 即报错）。

**关键键**：
- `data.mode: png`、`cache_dir: cache/tensors_main`、`optional_keys: []`（不接受 organ/suv 可选键）
- `gabor`：enabled true 但 `inject_adapter/use_for_noise/use_for_hotspot/use_for_loss` 全 false（仅诊断用）
- 损失仅 `lesion_roi_l1(1.0)`/`topk_lesion(0.4)`/`outside_peak_ranking(0.1)`；`focal_frequency`/`patch_nce` 等条纹嫌疑项与全部 SUV/organ 损失关闭
- `num_epochs 300` + `early_stopping(40/50)` + `best_checkpoint(combined_alpha 0.5, stripe_penalty 0.3)`

**消费方**：`tests/test_png_baseline.py`、`tests/test_trainer_monitoring.py`；`scripts/evaluate.py`、`scripts/diagnose_stripes.py`、`scripts/validate_png_alignment.py`、`scripts/run_perceptual_x0_ablation.py`；`configs/experiments/perceptual_x0_ablation_plan_v1.yaml`（base_config）；`configs/cloud_worktree_sync.yaml`。

### `configs/experiments/slmf_png_fullstack_perceptual_p2_300e.yaml`

**用途**：全栈 + P2 感知联合臂（300e）：以 `slmf_png_prior_anchored_router_300e.yaml` 逐字节为基座，唯一功能增量是启用 Stage-A 获胜的 P2_FEAT_GLOBAL `perceptual_x0`——与 300e 对比即可在完整机制栈上隔离感知项的边际贡献。

**关键键**：
- `losses.perceptual_x0`：`weight 1.0`、`encoder_kind: pretrained`（`pet_feature_encoder_v1/encoder_best.pt`）、charbonnier、`region_mode: global`、`timestep_weighting: uniform`、`require_checkpoint_lineage: true`
- `experiment.name: fullstack_perceptual_p2_300e` + 独立 `checkpoint_dir: results/fullstack_perceptual_p2_300e/checkpoints`
- `data.cache_dir/cache_lineage` 钉死为云端绝对路径 `D:/ECPC-IDS-SEVEN-Work3/My_diffusion/cache/tensors_main`
- 头注释明确 bridge 空间语义：residual_bridge 下感知损失比较完整重建 PET（梯度只达残差分支）

**消费方**：`artifacts/README_fullstack_p2.md`（部署与结果说明）；训练入口 `scripts/train_v2.py`（配置头注释给出云端启动命令）。

### `configs/dataset_contract_stage0a_v1.json`

**用途**：Stage-0A 数据契约（`contract_status: LOCKED`，内容寻址）：以三重 sha256 锁定 split_manifest、3573 张 raw PNG（ct/pet/mask 各 1191）与预处理 schema，是所有云端训练与机制验证的数据指纹根；任何字段改动即令 contract_sha256 失效。

**关键键**：
- 指纹：`manifest.semantic_sha256 2d063880…`、`raw_png.combined_sha256 b1173443…`、`preprocessing_config_sha256 b0d3cf7a…`、`contract_sha256 f43229d1…`
- 预处理：192×192；CT 双线性 + `uint8/127.5-1`；PET 反相（白底黑摄取 255-pixel）后归一；mask 最近邻 + `>127` 阈值二值化
- `splits`：train 954 样本/124 患者、val 237/31、test 0/0；`claim_boundary`：允许数据假设、禁止模型机制声明

**消费方**：全仓 88 处引用——`scripts/audit_checkpoint_lineage.py`、`scripts/estimate_h3_prior_from_png.py`、`scripts/run_all_mechanism_validations.py`、`scripts/run_cloud_stage0c_gate.py`、`scripts/run_cloud_v2_stage03.ps1`、`scripts/retrain_excluded_mean.ps1`、`scripts/develop_h4_v2_uncertainty_aware.py`、`tests/test_v2_cloud_stage03_runner.py`，以及 prior_anchored 100e/300e/identifiable 等 yaml 的 `data.dataset_contract` 字段。

### `configs/mechanism_validation_pipeline_v1.json`

**用途**：机制验证主管线 v1（`ct_pet_residual_brownian_bridge_mechanism_v1`）：以「先证伪后实现」策略编排 H1-H6 共 11 个 stage——00A 数据审计、00B H1 谱不对称、00C 云缓存血缘门、01 H2 残差富集、02 H3 可恢复性曲线、03 H4 噪声校准、04 CT support head、05 curriculum、06 artifact safety、07 H5 路由角色、08 H6 最终集成；失败门阻断全部下游。

**关键键**：
- `policy`：阈值只准来自 mechanism_train/calibration、禁止验证/测试驱动训练、患者级 bootstrap 95CI、每 stage 必写 decision.json、未验证 checkpoint 禁用
- 每 stage 骨架：`id / title / hypothesis / implemented / output(results/mechanism_validation/…) / command / decision_file / pass_path / pass_value / requires` 依赖图
- `decision_profiles.model_mechanism.required_assertions`；所有 stage 的 `contract_path` 均校验 `dataset_contract_sha256`

**消费方**：`scripts/run_all_mechanism_validations.py`（编排入口，L34）；`tests/test_formal_mechanism_pipeline.py`、`tests/test_mechanism_validation_runner.py`；`configs/cloud_worktree_sync.yaml`。

### `configs/mechanism_validation_pipeline_v2.json`

**用途**：机制验证管线 v2 索引：登记 H4-v2 内部探索性嵌套交叉验证协议（`pixi run mechanism-v2-internal`），并显式保留 v1 的 `H4: FAIL` 历史结论（`preserve: true`，不得覆盖/重贴标签）。

**关键键**：
- `dataset`：155 患者/1191 样本，mechanism_train 99 / calibration 25 / validation 31（曾暴露）、无测试集、未假设新增队列
- `current_protocol`：`config: configs/h4_v2_internal_exploratory_nested_cv_v1.json`、患者单位 5×5 嵌套折、command `pixi run mechanism-v2-internal`
- `comparators`：no_route / h3_fixed_schedule / original_evidence / uncertainty_aware；`interpretation` 限定"现有数据集内部探索性支持"

**消费方**：`scripts/run_h4_v2_pipeline.py`（L87 组装 config 路径）；`configs/cloud_worktree_sync.yaml`；与 h4_v2 json 互为引用（本文件 current_protocol.config 指向它）。

### `configs/h4_v2_internal_exploratory_nested_cv_v1.json`

**用途**：H4-v2「不确定性感知路由」嵌套 CV 预注册协议（`FROZEN_BEFORE_OUTER_EVALUATION`）：患者级 5 外折×5 内折、四 comparator、sign-flip + bootstrap 双门，并把允许的结论措辞预先冻结为两种（支持/不支持）。

**关键键**：
- `partition`：outer/inner seed 20260811/20260911、平衡候选 4096/2048、按 slice_count/mask_area/LL2 对比度分箱平衡；内层角色 mechanism_train/calibration
- `mechanism`：患者排除的 band×timestep robust 统计、`shrinkage_k 60`、`ridge_alpha 1.0`、置信分位 0.25、低置信回退 `h3_fixed_schedule`
- `gates`：bootstrap 10000、主对比 CI95 low>0 且单侧 p<0.05、≥4/5 外折为正、router 活跃患者均值≥0.5、三个难例子组各≥10 患者；必报 044/153 难例与原始分区披露

**消费方**：`scripts/run_h4_v2_internal_exploratory_pipeline.py`、`scripts/validate_h4_v2_internal_cv.py`、`scripts/freeze_h4_v2_internal_cv_plan.py`；`configs/mechanism_validation_pipeline_v2.json`（L20）；`docs/mechanism_validation_h4_v2.md`；`configs/cloud_worktree_sync.yaml`。

### `configs/h3_fixed_schedule_inference_v1.json`

**用途**：05B「固定 H3 schedule 推理可辨识性」预注册审计契约：从生产校准 bundle 冻结 6 频带×60 条目的标量可恢复性网格，审查「单标量 → native/shallow/null 三路由分布」的映射是否可辨识；预注册期望决策为 FAIL 并在模型改动前停止。

**关键键**：
- `runtime_sources` 以 sha256 锁定 `slmf_bbdm.py`/`spectral_router.py`/生产 v5 配置；`production_timestep_contract`：20 步 eval 网格有 19 个 off-grid 点、resolver `UNDEFINED_FAIL_CLOSED`（禁止 nearest/插值/全局均值回退）
- `h3_schedule_contract`：冻结网格精确查找（`off_grid_lookup_authorized: false`），runtime 禁止 target_pet/lesion_mask/recoverability 标签输入
- `mapping_contract`：`authorized_mapping: null`，identifiability_rule 声明标量无法唯一确定三路由分布；`decision_policy`：`expected_decision: FAIL`、next_stage/model_evaluation/production_activation 全禁

**消费方**：`scripts/audit_h3_fixed_schedule_inference.py`（DEFAULT_CONFIG）；`scripts/run_h3_fixed_schedule_overnight.py`（H3_CONFIG）；`configs/cloud_worktree_sync.yaml`。

### `configs/h3_v2_full_timestep_native_null_v1.json`

**用途**：05C H3-v2「全时间步 native/null 可用性」校准+实验预注册（`PREREGISTERED_BEFORE_H3_V2_CALIBRATION_OR_MODEL_EVALUATION`）：Stage A1 只测 active_mass 一个可辨识自由度（shallow 结构性为零），先在 0..999 全网格直接测量并校准各向同性非增曲线，全部完整性/科学门通过后才允许 no_route vs h3_v2_native_null 探索实验。

**关键键**：
- `calibration`：直接测量网格 step=1（插值/最近邻/均值回退全禁止）、映射 `[native, shallow, null] = [active_mass, 0, 1-active_mass]`、可恢复性=病灶掩膜余弦对齐、患者等权
- `calibration_gates`：`a(t0)≥0.98`、`a(t999)≤0.1`、全曲线非增、vs band-constant 的 MSE 差 CI95 high<0 且改善患者比例≥0.8，失败则训练前停止
- `exploratory_model_experiment`：seed 4242、50 epoch、主门 `small_lesion_topq_peak_error_norm` CI95 high<0、mae/stripe/failure/false_hotspot 非劣安全门、必报 044/153 且不得隐藏 002/022/080
- `stop_and_claim_rules` 14 项全 false（禁生产激活与三元路由/方向/课程/特异性等一切声明）；`runtime_sources` 锁 20+ 个脚本与源文件 sha256

**消费方**：`scripts/calibrate_h3_v2_full_timestep_native_null.py`、`scripts/build_h3_v2_experiment_configs.py`、`scripts/analyze_h3_v2_experiment.py`、`scripts/run_h3_v2_overnight.py`；`tests/test_h3_v2_full_timestep_native_null.py`；`docs/current_model_status.md`、`docs/h3_direct_png_prior.md`；`configs/cloud_worktree_sync.yaml`。

### `configs/cloud_worktree_sync.yaml`

**用途**：本地→云端「一次上传、清单驱动」的双 worktree 分发配置：router 主线（`.worktrees/spectral-router-v5`）与 feature 可解释性分支（`.worktrees/lesion-feature-emergence`）物理共存、逻辑隔离；数据/缓存/权重在云机只读共享、绝不随代码复制。

**关键键**：
- `roots`：source `.` + 两个 worktree 根 + manifests/backups；改 root 越界即拒绝拷贝
- `groups.common / router_only / feature_only` 显式白名单（字面量 + `*`/`**` glob）：router_only 含 `src/model/frequency/**`、run_spectral_router_v5/run_prior_anchored_router_100e 等训练脚本与 background_tail/v4 配置；feature_only 仅 feature_emergence/causality/provenance
- `exclude`：正则黑名单——`.git/main_data/Data/cache/checkpoints/results` 等目录树与 `.pt/.npy/.png/.zip` 等产物后缀一律不分发
- 本清单逐一登记全部实验/管线配置的组归属（common 组集中列出 19 个配置文件）

**消费方**：`scripts/sync_cloud_worktrees.ps1`（`-Config` 默认值）与 `scripts/sync_cloud_worktrees.py`（DEFAULT_CONFIG）；`docs/cloud_dual_experiment_workflow.md`（L55-191 工作流说明）。

### 小结：生产配置演进脉络（png_baseline -> spectral_router_v5 -> prior_anchored -> perceptual_x0）

- **png_baseline**：无条纹底座——raw CT concat + 三个基础病灶损失，Gabor 路由全关，`validate_png_baseline_config` 把关 PNG 纯净性；后续所有配置都继承这份 schema 与数据路径。
- **residual_frequency / boundary_reliable（BR-F → BR-D v4）**：引入条件均值（先联合训练、v4 起冻结 pretrain_v2）与频率残差模块，边界可靠性门控与峰值损失进入消融（plan v3/v4）。
- **spectral_router_v5**：路由显式化——DCT/Gabor 描述子 + learned 三路 softmax + `spectral_router_regularization`；v6/v7/v8 计划修复冷启动与终校准，生产 v5 配置被 H3 契约按 sha256 冻结。
- **prior_anchored（100e → 300e → identifiable_full → background_tail_repair）**：H3 direct-PNG 先验锚定路由（`prior_anchored_learned`），Experiment D 解浅层冷启动死锁；identifiable_full 以三阶段训练 + 效用监督换取可归因性；background_tail_repair 冻结路由专修背景低频尾部。
- **perceptual_x0（plan_v1 → fullstack_p2_300e）**：P2_FEAT_GLOBAL 在五臂 Stage-A 胜出后逐字节叠加到 300e 全栈，成为最新生产候选；A/B/C/D 配对消融因契约不可配对而 BLOCKED，体现"先证伪后实现"。
- **契约侧并行线**：dataset_contract（数据指纹）→ pipeline_v1 十一门 H1-H6 → H4 FAIL → 05B 固定 schedule FAIL → H3-v2 全时间步 native/null 重校准 + H4-v2 嵌套 CV，构成与训练配置严格咬合的证据链。

