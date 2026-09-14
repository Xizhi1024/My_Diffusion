# SLMF-BBDM：CT→PET 图像合成与病灶小目标保留

面向子宫内膜癌 PET/CT 的研究型扩散模型仓库。核心模型 **SLMF-BBDM**：基于条件均值（conditional mean）残差的
Brownian Bridge 扩散（BBDM）骨干，叠加时间步条件化的层级频带（L2/L1 × LH/HL/HH）**spectral-evidence router**
（含 CT 支持底板），配合 τ 门控的病灶保留监督与归一化摄取代理（uptake proxy）监督，并集成 Gabor / Hotspot /
Organ / Semantic 等解剖-代谢先验与约 25 个损失项。研究目标：在由 CT 合成 PET 图像时，可靠保留 CT 上可见的
小病灶摄取区域。

> 本仓库只包含代码与配置。医学影像数据、训练产物（checkpoint / 日志 / 评估输出）与参考文献均不入库。

## 目录结构

| 目录 | 说明 |
|---|---|
| `src/data/` | 数据层：DICOM/PNG 清单、张量缓存、五折划分、器官/语义预处理、血统(lineage)追踪 |
| `src/model/` | 模型核心：SLMF-BBDM、UNet 骨干（bbdm/wavelet）、trainer、registry、EMA、均值预测器 |
| `src/model/loss_terms/` | ~25 个损失项与损失聚合体系 |
| `src/model/{conditioning,noise,priors,frequency}/` | 条件适配 / β 调度、噪声调度、先验生成、频域分析与路由 |
| `src/mechanism_validation/` | 机制验证库：H1–H6 假设检验、H4-v2 内部 CV、特征溯源/因果/涌现审计 |
| `scripts/` | 训练/评估入口、消融 runner、机制验证与 V2 管线脚本 |
| `configs/` | 实验配置（yaml 消融计划 + json 机制管线契约） |
| `tests/` | pytest 测试套件 |
| `wuzhe/comparison_experiments/` | 对比 baseline：pix2pix / cyclegan / reggan / district_gan / cpdm |
| `CODEMAP.md` | 全仓库代码地图（职责 / 接口 / 依赖 / 数据流，中文） |

## 环境安装

依赖通过 [pixi](https://pixi.sh) 管理（Python 3.11–3.13，PyTorch ≥ 2.8 + CUDA 12.8）：

```bash
pip install pixi  # 或参考 pixi 官方安装方式
pixi install
```

> `pixi.toml` 内置了清华 conda/PyPI 镜像与 PyTorch cu128 源；如在海外环境可自行替换为官方源。

## 数据准备（数据不入库）

**路线 A：PNG 数据（当前推荐的生产路线，无 DICOM/SUV 依赖）**

将 CT/PET/mask 的 PNG 数据与 `split_manifest.csv` 放置于 `main_data/`，然后：

```bash
pixi run data-cache-png-main   # 生成 cache/tensors_main 张量缓存
```

**路线 B：DICOM 数据**

将原始 DICOM 放置于 `Data/`，然后：

```bash
pixi run data-setup   # manifest → npz 张量缓存 → 划分清单 → 器官先验（需 GPU + TotalSegmentator）
```

## 训练与评估

```bash
pixi run train-png-baseline   # PNG baseline（推荐的稳定入口）
pixi run train                # 全模块启用（DICOM 路线）
pixi run train-smoke          # 假数据冒烟，验证管线
pixi run eval-png-baseline    # 测试集评估
```

内置消融任务（`train-no-gabor` / `train-no-hotspot` / `train-no-organ` / `train-base-only` 等）与
频带筛选管线（`train-frequency-ablations[-v2]`、`train-spectral-router-v5`）见 `pixi.toml` [tasks] 段。

## 机制验证

```bash
pixi run mechanism-validate-local   # 本地范围 H1–H6 假设检验
pixi run mechanism-validate-all     # 全量（fail-closed：任一验证器未实现即失败）
pixi run mechanism-v2-freeze        # 冻结 H4-v2 内部 CV 计划
```

## 测试

```bash
pixi run test          # 完整测试套件
pixi run test-smoke    # 冒烟
```

## 声明

本项目仅用于科学研究。医学影像数据不随仓库分发；使用本项目处理真实临床数据时，请遵守所在机构的
伦理与数据合规要求。
