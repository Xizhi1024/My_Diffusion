# 第一个月进展报告：基于 CT 到 PET 模态转换的子宫内膜小目标分割

**日期：** 2026-01-14
**项目：** 基于 CT 到 PET 模态转换的子宫内膜小目标分割
**汇报人：** [您的名字]

---

## 执行摘要 (Executive Summary)
在过去的一个月中，项目经历了三个明显的阶段：从最初的架构分析，到定制化实现的对比，最后发展到目前的“MSRD-Diffusion”高级架构。本报告详细阐述了技术演进过程，早期阶段遇到的挑战（如预训练权重不匹配、模型不收敛），以及目前正在实施的解决方案。

---

## 第一阶段：初始模型架构分析
**状态：** 已完成
**结果：** 发现由于预训练模型不兼容导致的性能瓶颈。

### 1. 架构概览
*   **类型：** 基于 VAE 预训练模型的潜在扩散 (Latent Diffusion with VAE Pretrained Weights)。
*   **骨干网络 (Backbone)：** U-Net (2输入通道 [噪声+CT], 1输出通道 [噪声])。
    *   下采样路径：$128 \to 64 \to 32 \to 16 \to 8$。
    *   注意力机制：仅在最深层应用 ($16 \to 8$)。
*   **调度器 (Schedulers)：** DDPM (训练, 1000步, 线性 $\beta$) / DDIM (推理, 50步)。
*   **优化策略：** EMA (衰减率 0.995), 混合精度训练 (AMP), AdamW ($\lambda=0.01$)。

### 2. 医学图像特定处理
*   **预处理：** CT 窗宽窗位调整 (W:255, C:127.5), PET 归一化到 $[-1, 1]$。
*   **最佳性能指标：**
    *   PSNR: 9.41 | SSIM: 0.478
    *   MAE: 0.306 | MSE: 0.114

### 3. 遇到的关键问题
*   **预训练模型不匹配：** 直接使用标准预训练扩散模型的权重，对于医学 CT/PET 数据的特定分布效果不佳。
*   **配合度差：** 简单的条件输入机制（直接拼接）无法有效引导生成过程，导致生成的图像结构出现大量幻觉，且定量指标（PSNR/SSIM）极低。

---

## 第二阶段：My_diffusion 与 Diffusion/diffusion 的对比分析
**状态：** 已完成
**结果：** 完成架构重构以解决灵活性问题；识别出导致模型无法收敛的关键模式。

### 1. 对比分析
为了解决第一阶段的局限性，开发了定制代码库 `My_diffusion` 以更好地支持医学图像的条件生成。

| 特性 | 原始基准 (`Diffusion`) | 定制实现 (`My_diffusion`) |
| :--- | :--- | :--- |
| **项目性质** | 通用图像生成 | **医学条件生成 (CT $\rightarrow$ PET)** |
| **核心模型** | `ContinuousTimeGaussianDiffusion` | 新增 **`Conditional`** 子类以支持 CT 引导 |
| **骨干架构** | 标准 `KarrasUnet` | **`KarrasUnetWrapper`** (兼容 `diffusers` 接口) |
| **潜在空间** | 无 | **`LatentDiffusionModel`** (支持 Noise/x0/v-pred, Min-SNR loss) |
| **训练器** | 基础循环 | **`DiffusionTrainerV2`** (集成 EMA, AMP, 梯度累积, WandB) |
| **数据处理** | 通用 | **`MedicalImageDataset`** (由于配对 CT/PET 加载与归一化) |

### 2. 实验数据与量化分析 (Phase 2 Experimental Data)
这是一个关键的对比实验，数据揭示了定制模型在早期阶段的严重失效（模式崩溃）：

*   **My_diffusion (Phase 2) 评估指标:**
    *   **MAE:** 1.0129 (远高于第一阶段的 0.3067)
    *   **MSE:** 1.3532
    *   **RMSE:** 1.1633
    *   **NRMSE:** 0.5816
    *   **PSNR:** 4.7077 (极低，意味着图像质量不可用)
    *   **SSIM:** -0.0001 (结构相似性崩溃，说明模型未能生成与 CT 对应的解剖结构)
*   **训练状态 (Epoch 600):**
    *   Train Loss: 0.0006, Val Loss: 0.0007
    *   **解析：** 极低的 Loss 配合接近 0 的 SSIM，表明模型陷入了**收敛假象**。它极有可能是在输出“平均值”或“纯黑色/纯灰色”背景，从而在 MSE Loss 上表现良好，但完全丢失了语义信息。

相比之下，**第一阶段 (VAE Pretrained)** 虽然效果不佳（PSNR 9.41, SSIM 0.478），但至少保留了部分解剖结构。这一对比强有力地支持了我们在第三阶段转向 **MSRD-Net 双流架构** 的决定，因为单一的 U-Net encoder 显然不足以完成跨模态特征提取。

### 3. 遇到的关键问题
*   **模型由于无法收敛：** 尽管改进了架构，模型在训练过程中仍难以收敛。损失曲线过早进入平台期，且生成的 PET 图像缺乏明显的代谢热点（高 SUV 区域），整体偏灰暗或模糊。
*   **原因假设：** 标准 U-Net 编码器能力不足，难以从低对比度的 CT 图像中提取出预测代谢活动所需的复杂解剖特征。

---

## 第三阶段：当前架构 (MSRD-Diffusion)
**状态：** 进行中 (训练与验证阶段)
**目标：** 利用受 MSRD-Net 启发的双流架构，解决收敛和特征提取问题。

### 1. 架构创新
为了克服第二阶段的“无法收敛”问题，本阶段引入了更复杂的条件注入机制：

*   **双流条件编码器 (Dual-Stream Condition Encoder)：**
    *   替代了简单的通道拼接。
    *   **流 A (CNN)：** 提取高分辨率的纹理和边界特征。
    *   **流 B (Transformer/Global)：** 提取语义解剖理解（例如：定位特定器官区域）。
*   **多尺度 Stem (Multi-Scale Stem)：**
    *   在输入层并行使用 $3\times3$ 和 $7\times7$ 卷积，同时捕捉细微细节和更广阔的上下文信息。
*   **ControlNet 注入：**
    *   CT 特征通过 **ControlNet** (零卷积 Zero Convolutions) 注入到 Diffusion U-Net 中，确保扩散过程严格受解剖结构引导，同时不破坏生成的先验分布。

### 2. 高级损失策略
*   **加权肿瘤损失 (Weighted Tumor Loss)：** 针对高 SUV 区域赋予更高权重，专门解决“预测全黑”的问题。
*   **频域引导 (规划中)：** 计划引入 Focal Frequency Loss，以对齐纹理分布，减少图像模糊。

### 3. 当前进展状态
*   代码库已完全重构为符合 `MSRD-Diffusion` 标准。
*   **双流编码器** 和 **ControlNet** 模块已实现。
*   训练正在进行中，目前重点监控：
    *   收敛稳定性，确之前的无法收敛问题得到解决。
    *   肿瘤“热点”（高 SUV 值）的重建准确度。

### 4. 当前模型架构图解描述 (Architecture Diagram Description)
若要绘制当前 **MSRD-Diffusion** 的架构图，请参考以下逻辑流与组件描述：

### 4. 当前模型架构详述 (Detailed Architecture Specification)
以下描述基于 `src/model/` 下的代码实现（`encoder.py`, `control_net.py`），为架构图绘制提供精确参数：

1.  **输入层 (Input Layer)**
    *   **Main Input:** `Noisy PET` $(B, 1, 128, 128)$
    *   **Condition Input:** `Clean CT` $(B, 1, 128, 128)$

2.  **MSRD 双流条件编码器 (Dual-Stream Condition Encoder)**
    *   **位置:** `src/model/encoder.py` - `DualStreamCTEncoder`
    *   **(A) Multi-Scale Stem (多尺度茎块):**
        *   **Path 1:** $3\times3$ Conv (Padding 1) $\rightarrow$ 64 channels
        *   **Path 2:** $3\times3$ Conv $\rightarrow$ ReLU $\rightarrow$ $7\times7$ Conv (Padding 3) $\rightarrow$ 64 channels
        *   **Fusion:** Element-wise Sum $\rightarrow$ $1\times1$ Conv $\rightarrow$ GroupNorm(8) $\rightarrow$ SiLU $\rightarrow$ **Output: 64 channels**
    *   **(B) Stream 1: Texture Stream (CNN):**
        *   **结构:** 3个堆叠的 `ResnetBlock2D`
        *   **参数:** Kernel $3\times3$, GroupNorm(8), SiLU. **Maintains 64 channels.**
    *   **(C) Stream 2: Semantic Stream (Transformer):**
        *   **结构:** `EfficientChannelAttention` (Restormer-style)
        *   **机制:** Conv1x1 (QKV) $\rightarrow$ Channel-wise Attention (Heads=8, Dim=64) $\rightarrow$ Depthwise Conv $3\times3$ $\rightarrow$ Proj.
    *   **(D) Fusion Layer:**
        *   Concat(Stream1, Stream2) $\rightarrow$ 128 channels.
        *   Conv $1\times1$ $\rightarrow$ GroupNorm $\rightarrow$ SiLU.
        *   **Final Output:** **320 channels** (Matching SD latent dim).

3.  **ControlNet 注入模块 (Injection Module)**
    *   **位置:** `src/model/control_net.py`
    *   **Input:** 320 channel feature map from Encoder.
    *   **Control Model:** Trainable copy of U-Net Encoder blocks (Locked copy of SD Encoder).
    *   **Zero Convolution:** 初始化为 0 的 $1\times1$ 卷积，连接 Control Model 的每一层输出到 Main U-Net 的对应 Decoder 层。
    *   **Injection Points:** 12个注入点 (4 resolutions $\times$ 3 blocks/level) + Middle Block。

4.  **主生成模型 (Main Backbone)**
    *   **结构:** U-Net (Pixel-space variant or Latent variant depending on config).
    *   **Encoder:** $128 \to 64 \to 32 \to 16 \to 8$. (Downsampling)
    *   **Decoder:** $8 \to 16 \to 32 \to 64 \to 128$. (Upsampling + Skip Connections + **Control Signal Addition**).

---

## 下一步计划与预期目标 (Future Roadmap: Phase 6 & 7)
基于目前的 MSRD-Diffusion 架构，我们将目标延伸至模型的前沿优化与临床验证（对应原计划的阶段六与阶段七）：

### 1. 前沿模型优化 (Phase 6: SOTA Enhancements)
目标是将模型从“可用”提升至“最先进 (SOTA)”，解决生成质量的深层数学问题。
*   **引入 Brownian Bridge Diffusion (BBDM):**
    *   **目标：** 建立 CT 与 PET 之间的直接扩散桥梁，而非通过纯高斯噪声。这能最大程度保留 CT 的解剖结构，防止器官变形。
*   **2.5D 多视角一致性 (Multi-View Consistency):**
    *   **目标：** 同时利用 Axial/Coronal/Sagittal 三个切片训练，解决 2D 生成导致的层间不连续问题，实现伪 3D 效果。
*   **频域引导 (Frequency Guidance):**
    *   **目标：** 引入 Focal Frequency Loss，强制模型不仅在像素上，也在频率分布上对齐真实 PET，消除模糊伪影。

### 2. 临床可靠性验证 (Phase 7: Clinical Reliability)
目标是证明模型在医学临床上的安全性和有效性。
*   **不确定性量化 (Uncertainty Quantification):**
    *   **产出：** 为每一张生成的 PET 附带一张“不确定性热图 (Variance Map)”。帮助医生判断模型哪里生成的不可靠。
*   **临床特异性指标 (Clinical Metrics):**
    *   **SUV 误差分析：** 专门统计肿瘤区域的 SUVmax 绝对误差。
    *   **肿瘤检出率：** 验证生成的 PET 是否能辅助分割网络检测出微小病灶 (Small Object Segmentation)。
*   **频率一致性 (FMD):** 使用 Fréchet Medical Distance 替代传统的 FID，更准确评估医学图像质量。

---
可以，而且你的落脚点比“单纯 CT→PET 好看”更强：**生成一张对分割有用、在病灶/label 区域置信度高的 synthetic PET**。这时生成模型不必在全图每个像素都完美，重点是 ROI 热点、边界附近、假阳性控制。

**我的推荐主线**
做一个三分支系统：

`CT -> PET generator -> synthetic PET + hotspot/confidence map -> segmentation network`

生成器不只输出 PET，还输出：
- `PET/BQML_pred`
- `hotspot_prior`：疑似高摄取区域概率图
- `uncertainty/confidence`：哪里可信，哪里别太信

然后分割网络输入：

`[CT, synthetic PET, hotspot_prior, confidence]`

最后用真实 label 评估：生成 PET 是否让 Dice、Recall、HD95、病灶检出率提升。

**除了潜空间和桥扩散，还能做这些改进**
1. **ROI-aware / lesion-aware loss**
   不要只用全图 MSE/L1。你的任务是小目标，应该加：
   - mask 区域加权 L1/MSE
   - hotspot focal loss
   - top-k uptake loss：只管 PET 最亮的前 1% 或 5% 区域
   - background suppression：压低非病灶区假热点
   - segmentation-guided loss：把生成 PET 喂给冻结分割器，要求分割结果接近真实 label

2. **不确定性建模**
   生成器输出 `mean + logvar`，或者 MC sampling 多次生成 PET，得到 variance map。分割时让网络学会：高不确定区域少信，低不确定热点多信。你之前代码里已经有 heteroscedastic 相关雏形，这条很适合接上。

3. **2.5D 上下文**
   你的 PET 是 192×192，CT 是 512×512，单 slice 容易丢层间信息。比起直接 3D，大概率更稳的是：
   `CT_{z-2:z+2} -> PET_z`
   或者 `CT/PET 5-slice stack -> segmentation_z`。这对小病灶很有用，成本也比 3D 小。

4. **候选区域先验**
   先训练一个轻量 `CT -> ROI/hotspot prior` 网络。这个 prior 不必很准，只要给 diffusion 一个“多看这里”的软提示。可以监督它去拟合：
   - label mask
   - PET 高摄取阈值图
   - label 的 distance transform
   - label 边界图

**ControlNet 条件注入可以做，而且很适合**
但建议叫 **ControlNet-style medical control branch**，不要直接照搬 Stable Diffusion 的大模型设定。

可注入条件分三类：

**推理时可用条件**
- CT HU 图像，最好多窗：软组织窗、骨窗、宽窗
- CT edge / gradient / Canny-like 边缘图
- body mask / pelvic ROI / uterus-region coarse prior
- z 位置编码、spacing、slice thickness、FOV 等 metadata
- CT 邻近切片 2.5D stack

**训练时监督，但推理时不能直接输入**
- 真实 label mask
- 真实 PET hotspot mask
- 真实 PET/BQML top-k 区域
这些可以做 loss 或训练一个 prior predictor，但不能直接喂给生成器作为推理条件，否则任务泄漏。

**分割阶段可用条件**
- synthetic PET
- generated hotspot prior
- uncertainty/confidence map
- CT 原图

注入方式我推荐从轻到重：
- `concat`：最简单，CT/PET/noise 拼通道，作为 baseline。
- `FiLM/AdaGN`：把 z、spacing、slice thickness、病人级 metadata 注入归一化层。
- `ControlNet residual`：CT encoder 输出多尺度特征，经 zero conv 注入 diffusion U-Net 每个尺度。
- `cross-attention`：把 ROI token / hotspot token / CT global token 注入低分辨率层。

**TransUNet 要不要上？**
我的判断：**不要先把 diffusion U-Net 整体换成 TransUNet**。TransUNet强在分割，能补 U-Net 长程依赖，但 diffusion denoiser 还要处理 timestep/noise conditioning，直接换主干风险大。

更稳的用法：
- 分割网络可以用 `TransUNet / SwinUNETR / nnU-Net` 做 baseline 和最终 head。
- diffusion 主干保留 U-Net，只在 bottleneck 或低分辨率层加 transformer/self-attention。
- CT 条件 encoder 可以做成小型 TransUNet/Swin encoder，但不要把整个生成器一口气 transformer 化。

一句话设计：**生成器用 ControlNet-style 条件扩散，分割器用强医学分割网络，二者通过 synthetic PET + confidence map 连接。**

参考方向：CPDM 用 Attention/Attenuation maps 做 CT→PET 条件；ControlNet 用 zero-conv 注入空间条件；LDM 降低生成成本；BBDM 适合 image-to-image 桥接；TransUNet/SwinUNETR/nnU-Net 更适合作为分割侧强基线。  
来源：[CPDM](https://arxiv.org/abs/2410.21932)、[ControlNet](https://arxiv.org/abs/2302.05543)、[LDM](https://arxiv.org/abs/2112.10752)、[BBDM](https://arxiv.org/abs/2205.07680)、[TransUNet](https://arxiv.org/abs/2102.04306)、[Swin UNETR](https://arxiv.org/abs/2111.14791)、[nnU-Net](https://www.nature.com/articles/s41592-020-01008-z)。

我建议你先选一条实验主线：**A. ControlNet-style PET 生成 + 分割辅助**，还是 **B. 先做 PET/hotspot prior 轻量生成，再接 nnU-Net/TransUNet 分割**？