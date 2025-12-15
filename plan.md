这是一份为您定制的**CT to PET 模态转换模型实施计划**。

这份计划完全基于 **Latent Diffusion Model (LDM)** 架构，并将 **MSRD-Net** 论文中的核心创新点（双流架构、多尺度Stem、特征融合）提取为**具体的工程需求**。

**关键策略：** 我们将 MSRD-Net 的架构思想“移植”到 Diffusion 模型的**条件编码器（Condition Encoder）**中。我们将使用这个自定义编码器来提取 CT 的特征，并通过 ControlNet 的方式注入到 Diffusion 的生成过程中。

---

### **总体架构设计：MSRD-Diffusion (CT $\rightarrow$ PET)**

* **基座模型 (Base Model):** 使用 Stable Diffusion (或类似 U-Net) 的 AutoencoderKL 将图像压缩到潜空间 (Latent Space)。
* **输入 (Condition):** CT 图像 (高分辨率解剖结构)。
* **输出 (Target):** PET 图像 (代谢功能热图)。
* **核心创新模块 (The "Brain"):** 一个自定义的 **Dual-Stream Condition Encoder**。它替代了普通的简单的 CNN 编码器，负责从 CT 中提取深层的解剖与语义信息。

---

### **实施路线图**

#### **第一阶段：数据预处理 (针对 CT $\rightarrow$ PET 优化)**

**核心痛点：** CT 的数值范围（HU值）与 PET（SUV值）完全不同，且 PET 数据极度稀疏（大部分是背景）。

1.  **CT 标准化 (Windowing & Normalization):**
    * **动作：** 不要直接归一化。先应用**软组织窗 (Soft Tissue Window)**（例如：窗宽 400，窗位 50），截断范围之外的数值。
    * **归一化：** 将截断后的数据线性映射到 `[-1, 1]`。
2.  **PET 标准化:**
    * **动作：** 计算整个数据集的 `max_SUV`（通常是 10-20 左右），将数值除以该最大值归一化到 `[0, 1]` 或 `[-1, 1]`。

#### **第二阶段：构建“双流”条件编码器 (MSRD-Style Encoder)**

这是从论文中提取的精华。我们需要构建一个编码器，包含以下三个部分：

1.  [cite_start]**Multi-Scale Stem (多尺度茎块):** [cite: 619]
    * **原理：** 传统的卷积核太单一。MSRD-Net 使用 $3\times3$ 和 $7\times7$ 并行卷积来同时捕捉细微边缘和大感受野特征。
    * **移植：** 作为编码器的第一层入口。
2.  [cite_start]**Dual-Stream Body (双流主体):** [cite: 501, 666]
    * **Stream A (CNN / Local):** 使用残差块 (ResBlock) 提取 CT 丰富的纹理和边界信息。
    * **Stream B (Transformer / Global):** 使用轻量级 Transformer (如 Restormer 块) 提取解剖语义（例如：“这是肝脏区域，通常有特定的代谢水平”）。
3.  **Fusion (融合):**
    * 将两路特征在通道维度拼接 (Concat)，然后通过线性层 (1x1 Conv) 融合。

#### **第三阶段：注入与生成 (ControlNet 模式)**

1.  **注入机制：** 使用 **ControlNet** 架构。
2.  **流程：**
    * CT 图像 $\rightarrow$ **双流编码器** $\rightarrow$ 多尺度特征图。
    * 多尺度特征图 $\rightarrow$ **Zero Convolution** $\rightarrow$ 加或是注入到 Diffusion U-Net 的 Decoder 层。

#### **第四阶段：损失函数 (Loss Function)**

1.  **基础 Loss:** 噪声预测 MSE Loss。
2.  **功能一致性 Loss (针对 PET 优化):**
    * PET 只有肿瘤或特定器官亮。普通的 MSE 会导致模型倾向于生成全黑图像。
    * **策略：** 实现一个 **Weighted L1 Loss**，对高像素值区域（疑似肿瘤区域）赋予 10倍~50倍 的权重。

---

### **AI 编程助手提示词 (Prompts)**

请按顺序将以下 Prompt 发送给你的代码生成助手。这些 Prompt 已经**隐式包含了 MSRD-Net 的论文细节**，无需提供原论文。

#### **Prompt 1: 定义多尺度 Stem 模块 (The Multi-Scale Input)**

> **Prompt:**
> "I need to implement a specialized PyTorch module called `MultiScaleStem` for processing medical CT images.
>
> **Requirements:**
> 1.  **Input:** Takes a tensor `x` (Batch, Channels, Height, Width).
> 2.  **Logic:** It should implement a specific multi-scale convolution path inspired by advanced feature extractors. The formula is:
>     `Output = Conv1x1( Conv3x3(x) + Conv7x7(Conv3x3(x)) )`
> 3.  **Details:**
>     * Use `padding='same'` equivalent so output spatial dimensions match input.
>     * Include Batch Normalization and ReLU after the $3\times3$ and $7\times7$ convolutions.
>     * The final `Conv1x1` fuses the features into the desired `out_channels`.
>
> Please write the clean PyTorch code for this class."

#### **Prompt 2: 定义双流条件编码器 (The Core Dual-Stream Encoder)**

> **Prompt:**
> "Now, I need a feature extractor named `DualStreamCTEncoder` that will condition a Diffusion model. This encoder takes a CT image and outputs a high-level feature map.
>
> **Architecture Design:**
> It must process the input through two parallel streams to capture both **Local Details** (Texture) and **Global Context** (Anatomy):
>
> 1.  **Entry:** Use the `MultiScaleStem` module (from the previous step) to map input to 64 channels.
> 2.  **Stream 1 (CNN Path):** A stack of 3 standard ResNet Blocks. This captures the sharp edges of the CT scan.
> 3.  **Stream 2 (Transformer Path):** A simplified Vision Transformer block (or Restormer block). It should:
>     * Reshape the image to a sequence.
>     * Apply LayerNorm and Self-Attention (capturing long-range dependencies).
>     * Reshape back to image format.
> 4.  **Fusion:** Concatenate the outputs of Stream 1 and Stream 2 along the channel dimension, then use a $1\times1$ Convolution to reduce channels back to the target embedding dimension.
>
> **Constraint:** Pure PyTorch implementation. Ensure input and output resolutions remain the same."

#### **Prompt 3: 定义 ControlNet 注入类**

> **Prompt:**
> "I am building a **ControlNet-based** Latent Diffusion Model for CT-to-PET translation.
>
> Please implement the `CTControlNet` model class.
>
> **Structure:**
> 1.  **Inputs:** `noisy_latents` (from VAE), `timesteps`, and `control_image` (the CT image).
> 2.  **Conditioning:** Pass the `control_image` through the `DualStreamCTEncoder` defined above.
> 3.  **Control Mechanism:**
>     * The model should contain a trainable copy of the Stable Diffusion U-Net's **Encoder** blocks.
>     * The features from `DualStreamCTEncoder` are added to the input of these encoder blocks.
>     * **Zero Convolutions:** Implement a `ZeroConv2d` layer at the output of each encoder block before returning the control signals.
>
> **Goal:** This module will output a list of feature maps to be added to the main Frozen U-Net's decoder."

#### **Prompt 4: 定义加权损失函数 (Tumor-Aware Loss)**

> **Prompt:**
> "The target domain is **PET images**, which are sparse (mostly dark background with bright hotspots/tumors). Standard MSE loss often causes the model to generate blurry or blank images.
>
> Please write a custom loss function `WeightedPETLoss`.
>
> **Logic:**
> 1.  Calculate the L1 difference (absolute error) between the `predicted_pet` and `ground_truth_pet`.
> 2.  **Weighting:** Create a weight map based on the `ground_truth_pet`.
>     * Pixels with high SUV values (e.g., > 0.5 in normalized scale) should have a weight of 10.0.
>     * Background pixels have a weight of 1.0.
> 3.  Return the weighted mean of the L1 loss.
>
> This will ensure the model focuses on reconstructing the tumors accurately."

### **总结**

这套计划做到了：
1.  **不依赖论文文件：** 所有 MSRD-Net 的具体公式（如 $1\times1(3\times3 + 7\times7)$）都已经拆解在 Prompt 中。
2.  **方向正确：** 针对 CT $\rightarrow$ PET 进行了逻辑适配（强调从 CT 提取结构，Loss 强调 PET 的热点）。
3.  **技术栈主流：** 使用 Diffusion + ControlNet，这是目前图像翻译任务的最优解 (SOTA)。