这是一个非常棒的学习路径。既然你无法直接获取 MSRD-Net 的代码，我们可以提取它的**核心设计哲学（Dual-Stream, Global-Local Fusion）**，并将其“移植”到一个标准的、开源社区支持良好的 **Latent Diffusion Model (LDM)** 架构中。

我们不依赖 MSRD-Net 的具体实现，而是用标准的 PyTorch 组件重构它的思想。

### **总体架构蓝图 (Conceptual Architecture)**

我们要构建一个 **Conditional Latent Diffusion Model (cLDM)**。

1.  **主干 (Backbone):** 使用标准的 **U-Net** (带 Cross-Attention) 进行去噪。
2.  **潜空间 (Latent Space):** 使用预训练好的 **VQ-GAN** 或 **AutoencoderKL** (如 Stable Diffusion 用的那个) 将高分辨 CT 压缩为潜变量，降低显存需求。
3.  **条件编码器 (The "MSRD-Net" Spirit):** 这是我们需要自己写的部分。代替简单的 `cat(PET, Noise)`，我们设计一个 **Dual-Stream Encoder** 来提取 PET 特征，分别捕捉**解剖轮廓 (Global)** 和 **纹理边界 (Local)**，然后注入到主干中。

-----

### **实施计划 (Implementation Roadmap)**

#### **第一阶段：搭建基础 Diffusion 框架**

不要从零写 Diffusion。使用成熟的库，如 HuggingFace 的 `diffusers`。

  * **目标：** 跑通一个无条件的 CT 生成模型，或者简单的 `Concat` 条件模型。
  * **关键点：** 搞定数据加载器（Dataloader），加载成对的 PET/CT NIfTI/DICOM 数据。

#### **第二阶段：复现“双流”思想 (The Dual-Stream Encoder)**

这是从 MSRD-Net 借鉴的核心。我们需要一个编码器将 PET 图像转换成 Embedding。

  * **Stream 1 (CNN):** 提取局部特征（模拟 MSRD-Net 的 CNN 分支）。用简单的 ResNet Block 即可。
  * **Stream 2 (ViT/Attention):** 提取全局特征（模拟 MSRD-Net 的 Transformer 分支）。用简单的 Self-Attention Block。
  * **融合：** 将两路特征相加或拼接。

#### **第三阶段：特征注入 (Conditioning)**

将第二阶段提取的 PET Embedding 注入到 Diffusion U-Net 中。

  * **方法 A (ControlNet 方式 - 推荐):** 复制 U-Net 的 Encoder 部分作为“控制流”，将 PET 特征加进去。这是目前最强的空间控制方法。
  * **方法 B (Cross-Attention):** 将 PET Embedding 视为 "Prompt"，通过 Cross-Attention 层注入。

#### **第四阶段：训练与损失函数**

  * **Loss:** 标准的 MSE (噪声预测) + **Perceptual Loss** (参考 MSRD-Net 的 Sensing Loss，用预训练的 VGG 或医学 ResNet 计算特征距离，防止生成的 CT 纹理失真)。

-----

### **AI 编程助手提示词 (Prompts)**

你可以按顺序将以下提示词发送给 ChatGPT/Claude/Cursor，让它帮你生成代码框架。

#### **Step 1: 定义双流条件编码器 (The "MSRD-Net" Inspired Module)**

> **Prompt:**
> "I need to implement a custom feature extractor for a medical image translation task (PET to CT) in PyTorch.
>
> Inspired by 'Dual-Stream' architectures, I want a module named `PETConditionEncoder` that takes a 1-channel PET image input and outputs a feature map.
>
> The module should have two parallel branches:
>
> 1.  **Local Stream:** A stack of 3 ResNet blocks (use standard convolutions) to capture edges and textures.
> 2.  **Global Stream:** A simplified Vision Transformer (ViT) block or a Self-Attention layer to capture global anatomical shape from the sparse PET signals.
>
> **Requirements:**
>
>   * Fuse the outputs of both streams (e.g., concatenation + 1x1 conv).
>   * Do NOT use external complex libraries, just pure `torch` and `torch.nn`.
>   * The output resolution should match the input resolution (maintain spatial dimensions).
>   * Include comments explaining that the Transformer stream is for 'Global Context' and CNN is for 'Local Details'."

#### **Step 2: 定义基于 ControlNet 思想的注入机制**

*ControlNet 是目前将空间条件（如 PET）注入 Diffusion 的最佳实践，它完美契合 MSRD-Net 想要保留空间结构的初衷。*

> **Prompt:**
> "Now, I want to use the `PETConditionEncoder` defined above to condition a Latent Diffusion Model. I want to adopt a **ControlNet-like** architecture.
>
> Please write a PyTorch definition for a `PETControlNet` model.
>
> **Structure:**
>
> 1.  It should accept the noisy latent `x_t` and the condition image `c` (PET).
> 2.  It uses the `PETConditionEncoder` to process `c`.
> 3.  It contains a copy of the Stable Diffusion U-Net Encoder (just placeholder code or use `diffusers.models.UNet2DConditionModel` encoder blocks).
> 4.  Implement the 'Zero Convolution' layer logic to connect the encoded PET features to the main U-Net.
>
> **Goal:** Show me how the forward pass allows the PET features to be injected into the main network's decoder."

#### **Step 3: 定义感知损失 (Perceptual Loss)**

*MSRD-Net 强调了 Sensing Loss 的重要性，用于强制生成图像保留源图像的信息。*

> **Prompt:**
> "For the loss function of this PET-to-CT Diffusion model, purely using MSE on noise prediction might result in blurry textures.
>
> Please implement a custom Loss Module `MedicalPerceptualLoss`.
>
> **Requirements:**
>
> 1.  It should use a pre-trained ResNet (e.g., ResNet18 or VGG) as a feature extractor.
> 2.  The loss should calculate the L1 distance between the feature maps of the **Predicted CT** (reconstructed from predicted noise) and the **Ground Truth CT**.
> 3.  Explain how to combine this with the standard Diffusion MSE Noise Loss.
> 4.  (Optional) If possible, simulate the 'Sensing Loss' idea: ensuring the generated CT, when downsampled/encoded, structurally matches the PET input's anatomical layout."

-----

### **核心代码片段预览 (学习用)**

为了让你更好地理解思路，这里写一段 Step 1 中提到的 **不依赖 MSRD-Net 代码的双流模块** 雏形：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalStream(nn.Module):
    """
    CNN Branch: Focuses on edges and local texture.
    Mimics the 'ResBlock' or 'ScConv' part of MSRD-Net.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        return out + residual

class GlobalStream(nn.Module):
    """
    Transformer Branch: Focuses on global shape and distant dependencies.
    Mimics the 'Restormer' part of MSRD-Net.
    """
    def __init__(self, channels, size):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        # Simplified Self-Attention for demonstration
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        
    def forward(self, x):
        b, c, h, w = x.shape
        # Flatten spatial dims for Transformer: [B, C, H, W] -> [B, H*W, C]
        flat_x = x.flatten(2).transpose(1, 2)
        norm_x = self.norm(flat_x)
        
        # Self-Attention
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        
        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        return out + x # Residual connection

class DualStreamPETEncoder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, img_size=128):
        super().__init__()
        # Initial projection
        self.stem = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        
        # The Two Streams
        self.local_stream = LocalStream(embed_dim)
        self.global_stream = GlobalStream(embed_dim, img_size)
        
        # Fusion Layer (1x1 Conv)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        
        # Parallel Processing
        local_feat = self.local_stream(x)
        global_feat = self.global_stream(x)
        
        # Concatenate and Fuse
        combined = torch.cat([local_feat, global_feat], dim=1) # [B, 2*C, H, W]
        out = self.fusion(combined)
        
        return out
```

### **总结**

通过这一套流程，你虽然没有用一行 MSRD-Net 的源码，但你通过 **DualStreamPETEncoder** 实现了它最核心的“CNN+Transformer”互补思想，并利用 **ControlNet** 这一现代 Diffusion 架构实现了比原论文可能更强大的生成能力。