# 对比模型复现状态

目标：在统一 `batch["ct"] -> result["synthetic_pet"]` 输入输出接口的前提下，尽可能贴近原文或官方源码。

| 模型 | 当前复现状态 | 估计复现度 |
|---|---|---:|
| Pix2Pix | 官方式 U-Net/ResNet 风格生成器可配置、PatchGAN、LSGAN、L1，生成器/判别器分步优化 | 85–90% |
| CycleGAN | 双生成器、双判别器、cycle loss、identity loss、LSGAN、fake image pool，训练流程贴近官方 PyTorch 代码 | 85–90% |
| RegGAN | 保留生成网络、注册网络、形变校正监督、flow smoothness、判别器分步优化 | 75–85% |
| CPDM | 已从直接图像预测升级为 VQGAN-like first stage、3 通道 latent、9→3 UNet、Brownian Bridge、`objective=grad`、attention/attenuation 条件 | 70–80% |
| District-specific GAN | 区域特异性生成器/判别器、软区域 mask、随机 patch 训练、滑窗重叠平均推理；无真实 `district_mask` 时使用 2D 近似区域 | 65–75% |

## 仍然存在的客观差距

- CPDM：当前实现是统一接口下的内置 VQGAN-like first stage；若要进一步逼近官方代码，需要提供或训练官方 VQGAN checkpoint，并接入官方 latent checkpoint 加载逻辑。
- District-specific GAN：原文是 whole-body 3D patch。当前项目数据接口以 2D CT/PET slice 为主，因此 3D patch、真实 head/trunk/arms/legs 标签只能做接口兼容和 2D 近似。若提供真实 `district_mask` 或 3D volume dataloader，复现度还能继续提高。

