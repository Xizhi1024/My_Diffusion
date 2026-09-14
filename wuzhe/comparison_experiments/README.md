# CT→PET 对比实验模型

这个目录是独立新增的对比实验代码，不改动主模型 `SLMF-BBDM / MF_DIFFUSION` 的实现文件。

统一接口：

```python
loss, logs = model(batch)
result = model.sample(batch)
synthetic_pet = result["synthetic_pet"]
```

输入与主模型保持一致：

- `batch["ct"]`: CT，形状 `[B, 1, H, W]`
- `batch["pet"]`: PET，形状 `[B, 1, H, W]`
- 可选：`mask`, `mu_map`, `ct_hu`, `district_mask`

## 五个模型

| 模型 | 年份 | 复现依据 | 损失函数策略 |
|---|---:|---|---|
| Pix2Pix | 2017 | 官方 PyTorch CycleGAN-and-pix2pix 代码结构 | 原版 LSGAN + L1 |
| CycleGAN | 2017 | 官方 PyTorch CycleGAN-and-pix2pix 代码结构 | 原版 LSGAN + cycle loss + identity loss |
| RegGAN | 2021 | 公开 Reg-GAN 代码与论文描述 | 对抗损失 + 注册校正 L1 + flow smoothness |
| CPDM | 2024/2025 | 官方 CPDM/WACV 2025 代码与论文描述 | Brownian Bridge CT→PET 重建损失 + attention 加权损失；基础重建项与主模型对齐 |
| District-specific GAN | 2025 | 论文描述复现 | 区域 Pix2Pix 对抗损失 + 区域 L1；无官方可复用代码时按论文实现 |

## 复现策略

本目录的目标不是简单给主模型 Trainer 套一个 baseline，而是在输入输出接口统一的前提下，尽量保留原文/源码的训练协议：

- Pix2Pix：生成器和判别器分开优化，PatchGAN + LSGAN + L1。
- CycleGAN：双生成器、双判别器、cycle loss、identity loss，并加入官方实现中的 fake image pool。
- RegGAN：生成网络、注册网络和判别器分开组织，使用形变校正后的目标图像监督生成结果。
- CPDM：保留 CT→PET Brownian Bridge、attention map、attenuation map 条件；内部已改为 VQGAN-like first stage + 3 通道 latent + 9→3 UNet + `objective=grad`，尽量贴近官方 CPDM 配置。若后续提供官方 VQGAN 权重，可继续接入 checkpoint 加载。
- District-specific GAN：按论文思想复现区域特异性生成和判别；训练时使用随机 patch，推理时使用滑窗重叠平均拼接。若数据中没有 `district_mask`，默认用 2D 解剖顺序软区域近似 head/trunk/arms/legs。

## 训练示例

```bash
python comparison_experiments/train_comparison.py --config comparison_experiments/configs/pix2pix.yaml
python comparison_experiments/train_comparison.py --config comparison_experiments/configs/cpdm.yaml
```

PowerShell 一次性排五个对比实验：

```powershell
powershell -ExecutionPolicy Bypass -File comparison_experiments/run_all_comparisons.ps1 -Epochs 1000
```

## 主实验 + 五折交叉验证总流程

当前项目数据已对接：

- 主实验：`main_data/train` 和 `main_data/val`
- 五折交叉验证：`wuzhe_data/fold_1` 到 `wuzhe_data/fold_5`，每折内部均为 `train` / `val`
- 默认 CT 输入目录：`ct`
- 默认 PET 目标目录：`pet_peizhuan`，即优先使用配准 PET；如需改回原始 PET，可传 `-PetDir pet`
- 标签目录：`label`

一键运行顺序固定为：

1. 先跑完所有模型的主实验；
2. 再依次跑五折交叉验证；
3. 每个实验都用训练集训练、验证集选择 `ckpt_best.pt`；
4. 最终指标在验证集上由 `ckpt_best.pt` 计算。

```powershell
powershell -ExecutionPolicy Bypass -File comparison_experiments/run_main_then_fivefold.ps1 -Epochs 1000
```

默认早停机制：

- `warmup_epochs = 50`
- `patience = 80`
- `min_delta = 1e-4`

含义：前 50 个 epoch 不触发早停；之后如果验证集 loss 连续 80 个 epoch 没有至少下降 `1e-4`，则停止该模型当前实验。验证集最优权重保存为 `ckpt_best.pt`。

可手动调整：

```powershell
powershell -ExecutionPolicy Bypass -File comparison_experiments/run_main_then_fivefold.ps1 `
  -Epochs 1000 `
  -EarlyStopPatience 100 `
  -EarlyStopMinDelta 0.0001 `
  -EarlyStopWarmup 100
```

指标输出位置：

```text
output/comparison_metrics/
```

其中：

- `all_metrics_summary.csv`：所有模型、主实验和五折的汇总指标
- `all_metrics_summary.json`：同样内容的 JSON 版
- 每个实验子目录下有 `metrics_summary.json` 和 `metrics_per_sample.csv`

先预览命令、不真正训练：

```powershell
python comparison_experiments/run_all_png_experiments.py --dry-run
```

快速冒烟测试可覆盖为假数据和小模型：

```bash
python comparison_experiments/train_comparison.py --config comparison_experiments/configs/pix2pix.yaml --override data.use_fake_data=true --override data.cache_dir= --override data.image_size=32 --override data.batch_size=1 --override data.val_batch_size=1 --override comparison_model.base_channels=8 --override training.num_epochs=1 --override runtime.num_workers=0 --override runtime.amp=false --override runtime.channels_last=false --override runtime.eval_interval=999 --override runtime.sample_interval=999 --override runtime.save_interval=999
```

输出 checkpoint 与元信息会放到当前项目目录下的 `checkpoints/<experiment.name>/`。
