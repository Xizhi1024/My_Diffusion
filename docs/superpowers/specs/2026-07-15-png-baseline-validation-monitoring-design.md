# PNG Baseline 验证监控修复设计

日期：2026-07-15  
状态：已批准，待实施

## 背景

PNG-only baseline 已完成 300 epoch 首训。训练数据链路与 PET 极性已经验证正确，但当前验证监控不能可靠回答模型是否学到病灶：

- PET 张量位于 `[-1, 1]`，现有指标却通过 `prediction * mask` 选择区域。病灶内预测为负时，mask 外人为产生的 0 会赢得 `argmax`，造成约 50 px 的假质心距离以及数量级异常的峰值比。
- `val/mae`、`val/ssim` 和病灶指标只在 6 个可视化追踪样本上计算，不代表 237 个验证样本；配置中的 `runtime.eval_num_samples` 尚未参与该流程。
- 每次采样使用新的随机噪声，跨 epoch 指标和可视化不完全可比。
- best checkpoint 保存先更新 `_best_combined_score`，early-stop 随后再与该值比较，导致同一次提升无法被 early-stop 识别。
- `patience` 的注释声称单位为 epoch，实际却按每次评估调用累计；当 `eval_interval=50` 时，`patience=40` 等价于最多 2000 epoch。
- 日志只打印 total loss，无法判断 base、ROI、top-k 和 ranking 项的相对量级。

现有 loss 实现已经按 ROI 或 mask 像素数归一化，因此本修复不以“病灶面积造成约 200:1 稀释”为前提，也不立即重调 loss 权重。

## 目标

1. 使所有病灶监控指标在 `[-1, 1]` 模型空间下数学正确、量纲明确。
2. 使用固定、可重复、覆盖小中大病灶的验证子集进行模型选择。
3. 使 best checkpoint 与 early stopping 共享同一个、正确的“本次是否提升”判断。
4. 让 patience 真正表示 epoch 数。
5. 在终端输出关键 loss 分量，支持下一轮是否调权重的证据判断。
6. 保持模型结构、缓存格式和训练 loss 公式不变。

## 非目标

- 不修改 UNet、BBDM 采样公式或条件输入。
- 不把 lesion mask 作为推理输入。
- 不在本次改动中提高 lesion loss 权重。
- 不重建已反转的 PNG cache。
- 不把 237 个验证样本全部执行 20-step sampling，以避免显著增加云端费用。

## 设计

### 1. 验证样本与可视化样本

`Trainer` 从验证集按 mask 面积排序，并确定性选择 `runtime.eval_num_samples` 个近似等分位样本。默认配置继续使用 16 个样本，覆盖从小到大的病灶面积分布。

同一固定批次用于采样指标和 checkpoint 选择；样本图仍只绘制前 4 个，避免扩大图片。若验证集不足 16 个，则使用全部样本。显式配置的 `tracked_sample_ids` 仍优先，并只选择匹配样本。

### 2. 确定性采样

增加 `runtime.eval_seed`，默认继承实验 seed。验证采样在隔离的 RNG 上下文内使用固定 seed，保证：

- 不同 epoch 使用相同初始噪声；
- 指标采样与同 epoch 保存的可视化复用同一结果，避免重复采样和噪声不一致；
- 验证过程不永久改变训练 RNG 状态。

### 3. Signed-range 安全指标

指标计算保留 SSIM 的 `data_range=2.0`，但与强度和比例有关的 PET 值先通过 `(x + 1) / 2` 转换到 `[0, 1]` 并裁剪。

区域选择一律使用布尔索引，不使用乘 mask 后取最大值：

- `lesion_peak_error_norm`：`abs(max(pred[mask]) - max(target[mask]))`。
- `lesion_centroid_distance`：在 mask 内找到预测峰值坐标，再计算其与 mask 质心的欧氏距离。
- `outside_inside_peak_ratio`：`max(pred[outside]) / max(max(pred[mask]), eps)`。
- `failure_rate`：外部峰值严格高于内部峰值的样本比例。
- `lesion_roi_l1`：在 `[0, 1]` 空间按有效 mask 像素归一化。

无 mask 或空 mask 样本不进入病灶指标分母；同时报告有效病灶样本数，避免静默把无效样本当作 0。

### 4. 模型选择与 early stopping

每次评估只计算一次 lesion、image 和 combined score。checkpoint 保存函数返回 `combined_improved`，early-stop 直接消费这个布尔值，不再二次比较已经更新的 best score。

Trainer 记录 combined 最近一次提升的 epoch。若当前评估 epoch 与最近提升 epoch 的差达到 `early_stopping.patience`，才停止训练。因此 patience 的单位与 YAML 注释一致，确实是 epoch；停止只能发生在评估点。

best checkpoint 继续分别保存 `best_lesion`、`best_image` 和 `best_combined`。模型选择默认使用 `best_combined`，不把 `best_image` 当成病灶任务的默认结果。

### 5. 可观测性

每次训练 epoch 的简要行保持不变。每次验证额外打印下列平均分量：

- base diffusion；
- lesion ROI L1；
- top-k lesion；
- outside peak ranking；
- total loss。

这些值来自现有日志字典，不改变反向传播。下一轮只有在证据显示辅助项实际贡献过弱或验证病灶指标无改善时，才进入单变量 loss 调参。

## 测试设计

1. 构造病灶内预测为负、mask 外为更低负值的张量，证明峰值坐标仍被限制在 mask 内，比例有限且强度转换正确。
2. 构造空 mask 与有效 mask 混合批次，证明病灶指标只统计有效样本。
3. 构造连续 combined score 序列，证明提升会重置最近提升 epoch，未提升达到指定 epoch patience 后才停止。
4. 证明 checkpoint 保存返回正确的 `combined_improved`，且首次评估被视为提升。
5. 证明 `eval_num_samples=16` 时选择结果固定、无重复并覆盖病灶面积分布。
6. 证明相同 `eval_seed` 的采样输出可重复，且不会永久改变外部 RNG 状态。
7. 运行现有 `tests/test_png_baseline.py` 与 `tests/test_smoke.py`，确认数据链、loss 和训练入口无回归。

## 云端验收

同步修改后，优先加载现有 epoch 100 的 `ckpt_best_combined.pt`，用修复后的固定 16 例协议重新评估。只有当修复后的指标仍显示病灶强度或定位明显不足时，才启动第二轮训练。

若需要第二轮训练，先沿用原 loss 权重，并将 `eval_interval` 调整到足以执行 epoch-based patience 的频率；loss 重加权作为后续独立实验，不与本修复混合。
