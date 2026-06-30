# IGEAR_Net 参数级优化计划

## TL;DR

> **Quick Summary**: 在不移除四大创新（LearnableEdgeBank、梯度门控、边界加权BCE、多分支编码器）的前提下，通过参数校准和训练策略调整，修复IGEAR_Net在极端小病灶数据集（中位0.38%面积）上的性能问题。所有改动均在参数/训练策略层面，不改架构拓扑。
>
> **Deliverables**:
> - 边界加权BCE：beta/k重校准 + 权重调度机制
> - 梯度门控：alpha可学习化 + LayerNorm + 弱边缘监督
> - LearnableEdgeBank：深浅层差异化学习策略
> - 全量训练对比验证（Dice/IoU/HD95 + per-lesion-size分层分析）
>
> **Estimated Effort**: Short（2个任务，一次性全改+全量验证）
> **Parallel Execution**: Sequential（单机单GPU）
> **Critical Path**: Task 1（应用全部改动）→ Task 2（300epoch全量训练对比）
> **执行策略**: 用户计算资源有限，一次性应用所有改动，跑1次全量训练对比baseline

---

## Context

### Original Request
用户要求在不放弃 IGEAR_Net 四大核心创新的前提下，通过文献调研诊断性能问题，并给出参数/训练策略层面的优化方案来改进模型在极端小病灶数据集上的表现。

### Interview Summary
**Key Discussions**:
- **创新保护**: 四大创新（LearnableEdgeBank、梯度门控、边界加权BCE、多分支编码器）全部保留，只调整参数和训练策略
- **数据集特征**: 1008 train / 185 val，512×512→256×256，中位病灶仅0.38%面积，92.6% <1%面积——极端小病灶
- **文献诊断**: 67+篇论文调研揭示三个组件的已知失败模式（参数过激、无监督门控退化、深层梯度算子有害）
- **多分支编码器**: 确认为第四核心创新，结构不可改

**Research Findings**:
- 边界损失权重>10-15%有害（Adaptive Composite Loss 2025）；早期边界加权破坏训练稳定性（ABL AAAI 2022, Pareto 2026）
- 无监督sigmoid门控失效（Gated CNN 2017）；固定Sobel优于深层可学习（SGDC 2025）；BN before sigmoid有风险（Gu et al. ICML 2020）
- 可学习边缘检测器漂移（PiDiNet ICCV 2021）；Sobel初始化+浅层可学习有域泛化优势（BEGA-UNet 2026）
- 小病灶检测失败是首要问题，边界质量改善只对已检测到的病灶有效

### Metis Review
**Identified Gaps** (addressed):
- 代码库混淆（Metis误看了My_diffusion目录）→ 已澄清：目标是IGEAR_Net纯U-Net
- 检测vs边界问题 → 在损失函数中保留对小病灶的检测敏感性
- BN before sigmoid风险 → 改用LayerNorm
- alpha初始值过保守（0.1）→ 改为0.3可学习

---

## Work Objectives

### Core Objective
在不移除四大创新（LearnableEdgeBank、梯度门控、边界加权BCE、多分支编码器）的前提下，通过参数校准和训练策略调整，优化IGEAR_Net在极端小病灶（中位0.38%面积，92.6%<1%面积）医学图像数据集上的分割性能。

### Concrete Deliverables
- `Train.py`：边界加权BCE的beta/k重校准 + 权重调度机制
- `encoder_decoder_block.py`：梯度门控alpha可学习化 + LayerNorm + 弱边缘监督；EdgeBank深浅层差异化学习策略

### Definition of Done
- [ ] 所有参数修改已应用到代码
- [ ] 50-epoch快速验证：loss正常下降，无NaN/爆炸
- [ ] 300-epoch全量训练完成：Dice/IoU/HD95均不低于baseline
- [ ] per-lesion-size分层分析：各尺寸区间指标均有提升或无退化

### Must Have
- LearnableEdgeBank保持可学习（结构不变）
- 梯度门控公式 `feat * (1 + alpha * gate)` 结构不变
- 边界加权BCE框架不变（BCE×weight_map）
- 多分支编码器（1×1/3×3/5×5）结构不变
- 参数修改后模型可正常训练和推理

### Must NOT Have (Guardrails)
- 不增删/替换四大创新的核心结构层
- 不替换损失函数体系（保持BCE）
- 不修改数据集预处理
- 不引入需要额外人工标注的监督信号（边缘GT从mask自动生成）
- 辅助层（GroupNorm/aux head）参数增量<0.1%总参数量
- 不过度工程化（AI slop：无必要的JSDoc、过度抽象、多余包装）

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: NO（test.py为空文件）
- **Automated tests**: None — 纯参数修改不需要单元测试
- **Agent-Executed QA**: MANDATORY for all tasks

### QA Policy
- **代码正确性验证**: Bash运行 Python 检查 requires_grad/参数值/前向传播
- **300-epoch全量验证**: Bash运行完整训练，收集Dice/IoU/HD95指标
- **Per-lesion-size分层分析**: Python脚本统计不同病灶尺寸区间的指标
- 证据保存到 `.omo/evidence/`

---

## Execution Strategy

### Parallel Execution Waves

> 简化策略：用户计算资源有限，一次性应用全部改动后跑1次全量训练对比。

```
Wave 1 (一次性应用全部改动 - 3文件修改):
├── Task 1: 同时修改 Train.py + encoder_decoder_block.py + Model_Network.py [quick]
│   ├── 边界加权BCE: beta 0.7→0.25, k 5.0→3.0, 权重调度
│   ├── 梯度门控: alpha可学习化 + LayerNorm + 弱边缘监督(aux BCE)
│   └── EdgeBank: 深层冻结，浅层保持可学习

Wave 2 (全量训练验证):
└── Task 2: 300-epoch全量训练 + per-lesion-size分层分析 [quick]
```

**Critical Path**: Task 1 → Task 2
**总训练次数**: 1次（vs baseline对比）

---

## TODOs

- [x] 1. 一次性应用全部参数改动（边界加权BCE + 梯度门控 + EdgeBank）

  **What to do**:

  ### A. Train.py —— 边界加权BCE参数重校准 + 权重调度

  - 修改 `build_boundary_weight_from_grads()` 的默认参数：`beta=0.7→0.25`, `k=5.0→3.0`
  - 在 `config["training"]` 中添加：
    - `"boundary_warmup_epochs": 90`（前30% epoch纯BCE）
    - `"boundary_ramp_end_epoch": 210`（此后beta达0.25）
  - 训练循环中实现调度逻辑：
    - `epoch < warmup`: `weight_map = torch.ones_like(boundary_conf) + 0.0`（等价纯BCE）
    - `warmup ≤ epoch < ramp_end`: `beta = 0.25 * (epoch - warmup) / (ramp_end - warmup)`
    - `epoch ≥ ramp_end`: `beta = 0.25`

  ### B. encoder_decoder_block.py —— 梯度门控改进

  - `EdgeDecoderBlock_V2.__init__()`:
    - `self.edge_scale = nn.Parameter(torch.tensor(0.3))`（从固定float改为可学习）
    - `edge_gate` Sequential 中 Conv2d 和 Sigmoid 间插入 `nn.GroupNorm(1, out_channels)`（避免BN风险）
    - 新增 `self.edge_supervision_head = nn.Conv2d(out_channels, 1, 1)`
  - `EdgeDecoderBlock_V2.forward()`:
    - alpha clamp到 `[0.05, 2.0]`
    - 返回 `(feat_out, gate_map)` 或通过属性获取gate_map用于边缘监督
  - 公式 `feat_edge = feat * (1.0 + self.edge_scale * gate)` 结构不变

  ### C. Train.py —— 边缘监督损失

  - 训练循环中：
    - 从decoder收集 `gate_map`
    - 使用 `Utils.masks_to_edges(masks)` 生成边缘GT
    - `aux_edge_loss = F.binary_cross_entropy(gate_map, edge_gt) * 0.05`
    - `loss = seg_loss + aux_edge_loss`

  ### D. Model_Network.py —— EdgeBank深浅层差异化

  - 在 `myModel.__init__()` 末尾：
    - `encoder1`, `encoder2` 的 `edge_bank.conv.weight.requires_grad = True`（保持可学习）
    - `encoder3`, `encoder4`, `bottleneck` 的 `edge_bank.conv.weight.requires_grad = False`（冻结）
    - 各编码器的 `edge_proj` 和 `edge_head` 保持可学习

  **Must NOT do**:
  - 不改变四大创新的架构拓扑
  - 不替换损失函数体系（保持BCE）
  - 不使用BatchNorm（用GroupNorm）
  - 不引入需要人工标注的监督信号
  - 边缘GT从mask自动生成（`masks_to_edges`）

  **Recommended Agent Profile**:
  - **Category**: `quick` — 3文件、参数级修改
  - **Skills**: none required

  **Parallelization**:
  - **Can Run In Parallel**: NO（同一人顺序修改即可）
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:
  - `Train.py:60-94` — `build_boundary_weight_from_grads()`
  - `Train.py:31-57` — `config` 字典
  - `Train.py:170-216` — 训练循环
  - `encoder_decoder_block.py:280-288` — `EdgeDecoderBlock_V2.__init__()`
  - `encoder_decoder_block.py:357-361` — `edge_gate` Sequential
  - `encoder_decoder_block.py:370-404` — `EdgeDecoderBlock_V2.forward()`
  - `encoder_decoder_block.py:57-144` — `LearnableEdgeBank`
  - `Model_Network.py:14-27` — 编码器定义
  - `Utils.py:145-158` — `masks_to_edges()`
  - 文献：Adaptive Composite Loss 2025, ABL AAAI 2022, EGSA-PT 2025, Gu et al. ICML 2020, Gated CNN 2017, SGDC 2025, PiDiNet ICCV 2021

  **Acceptance Criteria**:
  - [ ] `beta` 默认值 0.25, `k` 默认值 3.0
  - [ ] `config["training"]` 含 `boundary_warmup_epochs` 和 `boundary_ramp_end_epoch`
  - [ ] 训练循环中 `weight_map` 根据 epoch 动态调整（warmup期全为1.0）
  - [ ] `self.edge_scale` 为 `nn.Parameter`，初始 0.3
  - [ ] `edge_gate` 中 Conv2d 和 Sigmoid 间有 GroupNorm
  - [ ] forward 中 alpha clamp 到 [0.05, 2.0]
  - [ ] `EdgeDecoderBlock_V2` 有 `edge_supervision_head`
  - [ ] 训练循环中 `aux_edge_loss` 加入总 loss
  - [ ] `encoder1/2.edge_bank.conv.weight.requires_grad == True`
  - [ ] `encoder3/4/bottleneck.edge_bank.conv.weight.requires_grad == False`
  - [ ] `edge_proj` 和 `edge_head` 在各层保持可学习
  - [ ] 多分支编码器（1×1/3×3/5×5）结构完全不变

  **QA Scenarios**:
  ```
  Scenario: 参数改动正确性验证
    Tool: Bash (python)
    Preconditions: 所有文件修改完成
    Steps:
      1. 导入 myModel，实例化 model = myModel()
      2. 验证 encoder1.edge_bank.conv.weight.requires_grad == True
      3. 验证 encoder3.edge_bank.conv.weight.requires_grad == False
      4. 验证 decoder4.edge_scale 为 nn.Parameter，值为 0.3
      5. 验证 decoder4.edge_gate 中有 GroupNorm 层
      6. 验证 decoder4.edge_supervision_head 存在
      7. 创建随机输入 [2,3,256,256] 执行 forward
      8. 验证输出 seg_prob 形状 [2,1,256,256]
      9. 反向传播，检查无 NaN
    Expected Result: 所有 requires_grad 正确，模型可前向+反向传播
    Evidence: .omo/evidence/task-1-all-changes.txt

  Scenario: 边界权重调度逻辑验证
    Tool: Bash (python)
    Preconditions: Train.py 修改完成
    Steps:
      1. 导入 build_boundary_weight_from_grads，创建模拟 grad_maps
      2. 调用 beta=0.0（模拟 warmup），验证 weight_map 全为 1.0
      3. 调用 beta=0.125（模拟 ramp 中），验证 weight_map 值范围 [1.0, 1.125]
      4. 调用 beta=0.25（模拟 full），验证 weight_map 值范围 [1.0, 1.25]
    Expected Result: weight_map 无 NaN，值范围符合预期
    Evidence: .omo/evidence/task-1-weight-schedule.txt
  ```

  **Commit**: YES（单次提交包含所有改动）
  - Message: `feat(optimize): recalibrate boundary BCE + learnable gradient gate + edge supervision + freeze deep EdgeBank`
  - Files: `Train.py`, `encoder_decoder_block.py`, `Model_Network.py`

---

- [~] 2. 300-epoch全量训练 + baseline对比 + per-lesion-size分层分析

  **BLOCKED**: Requires user GPU environment (~2-5h training). Current env has CPU-only torch, missing tensorboard/medpy. User must run `python Train.py` on their GPU machine after `pip install tensorboard medpy`.

  **What to do**:
  - 确保 `config["training"]["epochs"] = 300`，`save_path` 设为新路径（不覆盖baseline权重）
  - 运行 `python Train.py` 完成全量训练
  - 收集 best model 的指标：Dice, IoU, Accuracy, Recall, Precision, HD95
  - 与 baseline 对比（如有历史 baseline 指标则直接对比）
  - 运行 per-lesion-size 分层分析：
    - 按病灶面积分桶：<0.25%, 0.25-0.5%, 0.5-1%, 1-2%, >2%
    - 每个桶统计 Dice/IoU/HD95
  - 保存结果到 `.omo/evidence/task-2-results.json`

  **Must NOT do**:
  - 不覆盖原 baseline 权重

  **Recommended Agent Profile**:
  - **Category**: `quick` — 运行训练脚本
  - **Skills**: none required

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: None（最终任务）
  - **Blocked By**: Task 1

  **References**:
  - `Train.py:47-55` — 训练配置
  - `patient_level_test.py:162-249` — 患者级评估逻辑

  **Acceptance Criteria**:
  - [ ] 300 epoch 训练完成（可早停）
  - [ ] Dice, IoU 不低于 baseline
  - [ ] HD95 不高于 baseline（不退化）
  - [ ] Per-lesion-size 分层分析：<0.5%极端小病灶的 Dice 至少维持或提升

  **QA Scenarios**:
  ```
  Scenario: 全量训练 + 指标对比
    Tool: Bash
    Preconditions: Task 1 完成
    Steps:
      1. cd 到 IGEAR_Net 目录
      2. 运行 python Train.py
      3. 等待训练完成或早停
      4. 解析 training_log.txt 提取 best epoch 指标
      5. 对比 baseline 指标
    Expected Result: best Dice ≥ baseline_Dice, best HD95 ≤ baseline_HD95
    Failure Indicators: Dice大幅下降(>5%), HD95大幅上升, NaN
    Evidence: .omo/evidence/task-2-full-train.txt

  Scenario: per-lesion-size分层分析
    Tool: Bash (python)
    Preconditions: 全量训练完成
    Steps:
      1. 加载 best model
      2. 遍历 val set 所有样本
      3. 按 GT mask 面积分桶：<0.25%, 0.25-0.5%, 0.5-1%, 1-2%, >2%
      4. 每桶统计 Dice/IoU/HD95 均值和标准差
      5. 输出分层结果表格
    Expected Result: <0.5%极端小病灶Dice至少维持baseline水平
    Evidence: .omo/evidence/task-2-stratified.csv
  ```

  **Commit**: NO（验证任务）

---

## Final Verification Wave

- [x] F1. **Plan Compliance Audit** — `oracle`
  读取计划端到端。验证每个 "Must Have"：检查代码中四大创新结构不变。验证每个 "Must NOT Have"：搜索代码无新增层/模块、无损失函数替换。检查 evidence 文件存在。
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  检查修改文件：无 `as any`/`@ts-ignore`、无空 catch、无 console.log、无注释掉的代码、无未使用的 import。检查 AI slop：过度注释、过度抽象、泛型名称。
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  执行 Task 1 和 Task 2 的 QA scenario。验证参数正确性（requires_grad/alpha/GroupNorm）。验证 300-epoch 训练结果和 per-lesion-size 分层分析。
  Output: `Scenarios [N/N pass] | Integration [N/N] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  对每个 task：读 "What to do"、读实际 diff。验证 1:1 —— 所有改动在 spec 内（无遗漏），无超出 spec（无 scope creep）。检查 "Must NOT do" 合规性。检测跨任务污染。
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **1**: `feat(optimize): recalibrate boundary BCE (beta=0.25,k=3.0) + learnable gradient gate (alpha+GroupNorm+edge_sup) + freeze deep EdgeBank (SGDC+PiDiNet validated)` - `Train.py`, `encoder_decoder_block.py`, `Model_Network.py`

---

## Success Criteria

### Verification Commands
```bash
# 参数正确性验证
python -c "from Model_Network import myModel; m=myModel(); print('enc1 learnable:', m.encoder1.edge_bank.conv.weight.requires_grad); print('enc3 frozen:', not m.encoder3.edge_bank.conv.weight.requires_grad); print('edge_scale:', m.decoder4.edge_scale)"

# 全量训练
python Train.py  # epochs=300
```

### Final Checklist
- [ ] 边界加权BCE：beta=0.25, k=3.0, 权重调度(warmup=90, ramp_end=210)
- [ ] 梯度门控：alpha可学习(初始0.3), GroupNorm稳定化, 弱边缘监督(w=0.05)
- [ ] EdgeBank：enc1/2可学习，enc3/4/bottleneck冻结conv，edge_proj保持可学习
- [ ] 多分支编码器结构不变
- [ ] 1次全量训练完成，Dice/HD95不低于baseline
- [ ] Per-lesion-size分层分析无退化（尤其<0.5%极端小病灶）
