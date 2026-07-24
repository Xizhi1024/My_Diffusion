# H4-v2：现有 155 位患者的内部探索性验证

本轮协议只使用现有 155 位患者、1191 张切片。原始患者角色为
99 位 mechanism_train、25 位 calibration 和 31 位 validation；不存在 test
set。31 位 validation 患者已经在 v1 中使用，因此本轮不会把它们描述为新的确认集。
H4-v1 的 `FAIL` 保持不变。

## 固定分析设计

H4-v2 使用 5 折 patient-grouped nested cross-validation：

- 每位患者只进入一个 outer fold，并且恰好得到一次 out-of-fold 结果；
- 每个 outer fold 内重新划分 mechanism_train 和 calibration；
- 对应 outer 患者不参与 population statistics、模型系数、置信度阈值、亚组阈值
  或非劣效 margin 的确定；
- outer 与 inner 分区只依据有效切片数、平均病灶面积、患者级 PET 对比度和原始患者
  划分标签进行平衡，不读取 recoverability 或 comparator error；
- 患者 044 和 153 强制保留在总体结果中，并单列透明报告。

固定比较为：

1. 无 evidence（`no_route`）；
2. H3 固定可恢复性日程；
3. 原始 H4 单切片 evidence；
4. uncertainty-aware evidence router，低置信度时回退到 H3 固定日程。

固定稳健性门包括低有效切片数、小病灶和低 PET 对比度患者。所有总体效应、改善患者
比例和亚组结果均按患者计算，并报告患者级 bootstrap 95% CI。

完整冻结配置位于
`configs/h4_v2_internal_exploratory_nested_cv_v1.json`。运行前先生成带自校验 hash 的
`frozen_plan.json`、partition fingerprint 和逐患者 inner/outer 角色表。

## 一键运行

查看命令但不运行：

```powershell
pixi run mechanism-v2-plan
```

冻结患者分区并执行一次 outer 评估：

```powershell
pixi run mechanism-v2-internal
```

结果写入：

```text
results/mechanism_validation_v2/
  02_h4_v2_internal_exploratory_nested_cv/
    00_frozen_plan/
    01_outer_evaluation/
    99_pipeline/
```

最终 `decision.json` 的结论只会使用下列两种措辞之一：

- `v2 获得现有数据集内部探索性支持`
- `v2 未获得现有数据集内部支持`

本轮结果只用于判断 v2 是否值得后续内部使用，不改变 H4-v1 的历史结论。
