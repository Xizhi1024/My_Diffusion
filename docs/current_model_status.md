# 当前模型现状(Current Model Status)

> 本文件描述**当前代码与配置实际实现并启用的模型**,不是冻结目标。目标定位见
> `memory/mechanism-freeze-and-positioning.md`(及待写的 `docs/mechanism_freeze.md`)。
> 本文件随实现变更更新;每项区分三层:**已实现 / 已启用 / 已验证**。
>
> 基准日期:2026-07-24。生产配置:`configs/experiments/slmf_png_spectral_router_v5.yaml`。

## 0. 最准确的模型名(现状)

> 基于条件均值 residual Brownian bridge 的**时间步条件化层级频带 spectral-evidence
> router**(含 CT 支持底板),配合 **τ 门控的病灶保留 + 归一化摄取代理(uptake proxy)** 监督。

**不是**冻结框架的强表述——"可恢复性驱动的时间步—层级—频带可靠性路由 + 病灶保留、代谢感知的
渐进 residual 学习"。五点差距见第 3 节。

### 0.1 机制声明必须分层

| 机制组件 | 当前证据/实现状态 | 当前允许的表述 |
|---|---|---|
| Availability | 原始 H3 时序 PASS；H3-v2 Stage A1 尚未实跑 | 时间—频带 native/null availability 候选 |
| Destination | native/shallow 独立干预未运行 | 不得声称层级路由已验证 |
| Safety | 只有输出安全指标门；`e^safe` 机制未验证 | 不得声称安全门已阻止伪影 |
| Curriculum | 当前仍为手设 τ 门 | 不得声称 H3-driven 渐进课程 |

H3-v2 的 `[native, shallow, null]=[a,0,1-a]` 只测试 Availability。它不恢复
ternary destination，也不替代后续 Destination、Safety、Curriculum 实验。

## 1. 半边 A:路由层

| 冻结框架要求 | 代码现状 | 已启用 | 已验证 |
|---|---|---|---|
| 时间步 + log-SNR 条件 | [_time_features](../src/model/frequency/spectral_router.py#L536) `use_noise_release=true` | ✅ | — |
| 层级(L2/L1)× 频带(LH/HL/HH)→ native/shallow/null | [_acquire_routes](../src/model/frequency/spectral_router.py#L453) `policy=learned` | ✅ | **H5 未跑** |
| CT 结构支持 + 非零 floor | [_ct_support_field](../src/model/frequency/spectral_router.py#L772) floor 0.25/0.50 | ✅ | H1 PASS(定位) |
| residual 可恢复性证据 e^res **作主驱动** | [noise_calibrated_band_evidence](../src/model/frequency/spectral_router.py#L563) 仅 48 维特征之一 | 部分 | **H4-v1 FAIL**(就是此信号) |
| availability × destination 分解 `a=π·eCT·eRes·eSafe`,`null=1−a` | amplitude_head + route_head 分开,但非框架乘法分解 | 部分 | — |
| **artifact 安全因子 e^safe** | **不存在**;gabor 各向异性只是输入特征 | ❌ | — |
| uncertainty-aware 低置信度回退(H4-v2 PASS 机制) | [UncertaintyAwareRouteSelector](../src/model/frequency/spectral_router.py#L127) 存在 | **❌ 未接入**(仅被 test 调用) | v2 探索性 PASS 的是它 |
| π^open 绑定 H3 crossing 曲线 | 否;开放由学习头隐式驱动 | ❌ | — |

**半边 A 小结**:只成立"时间步条件化的层级频带路由"这一层。可恢复性驱动、可靠性因子
(e^res/e^safe)、H3 匹配的开放调度,均未在生产 router 实现/验证。H4-v2 唯一过门的机制
(uncertainty-aware 回退)被代码注释明确挡在生产 router 之外
([spectral_router.py:130-132](../src/model/frequency/spectral_router.py#L130))。

## 2. 半边 B:residual 构造 + 渐进课程

| 冻结框架要求 | 代码现状 | 已启用 | 已验证 |
|---|---|---|---|
| `r₀ = y − μ_bg(c)` pathology-**excluded** mean | 云端已生成并冻结 excluded candidate；生产 v5 仍指向 included，H3-v2 Stage A1 显式使用 excluded | ✅候选/❌生产切换 | H2 PASS；V2-03 candidate PASS |
| `R_enrich>1` + 边界不恶化 | H2 在 excluded 上测,contrast +0.00547 CI[0.00506,0.00589],边界非劣 | — | **H2 PASS(excluded checkpoint)** |
| continuation/homotopy 课程:逐统计量阈值(H3 驱动)+ 模糊目标递减 σ + 期望归一化 + 截断 EMA | 只有 [active_tau_max](../configs/experiments/slmf_png_spectral_router_v5.yaml#L157) 粗两段门:ROI τ≤0.7、TopK/peak τ≤0.4,手设 | 部分(简化) | stage 05_curriculum **未跑** |
| 解剖/代谢目标分离 `T_anat` / `T_metab` | 无两 head;有 ROI L1(空间)+ normalized_peak(强度) | 部分 | — |
| 代谢感知(uptake proxy,非 SUV) | [normalized_lesion_peak + cold_weight](../configs/experiments/slmf_png_spectral_router_v5.yaml#L163);`physical_suv_available:false` | ✅ proxy | 最终病灶收益 **H6 未跑** |

**半边 B 小结**:病灶保留目前靠 ROI/TopK/peak **损失**兜底;H2 对齐的 excluded mean
已在云端重训并冻结，但**尚未切换生产 v5**；H3-v2 Stage A1 会在隔离实验中显式使用它。
课程仍是手设两段 τ 门,不是 H3 驱动的 continuation/homotopy。

## 3. 现状与冻结框架的五点差距

1. **可恢复性证据没驱动 router**:H4-v1 FAIL 的 noise-band evidence 只是 48 维特征之一;
   v2 PASS 的 uncertainty-aware 回退机制没接入。
2. **没有 artifact 安全因子 e^safe**:gabor 各向异性只是特征,不是乘性安全门。
3. **课程是手设两段 τ 门,不是 H3 驱动的 continuation**:且按 H3 实测顺序应是
   intensity→coarse→shape→frequency(crossing log-SNR 分别 −1.32 / 2.90 / 7.67 / 17.87),
   不是手设的 ROI→peak;frequency-band 监督阶段当前全部关闭。
4. **生产 mean 仍是 pathology-included**:V2-03 已生成带 lineage 和 exclusion policy 的
   excluded candidate，但生产配置路径尚未 cut over；两者必须继续区分。
5. **H5/H6 从未运行**:"路由收益来自机制而非参数量""最终小病灶增益"无证据。

### 3.4 hash 证据:生产 mean ≠ H2 验证对象

| checkpoint | SHA256 | 训练数据 | mask |
|---|---|---|---|
| 生产 v5 `freq_mean_pretrain_v2/mean_best.pt` | `a2035873dbe466dedbf4f88d41b08d5442d8414bf8c2eac4cba0339a9919fcca` | `main_data/split_manifest.csv`(全 train split) | 无(included) |
| V2-03 `freq_mean_excluded_v1/mean_best.pt` | `f04c1d79214247c11cd1cff296fffc3020a835af0785aa5d084cae601768c5e8` | mechanism-train，云端重训 | enabled，guard radius 8 |
| H2 `mean_excluded_fixed.pt` | `26b13361d0fb28f7f95bf175f17fbce8da7fdead7a2b99028d3e4244f899b3e6` | mechanism partition(99 患者) | excluded |
| H2 `mean_included_fixed.pt` | `c5b53eac41a3c5563704b33fc9c91473a7d4eae263614be5b1a716f1e8d8e460` | mechanism partition(99 患者) | included |

依据:[mean_pretraining.py:82-90](../src/model/mean_pretraining.py#L82)(`_batch_loss` 只用
ct/pet)、[resolved_config.yaml](../checkpoints/freq_mean_pretrain_v2/resolved_config.yaml)
(无 exclusion 字段、`split_manifest: ../main_data/split_manifest.csv`)、
[history.json](../checkpoints/freq_mean_pretrain_v2/history.json)(实跑 30 epoch)。
H2 决策:`results/mechanism_validation/01_h2_residual_enrichment/decision.json`。
V2-03 决策:`results/mechanism_validation_v2/03_excluded_mean_production/decision.json`；
该 PASS 只确认候选 checkpoint 已生成并验证，不代表生产 cutover。

## 4. 机制层验证状态(不含最终性能)

| 假设/阶段 | 状态 | 关键数字 | 文件 |
|---|---|---|---|
| H1 CT 频谱不对称 | **PASS** | CT 定位 AUC 0.826;PET band 增量 skill −0.12(不决定幅值) | [00B report](../results/mechanism_validation/00B_h1_spectral_asymmetry/report.md) |
| H2 excluded residual 富集 | **PASS**(excluded) | contrast +0.00547;边界非劣 | [01_h2 decision](../results/mechanism_validation/01_h2_residual_enrichment/decision.json) |
| H3 可恢复时序 | **PASS** | crossing:intensity −1.32 < coarse 2.90 < shape 7.67 < freq 17.87 | [02_h3 decision](../results/mechanism_validation/02_h3_recoverability_curves/decision.json) |
| 05B H3 fixed-schedule inference admissibility | **FAIL** | 19/20 推理 timestep off-grid；scalar→ternary mapping 未定义 | `05B_h3_fixed_schedule_inference/decision.json` |
| H3-v2 Stage A1 Availability | **未运行** | 全 0..999 timestep；`[a,0,1-a]`；只比较 no-route 候选筛选 | `configs/h3_v2_full_timestep_native_null_v1.json` |
| H4-v1 noise-band evidence | **FAIL** | incremental skill 0.155 vs null q95 0.143;`next_stage_allowed:false` | [03_h4 decision](../results/mechanism_validation/03_h4_noise_calibration/decision.json) |
| H4-v2 uncertainty-aware(内部探索性) | **PASS(探索性)** | 改善 153/155;原 31 validation 患者已纳入 | [v2 outer eval](../results/mechanism_validation_v2/02_h4_v2_internal_exploratory_nested_cv/01_outer_evaluation/decision.json) |
| H4-v2 外部确认 | **未运行** | — | `scripts/validate_h4_v2_external_confirmation.py` |
| 04 CT support / 05 curriculum / 06 artifact | **未运行** | — | `99_full_pipeline`(全 DRY_RUN) |
| H5 路由角色分离 | **未运行** | — | `scripts/validate_h5_router_role_separation.py` |
| H6 最终增益 | **未运行** | — | `scripts/validate_h6_final_integration.py` |

## 5. 数据层限制(影响可主张范围,与模块无关)

- 当前数据**全部为病灶层**,缺真正阴性切片 → 不能充分证明"不会凭空生成病灶"。
- 自定义划分**无独立 test 集**;正式论文需固定 untouched test 或外部数据。
- `physical_suv_available:false` → "代谢感知"只能表述为 normalized uptake proxy。

## 6. 证据指针

- 生产配置:[slmf_png_spectral_router_v5.yaml](../configs/experiments/slmf_png_spectral_router_v5.yaml)
- Router 实现:[spectral_router.py](../src/model/frequency/spectral_router.py)
- Mean 训练:[mean_pretraining.py](../src/model/mean_pretraining.py)
- 现状→目标的最小改动清单:见 `docs/freeze_gap_checklist.md`
