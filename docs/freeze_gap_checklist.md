# 现状 → 冻结框架:最小改动清单

> 把 [current_model_status.md](current_model_status.md) 描述的现状,挪到冻结框架
> (`memory/mechanism-freeze-and-positioning.md`)所需的最小改动。每条标注**依赖哪个
> 未跑假设/阶段**,以及"现在能否做"。
>
> 治理原则(冻结):**falsify before implementation**——任何未通过门的假设,其依赖模块
> 不进入生产。因此多数改动在被依赖的门通过前**不应实现**,本清单只标出最小触点。

## 改动表

| # | 改动 | 代码触点 | 依赖(未跑除非注明) | 现在能否做 |
|---|---|---|---|---|
| 1 | 生产 mean 换成 **pathology-excluded**(在 `main_data` 全 split 上重训,非复用 H2 的 mechanism-partition checkpoint) | [mean_pretraining.py](../src/model/mean_pretraining.py) 增 mask-exclusion 路径;[v5 config:77](../configs/experiments/slmf_png_spectral_router_v5.yaml#L77) checkpoint 路径 | **H2(PASS)** | ✅ **能(代码已就绪,待云端跑)**。`MeanPretrainer` 已加 `pathology_exclusion` 开关(默认关,复刻 H2 validated loss;`tests/test_mean_pretraining.py` 8 测试通过);云端脚本 [scripts/retrain_excluded_mean.ps1](../scripts/retrain_excluded_mean.ps1) 待跑。跑完后:验 `mean_config.pathology_exclusion.enabled=true`、记 SHA256、把 [v5:77](../configs/experiments/slmf_png_spectral_router_v5.yaml#L77) 指向新 checkpoint。复用 H2 checkpoint 不可行(患者集不同)。建议追加:在新 split 上重算一次 H2 富集对比,确认 excluded 仍在生产 split 富集 residual |
| 2 | 课程开放阈值**绑 H3 crossing**,按 `intensity→coarse→shape→frequency` 排序 | loss curriculum 调度;**新增 frequency-band 监督阶段**(现 `residual_wavelet`/`boundary_frequency` 关闭) | **H3(PASS)** 提供阈值;课程**收益**需 **stage 05_curriculum(未跑)** | ⚠️ 阈值能设且应按 H3 排序纠正现状;但"课程是否有效"=stage 05,未验证前不能宣称 continuation 收益 |
| 3 | **接入 UncertaintyAwareRouteSelector**(低置信度回退 H3 固定日程) | [spectral_router forward](../src/model/frequency/spectral_router.py#L909) 调用 selector;提供 confidence + H3 固定 route 作为 fallback | **H4-v2 外部确认(未跑)**;selector docstring 要求 formal zero-overlap H4-v2 PASS | ❌ **不能**。卡在 H4-v2 外部确认。内部探索性 PASS 含原 31 validation 患者,不构成独立确认 |
| 4 | **加 artifact 安全因子 e^safe**(方向集中度 × 跨尺度不一致 × 接近噪声底) | router 新增乘性 gate;合成条纹/假热点/孤立热点校准 | **stage 06_artifact_safety(未跑)**;预注册证伪行"安全项有效" | ❌ **不能**。卡在 stage 06:artifact score 必须先被证明能识别条纹且不掉病灶 recall |
| 5 | router **重构成** `a=π·eCT·eRes·eSafe` × `p=softmax(native/shallow/null)` × `null=1−a` 的分解 | [spectral_router.py](../src/model/frequency/spectral_router.py) 结构重构 | #3 + #4(依赖 e^res 稳定 + e^safe 存在) | ❌ **不能**。依赖 #3#4。注意:把 H4-v1 FAIL 的 raw e^res 直接当主驱动,等于复活一个已 FAIL 的门,必须走 #3 的 uncertainty-aware 路径 |
| 6 | (验证)**H5** 路由消融:fixed / shuffle / wrong-band / timestep-permutation,证明收益来自时间—层级而非参数量 | [validate_h5_router_role_separation.py](../scripts/validate_h5_router_role_separation.py)(已存在,dry-run 计划 epochs=50 mc=20) | router 定形(依赖 #3#4 或至少冻结一种稳定 router 形态) | ⏳ 脚本就绪;需 router 定形后正式跑 |
| 7 | (验证)**H6** 最终小病灶增益 + 假热点/条纹/全图误差非劣,对比 plain residual diffusion 与固定频率注入 | [validate_h6_final_integration.py](../scripts/validate_h6_final_integration.py)(已存在,dry-run) | **H5 PASS** + #1–#5 | ⏳ 末步,全部上游就绪后 |

## 排序与门控

```
现在可做:  #1(excluded mean 重训,H2 已 PASS)
           #2 的阈值设置部分(按 H3 纠正排序;收益待 stage 05)
卡门:      #3 → H4-v2 外部确认
           #4 → stage 06 artifact_safety
           #5 → #3 + #4
           #2 的收益主张 → stage 05
验证链:    router 定形 → #6(H5) → #7(H6)
```

## 与"停止加模块"原则的一致性

本清单**不是**新增功能 backlog。#1 是修正现状(included→excluded,对齐 H2);#2 是把课程
对齐数据;#3–#5 是把框架定义但**尚未实现**的部件按门控补齐;#6#7 是验证。除 #1 外,所有
改动都在对应门通过后才动手——门未过则不实现,符合"falsify before implementation"。

## 数据层前置限制(影响 H6 强度,与改动无关)

- 全病灶层、无阴性切片 → H6 无法充分证明"不凭空生成病灶"。
- 无独立 test → H6 需固定 untouched test 或外部数据才有正式论文强度。
- 因此即便 #1–#7 全部完成,H6 的主张范围仍受这两条限制。
