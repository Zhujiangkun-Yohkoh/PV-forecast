# Site 17 Sanyo ALICD 最小真实可行性筛选

## 结论

**终止 ALICD。** 3/3 个固定 seed 的真实 GPU 训练均完成且数值稳定，但主要设计依据 Validation 不支持候选：高变化日间 RMSE 平均由 0.71538 增至 0.71608 kW（+0.10%），仅 seed 44 改善；H3 RMSE 明显恶化 2.40%；总变差比反而由 0.43719 降至 0.42543，离 1 更远。ALICD 没有修复过度平滑，不能用 Test 或单个 seed 挽救。因此不调整锚点、不搜索损失权重、不开发 v2/v3，回到 TRAJECTORY_ONLY，并重新评估 ASOC 投稿方法与期刊目标。

2022 Test 已在前序多轮中查看，本报告只把它作为探索性补充，不称为未使用的最终独立确认集。

## 协议与实现

- Site 17 Sanyo，2022；权威时间 UTC，转换 ACST = UTC+09:30；5 分钟网格。
- lookback=72，直接 H12；Train 2022-01-01–08-31，Validation 09-01–10-31，Test 11-01–12-31。
- Train-only 预处理；split 内连续窗口；Validation-only early stopping；Test 不进入训练函数或checkpoint选择。
- 输入与既有 MEAN_ONLY 完全相同；三个 seed 均为 42/43/44；Validation/Test origins、labels、timestamps 和 mask 与基线逐元素一致。
- `y0_scaled` 来自 forecast origin 最后一个真实功率，并由同一个 Train-only target scaler 转换；未使用 feature scaler 中的功率替代，也未使用未来第一个标签。
- ALICD 与基线共享完全相同的 ModernTCN backbone，只把 H12 线性头替换为 12维增量头、H3/H6/H12 三锚点头和固定最小范数投影。投影在target标准化空间完成。
- 损失固定为 `trajectory MSE + 0.2 × anchor MSE + 0.1 × projected-increment MSE`。Anchor loss 与轨迹锚点位置存在监督重叠；它只是提高关键水平权重，不是独立监督信息。

实际训练配置：AdamW，learning rate=0.001，weight decay=1e-5，batch=256，max epochs=25，patience=5，min_delta=1e-8，gradient clipping=1.0，mixed precision=false，num_workers=0，无scheduler。最佳checkpoint只按 Validation total loss。

## 执行与测试

| Seed | 完成 | Best epoch | 实际epoch | 停止原因 | 训练时间(s) | 单epoch(s) |
|---:|:---:|---:|---:|---|---:|---:|
| 42 | 是 | 9 | 14 | early stopping | 66.27 | 4.72 |
| 43 | 是 | 8 | 13 | early stopping | 61.26 | 4.70 |
| 44 | 是 | 8 | 13 | early stopping | 61.21 | 4.70 |

没有非有限loss或梯度。首次执行在训练结束后的本地checkpoint安全读取处暴露NumPy标量兼容问题；将checkpoint元数据改为普通Python `float` 后，以原配置重新完整执行。该修复未改变模型、数据、损失或选择规则。

普通协议测试 **25/25通过**，覆盖投影矩阵/秩/精确约束/可微性、双头梯度、两类batch有限loss、`y0_scaled`来源、真实增量构造、Train-only scaler、split边界、无未来输入、Validation-only checkpoint、artifact公平性、forward/backward及单batch过拟合。CSV由 `@oai/artifact-tool` 重新导入检查为 289行×43列（含表头），结构可解析。

## Validation：主要判断证据

### 整体提前步（mean±SD RMSE, kW）

| Horizon | TRAJECTORY_ONLY | ALICD | ALICD相对变化 |
|---|---:|---:|---:|
| H3 | 0.40364±0.00173 | 0.41332±0.00327 | +2.40% |
| H6 | 0.44803±0.00425 | 0.44870±0.00307 | +0.16% |
| H12 | 0.50632±0.00557 | 0.50439±0.00428 | -0.37% |

H12 的微小均值改善不稳定：seed 42/43 分别恶化 +0.43%/+0.68%，只有 seed 44 改善 -2.22%。H3 明显退化，故不是整体保持下的稳定收益。

### 关键场景（窗口轨迹RMSE, kW）

| 场景 | N | 基线 mean | ALICD mean | 相对变化 | seed方向 |
|---|---:|---:|---:|---:|---|
| Full timeline | 17,485 | 0.44564 | 0.44549 | -0.03% | 仅1/3改善 |
| Daylight | 9,032 | 0.61869 | 0.61857 | -0.02% | 仅1/3改善 |
| High-change daylight | 6,623 | 0.71538 | 0.71608 | +0.10% | 仅1/3改善 |
| H9–H12 | 17,485 | 0.49345 | 0.49258 | -0.17% | 仅1/3改善 |
| Midday | 3,660 | 0.77842 | 0.77524 | -0.41% | 3/3改善 |
| Sunrise | 2,917 | 0.40824 | 0.40803 | -0.05% | 1/3明确改善 |
| Sunset | 2,928 | 0.20890 | 0.21740 | +4.15% | 仅1/3改善 |

Stable/low-change 场景均值由 0.03944 降至 0.03695 kW，但方向高度不稳定（-6.61%、+21.12%、-28.73%），不能作为成功证据。高/低历史辐照波动场景完整数值保留在 `metrics_per_seed.csv`；它们未改变主要结论。

### 轨迹变化与虚假波动检查

| 指标（Full Validation） | 基线 | ALICD | 判断 |
|---|---:|---:|---|
| 一阶差分MAE (kW) | 0.11757 | 0.11749 | 几乎不变 |
| 预测/真实变化幅度比 | 0.43719 | 0.42543 | 离1更远 |
| 预测/真实总变差比 | 0.43719 | 0.42543 | 更平滑 |
| 方向一致率 | 0.36428 | 0.36426 | 无改善 |
| 真实变化但预测平坦比例 | 0.08557 | 0.08828 | 恶化 |
| 真实平稳但预测变化比例 | 0.09538 | 0.09750 | 恶化 |

ALICD没有靠虚假波动“美化”总变差；相反，它进一步压缩变化，同时轻微增加两类方向/平坦错误。因此核心机制没有解决局部轨迹形状问题。

## 投影与锚点诊断

- H3/H6/H12锚点 Validation RMSE分别为 0.41332/0.44870/0.50439 kW；锚点正是最终轨迹相应位置。
- 三个约束的最大绝对误差在保存精度下为0，固定投影确实实现了精确一致性。
- 投影前累计增量轨迹 RMSE=0.66075 kW，投影后=0.44549 kW，说明锚点成功限制了原始增量累计漂移。
- 平均投影修正范数=0.10781（标准化空间），修正量/原始增量范数=1.44767：投影修正大于原始增量尺度，表明两头预测存在较强冲突。
- 投影前总变差比=0.26470，投影后=0.42543；投影恢复了一部分变化，但仍低于基线0.43719，更远低于真实值1。

因此，“投影有效”只成立于代数一致性和抑制漂移，不成立于目标科学问题所需的误差/形状改善。锚点之间仍过度平滑。

## Test：已使用时间段的补充

| Horizon | TRAJECTORY_ONLY RMSE | ALICD RMSE | 相对变化 |
|---|---:|---:|---:|
| H3 | 0.44708±0.00172 | 0.45868±0.00312 | +2.60% |
| H6 | 0.49142±0.00479 | 0.49275±0.00229 | +0.28% |
| H12 | 0.54525±0.00736 | 0.54909±0.00206 | +0.72% |

Test中 full/daylight/high-change daylight/H9–H12 RMSE分别恶化约0.37%/0.39%/0.34%/0.53%，而总变差比由0.38020降至0.36741。Test方向与Validation“未修复过度平滑”一致，且整体精度更差，但它不承担独立确认作用。

## 参数与效率

- TRAJECTORY_ONLY：74,444参数。
- ALICD：88,271参数，其中backbone 19,136、增量头55,308、锚点头13,827；增加13,827（+18.57%）。这里原基线H12头参数被替换，不应把双头总量误称为纯新增双份容量。
- ALICD平均单epoch 4.71 s；三次总训练188.74 s。
- ALICD Test推理约0.0302 ms/样本；本次同进程基线约0.0207 ms/样本，约增加46%。该单机短任务计时仅作相对诊断。

额外参数与推理开销没有换来稳定的Validation主要场景收益。

## 有限新颖性边界核查

正式文献已经覆盖了“可微投影保证预测一致性”这一广义思想，不能把固定线性投影本身声明为创新：Rangapuram等的AISTATS 2023 temporal hierarchy工作把不同时间粒度预测通过可微 reconciliation 保持一致；ICML 2021工作已将闭式投影嵌入端到端相干概率预测；ICML 2024进一步学习最优斜投影。最接近的边界来源为：

- [Coherent Probabilistic Forecasting of Temporal Hierarchies, AISTATS 2023](https://proceedings.mlr.press/v206/rangapuram23a.html)
- [End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series, ICML 2021](https://proceedings.mlr.press/v139/rangapuram21a.html)
- [Learning Optimal Projection for Forecast Reconciliation of Hierarchical Time Series, ICML 2024](https://proceedings.mlr.press/v235/tsiourvas24b.html)

在本次有界检索中，没有找到同时明确采用“PV直接多步轨迹 + 逐步增量 + H3/H6/H12稀疏水平锚点 + 固定最小范数投影 + 锚点/累计增量精确一致”的正式论文。这只能支持“特定耦合尚未在检索范围内发现”的谨慎表述，不能证明全球首创。潜在新颖性若存在，只能来自该耦合针对PV轨迹过度平滑的任务化设计，而不是ModernTCN、多输出、差分损失、锚点或投影任一单项。由于实验证据失败，新颖性边界不构成继续开发的理由。

## 对研究问题的逐项回答

1. 未降低Validation high-change daylight RMSE；均值恶化且仅1/3 seed改善。
2. 改善方向不一致。
3. 总变差比从0.437降至0.425，背离目标1。
4. 变化幅度比同样降至0.425，未从既有约0.377问题向1稳定推进。
5. 一阶差分MAE几乎不变，方向指标略恶化。
6. H9–H12均值仅微改善，且只有1/3 seed改善。
7. H3明显恶化；H6基本持平；H12小幅且不稳定改善，整体不满足保持/改善。
8. 没有靠虚假波动提高总变差；实际更平滑，但两类误判略增。
9. Stable均值改善但跨seed极不稳定，其中seed43恶化21.1%。
10. 锚点确实限制增量漂移，投影前后RMSE大幅变化且约束精确满足。
11. 大修正量与无最终收益说明锚点和增量头冲突，机制的代数有效性没有转化为预测收益。
12. +18.57%参数及约46%推理计时增加与收益不相称。
13. Test补充结果同样不支持，并显示整体恶化。
14. 不值得进入完整baseline、消融或新时间段确认。
15. 候选的唯一可辩护边界是固定锚点—增量精确一致投影的特定耦合；本轮结果不足以把它发展为ASOC方法。

**下一步唯一建议：** 停止ALICD与继续模块设计，保留TRAJECTORY_ONLY作为可信工程基线，转而重新评估是否有足够独立科学问题与未使用确认数据支撑ASOC投稿；若没有，应调整期刊目标，而不是再造模型版本。

## 可复算材料

- 实现与训练入口：`run_alicd_screen.py`
- 固定配置：`config.json`
- 25项普通测试：`test_protocol.py`
- 逐seed、split、lead time、场景和投影诊断：`metrics_per_seed.csv`
- 本地未提交artifact：`results/ALICD/{42,43,44}/`

