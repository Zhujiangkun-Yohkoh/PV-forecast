# Scheme A Multisite M2：冻结训练与独立证据复算

**SCHEME_A_MULTISITE_RESULTS_READY_FOR_MANUSCRIPT_REVIEW**

## 执行完整性

24/24 runs完成，24个Validation最优checkpoint均可加载并无梯度复现保存预测。无数值发散，无额外run、seed或超参数搜索。神经模型是否胜过基线不作为软件门禁。训练与Test预测均按本轮用户授权执行。

M1-R前置测试43/43，M2普通测试17/17，artifact测试10/10；failed/errors/skipped均为0。独立数值比较10432/10432通过，最大绝对差4.55e-13。独立验证器不导入训练主脚本的指标函数，重新从原始功率构造标签与timestamp−24h Daily，再复算所有逐seed指标、两类skill、均值和sample SD、排名与胜负数。

GPU：NVIDIA GeForce RTX 3060 Laptop GPU；PyTorch 2.7.1+cu118，CUDA 11.8。累计训练时间3.607小时；训练脚本端到端（含预处理与首次预测）3.680小时。Float32、无AMP，batch=256，4个CPU线程，num_workers=0。逐run训练时间、参数量、best epoch及Validation global MSE见下表。

原始366个文件size/mtime_ns与M1-R记录及本轮读取前后完全一致。原Scheme A结果是否改变：否。原checkpoint是否修改：否，M2未读取或写入原17通道checkpoint，所有新产物限制在独立worktree的忽略目录。未访问master主工作树，未改C1/NWP、论文正文、补充材料、PDF或图表。原12/9/2/1/0、Daily 22/24、Hanwha H12两个例外与Qcells计数保持冻结。

首次artifact检查为9 passed、0 failed、1 error、0 skipped：数值比较完成后，审计JSON写出遇到NumPy int64序列化错误。修正为将NumPy标量转换成原生Python标量后，完整重跑10项检查全部通过（138.110秒）。此修正不改变训练、checkpoint、预测、指标计算或冻结协议，也没有增加训练run。

## 预注册主模型结果

主模型始终为INVERTED_VARIATE_TRAJECTORY，未根据外部成绩重新选择。下表为三个seed的指标均值±sample SD（ddof=1），不是预测平均后的ensemble。RMSE/MAE单位kW；nRMSE与skill表中以百分比显示，CSV以比例保存。Daily RMSE与Daily skill使用supplementary_daily_matched目标交集，和primary结果分开。

| 场址 | H | Scope | RMSE | MAE | nRMSE % | Last skill % | Daily-matched RMSE | Daily skill % |
|---|---:|---|---|---|---|---|---|---|
| Yulara | 12 | full | 10.2894 ± 0.0320 | 5.8883 ± 0.2815 | 9.3580 ± 0.0291 | 17.3445 ± 0.2573 | 10.2894 ± 0.0320 | 36.2783 ± 0.1984 |
| Yulara | 12 | daylight | 13.9882 ± 0.0627 | 9.4546 ± 0.3257 | 12.7220 ± 0.0570 | 18.7298 ± 0.3641 | 13.9882 ± 0.0627 | 37.2786 ± 0.2810 |
| Yulara | 48 | full | 13.3676 ± 0.1396 | 8.3598 ± 0.1466 | 12.1576 ± 0.1270 | 49.2413 ± 0.5302 | 13.3676 ± 0.1396 | 17.2469 ± 0.8644 |
| Yulara | 48 | daylight | 17.6587 ± 0.2419 | 12.8699 ± 0.0307 | 16.0603 ± 0.2200 | 48.5792 ± 0.7044 | 17.6587 ± 0.2419 | 20.8304 ± 1.0845 |
| Yulara | 96 | full | 15.6990 ± 0.0556 | 10.0281 ± 0.0671 | 14.2779 ± 0.0506 | 61.3217 ± 0.1371 | 15.6990 ± 0.0556 | 2.7132 ± 0.3448 |
| Yulara | 96 | daylight | 20.9265 ± 0.1501 | 16.0784 ± 0.0327 | 19.0322 ± 0.1365 | 56.0975 ± 0.3148 | 20.9265 ± 0.1501 | 6.1000 ± 0.6734 |
| Yulara | 144 | full | 15.8051 ± 0.1553 | 10.2257 ± 0.0838 | 14.3744 ± 0.1412 | 67.1005 ± 0.3233 | 15.8051 ± 0.1553 | 1.6245 ± 0.9666 |
| Yulara | 144 | daylight | 21.2026 ± 0.2208 | 16.5189 ± 0.3122 | 19.2833 ± 0.2008 | 59.4622 ± 0.4222 | 21.2026 ± 0.2208 | 4.4970 ± 0.9946 |
| NIST Ground | 12 | full | 17.8224 ± 0.1131 | 10.2601 ± 0.3503 | 6.9083 ± 0.0439 | 17.9290 ± 0.5210 | 17.7851 ± 0.1158 | 57.7292 ± 0.2753 |
| NIST Ground | 12 | daylight | 28.4044 ± 0.0934 | 20.1259 ± 0.2955 | 11.0100 ± 0.0362 | 20.2400 ± 0.2624 | 28.3641 ± 0.0903 | 59.0101 ± 0.1305 |
| NIST Ground | 48 | full | 28.5043 ± 0.2474 | 17.0150 ± 0.6350 | 11.0488 ± 0.0959 | 41.1576 ± 0.5106 | 28.4930 ± 0.2419 | 32.0866 ± 0.5767 |
| NIST Ground | 48 | daylight | 44.4344 ± 0.9937 | 32.9668 ± 0.7233 | 17.2235 ± 0.3852 | 37.9786 ± 1.3870 | 44.4196 ± 0.9822 | 36.0656 ± 1.4137 |
| NIST Ground | 96 | full | 40.2155 ± 0.7406 | 25.0934 ± 0.7715 | 15.5882 ± 0.2871 | 42.1821 ± 1.0647 | 40.2074 ± 0.7514 | 3.8822 ± 1.7963 |
| NIST Ground | 96 | daylight | 55.6752 ± 0.7810 | 43.2489 ± 0.4544 | 21.5807 ± 0.3027 | 39.2546 ± 0.8521 | 55.6119 ± 0.7839 | 20.2194 ± 1.1245 |
| NIST Ground | 144 | full | 44.6173 ± 0.6843 | 28.5578 ± 0.4564 | 17.2944 ± 0.2652 | 43.2438 ± 0.8705 | 44.6093 ± 0.6903 | -6.4675 ± 1.6476 |
| NIST Ground | 144 | daylight | 60.6101 ± 1.0335 | 48.4096 ± 0.6992 | 23.4935 ± 0.4006 | 37.7957 ± 1.0606 | 60.5445 ± 1.0406 | 13.1951 ± 1.4919 |

## 科学结论与原结果的关系

- NIST Ground：主模型在8/8个horizon×scope组合中优于Last-value；在Daily共同目标交集上赢7/8，Daily赢1/8，平局0。这些是相关的描述性比较，不是8个独立统计试验。
- Yulara：主模型在8/8个horizon×scope组合中优于Last-value；在Daily共同目标交集上赢8/8，Daily赢0/8，平局0。这些是相关的描述性比较，不是8个独立统计试验。
- Yulara四神经模型场址内平均排名：Inverted-variate 1.000；Joint-patch 2.500；Depthwise TCN 3.000；Discrete recurrent 3.500。平均排名仅汇总该场址八个描述性范围，不汇总跨场址kW误差。
  各范围首位：H12/daylight: Inverted-variate；H12/full: Inverted-variate；H48/daylight: Inverted-variate；H48/full: Inverted-variate；H96/daylight: Inverted-variate；H96/full: Inverted-variate；H144/daylight: Inverted-variate；H144/full: Inverted-variate。
- NIST Ground四神经模型场址内平均排名：Discrete recurrent 1.750；Depthwise TCN 2.000；Inverted-variate 2.250；Joint-patch 4.000。平均排名仅汇总该场址八个描述性范围，不汇总跨场址kW误差。
  各范围首位：H12/daylight: Inverted-variate；H12/full: Discrete recurrent；H48/daylight: Depthwise TCN；H48/full: Discrete recurrent；H96/daylight: Depthwise TCN；H96/full: Discrete recurrent；H144/daylight: Depthwise TCN；H144/full: Discrete recurrent。

主模型相对Last-value的优势在两个外部场址均重现。Yulara的RMSE skill为17.34%–67.10%，NIST为17.93%–43.24%。采用Daily共同目标交集后，Yulara八个范围仍全部为正skill，NIST七个范围为正；NIST H144/full的skill为−6.47% ± 1.65%，即主模型在该交集上的RMSE比Daily高6.47%。同一场址H144/daylight的skill为+13.20% ± 1.49%。这说明长时域结论取决于评价scope，不能只用full或daylight之一概括全部结果。

模型排序也依赖场址和scope。Yulara中Inverted-variate在八个范围均为四神经模型首位；NIST中Discrete recurrent在四个full范围居首，Inverted-variate在H12/daylight居首，Depthwise TCN在H48/H96/H144的daylight居首。因此，Alice Springs中Inverted-variate取得12/24 primary wins和最佳平均排名的架构排序在Yulara重现，在NIST未重现；预注册主模型仍保持不变。

Alice Springs原Daily在22/24共同目标比较中优于best-of-four神经结果，该广泛优势未在这两个外部场址重现：预指定主模型已分别在8/8和7/8个范围优于Daily。外部实验支持“基线信息策略和评价范围会改变结论”的判断，不支持“Daily在不同场址普遍占优”或“同一神经架构处处最佳”的泛化。没有据此调整任何模型或数据规则。外部结果单独报告，不把原24组合与新16组合相加，也不通过胜场数构造p值。

七通道外部实验与原17通道实验具有不同可用输入、缺失结构、年份、气候及功率规模；它们是按各自Train拟合的场址内复现实验，不是原checkpoint迁移或纯架构效应的统一排行榜。相同point mask保证相同评价目标，不表示相同历史信息：Daily使用准确24小时历史，神经模型使用六小时输入。

## 四模型次要结果

完整MAE、bias、R²、Train-range nRMSE、两类skill和每seed结果见metrics_per_seed.csv及metrics_summary_mean_sd.csv。负R²保留；基线RMSE为0时skill缺省，不用epsilon。表中rank仅比较四神经模型；同值使用平均秩。

| 场址 | H | Scope | 模型 | Primary RMSE±SD | Rank | Daily-matched RMSE±SD | Daily skill %±SD |
|---|---:|---|---|---|---:|---|---|
| Yulara | 12 | full | Discrete recurrent | 12.3203 ± 0.3691 | 3 | 12.3203 ± 0.3691 | 23.7012 ± 2.2861 |
| Yulara | 12 | full | Inverted-variate | 10.2894 ± 0.0320 | 1 | 10.2894 ± 0.0320 | 36.2783 ± 0.1984 |
| Yulara | 12 | full | Joint-patch | 12.5982 ± 0.2587 | 4 | 12.5982 ± 0.2587 | 21.9803 ± 1.6022 |
| Yulara | 12 | full | Depthwise TCN | 10.5616 ± 0.9289 | 2 | 10.5616 ± 0.9289 | 34.5927 ± 5.7527 |
| Yulara | 12 | daylight | Discrete recurrent | 16.6717 ± 1.0603 | 3 | 16.6717 ± 1.0603 | 25.2461 ± 4.7542 |
| Yulara | 12 | daylight | Inverted-variate | 13.9882 ± 0.0627 | 1 | 13.9882 ± 0.0627 | 37.2786 ± 0.2810 |
| Yulara | 12 | daylight | Joint-patch | 16.7598 ± 0.6670 | 4 | 16.7598 ± 0.6670 | 24.8510 ± 2.9909 |
| Yulara | 12 | daylight | Depthwise TCN | 14.0346 ± 0.9125 | 2 | 14.0346 ± 0.9125 | 37.0705 ± 4.0915 |
| Yulara | 48 | full | Discrete recurrent | 15.8157 ± 0.2012 | 4 | 15.8157 ± 0.2012 | 2.0921 ± 1.2455 |
| Yulara | 48 | full | Inverted-variate | 13.3676 ± 0.1396 | 1 | 13.3676 ± 0.1396 | 17.2469 ± 0.8644 |
| Yulara | 48 | full | Joint-patch | 14.6332 ± 0.0848 | 2 | 14.6332 ± 0.0848 | 9.4122 ± 0.5253 |
| Yulara | 48 | full | Depthwise TCN | 15.3827 ± 1.2281 | 3 | 15.3827 ± 1.2281 | 4.7728 ± 7.6023 |
| Yulara | 48 | daylight | Discrete recurrent | 21.2708 ± 0.7595 | 4 | 21.2708 ± 0.7595 | 4.6361 ± 3.4049 |
| Yulara | 48 | daylight | Inverted-variate | 17.6587 ± 0.2419 | 1 | 17.6587 ± 0.2419 | 20.8304 ± 1.0845 |
| Yulara | 48 | daylight | Joint-patch | 18.7326 ± 0.3381 | 2 | 18.7326 ± 0.3381 | 16.0159 ± 1.5158 |
| Yulara | 48 | daylight | Depthwise TCN | 19.0960 ± 1.5641 | 3 | 19.0960 ± 1.5641 | 14.3868 ± 7.0125 |
| Yulara | 96 | full | Discrete recurrent | 17.8657 ± 0.4556 | 3 | 17.8657 ± 0.4556 | -10.7140 ± 2.8235 |
| Yulara | 96 | full | Inverted-variate | 15.6990 ± 0.0556 | 1 | 15.6990 ± 0.0556 | 2.7132 ± 0.3448 |
| Yulara | 96 | full | Joint-patch | 16.7954 ± 0.3375 | 2 | 16.7954 ± 0.3375 | -4.0813 ± 2.0918 |
| Yulara | 96 | full | Depthwise TCN | 18.4349 ± 1.4216 | 4 | 18.4349 ± 1.4216 | -14.2414 ± 8.8094 |
| Yulara | 96 | daylight | Discrete recurrent | 24.2892 ± 0.8586 | 4 | 24.2892 ± 0.8586 | -8.9886 ± 3.8528 |
| Yulara | 96 | daylight | Inverted-variate | 20.9265 ± 0.1501 | 1 | 20.9265 ± 0.1501 | 6.1000 ± 0.6734 |
| Yulara | 96 | daylight | Joint-patch | 21.4260 ± 0.2821 | 2 | 21.4260 ± 0.2821 | 3.8589 ± 1.2657 |
| Yulara | 96 | daylight | Depthwise TCN | 23.4934 ± 1.9473 | 3 | 23.4934 ± 1.9473 | -5.4179 ± 8.7378 |
| Yulara | 144 | full | Discrete recurrent | 17.8827 ± 0.7591 | 3 | 17.8827 ± 0.7591 | -11.3068 ± 4.7249 |
| Yulara | 144 | full | Inverted-variate | 15.8051 ± 0.1553 | 1 | 15.8051 ± 0.1553 | 1.6245 ± 0.9666 |
| Yulara | 144 | full | Joint-patch | 17.0117 ± 0.3885 | 2 | 17.0117 ± 0.3885 | -5.8856 ± 2.4182 |
| Yulara | 144 | full | Depthwise TCN | 18.8751 ± 1.7983 | 4 | 18.8751 ± 1.7983 | -17.4840 ± 11.1934 |
| Yulara | 144 | daylight | Discrete recurrent | 24.4604 ± 1.2063 | 4 | 24.4604 ± 1.2063 | -10.1770 ± 5.4338 |
| Yulara | 144 | daylight | Inverted-variate | 21.2026 ± 0.2208 | 1 | 21.2026 ± 0.2208 | 4.4970 ± 0.9946 |
| Yulara | 144 | daylight | Joint-patch | 22.1512 ± 0.3607 | 2 | 22.1512 ± 0.3607 | 0.2242 ± 1.6249 |
| Yulara | 144 | daylight | Depthwise TCN | 24.2271 ± 2.4573 | 3 | 24.2271 ± 2.4573 | -9.1263 ± 11.0684 |
| NIST Ground | 12 | full | Discrete recurrent | 17.3029 ± 0.4624 | 1 | 17.2851 ± 0.4597 | 58.9176 ± 1.0925 |
| NIST Ground | 12 | full | Inverted-variate | 17.8224 ± 0.1131 | 3 | 17.7851 ± 0.1158 | 57.7292 ± 0.2753 |
| NIST Ground | 12 | full | Joint-patch | 19.4147 ± 0.4149 | 4 | 19.3928 ± 0.4146 | 53.9081 ± 0.9853 |
| NIST Ground | 12 | full | Depthwise TCN | 17.7360 ± 0.0794 | 2 | 17.7145 ± 0.0799 | 57.8969 ± 0.1899 |
| NIST Ground | 12 | daylight | Discrete recurrent | 28.4310 ± 0.8365 | 3 | 28.4122 ± 0.8307 | 58.9405 ± 1.2005 |
| NIST Ground | 12 | daylight | Inverted-variate | 28.4044 ± 0.0934 | 1 | 28.3641 ± 0.0903 | 59.0101 ± 0.1305 |
| NIST Ground | 12 | daylight | Joint-patch | 31.1675 ± 0.5808 | 4 | 31.1420 ± 0.5829 | 54.9956 ± 0.8424 |
| NIST Ground | 12 | daylight | Depthwise TCN | 28.4257 ± 0.1519 | 2 | 28.3963 ± 0.1527 | 58.9635 ± 0.2207 |
| NIST Ground | 48 | full | Discrete recurrent | 27.6673 ± 0.2493 | 1 | 27.6620 ± 0.2416 | 34.0671 ± 0.5759 |
| NIST Ground | 48 | full | Inverted-variate | 28.5043 ± 0.2474 | 2 | 28.4930 ± 0.2419 | 32.0866 ± 0.5767 |
| NIST Ground | 48 | full | Joint-patch | 29.5292 ± 0.4191 | 4 | 29.5359 ± 0.4258 | 29.6007 ± 1.0150 |
| NIST Ground | 48 | full | Depthwise TCN | 29.1156 ± 0.7780 | 3 | 29.1095 ± 0.7845 | 30.6170 ± 1.8699 |
| NIST Ground | 48 | daylight | Discrete recurrent | 43.5225 ± 0.4867 | 2 | 43.5067 ± 0.5042 | 37.3796 ± 0.7257 |
| NIST Ground | 48 | daylight | Inverted-variate | 44.4344 ± 0.9937 | 3 | 44.4196 ± 0.9822 | 36.0656 ± 1.4137 |
| NIST Ground | 48 | daylight | Joint-patch | 45.0398 ± 1.0374 | 4 | 45.0483 ± 1.0269 | 35.1607 ± 1.4780 |
| NIST Ground | 48 | daylight | Depthwise TCN | 42.7742 ± 0.1286 | 1 | 42.7454 ± 0.1393 | 38.4753 ± 0.2005 |
| NIST Ground | 96 | full | Discrete recurrent | 39.8472 ± 1.2537 | 1 | 39.8496 ± 1.2590 | 4.7375 ± 3.0096 |
| NIST Ground | 96 | full | Inverted-variate | 40.2155 ± 0.7406 | 2 | 40.2074 ± 0.7514 | 3.8822 ± 1.7963 |
| NIST Ground | 96 | full | Joint-patch | 41.7544 ± 0.7185 | 4 | 41.7626 ± 0.7264 | 0.1644 ± 1.7365 |
| NIST Ground | 96 | full | Depthwise TCN | 41.1586 ± 1.8057 | 3 | 41.1545 ± 1.8191 | 1.6179 ± 4.3486 |
| NIST Ground | 96 | daylight | Discrete recurrent | 55.5035 ± 0.4465 | 2 | 55.4692 ± 0.4479 | 20.4241 ± 0.6426 |
| NIST Ground | 96 | daylight | Inverted-variate | 55.6752 ± 0.7810 | 3 | 55.6119 ± 0.7839 | 20.2194 ± 1.1245 |
| NIST Ground | 96 | daylight | Joint-patch | 55.8228 ± 0.5182 | 4 | 55.7922 ± 0.5078 | 19.9608 ± 0.7284 |
| NIST Ground | 96 | daylight | Depthwise TCN | 54.6553 ± 0.5142 | 1 | 54.5987 ± 0.5259 | 21.6730 ± 0.7544 |
| NIST Ground | 144 | full | Discrete recurrent | 44.3829 ± 1.2353 | 1 | 44.3881 ± 1.2446 | -5.9397 ± 2.9703 |
| NIST Ground | 144 | full | Inverted-variate | 44.6173 ± 0.6843 | 2 | 44.6093 ± 0.6903 | -6.4675 ± 1.6476 |
| NIST Ground | 144 | full | Joint-patch | 46.4853 ± 0.9207 | 4 | 46.4945 ± 0.9244 | -10.9670 ± 2.2061 |
| NIST Ground | 144 | full | Depthwise TCN | 45.2858 ± 2.4934 | 3 | 45.2792 ± 2.5090 | -8.0663 ± 5.9881 |
| NIST Ground | 144 | daylight | Discrete recurrent | 61.4775 ± 0.7739 | 3 | 61.4452 ± 0.7796 | 11.9037 ± 1.1178 |
| NIST Ground | 144 | daylight | Inverted-variate | 60.6101 ± 1.0335 | 2 | 60.5445 ± 1.0406 | 13.1951 ± 1.4919 |
| NIST Ground | 144 | daylight | Joint-patch | 62.6846 ± 0.7809 | 4 | 62.6530 ± 0.7845 | 10.1720 ± 1.1248 |
| NIST Ground | 144 | daylight | Depthwise TCN | 60.1628 ± 1.8053 | 1 | 60.0995 ± 1.8240 | 13.8330 ± 2.6151 |

M2_INDEPENDENT_AUDIT.json保存每场址/horizon/scope的post hoc descriptive envelope。该包络不是预注册主模型、不是可部署模型，也不作为主假设成功依据。

## 训练记录

| 场址 | 模型 | Seed | 参数量 | 实际epochs | Best epoch | Validation global MSE | 训练秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| Yulara | Discrete recurrent | 42 | 96562 | 13 | 8 | 0.0423451287 | 1394.83 |
| Yulara | Discrete recurrent | 43 | 96562 | 18 | 13 | 0.0291177843 | 2175.77 |
| Yulara | Discrete recurrent | 44 | 96562 | 11 | 6 | 0.0422639589 | 1152.86 |
| Yulara | Inverted-variate | 42 | 102800 | 25 | 20 | 0.015971821 | 198.39 |
| Yulara | Inverted-variate | 43 | 102800 | 16 | 11 | 0.0168455096 | 128.18 |
| Yulara | Inverted-variate | 44 | 102800 | 15 | 10 | 0.0178320063 | 120.14 |
| Yulara | Joint-patch | 42 | 140432 | 12 | 7 | 0.0314961706 | 96.32 |
| Yulara | Joint-patch | 43 | 140432 | 17 | 12 | 0.0275610553 | 138.18 |
| Yulara | Joint-patch | 44 | 140432 | 13 | 8 | 0.0315627834 | 103.31 |
| Yulara | Depthwise TCN | 42 | 682384 | 6 | 1 | 0.0460016082 | 47.93 |
| Yulara | Depthwise TCN | 43 | 682384 | 15 | 10 | 0.0280538075 | 126.44 |
| Yulara | Depthwise TCN | 44 | 682384 | 14 | 9 | 0.0212051758 | 117.30 |
| NIST Ground | Discrete recurrent | 42 | 96562 | 21 | 16 | 0.0162022513 | 1850.30 |
| NIST Ground | Discrete recurrent | 43 | 96562 | 16 | 11 | 0.0168051737 | 1404.92 |
| NIST Ground | Discrete recurrent | 44 | 96562 | 20 | 15 | 0.0160847985 | 2360.15 |
| NIST Ground | Inverted-variate | 42 | 102800 | 24 | 19 | 0.0179006374 | 171.11 |
| NIST Ground | Inverted-variate | 43 | 102800 | 21 | 16 | 0.0185443183 | 150.32 |
| NIST Ground | Inverted-variate | 44 | 102800 | 25 | 24 | 0.0179460971 | 179.37 |
| NIST Ground | Joint-patch | 42 | 140432 | 25 | 24 | 0.0174976746 | 179.48 |
| NIST Ground | Joint-patch | 43 | 140432 | 25 | 21 | 0.0175885667 | 178.59 |
| NIST Ground | Joint-patch | 44 | 140432 | 23 | 18 | 0.0172189594 | 164.64 |
| NIST Ground | Depthwise TCN | 42 | 682384 | 25 | 25 | 0.017714833 | 184.40 |
| NIST Ground | Depthwise TCN | 43 | 682384 | 25 | 25 | 0.0182565444 | 181.15 |
| NIST Ground | Depthwise TCN | 44 | 682384 | 25 | 25 | 0.0187940589 | 181.88 |

## 协议与边界

全部24个best checkpoint、两个Train-fitted处理器、daylight阈值、Train range及评价配置先固定并记录在本地test_release.json，随后才构建Test loader。Test是此前用于数据质量审计的held-out performance split，不称为完全未查看的独立验证集。完整H144 Train/Validation窗口用于训练和global SSE/count选择；Test H12/H48/H96/H144各自独立判定合法prefix，不被后续缺失标签连带删除。

NIST保持FIXED_EST_LST/−05:00、[T−5,T)五个不同有限分钟及availability=T。Yulara保持provider local coordinate、两条离网格排除及+5min可用规则。原M1-R配置逐项未变。NIST_2017_OPERATION_LOG_UNAVAILABLE_NONBLOCKING和YULARA_PROVIDER_METADATA_PARTIAL_NONBLOCKING仍作为数据限制保留；没有根据Test表现删除负功率、负GHI、高GHI低功率或疑似停机记录。

确定性基线为配对CSV键而在三个seed下重复，不代表三次独立拟合；其SD为0。所有方法共享point masks，神经非有限预测会令整个run失败。24个checkpoint均验证无梯度复现预测，labels/origins/target starts/masks与当前读取路径逐元素一致。独立验证还确认所有Test支持计数与M1-R相同。

源提交e60f3482248cc657397db2b665ea0f8957dee7c5；分支research/scheme-a-multisite-frozen-training；Draft PR base为research/scheme-a-multisite-data-confirmation-r1。不修改或合并PR #14/#16/#17，不rebase或force push。原始/派生全量数据、checkpoint、NPZ和运行缓存不提交Git。M3论文整合等待导师审核，本轮不执行。
