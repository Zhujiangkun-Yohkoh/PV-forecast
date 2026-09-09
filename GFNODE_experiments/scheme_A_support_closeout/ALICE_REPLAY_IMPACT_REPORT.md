# Alice回放差异与指标影响

基线89e494f；36组原起报与上一轮保存的新前向数组逐元素核对。原rtol=atol=2e-5不变：16通过、20未通过。TCN与Recurrent各9组未通过，Sanyo Inverted43/44未通过；其余16组通过。不是checkpoint损坏或模型失效判定。

最大单点差为1.286983490 W（0.001286983490 kW），Qcells TCN44，2018-08-19 09:45:00，提前110分钟。其全输出预测差RMS为0.020311584 W。逐组输出包含完全相同比例、超过容差点数/比例/起报、median/P95/P99/max、预测差RMS，以及full/active/low三个目标范围。CSV同时列kW与W。数值全输出统计包含原不可评分目标位置；指标影响仅在真实有效且Daily匹配的同一标签/掩码上计算。

在1h/12h、full/active/low各原支持中，RMSE变化最大0.000566719 W，MAE变化最大0.000424838 W，bias变化最大0.001044245 W。已比较的四模型均值排名改变0项，相对Daily效应符号改变0项。未重新定义或替换旧指标。

界限：同一支持的 |RMSE(new)-RMSE(old)| <= RMS(new-old)，|MAE(new)-MAE(old)| <= mean|new-old|；所有216个指标影响行均满足。界限不覆盖新增起报，也不证明历史输入状态相同。参考RMSE接近零时，skill的相对变化可被放大；CSV保留old/new skill与Daily分母，不使用epsilon。

## 代表诊断（固定5组，未重复CPU/parser搜索）
| site | model | seed | mode | batch_size | max_difference_W | passes_original_tolerance |
| --- | --- | --- | --- | --- | --- | --- |
| Qcells | Depthwise convolutional TCN | 44 | original_batch | 256 | 1.28698 | False |
| Qcells | Depthwise convolutional TCN | 44 | mixed_batch | 256 | 1.28698 | False |
| Qcells | Depthwise convolutional TCN | 44 | singleton | 1 | 1.52294 | False |
| Qcells | Depthwise convolutional TCN | 44 | original_last_incomplete | 63 | 0 | True |
| Qcells | Depthwise convolutional TCN | 44 | mixed_last_incomplete | 184 | 0 | True |
| Sanyo | Depthwise convolutional TCN | 42 | original_batch | 256 | 0.247717 | False |
| Sanyo | Depthwise convolutional TCN | 42 | mixed_batch | 256 | 0.247717 | False |
| Sanyo | Depthwise convolutional TCN | 42 | singleton | 1 | 0.750929 | False |
| Sanyo | Depthwise convolutional TCN | 42 | original_last_incomplete | 189 | 0.0188798 | True |
| Sanyo | Depthwise convolutional TCN | 42 | mixed_last_incomplete | 184 | 0 | True |
| Sanyo | Inverted-variate Transformer | 43 | original_batch | 256 | 0.0278056 | False |
| Sanyo | Inverted-variate Transformer | 43 | mixed_batch | 256 | 0.0278056 | False |
| Sanyo | Inverted-variate Transformer | 43 | singleton | 1 | 0.0278503 | False |
| Sanyo | Inverted-variate Transformer | 43 | original_last_incomplete | 189 | 0.0106543 | True |
| Sanyo | Inverted-variate Transformer | 43 | mixed_last_incomplete | 184 | 0 | True |
| Sanyo | Inverted-variate Transformer | 44 | original_batch | 256 | 0.0300556 | False |
| Sanyo | Inverted-variate Transformer | 44 | mixed_batch | 256 | 0.0300556 | False |
| Sanyo | Inverted-variate Transformer | 44 | singleton | 1 | 0.031054 | False |
| Sanyo | Inverted-variate Transformer | 44 | original_last_incomplete | 189 | 0.00406802 | True |
| Sanyo | Inverted-variate Transformer | 44 | mixed_last_incomplete | 184 | 0 | True |
| Qcells | Inverted-variate Transformer | 42 | original_batch | 256 | 0.0109673 | True |
| Qcells | Inverted-variate Transformer | 42 | mixed_batch | 256 | 0.0109673 | True |
| Qcells | Inverted-variate Transformer | 42 | singleton | 1 | 0.00977516 | True |
| Qcells | Inverted-variate Transformer | 42 | original_last_incomplete | 63 | 0.00485778 | True |
| Qcells | Inverted-variate Transformer | 42 | mixed_last_incomplete | 184 | 0 | True |

原/混合256批中被选中的Alice极值轨迹数值相同，换回原批次未解决这些Alice差异；singleton可变化，不能将其视为真值。最后不完整批次的选定轨迹则可能通过，这说明不能由单个批次外推全部起报。

五组checkpoint均strict state_dict加载，无缺失/额外键，eval模式、float32、72×17输入和目标逆变换已核对。输入通道、KNN拟合矩阵维度、scaler已拟合数值、权重键与参数量见REPRESENTATIVE_IDENTITY.json。处理器是后来保存的Train-only Ridge处理器；没有单独原神经KNN/IF/scaler对象或原输入数组可直接比对。已保存IF在冻结增强路径中调用，但无法与未保存的历史IF对象逐树比较。checkpoint本身只含epoch/state_dict/validation_global_mse；run_summary为运行记录列表，不提供完整历史cuDNN/backend快照。

当前Python/PyTorch/CUDA/cuDNN/GPU及TF32配置见CURRENT_FORWARD_ENVIRONMENT.json。实际模型源码为基线89e494f中的冻结run_corrected_benchmark.py，本次未修改；保护清单提供内容校验。旧parser/CPU结果保留为历史诊断，本次不重复。当前证据支持“已评分指标稳定，但历史数值状态未完全恢复”，不支持把GPU后端或输入重建断言为唯一原因。

## 按比较推进
Hanwha和Qcells完整三个Inverted种子可用于接受的主比较。Sanyo仅Inverted42可单独报告；不补齐假想三seed均值。三个系统的Ridge及参考均接受；Joint-patch为接受的次要结果。TCN/Recurrent和Sanyo43/44只在独立diagnostic指标CSV保留，不进入正式新支持均值或排名。
