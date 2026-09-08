# P01 Yulara Ridge 真实矩阵诊断（2026-09-08—09）

结论：原 Yulara Ridge 极端误差不是此次检查发现的时间索引、单位、逆变换或求解器错误；主要证据指向罕见缺失模式的特征外推，以及原正则化候选范围不足。无约束线性预测可以远超 Train 功率范围。本轮没有删除高误差窗口、裁剪功率或改写原预测。

## 实际检查与定义

`designs.py` 从保存的 Train-only 处理器和原始三变量重建七通道及72步历史。A为504列；B额外加入144个逐目标−24h功率和144个缺失指示，共792列。Yulara维持provider-local时间与+5min availability；NIST保持固定−05:00。每个未来目标为origin+j×5min；Daily/B取该目标准确提前24h，在12h轨迹中最晚为origin−12h。因此均在起报时可用。比较实际origins、labels、target masks，不仅比较点数。

所有原始KNN、IF、scaler和最终标准化参数仅来自Train并直接复用。额外敏感性仅重新求解Ridge系数。目标函数是标准化输入、中心化目标上的 ||Y−XB||²_F+alpha||B||²_F，截距不惩罚；不能把alpha与按样本数平均的目标函数混用。

## 分布与误差集中

原A在完整H144、Daily匹配支持的Test RMSE=119.584750772 kW，B=35.950323857 kW。分布诊断表采用保存候选轨迹的全部有限标签，支持更宽，所以其Test RMSE=119.429239不能与headline混用。A Validation MSE=47033.794852 kW²（RMSE216.872762），Train最大真实功率109.903154 kW。A Validation预测约−1090.59至723.47，Test约−1124.40至729.39。

Train/Val/Test预测分位数、负值与超Train范围比例见 `ridge_distributions.csv`，逐分割最差20个起报见 `ridge_worst_origins.csv`。A的Test最高误差1%/5%窗口贡献总SSE的20.9512%/90.6781%；B为17.0578%/80.1562%。是集中爆发，不是所有时刻均同样恶化。诊断极端日期不作为代表性案例，全部仍进入原指标。

## 特征机制与数值核对

起报位置气温/GHI缺失指示在69333个Train窗口中各出现4次（0.0058%）；Validation18.5641%，Test5.4975%。Train标准差约0.007595，缺失标记1变成约131.65个标准化单位。两列一致、贡献相关；原A两通道各自Validation预测分量RMS约107.638 kW，Test各58.426 kW。不能把这些相关分量平方后当成独立MSE贡献相加。完整近常数、稀疏度、偏移、尺度见 `ridge_features.csv`，加性预测分量见 `ridge_feature_contributions.csv`。

使用实际69333×504/792矩阵的Cholesky独立求解，匹配标准化、alpha、截距和目标函数。原A Test最大差8.98945e−9 kW，B约2.91e−11；相对正规方程残差约1e−15，目标函数一致。Gram矩阵微小负特征值约−1.5e−10属于数值舍入，原A正则化条件数约8.96e7。真实矩阵结论不依赖之前小型合成测试。证据见 `ridge_solver_comparison.csv`、`ridge_alignment.csv`。

## 统一扩展网格：补充敏感性，不替换原结果

原10次选择中7次在原0.1—1000端点。本轮诊断后先记录统一13点网格10^−4至10^8，再对五系统A/B全部按完整H144 Validation MSE选择。130个Validation分数全部保留；Test未用于选择、决定是否报告或裁剪。只有NIST A仍在新下端点；没有继续搜索。

site | model | selected_alpha | old_RMSE | new_RMSE | validation_MSE_kW2
--- | --- | --- | --- | --- | ---
Sanyo | Ridge | 100 | 0.913715581 | 0.913715581 | 0.819793084
Sanyo | Ridge+day | 1000 | 0.211927137 | 0.211927137 | 0.130711725
Hanwha | Ridge | 1 | 0.844085435 | 0.844085435 | 0.605167072
Hanwha | Ridge+day | 100 | 0.248024217 | 0.248024217 | 0.101223667
Qcells | Ridge | 1000 | 0.839062914 | 0.839062914 | 0.647827757
Qcells | Ridge+day | 10000 | 0.236520307 | 0.259459041 | 0.137887056
YULARA_COMBINED | Ridge | 10000000 | 119.584751 | 29.8382291 | 816.895458
YULARA_COMBINED | Ridge+day | 1000000 | 35.9503239 | 15.8421097 | 268.330155
NIST_GROUND | Ridge | 0.0001 | 48.9560564 | 48.9560691 | 2280.44749
NIST_GROUND | Ridge+day | 1000 | 34.9950682 | 34.9950682 | 1152.26021

Yulara A/B选择1e7/1e6，Validation MSE降至816.895458/268.330155。Qcells B的Validation略改善但Test从0.236520变成0.259459，保留这个不利变化。不能以扩展后的Yulara结果反推此前方案正确，也不能把新拟合写成预注册结果。

NIST A新alpha1e−4时谱解/Cholesky出现0.002390 kW局部差；独立增广QR秩504，full RMSE差−1.834905e−7 kW。该微小数值敏感性已单独记录，QR预测另存、未替换谱解。见 `NIST_GROUND_Ridge_QR_CHECK.json`、`nist_QR_metric_sensitivity.csv`。它不是原Yulara异常的解释。

影响范围：主文摘要、线性参考结果和讨论；补充诊断方法、原Ridge与扩展网格图；中文报告。原神经指标、原Ridge表以及P03原网格区间全部保留。受影响图号见 `FIGURE_NUMBER_MAP.csv`。进一步限制：未做特征机制的独立因果干预；这里是实际矩阵分量和分布漂移支持的诊断，不是气象/运行事件归因。
