# 本轮实际验证与命令范围

运行时间：2026-09-08—09；起始提交14942f4b18bc85d43b168cafc5e7bad81eb9d311。命令在仓库根执行；PY指复现指南中的Python环境，路径参数仅位于忽略本地配置。

|实际命令|结果与边界|
|---|---|
|PY protect_sources.py --paths .local/review_paths.json --phase before/after|764文件SHA256、size、mtime_ns完全相同；无神经训练/原文件改写|
|PY diagnose_ridge.py --paths .local/review_paths.json|Yulara真实矩阵、三个split分布、原解/Cholesky、预测对齐；6求解记录|
|PY diagnose_qcells.py --paths .local/review_paths.json|三个split候选筛选，冻结Test timestamps/labels/masks逐元素一致|
|PY ridge_uncertainty.py --paths .local/review_paths.json|原A/B/神经预测配对，270区间行与块SSE；不是270独立实验|
|PY expanded_ridge.py --paths .local/review_paths.json --destination LOCAL_OUTPUT [--sites ...]|五系统×A/B，统一13alpha；130 Validation候选，10个选定拟合，30范围指标；无神经拟合|
|PY check_nist_qr.py --cache LOCAL_MATRIX_CACHE/NIST_GROUND.pkl --folder LOCAL_EXPANDED/NIST_GROUND|真实增广矩阵QR复核；数值敏感性与RMSE影响单独保存；LOCAL参数为未提交的明确输出目录|
|PY verify_light.py|独立CSV/块SSE算术复算，失败0、skip0；不是checkpoint前向复现|

原M1-R 43、M2普通17、artifact10、10432独立比较及上轮60checkpoint前向/训练日志记录是历史证据，本轮未完整重跑，不计作本轮通过。新增真实矩阵诊断、哈希和块重采样是本轮执行。

运行中真实问题：NIST原逐Timestamp列表设计导致内存/分页压力，C盘空间不足；仅清理本轮可重建临时目录并移至E盘，将等价时间连接改为纳秒向量化，保留固定时区。诊断源文件曾因空间不足写入中断，已恢复后完成实际矩阵运行。低alpha NIST求解差异没有隐去，改用增广QR量化。部分站点恢复运行曾将Yulara诊断小CSV写为空，已从本轮备份恢复，并修复只在有该站点数据时写出；旧证据未改动。TeX最初参数未引用及工作目录错误已修正，最终PDF/包验证另见QA记录。

不将语法、文件存在或历史记录冒充新模型预测复现。原始训练完整重跑未进行。包内轻量验证仅复算聚合证据，完整包提供实际矩阵/逐点复算所需数据但不宣称全训练已试运行。

## 归档验证发现并修复的可移植性问题

首个004110版本完整ZIP的CRC/全量解压及文件SHA通过，但新进程默认GBK读取中文清单失败，故不作为成功交付。新诊断代码的Path文本读写统一显式UTF-8，打包子进程也明确UTF-8。未启用-X utf8的独立verify_light命令已实际通过；随后新时间戳包重新执行完整检查。Git提升进程的worktree所有权校验使用当前命令明确safe.directory（正斜杠），未改变全局配置或remote。
