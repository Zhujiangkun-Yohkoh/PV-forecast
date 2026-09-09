# 本次实际验证及边界

本次固定时期工作从 ad59ed2 开始；旧轮次检查数只作历史记录。

- `expanded_intervals.py --paths LOCAL.json`：保存原/扩展数组逐元素起报、目标和掩码对齐，360行配对区间。
- `frozen_evaluation.py --paths LOCAL.json --sites Sanyo Hanwha Qcells`：36组加载/前向；16通过历史2e-5相对/绝对容差、20未通过。12个Alice原/扩展Ridge回放通过。支持统计输出有效；S1/S2全模型误差不接受。
- `replay_probe.py`、`parser_probe.py`：历史窗口构造、批量/backend、CSV精度核查，未消除差异。当前输入与当前旧代码相同不证明它等于未保存的历史神经输入。
- `frozen_evaluation.py --paths LOCAL.json --sites YULARA_COMBINED NIST_GROUND`：6个2017外部神经、8个Ridge原起报复现通过，再执行固定2018时期。运行时禁止预处理fit、backward与权重保存。
- `audit_period_source.py`、`input_exposure.py`：94个NIST文件、8个缓冲期缺失分钟、固定EST、字段与2018缺失模式/标准化范围实际检查。
- `verify_and_summarize.py --paths LOCAL.json`：216行2018指标从新保存逐点数组独立复算，通过216、失败0、skipped0；加权SSE分解成立。
- `verify_light.py`：1368行原/扩展/新时期配对块结果重放。它读取216行审计记录，不再次加载NPZ，不能称为重复前向验证。
- `protect.py`前后：795个原始/冻结证据文件SHA、size、mtime_ns完全一致。
- latexmk编译两份PDF；`pdf_qa.py`逐页渲染；58页实际视觉检查。14/44页，21图组，字体嵌入，无越界/未定义引用；小号对数指数单独记录。

未执行：神经重训、Ridge重新拟合/选参、全套M1-R/M2历史测试。未完成：Alice恢复起报正式误差和区间。检查行数不是独立实验数量。打包的CRC、全量试解压、清单哈希与轻量运行结果记录于各包sidecar，不冒称训练复现。
