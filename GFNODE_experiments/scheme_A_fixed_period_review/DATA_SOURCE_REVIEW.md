# 数据来源与负值规则复核（2026-09-09）

## Alice 有限负功率
本次重新读取官方 [Glossary](https://dkasolarcentre.com.au/glossary) 的历史 DKASC 数据参数：五分钟 AC 功率来自更高频率计量平均。术语页也描述了累计 delivered-minus-received 电量，但不能据此证明下载 CSV 的每一条负 Active_Power 都是正常净功率。没有找到把所有有限负功率定义为无效码的官方规则。

项目 `scheme_A_submission_correction/run_corrected_benchmark.py` 的 `_valid_power` 明确实施 `finite & >=0`；完整 H144 窗口要求全部未来标签满足该条件。可以确认这是项目实现的清洗规则，尚不能确认其提供方依据。历史结果保留是为了可追溯，不表示已证明该规则物理合理。数值接近零也不足以判定记录正常。

官方 [Notes on the Data](https://www.dkasolarcentre.com.au/download/notes-on-the-data) 明示记录并非所有短时事件的完整清单。本次可访问页面没有解决研究期负记录的物理身份。未找到不等于没有设备、运行或仪表异常。未向提供方发邮件。作者可后续询问：各阵列 Active_Power 的符号约定、明确无效码、2018 年 4–8 月负记录与仪表/逆变器运行的对应关系。

## NIST 2018 固定时段
从 [NIST 官方 PV 门户](https://pvdata.nist.gov/) 公开下载接口取得 Ground、1-min avg.、2018 年档案，Box 文件标识 501051140107。接口提供的访问 token 不写入日志、代码或归档。原 ZIP 的 CRC 检查通过；提取 2018-03-30 至 2018-07-01 的 94 个 CSV，包含前后缓冲，未按模型误差选择月份。具体文件及字节数见随原数据保存的 DOWNLOAD_RECORD.json。

字段沿用已核实的 [NIST 数据论文](https://nvlpubs.nist.gov/nistpubs/jres/122/jres.122.040.pdf) 和 [数据字典](https://www.nist.gov/document/datadictionarysupplementalcontentpdf)：电表 AC real power、环境气温、Ground GHI。不能将逆变器的 -999 列代替功率标签。新文件需通过实际表头、单位列及固定 -05:00 时间解析检查后才参与评价。

目前没有取得完整 2018 Ground 设备/运行变更日志，因此不能声称跨年设备完全未改变。相同官方 Ground 序列和计量字段支持同场址固定时期比较；设备状态未完整核实属于其解释限制，不按功率-GHI形态或预测难度删除观测。

## Yulara 2018
使用用户此前提供的同一个 106.6 kW combined-system CSV。它的覆盖跨越 2017 与本次固定 2018 时段，并非新增场址。继续以提供方本地坐标、规则网格和 +5 min 可用时刻处理；不推测 UTC 偏移。详细字段/运行元数据仍部分缺失，不能据此声称无设备变动或完整质量标识。原生 GHI 不做额外换算。

以上是本次重新访问的来源及其实际支持范围。2018 数据覆盖审计与模型误差分开输出；没有用模型分数决定资料适用性或更换时期。
