# 导师审核入口：2026-09-09诊断修订

先读PROJECT_REPORT_CN.md，然后新诊断目录中的P01/P02/P03报告与MANUSCRIPT_CHANGELOG.md，再读最新主文PDF和Supplement。REVIEW_RESPONSE_MATRIX.md区分已解决问题与剩余科学限制。

本轮不重训神经网络，旧结果完整保留。重点是Yulara缺失模式外推、Qcells负标签与完整窗口选择、原Ridge的条件时间区间。新增统一alpha为事后敏感性，不替换原网格。选择期刊前需机构核实CAS版本和分区。

完整包支持重型证据审核；轻量包仅支持聚合/块SSE复算；图件包支持图数据与版式交接。各包清单、校验与轻量验证见PACKAGE_VERIFICATION.md。不得将其解释为重新跑通60次训练。
