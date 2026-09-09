# 本轮验证说明（2026-09-08）

本轮实际执行：60个NPZ读取归约；2400个SSE分解恒等式；8个synthetic forward检查（17/7通道、模块参与）；565个Ridge/训练记录检查（0失败0skip）；6720个新归约与冻结CSV指标/计数比较（0失败0skip）。10个Ridge闭式拟合完成，50个Validation候选全部保留；没有神经训练。605个原保护文件快照再次对比无变化。完整包另做全部源文件打包前后size/mtime_ns对比。

历史而非本轮全套重跑：M1-R43项、M2普通17项、artifact10项、独立10432项、Alice旧4414项。checkpoint前向复现依据旧M3记录，本轮不冒称重跑。Alice原始CSV最初读取前缺少独立快照，代码仅只读；打包前后会实测全部原始CSV。

论文：主文10页、5图5表；Supplement36页、9图18表；摘要TeXcount196词。正文2895（旧4245）不含摘要、图表/图注、声明和参考，包含标题及TeXcount公式权重；图注287，表格源词/数值token428，参考源token853，后两者为单独词法计数，不能与正文TeXcount机械相加。所有数值口径见WORD_COUNTS.json。

两份PDF完整编译，全部页面渲染并以逐页联系表查看，重点单图复核；无missing citation、undefined reference、duplicate label或overfull。字体嵌入、图件矢量性做程序检查；热图改用矢量色块，避免imshow栅格化。模板可能有underfull排版提示，未发现实际裁切/遮挡。详细视觉事项在FIGURE_QA.md。

复现①CSV→图件与②保存预测→指标本轮实际执行。③官方数据→训练仅生成独立副本、导入与路径检查通过，没有执行60次训练复现。原历史git-show检查保留在历史源码，当前便携入口不依赖该历史对象。没有基于测试结果修改alpha集合或神经设置。
