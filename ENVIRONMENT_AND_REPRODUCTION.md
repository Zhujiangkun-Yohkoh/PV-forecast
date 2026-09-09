# Current closeout entries

From project root: `python GFNODE_experiments/scheme_A_support_closeout/verify_light.py` independently recomputes block-weight intervals without importing production interval functions, and reports observed replay status without expecting a fixed failure count. `audit_points.py --paths LOCAL.json` needs the full saved prediction evidence. `reduce_periods.py --paths LOCAL.json` only reduces existing arrays. `period_forward.py` refuses an existing output file; use an explicitly fresh destination for new inference, never archive evidence. No processor fitting or model training is invoked.

Current figure entry: `python manuscript/clean_pv_benchmark/build_figures.py`; current folder closeout_figures. Only two affected plots are regenerated; twenty unchanged figures are retained with their source scripts. The full handoff includes earlier builders and their CSV inputs. Core scientific state is B, not a claim of complete historical neural-state recovery. The original fixed_period_review/verify_light.py remains a frozen historical 20-failure snapshot.

## Earlier entries (historical)

# Current fixed-period reproduction scope

Run from the project root. Current plotting entry: `python manuscript/clean_pv_benchmark/build_figures.py` (NumPy, pandas, Matplotlib); it creates/refreshes fixed_period_figures from versioned CSVs and preserved historical figure sources. Use a standard scientific Python environment with DejaVu Sans, without private Windows font paths.

Light arithmetic: `python GFNODE_experiments/scheme_A_fixed_period_review/verify_light.py` replays 1368 rows from block SSE/counts. It does not rerun the recorded 216-row point-array audit or checkpoint inference. Heavy inference uses `frozen_evaluation.py --paths LOCAL.json`; explicit paths must resolve to the full package evidence. Never point outputs at frozen evidence. The full package has relative input configuration and a separate outputs destination. Missing raw data/checkpoints must be supplied; aggregate CSVs cannot replace them.

This revision loaded all 36 Alice and six external checkpoint groups for inference, with no fit/search/training. External replay passed; 20 Alice groups remain unresolved. Original Alice neural preprocessing objects were not saved separately. Whole training reproduction is NOT tested. Package extraction verifies only its stated lightweight scope.

## Earlier environment and reproduction documentation (historical scope)

# 本轮入口（2026-09-09，优先于下方历史指南）

①CSV→最新20组图：`python -B GFNODE_experiments/scheme_A_diagnostic_revision/build_figures.py`。依赖numpy/pandas/matplotlib，输出diagnostic_figures，使用DejaVu，无私人字体路径。本轮实际运行并在最终PDF检查；旧portable_entry figures输出上轮图，不是当前入口。

②保存预测→新诊断：`python -B GFNODE_experiments/scheme_A_diagnostic_revision/ridge_uncertainty.py --paths review_paths.json`。真实矩阵：`diagnose_ridge.py --paths review_paths.json`；Qcells：`diagnose_qcells.py --paths review_paths.json`。本轮均执行；完整包在project目录提供相对配置。轻量包仅运行`python -B GFNODE_experiments/scheme_A_diagnostic_revision/verify_light.py`，不需要raw或PyTorch，只复算CSV/块SSE。缺失重型路径会明确失败，不搜索磁盘。已有可信pickle需匹配环境：Python3.12、numpy2.0、pandas2.2.2、sklearn1.5、scipy1.13.1、torch2.7.1+cu118。

③官方数据→训练：保留原代码、明确路径及上轮prepare_training_copy；本轮未运行60次神经训练，不能将包内验证称为全训练复现。新增Ridge网格的expanded_ridge.py是明确的补充拟合入口，不是轻量检查。QR脚本需实际矩阵cache；可由expanded_ridge --matrix-cache在显式目录产生，cache不打包，因为原始数据/系数/处理器足以重建。

本轮命令和运行中修复见新诊断VALIDATION_REPORT.md。完整包包含全部已知本地证据；原Alice神经处理器并未另存，不能把新的Ridge处理器称为该历史对象。下文所有“本轮”字样指9月8日上一轮历史记录。

## 旧指南（历史记录）

# 环境与三条复现路径

本轮日期：2026-09-08。所有命令从项目根目录执行；不包含私人绝对路径、自动全盘搜索或需要历史 git show 的默认入口。

## ① CSV → 正式图件（本轮实际执行）

```text
python -B GFNODE_experiments/scheme_A_review_extension/portable_entry.py figures
```

依赖 numpy、pandas、matplotlib。默认使用 Matplotlib 自带的 DejaVu Sans，不依赖 Windows Arial 路径。输出位于 manuscript/clean_pv_benchmark/review_figures：14组 PDF/SVG/320dpi PNG、源数据CSV、caption和alt。PDF均为矢量统计图，SVG保留文字，可进一步编辑。build_figures.py 已改为本轮入口；build_figures_legacy.py 保留历史M3绘图实现，会覆盖旧表格，**不要用 legacy 入口重建当前稿件**。历史源码保留供溯源，当前入口如上。

论文编译：进入 manuscript/clean_pv_benchmark，使用 TeX Live 2025 的 latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex；Supplement同理。原REVTeX模板暂作导师审阅排版，选刊后调整，不代表已决定投JRSE。本轮已实际完整编译和逐页查看；详细计数见 REVIEW_VALIDATION.md。

## ② 保存预测 → 指标（本轮实际执行）

```text
python -B GFNODE_experiments/scheme_A_review_extension/portable_entry.py inspect --paths review_paths.json
python -B GFNODE_experiments/scheme_A_review_extension/portable_entry.py metrics --paths review_paths.json
python -B GFNODE_experiments/scheme_A_review_extension/portable_entry.py verify --paths review_paths.json
python -B GFNODE_experiments/scheme_A_review_extension/compare_frozen.py
```

review_paths.json 明确给出 alice_results、external_results、alice_raw（三个路径）、external_paths（外部原始路径JSON）和 new_results（本轮Ridge目录）。相对路径相对于命令工作目录。私下完整包提供可直接使用的相对路径配置；Git不提交本机绝对路径配置。该入口缺文件时明确报错，不猜测 master 数据目录。

原60份NPZ已含逐点预测、标签、时间戳、有效mask与真实origin power；Alice的Daily参考需从原始CSV精确时间连接，外部NPZ含Daily和阈值。Ridge保存10份预测NPZ、10份系数NPZ、完成元数据及5个新拟合处理对象。不要把历史 pickle 当成跨Python版本的通用格式：在本轮环境读取，或按明确Train流程重新拟合输入处理器。加载pickle只限可信的本项目文件。

## ③ 官方数据 → 神经训练（历史执行，本轮未重跑）

完整实现保留在 scheme_A_submission_correction 与 scheme_A_multisite_extension。原 Alice 实现支持 SCHEME_A_CORRECTION_RESULTS_ROOT 输出目录；原外部实现的 frozen_config() 调用历史 git show。这是历史证据检查的依赖，未删除历史记录，也没有声称 ZIP 无Git时直接运行该历史主函数已通过。

公开重现时使用 `prepare_training_copy.py --paths review_paths.json --output <新空目录>` 生成独立训练副本：从已附的当前正式配置读取，改为显式数据路径；历史配置来源在 PROVENANCE.md 保留。该准备步骤不训练。本轮测试了副本生成及导入/显式路径检查，**未执行60次训练复现**，因此路径③只能标为“准备与代码检查完成，完整训练未测试”。训练需GPU及数小时预算，必须由复现者有意运行副本入口。

## 本轮运行环境

Windows、TeX Live 2025；历史神经环境 PyTorch 2.7.1+cu118，RTX 3060 Laptop GPU（具体历史信息以原environment和run记录为准）。本轮Ridge使用CPU确定性闭式解，没有调用神经训练、optimizer或反向传播。新图用单独的绘图库，不修改原神经虚拟环境。确切本轮Python/numpy/pandas/scikit-learn/scipy/matplotlib版本由 review_environment.json 记录；不要将本机运行目录或虚拟环境打包。

## 数据及公开边界

官方数据来源、字段、下载页与条款保留在 M1-R MULTISITE_PROTOCOL.md、DOCUMENTATION_LIMITATIONS.md、论文references.bib及DATA_AUDIT_SUMMARY.csv。复现者应从提供方下载并遵守条款。此次导师私下审核包中的数据和artifact用于本项目核查，**不意味着公开再分发许可或代码许可证已选择**。Git提交不含raw、checkpoint、NPZ或处理器。未更改仓库可见性、未创建Release。
