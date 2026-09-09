from pathlib import Path
import runpy
root=Path(__file__).resolve().parents[2]
runpy.run_path(str(root/'GFNODE_experiments/scheme_A_fixed_period_review/build_figures.py'),run_name='__main__')
