"""Current portable figure entry; frozen historical builder is build_figures_legacy.py."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2]/"GFNODE_experiments/scheme_A_review_extension/build_review_figures.py"),run_name="__main__")
