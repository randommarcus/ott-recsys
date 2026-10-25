"""Module 2: tune and evaluate baselines on VALIDATION only (test stays locked)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd

from ott_recsys.config import Config
from ott_recsys.data import load_movies, load_user_split
from ott_recsys.experiments import results_table, run_validation, sample_users

pd.set_option("display.width", 200, "display.max_colwidth", 60)
cfg = Config()
split = load_user_split(cfg)
movies = load_movies(cfg)
out = run_validation(split, movies)

lines = [f"== VALIDATION RESULTS ({cfg.dataset_name}; fit on train, test locked) ==",
         f"best params: {out['best']}",
         f"validation vs train (cold report): {out['cold']}", ""]
for view, label in [("all", "ALL validation likes"), ("warm", "WARM likes (item seen in train)"),
                    ("cold", "COLD likes (item never seen in train)")]:
    lines += [f"--- {label} (mean [95% CI over users]) ---", results_table(out["tables"], view).to_string(), ""]
lines.append(sample_users(out, split, movies))
text = "\n".join(lines)
print(text)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "module2_validation.txt").write_text(text)
