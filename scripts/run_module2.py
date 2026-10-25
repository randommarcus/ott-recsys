"""Module 2: tune and evaluate baselines on VALIDATION only (test stays locked)."""
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd

from ott_recsys.config import Config
from ott_recsys.data import load_movies, load_user_split
from ott_recsys.experiments import DEFAULT_GRIDS, results_table, run_validation, sample_users
from ott_recsys.memguard import Stages

pd.set_option("display.width", 200, "display.max_colwidth", 60)
cfg = Config()
stages = Stages()
split = load_user_split(cfg)
movies = load_movies(cfg)
stages.done("load split")
out = run_validation(split, movies, tune_users=cfg.tune_users, log=lambda m: print(f"  {m}", flush=True))
stages.done("tune + evaluate")

lines = ["== CONFIG ==", *(f"{k:26s} {v}" for k, v in dataclasses.asdict(cfg).items()), "",
         f"grids: {DEFAULT_GRIDS}",
         f"== DATA == users={split.train.user_id.nunique()} movies(catalog)={len(movies)} "
         f"train/val rows={len(split.train)}/{len(split.val)}", "",
         f"== VALIDATION RESULTS ({cfg.run_name}; fit on train, test locked) ==",
         f"tuned on {out['tuned_on_users']} validation users (seeded subsample); table uses all",
         f"best params: {out['best']}",
         "parameters at the edge of their grid: "
         + (str({k: v for k, v in out['edges'].items() if v}) or "none") if any(out["edges"].values())
         else "parameters at the edge of their grid: none",
         f"validation vs train (cold report): {out['cold']}", ""]
lines += ["--- PAIRED DIFFERENCES over the same users: mean [95% CI]; CI excluding 0 = real difference ---"]
lines += [f"{k:58s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("  *" if lo > 0 or hi < 0 else "  (not distinguishable from 0)")
          for k, (m, lo, hi) in out["paired"].items()] + [""]
for view, label in [("all", "ALL validation likes"), ("warm", "WARM likes (item seen in train)"),
                    ("cold", "COLD likes (item never seen in train)")]:
    lines += [f"--- {label} (mean [95% CI over users]) ---", results_table(out["tables"], view).to_string(), ""]
lines.append(sample_users(out, split, movies))
text = "\n".join(lines)
print(text)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / f"module2_validation_{cfg.run_name}.txt").write_text(text)
(ROOT / "reports" / f"module2_validation_{cfg.run_name}.json").write_text(json.dumps({
    "config": {k: str(v) for k, v in dataclasses.asdict(cfg).items()}, "best": out["best"],
    "edges": out["edges"], "cold": out["cold"], "paired": out["paired"],
    "tables": {n: t.to_dict() for n, t in out["tables"].items()}}, default=float))
stages.done("write reports")
