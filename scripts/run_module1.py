"""Run Module 1: audit, sessions, split; save to data/processed/<dataset>/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from ott_recsys.config import Config
from ott_recsys.data import build_all

cfg = Config()
r = build_all(cfg)
cfg.out_dir.mkdir(parents=True, exist_ok=True)
for name in ["train", "val", "test"]:
    r[name].to_parquet(cfg.out_dir / f"{name}.parquet", index=False)
r["movies"].to_parquet(cfg.out_dir / "movies.parquet", index=False)
r["links"].to_parquet(cfg.out_dir / "links.parquet", index=False)

a = r["audit"]
print(f"== AUDIT ({cfg.dataset_name}) ==")
for k in ["ratings", "users", "movies_rated", "movies_in_catalog", "date_min", "date_max",
          "sparsity", "duplicates", "missing_ratings", "missing_tmdb_ids", "missing_genres",
          "movies_with_1_rating", "movies_with_le_5_ratings"]:
    print(f"{k:28s} {a[k]}")
print("\nratings per user\n", a["ratings_per_user"].round(1).to_string())
print("\nratings per movie\n", a["ratings_per_movie"].round(1).to_string())
print("\nrating distribution\n", a["rating_distribution"].to_string())
df = r["df"]
print(f"\npositive share (rating>=4): {df['label'].mean():.3f}")
print("\n== SESSIONS / BURSTS ==")
for k, v in r["sessions"].items():
    print(f"{k:34s} {v:.3f}" if isinstance(v, float) else f"{k:34s} {v}")
print("\nhour (UTC) share:\n", (df["hour"].value_counts(normalize=True).sort_index().round(3)).to_string())
print("\ndow share (0=Mon):\n", (df["dow"].value_counts(normalize=True).sort_index().round(3)).to_string())
print("\n== SPLIT ==")
for n in ["train", "val", "test"]:
    p = r[n]; print(f"{n:5s} rows={len(p):6d} {p['ts'].min():%Y-%m-%d} -> {p['ts'].max():%Y-%m-%d} pos={p['label'].mean():.3f}")
print("cutoffs:", {k: pd.to_datetime(v, unit='s') for k, v in r["cuts"].items()})
print("\n== COLD START ==")
for n, c in r["cold"].items():
    print(n, c)
