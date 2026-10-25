"""Fetch TMDB details for every movie in the subset. Standalone, resumable.

    nohup setsid .venv/bin/python -I scripts/fetch_tmdb.py >> logs/fetch_tmdb.log 2>&1 &

Exit codes: 0 done, 2 bad/missing key, 3 aborted (network), 4 some ids failed."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import requests
from dotenv import load_dotenv

from ott_recsys.config import Config
from ott_recsys.tmdb import fetch_all

limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
cfg = Config()
load_dotenv(ROOT / ".env")
key = os.environ.get("TMDB_API_KEY", "").strip()
if not key:
    print("TMDB_API_KEY is missing from .env", flush=True)
    sys.exit(2)

links = pd.read_parquet(cfg.out_dir / "links.parquet")
ids = links["tmdbId"].dropna().astype(int).unique().tolist()
if limit:
    ids = ids[:limit]
stamp = lambda m: print(f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {m}", flush=True)
stamp(f"start: dataset={cfg.run_name}, {len(ids)} ids")
stats = fetch_all(ids, ROOT / "data" / "raw" / "tmdb", key, requests.Session, log=stamp)
sys.exit(2 if (stats["aborted"] or "").startswith("TMDB rejected") else 3 if stats["aborted"] else 4 if stats["failed"] else 0)
