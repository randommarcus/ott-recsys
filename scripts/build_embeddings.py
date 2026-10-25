"""Build item texts from the TMDB cache, report coverage, embed with SBERT (CPU),
sanity-check with nearest neighbours. Needs the TMDB fetch to be (nearly) done."""
import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from ott_recsys import content
from ott_recsys.config import Config
from ott_recsys.data import load_movies
from ott_recsys.memguard import Stages

cfg = Config()
stages = Stages()
movies, links = load_movies(cfg), pd.read_parquet(cfg.out_dir / "links.parquet")
cutoff = int(pd.read_parquet(cfg.out_dir / "train.parquet", columns=["timestamp"])["timestamp"].max())
tmdb = content.load_tmdb_cache(ROOT / "data" / "raw" / "tmdb", links["tmdbId"])
missing = int(links["tmdbId"].dropna().astype(int).nunique() - len(tmdb))
if missing > 0.01 * len(movies) and "--allow-partial" not in sys.argv:
    sys.exit(f"{missing} TMDB ids are not cached yet; finish scripts/fetch_tmdb.py first (or pass --allow-partial)")
tags = content.movie_tags(cfg.dataset_dir / "tags.csv", cutoff, movies["movieId"])
texts = content.build_texts(movies, links, tmdb, tags)
cov = content.coverage(movies, links, tmdb, texts)
stages.done("texts + coverage")

lines = ["== CONFIG ==", *(f"{k:26s} {v}" for k, v in dataclasses.asdict(cfg).items()), "",
         f"tags used: dated <= global train cutoff {pd.to_datetime(cutoff, unit='s')} (movies with such tags: {len(tags)})",
         "== TMDB COVERAGE ==", *(f"{k:30s} {v}" for k, v in cov.items()),
         f"{'overview threshold (chars)':30s} {content.MIN_OVERVIEW_CHARS}", ""]
print("\n".join(lines), flush=True)

t0 = time.perf_counter()
emb384 = content.encode_texts(texts["text"].tolist())
enc_s = time.perf_counter() - t0
emb64 = content.pca_reduce(emb384, 64)
ids = texts["movieId"].to_numpy()
content.save_embeddings(cfg.out_dir / "item_embeddings.npz", ids, emb384, emb64)
texts.to_parquet(cfg.out_dir / "item_texts.parquet", index=False)
lines.append(f"SBERT {content.MODEL_NAME}: {len(texts)} texts, CPU, batch 64: {enc_s:.0f}s ({len(texts) / enc_s:.0f} texts/s); "
             f"dims 384 and PCA-64 (PCA is unsupervised, fit on item content only)")
stages.done("embeddings")

titles = movies.set_index("movieId")["title"].to_dict()
lines.append("\n== NEAREST NEIGHBOURS (cosine, 384-d | 64-d) ==")
for t in ["Toy Story (1995)", "Matrix, The (1999)", "Titanic (1997)", "Godfather, The (1972)", "Dark Knight, The (2008)"]:
    hit = movies[movies["title"] == t]
    if hit.empty:
        lines.append(f"{t}: not in subset")
        continue
    q = int(hit["movieId"].iloc[0])
    lines.append(f"{t}\n   384: " + "; ".join(content.neighbours(emb384, ids, titles, q)) +
                 "\n    64: " + "; ".join(content.neighbours(emb64, ids, titles, q)))
text = "\n".join(lines)
print(text)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / f"module3_embeddings_{cfg.run_name}.txt").write_text(text)
