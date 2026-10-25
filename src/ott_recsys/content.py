"""Item text, TMDB coverage, release dates and SBERT embeddings."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_OVERVIEW_CHARS = 40  # shorter overviews count as "very short" -> fallback text
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_tmdb_cache(cache_dir: Path, tmdb_ids) -> pd.DataFrame:
    """One row per cached tmdbId; ids not cached are simply absent."""
    rows = []
    for i in pd.unique(pd.Series(tmdb_ids).dropna().astype(int)):
        p = Path(cache_dir) / f"{i}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows.append({"tmdbId": i, "not_found": bool(d.get("not_found")), "overview": d.get("overview") or "",
                     "release_date": d.get("release_date") or "", "genres": d.get("genres") or [],
                     "keywords": d.get("keywords") or []})
    cols = ["tmdbId", "not_found", "overview", "release_date", "genres", "keywords"]
    return pd.DataFrame(rows, columns=cols).set_index("tmdbId")


def movie_tags(tags_csv: Path, cutoff_ts: int, movie_ids, top: int = 5) -> dict:
    """Top MovieLens tags per movie, using ONLY tags dated on or before cutoff_ts
    (so text never contains information from after the global train cutoff)."""
    t = pd.read_csv(tags_csv, usecols=["userId", "movieId", "tag", "timestamp"])
    t = t[(t["timestamp"] <= cutoff_ts) & t["movieId"].isin(set(movie_ids))].dropna(subset=["tag"])
    t["tag"] = t["tag"].str.lower().str.strip()
    c = t.groupby(["movieId", "tag"])["userId"].nunique().reset_index(name="n")
    c = c.sort_values(["movieId", "n", "tag"], ascending=[True, False, True])
    return c.groupby("movieId")["tag"].apply(lambda s: list(s.head(top))).to_dict()


def build_texts(movies: pd.DataFrame, links: pd.DataFrame, tmdb: pd.DataFrame, tags: dict) -> pd.DataFrame:
    """text = 'title. genres. overview'; if the overview is missing/very short:
    'title. genres. [short overview]. Keywords: ... Tags: ...' (fallback)."""
    tm = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    rows = []
    for r in tm.itertuples(index=False):
        genres = "" if r.genres == "(no genres listed)" else r.genres.replace("|", ", ")
        base = ". ".join(x for x in [r.title, genres] if x)
        info = tmdb.loc[int(r.tmdbId)] if pd.notna(r.tmdbId) and int(r.tmdbId) in tmdb.index else None
        ov = "" if info is None else info["overview"].strip()
        if len(ov) >= MIN_OVERVIEW_CHARS:
            rows.append((r.movieId, f"{base}. {ov}", "overview", False, False))
            continue
        kw = [] if info is None else list(info["keywords"])[:10]
        tg = tags.get(r.movieId, [])
        parts = [base] + ([ov] if ov else []) + ([f"Keywords: {', '.join(kw)}"] if kw else []) \
            + ([f"Tags: {', '.join(tg)}"] if tg else [])
        rows.append((r.movieId, ". ".join(parts), "fallback", bool(kw), bool(tg)))
    return pd.DataFrame(rows, columns=["movieId", "text", "source", "has_keywords", "has_tags"])


def release_dates(movies: pd.DataFrame, links: pd.DataFrame, tmdb: pd.DataFrame) -> pd.Series:
    """TMDB release date per movieId (NaT if unknown)."""
    tm = movies[["movieId"]].merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    rd = tm["tmdbId"].map(lambda i: tmdb["release_date"].get(int(i), "") if pd.notna(i) else "")
    return pd.Series(pd.to_datetime(rd.replace("", None), errors="coerce").to_numpy(), index=tm["movieId"].to_numpy())


def coverage(movies, links, tmdb, texts) -> dict:
    tm = movies[["movieId"]].merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    has_id = tm["tmdbId"].notna()
    info = tm.loc[has_id, "tmdbId"].astype(int)
    cached = info.isin(tmdb.index)
    ok = info[cached].map(lambda i: not tmdb.loc[i, "not_found"])
    ov = info[cached].map(lambda i: len(tmdb.loc[i, "overview"].strip()))
    fb = texts[texts["source"] == "fallback"]
    return {"movies": len(tm), "no_tmdb_id": int((~has_id).sum()), "tmdb_id_not_cached_yet": int((~cached).sum()),
            "tmdb_404": int((~ok).sum()), "overview_ok": int((texts["source"] == "overview").sum()),
            "overview_empty": int((ov == 0).sum()), "overview_very_short": int(((ov > 0) & (ov < MIN_OVERVIEW_CHARS)).sum()),
            "use_fallback": len(fb), "fallback_with_keywords": int(fb["has_keywords"].sum()),
            "fallback_with_tags": int(fb["has_tags"].sum()),
            "fallback_title_genres_only": int((~fb["has_keywords"] & ~fb["has_tags"]).sum()),
            "with_release_date": int(release_dates(movies, links, tmdb).notna().sum())}


def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def pca_reduce(emb: np.ndarray, dim: int = 64, seed: int = 0) -> np.ndarray:
    """Unsupervised PCA on item content only; rows re-normalised to unit length."""
    from sklearn.decomposition import PCA
    z = PCA(n_components=dim, random_state=seed).fit_transform(emb).astype(np.float32)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


def save_embeddings(path: Path, item_ids, emb384, emb64) -> None:
    np.savez(path, item_ids=np.asarray(item_ids), emb384=emb384, emb64=emb64)


def load_embeddings(path: Path, catalog: np.ndarray, which: str = "emb384") -> np.ndarray:
    z = np.load(path)
    pos = pd.Index(z["item_ids"]).get_indexer(catalog)
    assert (pos >= 0).all(), "catalog has movies without embeddings"
    return z[which][pos]


def neighbours(emb: np.ndarray, ids: np.ndarray, titles: dict, query_id: int, n: int = 5) -> list[str]:
    i = int(np.where(ids == query_id)[0][0])
    sims = emb @ emb[i]
    sims[i] = -np.inf
    return [f"{titles[ids[j]]} ({sims[j]:.2f})" for j in np.argsort(-sims)[:n]]
