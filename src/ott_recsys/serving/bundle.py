"""The artifacts shipped inside the Lambda zip, loaded once per container (built offline by scripts/build_serving_artifacts.py).

items.npz    catalog (movie ids, sorted), V (two-tower item vectors, 64-d), emb_c (centred text embeddings, 384-d), genre_m, genre_names, rel_ts,
             known (item had >= 1 train like: the frozen ID models know it), pop_ts/pop_start (sorted like timestamps per item, for 'popularity as of
             the request time'), pop_order (items by history popularity, for users without a vector)
ncf.npz      the NeuMF weights (item side + MLP), item_ids
ranker_*.npz numpy tree ensembles (M2 real model; M3b synthetic-device demo model)
titles.json  movie titles in catalog order"""
import json
from pathlib import Path

import numpy as np

from .ncf_np import NCFForward
from .trees import TreeEnsemble

DEVICES = ("tv", "mobile", "web", "tablet")  # same order as ott_recsys.context.DEVICES (tests assert it)


class Bundle:
    def __init__(self, d: Path):
        d = Path(d)
        z = np.load(d / "items.npz")
        self.catalog = z["catalog"]
        self.V, self.emb, self.genre_m, self.rel_ts, self.known = z["V"], z["emb_c"], z["genre_m"].astype(np.float32), z["rel_ts"], z["known"]
        self.genre_names = [str(x) for x in z["genre_names"]]
        self.pop_ts, self.pop_start, self.pop_order = z["pop_ts"], z["pop_start"], z["pop_order"]
        n = np.load(d / "ncf.npz")
        self.ncf = NCFForward({k: n[k] for k in n.files if k != "item_ids"}, n["item_ids"], self.catalog)
        self.rankers = {p.stem.replace("ranker_", ""): TreeEnsemble.load_npz(p) for p in sorted(d.glob("ranker_*.npz"))}
        self.titles = json.loads((d / "titles.json").read_text())

    def cat_index(self, movie_ids) -> np.ndarray:
        """Movie ids -> catalog indices (-1 if absent)."""
        movie_ids = np.asarray(movie_ids, dtype=np.int64)
        pos = np.searchsorted(self.catalog, movie_ids)
        pos = np.minimum(pos, len(self.catalog) - 1)
        return np.where(self.catalog[pos] == movie_ids, pos, -1)
