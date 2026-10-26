"""Live retrieval (serving mode B): two Zilliz searches per request, then the same list layout as the precomputed lists (200 main + 30 lane).

* main lane: collection movie_items_twotower_id (warm items only), query = the user's two-tower vector, inner product.
  Zilliz cannot exclude 'already seen' cheaply, so we ask for 200 + (seen warm items) hits (max 1,024 per search on the free plan) and drop seen items here.
  Users with so many seen warm items that 200 unseen ones do not fit in 1,024 hits get a SHORTER main lane (flagged 'truncated_main').
* new-release lane: collection movie_items_content, filter is_cold == true (no train like = the lane pool), query = the user's content profile (vec384).
* users without a two-tower vector get a popularity list from the bundle (no search); users without a content profile get no lane.

The REST call shape follows Zilliz's v2 REST API and is UNVERIFIED until the first real call (checkpoint 3). NumpySearch is an exact local stand-in used by the tests."""
import json
import urllib.request

import numpy as np

COLL_TT, FIELD_TT = "movie_items_twotower_id", "vector"
COLL_CONTENT, FIELD_CONTENT = "movie_items_content", "vec384"
MAX_LIMIT = 1024  # Zilliz free plan: topK <= 1,024
N_MAIN, N_LANE = 200, 30


class SearchError(Exception):
    """Message holds an exception type name or a numeric code only."""


class ZillizREST:
    def __init__(self, config, timeout: float = 3.0):
        self._cfg, self.timeout = config, timeout

    def search(self, collection: str, field: str, vec, limit: int, flt: str | None = None):
        body = {"collectionName": collection, "data": [np.asarray(vec, np.float32).tolist()], "annsField": field, "limit": int(limit),
                "outputFields": ["item_id"], "searchParams": {"metric_type": "IP", "params": {"ef": max(64, int(limit))}}}
        if flt:
            body["filter"] = flt
        req = urllib.request.Request(self._cfg.uri.rstrip("/") + "/v2/vectordb/entities/search", data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + self._cfg.token, "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.loads(r.read())
            code = payload.get("code", 0)
            if code not in (0, 200):
                raise SearchError(f"zilliz-code-{int(code)}")
            rows = payload["data"]
            return np.array([d.get("item_id", d.get("id")) for d in rows], dtype=np.int64), np.array([d["distance"] for d in rows], dtype=np.float32)
        except SearchError:
            raise
        except Exception as exc:
            raise SearchError(type(exc).__name__) from None


class NumpySearch:
    """Brute-force stand-in with the same interface (exact inner product over the same item sets the Zilliz collections hold)."""

    def __init__(self, b):
        self.b = b

    def search(self, collection: str, field: str, vec, limit: int, flt: str | None = None):
        b = self.b
        if collection == COLL_TT:
            idx = np.flatnonzero(b.known)
            scores = b.V[idx] @ np.asarray(vec, np.float32)
        else:
            idx = np.flatnonzero(~b.known) if flt and "is_cold == true" in flt else np.arange(len(b.catalog))
            scores = b.emb[idx].astype(np.float32) @ np.asarray(vec, np.float32)
        top = np.argsort(-scores, kind="stable")[:limit]
        return b.catalog[idx[top]].astype(np.int64), scores[top].astype(np.float32)


def live_candidates(b, rec: dict, searcher, n_main: int = N_MAIN, n_lane: int = N_LANE):
    """Build the 230-slot candidate array (catalog indices, -1 padded) for one user with live searches. Returns (cand, info)."""
    seen = rec["seen"] if rec.get("seen") is not None else np.empty(0, np.int32)
    info = {"searches": 0, "truncated_main": False, "popularity_fallback": False, "no_lane": False}
    if rec.get("ttvec") is not None:
        limit = min(MAX_LIMIT, n_main + int(b.known[seen].sum()))
        ids, _ = searcher.search(COLL_TT, FIELD_TT, rec["ttvec"], limit)
        info["searches"] += 1
        ci = b.cat_index(ids)
        main = ci[(ci >= 0) & ~np.isin(ci, seen)][:n_main]
        info["truncated_main"] = bool(len(main) < n_main and n_main + int(b.known[seen].sum()) > MAX_LIMIT)
    else:
        info["popularity_fallback"] = True
        po = b.pop_order
        main = po[~np.isin(po, seen)][:n_main]
    lane = np.empty(0, np.int64)
    if rec.get("prof") is not None:
        limit = min(MAX_LIMIT, n_lane + int((~b.known[seen]).sum()))
        ids, _ = searcher.search(COLL_CONTENT, FIELD_CONTENT, rec["prof"], limit, "is_cold == true")
        info["searches"] += 1
        ci = b.cat_index(ids)
        lane = ci[(ci >= 0) & ~np.isin(ci, seen) & ~np.isin(ci, main)][:n_lane]
    else:
        info["no_lane"] = True
    cand = np.full(n_main + n_lane, -1, np.int32)
    cand[:len(main)] = main
    cand[n_main:n_main + len(lane)] = lane
    return cand, info
