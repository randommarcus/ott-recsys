"""Export two-tower artefacts in a Milvus-ready layout (no Zilliz connection here).

item_vectors.parquet : item_id (INT64 primary key), vector (FLOAT_VECTOR 64), metadata
milvus_schema.json   : collection fields and index settings (vectors are unit length, so
                       inner product == cosine similarity)
user_tower.npz       : dense-layer weights; with item_vectors a user vector needs only numpy
user_vectors.parquet : precomputed user vectors for offline use"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _vector_column(vecs: np.ndarray) -> pa.Array:
    return pa.array(vecs.astype(np.float32).tolist(), type=pa.list_(pa.float32()))


def export_milvus(out_dir: Path, catalog, item_vecs, movies: pd.DataFrame, train_like_counts, W, b,
                  history_len: int, user_ids, user_vecs, collection: str = "movie_items") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dim = item_vecs.shape[1]
    assert item_vecs.shape[0] == len(catalog) and np.allclose(np.linalg.norm(item_vecs, axis=1), 1, atol=1e-4)
    meta = movies.set_index("movieId").reindex(catalog)
    year = pd.to_numeric(meta["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce").fillna(0).astype("int32")
    table = pa.table({
        "item_id": pa.array(np.asarray(catalog, dtype=np.int64)),
        "vector": _vector_column(item_vecs),
        "title": pa.array(meta["title"].str.slice(0, 256).tolist()),
        "year": pa.array(year.to_numpy()),
        "genres": pa.array(meta["genres"].str.slice(0, 128).tolist()),
        "n_train_likes": pa.array(np.asarray(train_like_counts, dtype=np.int32)),
        "is_cold": pa.array(np.asarray(train_like_counts) == 0)})
    pq.write_table(table, out_dir / "item_vectors.parquet")
    np.save(out_dir / "item_vectors.npy", item_vecs.astype(np.float32))
    pq.write_table(pa.table({"user_id": pa.array(np.asarray(user_ids, dtype=np.int64)),
                             "vector": _vector_column(user_vecs)}), out_dir / "user_vectors.parquet")
    np.savez(out_dir / "user_tower.npz", W=W, b=b, history_len=history_len)
    schema = {"collection": collection,
              "fields": [{"name": "item_id", "type": "INT64", "is_primary": True, "auto_id": False},
                         {"name": "vector", "type": "FLOAT_VECTOR", "dim": dim},
                         {"name": "title", "type": "VARCHAR", "max_length": 256},
                         {"name": "year", "type": "INT32"},
                         {"name": "genres", "type": "VARCHAR", "max_length": 128},
                         {"name": "n_train_likes", "type": "INT32"},
                         {"name": "is_cold", "type": "BOOL"}],
              "index": {"field": "vector", "index_type": "HNSW", "metric_type": "IP",
                        "params": {"M": 16, "efConstruction": 200}},
              "search_params": {"metric_type": "IP", "params": {"ef": 64}},
              "notes": "Vectors are L2-normalised, so IP equals cosine. Remove items the user already saw "
                       "at query time with a filter on item_id."}
    (out_dir / "milvus_schema.json").write_text(json.dumps(schema, indent=2))
    return {"items": len(catalog), "dim": dim, "users": len(user_ids), "cold_items": int((np.asarray(train_like_counts) == 0).sum())}


def export_collection(out_dir: Path, collection: str, catalog, vectors: dict, movies: pd.DataFrame,
                      train_like_counts, include=None, notes: str = "") -> dict:
    """Milvus-ready collection with one or more unit-length vector fields (name -> [n_items, dim]).
    `include` (bool mask over the catalog) drops rows, e.g. items whose vector carries no information."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog, counts = np.asarray(catalog), np.asarray(train_like_counts)
    keep = np.ones(len(catalog), dtype=bool) if include is None else np.asarray(include, dtype=bool)
    meta = movies.set_index("movieId").reindex(catalog[keep])
    year = pd.to_numeric(meta["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce").fillna(0).astype("int32")
    cols = {"item_id": pa.array(catalog[keep].astype(np.int64))}
    fields = [{"name": "item_id", "type": "INT64", "is_primary": True, "auto_id": False}]
    indexes = []
    for name, v in vectors.items():
        assert v.shape[0] == len(catalog) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4), f"{name}: not unit length"
        cols[name] = _vector_column(v[keep])
        np.save(out_dir / f"{name}.npy", v[keep].astype(np.float32))
        fields.append({"name": name, "type": "FLOAT_VECTOR", "dim": int(v.shape[1])})
        indexes.append({"field": name, "index_type": "HNSW", "metric_type": "IP", "params": {"M": 16, "efConstruction": 200}})
    cols.update({"title": pa.array(meta["title"].str.slice(0, 256).tolist()), "year": pa.array(year.to_numpy()),
                 "genres": pa.array(meta["genres"].str.slice(0, 128).tolist()),
                 "n_train_likes": pa.array(counts[keep].astype(np.int32)), "is_cold": pa.array(counts[keep] == 0)})
    fields += [{"name": "title", "type": "VARCHAR", "max_length": 256}, {"name": "year", "type": "INT32"},
               {"name": "genres", "type": "VARCHAR", "max_length": 128}, {"name": "n_train_likes", "type": "INT32"},
               {"name": "is_cold", "type": "BOOL"}]
    pq.write_table(pa.table(cols), out_dir / "items.parquet")
    np.save(out_dir / "item_ids.npy", catalog[keep].astype(np.int64))
    (out_dir / "milvus_schema.json").write_text(json.dumps({
        "collection": collection, "fields": fields, "indexes": indexes,
        "search_params": {"metric_type": "IP", "params": {"ef": 64}}, "notes": notes}, indent=2))
    return {"collection": collection, "items": int(keep.sum()), "dropped": int((~keep).sum()),
            "vector_fields": {k: int(v.shape[1]) for k, v in vectors.items()}, "cold_rows": int((counts[keep] == 0).sum())}


def export_user_vectors(path: Path, user_ids, vecs: dict) -> None:
    """Precomputed user vectors (one column per vector name)."""
    cols = {"user_id": pa.array(np.asarray(user_ids, dtype=np.int64))}
    cols.update({k: _vector_column(v) for k, v in vecs.items()})
    pq.write_table(pa.table(cols), path)
