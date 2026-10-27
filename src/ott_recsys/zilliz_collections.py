"""Checkpoint 3 (Module 9): the Zilliz collections built from the Module 4b exports. OFFLINE code only (never runs in Lambda).

The ONLY collection names this project may create, write, search or drop are in NAMES. Index: AUTOINDEX with metric IP on every vector field (Zilliz Cloud
manages the index; no ef). Secrets: the URI and tokens are read from the environment (`.env` is loaded inside Python by load_env), handed straight to the client,
and never printed, logged or put in an exception message. quiet() swallows whatever the client library prints."""
import contextlib
import io
import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TT, CONTENT, PROBE = "movie_items_twotower_id", "movie_items_content", "ott_write_probe"
NAMES = (TT, CONTENT, PROBE)  # load order: most important first, scratch probe last
SPEC = {
    TT: {"dir": "two_tower/export_id_only", "vectors": {"vector": 64}, "rows": 16450},
    CONTENT: {"dir": "content/export_content", "vectors": {"vec384": 384, "vec64": 64}, "rows": 33392},
    PROBE: {"dir": None, "vectors": {"vector": 2}, "rows": 1},
}
SCALARS = (("title", "VARCHAR", 256), ("year", "INT32", None), ("genres", "VARCHAR", 128), ("n_train_likes", "INT32", None), ("is_cold", "BOOL", None))
PROBE_ROW = {"item_id": 1, "vector": [1.0, 0.0]}  # the one row of the scratch collection
WRITE_TEST_ROW = {"item_id": -1, "vector": [0.0, 1.0]}  # what the read-only key must NOT be able to insert


def load_env() -> None:
    """Load .env into the process environment (never overriding what is already set). OTT_SKIP_DOTENV=1 skips it (tests)."""
    if os.environ.get("OTT_SKIP_DOTENV") == "1":
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")


@contextlib.contextmanager
def quiet():
    """Swallow library stdout/stderr, logging and warnings for the duration (messages can contain the endpoint)."""
    sink = io.StringIO()
    logging.disable(logging.CRITICAL)
    with warnings.catch_warnings(), contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        warnings.simplefilter("ignore")
        try:
            yield
        finally:
            logging.disable(logging.NOTSET)


def default_client_factory(endpoint: str, key: str):
    from pymilvus import MilvusClient
    return MilvusClient(uri=endpoint, token=key, timeout=30)


# ------------------------------------------------------------------------------------------------ exports
def read_export(out_dir: Path, name: str) -> dict:
    """Columns of one collection, read from the Module 4b export and checked against its milvus_schema.json. Raises AssertionError on any mismatch."""
    spec = SPEC[name]
    if name == PROBE:
        return {"item_id": np.array([PROBE_ROW["item_id"]]), "vector": np.array([PROBE_ROW["vector"]], np.float32)}
    import pyarrow.parquet as pq
    d = Path(out_dir) / spec["dir"]
    schema = json.loads((d / "milvus_schema.json").read_text())
    assert schema["collection"] == name, f"{name}: export is for {schema['collection']}"
    fields = {f["name"]: f for f in schema["fields"]}
    for vf, dim in spec["vectors"].items():
        assert fields[vf]["type"] == "FLOAT_VECTOR" and fields[vf]["dim"] == dim, f"{name}.{vf}: schema dim"
    for sf, typ, max_len in SCALARS:
        assert fields[sf]["type"] == typ and (max_len is None or fields[sf]["max_length"] == max_len), f"{name}.{sf}: schema"
    t = pq.read_table(d / "items.parquet")
    cols = {"item_id": t.column("item_id").to_numpy()}
    for sf, _, _ in SCALARS:
        cols[sf] = t.column(sf).to_numpy(zero_copy_only=False)
    ids = np.load(d / "item_ids.npy")
    assert np.array_equal(ids, cols["item_id"]), f"{name}: item_ids.npy differs from items.parquet"
    assert len(ids) == spec["rows"] and len(np.unique(ids)) == len(ids), f"{name}: {len(ids)} rows (expected {spec['rows']}, unique keys)"
    for vf, dim in spec["vectors"].items():
        v = np.load(d / f"{vf}.npy")
        assert v.shape == (spec["rows"], dim) and v.dtype == np.float32, f"{name}.{vf}: shape {v.shape}"
        flat = t.column(vf).combine_chunks().flatten().to_numpy().reshape(-1, dim)
        assert np.array_equal(flat, v), f"{name}.{vf}: parquet vectors differ from {vf}.npy"
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4), f"{name}.{vf}: not unit length"
        cols[vf] = v
    for sf, typ, max_len in SCALARS:
        if typ == "VARCHAR":
            assert max(len(s) for s in cols[sf]) <= max_len, f"{name}.{sf}: string too long"
    return cols


def rows(cols: dict, a: int, b: int) -> list:
    """Rows [a, b) as plain Python values (what the client sends)."""
    out = []
    for i in range(a, b):
        r = {}
        for k, v in cols.items():
            x = v[i]
            r[k] = x.tolist() if isinstance(x, np.ndarray) else (x.item() if hasattr(x, "item") else x)
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------ schema + index
def create(client, name: str) -> None:
    """Create one collection with its schema and an AUTOINDEX (metric IP) on every vector field."""
    from pymilvus import DataType
    spec = SPEC[name]
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="item_id", datatype=DataType.INT64, is_primary=True)
    for vf, dim in spec["vectors"].items():
        schema.add_field(field_name=vf, datatype=DataType.FLOAT_VECTOR, dim=dim)
    if name != PROBE:
        for sf, typ, max_len in SCALARS:
            kw = {"max_length": max_len} if max_len else {}
            schema.add_field(field_name=sf, datatype=getattr(DataType, typ), **kw)
    index = client.prepare_index_params()
    for vf in spec["vectors"]:
        index.add_index(field_name=vf, index_type="AUTOINDEX", metric_type="IP")
    client.create_collection(collection_name=name, schema=schema, index_params=index)


def row_count(client, name: str) -> int:
    res = client.query(collection_name=name, filter="", output_fields=["count(*)"])
    return int(res[0]["count(*)"])


def error_code(exc) -> str:
    """A short, safe description of a client error: the class name plus a numeric code if the error carries one (never the message)."""
    code = getattr(exc, "code", None)
    return f"{type(exc).__name__}" + (f" code {int(code)}" if isinstance(code, int) else "")


PERMISSION_WORDS = ("permission", "privilege", "denied", "forbidden", "unauthorized", "not authorized", "no access", "access deny", "rbac")


def looks_like_permission_error(text: str) -> bool:
    """Inspect an error text WITHOUT printing it: does it say the caller lacks the right?"""
    t = str(text).lower()
    return any(w in t for w in PERMISSION_WORDS)
