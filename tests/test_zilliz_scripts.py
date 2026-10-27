"""Checkpoint 3 scripts (load_zilliz, verify_zilliz, zilliz_teardown) against an in-memory fake Zilliz. No network, no .env: secrets are canaries.
Real-export tests are skipped when the Module 4b exports are not on disk."""
import io
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pymilvus  # noqa: F401  Imported at COLLECTION time on purpose: importing pymilvus runs logging.config.dictConfig, which closes every
#                 existing log handler. If TensorFlow (tests/test_environment.py) has already attached absl's handler to pytest's captured stderr,
#                 a later first import of pymilvus would close pytest's stream and crash the run. The scripts never import TensorFlow (tested below).
import pytest

from ott_recsys import zilliz_collections as zc

from .aws_fakes import load_script
from .serving_real import OUT

ROOT = Path(__file__).resolve().parents[1]
loader, verifier, teardown = load_script("load_zilliz"), load_script("verify_zilliz"), load_script("zilliz_teardown")
HAVE_EXPORTS = (OUT / "two_tower" / "export_id_only" / "items.parquet").exists() and (OUT / "content" / "export_content" / "items.parquet").exists()
real = pytest.mark.skipif(not HAVE_EXPORTS, reason="Module 4b exports not on disk")
CANARY_URI, FULL, RO = "https://CANARY-HOST-7f3a9c.invalid:19530", "CANARY-FULL-5b6e1d", "CANARY-RO-91c2d4e8"
SECRET_BITS = (CANARY_URI, FULL, RO, "7f3a9c", "5b6e1d", "91c2d4e8")


def clean(text):
    return [s for s in SECRET_BITS if s in text]


class FakeError(Exception):
    def __init__(self, message, code=1100):
        super().__init__(message)
        self.code, self.message = code, message


class Store:
    """The cluster: collections -> {fields, vectors, index, rows, loaded}. `ro_can_write` simulates a wrongly-scoped key."""

    def __init__(self):
        self.cols, self.calls = {}, []
        self.ro_can_write, self.ro_can_search = False, True  # a proper read-only key: can search, cannot write
        self.fail_upserts = 0
        self.insert_error = None

    def search(self, name, field, vec, limit, flt=""):
        c = self.cols[name]
        ids = [i for i, r in c["rows"].items() if not (flt and "is_cold == true" in flt and not r.get("is_cold"))]
        q = np.asarray(vec, np.float32)
        scores = [float(np.dot(q, np.asarray(c["rows"][i][field], np.float32))) for i in ids]
        order = np.argsort(-np.asarray(scores), kind="stable")[:limit]
        return [(ids[j], scores[j]) for j in order]


class Schema:
    def __init__(self): self.fields = []
    def add_field(self, **kw): self.fields.append(kw)


class Index:
    def __init__(self): self.items = []
    def add_index(self, **kw): self.items.append(kw)


class Client:
    def __init__(self, store, endpoint, key):
        self.s, self.role = store, ("ro" if key == RO else "full" if key == FULL else "bad")
        if endpoint != CANARY_URI or self.role == "bad":
            raise FakeError(f"cannot connect to {endpoint} with {key}", code=2)

    def _write(self, what):
        self.s.calls.append(what)
        if self.role == "ro" and not self.s.ro_can_write:
            raise FakeError(f"PERMISSION DENIED: privilege Insert for key {RO} at {CANARY_URI}", code=65535)

    def list_collections(self): return list(self.s.cols)
    def has_collection(self, collection_name): return collection_name in self.s.cols
    def create_schema(self, **kw): return Schema()
    def prepare_index_params(self): return Index()

    def create_collection(self, collection_name, schema, index_params):
        self._write(("create", collection_name))
        self.s.cols[collection_name] = {"fields": schema.fields, "index": index_params.items, "rows": {}, "loaded": False}

    def upsert(self, collection_name, data):
        self._write(("upsert", collection_name))
        if self.s.fail_upserts:
            self.s.fail_upserts -= 1
            raise FakeError("rate limit exceeded 2MB/s", code=8)
        for r in data:
            assert all(not isinstance(v, np.generic) for v in r.values()), "rows must be plain Python values"
            self.s.cols[collection_name]["rows"][r["item_id"]] = r

    def insert(self, collection_name, data):
        if self.s.insert_error:
            raise self.s.insert_error
        self._write(("insert", collection_name))
        for r in data:
            self.s.cols[collection_name]["rows"][r["item_id"]] = r

    def flush(self, collection_name): pass
    def load_collection(self, collection_name): self.s.cols[collection_name]["loaded"] = True
    def get_load_state(self, collection_name): return {"state": "<LoadState: Loaded>" if self.s.cols[collection_name]["loaded"] else "<LoadState: NotLoad>"}

    def drop_collection(self, collection_name):
        self._write(("drop", collection_name))
        del self.s.cols[collection_name]

    def query(self, collection_name, filter, output_fields):
        return [{"count(*)": len(self.s.cols[collection_name]["rows"])}]

    def search(self, collection_name, data, anns_field, limit, filter, search_params, output_fields):
        if self.role == "ro" and not self.s.ro_can_search:
            raise FakeError(f"permission denied: privilege Search ({RO})", code=65535)
        assert search_params == {"metric_type": "IP"}
        return [[{"id": i, "distance": d, "entity": {"item_id": i}} for i, d in self.s.search(collection_name, anns_field, data[0], limit, filter)]]


def factory(store):
    return lambda endpoint, key: Client(store, endpoint, key)


class _Resp:
    def __init__(self, body): self.body = body
    def read(self): return self.body
    def __enter__(self): return self
    def __exit__(self, *a): return False


def rest_fake(store):
    """urllib.request.urlopen replacement serving the REST v2 search/insert endpoints from the same store."""
    def urlopen(req, timeout=None):
        assert req.full_url.startswith(CANARY_URI)
        key = req.get_header("Authorization").split(" ", 1)[1]
        body = json.loads(req.data)
        if req.full_url.endswith("/entities/search"):
            if key == RO and not store.ro_can_search:
                return _Resp(json.dumps({"code": 65535, "message": f"permission denied {RO}"}).encode())
            assert body["searchParams"] == {"metricType": "IP"}
            hits = store.search(body["collectionName"], body["annsField"], body["data"][0], body["limit"], body.get("filter", ""))
            return _Resp(json.dumps({"code": 0, "data": [{"item_id": i, "distance": d} for i, d in hits]}).encode())
        if req.full_url.endswith("/entities/insert"):
            store.calls.append(("rest-insert", body["collectionName"]))
            if key == RO and not store.ro_can_write:
                return _Resp(json.dumps({"code": 65535, "message": f"permission deny: privilege Insert {RO} {CANARY_URI}"}).encode())
            for r in body["data"]:
                store.cols[body["collectionName"]]["rows"][r["item_id"]] = r
            return _Resp(json.dumps({"code": 0, "data": {"insertCount": 1}}).encode())
        raise AssertionError(req.full_url)
    return urlopen


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OTT_SKIP_DOTENV", "1")
    monkeypatch.setenv("ZILLIZ_URI", CANARY_URI)
    monkeypatch.setenv("ZILLIZ_TOKEN", FULL)
    monkeypatch.setenv("ZILLIZ_TOKEN_READONLY", RO)


# ------------------------------------------------------------------------------------------------ module
@real
def test_export_integrity_and_plain_rows():
    for name in zc.NAMES:
        cols = zc.read_export(OUT, name)
        assert len(cols["item_id"]) == zc.SPEC[name]["rows"]
        row = zc.rows(cols, 0, 1)[0]
        assert all(not isinstance(v, np.generic) for v in row.values())


def test_pymilvus_logging_is_silenced_and_never_bound_to_a_stream():
    import logging
    zc.silence_pymilvus_logging()
    for name, lg in logging.Logger.manager.loggerDict.items():
        if name.startswith("pymilvus") and isinstance(lg, logging.Logger):
            assert lg.propagate is False and all(isinstance(h, logging.NullHandler) for h in lg.handlers), name


def test_zilliz_scripts_never_import_tensorflow():
    """pymilvus' import closes all existing log handlers; with TensorFlow's absl handler on the real stderr that would close stderr itself."""
    code = ("import importlib.util, sys\n"
            "for p in ('scripts/load_zilliz.py', 'scripts/verify_zilliz.py', 'scripts/zilliz_teardown.py'):\n"
            "    spec = importlib.util.spec_from_file_location('m', p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "print('tensorflow' in sys.modules, 'absl.logging' in sys.modules)\n")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "False False", r.stdout + r.stderr[-300:]


def test_permission_word_detection():
    assert zc.looks_like_permission_error("PERMISSION DENIED: privilege Insert")
    assert zc.looks_like_permission_error("access deny") and not zc.looks_like_permission_error("rate limit exceeded")
    assert zc.error_code(FakeError("secret text", code=65535)) == "FakeError code 65535"


# ------------------------------------------------------------------------------------------------ load_zilliz
@real
def test_load_dry_run_writes_nothing(env, capsys):
    s = Store()
    assert loader.main([], client_factory=factory(s)) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and out.count("integrity: ok") == 3 and s.calls == [] and not clean(out)


@real
def test_load_creates_exactly_three_collections_with_autoindex_ip(env, capsys):
    s = Store()
    s.cols["someone_elses_collection"] = {"rows": {}, "loaded": True}
    assert loader.main(["--confirm", "LOAD-ZILLIZ"], client_factory=factory(s), sleep=lambda x: None) == 0
    out = capsys.readouterr().out
    assert "load complete" in out and not clean(out)
    assert set(s.cols) == set(zc.NAMES) | {"someone_elses_collection"}
    for name in zc.NAMES:
        c = s.cols[name]
        assert len(c["rows"]) == zc.SPEC[name]["rows"] and c["loaded"]
        assert {i["field_name"] for i in c["index"]} == set(zc.SPEC[name]["vectors"])
        assert all(i["index_type"] == "AUTOINDEX" and i["metric_type"] == "IP" for i in c["index"])
        dims = {f["field_name"]: f.get("dim") for f in c["fields"] if "dim" in f}
        assert dims == zc.SPEC[name]["vectors"]
    lens = {f["field_name"]: f.get("max_length") for f in s.cols[zc.CONTENT]["fields"]}
    assert lens["title"] == 256 and lens["genres"] == 128
    assert all(c[1] in zc.NAMES for c in s.calls), "only our three names may ever be written"


@real
def test_load_refuses_existing_collections_unless_recreate(env, capsys):
    s = Store()
    loader.main(["--confirm", "LOAD-ZILLIZ"], client_factory=factory(s), sleep=lambda x: None)
    s.cols["someone_elses_collection"] = {"rows": {}, "loaded": True}
    s.calls.clear()
    assert loader.main(["--confirm", "LOAD-ZILLIZ"], client_factory=factory(s)) == 1
    assert "refused" in capsys.readouterr().out and s.calls == []
    assert loader.main(["--confirm", "LOAD-ZILLIZ", "--recreate"], client_factory=factory(s), sleep=lambda x: None) == 0
    assert [c for c in s.calls if c[0] == "drop"] == [("drop", n) for n in zc.NAMES]
    assert "someone_elses_collection" in s.cols


@real
def test_load_retries_throttled_batches_and_reports_failures_by_type_only(env, capsys):
    s = Store()
    s.fail_upserts = 2
    waits = []
    assert loader.main(["--confirm", "LOAD-ZILLIZ"], client_factory=factory(s), sleep=waits.append) == 0
    assert waits[:2] == [1, 2]
    s2 = Store()
    s2.fail_upserts = 10 ** 6
    assert loader.main(["--confirm", "LOAD-ZILLIZ"], client_factory=factory(s2), sleep=lambda x: None) == 1
    out = capsys.readouterr().out
    assert "FAILED (error type: FakeError code 8)" in out and "rate limit" not in out and not clean(out)


def test_load_needs_credentials_and_the_exact_word(monkeypatch, capsys):
    monkeypatch.setenv("OTT_SKIP_DOTENV", "1")
    monkeypatch.delenv("ZILLIZ_URI", raising=False)
    monkeypatch.delenv("ZILLIZ_TOKEN", raising=False)
    assert loader.main(["--confirm", "yes"]) == 2
    monkeypatch.setattr(loader, "read_export", lambda d, n: {"item_id": np.arange(zc.SPEC[n]["rows"])})
    assert loader.main([], client_factory=lambda *a: pytest.fail("must not connect")) == 1
    assert "must be set" in capsys.readouterr().out


def test_load_stops_on_a_failed_integrity_check_before_connecting(env, monkeypatch, capsys):
    def bad(d, n): raise AssertionError(f"{n}: item_ids.npy differs from items.parquet")
    monkeypatch.setattr(loader, "read_export", bad)
    assert loader.main(["--confirm", "LOAD-ZILLIZ"], client_factory=lambda *a: pytest.fail("must not connect")) == 1
    assert "integrity: FAILED" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ zilliz_teardown
def test_teardown_dry_run_and_drop_touch_only_our_three_names(env, capsys):
    s = Store()
    for n in (*zc.NAMES, "someone_elses_collection"):
        s.cols[n] = {"rows": {}, "loaded": True}
    assert teardown.main([], client_factory=factory(s)) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "existing: 3" in out and s.calls == [] and "someone_elses" not in out
    assert teardown.main(["--confirm", "DROP-ZILLIZ"], client_factory=factory(s)) == 0
    out = capsys.readouterr().out
    assert "remaining: 0" in out and set(s.cols) == {"someone_elses_collection"} and not clean(out)
    assert {c[1] for c in s.calls} == set(zc.NAMES)


def test_teardown_wrong_word_and_unverifiable_checks(env, capsys):
    s = Store()
    assert teardown.main(["--confirm", "drop"], client_factory=factory(s)) == 2

    class Blind(Client):
        def has_collection(self, collection_name): raise FakeError(f"denied {FULL}", code=7)
    assert teardown.main([], client_factory=lambda e, k: Blind(s, e, k)) == 1
    out = capsys.readouterr().out
    assert "unverified: 3" in out and not clean(out)


def test_teardown_only_ever_uses_literal_names():
    src = (ROOT / "scripts" / "zilliz_teardown.py").read_text()
    assert "list_collections" not in src, "teardown must never decide what to drop from a listing"


# ------------------------------------------------------------------------------------------------ verify_zilliz
def loaded_store():
    s = Store()
    assert loader.main(["--confirm", "LOAD-ZILLIZ"], client_factory=factory(s), sleep=lambda x: None) == 0
    return s


@real
def test_verify_passes_with_a_proper_read_only_key(env, monkeypatch, capsys):
    s = loaded_store()
    capsys.readouterr()
    monkeypatch.setattr(urllib.request, "urlopen", rest_fake(s))
    assert verifier.main([], client_factory=factory(s)) == 0
    out = capsys.readouterr().out
    assert "A1 search (pymilvus): two-tower 10 results, query item first: yes; lane 30 results, all cold: yes; vec64 5 results  -> PASS" in out
    assert "(10/10 same as A1)" in out and "(30/30 same as A1)" in out
    assert "pymilvus REFUSED (permission); REST REFUSED (permission, code 65535); probe unchanged: yes  -> PASS" in out
    assert "verdict: PASS" in out and not clean(out) and "privilege Insert" not in out
    assert len(s.cols[zc.PROBE]["rows"]) == 1


@real
def test_verify_fails_if_the_key_can_write(env, monkeypatch, capsys):
    s = loaded_store()
    s.ro_can_write = True
    monkeypatch.setattr(urllib.request, "urlopen", rest_fake(s))
    assert verifier.main([], client_factory=factory(s)) == 1
    out = capsys.readouterr().out
    assert "ALLOWED (UNEXPECTED" in out and "probe unchanged: NO" in out and "verdict: FAIL" in out


@real
def test_verify_treats_unrelated_errors_as_inconclusive(env, monkeypatch, capsys):
    s = loaded_store()
    s.insert_error = ConnectionError(f"reset by peer {CANARY_URI}")
    monkeypatch.setattr(urllib.request, "urlopen", rest_fake(s))
    assert verifier.main([], client_factory=factory(s)) == 1
    out = capsys.readouterr().out
    assert "pymilvus INCONCLUSIVE (error type: ConnectionError)" in out and not clean(out)


@real
def test_verify_reports_a_key_without_search_rights(env, monkeypatch, capsys):
    s = loaded_store()
    s.ro_can_search = False
    monkeypatch.setattr(urllib.request, "urlopen", rest_fake(s))
    assert verifier.main([], client_factory=factory(s)) == 1
    out = capsys.readouterr().out
    assert "A1 search (pymilvus): FAIL (error type: FakeError code 65535); the key may not have search rights" in out
    assert "A2 search (Lambda REST client): FAIL (zilliz-code-65535)" in out and not clean(out)


@pytest.mark.parametrize("ro_value", ["", FULL])
def test_verify_refuses_a_blank_or_full_access_key(env, monkeypatch, capsys, ro_value):
    monkeypatch.setenv("ZILLIZ_TOKEN_READONLY", ro_value)
    assert verifier.main([], client_factory=lambda *a: pytest.fail("must not connect")) == 1
    out = capsys.readouterr().out
    assert out.startswith("refused") and not clean(out)


def test_rest_write_attempt_classification(monkeypatch):
    def http(code):
        def f(req, timeout=None): raise urllib.error.HTTPError(req.full_url, code, f"no {RO}", {}, io.BytesIO(b""))
        return f
    monkeypatch.setattr(urllib.request, "urlopen", http(403))
    assert verifier.write_attempt_rest(CANARY_URI, RO) == "REFUSED (permission, HTTP 403)"
    monkeypatch.setattr(urllib.request, "urlopen", http(401))
    assert verifier.write_attempt_rest(CANARY_URI, RO) == "INCONCLUSIVE (HTTP 401)"
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(b'{"code": 1100, "message": "invalid parameter"}'))
    assert verifier.write_attempt_rest(CANARY_URI, RO) == "INCONCLUSIVE (code 1100)"


@pytest.mark.skipif(not (HAVE_EXPORTS and (OUT / "serving" / "users.npz").exists()), reason="serving artifacts not built")
def test_verify_recall_check_with_exact_search_matches_the_precomputed_lists(env, monkeypatch, capsys):
    s = loaded_store()
    monkeypatch.setattr(urllib.request, "urlopen", rest_fake(s))
    assert verifier.main(["--recall"], client_factory=factory(s)) == 0
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if l.startswith("C recall")][0]
    main_pct = float(line.split("main overlap ")[1].split("%")[0])
    assert main_pct >= 99.0, line  # the fake search is exact, like the earlier numpy check


# ------------------------------------------------------------------------------------------------ canary subprocesses (real pymilvus, unreachable host)
@pytest.mark.parametrize("script,args", [("load_zilliz.py", []), ("verify_zilliz.py", []), ("zilliz_teardown.py", [])])
def test_real_scripts_never_print_secrets_when_the_connection_fails(script, args):
    if script in ("load_zilliz.py", "verify_zilliz.py") and not HAVE_EXPORTS:
        pytest.skip("exports needed")
    e = {k: v for k, v in os.environ.items() if not k.startswith("ZILLIZ_")}
    e.update({"OTT_SKIP_DOTENV": "1", "ZILLIZ_URI": "http://canary-host-7f3a9c.invalid:19530", "ZILLIZ_TOKEN": FULL, "ZILLIZ_TOKEN_READONLY": RO})
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / script), *args], capture_output=True, text=True, env=e, cwd=ROOT, timeout=240)
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "error type" in out, out[-500:]
    for bad in ("canary-host", "7f3a9c", FULL, RO, "5b6e1d", "91c2d4e8", "Traceback"):
        assert bad not in out, bad
