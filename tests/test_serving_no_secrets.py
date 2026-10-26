"""No log line, print, error message, response body or traceback of the serving code may contain the Zilliz URI or token.
Two layers: (1) a static check of the source (also self-tested on deliberately bad snippets); (2) a canary run of the real handler through every
error path with a fake URI and token that must never show up in stdout, stderr, log records, tracebacks or responses."""
import ast
import io
import json
import logging
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from ott_recsys.serving import handler as H
from ott_recsys.serving.search import SearchError, ZillizREST
from ott_recsys.serving.secrets import ConfigError, ZillizConfig, load_from_ssm

ROOT = Path(__file__).resolve().parents[1]
CANARY_URI, CANARY_TOKEN = "https://CANARY-URI-7f3a9c.invalid:19530", "CANARY-TOKEN-91c2d4e8"
LOG_CALLS = {"print", "write", "debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
SECRET_IDS = {"uri", "token", "secret", "password", "cfg", "config", "_cfg", "_uri", "_token"}
EXC_IDS = {"exc", "e", "err", "error", "ex"}


def violations(src: str) -> list:
    """Static rules. `type(exc).__name__` is the only way an exception may be mentioned in a message."""
    tree = ast.parse(src)
    out = []

    def bad_names(node):
        hidden = set()
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "type":
                hidden |= {id(n) for n in ast.walk(inner)}
        found = set()
        for n in ast.walk(node):
            if id(n) in hidden:
                continue
            if isinstance(n, ast.Name) and (n.id in SECRET_IDS or n.id in EXC_IDS):
                found.add(n.id)
            if isinstance(n, ast.Attribute) and n.attr in SECRET_IDS:
                found.add(n.attr)
        return found

    for n in ast.walk(tree):
        if isinstance(n, ast.Import) and any(a.name == "traceback" for a in n.names):
            out.append(f"line {n.lineno}: imports traceback")
        if isinstance(n, ast.Attribute) and n.attr in {"print_exc", "format_exc", "exception"}:
            out.append(f"line {n.lineno}: traceback / exception logging")
        if isinstance(n, ast.keyword) and n.arg == "exc_info":
            out.append(f"line {n.value.lineno}: exc_info")
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            if name in LOG_CALLS:
                for arg in list(n.args) + [k.value for k in n.keywords]:
                    if bad_names(arg):
                        out.append(f"line {n.lineno}: {name}(...) mentions {sorted(bad_names(arg))}")
            if name in {"str", "repr", "format"} and any(isinstance(a, ast.Name) and a.id in EXC_IDS for a in n.args):
                out.append(f"line {n.lineno}: exception text formatted with {name}()")
        if isinstance(n, ast.Raise) and n.exc is not None and bad_names(n.exc):
            out.append(f"line {n.lineno}: raise mentions {sorted(bad_names(n.exc))}")
        if isinstance(n, ast.JoinedStr):
            for v in n.values:
                if isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name) and v.value.id in EXC_IDS:
                    out.append(f"line {n.lineno}: exception interpolated into an f-string")
    return out


def _files():
    files = sorted((ROOT / "src" / "ott_recsys" / "serving").glob("*.py"))
    for pat in ("put_ssm_secrets.py", "aws_*.py", "deploy*.py", "demo*.py", "latency*.py", "load_dynamodb*.py", "zilliz_*.py", "build_lambda_zip.py",
                "export_dynamodb_items.py", "cloudshell/*.py"):
        files += sorted((ROOT / "scripts").glob(pat))
    return files


@pytest.mark.parametrize("path", _files(), ids=lambda p: p.name)
def test_source_has_no_secret_or_exception_text_in_output_paths(path):
    assert violations(path.read_text()) == []


@pytest.mark.parametrize("snippet", [
    "print(token)", "log.error('bad %s', exc)", "raise RuntimeError(f'cannot reach {uri}')", "logging.exception('x')",
    "import traceback", "traceback.format_exc()", "log.warning(str(exc))", "print(f'{err}')", "log.info('cfg=%s', self._cfg)",
    "log.error('x', exc_info=True)",
])
def test_the_checker_catches_bad_code(snippet):
    assert violations(snippet), snippet


def test_the_checker_allows_exception_type_names():
    assert violations("log.warning('failed: %s', type(exc).__name__)\nprint(type(e).__name__)") == []


def test_bad_request_messages_are_literals():
    tree = ast.parse((ROOT / "src" / "ott_recsys" / "serving" / "handler.py").read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call) and getattr(n.exc.func, "id", "") == "BadRequest":
            assert isinstance(n.exc.args[0], ast.Constant), f"line {n.lineno}: BadRequest message must be a literal"


def test_env_example_has_a_blank_read_only_token_placeholder():
    vals = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            vals[k] = v
    assert "ZILLIZ_TOKEN_READONLY" in vals
    assert all(v == "" for v in vals.values()), "every .env.example value must be blank"


# ---------------------------------------------------------------------------------------------- canary runs
def test_config_object_hides_its_values():
    c = ZillizConfig(CANARY_URI, CANARY_TOKEN)
    for text in (repr(c), str(c), f"{c}", "%s" % c, format(c, "")):
        assert CANARY_URI not in text and CANARY_TOKEN not in text
    with pytest.raises(TypeError):
        import pickle
        pickle.dumps(c)


class _Boom:
    """A fake SSM client whose error message contains the canaries (like a badly behaved SDK could)."""
    def __init__(self, exc): self.exc = exc
    def get_parameter(self, **kw): raise self.exc


class _Resp:
    def __init__(self, body): self.body = body
    def read(self): return self.body
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _leaks(*texts):
    # value fragments, not the word CANARY: tracebacks print source lines, and this file's own lines name CANARY_URI
    return [s for t in texts for s in (CANARY_URI, CANARY_TOKEN, "7f3a9c", "91c2d4e8") if s in str(t)]


@pytest.mark.parametrize("make_exc", [
    lambda: RuntimeError(f"cannot read {CANARY_URI} with {CANARY_TOKEN}"),
    lambda: KeyError(CANARY_TOKEN),
    lambda: ValueError(f"bad value {CANARY_URI}"),
])
def test_ssm_errors_do_not_leak(make_exc):
    with pytest.raises(ConfigError) as ei:
        load_from_ssm(client=_Boom(make_exc()))
    assert _leaks(ei.value, "".join(traceback.format_exception(ei.value))) == []


def test_ssm_malformed_value_does_not_leak():
    class C:
        def get_parameter(self, **kw): return {"Parameter": {"Value": '{"uri": "' + CANARY_URI + '", "token": "' + CANARY_TOKEN}}  # truncated JSON
    with pytest.raises(ConfigError) as ei:
        load_from_ssm(client=C())
    assert _leaks(ei.value, "".join(traceback.format_exception(ei.value))) == []


@pytest.fixture
def app_and_store():
    from ott_recsys.serving.store import MemoryStore
    class FakeBundle:  # only what the error paths touch
        pass
    return H, MemoryStore


def _canary_app(monkeypatch, urlopen_behaviour):
    """The real ZillizREST (canary config) inside the real App, with urllib patched to fail in the given way. A tiny synthetic bundle and user."""
    from ott_recsys.serving.store import MemoryStore
    from ott_recsys.serving.bundle import Bundle
    b = Bundle.__new__(Bundle)
    n, g = 30, 3
    rng = np.random.default_rng(0)
    b.catalog = np.arange(100, 100 + n)
    b.V = rng.normal(size=(n, 4)).astype(np.float32)
    b.emb = rng.normal(size=(n, 5)).astype(np.float32)
    b.genre_m, b.genre_names = np.eye(g, dtype=np.float32)[rng.integers(0, g, n)], ["a", "b", "c"]
    b.rel_ts = np.full(n, 1.0e9)
    b.known = np.arange(n) < 20
    b.pop_ts, b.pop_start, b.pop_order = np.arange(10, dtype=np.uint32), np.minimum(np.arange(n + 1), 10).astype(np.int64), np.arange(n, dtype=np.int32)
    b.titles = [f"t{i}" for i in range(n)]
    from ott_recsys.serving.ncf_np import NCFForward
    b.ncf = NCFForward({}, [], b.catalog)
    class Fixed:  # a ranker that just returns a column
        feature_names = ["tt_score"]
        def predict(self, X): return np.nan_to_num(X[:, 0])
    b.rankers = {"M2": Fixed()}
    store = MemoryStore()
    store.put_user({"user_id": 1, "cand": np.r_[np.arange(10), np.full(190, -1), np.arange(20, 25), np.full(25, -1)].astype(np.int32), "last_ts": 1_500_000_000,
                    "sess_len": 3, "sess_likes": 1, "ncf_known": 0, "ttvec": rng.normal(size=4).astype(np.float32), "prof": rng.normal(size=5).astype(np.float32),
                    "share": np.array([.5, .5, 0], np.float32), "seen": np.array([0, 1], np.int32), "sess_items": np.array([0, 1], np.int32)})
    monkeypatch.setattr(urllib.request, "urlopen", urlopen_behaviour)
    app = H.App(b, store, store, searcher_factory=lambda: ZillizREST(ZillizConfig(CANARY_URI, CANARY_TOKEN)))
    return app


def _fail_with(exc):
    def f(req, timeout=None): raise exc
    return f


@pytest.mark.parametrize("behaviour", [
    _fail_with(urllib.error.URLError(f"unreachable {CANARY_URI}")),
    _fail_with(urllib.error.HTTPError(CANARY_URI, 401, f"unauthorized {CANARY_TOKEN}", {}, io.BytesIO(CANARY_TOKEN.encode()))),
    _fail_with(RuntimeError(f"boom {CANARY_URI} {CANARY_TOKEN}")),
    lambda req, timeout=None: _Resp(b"<html>" + CANARY_TOKEN.encode()),  # not JSON
    lambda req, timeout=None: _Resp(json.dumps({"code": 80001, "message": f"invalid token {CANARY_TOKEN} at {CANARY_URI}"}).encode()),
    lambda req, timeout=None: _Resp(json.dumps({"code": 0}).encode()),  # missing 'data'
], ids=["urlerror", "http401", "runtime", "not-json", "zilliz-error-code", "no-data"])
def test_live_mode_failures_fall_back_without_leaking(monkeypatch, capsys, caplog, behaviour):
    caplog.set_level(logging.DEBUG)
    app = _canary_app(monkeypatch, behaviour)
    status, resp = H.handle(app, "POST", "/recommend", {"user_id": 1, "mode": "live"})
    out = capsys.readouterr()
    assert status == 200, resp
    assert resp["diagnostics"]["fallback"] and "live retrieval failed" in resp["diagnostics"]["fallback"]
    assert _leaks(json.dumps(resp), out.out, out.err, caplog.text) == []


@pytest.mark.parametrize("behaviour", [
    _fail_with(urllib.error.URLError(f"unreachable {CANARY_URI}")),
    _fail_with(urllib.error.HTTPError(CANARY_URI, 401, f"unauthorized {CANARY_TOKEN}", {}, io.BytesIO(CANARY_TOKEN.encode()))),
    _fail_with(RuntimeError(f"boom {CANARY_URI} {CANARY_TOKEN}")),
    lambda req, timeout=None: _Resp(b"<html>" + CANARY_TOKEN.encode()),
    lambda req, timeout=None: _Resp(json.dumps({"code": 80001, "message": f"invalid token {CANARY_TOKEN} at {CANARY_URI}"}).encode()),
], ids=["urlerror", "http401", "runtime", "not-json", "zilliz-error-code"])
def test_search_client_errors_carry_no_secret_text(monkeypatch, behaviour):
    """The client's own exception (message AND formatted traceback) must be clean, not just what the handler chooses to show."""
    monkeypatch.setattr(urllib.request, "urlopen", behaviour)
    with pytest.raises(SearchError) as ei:
        ZillizREST(ZillizConfig(CANARY_URI, CANARY_TOKEN)).search("c", "vector", [0.1, 0.2], 5)
    assert _leaks(ei.value, repr(ei.value), "".join(traceback.format_exception(ei.value))) == []


def test_unexpected_crash_returns_only_the_exception_type(monkeypatch, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    app = _canary_app(monkeypatch, _fail_with(RuntimeError("unused")))
    def explode(user_id): raise RuntimeError(f"db error for {CANARY_URI} {CANARY_TOKEN}")
    app.users.get_user = explode
    status, resp = H.handle(app, "POST", "/recommend", {"user_id": 1})
    out = capsys.readouterr()
    assert status == 500 and resp == {"error": "internal error", "type": "RuntimeError"}
    assert _leaks(json.dumps(resp), out.out, out.err, caplog.text) == []


def test_missing_ssm_config_in_live_mode_falls_back_without_leaking(monkeypatch, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    app = _canary_app(monkeypatch, _fail_with(RuntimeError("unused")))
    def factory(): raise ConfigError("could not load the Zilliz settings: ClientError")
    app._factory = factory
    status, resp = H.handle(app, "POST", "/recommend", {"user_id": 1, "mode": "live"})
    out = capsys.readouterr()
    assert status == 200 and "ConfigError" in resp["diagnostics"]["fallback"]
    assert _leaks(json.dumps(resp), out.out, out.err, caplog.text) == []


def test_lambda_event_wrapper_never_echoes_the_body_or_errors(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    H._APP = _canary_app(monkeypatch, _fail_with(RuntimeError("x")))
    try:
        r = H.handler({"requestContext": {"http": {"method": "POST"}}, "rawPath": "/recommend", "body": "{not json " + CANARY_TOKEN})
        assert r["statusCode"] == 400 and _leaks(r["body"], caplog.text) == []
        r = H.handler({"requestContext": {"http": {"method": "POST"}}, "rawPath": "/recommend", "body": json.dumps({"user_id": CANARY_TOKEN})})
        assert r["statusCode"] == 400 and _leaks(r["body"], caplog.text) == []
    finally:
        H._APP = None
