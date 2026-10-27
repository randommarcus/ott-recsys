"""Checkpoint 5 (demo + latency) without AWS: the scripts run against the REAL request handler in-process (real bundle, in-memory store, exact numpy
search) through a stand-in client; the signed Function URL client is checked against a fake connection."""
import json
from pathlib import Path

import numpy as np
import pytest

from ott_recsys import serving_client as sc
from ott_recsys.serving import LABEL_BANDIT, LABEL_CLOCK, LABEL_DEVICE, LABEL_M3B

from .aws_fakes import load_script
from .serving_real import OUT

demo, study = load_script("demo"), load_script("latency_study")
HAVE = (OUT / "serving" / "users.npz").exists() and (OUT / "serving" / "bundle" / "items.npz").exists()
real = pytest.mark.skipif(not HAVE, reason="serving artifacts not built")


@pytest.fixture(scope="module")
def local():
    from ott_recsys.serving import handler as H
    from ott_recsys.serving.bundle import Bundle
    from ott_recsys.serving.search import NumpySearch
    from ott_recsys.serving.store import MemoryStore
    from ott_recsys.serving_offline import record_at
    b = Bundle(OUT / "serving" / "bundle")
    a = dict(np.load(OUT / "serving" / "users.npz"))
    store = MemoryStore()
    for i in range(len(a["user_id"])):
        store.put_user(record_at(a, i))
    app = H.App(b, store, store, searcher_factory=lambda: NumpySearch(b))

    class LocalClient:
        host_for_connect_test = "localhost"
        calls = []

        def call(self, method, path, body=None):
            LocalClient.calls.append((method, path, dict(body or {})))
            status, payload = H.handle(app, method, path, body or {})
            return status, json.loads(json.dumps(payload)), 0.002

    class LocalInvoker:
        def __init__(self):
            self.cold, self.cleared, self.nonces = False, 0, []

        def force_cold(self, nonce, sleep=None):
            self.cold = True
            self.nonces.append(nonce)

        def clear_environment(self):
            self.cleared += 1

        def invoke(self, method, path, body=None):
            status, payload = H.handle(app, method, path, body or {})
            rep = {"init_ms": 1400.0 if self.cold else None, "duration_ms": payload["timings_ms"]["server_total"] + 3, "max_memory_mb": 260.0}
            self.cold = False
            return status, json.loads(json.dumps(payload)), rep
    return b, LocalClient, LocalInvoker


# ------------------------------------------------------------------------------------------------ demo
@real
def test_demo_page_has_every_label_and_no_endpoint(local, tmp_path):
    b, LocalClient, _ = local
    out = tmp_path / "demo.html"
    assert demo.main(["--users", "8", "--bandit-requests", "20", "--out", str(out)], client=LocalClient(), bundle=b,
                     rates=np.full(len(b.catalog), 0.5), progress=lambda *a, **k: None) == 0
    page = out.read_text()
    for label in (LABEL_M3B, LABEL_BANDIT, LABEL_CLOCK, LABEL_DEVICE):
        assert label in page or label.replace("'", "&#x27;") in page, label
    assert "Synthetic device effect, pipeline demonstration, not a finding" in page
    for bad in ("lambda-url", "amazonaws", "arn:", "AKIA"):
        assert bad not in page
    s = json.loads(out.with_suffix(".json").read_text())
    assert set(s) >= {"A_changed_titles", "B_mean_drama_without_comedy_in_top18", "C_top18_overlap", "D_impressions"}
    assert s["C_top18_overlap"] == 18 and s["C_live_fallback"] is None  # exact local search: live == precomputed
    assert s["D_impressions"] > 0


@real
def test_demo_shows_the_planted_device_effect_and_m2_ignores_device(local, tmp_path):
    b, LocalClient, _ = local
    out = tmp_path / "demo.html"
    demo.main(["--users", "30", "--bandit-requests", "0", "--out", str(out)], client=LocalClient(), bundle=b, progress=lambda *a, **k: None)
    m = json.loads(out.with_suffix(".json").read_text())["B_mean_drama_without_comedy_in_top18"]
    assert m["M3b mobile"] < m["M3b tv"], m  # the planted rule: fewer Drama-without-Comedy titles on mobile


@real
def test_demo_bandit_feedback_only_on_shown_new_release_arms(local, tmp_path):
    b, LocalClient, _ = local
    LocalClient.calls.clear()
    demo.main(["--users", "5", "--bandit-requests", "10", "--out", str(tmp_path / "d.html")], client=LocalClient(), bundle=b,
              rates=np.full(len(b.catalog), 1.0), progress=lambda *a, **k: None)
    fb = [body for m, p, body in LocalClient.calls if p == "/feedback"]
    assert fb and all(body["reward"] == 1 for body in fb) and all(not b.known[body["arm_id"]] for body in fb)
    recs = [body for m, p, body in LocalClient.calls if p == "/recommend"]
    assert sum(body.get("record_impressions") is True for body in recs) == 10, "only the bandit section may record impressions"


def test_demo_escapes_titles():
    html_ = demo.table([{"position": 1, "title": "<script>x</script> & Co", "source": "ranker", "movie_id": 1}])
    assert "<script>" not in html_ and "&lt;script&gt;" in html_


# ------------------------------------------------------------------------------------------------ latency study
@real
def test_latency_study_end_to_end_with_the_real_handler(local, tmp_path, capsys):
    b, LocalClient, LocalInvoker = local
    inv = LocalInvoker()
    out = tmp_path / "res.json"
    assert study.main(["--users", "12", "--cold", "4", "--out", str(out)], client=LocalClient(), invoker=inv,
                      connect=lambda host, n: [120.0] * n, progress=lambda *a, **k: None) == 0
    r = json.loads(out.read_text())
    assert r["warm"]["precomputed"]["server_ms"]["n"] == 12 and r["warm"]["live"]["server_ms"]["n"] == 12
    assert r["warm"]["live"]["fallbacks"] == 0 and r["warm"]["precomputed"]["errors"] == 0
    assert len(r["cold"]) == 4 and [c["mode"] for c in r["cold"]] == ["precomputed", "live"] * 2
    assert all(c["init_ms"] == 1400.0 for c in r["cold"]) and inv.cleared == 1 and inv.nonces == ["1", "2", "3", "4"]
    ids = [v["id"] for v in r["vs_predictions"]]
    assert ids == [f"L{i}" for i in range(1, 9)]
    by = {v["id"]: v for v in r["vs_predictions"]}
    assert by["L5"]["measured"] == 1400.0 and by["L5"]["verdict"] == "inside" and by["L8"]["verdict"] == "inside"
    text = capsys.readouterr().out + out.read_text()
    for bad in ("lambda-url", "amazonaws", "arn:", "AKIA"):
        assert bad not in text
    assert "NOT a property of the design" in text


def test_latency_study_refuses_uncommitted_predictions(tmp_path, monkeypatch):
    fake = tmp_path / "module9_latency_predictions.md"
    fake.write_text(study.PRED.read_text() + "\nedited after the fact\n")
    monkeypatch.setattr(study, "PRED", fake)
    with pytest.raises(SystemExit) as ei:
        study.predictions_committed()
    assert "must be committed and unchanged" in str(ei.value)


def test_latency_study_clears_the_environment_even_if_a_cold_start_fails():
    class Broken:
        cleared = 0
        def force_cold(self, nonce, sleep=None): raise RuntimeError("boom")
        def clear_environment(self): Broken.cleared += 1
    with pytest.raises(RuntimeError):
        study.cold_run(Broken(), [1], 2, progress=lambda *a, **k: None)
    assert Broken.cleared == 1


# ------------------------------------------------------------------------------------------------ client helpers
def test_parse_report_reads_lambda_numbers():
    log = ("START RequestId: x\nsome app line\nREPORT RequestId: x\tDuration: 12.34 ms\tBilled Duration: 13 ms\tMemory Size: 1024 MB\t"
           "Max Memory Used: 243 MB\tInit Duration: 1712.55 ms\n")
    assert sc.parse_report(log) == {"duration_ms": 12.34, "billed_ms": 13.0, "memory_mb": 1024.0, "max_memory_mb": 243.0, "init_ms": 1712.55}
    assert sc.parse_report(log.replace("\tInit Duration: 1712.55 ms", ""))["init_ms"] is None
    assert sc.parse_report("") == {k: None for k in sc.REPORT_RE}


def test_percentiles_and_bootstrap():
    xs = list(range(1, 101))
    assert sc.pct(xs, 50) == 50.5
    lo, hi = sc.bootstrap_ci(xs, 50)
    assert lo < 50.5 < hi


def test_function_url_client_signs_requests_and_hides_the_url(monkeypatch):
    from botocore.credentials import Credentials
    sent = {}

    class Conn:
        def __init__(self, host, timeout=None): sent["host"] = host
        def request(self, method, path, body=None, headers=None): sent.update(method=method, path=path, body=body, headers=headers)
        def getresponse(self):
            class R:
                status = 200
                def read(self): return b'{"ok": true}'
            return R()
        def close(self): pass

    class Lam:
        def get_function_url_config(self, FunctionName): return {"FunctionUrl": "https://abc123xyz.lambda-url.eu-central-1.on.aws/"}

    class Session:
        def client(self, name, region_name=None): return Lam()
        def get_credentials(self):
            class C:
                def get_frozen_credentials(self): return Credentials("AKIDEXAMPLETEST", "secretkeyexample")
            return C()
    monkeypatch.setattr(sc, "HTTPSConnection", Conn)
    c = sc.FunctionUrlClient(Session(), "eu-central-1")
    status, payload, secs = c.call("POST", "/recommend", {"user_id": 1})
    assert status == 200 and payload == {"ok": True} and secs >= 0
    h = {k.lower(): v for k, v in sent["headers"].items()}
    assert h["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLETEST/") and "/eu-central-1/lambda/aws4_request" in h["authorization"]
    assert "x-amz-date" in h and sent["path"] == "/recommend" and json.loads(sent["body"]) == {"user_id": 1}
    assert "abc123xyz" not in repr(c) and "lambda-url" not in repr(c)
    c.call("GET", "/health")
    assert c.connections_opened == 1, "one connection for all requests"
