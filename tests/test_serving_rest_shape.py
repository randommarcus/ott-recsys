"""The Lambda's Zilliz REST search request must match Zilliz's REST v2 docs (checked 2026-10-27): POST /v2/vectordb/entities/search, Bearer auth,
searchParams {"metricType": "IP"}. AUTOINDEX takes no ef, so none is sent. No network: urllib is patched."""
import json
import urllib.request

import numpy as np

from ott_recsys.serving.search import ZillizREST
from ott_recsys.serving.secrets import ZillizConfig


class _Resp:
    def __init__(self, body): self.body = body
    def read(self): return self.body
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _capture(monkeypatch, payload):
    seen = {}

    def fake(req, timeout=None):
        seen["req"], seen["timeout"] = req, timeout
        return _Resp(json.dumps(payload).encode())
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return seen


def test_search_request_shape(monkeypatch):
    seen = _capture(monkeypatch, {"code": 0, "data": [{"item_id": 7, "distance": 0.9}, {"item_id": 3, "distance": 0.5}]})
    ids, scores = ZillizREST(ZillizConfig("https://example.invalid:19530/", "tok")).search("movie_items_content", "vec384", np.ones(4), 30, "is_cold == true")
    req = seen["req"]
    assert req.full_url == "https://example.invalid:19530/v2/vectordb/entities/search" and req.get_method() == "POST"
    assert req.get_header("Authorization") == "Bearer tok" and req.get_header("Content-type") == "application/json"
    body = json.loads(req.data)
    assert body == {"collectionName": "movie_items_content", "data": [[1.0, 1.0, 1.0, 1.0]], "annsField": "vec384", "limit": 30,
                    "outputFields": ["item_id"], "searchParams": {"metricType": "IP"}, "filter": "is_cold == true"}
    assert "ef" not in json.dumps(body["searchParams"]) and "metric_type" not in req.data.decode()
    assert ids.tolist() == [7, 3] and np.allclose(scores, [0.9, 0.5])


def test_no_filter_key_without_a_filter_and_id_fallback(monkeypatch):
    seen = _capture(monkeypatch, {"code": 0, "data": [{"id": 11, "distance": 0.2}]})
    ids, _ = ZillizREST(ZillizConfig("https://example.invalid", "tok")).search("movie_items_twotower_id", "vector", [0.5, 0.5], 230)
    body = json.loads(seen["req"].data)
    assert "filter" not in body and body["limit"] == 230 and body["annsField"] == "vector"
    assert ids.tolist() == [11]
