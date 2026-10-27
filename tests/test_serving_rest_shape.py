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


def test_search_level_goes_in_search_params_params_level(monkeypatch):
    """Zilliz "Tune recall rate" docs: REST searchParams.params.level, an integer 1-10 (default 1 when absent)."""
    import pytest
    from ott_recsys.serving.search import SEARCH_LEVEL, search_params_rest
    seen = _capture(monkeypatch, {"code": 0, "data": []})
    ZillizREST(ZillizConfig("https://example.invalid", "tok"), level=7).search("movie_items_twotower_id", "vector", [1.0, 0.0], 230)
    body = json.loads(seen["req"].data)
    assert body["searchParams"] == {"metricType": "IP", "params": {"level": 7}}
    assert search_params_rest(None) == {"metricType": "IP"}  # no level -> server default; nothing else added
    for bad in (0, 11, 2.5, "5"):
        with pytest.raises(ValueError):
            search_params_rest(bad)
    assert SEARCH_LEVEL is None or 1 <= SEARCH_LEVEL <= 10


def test_the_lambda_uses_search_level_5():
    """Chosen from the level study (2026-10-27): the lowest level at the recall plateau."""
    from ott_recsys.serving import handler
    from ott_recsys.serving.search import SEARCH_LEVEL
    assert SEARCH_LEVEL == 5
    import inspect
    assert "level=SEARCH_LEVEL" in inspect.getsource(handler._default_app)


def test_one_connection_is_reused_across_searches(monkeypatch, route_zilliz_rest_through_urlopen):
    _capture(monkeypatch, {"code": 0, "data": [{"item_id": 1, "distance": 0.5}]})
    client = ZillizREST(ZillizConfig("https://example.invalid:19530", "tok"), level=5)
    for _ in range(3):
        client.search("movie_items_twotower_id", "vector", [1.0, 0.0], 10)
    assert route_zilliz_rest_through_urlopen.opened == 1 and client.connections_opened == 1


def test_a_stale_kept_alive_connection_is_retried_once(monkeypatch, route_zilliz_rest_through_urlopen):
    from http.client import RemoteDisconnected
    _capture(monkeypatch, {"code": 0, "data": [{"item_id": 4, "distance": 0.5}]})
    client = ZillizREST(ZillizConfig("https://example.invalid", "tok"))
    client.search("c", "vector", [1.0], 1)
    route_zilliz_rest_through_urlopen.fail_next = [RemoteDisconnected("server closed the idle connection")]
    ids, _ = client.search("c", "vector", [1.0], 1)
    assert ids.tolist() == [4] and client.connections_opened == 2


def test_two_stale_failures_in_a_row_give_up_with_a_clean_error(monkeypatch, route_zilliz_rest_through_urlopen):
    import pytest
    from http.client import RemoteDisconnected
    from ott_recsys.serving.search import SearchError
    _capture(monkeypatch, {"code": 0, "data": []})
    route_zilliz_rest_through_urlopen.fail_next = [RemoteDisconnected("tok"), ConnectionResetError("tok")]
    with pytest.raises(SearchError) as ei:
        ZillizREST(ZillizConfig("https://example.invalid", "tok")).search("c", "vector", [1.0], 1)
    assert ei.value.args == ("connection-closed",)


def test_http_errors_and_bad_json_close_the_connection(monkeypatch):
    import io, pytest, urllib.error
    from ott_recsys.serving.search import SearchError

    def http401(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b""))
    monkeypatch.setattr(urllib.request, "urlopen", http401)
    client = ZillizREST(ZillizConfig("https://example.invalid", "tok"))
    with pytest.raises(SearchError) as ei:
        client.search("c", "vector", [1.0], 1)
    assert ei.value.args == ("http-401",) and client._conn is None
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(b"<html>"))
    with pytest.raises(SearchError) as ei:
        client.search("c", "vector", [1.0], 1)
    assert ei.value.args == ("bad-json",)
