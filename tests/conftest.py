"""Shared test seam for the Lambda's Zilliz REST client (serving/search.py), which talks over a reused http.client connection.

Many tests fake the network by replacing urllib.request.urlopen with a function that takes a urllib Request and returns a response or raises. This autouse
fixture swaps the client's connection classes for a FakeConnection that turns each request into a urllib Request and hands it to whatever
urllib.request.urlopen is at that moment, so those fakes keep working unchanged. If no test replaced urlopen, the real one runs, exactly as before
(tests never reach a real cluster: they use fake or '.invalid' endpoints). FakeConnection.opened counts connections, to test reuse."""
import urllib.error
import urllib.request

import pytest

from ott_recsys.serving import search as search_mod


class _Resp:
    def __init__(self, status, data):
        self.status, self._data = status, data

    def read(self):
        return self._data


class FakeConnection:
    opened = 0
    fail_next = []  # exceptions to raise on the next request() calls, in order (e.g. a stale keep-alive connection)

    def __init__(self, host, port=None, timeout=None, **kw):
        self.host, self.port, self.timeout, self.scheme = host, port, timeout, "https"
        FakeConnection.opened += 1
        self._req = None

    def request(self, method, url, body=None, headers=None):
        if FakeConnection.fail_next:
            raise FakeConnection.fail_next.pop(0)
        netloc = self.host + (f":{self.port}" if self.port else "")
        self._req = urllib.request.Request(f"{self.scheme}://{netloc}{url}", data=body, method=method, headers=headers or {})

    def getresponse(self):
        try:
            r = urllib.request.urlopen(self._req, timeout=self.timeout)
            return _Resp(getattr(r, "status", 200), r.read())
        except urllib.error.HTTPError as http:
            return _Resp(http.code, b"")

    def close(self):
        pass


class FakeHTTPConnection(FakeConnection):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.scheme = "http"


@pytest.fixture(autouse=True)
def route_zilliz_rest_through_urlopen(monkeypatch):
    FakeConnection.opened = 0
    FakeConnection.fail_next = []
    monkeypatch.setattr(search_mod, "HTTPSConnection", FakeConnection)
    monkeypatch.setattr(search_mod, "HTTPConnection", FakeHTTPConnection)
    yield FakeConnection
