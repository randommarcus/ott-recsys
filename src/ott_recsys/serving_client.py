"""Laptop-side client for the deployed Lambda (checkpoint 5: demo and latency). OFFLINE code: never runs in Lambda.

FunctionUrlClient  signs each request with SigV4 (service 'lambda') using the profile's credentials and sends it over ONE reused HTTPS connection to the
                   IAM-only Function URL. The URL is looked up at start-up and kept in memory: never printed, logged or written to a file.
DirectInvoker      calls the function through the Lambda API with LogType=Tail, so Lambda's own REPORT line (Duration, Init Duration, Max Memory Used)
                   can be read for cold starts. Only those numbers are parsed out; the log text itself is never printed.
Also: a TCP+TLS connect timer (no credentials) and small statistics helpers (percentiles, bootstrap intervals)."""
import base64
import json
import re
import socket
import ssl
import time
import urllib.parse
from http.client import HTTPSConnection, CannotSendRequest, RemoteDisconnected

import numpy as np

FN = "ott-recsys-serve"
STALE = (RemoteDisconnected, ConnectionResetError, BrokenPipeError, CannotSendRequest)


class ClientError(Exception):
    """Message holds an exception type name or an HTTP status only."""


class FunctionUrlClient:
    def __init__(self, session, region: str, function: str = FN, timeout: float = 30.0):
        lam = session.client("lambda", region_name=region)
        target = urllib.parse.urlsplit(lam.get_function_url_config(FunctionName=function)["FunctionUrl"])
        self._host, self._region, self.timeout = target.hostname, region, timeout
        self._creds = session.get_credentials().get_frozen_credentials()
        self._conn = None
        self.connections_opened = 0

    def __repr__(self) -> str:
        return "FunctionUrlClient(<hidden>)"

    @property
    def host_for_connect_test(self) -> str:
        return self._host

    def _connection(self):
        if self._conn is None:
            self._conn = HTTPSConnection(self._host, timeout=self.timeout)
            self.connections_opened += 1
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def call(self, method: str, path: str, body: dict | None = None):
        """Returns (status, payload dict, client seconds). Client seconds = from sending the request to having read the full response."""
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest
        data = json.dumps(body or {}).encode() if method == "POST" else None
        for attempt in (1, 2):
            req = AWSRequest(method=method, url=f"https://{self._host}{path}", data=data, headers={"Content-Type": "application/json"} if data else {})
            SigV4Auth(self._creds, "lambda", self._region).add_auth(req)
            try:
                t0 = time.perf_counter()
                conn = self._connection()
                conn.request(method, path, body=data, headers=dict(req.headers.items()))
                resp = conn.getresponse()
                raw = resp.read()
                secs = time.perf_counter() - t0
            except STALE:
                self.close()
                if attempt == 2:
                    raise ClientError("connection-closed") from None
                continue
            except Exception as exc:
                self.close()
                raise ClientError(type(exc).__name__) from None
            try:
                payload = json.loads(raw) if raw else {}
            except ValueError:
                payload = {}
            return resp.status, payload, secs
        raise ClientError("connection-closed")


REPORT_RE = {"duration_ms": r"\bDuration: ([\d.]+) ms", "billed_ms": r"Billed Duration: ([\d.]+) ms", "memory_mb": r"Memory Size: (\d+) MB",
             "max_memory_mb": r"Max Memory Used: (\d+) MB", "init_ms": r"Init Duration: ([\d.]+) ms"}


def parse_report(log_text: str) -> dict:
    """Numbers from Lambda's REPORT line. init_ms is None on a warm invocation."""
    line = next((l for l in log_text.splitlines() if l.startswith("REPORT")), "")
    out = {}
    for k, rx in REPORT_RE.items():
        m = re.search(rx, line)
        out[k] = float(m.group(1)) if m else None
    return out


class DirectInvoker:
    def __init__(self, session, region: str, function: str = FN):
        self.lam, self.fn = session.client("lambda", region_name=region), function

    def invoke(self, method: str, path: str, body: dict | None = None):
        """Returns (status, payload dict, REPORT numbers)."""
        event = {"requestContext": {"http": {"method": method}}, "rawPath": path, "body": json.dumps(body or {})}
        r = self.lam.invoke(FunctionName=self.fn, Payload=json.dumps(event).encode(), LogType="Tail")
        out = json.loads(r["Payload"].read())
        report = parse_report(base64.b64decode(r.get("LogResult", "")).decode("utf-8", "replace"))
        payload = json.loads(out.get("body", "{}")) if isinstance(out, dict) and isinstance(out.get("body"), str) else {}
        return (out.get("statusCode", 0) if isinstance(out, dict) else 0), payload, report

    def force_cold(self, nonce: str, sleep=time.sleep) -> None:
        """A configuration change makes Lambda start NEW execution environments for the next invocations. The variable holds only a counter."""
        self.lam.update_function_configuration(FunctionName=self.fn, Environment={"Variables": {"OTT_COLD_NONCE": nonce}})
        self.lam.get_waiter("function_updated_v2").wait(FunctionName=self.fn)

    def clear_environment(self) -> None:
        self.lam.update_function_configuration(FunctionName=self.fn, Environment={"Variables": {}})
        self.lam.get_waiter("function_updated_v2").wait(FunctionName=self.fn)


def connect_times_ms(host: str, n: int = 20, port: int = 443, timeout: float = 10.0) -> list:
    """TCP connect + TLS handshake to `host`, n times, in milliseconds. No request and no credentials are sent."""
    ctx = ssl.create_default_context()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host):
                out.append((time.perf_counter() - t0) * 1000)
    return out


def pct(xs, q: float) -> float:
    return float(np.percentile(np.asarray(xs, float), q)) if len(xs) else float("nan")


def bootstrap_ci(xs, q: float, n_boot: int = 1000, seed: int = 0):
    """95% percentile-bootstrap interval for the q-th percentile."""
    xs = np.asarray(xs, float)
    if len(xs) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.percentile(xs[rng.integers(0, len(xs), (n_boot, len(xs)))], q, axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
