"""Check the Zilliz Cloud connection. READ-ONLY: creates nothing.

Prints ONLY: whether the connection succeeded, the server version (if available) and the collection names.
SECURITY: the URI and token (ZILLIZ_URI / ZILLIZ_TOKEN from .env, loaded like scripts/fetch_tmdb.py) are never printed, logged
or interpolated into any message. Errors are reported by exception TYPE only (messages can contain the URI), library
output and logging are swallowed, and no traceback is ever shown. tests/test_check_zilliz.py enforces this."""
import contextlib
import io
import logging
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # does not override variables already set in the environment
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


def main() -> int:
    uri = os.environ.get("ZILLIZ_URI", "").strip()
    token = os.environ.get("ZILLIZ_TOKEN", "").strip()
    if not uri or not token:
        print("connection succeeded: no (Zilliz credentials are not set)")
        return 1
    sink = io.StringIO()  # swallow anything the client library prints
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            from pymilvus import MilvusClient
            client = MilvusClient(uri=uri, token=token, timeout=20)
    except Exception as exc:
        print(f"connection succeeded: no (error type: {type(exc).__name__})")
        return 1
    print("connection succeeded: yes")
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            version = str(client.get_server_version())
    except Exception:
        version = "not available"
    print(f"server version: {version}")
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            names = sorted(client.list_collections())
        print(f"collections ({len(names)}): {names}")
    except Exception as exc:
        print(f"collections: could not be listed (error type: {type(exc).__name__})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
