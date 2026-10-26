"""scripts/check_zilliz.py must never print, log or echo the Zilliz URI or token, including in error messages."""
import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_zilliz.py"
SECRET_NAMES = {"uri", "token", "exc", "client"}  # the secrets, the exception object (its message may hold the URI) and the connected client


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def test_no_print_or_log_call_mentions_a_secret():
    tree = ast.parse(SCRIPT.read_text())
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        f = call.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
        if name in {"print", "write", "debug", "info", "warning", "error", "exception", "critical", "log"}:
            for arg in list(call.args) + [k.value for k in call.keywords]:
                bad = _names(arg) & SECRET_NAMES
                # `type(exc).__name__` is allowed: it names the exception CLASS, never its message
                for inner in ast.walk(arg):
                    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "type":
                        bad -= _names(inner)
                assert not bad, f"line {call.lineno}: {name}(...) mentions {bad}"


def test_source_never_reaches_for_tracebacks_or_exception_text():
    src = SCRIPT.read_text()
    tree = ast.parse(src)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert "traceback" not in imported
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"str", "repr", "format"} and "exc" in _names(n):
            raise AssertionError(f"line {n.lineno}: exception text is formatted")
        if isinstance(n, ast.Attribute) and n.attr in {"print_exc", "format_exc", "exception"}:
            raise AssertionError(f"line {n.lineno}: traceback/exception logging is used")
    assert "{uri}" not in src and "{token}" not in src and "ZILLIZ_URI}" not in src


def _run(env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ZILLIZ_")}
    env.update(env_extra)
    return subprocess.run([sys.executable, "-I", str(SCRIPT)], capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)


def test_canary_secrets_never_appear_in_any_output_when_the_connection_fails():
    uri, token = "http://canary-host-7c1e.invalid:19530", "CANARY_TOKEN_9f3a1b"
    r = _run({"ZILLIZ_URI": uri, "ZILLIZ_TOKEN": token})
    out = r.stdout + r.stderr
    assert r.returncode == 1 and "connection succeeded: no (error type:" in out
    for secret in (uri, token, "canary-host", "CANARY_TOKEN", "9f3a1b", "Traceback"):
        assert secret not in out, f"leaked {secret!r}"


def test_missing_credentials_are_reported_without_any_value():
    r = _run({"ZILLIZ_URI": "", "ZILLIZ_TOKEN": ""})
    assert r.returncode == 1 and "credentials are not set" in r.stdout and "Traceback" not in r.stdout + r.stderr
