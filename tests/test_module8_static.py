"""Repository-wide static guards for the one-shot test evaluation. These replace per-file lists: any NEW file under src/ or
scripts/ is covered automatically."""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNLOCK = ROOT / "scripts" / "unlock_test.py"
PY = [p for d in ("src", "scripts") for p in sorted((ROOT / d).rglob("*.py"))]
# files that may legitimately mention the unlock machinery: the script itself, the guard module, and the sealed-split class that DEFINES the lock
EXEMPT = {UNLOCK, ROOT / "src" / "ott_recsys" / "unlockguard.py", ROOT / "src" / "ott_recsys" / "data.py"}


def _tree(p):
    return ast.parse(p.read_text())


def test_nothing_else_references_the_unlock_machinery():
    bad = []
    for p in PY:
        if p in EXEMPT:
            continue
        for n in ast.walk(_tree(p)):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in n.names] + ([n.module] if isinstance(n, ast.ImportFrom) and n.module else []) + \
                       [a.name for a in n.names if isinstance(n, ast.ImportFrom)]
                if any("unlock_test" in m or "unlockguard" in m for m in mods):
                    bad.append(f"{p.name}:{n.lineno} imports the unlock machinery")
            elif isinstance(n, ast.Attribute) and n.attr in {"unlock_test", "_test_loader"}:
                bad.append(f"{p.name}:{n.lineno} touches .{n.attr}")
            elif isinstance(n, ast.Name) and n.id in {"unlock_test", "unlockguard"}:
                bad.append(f"{p.name}:{n.lineno} uses the name {n.id}")
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in {"unlock_test"}:
                bad.append(f"{p.name}:{n.lineno} defines {n.name}")
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and any(t in n.value for t in ("unlock_test.py", "TEST_UNLOCK_LOG", "unlockguard")):
                bad.append(f"{p.name}:{n.lineno} mentions the unlock script / log in a string")
    assert not bad, "\n".join(bad)


def test_unlock_script_cannot_be_imported():
    tree = _tree(UNLOCK)
    first = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))][:2]
    guard = first[1]
    assert isinstance(first[0], ast.Import) and isinstance(guard, ast.If) and "__name__" in ast.dump(guard.test) and "NotEq" in ast.dump(guard.test)
    assert any(isinstance(n, ast.Raise) and "ImportError" in ast.dump(n) for n in guard.body)
    code = ("import importlib.util as u, sys; spec = u.spec_from_file_location('unlock_test', r'%s'); m = u.module_from_spec(spec); "
            "spec.loader.exec_module(m)" % UNLOCK)
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert r.returncode != 0 and "ImportError" in r.stderr and "never be imported" in r.stderr


FORBIDDEN_CALLS = {"fit", "fit_transform", "partial_fit", "fit_ranker", "oof_scores", "select_param", "fit_prior", "train", "knn_neighbors",
                   "build_pairs", "fold_in_users", "XGBRanker", "Ridge", "KFold", "PCA", "sample_negatives", "eb_weight", "encode_texts", "pca_reduce",
                   "set_weights", "compile", "minimize_scalar"}


def test_unlock_script_trains_and_tunes_nothing():
    called = set()
    for n in ast.walk(_tree(UNLOCK)):
        if isinstance(n, ast.Call):
            f = n.func
            called.add(f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else "")
    assert not (called & FORBIDDEN_CALLS), called & FORBIDDEN_CALLS


def _first_call_line(tree, name):
    lines = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and
             ((isinstance(n.func, ast.Attribute) and n.func.attr == name) or (isinstance(n.func, ast.Name) and n.func.id == name))]
    return min(lines)


def test_unlock_script_orders_log_then_test_read_then_metrics_flag_then_report():
    t = _tree(UNLOCK)
    log, test_read, flag, emit = (_first_call_line(t, n) for n in ("write_start_log", "unlock_test", "mark_metrics_emitted", "emit_report"))
    first_eval = min(_first_call_line(t, n) for n in ("per_user_evaluation", "bandit_evaluation"))
    assert log < test_read < first_eval < flag < emit
    src = UNLOCK.read_text()
    assert "UNLOCK-TEST-ONCE" in src and "check_not_spent" in src and "verify_manifest" in src and "preregistration" in src and "clean_tree" in src


def test_unlock_script_never_prints_inside_the_computation_functions():
    """Nothing test-derived may be printed or written before the metrics-emitted flag: the compute functions must not print or write."""
    t = _tree(UNLOCK)
    for fn in (n for n in t.body if isinstance(n, ast.FunctionDef) and n.name in {"per_user_evaluation", "bandit_evaluation"}):
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                name = n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else ""
                assert name not in {"print", "write_text", "write_bytes", "open", "to_parquet", "to_csv", "save", "savez"}, f"{fn.name} calls {name}"


def test_sealed_split_still_locked_by_default():
    import pandas as pd
    import pytest
    from ott_recsys.data import TestLocked, UserSplit
    s = UserSplit(pd.DataFrame(), pd.DataFrame(), lambda: (_ for _ in ()).throw(AssertionError("loaded")))
    with pytest.raises(TestLocked):
        _ = s.test
