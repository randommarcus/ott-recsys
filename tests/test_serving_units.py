"""Small exactness checks for the Lambda-side re-implementations, plus the rule that the serving package stays numpy + standard library only."""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ott_recsys.bandit import strategy_keys
from ott_recsys.serving.features import hour_dow
from ott_recsys.serving.ncf_np import NCFForward
from ott_recsys.serving.ucb import C_PREREGISTERED, pick, ucb_keys

from .serving_real import HAVE_REAL, OUT

SERVING = Path(__file__).resolve().parents[1] / "src" / "ott_recsys" / "serving"
ALLOWED_TOP = set(sys.stdlib_module_names) | {"numpy"}
LAZY_ONLY = {"boto3"}  # may be imported inside a function (real AWS client), never at module level


def test_serving_imports_only_numpy_and_the_standard_library():
    for f in sorted(SERVING.glob("*.py")):
        tree = ast.parse(f.read_text())
        top_level = {id(n) for n in tree.body}
        for n in ast.walk(tree):
            names = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else \
                    [] if not isinstance(n, ast.ImportFrom) or n.level else [n.module.split(".")[0]]
            for name in names:
                if name in LAZY_ONLY:
                    assert id(n) not in top_level, f"{f.name}: {name} must be imported lazily"
                else:
                    assert name in ALLOWED_TOP, f"{f.name} imports {name}, which is not available / too big for Lambda"
            if isinstance(n, ast.ImportFrom) and n.level and n.module:
                assert (SERVING / (n.module.split(".")[0] + ".py")).exists() or n.module == "", f"{f.name}: relative import of {n.module}"
            if isinstance(n, ast.ImportFrom) and n.level == 2:
                pytest.fail(f"{f.name}: imports from outside the serving package")


def test_ucb_keys_equal_the_simulation_rule():
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = int(rng.integers(2, 30))
        n = rng.integers(0, 40, m).astype(float)
        s = np.minimum(rng.integers(0, 40, m), n).astype(float)
        mine = ucb_keys(n, s, C_PREREGISTERED, np.random.default_rng(1))
        ref = strategy_keys("ucb_c", n[None], s[None], None, None, 0.0, 0.0, 0.0, np.random.default_rng(1), c=C_PREREGISTERED)[0]
        seen = n > 0
        assert np.allclose(mine[seen], ref[seen])
        assert (mine[~seen] >= 1e9).all() and (ref[~seen] >= 1e9).all()
    assert C_PREREGISTERED == 0.2


def test_ucb_prefers_unseen_then_the_promising_arm():
    assert set(pick([0, 0, 5, 9], [0, 0, 1, 8], 2, rng=np.random.default_rng(0))) == {0, 1}
    assert pick([10, 10], [2, 8], 1)[0] == 1
    assert len(pick([3], [1], 2)) == 1


def test_hour_and_day_of_week_match_pandas():
    rng = np.random.default_rng(0)
    for t in rng.integers(0, 1_700_000_000, 500):
        ts = pd.to_datetime(int(t), unit="s")
        assert hour_dow(int(t)) == (ts.hour, ts.day_of_week)


@pytest.mark.skipif(not HAVE_REAL, reason="NCF weights not on disk")
def test_ncf_forward_matches_the_training_side_scorer():
    from ott_recsys.ncf import NCFScorer
    fold = NCFScorer.load(OUT / "test_prep" / "real" / "ncf_foldin.npz")
    stale = NCFScorer.load(OUT / "ncf" / "ncf_main" / "weights.npz")
    catalog = np.sort(pd.read_csv(OUT.parents[1] / "raw" / "ml-25m" / "movies.csv")["movieId"].to_numpy())
    fwd = NCFForward({k: stale.a[k] for k in ("ig", "im", "Wo", "bo", "W0", "b0", "W1", "b1", "W2", "b2")}, stale.item_ids, catalog)
    rng = np.random.default_rng(0)
    for u in rng.choice(fold.user_ids, 25, replace=False):
        cats = rng.choice(len(catalog), 300, replace=False)
        ref = fold.score(np.full(300, u), catalog[cats])
        i = fold._u.get_indexer([u])[0]
        got = fwd.score(fold.a["ug"][i], fold.a["um"][i], cats)
        assert np.array_equal(np.isnan(ref), np.isnan(got)) and np.allclose(ref, got, atol=1e-5, equal_nan=True)
    assert np.isnan(fwd.score(None, None, np.arange(5))).all()
