"""Module 1 tests. Logic tests use tiny synthetic data; the last tests check
the real processed files (skipped if data hasn't been built)."""
import pandas as pd
import pytest

from ott_recsys.config import Config
from ott_recsys.data import (add_sessions, add_time_features, audit, burst_report,
                             make_interactions, temporal_split)

cfg = Config()


def _ratings(rows):
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


def test_label_threshold():
    df = make_interactions(_ratings([(1, 1, 3.5, 100), (1, 2, 4.0, 200), (1, 3, 5.0, 300)]), cfg)
    assert df["label"].tolist() == [0, 1, 1]


def test_session_gap_is_30_minutes():
    t0, gap = 1_000_000, 30 * 60
    rows = [(1, 1, 4, t0), (1, 2, 4, t0 + gap),          # exactly 30 min -> same session
            (1, 3, 4, t0 + gap + gap + 1),                # > 30 min -> new session
            (2, 1, 4, t0)]                                # other user -> own session
    df = add_sessions(make_interactions(_ratings(rows), cfg), cfg)
    u1 = df[df.user_id == 1]
    assert u1["session_id"].nunique() == 2
    assert df["session_id"].nunique() == 3
    assert u1["session_len_so_far"].tolist() == [1, 2, 1]


def test_burst_detection():
    t0 = 1_000_000
    rows = [(1, i, 4, t0 + i) for i in range(40)] + [(2, i, 4, t0 + i * 10_000) for i in range(5)]
    df = add_sessions(make_interactions(_ratings(rows), cfg), cfg)
    rep, flags = burst_report(df, cfg)
    assert rep["burst_ratings"] == 40
    assert df.loc[flags, "user_id"].unique().tolist() == [1]


def test_time_features_utc():
    df = make_interactions(_ratings([(1, 1, 4, 0)]), cfg)  # 1970-01-01 00:00 UTC, Thursday
    df = add_time_features(df)
    assert (df.loc[0, "hour"], df.loc[0, "dow"]) == (0, 3)


def test_split_has_no_time_leakage():
    rows = [(u, i, 4, 1_000 + 100 * (u * 10 + i)) for u in range(1, 11) for i in range(10)]
    df = make_interactions(_ratings(rows), cfg)
    train, val, test, _ = temporal_split(df, cfg)
    assert test["timestamp"].min() >= train["timestamp"].max()
    assert val["timestamp"].min() >= train["timestamp"].max()
    assert test["timestamp"].min() >= val["timestamp"].max()
    assert len(train) + len(val) + len(test) == len(df)


def test_audit_catches_duplicates():
    r = _ratings([(1, 1, 4, 1_000_000_000), (1, 1, 5, 1_000_000_100)])
    movies = pd.DataFrame({"movieId": [1], "title": ["a"], "genres": ["x"]})
    links = pd.DataFrame({"movieId": [1], "imdbId": ["1"], "tmdbId": [1]})
    with pytest.raises(AssertionError):
        audit(r, movies, links)


@pytest.fixture(scope="module")
def processed():
    if not (cfg.out_dir / "train.parquet").exists():
        pytest.skip("run scripts/run_module1.py first")
    return {n: pd.read_parquet(cfg.out_dir / f"{n}.parquet") for n in ["train", "val", "test"]}


def test_processed_files_are_time_ordered(processed):
    assert processed["test"]["timestamp"].min() >= processed["train"]["timestamp"].max()
    assert processed["val"]["timestamp"].min() >= processed["train"]["timestamp"].max()


def test_processed_columns(processed):
    need = {"user_id", "item_id", "rating", "label", "timestamp", "session_id",
            "session_len_so_far", "hour", "dow", "in_burst"}
    assert need <= set(processed["train"].columns)
