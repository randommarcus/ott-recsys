"""The single data module: loading, audit, interactions, sessions, time
features and temporal split. Nothing else in the project reads raw files."""
import json

import numpy as np
import pandas as pd

from .config import Config


def _sample_user_ids(cfg: Config) -> np.ndarray:
    """Seeded random sample of users with >= min_sample_interactions ratings."""
    users = pd.read_csv(cfg.dataset_dir / "ratings.csv", usecols=["userId"], dtype="int32")["userId"]
    counts = users.value_counts()
    eligible = np.sort(counts[counts >= cfg.min_sample_interactions].index.to_numpy())
    rng = np.random.default_rng(cfg.sample_seed)
    n = min(cfg.sample_users, len(eligible))
    return np.sort(rng.choice(eligible, size=n, replace=False))


def load_raw(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ratings/movies/links. With cfg.sample_users set, keep a random
    sample of users and ALL of their ratings (never drop individual rows), and
    shrink the catalog to movies those users rated. The sampled ids and seed are
    saved to <out_dir>/sampled_users.json so the subset is reproducible."""
    d = cfg.dataset_dir
    dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    movies = pd.read_csv(d / "movies.csv")
    links = pd.read_csv(d / "links.csv", dtype={"imdbId": str})
    if not cfg.sample_users:
        return pd.read_csv(d / "ratings.csv", dtype=dtypes), movies, links

    chosen = _sample_user_ids(cfg)
    parts = [c[c["userId"].isin(chosen)]
             for c in pd.read_csv(d / "ratings.csv", dtype=dtypes, chunksize=2_000_000)]
    ratings = pd.concat(parts, ignore_index=True)
    assert set(ratings["userId"].unique()) == set(chosen.tolist()), "a sampled user has no ratings"
    rated = ratings["movieId"].unique()
    movies, links = movies[movies["movieId"].isin(rated)], links[links["movieId"].isin(rated)]

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "sampled_users.json").write_text(json.dumps({
        "dataset": cfg.dataset_name, "seed": cfg.sample_seed, "n_users": int(len(chosen)),
        "min_interactions": cfg.min_sample_interactions, "user_ids": chosen.tolist()}))
    return ratings, movies.reset_index(drop=True), links.reset_index(drop=True)


def audit(ratings: pd.DataFrame, movies: pd.DataFrame, links: pd.DataFrame) -> dict:
    """Assertion-style data checks. Raises AssertionError if data is broken;
    otherwise returns the numbers for the report."""
    n_users, n_items = ratings["userId"].nunique(), ratings["movieId"].nunique()
    per_user = ratings.groupby("userId").size()
    per_item = ratings.groupby("movieId").size()
    ts = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)

    assert ratings[["userId", "movieId", "rating", "timestamp"]].notna().all().all(), "missing in ratings"
    assert not ratings.duplicated(["userId", "movieId"]).any(), "user rated same movie twice"
    assert not ratings.duplicated().any(), "exact duplicate rows"
    assert ratings["rating"].between(0.5, 5.0).all(), "rating out of 0.5-5 range"
    assert movies["movieId"].is_unique, "duplicate movieId in movies"
    assert set(ratings["movieId"]) <= set(movies["movieId"]), "rated movie missing from movies.csv"
    assert (ts.min() > pd.Timestamp("1990-01-01", tz="UTC")) and (ts.max() < pd.Timestamp.now(tz="UTC")), "implausible timestamps"

    return {
        "ratings": len(ratings), "users": n_users, "movies_rated": n_items,
        "movies_in_catalog": len(movies),
        "date_min": ts.min(), "date_max": ts.max(),
        "ratings_per_user": per_user.describe(percentiles=[.25, .5, .75, .9]),
        "ratings_per_movie": per_item.describe(percentiles=[.25, .5, .75, .9]),
        "movies_with_1_rating": int((per_item == 1).sum()),
        "movies_with_le_5_ratings": int((per_item <= 5).sum()),
        "sparsity": 1 - len(ratings) / (n_users * n_items),
        "duplicates": int(ratings.duplicated(["userId", "movieId"]).sum()),
        "missing_ratings": int(ratings.isna().sum().sum()),
        "missing_tmdb_ids": int(links["tmdbId"].isna().sum()),
        "missing_genres": int((movies["genres"] == "(no genres listed)").sum()),
        "rating_distribution": ratings["rating"].value_counts().sort_index(),
    }


def make_interactions(ratings: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Every rating is a 'watch'. label=1 if rating >= threshold."""
    df = ratings.rename(columns={"userId": "user_id", "movieId": "item_id"}).copy()
    df["label"] = (df["rating"] >= cfg.positive_threshold).astype("int8")
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)


def add_sessions(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """New session when the gap to the user's previous interaction > N minutes."""
    df = df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    gap = df.groupby("user_id")["timestamp"].diff()
    new = gap.isna() | (gap > cfg.session_gap_minutes * 60)
    df["session_id"] = new.cumsum().astype("int64")  # globally unique
    df["pos_in_session"] = df.groupby("session_id").cumcount()  # 0 = first item
    df["session_len_so_far"] = df["pos_in_session"] + 1  # no peeking at the future
    return df


def burst_report(df: pd.DataFrame, cfg: Config) -> dict:
    """How many ratings come from bulk bursts: >= N ratings inside a short window."""
    win, k = cfg.burst_window_minutes * 60, cfg.burst_min_ratings
    in_burst = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("user_id").indices.items():
        t = df["timestamp"].to_numpy()[idx]  # already sorted within user
        # for each i, count of ratings in [t_i, t_i + win]
        right = np.searchsorted(t, t + win, side="right")
        start = np.flatnonzero(right - np.arange(len(t)) >= k)
        for i in start:
            in_burst[idx[i]:idx[right[i] - 1] + 1] = True
    s = df.groupby("session_id").size()
    return {
        "burst_ratings": int(in_burst.sum()),
        "burst_share": float(in_burst.mean()),
        "users_with_burst": int(df.loc[in_burst, "user_id"].nunique()),
        "sessions": int(s.size), "avg_session_len": float(s.mean()),
        "median_session_len": float(s.median()),
        "single_item_sessions": int((s == 1).sum()),
        "sessions_ge_20": int((s >= 20).sum()),
        "share_ratings_in_sessions_ge_20": float(s[s >= 20].sum() / s.sum()),
    }, in_burst


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """hour_of_day (0-23) and day_of_week (0=Mon) in UTC. MovieLens has no user
    timezone, so these are NOT the viewer's local time."""
    df["hour"] = df["ts"].dt.hour.astype("int8")
    df["dow"] = df["ts"].dt.day_of_week.astype("int8")
    return df


def temporal_split(df: pd.DataFrame, cfg: Config):
    """Global time cutoffs: first 70% of interactions (by time) = train, next 15%
    = val, last 15% = test. Same calendar cutoff for every user."""
    ts = df["timestamp"].to_numpy()
    t_train = int(np.quantile(ts, cfg.train_frac))
    t_val = int(np.quantile(ts, cfg.train_frac + cfg.val_frac))
    train = df[df["timestamp"] <= t_train]
    val = df[(df["timestamp"] > t_train) & (df["timestamp"] <= t_val)]
    test = df[df["timestamp"] > t_val]
    assert len(train) and len(val) and len(test), "empty split"
    assert train["timestamp"].max() <= val["timestamp"].min(), "val leaks into train"
    assert val["timestamp"].max() <= test["timestamp"].min(), "test leaks into val"
    assert test["timestamp"].min() >= train["timestamp"].max(), "test earlier than train"
    assert len(train) + len(val) + len(test) == len(df)
    return train, val, test, {"t_train": t_train, "t_val": t_val}


def cold_start_report(train, val, test) -> dict:
    out = {}
    for name, part in [("val", val), ("test", test)]:
        new_u = ~part["user_id"].isin(set(train["user_id"]))
        new_i = ~part["item_id"].isin(set(train["item_id"]))
        out[name] = {
            "rows": len(part),
            "users": part["user_id"].nunique(),
            "users_not_in_train": int(part.loc[new_u, "user_id"].nunique()),
            "items": part["item_id"].nunique(),
            "items_not_in_train": int(part.loc[new_i, "item_id"].nunique()),
            "rows_with_new_user": int(new_u.sum()),
            "rows_with_new_item": int(new_i.sum()),
            "rows_cold_either": int((new_u | new_i).sum()),
        }
    return out


def build_all(cfg: Config):
    ratings, movies, links = load_raw(cfg)
    stats = audit(ratings, movies, links)
    df = make_interactions(ratings, cfg)
    df = add_sessions(df, cfg)
    sessions, in_burst = burst_report(df, cfg)
    df["in_burst"] = in_burst
    df = add_time_features(df)
    train, val, test, cuts = temporal_split(df, cfg)
    cold = cold_start_report(train, val, test)
    return dict(movies=movies, links=links, df=df, train=train, val=val, test=test,
                audit=stats, sessions=sessions, cuts=cuts, cold=cold)


# ---------------------------------------------------------------------------
# Per-user temporal split (main evaluation split) and the test-set lock
# ---------------------------------------------------------------------------
class TestLocked(RuntimeError):
    """Raised when code touches the test portion before the user approved it."""

    __test__ = False  # not a pytest class


class UserSplit:
    """train/val are plain DataFrames. The test portion is loaded lazily and
    only after unlock_test(); tuning code must never call unlock_test()."""

    def __init__(self, train: pd.DataFrame, val: pd.DataFrame, test_loader):
        self.train, self.val = train, val
        self._test_loader, self._test, self._unlocked = test_loader, None, False

    def unlock_test(self) -> None:
        self._unlocked = True

    @property
    def test(self) -> pd.DataFrame:
        if not self._unlocked:
            raise TestLocked("Test portion is locked until the user approves its use.")
        if self._test is None:
            self._test = self._test_loader()
        return self._test


def per_user_split(df: pd.DataFrame, cfg: Config):
    """For each user: sort by (timestamp, item_id); last 20% -> test, the 10%
    before -> val, rest -> train. Users below the minimum are dropped.
    Only timestamps are inspected here (ordering asserts), never labels."""
    df = df.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort").reset_index(drop=True)
    n = df.groupby("user_id")["item_id"].transform("size")
    df, n = df[n >= cfg.min_user_interactions].copy(), n[n >= cfg.min_user_interactions]
    pos = df.groupby("user_id").cumcount()
    n_test = np.floor(n * cfg.user_test_frac + 1e-9).astype(int)
    n_val = np.floor(n * cfg.user_val_frac + 1e-9).astype(int)
    part = np.where(pos >= n - n_test, "test", np.where(pos >= n - n_test - n_val, "val", "train"))
    train, val, test = (df[part == p].reset_index(drop=True) for p in ["train", "val", "test"])

    g = lambda d, f: d.groupby("user_id")["timestamp"].agg(f)
    tr_max, va_min, va_max, te_min = g(train, "max"), g(val, "min"), g(val, "max"), g(test, "min")
    users = df["user_id"].unique()
    assert all(len(x) == len(users) for x in (tr_max, va_min, te_min)), "a user lacks train/val/test rows"
    assert (tr_max <= va_min).all(), "val earlier than train for some user"
    assert (va_max <= te_min).all(), "test earlier than val for some user"
    assert len(train) + len(val) + len(test) == len(df)
    return train, val, test


def build_user_split(cfg: Config, df: pd.DataFrame) -> dict:
    train, val, test = per_user_split(df, cfg)
    d = cfg.out_dir / "user_split"
    d.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(d / f"{name}.parquet", index=False)
    return {"train": len(train), "val": len(val), "test": len(test), "users": train["user_id"].nunique()}


def load_user_split(cfg: Config) -> UserSplit:
    d = cfg.out_dir / "user_split"
    return UserSplit(pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "val.parquet"),
                     lambda: pd.read_parquet(d / "test.parquet"))


def load_movies(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(cfg.out_dir / "movies.parquet")
