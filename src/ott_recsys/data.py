"""The single data module: loading, audit, interactions, sessions, time
features and temporal split. Nothing else in the project reads raw files."""
import numpy as np
import pandas as pd

from .config import Config


def load_raw(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = cfg.dataset_dir
    ratings = pd.read_csv(d / "ratings.csv")
    movies = pd.read_csv(d / "movies.csv")
    links = pd.read_csv(d / "links.csv", dtype={"imdbId": str})
    if cfg.max_users:
        top = ratings["userId"].value_counts().head(cfg.max_users).index
        ratings = ratings[ratings["userId"].isin(top)]
    return ratings, movies, links


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
