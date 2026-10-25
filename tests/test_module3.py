"""Module 3 tests: TMDB fetcher (mocked, no network/key), item text, content
retrieval, release sets and the new-release evaluation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ott_recsys import content
from ott_recsys.baselines import ContentRetrieval, GlobalPop, RandomTies
from ott_recsys.eval import evaluate
from ott_recsys.newrelease import evaluate_new_release, release_sets
from ott_recsys.tmdb import fetch_all

KEY = "SECRETKEY123"


# ------------------------------------------------------------ TMDB fetcher
class Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._b, self.headers = status, body or {}, headers or {}

    def json(self):
        return self._b


class FakeSession:
    """script: dict tmdb_id -> list of responses/exceptions, consumed in order."""
    def __init__(self, script, calls):
        self.script, self.calls = script, calls

    def get(self, url, timeout=None, params=None):
        assert params["api_key"] == KEY
        i = int(url.rsplit("/", 1)[1])
        self.calls.append(i)
        step = self.script[i].pop(0) if len(self.script[i]) > 1 else self.script[i][0]
        if isinstance(step, Exception):
            raise step
        return step


def _ok(i):
    return Resp(200, {"id": i, "title": f"M{i}", "overview": "o" * 50, "release_date": "2000-01-01",
                      "genres": [{"id": 1, "name": "Drama"}], "keywords": {"keywords": [{"id": 9, "name": "kw"}]},
                      "popularity": 99.9, "vote_count": 5})


def _run(tmp_path, script, ids, **kw):
    calls, logs = [], []
    stats = fetch_all(ids, tmp_path / "c", KEY, lambda: FakeSession(script, calls), per_sec=1000, workers=2,
                      log=logs.append, sleep=lambda s: None, **kw)
    return stats, calls, logs


def test_fetch_caches_resumes_and_keeps_only_whitelisted_fields(tmp_path):
    script = {1: [_ok(1)], 2: [Resp(404)], 3: [_ok(3)]}
    stats, calls, _ = _run(tmp_path, script, [1, 2, 3, 3])
    assert (stats["ok"], stats["not_found"], stats["failed"]) == (2, 1, 0) and sorted(calls) == [1, 2, 3]
    d = json.loads((tmp_path / "c" / "1.json").read_text())
    assert d["genres"] == ["Drama"] and d["keywords"] == ["kw"] and "popularity" not in d and "vote_count" not in d
    assert json.loads((tmp_path / "c" / "2.json").read_text()) == {"not_found": True}
    stats2, calls2, _ = _run(tmp_path, script, [1, 2, 3])  # resume: nothing left to fetch
    assert calls2 == [] and stats2["already_cached"] == 3
    assert KEY not in "".join(p.read_text() for p in (tmp_path / "c").glob("*.json"))


def test_fetch_retries_then_succeeds_and_never_logs_the_key(tmp_path):
    boom = ConnectionError(f"https://api.themoviedb.org/3/movie/1?api_key={KEY} failed")
    script = {1: [boom, boom, _ok(1)], 2: [Resp(500), _ok(2)]}
    stats, calls, logs = _run(tmp_path, script, [1, 2])
    assert stats["ok"] == 2 and stats["failed"] == 0
    assert logs and all(KEY not in line for line in logs)


def test_fetch_stops_on_bad_key(tmp_path):
    stats, calls, logs = _run(tmp_path, {i: [Resp(401)] for i in range(1, 30)}, list(range(1, 30)))
    assert "rejected the API key" in stats["aborted"] and stats["ok"] == 0
    assert not list((tmp_path / "c").glob("*.json"))


def test_fetch_aborts_after_many_consecutive_failures_and_leaves_failed_uncached(tmp_path):
    script = {i: [ConnectionError("x")] for i in range(1, 40)}
    stats, _, _ = _run(tmp_path, script, list(range(1, 40)), max_consecutive_failures=5)
    assert stats["aborted"] and "consecutive" in stats["aborted"]
    assert not list((tmp_path / "c").glob("*.json"))  # failed ids are retried on the next run


# ------------------------------------------------------------ text, tags, dates
def _meta():
    movies = pd.DataFrame({"movieId": [1, 2, 3, 4], "title": ["A (2000)", "B (2001)", "C (2002)", "D (2003)"],
                           "genres": ["Drama|Comedy", "(no genres listed)", "Action", "Action"]})
    links = pd.DataFrame({"movieId": [1, 2, 3, 4], "tmdbId": [10, 20, 30, np.nan]})
    tmdb = pd.DataFrame({"not_found": [False, False, True], "overview": ["x" * 60, "short", ""],
                         "release_date": ["2000-05-01", "", ""], "genres": [[], [], []],
                         "keywords": [[], ["space", "war"], []]}, index=pd.Index([10, 20, 30], name="tmdbId"))
    return movies, links, tmdb


def test_build_texts_uses_overview_or_fallback():
    movies, links, tmdb = _meta()
    t = content.build_texts(movies, links, tmdb, {3: ["noir"], 4: ["t1", "t2"]}).set_index("movieId")
    assert t.loc[1, "source"] == "overview" and t.loc[1, "text"].startswith("A (2000). Drama, Comedy. xxx")
    assert t.loc[2, "source"] == "fallback" and t.loc[2, "text"] == "B (2001). short. Keywords: space, war"  # no genres listed -> omitted
    assert t.loc[3, "text"] == "C (2002). Action. Tags: noir" and t.loc[3, "has_tags"]  # 404 -> title/genres/tags
    assert t.loc[4, "text"] == "D (2003). Action. Tags: t1, t2"  # no tmdb id at all
    cov = content.coverage(movies, links, tmdb, t.reset_index())
    assert (cov["no_tmdb_id"], cov["tmdb_404"], cov["overview_ok"], cov["use_fallback"]) == (1, 1, 1, 3)
    assert cov["overview_very_short"] == 1 and cov["with_release_date"] == 1


def test_movie_tags_ignores_tags_after_cutoff(tmp_path):
    f = tmp_path / "tags.csv"
    pd.DataFrame({"userId": [1, 2, 3, 1], "movieId": [5, 5, 5, 5], "tag": ["Funny ", "funny", "late", "dark"],
                  "timestamp": [100, 100, 999, 100]}).to_csv(f, index=False)
    tags = content.movie_tags(f, cutoff_ts=500, movie_ids=[5])
    assert tags == {5: ["funny", "dark"]}  # 'late' is after the cutoff; counts by distinct users, lowercased


def test_release_dates_parse():
    movies, links, tmdb = _meta()
    rd = content.release_dates(movies, links, tmdb)
    assert rd[1] == pd.Timestamp("2000-05-01") and rd.isna().sum() == 3


# ------------------------------------------------------------ content retrieval
EMB = np.array([[1, 0], [0.6, 0.8], [0, 1], [-1, 0]], dtype=np.float32)
CAT = np.array([1, 2, 3, 4])


def _fit(rows):
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "label"]).assign(timestamp=0, hour=0)
    m = ContentRetrieval(EMB)
    m.fit(df, CAT)
    return m


def test_content_profile_cosine_hand_checked():
    m = _fit([(1, 1, 1), (2, 1, 1), (2, 3, 1), (3, 1, 0)])
    s = m.score(np.array([1, 2, 3]))
    np.testing.assert_allclose(s[0], [1.0, 0.6, 0.0, -1.0], atol=1e-5)          # likes item 1 -> profile (1,0)
    r = 1 / np.sqrt(2)
    np.testing.assert_allclose(s[1], EMB @ np.array([r, r]), atol=1e-5)          # mean of items 1 and 3, renormalised
    assert s[1][1] == pytest.approx(0.6 * r + 0.8 * r, abs=1e-5)
    assert np.allclose(s[2], 0, atol=1e-5)                                       # no likes -> zero profile


def test_content_requires_aligned_embeddings():
    with pytest.raises(AssertionError):
        ContentRetrieval(EMB[:3]).fit(pd.DataFrame({"user_id": [1], "item_id": [1], "label": [1]}), CAT)


def test_candidate_pool_restricts_ranking():
    m = _fit([(1, 1, 1)])
    fit = pd.DataFrame({"user_id": [1], "item_id": [1], "label": [1]})
    ev = pd.DataFrame({"user_id": [1], "item_id": [3], "label": [1]})
    res = evaluate(m, fit, ev, CAT, ks=(2,), ndcg_k=2, candidate_items=np.array([3, 4]))
    assert res.top[0].tolist()[:2] == [2, 3]  # catalog indices of items 3 and 4; item 2 (more similar) is excluded


# ------------------------------------------------------------ new releases
def _global(n_old=30, n_new=25):
    """items 1..30 old; items 101..125 new (first rated after the cutoff). Items 101 is the one
    user 1 will like; its embedding equals user 1's liked old item."""
    old, new = list(range(1, n_old + 1)), list(range(101, 101 + n_new))
    train = pd.DataFrame([(u, i, 1, 100) for u in (1, 2, 3) for i in old[:10]], columns=["user_id", "item_id", "label", "timestamp"])
    train["hour"] = 0
    val = pd.DataFrame([(1, 101, 1, 200), (1, 102, 0, 200), (2, 102, 1, 200)], columns=["user_id", "item_id", "label", "timestamp"])
    val["hour"] = 0
    return train, val, np.array(old + new)


def test_release_sets_boundaries():
    train, val, cat = _global()
    cutoff = pd.to_datetime(100, unit="s")
    rel = pd.Series(pd.NaT, index=cat, dtype="datetime64[ns]")
    rel[1] = cutoff - pd.DateOffset(months=6)                    # exactly 6 months before: included
    rel[2] = cutoff - pd.DateOffset(months=6) - pd.Timedelta(days=1)  # a day earlier: excluded
    rel[101] = cutoff + pd.Timedelta(seconds=50)                 # after cutoff, before window end (200s): included
    rel[102] = cutoff + pd.Timedelta(days=400)                   # released after the window ends: excluded
    sets = release_sets(train, val, rel)
    assert sorted(sets["release_date"].tolist()) == [1, 101]
    assert sorted(sets["first_rated"].tolist()) == [101, 102]    # in val, never in train
    assert sets["overlap"] == 1 and sets["primary_with_no_train_ratings"] == 1
    assert sorted(sets["release_cold"].tolist()) == [101] and sorted(sets["release_warm"].tolist()) == [1]  # item 1 is in train


def test_new_release_eval_content_finds_cold_item_popularity_cannot():
    train, val, cat = _global()
    emb = np.zeros((len(cat), 4), dtype=np.float32)
    emb[:, 3] = 1.0                                               # every new item: generic direction
    emb[:30] = [0, 1, 0, 0]                                       # old items
    emb[0] = [1, 0, 0, 0]                                         # old item 1: liked by everyone
    emb[list(cat).index(101)] = [1, 0, 0, 0]                      # new item 101 looks just like item 1
    cm = ContentRetrieval(emb); cm.fit(train, cat)
    pop = RandomTies(GlobalPop(), seed=3); pop.fit(train, cat)
    sets = {"release_date": np.array([101, 102]), "first_rated": np.arange(101, 126), "cutoff": None}
    out = evaluate_new_release({"content": RandomTies(cm, seed=3), "global_pop": pop}, train, val, cat, sets, ks=(1, 5))
    t = out[("first_rated", "pool")]["tables"]
    assert out[("first_rated", "pool")]["n_users"] == 2 and out[("first_rated", "pool")]["n_items"] == 25
    assert 0.5 <= t["content"].loc["all_recall@1", "mean"] <= 1.0           # user 1 hit at rank 1 (user 2 only by luck)
    assert t["global_pop"].loc["all_recall@5", "mean"] <= t["content"].loc["all_recall@5", "mean"]
    full = out[("first_rated", "full")]["tables"]
    assert full["global_pop"].loc["all_recall@5", "mean"] == pytest.approx(0.0)  # old items fill the top; new items never surface
