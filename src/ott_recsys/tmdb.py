"""Polite, resumable TMDB fetcher. One JSON file per movie in the cache dir.

SECURITY: the API key is only ever placed in the request params. It is never
logged, never written to the cache, and exceptions are logged by class name
only (their text can contain the full URL, key included)."""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KEEP = ("id", "title", "overview", "tagline", "release_date", "original_language", "runtime", "status")


class RateLimiter:
    """At most `per_sec` request starts per second across all threads."""

    def __init__(self, per_sec: float):
        self.gap, self.next, self.lock = 1.0 / per_sec, 0.0, threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            self.next = max(self.next, now) + self.gap
            delay = self.next - self.gap - now
        if delay > 0:
            time.sleep(delay)


class BadKey(RuntimeError):
    pass


def slim(raw: dict) -> dict:
    """Keep only whitelisted fields (+ genre and keyword names)."""
    out = {k: raw.get(k) for k in KEEP}
    out["genres"] = [g["name"] for g in raw.get("genres", [])]
    out["keywords"] = [k["name"] for k in (raw.get("keywords") or {}).get("keywords", [])]
    return out


def cache_path(cache_dir: Path, tmdb_id: int) -> Path:
    return cache_dir / f"{int(tmdb_id)}.json"


def fetch_one(session, tmdb_id: int, key: str, limiter: RateLimiter, log, retries: int = 6,
              base_url: str = "https://api.themoviedb.org/3/movie/", sleep=time.sleep):
    """Return the slim dict, {"not_found": True}, or None if all retries failed."""
    for attempt in range(retries):
        limiter.wait()
        try:
            r = session.get(f"{base_url}{int(tmdb_id)}", timeout=20,
                            params={"api_key": key, "append_to_response": "keywords"})
        except Exception as e:  # network/TLS errors: log the class only, never str(e)
            log(f"id {tmdb_id}: {type(e).__name__} (attempt {attempt + 1})")
            sleep(min(60, 2 ** attempt))
            continue
        if r.status_code == 200:
            return slim(r.json())
        if r.status_code == 404:
            return {"not_found": True}
        if r.status_code == 401:
            raise BadKey("TMDB rejected the API key (HTTP 401)")
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", 5))
            log(f"id {tmdb_id}: rate limited, waiting {wait:.0f}s")
            sleep(wait + 1)
            continue
        log(f"id {tmdb_id}: HTTP {r.status_code} (attempt {attempt + 1})")
        sleep(min(60, 2 ** attempt))
    return None


def fetch_all(ids, cache_dir: Path, key: str, session_factory, per_sec: float = 12.0, workers: int = 4,
              log=print, max_consecutive_failures: int = 50, sleep=time.sleep) -> dict:
    """Fetch every id not already cached. Resumable: reruns skip cached ids."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [int(i) for i in dict.fromkeys(ids) if not cache_path(cache_dir, i).exists()]
    stats = {"total": len(set(map(int, ids))), "already_cached": len(set(map(int, ids))) - len(todo),
             "ok": 0, "not_found": 0, "failed": 0, "aborted": None}
    log(f"{stats['total']} unique ids; {stats['already_cached']} cached; {len(todo)} to fetch")
    limiter, lock, local = RateLimiter(per_sec), threading.Lock(), threading.local()
    state = {"streak": 0, "stop": False}
    t0 = time.time()

    def work(tmdb_id):
        if state["stop"]:
            return
        if not hasattr(local, "s"):
            local.s = session_factory()
        try:
            res = fetch_one(local.s, tmdb_id, key, limiter, log, sleep=sleep)
        except BadKey as e:
            with lock:
                state["stop"], stats["aborted"] = True, str(e)
            return
        with lock:
            if res is None:
                stats["failed"] += 1
                state["streak"] += 1
                if state["streak"] >= max_consecutive_failures:
                    state["stop"] = True
                    stats["aborted"] = f"{state['streak']} consecutive failures (network down?)"
                return
            state["streak"] = 0
            stats["not_found" if res.get("not_found") else "ok"] += 1
            done = stats["ok"] + stats["not_found"]
            if done % 500 == 0:
                rate = done / (time.time() - t0)
                log(f"progress: {done}/{len(todo)} fetched ({rate:.1f}/s, ETA {(len(todo) - done) / rate / 60:.0f} min)")
        tmp = cache_path(cache_dir, tmdb_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(res))
        tmp.replace(cache_path(cache_dir, tmdb_id))  # atomic: a killed run never leaves a half file

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, todo))
    log(f"finished: {stats}")
    return stats
