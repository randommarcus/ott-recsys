"""Checkpoint 3: load the Module 4b exports into the Zilliz free cluster. DRY RUN BY DEFAULT.

    python scripts/load_zilliz.py                          # local integrity check + read-only preflight (connect, list collections); writes nothing
    python scripts/load_zilliz.py --confirm LOAD-ZILLIZ    # FIRST ZILLIZ WRITE: creates and fills the three collections
    python scripts/load_zilliz.py --confirm LOAD-ZILLIZ --recreate   # drops ONLY our three collections first, then loads again
    python scripts/load_zilliz.py --confirm LOAD-ZILLIZ --reset-probe   # drops and rebuilds ONLY ott_write_probe (1 row); nothing else is touched

Creates exactly movie_items_twotower_id (16,450 rows), movie_items_content (33,392 rows) and ott_write_probe (1 row), AUTOINDEX with metric IP on every
vector field. Uses ZILLIZ_URI and the full-access ZILLIZ_TOKEN from .env (loaded inside Python). Prints only step names, row counts, seconds and error TYPE
names: never the URI, a token or an error message. Refuses to touch a collection with one of our names unless --recreate is given; never touches any other."""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ott_recsys.config import Config
from ott_recsys.zilliz_collections import NAMES, PROBE, SPEC, create, default_client_factory, error_code, load_env, quiet, read_export, row_count, rows

CONFIRM_WORD = "LOAD-ZILLIZ"


def upsert_batches(client, name: str, cols: dict, batch: int, sleep=time.sleep) -> int:
    """Upsert (idempotent, so a retried batch can never duplicate a key) in batches, with back-off on errors such as the free plan's 2 MB/s cap."""
    n = len(cols["item_id"])
    for a in range(0, n, batch):
        data = rows(cols, a, min(a + batch, n))
        for attempt in range(6):
            try:
                with quiet():
                    client.upsert(collection_name=name, data=data)
                break
            except Exception:
                if attempt == 5:
                    raise
                sleep(2 ** attempt)
    return n


def wait_loaded(client, name: str, sleep=time.sleep, tries: int = 60) -> bool:
    for _ in range(tries):
        with quiet():
            st = client.get_load_state(collection_name=name).get("state", "")
        if "Loaded" in (getattr(st, "name", None) or str(st)):  # LoadState enum (Loaded / Loading / NotLoad) or plain text
            return True
        sleep(2)
    return False


def main(argv=None, client_factory=None, out_dir=None, sleep=time.sleep) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--confirm", default=None)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--reset-probe", action="store_true", help="rebuild only the scratch collection ott_write_probe")
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args(argv)
    if args.reset_probe and args.recreate:
        ap.error("--reset-probe and --recreate cannot be combined")
    targets = (PROBE,) if args.reset_probe else NAMES
    recreate = args.recreate or args.reset_probe
    if args.confirm is not None and args.confirm != CONFIRM_WORD:
        print(f"refused: --confirm must be exactly {CONFIRM_WORD}")
        return 2
    write = args.confirm == CONFIRM_WORD
    out_dir = out_dir or Config(dataset_name="ml-25m", sample_users=20000, sample_seed=42).out_dir

    # 1. local integrity check (no network)
    data = {}
    for name in targets:
        try:
            data[name] = read_export(out_dir, name)
        except AssertionError as bad:
            print(f"integrity: FAILED for {name} ({bad.args[0] if bad.args else 'check failed'})")
            return 1
        print(f"integrity: ok  {name:<26s} {len(data[name]['item_id']):>6d} rows, vector fields {', '.join(f'{k}({v})' for k, v in SPEC[name]['vectors'].items())}")

    # 2. read-only preflight
    load_env()
    endpoint, key = os.environ.get("ZILLIZ_URI", "").strip(), os.environ.get("ZILLIZ_TOKEN", "").strip()
    if not endpoint or not key:
        print("connection: no (ZILLIZ_URI and ZILLIZ_TOKEN must be set in .env)")
        return 1
    try:
        with quiet():
            client = (client_factory or default_client_factory)(endpoint, key)
            existing = set(client.list_collections())
    except Exception as exc:
        shown = error_code(exc)
        print(f"connection: no (error type: {shown})")
        return 1
    ours = [n for n in targets if n in existing]
    print(f"connection: yes; collections in the cluster: {len(existing)} (ours already present: {', '.join(ours) if ours else 'none'}; others are never touched)")
    if not write:
        verb = "would REFUSE (ours exist; add --recreate)" if ours and not recreate else ("would rebuild" if ours else "would create")
        print(f"DRY RUN: nothing written. With --confirm {CONFIRM_WORD} it {verb}: " + ", ".join(f"{n} ({SPEC[n]['rows']} rows)" for n in targets))
        return 0
    if ours and not recreate:
        print(f"refused: {', '.join(ours)} already exist; add --recreate to drop and rebuild ONLY our three collections")
        return 1

    # 3. write
    for name in targets:
        t0 = time.time()
        try:
            if name in ours:
                with quiet():
                    client.drop_collection(collection_name=name)
                print(f"dropped: {name} ({'--reset-probe' if args.reset_probe else '--recreate'})")
            with quiet():
                create(client, name)
            n = upsert_batches(client, name, data[name], args.batch, sleep)
            with quiet():
                client.flush(collection_name=name)
                client.load_collection(collection_name=name)
            if not wait_loaded(client, name, sleep):
                print(f"{name}: not loaded after waiting; re-run with --recreate")
                return 1
            with quiet():
                count = row_count(client, name)
        except Exception as exc:
            shown = error_code(exc)
            print(f"{name}: FAILED (error type: {shown}); re-run with --recreate")
            return 1
        ok = count == n == SPEC[name]["rows"]
        print(f"loaded: {name:<26s} {count:>6d} rows (expected {SPEC[name]['rows']}) {'ok' if ok else 'MISMATCH'}  {time.time() - t0:.0f}s", flush=True)
        if not ok:
            return 1
    print("probe reset: ott_write_probe rebuilt with its one original row; no other collection touched" if args.reset_probe
          else "load complete: 3 collections, AUTOINDEX (metric IP), loaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
