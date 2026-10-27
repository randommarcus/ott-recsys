"""Remove this project's Zilliz collections. DRY RUN BY DEFAULT.

    python scripts/zilliz_teardown.py                         # report which of our three collections exist
    python scripts/zilliz_teardown.py --confirm DROP-ZILLIZ   # drop them, then re-check

It can only ever act on these three literal names: movie_items_twotower_id, movie_items_content, ott_write_probe. It never lists collections to decide
what to drop, so any other collection in the cluster is untouched. Uses ZILLIZ_URI and the full-access ZILLIZ_TOKEN from .env (loaded inside Python).
Prints names and exists / absent / unknown(error type) only."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ott_recsys.zilliz_collections import NAMES, default_client_factory, error_code, load_env, quiet

CONFIRM_WORD = "DROP-ZILLIZ"


def state(client, name) -> str:
    try:
        with quiet():
            return "exists" if client.has_collection(collection_name=name) else "absent"
    except Exception as exc:
        shown = error_code(exc)
        return f"unknown({shown})"


def main(argv=None, client_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--confirm", default=None)
    args = ap.parse_args(argv)
    if args.confirm is not None and args.confirm != CONFIRM_WORD:
        print(f"refused: --confirm must be exactly {CONFIRM_WORD}")
        return 2
    drop = args.confirm == CONFIRM_WORD
    load_env()
    endpoint, key = os.environ.get("ZILLIZ_URI", "").strip(), os.environ.get("ZILLIZ_TOKEN", "").strip()
    if not endpoint or not key:
        print("connection: no (ZILLIZ_URI and ZILLIZ_TOKEN must be set in .env)")
        return 1
    try:
        with quiet():
            client = (client_factory or default_client_factory)(endpoint, key)
    except Exception as exc:
        shown = error_code(exc)
        print(f"connection: no (error type: {shown})")
        return 1
    print("DROP MODE" if drop else f"DRY RUN (nothing is changed; add --confirm {CONFIRM_WORD} to drop)")
    states = {n: state(client, n) for n in NAMES}
    for n in NAMES:
        print(f"  {n:<26s} {states[n]}")
    if drop:
        for n in NAMES:
            if states[n] != "absent":
                try:
                    with quiet():
                        client.drop_collection(collection_name=n)
                    print(f"  dropped {n}")
                except Exception as exc:
                    shown = error_code(exc)
                    print(f"  drop {n}: FAILED ({shown})")
        states = {n: state(client, n) for n in NAMES}
        print("  re-check: " + ", ".join(f"{n} {s}" for n, s in states.items()))
    left = sum(s == "exists" for s in states.values())
    unverified = sum(s.startswith("unknown") for s in states.values())
    print(f"{'remaining' if drop else 'existing'}: {left}" + (f"; unverified: {unverified}" if unverified else ""))
    if drop:
        return 0 if left == 0 and unverified == 0 else 1
    return 1 if unverified else 0


if __name__ == "__main__":
    sys.exit(main())
