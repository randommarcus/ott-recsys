"""AWS CloudShell, one standalone file (standard library + boto3). Creates the two on-demand DynamoDB tables and loads the user records.

Upload this file plus the files made by scripts/export_dynamodb_items.py (manifest.json and users_items_part*.jsonl.gz) into ONE folder, then:

    python3 load_dynamodb.py --region eu-central-1 --create-tables            # creates ott-users and ott-bandit (on-demand), waits until active
    python3 load_dynamodb.py --region eu-central-1 --load                     # verifies each file's sha256 against manifest.json, then writes
    python3 load_dynamodb.py --region eu-central-1 --check                    # reads back 40 sample users

Only the fixed table names ott-users and ott-bandit are ever touched. Uses boto3's default credential chain (the console identity). Prints counts and words
only: no account id, no ARN, no request id, no record contents. Re-running --load is safe (items are overwritten by key); --start-part N resumes."""
import argparse
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

REGIONS = ("eu-central-1", "ap-south-1")
USERS, BANDIT = "ott-users", "ott-bandit"


def code_of(exc) -> str:
    return str((getattr(exc, "response", None) or {}).get("Error", {}).get("Code", "")) or type(exc).__name__


def word_of(exc) -> str:
    c = code_of(exc)
    return "DENIED" if "AccessDenied" in c else f"ERROR({c})"


def decode(line: str) -> dict:
    return {k: (base64.b64decode(v["b"]) if isinstance(v, dict) else v) for k, v in json.loads(line).items()}


def create_tables(session, region) -> int:
    ddb = session.client("dynamodb", region_name=region)
    bad = 0
    for name in (USERS, BANDIT):
        key = "user_id" if name == USERS else "arm_id"
        try:
            ddb.create_table(TableName=name, BillingMode="PAY_PER_REQUEST", AttributeDefinitions=[{"AttributeName": key, "AttributeType": "N"}],
                             KeySchema=[{"AttributeName": key, "KeyType": "HASH"}])
            ddb.get_waiter("table_exists").wait(TableName=name, WaiterConfig={"Delay": 3, "MaxAttempts": 40})
            print(f"  {name:<10s} created (on-demand)")
        except Exception as exc:
            c = code_of(exc)
            shown = "already exists" if c == "ResourceInUseException" else word_of(exc)
            print(f"  {name:<10s} {shown}")
            bad += c != "ResourceInUseException"
    return 1 if bad else 0


def load(session, region, folder: Path, start_part: int = 1) -> int:
    manifest = json.loads((folder / "manifest.json").read_text())
    table = session.resource("dynamodb", region_name=region).Table(USERS)
    total = 0
    for i, ch in enumerate(manifest["chunks"], 1):
        if i < start_part:
            continue
        path = folder / ch["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != ch["sha256"]:
            print(f"  part {i}: REFUSED (the file does not match manifest.json; upload it again)")
            return 1
        n = 0
        try:
            with table.batch_writer(overwrite_by_pkeys=["user_id"]) as bw, gzip.open(path, "rt") as f:
                for line in f:
                    bw.put_item(Item=decode(line))
                    n += 1
        except Exception as exc:
            why = word_of(exc)
            print(f"  part {i}: stopped after {n} items ({why}); resume with --start-part {i}")
            return 1
        total += n
        print(f"  part {i}/{len(manifest['chunks'])}: {n} items written", flush=True)
    print(f"loaded: {total} items into {USERS}")
    return 0 if start_part > 1 or total == manifest["users"] else 1


def check(session, region, folder: Path) -> int:
    manifest = json.loads((folder / "manifest.json").read_text())
    table = session.resource("dynamodb", region_name=region).Table(USERS)
    ids = []
    for ch in manifest["chunks"]:
        with gzip.open(folder / ch["file"], "rt") as f:
            for j, line in enumerate(f):
                if j % max(1, ch["users"] // 10) == 0:
                    ids.append(int(json.loads(line)["user_id"]))
    found = 0
    try:
        for uid in ids[:40]:
            found += "Item" in table.get_item(Key={"user_id": uid})
    except Exception as exc:
        why = word_of(exc)
        print(f"check: {why}")
        return 1
    print(f"check: {found} of {len(ids[:40])} sample users found")
    return 0 if found == len(ids[:40]) else 1


def main(argv=None, session_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--region", required=True, choices=REGIONS)
    ap.add_argument("--dir", default=".", help="folder with manifest.json and the part files")
    ap.add_argument("--start-part", type=int, default=1)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--create-tables", action="store_true")
    g.add_argument("--load", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if session_factory is None:
        import boto3
        session_factory = lambda region: boto3.Session(region_name=region)
    session = session_factory(args.region)
    print(f"region {args.region}, credentials: default chain (no profile)")
    folder = Path(args.dir)
    if args.create_tables:
        return create_tables(session, args.region)
    return load(session, args.region, folder, args.start_part) if args.load else check(session, args.region, folder)


if __name__ == "__main__":
    sys.exit(main())
