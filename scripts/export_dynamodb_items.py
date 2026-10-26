"""Module 9 (LOCAL ONLY): write the per-user DynamoDB items as gzip'ed JSON-lines chunks that scripts/cloudshell/load_dynamodb.py can load.

Reads data/processed/<run>/serving/users.npz (built by scripts/build_serving_artifacts.py). Contacts nothing. Binary fields are base64 inside the JSON;
counters are plain integers (DynamoDB's Python client rejects floats, and none are used). A manifest lists each chunk's size and sha256 so the loader can
verify what you uploaded.
    python scripts/export_dynamodb_items.py [--chunks 4] [--out build/dynamodb]"""
import argparse
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from ott_recsys.config import Config
from ott_recsys.serving.records import pack
from ott_recsys.serving_offline import record_at


def encode(item: dict) -> str:
    return json.dumps({k: ({"b": base64.b64encode(v).decode("ascii")} if isinstance(v, bytes) else v) for k, v in item.items()}, separators=(",", ":"))


def main(argv=None) -> int:
    run = Config(dataset_name="ml-25m", sample_users=20000, sample_seed=42)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "build" / "dynamodb"))
    ap.add_argument("--users-file", default=str(run.out_dir / "serving" / "users.npz"))
    args = ap.parse_args(argv)
    a = dict(np.load(args.users_file))
    n = len(a["user_id"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0, n, args.chunks + 1).astype(int)
    manifest = {"table": "ott-users", "key": "user_id", "users": int(n), "chunks": []}
    raw_bytes = 0
    for c in range(args.chunks):
        path = out / f"users_items_part{c + 1}.jsonl.gz"
        with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", compresslevel=9, mtime=0) as gz:
            for i in range(edges[c], edges[c + 1]):
                item = pack(record_at(a, i))
                raw_bytes += sum(len(v) if isinstance(v, bytes) else 8 for v in item.values())
                gz.write((encode(item) + "\n").encode())
        manifest["chunks"].append({"file": path.name, "users": int(edges[c + 1] - edges[c]), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest["total_gz_bytes"] = sum(c["bytes"] for c in manifest["chunks"])
    manifest["item_bytes_uncompressed_estimate"] = raw_bytes
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"users": n, "chunks": args.chunks, "total_gz_mib": round(manifest["total_gz_bytes"] / 2 ** 20, 1),
                      "largest_chunk_mib": round(max(c["bytes"] for c in manifest["chunks"]) / 2 ** 20, 1), "items_mib": round(raw_bytes / 2 ** 20, 1),
                      "estimated_write_request_units": int(sum(-(-(sum(len(v) if isinstance(v, bytes) else 8 for v in pack(record_at(a, i)).values())) // 1024) for i in range(n)))}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
