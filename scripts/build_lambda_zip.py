"""Module 9, step 2 (local only): assemble the Lambda deployment zip and smoke-test it WITHOUT AWS.

Zip = ott_recsys/serving code (numpy + standard library only) + bundle/ (artifacts from scripts/build_serving_artifacts.py) + the numpy Linux wheel
(downloaded from PyPI, sha256 checked, cached under build/wheels/). Checks the zip against Lambda's documented limits (50 MB zipped for a direct upload,
250 MB unzipped). The smoke test unpacks the zip and runs the handler in an interpreter with NO site-packages (python -I -S), so anything that is not
in the zip or the standard library fails here exactly as it would in Lambda. It never contacts AWS or Zilliz.
Usage: python scripts/build_lambda_zip.py [--bundle DIR] [--out build/lambda.zip] [--no-smoke]"""
import argparse
import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from ott_recsys.config import Config

LIMIT_ZIPPED_MB, LIMIT_UNZIPPED_MB = 50, 250  # AWS Lambda documentation (checked 2026-10-26)
SKIP = ("/tests/", "/f2py/", ".pyi", "__pycache__")


def numpy_wheel(version: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/numpy/{version}/json", timeout=30))
    files = [f for f in meta["urls"] if f["filename"].endswith(".whl") and "-cp312-cp312-" in f["filename"] and "manylinux" in f["filename"] and "x86_64" in f["filename"]]
    f = files[0]
    path = cache / f["filename"]
    if not path.exists():
        urllib.request.urlretrieve(f["url"], path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == f["digests"]["sha256"], "numpy wheel hash mismatch"
    return path


def build(bundle: Path, out: Path, version: str) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    wheel = numpy_wheel(version, out.parent / "wheels")
    entries = {"ott_recsys/__init__.py": (ROOT / "src" / "ott_recsys" / "__init__.py").read_bytes()}
    for p in sorted((ROOT / "src" / "ott_recsys" / "serving").glob("*.py")):
        entries[f"ott_recsys/serving/{p.name}"] = p.read_bytes()
    for p in sorted(bundle.glob("*")):
        entries[f"bundle/{p.name}"] = p.read_bytes()
    with zipfile.ZipFile(wheel) as z:
        for info in z.infolist():
            if not info.is_dir() and not any(s in "/" + info.filename for s in SKIP) and ".dist-info/" not in info.filename:
                entries[info.filename] = z.read(info)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in entries.items():
            z.writestr(zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0)), data, zipfile.ZIP_DEFLATED, 9)
    zipped, unzipped = out.stat().st_size, sum(len(d) for d in entries.values())
    mib = 2 ** 20  # the Lambda documentation says its "MB" means 1,024 KB
    return {"zip": str(out.relative_to(ROOT)), "zipped_bytes": zipped, "zipped_mib": round(zipped / mib, 2), "unzipped_mib": round(unzipped / mib, 1),
            "limit_zipped_mib": LIMIT_ZIPPED_MB, "margin_mib": round(LIMIT_ZIPPED_MB - zipped / mib, 2),
            "fits_direct_upload": zipped / mib < LIMIT_ZIPPED_MB, "fits_unzipped_limit": unzipped / mib < LIMIT_UNZIPPED_MB,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "numpy": version,
            "bundle_parts_mb": {p.name: round(p.stat().st_size / 1e6, 1) for p in sorted(bundle.glob("*"))}}


SMOKE = r'''
import sys, json, base64, time, resource
sys.path.insert(0, sys.argv[1])
t0 = time.perf_counter()
import numpy as np
from pathlib import Path
from ott_recsys.serving import handler as H
from ott_recsys.serving.bundle import Bundle
from ott_recsys.serving.store import MemoryStore
from ott_recsys.serving.search import NumpySearch
assert "pandas" not in sys.modules and "scipy" not in sys.modules and "boto3" not in sys.modules
b = Bundle(Path(sys.argv[1]) / "bundle")
init_ms = (time.perf_counter() - t0) * 1000
store = MemoryStore()
for item in json.loads(sys.stdin.read()):
    store._users[item["uid"]] = {k: (base64.b64decode(v["b"]) if isinstance(v, dict) else v) for k, v in item["rec"].items()}
app = H.App(b, store, store, searcher_factory=lambda: NumpySearch(b))
out = {}
for uid in sorted(store._users):
    t1 = time.perf_counter()
    s, r = H.handle(app, "POST", "/recommend", {"user_id": uid, "seed": 1, "record_impressions": False})
    first_ms = (time.perf_counter() - t1) * 1000
    out[uid] = {"status": s, "movies": [x["movie_id"] for x in r["items"]], "first_ms": round(first_ms, 1)}
    for mode in ("live",):
        s2, r2 = H.handle(app, "POST", "/recommend", {"user_id": uid, "seed": 1, "mode": mode, "record_impressions": False})
        out[uid][mode] = [x["movie_id"] for x in r2["items"]]
    if "M3b" in b.rankers:
        s3, r3 = H.handle(app, "POST", "/recommend", {"user_id": uid, "seed": 1, "model": "M3b", "device": "mobile", "record_impressions": False})
        out[uid]["m3b_ok"] = s3 == 200
print(json.dumps({"init_ms": round(init_ms, 1), "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), "users": out}))
'''


def smoke(zip_path: Path, data_run: Config) -> dict:
    from ott_recsys.serving import handler as H
    from ott_recsys.serving.bundle import Bundle
    from ott_recsys.serving.records import pack
    from ott_recsys.serving.search import NumpySearch
    from ott_recsys.serving.store import MemoryStore
    from ott_recsys.serving_offline import record_at
    sv = data_run.out_dir / "serving"
    a = dict(np.load(sv / "users.npz"))
    pick = [0, 5, 100, int(np.flatnonzero(~a["has_ttvec"])[0])]
    payload = [{"uid": int(a["user_id"][i]), "rec": {k: ({"b": base64.b64encode(v).decode()} if isinstance(v, bytes) else v) for k, v in pack(record_at(a, i)).items()}} for i in pick]
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        r = subprocess.run([sys.executable, "-I", "-S", "-c", SMOKE, tmp], input=json.dumps(payload), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, "smoke run failed: " + r.stderr[-600:]
    got = json.loads(r.stdout)
    b = Bundle(sv / "bundle")
    store = MemoryStore()
    for i in pick:
        store.put_user(record_at(a, i))
    app = H.App(b, store, store, searcher_factory=lambda: NumpySearch(b))
    same = {}
    for i in pick:
        uid = int(a["user_id"][i])
        s, ref = H.handle(app, "POST", "/recommend", {"user_id": uid, "seed": 1, "record_impressions": False})
        same[uid] = got["users"][str(uid)]["movies"] == [x["movie_id"] for x in ref["items"]]
    return {"python_flags": "-I -S (no site-packages)", "lists_identical_to_dev_environment": same, "local_init_ms": got["init_ms"], "local_max_rss_mb": got["max_rss_mb"],
            "local_first_request_ms": {u: v["first_ms"] for u, v in got["users"].items()},
            "m3b_ok": all(v.get("m3b_ok", True) for v in got["users"].values()), "note": "LAPTOP numbers, not Lambda numbers"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    data_run = Config(dataset_name="ml-25m", sample_users=20000, sample_seed=42)
    ap.add_argument("--bundle", default=str(data_run.out_dir / "serving" / "bundle"))
    ap.add_argument("--out", default=str(ROOT / "build" / "lambda.zip"))
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    info = build(Path(args.bundle), Path(args.out), np.__version__)
    print(json.dumps(info, indent=1))
    if not args.no_smoke:
        print(json.dumps(smoke(Path(args.out), data_run), indent=1))
    print(f"done in {time.time() - t0:.0f}s")
