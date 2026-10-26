"""One launch for Module 5: candidate lists, 3 validation runs, 1 global-cutoff run, then the report.

  OTT_DATASET=ml-25m OTT_SAMPLE_USERS=20000 OTT_SAMPLE_SEED=42 \
    setsid nohup .venv/bin/python -I scripts/run_module5_all.py >> logs/module5.log 2>&1 &
Resumable (finished steps are skipped). Stops at the first failure."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ott_recsys.config import Config

cfg = Config()
env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}


def stamp(m):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [driver] {m}", flush=True)


def run(script, *args) -> int:
    return subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "guarded_run.py"), str(ROOT / "scripts" / script), *args],
                          cwd=ROOT, env=env).returncode


def fail(what, rc):
    stamp(f"DRIVER FAILED: {what} exit {rc} ({'memory guard' if rc in (42, 43) else 'error'})")
    sys.exit(rc)


t_all = time.time()
stamp(f"start: {cfg.run_name}")
if not (cfg.out_dir / "candidates" / "candidates_global.npz").exists():
    stamp("build candidates")
    if rc := run("build_candidates.py"):
        fail("build_candidates.py", rc)
RUNS = [("ncf_main", ["--neg-k", "4", "--pop-frac", "0.5"]), ("ncf_uniform", ["--neg-k", "4", "--pop-frac", "0.0"]),
        ("ncf_k8", ["--neg-k", "8", "--pop-frac", "0.5"])]
for name, a in RUNS:
    if (cfg.out_dir / "ncf" / name / "meta.json").exists():
        stamp(f"skip {name}"); continue
    stamp(f"run {name}")
    if rc := run("train_ncf.py", "--name", name, "--split", "user", *a):
        fail(name, rc)
if not (cfg.out_dir / "ncf" / "ncf_global" / "meta.json").exists():
    ep = int(json.loads((cfg.out_dir / "ncf" / "ncf_main" / "meta.json").read_text())["best_epoch"])
    stamp(f"run ncf_global ({ep} epochs from ncf_main)")
    if rc := run("train_ncf.py", "--name", "ncf_global", "--split", "global", "--epochs", str(ep), "--neg-k", "4", "--pop-frac", "0.5"):
        fail("ncf_global", rc)
stamp("report")
if rc := run("run_module5.py"):
    fail("run_module5.py", rc)
stamp(f"DRIVER ALL DONE in {(time.time() - t_all) / 60:.0f} min")
