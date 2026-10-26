"""One launch for all Module 4 experiments, each under the memory guard, then the report.

  OTT_DATASET=ml-25m OTT_SAMPLE_USERS=20000 OTT_SAMPLE_SEED=42 \
    setsid nohup .venv/bin/python -I scripts/run_module4_all.py >> logs/module4.log 2>&1 &

Resumable (a run with a meta.json is skipped). Stops at the first failure."""
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
TT = cfg.out_dir / "two_tower"
env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}
PY = sys.executable


def stamp(m):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [driver] {m}", flush=True)


def run(script, *args) -> int:
    return subprocess.run([PY, "-I", str(ROOT / "scripts" / "guarded_run.py"), str(ROOT / "scripts" / script), *args],
                          cwd=ROOT, env=env).returncode


def best_epoch(name) -> int:
    return int(json.loads((TT / name / "meta.json").read_text())["best_epoch"])


USER_RUNS = [("a_both_n20", ["--variant", "both", "--history", "20"]),
             ("a_id_n20", ["--variant", "id", "--history", "20"]),
             ("a_content_n20", ["--variant", "content", "--history", "20"]),
             ("a_both_n50", ["--variant", "both", "--history", "50"]),
             ("a_both_n20_nologq", ["--variant", "both", "--history", "20", "--no-logq"])]
GLOBAL_RUNS = [("g_id_n20", "a_id_n20", ["--variant", "id"]), ("g_content_n20", "a_content_n20", ["--variant", "content"]),
               ("g_both_n20", "a_both_n20", ["--variant", "both"]),
               ("g_both_n20_nologq", "a_both_n20_nologq", ["--variant", "both", "--no-logq"])]
t_all = time.time()
stamp(f"start: {cfg.run_name}; 5 per-user runs, 4 global runs, then the report")
for name, a in USER_RUNS:
    if (TT / name / "meta.json").exists():
        stamp(f"skip {name} (done)"); continue
    stamp(f"run {name}")
    rc = run("train_two_tower.py", "--name", name, "--split", "user", *a)
    if rc:
        stamp(f"DRIVER FAILED: {name} exit {rc} ({'memory guard' if rc in (42, 43) else 'error'})"); sys.exit(rc)
for name, src, a in GLOBAL_RUNS:
    if (TT / name / "meta.json").exists():
        stamp(f"skip {name} (done)"); continue
    ep = best_epoch(src)
    stamp(f"run {name} ({ep} epochs, from {src})")
    rc = run("train_two_tower.py", "--name", name, "--split", "global", "--history", "20", "--epochs", str(ep), *a)
    if rc:
        stamp(f"DRIVER FAILED: {name} exit {rc} ({'memory guard' if rc in (42, 43) else 'error'})"); sys.exit(rc)
stamp("report")
rc = run("run_module4.py")
if rc:
    stamp(f"DRIVER FAILED: run_module4.py exit {rc}"); sys.exit(rc)
stamp(f"DRIVER ALL DONE in {(time.time() - t_all) / 60:.0f} min")
