"""One launch for Module 7. Order: bandit simulation first (the new code, so any problem shows early), then the
ranker robustness check (long). Each step under the memory guard; both resumable/independent.

  OTT_DATASET=ml-25m OTT_SAMPLE_USERS=20000 OTT_SAMPLE_SEED=42 \
    setsid nohup .venv/bin/python -I scripts/run_module7_all.py >> logs/module7.log 2>&1 &"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}
stamp = lambda m: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [driver] {m}", flush=True)
run = lambda s: subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "guarded_run.py"), str(ROOT / "scripts" / s)], cwd=ROOT, env=env).returncode
t0 = time.time()
stamp("start")
for script, label in [("run_module7.py", "bandit simulation"), ("run_module7_ranker.py", "ranker robustness check")]:
    stamp(label)
    if rc := run(script):
        stamp(f"DRIVER FAILED: {script} exit {rc} ({'memory guard' if rc in (42, 43) else 'error'})"); sys.exit(rc)
stamp(f"DRIVER ALL DONE in {(time.time() - t0) / 60:.0f} min")
