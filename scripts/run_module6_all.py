"""One launch for Module 6: build features (if missing), then the ladder + report. Resumable.

  OTT_DATASET=ml-25m OTT_SAMPLE_USERS=20000 OTT_SAMPLE_SEED=42 \
    setsid nohup .venv/bin/python -I scripts/run_module6_all.py >> logs/module6.log 2>&1 &"""
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
stamp = lambda m: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [driver] {m}", flush=True)
run = lambda s: subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "guarded_run.py"), str(ROOT / "scripts" / s)], cwd=ROOT, env=env).returncode
t0 = time.time()
stamp(f"start: {cfg.run_name}")
if not (cfg.out_dir / "ranker" / "features.npz").exists():
    stamp("build features")
    if rc := run("build_ranker_features.py"):
        stamp(f"DRIVER FAILED: build_ranker_features.py exit {rc} ({'memory guard' if rc in (42, 43) else 'error'})"); sys.exit(rc)
stamp("ladder + report")
if rc := run("run_module6.py"):
    stamp(f"DRIVER FAILED: run_module6.py exit {rc} ({'memory guard' if rc in (42, 43) else 'error'})"); sys.exit(rc)
stamp(f"DRIVER ALL DONE in {(time.time() - t0) / 60:.0f} min")
