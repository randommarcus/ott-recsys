"""Run a script as a child process and kill it if memory gets too high.

    python scripts/guarded_run.py [--limit-gb 7.5] scripts/run_module2.py [args]

Polls the child's resident memory every 0.2s (from a separate process, so it
still works while the child is busy inside a long numpy/scipy call). Exits 42
if it had to stop the run, so nothing keeps swapping and freezes WSL.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ott_recsys import memguard

ap = argparse.ArgumentParser()
ap.add_argument("--limit-gb", type=float, default=float(os.getenv("OTT_MEM_LIMIT_GB", "7.5")))
ap.add_argument("--min-available-gb", type=float, default=0.8)
ap.add_argument("script")
ap.add_argument("rest", nargs=argparse.REMAINDER)
args = ap.parse_args()

t0 = time.perf_counter()
proc = subprocess.Popen([sys.executable, "-I", args.script, *args.rest], cwd=ROOT)
peak = 0.0
try:
    while proc.poll() is None:
        try:
            rss, avail = memguard.rss_gb(proc.pid), memguard.available_gb()
            peak = max(peak, memguard.peak_gb(proc.pid))
        except (FileNotFoundError, ProcessLookupError):
            break
        if rss > args.limit_gb or avail < args.min_available_gb:
            proc.kill()
            proc.wait()
            why = (f"process memory {rss:.2f} GB exceeded the {args.limit_gb} GB limit"
                   if rss > args.limit_gb else f"system free memory fell to {avail:.2f} GB")
            print(f"\n*** MEMORY GUARD TRIPPED after {time.perf_counter() - t0:.0f}s: {why}. "
                  f"Run stopped. Next step: use fewer users (10-15k). ***", flush=True)
            sys.exit(42)
        time.sleep(0.2)
finally:
    if proc.poll() is None:
        proc.kill()

elapsed = time.perf_counter() - t0
print(f"\n[guarded_run] {args.script}: exit {proc.returncode}, wall time {elapsed:.0f}s "
      f"({elapsed / 60:.1f} min), peak RSS {peak:.2f} GB (limit {args.limit_gb} GB)", flush=True)
if peak > args.limit_gb:
    print("*** peak memory exceeded the limit between checks; treat this run as unsafe ***")
    sys.exit(43)
sys.exit(proc.returncode)
