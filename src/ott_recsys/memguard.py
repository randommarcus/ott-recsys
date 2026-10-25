"""Memory helpers for the guarded runner (scripts/guarded_run.py) and stage logs."""
import time


def _status_kb(pid: int | str, key: str) -> float:
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return float(line.split()[1])
    return 0.0


def rss_gb(pid: int | str = "self") -> float:
    return _status_kb(pid, "VmRSS") / 1024**2


def peak_gb(pid: int | str = "self") -> float:
    return _status_kb(pid, "VmHWM") / 1024**2


def available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024**2
    return float("inf")


class Stages:
    """Print elapsed time and peak memory after each stage."""

    def __init__(self):
        self.t0 = self.last = time.perf_counter()

    def done(self, name: str) -> None:
        now = time.perf_counter()
        print(f"[stage] {name}: {now - self.last:6.1f}s  (total {now - self.t0:6.1f}s, "
              f"peak RSS {peak_gb():.2f} GB)", flush=True)
        self.last = now
