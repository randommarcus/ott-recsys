"""Build and save the per-user temporal split. Prints sizes only; the test
portion is written to disk but not inspected."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ott_recsys.config import Config
from ott_recsys.data import build_all, build_user_split

cfg = Config()
print(build_user_split(cfg, build_all(cfg)["df"]))
