"""Build and save the per-user temporal split from the saved global files
(train+val+test of the global split together are every interaction). Prints
sizes only; the test portion is written to disk but not inspected."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from ott_recsys.config import Config
from ott_recsys.data import build_user_split
from ott_recsys.memguard import Stages

cfg = Config()
stages = Stages()
df = pd.concat([pd.read_parquet(cfg.out_dir / f"{n}.parquet") for n in ["train", "val", "test"]],
               ignore_index=True)
stages.done("read interactions")
print(build_user_split(cfg, df))
stages.done("per-user split + save")
