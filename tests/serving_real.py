"""Shared by the Module 9 tests that use the real 20,000-user MovieLens 25M artifacts (skipped when they are not on disk)."""
from ott_recsys.config import Config

CFG = Config(dataset_name="ml-25m", sample_users=20000, sample_seed=42)
OUT = CFG.out_dir
HAVE_REAL = (OUT / "test_prep" / "real" / "lists.npz").exists() and (OUT / "ranker" / "features.npz").exists()
