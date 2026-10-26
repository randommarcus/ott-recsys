"""All dataset-specific settings live here. Change this file (or set env vars)
to point the project at a bigger MovieLens version later."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # Which MovieLens folder under data/raw/ to use, e.g. ml-latest-small, ml-25m
    dataset_name: str = os.getenv("OTT_DATASET", "ml-latest-small")
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"

    # Big datasets: random sample of N users (all of their ratings are kept).
    # None = use every user.
    sample_users: int | None = (
        int(os.environ["OTT_SAMPLE_USERS"]) if os.getenv("OTT_SAMPLE_USERS") else None
    )
    sample_seed: int = int(os.getenv("OTT_SAMPLE_SEED", "42"))
    min_sample_interactions: int = 20  # only users with at least this many are sampled

    # Memory guard (scripts/guarded_run.py kills a run that exceeds this)
    mem_limit_gb: float = float(os.getenv("OTT_MEM_LIMIT_GB", "7.5"))

    # Tuning uses a seeded subsample of validation users (final table uses all)
    tune_users: int = 3000

    # Interactions
    positive_threshold: float = 4.0  # rating >= this counts as a positive label

    # Sessions
    session_gap_minutes: int = 30
    burst_window_minutes: int = 10  # "bulk burst" = many ratings inside this window
    burst_min_ratings: int = 30

    # Temporal split: fractions of all interactions, ordered by time
    train_frac: float = 0.70
    val_frac: float = 0.15  # test gets the remainder

    # Per-user temporal split (the main evaluation split)
    min_user_interactions: int = 10
    user_test_frac: float = 0.20  # last 20% of each user's interactions
    user_val_frac: float = 0.10   # the 10% before that

    # Two-tower retrieval (Module 4)
    history_len: int = int(os.getenv("OTT_HISTORY_LEN", "20"))  # last N liked items form the user vector
    embed_dim: int = 64
    temperature: float = 0.07
    batch_size: int = 2048
    learning_rate: float = 1e-3
    id_dropout: float = 0.25  # share of item-ID embeddings replaced by an "unknown" vector per step
    max_epochs: int = 15
    patience: int = 2
    model_seed: int = 0

    @property
    def dataset_dir(self) -> Path:
        return self.raw_dir / self.dataset_name

    @property
    def run_name(self) -> str:
        if self.sample_users:
            return f"{self.dataset_name}_u{self.sample_users}_s{self.sample_seed}"
        return self.dataset_name

    @property
    def out_dir(self) -> Path:
        return self.processed_dir / self.run_name
