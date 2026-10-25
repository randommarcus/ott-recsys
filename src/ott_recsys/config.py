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

    # Optional cap for big datasets: keep only the N most active users (None = all)
    max_users: int | None = (
        int(os.environ["OTT_MAX_USERS"]) if os.getenv("OTT_MAX_USERS") else None
    )

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

    @property
    def dataset_dir(self) -> Path:
        return self.raw_dir / self.dataset_name

    @property
    def out_dir(self) -> Path:
        return self.processed_dir / self.dataset_name
