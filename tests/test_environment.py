"""Smoke tests: the environment and raw data are set up."""
from pathlib import Path

RATINGS = Path(__file__).resolve().parents[1] / "data" / "raw" / "ml-latest-small" / "ratings.csv"


def test_tensorflow_imports_and_runs():
    import tensorflow as tf
    layer = tf.keras.layers.Dense(4)
    assert layer(tf.ones((1, 3))).shape == (1, 4)


def test_movielens_downloaded():
    assert RATINGS.exists(), "run: python scripts/download_movielens.py"
