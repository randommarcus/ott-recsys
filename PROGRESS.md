# Progress Log

## 2026-10-25 — Module 0: Project setup
- Python 3.12 venv (.venv) via uv; system Python is 3.14, which TensorFlow does not support.
- TensorFlow 2.20 (CPU) + Keras 3 verified. Also installed: xgboost, sentence-transformers, pymilvus, boto3.
- Folder structure, .gitignore, .env.example created.
- MovieLens small (100k ratings) downloaded to data/raw/ (not committed; re-run scripts/download_movielens.py).
- Note: WSL reports ~9 GB RAM, not 16 GB. Keep subsets small; heavy jobs to Colab.

## Next
- Module 1: config + data module (time-based split, baselines).
