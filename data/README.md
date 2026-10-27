# Data Folder

Nothing here is committed except this note and the empty-folder markers.

- `raw/` gets MovieLens from `scripts/download_movielens.py` (`small` or `25m`).
- `processed/<dataset>/` gets the splits, TMDB cache, embeddings and candidate lists written by the module scripts.

MovieLens belongs to GroupLens Research and may not be redistributed, so download it yourself. TMDB descriptions are fetched with your own API key (`TMDB_API_KEY` in `.env`).
