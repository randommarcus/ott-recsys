"""Download MovieLens (small by default) into data/raw/."""
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASETS = {
    "small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}

def main(name: str = "small") -> None:
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    zip_path = raw / f"ml-{name}.zip"
    print(f"Downloading {DATASETS[name]}")
    urllib.request.urlretrieve(DATASETS[name], zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw)
    zip_path.unlink()
    print("Done:", sorted(p.name for p in raw.iterdir()))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "small")
