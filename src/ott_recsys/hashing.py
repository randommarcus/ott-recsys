"""Plain file hashing, shared by the preparation script and the unlock guards."""
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_files(paths) -> dict:
    return {str(p): sha256_file(Path(p)) for p in paths}
