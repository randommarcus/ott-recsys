"""Check the final-report chapters against the "never" list in reports/final/STYLE.md: no em or en dashes, none of the banned AI-sounding words.

    python scripts/check_report_style.py            # all chapters in reports/final/
    python scripts/check_report_style.py FILE ...   # given files

Prints each hit as file:line: word, then a total. Exit code 1 if anything is found."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "reports" / "final"
DASHES = {"—": "em dash", "–": "en dash", "―": "horizontal bar"}
BANNED = ["delve", "tapestry", "landscape", "realm", "robust", "leverage", "seamless", "seamlessly", "crucial", "pivotal", "vital", "paramount", "comprehensive",
          "nuanced", "intricate", "underscore", "underscores", "showcase", "showcases", "navigate", "navigating", "elevate", "unlock", "unlocks", "harness",
          "foster", "testament", "holistic", "game-changer", "cutting-edge", "state-of-the-art", "transformative", "meticulous", "meticulously", "embark",
          "journey", "ever-evolving", "in today's", "it is worth noting", "it's important to note", "it is important to note", "notably", "furthermore",
          "in conclusion", "in summary", "myriad", "plethora", "stands as", "serves as a", "boasts", "let's", "dive", "deep dive", "key takeaway",
          "valuable insights", "vibrant", "empower", "empowers", "streamline", "synergy", "paradigm", "noteworthy", "commendable", "remarkable",
          "ultimately", "plays a role", "a wide range of", "in the world of"]
SKIP = {"STYLE.md"}  # the style guide lists the banned words on purpose
TITLES_ONLY = {"99_references.md"}  # paper titles must be quoted exactly (one contains "Robust"): only dashes are checked there


def check_text(text: str, dashes_only: bool = False) -> list:
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        for ch, name in DASHES.items():
            if ch in line:
                hits.append((n, name))
        if dashes_only:
            continue
        low = line.lower()
        for w in BANNED:
            if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low):
                hits.append((n, w))
    return hits


def main(argv=None) -> int:
    files = [Path(a) for a in (argv if argv is not None else sys.argv[1:])] or sorted(p for p in FINAL.glob("*.md") if p.name not in SKIP)
    total = 0
    for f in files:
        for n, what in check_text(f.read_text(), dashes_only=f.name in TITLES_ONLY):
            print(f"{f.name}:{n}: {what}")
            total += 1
    print(f"style check: {total} problem(s) in {len(files)} file(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
