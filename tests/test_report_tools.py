"""The final-report tooling: the style checker catches what STYLE.md bans, every written chapter passes it, and the Word builder produces a document."""
from pathlib import Path

from .aws_fakes import load_script

ROOT = Path(__file__).resolve().parents[1]
style, builder = load_script("check_report_style"), load_script("build_report_docx")
CHAPTERS = sorted((ROOT / "reports" / "final").glob("[0-9][0-9]_*.md"))


def test_checker_catches_dashes_and_stock_phrases():
    bad = "This is pivotal — we delve into a robust landscape.\nRanges like 1–2 too.\nIt is worth noting that."
    found = {w for _, w in style.check_text(bad)}
    assert {"em dash", "en dash", "pivotal", "delve", "robust", "landscape", "it is worth noting"} <= found


def test_checker_does_not_flag_ordinary_words():
    ok = "However, the ranker landed 91 ms. I drove the analysis with a hyphenated well-known word and the word 'diverse'."
    assert style.check_text(ok) == []


def test_every_chapter_passes_the_style_check():
    assert CHAPTERS, "no chapters yet"
    for ch in CHAPTERS:
        assert style.check_text(ch.read_text(), dashes_only=ch.name in style.TITLES_ONLY) == [], ch.name


def test_reference_titles_are_exempt_from_words_but_not_from_dashes():
    assert style.check_text("Efficient and Robust Approximate Nearest Neighbor Search", dashes_only=True) == []
    assert style.check_text("pages 235\u2013256", dashes_only=True) == [(1, "en dash")]


def test_word_builder_makes_a_document(tmp_path):
    from docx import Document
    out = tmp_path / "r.docx"
    assert builder.main(["--out", str(out), CHAPTERS[0].name]) == 0
    d = Document(str(out))
    assert any(p.style.name == "Heading 1" for p in d.paragraphs)
    assert len(d.tables) >= 1
