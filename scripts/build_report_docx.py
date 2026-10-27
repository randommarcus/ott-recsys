"""Assemble the final report as a Word document from the Markdown chapters in reports/final/ (files starting with a two-digit number, in order).

    python scripts/build_report_docx.py                          # -> reports/final/OTT_Recommender_Report.docx
    python scripts/build_report_docx.py --out build/preview.docx 03_data.md   # selected chapters only

Supported Markdown: # / ## / ### headings, paragraphs, **bold**, *italic*, bullet (- ) and numbered (1. ) lists, pipe tables, images
![caption](figures/file.png) with the caption printed under the figure. A4, Times New Roman 12 pt, 1.5 line spacing, 2.54 cm margins, page numbers.
A missing figure file is replaced by a visible placeholder line, so a draft can be built before every figure exists."""
import argparse
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "reports" / "final"
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")


def add_runs(par, text: str):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


def page_number_footer(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


def setup(doc):
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.54))
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Times New Roman", Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)
    for lvl, size in ((1, 16), (2, 13), (3, 12)):
        st = doc.styles[f"Heading {lvl}"]
        st.font.name, st.font.size, st.font.bold = "Times New Roman", Pt(size), True
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    page_number_footer(sec)


def flush_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows if not re.match(r"^\s*\|?\s*:?-{3,}", r)]
    t = doc.add_table(rows=len(cells), cols=len(cells[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(cells):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.paragraphs[0].paragraph_format.line_spacing = 1.0
            add_runs(cell.paragraphs[0], val)
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(10)
                if i == 0:
                    run.bold = True
    doc.add_paragraph()


def add_chapter(doc, path: Path, first: bool):
    lines = path.read_text().splitlines()
    table, para = [], []

    def flush_para():
        if para:
            p = doc.add_paragraph()
            add_runs(p, " ".join(s.strip() for s in para))
            para.clear()

    for line in lines + [""]:
        if line.strip().startswith("|"):
            flush_para()
            table.append(line)
            continue
        if table:
            flush_table(doc, table)
            table = []
        m = re.match(r"^(#{1,3}) (.*)", line)
        img = re.match(r"^!\[(.*)\]\((.*)\)\s*$", line.strip())
        if m:
            flush_para()
            if len(m.group(1)) == 1 and not first:
                doc.add_page_break()
            doc.add_heading(m.group(2), level=len(m.group(1)))
        elif img:
            flush_para()
            fpath = (path.parent / img.group(2)).resolve()
            if fpath.exists():
                doc.add_picture(str(fpath), width=Cm(15.5))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                doc.add_paragraph(f"[figure to be added: {img.group(2)}]")
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            parts = img.group(1).split(" ", 2)
            if len(parts) == 3 and parts[0] == "Figure":
                cap.add_run(f"{parts[0]} {parts[1]} ").bold = True
                cap.add_run(parts[2]).italic = True
            else:
                cap.add_run(img.group(1)).italic = True
        elif re.match(r"^\s*- ", line):
            flush_para()
            add_runs(doc.add_paragraph(style="List Bullet"), line.strip()[2:])
        elif re.match(r"^\s*\d+\. ", line) and not line.startswith("#"):
            flush_para()
            add_runs(doc.add_paragraph(style="List Number"), re.sub(r"^\s*\d+\. ", "", line))
        elif not line.strip():
            flush_para()
        else:
            para.append(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out", default=str(FINAL / "OTT_Recommender_Report.docx"))
    args = ap.parse_args(argv)
    files = [FINAL / f for f in args.files] if args.files else sorted(p for p in FINAL.glob("[0-9][0-9]_*.md"))
    doc = Document()
    setup(doc)
    for i, f in enumerate(files):
        add_chapter(doc, f, first=(i == 0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} ({len(files)} chapter file(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
