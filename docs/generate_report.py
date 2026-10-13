"""Render docs/churnlens-report.pdf from ARCHITECTURE.md.

Used by the day-7 close-out; kept as a standalone script so it can also be
re-run any time the architecture doc changes.

**Adapted from week 01 (retailpulse/docs/generate_report.py) rather than
rewritten.** That renderer already paid for three bugs that a fresh version
would have reintroduced, each of which had shipped in a committed PDF:

1. Stripping `_` as Markdown emphasis mangled every snake_case identifier --
   `subscriber_id` rendered as "subscriberid" across six pages. This document
   is even more identifier-dense than week 01's, and it uses `**bold**`
   exclusively, so underscores are left alone and nothing is traded away.
2. Markdown tables rendered as raw pipe text. They go through fpdf2's own
   table API, which measures the columns.
3. The committed PDF was byte-unstable, because fpdf2 stamps the current time
   into the file. The creation date is pinned, so a diff on the PDF means the
   report changed.

What is new here: the appendix lists the pinned stack *and* the test counts per
module, since "the suite backs these claims" is the central assertion of this
project and a reader of the PDF has no `pytest` output to check it against.
"""
from __future__ import annotations

import ast
import re
from datetime import datetime
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

DOCS_DIR = Path(__file__).parent

# The day the week-02 report was produced. A constant rather than "now" so the
# committed PDF is reproducible -- see set_creation_date in build_report().
REPORT_DATE = datetime(2026, 10, 13)

# The core Helvetica/Courier fonts only support latin-1 -- ARCHITECTURE.md's
# em/en dashes, curly quotes, and arrows all fall outside that range.
_UNSUPPORTED = {
    "→": "->", "←": "<-", "↔": "<->",
    "—": "--", "–": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
}


def _sanitize(text: str) -> str:
    for char, replacement in _UNSUPPORTED.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


_TABLE_SEPARATOR = re.compile(r"^:?-+:?$")


_CODE_SPAN = re.compile(r"`([^`]+)`")


def _inline(text: str) -> str:
    """Strip the inline Markdown fpdf2 renders no differently anyway.

    Underscores are deliberately left alone. This document is mostly
    snake_case identifiers -- stripping `_` as emphasis turned session_id into
    "sessionid" and CATALOG_SEED into "CATALOGSEED" throughout the report. The
    doc uses no `_emphasis_` at all (it is all `**bold**`), so there is nothing
    to trade away.

    Asterisks are stripped, but not inside a code span: `assert_*.sql` is a
    glob, not an italic.
    """
    out = []
    last = 0
    for span in _CODE_SPAN.finditer(text):
        out.append(re.sub(r"\*", "", text[last:span.start()]))
        out.append(span.group(1))  # verbatim -- it is code
        last = span.end()
    out.append(re.sub(r"\*", "", text[last:]))
    return "".join(out)


def _split_row(line: str) -> list[str]:
    return [_inline(cell.strip()) for cell in line.strip().strip("|").split("|")]


def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(_TABLE_SEPARATOR.match(cell) for cell in cells if cell)


def _render_table(pdf: FPDF, rows: list[list[str]]) -> None:
    """Draw collected Markdown rows with fpdf2's own table API.

    Hand-rolling column widths was the alternative; fpdf2 already measures text
    and splits the available width, and it keeps the header row styling.
    """
    if not rows:
        return
    width = len(rows[0])
    pdf.set_font("Helvetica", "", 8)
    pdf.ln(2)
    with pdf.table(line_height=5, padding=1) as table:
        for cells in rows:
            # A ragged row would otherwise raise; pad/truncate to the header.
            cells = (cells + [""] * width)[:width]
            row = table.row()
            for cell in cells:
                row.cell(cell)
    pdf.ln(3)


def _add_markdown(pdf: FPDF, text: str) -> None:
    in_code_block = False
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_rows
        _render_table(pdf, table_rows)
        table_rows = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_table()
            in_code_block = not in_code_block
            continue
        if in_code_block:
            # Checked before the table branch: the data-flow diagram is drawn
            # with pipes inside a fence and is not a table.
            # cell() clips instead of wrapping -- code/diagram lines are
            # meant to stay on one line, and some (box-drawing ASCII) have no
            # safe wrap point that multi_cell can break on.
            pdf.set_font("Courier", "", 7)
            pdf.cell(0, 4, line[:180], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            continue
        if stripped.startswith("|"):
            cells = _split_row(stripped)
            if not _is_separator(cells):
                table_rows.append(cells)
            continue
        flush_table()
        if not stripped:
            pdf.ln(2)
            continue
        if stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.ln(2)
            pdf.multi_cell(0, 7, stripped[4:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.ln(3)
            pdf.multi_cell(0, 8, stripped[3:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        elif stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 10, stripped[2:], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(
                0, 6, _inline(stripped), new_x=XPos.LMARGIN, new_y=YPos.NEXT
            )

    # A table ending the document has no following line to trigger the flush.
    flush_table()


def _add_pinned_stack(pdf: FPDF, requirements: Path) -> None:
    """Append requirements.txt as the stack appendix.

    Restating versions in ARCHITECTURE.md was the other option and was rejected:
    a reader of the PDF has no requirements.txt to check it against, and two
    copies of the same pins drift -- the same reason the product seed is
    exported rather than hand-maintained.
    """
    pdf.set_font("Helvetica", "B", 14)
    pdf.ln(3)
    pdf.multi_cell(0, 8, "Appendix: pinned stack", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        6,
        "Verbatim from requirements.txt, so this page cannot drift from what the "
        "project actually installs.",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    rows = [["Package", "Version"]]
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        package, _, version = line.partition("==")
        rows.append([package, version or "unpinned"])
    _render_table(pdf, rows)


def _add_test_counts(pdf: FPDF, tests_dir: Path) -> None:
    """Append the test inventory, counted from the files rather than typed in.

    "The suite backs these claims" is this project's central assertion, and a
    reader of the PDF has no `pytest` output to hold it to. So the appendix
    counts `def test_*` functions per module by parsing the test files -- which
    means it cannot drift from the suite the way a number written into prose
    would.

    Counted as *functions*, and labelled that way: a parametrised function is
    one function and several cases, and `pytest -q` reports the larger number.
    Claiming the case count here would mean either running the suite from the
    renderer or hardcoding a figure that goes stale on the next test added.
    """
    pdf.set_font("Helvetica", "B", 14)
    pdf.ln(3)
    pdf.multi_cell(0, 8, "Appendix: test inventory", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        6,
        "Counted from the test files at render time. One row per module, one "
        "count per test function -- a parametrised function counts once here "
        "and several times in pytest's own total, which is the larger number.",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )

    rows = [["Test module", "Test functions"]]
    total = 0
    for path in sorted(tests_dir.glob("test_*.py")):
        count = sum(
            1
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        )
        total += count
        rows.append([path.name, str(count)])
    rows.append(["total", str(total)])
    _render_table(pdf, rows)


def build_report(output_path: Path | None = None) -> Path:
    output_path = output_path or DOCS_DIR / "churnlens-report.pdf"
    architecture = _sanitize((DOCS_DIR / "ARCHITECTURE.md").read_text(encoding="utf-8"))

    pdf = FPDF()
    # Without a fixed creation date fpdf2 stamps "now" into the file, so the
    # committed PDF comes out byte-different on every regeneration and git shows
    # a change where the content has none. Pinning it means a diff on this file
    # means the report actually changed -- the same reason .gitattributes pins
    # the exported product seed to LF.
    pdf.set_creation_date(REPORT_DATE)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.multi_cell(0, 14, "ChurnLens", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(
        0,
        7,
        "Week 02 - Data Analytics - github.com/SaurabhMastud/churnlens",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(6)

    _add_markdown(pdf, architecture)
    _add_pinned_stack(pdf, DOCS_DIR.parent / "requirements.txt")
    _add_test_counts(pdf, DOCS_DIR.parent / "tests")

    pdf.output(str(output_path))
    return output_path


if __name__ == "__main__":
    path = build_report()
    print(f"wrote {path}")
