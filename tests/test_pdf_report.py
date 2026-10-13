"""Checks on the committed architecture PDF.

Asserting that a non-empty file was written is not a test of a report: it
passes on a PDF full of mangled text, which is exactly what week 01 shipped for
two days (`subscriber_id` rendered as "subscriberid" on every page, and tables
printed as raw pipe characters). So these read the **committed** PDF back,
extract its text, and assert on what a reader would actually see.

Split by what they need:

* Structural checks read the committed file and run anywhere, including a fresh
  clone with nothing installed beyond `pypdf`.
* The rebuild check regenerates the report and compares bytes, which needs
  `fpdf2` and the architecture doc.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pypdf = pytest.importorskip("pypdf")

DOCS = Path(__file__).parent.parent / "docs"
REPORT = DOCS / "churnlens-report.pdf"


@pytest.fixture(scope="module")
def pages() -> list[str]:
    if not REPORT.exists():
        pytest.fail(
            f"{REPORT.name} is missing -- it is a committed artifact; "
            "run `python docs/generate_report.py`"
        )
    reader = pypdf.PdfReader(str(REPORT))
    return [page.extract_text() for page in reader.pages]


@pytest.fixture(scope="module")
def text(pages: list[str]) -> str:
    return "\n".join(pages)


def test_the_report_is_a_plausible_length(pages):
    """A range rather than an exact count: the doc grows as decisions are
    recorded, and pinning the page count would fail on every honest edit. Too
    short means the render dropped content; far too long means something is
    repeating."""
    assert 8 <= len(pages) <= 25, len(pages)


def test_every_section_of_the_architecture_doc_survived_the_render(text):
    for heading in (
        "Why this project",
        "Components",
        "Data flow",
        "The ground-truth contract",
        "Day-by-day plan",
        "Decisions log",
        "Appendix: pinned stack",
        "Appendix: test inventory",
    ):
        assert heading in text, heading


def test_snake_case_identifiers_are_not_mangled(text):
    """Week 01's worst report bug, as a regression test.

    `_` is Markdown emphasis, and stripping it turned every identifier in the
    document into an unsearchable run-on. This document is denser in
    identifiers than that one was, so the check is explicit: the names have to
    appear with their underscores, and the mangled forms must not appear at all.
    """
    for identifier in (
        "subscriber_id",
        "subscription_months",
        "engagement_events",
        "hazard_truth",
        "subscriber_truth",
        "tenure_months",
        "run_meta",
        "plan_effect",
    ):
        assert identifier in text, identifier

    for mangled in ("subscriberid", "hazardtruth", "tenuremonths", "runmeta"):
        assert mangled not in text, mangled


def test_tables_rendered_as_tables_and_not_as_pipe_soup(text):
    """The second week-01 bug. A Markdown table that fell through to the text
    renderer appeared as `| Component | Tool | Role |`, which is both ugly and
    a sign the table API was never reached."""
    assert "| Component |" not in text
    assert "|---|" not in text
    # The content is still there, just drawn as a table.
    assert "Lifecycle generator" in text
    assert "Kaplan" in text


def test_the_data_flow_diagram_survived_the_code_fence(text):
    """The diagram is ASCII inside a fence, drawn with pipes -- which is why
    the renderer checks the fence state *before* the table branch. If that
    order ever flips, the diagram becomes a malformed table and disappears."""
    assert "hazard config (known coefficients)" in text
    assert "quarantined -- tests only, never read by an analysis" in text


def test_the_appendices_report_real_numbers(text):
    """Both appendices are generated from files rather than typed, so they
    cannot drift. This checks they actually contain what they claim to."""
    for package in ("duckdb", "pandas", "scikit-learn", "streamlit", "fpdf2"):
        assert package in text, package
    for module in ("test_cohorts.py", "test_survival.py", "test_dashboard_runs.py"):
        assert module in text, module


def test_the_report_carries_the_repo_link_and_the_week(text):
    assert "github.com/SaurabhMastud/churnlens" in text
    assert "Week 02" in text
    assert "Data Analytics" in text


def test_no_unrendered_markdown_markers_are_left_in_the_text(text):
    """Bold markers and heading hashes should be consumed by the renderer. A
    `**` in the output means a line took the wrong branch."""
    assert "**" not in text
    assert "## " not in text


def test_regenerating_the_report_is_byte_identical():
    """The committed PDF has to be reproducible, or its history is noise.

    fpdf2 stamps a creation timestamp into the file, so without a pinned date
    every regeneration produces a different file and git reports a change where
    the content has none. Week 01 shipped that for two days before anyone
    compared hashes.
    """
    build_report = pytest.importorskip("docs.generate_report").build_report

    rebuilt = build_report(DOCS / "churnlens-report.rebuild.pdf")
    try:
        assert hashlib.sha256(rebuilt.read_bytes()).hexdigest() == hashlib.sha256(
            REPORT.read_bytes()
        ).hexdigest(), (
            "the committed report differs from a fresh render -- either the "
            "architecture doc changed without the PDF being regenerated, or "
            "the render has become non-deterministic again"
        )
    finally:
        rebuilt.unlink(missing_ok=True)
