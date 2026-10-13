"""Checks on the committed IEEE paper, and on the claims in its source.

Three kinds, deliberately separated by what they need:

* **Structural**, reading the committed PDF: page count, sections, no
  unresolved cross-references. These run in a clone with no LaTeX installed,
  which is most of the point of committing a generated artifact.
* **Source integrity**, reading the `.tex`: every `\\cite` resolves and every
  bibliography entry is cited. Both directions matter -- a dangling citation
  renders as `[?]`, and an uncited reference in a four-page paper is padding.
* **Rebuild**, which needs a toolchain and skips without one.

What these cannot check is whether a number in the paper is true. That is what
`test_the_papers_repository_statistics_match_the_repository` is for: the
figures the paper quotes about its own repository are re-derived here, so the
paper cannot quietly go stale as the suite grows.
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import pytest

pypdf = pytest.importorskip("pypdf")

PROJECT = Path(__file__).parent.parent
DOCS = PROJECT / "docs"
PAPER = DOCS / "churnlens-ieee-report.pdf"
SOURCE = DOCS / "churnlens-ieee.tex"


@pytest.fixture(scope="module")
def pages() -> list[str]:
    if not PAPER.exists():
        pytest.fail(
            f"{PAPER.name} is missing -- it is a committed artifact; "
            "run `python docs/generate_ieee_report.py`"
        )
    return [page.extract_text() for page in pypdf.PdfReader(str(PAPER)).pages]


@pytest.fixture(scope="module")
def text(pages: list[str]) -> str:
    """Extracted text with runs of whitespace collapsed.

    A two-column layout wraps mid-sentence, so a phrase that reads as one line
    in the source arrives with a newline through the middle of it. Collapsing
    whitespace is what makes a probe for a sentence meaningful; without it the
    assertions silently become assertions about line breaks.
    """
    return " ".join(" ".join(pages).split())


@pytest.fixture(scope="module")
def source() -> str:
    return SOURCE.read_text(encoding="utf-8")


def test_the_paper_is_three_to_four_pages(pages):
    """The brief was a 3-4 page paper, and length is the easiest thing to let
    slip. Four pages in IEEEtran two-column is roughly 3,500 words."""
    assert 3 <= len(pages) <= 4, len(pages)


def test_every_required_section_is_present(text):
    """IEEEtran sets section titles in small caps, which extract as uppercase
    with the spaces collapsed ('BACKGROUND ANDRELATEDWORK'), so the assertions
    are on uppercase fragments rather than on the title as written."""
    for fragment in (
        "NTRODUCTION",
        "ACKGROUND",
        "RCHITECTURE",
        "MPLEMENTATION",
        "ERIFICATION",
        "VALUATION",
        "ISCUSSION",
        "ONCLUSION",
        "EFERENCES",
    ):
        assert fragment in text, fragment

    assert "Abstract" in text
    assert "Index Terms" in text, "IEEEkeywords should render as Index Terms"


def test_no_unresolved_cross_references(text):
    """A `??` in the output is LaTeX saying a \\ref found no \\label -- the
    classic one-pass artifact, and invisible unless the text is read back."""
    assert "??" not in text
    assert "[?]" not in text, "an unresolved citation rendered"


def test_the_honest_sections_are_actually_there(text):
    """The paper is a portfolio document, which is exactly why the awkward
    parts have to survive into the PDF rather than being quietly cut."""
    assert "Threats to Validity" in text or "HREATS" in text
    for claim in (
        "synthetic",
        "attenuat",
        "not shown",
        # Stops before "anyone": LaTeX hyphenates across the line break and the
        # extracted text really does contain "any- one". Probing a phrase in a
        # justified two-column PDF means probing a short one.
        "never been used by any",
    ):
        assert claim in text, claim


def test_the_defect_table_reached_the_pdf(text):
    """The evaluation rests on the defect table; a float that silently went
    missing would gut the section it belongs to."""
    assert "Root cause" in text
    assert "Caught by" in text
    for row in ("perfect target separator", "frozen", "tolerance", "slider"):
        assert row in text, row


def test_every_citation_resolves_and_every_reference_is_cited(source):
    """Both directions, because they fail differently.

    An unresolved `\\cite` renders as `[?]` and is caught above as well. An
    uncited `\\bibitem` renders as a numbered entry nothing points at, which in
    a four-page paper reads as padding and in a worse case is a reference that
    was never actually used.
    """
    cited = set()
    for match in re.finditer(r"\\cite\{([^}]*)\}", source):
        cited.update(key.strip() for key in match.group(1).split(","))
    listed = {match.group(1) for match in re.finditer(r"\\bibitem\{([^}]*)\}", source)}

    assert cited <= listed, f"cited but not listed: {sorted(cited - listed)}"
    assert listed <= cited, f"listed but never cited: {sorted(listed - cited)}"
    assert len(listed) >= 10, len(listed)


def test_every_reference_names_a_real_venue(source):
    """A weak check against a fabricated citation -- the thing that would most
    damage this document's credibility.

    It cannot verify a source exists; it can insist each entry carries the
    author, title, venue and year a reader needs to go and look, which is what
    an invented reference usually lacks.
    """
    entries = re.findall(r"\\bibitem\{[^}]*\}(.*?)(?=\\bibitem|\\end\{thebibliography\})",
                         source, re.DOTALL)
    assert entries

    for entry in entries:
        body = " ".join(entry.split())
        assert re.search(r"(19|20)\d{2}", body), f"no year: {body[:60]}"
        assert "``" in body or r"\emph" in body, f"no title: {body[:60]}"
        assert len(body) > 60, f"suspiciously thin entry: {body}"


def test_the_papers_repository_statistics_match_the_repository(source):
    """The paper quotes figures about its own repository. Re-derived here.

    Every number in a portfolio document is a liability if nothing checks it,
    and these are the ones that go stale first: they change whenever a test or
    a module is added. If this test fails, the paper needs updating and
    rebuilding -- which is the correct outcome, not an inconvenience.
    """
    # The .tex wraps, so collapse whitespace before matching a phrase.
    flat = " ".join(source.split())
    quoted_functions = int(
        re.search(r"suite contains ([\d,]+) test functions", flat).group(1).replace(",", "")
    )
    quoted_modules = {
        "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    }[re.search(r"in (\w+) modules", flat).group(1)]

    test_files = sorted((PROJECT / "tests").glob("test_*.py"))
    functions = sum(
        1
        for path in test_files
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )

    assert quoted_modules == len(test_files), (quoted_modules, len(test_files))
    assert quoted_functions == functions, (quoted_functions, functions)

    quoted_source_lines = int(
        re.search(r"holds ([\d{,}]+) lines of application", flat)
        .group(1)
        .replace("{,}", "")
        .replace(",", "")
    )
    actual = sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for root in ("src", "dashboard")
        for path in sorted((PROJECT / root).glob("*.py"))
    )
    assert quoted_source_lines == actual, (quoted_source_lines, actual)


def test_rebuilding_the_paper_is_byte_identical():
    """Reproducibility, same requirement as the architecture report.

    pdfTeX stamps the build time in unless SOURCE_DATE_EPOCH is pinned, so
    without this the committed PDF changes on every rebuild and its history
    says nothing. Both drivers were checked by hand and agree byte for byte;
    this asserts whichever one the machine has.
    """
    builder = pytest.importorskip("docs.generate_ieee_report")
    if builder.toolchain() is None:
        pytest.skip("no LaTeX toolchain; the committed PDF is still checked above")

    rebuilt = builder.build_report(DOCS / "churnlens-ieee.rebuild.pdf")
    try:
        assert hashlib.sha256(rebuilt.read_bytes()).hexdigest() == hashlib.sha256(
            PAPER.read_bytes()
        ).hexdigest(), (
            "the committed paper differs from a fresh build -- either the .tex "
            "changed without the PDF being rebuilt, or the build has become "
            "non-deterministic"
        )
    finally:
        rebuilt.unlink(missing_ok=True)
