"""Compile docs/churnlens-ieee.tex into the IEEE-format paper.

Separate from generate_report.py on purpose: that one renders ARCHITECTURE.md
with fpdf2 and needs no system dependency, while this is a LaTeX document with
its own source. They are different documents for different readers -- the
architecture report is the full written record, this is a four-page paper --
and deriving one from the other gives either a padded paper or a gutted record.

**latexmk is tried first here, and it works**, which is a difference from week
01: that project's notes record latexmk failing before reading the document,
because MiKTeX had no Perl script engine. The same MiKTeX install now runs it
(MiKTeX 25.12, latexmk on PATH), so the preferred driver is used and the
hand-rolled multi-pass loop is kept as the fallback rather than as the only
path. The fallback is exercised by a test that forces it, so it cannot rot into
a branch that has never run.

Reproducibility has two parts, and the second was found the hard way.

1. pdfTeX stamps the build time into the PDF unless SOURCE_DATE_EPOCH is pinned
   (with FORCE_SOURCE_DATE, since pdfTeX otherwise honours it only for
   \\pdfcreationdate). The committed PDF is a generated artifact under version
   control, so it has to be byte-stable or its history is noise.
2. **pdfTeX derives the trailer `/ID` from the build path**, so the same source
   in a different directory gave a PDF that differed in exactly those 32 bytes
   -- identical content, identical length, identical xref offsets. The
   committed paper therefore failed its own byte-comparison test in a fresh
   clone while passing in the directory it was built in, which is the only
   place the claim had ever been checked.

   The first diagnosis was wrong, and is recorded because the correction is the
   useful part: line endings were blamed (`.gitattributes` checks the `.tex`
   out as LF, the local copy was CRLF), the working tree was renormalised, and
   the clone still failed. Comparing the two files byte for byte -- rather than
   theorising about them -- found a 32-byte difference in the trailer and
   nothing else. Fixed in the document preamble with `\pdftrailerid{}`,
   `\pdfinfoomitdate` and `\pdfsuppressptexinfo`, each guarded by
   `\ifdefined` so the source still compiles under an engine without them.
   The LF normalisation was good hygiene and was not the bug.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

DOCS_DIR = Path(__file__).parent
SOURCE = DOCS_DIR / "churnlens-ieee.tex"
OUTPUT = DOCS_DIR / "churnlens-ieee-report.pdf"
BUILD_DIR = DOCS_DIR / "latex-build"

# Matches the architecture report's pinned date (2026-10-13) closely enough to
# be the same build day; the exact value only has to be constant.
SOURCE_DATE_EPOCH = "1792400000"

MAX_PASSES = 3


def toolchain() -> str | None:
    """Which driver is available: latexmk, pdflatex, or neither."""
    for candidate in ("latexmk", "pdflatex"):
        if shutil.which(candidate):
            return candidate
    return None


def _environment() -> dict[str, str]:
    return {
        **os.environ,
        "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH,
        "FORCE_SOURCE_DATE": "1",
    }


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=DOCS_DIR,
        env=_environment(),
        capture_output=True,
        text=True,
    )


def _build_with_latexmk() -> None:
    result = _run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-output-directory={BUILD_DIR.name}",
            SOURCE.name,
        ]
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )


def _build_with_pdflatex() -> None:
    """Bounded multi-pass loop: run again only while the log asks for it.

    Safe for this document specifically -- the bibliography is an inline
    `thebibliography` (no BibTeX pass) and the only cross-references are
    \\label/\\ref, which settle in two passes.
    """
    for _ in range(MAX_PASSES):
        result = _run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                f"-output-directory={BUILD_DIR.name}",
                SOURCE.name,
            ]
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )
        if "Rerun" not in result.stdout:
            break


def build_report(output_path: Path | None = None, driver: str | None = None) -> Path:
    """Compile the paper. Raises CalledProcessError carrying the LaTeX output."""
    output_path = output_path or OUTPUT
    driver = driver or toolchain()
    if driver is None:
        raise RuntimeError(
            "no LaTeX toolchain found -- install MiKTeX or TeX Live, or read the "
            "committed docs/churnlens-ieee-report.pdf, which needs neither"
        )

    BUILD_DIR.mkdir(exist_ok=True)
    if driver == "latexmk":
        _build_with_latexmk()
    elif driver == "pdflatex":
        _build_with_pdflatex()
    else:
        raise ValueError(f"unknown driver {driver!r}; expected latexmk or pdflatex")

    shutil.copyfile(BUILD_DIR / f"{SOURCE.stem}.pdf", output_path)
    return output_path


if __name__ == "__main__":
    path = build_report()
    print(f"wrote {path} via {toolchain()}")
