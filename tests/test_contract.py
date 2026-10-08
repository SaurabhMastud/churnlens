"""The ground-truth contract, enforced across *every* analysis module.

Day 1 guarded one module by name, which is how the latent engagement level
stayed readable: the guard existed, it just did not cover the thing that
mattered. Days 3-6 add survival analysis, a driver model and a dashboard, so a
per-module opt-in is one forgotten import away from guarding nothing -- and
forgetting fails silently, in the direction of a better-looking result.

So the module list is discovered, not written down. Any module in a code root
(`src/`, `dashboard/`) that is not on the generator side is treated as an
analysis and must obey the contract. Adding `src/survival.py` on day 3 puts it
under this guard with no edit here; the only way out is to add it to
GENERATOR_MODULES deliberately, which is a visible choice in a diff rather than
an omission.

Discovery is only as wide as its roots, so `dashboard/` is included now rather
than on day 6 -- the Streamlit app reads the same store and can read the answer
just as easily -- and a further test fails if Python turns up in a directory
that is in neither list, which is the general form of that mistake.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

from src import store

PROJECT = Path(__file__).parent.parent
SRC = PROJECT / "src"

# Where analysis code is allowed to live. `dashboard/` is listed because the
# day-6 Streamlit app goes there per the architecture doc, and a guard that
# only scanned `src/` would have missed it completely -- the dashboard reads
# the same store and is every bit as capable of reading the answer.
# A test asserts no Python outside these roots, so a new directory cannot
# become a place the contract does not apply.
CODE_ROOTS = ("src", "dashboard")

# Directories that are not analysis code and are not expected to be.
NON_CODE_DIRS = {"tests", "docs", "data", ".git", "__pycache__", ".pytest_cache"}

# The generator side *owns* the truth -- it draws from the hazard model and
# writes both truth tables. These are the only modules exempt, and naming one
# here is a deliberate, reviewable act.
GENERATOR_MODULES = frozenset({"hazard", "generate", "store"})

# Tables in the store that hold ground truth.
QUARANTINED_TABLES = ("hazard_truth", "subscriber_truth")


def analysis_modules() -> list[str]:
    """Every importable module in a code root that is not on the generator side.

    Returned as dotted import paths (`src.report`, `dashboard.app`) so a module
    outside `src/` is covered on the same footing.
    """
    found = []
    for root in CODE_ROOTS:
        directory = PROJECT / root
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.stem == "__init__" or path.stem in GENERATOR_MODULES:
                continue
            found.append(f"{root}.{path.stem}")
    return found


def code_strings(module) -> list[str]:
    """String literals the module's *code* uses, docstrings excluded.

    A substring scan of the source cannot do this job: modules explain the
    contract in their own docstrings, so prose naming a table would fail the
    check while an actual query is the only thing that matters. SQL lives in
    string literals -- including the fragments of f-strings, which is how the
    bucketed query is assembled.
    """
    tree = ast.parse(inspect.getsource(module))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def imported_modules(module) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_there_is_at_least_one_analysis_module_to_check():
    """Discovery that silently finds nothing would make every test below pass
    for the wrong reason."""
    assert analysis_modules(), "no analysis modules discovered under src/"


@pytest.mark.parametrize("name", analysis_modules())
def test_analysis_module_never_queries_a_truth_table(name):
    module = importlib.import_module(name)

    for table in QUARANTINED_TABLES:
        offenders = [s for s in code_strings(module) if table in s]
        assert not offenders, (name, table, offenders)


@pytest.mark.parametrize("name", analysis_modules())
def test_analysis_module_never_imports_the_hazard_model(name):
    """Reading the coefficients straight from Python is the same breach as
    reading them from the store, and easier to do by accident."""
    module = importlib.import_module(name)

    assert not [m for m in imported_modules(module) if m.startswith("src.hazard")]


@pytest.mark.parametrize("name", analysis_modules())
def test_analysis_module_never_references_a_latent_column(name):
    """Second line behind the structural quarantine: this still fails if
    someone puts a latent column back on an analysis-facing table.

    `\\bengagement\\b` does not match inside `engagement_events`, since `_` is a
    word character, so the observable table is allowed and a bare column
    reference is not.
    """
    module = importlib.import_module(name)
    patterns = [rf"\b{re.escape(field)}\b" for field in store.LATENT_FIELDS]

    for pattern in patterns:
        offenders = [s for s in code_strings(module) if re.search(pattern, s)]
        assert not offenders, (name, pattern, offenders)


def test_the_guards_are_checking_against_reality():
    """Every guard above is a string match, so each one is only worth something
    if it still matches the real schema. Asserted, not assumed."""
    for table in QUARANTINED_TABLES:
        assert f"TABLE {table}" in store.SCHEMA, table
    assert store.LATENT_FIELDS, "no latent fields declared"
    for field in store.LATENT_FIELDS:
        assert re.search(rf"\b{re.escape(field)}\b", store.SCHEMA), field


def test_every_exemption_names_a_module_that_exists():
    """The exemption list is the only way out of the contract, so it must not
    accumulate stale names. A dead entry is how a future module called
    `store_v2` would arrive pre-exempted by a leftover rename.

    Deliberately *not* pinning the analysis list to `["report"]` as well: that
    would fail on every legitimate new module, which is friction rather than
    safety -- a new analysis module is already covered by the parametrized
    guards above, as adding one and watching them fail confirms.
    """
    present = {path.stem for path in SRC.glob("*.py")}

    assert GENERATOR_MODULES <= present, GENERATOR_MODULES - present


def test_the_generator_side_is_exactly_what_it_should_be():
    """Growth on the analysis side is routine; growth on the *exempt* side is
    not, and should never happen quietly. This fails if a module is added to
    GENERATOR_MODULES, which puts the choice in a diff where it can be read.
    """
    assert GENERATOR_MODULES == {"hazard", "generate", "store"}
    assert "src.report" in analysis_modules()


def test_no_analysis_code_lives_outside_the_scanned_roots():
    """The guard is only as wide as its discovery. `dashboard/` is already in
    CODE_ROOTS because the day-6 Streamlit app goes there and scanning only
    `src/` would have missed it entirely -- the dashboard reads the same store
    and can read the answer just as easily.

    This fails if Python appears in some third directory, which is the general
    version of that mistake: a new home for code is a new place the contract
    silently does not apply.
    """
    stray = [
        path.relative_to(PROJECT).as_posix()
        for path in PROJECT.rglob("*.py")
        if path.parent != PROJECT
        and path.parent.name not in CODE_ROOTS
        and not set(path.relative_to(PROJECT).parts) & NON_CODE_DIRS
    ]

    assert not stray, f"Python outside {CODE_ROOTS}: {stray}"


def test_the_contract_check_can_see_queries_that_are_really_there():
    """The guards pass trivially if code_strings is broken, so prove it reads
    the SQL the report genuinely contains."""
    report = importlib.import_module("src.report")
    strings = code_strings(report)

    assert [s for s in strings if "from subscribers" in s]
    assert [s for s in strings if "subscription_months" in s]
    assert [s for s in strings if "engagement_events" in s]
