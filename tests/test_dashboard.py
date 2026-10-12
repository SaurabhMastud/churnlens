"""Dashboard checks: the data the page shows, not the page.

Streamlit's own rendering is not what can be wrong here. What can be wrong is
the reshaping between an analysis function and a chart -- a pivot that fills
unreachable cells with zero, two series plotted on axes that are off by one, a
frame that silently drops a plan -- and all of that is in plain functions that
take a connection and return a DataFrame, so it is testable without a browser.

The app's own startup is covered separately, by actually running it.
"""
from __future__ import annotations

import duckdb
import pytest

from dashboard import app
from src import cohorts, drivers, ltv, store, survival

BUILD = dict(count=4000, window_months=18, seed=67)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("dashboard") / "churnlens.duckdb"
    store.build_store(store=path, **BUILD)
    return path


@pytest.fixture(scope="module")
def con(built):
    with duckdb.connect(str(built), read_only=True) as connection:
        yield connection


def test_importing_the_dashboard_renders_nothing():
    """A module-level `st.title(...)` would make importing the app draw it.

    The contract test imports this module to walk its AST, so an import with
    side effects would mean the test suite was rendering a dashboard; and
    `streamlit run` executes the file as `__main__` either way. Asserted by
    checking the module body holds no bare expression statements beyond its
    docstring -- which is what a stray `st.something(...)` would be.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(app))
    statements = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant)
    ]

    assert not statements, [ast.dump(node)[:80] for node in statements]
    assert isinstance(tree.body[-1], ast.If), "entry point should stay guarded"


def test_the_store_is_opened_read_only(built):
    """The dashboard must not be able to write to the store.

    Streamlit reruns the script on every widget interaction, so a writable
    handle is both unnecessary and a way to lose the store to a lock while
    another process reads it.
    """
    con = app.open_store(built)
    try:
        with pytest.raises(duckdb.Error):
            con.execute("delete from subscribers")
    finally:
        con.close()


def test_a_missing_store_says_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.store"):
        app.open_store(tmp_path / "nothing.duckdb")


def test_unreachable_retention_cells_arrive_as_blanks_not_zeros(con):
    """The day-2 rule has to survive the pivot into a DataFrame.

    `pivot` fills absent cells with NaN, which Streamlit renders as empty. A
    `fillna(0)` anywhere on this path would put the zero-filled table -- the
    one that decays faster than the truth and looks like a finding -- straight
    into the UI.
    """
    frame = app.retention_frame(con, max_tenure=12)
    matrix = cohorts.retention_matrix(con)
    present = {(row["cohort"], row["tenure"]) for row in matrix}

    assert not frame.empty
    blanks = 0
    for cohort in frame.index:
        for tenure in frame.columns:
            value = frame.loc[cohort, tenure]
            if (cohort, tenure) in present:
                assert value == pytest.approx(
                    next(
                        row["retention"]
                        for row in matrix
                        if row["cohort"] == cohort and row["tenure"] == tenure
                    )
                )
            else:
                assert value != value, (cohort, tenure, value)  # NaN
                blanks += 1

    assert blanks > 0, "no unreachable cells in this frame, so nothing was proven"


def test_the_two_curves_are_plotted_on_the_same_convention(con):
    """Cohort retention at t and KM's S(t-1) are the same quantity.

    Plotting S(t) against retention(t) would draw two lines a month apart and
    invite the reader to explain a gap that is purely a convention. The offset
    is applied in `curve_frame`; this checks it landed the right way round by
    comparing against the underlying functions.
    """
    frame = app.curve_frame(con)
    pooled = {row["tenure"]: row["retention"] for row in cohorts.retention_curve(con)}
    km = survival.kaplan_meier(con)

    assert frame.loc[1, "cohort retention"] == pytest.approx(pooled[1])
    assert frame.loc[1, "Kaplan-Meier"] == pytest.approx(km[0]["survival"])
    # And the two lines sit on top of each other, which is the point of showing
    # both: they come from different risk sets.
    overlap = frame.dropna()
    assert not overlap.empty
    assert (overlap["cohort retention"] - overlap["Kaplan-Meier"]).abs().max() < 0.02


def test_the_driver_frame_carries_every_term_with_an_interval(con):
    frame = app.driver_frame(con)
    rows = drivers.design_rows(con)
    fitted = drivers.fit_with_features(rows)

    assert list(frame.index) == ["intercept", *fitted["features"]]
    for term in frame.index:
        assert frame.loc[term, "low"] < frame.loc[term, "coefficient"] < frame.loc[term, "high"]
        assert frame.loc[term, "se"] > 0
    # The attenuated activity coefficient is still the largest single effect,
    # which is what the caption beside the chart claims.
    assert frame.loc["activity", "coefficient"] < frame.loc["plan_premium", "coefficient"]


def test_the_segment_frame_shows_both_cuts_side_by_side(con):
    frame = app.segment_frame(con, tenure=11)

    assert list(frame.columns) == ["baseline cut (honest)", "lifetime cut (look-ahead)"]
    assert len(frame) == 4
    # Both columns rise with activity; the look-ahead one rises further, which
    # is the whole reason it is displayed.
    for column in frame.columns:
        assert list(frame[column]) == sorted(frame[column]), column
    honest = frame["baseline cut (honest)"]
    naive = frame["lifetime cut (look-ahead)"]
    assert (naive.max() - naive.min()) > (honest.max() - honest.min())


def test_the_value_frame_is_the_full_plan_by_segment_grid(con):
    frame = app.value_frame(con, horizon=12)

    assert list(frame.index) == ["basic", "standard", "premium"], "ordered by price"
    assert len(frame.columns) == 4
    assert not frame.isna().any().any()
    # Matches the module it comes from, cell for cell.
    for row in ltv.by_plan_and_activity(con, 12):
        assert frame.loc[row["plan"], row["segment"]] == pytest.approx(row["value"])


def test_the_headline_numbers_come_from_the_store(con):
    facts = app.headline(con)
    subscribers, censored = con.execute(
        "select count(*), count(*) filter (where censored) from subscribers"
    ).fetchone()

    assert facts["subscribers"] == subscribers == BUILD["count"]
    assert facts["still_subscribed"] == censored
    assert facts["window"] == BUILD["window_months"]
    assert facts["final_tenure"] == BUILD["window_months"] - 1
    assert 0.5 < facts["final_survival"] < 1.0
    # Reported as "not reached" in the UI rather than as a number.
    assert facts["median"] is None
