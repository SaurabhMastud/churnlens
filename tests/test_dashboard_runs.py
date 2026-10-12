"""Does the dashboard actually run? Asked properly, because last week it was not.

Week 01 shipped an Airflow DAG that was never once executed, and had to record
that in its limitations. The lesson transfers directly: a component that is
imported but never *run* is untested, and a dashboard is the easiest place in a
project for that to hide, since every other test here can pass while the page
raises on load.

Two things were tried, and the difference between them is the point.

**A headless server probe** -- `streamlit run --server.headless true`, then
`GET /_stcore/health` -- returns `ok` in about three seconds, and proves almost
nothing. Streamlit executes the script per *session*, when a browser connects
over a websocket; with no client attached the script never runs, so a page that
would crash on line one still answers 200 on `/` and `ok` on the health
endpoint. It is a test that the web server started.

**`streamlit.testing.v1.AppTest`** runs the script the way a session does,
collects the elements it produced, and exposes anything it raised. That is the
check worth having, and it is what this file uses: it fails if the app raises,
and asserts on what it actually rendered.

The store comes from `CHURNLENS_STORE`, so this builds its own temporary one
rather than depending on a store being lying around -- which would have made
the test pass or fail on the state of a developer's working directory.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from dashboard import app
from src import store

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).parent.parent / "dashboard" / "app.py")
BUILD = dict(count=1500, window_months=12, seed=67)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Run the app once against a temporary store and return the result.

    A small store (1500 subscribers, 12 months) because this exercises every
    analysis the page shows -- cohort matrix, KM, the fitted model, segments
    and the value grid -- and the point is that they all run, not that they run
    on a lot of rows.
    """
    path = tmp_path_factory.mktemp("dashboard_run") / "churnlens.duckdb"
    store.build_store(store=path, **BUILD)

    previous = os.environ.get(app.STORE_ENV)
    os.environ[app.STORE_ENV] = str(path)
    try:
        yield AppTest.from_file(APP, default_timeout=120).run()
    finally:
        if previous is None:
            os.environ.pop(app.STORE_ENV, None)
        else:
            os.environ[app.STORE_ENV] = previous


def test_the_app_runs_without_raising(run):
    """The assertion the health probe could not make."""
    assert not run.exception, [exception.value for exception in run.exception]


def test_the_page_shows_its_headline_metrics(run):
    labels = [metric.label for metric in run.metric]

    assert "subscribers" in labels
    assert "still subscribed" in labels
    assert "median survival" in labels
    median = next(metric for metric in run.metric if metric.label == "median survival")
    assert median.value == "not reached", (
        "the curve never reaches 50% here, and the page has to say so rather "
        "than print the last observed tenure"
    )


def test_every_analysis_tab_rendered_something(run):
    """Five tabs, and each one has to have produced output.

    An empty tab is the failure mode a smoke test catches and nothing else
    does: the module imports, the frame builds, and the chart silently gets
    nothing because a widget default filtered everything out.
    """
    assert len(run.tabs) == 5
    assert len(run.dataframe) >= 4, "expected a table in most tabs"
    assert len(run.markdown) >= 1


def test_the_page_carries_the_caveats_and_not_just_the_numbers(run):
    """The captions are load-bearing: a retention dashboard without the
    blank-cell and horizon caveats is the kind of page that gets screenshotted
    into a deck and misread."""
    text = " ".join(element.value for element in run.caption)

    assert "never reads the truth tables" in text
    assert "not a month everyone left" in text
    assert "extrapolation" in text


def test_a_missing_store_is_reported_in_the_page_rather_than_raising(tmp_path):
    """A dashboard pointed at nothing should explain itself, not traceback.

    This is the one failure a real user hits first -- cloning the repo and
    running the app before building a store -- so it gets the error element
    rather than an exception.
    """
    previous = os.environ.get(app.STORE_ENV)
    os.environ[app.STORE_ENV] = str(tmp_path / "absent.duckdb")
    try:
        result = AppTest.from_file(APP, default_timeout=60).run()
    finally:
        if previous is None:
            os.environ.pop(app.STORE_ENV, None)
        else:
            os.environ[app.STORE_ENV] = previous

    assert not result.exception
    assert result.error, "expected the page to show an error element"
    assert "src.store" in result.error[0].value


def test_the_horizon_slider_cannot_be_dragged_past_the_data(run):
    """Regression test for a real crash.

    The slider was a hardcoded 3-24 months. On a store with a shorter window --
    this one observes 12 -- dragging it past the end skipped every cell of the
    value grid, which left an empty frame, which made pandas raise
    `KeyError: 'plan'` from the pivot. The page died with an error message
    about a missing column when the actual problem was a horizon outside the
    data.

    Two fixes, both asserted: the slider's maximum now comes from the store's
    own window, and `ltv.by_plan_and_activity` refuses an impossible horizon
    with a message that says so (`tests/test_ltv.py` covers that half).
    """
    slider = run.slider[0]

    assert slider.max == BUILD["window_months"]
    assert slider.value <= BUILD["window_months"]
    assert "extrapolate" in slider.help


def test_moving_the_slider_to_the_window_edge_still_renders(run):
    """And the boundary itself has to work, since it is now reachable."""
    result = run.slider[0].set_value(BUILD["window_months"]).run()

    assert not result.exception, [e.value for e in result.exception]
    assert result.dataframe
