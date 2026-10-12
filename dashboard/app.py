"""Streamlit dashboard over the ChurnLens analyses.

Run it with:

    streamlit run dashboard/app.py

**Nothing happens at import time.** Every Streamlit call lives inside a
function and the entry point is guarded, which matters for two reasons: the
contract test imports this module to walk its AST, and `streamlit run` executes
the file as `__main__`, so a guarded `main()` is both testable and runnable.
Putting `st.title(...)` at module level would have made importing the dashboard
render it.

This module is an **analysis** as far as the ground-truth contract goes, and
`tests/test_contract.py` discovers it automatically -- `dashboard/` was added
to the guard's roots on day 2, before this file existed, precisely because a
dashboard is the thing most likely to reach for a convenient column. It reads
only the analysis-facing tables, through the same functions the CLI reports
use. There is no dashboard-specific SQL here at all: a query written twice is a
query that can disagree with itself, and the version in a dashboard is the one
nobody tests.

Charts are Streamlit's own (`st.line_chart`, `st.dataframe`, `st.bar_chart`).
No plotting library was added: the retention matrix is a table of percentages
rather than a colour heatmap because a gradient-styled DataFrame needs
matplotlib, and a 300 KB dependency for cell shading is not a trade worth
making -- recorded in the decisions log rather than left as a silent omission.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from src import cohorts, drivers, ltv, segments, survival
from src.store import DEFAULT_STORE

TITLE = "ChurnLens - retention analytics on a known churn process"

# Which store the page reads. An environment variable rather than a CLI flag
# because `streamlit run` owns the command line, and a dashboard that can only
# ever open one hardcoded path is awkward both to deploy and to test -- the
# headless test in tests/test_dashboard_runs.py points this at a temporary
# store so it does not depend on a built one lying around.
STORE_ENV = "CHURNLENS_STORE"


def configured_store() -> Path:
    return Path(os.environ.get(STORE_ENV) or DEFAULT_STORE)


def open_store(path: str | Path = DEFAULT_STORE) -> duckdb.DuckDBPyConnection:
    """Read-only connection, so the dashboard cannot modify the store.

    `read_only=True` is not decoration: Streamlit reruns the script on every
    widget interaction, and a writable handle to a file another process may
    also be reading is a way to lose a store to a lock or a truncation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no store at {path} -- run `python -m src.store` to build one first"
        )
    return duckdb.connect(str(path), read_only=True)


def retention_frame(con: duckdb.DuckDBPyConnection, max_tenure: int = 12) -> pd.DataFrame:
    """Cohort retention as a wide cohort x tenure frame, unreachable cells NaN.

    NaN rather than 0 for a cell a cohort could not reach, which is the same
    rule the day-2 module follows and the reason it exists: a zero-filled
    retention table decays faster than the truth and looks like a finding.
    Streamlit renders NaN as an empty cell, so the distinction survives into
    the UI instead of being flattened on the way out.
    """
    matrix = cohorts.retention_matrix(con)
    frame = pd.DataFrame(matrix).pivot(index="cohort", columns="tenure", values="retention")
    return frame.loc[:, [t for t in frame.columns if t <= max_tenure]]


def curve_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Pooled retention and the KM curve on one tenure axis.

    Both estimators in one frame on purpose: they come from different risk sets
    and agreeing is evidence rather than tautology (the test suite checks it to
    a standard error). A dashboard showing only one of them hides the fact that
    there are two ways to compute it.
    """
    pooled = {row["tenure"]: row["retention"] for row in cohorts.retention_curve(con)}
    km = survival.kaplan_meier(con)

    return pd.DataFrame(
        {
            "cohort retention": pd.Series(pooled),
            # KM's S(t-1) is "present at the start of month t", which is what
            # the cohort curve reports at t -- the same offset the cross-module
            # test uses.
            "Kaplan-Meier": pd.Series(
                {row["tenure"] + 1: row["survival"] for row in km}
            ),
        }
    ).sort_index()


def driver_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = drivers.design_rows(con)
    fitted = drivers.fit_with_features(rows)
    errors = drivers.standard_errors(fitted, rows)

    return pd.DataFrame(
        [
            {
                "term": term,
                "coefficient": value,
                "se": errors[term],
                "low": value - 1.96 * errors[term],
                "high": value + 1.96 * errors[term],
            }
            for term, value in (
                {"intercept": fitted["intercept"]} | fitted["coefficients"]
            ).items()
        ]
    ).set_index("term")


def segment_frame(con: duckdb.DuckDBPyConnection, tenure: int = 11) -> pd.DataFrame:
    """Honest and look-ahead segment cuts in one frame, both labelled."""
    honest = segments.survival_at(segments.survival_by_activity(con, "baseline"), tenure)
    naive = segments.survival_at(segments.survival_by_activity(con, "lifetime"), tenure)

    return pd.DataFrame(
        {
            "baseline cut (honest)": pd.Series(honest),
            "lifetime cut (look-ahead)": pd.Series(naive),
        }
    )


def value_frame(con: duckdb.DuckDBPyConnection, horizon: int = 12) -> pd.DataFrame:
    grid = ltv.by_plan_and_activity(con, horizon)
    return (
        pd.DataFrame(grid)
        .pivot(index="plan", columns="segment", values="value")
        .reindex(sorted(ltv.MONTHLY_PRICE, key=ltv.MONTHLY_PRICE.get))
    )


def headline(con: duckdb.DuckDBPyConnection) -> dict:
    """The four numbers worth putting at the top, with their caveats attached."""
    km = survival.kaplan_meier(con)
    subscribers, censored = con.execute(
        "select count(*), count(*) filter (where censored) from subscribers"
    ).fetchone()
    window = cohorts.window_months(con)

    return {
        "subscribers": subscribers,
        "still_subscribed": censored,
        "window": window,
        "final_survival": km[-1]["survival"],
        "final_tenure": km[-1]["tenure"],
        "median": survival.median_survival(km),
    }


def main(store: str | Path | None = None) -> None:
    store = Path(store) if store is not None else configured_store()
    st.set_page_config(page_title="ChurnLens", layout="wide")
    st.title(TITLE)

    try:
        con = open_store(store)
    except FileNotFoundError as missing:
        st.error(str(missing))
        return

    facts = headline(con)
    columns = st.columns(4)
    columns[0].metric("subscribers", f"{facts['subscribers']:,}")
    columns[1].metric("still subscribed", f"{facts['still_subscribed']:,}")
    columns[2].metric(
        f"survival at {facts['final_tenure']}mo", f"{facts['final_survival']:.1%}"
    )
    columns[3].metric(
        "median survival",
        "not reached" if facts["median"] is None else f"{facts['median']} months",
        help="The curve never falls to 50%, so there is no median to report. "
        "Printing the last observed tenure instead would report a number set "
        "by the window length rather than by churn.",
    )

    st.caption(
        "Data is generated from a known hazard model, so every figure here has "
        "a ground-truth counterpart the test suite checks it against. The "
        "dashboard itself never reads the truth tables."
    )

    retention, survival_tab, driver_tab, segment_tab, value_tab = st.tabs(
        ["Cohort retention", "Survival", "Churn drivers", "Segments", "Value"]
    )

    with retention:
        st.subheader("Signup-cohort retention")
        st.caption(
            "A blank cell is a month the cohort has not had time to reach, not "
            "a month everyone left. Zero-filling those cells drags tenure-12 "
            "retention from 82.5% to 40.5% on this dataset."
        )
        st.dataframe(
            retention_frame(con).style.format("{:.1%}", na_rep=""),
            use_container_width=True,
        )
        st.subheader("Two estimators of the same curve")
        st.line_chart(curve_frame(con))

    with survival_tab:
        st.subheader("Kaplan-Meier, with the risk set beside it")
        curve = pd.DataFrame(survival.kaplan_meier(con)).set_index("tenure")
        st.line_chart(curve[["survival", "ci_low", "ci_high"]])
        st.dataframe(
            curve[["at_risk", "events", "censored", "hazard", "survival"]].style.format(
                {"hazard": "{:.2%}", "survival": "{:.1%}"}
            ),
            use_container_width=True,
        )
        test = survival.logrank(con, "plan")
        st.markdown(
            f"**Log-rank across plans:** chi2 = {test['statistic']:.1f} on "
            f"{test['df']} df, critical value {test['critical_05']} at alpha 0.05. "
            "The statistic is the conservative (O-E)^2/E form, so a "
            "non-significant result from it means *not shown*, not *no difference*."
        )

    with driver_tab:
        st.subheader("Fitted hazard coefficients")
        frame = driver_frame(con)
        st.bar_chart(frame["coefficient"])
        st.dataframe(frame.style.format("{:.3f}"), use_container_width=True)
        st.caption(
            "Fitted on observable data only. The engagement effect comes back "
            "attenuated to about 58% of the planted value because the regressor "
            "is a noisy proxy for the latent level -- measurement error, not a "
            "bug, and the test suite measures it."
        )

    with segment_tab:
        st.subheader("Survival by activity segment")
        frame = segment_frame(con)
        st.line_chart(frame)
        st.dataframe(frame.style.format("{:.1%}"), use_container_width=True)
        st.caption(
            "The look-ahead column groups subscribers by activity averaged over "
            "all observed months, which exist only because they had not churned "
            "yet. It inflates the top-to-bottom gap by about 1.5x here."
        )

    with value_tab:
        # Bounded by the window this store actually observed, not by a constant.
        # With a hardcoded 3-24 range the slider could be dragged past the data
        # on a shorter store: every cell was then skipped, the grid came back
        # empty, and the page died with `KeyError: 'plan'` out of a pandas
        # pivot -- an error message about the wrong thing entirely. The module
        # now refuses that horizon explicitly, and the UI no longer offers it.
        window = facts["window"]
        horizon = st.slider(
            "horizon (months)",
            min_value=3,
            max_value=window,
            value=min(12, window),
            help=f"This store observed {window} months. A longer horizon would "
            "have to extrapolate the survival curve, which is a modelling "
            "assumption and not something a slider should make quietly.",
        )
        st.subheader(f"Expected value per subscriber over {horizon} months")
        st.dataframe(
            value_frame(con, horizon).style.format("{:.0f}"),
            use_container_width=True,
        )
        st.caption(
            "Restricted to a stated horizon: survival never reaches zero in the "
            "window, so a lifetime figure would be extrapolation. Prices are an "
            "assumption, and every figure scales linearly in them."
        )

    con.close()


if __name__ == "__main__":
    main()
