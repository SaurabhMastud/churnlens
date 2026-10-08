"""Cohort retention checks, against the survival function the data came from.

Retention at tenure t implied by the hazard model is the product of
(1 - monthly hazard) over months 0..t-1, averaged across the subscribers in
scope. That is a real prediction, so the matrix can be checked against it
rather than merely inspected for plausibility.

The comparison is made over the *same* subscriber set the curve pools, i.e.
cohorts with exposure at that tenure -- comparing a pooled estimate to a
prediction averaged over a different population would be its own bug.
"""
from __future__ import annotations

import math

import duckdb
import pytest

from src import cohorts, store

BUILD = dict(count=6000, window_months=18, seed=67)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("cohorts") / "churnlens.duckdb"
    store.build_store(store=path, **BUILD)
    return path


@pytest.fixture(scope="module")
def con(built):
    with duckdb.connect(str(built), read_only=True) as connection:
        yield connection


@pytest.fixture(scope="module")
def planted(con):
    """Coefficients plus each subscriber's cohort, plan and latent level.

    A test may read both truth tables; an analysis may not.
    """
    truth = dict(con.execute("select term, value from hazard_truth").fetchall())
    subscribers = con.execute(
        """
        select s.signup_month, s.plan, t.engagement
        from subscribers s
        join subscriber_truth t using (subscriber_id)
        """
    ).fetchall()
    last_month = con.execute(
        """
        select max(signup_month + tenure_months)
        from subscribers join subscription_months using (subscriber_id)
        """
    ).fetchone()[0]
    return truth, subscribers, last_month


def _hazard(truth: dict, plan: str, tenure: int, engagement: float) -> float:
    logit = (
        truth["intercept"]
        + truth[f"plan_{plan}"]
        + truth["tenure_log1p"] * math.log1p(tenure)
        + truth["engagement"] * engagement
    )
    return 1.0 / (1.0 + math.exp(-logit))


def _planted_retention(planted, tenure: int) -> tuple[float, int]:
    """Mean planted survival to `tenure`, over the cohorts exposed at it."""
    truth, subscribers, last_month = planted
    pool = [
        (plan, engagement)
        for cohort, plan, engagement in subscribers
        if cohort + tenure <= last_month
    ]
    survival = [
        math.prod(1.0 - _hazard(truth, plan, k, engagement) for k in range(tenure))
        for plan, engagement in pool
    ]
    return sum(survival) / len(survival), len(pool)


def test_window_comes_from_the_recorded_metadata(con):
    assert cohorts.window_months(con) == BUILD["window_months"]
    assert cohorts.last_month(con) == BUILD["window_months"] - 1


def test_exposure_does_not_shrink_when_subscribers_churn_early(tmp_path):
    """Regression for a real bug. The window used to be inferred as
    max(signup_month + tenure_months) + 1, which is exact only if somebody is
    observed to the final month. This store -- one subscriber, who signed up in
    month 15 of 24 and churned at tenure 4 -- reported a 20-month window, so
    its cohort silently lost four months of exposure, and nothing looked wrong.
    """
    path = tmp_path / "degenerate.duckdb"
    store.build_store(store=path, count=1, window_months=24, seed=21)

    with duckdb.connect(str(path), read_only=True) as con:
        assert cohorts.window_months(con) == 24

        signup = con.execute("select signup_month from subscribers").fetchone()[0]
        reached = max(r["tenure"] for r in cohorts.retention_matrix(con))
        # The cohort's exposure is set by the window, not by how long its one
        # member happened to last.
        assert reached == 24 - 1 - signup


def test_window_lookup_fails_loudly_on_a_store_without_metadata(tmp_path):
    """An older store has no run_meta. Better a clear error than a default that
    quietly mis-sizes every cohort's exposure."""
    path = tmp_path / "no_meta.duckdb"
    store.build_store(store=path, count=50, window_months=6, seed=5)
    with duckdb.connect(str(path)) as con:
        con.execute("delete from run_meta")

    with duckdb.connect(str(path), read_only=True) as con:
        with pytest.raises(ValueError, match="run_meta"):
            cohorts.window_months(con)


def test_every_cohort_starts_at_full_retention(con):
    matrix = cohorts.retention_matrix(con)

    at_zero = [r for r in matrix if r["tenure"] == 0]
    assert len(at_zero) == BUILD["window_months"], "one cohort per signup month"
    for row in at_zero:
        assert row["retention"] == 1.0
        assert row["retained"] == row["cohort_size"]


def test_unobservable_cells_are_absent_rather_than_zero(con):
    """The cell a cohort has not had time to reach is unknown, not empty. If it
    were emitted as 0% the curve would decay faster than the truth purely
    because young cohorts were counted as total losses.
    """
    matrix = cohorts.retention_matrix(con)
    last_month = BUILD["window_months"] - 1

    present = {(r["cohort"], r["tenure"]) for r in matrix}
    for cohort, tenure in present:
        assert cohort + tenure <= last_month, (cohort, tenure)

    # And the matrix is the full triangle -- nothing observable is missing.
    expected = {
        (cohort, tenure)
        for cohort in range(BUILD["window_months"])
        for tenure in range(last_month - cohort + 1)
    }
    assert present == expected

    assert not [r for r in matrix if r["retention"] == 0.0]


def test_retention_never_increases_within_a_cohort(con):
    """Subscribers do not come back in this model, so a rise would mean the
    retained count is being computed over the wrong risk set."""
    by_cohort: dict[int, list[tuple[int, float]]] = {}
    for row in cohorts.retention_matrix(con):
        by_cohort.setdefault(row["cohort"], []).append(
            (row["tenure"], row["retention"])
        )

    for cohort, cells in by_cohort.items():
        series = [retention for _, retention in sorted(cells)]
        assert series == sorted(series, reverse=True), (cohort, series)


def test_the_curve_pools_exactly_the_cells_the_matrix_shows(con):
    """Both come from one SQL definition of observability; this asserts they
    agree, so the printed table and the curve can never tell different
    stories about which cohorts were counted.
    """
    matrix = cohorts.retention_matrix(con)
    curve = {r["tenure"]: r for r in cohorts.retention_curve(con)}

    from collections import Counter

    cells_per_tenure = Counter(r["tenure"] for r in matrix)
    assert set(curve) == set(cells_per_tenure)
    for tenure, count in cells_per_tenure.items():
        assert curve[tenure]["cohorts"] == count
        assert curve[tenure]["at_risk"] == sum(
            r["cohort_size"] for r in matrix if r["tenure"] == tenure
        )
        assert curve[tenure]["retained"] == sum(
            r["retained"] for r in matrix if r["tenure"] == tenure
        )


def test_the_matrix_reconciles_with_the_month_rows(con):
    """The same fact, counted two independent ways, must agree in every cell.

    The matrix is built from `subscribers.churn_month` and the recorded window.
    `subscription_months` carries one row per observed subscriber-month, written
    by a different part of the generator. Retention at tenure t and "has a month
    row at tenure t" are the same statement, so a disagreement means the
    retention convention and the month rows have drifted apart -- exactly the
    off-by-one the module docstring warns about (a churner counted as retained
    at tenure c + 1 instead of c), which no single-source test can see because
    both the matrix and the curve would move together.

    Asserted cell by cell rather than on the totals: a cell shifted one tenure
    earlier and another one later cancels in a total.
    """
    matrix = {(r["cohort"], r["tenure"]): r["retained"] for r in cohorts.retention_matrix(con)}
    month_rows = {
        (cohort, tenure): n
        for cohort, tenure, n in con.execute(
            """
            select s.signup_month, m.tenure_months, count(*)
            from subscribers s join subscription_months m using (subscriber_id)
            group by 1, 2
            """
        ).fetchall()
    }

    assert set(matrix) == set(month_rows), "the two sources disagree on which cells exist"
    assert matrix == month_rows


def test_the_naive_churn_rate_is_exposure_biased_and_the_matrix_is_not(con):
    """Why this table exists at all, stated as a measurement.

    "Churn rate" as usually computed -- churned subscribers over all
    subscribers -- is not a rate of anything, because the denominator mixes
    populations observed for wildly different lengths of time. The cohort that
    signed up in the first month of the window has had the whole window to
    churn in; the last one has had a month.

    So the naive number is strongly correlated with signup month, and if those
    cohorts also differed in some business-relevant way (an acquisition channel
    ramping up, a price change) the artifact would be read as its effect.
    Retention at a *fixed tenure* compares like with like and shows no such
    gradient, which is the whole argument for the matrix over the headline.

    This test is here rather than in a comment because the claim is empirical
    and the numbers are the point.
    """
    naive_spread, naive_corr = con.execute(
        """
        with by_cohort as (
            select signup_month,
                   avg(case when censored then 0.0 else 1.0 end) as churn_rate
            from subscribers
            group by signup_month
        )
        select max(churn_rate) - min(churn_rate), corr(churn_rate, signup_month)
        from by_cohort
        """
    ).fetchone()

    # Exposure alone moves the naive rate across a wide band, almost perfectly
    # ordered by signup month. On the committed 24-month store it runs 27.4%
    # down to 5.5% with r = -0.86.
    assert naive_spread > 0.10, naive_spread
    assert naive_corr < -0.7, naive_corr

    matrix = cohorts.retention_matrix(con)
    for tenure in (1, 3):
        cells = sorted(
            (r["cohort"], r["retention"]) for r in matrix if r["tenure"] == tenure
        )
        retention = [value for _, value in cells]
        spread = max(retention) - min(retention)
        # Same quantity, measured at equal tenure: a few points of binomial
        # noise, and no gradient in signup month left to mistake for a trend.
        assert spread < 0.10, (tenure, spread)

        cohort_index = [cohort for cohort, _ in cells]
        assert abs(_pearson(cohort_index, retention)) < 0.5, (tenure, cells)


def _pearson(xs: list[float], ys: list[float]) -> float:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    spread = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return cov / spread


@pytest.mark.parametrize("tenure", [1, 3, 6, 10])
def test_cohorts_agree_at_a_given_tenure_because_no_cohort_effect_was_planted(
    con, tenure
):
    """The hazard depends on plan, tenure and engagement -- not on signup month.

    So retention at a fixed tenure should differ between cohorts only by
    sampling noise, and a cohort sitting well outside that is a real finding:
    either the generator leaked a calendar effect, or the matrix is indexing
    cohorts against the wrong tenure column (a classic retention bug, where
    tenure is computed from the calendar month and every cohort is shifted by
    its own signup month). The curve alone cannot show this -- pooling averages
    the shift away.

    Bound is 4.5 binomial se on the pooled rate, with the cohort size as n.
    Both halves of that choice were measured, not assumed. Noise: the worst |z|
    over every cell came out at 1.9 / 2.5 / 2.7 on seeds 67 / 11 / 404, so 4.5
    leaves headroom for the ~70 comparisons this makes. Sensitivity: simulating
    the calendar-vs-tenure bug -- cohort k reading its tenure k+1 cell -- puts
    the older cohorts at 10-12 se, far outside the bound.

    What this test does *not* catch is a shift applied uniformly to every
    cohort, which only moves a cell ~2.5 se here because it has no cohort to
    disagree with. That one is the planted-survival test's job, which compares
    against the hazard rather than within the table.
    """
    pooled = next(
        r["retention"] for r in cohorts.retention_curve(con) if r["tenure"] == tenure
    )
    cells = [r for r in cohorts.retention_matrix(con) if r["tenure"] == tenure]
    assert len(cells) >= 5, "too few exposed cohorts at this tenure to say anything"

    for cell in cells:
        se = math.sqrt(pooled * (1.0 - pooled) / cell["cohort_size"])
        z = abs(cell["retention"] - pooled) / se
        assert z < 4.5, (cell, pooled, round(z, 2))


@pytest.mark.parametrize("tenure", [1, 2, 3, 5, 8, 12])
def test_pooled_retention_matches_the_planted_survival_curve(con, planted, tenure):
    """The day-2 result: the retention table recovers the survival function
    implied by the hazard the data was drawn from.

    Tolerance is 4 binomial standard errors on the pool at that tenure, which
    widens as the pool thins -- the honest shape, since a tenure backed by six
    cohorts genuinely carries more sampling noise than one backed by eighteen.
    """
    expected, n = _planted_retention(planted, tenure)
    observed = next(
        r["retention"] for r in cohorts.retention_curve(con) if r["tenure"] == tenure
    )
    se = math.sqrt(expected * (1.0 - expected) / n)

    assert observed == pytest.approx(expected, abs=4 * se), (
        tenure,
        observed,
        expected,
        se,
    )


def test_per_plan_curves_recover_the_planted_plan_ordering(con, planted):
    """Retention by plan has to order the way the coefficients do, at every
    tenure -- not just on average, where one noisy tenure could hide a sign
    error."""
    truth, _, _ = planted
    ranked = sorted(
        (term for term in truth if term.startswith("plan_")),
        key=lambda term: truth[term],
    )
    best_to_worst = [term.removeprefix("plan_") for term in ranked]

    rows = cohorts.retention_curve_by_segment(con, "plan")
    by_tenure: dict[int, dict[str, float]] = {}
    for row in rows:
        by_tenure.setdefault(row["tenure"], {})[row["plan"]] = row["retention"]

    for tenure, plans in by_tenure.items():
        if tenure == 0:
            continue  # everyone is retained at tenure 0 by construction
        observed = [plans[p] for p in best_to_worst]
        assert observed == sorted(observed, reverse=True), (tenure, plans)


def test_the_plan_gap_widens_with_tenure(con, planted):
    """A per-month hazard difference compounds, so the spread between the best
    and worst plan has to grow. A table that only differenced the final month
    would show a flat gap.
    """
    rows = cohorts.retention_curve_by_segment(con, "plan")
    by_tenure: dict[int, dict[str, float]] = {}
    for row in rows:
        by_tenure.setdefault(row["tenure"], {})[row["plan"]] = row["retention"]

    def spread(tenure: int) -> float:
        values = by_tenure[tenure].values()
        return max(values) - min(values)

    assert spread(12) > spread(6) > spread(1)


def test_per_plan_pools_reconcile_with_the_overall_pool(con):
    """Splitting by plan must partition the subscribers, not resample them."""
    overall = {r["tenure"]: r for r in cohorts.retention_curve(con)}
    per_plan = cohorts.retention_curve_by_segment(con, "plan")

    totals: dict[int, list[int]] = {}
    for row in per_plan:
        bucket = totals.setdefault(row["tenure"], [0, 0])
        bucket[0] += row["at_risk"]
        bucket[1] += row["retained"]

    assert set(totals) == set(overall)
    for tenure, (at_risk, retained) in totals.items():
        assert at_risk == overall[tenure]["at_risk"], tenure
        assert retained == overall[tenure]["retained"], tenure


@pytest.mark.parametrize("plan", ["basic", "standard", "premium"])
def test_each_plan_curve_matches_its_own_planted_survival(con, planted, plan):
    """Stronger than the ordering test: each curve is compared to the survival
    function for that plan specifically, so a model that got the ranking right
    with the wrong magnitudes still fails."""
    truth, subscribers, last_month = planted
    tenure = 8

    pool = [
        engagement
        for cohort, row_plan, engagement in subscribers
        if row_plan == plan and cohort + tenure <= last_month
    ]
    survival = [
        math.prod(1.0 - _hazard(truth, plan, k, engagement) for k in range(tenure))
        for engagement in pool
    ]
    expected = sum(survival) / len(survival)
    se = math.sqrt(expected * (1.0 - expected) / len(pool))

    observed = next(
        r["retention"]
        for r in cohorts.retention_curve_by_segment(con, "plan")
        if r["plan"] == plan and r["tenure"] == tenure
    )

    assert observed == pytest.approx(expected, abs=4 * se), (plan, observed, expected)


def test_an_unknown_segment_is_refused():
    """The segment name is interpolated into SQL, so it is validated against an
    allowlist rather than trusted."""
    with pytest.raises(ValueError, match="unknown segment"):
        cohorts._matrix_sql("engagement")
    with pytest.raises(ValueError, match="unknown segment"):
        cohorts.retention_curve_by_segment(None, "subscriber_id; drop table")


def test_zero_filling_unobservable_cells_would_visibly_break_the_curve(con, planted):
    """The bug this module is built to avoid, demonstrated rather than asserted
    in a comment.

    Rebuilding the curve with unreachable cells counted as zero retained --
    the natural result of a `cross join` without the exposure filter -- pulls
    retention far below the planted survival, and the error grows with tenure
    because that is where the most cohorts are unreachable. If this ever stops
    being true, the exposure filter has stopped mattering and these tests are
    no longer guarding anything.
    """
    tenure = 12
    matrix = cohorts.retention_matrix(con)
    expected, _ = _planted_retention(planted, tenure)

    all_cohorts = {r["cohort"]: r["cohort_size"] for r in matrix}
    retained = {r["cohort"]: r["retained"] for r in matrix if r["tenure"] == tenure}

    # Every cohort in the denominator; cohorts that never reached this tenure
    # contribute 0 retained.
    naive = sum(retained.get(c, 0) for c in all_cohorts) / sum(all_cohorts.values())
    honest = next(
        r["retention"] for r in cohorts.retention_curve(con) if r["tenure"] == tenure
    )

    assert naive < honest, (naive, honest)
    assert abs(naive - expected) > 5 * abs(honest - expected), (naive, honest, expected)


def test_render_marks_unreachable_cells_blank(built):
    text = cohorts.render(built, max_tenure=6)

    assert "Signup-cohort retention" in text
    assert "not a month everyone left" in text
    assert "Pooled retention by tenure" in text
    # The last cohort can only be observed at tenure 0, so its row must trail
    # off into blanks rather than showing a number at tenure 6.
    last_row = next(
        line
        for line in text.splitlines()
        if line.strip().startswith(str(BUILD["window_months"] - 1) + " ")
        or line.startswith(f"{BUILD['window_months'] - 1:>6}  ")
    )
    assert last_row.count("%") == 1, last_row


def test_render_explains_itself_when_there_is_no_store(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.store"):
        cohorts.render(tmp_path / "absent.duckdb")


def test_cohorts_cli_prints_the_matrix(built, capsys):
    cohorts.main(["--store", str(built), "--max-tenure", "4"])

    assert "Signup-cohort retention" in capsys.readouterr().out
