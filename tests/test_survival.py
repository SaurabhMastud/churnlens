"""Kaplan-Meier checks: the bookkeeping first, the estimate against truth after.

Survival estimates fail in two very different ways, so they get two different
kinds of test. The first is arithmetic -- a risk set that does not balance, a
censored subscriber counted as an event, an interval that claims survival above
1 -- and those are exact assertions with no tolerance. The second is whether
the curve recovers the hazard the data was drawn from, which is a statistical
comparison and needs a stated tolerance and a stated population.

Mixing them produces a test suite that is loose where it could be exact.
"""
from __future__ import annotations

import math

import duckdb
import pytest

from src import store, survival

BUILD = dict(count=6000, window_months=18, seed=67)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("survival") / "churnlens.duckdb"
    store.build_store(store=path, **BUILD)
    return path


@pytest.fixture(scope="module")
def con(built):
    with duckdb.connect(str(built), read_only=True) as connection:
        yield connection


def test_the_risk_set_balances_month_to_month(con):
    """n(t+1) = n(t) - churned(t) - censored(t), exactly, with no slack.

    This is the whole of KM's bookkeeping in one line, and it is the assertion
    that catches the two ways the risk set goes wrong: a censored subscriber
    left in the denominator after their last observed month (retention looks
    worse than it is, since they can never contribute an event again) or
    removed a month too early (the denominator shrinks and the hazard inflates).
    Neither shows up in the shape of the curve.
    """
    table = survival.risk_table(con)
    assert table[0]["tenure"] == 0
    assert [row["tenure"] for row in table] == list(range(len(table)))

    for current, following in zip(table, table[1:]):
        leaving = current["events"] + current["censored"]
        assert current["at_risk"] - leaving == following["at_risk"], current


def test_censoring_counts_reconcile_with_the_subscriber_flags(con):
    """The risk table derives censoring from the month rows (`max(tenure)` per
    subscriber); `subscribers.censored` is set by the generator. Two sources,
    one fact, so they have to agree in total -- and this is why the risk table
    does not simply read the flag, which would make this test vacuous.
    """
    table = survival.risk_table(con)
    flags = con.execute(
        """
        select count(*) filter (where censored),
               count(*) filter (where not censored),
               count(*)
        from subscribers
        """
    ).fetchone()

    assert sum(row["censored"] for row in table) == flags[0]
    assert sum(row["events"] for row in table) == flags[1]
    # Everyone is observed for at least their first month, so tenure 0's risk
    # set is the whole population -- the one place the denominator is known
    # without any censoring logic at all.
    assert table[0]["at_risk"] == flags[2]


def test_survival_is_a_non_increasing_product_of_monthly_survivals(con):
    curve = survival.kaplan_meier(con)

    assert curve[0]["survival"] == pytest.approx(1.0 - curve[0]["hazard"])
    running = 1.0
    for row in curve:
        running *= 1.0 - row["hazard"]
        assert row["survival"] == pytest.approx(running)
        assert 0.0 < row["survival"] <= 1.0

    series = [row["survival"] for row in curve]
    assert series == sorted(series, reverse=True)


def test_intervals_stay_inside_zero_one_and_bracket_the_estimate(con):
    curve = survival.kaplan_meier(con)

    for row in curve:
        assert 0.0 <= row["ci_low"] <= row["survival"] <= row["ci_high"] <= 1.0, row


def test_the_log_log_interval_earns_itself_on_a_small_risk_set(tmp_path):
    """Where the symmetric interval actually breaks, measured rather than claimed.

    `S +- 1.96 se` is the version usually written and it can put the upper
    bound above 1 -- a claim that more than everyone survived. On the main
    6000-subscriber store it never does, because the standard error is tiny, so
    asserting it there would have been a test of nothing. It does break on a
    small risk set: 40 subscribers over 6 months puts the symmetric upper bound
    at 1.023 at tenure 0, while the log-log interval stays at 0.996.

    Worth keeping as a test rather than a comment, since the simplification is
    exactly the kind someone makes on the way past.
    """
    path = tmp_path / "small.duckdb"
    store.build_store(store=path, count=40, window_months=6, seed=3)

    with duckdb.connect(str(path), read_only=True) as con:
        curve = survival.kaplan_meier(con)

    symmetric_overflows = [
        row for row in curve if row["survival"] + 1.96 * row["se"] > 1.0
    ]
    assert symmetric_overflows, "expected the symmetric interval to overflow here"
    for row in curve:
        assert 0.0 <= row["ci_low"] <= row["survival"] <= row["ci_high"] <= 1.0, row


def test_the_interval_widens_as_the_risk_set_thins(con):
    """Later tenures rest on fewer subscribers, and the interval has to say so.

    A curve whose uncertainty does not grow as the pool shrinks is reporting
    the same confidence in a number backed by 160 subscribers as one backed by
    6000, which is the quiet failure mode of a hand-rolled variance.
    """
    curve = survival.kaplan_meier(con)
    widths = [row["ci_high"] - row["ci_low"] for row in curve]

    assert widths[-1] > widths[1] * 2, (widths[1], widths[-1])
    assert curve[-1]["at_risk"] < curve[1]["at_risk"] / 5


def test_a_month_with_no_churn_leaves_survival_and_the_interval_alone():
    """Greenwood accumulates only on event months. A no-event month must not
    move the estimate or widen the interval -- there is no new information in
    it, and a variance term that fired anyway would make the curve drift with
    window length rather than with the data.

    Asserted on a hand-built risk table rather than on a generated store: this
    dataset has churn in every month, so a test over it would have skipped and
    the accumulator's quiet path would never have been exercised at all.
    """
    rows = [
        {"tenure": 0, "at_risk": 100, "events": 10, "censored": 0},
        {"tenure": 1, "at_risk": 90, "events": 0, "censored": 0},
        {"tenure": 2, "at_risk": 90, "events": 9, "censored": 0},
    ]
    curve = survival.accumulate(rows)

    assert curve[0]["survival"] == pytest.approx(0.9)
    assert curve[1]["survival"] == pytest.approx(curve[0]["survival"])
    assert curve[1]["se"] == pytest.approx(curve[0]["se"])
    assert curve[1]["hazard"] == 0.0
    # And the curve picks straight back up on the next event month.
    assert curve[2]["survival"] == pytest.approx(0.9 * 0.9)
    assert curve[2]["se"] > curve[1]["se"]


@pytest.fixture(scope="module")
def planted(con):
    """Coefficients and each subscriber's latent level. Tests may read truth."""
    truth = dict(con.execute("select term, value from hazard_truth").fetchall())
    subscribers = con.execute(
        """
        select s.plan, t.engagement
        from subscribers s join subscriber_truth t using (subscriber_id)
        """
    ).fetchall()
    return truth, subscribers


def _planted_hazard(truth: dict, plan: str, tenure: int, engagement: float) -> float:
    logit = (
        truth["intercept"]
        + truth[f"plan_{plan}"]
        + truth["tenure_log1p"] * math.log1p(tenure)
        + truth["engagement"] * engagement
    )
    return 1.0 / (1.0 + math.exp(-logit))


def _planted_survival(planted, tenure: int) -> float:
    """Mean survival through `tenure`, over the whole generated population.

    Over everyone rather than over the risk set at `tenure`, deliberately. KM
    estimates the *marginal* survival function, and it is consistent for it
    when censoring is independent of the event -- which holds here because
    censoring is administrative: it is set by signup month, which the hazard
    does not depend on. Averaging the prediction over the survivors instead
    would compare KM against a different quantity and the gap would widen with
    tenure as the risk set selects for low-hazard subscribers.
    """
    truth, subscribers = planted
    return sum(
        math.prod(
            1.0 - _planted_hazard(truth, plan, k, engagement)
            for k in range(tenure + 1)
        )
        for plan, engagement in subscribers
    ) / len(subscribers)


@pytest.mark.parametrize("tenure", [0, 1, 3, 6, 11, 17])
def test_the_km_curve_recovers_the_planted_survival_function(con, planted, tenure):
    """The day-3 result: the estimate lands on the process that generated it.

    Tolerance is 4 Greenwood standard errors. Measured gaps on seed 67 are
    1.03 / 1.14 / 1.55 / 0.75 / 0.38 / 0.17 se at the tenures checked, and the
    worst gap over seeds 11 and 404 was 2.18 and 1.00 se, so 4 is loose enough
    not to flake and tight enough to catch a hazard off by a few points.
    """
    observed = survival.kaplan_meier(con)[tenure]
    expected = _planted_survival(planted, tenure)

    assert observed["survival"] == pytest.approx(
        expected, abs=4 * observed["se"]
    ), (tenure, observed["survival"], expected, observed["se"])


def test_ignoring_censoring_flatters_retention_by_a_measurable_amount(con, planted):
    """The error KM exists to avoid, quantified on this dataset.

    "1 - churned/total" treats every censored subscriber as a permanent
    survivor, which is the single most common retention number in circulation.
    On this store it reports 84.2% against a KM estimate of 77.2% at the last
    observed tenure -- seven points of optimism, and it would get *better*
    looking if the window were shortened, since a shorter window censors more
    people.
    """
    curve = survival.kaplan_meier(con)
    total, churned = con.execute(
        "select count(*), count(*) filter (where not censored) from subscribers"
    ).fetchone()

    naive = 1.0 - churned / total
    final = curve[-1]["survival"]
    truth = _planted_survival(planted, curve[-1]["tenure"])

    assert naive > final + 0.04, (naive, final)
    # And it is the naive number that is wrong, not KM: KM sits on the truth.
    assert abs(final - truth) < abs(naive - truth)


def test_treating_censored_subscribers_as_churn_destroys_the_curve(con):
    """The opposite error, and the reason the risk table derives censoring
    instead of assuming the last observed month is an exit.

    Anyone whose observation ends gets counted as having churned: survival
    collapses from 77.2% to 0.0% here, because in a window where most people
    are censored, that assumption says almost everybody left.
    """
    table = survival.risk_table(con)
    mutated = [
        {**row, "events": row["events"] + row["censored"], "censored": 0}
        for row in table
    ]

    correct = survival.accumulate(table)[-1]["survival"]
    broken = survival.accumulate(mutated)[-1]["survival"]

    assert correct > 0.7
    assert broken < 0.01, broken


def test_dropping_censored_subscribers_entirely_is_worse_than_either(con):
    """Complete-case analysis, the third way this goes wrong.

    Restricting to subscribers with an observed churn means restricting to
    people who churned, so survival goes to zero by construction -- the
    selection is on the outcome. Included because it looks the most defensible
    of the three ("only use complete data") and is the most badly wrong.
    """
    rows = con.execute(
        """
        select m.tenure_months, count(*), count(*) filter (where m.churned)
        from subscription_months m join subscribers s using (subscriber_id)
        where not s.censored
        group by 1 order by 1
        """
    ).fetchall()

    survival_estimate = 1.0
    for _, at_risk, events in rows:
        survival_estimate *= 1.0 - events / at_risk

    assert survival_estimate < 0.01, survival_estimate


def test_km_and_the_day_two_retention_curve_tell_the_same_story(con):
    """Two estimators, two definitions of the risk set, one truth.

    Day 2's cohort curve pools only cohorts with exposure at a tenure. KM keeps
    everyone for the months they were observed and drops them after. They are
    not the same computation, and they are not forced to agree -- which is what
    makes the comparison worth making: both are attempts at the same marginal
    survival function from different directions.

    The offset is a convention, not a fudge. Cohort retention at tenure t is
    "still subscribed at the *start* of month t", while KM's S(t) is "survived
    *through* month t", so retention(t) lines up with S(t-1). Gaps come out at
    0.05-1.14 Greenwood se on the committed store, widening at long tenures
    where the two risk sets differ most.
    """
    from src import cohorts

    km = survival.kaplan_meier(con)
    retention = {row["tenure"]: row["retention"] for row in cohorts.retention_curve(con)}

    for tenure in (1, 3, 6, 12, 17):
        if tenure not in retention:
            continue
        estimate = km[tenure - 1]
        assert estimate["survival"] == pytest.approx(
            retention[tenure], abs=4 * estimate["se"]
        ), (tenure, estimate["survival"], retention[tenure])


def test_an_unknown_segment_is_refused():
    with duckdb.connect(":memory:") as con:
        for call in (
            lambda: survival.risk_table(con, "engagement"),
            lambda: survival.kaplan_meier_by_segment(con, "signup_month"),
        ):
            with pytest.raises(ValueError, match="unknown segment"):
                call()


def test_per_plan_risk_sets_partition_the_pooled_one(con):
    """Splitting by plan must not create or lose subscriber-months. Plan is
    fixed per subscriber in this model, so every month belongs to exactly one
    plan's risk set."""
    pooled = {row["tenure"]: row for row in survival.risk_table(con)}
    split: dict[int, dict[str, int]] = {}
    for row in survival.risk_table(con, "plan"):
        bucket = split.setdefault(row["tenure"], {"at_risk": 0, "events": 0, "censored": 0})
        for key in bucket:
            bucket[key] += row[key]

    assert set(split) == set(pooled)
    for tenure, totals in split.items():
        for key, value in totals.items():
            assert value == pooled[tenure][key], (tenure, key)


def test_per_plan_curves_recover_the_planted_tier_ordering(con, planted):
    """Each plan's own curve, checked against its own planted survival.

    The pooled curve can be right while the split is wrong -- a segment column
    joined on the wrong key would mix the groups and leave the total intact --
    so the per-plan curves are compared to per-plan predictions rather than
    only to each other.
    """
    truth, subscribers = planted
    curves = survival.kaplan_meier_by_segment(con, "plan")
    assert set(curves) == {"basic", "standard", "premium"}

    tenure = 6
    for plan, curve in curves.items():
        cohort = [e for subscriber_plan, e in subscribers if subscriber_plan == plan]
        expected = sum(
            math.prod(
                1.0 - _planted_hazard(truth, plan, k, engagement)
                for k in range(tenure + 1)
            )
            for engagement in cohort
        ) / len(cohort)

        row = curve[tenure]
        assert row["survival"] == pytest.approx(expected, abs=4 * row["se"]), (
            plan,
            row["survival"],
            expected,
        )

    at_six = {plan: curve[tenure]["survival"] for plan, curve in curves.items()}
    assert at_six["premium"] > at_six["standard"] > at_six["basic"], at_six


def test_the_log_rank_fires_on_the_planted_plan_effect_in_the_right_direction(con):
    """A significant statistic is only half of it: the observed-minus-expected
    per group has to point the way the coefficients were planted. A test on the
    statistic alone passes just as well if the groups are swapped.
    """
    result = survival.logrank(con, "plan")

    assert result["df"] == 2
    assert result["statistic"] > result["critical_05"], result
    assert result["significant"] is True

    # Churn is planted highest on basic and lowest on premium, so basic must
    # churn more than a shared hazard predicts and premium less.
    assert result["observed"]["basic"] > result["expected"]["basic"]
    assert result["observed"]["premium"] < result["expected"]["premium"]
    # Total events are conserved: expectations redistribute them, nothing else.
    assert sum(result["observed"].values()) == pytest.approx(
        sum(result["expected"].values()), rel=1e-9
    )


def test_the_log_rank_does_not_fire_on_a_dataset_with_no_plan_effect(tmp_path):
    """Calibration, which a significance test is not much use without.

    The planted hazard can be rebuilt with every plan effect set to zero, which
    gives a dataset where the null hypothesis is *true* by construction. Across
    seeds 0-9 (1500 subscribers, 12 months) the statistic came out between 0.59
    and 7.26, crossing the 5.991 critical value once -- about the 5% false
    positive rate the test is supposed to have. So the assertion is on the rate,
    not on a single draw: a test demanding zero rejections from a correct 5%
    test would be a test that the test is broken.

    Worth recording: at 2000 subscribers over 20 seeds nothing fired at all
    (max statistic 4.5). The simplified (O-E)^2/E form used here is known to be
    conservative relative to the full variance version, so a non-significant
    result from it means "not shown", not "no difference".
    """
    from src.hazard import HazardModel

    flat = HazardModel(plan_effect={plan: 0.0 for plan in ("basic", "standard", "premium")})
    rejections = []
    for seed in range(10):
        path = tmp_path / f"null{seed}.duckdb"
        store.build_store(store=path, count=1500, window_months=12, seed=seed, model=flat)
        with duckdb.connect(str(path), read_only=True) as con:
            result = survival.logrank(con, "plan")
        if result["significant"]:
            rejections.append((seed, round(result["statistic"], 2)))

    assert len(rejections) <= 2, rejections


def test_render_reports_the_curve_and_the_median_honestly(built):
    text = survival.render(built, max_tenure=6)

    assert "Kaplan-Meier survival" in text
    assert "95% CI" in text
    assert "Median survival: not reached" in text, (
        "most subscribers in this dataset are still subscribed, so there is no "
        "median to report -- saying otherwise would be an extrapolation"
    )
    assert "Log-rank across plans" in text
    # The deepest printed tenure honours --max-tenure.
    assert "\n     6  " in text
    assert "\n     7  " not in text


def test_render_explains_itself_when_there_is_no_store(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.store"):
        survival.render(tmp_path / "missing.duckdb")


def test_survival_cli_prints_the_curve(built, capsys):
    survival.main(["--store", str(built), "--max-tenure", "3"])
    assert "Kaplan-Meier survival" in capsys.readouterr().out
