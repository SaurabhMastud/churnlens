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
    curve = survival._accumulate(rows)

    assert curve[0]["survival"] == pytest.approx(0.9)
    assert curve[1]["survival"] == pytest.approx(curve[0]["survival"])
    assert curve[1]["se"] == pytest.approx(curve[0]["se"])
    assert curve[1]["hazard"] == 0.0
    # And the curve picks straight back up on the next event month.
    assert curve[2]["survival"] == pytest.approx(0.9 * 0.9)
    assert curve[2]["se"] > curve[1]["se"]


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
