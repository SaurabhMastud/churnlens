"""LTV checks. The arithmetic is simple, so the tests are about the claims.

Three things can be wrong here even when every sum is right: the boundary
month (is month 0 paid by everyone?), the horizon (is it stated, and is it
inside the data?), and the comparison with the formula a business already uses.
Each gets a test; the planted-truth comparison lives in the second half.
"""
from __future__ import annotations

import math

import duckdb
import pytest

from src import ltv, store, survival

BUILD = dict(count=6000, window_months=18, seed=67)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("ltv") / "churnlens.duckdb"
    store.build_store(store=path, **BUILD)
    return path


@pytest.fixture(scope="module")
def con(built):
    with duckdb.connect(str(built), read_only=True) as connection:
        yield connection


@pytest.fixture(scope="module")
def curves(con):
    return survival.kaplan_meier_by_segment(con, "plan")


def test_a_flat_curve_pays_the_price_every_month():
    """The simplest case, which pins the boundary month.

    With survival fixed at 1.0 -- nobody ever churns -- the value over H months
    must be exactly `H * price`. If month 0 were charged at S(0) instead of 1.0,
    this would come out a fraction of a month short, and on a 12-month horizon
    that single month is comparable to the whole gap between two plans.
    """
    flat = [{"tenure": t, "at_risk": 100, "events": 0, "censored": 0} for t in range(12)]
    curve = survival.accumulate(flat)

    assert ltv.restricted_value(curve, 10.0, 12) == pytest.approx(120.0)
    assert ltv.restricted_value(curve, 10.0, 1) == pytest.approx(10.0)


def test_a_known_curve_matches_the_series_computed_by_hand():
    """One hand-computed case, so the implementation is checked against
    arithmetic rather than against itself."""
    rows = [
        {"tenure": 0, "at_risk": 100, "events": 10, "censored": 0},  # S(0) = 0.9
        {"tenure": 1, "at_risk": 90, "events": 9, "censored": 0},    # S(1) = 0.81
    ]
    curve = survival.accumulate(rows)

    # 1 + 0.9 + 0.81 months of revenue, at 100 each.
    assert ltv.restricted_value(curve, 100.0, 3) == pytest.approx(271.0)
    # With a 10%/month discount: 1 + 0.9/1.1 + 0.81/1.21.
    expected = 100.0 * (1.0 + 0.9 / 1.1 + 0.81 / 1.21)
    assert ltv.restricted_value(curve, 100.0, 3, 0.1) == pytest.approx(expected)


def test_discounting_lowers_value_and_longer_horizons_raise_it(curves):
    for curve in curves.values():
        plain = ltv.restricted_value(curve, 10.0, 12)
        discounted = ltv.restricted_value(curve, 10.0, 12, 0.01)
        assert discounted < plain
        # Survival is non-increasing, so each extra month adds something, but
        # less than the last.
        steps = [ltv.restricted_value(curve, 10.0, h) for h in range(1, 13)]
        increments = [b - a for a, b in zip(steps, steps[1:])]
        assert all(step > 0 for step in increments)
        assert increments == sorted(increments, reverse=True)


def test_value_scales_linearly_in_price(curves):
    """Stated in the output, so it had better be true: the price is an
    assumption, and the claim is that getting it wrong moves the magnitudes
    without reordering the plans."""
    for curve in curves.values():
        single = ltv.restricted_value(curve, 1.0, 12)
        assert ltv.restricted_value(curve, 37.0, 12) == pytest.approx(37.0 * single)


def test_a_horizon_beyond_the_data_is_refused(con):
    """The one error this module exists to prevent: a 36-month "lifetime"
    number computed from 18 months of observation. Extrapolation is a
    modelling choice and this is not the place to make it silently."""
    with pytest.raises(ValueError, match="exceeds"):
        ltv.by_plan(con, horizon=BUILD["window_months"] + 6)

    with pytest.raises(ValueError, match="at least 1 month"):
        ltv.restricted_value([], 10.0, 0)


def test_a_missing_price_is_refused(con):
    with pytest.raises(ValueError, match="no price given"):
        ltv.by_plan(con, 12, prices={"basic": 9.0})


def test_plan_ordering_follows_price_and_retention_together(con):
    rows = {row["plan"]: row for row in ltv.by_plan(con, 12)}

    assert rows["premium"]["value"] > rows["standard"]["value"] > rows["basic"]["value"]
    # Premium wins on both terms, so its value has to beat the pure price ratio
    # against basic: it is both dearer and better retained.
    price_ratio = rows["premium"]["price"] / rows["basic"]["price"]
    value_ratio = rows["premium"]["value"] / rows["basic"]["value"]
    assert value_ratio > price_ratio


def test_the_restricted_value_matches_the_planted_truth(con):
    """Does the LTV figure recover the value implied by the planted hazard?

    The planted survival for each subscriber is the product of
    `1 - hazard(plan, k, engagement)` over months, so the expected revenue over
    a horizon can be computed directly from the coefficients and averaged over
    the subscribers on each plan. That is the number the KM-based estimate
    should land on, and it does: within 1.5% at a 12-month horizon on this
    store, against a figure of several hundred currency units.

    A 2% tolerance rather than a standard-error band, because this comparison
    folds several estimates into one scalar (a survival curve per plan, each
    with its own Greenwood error) and a composite se would be more elaborate
    than the claim needs.
    """
    truth = dict(con.execute("select term, value from hazard_truth").fetchall())
    subscribers = con.execute(
        """
        select s.plan, t.engagement
        from subscribers s join subscriber_truth t using (subscriber_id)
        """
    ).fetchall()

    horizon = 12
    planted: dict[str, list[float]] = {}
    for plan, engagement in subscribers:
        survival_through = 1.0
        value = 0.0
        for month in range(horizon):
            value += survival_through
            logit = (
                truth["intercept"]
                + truth[f"plan_{plan}"]
                + truth["tenure_log1p"] * math.log1p(month)
                + truth["engagement"] * engagement
            )
            survival_through *= 1.0 - 1.0 / (1.0 + math.exp(-logit))
        planted.setdefault(plan, []).append(value)

    for row in ltv.by_plan(con, horizon):
        months = planted[row["plan"]]
        expected = row["price"] * sum(months) / len(months)
        assert row["value"] == pytest.approx(expected, rel=0.02), (
            row["plan"],
            row["value"],
            expected,
        )


def test_the_constant_hazard_shortcut_errs_in_a_predictable_direction(con):
    """The finding, with its sign the right way round.

    `price / churn_rate` truncated to the same horizon is close -- within a few
    percent -- but it is not neutral. The pooled rate averages a hazard that
    starts at 4.2% and ends under 1.3%, so over a horizon shorter than the
    averaging window it under-prices the early risk and **over-states** value.
    On the committed store that is +3.6% / +3.0% / +1.5% over 12 months.

    Over a horizon equal to the whole window the error nearly vanishes (under
    0.2%), which is not the shortcut being right: the horizon and the window
    the rate was averaged over simply coincide there. Both facts are asserted,
    since the second is what stops the first being read as a general result.
    """
    short = ltv.by_plan(con, 12)
    assert all(row["shortcut_error"] > 0 for row in short), short

    full = ltv.by_plan(con, BUILD["window_months"])
    assert all(abs(row["shortcut_error"]) < 0.01 for row in full), full


def test_the_unbounded_shortcut_implies_a_lifetime_the_data_cannot_support(curves):
    """And the version people actually quote.

    Unbounded, `price / churn_rate` implies an expected lifetime of 1/rate
    months -- 50 to 113 months per plan on the committed store, against 24
    months of observation, with the observed hazard still falling at the end of
    the window. It is not an arithmetic error; it is an extrapolation reported
    as a measurement, and it comes out several times the longest restricted
    figure the data supports.
    """
    for curve in curves.values():
        unbounded = ltv.geometric_shortcut(curve, 10.0)
        longest = ltv.restricted_value(curve, 10.0, len(curve))
        implied_months = unbounded / 10.0

        assert implied_months > 2 * len(curve), implied_months
        assert unbounded > 2 * longest


def test_render_states_the_horizon_and_the_price_assumption(built):
    text = ltv.render(built, horizon=12)

    assert "over **12 months**" in text
    assert "Prices are an assumption, not data" in text
    assert "extrapolation" in text
    assert "implied lifetime" in text


def test_ltv_cli_accepts_a_price_override(built, capsys):
    ltv.main(["--store", str(built), "--horizon", "6", "--price", "basic=99"])
    out = capsys.readouterr().out

    assert "basic 99" in out
    assert "over **6 months**" in out


def test_ltv_cli_rejects_an_unknown_plan_price(built):
    with pytest.raises(SystemExit):
        ltv.main(["--store", str(built), "--price", "gold=5"])
