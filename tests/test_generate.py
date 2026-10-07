"""Generator checks, with the emphasis on censoring and on the generated data
actually reflecting the hazard it was drawn from.
"""
from datetime import date

import pytest

from src import generate
from src.hazard import PLANS


def test_generation_is_deterministic_for_a_seed():
    a = generate.generate_subscribers(count=200, seed=42)
    b = generate.generate_subscribers(count=200, seed=42)

    assert a == b
    assert a != generate.generate_subscribers(count=200, seed=43)


def test_both_churned_and_censored_subscribers_are_produced():
    """A run with no censoring would let a broken survival analysis pass on day
    3, since censoring is the thing it exists to handle."""
    subs = generate.generate_subscribers(count=1000, window_months=24, seed=1)

    assert any(s.censored for s in subs)
    assert any(not s.censored for s in subs)


def test_churn_month_never_exceeds_the_observable_window():
    window = 24
    for sub in generate.generate_subscribers(count=1000, window_months=window, seed=3):
        if sub.churn_month is not None:
            assert sub.churn_month < window - sub.signup_month


def test_censored_subscribers_have_no_churn_month():
    for sub in generate.generate_subscribers(count=500, seed=5):
        assert sub.censored == (sub.churn_month is None)


def test_higher_tiers_churn_less_in_the_generated_data():
    """The planted effect has to survive sampling, not just exist in the model."""
    subs = generate.generate_subscribers(count=6000, window_months=24, seed=11)

    rates = {}
    for plan in PLANS:
        cohort = [s for s in subs if s.plan == plan]
        rates[plan] = sum(1 for s in cohort if not s.censored) / len(cohort)

    assert rates["basic"] > rates["standard"] > rates["premium"], rates


def test_engaged_subscribers_churn_less_in_the_generated_data():
    subs = generate.generate_subscribers(count=6000, window_months=24, seed=13)
    low = [s for s in subs if s.engagement < 0.3]
    high = [s for s in subs if s.engagement > 0.7]

    low_rate = sum(1 for s in low if not s.censored) / len(low)
    high_rate = sum(1 for s in high if not s.censored) / len(high)

    assert low_rate > high_rate, (low_rate, high_rate)


def test_month_rows_cover_every_observed_month_and_one_churn_row():
    subs = generate.generate_subscribers(count=300, window_months=24, seed=17)
    rows = generate.month_rows(subs, date(2025, 1, 1), 24)

    by_subscriber: dict[str, list[dict]] = {}
    for row in rows:
        by_subscriber.setdefault(row["subscriber_id"], []).append(row)

    for sub in subs:
        own = by_subscriber.get(sub.subscriber_id, [])
        churn_rows = [r for r in own if r["churned"]]
        assert len(churn_rows) == (0 if sub.censored else 1)
        tenures = sorted(r["tenure_months"] for r in own)
        # Contiguous from 0, no gaps -- a gap would silently break retention.
        assert tenures == list(range(len(tenures)))


def test_no_row_falls_outside_the_observation_window():
    """`recorded_months` adds a month for the churn row, so the subscriber who
    churns in their last observable month sits exactly on the window edge. One
    off-by-one there would date rows past the end of the window and quietly
    extend the last calendar cohort.
    """
    window, start = 24, date(2025, 1, 1)
    subs = generate.generate_subscribers(count=5000, window_months=window, seed=41)
    rows = generate.month_rows(subs, start, window) + generate.engagement_rows(
        subs, start, window, seed=41
    )

    offsets = [
        (r["calendar_month"].year - start.year) * 12
        + r["calendar_month"].month
        - start.month
        for r in rows
    ]
    assert min(offsets) == 0
    assert max(offsets) == window - 1

    # The edge case is only guarded if the sample actually contains it.
    on_edge = [
        s
        for s in subs
        if s.churn_month is not None
        and s.churn_month == window - s.signup_month - 1
    ]
    assert on_edge, "no subscriber churned in their last observable month"


def test_engagement_rows_are_deterministic_for_a_seed():
    """The sampler moved from summed exponentials to numpy for speed; it still
    has to be reproducible, or no stored dataset can be regenerated."""
    args = (generate.generate_subscribers(count=150, seed=29), date(2025, 1, 1), 24)

    assert generate.engagement_rows(*args, seed=29) == generate.engagement_rows(
        *args, seed=29
    )
    assert generate.engagement_rows(*args, seed=29) != generate.engagement_rows(
        *args, seed=30
    )


def test_engagement_and_month_rows_cover_exactly_the_same_months():
    """The invariant day 4 depends on. Engagement originally stopped one month
    short of the churn row, which made the two tables disagree on precisely the
    churned months -- i.e. the target. Keyed on (subscriber, tenure) rather than
    on row counts, so a count that happens to match but lands on the wrong
    months still fails.
    """
    subs = generate.generate_subscribers(count=400, window_months=24, seed=31)
    months = generate.month_rows(subs, date(2025, 1, 1), 24)
    events = generate.engagement_rows(subs, date(2025, 1, 1), 24, seed=31)

    keyed = {(r["subscriber_id"], r["tenure_months"]) for r in months}
    assert keyed == {(r["subscriber_id"], r["tenure_months"]) for r in events}

    # And the churn months specifically are in there, not just balanced totals.
    churn_keys = {
        (r["subscriber_id"], r["tenure_months"]) for r in months if r["churned"]
    }
    assert churn_keys and churn_keys <= keyed


def test_engagement_rows_have_one_row_per_recorded_month():
    subs = generate.generate_subscribers(count=400, window_months=24, seed=31)
    rows = generate.engagement_rows(subs, date(2025, 1, 1), 24, seed=31)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["subscriber_id"]] = counts.get(row["subscriber_id"], 0) + 1

    for sub in subs:
        expected = generate.recorded_months(sub, 24)
        assert counts.get(sub.subscriber_id, 0) == expected


def test_churn_month_activity_is_not_a_giveaway():
    """The leakage check stated as the analyst would hit it: activity in a
    churn month must look like activity in any other month. If the churn month
    were systematically empty (or absent), engagement would predict churn
    perfectly and the day-4 coefficient would be meaningless.
    """
    subs = generate.generate_subscribers(count=3000, window_months=24, seed=37)
    months = generate.month_rows(subs, date(2025, 1, 1), 24)
    events = {
        (r["subscriber_id"], r["tenure_months"]): r["events"]
        for r in generate.engagement_rows(subs, date(2025, 1, 1), 24, seed=37)
    }
    level = {s.subscriber_id: s.engagement for s in subs}

    # Compare like with like: churn-month activity against non-churn-month
    # activity for the same engagement band, so the plan/engagement effect on
    # churn doesn't masquerade as a churn-month effect.
    churn, other = [], []
    for row in months:
        key = (row["subscriber_id"], row["tenure_months"])
        if level[row["subscriber_id"]] <= 0.4 or level[row["subscriber_id"]] >= 0.6:
            continue
        (churn if row["churned"] else other).append(events[key])

    assert churn, "no churn months in the mid-engagement band to compare"
    churn_mean = sum(churn) / len(churn)
    other_mean = sum(other) / len(other)
    assert churn_mean == pytest.approx(other_mean, rel=0.25), (churn_mean, other_mean)
    assert min(churn) >= 0 and max(churn) > 0


def test_engagement_counts_are_non_negative_integers():
    subs = generate.generate_subscribers(count=300, seed=33)
    rows = generate.engagement_rows(subs, date(2025, 1, 1), 24, seed=33)

    assert all(isinstance(r["events"], int) and r["events"] >= 0 for r in rows)


def test_engagement_rows_track_the_engagement_level():
    subs = generate.generate_subscribers(count=2000, window_months=24, seed=19)
    rows = generate.engagement_rows(subs, date(2025, 1, 1), 24, seed=19)
    level = {s.subscriber_id: s.engagement for s in subs}

    low = [r["events"] for r in rows if level[r["subscriber_id"]] < 0.3]
    high = [r["events"] for r in rows if level[r["subscriber_id"]] > 0.7]

    assert sum(low) / len(low) < sum(high) / len(high)
