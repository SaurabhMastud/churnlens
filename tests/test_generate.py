"""Generator checks, with the emphasis on censoring and on the generated data
actually reflecting the hazard it was drawn from.
"""
from datetime import date

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


def test_engagement_rows_track_the_engagement_level():
    subs = generate.generate_subscribers(count=2000, window_months=24, seed=19)
    rows = generate.engagement_rows(subs, date(2025, 1, 1), 24, seed=19)
    level = {s.subscriber_id: s.engagement for s in subs}

    low = [r["events"] for r in rows if level[r["subscriber_id"]] < 0.3]
    high = [r["events"] for r in rows if level[r["subscriber_id"]] > 0.7]

    assert sum(low) / len(low) < sum(high) / len(high)
