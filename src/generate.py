"""Generate subscriber lifecycles from the hazard model.

One subscriber per row in `subscribers`, one row per observed subscriber-month
in `subscription_months`, and engagement events per month. Subscribers still
active when the observation window closes are right-censored -- churn_month is
None and `censored` is true, which is the case the survival analysis on day 3
has to handle rather than drop.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date

from src.hazard import DEFAULT_MODEL, PLANS, HazardModel

# Signups are spread across the window so cohorts differ in how long they have
# been observable -- without that, every cohort has identical exposure and the
# censoring the survival analysis exists to handle never appears.
PLAN_WEIGHTS = (0.55, 0.30, 0.15)


@dataclass(frozen=True)
class Subscriber:
    subscriber_id: str
    plan: str
    engagement: float
    signup_month: int  # months since the window opened
    churn_month: int | None  # months since signup; None if censored
    censored: bool


def _month_to_date(window_start: date, offset: int) -> date:
    """First of the month `offset` months after window_start."""
    month_index = window_start.month - 1 + offset
    return date(window_start.year + month_index // 12, month_index % 12 + 1, 1)


def generate_subscribers(
    count: int = 4000,
    window_months: int = 24,
    seed: int = 7,
    model: HazardModel = DEFAULT_MODEL,
) -> list[Subscriber]:
    rng = random.Random(seed)
    subscribers: list[Subscriber] = []

    for n in range(count):
        plan = rng.choices(PLANS, weights=PLAN_WEIGHTS, k=1)[0]
        # Beta(2, 2) keeps engagement inside (0, 1) and off the extremes, so no
        # subscriber is immortal or doomed by construction.
        engagement = rng.betavariate(2.0, 2.0)
        signup_month = rng.randrange(window_months)
        observable = window_months - signup_month

        churn_month: int | None = None
        for tenure in range(observable):
            p = model.monthly_churn_probability(plan, tenure, engagement)
            if rng.random() < p:
                churn_month = tenure
                break

        subscribers.append(
            Subscriber(
                subscriber_id=f"S{n:06d}",
                plan=plan,
                engagement=round(engagement, 4),
                signup_month=signup_month,
                churn_month=churn_month,
                censored=churn_month is None,
            )
        )

    return subscribers


def observed_months(subscriber: Subscriber, window_months: int = 24) -> int:
    """How many months this subscriber was actually observed as active.

    A churn in month t means t active months were observed before leaving; a
    censored subscriber was observed for the whole remainder of the window.
    """
    if subscriber.churn_month is not None:
        return subscriber.churn_month
    return window_months - subscriber.signup_month


def month_rows(
    subscribers: list[Subscriber], window_start: date, window_months: int = 24
) -> list[dict]:
    """Expand to one row per observed subscriber-month."""
    rows = []
    for sub in subscribers:
        active_months = observed_months(sub, window_months)
        for tenure in range(active_months + (0 if sub.censored else 1)):
            churned_here = (not sub.censored) and tenure == sub.churn_month
            rows.append(
                {
                    "subscriber_id": sub.subscriber_id,
                    "plan": sub.plan,
                    "tenure_months": tenure,
                    "calendar_month": _month_to_date(
                        window_start, sub.signup_month + tenure
                    ),
                    "churned": churned_here,
                }
            )
    return rows


def engagement_rows(
    subscribers: list[Subscriber],
    window_start: date,
    window_months: int = 24,
    seed: int = 7,
    base_events: float = 12.0,
) -> list[dict]:
    """Monthly engagement event counts, Poisson-ish around engagement level.

    The analysis sees these counts, not the latent engagement value -- that is
    the point. Recovering the engagement effect from observable activity is the
    day-4 problem.
    """
    rng = random.Random(seed + 1)
    rows = []
    for sub in subscribers:
        active_months = observed_months(sub, window_months)
        mean = base_events * sub.engagement
        for tenure in range(active_months):
            # Poisson via sum of exponentials is enough here and keeps the
            # dependency surface at the stdlib.
            events, total = 0, 0.0
            while True:
                total += rng.expovariate(1.0)
                if total > mean:
                    break
                events += 1
            rows.append(
                {
                    "subscriber_id": sub.subscriber_id,
                    "tenure_months": tenure,
                    "calendar_month": _month_to_date(
                        window_start, sub.signup_month + tenure
                    ),
                    "events": events,
                }
            )
    return rows
