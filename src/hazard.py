"""The churn process the data is generated from.

This module is the project's ground truth. Coefficients live here, the
generator draws from them, and tests compare estimates against them. Analyses
must never import this module -- see the ground-truth contract in
docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

PLANS = ("basic", "standard", "premium")


@dataclass(frozen=True)
class HazardModel:
    """Monthly churn hazard on the logit scale.

    logit(p) = intercept
             + plan_effect[plan]
             + tenure_coef * log1p(tenure_months)
             + engagement_coef * engagement

    `engagement` is the subscriber's latent engagement level in [0, 1].
    Signs are chosen so the planted story is unambiguous and testable: higher
    tiers churn less, churn risk falls as tenure accumulates, and engaged
    subscribers churn far less.
    """

    intercept: float = -1.9
    plan_effect: dict[str, float] = field(
        default_factory=lambda: {"basic": 0.0, "standard": -0.45, "premium": -0.95}
    )
    tenure_coef: float = -0.55
    engagement_coef: float = -2.4

    def monthly_churn_probability(
        self, plan: str, tenure_months: int, engagement: float
    ) -> float:
        if plan not in self.plan_effect:
            raise ValueError(f"unknown plan {plan!r}; expected one of {PLANS}")
        if not 0.0 <= engagement <= 1.0:
            raise ValueError(f"engagement must be in [0, 1], got {engagement}")
        if tenure_months < 0:
            raise ValueError(f"tenure_months must be >= 0, got {tenure_months}")

        logit = (
            self.intercept
            + self.plan_effect[plan]
            + self.tenure_coef * math.log1p(tenure_months)
            + self.engagement_coef * engagement
        )
        return 1.0 / (1.0 + math.exp(-logit))

    def as_rows(self) -> list[tuple[str, str, float]]:
        """Flatten to (term, kind, value) rows for storage beside the data."""
        rows = [
            ("intercept", "intercept", self.intercept),
            ("tenure_log1p", "tenure", self.tenure_coef),
            ("engagement", "engagement", self.engagement_coef),
        ]
        rows += [(f"plan_{plan}", "plan", value) for plan, value in self.plan_effect.items()]
        return rows


DEFAULT_MODEL = HazardModel()


def demo() -> None:
    """Self-check: the planted story has to actually hold in the probabilities."""
    m = DEFAULT_MODEL

    # Higher tiers churn less, all else equal.
    basic = m.monthly_churn_probability("basic", 0, 0.5)
    standard = m.monthly_churn_probability("standard", 0, 0.5)
    premium = m.monthly_churn_probability("premium", 0, 0.5)
    assert basic > standard > premium, (basic, standard, premium)

    # Churn risk falls as tenure accumulates.
    early = m.monthly_churn_probability("basic", 0, 0.5)
    late = m.monthly_churn_probability("basic", 24, 0.5)
    assert early > late, (early, late)

    # Engagement is the strongest single lever by construction.
    disengaged = m.monthly_churn_probability("basic", 0, 0.0)
    engaged = m.monthly_churn_probability("basic", 0, 1.0)
    assert disengaged > engaged
    assert disengaged - engaged > basic - premium, "engagement should dominate plan"

    # Probabilities stay probabilities at the extremes.
    for plan in PLANS:
        for tenure in (0, 1, 120):
            for engagement in (0.0, 1.0):
                p = m.monthly_churn_probability(plan, tenure, engagement)
                assert 0.0 < p < 1.0, (plan, tenure, engagement, p)

    print("hazard model self-check passed")


if __name__ == "__main__":
    demo()
