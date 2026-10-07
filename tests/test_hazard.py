"""The hazard model is the project's ground truth, so its planted story has to
actually hold -- otherwise every later test that compares an estimate against
it is comparing against nonsense.
"""
import pytest

from src.hazard import DEFAULT_MODEL, PLANS, HazardModel


def test_higher_tiers_churn_less_all_else_equal():
    probabilities = [
        DEFAULT_MODEL.monthly_churn_probability(plan, 0, 0.5) for plan in PLANS
    ]

    assert probabilities == sorted(probabilities, reverse=True)
    assert len(set(probabilities)) == len(PLANS), "plan effects must be distinguishable"


def test_churn_risk_falls_with_tenure():
    risks = [
        DEFAULT_MODEL.monthly_churn_probability("basic", tenure, 0.5)
        for tenure in (0, 3, 12, 24)
    ]

    assert risks == sorted(risks, reverse=True)


def test_engagement_dominates_plan():
    """Day 4 asks whether a driver model recovers the effect *ordering*. If
    engagement were not the larger lever by construction, that test could pass
    on a model that had learned nothing."""
    engagement_swing = DEFAULT_MODEL.monthly_churn_probability(
        "basic", 0, 0.0
    ) - DEFAULT_MODEL.monthly_churn_probability("basic", 0, 1.0)
    plan_swing = DEFAULT_MODEL.monthly_churn_probability(
        "basic", 0, 0.5
    ) - DEFAULT_MODEL.monthly_churn_probability("premium", 0, 0.5)

    assert engagement_swing > plan_swing


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("tenure", [0, 1, 120])
@pytest.mark.parametrize("engagement", [0.0, 0.5, 1.0])
def test_probabilities_stay_in_the_open_unit_interval(plan, tenure, engagement):
    p = DEFAULT_MODEL.monthly_churn_probability(plan, tenure, engagement)

    assert 0.0 < p < 1.0


def test_rejects_inputs_it_cannot_model():
    with pytest.raises(ValueError, match="unknown plan"):
        DEFAULT_MODEL.monthly_churn_probability("enterprise", 0, 0.5)
    with pytest.raises(ValueError, match="engagement"):
        DEFAULT_MODEL.monthly_churn_probability("basic", 0, 1.5)
    with pytest.raises(ValueError, match="tenure_months"):
        DEFAULT_MODEL.monthly_churn_probability("basic", -1, 0.5)


def test_planted_coefficients_cannot_be_mutated():
    """"Fixed coefficients" is the claim the whole project rests on, and a
    frozen dataclass does not deliver it on its own: the dict behind
    plan_effect stayed writable, and DEFAULT_MODEL is one shared instance, so
    a single stray write moved the ground truth for the generator, the store
    and every test at once.
    """
    with pytest.raises(TypeError):
        DEFAULT_MODEL.plan_effect["premium"] = 99.0
    with pytest.raises(TypeError):
        del DEFAULT_MODEL.plan_effect["premium"]
    with pytest.raises(AttributeError):
        DEFAULT_MODEL.intercept = 99.0

    assert DEFAULT_MODEL.plan_effect["premium"] == -0.95


def test_a_caller_cannot_mutate_the_model_through_their_own_dict():
    """Passing a dict in and keeping a reference to it would otherwise be a
    second way around the same guarantee."""
    mine = {"basic": 0.0, "standard": -0.4, "premium": -0.8}
    model = HazardModel(plan_effect=mine)

    mine["premium"] = 99.0

    assert model.plan_effect["premium"] == -0.8


def test_equality_survives_the_read_only_wrapper():
    """as_rows and the store's truth check both compare models; wrapping the
    mapping must not make two identically-configured models unequal."""
    assert HazardModel() == HazardModel()
    assert HazardModel() == DEFAULT_MODEL
    assert HazardModel(intercept=-1.0) != HazardModel()


def test_coefficients_round_trip_to_storable_rows():
    rows = HazardModel().as_rows()
    terms = {term for term, _, _ in rows}

    assert {"intercept", "tenure_log1p", "engagement"} <= terms
    assert {f"plan_{plan}" for plan in PLANS} <= terms
    # Every planted coefficient must be stored, or a test could never check it.
    assert len(rows) == 3 + len(PLANS)
