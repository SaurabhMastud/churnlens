"""Driver-model checks: the design matrix first, coefficient recovery after.

A churn model is the easiest thing in this project to get wrong in a way that
looks like success, so the tests are ordered from "what is in the matrix" to
"does the fit mean anything". Most of the damage happens in the first part:
a feature that encodes the outcome, a row count that silently dropped the churn
months, a split that puts the same subscriber on both sides.
"""
from __future__ import annotations

import math

import duckdb
import numpy as np
import pytest

from src import drivers, store

BUILD = dict(count=6000, window_months=18, seed=67)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("drivers") / "churnlens.duckdb"
    store.build_store(store=path, **BUILD)
    return path


@pytest.fixture(scope="module")
def con(built):
    with duckdb.connect(str(built), read_only=True) as connection:
        yield connection


@pytest.fixture(scope="module")
def rows(con):
    return drivers.design_rows(con)


@pytest.fixture(scope="module")
def fitted(rows):
    return drivers.fit_with_features(rows)


def test_the_design_keeps_every_subscriber_month(con, rows):
    """The inner join to `engagement_events` must not drop a single row.

    This is the day-2 bug's test, moved to where it would now do damage. When
    engagement stopped a month short of the churn month, this join dropped
    *every churn event* -- and a model fitted on the remainder has nothing to
    predict, which at least fails loudly. The other half of that bug is worse:
    a left join with `fillna(0)` keeps the rows and makes "no activity" a
    perfect churn predictor.
    """
    months, events = con.execute(
        "select (select count(*) from subscription_months), "
        "(select count(*) from engagement_events)"
    ).fetchone()

    assert len(rows) == months == events
    churned = sum(row["churned"] for row in rows)
    assert churned == con.execute(
        "select count(*) from subscription_months where churned"
    ).fetchone()[0]
    assert churned > 0, "a design with no churn events cannot be fitted"


def test_no_feature_is_a_restatement_of_the_outcome(rows):
    """Every feature must be computable without knowing the outcome.

    Checked by the one thing that actually catches leakage: fit on each feature
    alone and look for a suspiciously perfect separation. A leaked outcome
    gives an AUC at or near 1.0; here no single honest feature beats 0.75.
    """
    for feature in sorted(drivers._FEATURE_BUILDERS):
        single = drivers.fit_with_features(rows, (feature,))
        score = drivers.auc(single, rows)
        assert 0.4 < score < 0.8, (feature, score)


def test_a_leaked_feature_would_be_caught_by_that_same_check(rows):
    """The mutation that proves the previous test can see anything at all.

    `churned` as a feature is the crudest possible leak. It scores 1.000 --
    perfect separation, which no real driver of churn ever produces. The point
    of keeping this as a test is that the number 1.000 is the signature to
    recognise: a model that looks this good has been handed the answer.
    """
    leaked = dict(drivers._FEATURE_BUILDERS)
    leaked["is_churn_month"] = lambda row: float(row["churned"])
    original = drivers._FEATURE_BUILDERS
    try:
        drivers._FEATURE_BUILDERS = leaked
        fitted = drivers.fit_with_features(rows, ("is_churn_month",))
        assert drivers.auc(fitted, rows) == pytest.approx(1.0)
    finally:
        drivers._FEATURE_BUILDERS = original


def test_unknown_features_and_specifications_are_refused(rows, con):
    with pytest.raises(ValueError, match="unknown features"):
        drivers.design_matrix(rows, ("engagement",))
    with pytest.raises(ValueError, match="unknown specification"):
        drivers.fit(con, "kitchen_sink")


def test_the_plan_dummies_use_basic_as_the_reference(rows):
    """Two dummies for three plans, with basic absent from both.

    The reference level has to match the planted model -- basic's plan effect
    is 0 there -- or the fitted plan coefficients are measuring the gap to a
    different baseline and the comparison against truth is meaningless.
    """
    matrix, _ = drivers.design_matrix(rows, ("plan_standard", "plan_premium"))

    for row, (standard, premium) in zip(rows, matrix):
        if row["plan"] == "basic":
            assert (standard, premium) == (0.0, 0.0)
        elif row["plan"] == "standard":
            assert (standard, premium) == (1.0, 0.0)
        else:
            assert (standard, premium) == (0.0, 1.0)

    # And no row claims two plans at once.
    assert matrix.sum(axis=1).max() == 1.0


def test_the_tenure_feature_is_the_planted_transform(rows):
    for row in rows[:500]:
        assert row["tenure_log1p"] == pytest.approx(math.log1p(row["tenure"]))


def test_the_split_shares_no_subscriber_between_train_and_test(rows):
    """Splitting on subscribers rather than rows, asserted rather than assumed.

    A random row split leaves months from the same subscriber on both sides, so
    their plan and their engagement level are present in training and test
    alike and the score partly measures memory of the training set. The split
    here is on subscriber ids; this checks that nothing leaks across and that
    both sides still contain churn events to score on.
    """
    train, test = drivers.split_by_subscriber(rows, holdout=0.3)

    train_ids = {row["subscriber_id"] for row in train}
    test_ids = {row["subscriber_id"] for row in test}
    assert not train_ids & test_ids
    assert train_ids | test_ids == {row["subscriber_id"] for row in rows}
    assert len(train) + len(test) == len(rows)
    # Roughly the requested proportion, by subscribers rather than by rows --
    # rows cannot hit it exactly, since subscribers differ in exposure.
    assert 0.25 < len(test_ids) / (len(train_ids) + len(test_ids)) < 0.35
    assert sum(row["churned"] for row in test) > 0


def test_the_split_is_deterministic_for_a_seed_and_moves_with_it(rows):
    first, _ = drivers.split_by_subscriber(rows, seed=11)
    again, _ = drivers.split_by_subscriber(rows, seed=11)
    other, _ = drivers.split_by_subscriber(rows, seed=12)

    assert [row["subscriber_id"] for row in first] == [
        row["subscriber_id"] for row in again
    ]
    assert [row["subscriber_id"] for row in first] != [
        row["subscriber_id"] for row in other
    ]


@pytest.fixture(scope="module")
def planted(con):
    """Planted coefficients, plus each subscriber's latent engagement level.

    Tests may read the truth tables; analysis modules may not. The latent level
    is needed here for the one comparison that matters most on day 4 -- fitting
    the *same* model on the true regressor instead of its observable proxy.
    """
    truth = dict(con.execute("select term, value from hazard_truth").fetchall())
    latent = dict(
        con.execute("select subscriber_id, engagement from subscriber_truth").fetchall()
    )
    return truth, latent


PLANTED_TERM = {
    "plan_standard": "plan_standard",
    "plan_premium": "plan_premium",
    "tenure_log1p": "tenure_log1p",
    "activity": "engagement",
}


def test_the_fit_recovers_the_planted_signs_and_the_tier_ordering(fitted):
    """The weakest useful claim, asserted separately from the numeric one.

    Signs and ordering are what a stakeholder actually reads off a driver
    model -- "premium churns less than standard, which churns less than basic"
    -- and they survive attenuation, so they can be asserted tightly while the
    magnitudes cannot.
    """
    coefficients = fitted["coefficients"]

    assert all(value < 0 for value in coefficients.values()), coefficients
    assert coefficients["plan_premium"] < coefficients["plan_standard"] < 0
    # Engagement is the strongest single lever by construction, and stays so
    # even after measurement error eats part of it.
    assert abs(coefficients["activity"]) > abs(coefficients["plan_premium"])


def test_fitting_on_the_true_regressor_recovers_every_planted_coefficient(
    con, rows, planted
):
    """The day-4 result: with the true regressor, the model lands on the truth.

    Substituting the latent engagement level for the observed event count --
    something only a test may do -- makes the specification exactly the one the
    data was generated from. Every coefficient then comes back within 1.7
    asymptotic standard errors of its planted value:

        plan_standard  -0.444 vs -0.45   (0.07 se)
        plan_premium   -0.839 vs -0.95   (0.81 se)
        tenure_log1p   -0.618 vs -0.55   (1.63 se)
        engagement     -2.431 vs -2.40   (0.17 se)

    This is the test that separates "the analysis is wrong" from "the data
    hides the regressor". If this one fails, the model code is broken; the
    attenuated fit below is the honest consequence of only having a proxy.
    """
    truth, latent = planted
    on_truth = [
        {**row, "activity": latent[row["subscriber_id"]]} for row in rows
    ]
    fitted = drivers.fit_with_features(on_truth)
    errors = drivers.standard_errors(fitted, on_truth)

    for term, value in fitted["coefficients"].items():
        expected = truth[PLANTED_TERM[term]]
        assert value == pytest.approx(expected, abs=3 * errors[term]), (
            term,
            value,
            expected,
            errors[term],
        )

    assert fitted["intercept"] == pytest.approx(
        truth["intercept"], abs=3 * errors["intercept"]
    )


def test_the_observable_proxy_attenuates_the_engagement_effect(con, rows, fitted, planted):
    """And the day's most useful negative result, with a number on it.

    Fitted on observed activity, the engagement coefficient comes back at
    -1.368 against a planted -2.40: **57% of the true effect**, and the gap is
    7.2 times its own standard error (8.4 on the test store), so it is bias
    rather than sampling noise. Reporting
    -1.368 as "the engagement effect" would understate the most important lever
    in the business by nearly half.

    The cause is measurement error in the regressor, and the size is roughly
    what theory predicts. The classic reliability ratio -- between-subscriber
    variance over total variance -- is var(e) / (var(e) + E[e]/12) = 0.0489 /
    (0.0489 + 0.0427) = 0.53 for this dataset, against an observed 0.57. The
    remaining gap is the logistic link's non-linearity plus the proxy being a
    single month rather than a subscriber average, which is why this is
    asserted as a band rather than a point.

    Everything else in the model survives: the plan coefficients and the
    tenure term stay close to their planted values, because the measurement
    error is in a regressor roughly orthogonal to them.
    """
    truth, latent = planted
    observed = fitted["coefficients"]["activity"]
    errors = drivers.standard_errors(fitted, rows)

    ratio = observed / truth["engagement"]
    assert 0.45 < ratio < 0.70, ratio
    # Bias, not sampling noise: several standard errors from the truth, where
    # the latent-regressor fit above sits within one.
    assert abs(observed - truth["engagement"]) > 5 * errors["activity"]

    # The attenuation is specific to the mismeasured term.
    for term in ("plan_standard", "plan_premium", "tenure_log1p"):
        assert fitted["coefficients"][term] == pytest.approx(
            truth[PLANTED_TERM[term]], abs=0.2
        ), term


def test_standard_errors_scale_down_as_the_dataset_grows(rows, fitted, tmp_path):
    """se ~ 1/sqrt(n): a quarter of the rows should roughly double them.

    Cheap, and it catches the two ways a hand-written information matrix goes
    wrong -- a missing weight term or a transposed product -- both of which
    produce errors that do not scale with n at all.
    """
    errors = drivers.standard_errors(fitted, rows)
    quarter = rows[: len(rows) // 4]
    small = drivers.fit_with_features(quarter)
    small_errors = drivers.standard_errors(small, quarter)

    for term, value in errors.items():
        assert 1.4 < small_errors[term] / value < 3.0, (term, value, small_errors[term])


def test_log_loss_beats_the_base_rate_and_accuracy_would_not(rows, fitted):
    """Why the headline metric is log loss and not accuracy.

    The monthly churn rate is about 1.6%, so "nobody ever churns" is 98.4%
    accurate. Log loss is a proper scoring rule: the fitted model has to beat
    the constant base-rate prediction, which it does, and could not do by
    predicting a constant.
    """
    outcome = np.array([row["churned"] for row in rows])
    base_rate = outcome.mean()
    constant = -np.mean(
        outcome * np.log(base_rate) + (1 - outcome) * np.log1p(-base_rate)
    )

    assert drivers.log_loss(fitted, rows) < constant
    # The accuracy trap, stated as a number so it is not just a claim.
    assert 1.0 - base_rate > 0.98


def test_auc_is_modest_and_that_is_the_expected_result(rows, fitted):
    """A correctly specified model on this data scores around 0.70.

    Worth asserting an *upper* bound as well as a lower one. The planted
    hazard sits close to the base rate for most subscriber-months, so the
    classes genuinely overlap and no model can separate them cleanly. An AUC
    above 0.9 here would mean something had leaked, not that the model got
    better -- so that is a failure condition, not a success.
    """
    score = drivers.auc(fitted, rows)
    assert 0.6 < score < 0.85, score
