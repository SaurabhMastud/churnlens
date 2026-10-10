"""Churn drivers: a discrete-time hazard model fitted on observable data.

The question this module exists to answer is not "can a model predict churn"
-- with a planted hazard, something will always score above chance. It is
whether a model fitted the ordinary way **recovers the process the data came
from**: the right signs, the right ordering of plan tiers, and a tenure effect
of the right shape.

Specification, and why it is this one:

* **One row per subscriber-month, not per subscriber.** The generator draws a
  Bernoulli churn decision every month from a hazard that depends on tenure, so
  the subscriber-month is the unit the process actually operates on. A
  per-subscriber model would have to pick a single "tenure" per person, which
  does not exist, and would throw away the censoring structure day 3 was built
  around.
* **Logistic link.** The hazard was planted on the logit scale, so a logistic
  regression on subscriber-months is the correctly specified model for it. This
  is the one place in the project where the analysis gets to be *right* by
  construction rather than approximately right, which makes any failure to
  recover the coefficients a bug in the analysis rather than a modelling
  mismatch. Noted as a limitation in the architecture doc: real data does not
  hand you the correct functional form.
* **`log1p(tenure)` as the tenure feature**, matching the planted shape. A
  linear tenure term would be a misspecification, and the comparison between
  the two is itself a test -- see `fit_with_features`.
* **Observed activity, never the latent engagement level.** The hazard was
  driven by a latent engagement value in [0, 1]; what an analyst can see is a
  monthly event count. Fitting on the observable means the recovered
  engagement coefficient is attenuated by measurement error, and quantifying
  that attenuation is one of the day's results rather than something to hide.

Features are built in SQL over the analysis-facing tables only. Nothing here
reads `hazard_truth` or `subscriber_truth`; the contract test enforces it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as sk_log_loss, roc_auc_score

from src.store import DEFAULT_STORE

FEATURE_SETS = {
    # The specification that matches the planted hazard.
    "planted_shape": ("plan_standard", "plan_premium", "tenure_log1p", "activity"),
    # A deliberate misspecification of the tenure term, kept as a comparison.
    "linear_tenure": ("plan_standard", "plan_premium", "tenure", "activity"),
    # No activity term at all: what an analyst who never joined the engagement
    # table would get, and the baseline the activity feature has to beat.
    "no_activity": ("plan_standard", "plan_premium", "tenure_log1p"),
}


def design_rows(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """One row per observed subscriber-month, with features and the outcome.

    `activity` is that month's own event count, scaled to events-per-12 so the
    coefficient is on a comparable footing with the planted engagement
    coefficient (the generator draws counts around `12 * engagement`). The
    scaling is a presentational convenience and does not change the fit.

    The join to `engagement_events` is an inner join on purpose. The two tables
    share a key set exactly -- that invariant is tested in both Python and SQL
    -- so an inner join cannot drop a row here. If that ever stops being true,
    this returns fewer rows than `subscription_months` has and the row-count
    test fails loudly, which is better than a `coalesce` quietly inventing
    zero-activity months. A zero-filled left join is precisely the shape the
    day-2 leakage bug would have taken.
    """
    return [
        dict(
            zip(
                (
                    "subscriber_id",
                    "plan",
                    "tenure",
                    "tenure_log1p",
                    "activity",
                    "churned",
                ),
                row,
            )
        )
        for row in con.execute(
            """
            select m.subscriber_id,
                   m.plan,
                   m.tenure_months,
                   ln(1 + m.tenure_months)      as tenure_log1p,
                   e.events / 12.0              as activity,
                   case when m.churned then 1 else 0 end as churned
            from subscription_months m
            join engagement_events e
              on e.subscriber_id = m.subscriber_id
             and e.tenure_months = m.tenure_months
            order by m.subscriber_id, m.tenure_months
            """
        ).fetchall()
    ]


def design_matrix(
    rows: list[dict], features: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Expand feature names into a numeric matrix, plan one-hot included.

    Dummies are built here rather than by `pd.get_dummies` so the reference
    level is pinned explicitly. `get_dummies(drop_first=True)` drops in
    alphabetical order, which would make `basic` the reference today and
    silently change it if a plan named `alpha` were ever added -- and every
    fitted plan coefficient would shift meaning without anything failing.
    """
    unknown = set(features) - set(_FEATURE_BUILDERS)
    if unknown:
        raise ValueError(f"unknown features {sorted(unknown)}")

    matrix = np.column_stack(
        [[_FEATURE_BUILDERS[name](row) for row in rows] for name in features]
    )
    outcome = np.array([row["churned"] for row in rows])
    return matrix, outcome


# `basic` is the reference level -- it has no dummy -- matching how the hazard
# was planted, where basic's plan effect is 0. Keeping the same reference is
# what makes a fitted plan coefficient comparable to a planted one at all.
_FEATURE_BUILDERS = {
    "plan_standard": lambda row: float(row["plan"] == "standard"),
    "plan_premium": lambda row: float(row["plan"] == "premium"),
    "tenure": lambda row: float(row["tenure"]),
    "tenure_log1p": lambda row: float(row["tenure_log1p"]),
    "activity": lambda row: float(row["activity"]),
}


def fit_with_features(
    rows: list[dict], features: tuple[str, ...] = FEATURE_SETS["planted_shape"]
) -> dict:
    """Fit the discrete-time hazard model and return the coefficients.

    Unpenalised, deliberately. scikit-learn's default is L2 with C=1.0, which
    shrinks coefficients toward zero -- harmless for prediction and actively
    wrong here, because the point is to compare estimates against planted
    values. A penalised fit would be biased low by an amount nobody set, and
    the comparison would be measuring the default rather than the data.

    Spelled `C=np.inf` rather than `penalty=None`: the latter is the obvious
    way to write it, works, and emits a `FutureWarning` on scikit-learn 1.8+
    saying the argument is going away in 1.10. Both forms were run and give
    identical coefficients to 1e-12, so this is the same fit written the way
    that will still exist.

    `tol=1e-8` rather than the default 1e-4, and this one was a real bug rather
    than a preference. At the default, lbfgs stops *short of the maximum
    likelihood estimate*: the score equation for the intercept requires
    `sum(y - p) = 0`, and the default fit left a residual of 0.98 churn events
    on 43,467 rows. The coefficients it returned were visibly off --
    `plan_standard` came back at -0.425 where the converged fit gives -0.4516
    against a planted -0.45 -- so the under-converged estimate looked like a
    small recovery failure in the analysis when it was the optimiser quitting
    early. `newton-cholesky` converges to the same place; lbfgs at a tighter
    tolerance is the smaller change. The score identity is now a test.
    """
    matrix, outcome = design_matrix(rows, features)
    model = LogisticRegression(C=np.inf, max_iter=2000, tol=1e-8)
    model.fit(matrix, outcome)

    return {
        "features": features,
        "intercept": float(model.intercept_[0]),
        "coefficients": dict(zip(features, (float(c) for c in model.coef_[0]))),
        "rows": len(rows),
        "events": int(outcome.sum()),
        "model": model,
    }


def fit(con: duckdb.DuckDBPyConnection, specification: str = "planted_shape") -> dict:
    if specification not in FEATURE_SETS:
        raise ValueError(
            f"unknown specification {specification!r}; "
            f"expected one of {tuple(FEATURE_SETS)}"
        )
    return fit_with_features(design_rows(con), FEATURE_SETS[specification])


def standard_errors(fitted: dict, rows: list[dict]) -> dict[str, float]:
    """Asymptotic standard errors for the fitted coefficients.

    scikit-learn is a prediction library and does not return these, which is a
    reasonable choice for its purpose and a problem for this one: a coefficient
    without a standard error cannot be compared to a planted value except by
    eye. They come from the inverse Fisher information for a logistic model,

        cov = (X' W X)^-1,  W = diag(p_i (1 - p_i))

    which is six lines of numpy over the design matrix the fit already used.
    This is also the honest reason the project does not reach for statsmodels:
    one matrix inverse is cheaper than a dependency, and the day-1 pins already
    carried one package nobody had installed.

    Asymptotic, so they are standard errors of the maximum-likelihood estimate
    and say nothing about whether the specification is right. A misspecified
    model gets tight errors around the wrong number, which is precisely why the
    comparison against the planted coefficients exists.
    """
    matrix, _ = design_matrix(rows, fitted["features"])
    with_intercept = np.column_stack([np.ones(len(matrix)), matrix])
    probability = predicted_hazard(fitted, rows)
    weights = probability * (1.0 - probability)

    information = with_intercept.T @ (with_intercept * weights[:, None])
    errors = np.sqrt(np.diag(np.linalg.inv(information)))
    return dict(zip(("intercept", *fitted["features"]), (float(e) for e in errors)))


def predicted_hazard(fitted: dict, rows: list[dict]) -> np.ndarray:
    matrix, _ = design_matrix(rows, fitted["features"])
    return fitted["model"].predict_proba(matrix)[:, 1]


def log_loss(fitted: dict, rows: list[dict]) -> float:
    """Mean negative log-likelihood per subscriber-month.

    Log loss rather than accuracy. The monthly churn rate here is under 2%, so
    a model that predicts "nobody churns" is 98% accurate and useless; log loss
    is a proper scoring rule and cannot be gamed that way.
    """
    return float(sk_log_loss(_outcome(rows), predicted_hazard(fitted, rows)))


def auc(fitted: dict, rows: list[dict]) -> float:
    """Rank-based AUC (Mann-Whitney), straight from scikit-learn.

    Both metrics were written out by hand first -- a clipped log-likelihood and
    a tie-averaged rank sum -- and then deleted: the versions here agree with
    scikit-learn's to 1e-12 on this data, scikit-learn is already a dependency
    for the model itself, and its implementations handle the tie and
    degenerate-input cases that the hand-rolled ones got right only by
    accident. Twenty lines removed for no change in behaviour.
    """
    return float(roc_auc_score(_outcome(rows), predicted_hazard(fitted, rows)))


def _outcome(rows: list[dict]) -> np.ndarray:
    return np.array([row["churned"] for row in rows])


def split_by_subscriber(
    rows: list[dict], holdout: float = 0.3, seed: int = 11
) -> tuple[list[dict], list[dict]]:
    """Split on subscribers, never on rows.

    Splitting rows at random puts months from the same subscriber on both sides
    of the split, and subscriber-level traits (their plan, their engagement
    level) then appear in training and test alike. The score that comes back is
    partly a memory of the training subscribers, which is the most common
    leak in a panel dataset and looks exactly like a good result.
    """
    subscribers = sorted({row["subscriber_id"] for row in rows})
    generator = np.random.default_rng(seed)
    held = set(
        generator.choice(
            subscribers, size=max(1, int(len(subscribers) * holdout)), replace=False
        )
    )
    return (
        [row for row in rows if row["subscriber_id"] not in held],
        [row for row in rows if row["subscriber_id"] in held],
    )


def evaluate(con: duckdb.DuckDBPyConnection, specification: str = "planted_shape") -> dict:
    """Fit on a training split of subscribers, score on held-out subscribers."""
    rows = design_rows(con)
    train, test = split_by_subscriber(rows)
    fitted = fit_with_features(train, FEATURE_SETS[specification])

    return {
        "specification": specification,
        "fitted": fitted,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_log_loss": log_loss(fitted, train),
        "test_log_loss": log_loss(fitted, test),
        "test_auc": auc(fitted, test),
        "base_rate": float(_outcome(rows).mean()),
    }


def render(store: str | Path = DEFAULT_STORE) -> str:
    store = Path(store)
    if not store.exists():
        raise FileNotFoundError(
            f"no store at {store} -- run `python -m src.store` to build one first"
        )

    with duckdb.connect(str(store), read_only=True) as con:
        rows = design_rows(con)
        full = fit_with_features(rows)
        errors = standard_errors(full, rows)
        results = {name: evaluate(con, name) for name in FEATURE_SETS}

    lines = [
        f"Churn drivers -- {store}",
        "",
        f"Discrete-time logistic hazard on {full['rows']:,} subscriber-months "
        f"({full['events']:,} churn events, {full['events'] / full['rows']:.2%} "
        "monthly base rate).",
        "Fitted on observable data only: plan, tenure, and that month's event count.",
        "",
        "term             coefficient     se  reads as",
        "---------------  -----------  -----  -------------------------------------",
        f"intercept        {full['intercept']:>11.3f}  {errors['intercept']:>5.3f}  "
        "baseline log-odds, basic at tenure 0",
    ]
    reads = {
        "plan_standard": "standard vs basic, holding the rest",
        "plan_premium": "premium vs basic, holding the rest",
        "tenure_log1p": "per log1p(month) of tenure",
        "activity": "per 12 events in the month",
    }
    lines += [
        f"{term:<15}  {value:>11.3f}  {errors[term]:>5.3f}  {reads.get(term, '')}"
        for term, value in full["coefficients"].items()
    ]
    lines += [
        "",
        "Standard errors are asymptotic, from the inverse Fisher information.",
        "They describe sampling error in the estimate, not whether the model is",
        "the right one -- a misspecified model gets tight errors around a wrong",
        "number, which is what the comparison against the planted coefficients",
        "in the test suite is for.",
    ]

    lines += [
        "",
        "Held-out evaluation, split by subscriber (never by row -- months from",
        "one subscriber on both sides of a split leak their traits across it).",
        "",
        "specification     test log loss  test AUC",
        "----------------  -------------  --------",
    ]
    lines += [
        f"{name:<16}  {result['test_log_loss']:>13.4f}  {result['test_auc']:>8.3f}"
        for name, result in results.items()
    ]

    planted = results["planted_shape"]
    lines += [
        "",
        f"AUC on held-out subscribers: {planted['test_auc']:.3f}. That is a modest",
        "number and it should be: the planted hazard is genuinely close to the",
        "base rate for most subscriber-months, so no model can separate them",
        "cleanly. A high AUC here would be evidence of leakage, not of skill.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default=DEFAULT_STORE, type=Path)
    args = parser.parse_args(argv)
    print(render(args.store))


if __name__ == "__main__":
    main()
