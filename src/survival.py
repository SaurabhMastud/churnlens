"""Kaplan-Meier survival estimates over subscriber tenure.

The reason survival analysis earns its place here rather than a plain retention
percentage is **censoring**. Most subscribers in this dataset have not churned
and have not been observed to the end of their life either -- they were still
subscribed when the observation window closed. A naive "share who churned"
treats those people as survivors forever, which flatters retention; dropping
them treats a 2-month-old signup the same as a 2-year-old one, which is worse.

Kaplan-Meier keeps a censored subscriber in the denominator for exactly the
months they were actually observed and removes them afterwards. Written out for
discrete tenure months:

    S(t) = prod over k <= t of (1 - d_k / n_k)

where `n_k` is the number of subscribers observed *during* tenure month k and
`d_k` the number who churned in it. S(t) reads as "still subscribed after
surviving month t".

Grain note: this data is monthly by construction, so the estimator is the
discrete-time one. There are no ties to break and no continuous-time
approximation involved -- a month is the unit the hazard itself was defined on.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import duckdb

from src.store import DEFAULT_STORE

# Chi-square upper-tail critical values at alpha = 0.05, by degrees of freedom.
# Two of them, hardcoded with their source, rather than a scipy dependency for
# two numbers: scipy is a large install and nothing else here needs it.
# Values from the standard chi-square table (e.g. NIST/SEMATECH e-Handbook
# 1.3.6.7.4); reproducible as scipy.stats.chi2.ppf(0.95, df).
CHI2_CRITICAL_05 = {1: 3.841, 2: 5.991}

# Columns of `subscription_months` a survival curve may be split by,
# allowlisted because the name is interpolated into SQL. Reading the segment
# off the month rows rather than joining `subscribers` keeps the risk-table
# query to one join; a subscriber-level segment (engagement decile, day 5) will
# need that join and can add it then.
SEGMENTS = ("plan",)


def risk_table(
    con: duckdb.DuckDBPyConnection, segment: str | None = None
) -> list[dict]:
    """At-risk, churned and censored counts per tenure month.

    `at_risk` counts subscribers with a `subscription_months` row at that
    tenure, i.e. observed during the month. `censored` counts those whose last
    observed month *is* that month and who did not churn in it -- they leave
    the risk set from the next month on, because nothing is known about them
    after the window closed.

    Deriving the censored count from the rows (`max(tenure_months)` per
    subscriber) rather than from `subscribers.censored` is deliberate: the two
    have to agree, and a test asserts they do. If instead this read the flag,
    that test would be comparing a column to itself.
    """
    if segment is not None and segment not in SEGMENTS:
        raise ValueError(f"unknown segment {segment!r}; expected one of {SEGMENTS}")

    column = f", m.{segment}" if segment else ""
    group = f", m.{segment}" if segment else ""

    return [
        dict(
            zip(
                ("tenure", *( (segment,) if segment else () ),
                 "at_risk", "events", "censored"),
                row,
            )
        )
        for row in con.execute(
            f"""
            with last_seen as (
                select subscriber_id, max(tenure_months) as last_tenure
                from subscription_months
                group by subscriber_id
            )
            select m.tenure_months as tenure{column},
                   count(*)                                   as at_risk,
                   count(*) filter (where m.churned)          as events,
                   count(*) filter (
                       where not m.churned
                         and m.tenure_months = l.last_tenure
                   )                                          as censored
            from subscription_months m
            join last_seen l using (subscriber_id)
            group by m.tenure_months{group}
            order by m.tenure_months{group}
            """
        ).fetchall()
    ]


def _accumulate(rows: list[dict]) -> list[dict]:
    """Walk a risk table in tenure order, folding in survival and its error.

    Greenwood's formula for the variance of S(t):

        var(S(t)) = S(t)^2 * sum over k <= t of d_k / (n_k * (n_k - d_k))

    The interval is built on the log-log scale rather than as S +- 1.96 se.
    The symmetric version is the one most people write and it has a real
    defect: near S = 1 (which is where every survival curve starts) it produces
    an upper bound above 1, i.e. a claim that more than everyone survived. The
    log-log transform maps to the whole real line, so the interval stays inside
    [0, 1] by construction.
    """
    out = []
    survival = 1.0
    greenwood = 0.0
    for row in rows:
        n, d = row["at_risk"], row["events"]
        hazard = d / n
        survival *= 1.0 - hazard
        if d and n > d:
            greenwood += d / (n * (n - d))

        se = survival * math.sqrt(greenwood)
        out.append(
            {
                **row,
                "hazard": hazard,
                "survival": survival,
                "se": se,
                **dict(zip(("ci_low", "ci_high"), _loglog_ci(survival, greenwood))),
            }
        )
    return out


def _loglog_ci(survival: float, greenwood: float) -> tuple[float, float]:
    """95% log-log interval for S, degenerating gracefully at the ends.

    At S = 1 (no events yet) the interval is a point: there is no sampling
    error in "nobody has churned yet" to express, and log(1) = 0 would divide
    by zero on the way to pretending otherwise.
    """
    if survival >= 1.0 or survival <= 0.0 or greenwood <= 0.0:
        return survival, survival

    log_s = math.log(survival)
    # se of log(-log S)
    se = math.sqrt(greenwood) / abs(log_s)
    low, high = (
        survival ** math.exp(1.96 * se),
        survival ** math.exp(-1.96 * se),
    )
    return low, high


def kaplan_meier(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Pooled KM curve: one row per tenure month, in order."""
    return _accumulate(risk_table(con))


def kaplan_meier_by_segment(
    con: duckdb.DuckDBPyConnection, segment: str = "plan"
) -> dict[str, list[dict]]:
    """One KM curve per level of `segment`, each accumulated independently."""
    if segment not in SEGMENTS:
        raise ValueError(f"unknown segment {segment!r}; expected one of {SEGMENTS}")

    grouped: dict[str, list[dict]] = {}
    for row in risk_table(con, segment):
        grouped.setdefault(row[segment], []).append(row)
    return {level: _accumulate(rows) for level, rows in sorted(grouped.items())}


def median_survival(curve: list[dict]) -> int | None:
    """First tenure month where S(t) <= 0.5, or None if it never gets there.

    Returning None matters more than it looks. Median survival is the headline
    number people ask for, and when a curve never reaches 0.5 -- which is the
    case here, since most subscribers are still subscribed -- the honest answer
    is "not reached within the observation window". The alternatives are to
    report the last observed tenure (wrong, and biased by window length) or to
    extrapolate the curve (an assumption dressed as a measurement).
    """
    for row in curve:
        if row["survival"] <= 0.5:
            return row["tenure"]
    return None


def logrank(con: duckdb.DuckDBPyConnection, segment: str = "plan") -> dict:
    """Log-rank test across the levels of `segment`.

    Per tenure month, each group's churn count is compared with what it would
    have been if all groups shared one hazard that month -- expected
    `e_gk = n_gk * d_k / n_k`. Summed over months, the statistic is

        chi2 = sum over groups of (O_g - E_g)^2 / E_g

    with `len(groups) - 1` degrees of freedom. That is the standard
    approximation to the full form (which uses the covariance matrix of the
    hypergeometric variances); it is slightly conservative and does not need a
    matrix inverse. Reported with the statistic, the per-group observed and
    expected counts, and the critical value, so a reader can see the comparison
    rather than take a verdict on trust.
    """
    if segment not in SEGMENTS:
        raise ValueError(f"unknown segment {segment!r}; expected one of {SEGMENTS}")

    pooled = {row["tenure"]: row for row in risk_table(con)}
    observed: dict[str, float] = {}
    expected: dict[str, float] = {}
    for row in risk_table(con, segment):
        level, overall = row[segment], pooled[row["tenure"]]
        observed[level] = observed.get(level, 0.0) + row["events"]
        expected[level] = expected.get(level, 0.0) + row["at_risk"] * (
            overall["events"] / overall["at_risk"]
        )

    statistic = sum(
        (observed[level] - expected[level]) ** 2 / expected[level]
        for level in observed
        if expected[level] > 0
    )
    df = len(observed) - 1
    critical = CHI2_CRITICAL_05.get(df)
    return {
        "segment": segment,
        "observed": observed,
        "expected": expected,
        "statistic": statistic,
        "df": df,
        "critical_05": critical,
        "significant": None if critical is None else statistic > critical,
    }


def render(store: str | Path = DEFAULT_STORE, max_tenure: int = 12) -> str:
    store = Path(store)
    if not store.exists():
        raise FileNotFoundError(
            f"no store at {store} -- run `python -m src.store` to build one first"
        )

    with duckdb.connect(str(store), read_only=True) as con:
        curve = kaplan_meier(con)
        by_plan = kaplan_meier_by_segment(con, "plan")
        test = logrank(con, "plan")

    median = median_survival(curve)
    lines = [
        f"Kaplan-Meier survival -- {store}",
        "",
        "A censored subscriber counts in the denominator for the months they",
        "were observed and leaves the risk set afterwards. Intervals are 95%",
        "log-log, so they cannot claim survival above 1.",
        "",
        "tenure  at risk  churned  censored  hazard  survival      95% CI",
        "------  -------  -------  --------  ------  --------  --------------",
    ]
    lines += [
        f"{r['tenure']:>6}  {r['at_risk']:>7}  {r['events']:>7}  {r['censored']:>8}  "
        f"{r['hazard']:>6.2%}  {r['survival']:>8.1%}  "
        f"{r['ci_low']:>6.1%}-{r['ci_high']:<7.1%}"
        for r in curve
        if r["tenure"] <= max_tenure
    ]

    lines += [
        "",
        "Median survival: "
        + (
            f"tenure {median}"
            if median is not None
            else "not reached within the observation window "
            f"(lowest S = {curve[-1]['survival']:.1%} at tenure {curve[-1]['tenure']})"
        ),
        "",
        "Survival by plan",
        "",
        "tenure  " + "".join(f"{plan:>10}" for plan in by_plan),
        "------  " + "".join("  --------" for _ in by_plan),
    ]
    depth = min(max_tenure, min(len(rows) for rows in by_plan.values()) - 1)
    lines += [
        f"{tenure:>6}  "
        + "".join(f"{rows[tenure]['survival']:>9.1%} " for rows in by_plan.values())
        for tenure in range(depth + 1)
    ]

    verdict = (
        "a difference this large is not sampling noise"
        if test["significant"]
        else "not distinguishable from sampling noise"
    )
    lines += [
        "",
        f"Log-rank across plans: chi2 = {test['statistic']:.1f} on {test['df']} df, "
        f"critical value {test['critical_05']} at alpha 0.05 -- {verdict}.",
        "",
        "plan        churned  expected if plans shared one hazard",
        "--------  ---------  -----------------------------------",
    ]
    lines += [
        f"{level:<8}  {test['observed'][level]:>9.0f}  {test['expected'][level]:>35.1f}"
        for level in sorted(test["observed"])
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default=DEFAULT_STORE, type=Path)
    parser.add_argument(
        "--max-tenure",
        type=int,
        default=12,
        help="deepest tenure month to print (the curve is computed in full)",
    )
    args = parser.parse_args(argv)
    print(render(args.store, args.max_tenure))


if __name__ == "__main__":
    main()
