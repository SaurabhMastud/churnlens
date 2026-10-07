"""Read-only sanity report over the generated store.

This is the first *analysis* in the project, so it is the first code the
ground-truth contract applies to: nothing here may read `hazard_truth`. The
tests read it and check these numbers against the planted coefficients -- the
channel stays one-way, which is what makes that check evidence rather than a
tautology. A test asserts this module never names the table.

Deliberately plain SQL aggregates, not the day-2 cohort matrix: the point is a
first look that can be compared to the process the data came from.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from src.store import DEFAULT_STORE

# Buckets over tenure rather than raw months: the monthly hazard at a single
# tenure is a thin slice (few hundred subscriber-months out at the tail), so
# raw per-month rates are too noisy to read a trend off. Open-ended at the top
# because exposure thins out there.
TENURE_BUCKETS = ((0, 0), (1, 2), (3, 5), (6, 11), (12, None))


def _label(low: int, high: int | None) -> str:
    """The one place a bucket's label is spelled."""
    if high is None:
        return f"{low}+"
    return f"{low}" if low == high else f"{low}-{high}"


def _bucket_case(column: str = "tenure_months") -> str:
    """SQL CASE mapping a tenure to its bucket label.

    Shares `_label` with `bucket_labels`, so the string SQL writes and the
    string the report looks up cannot drift apart -- the first version of this
    had the formatting written out twice, which is the drift it was supposed to
    prevent.
    """
    whens = []
    for low, high in TENURE_BUCKETS:
        label = _label(low, high)
        if high is None:
            whens.append(f"when {column} >= {low} then '{label}'")
        elif low == high:
            whens.append(f"when {column} = {low} then '{label}'")
        else:
            whens.append(f"when {column} between {low} and {high} then '{label}'")
    return "case " + " ".join(whens) + " end"


def bucket_labels() -> list[str]:
    """Bucket labels in tenure order -- the order the report prints them in."""
    return [_label(low, high) for low, high in TENURE_BUCKETS]


def churn_by_plan(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """One row per plan: how many subscribers, how many churned, what rate.

    Subscriber-level, so this is cumulative churn over the observation window,
    not a monthly hazard -- and it is sensitive to censoring, since a censored
    subscriber may yet churn after the window closes. That is exactly why day 3
    needs survival analysis; this number is a first look, not a retention rate.
    """
    return [
        dict(zip(("plan", "subscribers", "churned", "churn_rate"), row))
        for row in con.execute(
            """
            select plan,
                   count(*)                                        as subscribers,
                   count(*) filter (where not censored)            as churned,
                   count(*) filter (where not censored) / count(*) as churn_rate
            from subscribers
            group by plan
            order by churn_rate desc
            """
        ).fetchall()
    ]


def hazard_by_tenure(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Discrete-time monthly hazard by tenure bucket.

    Numerator is churn events in the bucket, denominator is subscriber-months
    *at risk* in it -- not subscribers. Dividing by subscribers instead is the
    standard way this comes out wrong, since a subscriber contributes a month of
    risk for every month they stayed, so long-tenured subscribers would be
    counted once while carrying a dozen chances to churn.
    """
    case = _bucket_case("m.tenure_months")
    rows = con.execute(
        f"""
        select {case}                                      as bucket,
               count(*)                                    as exposed_months,
               count(*) filter (where m.churned)           as churn_events,
               count(*) filter (where m.churned) / count(*) as monthly_hazard
        from subscription_months m
        group by bucket
        """
    ).fetchall()

    keyed = {
        row[0]: dict(
            zip(("bucket", "exposed_months", "churn_events", "monthly_hazard"), row)
        )
        for row in rows
    }
    # Order by the bucket definition, not by SQL -- '12+' sorts before '3-5'
    # as a string, which would print the trend out of sequence.
    return [keyed[label] for label in bucket_labels() if label in keyed]


def churn_by_activity_quartile(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Churn rate by quartile of *observed* mean monthly activity.

    The first look at the engagement signal as an analyst actually has it --
    event counts, not the latent level on `subscribers`, which this must not
    use. Engagement is the model's dominant lever, so a clear gradient here is
    the strongest single sanity check that the generated data carries it.

    Every subscriber has at least one engagement month, so the join drops
    nobody. That is only true since the churn month gained an engagement row:
    before that, a subscriber who churned in their first month had no
    engagement rows at all and would have vanished from this cut entirely --
    and they are the highest-churn subscribers there are.
    """
    return [
        dict(
            zip(
                (
                    "quartile",
                    "subscribers",
                    "min_mean_events",
                    "max_mean_events",
                    "churn_rate",
                ),
                row,
            )
        )
        for row in con.execute(
            """
            with activity as (
                select subscriber_id, avg(events) as mean_events
                from engagement_events
                group by subscriber_id
            ),
            ranked as (
                select a.mean_events,
                       s.censored,
                       ntile(4) over (order by a.mean_events) as quartile
                from activity a
                join subscribers s using (subscriber_id)
            )
            select quartile,
                   count(*)                                        as subscribers,
                   min(mean_events)                                as min_mean_events,
                   max(mean_events)                                as max_mean_events,
                   count(*) filter (where not censored) / count(*) as churn_rate
            from ranked
            group by quartile
            order by quartile
            """
        ).fetchall()
    ]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    lines = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in rows]
    return "\n".join(lines)


def render(store: str | Path = DEFAULT_STORE) -> str:
    store = Path(store)
    if not store.exists():
        raise FileNotFoundError(
            f"no store at {store} -- run `python -m src.store` to build one first"
        )

    with duckdb.connect(str(store), read_only=True) as con:
        plans = churn_by_plan(con)
        tenure = hazard_by_tenure(con)
        activity = churn_by_activity_quartile(con)

    out = [
        f"ChurnLens sanity report -- {store}",
        "",
        "Cumulative churn by plan (observation window, censoring-sensitive)",
        _table(
            ["plan", "subscribers", "churned", "churn rate"],
            [
                [
                    r["plan"],
                    str(r["subscribers"]),
                    str(r["churned"]),
                    f"{r['churn_rate']:.1%}",
                ]
                for r in plans
            ],
        ),
        "",
        "Monthly churn hazard by tenure bucket (per subscriber-month at risk)",
        _table(
            ["tenure", "months at risk", "churn events", "monthly hazard"],
            [
                [
                    r["bucket"],
                    str(r["exposed_months"]),
                    str(r["churn_events"]),
                    f"{r['monthly_hazard']:.2%}",
                ]
                for r in tenure
            ],
        ),
        "",
        "Cumulative churn by quartile of observed mean monthly activity",
        _table(
            ["quartile", "subscribers", "mean events/month", "churn rate"],
            [
                [
                    f"Q{r['quartile']}",
                    str(r["subscribers"]),
                    f"{r['min_mean_events']:.1f}-{r['max_mean_events']:.1f}",
                    f"{r['churn_rate']:.1%}",
                ]
                for r in activity
            ],
        ),
        "",
        "  Quartiles are equal-sized, so subscribers tied on mean activity can",
        "  fall either side of a boundary -- adjacent ranges may share an",
        "  endpoint. Ties are common here because a short-tenured subscriber's",
        "  mean over one or two months is often a whole number.",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--store", default=DEFAULT_STORE, type=Path, help="DuckDB store to read"
    )
    args = parser.parse_args(argv)
    print(render(args.store))


if __name__ == "__main__":
    main()
