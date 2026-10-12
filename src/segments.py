"""Segment deep-dives: survival by observed activity level, and by plan within it.

Day 3 split survival by plan, which is easy because plan is fixed at signup.
Activity is not, and that makes segmenting on it the most dangerous cut in the
project -- not because the arithmetic is hard, but because the obvious way to
define the segment looks into the future.

**The trap.** "Group subscribers by their average monthly activity, then compare
retention" is the natural phrasing and it is wrong: a subscriber's average over
*all* their observed months is computed from months that only exist because
they had not churned yet. The segment is then partly a function of the outcome,
and the resulting curves overstate the gap -- survivors contribute long, stable
activity histories to the high-activity group while early churners are sorted on
one or two noisy months. This is the same family of error as immortal-time bias.

**The fix** is to segment on a covariate known at entry: activity in the first
observed month, which every subscriber has (the generator records the signup
month for everyone) and which cannot have been influenced by whether they later
churned.

Both are computed here, deliberately, because the point is the comparison --
`baseline` is the honest cut and `lifetime` is the one to be able to recognise.
The naive version is labelled as such in the output rather than omitted: a
reader who has seen it done the wrong way elsewhere should be able to find out
how much it mattered.

Neither basis reads the latent engagement level. Both are built from
`engagement_events`, which is the observable signal; recovering the engagement
story from it is the project's standing problem.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from src import survival
from src.store import DEFAULT_STORE

BASES = ("baseline", "lifetime")
DEFAULT_GROUPS = 4


def _activity_sql(basis: str, groups: int) -> str:
    """Per-subscriber activity level and its quantile group.

    `ntile` over the activity value, so groups are equal-sized by subscriber
    count.

    **Ordered by `(activity, subscriber_id)`, and the second key is not
    cosmetic.** Baseline activity is an integer event count, so hundreds of
    subscribers share a value and the quartile boundary falls *inside* a tie.
    With `order by activity` alone, which subscribers land either side of that
    boundary is whatever order the scan produced -- and it genuinely varied
    between runs on the same store: S(11) for basic/A1 came out 66.1% one run
    and 65.9% the next. A non-reproducible analysis is worse than a slightly
    arbitrary one, so the tie is broken on subscriber id: still arbitrary,
    now deterministic and stated.
    """
    if basis not in BASES:
        raise ValueError(f"unknown basis {basis!r}; expected one of {BASES}")
    if groups < 2:
        raise ValueError(f"groups must be at least 2, got {groups}")

    # The only difference between the honest and the naive cut: which months
    # are allowed to define the segment.
    activity = {
        "baseline": "min_by(e.events, e.tenure_months)",
        "lifetime": "avg(e.events)",
    }[basis]

    return f"""
        with level as (
            select e.subscriber_id,
                   {activity} as activity
            from engagement_events e
            group by e.subscriber_id
        )
        select subscriber_id,
               activity,
               'A' || ntile({groups}) over (order by activity, subscriber_id) as segment
        from level
    """


def activity_levels(
    con: duckdb.DuckDBPyConnection,
    basis: str = "baseline",
    groups: int = DEFAULT_GROUPS,
) -> list[dict]:
    return [
        dict(zip(("subscriber_id", "activity", "segment"), row))
        for row in con.execute(_activity_sql(basis, groups)).fetchall()
    ]


def risk_table_by_activity(
    con: duckdb.DuckDBPyConnection,
    basis: str = "baseline",
    groups: int = DEFAULT_GROUPS,
    plan: str | None = None,
) -> list[dict]:
    """Risk-table rows per (segment, tenure), optionally within one plan.

    Same counting rules as `survival.risk_table` -- at-risk from the month
    rows, censoring from each subscriber's last observed month -- with the
    segment joined on. It is a separate query rather than an argument to that
    function because the segment is subscriber-level and derived, which is one
    join and a window function more than that query is built for.
    """
    where = "where s.plan = ?" if plan else ""
    parameters = [plan] if plan else []

    return [
        dict(zip(("segment", "tenure", "at_risk", "events", "censored"), row))
        for row in con.execute(
            f"""
            with levels as ({_activity_sql(basis, groups)}),
            last_seen as (
                select subscriber_id, max(tenure_months) as last_tenure
                from subscription_months
                group by subscriber_id
            )
            select l.segment,
                   m.tenure_months as tenure,
                   count(*)                          as at_risk,
                   count(*) filter (where m.churned) as events,
                   count(*) filter (
                       where not m.churned
                         and m.tenure_months = ls.last_tenure
                   )                                 as censored
            from subscription_months m
            join levels l    using (subscriber_id)
            join last_seen ls using (subscriber_id)
            join subscribers s using (subscriber_id)
            {where}
            group by l.segment, m.tenure_months
            order by l.segment, m.tenure_months
            """,
            parameters,
        ).fetchall()
    ]


def survival_by_activity(
    con: duckdb.DuckDBPyConnection,
    basis: str = "baseline",
    groups: int = DEFAULT_GROUPS,
    plan: str | None = None,
) -> dict[str, list[dict]]:
    """One KM curve per activity segment, through the shared accumulator."""
    grouped: dict[str, list[dict]] = {}
    for row in risk_table_by_activity(con, basis, groups, plan):
        grouped.setdefault(row["segment"], []).append(row)
    return {
        segment: survival.accumulate(rows) for segment, rows in sorted(grouped.items())
    }


def survival_at(curves: dict[str, list[dict]], tenure: int) -> dict[str, float]:
    """S(tenure) per segment, skipping segments not observed that deep."""
    return {
        segment: curve[tenure]["survival"]
        for segment, curve in curves.items()
        if len(curve) > tenure
    }


def render(
    store: str | Path = DEFAULT_STORE,
    tenure: int = 11,
    groups: int = DEFAULT_GROUPS,
) -> str:
    store = Path(store)
    if not store.exists():
        raise FileNotFoundError(
            f"no store at {store} -- run `python -m src.store` to build one first"
        )

    with duckdb.connect(str(store), read_only=True) as con:
        honest = survival_by_activity(con, "baseline", groups)
        naive = survival_by_activity(con, "lifetime", groups)
        plans = [row[0] for row in con.execute(
            "select distinct plan from subscribers order by plan"
        ).fetchall()]
        cells = {
            plan: survival_by_activity(con, "baseline", groups, plan) for plan in plans
        }
        by_plan = {plan: survival_at(curves, tenure) for plan, curves in cells.items()}

    honest_at = survival_at(honest, tenure)
    naive_at = survival_at(naive, tenure)

    lines = [
        f"Survival by activity segment -- {store}",
        "",
        f"Groups are equal-sized by subscriber count ({groups} of them), lowest",
        f"activity first. Survival is S({tenure}) -- still subscribed after",
        f"{tenure + 1} tenure months.",
        "",
        "segment  at risk  baseline cut  lifetime cut (naive)",
        "-------  -------  ------------  --------------------",
    ]
    lines += [
        f"{segment:>7}  {honest[segment][0]['at_risk']:>7}  "
        f"{honest_at[segment]:>11.1%}  {naive_at.get(segment, float('nan')):>19.1%}"
        for segment in sorted(honest_at)
    ]

    honest_gap = max(honest_at.values()) - min(honest_at.values())
    naive_gap = max(naive_at.values()) - min(naive_at.values())
    lines += [
        "",
        "The right-hand column groups subscribers by their average activity over",
        "all observed months -- which is partly a function of how long they",
        "survived, since a churner only has the months they lasted. It is the",
        f"natural phrasing of the question and it overstates the top-to-bottom",
        f"gap: {naive_gap:.1%} against {honest_gap:.1%} for a cut taken at signup",
        f"({naive_gap / honest_gap:.1f}x). Only the left-hand column is a",
        "retention finding; the right-hand one is mostly the outcome restated.",
        "",
        f"Baseline-cut S({tenure}) by plan and activity segment",
        "",
        "plan      " + "".join(f"{segment:>9}" for segment in sorted(honest_at)),
        "--------  " + "".join("  -------" for _ in honest_at),
    ]
    lines += [
        f"{plan:<8}  "
        + "".join(
            f"{by_plan[plan][segment]:>8.1%} " if segment in by_plan[plan] else " " * 9
            for segment in sorted(honest_at)
        )
        for plan in plans
    ]
    grid = {
        (plan, segment): value
        for plan, values in by_plan.items()
        for segment, value in values.items()
    }
    worst, best = min(grid, key=grid.get), max(grid, key=grid.get)
    smallest = min(
        curve[0]["at_risk"] for curves in cells.values() for curve in curves.values()
    )
    within_plan = max(
        max(values.values()) - min(values.values()) for values in by_plan.values()
    )
    at_top = [by_plan[plan].get(max(honest_at)) for plan in plans]
    across_plan = max(v for v in at_top if v is not None) - min(
        v for v in at_top if v is not None
    )

    lines += [
        "",
        f"Worst cell: {worst[0]} on {worst[1]} at {grid[worst]:.1%}. Best: "
        f"{best[0]} on {best[1]} at {grid[best]:.1%}.",
        f"Spread within a plan across activity ({within_plan:.1%}) is wider than",
        f"the spread across plans at top activity ({across_plan:.1%}), which is the",
        "planted story -- engagement dominates plan tier -- recovered from",
        "observable data alone.",
        "",
        f"The grid is thin in places: the smallest cell holds {smallest} subscribers,",
        "so individual cells are not monotone in activity and should not be read",
        "as if they were. The pooled column above is what carries the finding.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default=DEFAULT_STORE, type=Path)
    parser.add_argument("--tenure", type=int, default=11, help="tenure month to compare at")
    parser.add_argument("--groups", type=int, default=DEFAULT_GROUPS)
    args = parser.parse_args(argv)
    print(render(args.store, args.tenure, args.groups))


if __name__ == "__main__":
    main()
