"""Signup-cohort retention matrix and the pooled retention curve.

The whole difficulty here is **unequal exposure**. Signups are spread across the
observation window, so the cohort that joined in the last month has one month
of exposure while the first cohort has the full window. A cohort cannot be
observed at a tenure it has not had time to reach.

That produces the standard way a retention table comes out wrong: treating an
unreachable cell as 0% retained instead of as unknown. It looks plausible --
retention appears to decay faster at long tenures -- and it is entirely an
artifact of young cohorts being counted as total losses. So an unobservable
cell is omitted here, never zeroed, and the pooled curve at tenure t averages
only over cohorts with exposure at t.

Retention convention: a subscriber who churns in month c was subscribed at the
start of month c, so they count as retained at tenure c and not at c + 1.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from src.store import DEFAULT_STORE


def window_months(con: duckdb.DuckDBPyConnection) -> int:
    """Length of the observation window, read from the store's run metadata.

    This was originally derived as `max(signup_month + tenure_months) + 1`,
    which is exact only when some subscriber is actually observed to the final
    month. That holds for any realistic size -- most subscribers are censored
    -- and fails silently otherwise: a one-subscriber 24-month store whose only
    subscriber signed up in month 15 and churned at tenure 4 reported a
    20-month window, with nothing to indicate it was wrong. A recorded integer
    beats an inference that is usually right.
    """
    row = con.execute("select window_months from run_meta").fetchone()
    if row is None:
        raise ValueError("store has no run_meta row -- rebuild it with src.store")
    return row[0]


def last_month(con: duckdb.DuckDBPyConnection) -> int:
    """Index of the final calendar month in the window."""
    return window_months(con) - 1


def retention_matrix(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """One row per (cohort, tenure) cell that is actually observable.

    `retained` counts subscribers still subscribed at the start of that tenure
    month: censored subscribers always, churned ones while `churn_month >=
    tenure`. Cells a cohort could not reach are absent from the result rather
    than present with a zero -- see the module docstring.
    """
    return [
        dict(
            zip(
                ("cohort", "tenure", "cohort_size", "retained", "retention"),
                row,
            )
        )
        for row in con.execute(
            f"{_matrix_sql()} order by cohort, tenure"
        ).fetchall()
    ]


def retention_curve(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Pooled retention by tenure, over cohorts with exposure at that tenure.

    Pools subscribers rather than averaging cohort percentages: an average of
    rates weights a 150-subscriber cohort the same as a 180-subscriber one, and
    because exposure correlates with cohort age that bias is not random. The
    `cohorts` count is reported so a reader can see the pool thinning.
    """
    return [
        dict(
            zip(
                ("tenure", "cohorts", "at_risk", "retained", "retention"),
                row,
            )
        )
        for row in con.execute(
            f"""
            with matrix as ({_matrix_sql()})
            select tenure,
                   count(*)                       as cohorts,
                   sum(cohort_size)               as at_risk,
                   sum(retained)                  as retained,
                   sum(retained) / sum(cohort_size) as retention
            from matrix
            group by tenure
            order by tenure
            """
        ).fetchall()
    ]


# Columns of `subscribers` a retention table may be split by. An allowlist
# because the name is interpolated into SQL; these are internal callers today,
# but a validated identifier costs one line and removes the question.
SEGMENTS = ("plan",)


def _matrix_sql(segment: str | None = None) -> str:
    """The cell-level retention query, optionally split by a segment column.

    Parameterised rather than copied so that **the exposure rule exists once**.
    Which cells are observable is the one piece of logic every function here
    depends on; a second hand-written copy drifting from this one would make
    the matrix and the curve disagree about which cohorts were counted, and
    nothing would obviously look wrong.
    """
    if segment is not None and segment not in SEGMENTS:
        raise ValueError(f"unknown segment {segment!r}; expected one of {SEGMENTS}")

    column = f", {segment}" if segment else ""
    alias = f", c.{segment}" if segment else ""
    match = f" and sub.{segment} = c.{segment}" if segment else ""

    return f"""
        with sized as (
            select signup_month as cohort{column}, count(*) as cohort_size
            from subscribers
            group by signup_month{column}
        ),
        -- The recorded window, not one inferred from the rows present: a
        -- cohort's exposure must not shrink just because its subscribers
        -- happened to churn early.
        bounds as (
            select window_months - 1 as last_month from run_meta
        ),
        cells as (
            select s.cohort{column}, s.cohort_size, t.tenure
            from sized s
            cross join (
                select unnest(range(0, (select last_month + 1 from bounds)))
                    as tenure
            ) t
            where s.cohort + t.tenure <= (select last_month from bounds)
        )
        select c.cohort,
               c.tenure{alias},
               c.cohort_size,
               count(sub.subscriber_id) as retained,
               count(sub.subscriber_id) / c.cohort_size as retention
        from cells c
        left join subscribers sub
          on sub.signup_month = c.cohort
         and (sub.churn_month is null or sub.churn_month >= c.tenure){match}
        group by c.cohort, c.tenure{alias}, c.cohort_size
    """


def retention_curve_by_segment(
    con: duckdb.DuckDBPyConnection, segment: str = "plan"
) -> list[dict]:
    """Pooled retention by tenure within each level of `segment`.

    Same exposure handling as the pooled curve -- a cohort still only
    contributes at tenures it reached -- so the per-plan curves are comparable
    to each other and to the overall one.
    """
    if segment not in SEGMENTS:
        raise ValueError(f"unknown segment {segment!r}; expected one of {SEGMENTS}")

    return [
        dict(zip((segment, "tenure", "cohorts", "at_risk", "retained", "retention"), row))
        for row in con.execute(
            f"""
            with matrix as ({_matrix_sql(segment)})
            select {segment},
                   tenure,
                   count(*)                         as cohorts,
                   sum(cohort_size)                 as at_risk,
                   sum(retained)                    as retained,
                   sum(retained) / sum(cohort_size) as retention
            from matrix
            group by {segment}, tenure
            order by {segment}, tenure
            """
        ).fetchall()
    ]


def render(store: str | Path = DEFAULT_STORE, max_tenure: int = 12) -> str:
    store = Path(store)
    if not store.exists():
        raise FileNotFoundError(
            f"no store at {store} -- run `python -m src.store` to build one first"
        )

    with duckdb.connect(str(store), read_only=True) as con:
        matrix = retention_matrix(con)
        curve = retention_curve(con)
        by_plan = retention_curve_by_segment(con, "plan")
        window = window_months(con)

    by_cell = {(r["cohort"], r["tenure"]): r["retention"] for r in matrix}
    cohorts = sorted({r["cohort"] for r in matrix})
    tenures = list(range(min(max_tenure + 1, window)))

    header = "cohort  " + "".join(f"{t:>7}" for t in tenures)
    lines = [
        f"Signup-cohort retention -- {store}",
        "",
        f"Tenure months 0-{tenures[-1]} of a {window}-month window. A blank cell is a",
        "month the cohort has not had time to reach -- not a month everyone left.",
        "",
        header,
        "-" * len(header),
    ]
    for cohort in cohorts:
        cells = "".join(
            f"{by_cell[(cohort, t)]:>6.0%} " if (cohort, t) in by_cell else " " * 7
            for t in tenures
        )
        lines.append(f"{cohort:>6}  {cells}")

    lines += [
        "",
        "Pooled retention by tenure (only cohorts exposed at that tenure)",
        "",
        "tenure  cohorts  at risk  retained  retention",
        "------  -------  -------  --------  ---------",
    ]
    lines += [
        f"{r['tenure']:>6}  {r['cohorts']:>7}  {r['at_risk']:>7}  "
        f"{r['retained']:>8}  {r['retention']:>8.1%}"
        for r in curve
        if r["tenure"] in tenures
    ]

    by_plan_cell = {(r["plan"], r["tenure"]): r["retention"] for r in by_plan}
    # Ordered best-retaining first, from the data. Alphabetical would put
    # 'premium' between 'basic' and 'standard' and read as a tier order that
    # isn't one; the tier list itself lives in the hazard module, which an
    # analysis may not import. Sorting on an observed column is a display
    # choice, not an inference.
    deepest = max(r["tenure"] for r in by_plan)
    plans = sorted(
        {r["plan"] for r in by_plan},
        key=lambda p: by_plan_cell.get((p, deepest), 0.0),
        reverse=True,
    )
    plan_header = "tenure  " + "".join(f"{p:>10}" for p in plans)
    lines += [
        "",
        "Retention by plan (same exposure handling, so the curves are comparable)",
        "",
        plan_header,
        "-" * len(plan_header),
    ]
    lines += [
        f"{t:>6}  "
        + "".join(
            f"{by_plan_cell[(p, t)]:>9.1%} "
            if (p, t) in by_plan_cell
            else " " * 10
            for p in plans
        )
        for t in tenures
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default=DEFAULT_STORE, type=Path)
    parser.add_argument(
        "--max-tenure",
        type=int,
        default=12,
        help="widest tenure column to print (the matrix stays full internally)",
    )
    args = parser.parse_args(argv)
    print(render(args.store, args.max_tenure))


if __name__ == "__main__":
    main()
