"""DuckDB load for generated lifecycles.

The store is split by the ground-truth contract. Two tables are quarantined --
`hazard_truth` (the planted coefficients) and `subscriber_truth` (each
subscriber's latent engagement level) -- and analyses must not read either;
tests do, to check an estimate against the process it came from. Everything
else is analysis-facing, including `run_meta`, since the length of the
observation window is something an analyst legitimately knows.

Splitting the subscriber row across two tables is what makes the quarantine
structural rather than a promise. See docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from src import generate
from src.hazard import DEFAULT_MODEL, HazardModel

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_STORE = PROJECT_ROOT / "data" / "churnlens.duckdb"
WINDOW_START = date(2025, 1, 1)

# Subscriber fields that are ground truth rather than things an analyst could
# observe. These go to subscriber_truth; everything else goes to subscribers.
LATENT_FIELDS = frozenset({"engagement"})

# Why the split: leaving the latent level on an analysis-facing table let an
# analysis read the answer without ever touching hazard_truth, which a mutation
# test caught doing exactly that -- and it produced a *cleaner* result, so
# nothing would have looked wrong. A structural quarantine beats a per-module
# guard because days 3-6 add three more analysis modules.
SCHEMA = """
CREATE OR REPLACE TABLE subscribers (
    subscriber_id  VARCHAR PRIMARY KEY,
    plan           VARCHAR NOT NULL,
    signup_month   INTEGER NOT NULL,
    churn_month    INTEGER,
    censored       BOOLEAN NOT NULL
);
CREATE OR REPLACE TABLE subscriber_truth (
    subscriber_id  VARCHAR PRIMARY KEY,
    engagement     DOUBLE  NOT NULL
);
CREATE OR REPLACE TABLE subscription_months (
    subscriber_id  VARCHAR NOT NULL,
    plan           VARCHAR NOT NULL,
    tenure_months  INTEGER NOT NULL,
    calendar_month DATE    NOT NULL,
    churned        BOOLEAN NOT NULL
);
CREATE OR REPLACE TABLE engagement_events (
    subscriber_id  VARCHAR NOT NULL,
    tenure_months  INTEGER NOT NULL,
    calendar_month DATE    NOT NULL,
    events         INTEGER NOT NULL
);
CREATE OR REPLACE TABLE hazard_truth (
    term  VARCHAR NOT NULL,
    kind  VARCHAR NOT NULL,
    value DOUBLE  NOT NULL
);
-- Run provenance, one row. Analysis-facing: the length of the observation
-- window is something an analyst legitimately knows, not a fact about the
-- churn process. Recorded rather than inferred because inferring it from
-- max(signup_month + tenure_months) is only exact when some subscriber is
-- observed to the final month -- true for any realistic size, silently wrong
-- on small data (demonstrated: a one-subscriber, 24-month store whose only
-- subscriber churned early reported a 20-month window).
CREATE OR REPLACE TABLE run_meta (
    window_months    INTEGER NOT NULL,
    subscriber_count INTEGER NOT NULL,
    seed             INTEGER NOT NULL,
    window_start     DATE    NOT NULL
);
"""


def _bulk_insert(con, table: str, rows: list[dict]) -> None:
    """Load rows through a DataFrame scan rather than row-at-a-time INSERT.

    `executemany` autocommits per statement, so a 12k-row load paid 12k disk
    syncs -- measured at ~34s, which was most of the test suite's runtime. One
    explicit transaction brings that to ~1.9s and a DataFrame scan to ~0.06s;
    pandas is already a dependency for the analysis, so the scan is free.

    BY NAME binds on column name, so a reordered dataclass field or dict key
    can't silently shift values into the wrong column.
    """
    if not rows:
        return
    frame = pd.DataFrame(rows)  # noqa: F841 -- read by DuckDB's replacement scan
    con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM frame")


def build_store(
    store: str | Path = DEFAULT_STORE,
    count: int = 4000,
    window_months: int = 24,
    seed: int = 7,
    model: HazardModel = DEFAULT_MODEL,
) -> dict:
    store = Path(store)
    store.parent.mkdir(parents=True, exist_ok=True)

    subscribers = generate.generate_subscribers(
        count=count, window_months=window_months, seed=seed, model=model
    )
    months = generate.month_rows(subscribers, WINDOW_START, window_months)
    events = generate.engagement_rows(subscribers, WINDOW_START, window_months, seed=seed)

    with duckdb.connect(str(store)) as con:
        con.execute(SCHEMA)
        # Split by LATENT_FIELDS rather than by hand-listing each table's
        # columns: a field added to Subscriber then lands in one table or the
        # other automatically, and a test asserts the two together account for
        # every field, so a new one cannot be silently dropped from the store.
        _bulk_insert(
            con,
            "subscribers",
            [
                {k: v for k, v in asdict(s).items() if k not in LATENT_FIELDS}
                for s in subscribers
            ],
        )
        _bulk_insert(
            con,
            "subscriber_truth",
            [
                {"subscriber_id": s.subscriber_id}
                | {f: getattr(s, f) for f in LATENT_FIELDS}
                for s in subscribers
            ],
        )
        _bulk_insert(con, "subscription_months", months)
        _bulk_insert(con, "engagement_events", events)
        # Left on executemany: as_rows() is one row per coefficient, so there is
        # nothing for a bulk path to win here.
        con.executemany("INSERT INTO hazard_truth VALUES (?, ?, ?)", model.as_rows())
        con.execute(
            "INSERT INTO run_meta VALUES (?, ?, ?, ?)",
            # len(subscribers), not the requested count: provenance should
            # record what was actually loaded, not what was asked for.
            [window_months, len(subscribers), seed, WINDOW_START],
        )

    churned = sum(1 for s in subscribers if not s.censored)
    return {
        "store": str(store),
        "subscribers": len(subscribers),
        "churned": churned,
        "censored": len(subscribers) - churned,
        "subscriber_months": len(months),
        "engagement_rows": len(events),
    }


def summarise(stats: dict) -> str:
    return (
        f"{stats['subscribers']} subscribers "
        f"({stats['churned']} churned, {stats['censored']} censored), "
        f"{stats['subscriber_months']} subscriber-months -> {stats['store']}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the ChurnLens DuckDB store.")
    parser.add_argument("--count", type=int, default=4000, help="subscribers to draw")
    parser.add_argument(
        "--window-months", type=int, default=24, help="length of the observation window"
    )
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = parser.parse_args(argv)

    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.window_months < 1:
        parser.error("--window-months must be at least 1")

    print(
        summarise(
            build_store(
                store=args.store,
                count=args.count,
                window_months=args.window_months,
                seed=args.seed,
            )
        )
    )


if __name__ == "__main__":
    main()
