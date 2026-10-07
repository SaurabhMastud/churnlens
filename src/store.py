"""DuckDB load for generated lifecycles.

The hazard coefficients are written to `hazard_truth` alongside the data.
Analyses must not read that table -- tests do, to check an estimate against the
process it came from. See the ground-truth contract in docs/ARCHITECTURE.md.
"""
from __future__ import annotations

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

SCHEMA = """
CREATE OR REPLACE TABLE subscribers (
    subscriber_id  VARCHAR PRIMARY KEY,
    plan           VARCHAR NOT NULL,
    engagement     DOUBLE  NOT NULL,
    signup_month   INTEGER NOT NULL,
    churn_month    INTEGER,
    censored       BOOLEAN NOT NULL
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
        _bulk_insert(con, "subscribers", [asdict(s) for s in subscribers])
        _bulk_insert(con, "subscription_months", months)
        _bulk_insert(con, "engagement_events", events)
        # Left on executemany: as_rows() is one row per coefficient, so there is
        # nothing for a bulk path to win here.
        con.executemany("INSERT INTO hazard_truth VALUES (?, ?, ?)", model.as_rows())

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


if __name__ == "__main__":
    print(summarise(build_store()))
