# ChurnLens

Subscription retention analytics on data generated from a **known** churn
process — cohort retention, survival analysis and churn drivers, each checkable
against the hazard model the data actually came from.

**Domain:** Data Analytics · **Week:** 02 · **Started:** 2026-10-01

## Why this project

Cohort retention tables, Kaplan–Meier curves and "churn driver" models are
standard analytics output, and most published examples are unverifiable: they
compute a number from a dataset whose true churn process nobody knows, so a
correct analysis and a plausible-looking wrong one are indistinguishable.

ChurnLens generates subscriber lifecycles from an explicit hazard model with
fixed coefficients, so the ground truth exists. Every analysis can then be
asked a real question: does it recover the process it was drawn from?

Week 01 ([RetailPulse](https://github.com/SaurabhMastud/retailpulse)) was
pipeline engineering, where correctness means the data arrives intact. Here
correctness means an *inference* is right, which needs a different kind of
test — and ground truth is what makes those tests evidence rather than
decoration.

## The ground-truth contract

The generator writes the coefficients it used into a `hazard_truth` table
beside the data. **Analyses never read that table; tests do.** The channel is
deliberately one-way: an analysis cannot be tuned against the answer, but a
test can assert an estimate lands within tolerance of the truth.

This is the project's central design decision — see `docs/ARCHITECTURE.md`.

## Project layout

```
src/hazard.py       the churn process: coefficients, monthly hazard (ground truth)
src/generate.py     subscriber lifecycles, subscriber-months, engagement events
src/store.py        DuckDB load, including hazard_truth
tests/              pytest suite
docs/               architecture notes + the day-7 reports
data/               local DuckDB store (git-ignored)
```

## Running it

```bash
pip install -r requirements.txt
python -m src.store            # generate lifecycles and load data/churnlens.duckdb
python -m src.hazard           # hazard model self-check
python -m pytest tests/ -q
```

## Data model

| Table | Grain | What it holds |
|---|---|---|
| `subscribers` | one subscriber | plan, latent engagement, signup month, churn month, censoring flag |
| `subscription_months` | subscriber × tenure month | the observed active months, with the churn month flagged |
| `engagement_events` | subscriber × tenure month | monthly activity counts — the *observable* engagement signal |
| `hazard_truth` | one coefficient | the planted coefficients, for tests only |

Subscribers still active when the observation window closes are
**right-censored**: `churn_month` is null and `censored` is true. Signups are
spread across the window, so cohorts differ in how long they have been
observable — without that, every cohort has identical exposure and the
censoring the survival analysis exists to handle never appears.

`subscription_months` and `engagement_events` share the **same key set** —
every `(subscriber_id, tenure_months)` in one is in the other, churn months
included. That is deliberate and tested in both Python and SQL: when engagement
stopped a month short of the churn row, the two tables disagreed on exactly the
churned months, which makes an inner join drop every churn event and a
zero-filled left join predict churn perfectly from leakage alone.

Note that `engagement` on `subscribers` is the *latent* level the hazard uses,
while `engagement_events` is what an analyst can actually see. Recovering the
engagement effect from observed activity is the day-4 problem, not a lookup.

## Status

Day 1 of 7. The hazard model, the lifecycle generator with correct censoring,
and the DuckDB load are in place and tested (57 tests). Cohort retention is
day 2.
