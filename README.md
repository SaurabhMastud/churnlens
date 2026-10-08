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

The coefficients are not the only ground truth in the store. The *latent*
engagement level the hazard was given is per-subscriber truth, and on day 1 it
sat on `subscribers` — an analysis-facing table — so an analysis could cheat
without ever reading `hazard_truth`. A mutation test caught exactly that:
swapping the latent level in for observed activity passed every other test and
produced a *cleaner* result.

Day 2 moved it to `subscriber_truth`, so the quarantine is structural — an
analysis cannot read a column that is not in the table it queries. The AST
guards remain as a second line, and now discover every analysis module
automatically rather than being opted into one at a time.

This is the project's central design decision — see `docs/ARCHITECTURE.md`.

## Project layout

```
src/hazard.py       the churn process: coefficients, monthly hazard (ground truth)
src/generate.py     subscriber lifecycles, subscriber-months, engagement events
src/store.py        DuckDB load, including hazard_truth
src/report.py       read-only sanity report (churn by plan, hazard by tenure)
src/cohorts.py      signup-cohort retention matrix, pooled and per-plan curves
tests/              pytest suite
docs/               architecture notes + the day-7 reports
data/               local DuckDB store (git-ignored)
```

## Running it

```bash
pip install -r requirements.txt
python -m src.store            # generate lifecycles and load data/churnlens.duckdb
python -m src.report           # read-only sanity report over the store
python -m src.cohorts          # signup-cohort retention matrix + retention curves
python -m src.hazard           # hazard model self-check
python -m pytest tests/ -q
```

The store build takes parameters — `--count`, `--window-months`, `--seed` and
`--store` — so a differently-sized or differently-seeded dataset is one flag
away, and generation is deterministic per seed:

```bash
python -m src.store --count 20000 --window-months 36 --seed 11
```

`python -m src.report` prints three cuts, and all three show the planted
story: cumulative churn falls across basic → standard → premium (20.8% →
10.2%), the monthly hazard falls from 4.2% in the first month to 0.7% past a
year's tenure, and churn falls from 29.0% to 9.3% across quartiles of
*observed* monthly activity. That last one is recovered from event counts
alone, never from the latent engagement level.

## Cohort retention (day 2)

`python -m src.cohorts` prints a signup-cohort retention matrix, the pooled
curve, and the same curve split by plan. The thing it takes seriously is
**unequal exposure**: signups are spread across the window, so the cohort that
joined last month has one month of exposure and cannot be observed at tenure 6.

A cell a cohort has not had time to reach is **left blank, never zeroed**. That
single choice is the difference between a retention table and a misleading one
— zero-filling makes young cohorts look like total losses and the curve appear
to collapse at long tenures, which reads as a real finding and is pure artifact.
With this dataset, zero-filling drags pooled retention at tenure 12 from 82.5%
down to 40.5% (and at tenure 6 from 87.3% to 65.7%), and a test asserts the
distortion rather than trusting the comment.

```
tenure  cohorts  at risk  retention
     1       23     3855      95.8%
     3       21     3498      91.4%
     6       18     3013      87.3%
    12       12     1966      82.5%
    23        1      164      72.6%
```

The pool thins with tenure, so the `cohorts` and `at risk` columns are printed
beside the rate rather than left implicit — 72.6% at tenure 23 rests on one
cohort of 164 subscribers and should not be read as the same kind of number as
95.8%. Retention is pooled over subscribers, not averaged over cohort rates,
because an average weights a 150-subscriber cohort the same as a 180-subscriber
one and cohort size correlates with exposure.

Checked against the hazard the data came from, not just eyeballed: predicted
survival from the planted coefficients and observed pooled retention agree
within 4 binomial standard errors at tenures 1, 2, 3, 5, 8 and 12, and the
per-plan curves recover the planted tier ordering (at tenure 12: basic 78.0%,
standard 86.0%, premium 91.8%) with the gap widening as tenure grows.

## Data model

| Table | Grain | What it holds |
|---|---|---|
| `subscribers` | one subscriber | plan, signup month, churn month, censoring flag |
| `subscriber_truth` | one subscriber | the *latent* engagement level the hazard was given — quarantined, for tests only |
| `subscription_months` | subscriber × tenure month | the observed active months, with the churn month flagged |
| `engagement_events` | subscriber × tenure month | monthly activity counts — the *observable* engagement signal |
| `hazard_truth` | one coefficient | the planted coefficients, for tests only |
| `run_meta` | one row per build | observation window, subscriber count, seed — dataset provenance |

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

`subscriber_truth.engagement` is the *latent* level the hazard uses, and no
analysis module may read it; `engagement_events` is what an analyst can actually
see. Recovering the engagement effect from observed activity is the day-4
problem, not a lookup.

## Status

Day 2 of 7. The hazard model, the lifecycle generator with correct censoring,
the DuckDB load, a read-only sanity report and the cohort retention analysis are
in place and tested. The estimate-vs-truth comparisons hold at both grains:
averaging the planted model over each subscriber's own plan and engagement
predicts a 3.920% first-month churn hazard against 3.833% observed (0.34
binomial standard errors), and pooled cohort retention matches the planted
survival function within 4 se at every tenure checked. Survival analysis
(Kaplan–Meier) is day 3.
