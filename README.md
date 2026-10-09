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
src/survival.py     Kaplan-Meier survival, log-rank comparison between plans
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
python -m src.survival         # Kaplan-Meier curve, median, log-rank across plans
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

## Survival analysis (day 3)

`python -m src.survival` prints the Kaplan-Meier curve with its risk table,
median survival, per-plan curves and a log-rank test. Censoring is the reason
this exists rather than a retention percentage: most subscribers here have not
churned *and* have not been observed to the end of their life either.

```
tenure  at risk  churned  censored  hazard  survival      95% CI
     0     4000      168       137   4.20%     95.8%   95.1%-96.4%
     1     3695       94       173   2.54%     93.4%   92.5%-94.1%
     6     2629       34       146   1.29%     86.0%   84.9%-87.1%
    12     1936       19       141   0.98%     82.8%   81.5%-84.1%

Median survival: not reached within the observation window (lowest S = 75.9%)
Log-rank across plans: chi2 = 49.3 on 2 df, critical value 5.991 -- not noise
```

Three things in that output are deliberate:

- **"Not reached"** rather than a median. The curve bottoms out at 75.9%,
  nowhere near 0.5. Printing the last observed tenure instead would report a
  number set by the window length, not by churn.
- **Log-log intervals**, so the upper bound cannot exceed 1. The usual
  `S +/- 1.96*se` does exactly that near the top of a curve; on a
  40-subscriber store it reaches 1.023, which is a claim that more than
  everyone survived.
- **The risk table printed beside the estimate**, because `at risk` falling
  from 4000 to 1936 is the context that makes the later numbers readable.

The naive alternative is quantified rather than dismissed: "1 - churned/total"
gives 82.9% where KM gives 75.9%, seven points of optimism that gets *better*
the shorter the window, since a shorter window censors more people. Counting
the end of observation as an exit, or keeping only subscribers with an observed
churn, both collapse survival to 0.0%. All three are tests.

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

Day 3 of 7. The hazard model, the lifecycle generator with correct censoring,
the DuckDB load, the sanity report, cohort retention and Kaplan–Meier survival
are in place and tested. Every analysis so far lands on the process it was
drawn from: a 3.920% predicted first-month hazard against 3.833% observed (0.34
binomial se), cohort retention within 4 se of the planted survival function at
every tenure checked, and the KM curve within 0.17–2.18 se of it across three
seeds. The churn-driver model — does a fitted model recover the planted
coefficients and their ordering — is day 4.
