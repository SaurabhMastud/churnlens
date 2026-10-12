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
src/drivers.py      churn-driver model: coefficients, calibration, held-out scores
src/segments.py     survival by observed-activity segment, honest vs look-ahead cut
src/ltv.py          restricted expected value by plan and segment
dashboard/app.py    Streamlit dashboard over all of the above
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
python -m src.drivers          # churn drivers: coefficients, calibration, held-out scores
python -m src.segments         # survival by activity segment (and the naive cut beside it)
python -m src.ltv              # restricted expected value, by plan and activity segment
streamlit run dashboard/app.py # the dashboard (set CHURNLENS_STORE to point it elsewhere)
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

## Churn drivers (day 4)

`python -m src.drivers` fits a discrete-time logistic hazard on subscriber-months
— the unit the generator actually draws churn on — using only observable data:
plan, tenure, and that month's event count.

```
term             coefficient     se   planted
intercept             -2.290  0.092    -1.90
plan_standard         -0.452  0.092    -0.45
plan_premium          -0.813  0.137    -0.95
tenure_log1p          -0.625  0.041    -0.55
activity              -1.394  0.144    -2.40   <- attenuated
```

The `planted` column is not in the program's output — an analysis here may not
read the truth — but the test suite makes the comparison, and it is where the
day's two results come from:

1. **Fitted on the true regressor, the model recovers everything.** Substituting
   the latent engagement level for the observed event count (something only a
   test may do) puts every coefficient within 1.5 standard errors of its
   planted value, intercept included.
2. **Fitted on the observable proxy, the engagement effect attenuates to 58%
   of the truth** — −1.383 against −2.40, 8.3 standard errors away, so bias
   rather than noise. The size matches the classic reliability ratio (0.53
   predicted, 0.58 observed). Reporting −1.4 as "the engagement effect" would
   understate the business's biggest lever by nearly half, and nothing in the
   fit's own diagnostics would say so.

The evaluation is deliberately unglamorous. Held-out AUC is **0.680**, and an
upper bound is asserted as well as a lower one: the planted hazard sits close
to the base rate for most subscriber-months, so an AUC above 0.9 would be
evidence of leakage rather than skill. Log loss rather than accuracy, since
"nobody churns" is 98.4% accurate here. Calibration is checked separately,
because tripling every prediction leaves AUC identical to 12 decimal places
and makes every number unusable.

Two findings worth keeping from the day:

- **scikit-learn's default tolerance stopped the fit short of the maximum
  likelihood estimate.** `sum(y - p) = 0` is required at the optimum; the
  default left a residual of 0.98 events and `plan_standard` at −0.425 instead
  of −0.452. It looked exactly like the analysis failing to recover a
  coefficient. `tol=1e-8` fixes it and the identity is now a test.
- **The row-split leak is real in principle and small here.** A random row
  split scored 0.711 against 0.680 on the first seed tried — then 0.0025
  averaged over five, against a between-seed sd of 0.014. Four parameters leave
  nothing to memorise. The split stays by subscriber anyway; the claim in the
  test is the measured one.

## Segments and value (day 5)

Two reports, and between them the day's lesson: the same data answers "who
retains" and "who is worth more" differently.

`python -m src.segments` splits survival by observed activity. The interesting
part is the segment *definition*. Grouping subscribers by their average
activity over all observed months is the natural phrasing and it looks into the
future - those months exist only because the subscriber had not churned yet, so
the segment is partly the outcome. Segmenting on **first-month activity**, a
covariate known at signup, is the honest version. Both are printed:

```
segment  at risk  baseline cut  lifetime cut (naive)
     A1     1000        73.5%                68.1%
     A4     1000        88.0%                90.4%
```

The naive cut inflates the top-to-bottom gap by 1.54x on this store - and by
1.13x-1.31x on three other configurations, since a longer window gives the
lifetime average more months in which to encode survival. The spread of that
ratio is in the test, not just the headline number.

`python -m src.ltv` turns the curves into money, with two rules. The horizon is
part of the figure - survival is still 75.9% when the window closes, so a
"lifetime" number would be extrapolation - and a horizon longer than the data
is refused rather than extrapolated silently. Prices are a stated assumption.

```
plan      S(12mo)   value   value @1%/mo   price/churn   error
basic       77.9%      94             89            97  +3.6%
premium     89.3%     439            416           446  +1.5%
```

`price / churn_rate` is where most LTV estimates come from. Truncated to the
same horizon it over-states value by 1.5-3.6%, because the pooled rate averages
a hazard that runs from 4.2% down to 1.3%. Unbounded - the form actually quoted
- it implies expected lifetimes of 50 to 113 months against 24 months of
observation.

The value grid reverses the retention finding, and both are right: activity
moves retention further than plan does (18.7 points against 7.4), while plan
moves value further (4.5x against 1.1x at a 12-month horizon). Value multiplies
by price; retention can only buy months, and the horizon caps how many there
are to buy.

## Dashboard (day 6)

```bash
streamlit run dashboard/app.py
```

Five tabs -- cohort retention, survival, churn drivers, segments, value -- over
the same functions the CLI reports use. The dashboard contains **no SQL of its
own**: a query written twice can disagree with itself, and the copy in a
dashboard is the one nobody tests. It opens the store read-only, reads the
store path from `CHURNLENS_STORE` if set, and never touches the truth tables
(the contract guard discovers `dashboard/` automatically).

The caveats are on the page, not in a footnote: blank retention cells are
explained where they are shown, median survival reads "not reached", the
log-rank note says a non-significant result from a conservative statistic means
*not shown*, and the value tab says its horizon is part of the number.

Two things came out of making it run rather than just writing it:

- **A health probe proves nothing.** `--server.headless true` plus
  `GET /_stcore/health` returns `ok` in three seconds, but Streamlit executes
  the script per session, so with no browser attached the script never runs --
  a page that would crash on line one still answers 200. The real check is
  `streamlit.testing.v1.AppTest`, which runs the script and exposes what it
  raised. Week 01 shipped an Airflow DAG it never executed; that lesson is why
  this exists.
- **It found a crash.** The horizon slider was a hardcoded 3-24 months; on a
  store with a shorter window, every cell of the value grid was skipped and the
  page died with `KeyError: 'plan'` out of a pandas pivot. Fixed at both ends:
  the slider's range now comes from the store's window, and the value grid
  refuses an impossible horizon with a message that says so.

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

Day 6 of 7. The hazard model, the lifecycle generator with correct censoring,
the DuckDB load, the sanity report, cohort retention, Kaplan–Meier survival and
the churn-driver model are in place and tested. Every analysis lands on the
process it was drawn from: a 3.920% predicted first-month hazard against 3.833%
observed, cohort retention and the KM curve both within a few standard errors
of the planted survival function, and the driver model recovering every planted
coefficient within 1.5 se when given the true regressor — plus a measured 58%
attenuation when given only the observable proxy, which is the most useful
thing the week has produced. Segment deep-dives and restricted LTV landed on day 5,
including a measured look-ahead bias and a value grid, and the Streamlit
dashboard on day 6 -- which, once it was actually run headless, turned up a
crash in its own horizon slider. Both day-7 reports are what is left.
