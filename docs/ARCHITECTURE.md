# Architecture — ChurnLens

## Why this project

Retention is the metric subscription businesses are judged on, and the analysis
around it is unusually easy to get quietly wrong. Cohort retention tables,
survival curves and "churn driver" models are standard output in analytics
roles, yet most published examples are unverifiable: they compute a number from
a dataset whose true churn process is unknown, so there is no way to tell a
correct analysis from a plausible-looking one.

ChurnLens inverts that. Subscriber lifecycles are **generated from a known
hazard model** — churn probability is an explicit function of plan tier,
tenure and engagement, with coefficients fixed in configuration. The ground
truth is therefore available, and every analysis in the project can be checked
against it: a Kaplan–Meier curve can be compared to the hazard it was drawn
from, and a churn-driver model can be asked whether it recovers the planted
coefficients and their ordering.

That is the point of the project. Week 01 (RetailPulse) was pipeline
engineering, where correctness means data arrives intact. Here correctness
means an *inference* is right, which needs a different kind of test — and
having ground truth is what makes those tests possible rather than decorative.

## Components

| Component | Tool | Role |
|---|---|---|
| Lifecycle generator | Python (`src/generate`) | Synthetic subscribers with churn drawn from an explicit, configured hazard model — the ground truth |
| Store | DuckDB (`data/`) | Subscribers, subscription months, engagement events |
| Cohort analysis | Python + SQL (`src/analysis`) | Signup-cohort retention matrix, retention curves |
| Survival analysis | Python (`src/analysis`) | Kaplan–Meier estimates, hazard by tenure bucket |
| Driver model | scikit-learn / statsmodels | Churn driver estimation, scored against the planted coefficients |
| Dashboard | Streamlit (`dashboard/`) | Cohort heatmap, retention curves, driver effects |
| Reports | `docs/` | Day-7 architecture report and IEEE paper |

## Data flow

```
hazard config (known coefficients)
        |
        v
src/generate  -->  data/churnlens.duckdb
                     |-- subscribers        (one row per subscriber, incl. true churn month)
                     |-- subscription_months (one row per subscriber-month, active or churned)
                     |-- engagement_events   (logins/feature use, the engagement signal)
                               |
                   src/analysis (cohorts, survival, drivers)
                               |
                   Streamlit dashboard + day-7 reports
```

## The ground-truth contract

The generator writes the coefficients it used into the store alongside the
data. Analyses never read them; tests do. This keeps the analysis honest — it
cannot accidentally be tuned against the answer — while still allowing a test
to assert that the estimate lands within tolerance of the truth.

This is the project's central design decision and the reason the test suite can
say anything meaningful about analytical correctness.

## Day-by-day plan

| Day | Planned |
|---|---|
| 1 | Scaffolding, architecture doc, lifecycle generator with a configured hazard model, DuckDB load, first tests |
| 2 | Cohort retention matrix + retention curves, validated against the generated hazard |
| 3 | Survival analysis (Kaplan–Meier, hazard by tenure), censoring handled correctly |
| 4 | Churn-driver model; does it recover the planted coefficients and their ordering |
| 5 | Segment deep-dives (plan tier, engagement decile), LTV |
| 6 | Streamlit dashboard over the analyses |
| 7 | Both day-7 reports, tag `week02-complete` |

## Decisions log

- **Synthetic data with a known hazard, not a public churn dataset**: a public dataset has real messiness but no ground truth, so no analysis built on it can be verified — only sanity-checked. Since the project's goal is testable analytics, verifiability wins. The cost is that the data lacks real-world pathologies, which is recorded as a limitation rather than hidden.
- **Coefficients stored with the data but never read by the analysis**: the alternative — passing the truth into the analysis for comparison — would make it trivially possible to fit to the answer. Keeping the channel one-way (tests may read, analyses may not) is what makes the comparison evidence.
- **DuckDB again, deliberately**: the same reasoning as week 01 (no server, real SQL, good pandas interop) and it keeps the week's novelty in the analysis rather than in infrastructure setup. Not every project needs a new database.
- **`np.random.poisson` instead of a hand-rolled sampler**: the first engagement sampler drew Poisson counts by summing exponentials until they exceeded the mean, which is correct but O(mean) per row across tens of thousands of rows. numpy was already pinned for the analysis, so the stdlib-only argument bought nothing. Changed before any dataset was committed, so no stored data had to be regenerated.
- **Bulk-load through a DataFrame scan, not `executemany`** — and a correction to the day-1 diagnosis. Day 1 recorded the slow test suite as "mostly the engagement sampler". Profiling with `--durations` said otherwise: after the numpy fix the suite was still 138s, of which 137s was two DuckDB builds (57s fixture setup, 80s rebuild test) and 0.06s the sampler. The real cause was `con.executemany("INSERT ...")`, which autocommits per statement — a 12k-row load paying 12k disk syncs. Measured on this machine, 12k rows: **34s** by `executemany`, **1.9s** wrapped in one explicit transaction, **0.06s** via `INSERT INTO ... BY NAME SELECT * FROM frame` over a pandas DataFrame. Took the DataFrame path because pandas is already a dependency for the analysis; suite went **138s → 1.4s**. `BY NAME` rather than positional `SELECT *` so a reordered dataclass field cannot shift values into the wrong column — without it, a reordered dict raises on type mismatch but a same-type swap (`engagement` vs `signup_month`) would be silent.
- **The churn month gets an engagement row** (a real bug, found while profiling, not by a failing test). `month_rows` spanned `active_months + 1` for a churned subscriber — including the month they left — while `engagement_rows` spanned only `active_months`. At 1200 subscribers that left all 162 churn rows with no matching engagement row and all 10,228 other rows with one: a **perfect separator on the target**. Day 4 would have hit it one of two ways, and the second is the dangerous one — an inner join silently drops every churn event (churn rate 0, nothing to fit), while a left join with `fillna(0)` makes "no activity" predict churn exactly, so the model reports near-perfect accuracy and a badly wrong engagement coefficient. Leakage that looks like success is the failure mode this project exists to catch, so finding it planted in the generator is the point working as intended. Fixed by giving both builders one shared span function (`recorded_months`) instead of two copies of the expression, which is what let them drift apart in the first place.
- **`frozen=True` was not enough to make the coefficients fixed**: `HazardModel` is a frozen dataclass, which blocks attribute reassignment but not mutation of the dict behind `plan_effect` — `DEFAULT_MODEL.plan_effect["premium"] = 99.0` was accepted and took a premium subscriber's monthly churn probability to 1.0. Because `DEFAULT_MODEL` is one module-level instance shared by the generator, the store and the tests, a single stray write would have moved the ground truth for all of them simultaneously, and `hazard_truth` could have been written from different coefficients than the data was drawn from — which would invalidate every estimate-vs-truth comparison while leaving the suite green. Fixed with a `__post_init__` that copies the mapping and wraps it in `MappingProxyType`; the copy closes the second route, where a caller keeps a reference to the dict they passed in. The mutation check was instructive beyond the fix: with the wrapping removed, the immutability test's own write leaked into `DEFAULT_MODEL` and broke an unrelated equality test downstream, which is the cross-contamination in miniature.
- **Distinguishing the planted effect from the artifact**: after the fix, churn-month activity still averages lower than other months — 4.87 vs 6.14 events, a 21% gap. That one is *real signal*, not leakage: low-engagement subscribers churn more, so churn months are drawn disproportionately from low-engagement subscribers. Holding engagement in a 0.4–0.6 band collapses it to 5.44 vs 5.99 (~9%, n=151, within sampling noise plus the residual selection inside the band). The leakage test asserts on the banded comparison for exactly this reason; a test on the raw means would have to either tolerate 21% (and miss a real separator) or fail on the planted effect. Day 4 has to make the same distinction, which is the whole exercise.
- **Guarding the pandas round trip with a full-equality test**: routing a nullable `churn_month` through a DataFrame makes it float64 (`None` → `NaN`), so the risks are a float leaking back out and `NaN` landing where `NULL` was meant — the second would make the censored population invisible to `is null` and quietly break every survival estimate. Rather than eyeballing one row, the test reads every stored subscriber back and compares it to the object it was generated from, types included. Both guards were mutation-checked: switching the column to `DOUBLE` fails the round-trip test, and dropping `BY NAME` with reordered keys raises.
