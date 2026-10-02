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
