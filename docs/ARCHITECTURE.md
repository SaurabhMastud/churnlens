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
| Sanity report | Python + SQL (`src/report`) | Read-only first look: cumulative churn by plan, monthly hazard by tenure bucket |
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
                     |
                     |  analysis-facing
                     |-- subscribers         (plan, signup month, churn month, censoring)
                     |-- subscription_months (one row per subscriber-month, active or churned)
                     |-- engagement_events   (logins/feature use, the observable signal)
                     |-- run_meta            (window length, count, seed -- provenance)
                     |
                     |  quarantined -- tests only, never read by an analysis
                     |-- hazard_truth        (the planted coefficients)
                     |-- subscriber_truth    (each subscriber's latent engagement level)
                               |
                   src/report, src/analysis (cohorts, survival, drivers)
                   read only the analysis-facing tables
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

**The store is split in two by this contract.** `hazard_truth` and
`subscriber_truth` are quarantined; `subscribers`, `subscription_months` and
`engagement_events` are analysis-facing. Two things enforce the split, in that
order of strength:

1. **Structurally** — the truth is not in the tables an analysis queries, so
   there is nothing to read. This is what day 2 bought by moving the latent
   engagement level off `subscribers`.
2. **By test** — a guard walks each analysis module's AST and fails if it names
   a quarantined table or a latent column. It discovers analysis modules by
   convention rather than by a hand-maintained list, so a module added on day 3
   or 6 is covered without anyone remembering to opt it in. Discovery spans
   `src/` *and* `dashboard/`, and a further test fails if Python appears in a
   directory in neither list — a guard is only as wide as the places it looks,
   and the day-6 Streamlit app would otherwise have sat outside it.

Day 1 shipped only the second of these, and only for one module, which is why
the latent level was readable at all. The lesson generalises: a contract held
up by a test that someone has to remember to extend is one module away from
being held up by nothing.

## Day-by-day plan

| Day | Planned |
|---|---|
| 1 | Scaffolding, architecture doc, lifecycle generator with a configured hazard model, DuckDB load, first tests. *Also shipped:* parameterised CLI and a read-only sanity report (churn by plan, hazard by tenure bucket, churn by observed-activity quartile), plus the first estimate-vs-truth comparison |
| 2 | Move the latent engagement column to a truth-side table (decided — see the decisions log), then the cohort retention matrix + retention curves, validated against the generated hazard |
| 3 | Survival analysis (Kaplan–Meier, hazard by tenure), censoring handled correctly |
| 4 | Churn-driver model; does it recover the planted coefficients and their ordering |
| 5 | Segment deep-dives (plan tier, engagement decile), LTV |
| 6 | Streamlit dashboard over the analyses |
| 7 | Both day-7 reports, tag `week02-complete` |

## Decisions log

- **Synthetic data with a known hazard, not a public churn dataset**: a public dataset has real messiness but no ground truth, so no analysis built on it can be verified — only sanity-checked. Since the project's goal is testable analytics, verifiability wins. The cost is that the data lacks real-world pathologies, which is recorded as a limitation rather than hidden.
- **Coefficients stored with the data but never read by the analysis**: the alternative — passing the truth into the analysis for comparison — would make it trivially possible to fit to the answer. Keeping the channel one-way (tests may read, analyses may not) is what makes the comparison evidence.
- **DuckDB again, deliberately**: the same reasoning as week 01 (no server, real SQL, good pandas interop) and it keeps the week's novelty in the analysis rather than in infrastructure setup. Not every project needs a new database.
- **The day-1 pins described an environment nobody had built**: `requirements.txt` was written on day 1 from memory and never installed from. Checked against the live interpreter, five of nine pins had drifted (duckdb 1.0.0 vs 1.5.5 installed, pandas 2.2.2 vs 2.3.3, numpy 1.26.4 vs 2.4.6, scikit-learn 1.5.1 vs 1.9.0, pytest 8.3.2 vs 9.1.1) and `statsmodels` was pinned but not installed at all. That makes `pip install -r requirements.txt` produce an environment the suite had never run in — a portfolio repo's most embarrassing kind of broken, because it fails for the reader and not for the author. Re-pinned to the versions actually verified (Python 3.11.9), with the not-yet-imported packages kept in a labelled section so the file states plainly what has been exercised and what has not. `statsmodels` was dropped rather than carried at an unverified version; scikit-learn covers the day-4 driver model, and if statsmodels turns out to fit better it gets added at a version that has been run. This also matters more than it did yesterday: `pandas` became a hard runtime import of `src/store` with the bulk-load change, so it is no longer only an analysis dependency.
- **The ground-truth contract is enforced by a test, not by convention**: "analyses must not read `hazard_truth`" is a comment in every file, which is worth nothing once there are six modules and a dashboard. `test_the_report_never_reads_the_ground_truth` walks the report module's AST and asserts no string literal it executes names the table and that it does not import `src.hazard`. Deliberately AST-based rather than a substring scan of the source: the module explains the contract in its own docstring, so a text scan would fail on prose while missing nothing real. Docstrings are excluded, f-string fragments are included (that is how the bucketed query is assembled), and a second test asserts the check can still see the queries that *are* there, so it cannot pass by being broken. Verified by adding a `hazard_truth` read to the report, which fails it.
- **The observation window is recorded, not inferred** — reversing a call made an hour earlier the same day. The cohort code first derived the window as `max(signup_month + tenure_months) + 1`, on the reasoning that a metadata table for one integer was not worth it and that a derived value cannot drift from the rows. That reasoning has a hole: the derivation is exact only if *some* subscriber is observed to the final month. True for any realistic dataset, since most subscribers are censored — and silently wrong otherwise. Searched for a counterexample rather than assuming one existed, and found it immediately: a one-subscriber 24-month store whose only member signed up in month 15 and churned at tenure 4 reported a **20-month window**, shrinking that cohort's exposure by four months with nothing to indicate anything was wrong. Since cohort exposure is the one quantity the whole retention table rests on, an inference that is "usually right" is the wrong trade. Now a `run_meta` row carries the window alongside the subscriber count and seed, which the project wanted anyway as dataset provenance, and a missing row raises instead of defaulting. The original derivation survives as the *test* — exposure must not shrink when subscribers churn early.
- **`hazard_truth` was not the only ground truth, and the contract had a hole**: the contract as written on day 1 quarantined the *coefficients*, but `subscribers.engagement` — the latent engagement level the hazard was actually given — sits in an analysis-facing table. An analysis can therefore cheat completely without ever touching the quarantined table. This was not theoretical: swapping `order by s.engagement` in for `order by a.mean_events` in the activity-quartile cut **passed every other test in the file**, and produced a cleaner gradient, because it is reading the answer. Found by mutation-testing the new cut rather than by reasoning about it. Now guarded by the same AST mechanism, with `\bengagement\b` over the report's SQL — `_` is a word character, so `engagement_events` (the legitimate observable table) does not match while a bare column reference does, and a companion test asserts the regex still matches the real schema so the guard cannot rot silently. **Resolved on day 2: the column moves.** Surveying the actual reads first made the decision easy — the latent engagement is read *from the store* in exactly one place, and it is a test (`planted_probability`'s `select plan, engagement from subscribers`). Every other `.engagement` reference is the generator reading its own in-memory `Subscriber` dataclass, which is not the store and is not a channel an analysis can reach. So the ripple is far smaller than it looked when it was deferred: the schema, the load, and two tests.

The reason to do it rather than keep the AST guard is that the guard does not scale. It is per-module, and days 3–6 add survival analysis, a driver model and a dashboard — three more modules that each have to remember to opt into the check, where forgetting fails silently and in the direction of a better-looking result. Moving the column makes the quarantine structural: an analysis cannot read what is not in the table it queries. The AST guards stay as a second line (they also cover `hazard_truth`, which is not moving anywhere), but they stop being the only thing standing between an analysis and the answer.

Implemented the same day. The load now splits `Subscriber` on a named `LATENT_FIELDS` set rather than hand-listing each table's columns, so a field added to the dataclass lands in one table or the other automatically — and mutation-testing that found the system fails *closed*: adding an unassigned field raises a `BinderException` at load time, because `BY NAME` refuses a column the table does not have. A test also asserts the two tables partition every dataclass field, with `subscriber_id` the only legitimate overlap.

The AST guard was generalised at the same time, into `tests/test_contract.py`: analysis modules are now **discovered** (anything under `src/` not named in `GENERATOR_MODULES`) and each is parametrically checked for truth-table reads, `src.hazard` imports, and latent-column references. Proved by dropping a deliberately cheating `src/survival.py` into the tree — it was caught as `[survival]` with no edit to any test — and by dropping a clean one in, which passed. That exercise also caught a flaw in a test I had just written: pinning the discovered list to `["report"]` would have failed on every legitimate new module, which is friction rather than safety, since the parametrized guards already cover new modules. Narrowed to pinning the *exempt* side instead, where quiet growth is the actual risk. Removing the now-duplicated guards from `tests/test_report.py` deleted 104 lines.

Then a second gap in the same new guard: it scanned only `src/`, while the component table puts the day-6 Streamlit app in `dashboard/` — so the one analysis module most likely to be written quickly and least likely to be reviewed carefully would have sat entirely outside the contract. Discovery now spans both roots and reports modules as dotted paths, verified by planting a cheating `dashboard/app.py` and watching it fail as `[dashboard.app]`. A further test fails if Python appears in a directory in neither list, which generalises the mistake rather than patching this one instance of it. Worth noting that both holes in this guard were found by trying to break it, not by reading it.
- **First estimate-vs-truth result, and the shape every later one will take**: the tenure-0 monthly hazard is the cleanest possible comparison, because `log1p(0) = 0` removes the tenure term and a disagreement cannot hide in that coefficient. Taking each subscriber's own plan and engagement, asking the *stored* coefficients what their first-month churn probability was, and averaging gives 3.920%; the generated data did 3.833%, a gap of 0.34 binomial standard errors (n=6000). Rebuilding the probability from the stored rows rather than importing `src.hazard` also checks that what the store carries is usable as ground truth, not merely present. Tolerance is set at an absolute 0.01 (~4 se) — loose enough not to flake on a reseed, tight enough that a wrong exposure denominator or a dropped plan effect fails it. Checked against three seeds rather than assumed: the gap came out at 0.34, 0.67 and 1.18 se, with the plan ordering and the activity gradient holding in all three, so the result is not an artifact of the one seed the test happens to pin.
- **Exposure denominated in subscriber-months, not subscribers**: the monthly hazard divides churn events by subscriber-months *at risk*. Dividing by subscribers is the usual way this comes out wrong, since a subscriber who stayed twelve months carries twelve chances to churn but would be counted once, which deflates the hazard at exactly the long tenures where the trend is being read. Tenure 0 is the one bucket where both denominators agree (every subscriber has exactly one tenure-0 month), so a test pins it to the subscriber count there and requires the later buckets to exceed it.
- **Tenure bucketed, with the boundaries and labels generated from one tuple**: per-month hazards are too thin at the tail to read a trend off, so tenure is bucketed `0 / 1-2 / 3-5 / 6-11 / 12+`. The SQL `CASE` and the printed labels are both derived from `TENURE_BUCKETS`, because two hand-maintained copies drifting apart would silently misfile months. A test asserts no subscriber-month lands in a null bucket, that the buckets are contiguous, and that the display order follows the tuple — string ordering would print `12+` before `3-5` and show the decline out of sequence.
- **`np.random.poisson` instead of a hand-rolled sampler**: the first engagement sampler drew Poisson counts by summing exponentials until they exceeded the mean, which is correct but O(mean) per row across tens of thousands of rows. numpy was already pinned for the analysis, so the stdlib-only argument bought nothing. Changed before any dataset was committed, so no stored data had to be regenerated.
- **Bulk-load through a DataFrame scan, not `executemany`** — and a correction to the day-1 diagnosis. Day 1 recorded the slow test suite as "mostly the engagement sampler". Profiling with `--durations` said otherwise: after the numpy fix the suite was still 138s, of which 137s was two DuckDB builds (57s fixture setup, 80s rebuild test) and 0.06s the sampler. The real cause was `con.executemany("INSERT ...")`, which autocommits per statement — a 12k-row load paying 12k disk syncs. Measured on this machine, 12k rows: **34s** by `executemany`, **1.9s** wrapped in one explicit transaction, **0.06s** via `INSERT INTO ... BY NAME SELECT * FROM frame` over a pandas DataFrame. Took the DataFrame path because pandas is already a dependency for the analysis; suite went **138s → 1.4s**. `BY NAME` rather than positional `SELECT *` so a reordered dataclass field cannot shift values into the wrong column — without it, a reordered dict raises on type mismatch but a same-type swap (`engagement` vs `signup_month`) would be silent.
- **The churn month gets an engagement row** (a real bug, found while profiling, not by a failing test). `month_rows` spanned `active_months + 1` for a churned subscriber — including the month they left — while `engagement_rows` spanned only `active_months`. At 1200 subscribers that left all 162 churn rows with no matching engagement row and all 10,228 other rows with one: a **perfect separator on the target**. Day 4 would have hit it one of two ways, and the second is the dangerous one — an inner join silently drops every churn event (churn rate 0, nothing to fit), while a left join with `fillna(0)` makes "no activity" predict churn exactly, so the model reports near-perfect accuracy and a badly wrong engagement coefficient. Leakage that looks like success is the failure mode this project exists to catch, so finding it planted in the generator is the point working as intended. Fixed by giving both builders one shared span function (`recorded_months`) instead of two copies of the expression, which is what let them drift apart in the first place.
- **`frozen=True` was not enough to make the coefficients fixed**: `HazardModel` is a frozen dataclass, which blocks attribute reassignment but not mutation of the dict behind `plan_effect` — `DEFAULT_MODEL.plan_effect["premium"] = 99.0` was accepted and took a premium subscriber's monthly churn probability to 1.0. Because `DEFAULT_MODEL` is one module-level instance shared by the generator, the store and the tests, a single stray write would have moved the ground truth for all of them simultaneously, and `hazard_truth` could have been written from different coefficients than the data was drawn from — which would invalidate every estimate-vs-truth comparison while leaving the suite green. Fixed with a `__post_init__` that copies the mapping and wraps it in `MappingProxyType`; the copy closes the second route, where a caller keeps a reference to the dict they passed in. The mutation check was instructive beyond the fix: with the wrapping removed, the immutability test's own write leaked into `DEFAULT_MODEL` and broke an unrelated equality test downstream, which is the cross-contamination in miniature.
- **Distinguishing the planted effect from the artifact**: after the fix, churn-month activity still averages lower than other months — 4.87 vs 6.14 events, a 21% gap. That one is *real signal*, not leakage: low-engagement subscribers churn more, so churn months are drawn disproportionately from low-engagement subscribers. Holding engagement in a 0.4–0.6 band collapses it to 5.44 vs 5.99 (~9%, n=151, within sampling noise plus the residual selection inside the band). The leakage test asserts on the banded comparison for exactly this reason; a test on the raw means would have to either tolerate 21% (and miss a real separator) or fail on the planted effect. Day 4 has to make the same distinction, which is the whole exercise.
- **Guarding the pandas round trip with a full-equality test**: routing a nullable `churn_month` through a DataFrame makes it float64 (`None` → `NaN`), so the risks are a float leaking back out and `NaN` landing where `NULL` was meant — the second would make the censored population invisible to `is null` and quietly break every survival estimate. Rather than eyeballing one row, the test reads every stored subscriber back and compares it to the object it was generated from, types included. Both guards were mutation-checked: switching the column to `DOUBLE` fails the round-trip test, and dropping `BY NAME` with reordered keys raises.
