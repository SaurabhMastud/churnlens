"""Sanity-report checks -- the project's first estimate-vs-truth tests.

These read `hazard_truth`, which the report itself must not. The coefficients
are rebuilt into probabilities here from the *stored* rows rather than imported
from src.hazard, which also checks that what the store carries is actually
usable as ground truth rather than just present.
"""
import ast
import inspect
import math
import re

import duckdb
import pytest

from src import report, store

BUILD = dict(count=6000, window_months=18, seed=51)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("report") / "churnlens.duckdb"
    store.build_store(store=path, **BUILD)
    return path


@pytest.fixture(scope="module")
def con(built):
    with duckdb.connect(str(built), read_only=True) as connection:
        yield connection


@pytest.fixture(scope="module")
def truth(built):
    """The planted coefficients, read back out of the store."""
    with duckdb.connect(str(built), read_only=True) as connection:
        return dict(connection.execute("select term, value from hazard_truth").fetchall())


def planted_probability(truth: dict, plan: str, tenure: int, engagement: float) -> float:
    logit = (
        truth["intercept"]
        + truth[f"plan_{plan}"]
        + truth["tenure_log1p"] * math.log1p(tenure)
        + truth["engagement"] * engagement
    )
    return 1.0 / (1.0 + math.exp(-logit))


def _code_strings(module) -> list[str]:
    """Every string literal the module's *code* uses, docstrings excluded.

    A plain substring scan over the source cannot do this job: the module
    explains the contract in its own docstring, so naming the table in prose
    would fail the check while an actual query is what matters. SQL lives in
    string literals, so those are what gets inspected -- including the pieces
    of f-strings, which is how the bucketed query is built.
    """
    tree = ast.parse(inspect.getsource(module))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _imported_modules(module) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_the_report_never_reads_the_ground_truth():
    """The contract, enforced mechanically rather than by convention. An
    analysis that reads hazard_truth could be tuned against the answer, and
    every comparison in this file would stop being evidence.
    """
    assert not [s for s in _code_strings(report) if "hazard_truth" in s]
    assert not [m for m in _imported_modules(report) if m.startswith("src.hazard")]


def test_the_report_never_reads_the_latent_engagement_level():
    """`hazard_truth` is not the only ground truth in the store.

    `subscribers.engagement` is the latent level the hazard was actually given,
    and it sits in an analysis-facing table, so an analysis can reach it
    without touching the quarantined coefficients at all. Substituting it for
    observed activity in the quartile cut passed every other test in this file
    -- the gradient is cleaner with the latent value, which is precisely why
    using it would be cheating. Recovering the effect from `engagement_events`
    is the day-4 problem.

    `\\bengagement\\b` does not match inside `engagement_events`, since `_` is a
    word character -- so the legitimate table reference is allowed and a bare
    column reference is not.
    """
    offenders = [
        s for s in _code_strings(report) if re.search(r"\bengagement\b", s)
    ]

    assert not offenders, offenders


def test_the_latent_engagement_guard_matches_the_real_column_name():
    """The guard above is a regex over SQL, so it is only worth anything if the
    column is still called what it thinks. Asserted against the schema rather
    than trusted."""
    assert re.search(r"\bengagement\b", store.SCHEMA)
    assert "engagement_events" in store.SCHEMA


def test_the_contract_check_would_catch_a_violation():
    """The guard above passes trivially if _code_strings is broken, so prove it
    sees a query that is really there -- the report does read subscribers."""
    assert [s for s in _code_strings(report) if "from subscribers" in s]
    assert [s for s in _code_strings(report) if "subscription_months" in s]
    assert _imported_modules(report) >= {"src.store", "src.store.DEFAULT_STORE"}


def test_churn_by_plan_recovers_the_planted_plan_ordering(con, truth):
    rows = report.churn_by_plan(con)
    observed = [r["plan"] for r in rows]  # the query orders by rate, descending

    expected = sorted(
        (p for p in truth if p.startswith("plan_")),
        key=lambda term: truth[term],
        reverse=True,
    )
    assert observed == [term.removeprefix("plan_") for term in expected]
    assert all(0.0 < r["churn_rate"] < 1.0 for r in rows)
    assert sum(r["subscribers"] for r in rows) == BUILD["count"]


def test_hazard_falls_with_tenure_as_planted(con, truth):
    assert truth["tenure_log1p"] < 0, "the planted tenure effect must be protective"

    rows = report.hazard_by_tenure(con)
    hazards = [r["monthly_hazard"] for r in rows]

    assert len(rows) == len(report.bucket_labels())
    assert hazards == sorted(hazards, reverse=True), rows
    assert hazards[0] > hazards[-1] * 2, "the planted decline should be unmistakable"


def test_churn_falls_across_observed_activity_quartiles(con, truth):
    """The strongest sanity check available: engagement is the model's dominant
    lever, and this recovers the gradient from the *observable* event counts
    rather than the latent level the hazard was given.
    """
    assert truth["engagement"] < 0, "the planted engagement effect must be protective"

    rows = report.churn_by_activity_quartile(con)
    rates = [r["churn_rate"] for r in rows]

    assert [r["quartile"] for r in rows] == [1, 2, 3, 4]
    assert rates == sorted(rates, reverse=True), rows

    # Engagement dominating plan is asserted in test_hazard; it should show up
    # here as a wider spread than the plan cut produces on the same data.
    plan_rates = [r["churn_rate"] for r in report.churn_by_plan(con)]
    plan_spread = max(plan_rates) - min(plan_rates)
    assert rates[0] - rates[-1] > plan_spread, (rates, plan_spread)


def test_activity_quartiles_cover_every_subscriber(con):
    """The payoff of giving the churn month an engagement row. Before that fix a
    subscriber who churned in their first month had no engagement rows at all,
    so this join would have dropped them -- and they are the highest-churn
    subscribers in the data, which would bias the gradient toward looking
    weaker than it is.
    """
    rows = report.churn_by_activity_quartile(con)

    assert sum(r["subscribers"] for r in rows) == BUILD["count"]
    # ntile(4) splits as evenly as the row count allows.
    sizes = [r["subscribers"] for r in rows]
    assert max(sizes) - min(sizes) <= 1, sizes


def test_first_month_churners_are_present_in_the_activity_cut(con):
    """Stated directly rather than only implied by the totals: the subscribers
    the old generator would have dropped exist, and they have activity rows."""
    missing = con.execute(
        """
        select count(*) from subscribers s
        where not exists (
            select 1 from engagement_events e
            where e.subscriber_id = s.subscriber_id
        )
        """
    ).fetchone()[0]
    first_month_churners = con.execute(
        "select count(*) from subscribers where churn_month = 0"
    ).fetchone()[0]

    assert missing == 0
    assert first_month_churners > 0, "no first-month churners to be dropped"


def test_exposure_denominator_counts_subscriber_months_not_subscribers(con):
    """The bug this guards is using subscribers as the denominator. Tenure 0 is
    the one bucket where both give the same answer -- every subscriber has
    exactly one tenure-0 month -- so it pins the denominator exactly, while the
    later buckets must exceed the subscriber count to be months at all.
    """
    rows = {r["bucket"]: r for r in report.hazard_by_tenure(con)}

    assert rows["0"]["exposed_months"] == BUILD["count"]
    assert sum(r["exposed_months"] for r in rows.values()) > BUILD["count"]


def test_observed_tenure_zero_hazard_matches_the_planted_model(con, truth):
    """The real check: take every subscriber's own plan and engagement, ask the
    planted model what their first-month churn probability was, average it, and
    compare to what the data actually did. Tenure 0 is used because the tenure
    term is zero there, so a disagreement cannot hide in the tenure coefficient.
    """
    subscribers = con.execute("select plan, engagement from subscribers").fetchall()
    expected = sum(
        planted_probability(truth, plan, 0, engagement)
        for plan, engagement in subscribers
    ) / len(subscribers)

    observed = next(r for r in report.hazard_by_tenure(con) if r["bucket"] == "0")[
        "monthly_hazard"
    ]

    # n = 6000 at p ~ 0.04 puts the binomial standard error near 0.0026, so a
    # whole percentage point is roughly 4 se -- loose enough not to flake,
    # tight enough that a wrong denominator or a dropped plan effect fails.
    assert observed == pytest.approx(expected, abs=0.01), (observed, expected)


def test_tenure_buckets_cover_every_tenure_without_overlap(con):
    """The bucket CASE is generated from TENURE_BUCKETS, so a gap would put
    real subscriber-months in a null bucket and quietly drop them."""
    unbucketed = con.execute(
        f"""
        select count(*) from subscription_months
        where ({report._bucket_case()}) is null
        """
    ).fetchone()[0]

    assert unbucketed == 0

    bounds = report.TENURE_BUCKETS
    assert bounds[0][0] == 0, "tenure 0 must be covered"
    assert bounds[-1][1] is None, "the top bucket must be open-ended"
    for (_, prev_high), (low, _) in zip(bounds, bounds[1:]):
        assert low == prev_high + 1, (prev_high, low)


def test_sql_emits_exactly_the_labels_python_expects(con):
    """`hazard_by_tenure` looks its rows up by label, so a label SQL emits but
    Python does not expect silently drops a whole bucket from the report.
    Both now come from `_label`; this asserts the two sides agree in practice.
    """
    emitted = {
        row[0]
        for row in con.execute(
            f"select distinct ({report._bucket_case('tenure_months')}) "
            "from subscription_months"
        ).fetchall()
    }

    assert emitted == set(report.bucket_labels())


def test_buckets_print_in_tenure_order_not_string_order(con):
    """String ordering puts '12+' second and would show the hazard decline out
    of sequence, which is the one thing this table exists to communicate."""
    buckets = [r["bucket"] for r in report.hazard_by_tenure(con)]

    assert buckets == report.bucket_labels()
    assert buckets != sorted(buckets), "string order would be a different sequence"


def test_render_produces_every_table(built):
    text = report.render(built)

    assert "Cumulative churn by plan" in text
    assert "Monthly churn hazard by tenure bucket" in text
    assert "quartile of observed mean monthly activity" in text
    for label in report.bucket_labels():
        assert label in text
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        assert quartile in text
    assert "%" in text
    # The tie caveat is part of the output, not a comment: adjacent activity
    # ranges really can share an endpoint, and without the note that reads as
    # a bug in the report.
    assert "share an" in text


def test_render_explains_itself_when_there_is_no_store(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.store"):
        report.render(tmp_path / "absent.duckdb")


def test_report_cli_prints_the_report(built, capsys):
    report.main(["--store", str(built)])

    assert "Cumulative churn by plan" in capsys.readouterr().out


def test_store_cli_honours_its_parameters(tmp_path, capsys):
    path = tmp_path / "cli.duckdb"
    store.main(
        ["--count", "150", "--window-months", "6", "--seed", "3", "--store", str(path)]
    )

    assert "150 subscribers" in capsys.readouterr().out
    with duckdb.connect(str(path), read_only=True) as con:
        assert con.execute("select count(*) from subscribers").fetchone()[0] == 150
        assert (
            con.execute("select max(tenure_months) from subscription_months").fetchone()[
                0
            ]
            < 6
        )


def test_store_cli_rejects_impossible_sizes():
    for bad in (["--count", "0"], ["--window-months", "0"]):
        with pytest.raises(SystemExit):
            store.main(bad)
