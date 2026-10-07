"""Load checks, including the one that matters most: the ground truth has to
arrive in the store intact, because every later analytical test reads it from
there rather than from src.hazard.
"""
import duckdb
import pytest

from src import generate, store
from src.hazard import PLANS, HazardModel

BUILD = dict(count=1200, window_months=18, seed=23)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("store") / "churnlens.duckdb"
    stats = store.build_store(store=path, **BUILD)
    return path, stats


def test_build_reports_counts_that_match_the_tables(built):
    path, stats = built

    with duckdb.connect(str(path)) as con:
        assert con.execute("select count(*) from subscribers").fetchone()[0] == stats[
            "subscribers"
        ]
        assert con.execute("select count(*) from subscription_months").fetchone()[
            0
        ] == stats["subscriber_months"]
        assert con.execute("select count(*) from engagement_events").fetchone()[
            0
        ] == stats["engagement_rows"]


def test_churned_and_censored_partition_the_subscribers(built):
    _, stats = built

    assert stats["churned"] + stats["censored"] == stats["subscribers"]
    assert stats["churned"] > 0 and stats["censored"] > 0


def test_hazard_truth_lands_in_the_store_with_the_right_values(built):
    """If the stored coefficients drifted from the model the data was drawn
    from, every estimate-vs-truth test downstream would be meaningless."""
    path, _ = built
    expected = {term: value for term, _, value in HazardModel().as_rows()}

    with duckdb.connect(str(path)) as con:
        stored = dict(con.execute("select term, value from hazard_truth").fetchall())

    assert stored == pytest.approx(expected)
    assert len(stored) == 3 + len(PLANS)


def test_every_month_row_belongs_to_a_known_subscriber(built):
    path, _ = built

    with duckdb.connect(str(path)) as con:
        orphans = con.execute(
            """
            select count(*) from subscription_months m
            left join subscribers s using (subscriber_id)
            where s.subscriber_id is null
            """
        ).fetchone()[0]

    assert orphans == 0


def test_exactly_one_churn_row_per_churned_subscriber(built):
    path, _ = built

    with duckdb.connect(str(path)) as con:
        bad = con.execute(
            """
            select count(*) from (
                select s.subscriber_id, s.censored,
                       sum(case when m.churned then 1 else 0 end) as churn_rows
                from subscribers s
                join subscription_months m using (subscriber_id)
                group by s.subscriber_id, s.censored
            )
            where (censored and churn_rows <> 0)
               or (not censored and churn_rows <> 1)
            """
        ).fetchone()[0]

    assert bad == 0


def test_month_and_engagement_keys_match_in_sql(built):
    """The generator-level version of this invariant is in test_generate; this
    is the same check where day 4 will actually hit it -- as a join in the
    store. A full outer join reports both directions of mismatch at once.
    """
    path, _ = built

    with duckdb.connect(str(path)) as con:
        unmatched = con.execute(
            """
            select
                count(*) filter (where e.subscriber_id is null) as months_without_events,
                count(*) filter (where m.subscriber_id is null) as events_without_months
            from subscription_months m
            full outer join engagement_events e
              on m.subscriber_id = e.subscriber_id
             and m.tenure_months = e.tenure_months
            """
        ).fetchone()

    assert unmatched == (0, 0)


def test_subscriber_month_keys_are_unique(built):
    """Nothing in the schema enforces this, and a duplicated subscriber-month
    would inflate both the retention denominator and the activity total.
    """
    path, _ = built

    with duckdb.connect(str(path)) as con:
        for table in ("subscription_months", "engagement_events"):
            dupes = con.execute(
                f"""
                select count(*) from (
                    select subscriber_id, tenure_months
                    from {table}
                    group by subscriber_id, tenure_months
                    having count(*) > 1
                )
                """
            ).fetchone()[0]
            assert dupes == 0, table


def test_subscribers_round_trip_exactly(built):
    """The load goes through a pandas frame, which is where a column could land
    under the wrong name or an int could come back as a float. Comparing every
    stored row against the object it came from covers both at once -- generation
    is deterministic, so the same seed rebuilds the same subscribers.
    """
    path, _ = built
    expected = {
        s.subscriber_id: s
        for s in generate.generate_subscribers(
            count=BUILD["count"],
            window_months=BUILD["window_months"],
            seed=BUILD["seed"],
        )
    }

    with duckdb.connect(str(path)) as con:
        stored = con.execute(
            """
            select subscriber_id, plan, engagement, signup_month, churn_month,
                   censored
            from subscribers
            """
        ).fetchall()

    assert len(stored) == len(expected)
    for sid, plan, engagement, signup, churn, censored in stored:
        want = expected[sid]
        assert (plan, signup, censored) == (want.plan, want.signup_month, want.censored)
        assert engagement == pytest.approx(want.engagement)
        # Not just equal but the same type: pandas routes a nullable int column
        # through float64, so a churn month that came back as 3.0 would quietly
        # break any downstream integer bucketing.
        assert churn == want.churn_month and isinstance(churn, type(want.churn_month))


def test_censored_subscribers_store_a_real_null_churn_month(built):
    """NaN is not NULL. If the float64 round trip left NaN in the column, the
    censored population would stop being findable with `is null`.
    """
    path, stats = built

    with duckdb.connect(str(path)) as con:
        nulls = con.execute(
            "select count(*) from subscribers where churn_month is null"
        ).fetchone()[0]
        not_nan = con.execute(
            "select count(*) from subscribers where isnan(churn_month)"
        ).fetchone()[0]

    assert nulls == stats["censored"]
    assert not_nan == 0


def test_bulk_insert_tolerates_no_rows(built):
    """A zero-row table is a legitimate outcome of a narrow window, and the
    DataFrame path has no columns to bind in that case.
    """
    path, _ = built

    with duckdb.connect(str(path)) as con:
        before = con.execute("select count(*) from engagement_events").fetchone()[0]
        store._bulk_insert(con, "engagement_events", [])
        assert (
            con.execute("select count(*) from engagement_events").fetchone()[0]
            == before
        )


def test_rebuilding_replaces_rather_than_appends(built):
    """CREATE OR REPLACE, not CREATE IF NOT EXISTS -- a second build must not
    double the data, which is the failure mode that makes every rate wrong."""
    path, stats = built
    again = store.build_store(store=path, count=1200, window_months=18, seed=23)

    assert again["subscribers"] == stats["subscribers"]
    with duckdb.connect(str(path)) as con:
        assert (
            con.execute("select count(*) from subscribers").fetchone()[0]
            == stats["subscribers"]
        )
