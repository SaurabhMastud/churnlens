"""Load checks, including the one that matters most: the ground truth has to
arrive in the store intact, because every later analytical test reads it from
there rather than from src.hazard.
"""
import dataclasses

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


def test_two_builds_with_the_same_parameters_are_identical(tmp_path):
    """Reproducibility end to end, not just in the generator.

    `test_generation_is_deterministic_for_a_seed` covers the Python objects.
    This covers everything after them -- the DataFrame round trip, the column
    binding, the truth split -- by building two independent stores and
    comparing every table's full contents. A seeded dataset nobody else can
    reproduce would undercut the entire premise of the project.
    """
    params = dict(count=400, window_months=12, seed=91)
    first, second = tmp_path / "a.duckdb", tmp_path / "b.duckdb"
    store.build_store(store=first, **params)
    store.build_store(store=second, **params)

    tables = (
        "subscribers",
        "subscriber_truth",
        "subscription_months",
        "engagement_events",
        "hazard_truth",
    )
    with duckdb.connect(str(first)) as a, duckdb.connect(str(second)) as b:
        for table in tables:
            order = "all"
            rows_a = a.execute(f"select * from {table} order by {order}").fetchall()
            rows_b = b.execute(f"select * from {table} order by {order}").fetchall()
            assert rows_a == rows_b, table
            assert rows_a, f"{table} is empty, so comparing it proves nothing"

        # run_meta excludes nothing but is compared separately: it records the
        # parameters, so identical parameters must give an identical row.
        assert (
            a.execute("select * from run_meta").fetchall()
            == b.execute("select * from run_meta").fetchall()
        )


def test_different_seeds_produce_different_data(tmp_path):
    """Guards the test above from passing because the seed is ignored."""
    first, second = tmp_path / "a.duckdb", tmp_path / "b.duckdb"
    store.build_store(store=first, count=400, window_months=12, seed=91)
    store.build_store(store=second, count=400, window_months=12, seed=92)

    with duckdb.connect(str(first)) as a, duckdb.connect(str(second)) as b:
        assert (
            a.execute("select * from subscribers order by all").fetchall()
            != b.execute("select * from subscribers order by all").fetchall()
        )


def test_run_meta_records_the_build_parameters(built):
    """Provenance for the dataset, and the observation window in particular.

    The window has to be recorded rather than inferred from the rows: deriving
    it as max(signup_month + tenure_months) + 1 is exact only when some
    subscriber reaches the final month, which small stores can violate.
    """
    path, stats = built

    with duckdb.connect(str(path)) as con:
        rows = con.execute(
            "select window_months, subscriber_count, seed from run_meta"
        ).fetchall()

    assert len(rows) == 1, "run_meta is one row per build, and a rebuild replaces it"
    window, count, seed = rows[0]
    assert window == BUILD["window_months"]
    assert count == BUILD["count"] == stats["subscribers"]
    assert seed == BUILD["seed"]


def test_rebuilding_does_not_accumulate_run_meta_rows(built):
    """CREATE OR REPLACE covers this, but run_meta is the one table where a
    second row would make every reader's `fetchone()` silently arbitrary."""
    path, _ = built
    store.build_store(store=path, **BUILD)

    with duckdb.connect(str(path)) as con:
        assert con.execute("select count(*) from run_meta").fetchone()[0] == 1


def test_the_latent_level_is_not_on_the_analysis_facing_table(built):
    """The structural half of the ground-truth contract. Until day 2 the latent
    engagement sat on `subscribers`, where any analysis could read it; a test
    was the only thing stopping that, and tests are per-module. Now the column
    simply is not there to read.
    """
    path, _ = built

    with duckdb.connect(str(path)) as con:
        analysis_facing = {
            row[0]
            for row in con.execute("describe subscribers").fetchall()
        }
        quarantined = {
            row[0]
            for row in con.execute("describe subscriber_truth").fetchall()
        }

    assert "engagement" not in analysis_facing
    assert "engagement" in quarantined
    assert store.LATENT_FIELDS <= quarantined


def test_the_two_subscriber_tables_partition_every_dataclass_field(built):
    """A field added to `Subscriber` has to land in exactly one of the two
    tables. Without this, adding one would silently drop it from the store --
    the load splits on LATENT_FIELDS rather than listing columns, so there is
    nothing else to notice the omission.
    """
    path, _ = built
    fields = {f.name for f in dataclasses.fields(generate.Subscriber)}

    with duckdb.connect(str(path)) as con:
        plain = {row[0] for row in con.execute("describe subscribers").fetchall()}
        truth = {row[0] for row in con.execute("describe subscriber_truth").fetchall()}

    # subscriber_id is the join key, so it is legitimately in both.
    assert plain | truth == fields
    assert plain & truth == {"subscriber_id"}


def test_every_subscriber_has_exactly_one_truth_row(built):
    """The join in the estimate-vs-truth tests is only sound if it neither
    drops nor duplicates subscribers."""
    path, stats = built

    with duckdb.connect(str(path)) as con:
        assert (
            con.execute("select count(*) from subscriber_truth").fetchone()[0]
            == stats["subscribers"]
        )
        assert (
            con.execute(
                """
                select count(*) from subscribers s
                full outer join subscriber_truth t using (subscriber_id)
                where s.subscriber_id is null or t.subscriber_id is null
                """
            ).fetchone()[0]
            == 0
        )


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
        # engagement comes from subscriber_truth now, so the round trip has to
        # be checked across the join that splits the dataclass in two.
        stored = con.execute(
            """
            select s.subscriber_id, s.plan, t.engagement, s.signup_month,
                   s.churn_month, s.censored
            from subscribers s
            join subscriber_truth t using (subscriber_id)
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
