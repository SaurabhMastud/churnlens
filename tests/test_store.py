"""Load checks, including the one that matters most: the ground truth has to
arrive in the store intact, because every later analytical test reads it from
there rather than from src.hazard.
"""
import duckdb
import pytest

from src import store
from src.hazard import PLANS, HazardModel


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("store") / "churnlens.duckdb"
    stats = store.build_store(store=path, count=1200, window_months=18, seed=23)
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
