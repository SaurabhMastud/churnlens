"""Segment checks, with the look-ahead bias as the main event.

The arithmetic here is the same arithmetic day 3 already tests -- these curves
go through the same accumulator -- so these tests are about the *segment
definition* instead: whether it is computable at signup, whether it is
reproducible, and how much the naive version distorts the answer.
"""
from __future__ import annotations

import duckdb
import pytest

from src import segments, store, survival

BUILD = dict(count=6000, window_months=18, seed=67)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("segments") / "churnlens.duckdb"
    store.build_store(store=path, **BUILD)
    return path


@pytest.fixture(scope="module")
def con(built):
    with duckdb.connect(str(built), read_only=True) as connection:
        yield connection


def test_groups_are_equal_sized_and_ordered_by_activity(con):
    levels = segments.activity_levels(con, "baseline", 4)
    assert len(levels) == BUILD["count"]

    sizes: dict[str, int] = {}
    activity: dict[str, list[float]] = {}
    for row in levels:
        sizes[row["segment"]] = sizes.get(row["segment"], 0) + 1
        activity.setdefault(row["segment"], []).append(row["activity"])

    assert sorted(sizes) == ["A1", "A2", "A3", "A4"]
    assert set(sizes.values()) == {BUILD["count"] // 4}
    means = [sum(activity[s]) / len(activity[s]) for s in sorted(activity)]
    assert means == sorted(means), means


def test_the_segment_assignment_is_reproducible(con):
    """Regression test for a real non-determinism bug.

    Baseline activity is an integer event count, so the quartile boundary falls
    inside a large tie, and `ntile` over `order by activity` alone put those
    tied subscribers on either side depending on scan order. The same store
    gave S(11) for basic/A1 as 66.1% on one run and 65.9% on the next -- small,
    but an analysis that will not reproduce cannot be checked by anyone.

    Fixed by breaking the tie on `subscriber_id`. Asserted on repeated queries
    against one connection and on the derived curves, since the visible symptom
    was in the curves rather than in the assignment.
    """
    first = segments.activity_levels(con, "baseline", 4)
    again = segments.activity_levels(con, "baseline", 4)
    assert first == again

    curves = [segments.survival_at(segments.survival_by_activity(con), 11) for _ in range(3)]
    assert curves[0] == curves[1] == curves[2]


def test_the_baseline_cut_uses_only_the_first_observed_month(con):
    """The property that makes the segment legitimate, checked against the rows.

    A segment known at entry cannot have been influenced by whether the
    subscriber later churned. So every subscriber's baseline activity must
    equal their tenure-0 event count exactly -- no averaging, no later month
    leaking in.
    """
    levels = {row["subscriber_id"]: row["activity"] for row in segments.activity_levels(con)}
    first_month = dict(
        con.execute(
            "select subscriber_id, events from engagement_events where tenure_months = 0"
        ).fetchall()
    )

    assert len(levels) == len(first_month)
    for subscriber_id, value in levels.items():
        assert value == first_month[subscriber_id], subscriber_id


def test_the_lifetime_cut_is_the_one_that_sees_the_future(con):
    """And the naive basis has to differ, or the comparison is vacuous.

    A churner's lifetime average is taken over the months they survived, so for
    anyone who left early it is a different number from their first month.
    """
    baseline = {r["subscriber_id"]: r["activity"] for r in segments.activity_levels(con, "baseline")}
    lifetime = {r["subscriber_id"]: r["activity"] for r in segments.activity_levels(con, "lifetime")}

    differing = sum(1 for key in baseline if baseline[key] != lifetime[key])
    # 5178 of 6000 on this store. The remainder are subscribers whose lifetime
    # average happens to equal their first month -- mostly people with a single
    # observed month, for whom the two definitions coincide by construction.
    assert differing > 0.8 * len(baseline), differing


def test_the_look_ahead_cut_overstates_the_retention_gap(con):
    """The day-5 finding, as a number.

    Grouping on lifetime average activity -- the natural phrasing of "segment
    by the engagement signal" -- inflates the top-to-bottom S(11) gap. The
    segment is partly a restatement of the outcome: a subscriber who churned in
    month 2 has two months to be averaged over, and those months are also the
    only evidence about them.

    Measured across five store configurations rather than one, because the
    first number found was the largest and the effect turns out to depend on
    window length:

        6000 x 18, seed 67   1.13x
        6000 x 18, seed 11   1.31x
        6000 x 18, seed 404  1.25x
        4000 x 24, seed 7    1.54x   (the committed store)
        6000 x 36, seed 67   1.26x

    A longer window gives the lifetime average more months in which to encode
    survival, so the committed store's 1.54x is the high end and quoting it as
    "about half again" generally would be overstating. The assertion is
    therefore a direction plus a floor of 1.05, and the shallower tenure 5 is
    deliberately not used -- at 1.02x there the bias has barely had room to
    appear, which is itself worth knowing.
    """
    honest = segments.survival_at(segments.survival_by_activity(con, "baseline"), 11)
    naive = segments.survival_at(segments.survival_by_activity(con, "lifetime"), 11)

    honest_gap = max(honest.values()) - min(honest.values())
    naive_gap = max(naive.values()) - min(naive.values())

    assert honest_gap > 0.05, honest_gap
    assert naive_gap > honest_gap * 1.05, (naive_gap, honest_gap)


def test_the_activity_segments_recover_the_planted_engagement_ordering(con):
    """Observable activity has to sort subscribers by the latent level.

    A test may read `subscriber_truth`; the module may not. Mean latent
    engagement rises monotonically across the four baseline segments (0.288,
    0.465, 0.586, 0.712 on this store, correlation 0.72 between the single
    observed month and the latent level), which is what makes the segment a
    proxy for engagement rather than for noise.
    """
    rows = con.execute(
        f"""
        with levels as ({segments._activity_sql("baseline", 4)})
        select levels.segment, avg(t.engagement), count(*)
        from levels join subscriber_truth t using (subscriber_id)
        group by 1 order by 1
        """
    ).fetchall()

    latent = [mean for _, mean, _ in rows]
    assert latent == sorted(latent), latent
    assert latent[-1] - latent[0] > 0.3, latent


def test_segment_curves_reconcile_with_the_pooled_survival_curve(con):
    """Segments partition the population, so their risk sets have to add up.

    Checked against `survival.risk_table` rather than against each other: this
    module builds its own risk-table query (it needs a subscriber-level join
    the survival module's query does not do), and two queries that are supposed
    to count the same risk set are exactly the place a quiet disagreement
    appears.
    """
    pooled = {row["tenure"]: row for row in survival.risk_table(con)}
    totals: dict[int, dict[str, int]] = {}
    for row in segments.risk_table_by_activity(con):
        bucket = totals.setdefault(row["tenure"], {"at_risk": 0, "events": 0, "censored": 0})
        for key in bucket:
            bucket[key] += row[key]

    assert set(totals) == set(pooled)
    for tenure, sums in totals.items():
        for key, value in sums.items():
            assert value == pooled[tenure][key], (tenure, key)


def test_filtering_to_one_plan_keeps_the_population_whole(con):
    """The per-plan grid must partition the same risk set once more."""
    plans = [row[0] for row in con.execute("select distinct plan from subscribers").fetchall()]
    pooled = {row["tenure"]: row["at_risk"] for row in survival.risk_table(con)}

    totals: dict[int, int] = {}
    for plan in plans:
        for row in segments.risk_table_by_activity(con, plan=plan):
            totals[row["tenure"]] = totals.get(row["tenure"], 0) + row["at_risk"]

    assert totals == pooled


def test_an_unknown_basis_or_a_degenerate_group_count_is_refused(con):
    with pytest.raises(ValueError, match="unknown basis"):
        segments.activity_levels(con, "whatever")
    with pytest.raises(ValueError, match="at least 2"):
        segments.activity_levels(con, "baseline", 1)


def test_render_labels_the_naive_column_as_naive(built):
    text = segments.render(built, tenure=11, groups=4)

    assert "lifetime cut (naive)" in text
    assert "overstates" in text
    assert "Worst cell:" in text
    # The thin-cell caveat is part of the output, not an optional extra.
    assert "smallest cell holds" in text


def test_segments_cli_prints_the_table(built, capsys):
    segments.main(["--store", str(built), "--tenure", "6", "--groups", "3"])
    out = capsys.readouterr().out
    assert "Survival by activity segment" in out
    assert "A3" in out and "A4" not in out
