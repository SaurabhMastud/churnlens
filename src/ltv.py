"""Expected subscriber value over a **stated horizon**, built on the KM curve.

Lifetime value is where retention analysis usually stops being analysis. Two
things go wrong, and they go wrong in opposite directions:

1. **The horizon is left out.** "Lifetime" value needs the survival curve all
   the way to zero, and this one does not get there -- 76% of subscribers are
   still subscribed when the window closes. Anything claiming a lifetime number
   has extrapolated the tail, which is an assumption, not a measurement. So
   everything here is a *restricted* expected value over an explicit number of
   months, and the horizon is in the output next to the figure.

2. **The standard shortcut assumes a constant hazard.** `ARPU / churn_rate`
   -- the formula most LTV estimates come from -- is the sum of a geometric
   series, which is only the right sum if the monthly churn probability never
   changes. Here it falls from 4.2% in the first month to under 1% after a
   year, by construction. The shortcut is computed alongside the honest number
   precisely so the gap can be read off rather than argued about.

Prices are an input, not data: they are not in the store and cannot be
estimated from it. The defaults below are a stated assumption, overridable on
the command line, and every figure scales linearly in them -- which is itself
worth knowing, because it means a wrong price assumption moves the whole
comparison but not the ordering.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from src import survival
from src.store import DEFAULT_STORE

# A stated assumption, not a measurement. Monthly list price per plan.
MONTHLY_PRICE = {"basic": 9.0, "standard": 19.0, "premium": 39.0}

# Monthly discount rate. Default 0 so the headline number is a plain expected
# revenue total; 1%/month (about 12.7%/year) is reported beside it as the
# sensitivity, because discounting matters more the longer the horizon and
# quietly choosing one rate would hide that.
DEFAULT_DISCOUNT = 0.0


def restricted_value(
    curve: list[dict],
    price: float,
    horizon: int,
    discount: float = DEFAULT_DISCOUNT,
) -> float:
    """Expected revenue from one subscriber over `horizon` months.

        value = price * sum over t in 0..horizon-1 of S(t-1) / (1 + r)^t

    with `S(-1) = 1`: month 0 is paid by everyone, since a subscriber who
    churns in their first month was still there for it. Getting that boundary
    wrong is an off-by-one worth a whole month of revenue per subscriber, which
    on a 12-month horizon is most of the difference between plans.

    Reading `S(t-1)` rather than `S(t)` is the same convention day 2's
    retention table uses: survival through month t-1 is what says the
    subscriber is present at the start of month t.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 month, got {horizon}")
    if discount < 0:
        raise ValueError(f"discount must be non-negative, got {discount}")

    total = 0.0
    for month in range(horizon):
        alive = 1.0 if month == 0 else curve[month - 1]["survival"]
        total += price * alive / (1.0 + discount) ** month
    return total


def geometric_shortcut(curve: list[dict], price: float, horizon: int | None = None) -> float:
    """`price / churn_rate`, the constant-hazard formula, for comparison.

    Uses the pooled monthly hazard over all observed subscriber-months as the
    "churn rate", which is how the number is normally produced. With a horizon
    the series is truncated to the same number of months as
    `restricted_value`, so the two are comparable; without one it is the
    infinite sum, which is the version usually quoted.
    """
    events = sum(row["events"] for row in curve)
    at_risk = sum(row["at_risk"] for row in curve)
    rate = events / at_risk

    if horizon is None:
        return price / rate
    return price * sum((1.0 - rate) ** month for month in range(horizon))


def by_plan(
    con: duckdb.DuckDBPyConnection,
    horizon: int = 12,
    discount: float = DEFAULT_DISCOUNT,
    prices: dict[str, float] | None = None,
) -> list[dict]:
    """Restricted value per plan, with the shortcut and the gap beside it."""
    prices = prices or MONTHLY_PRICE
    curves = survival.kaplan_meier_by_segment(con, "plan")

    missing = set(curves) - set(prices)
    if missing:
        raise ValueError(f"no price given for plan(s) {sorted(missing)}")

    out = []
    for plan, curve in curves.items():
        if len(curve) < horizon:
            raise ValueError(
                f"horizon of {horizon} months exceeds the {len(curve)} months "
                f"observed for plan {plan!r} -- the curve would have to be "
                "extrapolated, which this does not do"
            )
        honest = restricted_value(curve, prices[plan], horizon, discount)
        shortcut = geometric_shortcut(curve, prices[plan], horizon)
        out.append(
            {
                "plan": plan,
                "price": prices[plan],
                "horizon": horizon,
                "value": honest,
                "shortcut": shortcut,
                "shortcut_error": shortcut / honest - 1.0,
                "survival_at_horizon": curve[horizon - 1]["survival"],
            }
        )
    return out


def render(
    store: str | Path = DEFAULT_STORE,
    horizon: int = 12,
    prices: dict[str, float] | None = None,
) -> str:
    store = Path(store)
    if not store.exists():
        raise FileNotFoundError(
            f"no store at {store} -- run `python -m src.store` to build one first"
        )

    prices = prices or MONTHLY_PRICE
    with duckdb.connect(str(store), read_only=True) as con:
        plain = by_plan(con, horizon, 0.0, prices)
        discounted = by_plan(con, horizon, 0.01, prices)
        pooled = survival.kaplan_meier(con)
        curves = survival.kaplan_meier_by_segment(con, "plan")

    lines = [
        f"Restricted subscriber value -- {store}",
        "",
        f"Expected revenue per subscriber over **{horizon} months**, from the",
        "Kaplan-Meier curve for their plan. Not a lifetime figure: survival is",
        f"still {pooled[-1]['survival']:.1%} when the observation window closes,",
        "so a lifetime number would be an extrapolation of the tail rather than",
        "a measurement. The horizon is part of the number.",
        "",
        "Prices are an assumption, not data: "
        + ", ".join(f"{plan} {price:.0f}" for plan, price in sorted(prices.items()))
        + " per month.",
        "",
        "plan      S(horizon)   value   value @1%/mo   price/churn   error",
        "--------  ----------  ------  -------------  ------------  ------",
    ]
    discounted_by_plan = {row["plan"]: row for row in discounted}
    for row in plain:
        lines.append(
            f"{row['plan']:<8}  {row['survival_at_horizon']:>10.1%}  "
            f"{row['value']:>6.0f}  {discounted_by_plan[row['plan']]['value']:>13.0f}  "
            f"{row['shortcut']:>12.0f}  {row['shortcut_error']:>+5.1%}"
        )

    worst = max(plain, key=lambda row: abs(row["shortcut_error"]))
    lines += [
        "",
        "The `price/churn` column is the constant-hazard shortcut -- a geometric",
        "series at the pooled monthly churn rate, truncated to the same horizon",
        "so the two are comparable. Truncated, it is close: wrong by "
        f"{worst['shortcut_error']:+.1%} at worst ({worst['plan']}) over {horizon} months,",
        "and the sign is the informative part. The pooled rate averages a hazard",
        "that starts at 4.2% and falls under 1.3%, so a horizon shorter than the",
        "averaging window is priced too cheaply in risk terms and the shortcut",
        "over-states value. At a 24-month horizon on this store the error falls",
        "to under 0.2%, which is arithmetic rather than virtue: the horizon and",
        "the window the rate was averaged over then coincide.",
        "",
        "The version actually quoted as LTV is the *unbounded* one, and that is",
        "a different animal:",
        "",
        "plan      price/churn, unbounded   implied lifetime   x the 24-month figure",
        "--------  ----------------------   ----------------   ---------------------",
    ]
    for plan, curve in curves.items():
        price = prices[plan]
        unbounded = geometric_shortcut(curve, price)
        longest = restricted_value(curve, price, len(curve))
        lines.append(
            f"{plan:<8}  {unbounded:>22.0f}   {unbounded / price:>13.0f} mo   "
            f"{unbounded / longest:>21.1f}"
        )

    lines += [
        "",
        "Those implied lifetimes (50-113 months) come from extrapolating a flat",
        "hazard well past the 24 months of data, and the observed hazard is still",
        "falling at the end of the window. The number is not wrong because the",
        "arithmetic is wrong; it is unknowable from this dataset, and the honest",
        "version of it is the restricted figure with its horizon attached.",
        "",
        "Every figure scales linearly in price, so a wrong price assumption",
        "moves the whole table and not the ordering.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", default=DEFAULT_STORE, type=Path)
    parser.add_argument("--horizon", type=int, default=12, help="months to value over")
    parser.add_argument(
        "--price",
        action="append",
        default=[],
        metavar="PLAN=AMOUNT",
        help="override a plan's monthly price (repeatable)",
    )
    args = parser.parse_args(argv)

    prices = dict(MONTHLY_PRICE)
    for override in args.price:
        plan, _, amount = override.partition("=")
        if plan not in prices or not amount:
            parser.error(f"--price expects PLAN=AMOUNT for a known plan, got {override!r}")
        prices[plan] = float(amount)

    print(render(args.store, args.horizon, prices))


if __name__ == "__main__":
    main()
