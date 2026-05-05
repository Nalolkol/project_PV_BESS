"""Single-day driver. Edit the two knobs below and run `python main.py`.

Pipeline (per scenario):
  1. Phase A — joint MILP for aFRR + DA using the chosen aFRR price source.
  2. Build aFRR bid matrix and clear vs actual aFRR prices.
  3. Phase B — re-solve DA with cleared aFRR commitments fixed.
  4. Sanity-check schedule, build DA bid matrix, compute realised profit.
  5. Save CSVs and show the overview plot.

Scenarios:
  - "forecast": Phase A uses the 7-day rolling-mean aFRR forecast.
  - "oracle":   Phase A uses the actual aFRR clearing prices →
                perfect-foresight upper bound for benchmarking.
"""
import pandas as pd

import bids
import config
import data
import forecast
import optimizer
import plots

# ── Knobs — edit and run ───────────────────────────────────────────────────
RUN_DATE   = "2026-05-01"   # Trading day, YYYY-MM-DD
RUN_ORACLE = True           # True → also run perfect-foresight benchmark
# ───────────────────────────────────────────────────────────────────────────


def sanity_check(schedule):
    """Hard asserts: solver output must respect the physical constraints."""
    soc = schedule["soc_mwh"].values
    ne  = schedule["net_export_mw"].values
    chg = schedule["bess_charge_mw"].values + schedule["pv_to_bess_mw"].values
    dis = schedule["bess_discharge_mw"].values
    tol = 1e-4

    assert (soc >= -tol).all() and (soc <= config.BESS_ENERGY_MWH + tol).all(), \
        f"SOC out of [0, {config.BESS_ENERGY_MWH}] (min={soc.min():.2f}, max={soc.max():.2f})"
    assert (abs(ne) <= config.GRID_LIMIT_MW + tol).all(), \
        f"|net_export| > {config.GRID_LIMIT_MW} (max |ne|={abs(ne).max():.2f})"
    assert (chg <= config.BESS_POWER_MW + tol).all(), \
        f"BESS charge power > {config.BESS_POWER_MW} (max={chg.max():.2f})"
    assert (dis <= config.BESS_POWER_MW + tol).all(), \
        f"BESS discharge power > {config.BESS_POWER_MW} (max={dis.max():.2f})"
    print("  [check] SOC bounds, grid limit, BESS power: PASS")


def run_scenario(day, spot, pv, actual_afrr, afrr_prices_for_phase_a,
                 *, label, suffix):
    """Phase A → clear → Phase B → outputs. Returns the profit dict."""
    print(f"== {day.date()}  scenario: {label} ==")

    a = optimizer.solve(spot, pv, config.BESS_SOC_INITIAL_MWH,
                        afrr_up_price=afrr_prices_for_phase_a["up"],
                        afrr_down_price=afrr_prices_for_phase_a["down"])

    volumes = {"bess_up":   a["bess_up"],   "bess_down": a["bess_down"],
               "pv_up":     a["pv_up"],     "pv_down":   a["pv_down"]}
    afrr_bids_long = bids.build_afrr_bids(volumes)
    cleared        = bids.clear_afrr(afrr_bids_long, actual_afrr)

    b = optimizer.solve(spot, pv, config.BESS_SOC_INITIAL_MWH,
                        afrr_fixed={
                            "bess_up":   cleared["bess_up"].to_numpy(),
                            "bess_down": cleared["bess_down"].to_numpy(),
                            "pv_up":     cleared["pv_up"].to_numpy(),
                            "pv_down":   cleared["pv_down"].to_numpy(),
                        })

    sanity_check(b["schedule"])

    da_bid_matrix = bids.build_da_bid_matrix(b["schedule"]["net_export_mw"], spot)
    profit        = bids.compute_profit(b["schedule"], spot, cleared)

    bids.save_outputs(day, afrr_bids_long, da_bid_matrix, b["schedule"], suffix=suffix)
    plots.plot_all(day, spot, pv, afrr_prices_for_phase_a, actual_afrr,
                   cleared, b["schedule"], profit, scenario_label=label)
    return profit


def print_summary(profit, label):
    print()
    print(f"  --- {label} ---")
    print(f"  DA revenue:        {profit['da_revenue_eur']:>10.0f} EUR")
    print(f"  aFRR UP revenue:   {profit['afrr_up_revenue_eur']:>10.0f} EUR")
    print(f"  aFRR DOWN revenue: {profit['afrr_down_revenue_eur']:>10.0f} EUR")
    print(f"  Cycle cost:        {-profit['cycle_cost_eur']:>10.0f} EUR")
    print(f"  -------------------------------------")
    print(f"  Total profit:      {profit['total_profit_eur']:>10.0f} EUR")


def main():
    day           = pd.Timestamp(RUN_DATE, tz="CET")
    next_day      = day + pd.Timedelta(days=1)
    history_start = day - pd.Timedelta(days=config.AFRR_FORECAST_LOOKBACK_DAYS)

    spot        = data.load_spot(day, next_day).iloc[:96]
    pv          = data.load_pv_forecast(day, next_day).iloc[:96]
    actual_afrr = data.load_afrr(day, next_day).iloc[:24]
    history     = data.load_afrr(history_start, day)
    fc          = forecast.forecast_afrr(history, day)

    # Forecast scenario
    forecast_profit = run_scenario(
        day, spot, pv, actual_afrr,
        afrr_prices_for_phase_a=fc,
        label="forecast", suffix="",
    )
    print_summary(forecast_profit, "FORECAST")

    # Oracle scenario (perfect foresight on aFRR prices) — optional
    if RUN_ORACLE:
        oracle_prices = pd.DataFrame({"up":   actual_afrr["up"].values,
                                      "down": actual_afrr["down"].values},
                                     index=fc.index)
        oracle_profit = run_scenario(
            day, spot, pv, actual_afrr,
            afrr_prices_for_phase_a=oracle_prices,
            label="oracle", suffix="oracle",
        )
        print_summary(oracle_profit, "ORACLE (perfect foresight)")

        gap = oracle_profit["total_profit_eur"] - forecast_profit["total_profit_eur"]
        pct = (100.0 * forecast_profit["total_profit_eur"] /
               oracle_profit["total_profit_eur"]) if oracle_profit["total_profit_eur"] else float("nan")
        print()
        print(f"  Forecast / Oracle: {pct:5.1f} %   "
              f"(forecast leaves {gap:+.0f} EUR on the table)")

    print()
    print(f"  Outputs in {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
