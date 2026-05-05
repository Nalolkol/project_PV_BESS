"""aFRR bid construction, pay-as-cleared, DA bid matrix and profit accounting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from helpers.bids import make_day_ahead_bids

# Bid pricing comes from config — one knob per asset/direction, easy to tweak.
_BID_PRICE = {
    ("bess", "up"):   config.BESS_AFRR_UP_BID_EUR_PER_MW,
    ("bess", "down"): config.BESS_AFRR_DOWN_BID_EUR_PER_MW,
    ("pv",   "up"):   config.PV_AFRR_UP_BID_EUR_PER_MW,
    ("pv",   "down"): config.PV_AFRR_DOWN_BID_EUR_PER_MW,
}


def build_afrr_bids(volumes):
    """One bid per hour, asset, direction at the config-specified flat price.

    volumes: dict with keys 'bess_up', 'bess_down', 'pv_up', 'pv_down',
    each a (24,) array of integer MW.
    """
    rows = []
    for asset in ("bess", "pv"):
        for direction in ("up", "down"):
            arr = volumes[f"{asset}_{direction}"]
            price = _BID_PRICE[(asset, direction)]
            for h in range(config.HOURS_PER_DAY):
                if arr[h] > 0:
                    rows.append({"hour": h, "asset": asset, "direction": direction,
                                 "volume_mw": int(arr[h]),
                                 "bid_price_eur_per_mw": round(price, 2)})
    return pd.DataFrame(rows, columns=["hour", "asset", "direction",
                                        "volume_mw", "bid_price_eur_per_mw"])


def clear_afrr(bids, actual_afrr):
    """Pay-as-cleared per asset/direction. Returns one row per hour with all 4 categories."""
    actual_up   = actual_afrr["up"].to_numpy()
    actual_down = actual_afrr["down"].to_numpy()
    rows = []
    for h in range(config.HOURS_PER_DAY):
        cleared = {"bess_up": 0, "bess_down": 0, "pv_up": 0, "pv_down": 0}
        for _, b in bids[bids["hour"] == h].iterrows():
            clearing = actual_up[h] if b["direction"] == "up" else actual_down[h]
            if b["bid_price_eur_per_mw"] <= clearing:
                cleared[f"{b['asset']}_{b['direction']}"] += int(b["volume_mw"])
        rows.append({
            "hour": h,
            **cleared,
            "up":   cleared["bess_up"]   + cleared["pv_up"],
            "down": cleared["bess_down"] + cleared["pv_down"],
            "clearing_up_eur_per_mw":   float(actual_up[h]),
            "clearing_down_eur_per_mw": float(actual_down[h]),
        })
    return pd.DataFrame(rows)


def build_da_bid_matrix(net_export_mw, spot_prices):
    """DA bid matrix in helpers.bids.make_day_ahead_bids format.

    Index = quarter timestamps; columns = price levels (EUR/MWh);
    values = cumulative net volume (positive = sell, negative = buy).

    Each quarter places its planned volume at its actual spot price — with perfect
    foresight on spot, this guarantees the planned position clears.
    """
    vols   = net_export_mw.round(1)
    prices = spot_prices.round(2)

    sell = pd.DataFrame(index=vols.index)
    buy  = pd.DataFrame(index=vols.index)

    for p in prices.unique():
        sell_mask = (prices == p) & (vols > 0)
        buy_mask  = (prices == p) & (vols < 0)
        if sell_mask.any():
            sell[p] = 0.0
            sell.loc[sell_mask, p] = vols[sell_mask]
        if buy_mask.any():
            buy[p] = 0.0
            buy.loc[buy_mask, p] = -vols[buy_mask]

    return make_day_ahead_bids(sell.fillna(0.0), buy.fillna(0.0))


def afrr_bids_to_wide(bids_long):
    """Convert long-format aFRR bids to a wide cumulative matrix per direction.

    Returns (up, down): each indexed by hour 0-23, columns = price levels,
    values = cumulative volume offered at that price or below.
    """
    def to_wide(subset):
        wide = pd.DataFrame(index=pd.RangeIndex(config.HOURS_PER_DAY))
        for _, row in subset.iterrows():
            h, p, v = int(row["hour"]), float(row["bid_price_eur_per_mw"]), float(row["volume_mw"])
            if p not in wide.columns:
                wide[p] = 0.0
            wide.loc[h, p] = wide.loc[h, p] + v
        wide = wide.fillna(0.0)
        if not wide.columns.empty:
            wide[round(wide.columns.min() - 0.01, 2)] = 0.0
            wide = wide.sort_index(axis=1).cumsum(axis=1)
            wide.columns = wide.columns.round(2)
        return wide
    return to_wide(bids_long[bids_long["direction"] == "up"]), to_wide(bids_long[bids_long["direction"] == "down"])


def compute_profit(schedule, spot_prices, cleared):
    """Realised profit: actual flows x actual prices."""
    dt = config.DT_HOURS
    da_revenue        = float((schedule["net_export_mw"] * spot_prices).sum() * dt)
    afrr_up_revenue   = float((cleared["up"]   * cleared["clearing_up_eur_per_mw"]).sum())
    afrr_down_revenue = float((cleared["down"] * cleared["clearing_down_eur_per_mw"]).sum())
    cycle_cost        = float(schedule["bess_discharge_mw"].sum() * dt * config.BESS_CYCLE_COST_EUR_PER_MWH)
    return {
        "da_revenue_eur":        da_revenue,
        "afrr_up_revenue_eur":   afrr_up_revenue,
        "afrr_down_revenue_eur": afrr_down_revenue,
        "cycle_cost_eur":        cycle_cost,
        "total_profit_eur":      da_revenue + afrr_up_revenue + afrr_down_revenue # - cycle_cost, # dont think cycle cost should be considered in the net profit!
    }


def save_outputs(date, afrr_bids_long, da_bid_matrix, schedule, suffix=""):
    """Write the two required bid matrices plus the schedule to CSV."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = date.strftime("%Y-%m-%d")
    sfx = f"_{suffix}" if suffix else ""
    up_wide, down_wide = afrr_bids_to_wide(afrr_bids_long)
    up_wide.to_csv(  config.OUTPUT_DIR / f"bids_afrr_up_{date_str}{sfx}.csv",   index=True)
    down_wide.to_csv(config.OUTPUT_DIR / f"bids_afrr_down_{date_str}{sfx}.csv", index=True)
    da_bid_matrix.to_csv(config.OUTPUT_DIR / f"bids_da_{date_str}{sfx}.csv",    index=True)
    schedule.to_csv(     config.OUTPUT_DIR / f"schedule_{date_str}{sfx}.csv")
