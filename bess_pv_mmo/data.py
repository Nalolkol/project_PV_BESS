"""Energidataservice loaders. Returns CET-indexed Series/DataFrames."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config
from helpers.data import get_capacity_market_data, get_forecasts, get_spot_price


def load_spot(t0, t1):
    """Day-ahead spot price (EUR/MWh), 15-min CET."""
    df = get_spot_price(t0, t1, config.PRICE_AREA)
    return (df.set_index("TimeUTC")["DayAheadPriceEUR"]
              .tz_convert("CET").astype(float).sort_index()
              .rename("spot_eur_per_mwh"))


def load_pv_forecast(t0, t1):
    """PV forecast (MW), 15-min CET, scaled to plant capacity."""
    df = get_forecasts(t0, t1, config.PRICE_AREA, "Solar")
    raw = df.set_index("Minutes5UTC")["ForecastDayAhead"].tz_convert("CET").astype(float)
    share = config.PV_CAPACITY_MW / config.DK1_SOLAR_PEAK_MW
    return (raw.resample("15min").mean()
               .mul(share)
               .clip(upper=config.PV_CAPACITY_MW)
               .fillna(0.0)
               .rename("pv_mw"))


def load_afrr(t0, t1):
    """aFRR up/down capacity prices (EUR/MW), hourly CET."""
    df = get_capacity_market_data(t0, t1, config.PRICE_AREA)
    return (df.set_index("TimeUTC")[["UpPriceEUR", "DownPriceEUR"]]
              .tz_convert("CET").astype(float).sort_index()
              .rename(columns={"UpPriceEUR": "up", "DownPriceEUR": "down"}))
