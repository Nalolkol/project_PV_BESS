"""aFRR price forecast — simple historic mean by hour-of-day.

Kept intentionally minimal given (a) we bid 0 EUR/MW so volume choice is the
only thing forecast accuracy affects, and (b) the ID-mitigation assumption
absorbs forecast error on activation impact.
"""
import pandas as pd

import config


def forecast_afrr(history, target_day):
    """24-row hourly forecast = mean by hour-of-day over the last 7 days."""
    fc = history.groupby(history.index.hour).mean().reindex(range(24)).ffill().bfill()
    fc.index = pd.date_range(target_day, periods=config.HOURS_PER_DAY, freq="h",
                             tz=target_day.tz)
    return fc
