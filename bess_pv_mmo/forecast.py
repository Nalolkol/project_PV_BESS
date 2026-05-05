"""aFRR price forecast — simple historic mean by hour-of-day.

Should have added 'remove outlier' prices like remove higest 1% of values or something similar
"""


import pandas as pd
import config


def forecast_afrr(history, target_day):
    """24-row hourly forecast = mean by hour-of-day over the last 7 days."""
    fc = history.groupby(history.index.hour).mean().reindex(range(24)).ffill().bfill()
    fc.index = pd.date_range(target_day, periods=config.HOURS_PER_DAY, freq="h",
                             tz=target_day.tz)
    return fc
