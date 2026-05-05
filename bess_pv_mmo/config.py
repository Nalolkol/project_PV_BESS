"""Asset, market and solver constants."""
from pathlib import Path

# Asset
GRID_LIMIT_MW = 95.0
PV_CAPACITY_MW = 100.0
DK1_SOLAR_PEAK_MW = 2368.0   # all-time DK1 solar forecast peak (simpel); used to scale DK1-wide forecast to plant. OUR SHARE OF THE TOTAL SOALR
BESS_POWER_MW = 50.0
BESS_ENERGY_MWH = 200.0
BESS_CYCLE_COST_EUR_PER_MWH = 25.0
BESS_DAILY_DISCHARGE_LIMIT_MWH = 400.0   # 2 cycles x 200 MWh
BESS_SOC_INITIAL_MWH = 100.0             # 50% start/end target

# Market
PRICE_AREA = "DK1"
QUARTERS_PER_DAY = 96
HOURS_PER_DAY = 24
QUARTERS_PER_HOUR = 4
DT_HOURS = 0.25

# Forecast
AFRR_FORECAST_LOOKBACK_DAYS = 7

# aFRR bid prices (EUR/MW) — flat, per asset and direction.
# Marginal pricing: revenue = actual clearing price, not our bid. So under the
BESS_AFRR_UP_BID_EUR_PER_MW   = 5.0
BESS_AFRR_DOWN_BID_EUR_PER_MW = 5.0
PV_AFRR_UP_BID_EUR_PER_MW     = 5.0
PV_AFRR_DOWN_BID_EUR_PER_MW   = 5.0

# Solver
SOLVER = "HIGHS"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
