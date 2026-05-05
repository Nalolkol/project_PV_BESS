"""Single Plotly figure with 4 stacked subplots — fig.show() displays in PyCharm.

Rows (shared x-axis):
  1. Allocations — stacked bars for every component (DA buy/sell, BESS charge,
     BESS discharge, PV→grid, PV→BESS) plus per-asset aFRR UP/DOWN headroom as
     hatched slices. Net-export line + ±95 MW grid limits overlayed.
  2. SOC over the day with min (0) / max (200) limit lines and 50 % target.
  3. PV forecast (MW).
  4. Prices — spot (line, left axis) + aFRR UP/DOWN forecast vs actual (bars, right).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import config


def _to_quarters(hourly):
    return np.repeat(np.asarray(hourly, dtype=float), config.QUARTERS_PER_HOUR)


# Market colour palette — used consistently across all subplots.
# Asset distinction: BESS = solid fill, PV = "/" pattern (diagonal lines).
COLOR_DA      = "gold"            # Day-Ahead market
COLOR_AFRR_UP = "crimson"         # aFRR UP capacity
COLOR_AFRR_DN = "royalblue"       # aFRR DOWN capacity
COLOR_ID      = "mediumpurple"    # Intraday (recourse / mitigation)
COLOR_PV      = "forestgreen"     # PV asset when not tied to a market (e.g. routing to BESS)
COLOR_DEPR    = "grey"            # BESS depreciation (informational)
COLOR_SOC     = "purple"          # SOC (BESS state, market-independent)
COLOR_NET_EXP = "black"           # Net export line
COLOR_GRID    = "red"             # Grid limit
PV_PATTERN    = "/"               # PV asset → diagonal lines
BESS_PATTERN  = ""                # BESS asset → solid fill


def plot_overview(date, spot, pv, forecast_afrr, actual_afrr, cleared, schedule, profit,
                  scenario_label=""):
    q_idx    = schedule.index           # 96 quarter timestamps (CET)
    hour_idx = forecast_afrr.index      # 24 hourly timestamps

    # Per-quarter expansion of cleared aFRR (24 → 96)
    up_bess_q = _to_quarters(cleared["bess_up"])
    up_pv_q   = _to_quarters(cleared["pv_up"])
    dn_bess_q = _to_quarters(cleared["bess_down"])
    dn_pv_q   = _to_quarters(cleared["pv_down"])

    # ID mitigation option per quarter (zero in hour 0, then prev-hour BESS commitment).
    # Only BESS commitments need ID restoration — PV UP/DOWN don't drain BESS energy
    # (PV UP redirects PV away from BESS; PV DOWN curtails PV→grid). Within-hour SOC
    # reserve already handles the planned-trajectory shortfall from PV UP.
    qph = config.QUARTERS_PER_HOUR
    id_buy_q  = np.zeros(len(q_idx))   # ID import — refills BESS after BESS UP activation
    id_sell_q = np.zeros(len(q_idx))   # ID export — drains  BESS after BESS DOWN activation
    bess_up   = cleared["bess_up"].to_numpy()
    bess_dn   = cleared["bess_down"].to_numpy()
    for h in range(1, config.HOURS_PER_DAY):
        sl = slice(h * qph, (h + 1) * qph)
        id_buy_q[sl]  = bess_up[h - 1]
        id_sell_q[sl] = bess_dn[h - 1]

    # Split dual-use slices
    pv_grid_total       = schedule["pv_to_grid_mw"].values
    pv_grid_dn          = np.minimum(dn_pv_q, pv_grid_total)
    pv_grid_firm        = np.maximum(0.0, pv_grid_total - pv_grid_dn)
    pv_to_bess_total    = schedule["pv_to_bess_mw"].values
    pv_to_bess_up_slice = np.minimum(up_pv_q, pv_to_bess_total)
    pv_to_bess_firm     = np.maximum(0.0, pv_to_bess_total - pv_to_bess_up_slice)

    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[0.28, 0.11, 0.11, 0.13, 0.18, 0.19],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": True}],
               [{"secondary_y": True}]],
        subplot_titles=(
            "Allocations — DA + aFRR per asset (positive = export, negative = import). "
            "Hatched = aFRR headroom",
            "BESS state of charge",
            "PV — forecast (line) vs actual usage (bars)",
            "BESS — power flows (bars) vs ±50 MW capacity (dashed)",
            "Prices — spot (line, left) and aFRR UP/DOWN (bars, right)",
            "Profit — per-quarter components (bars) and cumulative cash profit (line)",
        ),
    )

    # ── Row 1: Allocations (stacked bars + net export line + grid limits) ──
    # Colour = market (DA / aFRR UP / aFRR DOWN / ID).
    # Pattern: BESS = solid, PV = diagonal lines, dual-use DA+aFRR DOWN PV = cross.
    # Legend order: positive markets (DA → UP → DOWN → ID) then negative side, then lines.
    bars = [
        # label,                                 values,                                 color,         pattern
        # ── Positive (export) ─────────────────────────────────────────────
        ("DA sell – BESS",                       schedule["bess_discharge_mw"].values,   COLOR_DA,      BESS_PATTERN),
        ("DA sell – PV",                         pv_grid_firm,                           COLOR_DA,      PV_PATTERN),
        ("DA sell – PV  (also bid as aFRR DOWN)", pv_grid_dn,                            COLOR_DA,      "x"),
        ("aFRR UP – BESS (headroom)",            up_bess_q,                              COLOR_AFRR_UP, BESS_PATTERN),
        ("aFRR UP – PV (redirect)",              up_pv_q,                                COLOR_AFRR_UP, PV_PATTERN),
        # ── Negative (import / charging) ──────────────────────────────────
        ("DA buy – BESS",                        -schedule["bess_charge_mw"].values,     COLOR_DA,      BESS_PATTERN),
        ("aFRR DOWN – BESS (headroom)",          -dn_bess_q,                             COLOR_AFRR_DN, BESS_PATTERN),
        ("PV → BESS (internal)",                 -pv_to_bess_firm,                       COLOR_PV,      PV_PATTERN),
    ]
    for label, vals, color, pattern in bars:
        fig.add_trace(go.Bar(
            x=q_idx, y=vals, name=label,
            marker=dict(color=color, pattern_shape=pattern,
                        line=dict(color="white", width=0.3)),
        ), row=1, col=1)

    # ID mitigation option bars — contingent on activation, semi-opaque
    fig.add_trace(go.Bar(
        x=q_idx, y=id_sell_q, name="ID sell option (DOWN mitigation)",
        marker=dict(color=COLOR_ID, line=dict(color="white", width=0.3)),
        opacity=0.45,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=q_idx, y=-id_buy_q, name="ID buy option (UP mitigation)",
        marker=dict(color=COLOR_ID, line=dict(color="white", width=0.3)),
        opacity=0.45,
    ), row=1, col=1)

    # Net export line
    fig.add_trace(go.Scatter(
        x=q_idx, y=schedule["net_export_mw"].values,
        name="Net export", mode="lines",
        line=dict(color=COLOR_NET_EXP, width=2),
    ), row=1, col=1)

    # Grid limits ±95
    for y, show in [(config.GRID_LIMIT_MW, True), (-config.GRID_LIMIT_MW, False)]:
        fig.add_trace(go.Scatter(
            x=[q_idx[0], q_idx[-1]], y=[y, y], mode="lines",
            name="Grid limit ±95 MW", legendgroup="grid",
            line=dict(color=COLOR_GRID, dash="dash", width=1),
            showlegend=show,
        ), row=1, col=1)

    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)

    # ── Row 2: SOC ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=q_idx, y=schedule["soc_mwh"].values,
        name="SOC", mode="lines",
        line=dict(color=COLOR_SOC, width=2),
        fill="tozeroy", fillcolor="rgba(128,0,128,0.10)",
    ), row=2, col=1)
    for y, label, dash in [
        (config.BESS_ENERGY_MWH,      f"SOC max ({config.BESS_ENERGY_MWH:.0f} MWh)", "dash"),
        (config.BESS_SOC_INITIAL_MWH, f"50 % target ({config.BESS_SOC_INITIAL_MWH:.0f} MWh)", "dot"),
        (0,                            "SOC min (0 MWh)",                            "dash"),
    ]:
        fig.add_trace(go.Scatter(
            x=[q_idx[0], q_idx[-1]], y=[y, y], mode="lines",
            name=label, line=dict(color=COLOR_SOC, width=1, dash=dash),
        ), row=2, col=1)
    fig.update_yaxes(title_text="SOC (MWh)", row=2, col=1,
                     range=[-10, config.BESS_ENERGY_MWH + 15])

    # ── Row 3: PV forecast + actual usage (PV→grid + PV→BESS) ─────────────
    fig.add_trace(go.Scatter(
        x=pv.index, y=pv.values, name="PV forecast (available)",
        mode="lines", line=dict(color=COLOR_PV, width=1.5, dash="dash"),
        fill="tozeroy", fillcolor="rgba(34,139,34,0.12)",
    ), row=3, col=1)
    fig.add_trace(go.Bar(
        x=q_idx, y=pv_grid_total, name="PV used → grid (DA)",
        marker=dict(color=COLOR_DA, pattern_shape=PV_PATTERN,
                    line=dict(color="white", width=0.3)),
    ), row=3, col=1)
    fig.add_trace(go.Bar(
        x=q_idx, y=pv_to_bess_total, name="PV used → BESS (internal)",
        marker=dict(color=COLOR_PV, pattern_shape=PV_PATTERN,
                    line=dict(color="white", width=0.3)),
    ), row=3, col=1)
    fig.update_yaxes(title_text="PV (MW)", row=3, col=1,
                     range=[0, max(config.PV_CAPACITY_MW, float(pv.max()) * 1.1)])

    # ── Row 4: BESS power flows + capacity ─────────────────────────────────
    bess_dis_v  = schedule["bess_discharge_mw"].values
    bess_chg_v  = schedule["bess_charge_mw"].values
    pv2bess_v   = schedule["pv_to_bess_mw"].values

    # Positive (discharge) — DA sell + aFRR UP headroom
    fig.add_trace(go.Bar(x=q_idx, y=bess_dis_v, name="BESS discharge → grid (DA)",
                         marker=dict(color=COLOR_DA, line=dict(color="white", width=0.3))),
                  row=4, col=1)
    fig.add_trace(go.Bar(x=q_idx, y=up_bess_q, name="BESS aFRR UP (headroom)",
                         marker=dict(color=COLOR_AFRR_UP, pattern_shape="/",
                                     line=dict(color="white", width=0.3))),
                  row=4, col=1)
    # Negative (charge) — DA buy + PV→BESS + aFRR DOWN headroom
    fig.add_trace(go.Bar(x=q_idx, y=-bess_chg_v, name="BESS charge ← grid (DA)",
                         marker=dict(color=COLOR_DA, line=dict(color="white", width=0.3))),
                  row=4, col=1)
    fig.add_trace(go.Bar(x=q_idx, y=-pv2bess_v, name="BESS charge ← PV",
                         marker=dict(color=COLOR_PV, pattern_shape=PV_PATTERN,
                                     line=dict(color="white", width=0.3))),
                  row=4, col=1)
    fig.add_trace(go.Bar(x=q_idx, y=-dn_bess_q, name="BESS aFRR DOWN (headroom)",
                         marker=dict(color=COLOR_AFRR_DN, pattern_shape="/",
                                     line=dict(color="white", width=0.3))),
                  row=4, col=1)
    # Capacity limits ±50 MW
    for y, show in [(config.BESS_POWER_MW, True), (-config.BESS_POWER_MW, False)]:
        fig.add_trace(go.Scatter(
            x=[q_idx[0], q_idx[-1]], y=[y, y], mode="lines",
            name=f"BESS power limit ±{config.BESS_POWER_MW:.0f} MW",
            legendgroup="bess_lim",
            line=dict(color=COLOR_GRID, dash="dash", width=1),
            showlegend=show,
        ), row=4, col=1)
    fig.update_yaxes(title_text="BESS (MW)", row=4, col=1,
                     range=[-config.BESS_POWER_MW * 1.15, config.BESS_POWER_MW * 1.15])

    # ── Row 5: Prices ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=spot.index, y=spot.values, name="Spot (EUR/MWh)",
        mode="lines", line=dict(color=COLOR_DA, width=2),
    ), row=5, col=1, secondary_y=False)

    # 4 bars per hour, side-by-side (manual offset since barmode="relative" stacks
    # by default — relative ignores offsetgroup, so we shift x positions instead).
    # Forecast = lighter shade of market colour; actual = full market colour.
    bw_ms = 12 * 60 * 1000   # bar width 12 min in ms
    offsets = {
        "up_fc":   -22,
        "up_ac":   -10,
        "down_fc":   2,
        "down_ac":  14,
    }
    for key, vals, color, opacity, label in [
        ("up_fc",   forecast_afrr["up"].values,   COLOR_AFRR_UP, 0.40, "aFRR UP forecast"),
        ("up_ac",   actual_afrr["up"].values,     COLOR_AFRR_UP, 0.95, "aFRR UP actual"),
        ("down_fc", forecast_afrr["down"].values, COLOR_AFRR_DN, 0.40, "aFRR DOWN forecast"),
        ("down_ac", actual_afrr["down"].values,   COLOR_AFRR_DN, 0.95, "aFRR DOWN actual"),
    ]:
        fig.add_trace(go.Bar(
            x=hour_idx + pd.Timedelta(minutes=offsets[key]),
            y=vals, name=label,
            width=bw_ms, marker_color=color, opacity=opacity,
        ), row=5, col=1, secondary_y=True)

    fig.update_yaxes(title_text="Spot (EUR/MWh)", row=5, col=1, secondary_y=False)
    fig.update_yaxes(title_text="aFRR (EUR/MW)", row=5, col=1, secondary_y=True)

    # ── Row 5: Profit (per-quarter bars + cumulative line) ─────────────────
    dt = config.DT_HOURS
    da_q   = spot.values * schedule["net_export_mw"].values * dt
    # aFRR is hourly → spread evenly across the 4 quarters of each hour
    up_rev_h = cleared["up"].to_numpy()   * cleared["clearing_up_eur_per_mw"].to_numpy()
    dn_rev_h = cleared["down"].to_numpy() * cleared["clearing_down_eur_per_mw"].to_numpy()
    up_q   = _to_quarters(up_rev_h) / qph
    dn_q   = _to_quarters(dn_rev_h) / qph
    cycle_q = schedule["bess_discharge_mw"].values * dt * config.BESS_CYCLE_COST_EUR_PER_MWH
    cash_q  = da_q + up_q + dn_q
    cash_cum = np.cumsum(cash_q)

    fig.add_trace(go.Bar(x=q_idx, y=da_q, name="DA cash / quarter",
                         marker_color=COLOR_DA, opacity=0.85),
                  row=6, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=q_idx, y=up_q, name="aFRR UP cash / quarter",
                         marker_color=COLOR_AFRR_UP, opacity=0.85),
                  row=6, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=q_idx, y=dn_q, name="aFRR DOWN cash / quarter",
                         marker_color=COLOR_AFRR_DN, opacity=0.85),
                  row=6, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=q_idx, y=-cycle_q, name="BESS depreciation (excluded)",
                         marker_color=COLOR_DEPR, opacity=0.45),
                  row=6, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=q_idx, y=cash_cum, name="Cumulative cash profit",
                             mode="lines", line=dict(color=COLOR_NET_EXP, width=2.2)),
                  row=6, col=1, secondary_y=True)

    fig.update_yaxes(title_text="EUR / quarter", row=6, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative EUR", row=6, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Time of day (CET)", row=6, col=1)

    # ── Layout ─────────────────────────────────────────────────────────────
    label_html = f" — <span style='color:#b00020'>[{scenario_label.upper()}]</span>" if scenario_label else ""
    title = (f"<b>Day overview — {date.date()}{label_html}</b>   "
             f"Cash profit: {profit['total_profit_eur']:.0f} EUR   "
             f"(DA {profit['da_revenue_eur']:.0f} + "
             f"aFRR UP {profit['afrr_up_revenue_eur']:.0f} + "
             f"aFRR DOWN {profit['afrr_down_revenue_eur']:.0f})   "
             f"BESS depreciation: {profit['cycle_cost_eur']:.0f} EUR (excluded)")
    # Group legend entries by subplot row, with a header per group.
    # Map y-axis name → (row number, group title). With secondary_y on rows 5 and 6,
    # plotly assigns yaxis names: y(r1), y2(r2), y3(r3), y4(r4), y5/y6(r5), y7/y8(r6).
    _row_of_yaxis = {"y": (1, "1 ∙ Allocations"),
                     "y2": (2, "2 ∙ SOC"),
                     "y3": (3, "3 ∙ PV"),
                     "y4": (4, "4 ∙ BESS"),
                     "y5": (5, "5 ∙ Prices"), "y6": (5, "5 ∙ Prices"),
                     "y7": (6, "6 ∙ Profit"), "y8": (6, "6 ∙ Profit")}
    seen_groups = set()
    for trace in fig.data:
        row, title_text = _row_of_yaxis.get(trace.yaxis or "y", (0, ""))
        if row == 0:
            continue
        trace.legendgroup = f"row{row}"
        if trace.legendgroup not in seen_groups:
            trace.legendgrouptitle = dict(text=f"<b>{title_text}</b>",
                                          font=dict(size=10))
            seen_groups.add(trace.legendgroup)

    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=13)),
        autosize=True, height=1300,          # taller — 6 subplots now
        barmode="relative",
        legend=dict(orientation="v", yanchor="top", y=1, x=1.02,
                    font=dict(size=9), tracegroupgap=8),
        margin=dict(l=60, r=240, t=70, b=50),
    )
    return fig


def plot_all(date, spot, pv, forecast_afrr, actual_afrr, cleared, schedule, profit,
             scenario_label=""):
    """Build and display the overview figure (PyCharm SciView via fig.show())."""
    fig = plot_overview(date, spot, pv, forecast_afrr, actual_afrr, cleared, schedule, profit,
                        scenario_label=scenario_label)
    # PyCharm Professional: 'browser' opens in browser; default renderer in PyCharm's
    # scientific mode displays inline in SciView. Override here if needed.
    if pio.renderers.default in ("", None):
        pio.renderers.default = "browser"
    fig.show()
