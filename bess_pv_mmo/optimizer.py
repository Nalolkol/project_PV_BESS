"""Single-day MILP for joint aFRR + day-ahead optimisation.

Phases (one function `solve()`):
  - Phase A: pass forecast aFRR prices, integer aFRR volumes (per asset/direction)
    are decision variables.
  - Phase B: pass cleared aFRR volumes as fixed inputs (per asset/direction).

aFRR products (4 total):
  - BESS UP   : discharge headroom in BESS power
  - BESS DOWN : charge headroom in BESS power
  - PV UP     : redirect planned pv_to_bess back to grid (frees BESS charge slot)
  - PV DOWN   : curtail planned pv_to_grid

Simplifications:
  - aFRR feasibility uses a per-hour reserve (end-of-hour SOC must survive a full
    hour of activation in either direction); cumulative worst-case bands omitted.
  - Cycle limit is on planned DA discharge only.
"""
import cvxpy as cp
import numpy as np
import pandas as pd

import config


def solve(spot_prices, pv_forecast_mw, soc_start_mwh, *,
          afrr_up_price=None, afrr_down_price=None,
          afrr_fixed=None):
    """Return dict with schedule (DataFrame) and per-asset hourly aFRR ndarrays.

    afrr_fixed (Phase B only): dict with keys 'bess_up', 'bess_down', 'pv_up',
    'pv_down', each a (24,) array of cleared MW.
    """
    Q   = config.QUARTERS_PER_DAY
    H   = config.HOURS_PER_DAY
    QPH = config.QUARTERS_PER_HOUR
    dt  = config.DT_HOURS

    # ── Variables ───────────────────────────────────────────────────────────
    bess_chg    = cp.Variable(Q, nonneg=True)   # charge from grid
    bess_dis    = cp.Variable(Q, nonneg=True)   # discharge to grid
    pv_grid     = cp.Variable(Q, nonneg=True)   # PV exported to grid
    pv_to_bess  = cp.Variable(Q, nonneg=True)   # PV routed to BESS (no market settlement)
    soc         = cp.Variable(Q)                # SOc of the battery
    is_dis      = cp.Variable(Q, boolean=True)  # Variable (MILP) that makes sure battery does not charge and discharge sim.

    # This either makes aFRR a constant that is just true. Or a varaible that we optimize one.
    phase_a = afrr_fixed is None
    if phase_a:
        afrr_up_bess   = cp.Variable(H, integer=True)
        afrr_dn_bess   = cp.Variable(H, integer=True)
        afrr_up_pv     = cp.Variable(H, integer=True)
        afrr_dn_pv     = cp.Variable(H, integer=True)
    else:
        afrr_up_bess = cp.Constant(np.asarray(afrr_fixed["bess_up"],   dtype=float))
        afrr_dn_bess = cp.Constant(np.asarray(afrr_fixed["bess_down"], dtype=float))
        afrr_up_pv   = cp.Constant(np.asarray(afrr_fixed["pv_up"],     dtype=float))
        afrr_dn_pv   = cp.Constant(np.asarray(afrr_fixed["pv_down"],   dtype=float))

    # Quarter <-> hour expansion
    # - h_of_q — for each quarter q, its hour index. With QPH=4, it's [0,0,0,0,1,1,1,1,...,23,23,23,23].
    # - last_q_of_h — the index of the last quarter inside each hour: [3, 7, 11, ..., 95]. Used to pin the per-hour
    #   SOC reserve check to end-of-hour SOC.
    # - afrr_*_q = afrr_*[h_of_q] — fancy-indexing trick that "broadcasts" each hourly aFRR variable across its 4
    #   quarters, producing a length-96 CVXPY expression. Now hourly capacity commitments and quarterly dispatch live
    #   on the same axis and can be added/subtracted directly in constraints.
    h_of_q = np.arange(Q) // QPH
    last_q_of_h = np.arange(H) * QPH + (QPH - 1)
    afrr_up_bess_q = afrr_up_bess[h_of_q]
    afrr_dn_bess_q = afrr_dn_bess[h_of_q]
    afrr_up_pv_q   = afrr_up_pv[h_of_q]
    afrr_dn_pv_q   = afrr_dn_pv[h_of_q]

    # just make to an array
    pv   = pv_forecast_mw.to_numpy()
    spot = spot_prices.to_numpy()

    # Net export is based on pv and bess (excludes afrr)
    net_export = bess_dis - bess_chg + pv_grid

    # ── Constraints ─────────────────────────────────────────────────────────
    cons = [
        # PV balance: split between grid and BESS, never exceed forecast
        pv_grid + pv_to_bess <= pv,

        # BESS power: only charge OR discharge per quarter
        bess_dis <= config.BESS_POWER_MW * is_dis,
        bess_chg + pv_to_bess <= config.BESS_POWER_MW * (1 - is_dis),

        # BESS power headroom for aFRR (BESS-side only)
        bess_dis + afrr_up_bess_q                <= config.BESS_POWER_MW,
        bess_chg + pv_to_bess + afrr_dn_bess_q   <= config.BESS_POWER_MW,

        # PV UP: served from un-curtailed PV — direct PV-to-grid, BESS untouched, no SOC drift.
        # Equivalent to "PV not already routed to grid or BESS". The optimiser is free to
        # reduce pv_to_bess at planning time if more PV UP capacity is profitable.
        afrr_up_pv_q <= pv - pv_grid - pv_to_bess,
        # PV DOWN: can only curtail what is currently exported to grid
        afrr_dn_pv_q <= pv_grid,

        # Grid limit — activation flow during the hour itself.
        # PV DOWN excluded from the import side: curtailment can only REDUCE pv_to_grid
        # (already in net_export), never add new grid load. afrr_dn_pv ≤ pv_to_grid
        # caps it physically, so adding it here would just over-constrain.
        net_export + (afrr_up_bess_q + afrr_up_pv_q) <= config.GRID_LIMIT_MW,
        net_export - afrr_dn_bess_q                  >= -config.GRID_LIMIT_MW,

        # SOC dynamics (planned, no aFRR activation). pv_to_bess also charges.
        soc[0]  == soc_start_mwh + (bess_chg[0]  + pv_to_bess[0]  - bess_dis[0])  * dt,
        soc[1:] == soc[:-1]      + (bess_chg[1:] + pv_to_bess[1:] - bess_dis[1:]) * dt,
        soc >= 0,
        soc <= config.BESS_ENERGY_MWH,
        soc[Q - 1] == config.BESS_SOC_INITIAL_MWH,

        # Per-hour aFRR feasibility — end-of-hour SOC under full-hour activation:
        # UP scenario: only BESS UP drains SOC. PV UP is direct PV-to-grid (un-curtailment),
        # so it never touches the battery.
        soc[last_q_of_h] - afrr_up_bess * 1.0 >= 0,
        # DOWN scenario: only BESS DOWN raises SOC (PV DOWN is just curtailment).
        soc[last_q_of_h] + afrr_dn_bess * 1.0 <= config.BESS_ENERGY_MWH,

        # Cycle limit on planned DA discharge
        cp.sum(bess_dis) * dt <= config.BESS_DAILY_DISCHARGE_LIMIT_MWH,
    ]

    # ── Grid limit with ID restoration ──────────────────────────────────────
    # In any hour h ≥ 1 two flows can stack on the same side of the meter:
    #   1) this hour's aFRR activation
    #   2) an ID order placed this hour to restore SOC from h-1's activation
    # Both must fit inside ±95 MW together with the planned net_export.
    #
    # PV UP / PV DOWN are excluded from ID restoration: PV UP comes from
    # un-curtailed PV (battery untouched) and PV DOWN is curtailment (no flow).
    # Hour 0 is exempt — no prior commitment to restore.
    for h in range(1, H):
        q = slice(h * QPH, (h + 1) * QPH)

        # IMPORT side (negative net_export): DOWN absorbs, ID buy imports to refill UP
        activation_in = afrr_dn_bess[h]            # DOWN activation pulls power in
        id_refill_in  = afrr_up_bess[h - 1]        # buy back BESS UP delivered in h-1

        # EXPORT side (positive net_export): UP injects, ID sell exports to drain DOWN
        activation_out = afrr_up_bess[h] + afrr_up_pv[h]
        id_drain_out   = afrr_dn_bess[h - 1]       # sell off BESS DOWN absorbed in h-1

        cons += [
            net_export[q] - activation_in  - id_refill_in >= -config.GRID_LIMIT_MW,
            net_export[q] + activation_out + id_drain_out <=  config.GRID_LIMIT_MW,
        ]

    if phase_a:
        cons += [
            afrr_up_bess >= 0, afrr_up_bess <= config.BESS_POWER_MW,
            afrr_dn_bess >= 0, afrr_dn_bess <= config.BESS_POWER_MW,
            afrr_up_pv   >= 0, afrr_up_pv   <= config.PV_CAPACITY_MW,
            afrr_dn_pv   >= 0, afrr_dn_pv   <= config.PV_CAPACITY_MW,
            # No aFRR in the last hour — avoids end-SOC-target trap. VERY VERY SIMPLIFIED!
            afrr_up_bess[H - 1] == 0,
            afrr_dn_bess[H - 1] == 0,
            afrr_up_pv[H - 1]   == 0,
            afrr_dn_pv[H - 1]   == 0,
        ]

    # ── Objective ───────────────────────────────────────────────────────────

    #money made from allcating to DA market
    da_revenue = cp.sum(cp.multiply(spot, net_export)) * dt
    #
    cycle_cost = cp.sum(bess_dis) * dt * config.BESS_CYCLE_COST_EUR_PER_MWH

    # If Phase with aFRR we optimize on all. Else optimize with constant aFRR allocaiton and can only use DA!
    if afrr_up_price is not None:
        up_total = afrr_up_bess + afrr_up_pv
        dn_total = afrr_dn_bess + afrr_dn_pv
        afrr_revenue = (cp.sum(cp.multiply(afrr_up_price.to_numpy(),   up_total))
                        + cp.sum(cp.multiply(afrr_down_price.to_numpy(), dn_total)))
        objective = cp.Maximize(da_revenue + afrr_revenue - cycle_cost)
    # Phase only DA!
    else:
        objective = cp.Maximize(da_revenue - cycle_cost)

    problem = cp.Problem(objective, cons)
    problem.solve(solver=config.SOLVER, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Solver returned status: {problem.status}")

    solve_time_ms = (problem.solver_stats.solve_time or 0.0) * 1000
    phase_label = "A (joint aFRR + DA)" if phase_a else "B (DA, aFRR fixed)"
    print(f"    [solver] phase {phase_label}: status={problem.status}, "
          f"objective={problem.value:.0f} EUR, solve_time={solve_time_ms:.0f} ms")

    # ── Extract ─────────────────────────────────────────────────────────────
    def v(x):
        return np.asarray(x.value).flatten()

    schedule = pd.DataFrame({
        "bess_charge_mw":    v(bess_chg),
        "bess_discharge_mw": v(bess_dis),
        "pv_to_grid_mw":     v(pv_grid),
        "pv_to_bess_mw":     v(pv_to_bess),
        "net_export_mw":     v(bess_dis) - v(bess_chg) + v(pv_grid),
        "soc_mwh":           v(soc),
    }, index=spot_prices.index)

    return {
        "schedule":  schedule,
        "bess_up":   np.round(v(afrr_up_bess)).astype(int),
        "bess_down": np.round(v(afrr_dn_bess)).astype(int),
        "pv_up":     np.round(v(afrr_up_pv)).astype(int),
        "pv_down":   np.round(v(afrr_dn_pv)).astype(int),
        "objective": float(problem.value),
    }
