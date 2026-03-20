#!/usr/bin/env python3
"""
Pair Backtest — planLeg Engine on Both Legs
============================================
Each leg runs the full simulate() logic from sim.js (ported to Python):
  - Dynamic leverage profile (DECREASING_PROFILE / SINGLE_BUY / ATH_CONSERVATIVE)
  - Anchor system (S_A, E_A, beta)
  - planBuyProfile with tau/gamma shaping
  - Per-step fee + slippage + funding
  - clampLtoZero, boundStepDirection, randomizeStepSlip

Pair trade logic:
  - OPEN  when spread z-score crosses ±z_entry
  - CLOSE when spread z-score reverts inside ±z_exit
  - STOP  when spread z-score blows out to ±z_stop OR time > max_hold candles

Both legs run their simulate engine simultaneously during the open period.
The leg going against us (SHORT side) runs in reverse: DOWN steps = price UP.

Usage:
    python pair_backtest_planleg.py
    python pair_backtest_planleg.py --a BNB/USDT --b BTC/USDT
    python pair_backtest_planleg.py --a ETH/USDT --b LINK/USDT --leverage 5 --days 365
    python pair_backtest_planleg.py --a BNB/USDT --b BTC/USDT --z-entry 1.8 --z-stop 3.0
"""

import argparse
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
#  EXCHANGE
# ─────────────────────────────────────────────
exchange = ccxt.binance({
    "options": {"defaultType": "future"},
    "enableRateLimit": True,
})

# ─────────────────────────────────────────────
#  planLeg ENGINE  (direct port of sim.js)
# ─────────────────────────────────────────────


@dataclass
class LegState:
    """Mutable state of one leg's simulate engine."""
    S_A:   float        # anchor price
    E_A:   float        # anchor equity
    S:     float        # current price
    E:     float        # current equity
    N:     float        # current notional
    beta:  float        # computed from L_ATH, pUp
    total_fees: float = 0.0
    ath_breaks: int = 0
    steps_taken: int = 0


def make_leg(entry_price: float, equity: float, cfg: dict) -> LegState:
    pUp = cfg["pUp"]
    L_ATH = cfg["L_ATH"]
    beta = math.log(1 + L_ATH * pUp) / math.log(1 + pUp)
    N0 = cfg.get("L0", L_ATH) * equity
    return LegState(
        S_A=entry_price, E_A=equity,
        S=entry_price,   E=equity, N=N0,
        beta=beta,
    )


def e_star(S: float, state: LegState, cfg: dict) -> float:
    return state.E_A * (S / state.S_A) ** state.beta


def k_from_s(S: float, state: LegState, cfg: dict) -> int:
    pUp = cfg["pUp"]
    if S >= state.S_A:
        return 0
    return math.ceil(math.log(state.S_A / S) / math.log(1 + pUp))


def plan_buy_profile(S_after, E_cur, k_after, state, cfg):
    pUp = cfg["pUp"]
    L_ATH = cfg["L_ATH"]
    tau = cfg.get("planTau",   0.1)
    gamma = cfg.get("planGamma", 2.0)
    EPS = 1e-12
    m = k_after - 1
    if m <= 0:
        return []
    S_final = S_after * (1 + pUp) ** k_after
    E_objK = e_star(S_final, state, cfg)
    Emid = E_objK / (1 + L_ATH * pUp)
    Gtotal = Emid / max(E_cur, EPS)
    if Gtotal <= 0 or not math.isfinite(Gtotal):
        return []
    Leq = (Gtotal ** (1 / m) - 1) / pUp
    gprime = []
    for j in range(1, m + 1):
        frac = (m - j) / (m - 1) if m > 1 else 0
        fj = 1 + tau * frac ** gamma
        gprime.append(1 + (Leq * fj) * pUp)
    prod = 1.0
    for g in gprime:
        prod *= g
    C = (Gtotal / max(prod, EPS)) ** (1 / m)
    Lj = [(C * g - 1) / pUp for g in gprime]
    return Lj


def l_pre_buy_equal(E_obj_final, E_cur, m, L_ATH, pUp):
    EPS = 1e-12
    Emid = E_obj_final / (1 + L_ATH * pUp)
    return ((Emid / max(E_cur, EPS)) ** (1 / max(m, 1)) - 1) / pUp


def dynamic_pct(L_current, cfg):
    L_max = cfg.get("L_max", 15)
    pUpR = cfg.get("pUpRange", [0.004, 0.010])
    pDnR = cfg.get("pDnRange", [-0.010, -0.004])
    ratio = min(L_current / max(L_max, 1e-12), 1.0)
    adjUp = pUpR[0] + (pUpR[1] - pUpR[0]) * (1 - ratio)
    adjDn = pDnR[1] + (pDnR[0] - pDnR[1]) * (1 - ratio)
    adjUp = max(min(adjUp, pUpR[1]),  pUpR[0])
    adjDn = max(min(adjDn, pDnR[0]),  pDnR[1])
    return adjUp, adjDn


def step_leg(state: LegState, direction: str, cfg: dict) -> dict:
    """
    Apply one price step (UP or DOWN) to the leg using the planLeg engine.
    direction: 'UP' or 'DOWN'
    Returns dict with all step metrics.
    """
    EPS = 1e-12
    pUp = cfg["pUp"]
    L_ATH = cfg["L_ATH"]
    fee_rate = cfg.get("feeRate",       0.0004)
    slip_rate = cfg.get("slippageRate",  0.0001)
    funding_r = cfg.get("fundingRate",   0.0)
    slip_up = cfg.get("stepSlipUpAbs", 0.0002)
    slip_dn = cfg.get("stepSlipDnAbs", 0.0002)
    randomize = cfg.get("randomizeStepSlip", True)
    bound_dir = cfg.get("boundStepDirection", True)
    clamp_l = cfg.get("clampLtoZero", True)

    is_up = (direction == "UP")

    S_before = state.S
    E_before = state.E
    N_before = state.N
    L_used = N_before / max(E_before, EPS)
    k_before = k_from_s(S_before, state, cfg)

    adj_up, adj_dn = dynamic_pct(L_used, cfg)
    p_base = adj_up if is_up else adj_dn

    rnd = (random.random() * 2 - 1) if randomize else 1.0
    p_eff = (p_base + rnd * slip_up) if is_up else (p_base - rnd * slip_dn)

    if bound_dir:
        if is_up:
            p_eff = max(p_eff, 0)
        else:
            p_eff = min(p_eff, 0)

    if 1 + L_used * p_eff <= 0:
        p_eff = -0.9999 / max(L_used, EPS)

    S_after = S_before * (1 + p_eff)
    N_after = N_before * (1 + p_eff)
    E_after_gross = E_before + N_before * p_eff

    broke = is_up and (S_after > state.S_A)
    k_after = k_from_s(S_after, state, cfg)
    m_after = max(k_after - 1, 0)

    L_target = None

    if k_after == 0:
        L_target = L_ATH

    elif k_after == 1:
        S_end_up = S_after * (1 + pUp)
        E_obj = e_star(S_end_up, state, cfg)
        denom = max(E_after_gross, EPS) * pUp
        L_target = (E_obj / max(E_after_gross, EPS) - 1) / pUp

    else:
        seq_L = plan_buy_profile(S_after, E_after_gross, k_after, state, cfg)
        S_final = S_after * (1 + pUp) ** k_after
        E_objK = e_star(S_final, state, cfg)
        L_equal = l_pre_buy_equal(E_objK, E_after_gross, m_after, L_ATH, pUp)

        if seq_L:
            L_target = seq_L[0] if math.isfinite(seq_L[0]) else L_equal
            if m_after == 1:
                L_target = L_equal
        else:
            L_target = L_equal

    if clamp_l and L_target < 0:
        L_target = 0.0

    N_target = L_target * max(E_after_gross, EPS)
    dN = N_target - N_after
    trade_cost = abs(dN) * (fee_rate + slip_rate)
    fund_cost = abs(N_target) * funding_r
    E_after = E_after_gross - trade_cost - fund_cost

    # Update anchor if ATH broken
    if broke:
        state.S_A = S_after
        state.E_A = E_after
        state.ath_breaks += 1

    state.S = S_after
    state.E = E_after
    state.N = N_target
    state.total_fees += trade_cost + fund_cost
    state.steps_taken += 1

    return {
        "S_before": S_before, "S_after": S_after,
        "E_before": E_before, "E_after": E_after,
        "L_used": L_used, "L_target": L_target,
        "k_before": k_before, "k_after": k_after,
        "p_eff": p_eff, "dN": dN,
        "trade_cost": trade_cost, "broke_ath": broke,
    }


# ─────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────
def fetch(symbol: str, days: int, timeframe: str) -> pd.Series:
    print(f"  Fetching {symbol} ({days}d × {timeframe})…", end=" ", flush=True)
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_ohlcv = []
    while True:
        try:
            batch = exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since, limit=1000)
            if not batch:
                break
            all_ohlcv.extend(batch)
            since = batch[-1][0] + 1
            if since >= int(datetime.now().timestamp() * 1000):
                break
        except Exception as e:
            print(f"\n  Error: {e}")
            break
    if not all_ohlcv:
        return pd.Series(dtype=float)
    df = pd.DataFrame(all_ohlcv, columns=[
                      "ts", "open", "high", "low", "close", "vol"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("ts", inplace=True)
    print(f"{len(df)} candles")
    return df["close"]


# ─────────────────────────────────────────────
#  BACKTEST
# ─────────────────────────────────────────────
def run_backtest(cfg: dict) -> dict:
    pa = fetch(cfg["sym_a"], cfg["days"], cfg["timeframe"])
    pb = fetch(cfg["sym_b"], cfg["days"], cfg["timeframe"])

    df = pd.DataFrame({"A": pa, "B": pb}).dropna()
    if len(df) < cfg["z_window"] + 10:
        print("  Not enough data.")
        return {}

    # Z-score of log spread
    df["log_A"] = np.log(df["A"])
    df["log_B"] = np.log(df["B"])
    df["spread"] = df["log_A"] - df["log_B"]
    df["roll_mean"] = df["spread"].rolling(cfg["z_window"]).mean()
    df["roll_std"] = df["spread"].rolling(cfg["z_window"]).std()
    df["zscore"] = (df["spread"] - df["roll_mean"]) / df["roll_std"]
    df.dropna(inplace=True)

    # Per-candle direction labels for each asset
    # UP = price higher than prev candle, DOWN = lower
    df["dir_A"] = (df["A"] >= df["A"].shift(1)).map(
        {True: "UP", False: "DOWN"})
    df["dir_B"] = (df["B"] >= df["B"].shift(1)).map(
        {True: "UP", False: "DOWN"})
    df.dropna(inplace=True)

    # Correlation
    ret_A = np.log(df["A"] / df["A"].shift(1)).dropna()
    ret_B = np.log(df["B"] / df["B"].shift(1)).dropna()
    correlation = float(ret_A.corr(ret_B))

    # ── Leg engine config ────────────────────────────────
    leg_cfg = {
        "pUp":              cfg.get("pUp",     0.006),
        "L_ATH":            cfg.get("leverage", 5.0),
        "L_max":            cfg.get("L_max",   15.0),
        "pUpRange":         cfg.get("pUpRange", [0.004, 0.010]),
        "pDnRange":         cfg.get("pDnRange", [-0.010, -0.004]),
        "feeRate":          cfg.get("fee_rate",   0.0004),
        "slippageRate":     cfg.get("slippage",   0.0001),
        "fundingRate":      cfg.get("funding_8h", 0.0003) / 6,  # per 4h candle
        "stepSlipUpAbs":    0.0002,
        "stepSlipDnAbs":    0.0002,
        "randomizeStepSlip": True,
        "boundStepDirection": True,
        "planTau":          cfg.get("planTau",   0.1),
        "planGamma":        cfg.get("planGamma", 2.0),
        "clampLtoZero":     True,
    }

    capital = cfg["capital"]

    # ── State machine ────────────────────────────────────
    position = 0      # 0=flat, +1=long A short B, -1=short A long B
    entry_z = 0.0
    entry_idx = 0
    leg_long = None   # LegState for the LONG leg
    leg_short = None   # LegState for the SHORT leg
    # which symbol is LONG/SHORT this trade
    long_sym = None
    short_sym = None

    equity = capital * 2
    equity_curve = []
    trades = []
    peak_eq = equity

    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        z = df["zscore"].iloc[idx]
        dir_a = df["dir_A"].iloc[idx]
        dir_b = df["dir_B"].iloc[idx]
        p_a = float(df["A"].iloc[idx])
        p_b = float(df["B"].iloc[idx])

        # ── While in trade: step both leg engines ────────
        if position != 0:
            # LONG leg: moves in its actual price direction
            # SHORT leg: moves in OPPOSITE price direction
            #   (when price goes UP on the SHORT leg, we LOSE → feed DOWN to engine)
            long_dir = dir_a if position == +1 else dir_b
            short_dir_price = dir_b if position == +1 else dir_a
            # Invert direction for short leg engine
            short_dir = "DOWN" if short_dir_price == "UP" else "UP"

            step_long = step_leg(leg_long,  long_dir,  leg_cfg)
            step_short = step_leg(leg_short, short_dir, leg_cfg)

            # Combined equity = sum of both leg equities
            equity = leg_long.E + leg_short.E

        equity_curve.append(equity)
        peak_eq = max(peak_eq, equity)

        # ── Entry ────────────────────────────────────────
        if position == 0:
            if z < -cfg["z_entry"]:
                # Spread cheap: A underperforming B → LONG A, SHORT B
                position = +1
                entry_z = z
                entry_idx = idx
                long_sym = cfg["sym_a"]
                short_sym = cfg["sym_b"]
                # Initialise both leg engines at current prices
                leg_long = make_leg(p_a, capital, leg_cfg)
                leg_short = make_leg(p_b, capital, leg_cfg)

            elif z > cfg["z_entry"]:
                # Spread rich: A outperforming B → SHORT A, LONG B
                position = -1
                entry_z = z
                entry_idx = idx
                long_sym = cfg["sym_b"]
                short_sym = cfg["sym_a"]
                leg_long = make_leg(p_b, capital, leg_cfg)
                leg_short = make_leg(p_a, capital, leg_cfg)

        # ── Exit: target ─────────────────────────────────
        elif position == +1 and z >= -cfg["z_exit"]:
            _close_trade(trades, position, entry_z, z, entry_idx, idx,
                         leg_long, leg_short, equity, capital*2, "target")
            equity = leg_long.E + leg_short.E
            position = 0

        elif position == -1 and z <= cfg["z_exit"]:
            _close_trade(trades, position, entry_z, z, entry_idx, idx,
                         leg_long, leg_short, equity, capital*2, "target")
            equity = leg_long.E + leg_short.E
            position = 0

        # ── Exit: z stop ──────────────────────────────────
        elif abs(z) >= cfg["z_stop"]:
            _close_trade(trades, position, entry_z, z, entry_idx, idx,
                         leg_long, leg_short, equity, capital*2, "z_stop")
            equity = leg_long.E + leg_short.E
            position = 0

        # ── Exit: time stop ───────────────────────────────
        elif (idx - entry_idx) >= cfg["max_hold"]:
            _close_trade(trades, position, entry_z, z, entry_idx, idx,
                         leg_long, leg_short, equity, capital*2, "time_stop")
            equity = leg_long.E + leg_short.E
            position = 0

    # Close open position at end
    if position != 0:
        z_last = float(df["zscore"].iloc[-1])
        _close_trade(trades, position, entry_z, z_last, entry_idx,
                     len(df)-1, leg_long, leg_short, equity, capital*2, "eod")
        equity = leg_long.E + leg_short.E

    # ── Statistics ───────────────────────────────────────
    eq_arr = np.array(equity_curve)
    run_max = np.maximum.accumulate(eq_arr)
    drawdowns = (eq_arr - run_max) / np.maximum(run_max, 1e-9)
    max_dd = float(drawdowns.min())

    td = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_z", "exit_z", "duration", "pnl", "long_ath", "short_ath",
                 "long_fees", "short_fees", "exit_reason"])

    total_ret = (equity - capital * 2) / (capital * 2) * 100
    n_trades = len(td)
    win_rate = float((td["pnl"] > 0).mean() * 100) if n_trades else 0
    avg_pnl = float(td["pnl"].mean()) if n_trades else 0
    avg_dur = float(td["duration"].mean()) if n_trades else 0
    n_stops = int((td["exit_reason"] == "z_stop").sum()) if n_trades else 0
    n_time = int((td["exit_reason"] == "time_stop").sum()) if n_trades else 0
    total_fees = float(td["long_fees"].sum() +
                       td["short_fees"].sum()) if n_trades else 0
    avg_ath = float((td["long_ath"] + td["short_ath"]
                     ).mean()) if n_trades else 0

    candle_h = {"1h": 1, "4h": 4, "1d": 24}.get(cfg["timeframe"], 4)
    eq_s = pd.Series(eq_arr)
    dr = eq_s.pct_change().dropna()
    ann_f = math.sqrt(24 / candle_h * 365)
    sharpe = float(dr.mean() / dr.std() *
                   ann_f) if float(dr.std()) > 0 else 0.0

    return {
        "sym_a": cfg["sym_a"], "sym_b": cfg["sym_b"],
        "timeframe": cfg["timeframe"], "days": cfg["days"],
        "leverage": cfg["leverage"], "capital_per_leg": capital,
        "total_return_pct": round(total_ret, 2),
        "final_equity": round(equity, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 1),
        "avg_pnl_usd": round(avg_pnl, 2),
        "avg_duration_candles": round(avg_dur, 1),
        "avg_duration_hours": round(avg_dur * candle_h, 1),
        "n_z_stops": n_stops,
        "n_time_stops": n_time,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2),
        "total_fees_usd": round(total_fees, 2),
        "avg_ath_breaks": round(avg_ath, 1),
        "correlation": round(correlation, 4),
        "trades_df": td,
        "equity_curve": eq_arr,
    }


def _close_trade(trades, position, entry_z, exit_z,
                 entry_idx, exit_idx,
                 leg_long: LegState, leg_short: LegState,
                 equity, start_equity, reason):
    trades.append({
        "entry_z":    round(entry_z, 3),
        "exit_z":     round(exit_z, 3),
        "duration":   exit_idx - entry_idx,
        "pnl":        round((leg_long.E + leg_short.E) - start_equity, 2),
        "long_ath":   leg_long.ath_breaks,
        "short_ath":  leg_short.ath_breaks,
        "long_fees":  round(leg_long.total_fees, 2),
        "short_fees": round(leg_short.total_fees, 2),
        "exit_reason": reason,
    })


# ─────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────
def clr(v, pos_good=True):
    import sys
    if not sys.stdout.isatty():
        return str(v)
    good = (v > 0) == pos_good
    code = "92" if good else "91"
    return f"\033[{code}m{v}\033[0m"


def print_results(r: dict):
    if not r:
        return
    W = 65
    pair = f"{r['sym_a'].split('/')[0]} / {r['sym_b'].split('/')[0]}"

    print("\n" + "=" * W)
    print(
        f"  PLANLEG PAIR BACKTEST  {pair}  |  {r['timeframe']}  {r['days']}d")
    print(
        f"  Capital/leg ${r['capital_per_leg']:,.0f}  ×  {r['leverage']}× leverage (L_ATH)")
    print("=" * W)
    print(f"  Correlation (ρ)         : {r['correlation']}")
    print(f"  Total return            : {clr(r['total_return_pct'])}%")
    print(f"  Final equity            : ${r['final_equity']:,.2f}  "
          f"(started ${r['capital_per_leg']*2:,.0f})")
    print(f"  Sharpe ratio            : {clr(r['sharpe'])}")
    print(f"  Max drawdown            : {clr(r['max_drawdown_pct'], False)}%")
    print("-" * W)
    print(f"  Round-trip trades       : {r['n_trades']}")
    print(f"  Win rate                : {clr(r['win_rate'])}%")
    print(f"  Avg profit / trade      : ${clr(r['avg_pnl_usd'])}")
    print(f"  Avg hold                : {r['avg_duration_candles']} candles "
          f"({r['avg_duration_hours']}h)")
    print(
        f"  Z stops / time stops    : {r['n_z_stops']} / {r['n_time_stops']}")
    print(f"  Avg ATH breaks/trade    : {r['avg_ath_breaks']}  "
          f"(anchor resets per leg)")
    print(f"  Total fees (both legs)  : ${r['total_fees_usd']:,.2f}")
    print("=" * W)

    td = r["trades_df"]
    if not td.empty:
        print(f"\n  {'#':<4} {'EntZ':>7} {'ExtZ':>7} {'Hold':>6} "
              f"{'PnL $':>9} {'ATH_L':>6} {'ATH_S':>6} {'Fees $':>8} {'Exit'}")
        print("  " + "─" * 63)
        for i, row in td.head(20).iterrows():
            p_str = f"+${row['pnl']:.2f}" if row['pnl'] >= 0 else f"-${abs(row['pnl']):.2f}"
            flag = " ⚠" if row["exit_reason"] in (
                "z_stop", "time_stop") else ""
            fees = row["long_fees"] + row["short_fees"]
            print(
                f"  {i+1:<4} {row['entry_z']:>+7.3f} {row['exit_z']:>+7.3f} "
                f"{row['duration']:>6} {p_str:>9} "
                f"{row['long_ath']:>6} {row['short_ath']:>6} "
                f"${fees:>7.2f} {row['exit_reason']}{flag}"
            )
        if len(td) > 20:
            print(f"  … {len(td)-20} more trades")

    curve = r["equity_curve"]
    if len(curve):
        mn, mx = curve.min(), curve.max()
        bars = " ▁▂▃▄▅▆▇█"
        W2 = 58
        step = max(1, len(curve) // W2)
        samp = curve[::step]
        spark = "".join(bars[int((v-mn)/(mx-mn+1e-9)*(len(bars)-1))]
                        for v in samp)
        print(f"\n  Equity: {spark}")
        print(f"  ${curve[0]:,.0f} → ${curve[-1]:,.0f}\n")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--a",         default="BNB/USDT")
    p.add_argument("--b",         default="BTC/USDT")
    p.add_argument("--timeframe", default="4h")
    p.add_argument("--days",      type=int,   default=365)
    p.add_argument("--z-window",  type=int,   default=60,   dest="z_window")
    p.add_argument("--z-entry",   type=float, default=2.0,  dest="z_entry")
    p.add_argument("--z-exit",    type=float, default=0.3,  dest="z_exit")
    p.add_argument("--z-stop",    type=float, default=4.0,  dest="z_stop")
    p.add_argument("--max-hold",  type=int,   default=60,   dest="max_hold")
    p.add_argument("--leverage",  type=float, default=5.0)
    p.add_argument("--capital",   type=float, default=1000.0)
    p.add_argument("--fee",       type=float, default=0.0004, dest="fee_rate")
    p.add_argument("--pup",       type=float, default=0.006)
    p.add_argument("--tau",       type=float, default=0.1,  dest="planTau")
    p.add_argument("--gamma",     type=float, default=2.0,  dest="planGamma")
    a = p.parse_args()
    return dict(
        sym_a=a.a, sym_b=a.b, timeframe=a.timeframe, days=a.days,
        z_window=a.z_window, z_entry=a.z_entry, z_exit=a.z_exit,
        z_stop=a.z_stop, max_hold=a.max_hold,
        leverage=a.leverage, capital=a.capital,
        fee_rate=a.fee_rate, slippage=0.0001, funding_8h=0.0003,
        pUp=a.pup, planTau=a.planTau, planGamma=a.planGamma,
        L_max=15.0,
        pUpRange=[0.004, 0.010], pDnRange=[-0.010, -0.004],
    )


if __name__ == "__main__":
    cfg = parse_args()
    print(f"\n  PlanLeg pair backtest: {cfg['sym_a']} / {cfg['sym_b']}")
    print(f"  z_entry={cfg['z_entry']}  z_exit={cfg['z_exit']}  "
          f"z_stop={cfg['z_stop']}  max_hold={cfg['max_hold']}c  "
          f"L_ATH={cfg['leverage']}  pUp={cfg['pUp']*100:.2f}%\n")
    r = run_backtest(cfg)
    print_results(r)
