#!/usr/bin/env python3
"""
Corrected Pair Z-Score Backtester — Binance Futures
====================================================
Fixes vs original:
  1. Exit at z_exit=0.3 (not 0.0) — realistic partial reversion target
  2. Stop loss at z_stop=4.5 — prevents infinite drawdown
  3. Fee counted once per round trip (not on every diff() step)
  4. Returns are leveraged (notional = capital × leverage)
  5. Z-window widened to 60 candles (10 days on 4h) — more stable mean/std
  6. Per-trade stats: duration, PnL, drawdown
  7. Sharpe, max drawdown, win rate, avg hold time reported
  8. Works on any pair — pass symbols as CLI args

Usage:
    python pair_backtest_fixed.py
    python pair_backtest_fixed.py --a ETH/USDT --b LINK/USDT
    python pair_backtest_fixed.py --a BNB/USDT --b BTC/USDT --leverage 8 --z-entry 1.8
    python pair_backtest_fixed.py --a SOL/USDT --b BNB/USDT --timeframe 1h --days 180
"""

import argparse
from datetime import datetime, timedelta

import ccxt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
#  DEFAULTS
# ─────────────────────────────────────────────
DEFAULTS = dict(
    sym_a="BNB/USDT",
    sym_b="BTC/USDT",
    timeframe="4h",
    days=365,
    z_window=60,      # FIX 5: was 20 (only 3 days), now 60 (10 days on 4h)
    z_entry=2.0,
    z_exit=0.3,     # FIX 1: was 0.0 — realistic convergence target
    z_stop=4.5,     # FIX 3: was missing entirely
    fee_rate=0.0004,  # 0.04% taker (BNB discount tier)
    leverage=5.0,     # FIX 4: was unlevered
    capital=1000.0,  # USD per leg
    slippage=0.0001,  # 1bps slippage estimate per fill
)

exchange = ccxt.binance(
    {"options": {"defaultType": "future"}, "enableRateLimit": True})


# ─────────────────────────────────────────────
#  DATA
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
#  BACKTEST ENGINE
# ─────────────────────────────────────────────
def run_backtest(cfg: dict) -> dict:
    pa = fetch(cfg["sym_a"], cfg["days"], cfg["timeframe"])
    pb = fetch(cfg["sym_b"], cfg["days"], cfg["timeframe"])

    df = pd.DataFrame({"A": pa, "B": pb}).dropna()
    if len(df) < cfg["z_window"] + 10:
        print("  Not enough data.")
        return {}

    # ── Spread & z-score ─────────────────────────────────
    df["log_A"] = np.log(df["A"])
    df["log_B"] = np.log(df["B"])
    df["spread"] = df["log_A"] - df["log_B"]
    df["roll_mean"] = df["spread"].rolling(cfg["z_window"]).mean()
    df["roll_std"] = df["spread"].rolling(cfg["z_window"]).std()
    df["zscore"] = (df["spread"] - df["roll_mean"]) / df["roll_std"]
    df.dropna(inplace=True)

    # ── Per-candle log returns ────────────────────────────
    df["ret_A"] = df["log_A"].diff()
    df["ret_B"] = df["log_B"].diff()
    df.dropna(inplace=True)

    # ── State machine ─────────────────────────────────────
    # position: 0=flat, +1=long spread (long A short B), -1=short spread
    capital = cfg["capital"]
    leverage = cfg["leverage"]
    notional = capital * leverage          # per leg
    fee_rt = cfg["fee_rate"] + cfg["slippage"]
    round_trip_fee = notional * fee_rt * 2 * 2  # 2 legs × 2 (entry+exit)

    position = 0
    entry_z = 0.0
    entry_idx = 0
    equity = capital * 2   # total starting equity (both legs)
    equity_curve = []
    trades = []
    peak_eq = equity

    for i in range(len(df)):
        z = df["zscore"].iloc[i]
        ret_a = df["ret_A"].iloc[i] if i > 0 else 0.0
        ret_b = df["ret_B"].iloc[i] if i > 0 else 0.0

        # Mark-to-market while in trade
        if position != 0:
            # positive = spread moved our way
            spread_ret = (ret_a - ret_b) * position
            pnl_candle = spread_ret * notional         # leveraged P&L this candle
            equity += pnl_candle

        equity_curve.append(equity)
        peak_eq = max(peak_eq, equity)

        # ── Entry ────────────────────────────────────────
        if position == 0:
            if z < -cfg["z_entry"]:
                position = +1          # long spread: A cheap vs B
                entry_z = z
                entry_idx = i
                entry_eq = equity
                # FIX 2: entry fee (both legs once)
                equity -= notional * fee_rt * 2
            elif z > cfg["z_entry"]:
                position = -1          # short spread: A rich vs B
                entry_z = z
                entry_idx = i
                entry_eq = equity
                equity -= notional * fee_rt * 2

        # ── Exit: target ─────────────────────────────────
        # FIX 1: exit when spread reverts to z_exit, not zero
        elif position == +1 and z >= -cfg["z_exit"]:
            equity -= notional * fee_rt * 2          # FIX 2: exit fee once
            trades.append({
                "entry_z": round(entry_z, 3),
                "exit_z":  round(z, 3),
                "duration": i - entry_idx,
                "pnl":     round(equity - entry_eq, 4),
                "exit_reason": "target",
            })
            position = 0

        elif position == -1 and z <= cfg["z_exit"]:
            equity -= notional * fee_rt * 2
            trades.append({
                "entry_z": round(entry_z, 3),
                "exit_z":  round(z, 3),
                "duration": i - entry_idx,
                "pnl":     round(equity - entry_eq, 4),
                "exit_reason": "target",
            })
            position = 0

        # ── Exit: stop loss ───────────────────────────────
        # FIX 3: hard stop prevents blow-up trades
        elif abs(z) >= cfg["z_stop"]:
            equity -= notional * fee_rt * 2
            trades.append({
                "entry_z": round(entry_z, 3),
                "exit_z":  round(z, 3),
                "duration": i - entry_idx,
                "pnl":     round(equity - entry_eq, 4),
                "exit_reason": "stop",
            })
            position = 0

    # Close any open position at end
    if position != 0:
        equity -= notional * fee_rt * 2
        trades.append({
            "entry_z": round(entry_z, 3),
            "exit_z":  round(df["zscore"].iloc[-1], 3),
            "duration": len(df) - 1 - entry_idx,
            "pnl":     round(equity - entry_eq, 4),
            "exit_reason": "eod",
        })

    # ── Statistics ───────────────────────────────────────
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max) / running_max
    max_dd = drawdowns.min()

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_z", "exit_z", "duration", "pnl", "exit_reason"])

    total_ret = (equity - capital * 2) / (capital * 2) * 100
    n_trades = len(trades_df)
    n_wins = (trades_df["pnl"] > 0).sum() if n_trades else 0
    win_rate = n_wins / n_trades * 100 if n_trades else 0
    avg_pnl = trades_df["pnl"].mean() if n_trades else 0
    avg_dur = trades_df["duration"].mean() if n_trades else 0
    n_stops = (trades_df["exit_reason"] == "stop").sum() if n_trades else 0
    total_fees = n_trades * notional * fee_rt * 2 * 2

    # Daily returns for Sharpe
    candles_per_day = {"1h": 24, "4h": 6, "1d": 1}.get(cfg["timeframe"], 6)
    eq_df = pd.Series(equity_curve)
    daily_ret = eq_df.pct_change().dropna()
    # Annualise
    ann_factor = np.sqrt(candles_per_day * 365)
    sharpe = (daily_ret.mean() / daily_ret.std() * ann_factor
              if daily_ret.std() > 0 else 0)

    # Correlation check
    corr = df["ret_A"].corr(df["ret_B"])

    return {
        "sym_a": cfg["sym_a"], "sym_b": cfg["sym_b"],
        "timeframe": cfg["timeframe"], "days": cfg["days"],
        "leverage": leverage, "capital_per_leg": capital,
        "notional": notional,
        "total_return_pct": round(total_ret, 2),
        "final_equity": round(equity, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 1),
        "avg_pnl_usd": round(avg_pnl, 2),
        "avg_duration_candles": round(avg_dur, 1),
        "avg_duration_hours": round(avg_dur * {"1h": 1, "4h": 4, "1d": 24}.get(cfg["timeframe"], 4), 1),
        "n_stops": n_stops,
        "stop_rate": round(n_stops / n_trades * 100, 1) if n_trades else 0,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2),
        "total_fees_usd": round(total_fees, 2),
        "correlation": round(corr, 4),
        "trades_df": trades_df,
        "equity_curve": equity_arr,
    }


# ─────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────
def print_results(r: dict):
    if not r:
        return

    W = 60
    pair = f"{r['sym_a'].split('/')[0]} / {r['sym_b'].split('/')[0]}"
    ret = r["total_return_pct"]
    dd = r["max_drawdown_pct"]

    def clr(v, good_positive=True):
        """ANSI green/red."""
        import sys
        if not sys.stdout.isatty():
            return str(v)
        code = "92" if (v > 0) == good_positive else "91"
        return f"\033[{code}m{v}\033[0m"

    print("\n" + "=" * W)
    print(f"  BACKTEST  {pair}  |  {r['timeframe']}  {r['days']}d")
    print(
        f"  Capital/leg ${r['capital_per_leg']:,.0f}  ×  {r['leverage']}× leverage")
    print(f"  Notional/leg ${r['notional']:,.0f}")
    print("=" * W)
    print(f"  Pair correlation (ρ)    : {r['correlation']}")
    print(f"  Total return            : {clr(ret)}%")
    print(
        f"  Final equity            : ${r['final_equity']:,.2f}  (started ${r['capital_per_leg']*2:,.0f})")
    print(f"  Sharpe ratio            : {clr(r['sharpe'])}")
    print(f"  Max drawdown            : {clr(dd, good_positive=False)}%")
    print("-" * W)
    print(f"  Round-trip trades       : {r['n_trades']}")
    print(f"  Win rate                : {clr(r['win_rate'])}%")
    print(f"  Avg profit / trade      : ${clr(r['avg_pnl_usd'])}")
    print(
        f"  Avg hold time           : {r['avg_duration_candles']} candles  ({r['avg_duration_hours']}h)")
    print(
        f"  Stop-outs               : {r['n_stops']}  ({r['stop_rate']}% of trades)")
    print(f"  Total fees paid         : ${r['total_fees_usd']:,.2f}")
    print("=" * W)

    # Per-trade breakdown (first 10)
    td = r["trades_df"]
    if not td.empty:
        print(
            f"\n  {'#':<4} {'Entry Z':>9} {'Exit Z':>9} {'Hold(c)':>9} {'PnL $':>10} {'Exit'}")
        print("  " + "─" * 52)
        for i, row in td.head(15).iterrows():
            pnl_str = f"+${row['pnl']:.2f}" if row['pnl'] >= 0 else f"-${abs(row['pnl']):.2f}"
            stop_flag = " ⚠" if row["exit_reason"] == "stop" else ""
            print(f"  {i+1:<4} {row['entry_z']:>+9.3f} {row['exit_z']:>+9.3f} "
                  f"{row['duration']:>9} {pnl_str:>10} {row['exit_reason']}{stop_flag}")

        if len(td) > 15:
            print(f"  … {len(td) - 15} more trades")

    print()

    # Equity curve sparkline (ASCII)
    curve = r["equity_curve"]
    if len(curve) > 0:
        mn, mx = curve.min(), curve.max()
        W2 = 56
        step = max(1, len(curve) // W2)
        sampled = curve[::step]
        bars = " ▁▂▃▄▅▆▇█"
        spark = ""
        for v in sampled:
            idx = int((v - mn) / (mx - mn + 1e-9) * (len(bars) - 1))
            spark += bars[idx]
        print(f"  Equity curve: {spark}")
        print(f"  Start ${curve[0]:,.0f}  →  End ${curve[-1]:,.0f}\n")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def parse_args() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--a",         default=DEFAULTS["sym_a"])
    p.add_argument("--b",         default=DEFAULTS["sym_b"])
    p.add_argument("--timeframe", default=DEFAULTS["timeframe"])
    p.add_argument("--days",      type=int,   default=DEFAULTS["days"])
    p.add_argument("--z-window",  type=int,
                   default=DEFAULTS["z_window"],   dest="z_window")
    p.add_argument("--z-entry",   type=float,
                   default=DEFAULTS["z_entry"],    dest="z_entry")
    p.add_argument("--z-exit",    type=float,
                   default=DEFAULTS["z_exit"],     dest="z_exit")
    p.add_argument("--z-stop",    type=float,
                   default=DEFAULTS["z_stop"],     dest="z_stop")
    p.add_argument("--fee",       type=float,
                   default=DEFAULTS["fee_rate"],   dest="fee_rate")
    p.add_argument("--leverage",  type=float, default=DEFAULTS["leverage"])
    p.add_argument("--capital",   type=float, default=DEFAULTS["capital"])
    p.add_argument("--slippage",  type=float, default=DEFAULTS["slippage"])
    args = p.parse_args()
    cfg = {**DEFAULTS}
    cfg.update({
        "sym_a":     args.a,
        "sym_b":     args.b,
        "timeframe": args.timeframe,
        "days":      args.days,
        "z_window":  args.z_window,
        "z_entry":   args.z_entry,
        "z_exit":    args.z_exit,
        "z_stop":    args.z_stop,
        "fee_rate":  args.fee_rate,
        "leverage":  args.leverage,
        "capital":   args.capital,
        "slippage":  args.slippage,
    })
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    print(f"\n  Pair backtest: {cfg['sym_a']} / {cfg['sym_b']}")
    print(f"  z_entry={cfg['z_entry']}  z_exit={cfg['z_exit']}  "
          f"z_stop={cfg['z_stop']}  window={cfg['z_window']}c  "
          f"leverage={cfg['leverage']}×\n")
    results = run_backtest(cfg)
    print_results(results)
