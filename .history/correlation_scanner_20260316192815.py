#!/usr/bin/env python3
"""
Pair Correlation & Divergence Scanner — Binance Futures
========================================================
Scans ALL active USDT-margined futures, builds a correlation matrix,
detects pairs that are historically correlated but currently diverging,
and ranks them by spread z-score (how many σ away from mean).

Usage:
    python correlation_scanner.py               # single scan, print table
    python correlation_scanner.py --loop        # repeat every N minutes
    python correlation_scanner.py --top 20      # show top 20 pairs
    python correlation_scanner.py --min-corr 0.80  # stricter correlation filter
    python correlation_scanner.py --timeframe 4h   # use 4h candles
    python correlation_scanner.py --export         # save CSV on each scan
"""

import argparse
import os
import sys
import time
from datetime import datetime
from itertools import combinations

import ccxt
import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────
#  CONFIG DEFAULTS  (override via CLI flags)
# ─────────────────────────────────────────────
DEFAULTS = dict(
    timeframe="1d",       # candle size: 1m 5m 15m 1h 4h 1d
    lookback=30,         # candles for correlation window
    zscore_window=20,         # rolling window for spread z-score
    min_corr=0.75,       # minimum Pearson ρ to consider a pair
    min_zscore=1.5,        # minimum |z| to flag as diverging
    top_n=30,         # pairs to display
    scan_interval=5,          # minutes between scans (--loop mode)
    min_volume_usdt=1_000_000,  # 24h volume filter (skip illiquid coins)
    export=False,
    loop=False,
)


# ─────────────────────────────────────────────
#  EXCHANGE INIT
# ─────────────────────────────────────────────
exchange = ccxt.binance({
    "options": {"defaultType": "future"},
    "enableRateLimit": True,
})


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def clear():
    os.system("cls" if os.name == "nt" else "clear")


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def color(text, code):
    """ANSI color — falls back to plain text if terminal doesn't support it."""
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


def RED(t): return color(t, "91")
def GREEN(t): return color(t, "92")
def YELLOW(t): return color(t, "93")


def CYAN(t): return color(t, "96")
def BOLD(t): return color(t, "1")


# ─────────────────────────────────────────────
#  STEP 1 — fetch tickers & filter by volume
# ─────────────────────────────────────────────
def get_liquid_symbols(min_volume_usdt: float) -> list[str]:
    """Return USDT-perp symbols with sufficient 24h quote volume."""
    print(f"  [{ts()}] Fetching tickers…", end="\r")
    tickers = exchange.fetch_tickers()
    symbols = []
    for sym, t in tickers.items():
        if not sym.endswith(":USDT"):
            continue
        vol = t.get("quoteVolume") or 0
        if vol >= min_volume_usdt:
            symbols.append(sym)
    print(f"  [{ts()}] {len(symbols)} liquid USDT-perp symbols found.         ")
    return sorted(symbols)


# ─────────────────────────────────────────────
#  STEP 2 — fetch OHLCV & build returns matrix
# ─────────────────────────────────────────────
def fetch_close_matrix(symbols: list[str], timeframe: str, lookback: int) -> pd.DataFrame:
    """
    Returns a DataFrame of shape (lookback, n_symbols) with close prices.
    Symbols that fail to fetch are dropped silently.
    """
    closes = {}
    limit = lookback + 5            # small buffer in case of missing candles
    total = len(symbols)

    for i, sym in enumerate(symbols):
        print(f"  [{ts()}] Fetching {sym} ({i+1}/{total})…   ", end="\r")
        try:
            ohlcv = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            if len(ohlcv) < lookback:
                continue
            df_tmp = pd.DataFrame(
                ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
            closes[sym] = df_tmp["close"].values[-lookback:]
        except Exception:
            continue

    print(
        f"  [{ts()}] Price matrix built: {len(closes)} symbols × {lookback} candles.   ")
    if not closes:
        return pd.DataFrame()

    # Align all series to same length (drop any that are shorter)
    min_len = min(len(v) for v in closes.values())
    closes = {k: v[-min_len:] for k, v in closes.items()}
    return pd.DataFrame(closes)


# ─────────────────────────────────────────────
#  STEP 3 — compute log-return correlation matrix
# ─────────────────────────────────────────────
def build_correlation_matrix(price_df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(price_df / price_df.shift(1)).dropna()
    return log_ret.corr(method="pearson")


# ─────────────────────────────────────────────
#  STEP 4 — find correlated pairs
# ─────────────────────────────────────────────
def find_correlated_pairs(corr_matrix: pd.DataFrame, min_corr: float) -> list[tuple]:
    """Return list of (symA, symB, rho) above threshold, sorted by |rho| desc."""
    pairs = []
    syms = corr_matrix.columns.tolist()
    for a, b in combinations(syms, 2):
        rho = corr_matrix.loc[a, b]
        if abs(rho) >= min_corr:
            pairs.append((a, b, round(rho, 4)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


# ─────────────────────────────────────────────
#  STEP 5 — compute spread z-score for each pair
# ─────────────────────────────────────────────
def compute_spread_zscore(
    price_df: pd.DataFrame,
    pairs:    list[tuple],
    window:   int,
) -> list[dict]:
    """
    For each correlated pair compute:
      spread      = log(priceA) - log(priceB)    (log-price spread)
      z_score     = (spread_now - mean) / std     over rolling `window`
      half_life   = Ornstein-Uhlenbeck half-life  (mean reversion speed)
      direction   = which leg is currently rich / cheap
    """
    log_prices = np.log(price_df)
    results = []

    for sym_a, sym_b, rho in pairs:
        if sym_a not in log_prices or sym_b not in log_prices:
            continue

        spread = log_prices[sym_a] - log_prices[sym_b]
        if len(spread) < window:
            continue

        # Rolling stats on the last `window` candles
        roll_mean = spread.rolling(window).mean()
        roll_std = spread.rolling(window).std()

        current_spread = spread.iloc[-1]
        mean_now = roll_mean.iloc[-1]
        std_now = roll_std.iloc[-1]

        if std_now == 0 or np.isnan(std_now):
            continue

        z = (current_spread - mean_now) / std_now

        # Ornstein-Uhlenbeck half-life
        # lag-1 regression: Δspread = a + b*spread_lag + ε
        # half_life = -ln(2) / b
        half_life = None
        try:
            spread_vals = spread.values
            lag = spread_vals[:-1]
            delta = np.diff(spread_vals)
            if len(lag) >= 5:
                b, _, _, _, _ = stats.linregress(lag, delta)
                if b < 0:
                    half_life = round(-np.log(2) / b, 1)
        except Exception:
            pass

        # Which direction to trade
        if z > 0:
            rich, cheap = sym_a.split("/")[0], sym_b.split("/")[0]
            trade_note = f"SHORT {rich} / LONG {cheap}"
        else:
            rich, cheap = sym_b.split("/")[0], sym_a.split("/")[0]
            trade_note = f"SHORT {rich} / LONG {cheap}"

        # Recent price change for context
        pct_a = round((price_df[sym_a].iloc[-1] /
                      price_df[sym_a].iloc[-2] - 1) * 100, 2)
        pct_b = round((price_df[sym_b].iloc[-1] /
                      price_df[sym_b].iloc[-2] - 1) * 100, 2)

        results.append({
            "Pair":        f"{sym_a.split('/')[0]} / {sym_b.split('/')[0]}",
            "ρ":           rho,
            "Z-score":     round(z, 3),
            "|Z|":         round(abs(z), 3),
            "Spread mean": round(float(mean_now), 5),
            "Spread now":  round(float(current_spread), 5),
            "Half-life":   half_life,
            "Δ A (1c)":    pct_a,
            "Δ B (1c)":    pct_b,
            "Direction":   trade_note,
            "_sym_a":      sym_a,
            "_sym_b":      sym_b,
        })

    # Sort by absolute z-score descending
    results.sort(key=lambda x: x["|Z|"], reverse=True)
    return results


# ─────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────
def print_results(results: list[dict], min_zscore: float, top_n: int, cfg: dict):
    clear()
    now = datetime.now().strftime("%H:%M:%S")

    print(BOLD("=" * 105))
    print(BOLD(f"  PAIR CORRELATION & DIVERGENCE SCANNER   [{now}]"))
    print(f"  corr ≥ {cfg['min_corr']}  |  |z| ≥ {cfg['min_zscore']}  |  "
          f"timeframe={cfg['timeframe']}  lookback={cfg['lookback']}c  "
          f"z-window={cfg['zscore_window']}c")
    print(BOLD("=" * 105))

    flagged = [r for r in results if r["|Z|"] >= min_zscore]

    if not flagged:
        print(YELLOW("\n  No diverging pairs found under current thresholds.\n"))
        return

    display = flagged[:top_n]

    # Header
    hdr = (
        f"  {'Pair':<26} {'ρ':>6} {'Z-score':>9} {'HL(c)':>7} "
        f"{'ΔA%':>7} {'ΔB%':>7}   Direction"
    )
    print(CYAN(hdr))
    print(CYAN("  " + "─" * 101))

    for r in display:
        z = r["Z-score"]
        az = r["|Z|"]
        hl = str(r["Half-life"]) if r["Half-life"] else "—"
        da = r["Δ A (1c)"]
        db = r["Δ B (1c)"]
        pair = r["Pair"]
        rho = r["ρ"]
        direc = r["Direction"]

        # Color z-score by intensity
        if az >= 3.0:
            zstr = RED(f"{z:>+9.3f}")
        elif az >= 2.0:
            zstr = YELLOW(f"{z:>+9.3f}")
        else:
            zstr = f"{z:>+9.3f}"

        da_str = GREEN(f"{da:>+7.2f}") if da > 0 else RED(f"{da:>+7.2f}")
        db_str = GREEN(f"{db:>+7.2f}") if db > 0 else RED(f"{db:>+7.2f}")

        print(
            f"  {pair:<26} {rho:>6.3f} {zstr} {hl:>7} "
            f"{da_str} {db_str}   {direc}"
        )

    total_corr = len(results)
    total_flag = len(flagged)
    print(CYAN("\n  " + "─" * 101))
    print(f"  Correlated pairs: {BOLD(str(total_corr))}  |  "
          f"Diverging (|z|≥{min_zscore}): {BOLD(str(total_flag))}  |  "
          f"Showing top {min(top_n, total_flag)}")

    if cfg.get("loop"):
        print(
            f"\n  Next scan in {cfg['scan_interval']} min…  (Ctrl+C to stop)")
    print()


def export_csv(results: list[dict]):
    fname = f"divergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df = pd.DataFrame(results).drop(
        columns=["_sym_a", "_sym_b"], errors="ignore")
    df.to_csv(fname, index=False)
    print(f"  Exported → {fname}")


# ─────────────────────────────────────────────
#  MAIN SCAN CYCLE
# ─────────────────────────────────────────────
def scan_once(cfg: dict) -> list[dict]:
    print(f"\n[{ts()}] Starting scan…")

    # 1. Liquid symbols
    symbols = get_liquid_symbols(cfg["min_volume_usdt"])
    if not symbols:
        print("  No symbols found. Check connectivity.")
        return []

    # 2. Price matrix
    price_df = fetch_close_matrix(symbols, cfg["timeframe"], cfg["lookback"])
    if price_df.empty:
        print("  Could not build price matrix.")
        return []

    # 3. Correlation matrix
    print(
        f"  [{ts()}] Computing correlation matrix ({len(price_df.columns)} symbols)…")
    corr_matrix = build_correlation_matrix(price_df)

    # 4. Correlated pairs
    pairs = find_correlated_pairs(corr_matrix, cfg["min_corr"])
    print(f"  [{ts()}] {len(pairs)} pairs with ρ ≥ {cfg['min_corr']}")

    if not pairs:
        print("  No correlated pairs found. Try lowering --min-corr.")
        return []

    # 5. Spread z-scores
    print(f"  [{ts()}] Computing spread z-scores…")
    results = compute_spread_zscore(price_df, pairs, cfg["zscore_window"])

    return results


def run(cfg: dict):
    if cfg["loop"]:
        while True:
            try:
                results = scan_once(cfg)
                print_results(results, cfg["min_zscore"], cfg["top_n"], cfg)
                if cfg["export"] and results:
                    export_csv(results)
                time.sleep(cfg["scan_interval"] * 60)
            except KeyboardInterrupt:
                print(RED("\n  Stopped."))
                break
            except Exception as e:
                print(f"  Error: {e}")
                time.sleep(30)
    else:
        results = scan_once(cfg)
        print_results(results, cfg["min_zscore"], cfg["top_n"], cfg)
        if cfg["export"] and results:
            export_csv(results)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="Binance Futures Pair Correlation & Divergence Scanner"
    )
    p.add_argument("--timeframe",   default=DEFAULTS["timeframe"],
                   help="Candle timeframe (default: 1d)")
    p.add_argument("--lookback",    type=int, default=DEFAULTS["lookback"],
                   help="Candles for correlation (default: 30)")
    p.add_argument("--zscore-window", type=int, default=DEFAULTS["zscore_window"],
                   help="Rolling window for z-score (default: 20)")
    p.add_argument("--min-corr",    type=float, default=DEFAULTS["min_corr"],
                   help="Min Pearson ρ (default: 0.75)")
    p.add_argument("--min-zscore",  type=float, default=DEFAULTS["min_zscore"],
                   help="Min |z| to flag divergence (default: 1.5)")
    p.add_argument("--top",         type=int, default=DEFAULTS["top_n"],
                   dest="top_n",    help="Pairs to show (default: 30)")
    p.add_argument("--min-vol",     type=float, default=DEFAULTS["min_volume_usdt"],
                   dest="min_volume_usdt",
                   help="Min 24h quote volume in USDT (default: 1_000_000)")
    p.add_argument("--interval",    type=int, default=DEFAULTS["scan_interval"],
                   dest="scan_interval",
                   help="Minutes between scans in loop mode (default: 5)")
    p.add_argument("--loop",        action="store_true",
                   help="Run continuously")
    p.add_argument("--export",      action="store_true",
                   help="Export results to CSV")
    args = p.parse_args()
    return vars(args)


if __name__ == "__main__":
    cfg = {**DEFAULTS, **parse_args()}
    run(cfg)
