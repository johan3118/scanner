#!/usr/bin/env python3
"""
Advanced Pair Cointegration & Divergence Scanner — Binance Futures
==================================================================
Scans active USDT-margined futures asynchronously, builds a correlation matrix,
calculates OLS hedge ratios (beta), tests for cointegration (ADF test), 
and ranks stationary diverging pairs by spread z-score.

Usage:
    python correlation_scanner.py               # single scan, print table
    python correlation_scanner.py --loop        # repeat every N minutes
    python correlation_scanner.py --max-p 0.05  # strict cointegration (95% confidence)
"""

import argparse
import asyncio
import os
import sys
import time
import warnings
from datetime import datetime
from itertools import combinations

import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller

# Suppress harmless statsmodels warnings for clean terminal output
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG DEFAULTS  (override via CLI flags)
# ─────────────────────────────────────────────
DEFAULTS = dict(
    timeframe="1h",       # 1h is generally better for statistical arb than 1d
    lookback=300,         # Need more candles for a valid cointegration test
    zscore_window=20,     # rolling window for spread z-score
    min_corr=0.75,        # minimum Pearson ρ to consider a pair
    # max ADF p-value for cointegration (0.10 = 90% confidence)
    max_pvalue=0.10,
    min_zscore=1.5,       # minimum |z| to flag as diverging
    top_n=30,             # pairs to display
    scan_interval=5,      # minutes between scans (--loop mode)
    min_volume_usdt=5_000_000,  # 24h volume filter
    export=False,
    loop=False,
)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def color(text, code):
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


def RED(t): return color(t, "91")
def GREEN(t): return color(t, "92")
def YELLOW(t): return color(t, "93")
def BOLD(t): return color(t, "1")
def CYAN(t): return color(t, "96")

# ─────────────────────────────────────────────
#  STEP 1 — fetch tickers & filter by volume (ASYNC)
# ─────────────────────────────────────────────


async def get_liquid_symbols(exchange, min_volume_usdt: float) -> list[str]:
    print(f"  [{ts()}] Fetching tickers…", end="\r")
    tickers = await exchange.fetch_tickers()
    symbols = []
    for sym, t in tickers.items():
        if not sym.endswith(":USDT"):
            continue
        vol = t.get("quoteVolume") or 0
        if vol >= min_volume_usdt:
            symbols.append(sym)
    print(f"  [{ts()}] {len(symbols)} liquid USDT-perp symbols found.        ")
    return sorted(symbols)

# ─────────────────────────────────────────────
#  STEP 2 — fetch OHLCV concurrently (ASYNC)
# ─────────────────────────────────────────────


async def fetch_ohlcv_safe(exchange, sym, timeframe, limit, semaphore):
    """Fetch OHLCV for a single symbol, bounded by a concurrency semaphore."""
    async with semaphore:
        try:
            ohlcv = await exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=limit)
            return sym, ohlcv
        except Exception:
            return sym, None


async def fetch_close_matrix(exchange, symbols: list[str], timeframe: str, lookback: int) -> pd.DataFrame:
    limit = lookback + 5
    semaphore = asyncio.Semaphore(15)  # Prevent Binance rate limits

    print(
        f"  [{ts()}] Concurrently fetching {len(symbols)} OHLCV histories...", end="\r")
    tasks = [fetch_ohlcv_safe(exchange, sym, timeframe,
                              limit, semaphore) for sym in symbols]
    results = await asyncio.gather(*tasks)

    closes = {}
    for sym, ohlcv in results:
        if ohlcv and len(ohlcv) >= lookback:
            df_tmp = pd.DataFrame(
                ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
            closes[sym] = df_tmp["close"].values[-lookback:]

    print(
        f"  [{ts()}] Price matrix built: {len(closes)} symbols × {lookback} candles.   ")
    if not closes:
        return pd.DataFrame()

    min_len = min(len(v) for v in closes.values())
    closes = {k: v[-min_len:] for k, v in closes.items()}
    return pd.DataFrame(closes)

# ─────────────────────────────────────────────
#  STEP 3 — compute log-return correlation matrix
# ─────────────────────────────────────────────


def build_correlation_matrix(price_df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(price_df / price_df.shift(1)).dropna()
    return log_ret.corr(method="pearson")


def find_correlated_pairs(corr_matrix: pd.DataFrame, min_corr: float) -> list[tuple]:
    pairs = []
    syms = corr_matrix.columns.tolist()
    for a, b in combinations(syms, 2):
        rho = corr_matrix.loc[a, b]
        if abs(rho) >= min_corr:
            pairs.append((a, b, round(rho, 4)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs

# ─────────────────────────────────────────────
#  STEP 4 — Cointegration, Beta, & Z-Score
# ─────────────────────────────────────────────


def compute_arb_metrics(
    price_df: pd.DataFrame,
    pairs:    list[tuple],
    window:   int,
    max_pvalue: float
) -> list[dict]:
    log_prices = np.log(price_df)
    results = []

    for sym_a, sym_b, rho in pairs:
        if sym_a not in log_prices or sym_b not in log_prices:
            continue

        Y = log_prices[sym_a]
        X = log_prices[sym_b]

        # 1. Ordinary Least Squares to find Hedge Ratio (Beta)
        X_const = sm.add_constant(X)
        try:
            model = sm.OLS(Y, X_const).fit()
            beta = model.params.iloc[1]
        except Exception:
            continue

        # Spread = log(A) - (beta * log(B))
        spread = Y - (beta * X)

        # 2. Augmented Dickey-Fuller Test for Cointegration
        try:
            adf_result = adfuller(spread)
            p_value = adf_result[1]
        except Exception:
            continue

        # Skip if the spread is not statistically stationary
        if p_value > max_pvalue:
            continue

        # 3. Rolling Z-Score of the Spread
        if len(spread) < window:
            continue

        roll_mean = spread.rolling(window).mean()
        roll_std = spread.rolling(window).std()

        current_spread = spread.iloc[-1]
        mean_now = roll_mean.iloc[-1]
        std_now = roll_std.iloc[-1]

        if std_now == 0 or np.isnan(std_now):
            continue

        z = (current_spread - mean_now) / std_now

        # 4. Ornstein-Uhlenbeck half-life (Mean Reversion Speed)
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

        # Trade Direction
        leg_a, leg_b = sym_a.split("/")[0], sym_b.split("/")[0]
        if z > 0:
            trade_note = f"SHORT {leg_a} / LONG {leg_b}"
        else:
            trade_note = f"LONG {leg_a} / SHORT {leg_b}"

        pct_a = round((price_df[sym_a].iloc[-1] /
                      price_df[sym_a].iloc[-2] - 1) * 100, 2)
        pct_b = round((price_df[sym_b].iloc[-1] /
                      price_df[sym_b].iloc[-2] - 1) * 100, 2)

        results.append({
            "Pair":        f"{leg_a} / {leg_b}",
            "ρ":           rho,
            "Beta":        round(beta, 3),
            "Coin-P":      round(p_value, 3),
            "Z-score":     round(z, 3),
            "|Z|":         round(abs(z), 3),
            "Half-life":   half_life,
            "Δ A (1c)":    pct_a,
            "Δ B (1c)":    pct_b,
            "Direction":   trade_note,
        })

    results.sort(key=lambda x: x["|Z|"], reverse=True)
    return results

# ─────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────


def print_results(results: list[dict], min_zscore: float, top_n: int, cfg: dict):
    clear()
    now = datetime.now().strftime("%H:%M:%S")

    print(BOLD("=" * 115))
    print(BOLD(f"  COINTEGRATION & DIVERGENCE SCANNER   [{now}]"))
    print(f"  corr ≥ {cfg['min_corr']}  |  p-val ≤ {cfg['max_pvalue']}  |  |z| ≥ {cfg['min_zscore']}  |  "
          f"TF={cfg['timeframe']}  lookback={cfg['lookback']}c  z-window={cfg['zscore_window']}c")
    print(BOLD("=" * 115))

    flagged = [r for r in results if r["|Z|"] >= min_zscore]

    if not flagged:
        print(
            YELLOW("\n  No cointegrated diverging pairs found under current thresholds.\n"))
        return

    display = flagged[:top_n]

    hdr = (
        f"  {'Pair':<18} {'ρ':>6} {'Beta':>6} {'P-val':>6} {'Z-score':>9} {'HL(c)':>7} "
        f"{'ΔA%':>7} {'ΔB%':>7}   Direction"
    )
    print(CYAN(hdr))
    print(CYAN("  " + "─" * 111))

    for r in display:
        z = r["Z-score"]
        az = r["|Z|"]
        hl = str(r["Half-life"]) if r["Half-life"] else "—"
        da_str = GREEN(
            f"{r['Δ A (1c)']:>+7.2f}") if r["Δ A (1c)"] > 0 else RED(f"{r['Δ A (1c)']:>+7.2f}")
        db_str = GREEN(
            f"{r['Δ B (1c)']:>+7.2f}") if r["Δ B (1c)"] > 0 else RED(f"{r['Δ B (1c)']:>+7.2f}")

        if az >= 3.0:
            zstr = RED(f"{z:>+9.3f}")
        elif az >= 2.0:
            zstr = YELLOW(f"{z:>+9.3f}")
        else:
            zstr = f"{z:>+9.3f}"

        print(
            f"  {r['Pair']:<18} {r['ρ']:>6.3f} {r['Beta']:>6.3f} {r['Coin-P']:>6.3f} "
            f"{zstr} {hl:>7} {da_str} {db_str}   {r['Direction']}"
        )

    print(CYAN("\n  " + "─" * 111))
    print(f"  Stationary Pairs: {BOLD(str(len(results)))}  |  "
          f"Diverging (|z|≥{min_zscore}): {BOLD(str(len(flagged)))}  |  "
          f"Showing top {min(top_n, len(flagged))}")

    if cfg.get("loop"):
        print(
            f"\n  Next scan in {cfg['scan_interval']} min…  (Ctrl+C to stop)")
    print()


def export_csv(results: list[dict]):
    fname = f"divergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df = pd.DataFrame(results)
    df.to_csv(fname, index=False)
    print(f"  Exported → {fname}")

# ─────────────────────────────────────────────
#  MAIN ASYNC CYCLE
# ─────────────────────────────────────────────


async def scan_once(exchange, cfg: dict) -> list[dict]:
    print(f"\n[{ts()}] Starting async scan…")

    symbols = await get_liquid_symbols(exchange, cfg["min_volume_usdt"])
    if not symbols:
        return []

    price_df = await fetch_close_matrix(exchange, symbols, cfg["timeframe"], cfg["lookback"])
    if price_df.empty:
        return []

    print(
        f"  [{ts()}] Computing correlation matrix ({len(price_df.columns)} symbols)…")
    corr_matrix = build_correlation_matrix(price_df)

    pairs = find_correlated_pairs(corr_matrix, cfg["min_corr"])
    print(f"  [{ts()}] {len(pairs)} pairs with ρ ≥ {cfg['min_corr']}")

    if not pairs:
        return []

    print(f"  [{ts()}] Testing for cointegration & computing Z-scores…")
    results = compute_arb_metrics(
        price_df, pairs, cfg["zscore_window"], cfg["max_pvalue"])

    return results


async def main_loop(cfg: dict):
    exchange = ccxt_async.binance({
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    })

    try:
        if cfg["loop"]:
            while True:
                results = await scan_once(exchange, cfg)
                print_results(results, cfg["min_zscore"], cfg["top_n"], cfg)
                if cfg["export"] and results:
                    export_csv(results)
                await asyncio.sleep(cfg["scan_interval"] * 60)
        else:
            results = await scan_once(exchange, cfg)
            print_results(results, cfg["min_zscore"], cfg["top_n"], cfg)
            if cfg["export"] and results:
                export_csv(results)
    except KeyboardInterrupt:
        print(RED("\n  Stopped by user."))
    finally:
        await exchange.close()

# ─────────────────────────────────────────────
#  CLI ENTRY
# ─────────────────────────────────────────────


def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="Advanced Cointegration & Divergence Scanner")
    p.add_argument("--timeframe",   default=DEFAULTS["timeframe"])
    p.add_argument("--lookback",    type=int, default=DEFAULTS["lookback"])
    p.add_argument("--zscore-window", type=int,
                   default=DEFAULTS["zscore_window"])
    p.add_argument("--min-corr",    type=float, default=DEFAULTS["min_corr"])
    p.add_argument("--max-pvalue",  type=float,
                   default=DEFAULTS["max_pvalue"], dest="max_pvalue", help="Max ADF p-value (e.g. 0.05)")
    p.add_argument("--min-zscore",  type=float, default=DEFAULTS["min_zscore"])
    p.add_argument("--top",         type=int,
                   default=DEFAULTS["top_n"], dest="top_n")
    p.add_argument("--min-vol",     type=float,
                   default=DEFAULTS["min_volume_usdt"], dest="min_volume_usdt")
    p.add_argument("--interval",    type=int,
                   default=DEFAULTS["scan_interval"], dest="scan_interval")
    p.add_argument("--loop",        action="store_true")
    p.add_argument("--export",      action="store_true")
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = {**DEFAULTS, **parse_args()}

    # Run the asyncio event loop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main_loop(cfg))
    except KeyboardInterrupt:
        print("\nExited.")
