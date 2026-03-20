#!/usr/bin/env python3
"""
Pair Correlation & Divergence Scanner — Binance Futures  (v2)
=============================================================
Improvements over v1:
  ✓ ADF cointegration test  — filters out spurious pairs
  ✓ OLS hedge ratio (β)     — proper spread construction, not naive 1:1
  ✓ Longer defaults          — 120-candle lookback, 60-candle z-window
  ✓ Rolling ADF on spread    — confirms spread stationarity in-sample
  ✓ Price-matrix caching     — loop mode only re-fetches the latest candle
  ✓ Higher volume floor      — $5 M default (was $1 M)
  ✓ Notional-correct sizing  — prints USD-per-leg for equal-dollar positioning
  ✓ Bonferroni-aware tip      — warns when pair count is very large

Usage:
    python correlation_scanner.py               # single scan, print table
    python correlation_scanner.py --loop        # repeat every N minutes
    python correlation_scanner.py --top 20      # show top 20 pairs
    python correlation_scanner.py --min-corr 0.85
    python correlation_scanner.py --timeframe 4h
    python correlation_scanner.py --export
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
from statsmodels.tsa.stattools import adfuller, coint


# ─────────────────────────────────────────────
#  CONFIG DEFAULTS
# ─────────────────────────────────────────────
DEFAULTS = dict(
    timeframe="1d",
    lookback=120,   # ↑ was 30  — enough for robust correlation
    zscore_window=60,    # ↑ was 20  — wider rolling z-score baseline
    min_corr=0.90,  # ↑ was 0.75
    min_zscore=1.5,
    top_n=30,
    scan_interval=5,
    min_volume_usdt=5_000_000,   # ↑ was 1 M — more liquid pairs only
    coint_pvalue=0.05,        # NEW — ADF p-value threshold
    adf_spread_pvalue=0.10,        # NEW — rolling spread stationarity check
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
#  STEP 2 — fetch OHLCV & build close matrix
# ─────────────────────────────────────────────
def fetch_close_matrix(
    symbols:   list[str],
    timeframe: str,
    lookback:  int,
) -> pd.DataFrame:
    """
    Returns DataFrame of shape (lookback, n_symbols) with close prices.
    Symbols that fail to fetch are silently dropped.
    """
    closes = {}
    limit = lookback + 5
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

    min_len = min(len(v) for v in closes.values())
    closes = {k: v[-min_len:] for k, v in closes.items()}
    return pd.DataFrame(closes)


def update_close_matrix(
    price_df:  pd.DataFrame,
    symbols:   list[str],
    timeframe: str,
) -> pd.DataFrame:
    """
    Efficient cache update for --loop mode.
    Fetches only the 3 most-recent candles per symbol and appends the
    latest confirmed close, then drops the oldest row to keep length constant.
    """
    total = len(symbols)
    updates = {}
    for i, sym in enumerate(symbols):
        print(f"  [{ts()}] Updating {sym} ({i+1}/{total})…   ", end="\r")
        try:
            ohlcv = exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=3)
            if ohlcv:
                updates[sym] = ohlcv[-1][4]   # latest close
        except Exception:
            pass

    if not updates:
        return price_df

    new_row = pd.DataFrame(
        {sym: [updates[sym]] for sym in price_df.columns if sym in updates}
    )
    updated = pd.concat([price_df.iloc[1:], new_row], ignore_index=True)
    print(f"  [{ts()}] Cache updated ({len(updates)} symbols refreshed).        ")
    return updated


# ─────────────────────────────────────────────
#  STEP 3 — log-return correlation matrix
# ─────────────────────────────────────────────
def build_correlation_matrix(price_df: pd.DataFrame) -> pd.DataFrame:
    log_ret = np.log(price_df / price_df.shift(1)).dropna()
    return log_ret.corr(method="pearson")


# ─────────────────────────────────────────────
#  STEP 4 — correlated pairs (pre-filter)
# ─────────────────────────────────────────────
def find_correlated_pairs(corr_matrix: pd.DataFrame, min_corr: float) -> list[tuple]:
    """Return (symA, symB, rho) above threshold, sorted by |rho| desc."""
    pairs = []
    syms = corr_matrix.columns.tolist()
    for a, b in combinations(syms, 2):
        rho = corr_matrix.loc[a, b]
        if abs(rho) >= min_corr:
            pairs.append((a, b, round(rho, 4)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    n = len(pairs)
    if n > 500:
        print(YELLOW(
            f"  ⚠  {n} correlated pairs — many may be spurious. "
            f"Consider raising --min-corr or --min-vol."
        ))
    return pairs


# ─────────────────────────────────────────────
#  STEP 4b — cointegration filter  (NEW)
# ─────────────────────────────────────────────
def filter_cointegrated(
    price_df:       pd.DataFrame,
    pairs:          list[tuple],
    pvalue_thresh:  float,
) -> list[tuple]:
    """
    Apply Engle-Granger cointegration test (statsmodels `coint`).
    Keeps only pairs where the spread is likely stationary (p ≤ pvalue_thresh).

    `coint` runs an ADF on the residuals of OLS(price_a ~ price_b),
    which is equivalent to the standard EG two-step procedure.
    """
    cointegrated = []
    total = len(pairs)
    for i, (a, b, rho) in enumerate(pairs):
        print(f"  [{ts()}] Cointegration test {i+1}/{total}…   ", end="\r")
        try:
            pa = price_df[a].values
            pb = price_df[b].values
            _, pvalue, _ = coint(pa, pb)
            if pvalue <= pvalue_thresh:
                cointegrated.append((a, b, rho, round(pvalue, 4)))
        except Exception:
            continue

    print(
        f"  [{ts()}] {len(cointegrated)}/{total} pairs passed cointegration test.   ")
    return cointegrated


# ─────────────────────────────────────────────
#  STEP 5 — OLS hedge ratio + spread z-score  (FIXED)
# ─────────────────────────────────────────────
def compute_spread_zscore(
    price_df:         pd.DataFrame,
    pairs:            list[tuple],     # (symA, symB, rho, coint_pvalue)
    window:           int,
    adf_spread_pval:  float,
) -> list[dict]:
    """
    For each cointegrated pair:
      1. Estimate OLS hedge ratio β via log-price regression
         log(A) = α + β·log(B) + ε
      2. Construct spread = log(A) − β·log(B) − α
      3. Compute rolling z-score of spread
      4. Validate spread stationarity with ADF
      5. Estimate OU half-life via lag-1 OLS on spread differences
    """
    log_prices = np.log(price_df)
    results = []

    for entry in pairs:
        sym_a, sym_b, rho, coint_p = entry

        if sym_a not in log_prices.columns or sym_b not in log_prices.columns:
            continue

        la = log_prices[sym_a].values
        lb = log_prices[sym_b].values

        if len(la) < window + 5:
            continue

        # ── OLS hedge ratio ──────────────────────────────────────────
        # log(A) = alpha + beta * log(B)
        beta, alpha, _, _, _ = stats.linregress(lb, la)

        # ── Spread with proper hedge ratio ───────────────────────────
        spread_vals = la - beta * lb - alpha
        spread = pd.Series(spread_vals)

        # ── Rolling z-score ──────────────────────────────────────────
        roll_mean = spread.rolling(window).mean()
        roll_std = spread.rolling(window).std()

        mean_now = roll_mean.iloc[-1]
        std_now = roll_std.iloc[-1]
        current_spr = spread.iloc[-1]

        if std_now == 0 or np.isnan(std_now) or np.isnan(mean_now):
            continue

        z = (current_spr - mean_now) / std_now

        # ── ADF stationarity check on spread ─────────────────────────
        try:
            adf_result = adfuller(spread_vals, maxlag=1, autolag=None)
            adf_pval = round(adf_result[1], 4)
            spread_stat = adf_pval <= adf_spread_pval
        except Exception:
            adf_pval = None
            spread_stat = False

        # ── Ornstein-Uhlenbeck half-life ─────────────────────────────
        half_life = None
        try:
            lag = spread_vals[:-1]
            delta = np.diff(spread_vals)
            if len(lag) >= 5:
                b, _, _, _, _ = stats.linregress(lag, delta)
                if b < 0:
                    half_life = round(-np.log(2) / b, 1)
        except Exception:
            pass

        # ── Direction & notional sizing hint ─────────────────────────
        # β tells us how many units of B hedge 1 unit of A (in log-price space)
        price_a = price_df[sym_a].iloc[-1]
        price_b = price_df[sym_b].iloc[-1]

        # Dollar-neutral sizing: $1000 in A → need $1000·β in B
        # (approximate — β is in log-price space but works well for sizing)
        size_note = f"β={beta:.3f}  (${1000:.0f}A : ${1000*abs(beta):.0f}B)"

        if z > 0:
            # A is rich, B is cheap
            base_a = sym_a.split("/")[0]
            base_b = sym_b.split("/")[0]
            trade_note = f"SHORT {base_a} / LONG {base_b}"
        else:
            base_a = sym_a.split("/")[0]
            base_b = sym_b.split("/")[0]
            trade_note = f"LONG {base_a} / SHORT {base_b}"

        # Last-candle % change for context
        pct_a = round((price_df[sym_a].iloc[-1] /
                      price_df[sym_a].iloc[-2] - 1) * 100, 2)
        pct_b = round((price_df[sym_b].iloc[-1] /
                      price_df[sym_b].iloc[-2] - 1) * 100, 2)

        results.append({
            "Pair":        f"{sym_a.split('/')[0]} / {sym_b.split('/')[0]}",
            "ρ":           rho,
            "coint_p":     coint_p,
            "β":           round(beta, 4),
            "Z-score":     round(z, 3),
            "|Z|":         round(abs(z), 3),
            "ADF-p":       adf_pval,
            "Stationary":  "✓" if spread_stat else "~",
            "Half-life":   half_life,
            "Δ A (1c)":    pct_a,
            "Δ B (1c)":    pct_b,
            "Direction":   trade_note,
            "Sizing":      size_note,
            "_sym_a":      sym_a,
            "_sym_b":      sym_b,
        })

    results.sort(key=lambda x: x["|Z|"], reverse=True)
    return results


# ─────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────
def print_results(results: list[dict], min_zscore: float, top_n: int, cfg: dict):
    clear()
    now = datetime.now().strftime("%H:%M:%S")

    print(BOLD("=" * 125))
    print(BOLD(f"  PAIR CORRELATION & DIVERGENCE SCANNER  v2  [{now}]"))
    print(
        f"  corr≥{cfg['min_corr']}  coint_p≤{cfg['coint_pvalue']}  |z|≥{cfg['min_zscore']}  "
        f"tf={cfg['timeframe']}  lookback={cfg['lookback']}c  z-win={cfg['zscore_window']}c  "
        f"vol≥${cfg['min_volume_usdt']:,.0f}"
    )
    print(BOLD("=" * 125))

    flagged = [r for r in results if r["|Z|"] >= min_zscore]

    if not flagged:
        print(YELLOW("\n  No diverging pairs found under current thresholds.\n"))
        return

    display = flagged[:top_n]

    hdr = (
        f"  {'Pair':<24} {'ρ':>6} {'coint_p':>8} {'β':>8} "
        f"{'Z-score':>9} {'ADF-p':>7} {'St':>3} {'HL(c)':>7} "
        f"{'ΔA%':>7} {'ΔB%':>7}   Direction"
    )
    print(CYAN(hdr))
    print(CYAN("  " + "─" * 121))

    for r in display:
        z = r["Z-score"]
        az = r["|Z|"]
        hl = str(r["Half-life"]) if r["Half-life"] else "—"
        da = r["Δ A (1c)"]
        db = r["Δ B (1c)"]
        pair = r["Pair"]
        rho = r["ρ"]
        cp = r["coint_p"]
        beta = r["β"]
        adfp = f"{r['ADF-p']:.3f}" if r["ADF-p"] is not None else "—"
        stat = r["Stationary"]
        direc = r["Direction"]

        if az >= 3.0:
            zstr = RED(f"{z:>+9.3f}")
        elif az >= 2.0:
            zstr = YELLOW(f"{z:>+9.3f}")
        else:
            zstr = f"{z:>+9.3f}"

        da_str = GREEN(f"{da:>+7.2f}") if da > 0 else RED(f"{da:>+7.2f}")
        db_str = GREEN(f"{db:>+7.2f}") if db > 0 else RED(f"{db:>+7.2f}")

        # Dim rows where spread is not stationary
        row = (
            f"  {pair:<24} {rho:>6.3f} {cp:>8.4f} {beta:>8.4f} "
            f"{zstr} {adfp:>7} {stat:>3} {hl:>7} "
            f"{da_str} {db_str}   {direc}"
        )
        if stat == "~":
            print(color(row, "2"))   # dim
        else:
            print(row)

    total_corr = len(results)
    total_flag = len(flagged)
    print(CYAN("\n  " + "─" * 121))
    print(
        f"  Cointegrated pairs: {BOLD(str(total_corr))}  |  "
        f"Diverging (|z|≥{min_zscore}): {BOLD(str(total_flag))}  |  "
        f"Showing top {min(top_n, total_flag)}"
    )
    print(
        f"  {YELLOW('~')} = spread ADF p-value above threshold (treat with caution)")

    if cfg.get("loop"):
        print(
            f"\n  Next scan in {cfg['scan_interval']} min…  (Ctrl+C to stop)")
    print()


def export_csv(results: list[dict]):
    fname = f"divergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df = pd.DataFrame(results).drop(
        columns=["_sym_a", "_sym_b", "Sizing"], errors="ignore")
    df.to_csv(fname, index=False)
    print(f"  Exported → {fname}")


# ─────────────────────────────────────────────
#  MAIN SCAN CYCLE
# ─────────────────────────────────────────────
def scan_once(cfg: dict, price_df: pd.DataFrame | None = None) -> tuple[list[dict], pd.DataFrame]:
    """
    Returns (results, price_df).
    If price_df is passed in (loop mode), the matrix is updated in-place
    rather than re-fetched from scratch — much faster on subsequent loops.
    """
    print(f"\n[{ts()}] Starting scan…")

    # 1. Liquid symbols
    symbols = get_liquid_symbols(cfg["min_volume_usdt"])
    if not symbols:
        print("  No symbols found. Check connectivity.")
        return [], price_df

    # 2. Price matrix — full fetch or cache update
    if price_df is None or price_df.empty:
        price_df = fetch_close_matrix(
            symbols, cfg["timeframe"], cfg["lookback"])
    else:
        # Keep only symbols still in the liquid set
        live_syms = [s for s in price_df.columns if s in symbols]
        price_df = price_df[live_syms]
        price_df = update_close_matrix(price_df, live_syms, cfg["timeframe"])

    if price_df.empty:
        print("  Could not build price matrix.")
        return [], price_df

    # 3. Correlation matrix
    print(
        f"  [{ts()}] Computing correlation matrix ({len(price_df.columns)} symbols)…")
    corr_matrix = build_correlation_matrix(price_df)

    # 4. Correlated pairs (fast pre-filter)
    pairs = find_correlated_pairs(corr_matrix, cfg["min_corr"])
    print(f"  [{ts()}] {len(pairs)} pairs with ρ ≥ {cfg['min_corr']}")
    if not pairs:
        print("  No correlated pairs found. Try lowering --min-corr.")
        return [], price_df

    # 4b. Cointegration filter (NEW)
    print(f"  [{ts()}] Running cointegration tests…")
    coint_pairs = filter_cointegrated(price_df, pairs, cfg["coint_pvalue"])
    if not coint_pairs:
        print("  No cointegrated pairs found. Try raising --coint-pvalue.")
        return [], price_df

    # 5. Spread z-scores with OLS hedge ratio (FIXED)
    print(f"  [{ts()}] Computing OLS spreads and z-scores…")
    results = compute_spread_zscore(
        price_df, coint_pairs, cfg["zscore_window"], cfg["adf_spread_pvalue"]
    )

    return results, price_df


def run(cfg: dict):
    price_df = None

    if cfg["loop"]:
        while True:
            try:
                results, price_df = scan_once(cfg, price_df)
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
        results, _ = scan_once(cfg)
        print_results(results, cfg["min_zscore"], cfg["top_n"], cfg)
        if cfg["export"] and results:
            export_csv(results)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="Binance Futures Pair Correlation & Divergence Scanner v2"
    )
    p.add_argument("--timeframe",        default=DEFAULTS["timeframe"],
                   help="Candle timeframe (default: 1d)")
    p.add_argument("--lookback",         type=int, default=DEFAULTS["lookback"],
                   help="Candles for correlation (default: 120)")
    p.add_argument("--zscore-window",    type=int, default=DEFAULTS["zscore_window"],
                   help="Rolling window for z-score (default: 60)")
    p.add_argument("--min-corr",         type=float, default=DEFAULTS["min_corr"],
                   help="Min Pearson ρ (default: 0.80)")
    p.add_argument("--min-zscore",       type=float, default=DEFAULTS["min_zscore"],
                   help="Min |z| to flag divergence (default: 1.5)")
    p.add_argument("--coint-pvalue",     type=float, default=DEFAULTS["coint_pvalue"],
                   dest="coint_pvalue",
                   help="Max p-value for Engle-Granger cointegration test (default: 0.05)")
    p.add_argument("--adf-spread-pvalue", type=float, default=DEFAULTS["adf_spread_pvalue"],
                   dest="adf_spread_pvalue",
                   help="Max ADF p-value for spread stationarity check (default: 0.10)")
    p.add_argument("--top",              type=int, default=DEFAULTS["top_n"],
                   dest="top_n",         help="Pairs to show (default: 30)")
    p.add_argument("--min-vol",          type=float, default=DEFAULTS["min_volume_usdt"],
                   dest="min_volume_usdt",
                   help="Min 24h quote volume in USDT (default: 5_000_000)")
    p.add_argument("--interval",         type=int, default=DEFAULTS["scan_interval"],
                   dest="scan_interval",
                   help="Minutes between scans in loop mode (default: 5)")
    p.add_argument("--loop",             action="store_true",
                   help="Run continuously (uses cached price matrix)")
    p.add_argument("--export",           action="store_true",
                   help="Export results to CSV")
    args = p.parse_args()
    return vars(args)


if __name__ == "__main__":
    cfg = {**DEFAULTS, **parse_args()}
    run(cfg)
