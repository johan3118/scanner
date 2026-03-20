#!/usr/bin/env python3
"""
Proper Pairs / Spread Scanner — Binance USDT-M Futures
======================================================

What this version fixes:
- Uses timestamp alignment instead of raw array alignment
- Drops the still-open candle
- Uses positive-correlation pairs only
- Builds a beta-adjusted residual spread: y - (alpha + beta*x)
- Filters by ADF stationarity + Engle-Granger cointegration
- Computes z-score on the residual spread
- Estimates half-life on the residual, not a fake 1:1 spread

Install:
    pip install ccxt pandas numpy statsmodels

Examples:
    python pairs_scanner.py
    python pairs_scanner.py --timeframe 1h --lookback 300 --zscore-window 100
    python pairs_scanner.py --min-corr 0.85 --min-zscore 2.0
    python pairs_scanner.py --loop --interval 15
    python pairs_scanner.py --export
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
from statsmodels.tsa.stattools import adfuller, coint


DEFAULTS = dict(
    timeframe="4h",
    lookback=250,
    zscore_window=80,
    min_corr=0.80,
    min_zscore=2.0,
    min_common=150,
    max_adf_p=0.05,
    max_coint_p=0.10,
    max_half_life=200.0,
    min_volume_usdt=5_000_000,
    top_n=30,
    scan_interval=15,
    export=False,
    loop=False,
)


exchange = ccxt.binance({
    "options": {"defaultType": "future"},
    "enableRateLimit": True,
})


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def color(text: str, code: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text


def RED(t): return color(t, "91")
def GREEN(t): return color(t, "92")
def YELLOW(t): return color(t, "93")
def CYAN(t): return color(t, "96")
def BOLD(t): return color(t, "1")


def base_symbol(sym: str) -> str:
    return sym.split("/")[0]


def current_closed_candle_cutoff_ms(timeframe: str) -> int:
    tf_sec = exchange.parse_timeframe(timeframe)
    tf_ms = tf_sec * 1000
    now_ms = exchange.milliseconds()
    return (now_ms // tf_ms) * tf_ms


def get_liquid_symbols(min_volume_usdt: float) -> list[str]:
    """
    Return active linear USDT perpetuals above quote volume threshold.
    """
    print(f"  [{ts()}] Loading markets / tickers…", end="\r")
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()

    symbols = []
    for sym, market in markets.items():
        if not market.get("active", True):
            continue
        if not market.get("swap", False):
            continue
        if not market.get("contract", False):
            continue
        if market.get("quote") != "USDT":
            continue
        if not market.get("linear", False):
            continue

        t = tickers.get(sym, {}) or {}
        quote_vol = t.get("quoteVolume") or 0
        if quote_vol >= min_volume_usdt:
            symbols.append(sym)

    print(f"  [{ts()}] {len(symbols)} liquid USDT perpetuals found.                ")
    return sorted(symbols)


def fetch_symbol_close_series(symbol: str, timeframe: str, lookback: int, buffer: int = 20) -> pd.Series | None:
    """
    Fetch close prices for ONE symbol, drop the still-open candle,
    return timestamp-indexed Series of closes.
    """
    try:
        ohlcv = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, limit=lookback + buffer)
        if not ohlcv:
            return None

        df = pd.DataFrame(
            ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts")

        cutoff_ms = current_closed_candle_cutoff_ms(timeframe)
        df = df[df["ts"] < cutoff_ms]

        if len(df) < lookback:
            return None

        df = df.tail(lookback).copy()
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        s = pd.Series(df["close"].astype(float).values,
                      index=df["dt"], name=symbol)
        return s

    except Exception:
        return None


def fetch_close_matrix(symbols: list[str], timeframe: str, lookback: int) -> pd.DataFrame:
    """
    Build a wide DataFrame of closes indexed by timestamp.
    No fake array alignment. Everything is aligned by real candle time.
    """
    series_list: list[pd.Series] = []
    total = len(symbols)

    for i, sym in enumerate(symbols, start=1):
        print(f"  [{ts()}] Fetching {sym} ({i}/{total})…", end="\r")
        s = fetch_symbol_close_series(sym, timeframe, lookback)
        if s is not None and len(s) >= lookback:
            series_list.append(s)

    print(
        f"  [{ts()}] Fetched {len(series_list)} valid symbols.                         ")

    if not series_list:
        return pd.DataFrame()

    price_df = pd.concat(series_list, axis=1).sort_index()
    return price_df


def find_correlated_pairs(price_df: pd.DataFrame, min_corr: float, min_common: int) -> list[tuple]:
    """
    Compute pairwise return correlation with proper pairwise overlap.
    Positive-correlation pairs only.
    """
    log_ret = np.log(price_df).diff()

    symbols = list(log_ret.columns)
    pairs: list[tuple] = []

    for a, b in combinations(symbols, 2):
        pair_ret = log_ret[[a, b]].dropna()
        n = len(pair_ret)
        if n < min_common:
            continue

        rho = pair_ret[a].corr(pair_ret[b])
        if rho is None or not np.isfinite(rho):
            continue

        if rho >= min_corr:
            pairs.append((a, b, float(rho), n))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def fit_ols_spread(y: pd.Series, x: pd.Series) -> tuple[float, float, pd.Series]:
    """
    Fit y = alpha + beta*x + residual
    Returns alpha, beta, residual
    """
    X = np.column_stack([np.ones(len(x)), x.values])
    alpha, beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
    resid = y - (alpha + beta * x)
    resid.name = "spread"
    return float(alpha), float(beta), resid


def estimate_half_life(spread: pd.Series) -> float | None:
    """
    Estimate mean-reversion half-life from:
        Δspread_t = a + b*spread_{t-1} + e_t
    Half-life = -ln(2) / b   if b < 0
    """
    vals = spread.dropna().values
    if len(vals) < 20:
        return None

    lag = vals[:-1]
    delta = np.diff(vals)

    X = np.column_stack([np.ones(len(lag)), lag])
    a, b = np.linalg.lstsq(X, delta, rcond=None)[0]

    if not np.isfinite(b) or b >= 0:
        return None

    hl = -np.log(2) / b
    if not np.isfinite(hl) or hl <= 0:
        return None

    return float(round(hl, 1))


def compute_pair_scores(price_df: pd.DataFrame, pairs: list[tuple], cfg: dict) -> list[dict]:
    """
    For each correlated pair:
    - align prices pairwise
    - fit beta-adjusted residual spread
    - filter by cointegration + ADF
    - compute latest residual z-score
    - estimate half-life
    """
    results: list[dict] = []

    for sym_a, sym_b, rho, common_n in pairs:
        pair_prices = price_df[[sym_a, sym_b]].dropna()
        if len(pair_prices) < max(cfg["min_common"], cfg["zscore_window"] + 5):
            continue

        logp = np.log(pair_prices.astype(float))
        y = logp[sym_a]
        x = logp[sym_b]

        try:
            alpha, beta, spread = fit_ols_spread(y, x)
        except Exception:
            continue

        if not np.isfinite(beta):
            continue

        # Optional sanity guard: beta should usually be positive for a normal same-direction pair
        if beta <= 0:
            continue

        # Statistical tests
        try:
            adf_p = float(adfuller(spread.dropna(),
                          regression="c", autolag="AIC")[1])
        except Exception:
            continue

        try:
            coint_p = float(coint(y, x)[1])
        except Exception:
            continue

        if not np.isfinite(adf_p) or adf_p > cfg["max_adf_p"]:
            continue
        if not np.isfinite(coint_p) or coint_p > cfg["max_coint_p"]:
            continue

        recent = spread.dropna().iloc[-cfg["zscore_window"]:]
        if len(recent) < cfg["zscore_window"]:
            continue

        mean_now = float(recent.mean())
        std_now = float(recent.std(ddof=1))
        spread_now = float(spread.iloc[-1])

        if not np.isfinite(std_now) or std_now <= 0:
            continue

        z = (spread_now - mean_now) / std_now
        if not np.isfinite(z):
            continue

        half_life = estimate_half_life(spread)
        if half_life is not None and half_life > cfg["max_half_life"]:
            continue

        pct_a = round((pair_prices[sym_a].iloc[-1] /
                      pair_prices[sym_a].iloc[-2] - 1) * 100, 2)
        pct_b = round((pair_prices[sym_b].iloc[-1] /
                      pair_prices[sym_b].iloc[-2] - 1) * 100, 2)

        if z > 0:
            direction = f"SHORT {base_symbol(sym_a)} / LONG {beta:.2f}× {base_symbol(sym_b)}"
        else:
            direction = f"LONG {base_symbol(sym_a)} / SHORT {beta:.2f}× {base_symbol(sym_b)}"

        results.append({
            "Pair": f"{base_symbol(sym_a)} / {base_symbol(sym_b)}",
            "ρ": round(rho, 4),
            "Obs": common_n,
            "Beta": round(beta, 4),
            "Alpha": round(alpha, 5),
            "Z-score": round(float(z), 3),
            "|Z|": round(abs(float(z)), 3),
            "ADF p": round(adf_p, 4),
            "Coint p": round(coint_p, 4),
            "Half-life": half_life,
            "Spread now": round(spread_now, 5),
            "Spread mean": round(mean_now, 5),
            "Spread std": round(std_now, 5),
            "Δ A (1c)": pct_a,
            "Δ B (1c)": pct_b,
            "Direction": direction,
            "_sym_a": sym_a,
            "_sym_b": sym_b,
        })

    results.sort(key=lambda r: (
        r["|Z|"], -r["Coint p"], -r["ρ"]), reverse=True)
    return results


def print_results(results: list[dict], cfg: dict):
    clear()
    now = datetime.now().strftime("%H:%M:%S")
    flagged = [r for r in results if r["|Z|"] >= cfg["min_zscore"]]
    display = flagged[:cfg["top_n"]]

    print(BOLD("=" * 145))
    print(BOLD(f"  PROPER PAIRS / SPREAD SCANNER   [{now}]"))
    print(
        f"  timeframe={cfg['timeframe']}  lookback={cfg['lookback']}c  "
        f"z-window={cfg['zscore_window']}c  corr≥{cfg['min_corr']}  "
        f"|z|≥{cfg['min_zscore']}  ADF p≤{cfg['max_adf_p']}  "
        f"coint p≤{cfg['max_coint_p']}"
    )
    print(BOLD("=" * 145))

    if not display:
        print(YELLOW("\n  No valid pair spreads found under current thresholds.\n"))
        return

    hdr = (
        f"  {'Pair':<24} {'ρ':>6} {'β':>7} {'Z':>8} {'ADF':>7} {'Coint':>7} "
        f"{'HL(c)':>8} {'ΔA%':>7} {'ΔB%':>7}   Direction"
    )
    print(CYAN(hdr))
    print(CYAN("  " + "─" * 139))

    for r in display:
        z = r["Z-score"]
        az = r["|Z|"]
        hl = "—" if r["Half-life"] is None else f"{r['Half-life']:.1f}"

        if az >= 3.0:
            z_str = RED(f"{z:>+8.3f}")
        elif az >= 2.0:
            z_str = YELLOW(f"{z:>+8.3f}")
        else:
            z_str = f"{z:>+8.3f}"

        da = r["Δ A (1c)"]
        db = r["Δ B (1c)"]
        da_str = GREEN(f"{da:>+7.2f}") if da > 0 else RED(f"{da:>+7.2f}")
        db_str = GREEN(f"{db:>+7.2f}") if db > 0 else RED(f"{db:>+7.2f}")

        print(
            f"  {r['Pair']:<24} {r['ρ']:>6.3f} {r['Beta']:>7.3f} {z_str} "
            f"{r['ADF p']:>7.4f} {r['Coint p']:>7.4f} {hl:>8} "
            f"{da_str} {db_str}   {r['Direction']}"
        )

    print(CYAN("\n  " + "─" * 139))
    print(
        f"  Valid stationary pairs: {BOLD(str(len(results)))}  |  "
        f"Flagged (|z|≥{cfg['min_zscore']}): {BOLD(str(len(flagged)))}  |  "
        f"Showing top {min(cfg['top_n'], len(display))}"
    )

    if cfg["loop"]:
        print(
            f"\n  Next scan in {cfg['scan_interval']} min…  (Ctrl+C to stop)")
    print()


def export_csv(results: list[dict]):
    fname = f"pairs_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df = pd.DataFrame(results).drop(
        columns=["_sym_a", "_sym_b"], errors="ignore")
    df.to_csv(fname, index=False)
    print(f"  Exported → {fname}")


def scan_once(cfg: dict) -> list[dict]:
    print(f"\n[{ts()}] Starting scan…")

    symbols = get_liquid_symbols(cfg["min_volume_usdt"])
    if not symbols:
        print("  No symbols found. Check connectivity.")
        return []

    price_df = fetch_close_matrix(symbols, cfg["timeframe"], cfg["lookback"])
    if price_df.empty:
        print("  Could not build price matrix.")
        return []

    valid_cols = [
        c for c in price_df.columns if price_df[c].notna().sum() >= cfg["min_common"]]
    price_df = price_df[valid_cols]

    if price_df.shape[1] < 2:
        print("  Not enough valid symbols after cleaning.")
        return []

    print(f"  [{ts()}] Finding positively correlated pairs…")
    pairs = find_correlated_pairs(price_df, cfg["min_corr"], cfg["min_common"])
    print(f"  [{ts()}] {len(pairs)} candidate pairs with ρ ≥ {cfg['min_corr']}")

    if not pairs:
        print("  No correlated pairs found. Try lowering --min-corr or --min-common.")
        return []

    print(f"  [{ts()}] Fitting beta spreads + stationarity filters…")
    results = compute_pair_scores(price_df, pairs, cfg)

    return results


def run(cfg: dict):
    if cfg["loop"]:
        while True:
            try:
                results = scan_once(cfg)
                print_results(results, cfg)
                if cfg["export"] and results:
                    export_csv(results)
                time.sleep(cfg["scan_interval"] * 60)
            except KeyboardInterrupt:
                print(RED("\n  Stopped."))
                break
            except Exception as e:
                print(RED(f"\n  Error: {e}"))
                time.sleep(30)
    else:
        results = scan_once(cfg)
        print_results(results, cfg)
        if cfg["export"] and results:
            export_csv(results)


def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="Proper Binance Futures Pairs / Spread Scanner")
    p.add_argument(
        "--timeframe", default=DEFAULTS["timeframe"], help="Candle timeframe (default: 4h)")
    p.add_argument("--lookback", type=int,
                   default=DEFAULTS["lookback"], help="Candles to fetch (default: 250)")
    p.add_argument("--zscore-window", type=int,
                   default=DEFAULTS["zscore_window"], help="Residual z window (default: 80)")
    p.add_argument("--min-corr", type=float,
                   default=DEFAULTS["min_corr"], help="Min positive correlation (default: 0.80)")
    p.add_argument("--min-zscore", type=float,
                   default=DEFAULTS["min_zscore"], help="Min |z| to flag (default: 2.0)")
    p.add_argument("--min-common", type=int,
                   default=DEFAULTS["min_common"], help="Min overlapping return bars (default: 150)")
    p.add_argument("--max-adf-p", type=float,
                   default=DEFAULTS["max_adf_p"], help="Max ADF p-value (default: 0.05)")
    p.add_argument("--max-coint-p", type=float,
                   default=DEFAULTS["max_coint_p"], help="Max cointegration p-value (default: 0.10)")
    p.add_argument("--max-half-life", type=float,
                   default=DEFAULTS["max_half_life"], help="Max half-life in candles (default: 200)")
    p.add_argument("--min-vol", type=float,
                   default=DEFAULTS["min_volume_usdt"], dest="min_volume_usdt", help="Min 24h quote volume in USDT")
    p.add_argument("--top", type=int,
                   default=DEFAULTS["top_n"], dest="top_n", help="Pairs to show")
    p.add_argument("--interval", type=int,
                   default=DEFAULTS["scan_interval"], dest="scan_interval", help="Minutes between scans")
    p.add_argument("--loop", action="store_true", help="Run continuously")
    p.add_argument("--export", action="store_true",
                   help="Export results to CSV")
    args = p.parse_args()
    return vars(args)


if __name__ == "__main__":
    cfg = {**DEFAULTS, **parse_args()}
    run(cfg)
