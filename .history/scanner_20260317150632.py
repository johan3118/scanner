import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from statistics import mean
from zoneinfo import ZoneInfo

import requests

# ============================================================
# CONVEX ENGINE SCANNER - BINANCE USDⓈ-M FUTURES
# ------------------------------------------------------------
# Purpose:
#   Feed your convex/re-anchoring execution engine with better
#   candidates than a "violent candle only" scanner.
#
# What it does:
#   1) Scans Binance USDT perpetuals only
#   2) Uses CLOSED 15m candles only
#   3) Builds two lists:
#        - WATCHLIST  = "waking up"
#        - TRIGGER    = "confirmed enough to arm the engine"
#   4) Supports LONG and SHORT
#   5) Confirms with OI + taker buy/sell pressure
#   6) Optionally enriches with CoinGecko market cap
#   7) Writes scanner_output.json for your bot to consume
#
# Dependencies:
#   pip install requests
#
# Notes:
#   - This is the scanner layer for your algorithm.
#   - It DOES NOT place trades. It hands cleaner candidates
#     to your JS Coin execution engine.
# ============================================================

# -----------------------------
# USER CONFIG
# -----------------------------
INTERVAL = "15m"
SCAN_EVERY_CLOSED_BAR = True
RUN_LOOP = True
RUN_IMMEDIATELY_ON_START = True

# Binance universe
UNIVERSE_SIZE = 600
MIN_24H_QUOTE_VOLUME = 5_000_000
MAX_CONCURRENCY = 12

# Optional market-cap filter
USE_MCAP_FILTER = False
MIN_MCAP = 25_000_000
MAX_MCAP = 8_000_000_000
ALLOW_UNKNOWN_MCAP = True

# Optional catalyst enrichment (only for finalists)
USE_CRYPTOPANIC = False
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "").strip()

# Scanner mode
TIMEZONE_NAME = "America/Santo_Domingo"

# -----------------------------
# WATCHLIST thresholds
# Better for your convex engine:
# early expansion, not climax
# -----------------------------
WATCH_MAX_ACCUM_RANGE_PCT = 12.0
WATCH_RVOL_MIN = 1.8
WATCH_RVOL_MAX = 4.8
WATCH_CHANGE_MIN_LONG = 0.40
WATCH_CHANGE_MAX_LONG = 1.80
WATCH_CHANGE_MIN_SHORT = -1.80
WATCH_CHANGE_MAX_SHORT = -0.40
WATCH_OI_MIN_PCT = 1.0
WATCH_TAKER_RATIO_LONG = 1.05
WATCH_TAKER_RATIO_SHORT = 0.95
WATCH_BODY_QUALITY_MIN = 0.45
WATCH_NEAR_RANGE_BUFFER_PCT = 0.35
WATCH_DAY_CHANGE_ABS_MAX = 8.0

# -----------------------------
# TRIGGER thresholds
# Stronger confirmation before
# handing it to the engine
# -----------------------------
TRIGGER_MAX_ACCUM_RANGE_PCT = 14.0
TRIGGER_RVOL_MIN = 2.5
TRIGGER_CHANGE_MIN_LONG = 0.80
TRIGGER_CHANGE_MIN_SHORT = -0.80
TRIGGER_OI_MIN_PCT = 2.0
TRIGGER_TAKER_RATIO_LONG = 1.12
TRIGGER_TAKER_RATIO_SHORT = 0.88
TRIGGER_BODY_QUALITY_MIN = 0.55
TRIGGER_BREAK_BUFFER_PCT = 0.10
TRIGGER_DAY_CHANGE_ABS_MAX = 10.0

# -----------------------------
# URLs
# -----------------------------
BINANCE_FAPI = "https://fapi.binance.com"
COINGECKO_API = "https://api.coingecko.com/api/v3"
CRYPTOPANIC_API = "https://cryptopanic.com/api/developer/v2/posts/"

# -----------------------------
# Symbol overrides for tricky tickers
# Extend this if needed.
# -----------------------------
COINGECKO_ID_OVERRIDES = {
    "PEPE": "pepe",
    "FLOKI": "floki",
    "BONK": "bonk",
    "SHIB": "shiba-inu",
    "DOGE": "dogecoin",
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "SUI": "sui",
    "WIF": "dogwifcoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "TRX": "tron",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
}

session = requests.Session()
session.headers.update({"User-Agent": "convex-engine-scanner/1.0"})
_mcap_cache = {}
_news_cache = {}

# ============================================================
# HELPERS
# ============================================================


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


def now_utc():
    return datetime.now(timezone.utc)


def now_local():
    return now_utc().astimezone(ZoneInfo(TIMEZONE_NAME))


def fmt_money(x):
    if x is None:
        return "N/A"
    return f"${x:,.0f}"


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def ema(values, period):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    out = values[0]
    for v in values[1:]:
        out = (v * k) + (out * (1 - k))
    return out


def normalize_base_symbol(base):
    # 1000PEPE -> PEPE ; 1000BONK -> BONK ; remove leading digits
    return re.sub(r"^\d+", "", base.upper())


def signed_pct(a, b):
    if a == 0:
        return 0.0
    return ((b / a) - 1.0) * 100.0


def ratio(a, b, eps=1e-12):
    return a / max(b, eps)


def seconds_until_next_15m_close(buffer_seconds=8):
    current = time.time()
    bar = 15 * 60
    next_close = ((int(current) // bar) + 1) * bar
    return max(0, next_close - current + buffer_seconds)


def body_quality(candle):
    hi = candle["high"]
    lo = candle["low"]
    op = candle["open"]
    cl = candle["close"]
    rng = max(hi - lo, 1e-12)
    return abs(cl - op) / rng


def to_bars(raw_klines):
    bars = []
    for row in raw_klines:
        bars.append({
            "open_time": int(row[0]),
            "open": safe_float(row[1]),
            "high": safe_float(row[2]),
            "low": safe_float(row[3]),
            "close": safe_float(row[4]),
            "volume": safe_float(row[5]),
            "close_time": int(row[6]),
            "quote_volume": safe_float(row[7]),
            "trade_count": int(row[8]),
            "taker_buy_base": safe_float(row[9]),
            "taker_buy_quote": safe_float(row[10]),
        })
    return bars


def prune_incomplete_bar(bars):
    if not bars:
        return bars
    utc_ms = int(time.time() * 1000)
    if bars[-1]["close_time"] > utc_ms:
        return bars[:-1]
    return bars


def binance_get(path, params=None, timeout=12):
    url = f"{BINANCE_FAPI}{path}"
    r = session.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def coingecko_get(path, params=None, timeout=12):
    url = f"{COINGECKO_API}{path}"
    r = session.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def cryptopanic_get(params=None, timeout=12):
    r = session.get(CRYPTOPANIC_API, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


async def to_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


# ============================================================
# DATA SOURCES
# ============================================================

def load_universe():
    exchange_info = binance_get("/fapi/v1/exchangeInfo")
    symbols_meta = {}
    for s in exchange_info.get("symbols", []):
        if (
            s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING"
        ):
            symbols_meta[s["symbol"]] = s

    tickers = binance_get("/fapi/v1/ticker/24hr")
    rows = []
    for t in tickers:
        symbol = t.get("symbol")
        if symbol not in symbols_meta:
            continue
        quote_vol = safe_float(t.get("quoteVolume"))
        if quote_vol < MIN_24H_QUOTE_VOLUME:
            continue
        rows.append({
            "symbol": symbol,
            "base": symbols_meta[symbol]["baseAsset"],
            "quote_volume": quote_vol,
            "day_change_pct": safe_float(t.get("priceChangePercent")),
            "last_price": safe_float(t.get("lastPrice")),
        })

    rows.sort(key=lambda x: x["quote_volume"], reverse=True)
    return rows[:UNIVERSE_SIZE]


def fetch_klines(symbol, interval="15m", limit=110):
    raw = binance_get("/fapi/v1/klines",
                      {"symbol": symbol, "interval": interval, "limit": limit})
    return prune_incomplete_bar(to_bars(raw))


def fetch_oi_hist(symbol, period="15m", limit=3):
    url = f"{BINANCE_FAPI}/futures/data/openInterestHist"
    r = session.get(url, params={"symbol": symbol,
                    "period": period, "limit": limit}, timeout=12)
    r.raise_for_status()
    return r.json()


def fetch_taker_hist(symbol, period="15m", limit=3):
    # USDⓈ-M official endpoint:
    # GET /futures/data/takerlongshortRatio
    url = f"{BINANCE_FAPI}/futures/data/takerlongshortRatio"
    r = session.get(url, params={"symbol": symbol,
                    "period": period, "limit": limit}, timeout=12)
    r.raise_for_status()
    return r.json()


def fetch_premium_index(symbol):
    return binance_get("/fapi/v1/premiumIndex", {"symbol": symbol})


def resolve_coingecko_id(base_symbol):
    base = normalize_base_symbol(base_symbol)
    if base in COINGECKO_ID_OVERRIDES:
        return COINGECKO_ID_OVERRIDES[base]

    data = coingecko_get("/search", {"query": base})
    coins = data.get("coins", [])
    exact = [c for c in coins if str(c.get("symbol", "")).upper() == base]
    if not exact:
        return None

    def rank_key(c):
        rank = c.get("market_cap_rank")
        return rank if isinstance(rank, int) else 10**9

    exact.sort(key=rank_key)
    return exact[0].get("id")


def get_market_cap(base_symbol):
    base = normalize_base_symbol(base_symbol)
    if base in _mcap_cache:
        return _mcap_cache[base]

    try:
        coin_id = resolve_coingecko_id(base)
        if not coin_id:
            _mcap_cache[base] = None
            return None

        rows = coingecko_get(
            "/coins/markets", {"vs_currency": "usd", "ids": coin_id})
        if rows:
            mcap = rows[0].get("market_cap")
            _mcap_cache[base] = mcap
            return mcap
    except Exception:
        pass

    _mcap_cache[base] = None
    return None


def get_catalyst(base_symbol):
    if not USE_CRYPTOPANIC or not CRYPTOPANIC_API_KEY:
        return "N/A"

    base = normalize_base_symbol(base_symbol)
    if base in _news_cache:
        return _news_cache[base]

    try:
        data = cryptopanic_get({
            "auth_token": CRYPTOPANIC_API_KEY,
            "currencies": base,
            "kind": "news",
            "filter": "important",
        })
        results = data.get("results") or []
        if results:
            title = results[0].get("title", "N/A")
            title = (title[:80] + "...") if len(title) > 80 else title
            _news_cache[base] = title
            return title
    except Exception:
        pass

    _news_cache[base] = "N/A"
    return "N/A"


# ============================================================
# SIGNAL LOGIC
# ============================================================

def score_long(rvol, bar_change, oi_change, taker_ratio, accum_range, body_q, break_pct):
    score = 0.0
    score += min(rvol / 4.0, 2.0) * 20
    score += min(max(bar_change, 0) / 1.5, 2.0) * 15
    score += min(max(oi_change, 0) / 3.0, 2.0) * 20
    score += min(max((taker_ratio - 1.0) * 8.0, 0), 2.0) * 15
    score += min(max((12.0 - accum_range) / 12.0, 0), 1.0) * 10
    score += min(body_q / 0.8, 1.5) * 10
    score += min(max(break_pct, 0) / 0.5, 2.0) * 10
    return round(score, 1)


def score_short(rvol, bar_change, oi_change, taker_ratio, accum_range, body_q, break_pct):
    score = 0.0
    score += min(rvol / 4.0, 2.0) * 20
    score += min(abs(min(bar_change, 0)) / 1.5, 2.0) * 15
    score += min(max(oi_change, 0) / 3.0, 2.0) * 20
    score += min(max((1.0 - taker_ratio) * 8.0, 0), 2.0) * 15
    score += min(max((12.0 - accum_range) / 12.0, 0), 1.0) * 10
    score += min(body_q / 0.8, 1.5) * 10
    score += min(max(break_pct, 0) / 0.5, 2.0) * 10
    return round(score, 1)


def make_candidate(stage, side, meta):
    row = dict(meta)
    row["stage"] = stage
    row["side"] = side
    return row


def analyze_symbol_sync(row):
    symbol = row["symbol"]
    base = row["base"]
    day_change_pct = row["day_change_pct"]

    try:
        bars = fetch_klines(symbol, interval=INTERVAL, limit=120)
        if len(bars) < 100:
            return None

        signal = bars[-1]
        prior = bars[-97:-1]  # 96 x 15m bars = 24h before the signal bar

        range_high = max(b["high"] for b in prior)
        range_low = min(b["low"] for b in prior)
        accum_range_pct = signed_pct(range_low, range_high)

        close_series = [b["close"] for b in bars[-80:]]
        ema20 = ema(close_series, 20)
        ema50 = ema(close_series, 50)
        trend_up = (
            ema20 is not None and ema50 is not None and signal["close"] >= ema20 and ema20 >= ema50 * 0.995)
        trend_down = (
            ema20 is not None and ema50 is not None and signal["close"] <= ema20 and ema20 <= ema50 * 1.005)

        avg_vol = mean(b["volume"] for b in prior[-32:])
        rvol = ratio(signal["volume"], avg_vol)
        bar_change_pct = signed_pct(signal["open"], signal["close"])
        body_q = body_quality(signal)

        near_high = signal["close"] >= range_high * \
            (1 - WATCH_NEAR_RANGE_BUFFER_PCT / 100)
        near_low = signal["close"] <= range_low * \
            (1 + WATCH_NEAR_RANGE_BUFFER_PCT / 100)
        break_above_pct = signed_pct(range_high, signal["close"])
        break_below_pct = signed_pct(signal["close"], range_low)

        oi_hist = fetch_oi_hist(symbol, period=INTERVAL, limit=3)
        if len(oi_hist) < 2:
            return None
        oi_prev = safe_float(
            oi_hist[-2].get("sumOpenInterestValue") or oi_hist[-2].get("sumOpenInterest"))
        oi_last = safe_float(
            oi_hist[-1].get("sumOpenInterestValue") or oi_hist[-1].get("sumOpenInterest"))
        oi_change_pct = signed_pct(oi_prev, oi_last)

        taker_hist = fetch_taker_hist(symbol, period=INTERVAL, limit=3)
        if len(taker_hist) < 1:
            return None
        taker_last = taker_hist[-1]
        taker_ratio_value = safe_float(taker_last.get("buySellRatio"))
        if taker_ratio_value <= 0:
            buy_v = safe_float(taker_last.get("buyVol"))
            sell_v = safe_float(taker_last.get("sellVol"))
            taker_ratio_value = ratio(buy_v, sell_v)
            taker_imbalance = ratio((buy_v - sell_v), (buy_v + sell_v))
        else:
            taker_imbalance = (taker_ratio_value - 1) / (taker_ratio_value + 1)

        common = {
            "symbol": symbol,
            "base": base,
            "score": 0.0,
            "rvol": round(rvol, 2),
            "change_15m_pct": round(bar_change_pct, 2),
            "day_change_pct": round(day_change_pct, 2),
            "oi_15m_pct": round(oi_change_pct, 2),
            "taker_ratio": round(taker_ratio_value, 2),
            "taker_imbalance": round(taker_imbalance, 3),
            "accum_range_24h_pct": round(accum_range_pct, 2),
            "body_quality": round(body_q, 2),
            "break_pct": 0.0,
            "last_close": signal["close"],
            "range_high": range_high,
            "range_low": range_low,
            "quote_volume_24h": round(row["quote_volume"], 2),
            "market_cap": None,
            "catalyst": "N/A",
            "funding_rate": None,
            "next_funding_local": None,
        }

        watch_long = all([
            accum_range_pct <= WATCH_MAX_ACCUM_RANGE_PCT,
            WATCH_RVOL_MIN <= rvol <= WATCH_RVOL_MAX,
            WATCH_CHANGE_MIN_LONG <= bar_change_pct <= WATCH_CHANGE_MAX_LONG,
            oi_change_pct >= WATCH_OI_MIN_PCT,
            taker_ratio_value >= WATCH_TAKER_RATIO_LONG,
            body_q >= WATCH_BODY_QUALITY_MIN,
            abs(day_change_pct) <= WATCH_DAY_CHANGE_ABS_MAX,
            near_high,
            trend_up,
        ])

        watch_short = all([
            accum_range_pct <= WATCH_MAX_ACCUM_RANGE_PCT,
            WATCH_RVOL_MIN <= rvol <= WATCH_RVOL_MAX,
            WATCH_CHANGE_MIN_SHORT <= bar_change_pct <= WATCH_CHANGE_MAX_SHORT,
            oi_change_pct >= WATCH_OI_MIN_PCT,
            taker_ratio_value <= WATCH_TAKER_RATIO_SHORT,
            body_q >= WATCH_BODY_QUALITY_MIN,
            abs(day_change_pct) <= WATCH_DAY_CHANGE_ABS_MAX,
            near_low,
            trend_down,
        ])

        trigger_long = all([
            accum_range_pct <= TRIGGER_MAX_ACCUM_RANGE_PCT,
            rvol >= TRIGGER_RVOL_MIN,
            bar_change_pct >= TRIGGER_CHANGE_MIN_LONG,
            oi_change_pct >= TRIGGER_OI_MIN_PCT,
            taker_ratio_value >= TRIGGER_TAKER_RATIO_LONG,
            body_q >= TRIGGER_BODY_QUALITY_MIN,
            abs(day_change_pct) <= TRIGGER_DAY_CHANGE_ABS_MAX,
            trend_up,
            break_above_pct >= TRIGGER_BREAK_BUFFER_PCT,
        ])

        trigger_short = all([
            accum_range_pct <= TRIGGER_MAX_ACCUM_RANGE_PCT,
            rvol >= TRIGGER_RVOL_MIN,
            bar_change_pct <= TRIGGER_CHANGE_MIN_SHORT,
            oi_change_pct >= TRIGGER_OI_MIN_PCT,
            taker_ratio_value <= TRIGGER_TAKER_RATIO_SHORT,
            body_q >= TRIGGER_BODY_QUALITY_MIN,
            abs(day_change_pct) <= TRIGGER_DAY_CHANGE_ABS_MAX,
            trend_down,
            break_below_pct >= TRIGGER_BREAK_BUFFER_PCT,
        ])

        candidates = []

        if watch_long or trigger_long:
            meta = dict(common)
            meta["break_pct"] = round(break_above_pct, 2)
            meta["score"] = score_long(
                rvol, bar_change_pct, oi_change_pct,
                taker_ratio_value, accum_range_pct,
                body_q, break_above_pct
            )
            stage = "TRIGGER" if trigger_long else "WATCHLIST"
            candidates.append(make_candidate(stage, "LONG", meta))

        if watch_short or trigger_short:
            meta = dict(common)
            meta["break_pct"] = round(break_below_pct, 2)
            meta["score"] = score_short(
                rvol, bar_change_pct, oi_change_pct,
                taker_ratio_value, accum_range_pct,
                body_q, break_below_pct
            )
            stage = "TRIGGER" if trigger_short else "WATCHLIST"
            candidates.append(make_candidate(stage, "SHORT", meta))

        return candidates or None

    except Exception:
        return None


async def analyze_symbol(row, semaphore):
    async with semaphore:
        return await to_thread(analyze_symbol_sync, row)


async def enrich_finalists(candidates):
    async def enrich_one(c):
        base = c["base"]

        if USE_MCAP_FILTER:
            mcap = await to_thread(get_market_cap, base)
            c["market_cap"] = mcap

        if USE_CRYPTOPANIC and CRYPTOPANIC_API_KEY:
            c["catalyst"] = await to_thread(get_catalyst, base)

        try:
            premium = await to_thread(fetch_premium_index, c["symbol"])
            funding = safe_float(premium.get("lastFundingRate")) * 100
            nft = premium.get("nextFundingTime")
            next_funding_local = None
            if nft:
                dt = datetime.fromtimestamp(
                    int(nft) / 1000, tz=timezone.utc).astimezone(ZoneInfo(TIMEZONE_NAME))
                next_funding_local = dt.strftime("%Y-%m-%d %H:%M")
            c["funding_rate"] = round(funding, 4)
            c["next_funding_local"] = next_funding_local
        except Exception:
            pass

        return c

    tasks = [enrich_one(c) for c in candidates]
    return await asyncio.gather(*tasks)


def apply_market_cap_filter(candidates):
    if not USE_MCAP_FILTER:
        return candidates

    out = []
    for c in candidates:
        mcap = c.get("market_cap")
        if mcap is None and ALLOW_UNKNOWN_MCAP:
            out.append(c)
            continue
        if mcap is None and not ALLOW_UNKNOWN_MCAP:
            continue
        if MIN_MCAP <= mcap <= MAX_MCAP:
            out.append(c)
    return out


def split_lists(candidates):
    triggers = [c for c in candidates if c["stage"] == "TRIGGER"]
    watchlist = [c for c in candidates if c["stage"] == "WATCHLIST"]

    triggers.sort(key=lambda x: (-x["score"], x["symbol"]))
    watchlist.sort(key=lambda x: (-x["score"], x["symbol"]))

    return triggers, watchlist


def table(rows, columns):
    if not rows:
        return "(none)"
    widths = {c: len(c) for c in columns}
    for r in rows:
        for c in columns:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    lines = [header, sep]
    for r in rows:
        lines.append(" | ".join(
            str(r.get(c, "")).ljust(widths[c]) for c in columns))
    return "\n".join(lines)


def compact_rows(rows, max_rows=20):
    out = []
    for r in rows[:max_rows]:
        out.append({
            "Side": r["side"],
            "Symbol": r["symbol"],
            "Score": r["score"],
            "RVOL": r["rvol"],
            "%15m": r["change_15m_pct"],
            "%Day": r["day_change_pct"],
            "OI15m%": r["oi_15m_pct"],
            "Taker": r["taker_ratio"],
            "24hRange%": r["accum_range_24h_pct"],
            "Break%": r["break_pct"],
            "MCap": fmt_money(r.get("market_cap")),
            "Funding%": r.get("funding_rate", "N/A"),
            "NextFunding": r.get("next_funding_local", "N/A"),
            "Catalyst": r.get("catalyst", "N/A"),
        })
    return out


def save_output(triggers, watchlist):
    payload = {
        "generated_at_utc": now_utc().strftime("%Y-%m-%d %H:%M:%S"),
        "generated_at_local": now_local().strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": TIMEZONE_NAME,
        "engine_mode": "convex_execution_scanner_v2",
        "notes": [
            "This file is for the execution engine to consume.",
            "TRIGGER means strong enough to arm.",
            "WATCHLIST means waking-up candidate; usually wait for confirmation."
        ],
        "triggers": triggers,
        "watchlist": watchlist,
    }
    with open("scanner_output.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


async def scan_once():
    universe = await to_thread(load_universe)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = [analyze_symbol(row, semaphore) for row in universe]
    raw = await asyncio.gather(*tasks)

    candidates = []
    for x in raw:
        if x:
            if isinstance(x, list):
                candidates.extend(x)
            else:
                candidates.append(x)

    if not candidates:
        return [], []

    candidates = await enrich_finalists(candidates)
    candidates = apply_market_cap_filter(candidates)

    triggers, watchlist = split_lists(candidates)
    save_output(triggers, watchlist)
    return triggers, watchlist


def print_report(triggers, watchlist, elapsed):
    clear_console()
    print("=" * 124)
    print(
        f"🚀 CONVEX ENGINE SCANNER | {now_local().strftime('%Y-%m-%d %H:%M:%S')} ({TIMEZONE_NAME})")
    print("=" * 124)

    print("\n🎯 TRIGGERS (armable for the convex engine)")
    print("-" * 124)
    trigger_rows = compact_rows(triggers, max_rows=25)
    print(table(
        trigger_rows,
        ["Side", "Symbol", "Score", "RVOL", "%15m", "%Day", "OI15m%", "Taker",
            "24hRange%", "Break%", "MCap", "Funding%", "NextFunding", "Catalyst"]
    ))

    print("\n👀 WATCHLIST (waking up, usually wait for confirmation)")
    print("-" * 124)
    watch_rows = compact_rows(watchlist, max_rows=25)
    print(table(
        watch_rows,
        ["Side", "Symbol", "Score", "RVOL", "%15m", "%Day", "OI15m%", "Taker",
            "24hRange%", "Break%", "MCap", "Funding%", "NextFunding", "Catalyst"]
    ))

    print("\n" + "-" * 124)
    print(
        f"Universe scanned: {UNIVERSE_SIZE} top USDT perpetuals by 24h quote volume")
    print(f"Elapsed: {elapsed:.2f}s")
    print("Output file: scanner_output.json")


async def run_job():
    try:
        if RUN_IMMEDIATELY_ON_START:
            start = time.time()
            triggers, watchlist = await scan_once()
            print_report(triggers, watchlist, time.time() - start)

        if not RUN_LOOP:
            return

        while True:
            sleep_s = seconds_until_next_15m_close(
                buffer_seconds=8) if SCAN_EVERY_CLOSED_BAR else 15 * 60
            await asyncio.sleep(sleep_s)

            start = time.time()
            triggers, watchlist = await scan_once()
            print_report(triggers, watchlist, time.time() - start)

    except KeyboardInterrupt:
        print("\n🛑 Scanner stopped manually.")


if __name__ == "__main__":
    asyncio.run(run_job())
