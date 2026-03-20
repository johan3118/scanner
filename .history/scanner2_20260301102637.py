import ccxt.async_support as ccxt
import pandas as pd
import requests
import time
import os
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# =========================
# CONFIGURACIÓN
# =========================
SCAN_INTERVAL_MINUTES = 5

# Universo
UNIVERSE_SIZE = 120                 # Top N contratos por volumen 24h
MIN_24H_QUOTE_VOLUME = 10_000_000   # Liquidez mínima en USDT

# Market cap
MIN_MCAP = 50_000_000
MAX_MCAP = 900_000_000

# Etapa 1: pre-breakout / incubación
PRE_TF = '1h'
PRE_LOOKBACK = 72                   # 72h
MAX_PRE_RANGE_PCT = 18.0            # Compresión aceptable en 72h
MIN_PRE_PRICE_POSITION = 0.65       # Cierre en el 65% superior del rango
MIN_PRE_VOL_PRESSURE = 1.15         # Volumen reciente > 1.15x volumen previo

# Etapa 2: breakout / trigger
TRIGGER_TF = '15m'
TRIGGER_LOOKBACK = 24               # 24 velas de 15m = 6h
MIN_TRIGGER_RVOL = 2.5
MIN_TRIGGER_CHANGE_PCT = 1.5
MIN_CLOSE_IN_CANDLE = 0.60          # Cierre en el 60% superior de la vela
MIN_BREAKOUT_BUFFER_PCT = 0.15      # Qué tanto supera el high previo

# Open Interest
OI_PERIOD = '15m'
OI_LIMIT = 6
MIN_OI_DELTA_PCT = 5.0

# News
CRYPTOPANIC_API_KEY = ''  # opcional

# Rate limiting / concurrencia
MAX_CONCURRENT_TASKS = 12
REQUEST_TIMEOUT = 8
COINGECKO_CACHE_TTL = 30 * 60
NEWS_CACHE_TTL = 10 * 60

exchange = ccxt.binance({
    'options': {'defaultType': 'future'},
    'enableRateLimit': True,
})

mcap_cache: Dict[str, Dict[str, Any]] = {}
news_cache: Dict[str, Dict[str, Any]] = {}


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return ((a - b) / b) * 100.0


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def get_coin_market_data(symbol_base: str) -> Dict[str, Any]:
    """
    CoinGecko:
    - usa /coins/markets con symbols=<ticker>
    - luego toma el match exacto del símbolo con mayor market cap
    """
    now_ts = time.time()
    cached = mcap_cache.get(symbol_base)
    if cached and now_ts - cached['ts'] < COINGECKO_CACHE_TTL:
        return cached['data']

    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "symbols": symbol_base.lower(),
        "include_tokens": "all",
        "order": "market_cap_desc",
        "per_page": 10,
        "page": 1,
        "sparkline": "false"
    }

    data = {
        "market_cap": 0,
        "coin_id": None,
        "name": symbol_base,
        "market_cap_rank": None,
        "total_volume": 0
    }

    try:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        rows = r.json()

        exact = [x for x in rows if str(
            x.get("symbol", "")).lower() == symbol_base.lower()]
        if exact:
            best = max(exact, key=lambda x: safe_float(x.get("market_cap", 0)))
            data = {
                "market_cap": safe_float(best.get("market_cap", 0)),
                "coin_id": best.get("id"),
                "name": best.get("name", symbol_base),
                "market_cap_rank": best.get("market_cap_rank"),
                "total_volume": safe_float(best.get("total_volume", 0))
            }
    except Exception:
        pass

    mcap_cache[symbol_base] = {"ts": now_ts, "data": data}
    return data


def check_catalyst(symbol_base: str) -> Tuple[bool, str]:
    if not CRYPTOPANIC_API_KEY:
        return False, "Sin API key"

    now_ts = time.time()
    cached = news_cache.get(symbol_base)
    if cached and now_ts - cached['ts'] < NEWS_CACHE_TTL:
        d = cached['data']
        return d["has_news"], d["title"]

    url = "https://cryptopanic.com/api/developer/v2/posts/"
    params = {
        'auth_token': CRYPTOPANIC_API_KEY,
        'currencies': symbol_base,
        'kind': 'news',
        'filter': 'important'
    }

    result = {"has_news": False, "title": "N/A"}
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                first_news = data['results'][0]
                title = first_news.get('title', 'N/A')
                result = {
                    "has_news": True,
                    "title": (title[:85] + '...') if len(title) > 85 else title
                }
    except Exception:
        pass

    news_cache[symbol_base] = {"ts": now_ts, "data": result}
    return result["has_news"], result["title"]


def get_open_interest_delta(symbol_id: str) -> Dict[str, Any]:
    """
    symbol_id para Binance REST: BTCUSDT, DOGEUSDT...
    """
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {
        "symbol": symbol_id,
        "period": OI_PERIOD,
        "limit": OI_LIMIT
    }

    out = {
        "oi_delta_pct": 0.0,
        "oi_rising": False,
        "oi_last": 0.0
    }

    try:
        r = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        rows = r.json()

        if isinstance(rows, list) and len(rows) >= 3:
            oi_values = [safe_float(x.get("sumOpenInterest", 0)) for x in rows]
            last_oi = oi_values[-1]
            prev_avg = sum(oi_values[:-1]) / max(len(oi_values) - 1, 1)
            delta_pct = pct(last_oi, prev_avg)

            out = {
                "oi_delta_pct": round(delta_pct, 2),
                "oi_rising": delta_pct >= MIN_OI_DELTA_PCT,
                "oi_last": last_oi
            }
    except Exception:
        pass

    return out


def compute_prebreakout_metrics(df_raw: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(df_raw) < PRE_LOOKBACK + 2:
        return None

    df = df_raw.iloc[:-1].copy()  # quitar vela en formación
    window = df.tail(PRE_LOOKBACK)

    range_high = window['high'].max()
    range_low = window['low'].min()
    if range_low <= 0:
        return None

    range_pct = pct(range_high, range_low)
    last_close = window['close'].iloc[-1]

    denom = (range_high - range_low)
    price_position = ((last_close - range_low) / denom) if denom > 0 else 0.0

    recent_vol = window['volume'].tail(6).mean()
    prev_vol = window['volume'].iloc[:-
                                     6].tail(24).mean() if len(window) >= 30 else window['volume'].mean()
    vol_pressure = (recent_vol / prev_vol) if prev_vol > 0 else 0.0

    ret_24h = pct(
        last_close, window['close'].iloc[-24]) if len(window) >= 24 else 0.0

    # scoring
    compression_score = 35 * \
        clamp((MAX_PRE_RANGE_PCT - range_pct) / MAX_PRE_RANGE_PCT, 0, 1)
    vol_score = 25 * clamp((vol_pressure - 1.0) / (1.8 - 1.0), 0, 1)
    position_score = 25 * clamp((price_position - 0.50) / (0.90 - 0.50), 0, 1)
    trend_score = 15 * clamp((ret_24h + 2) / 10, 0, 1)
    score = compression_score + vol_score + position_score + trend_score

    watch_ready = (
        range_pct <= MAX_PRE_RANGE_PCT and
        price_position >= MIN_PRE_PRICE_POSITION and
        vol_pressure >= MIN_PRE_VOL_PRESSURE
    )

    return {
        "range_pct": round(range_pct, 2),
        "price_position": round(price_position, 2),
        "vol_pressure": round(vol_pressure, 2),
        "ret_24h_pct": round(ret_24h, 2),
        "watch_score": round(score, 1),
        "watch_ready": watch_ready,
        "range_high": range_high,
        "range_low": range_low
    }


def compute_breakout_metrics(df_raw: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(df_raw) < max(TRIGGER_LOOKBACK + 5, 40):
        return None

    df = df_raw.iloc[:-1].copy()   # quitar vela en formación
    signal = df.iloc[-1]           # última vela cerrada
    history = df.iloc[:-1]

    local_high = history['high'].tail(TRIGGER_LOOKBACK).max()
    local_vol_avg = history['volume'].tail(20).mean()

    signal_open = safe_float(signal['open'])
    signal_close = safe_float(signal['close'])
    signal_high = safe_float(signal['high'])
    signal_low = safe_float(signal['low'])
    signal_vol = safe_float(signal['volume'])

    if signal_open <= 0 or signal_low <= 0:
        return None

    candle_change_pct = pct(signal_close, signal_open)
    rvol = (signal_vol / local_vol_avg) if local_vol_avg > 0 else 0.0
    breakout_buffer_pct = pct(
        signal_close, local_high) if local_high > 0 else 0.0
    candle_range = max(signal_high - signal_low, 1e-12)
    close_in_candle = (signal_close - signal_low) / candle_range

    breakout_score = 35 * clamp((breakout_buffer_pct - 0.0) / 1.5, 0, 1)
    rvol_score = 30 * clamp((rvol - 1.0) / (4.0 - 1.0), 0, 1)
    change_score = 20 * clamp((candle_change_pct - 0.5) / (3.0 - 0.5), 0, 1)
    close_score = 15 * clamp((close_in_candle - 0.5) / (0.95 - 0.5), 0, 1)
    score = breakout_score + rvol_score + change_score + close_score

    trigger_ready = (
        signal_close > local_high and
        breakout_buffer_pct >= MIN_BREAKOUT_BUFFER_PCT and
        rvol >= MIN_TRIGGER_RVOL and
        candle_change_pct >= MIN_TRIGGER_CHANGE_PCT and
        close_in_candle >= MIN_CLOSE_IN_CANDLE
    )

    return {
        "trigger_score": round(score, 1),
        "rvol": round(rvol, 2),
        "change_15m_pct": round(candle_change_pct, 2),
        "breakout_buffer_pct": round(breakout_buffer_pct, 2),
        "close_in_candle": round(close_in_candle, 2),
        "trigger_ready": trigger_ready,
        "local_high": local_high,
        "signal_close": signal_close
    }


async def fetch_ohlcv_df(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        rows = await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not rows or len(rows) < limit - 2:
            return None
        df = pd.DataFrame(
            rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except Exception:
        return None


async def build_universe() -> List[str]:
    markets = await exchange.load_markets()

    eligible = []
    for s, m in markets.items():
        if not m.get('active'):
            continue

        # Solo derivados
        if not m.get('contract'):
            continue

        # Solo Binance Futures PERPETUALS (swap), no delivery futures
        if m.get('type') != 'swap':
            continue

        # Solo lineales USDⓈ-M
        if not m.get('linear'):
            continue

        # Si quieres solo USDT, deja esto así
        if m.get('quote') != 'USDT':
            continue

        # Asegura símbolo perp tipo BTC/USDT:USDT
        if ':USDT' not in s:
            continue

        eligible.append(s)

    if not eligible:
        return []

    tickers = await exchange.fetch_tickers(eligible)

    ranked = []
    for s in eligible:
        t = tickers.get(s, {})
        quote_volume = safe_float(
            t.get('quoteVolume', 0)
        )

        if quote_volume >= MIN_24H_QUOTE_VOLUME:
            ranked.append((s, quote_volume))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked[:UNIVERSE_SIZE]]


async def analyze_symbol(symbol: str, semaphore: asyncio.Semaphore) -> Optional[Dict[str, Any]]:
    async with semaphore:
        try:
            df_pre, df_trigger = await asyncio.gather(
                fetch_ohlcv_df(symbol, PRE_TF, PRE_LOOKBACK + 5),
                fetch_ohlcv_df(symbol, TRIGGER_TF, max(
                    TRIGGER_LOOKBACK + 25, 50))
            )

            if df_pre is None or df_trigger is None:
                return None

            pre = compute_prebreakout_metrics(df_pre)
            trig = compute_breakout_metrics(df_trigger)

            if not pre or not trig:
                return None

            # Si no está ni en watchlist ni en breakout, fuera
            if not pre['watch_ready'] and not trig['trigger_ready']:
                return None

            market = exchange.market(symbol)
            base_coin = market['base']
            symbol_id = market['id']  # Binance REST symbol, ej: BTCUSDT

            oi = await asyncio.to_thread(get_open_interest_delta, symbol_id)

            mcap = await asyncio.to_thread(get_coin_market_data, base_coin)
            market_cap = safe_float(mcap.get("market_cap", 0))
            if not (MIN_MCAP <= market_cap <= MAX_MCAP):
                return None

            has_news, news_title = await asyncio.to_thread(check_catalyst, base_coin)

            catalyst_bonus = 8 if has_news else 0
            oi_bonus = 12 if oi['oi_rising'] else 0
            total_score = round(
                pre['watch_score'] * 0.45 + trig['trigger_score'] * 0.45 + catalyst_bonus + oi_bonus, 1)

            stage = "BREAKOUT" if trig['trigger_ready'] else "WATCHLIST"

            return {
                "Stage": stage,
                "Symbol": base_coin,
                "Contract": symbol,
                "Total Score": total_score,
                "Watch Score": pre['watch_score'],
                "Trigger Score": trig['trigger_score'],
                "RVOL": trig['rvol'],
                "%Change 15m": trig['change_15m_pct'],
                "Breakout %": trig['breakout_buffer_pct'],
                "OI Δ%": oi['oi_delta_pct'],
                "OI Rising": oi['oi_rising'],
                "72h Range %": pre['range_pct'],
                "Vol Pressure": pre['vol_pressure'],
                "Price Pos": pre['price_position'],
                "MCap": f"${market_cap:,.0f}",
                "CG Name": mcap.get("name", base_coin),
                "Catalyst": news_title if has_news else "No catalyst"
            }
        except Exception:
            return None


async def scan_markets() -> List[Dict[str, Any]]:
    universe = await build_universe()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    tasks = [analyze_symbol(symbol, semaphore) for symbol in universe]
    results = await asyncio.gather(*tasks)

    candidates = [x for x in results if x is not None]

    # Orden: primero breakout, luego score
    candidates.sort(
        key=lambda x: (0 if x["Stage"] == "BREAKOUT" else 1, -x["Total Score"])
    )
    return candidates


def print_results(results: List[Dict[str, Any]], scan_seconds: float):
    clear_console()
    print("=" * 130)
    print(
        f"🚀 PRE-BREAKOUT + BREAKOUT SCANNER | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 130)

    if not results:
        print("\n💤 No hubo candidatos que pasaran compresión + trigger + market cap.")
        print(f"\n⏱️ Tiempo de escaneo: {round(scan_seconds, 2)} s")
        return

    df = pd.DataFrame(results)

    breakout_df = df[df["Stage"] == "BREAKOUT"].copy()
    watch_df = df[df["Stage"] == "WATCHLIST"].copy()

    show_cols = [
        "Stage", "Symbol", "Total Score", "Watch Score", "Trigger Score",
        "RVOL", "%Change 15m", "Breakout %", "OI Δ%", "72h Range %",
        "Vol Pressure", "Price Pos", "MCap", "Catalyst"
    ]

    if not breakout_df.empty:
        print("\n🔥 BREAKOUT TRIGGERS")
        print("-" * 130)
        print(breakout_df[show_cols].to_string(index=False))

    if not watch_df.empty:
        print("\n👀 PRE-BREAKOUT WATCHLIST")
        print("-" * 130)
        print(watch_df[show_cols].to_string(index=False))

    print("\n" + "-" * 130)
    print(f"⏱️ Tiempo de escaneo: {round(scan_seconds, 2)} s")
    print(
        f"🔎 Universo analizado: top {UNIVERSE_SIZE} contratos USDT por volumen 24h")
    print(f"⏳ Próximo escaneo en {SCAN_INTERVAL_MINUTES} min...")


async def run_job():
    try:
        while True:
            clear_console()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"⚡ Escaneo iniciado... [{now}]")

            start = time.time()
            results = await scan_markets()
            elapsed = time.time() - start

            print_results(results, elapsed)
            await asyncio.sleep(SCAN_INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        print("\n🛑 Escáner detenido manualmente.")
    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(run_job())
