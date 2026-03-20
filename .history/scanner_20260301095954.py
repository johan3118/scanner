import ccxt.async_support as ccxt  # Importante: usamos la versión asíncrona
import pandas as pd
import requests
import time
import os
import asyncio
from datetime import datetime

# --- CONFIGURACIÓN ESTRATÉGICA ---
# Buscamos una inyección de capital violenta (5x el promedio)
RVOL_THRESHOLD = 5.0
# Vela de intención clara alcista (ignoramos shorts)
MIN_15M_CHANGE_PCT = 2.0
# Olla de presión: Máxima variación del precio en las 24h previas
MAX_ACCUMULATION_RANGE_PCT = 10.0
MIN_MCAP = 50_000
MAX_MCAP = 900_000_000
CRYPTOPANIC_API_KEY = '489b4a063db51b4784e0f5c81221e6251b9e9de6'
SCAN_INTERVAL_MINUTES = 5

# Inicializar Binance Futures (Asíncrono)
exchange = ccxt.binance({
    'options': {'defaultType': 'future'},
    'enableRateLimit': True
})


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_market_cap(symbol_base):
    try:
        url = "https://api.coingecko.com/api/v3/search"
        response = requests.get(
            url, params={"query": symbol_base}, timeout=5).json()
        if response.get('coins'):
            coin_id = response['coins'][0]['id']
            market_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            market_data = requests.get(market_url, timeout=5).json()
            return market_data.get('market_data', {}).get('market_cap', {}).get('usd', 0)
    except:
        pass
    return 0


def check_catalyst(symbol_base):
    url = "https://cryptopanic.com/api/developer/v2/posts/"
    params = {
        'auth_token': CRYPTOPANIC_API_KEY,
        'currencies': symbol_base,
        'kind': 'news',
        'filter': 'important'
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                first_news = data['results'][0]
                title = first_news['title']
                clean_title = (
                    title[:75] + '...') if len(title) > 75 else title
                return True, clean_title
    except:
        pass
    return False, "N/A"


async def fetch_and_analyze(symbol, semaphore):
    # El semáforo evita que Binance nos bloquee por pedir 300 monedas a la vez
    async with semaphore:
        try:
            # Descargamos 100 velas de 15m (Aprox. 25 horas de historial)
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
            if len(ohlcv) < 100:
                return None

            df = pd.DataFrame(
                ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # 1. FILTRO DE ACUMULACIÓN (Las primeras 99 velas)
            history = df.iloc[:-1]
            max_price = history['high'].max()
            min_price = history['low'].min()

            # Calculamos cuánto se movió en las últimas 24h
            accumulation_range = ((max_price - min_price) / min_price) * 100

            # Si ya se movió más del límite, descartamos (ya llegó tarde a la fiesta)
            if accumulation_range > MAX_ACCUMULATION_RANGE_PCT:
                return None

            # 2. FILTRO DE EXPLOSIÓN (La vela actual de 15m)
            avg_volume = history['volume'].mean()
            current_volume = df['volume'].iloc[-1]
            rvol = current_volume / avg_volume if avg_volume > 0 else 0

            current_open = df['open'].iloc[-1]
            current_close = df['close'].iloc[-1]

            # Lógica exclusiva para compras (Longs)
            pct_change = (current_close - current_open) / current_open * 100

            if rvol >= RVOL_THRESHOLD and pct_change >= MIN_15M_CHANGE_PCT:
                base_coin = symbol.split('/')[0].split(':')[0]

                # Ejecutamos llamadas a APIs externas en hilos separados para no frenar la descarga de Binance
                mcap = await asyncio.to_thread(get_market_cap, base_coin)

                if MIN_MCAP <= mcap <= MAX_MCAP:
                    has_news, news_title = await asyncio.to_thread(check_catalyst, base_coin)
                    return {
                        'Symbol': base_coin,
                        'RVOL': round(rvol, 2),
                        '% Change (15m)': round(pct_change, 2),
                        'Market Cap': f"${mcap:,.0f}",
                        'Catalyst': news_title
                    }
        except Exception:
            pass
        return None


async def scan_markets():
    markets = await exchange.load_markets()
    symbols = [s for s in markets if markets[s]
               ['active'] and s.endswith(':USDT')]

    # Procesamos hasta 15 monedas simultáneamente
    semaphore = asyncio.Semaphore(15)

    # Creamos las tareas y las ejecutamos todas al mismo tiempo
    tasks = [fetch_and_analyze(symbol, semaphore) for symbol in symbols]
    results = await asyncio.gather(*tasks)

    # Filtramos los valores "None" para dejar solo las monedas que pasaron todos los filtros
    hot_coins = [res for res in results if res is not None]
    return hot_coins


async def run_job():
    try:
        while True:
            clear_console()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"⚡ Escaneo Asíncrono Iniciado... [{now}]")

            # Medimos el tiempo para que veas lo rápido que es ahora
            start_time = time.time()
            results = await scan_markets()
            end_time = time.time()

            clear_console()
            print("="*85)
            print(
                f"🏆 RUPTURAS INSTITUCIONALES DETECTADAS | {datetime.now().strftime('%H:%M:%S')}")
            print("="*85)

            if results:
                results_df = pd.DataFrame(results)
                print(results_df.to_string(index=False))
            else:
                print("\n💤 Mercado tranquilo. Ninguna moneda rompiendo acumulación.")

            print("\n" + "-"*85)
            print(
                f"⏱️ Tiempo de escaneo: {round(end_time - start_time, 2)} segundos")
            print(f"⏳ Próximo escaneo en {SCAN_INTERVAL_MINUTES} min...")

            await asyncio.sleep(SCAN_INTERVAL_MINUTES * 60)

    except KeyboardInterrupt:
        print("\n🛑 Escáner detenido manualmente.")
    finally:
        await exchange.close()


if __name__ == "__main__":
    # Inicia el bucle de eventos asíncrono
    asyncio.run(run_job())
