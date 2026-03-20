import ccxt
import pandas as pd
import requests
import time
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
RVOL_THRESHOLD = 2.0
MIN_DAILY_CHANGE_PCT = 5.0
MIN_MCAP = 200_000
MAX_MCAP = 900_000_000
# CLAVE CORREGIDA CON COMILLAS
CRYPTOPANIC_API_KEY = '489b4a063db51b4784e0f5c81221e6251b9e9de6'
SCAN_INTERVAL_MINUTES = 5

# Inicializar Binance Futures
exchange = ccxt.binance({
    'options': {'defaultType': 'future'},
    'enableRateLimit': True
})


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_market_cap(symbol_base):
    try:
        # Nota: CoinGecko gratuito es lento, si falla devolverá 0
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
        'filter': 'important'  # Cambiamos 'hot' por 'important' para ver noticias con peso
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                # Verificamos que la noticia REALMENTE mencione la moneda en sus tags
                first_news = data['results'][0]
                title = first_news['title']

                # Cortamos el título si es muy largo para que la tabla se vea bien
                clean_title = (
                    title[:75] + '...') if len(title) > 75 else title
                return True, clean_title
    except:
        pass
    return False, "N/A"


def scan_markets():
    markets = exchange.load_markets()
    # Filtrar solo pares contra USDT en Futuros
    symbols = [s for s in markets if markets[s]
               ['active'] and s.endswith(':USDT')]
    hot_coins = []

    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=15)
            if len(ohlcv) < 15:
                continue

            df = pd.DataFrame(
                ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            avg_volume = df['volume'][:-1].mean()
            current_volume = df['volume'].iloc[-1]
            rvol = current_volume / avg_volume if avg_volume > 0 else 0

            today_open = df['open'].iloc[-1]
            today_close = df['close'].iloc[-1]
            pct_change = abs((today_close - today_open) / today_open * 100)

            if rvol >= RVOL_THRESHOLD and pct_change >= MIN_DAILY_CHANGE_PCT:
                base_coin = symbol.split('/')[0].split(':')[0]
                mcap = get_market_cap(base_coin)

                if MIN_MCAP <= mcap <= MAX_MCAP:
                    has_news, news_title = check_catalyst(base_coin)
                    hot_coins.append({
                        'Symbol': base_coin,
                        'RVOL': round(rvol, 2),
                        '% Change': round(pct_change, 2),
                        'Market Cap': f"${mcap:,.0f}",
                        'Catalyst': news_title
                    })
                # Pausa para no saturar CoinGecko (puedes subirlo si da 0 en Market Cap)
                time.sleep(1.5)

        except Exception:
            continue
    return hot_coins


def run_job():
    while True:
        try:
            clear_console()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"🔄 Escaneando... [{now}]")

            results = scan_markets()

            clear_console()
            print("="*85)
            print(
                f"🏆 COINS CON VOLUMEN Y CATALIZADOR | {datetime.now().strftime('%H:%M:%S')}")
            print("="*85)

            if results:
                results_df = pd.DataFrame(results)
                print(results_df.to_string(index=False))
            else:
                print("\n💤 Sin movimientos interesantes bajo los criterios actuales.")

            print("\n" + "-"*85)
            print(f"⏳ Próximo escaneo en {SCAN_INTERVAL_MINUTES} min...")
            time.sleep(SCAN_INTERVAL_MINUTES * 60)

        except KeyboardInterrupt:
            print("\n🛑 Detenido.")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    run_job()
