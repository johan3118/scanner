import ccxt
import pandas as pd
import requests
import time
import os
from datetime import datetime

# --- CONFIGURACIÓN ---
RVOL_THRESHOLD = 2.0          # Volumen 2x por encima de la media
MIN_DAILY_CHANGE_PCT = 5.0    # Cambio diario mínimo del 5%
MIN_MCAP = 500_000            # 500k USD
MAX_MCAP = 500_000_000        # 500M USD
CRYPTOPANIC_API_KEY = '489b4a063db51b4784e0f5c81221e6251b9e9de6'
SCAN_INTERVAL_MINUTES = 5     # Cuántos minutos esperar entre cada escaneo completo

# Inicializar Binance Futures
exchange = ccxt.binance({
    'options': {'defaultType': 'future'},
    'enableRateLimit': True
})


def clear_console():
    """Limpia la consola para que parezca un dashboard actualizado."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_market_cap(symbol_base):
    try:
        url = "https://api.coingecko.com/api/v3/search"
        response = requests.get(url, params={"query": symbol_base}).json()
        if response.get('coins'):
            coin_id = response['coins'][0]['id']
            market_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            market_data = requests.get(market_url).json()
            return market_data.get('market_data', {}).get('market_cap', {}).get('usd', 0)
    except Exception as e:
        pass  # Silenciado para el log continuo
    return 0


def check_catalyst(symbol_base):
    if not CRYPTOPANIC_API_KEY or CRYPTOPANIC_API_KEY == 'TU_API_KEY_AQUI':
        return False, "API Key no configurada"

    url = f"https://cryptopanic.com/api/v1/posts/"
    params = {
        'auth_token': CRYPTOPANIC_API_KEY,
        'currencies': symbol_base,
        'filter': 'hot'
    }
    try:
        response = requests.get(url, params=params).json()
        if response.get('results'):
            latest_news = response['results'][0]['title']
            return True, latest_news
    except Exception as e:
        return False, "Error al buscar"
    return False, "N/A"


def scan_markets():
    markets = exchange.load_markets()
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

                base_coin = symbol.split('/')[0]
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
                time.sleep(1)  # Respetar rate limits de CoinGecko

        except Exception as e:
            continue

    return hot_coins


def run_job():
    """Ejecuta el escáner en un bucle infinito."""
    while True:
        try:
            clear_console()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"🔄 Iniciando escaneo del mercado... [{now}]")
            print(
                f"Criterios: RVOL > {RVOL_THRESHOLD}x | Mínimo {MIN_DAILY_CHANGE_PCT}% mov. | MCAP: {MIN_MCAP/1000}k - {MAX_MCAP/1000000}M\n")

            results = scan_markets()

            clear_console()
            print("="*75)
            print(
                f"🏆 RESULTADOS DEL ESCÁNER 🏆  | Última actualización: {datetime.now().strftime('%H:%M:%S')}")
            print("="*75)

            if results:
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values(by='RVOL', ascending=False)
                print(results_df.to_string(index=False))
            else:
                print(
                    "\n💤 El mercado está tranquilo. Ninguna moneda cumple los criterios.")

            print("\n" + "-"*75)
            print(
                f"⏳ Esperando {SCAN_INTERVAL_MINUTES} minutos para el próximo escaneo... (Presiona Ctrl+C para detener)")

            # Pausa el script durante X minutos
            time.sleep(SCAN_INTERVAL_MINUTES * 60)

        except KeyboardInterrupt:
            print("\n\n🛑 Escáner detenido manualmente. ¡Buen trading!")
            break
        except Exception as e:
            print(f"\n⚠️ Ocurrió un error inesperado: {e}")
            print(f"Reintentando en 60 segundos...")
            time.sleep(60)


if __name__ == "__main__":
    run_job()
