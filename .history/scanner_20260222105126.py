import ccxt
import pandas as pd
import requests
import time

# --- CONFIGURACIÓN ---
RVOL_THRESHOLD = 2.0          # Volumen 2x por encima de la media
# Cambio diario mínimo (fuerza bull o bear) del 5%
MIN_DAILY_CHANGE_PCT = 5.0
MIN_MCAP = 500_000            # 500k USD
MAX_MCAP = 500_000_000        # 500M USD
# Regístrate en cryptopanic.com para obtener una
CRYPTOPANIC_API_KEY = 'TU_API_KEY_AQUI'

# Inicializar Binance Futures
exchange = ccxt.binance({
    'options': {'defaultType': 'future'},
    'enableRateLimit': True
})


def get_market_cap(symbol_base):
    """Obtiene el Market Cap aproximado usando CoinGecko."""
    try:
        url = "https://api.coingecko.com/api/v3/search"
        response = requests.get(url, params={"query": symbol_base}).json()
        if response.get('coins'):
            coin_id = response['coins'][0]['id']
            # Obtener datos de mercado
            market_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            market_data = requests.get(market_url).json()
            return market_data.get('market_data', {}).get('market_cap', {}).get('usd', 0)
    except Exception as e:
        print(f"Error obteniendo Market Cap para {symbol_base}: {e}")
    return 0


def check_catalyst(symbol_base):
    """Verifica si hay noticias recientes en CryptoPanic."""
    if not CRYPTOPANIC_API_KEY or CRYPTOPANIC_API_KEY == 'TU_API_KEY_AQUI':
        return False, "API de CryptoPanic no configurada"

    url = f"https://cryptopanic.com/api/v1/posts/"
    params = {
        'auth_token': CRYPTOPANIC_API_KEY,
        'currencies': symbol_base,
        'filter': 'hot'  # Buscar noticias relevantes/calientes
    }
    try:
        response = requests.get(url, params=params).json()
        if response.get('results'):
            latest_news = response['results'][0]['title']
            return True, latest_news
    except Exception as e:
        return False, f"Error buscando noticias: {e}"
    return False, "Sin noticias recientes"


def scan_markets():
    print("Obteniendo mercados de Binance Futures...")
    markets = exchange.load_markets()

    # Filtrar solo pares USDT activos
    symbols = [s for s in markets if markets[s]
               ['active'] and s.endswith(':USDT')]

    hot_coins = []

    print(f"Escaneando {len(symbols)} pares...\n")

    for symbol in symbols:
        try:
            # 1. Analizar Gráfico Diario (Volumen y Fuerza)
            # Obtenemos las últimas 15 velas diarias
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=15)
            df = pd.DataFrame(
                ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            if len(df) < 15:
                continue

            # Calcular RVOL (Volumen de hoy vs media de los 14 días anteriores)
            avg_volume = df['volume'][:-1].mean()
            current_volume = df['volume'].iloc[-1]
            rvol = current_volume / avg_volume if avg_volume > 0 else 0

            # Calcular % de cambio de hoy (Fuerza absoluta)
            today_open = df['open'].iloc[-1]
            today_close = df['close'].iloc[-1]
            pct_change = abs((today_close - today_open) / today_open * 100)

            # Criterio 2 y 3: RVOL >= 2 y Fuerte movimiento de precio
            if rvol >= RVOL_THRESHOLD and pct_change >= MIN_DAILY_CHANGE_PCT:

                # Extraer "BTC" de "BTC/USDT:USDT"
                base_coin = symbol.split('/')[0]

                print(
                    f"🔥 {base_coin} detectada con fuerza! RVOL: {rvol:.2f}x | Cambio: {pct_change:.2f}%")

                # Criterio 1: Market Cap
                print(f"   Verificando Market Cap para {base_coin}...")
                mcap = get_market_cap(base_coin)

                if MIN_MCAP <= mcap <= MAX_MCAP:
                    print(f"   ✅ Market Cap válido: ${mcap:,.0f}")

                    # Criterio 4: Catalizador (Noticias)
                    print(f"   Buscando catalizadores...")
                    has_news, news_title = check_catalyst(base_coin)

                    if has_news:
                        print(f"   ✅ Noticia detectada: {news_title}")
                        hot_coins.append({
                            'Symbol': base_coin,
                            'RVOL': round(rvol, 2),
                            '% Change': round(pct_change, 2),
                            'Market Cap': mcap,
                            'Catalyst': news_title
                        })
                    else:
                        print("   ❌ Sin catalizadores fuertes recientes.")
                else:
                    print(f"   ❌ Market Cap fuera de rango (${mcap:,.0f}).")

                print("-" * 40)

                # Evitar saturar las APIs de CoinGecko y CryptoPanic
                time.sleep(1)

        except Exception as e:
            # Manejo de errores silencioso para que el loop no se rompa
            continue

    return hot_coins


if __name__ == "__main__":
    results = scan_markets()

    print("\n" + "="*50)
    print("🏆 RESULTADOS FINALES DEL ESCÁNER 🏆")
    print("="*50)

    if results:
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
    else:
        print("Ninguna moneda cumple todos los criterios en este momento.")
