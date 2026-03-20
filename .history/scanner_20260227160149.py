import requests

CRYPTOPANIC_API_KEY = '489b4a063db51b4784e0f5c81221e6251b9e9de6'


def probar_api_total():
    # Eliminamos el filtro 'hot' para ver TODO lo que llega
    url = "https://cryptopanic.com/api/developer/v2/posts/"
    params = {
        'auth_token': CRYPTOPANIC_API_KEY,
        'public': 'true'
    }

    print("--- Verificando flujo de datos de CryptoPanic ---")
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])

            if results:
                print(
                    f"✅ ¡ÉXITO! Se recibieron {len(results)} noticias generales.")
                print("\nÚltimos 3 titulares del mercado:")
                for i, news in enumerate(results[:3]):
                    # Mostramos el título y las monedas relacionadas
                    currencies = news.get('currencies', [])
                    symbols = [c['code']
                               for c in currencies] if currencies else "General"
                    print(f"{i+1}. [{symbols}] {news['title']}")
            else:
                print("❌ Conectado, pero la lista de noticias está vacía.")
        else:
            print(f"❌ Error de servidor: {response.status_code}")

    except Exception as e:
        print(f"❌ Error de red: {e}")


if __name__ == "__main__":
    probar_api_total()
