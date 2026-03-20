import requests

API_KEY = '489b4a063db51b4784e0f5c81221e6251b9e9de6'
SYMBOL = 'BTC'


def test_cryptopanic():
    url = "https://cryptopanic.com/api/v1/posts/"
    params = {
        'auth_token': API_KEY,
        'currencies': SYMBOL,
        'filter': 'hot'
    }

    print(f"--- Probando conexión con CryptoPanic para {SYMBOL} ---")
    try:
        response = requests.get(url, params=params)
        print(f"Status Code: {response.status_code}")  # Debería ser 200

        data = response.json()

        if 'results' in data:
            print("✅ ¡Conexión exitosa!")
            if len(data['results']) > 0:
                print(f"Última noticia: {data['results'][0]['title']}")
            else:
                print("No hay noticias 'hot' ahora mismo para esta moneda.")
        else:
            print("❌ Error en el formato de respuesta:")
            print(data)

    except Exception as e:
        print(f"❌ Error de red o código: {e}")


if __name__ == "__main__":
    test_cryptopanic()
