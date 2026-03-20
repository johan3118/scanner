import requests

API_KEY = '489b4a063db51b4784e0f5c81221e6251b9e9de6'
SYMBOL = 'BTC'


def test_cryptopanic():
    # URL actualizada sin el "posts" final si da problemas,
    # o asegurando la estructura correcta:
    url = "https://cryptopanic.com/api/v1/posts/"

    params = {
        'auth_token': API_KEY,
        'currencies': SYMBOL,
        'kind': 'news',  # Usamos 'kind' en lugar de filter para probar
    }

    # Añadimos un Header para parecer un navegador real
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print(f"--- Re-intentando conexión con {SYMBOL} ---")
    try:
        response = requests.get(url, params=params, headers=headers)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✅ ¡CONECTADO CON ÉXITO!")
            if data.get('results'):
                print(f"Noticia encontrada: {data['results'][0]['title']}")
        else:
            print(
                f"❌ Error {response.status_code}: La URL sigue sin ser encontrada.")
            # Ver los primeros 100 caracteres del error
            print("Respuesta del servidor:", response.text[:100])

    except Exception as e:
        print(f"❌ Error inesperado: {e}")


if __name__ == "__main__":
    test_cryptopanic()
