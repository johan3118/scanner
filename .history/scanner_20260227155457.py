import requests

# Tu clave tal cual la copiaste
API_KEY = '489b4a063db51b4784e0f5c81221e6251b9e9de6'


def test_final():
    # En la API de CryptoPanic, el token funciona mejor pegado directamente en la URL
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={API_KEY}&currencies=BTC"

    print(f"--- Probando URL directa ---")
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✅ ¡LOGRADO! Conexión establecida.")
            if data.get('results'):
                print(f"Noticia: {data['results'][0]['title']}")
        else:
            print(f"❌ Sigue fallando. Código: {response.status_code}")
            print(
                "Posible causa: Tu API Key tiene un espacio extra o no está activa aún.")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_final()
