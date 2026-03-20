def check_catalyst(symbol_base):
    # Actualizado a la ruta v2
    url = "https://cryptopanic.com/api/developer/v2/posts/"
    params = {
        'auth_token': CRYPTOPANIC_API_KEY,
        'currencies': symbol_base,
        'filter': 'hot'
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('results') and len(data['results']) > 0:
                latest_news = data['results'][0]['title']
                return True, latest_news
    except Exception:
        return False, "Error de conexión"

    return False, "N/A"
