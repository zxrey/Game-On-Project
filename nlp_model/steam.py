import requests


def get_steam_data(appid):
    # Consultamos la API de Steam con cc=pe para obtener precios en soles peruanos
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&cc=pe"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        # Verificamos que la respuesta sea exitosa
        if data[str(appid)]['success']:
            game = data[str(appid)]['data']
            price_overview = game.get('price_overview', {})

            return {
                # Precio final con descuento aplicado (ej: S/.16.40)
                'price': price_overview.get('final_formatted'),
                # Precio original sin descuento (ej: S/.82.00)
                'original_price': price_overview.get('initial_formatted'),
                # Porcentaje de descuento (ej: 80)
                'discount': price_overview.get('discount_percent'),
                # URL del trailer en formato HLS si existe
                'trailer': game.get('movies', [{}])[0].get('hls_h264') if game.get('movies') else None
            }
    except (requests.RequestException, ValueError, KeyError):
        # Precio/trailer son datos secundarios: si Steam falla, seguimos sin ellos
        pass

    return {}
