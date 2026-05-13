import requests

def get_steam_data(appid):

    url = (
        f"https://store.steampowered.com/api/appdetails"
        f"?appids={appid}"
    )

    try:

        response = requests.get(url, timeout=5)

        data = response.json()

        if data[str(appid)]["success"]:

            game = data[str(appid)]["data"]

            return {
                "price": game.get(
                    "price_overview",
                    {}
                ).get("final_formatted"),

                "discount": game.get(
                    "price_overview",
                    {}
                ).get("discount_percent"),

                "trailer": (
                    game.get("movies", [{}])[0]
                    .get("hls_h264")
                    if game.get("movies")
                    else None
                )
            }

    except Exception:
        pass

    return {}
