from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from pln_model.limpieza import limpieza
import requests
import os
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

# Configuración del modelo SBERT
model_name = 'paraphrase-multilingual-mpnet-base-v2'
model = SentenceTransformer(model_name)

# ---------------- GROQ ----------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def mejorar_consulta(consulta):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # modelo más grande y preciso
            messages=[{"role": "user", "content": f"""You are a video game expert.
        Given this search query: '{consulta}'

        1. Translate the query to English if it is in another language
        2. Write the translated query
        2. Extract the main theme/enemy/setting keywords
        3. Repeat those keywords 3 times to give them more weight
        4. Add 5 related gaming terms in English

        Example:
        'kill horde of demons' → 'demons demons demons, kill horde of demons,
        demon slayer, hellish, gore, FPS, shooter'

        Return only the words, no explanation."""}],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception:
        # Si Groq falla, seguimos la búsqueda con la consulta original sin mejorar
        return consulta

def generar_descripcion(game, consulta):
    # Genera una descripción personalizada explicando por qué el juego es una buena recomendación
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user",
                       "content": f"En 2 oraciones explica por qué el juego '{game['name']}' ({game['genre']}) es una buena recomendación para alguien que busca: {consulta}"}],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception:
        return "No se pudo generar una descripción para este juego en este momento."


# ---------------- SBERT ----------------


def embedding(df, df1):
    # Limpieza del dataframe
    df = limpieza(df, df1)

    # Eliminamos filas con embedding nulo y generamos los vectores
    df = df.dropna(subset=['embedding'])
    game_embeddings = model.encode(df['embedding'].tolist(), convert_to_tensor=True)

    return df, game_embeddings


# ---------------- STEAM API ----------------
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

#----------reordenar por quality------
def reordenar_por_calidad(resultados):
    for juego in resultados:
        quality = juego.get('quality_score') or 0
        if pd.isna(quality):
            quality = 0

        juego['score_final'] = (
            juego['match'] * 0.55 +  # antes era 0.70
            quality        * 0.45    # antes era 0.30
        )

    resultados = sorted(resultados, key=lambda x: x['score_final'], reverse=True)
    return resultados



# ---------------- QUERY ENGINE ----------------
def query(consulta, df, game_embeddings, n_top=1000):
    '''Esta funcion recibe una consulta y retorna los juegos más similares'''
    # Mejoramos la consulta del usuario con Gemini
    consulta_mejorada = mejorar_consulta(consulta)

    # Vectorizamos la consulta mejorada
    query_embedding = model.encode(
        consulta_mejorada,
        convert_to_tensor=True
    )

    # Cálculo de similitud coseno
    cosine_scores = util.cos_sim(query_embedding, game_embeddings)[0]

    # Obtenemos los mejores N resultados
    top_results = torch.topk(cosine_scores, k=n_top)

    # Retornamos resultados como lista de diccionarios
    resultados = []
    for score, idx in zip(top_results.values, top_results.indices):
        game = df.iloc[idx.item()]

        resultados.append({
            'name': game['name'],
            'header_image': game['header_image'],
            'match': round(score.item(), 4),
            'quality_score': float(game['quality_score']) if pd.notna(game['quality_score']) else 0.0,
            'genre': game['genre'],
            'popular_tags': game['popular_tags'],
            'review_percentage': game['review_percentage'],
            'appid': game['appid'],
            'url': game['url']
        })

    # Reordena y queda con los 5 mejores
    resultados = reordenar_por_calidad(resultados)
    resultados = resultados[:5]

    # Solo 5 llamadas a Groq y Steam
    for juego in resultados:
        juego['descripcion'] = generar_descripcion(juego, consulta)
        steam_data = get_steam_data(juego['appid'])
        juego['original_price'] = steam_data.get('original_price')
        juego['price'] = steam_data.get('price')
        juego['discount'] = steam_data.get('discount', 0)
        juego['trailer'] = steam_data.get('trailer')

    return resultados, consulta_mejorada
