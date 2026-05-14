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
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # modelo más grande y preciso
        messages=[{"role": "user", "content": f"""You are a video game expert.
        Given this search query: '{consulta}'

        1. Extract the main theme/enemy/setting keywords
        2. Repeat those keywords 3 times to give them more weight
        3. Add 5 related gaming terms in English

        Example:
        'kill horde of demons' → 'demons demons demons, kill horde of demons,
        demon slayer, hellish, gore, FPS, shooter'

        Return only the words, no explanation."""}],
        max_tokens=200
    )
    return response.choices[0].message.content

def generar_descripcion(game, consulta):
    # Genera una descripción personalizada explicando por qué el juego es una buena recomendación
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user",
                   "content": f"En 2 oraciones explica por qué el juego '{game['name']}' ({game['genre']}) es una buena recomendación para alguien que busca: {consulta}"}],
        max_tokens=150
    )
    return response.choices[0].message.content


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
    # Consulta la API de Steam para obtener datos en tiempo real del juego
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
    response = requests.get(url)
    data = response.json()

    # Verifica que la respuesta sea exitosa
    if data[str(appid)]['success']:
        game = data[str(appid)]['data']
        return {
            # Precio formateado (ej: "$19.99")
            'price': game.get('price_overview', {}).get('final_formatted'),
            # Porcentaje de descuento (ej: 50 para 50% de descuento)
            'discount': game.get('price_overview', {}).get('discount_percent'),
            # URL del trailer en mp4 a 480p si existe
            'trailer': game.get('movies', [{}])[0].get('hls_h264') if game.get('movies') else None
        }
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
#        descripcion = generar_descripcion(game, consulta)

        # Consultamos Steam en tiempo real para obtener precio y trailer actualizados
#        steam_data = get_steam_data(game['appid'])

        resultados.append({
            'name': game['name'],
            'header_image': game['header_image'],
            'match': round(score.item(), 4),
            'quality_score': float(game['quality_score']) if pd.notna(game['quality_score']) else 0.0,
#            'descripcion': descripcion,
            'genre': game['genre'],
            'popular_tags': game['popular_tags'],
#            'original_price': steam_data.get('price'),
#            'discount': steam_data.get('discount', 0),
#            'trailer': steam_data.get('trailer'),
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
        juego['original_price'] = steam_data.get('price')
        juego['discount'] = steam_data.get('discount', 0)
        juego['trailer'] = steam_data.get('trailer')

    return resultados, consulta_mejorada
