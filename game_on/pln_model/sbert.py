from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from pln_model.limpieza import limpieza
import requests
from dotenv import load_dotenv
load_dotenv()
#---------
import os
from groq import Groq

# Configuración de Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def mejorar_consulta(consulta):
    # Expande la consulta del usuario con palabras clave relevantes para mejorar la búsqueda
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"Eres un experto en videojuegos. Dado este texto de búsqueda: '{consulta}', devuelve exactamente la misma consulta original y agrega máximo 5 palabras clave en inglés separadas por comas que describan mejor el juego o género. Si es el nombre de un juego específico, agrega palabras que describan ese juego. Solo devuelve las palabras sin explicación ni formato adicional."}],
        max_tokens=200
    )
    return response.choices[0].message.content

def generar_descripcion(game, consulta):
    # Genera una descripción personalizada explicando por qué el juego es una buena recomendación
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"En 2 oraciones explica por qué el juego '{game['name']}' ({game['genre']}) es una buena recomendación para alguien que busca: {consulta}"}],
        max_tokens=150
    )
    return response.choices[0].message.content
#-------------

# Configuración del modelo SBERT
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

def embedding(df, df1):
    # Limpieza del dataframe
    df = limpieza(df, df1)

    # Eliminamos filas con embedding nulo y generamos los vectores
    df = df.dropna(subset=['embedding'])
    game_embeddings = model.encode(df['embedding'].tolist(), convert_to_tensor=True)

    return df, game_embeddings



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


def query(consulta, df, game_embeddings, n_top=5):
    '''Esta funcion recibe una consulta y retorna los juegos más similares'''
    # Mejoramos la consulta del usuario con Gemini
    consulta_mejorada = mejorar_consulta(consulta)

    # Vectorizamos la consulta mejorada
    query_embedding = model.encode(consulta_mejorada, convert_to_tensor=True)

    # Cálculo de similitud coseno
    cosine_scores = util.cos_sim(query_embedding, game_embeddings)[0]

    # Obtenemos los mejores N resultados
    top_results = torch.topk(cosine_scores, k=n_top)

    # Retornamos resultados como lista de diccionarios
    resultados = []
    for score, idx in zip(top_results.values, top_results.indices):
        game = df.iloc[idx.item()]
        descripcion = generar_descripcion(game, consulta)

        # Consultamos Steam en tiempo real para obtener precio y trailer actualizados
        steam_data = get_steam_data(game['appid'])

        resultados.append({
            'name': game['name'],
            'header_image': game['header_image'],
            'match': round(score.item(), 4),
            'descripcion': descripcion,
            'genre': game['genre'],
            'popular_tags': game['popular_tags'],
            'original_price': steam_data.get('price'),
            'discount': steam_data.get('discount', 0),
            'trailer': steam_data.get('trailer'),
            'review_percentage': game['review_percentage'],
            'url': game['url']
        })

    return resultados, consulta_mejorada
