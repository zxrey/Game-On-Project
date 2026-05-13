from sentence_transformers import SentenceTransformer, util
import torch
import requests
import os
from groq import Groq
from dotenv import load_dotenv

from pln_model.limpieza import limpieza

load_dotenv()

# ---------------- GROQ ----------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def mejorar_consulta(consulta):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": (
                f"Eres un experto en videojuegos. Dado este texto: '{consulta}', "
                "devuelve la misma consulta y agrega máximo 5 keywords en inglés "
                "separadas por comas."
            )
        }],
        max_tokens=200
    )
    return response.choices[0].message.content


def generar_descripcion(game, consulta):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": (
                f"En 2 oraciones explica por qué el juego '{game['name']}' "
                f"({game['genre']}) es buena recomendación para: {consulta}"
            )
        }],
        max_tokens=150
    )
    return response.choices[0].message.content


# ---------------- SBERT ----------------
model = SentenceTransformer('all-MiniLM-L6-v2')


def embedding(df, df1):
    df = limpieza(df, df1)

    df = df.dropna(subset=['embedding'])

    game_embeddings = model.encode(
        df['embedding'].tolist(),
        convert_to_tensor=True
    )

    return df, game_embeddings


# ---------------- STEAM API ----------------
def get_steam_data(appid):
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
    response = requests.get(url, timeout=10)
    data = response.json()

    if data[str(appid)]['success']:
        game = data[str(appid)]['data']
        return {
            'price': game.get('price_overview', {}).get('final_formatted'),
            'discount': game.get('price_overview', {}).get('discount_percent'),
            'trailer': game.get('movies', [{}])[0].get('hls_h264')
            if game.get('movies') else None
        }

    return {}


# ---------------- QUERY ENGINE ----------------
def query(consulta, df, game_embeddings, n_top=5):

    consulta_mejorada = mejorar_consulta(consulta)

    query_embedding = model.encode(
        consulta_mejorada,
        convert_to_tensor=True
    )

    cosine_scores = util.cos_sim(query_embedding, game_embeddings)[0]

    top_results = torch.topk(cosine_scores, k=n_top)

    resultados = []

    for score, idx in zip(top_results.values, top_results.indices):
        game = df.iloc[idx.item()]

        # 🔥 versión estable (sin optimizaciones aún)
        descripcion = generar_descripcion(game, consulta)
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
