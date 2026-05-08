from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from pln_model.limpieza import limpieza

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


def query(consulta, df, game_embeddings, n_top=5):
    '''Esta funcion recibe una consulta y retorna los juegos más similares'''
    # Vectorizamos la consulta del usuario
    query_embedding = model.encode(consulta, convert_to_tensor=True)

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
            'match': round(score.item(), 4),
            'genre': game['genre'],
            'popular_tags': game['popular_tags'],
            'original_price': game['original_price'],
            'review_percentage': game['review_percentage'],
            'url': game['url']
        })

    return resultados
