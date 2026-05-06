from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from pln_model.limpieza import limpieza

# Configuración del modelo SBERT
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

def embedding(df):
    # Limpieza del dataframe
    df = limpieza(df)

    # Eliminamos filas con embedding nulo y generamos los vectores
    df = df.dropna(subset=['embedding'])
    game_embeddings = model.encode(df['embedding'].tolist(), convert_to_tensor=True)

    return df, game_embeddings


def query(consulta, df, game_embeddings, n_top=5):
    '''Este funcion es para colocar la consulta y devolver la similitud con el embedding'''
    # Vectorizamos la consulta del usuario
    query_embedding = model.encode(consulta, convert_to_tensor=True)

    # Cálculo de similitud coseno
    cosine_scores = util.cos_sim(query_embedding, game_embeddings)[0]

    # Obtenemos los mejores N resultados
    top_results = torch.topk(cosine_scores, k=n_top)

    # Mostramos resultados
    print(f"🔎 Resultados para: '{consulta}'\n")
    print("="*50)

    for score, idx in zip(top_results.values, top_results.indices):
        game = df.iloc[idx.item()]
        print(f"🎮 JUEGO: {game['name']}")
        print(f"📊 Match: {score:.2%}")
        print(f"📂 {game['genre']}")
        print(f"{game['popular_tags']}")
        print(f"💰 Precio: {game['original_price']}")
        print(f"⭐ {game['review_percentage']}")
        print(f"🔗 Link: {game['url']}")
        print("-" * 50)
