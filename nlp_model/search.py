import torch
import pandas as pd
from sentence_transformers import util

from nlp_model.embeddings import model
from nlp_model.llm import mejorar_consulta, generar_descripcion
from nlp_model.steam import get_steam_data
from nlp_model.scoring import reordenar_por_calidad


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
