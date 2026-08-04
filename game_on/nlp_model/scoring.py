import pandas as pd


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
