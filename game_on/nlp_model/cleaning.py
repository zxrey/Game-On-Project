import numpy as np
import pandas as pd
import re

def calcular_quality_score(row, max_reviews):
    scores = []
    pesos = []

    # review_percentage (siempre que exista)
    if pd.notna(row['review_per']):
        scores.append(row['review_per'] / 100)
        pesos.append(0.30)

    # num_reviews normalizado logarítmicamente
    if pd.notna(row['total_review']) and row['total_review'] > 0:
        norm = np.log1p(row['total_review']) / np.log1p(max_reviews)
        scores.append(norm)
        pesos.append(0.50)

    # metacritic_score (solo si existe)
    if pd.notna(row['metacritic_score']):
        scores.append(row['metacritic_score'] / 100)
        pesos.append(0.20)

    if not scores:
        return None

    total_pesos = sum(pesos)
    quality_score = sum(s * p for s, p in zip(scores, pesos)) / total_pesos

    return round(quality_score, 4)



def limpieza(df, df1):
    # Limpieza de 'recent_reviews'
    df['total_review'] = df['all_reviews'].str.extract(r'\(([\d,]+)\)').iloc[:, 0].str.replace(',', '')
    df['total_review'] = pd.to_numeric(df['total_review'], errors='coerce')
    df['review_per'] = df['all_reviews'].str.extract(r'(\d+)%').iloc[:, 0]
    df['review_per'] = pd.to_numeric(df['review_per'], errors='coerce')

    # Se eliminan columnas que no son relevantes para el análisis
    df = df.drop(columns=['types', 'all_reviews', 'desc_snippet', 'recent_reviews', 'developer',
                           'publisher', 'achievements', 'mature_content', 'minimum_requirements',
                           'recommended_requirements', 'discount_price'])

    # Limpieza de 'genre', 'popular_tags', 'game_details', 'languages'
    for col in ['genre', 'popular_tags', 'game_details', 'languages']:
        df[col] = df[col].str.replace(',', ', ', regex=False)

    # Limpieza de 'release_date'
    df = df[df['release_date'].str.contains(r'\d{4}', na=False)]
    df['release_date'] = df['release_date'].str.replace(r'(\d{4})\d{4}', r'\1', regex=True).str.findall(r'\d{4}').str[-1]
    df['release_date'] = df['release_date'].str.extract(r'(\d{4})(?!.*\d{4})')
    df['release_date'] = df['release_date'].astype(int)

    #Creacion de columna APIid extraida de URL
    df['appid'] = df['url'].str.extract(r'/app/(\d+)')

    # Limpieza de 'game_description'
    def clean_text(text):
        if pd.isna(text):
            return text
        text = re.sub(r"\*+|\s+", ' ', text)
        text = text.lower()
        return text.strip()

    df['game_description'] = df['game_description'].apply(clean_text)

    # Limpieza de 'original_price'
    df['original_price'] = df['original_price'].replace({'Free': '0', '0': '0'})
    df['original_price'] = df['original_price'].astype(str).str.replace('$', '', regex=False)
    df['original_price'] = pd.to_numeric(df['original_price'], errors='coerce')
    df['original_price'] = df['original_price'].fillna(0.0)

    # Merge con segunda base de datos
    df1 = df1[['header_image', 'required_age', 'short_description', 'appid', 'metacritic_score']].copy()
    # Elimina el símbolo '+' de valores como '17+' para poder convertirlos a número
    df1['required_age'] = df1['required_age'].astype(str).str.replace('+', '', regex=False)

    # Convierte a número entero, los valores inválidos (javascript, texto) se reemplazan con 0
    df1['required_age'] = pd.to_numeric(df1['required_age'], errors='coerce').fillna(0).astype(int)

    # Si required_age es 0 se reemplaza con 'Para todas las edades',
    # si tiene un número mayor se reemplaza con 'Juego para mayores de X'
    df1['required_age'] = df1['required_age'].apply(
        lambda x: f"For all ages" if x == 0 else f"Game for people over {x} years old"
    )

    # Estandarizamos appid a string en ambos DataFrames para evitar errores en el merge
    df['appid'] = df['appid'].astype(str)
    df1['appid'] = df1['appid'].astype(str)

    # Unimos ambos DataFrames por appid, conservando solo los juegos presentes en ambas tablas
    df = pd.merge(df, df1, on='appid', how='inner')

    # Formateo de columnas para embedding
    df['review_percentage'] = df['review_per'].apply(lambda x: f"Percentage of player recommendations: {int(x)}%" if pd.notna(x) else x)
    df['popular_tags'] = df['popular_tags'].apply(lambda x: f"Popular tags: {x}" if pd.notna(x) else x)
    df['game_details'] = df['game_details'].apply(lambda x: f"Game details: {x}" if pd.notna(x) else x)
    df['genre'] = df['genre'].apply(lambda x: f"Game genre: {x}" if pd.notna(x) else x)

    df['embedding'] = (
        df['name'] + '\n' +
        df['genre'] + '\n' +
        df['popular_tags'] + '\n' +
        df['game_details'] + '\n' +
        df['review_percentage']+ '\n' +
        df['required_age'] + '\n' +
        df['game_description']
    )

    max_reviews = df['total_review'].max()
    df['quality_score'] = df.apply(
        lambda row: calcular_quality_score(row, max_reviews), axis=1
    )

    return df
