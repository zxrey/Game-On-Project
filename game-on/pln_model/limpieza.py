import numpy as np
import pandas as pd
import matplotlib
import re

file = "/mnt/c/Users/Usuario/Downloads/steam_games.csv"
df = pd.read_csv(file)

def limpieza(df):
    # Limpieza de 'recent_reviews'
    def extract_percentage(text):
        if not isinstance(text, str):
            return None
        match = re.search(r'([-]?\d+)%', text)
        if match:
            return int(match.group(1))
        return None

    df['review_percentage'] = df['recent_reviews'].apply(extract_percentage)

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

    # Formateo de columnas para embedding
    df['review_percentage'] = df['review_percentage'].apply(lambda x: f"Porcentaje de recomendación de jugadores: {int(x)}%" if pd.notna(x) else x)
    df['popular_tags'] = df['popular_tags'].apply(lambda x: f"Tags populares: {x}" if pd.notna(x) else x)
    df['game_details'] = df['game_details'].apply(lambda x: f"Tags populares: {x}" if pd.notna(x) else x)
    df['genre'] = df['genre'].apply(lambda x: f"Genero de juego: {x}" if pd.notna(x) else x)

    df['embedding'] = (
        df['game_description'] + '\n' +
        df['genre'] + '\n' +
        df['popular_tags'] + '\n' +
        df['game_details'] + '\n' +
        df['review_percentage']
    )

    return df
