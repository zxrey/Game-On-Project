from sentence_transformers import SentenceTransformer
from nlp_model.cleaning import limpieza

# Configuración del modelo SBERT
model_name = 'paraphrase-multilingual-mpnet-base-v2'
model = SentenceTransformer(model_name)


def embedding(df, df1):
    # Limpieza del dataframe
    df = limpieza(df, df1)

    # Eliminamos filas con embedding nulo y generamos los vectores
    df = df.dropna(subset=['embedding'])
    game_embeddings = model.encode(df['embedding'].tolist(), convert_to_tensor=True)

    return df, game_embeddings
