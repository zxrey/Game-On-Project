import pandas as pd
import torch
import os

from dotenv import load_dotenv
from pln_model.sbert import model
from pln_model.limpieza import limpieza

load_dotenv()

CACHE_DIR = "data/cache"

os.makedirs(CACHE_DIR, exist_ok=True)

df = pd.read_csv(os.getenv("CSV_PATH"))
df1 = pd.read_csv(os.getenv("CSV_PATH_IMG"))

df_clean = limpieza(df, df1)

df_clean = df_clean.dropna(subset=["embedding"])

embeddings = model.encode(
    df_clean["embedding"].tolist(),
    convert_to_tensor=True,
    show_progress_bar=True
)

df_clean.to_parquet(f"{CACHE_DIR}/cleaned_games.parquet")

torch.save(
    embeddings,
    f"{CACHE_DIR}/embeddings.pt"
)

print("Embeddings generados correctamente")
