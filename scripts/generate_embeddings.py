import pandas as pd
import torch
import os

from dotenv import load_dotenv
from nlp_model.embeddings import model
from nlp_model.cleaning import limpieza

load_dotenv()

CACHE_DIR = "data"

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

# 3. Guardar localmente
df_clean.to_pickle(f"{CACHE_DIR}/df_clean.pkl")
torch.save(embeddings, f"{CACHE_DIR}/game_embeddings.pt")
print("✅ Archivos guardados localmente")
