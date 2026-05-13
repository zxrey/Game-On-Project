import torch
import pandas as pd

CACHE_DIR = "data/cache"

def load_cached_data():

    df = pd.read_parquet(
        f"{CACHE_DIR}/cleaned_games.parquet"
    )

    embeddings = torch.load(
        f"{CACHE_DIR}/embeddings.pt"
    )

    return df, embeddings
