from fastapi import FastAPI
from pln_model.sbert import query
from pln_model.params import MODEL_TARGET, EMBEDDINGS_PATH, DATA_PATH, BUCKET_NAME
import pandas as pd
import torch

app = FastAPI()

if MODEL_TARGET == "local":
    print("✅ Cargando desde local...")
    app.state.game_embeddings = torch.load(EMBEDDINGS_PATH)
    app.state.data_limpia = pd.read_pickle(DATA_PATH)

elif MODEL_TARGET == "gcs":
    print("✅ Descargando desde GCS...")
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    bucket.blob("game_embeddings.pt").download_to_filename("game_embeddings.pt")
    bucket.blob("df_clean.pkl").download_to_filename("df_clean.pkl")

    app.state.game_embeddings = torch.load("game_embeddings.pt")
    app.state.data_limpia = pd.read_pickle("df_clean.pkl")

print("✅ API lista")

@app.post("/query")
def recomendar(payload: dict):
    consulta = payload["query"]

    resultado, consulta_mejorada = query(
        consulta,
        app.state.data_limpia,
        app.state.game_embeddings
    )

    return {
        "recommendations": resultado,
        "consulta_mejorada": consulta_mejorada
    }
