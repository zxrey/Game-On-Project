from fastapi import FastAPI
from pydantic import BaseModel
from nlp_model.search import query
from nlp_model.params import MODEL_TARGET, EMBEDDINGS_PATH, DATA_PATH, BUCKET_NAME, GCP_PROJECT
import pandas as pd
import torch

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


if MODEL_TARGET == "local":
    print("✅ Cargando desde local...")
    app.state.game_embeddings = torch.load(EMBEDDINGS_PATH, map_location=torch.device('cpu'))
    app.state.data_limpia = pd.read_pickle(DATA_PATH)

elif MODEL_TARGET == "gcs":
    print("✅ Descargando desde GCS...")
    from google.cloud import storage

    client = storage.Client(project=GCP_PROJECT)
    bucket = client.bucket(BUCKET_NAME)

    bucket.blob("game_embeddings.pt").download_to_filename("/tmp/game_embeddings.pt")
    bucket.blob("df_clean.pkl").download_to_filename("/tmp/df_clean.pkl")

    app.state.game_embeddings = torch.load(
        "/tmp/game_embeddings.pt",
        map_location=torch.device('cpu')
    )
    app.state.data_limpia = pd.read_pickle("/tmp/df_clean.pkl")

print("✅ API lista")

@app.post("/query")
def recomendar(payload: QueryRequest):
    resultado, consulta_mejorada = query(
        payload.query,
        app.state.data_limpia,
        app.state.game_embeddings
    )

    return {
        "recommendations": resultado,
        "consulta_mejorada": consulta_mejorada
    }
