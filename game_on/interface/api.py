from fastapi import FastAPI
from pln_model.sbert import embedding, query
import pandas as pd
import os

app = FastAPI()

df = pd.read_csv(os.getenv("CSV_PATH"))
data_limpia, game_embeddings = embedding(df)

@app.post("/query")
def recomendar(payload: dict):
    consulta = payload["query"]
    resultado = query(consulta, data_limpia, game_embeddings)
    return {"recommendations": resultado}
