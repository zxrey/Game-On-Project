from fastapi import FastAPI
from pln_model.sbert import embedding, query
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

app = FastAPI()

df = pd.read_csv(os.getenv("CSV_PATH"))
data_limpia, game_embeddings = embedding(df)

@app.post("/query")
def recomendar(payload: dict):
    consulta = payload["query"]
    resultado = query(consulta, data_limpia, game_embeddings)
    return {"recommendations": resultado}
