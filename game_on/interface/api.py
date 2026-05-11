from fastapi import FastAPI
from pln_model.sbert import embedding, query
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

app = FastAPI()

df = pd.read_csv(os.getenv("CSV_PATH"))
df1 = pd.read_csv(os.getenv("CSV_PATH_IMG"))
data_limpia, game_embeddings = embedding(df, df1)

@app.post("/query")
def recomendar(payload: dict):
    consulta = payload["query"]
    resultado, consulta_mejorada = query(consulta, data_limpia, game_embeddings)
    return {"recommendations": resultado, "consulta_mejorada": consulta_mejorada}
