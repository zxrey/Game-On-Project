from fastapi import FastAPI
from pln_model.sbert import embedding, query
import pandas as pd

app = FastAPI()

df = pd.read_csv('/mnt/c/Users/Usuario/Downloads/steam_games.csv')
data_limpia, game_embeddings = embedding(df)

@app.post("/query")
def recomendar(payload: dict):
    consulta = payload["query"]
    resultado = query(consulta, data_limpia, game_embeddings)
    return {"recommendations": resultado}
