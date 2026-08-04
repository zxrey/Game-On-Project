import pandas as pd
from nlp_model.embeddings import embedding
from nlp_model.search import query
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

def main():
    # Cargar dataset local, por ejemplo un CSV
    df = pd.read_csv(os.getenv("CSV_PATH"))
    df1 = pd.read_csv(os.getenv("CSV_PATH_IMG"))

    # Generar embeddings (incluye limpieza dentro)
    data_limpia, game_embeddings = embedding(df, df1)

    # Ejecutar query (consulta)
    resultado, consulta_mejorada = query("kill demon hordes, Gore, single player, shooter, first person", data_limpia, game_embeddings, n_top=5)

    print(resultado)

if __name__ == "__main__":
    main()
