import pandas as pd
from pln_model.limpieza import limpieza
from pln_model.sbert import embedding, query

def main():
    # Cargar dataset local, por ejemplo un CSV
    df = pd.read_csv('/mnt/c/Users/Usuario/Downloads/steam_games.csv')

    # Limpiar dataset
    data_limpia = limpieza(df)

    # Generar embeddings (incluye limpieza dentro)
    data_limpia, game_embeddings = embedding(df)

    # Ejecutar query (consulta)
    resultado = query("tu consulta", data_limpia, game_embeddings, n_top=5)

    print(resultado)

if __name__ == "__main__":
    main()
