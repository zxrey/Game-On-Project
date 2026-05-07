from pln_model.limpieza import limpieza
from pln_model.sbert import generar_embeddings, query

def main():
    # Cargar dataset local, por ejemplo un CSV
    df = pd.read_csv('/mnt/c/Users/Usuario/Downloads/steam_games.csv')

    # Limpiar dataset
    data_limpia = limpieza(df)

    # Generar embeddings con sbert.py
    embeddings = generar_embeddings(data_limpia)

    # Ejecutar query (consulta)
    resultado = query("tu consulta", data_limpia, embeddings, n_top=5)

    print(resultado)

if __name__ == "__main__":
    main()
