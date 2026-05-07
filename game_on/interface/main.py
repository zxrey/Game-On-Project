# interface/main.py

# Importa desde la subcarpeta 'pln_model'
from pln_model.limpieza import procesar_dataset
from pln_model.sbert import generar_embeddings

def main():
    # Procesamiento y limpieza
    procesar_dataset()
    # Generación de embeddings
    generar_embeddings()

if __name__ == "__main__":
    main()
