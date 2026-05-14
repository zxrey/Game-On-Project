import os

##################  VARIABLES  ##################

# Datasets
CSV_PATH = os.environ.get("CSV_PATH")
CSV_PATH_IMG = os.environ.get("CSV_PATH_IMG")

# APIs
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Google Cloud
BUCKET_NAME = os.environ.get("BUCKET_NAME")
GCP_PROJECT = os.environ.get("GCP_PROJECT")

# Archivos generados
EMBEDDINGS_PATH = os.environ.get("EMBEDDINGS_PATH")
DATA_PATH = os.environ.get("DATA_PATH")

# Entorno
MODEL_TARGET = os.environ.get("MODEL_TARGET")
