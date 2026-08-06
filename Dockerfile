FROM python:3.10-slim

WORKDIR /app

RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY interface/ ./interface/
COPY nlp_model/ ./nlp_model/

ENV MODEL_TARGET=${MODEL_TARGET} \
    BUCKET_NAME=${BUCKET_NAME} \
    GCP_PROJECT=${GCP_PROJECT}

CMD ["sh", "-c", "uvicorn interface.api:app --host 0.0.0.0 --port $PORT"]
