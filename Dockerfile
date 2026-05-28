FROM python:3.10-slim

WORKDIR /app

RUN apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY game_on/ ./game_on/

ENV MODEL_TARGET=gcs \
    BUCKET_NAME=gameon-models \
    GCP_PROJECT=wagon-bootcamp-490420 \
    PYTHONPATH=/app/game_on

CMD ["sh", "-c", "uvicorn game_on.interface.api:app --host 0.0.0.0 --port $PORT"]
