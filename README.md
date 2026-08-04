# 🎮 Game-On — Semantic Video Game Search Engine

A semantic search engine for video games that understands natural language queries in English and Spanish. Built as a final project for the Le Wagon Data Science & AI Bootcamp (2025).

**Team:** 4 members | **Dataset:** ~30,000 Steam games

🔗 [Live Demo](https://game-on-frontend-34uayvdzekl6rnqlg8xpiu.streamlit.app/)

---

## 📸 Demo

<img width="1292" height="874" alt="Captura de pantalla 2026-05-28 135932" src="https://github.com/user-attachments/assets/81347440-0dc2-49ed-83b3-39054e054729" />


---

## 🎯 Goal

Steam's catalog has grown to ~30,000 titles, but discovery still mostly relies on exact tags and filters — you have to already know what you're looking for. **Game-On** started from a simpler premise:

> What if you could just describe the game you want, in your own words, and get a match that actually fits?

The objective was to build a search engine that anyone — not just Steam power users — could use to find a game that matches their mood or interests, by typing a natural-language query in English or Spanish instead of navigating tag filters.

That objective shaped the questions the project set out to answer:

- Can a multilingual sentence-embedding model match free-text queries to games more effectively than keyword/tag search?
- Can an LLM clean up and enrich a vague or informal user query *before* the search to improve match relevance?
- Can semantic similarity be combined with a game-quality signal (reviews, Metacritic score) so results aren't just the most literal match, but also *good* games?

---

## 🔍 How It Works

1. **Dataset** — Steam Games dataset from Kaggle (~30,000 games after cleaning)
2. **Embeddings** — Relevant game features are grouped and encoded using **Sentence Transformers** (`paraphrase-multilingual-mpnet-base-v2`) — a multilingual model that enables searches in both English and Spanish
3. **Query Enhancement** — User queries are improved automatically using **Groq LLaMA 3** before the search
4. **Semantic Search** — The improved query is matched against game embeddings using cosine similarity
5. **Scoring** — A custom relevance score ranks results and identifies the best match
6. **Explanation** — The LLM generates a personalized explanation of why each game fits the user's query
7. **Output** — Top 5 recommendations displayed with match %, genre, tags, price, and Steam ratings

---

## ⚙️ Local Setup

### Requirements
- Python 3.10
- A [Groq](https://console.groq.com) API key (free tier works)
- The Steam Games dataset from Kaggle (two CSVs: game listings + app details/images)

### 1. Clone and install dependencies
```bash
git clone https://github.com/zxrey/Game-On-Project.git
cd Game-On-Project
pip install -r requirements.txt
```

### 2. Environment variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key
MODEL_TARGET=local
CSV_PATH=../raw_data/steam_games.csv
CSV_PATH_IMG=../raw_data/applications.csv
DATA_PATH=data/df_clean.pkl
EMBEDDINGS_PATH=data/game_embeddings.pt
```
`CSV_PATH`/`CSV_PATH_IMG` point to the raw Kaggle CSVs (place them under `raw_data/` at the repo root); `DATA_PATH`/`EMBEDDINGS_PATH` are resolved relative to `game_on/`, where the API expects to find them.

### 3. Generate the cleaned dataset and embeddings
The cleaned dataset and SBERT embeddings aren't shipped in the repo (they're large binary files, ~100MB). Generate them once:
```bash
cd game_on
python scripts/generate_embeddings.py
```
This cleans the raw CSVs and saves `df_clean.pkl` and `game_embeddings.pt` to `game_on/data/`.

### 4. Run the API
```bash
cd game_on
uvicorn interface.api:app --reload
```
Query it:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "relaxing farming game to play with my partner"}'
```

> The Streamlit frontend used in the [Live Demo](https://game-on-frontend-34uayvdzekl6rnqlg8xpiu.streamlit.app/) lives in a separate repo and simply calls this API.

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Embeddings | Sentence Transformers (`paraphrase-multilingual-mpnet-base-v2`) |
| LLM | Groq LLaMA 3 |
| Data processing | Pandas · NumPy |
| Frontend | Streamlit |
| Deployment | Google Cloud Platform |
| Data source | Kaggle — Steam Games dataset |

---

## ✨ Features

- 🌐 Multilingual search — accepts queries in English and Spanish
- 🤖 Automatic query enhancement via LLM
- 📊 Match percentage score per game
- 💬 AI-generated explanation for each recommendation
- 💰 Real-time price and discount data via Steam API (in Peruvian soles)
- 🎬 Trailers streamed directly from Steam
- 🛒 Direct link to Steam store

---

## 👥 Team

Built by a team of 4 at Le Wagon Data Science & AI Bootcamp — 2025.
