# 🎮 Game-On — Semantic Video Game Search Engine

A semantic search engine for video games that understands natural language queries in English and Spanish. Built as a final project for the Le Wagon Data Science & AI Bootcamp (2025).

**Team:** 4 members | **Dataset:** ~30,000 Steam games

🔗 [Live Demo](https://your-streamlit-link-here) <!-- reemplaza con tu link real -->

---

## 📸 Demo

![Game-On Demo](screenshot.png) <!-- reemplaza con tu screenshot -->

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
