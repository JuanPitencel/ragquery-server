# RAGQuery Server

A lightweight FastAPI server for RAG (Retrieval-Augmented Generation) queries. Designed to run on free tier hosting with minimal resources.

## 🎯 Purpose

This server handles the query side of a RAG system:
- Receives questions from users
- Generates embeddings for the question
- Searches for relevant chunks in Qdrant
- Uses an LLM to generate human-friendly answers

**No PDF processing here** - that's handled by [literag-ingest](https://github.com/JuanPitencel/literag-ingest) locally.

## 🚀 Features

- 🔍 Semantic search with Qdrant Cloud
- 🧠 LLM responses with Groq (Llama 3.1) - free tier
- 🌐 Automatic question translation (Spanish ↔ English)
- 🔒 CORS configured for production
- ⚡ Fast cold starts (~2 seconds)
- 💰 Runs on free tier hosting (Railway, Render, Fly.io)

## 📦 Installation
```bash
git clone https://github.com/JuanPitencel/ragquery-server.git
cd ragquery-server
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
HF_API_KEY=your_huggingface_token
COLLECTION_NAME=your_collection
ENVIRONMENT=development
FRONTEND_URL=https://your-frontend.vercel.app
```

### Getting API Keys (all free)

| Service | Free Tier | Get Key |
|---------|-----------|---------|
| Qdrant Cloud | 1GB storage | [cloud.qdrant.io](https://cloud.qdrant.io) |
| Groq | Generous limits | [console.groq.com](https://console.groq.com) |
| HuggingFace | Unlimited* | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

## 🖥️ Running Locally
```bash
uvicorn app.main:app --reload
```

Server runs at `http://localhost:8000`

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{"status": "ok"}
```

### Chat (with LLM response)
```bash
POST /chat
Content-Type: application/json

{
  "question": "What is the fuel tank capacity?",
  "collection": "toyota_corolla_cross",
  "top_k": 5
}
```

Response:
```json
{
  "question": "What is the fuel tank capacity?",
  "answer": "The fuel tank capacity for 2WD models is 36.0 L (9.5 gal.), and for AWD models is 43.0 L (11.4 gal.).",
  "sources": [
    {
      "text": "Fuel tank capacity...",
      "page_number": 435,
      "score": 0.56
    }
  ],
  "collection": "toyota_corolla_cross"
}
```

### Query (raw chunks without LLM)
```bash
POST /query
Content-Type: application/json

{
  "question": "fuel tank capacity",
  "collection": "toyota_corolla_cross",
  "top_k": 5
}
```

### Collection Info
```bash
GET /collections/{collection_name}/info
```

Response:
```json
{
  "name": "toyota_corolla_cross",
  "points_count": 476,
  "status": "green"
}
```

## 🏗️ Architecture
```
User Question
    │
    ▼
┌─────────────────┐
│ Translate to EN │  Groq (if question is not in English)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Generate        │  HuggingFace Inference API
│ Embedding       │  (all-MiniLM-L6-v2)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Search Qdrant   │  Find top-k similar chunks
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Generate Answer │  Groq (Llama 3.1)
└─────────────────┘
    │
    ▼
Response in user's language
```

## 📁 Project Structure
```
ragquery-server/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app & endpoints
│   ├── config.py         # Environment configuration
│   ├── qdrant_service.py # Qdrant client & search
│   └── llm_service.py    # Groq client & translation
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## 🚀 Deploy to Railway

1. Push to GitHub
2. Connect repo to [Railway](https://railway.app)
3. Add environment variables:
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `GROQ_API_KEY`
   - `HF_API_KEY`
   - `COLLECTION_NAME`
   - `ENVIRONMENT=production`
   - `FRONTEND_URL=https://your-frontend.vercel.app`
4. Deploy automatically

## 🔒 CORS Configuration

- **Development**: Allows `localhost:5173` and `localhost:3000`
- **Production**: Only allows the configured `FRONTEND_URL`

## 🔧 Tech Stack

- **FastAPI** - Web framework
- **Qdrant** - Vector database
- **Groq** - LLM inference (Llama 3.1)
- **HuggingFace** - Embeddings API
- **Uvicorn** - ASGI server

## 📊 Resource Usage

| Metric | Value |
|--------|-------|
| Memory | ~150MB |
| Cold Start | ~2 seconds |
| Response Time | ~1-3 seconds |

Perfect for free tier hosting!

## 🤝 Related Projects

This is part of a complete RAG system:

- **[literag-ingest](https://github.com/JuanPitencel/literag-ingest)** - Local PDF processing pipeline
- **[toyota-corolla-cross-bot](https://github.com/JuanPitencel/toyota-corolla-cross-bot)** - Frontend React application

## 📄 License

MIT

## 👤 Author

**Juan Pitencel**
- GitHub: [@JuanPitencel](https://github.com/JuanPitencel)
