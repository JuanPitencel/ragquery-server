from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.qdrant_service import qdrant_service
from app.llm_service import llm_service
from app.config import settings
import os


app = FastAPI(
    title="RAGQuery Server",
    description="Lightweight query server for RAG applications",
    version="0.1.0"
)

# CORS configuration
allowed_origins = []

# Only add localhost in development
if os.getenv("ENVIRONMENT") != "production":
    allowed_origins.extend([
        "http://localhost:5173",
        "http://localhost:3000",
    ])

# Always add production frontend
frontend_url = os.getenv("FRONTEND_URL", "")
if frontend_url:
    allowed_origins.append(frontend_url)

# Fallback: if no origins configured, allow the known frontend
if not allowed_origins:
    allowed_origins = ["https://toyota-corolla-cross-bot.vercel.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    collection: str | None = None
    top_k: int | None = 5


class QueryResult(BaseModel):
    text: str
    page_number: int | None
    score: float


class QueryResponse(BaseModel):
    question: str
    results: list[QueryResult]
    collection: str


class ChatRequest(BaseModel):
    question: str
    collection: str | None = None
    top_k: int | None = 5


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[QueryResult]
    collection: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/debug")
def debug():
    return {
        "environment": os.getenv("ENVIRONMENT"),
        "frontend_url": os.getenv("FRONTEND_URL"),
        "allowed_origins": allowed_origins
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Return raw chunks without LLM processing."""
    try:
        collection = request.collection or settings.COLLECTION_NAME
        
        # Translate question to English for better search
        english_question = llm_service.translate_to_english(request.question)
        
        results = qdrant_service.query(
            question=english_question,
            collection=collection,
            top_k=request.top_k
        )
        
        return QueryResponse(
            question=request.question,
            results=results,
            collection=collection
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Return LLM-generated answer based on relevant chunks."""
    try:
        collection = request.collection or settings.COLLECTION_NAME
        
        # Translate question to English for better search
        english_question = llm_service.translate_to_english(request.question)
        
        # Get relevant chunks using translated question
        chunks = qdrant_service.query(
            question=english_question,
            collection=collection,
            top_k=request.top_k
        )
        
        # Generate answer with original question (so it responds in same language)
        answer = llm_service.generate_response(request.question, chunks)
        
        return ChatResponse(
            question=request.question,
            answer=answer,
            sources=chunks,
            collection=collection
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/{collection_name}/info")
def collection_info(collection_name: str):
    try:
        client = qdrant_service.connect()
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "status": str(info.status)
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))