import httpx
from qdrant_client import AsyncQdrantClient
from app.config import settings


class QdrantService:
    def __init__(self):
        self.client = None
        self.hf_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
        self.hf_headers = {
            "Authorization": f"Bearer {settings.HF_API_KEY}",
            "Content-Type": "application/json"
        }
    
    async def connect(self):
        if self.client is None:
            self.client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
        return self.client
    
    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding from HuggingFace API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.hf_url,
                headers=self.hf_headers,
                json={"inputs": text},
                timeout=30.0
            )
            response.raise_for_status()
            embedding = response.json()
            
            # Handle nested list response
            if isinstance(embedding, list) and len(embedding) > 0:
                if isinstance(embedding[0], list):
                    embedding = embedding[0]
            
            return embedding
    
    async def query(self, question: str, collection: str, top_k: int = 5) -> list[dict]:
        """Query Qdrant for similar chunks."""
        client = await self.connect()
        
        # Get embedding for the question
        embedding = await self.get_embedding(question)
        
        # Search in Qdrant using query_points (async client method)
        from qdrant_client.models import models
        
        results = await client.query_points(
            collection_name=collection,
            query=embedding,
            limit=top_k
        )
        
        return [
            {
                "text": point.payload.get("text", ""),
                "page_number": point.payload.get("page_number"),
                "score": point.score
            }
            for point in results.points
        ]
    
    async def get_collection_info(self, collection_name: str):
        """Get collection info from Qdrant."""
        client = await self.connect()
        return await client.get_collection(collection_name)


qdrant_service = QdrantService()
