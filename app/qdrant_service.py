import requests
from qdrant_client import QdrantClient
from app.config import settings


class QdrantService:
    def __init__(self):
        self.client = None
    
    def connect(self):
        if self.client is None:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
        return self.client
    
    def get_embedding(self, text: str) -> list[float]:
        """Get embedding from Hugging Face Router API."""
        url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
        headers = {"Authorization": f"Bearer {settings.HF_API_KEY}"}
        
        response = requests.post(url, headers=headers, json={"inputs": text})
        response.raise_for_status()
        
        result = response.json()
        
        # Handle nested list response
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                return result[0]
            return result
        
        return result
    
    def query(self, question: str, collection: str = None, top_k: int = None) -> list[dict]:
        client = self.connect()
        
        collection_name = collection or settings.COLLECTION_NAME
        k = top_k or settings.TOP_K
        
        # Generate embedding via HF API
        query_embedding = self.get_embedding(question)
        
        # Search in Qdrant
        results = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=k
        )
        
        return [
            {
                "text": hit.payload.get("text", ""),
                "page_number": hit.payload.get("page_number"),
                "score": hit.score
            }
            for hit in results.points
        ]


qdrant_service = QdrantService()
