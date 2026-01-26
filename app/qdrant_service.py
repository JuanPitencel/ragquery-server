from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.config import settings


class QdrantService:
    def __init__(self):
        self.client = None
        self.model = None
    
    def connect(self):
        if self.client is None:
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY
            )
        return self.client
    
    def load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self.model
    
    def query(self, question: str, collection: str = None, top_k: int = None) -> list[dict]:
        client = self.connect()
        model = self.load_model()
        
        collection_name = collection or settings.COLLECTION_NAME
        k = top_k or settings.TOP_K
        
        # Generate embedding for the question
        query_embedding = model.encode(question).tolist()
        
        # Search in Qdrant (using query_points for newer versions)
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
