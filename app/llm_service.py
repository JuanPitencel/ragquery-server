from groq import Groq
from app.config import settings


class LLMService:
    def __init__(self):
        self.client = None
    
    def connect(self):
        if self.client is None:
            self.client = Groq(api_key=settings.GROQ_API_KEY)
        return self.client
    
    def generate_response(self, question: str, context_chunks: list[dict]) -> str:
        client = self.connect()
        
        # Build context from chunks
        context = "\n\n---\n\n".join([
            f"[Page {chunk['page_number']}]\n{chunk['text']}" 
            for chunk in context_chunks
        ])
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided document context.
Answer in the same language as the question.
If the answer is not in the context, say so.
Be concise and direct.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024
        )
        
        return response.choices[0].message.content


llm_service = LLMService()
