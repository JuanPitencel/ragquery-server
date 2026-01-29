import httpx
from app.config import settings


class LLMService:
    def __init__(self):
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
    
    async def translate_to_english(self, text: str) -> str:
        """Translate text to English if not already in English."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.groq_url,
                headers=self.headers,
                json={
                    "model": settings.LLM_MODEL,
                    "messages": [{
                        "role": "user",
                        "content": f"""If the following text is in English, return it exactly as is.
If it's in another language, translate it to English.
Return ONLY the text, nothing else.

Text: {text}"""
                    }],
                    "temperature": 0.1,
                    "max_tokens": 256
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
    
    async def generate_response(self, question: str, context_chunks: list[dict]) -> str:
        """Generate a response using the LLM."""
        context = "\n\n---\n\n".join([
            f"[Page {chunk['page_number']}]\n{chunk['text']}" 
            for chunk in context_chunks
        ])
        
        prompt = f"""You are a helpful Toyota Corolla Cross Hybrid assistant. Answer questions based on the provided owner's manual content.

INSTRUCTIONS:
- Provide detailed and complete answers using all relevant information from the context
- Include specific steps, warnings, and important notes when available
- If there are safety warnings, always include them
- Answer in the same language as the question
- If the answer is not in the context, say so

CONTEXT FROM OWNER'S MANUAL:
{context}

QUESTION: {question}

DETAILED ANSWER:"""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.groq_url,
                headers=self.headers,
                json={
                    "model": settings.LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 2048
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


llm_service = LLMService()
