from groq import Groq
from app.config import settings


class LLMService:
    def __init__(self):
        self.client = None
    
    def connect(self):
        if self.client is None:
            self.client = Groq(api_key=settings.GROQ_API_KEY)
        return self.client
    
    def translate_to_english(self, text: str) -> str:
        """Translate text to English if not already in English."""
        client = self.connect()
        
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{
                "role": "user", 
                "content": f"""If the following text is in English, return it exactly as is.
If it's in another language, translate it to English.
Return ONLY the text, nothing else.

Text: {text}"""
            }],
            temperature=0.1,
            max_tokens=256
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_response(self, question: str, context_chunks: list[dict]) -> str:
        client = self.connect()
        
        # Build context from chunks
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
        
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048
        )
        
        return response.choices[0].message.content


llm_service = LLMService()
