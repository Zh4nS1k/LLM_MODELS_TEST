"""Prompt builder for constructing strict legal RAG prompts."""
from typing import List, Dict
from config import settings
from models.schemas import Chunk

class PromptBuilder:
    """Constructs strict RAG prompts with context chunks."""
    
    def __init__(self, max_tokens: int = settings.max_context_tokens):
        """Initializes the PromptBuilder."""
        self.max_tokens = max_tokens
        # Approximate 1 token ≈ 4 characters
        self.chars_per_token = 4
        
        self.system_prompt = (
            "You are a precise legal assistant. Your task is to answer legal questions strictly based on the provided document excerpts.\n\n"
            "Rules you must follow without exception:\n"
            "1. Answer ONLY using information explicitly stated in the provided context chunks.\n"
            "2. If the answer is not found in the context, respond exactly: \"Контекстте жауап жоқ.\" (or the language-appropriate equivalent: \"No answer found in context.\")\n"
            "3. Do NOT use your general knowledge, training data, or external information.\n"
            "4. Do NOT speculate, infer beyond what is written, or extrapolate.\n"
            "5. Cite the source chunk number(s) in your answer (e.g., \"[Chunk 2]\").\n"
            "6. Keep your answer concise and direct. Do not repeat the question.\n"
            "7. If multiple chunks are relevant, synthesize from all of them."
        )

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncates text to a maximum number of characters."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def build(self, question: str, chunks: List[Chunk]) -> Dict[str, str]:
        """Builds system and user prompts with truncated, sorted context chunks."""
        # Chunks are assumed to be sorted by score descending (from retriever)
        
        formatted_chunks = []
        chars_used = 0
        max_chars = self.max_tokens * self.chars_per_token
        
        for i, chunk in enumerate(chunks, 1):
            chunk_header = f"[Chunk {i}] (score: {chunk.score:.2f})\n"
            header_chars = len(chunk_header)
            
            if chars_used + header_chars >= max_chars:
                break
                
            allowed_chars = max_chars - chars_used - header_chars
            truncated_text = self._truncate_text(chunk.text, allowed_chars)
            
            formatted_chunk = f"{chunk_header}{truncated_text}"
            formatted_chunks.append(formatted_chunk)
            
            chars_used += len(formatted_chunk)
            
            if chars_used >= max_chars:
                break
                
        chunks_str = "\n\n".join(formatted_chunks)
        
        user_prompt = (
            f"Context documents:\n{chunks_str}\n\n"
            f"Legal question:\n{question}\n\n"
            "Answer strictly based on the context above:"
        )
        
        return {
            "system": self.system_prompt,
            "user": user_prompt
        }
