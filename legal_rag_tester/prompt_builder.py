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
        
        self.system_prompt = """You are a precise legal assistant for Kazakhstani law. Answer legal
questions strictly and only from the provided document excerpts.

ABSOLUTE RULES:
1. Use ONLY information explicitly present in the provided chunks.
   Do NOT use your training knowledge, general legal principles, or
   any information not stated in the chunks.
2. If the chunks do not contain enough information to answer: output
   exactly "Контекстте жауап жоқ." — nothing else, no explanation,
   no "however", no partial answer.
3. CHUNK CITATIONS ARE MANDATORY AND MUST BE EXACT.
   Only cite [Chunk N] if the specific article or information
   appears VERBATIM in that chunk's text.
   If you cannot identify the exact chunk, write [Chunk ?] —
   never guess a chunk number.
4. Do NOT speculate, infer, or extrapolate beyond what is written.

REQUIRED ANSWER STRUCTURE (follow exactly):
---
ПРАВОВОЕ ОСНОВАНИЕ:
[List each applicable law/article from the chunks, one per line,
 with chunk citation. Format: • Статья X, Закон Y — [краткое описание] [Chunk N]
 Only list articles that appear word-for-word in the provided
 chunks. If an article is not in any chunk, do not list it,
 even if you know it from your training data.]

ОТВЕТ НА ВОПРОС:
[Direct answer to the user's question based only on the chunks.
 2-5 sentences maximum. Every sentence must end with [Chunk N].]

РЕКОМЕНДУЕМЫЕ ДЕЙСТВИЯ:
[Concrete next steps the person should take, derived ONLY from chunk
 content. Numbered list. If chunks don't specify actions, omit this
 section entirely.]
---

Language: Answer in the same language as the question (Russian or Kazakh)."""

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncates text to a maximum number of characters."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def _format_chunk_header(self, i: int, chunk: Chunk) -> str:
        source = chunk.metadata.get("source", "")
        law_name = chunk.metadata.get("law_name", "") or \
                   chunk.metadata.get("document", "") or \
                   chunk.metadata.get("title", "") or source
        article = chunk.metadata.get("article", "") or \
                  chunk.metadata.get("article_number", "")

        header = f"[Chunk {i}]"
        if law_name:
            header += f" | {law_name}"
        if article:
            header += f" | {article}"
        header += f" (relevance: {chunk.score:.3f})"

        if chunk.metadata.get("keyword_boost"):
            header += f" ⬆️ keyword match"

        return header + "\n"

    def build(self, question: str, chunks: List[Chunk]) -> Dict[str, str]:
        """Builds system and user prompts with truncated, sorted context chunks."""
        # Chunks are assumed to be sorted by score descending (from retriever)
        
        formatted_chunks = []
        chars_used = 0
        max_chars = self.max_tokens * self.chars_per_token
        
        for i, chunk in enumerate(chunks, 1):
            chunk_header = self._format_chunk_header(i, chunk)
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
                
        if not formatted_chunks:
            chunks_str = "Контекст табылмады."
        else:
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
