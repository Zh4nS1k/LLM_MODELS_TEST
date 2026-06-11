"""Pinecone retriever module."""
from pinecone import Pinecone
from typing import List
from logger import pipeline_logger
from config import Settings
from models.schemas import Chunk

class PineconeRetriever:
    """Handles vector search queries to Pinecone."""
    
    def __init__(self, settings: Settings):
        """Initializes the retriever with Pinecone credentials."""
        self.settings = settings
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.pc.Index(settings.pinecone_index_name)
        
        stats = self.index.describe_index_stats()
        self.index_dim = stats.dimension
        namespaces = list(stats.namespaces.keys())
        if self.settings.pinecone_namespace and self.settings.pinecone_namespace not in namespaces:
            pipeline_logger.log_warning(
                f"⚠️  Namespace '{self.settings.pinecone_namespace}' not found in index. "
                f"Available namespaces: {namespaces}. "
                f"Falling back to default namespace."
            )
        
    def _text_overlap(self, a: str, b: str) -> float:
        """Jaccard similarity on word sets as diversity proxy."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def _mmr_rerank(self, chunks: list[Chunk], final_k: int, lambda_param: float = 0.7) -> list[Chunk]:
        if len(chunks) <= final_k:
            return chunks

        selected = []
        remaining = list(chunks)

        while len(selected) < final_k and remaining:
            if not selected:
                best = remaining.pop(0)
                selected.append(best)
                continue

            best_chunk = None
            best_score = -1.0

            for candidate in remaining:
                relevance = candidate.score
                max_overlap = max(self._text_overlap(candidate.text, sel.text) for sel in selected)
                diversity = 1.0 - max_overlap
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_chunk = candidate

            remaining.remove(best_chunk)
            selected.append(best_chunk)

        return selected

    def _keyword_boost(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        import re
        articles = re.findall(r'[Сс]татья\s+\d+[\-\d]*', query)
        law_keywords = re.findall(
            r'(Гражданск\w+\s+[Кк]одекс|ГК\s+РК|ТК\s+РК|УК\s+РК|'
            r'адвокатск\w+|потребител\w+|трудов\w+|налогов\w+|'
            r'административн\w+|семейн\w+|жилищн\w+)',
            query, re.IGNORECASE
        )
        all_terms = [a.lower() for a in articles] + [k.lower() for k in law_keywords]

        if not all_terms:
            return chunks

        for chunk in chunks:
            text_lower = chunk.text.lower()
            matches = sum(1 for term in all_terms if term in text_lower)
            if matches > 0:
                chunk.score = min(1.0, chunk.score + min(matches * 0.05, 0.15))
                chunk.metadata["keyword_boost"] = matches

        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def query(self, embedding: List[float], query_text: str = "") -> List[Chunk]:
        """Queries the Pinecone index with the given embedding."""
        query_dim = len(embedding)
        assert self.index_dim == query_dim, (
            f"DIMENSION MISMATCH: index={self.index_dim}, embedder={query_dim}. "
            f"Wrong embedding model configured."
        )
        k = self.settings.pinecone_top_k
        final_k = self.settings.pinecone_final_k
        
        response = self.index.query(
            namespace=self.settings.pinecone_namespace,
            vector=embedding,
            top_k=k,
            include_values=False,
            include_metadata=True
        )
        
        matches = response.get("matches", [])
        if matches:
            sample_keys = list(matches[0].metadata.keys())
            pipeline_logger.log_simple_info(f"📄 Pinecone metadata keys: {sample_keys}")

        chunks = []
        for match in matches:
            score = match.get("score", 0.0)
            
            if score < self.settings.pinecone_score_threshold:
                continue
                
            metadata = match.get("metadata", {})
            text = metadata.get(self.settings.chunk_text_field, "")
            
            if not text:
                continue
                
            chunks.append(Chunk(
                id=match.get("id", ""),
                text=text,
                score=score,
                metadata=metadata
            ))
            
        chunks.sort(key=lambda x: x.score, reverse=True)
        
        if query_text:
            chunks = self._keyword_boost(query_text, chunks)
            
        chunks = self._mmr_rerank(chunks, final_k)
            
        return chunks