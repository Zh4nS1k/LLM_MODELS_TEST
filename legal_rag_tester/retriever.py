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
        
    def query(self, embedding: List[float], top_k: int = None) -> List[Chunk]:
        """Queries the Pinecone index with the given embedding."""
        query_dim = len(embedding)
        assert self.index_dim == query_dim, (
            f"DIMENSION MISMATCH: index={self.index_dim}, embedder={query_dim}. "
            f"Wrong embedding model configured."
        )
        k = top_k or self.settings.pinecone_top_k
        
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
            
        return chunks