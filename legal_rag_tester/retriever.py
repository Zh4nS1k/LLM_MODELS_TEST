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
        
    def query(self, embedding: List[float], top_k: int = None) -> List[Chunk]:
        """Queries the Pinecone index with the given embedding."""
        k = top_k or self.settings.pinecone_top_k
        
        response = self.index.query(
            namespace=self.settings.pinecone_namespace,
            vector=embedding,
            top_k=k,
            include_values=False,
            include_metadata=True
        )
        
        chunks = []
        for match in response.get("matches", []):
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
        
        if len(chunks) < 2:
            pipeline_logger.log_warning(f"Retrieved fewer than 2 chunks (found {len(chunks)})")
            
        return chunks