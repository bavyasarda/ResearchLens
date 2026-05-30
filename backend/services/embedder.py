"""
Dense Embedder Service - Uses sentence-transformers for dense embeddings.
"""
import logging
from typing import Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import EMBED_MODEL
from backend.models.schemas import Chunk

logger = logging.getLogger(__name__)


class DenseEmbedder:
    """
    Dense embedding service using sentence-transformers.

    Uses BAAI/bge-base-en-v1.5 by default for high-quality embeddings.
    Normalizes embeddings for cosine similarity computations.
    Caches embeddings in-memory per session.
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        """
        Initialize the dense embedder.

        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def load(self):
        """Load the embedding model (lazy loading for startup performance)."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")

    def embed_chunks(self, chunks: List[Chunk]) -> Dict[str, np.ndarray]:
        """
        Embed all chunks and return a mapping of chunk_id to embedding.

        Args:
            chunks: List of Chunk objects to embed

        Returns:
            Dictionary mapping chunk_id to normalized embedding vector
        """
        self.load()

        if not chunks:
            return {}

        # Check cache for already-embedded chunks
        uncached_chunks = []
        chunk_ids = []
        embeddings_dict = {}

        for chunk in chunks:
            if chunk.chunk_id in self._embedding_cache:
                embeddings_dict[chunk.chunk_id] = self._embedding_cache[chunk.chunk_id]
            else:
                uncached_chunks.append(chunk)
                chunk_ids.append(chunk.chunk_id)

        if not uncached_chunks:
            logger.info(f"All {len(chunks)} chunks found in cache")
            return embeddings_dict

        logger.info(f"Embedding {len(uncached_chunks)} new chunks (cache hit: {len(chunks) - len(uncached_chunks)})")

        # Get context_text for embedding (enriched with context window)
        texts = [chunk.context_text for chunk in uncached_chunks]

        # Batch encode all texts
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # Normalize for cosine similarity
        )

        # Store in cache and return dict
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            self._embedding_cache[chunk_id] = embedding
            embeddings_dict[chunk_id] = embedding

        return embeddings_dict

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.

        Args:
            query: Query text to embed

        Returns:
            Normalized embedding vector for the query
        """
        self.load()

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return embedding

    def compute_similarity(self, query_embedding: np.ndarray, chunk_embedding: np.ndarray) -> float:
        """
        Compute cosine similarity between query and chunk embeddings.

        Args:
            query_embedding: Query embedding vector
            chunk_embedding: Chunk embedding vector

        Returns:
            Cosine similarity score (already normalized, so dot product = cosine)
        """
        return float(np.dot(query_embedding, chunk_embedding))

    def get_top_chunks(
        self,
        query: str,
        chunk_embeddings: Dict[str, np.ndarray],
        chunks: List[Chunk],
        top_k: int,
    ) -> List[tuple]:
        """
        Get top-k chunks by cosine similarity to query.

        Args:
            query: Query string
            chunk_embeddings: Dictionary of chunk_id to embedding
            chunks: List of Chunk objects
            top_k: Number of top results to return

        Returns:
            List of (Chunk, similarity_score) tuples sorted by score descending
        """
        if not chunks:
            return []

        query_embedding = self.embed_query(query)

        # Create chunk_id to chunk map
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}

        # Compute similarities
        similarities = []
        for chunk_id, embedding in chunk_embeddings.items():
            if chunk_id in chunk_map:
                sim = self.compute_similarity(query_embedding, embedding)
                similarities.append((chunk_map[chunk_id], sim))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")


# Global embedder instance
_embedder: Optional[DenseEmbedder] = None


def get_embedder() -> DenseEmbedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = DenseEmbedder()
    return _embedder


def preload_embedder():
    """Preload the embedder model at startup."""
    embedder = get_embedder()
    embedder.load()
    logger.info("Embedder preloaded")