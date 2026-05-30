"""
Hybrid Retriever Service - Combines dense and sparse retrieval with RRF fusion.
"""
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np

from backend.config import RRF_K, DENSE_WEIGHT, SPARSE_WEIGHT
from backend.models.schemas import Chunk, Paper
from backend.services.embedder import DenseEmbedder
from backend.services.sparse_retriever import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Modern Hybrid RAG retrieval combining dense + sparse with RRF fusion.

    Strategy:
    1. Dense retrieval: cosine similarity of query embedding vs chunk embeddings
    2. Sparse retrieval: BM25 on tokenized chunks
    3. RRF fusion: score = Σ 1/(k + rank_i) for each retrieval list
    4. Dedup by paper_id (keep highest-scoring chunk per paper)
    5. Return top-N unique papers

    RRF constant k=60 (standard)
    Dense weight: 0.6, Sparse weight: 0.4 (tunable)
    """

    def __init__(
        self,
        embedder: DenseEmbedder,
        bm25: BM25Retriever,
        rrf_k: int = RRF_K,
        dense_weight: float = DENSE_WEIGHT,
        sparse_weight: float = SPARSE_WEIGHT,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            embedder: DenseEmbedder instance
            bm25: BM25Retriever instance
            rrf_k: RRF constant (default 60)
            dense_weight: Weight for dense scores (default 0.6)
            sparse_weight: Weight for sparse scores (default 0.4)
        """
        self.embedder = embedder
        self.bm25 = bm25
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(
        self,
        query: str,
        expanded_terms: List[str],
        chunks: List[Chunk],
        embeddings: Dict[str, np.ndarray],
        top_k: int,
    ) -> List[Tuple[Paper, float, Chunk]]:
        """
        Perform hybrid retrieval and return top-k papers with scores.

        Args:
            query: Search query string
            expanded_terms: Related terms for query expansion
            chunks: List of all chunks
            embeddings: Dictionary of chunk_id to embedding vectors
            top_k: Number of results to return

        Returns:
            List of (Paper, relevance_score, best_matching_chunk) tuples
        """
        if not chunks:
            logger.warning("No chunks provided for retrieval")
            return []

        # 1. Dense retrieval
        dense_results = self._dense_retrieve(query, chunks, embeddings, top_k * 2)

        # 2. Sparse retrieval
        sparse_results = self._sparse_retrieve(query, expanded_terms, top_k * 2)

        # 3. RRF fusion
        fused_scores = self._rrf_fusion(dense_results, sparse_results, top_k)

        # 4. Deduplicate by paper_id and return top-k
        paper_scores = self._deduplicate_by_paper(fused_scores, top_k)

        logger.info(f"Hybrid retrieval returned {len(paper_scores)} unique papers")

        return paper_scores

    def _dense_retrieve(
        self,
        query: str,
        chunks: List[Chunk],
        embeddings: Dict[str, np.ndarray],
        top_k: int,
    ) -> List[Tuple[Chunk, float, int]]:
        """
        Perform dense retrieval using cosine similarity.

        Returns:
            List of (Chunk, score, rank) tuples
        """
        # Create chunk_id to chunk map
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}

        # Get top chunks by similarity
        top_chunks = self.embedder.get_top_chunks(query, embeddings, chunks, top_k * 3)

        # Convert to (chunk, score, rank) format
        results = []
        for rank, (chunk, score) in enumerate(top_chunks[:top_k]):
            results.append((chunk, float(score), rank))

        return results

    def _sparse_retrieve(
        self,
        query: str,
        related_terms: List[str],
        top_k: int,
    ) -> List[Tuple[Chunk, float, int]]:
        """
        Perform sparse retrieval using BM25.

        Returns:
            List of (Chunk, score, rank) tuples
        """
        sparse_results = self.bm25.retrieve(query, top_k * 3, related_terms)

        # Convert to (chunk, score, rank) format
        results = []
        for rank, (chunk, score) in enumerate(sparse_results[:top_k]):
            results.append((chunk, float(score), rank))

        return results

    def _rrf_fusion(
        self,
        dense_results: List[Tuple[Chunk, float, int]],
        sparse_results: List[Tuple[Chunk, float, int]],
        top_k: int,
    ) -> List[Tuple[Chunk, float]]:
        """
        Reciprocal Rank Fusion combining dense and sparse results.

        RRF formula: score = Σ 1/(k + rank)

        Args:
            dense_results: (chunk, score, rank) from dense retrieval
            sparse_results: (chunk, score, rank) from sparse retrieval
            top_k: Number of results to return

        Returns:
            List of (Chunk, fused_score) sorted by score descending
        """
        # Build score maps
        chunk_scores: Dict[str, float] = {}

        # Add dense scores with weight
        for chunk, score, rank in dense_results:
            rrf_score = 1.0 / (self.rrf_k + rank)
            chunk_id = chunk.chunk_id

            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] += self.dense_weight * rrf_score
            else:
                chunk_scores[chunk_id] = self.dense_weight * rrf_score

        # Add sparse scores with weight
        for chunk, score, rank in sparse_results:
            rrf_score = 1.0 / (self.rrf_k + rank)
            chunk_id = chunk.chunk_id

            if chunk_id in chunk_scores:
                chunk_scores[chunk_id] += self.sparse_weight * rrf_score
            else:
                chunk_scores[chunk_id] = self.sparse_weight * rrf_score

        # Sort by fused score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

        # Create chunk_id to chunk map for lookup
        all_chunks = {r[0].chunk_id: r[0] for r in dense_results + sparse_results}

        # Return top-k chunks with scores
        results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            if chunk_id in all_chunks:
                results.append((all_chunks[chunk_id], score))

        return results

    def _deduplicate_by_paper(
        self,
        chunk_scores: List[Tuple[Chunk, float]],
        top_k: int,
    ) -> List[Tuple[Paper, float, Chunk]]:
        """
        Deduplicate chunks by paper_id, keeping the highest-scoring chunk per paper.

        Args:
            chunk_scores: (chunk, score) tuples
            top_k: Maximum number of papers to return

        Returns:
            List of (Paper, score, chunk) tuples (one per paper)
        """
        paper_best: Dict[str, Tuple[Chunk, float]] = {}

        for chunk, score in chunk_scores:
            paper_id = chunk.paper_id
            if paper_id not in paper_best or score > paper_best[paper_id][1]:
                paper_best[paper_id] = (chunk, score)

        # Convert to (Paper, score, chunk) format
        paper_chunks: List[Tuple[Paper, float, Chunk]] = []

        # We need access to the full paper data
        # For now, reconstruct paper info from chunk metadata
        for paper_id, (chunk, score) in list(paper_best.items())[:top_k]:
            metadata = chunk.metadata
            # Create a minimal Paper object from chunk metadata
            paper = Paper(
                paper_id=paper_id,
                title=metadata.get("title", ""),
                authors=metadata.get("authors", []),
                year=metadata.get("year", 2024),
                abstract=chunk.text,  # Use chunk text as abstract
                citation_count=metadata.get("citation_count", 0),
                url="",
                source=metadata.get("source", "unknown"),
            )
            paper_chunks.append((paper, score, chunk))

        # Sort by score
        paper_chunks.sort(key=lambda x: x[1], reverse=True)

        return paper_chunks[:top_k]


def create_hybrid_retriever(embedder: DenseEmbedder, bm25: BM25Retriever) -> HybridRetriever:
    """Factory function to create a HybridRetriever instance."""
    return HybridRetriever(embedder=embedder, bm25=bm25)