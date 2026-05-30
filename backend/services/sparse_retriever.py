"""
Sparse Retriever Service - BM25-based sparse retrieval.
"""
import logging
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi

from backend.models.schemas import Chunk

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25 sparse retrieval for keyword-based search.

    BM25 (Best Matching 25) is a probabilistic ranking function used
    for text retrieval. It handles term frequency saturation and
    document length normalization.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize the BM25 retriever.

        Args:
            k1: Term frequency saturation parameter (typical range: 1.2-2.0)
            b: Document length normalization parameter (typical range: 0.5-0.75)
        """
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_map: dict = {}  # index -> chunk
        self._tokenized_chunks: List[List[str]] = []

    def index(self, chunks: List[Chunk]):
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of Chunk objects to index
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return

        # Tokenize each chunk
        self._tokenized_chunks = []
        self.chunk_map = {}

        for i, chunk in enumerate(chunks):
            # Simple tokenization: lowercase, split on non-alphanumeric
            tokens = self._tokenize(chunk.text)
            self._tokenized_chunks.append(tokens)
            self.chunk_map[i] = chunk

        # Build BM25 index
        self.bm25 = BM25Okapi(self._tokenized_chunks)
        logger.info(f"BM25 index built with {len(chunks)} chunks")

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer for BM25.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Lowercase and split on non-alphanumeric characters
        import re
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())

        # Remove very short tokens (single characters)
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    def retrieve(
        self,
        query: str,
        top_k: int,
        related_terms: Optional[List[str]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k chunks using BM25.

        Args:
            query: Search query
            top_k: Number of results to return
            related_terms: Optional list of related/synonym terms to include

        Returns:
            List of (Chunk, bm25_score) tuples sorted by score descending
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built. Returning empty results.")
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Add related terms if provided
        if related_terms:
            for term in related_terms:
                term_tokens = self._tokenize(term)
                query_tokens.extend(term_tokens)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Create list of (index, score) sorted by score
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        results = []
        for idx, score in indexed_scores[:top_k]:
            if idx in self.chunk_map:
                results.append((self.chunk_map[idx], float(score)))

        return results

    def get_scores_for_chunks(
        self,
        query: str,
        chunks: List[Chunk],
    ) -> List[Tuple[Chunk, float]]:
        """
        Get BM25 scores for specific chunks (used in hybrid retrieval).

        Args:
            query: Search query
            chunks: List of specific chunks to score

        Returns:
            List of (Chunk, bm25_score) tuples
        """
        if not chunks or self.bm25 is None:
            return []

        # Build a temporary index for the specific chunks
        tokenized = [self._tokenize(chunk.text) for chunk in chunks]
        temp_bm25 = BM25Okapi(tokenized)

        query_tokens = self._tokenize(query)
        scores = temp_bm25.get_scores(query_tokens)

        return [(chunk, float(score)) for chunk, score in zip(chunks, scores)]