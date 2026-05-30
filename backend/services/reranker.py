"""
Cross-Encoder Reranker Service - Re-ranks retrieval results using cross-encoder model.
"""
import logging
from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import CrossEncoder

from backend.models.schemas import Chunk, Paper

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranking for improved relevance.

    Uses cross-encoder/ms-marco-MiniLM-L-6-v2 for scoring
    (query, document) pairs. Combines cross-encoder scores
    with hybrid retrieval scores for final ranking.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None

    def load(self):
        """Load the cross-encoder model (lazy loading)."""
        if self.model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Paper, float, Chunk]],
        top_k: int,
    ) -> List[Tuple[Paper, float]]:
        """
        Rerank candidate papers using cross-encoder.

        Args:
            query: Search query
            candidates: List of (Paper, hybrid_score, chunk) tuples
            top_k: Number of results to return

        Returns:
            List of (Paper, final_score) tuples sorted by final score descending
        """
        if not candidates:
            return []

        self.load()

        # Prepare (query, document) pairs for cross-encoder
        # Use paper title + abstract as the document
        pairs = []
        for paper, hybrid_score, chunk in candidates:
            # Construct document text from paper info
            title = paper.title or ""
            abstract = paper.abstract or chunk.text
            document = f"{title}\n\n{abstract}"
            pairs.append((query, document))

        # Get cross-encoder scores
        cross_scores = self.model.predict(pairs)

        # Normalize cross-scores to 0-1 range
        if isinstance(cross_scores, np.ndarray):
            cross_scores = cross_scores.flatten()

        min_score = cross_scores.min()
        max_score = cross_scores.max()
        if max_score > min_score:
            normalized_cross = (cross_scores - min_score) / (max_score - min_score)
        else:
            normalized_cross = np.ones_like(cross_scores) * 0.5

        # Combine cross-encoder scores with hybrid scores
        # final = 0.5 * normalized_cross + 0.5 * normalized_hybrid
        hybrid_scores = np.array([score for _, score, _ in candidates])

        # Normalize hybrid scores
        min_hybrid = hybrid_scores.min()
        max_hybrid = hybrid_scores.max()
        if max_hybrid > min_hybrid:
            normalized_hybrid = (hybrid_scores - min_hybrid) / (max_hybrid - min_hybrid)
        else:
            normalized_hybrid = np.ones_like(hybrid_scores) * 0.5

        # Combine scores
        final_scores = 0.5 * normalized_cross + 0.5 * normalized_hybrid

        # Create (paper, final_score) tuples
        reranked = []
        for i, (paper, _, _) in enumerate(candidates):
            reranked.append((paper, float(final_scores[i])))

        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Reranked {len(candidates)} candidates, returning top {top_k}")

        return reranked[:top_k]

    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.

        Args:
            query: Query string
            document: Document string

        Returns:
            Cross-encoder score
        """
        self.load()
        scores = self.model.predict([(query, document)])
        return float(scores[0])


# Global reranker instance
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker() -> CrossEncoderReranker:
    """Get or create the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker