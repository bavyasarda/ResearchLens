"""
Summarize API - Per-paper summarization endpoint.
"""
import logging
from fastapi import APIRouter, HTTPException

from backend.models.schemas import Paper, PaperSummary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/summarize", tags=["summarize"])


@router.post("", response_model=PaperSummary)
async def summarize(
    paper: Paper,
    user_query: str,
):
    """
    Generate an LLM summary for a single paper.

    Args:
        paper: Paper object to summarize
        user_query: Original user query for context

    Returns:
        PaperSummary with summary, methodology, findings, and relevance
    """
    from backend.services.summarizer import summarize_paper

    try:
        logger.info(f"Summarizing paper: {paper.title[:50]}...")
        summary = await summarize_paper(paper, user_query)
        return summary
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")