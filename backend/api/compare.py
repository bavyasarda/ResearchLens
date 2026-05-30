"""
Compare API - Comparative methodology table generation endpoint.
"""
import logging
from fastapi import APIRouter, HTTPException

from backend.models.schemas import CompareRequest, ComparisonTable

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/compare", tags=["compare"])


@router.post("", response_model=ComparisonTable)
async def compare(req: CompareRequest):
    """
    Generate a comparative methodology table for selected papers.

    Args:
        req: CompareRequest containing query, papers, and summaries

    Returns:
        ComparisonTable with headers, rows, and alignment analysis
    """
    from backend.services.comparator import generate_comparison

    try:
        logger.info(f"Comparing {len(req.papers)} papers")
        comparison = await generate_comparison(
            query=req.query,
            papers=req.papers,
            summaries=req.summaries,
        )
        return comparison
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")