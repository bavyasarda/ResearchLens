"""
Pydantic models for all request/response shapes in ResearchLens.
"""
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request model for the main search endpoint."""
    query: str = Field(..., description="User's keywords or problem statement")
    num_papers: int = Field(default=10, ge=1, le=30, description="Number of papers to fetch")
    preference: Literal["recent", "most_cited", "balanced"] = Field(
        default="balanced",
        description="Sorting preference for paper retrieval"
    )
    year_from: Optional[int] = Field(default=None, description="Filter papers from this year")
    year_to: Optional[int] = Field(default=None, description="Filter papers up to this year")


class Paper(BaseModel):
    """Model representing a research paper."""
    paper_id: str = Field(..., description="Unique paper identifier")
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    year: int = Field(..., description="Publication year")
    abstract: str = Field(..., description="Paper abstract text")
    citation_count: int = Field(default=0, description="Number of citations")
    url: str = Field(..., description="Paper URL")
    pdf_url: Optional[str] = Field(default=None, description="PDF download URL")
    venue: Optional[str] = Field(default=None, description="Journal or conference name")
    source: str = Field(..., description="Source API: semantic_scholar, arxiv, or core")


class PaperSummary(BaseModel):
    """LLM-generated summary of a paper."""
    paper_id: str = Field(..., description="Reference to paper")
    summary: str = Field(..., description="3-5 sentence plain-English summary")
    key_methodology: str = Field(..., description="Primary method/approach used")
    key_findings: str = Field(..., description="Most important results or conclusions")
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance to user query (0-1)"
    )
    relevance_reason: str = Field(..., description="Why this paper matches the query")


class CompareRequest(BaseModel):
    """Request model for comparative analysis."""
    query: str
    papers: List[Paper]
    summaries: List[PaperSummary]


class ComparisonTable(BaseModel):
    """Comparative methodology table with alignment analysis."""
    headers: List[str] = Field(
        default_factory=list,
        description="Table column headers"
    )
    rows: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Table rows with column values"
    )
    alignment_analysis: str = Field(
        ...,
        description="LLM paragraph on how papers align with query"
    )


class ChatRequest(BaseModel):
    """Request model for follow-up chat."""
    message: str = Field(..., description="User's follow-up question")
    context_papers: List[Paper] = Field(..., description="Papers from last search")
    context_summaries: List[PaperSummary] = Field(..., description="Summaries from last search")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Conversation history [{role, content}]"
    )


class SearchResponse(BaseModel):
    """Response model for the main search endpoint."""
    papers: List[Paper] = Field(..., description="Retrieved papers")
    summaries: List[PaperSummary] = Field(..., description="LLM-generated summaries")
    comparison: ComparisonTable = Field(..., description="Comparative analysis table")
    expanded_query: str = Field(..., description="LLM-expanded search query")
    total_fetched: int = Field(..., description="Total papers fetched from APIs")
    retrieval_method: str = Field(
        default="hybrid_rag",
        description="Retrieval method used"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="LLM-generated response")
    sources: Optional[List[str]] = Field(
        default=None,
        description="Referenced paper titles in response"
    )


class QueryExpansion(BaseModel):
    """LLM-generated query expansion."""
    expanded_query: str = Field(..., description="Optimized search query")
    key_concepts: List[str] = Field(default_factory=list, description="Core concepts")
    related_terms: List[str] = Field(default_factory=list, description="Synonyms for search")
    exclusion_terms: List[str] = Field(default_factory=list, description="Terms to exclude")


class Chunk(BaseModel):
    """A chunk of text from a paper for embedding and retrieval."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    paper_id: str = Field(..., description="Parent paper ID")
    text: str = Field(..., description="Actual chunk text (for display)")
    context_text: str = Field(..., description="Text with surrounding context (for embedding)")
    position: int = Field(..., description="Order in document")
    metadata: Dict = Field(
        default_factory=dict,
        description="Paper metadata (year, authors, venue, citation_count)"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    embedder_loaded: bool = Field(default=False)
    version: str = Field(default="1.0.0")