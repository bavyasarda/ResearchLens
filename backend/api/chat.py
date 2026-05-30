"""
Chat API - Follow-up conversation endpoint
Uses RAG context (papers, summaries, chunks) for accurate responses.
"""
import logging
from typing import List, Optional
import anthropic

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, LLM_MODEL
from backend.models.schemas import Paper, PaperSummary

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

# Initialize Anthropic client
_client: anthropic.Anthropic = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    base_url=ANTHROPIC_BASE_URL,
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    context_papers: List[Paper]
    context_summaries: List[PaperSummary]
    history: List[ChatMessage] = []
    chunks: Optional[List[dict]] = []  # Optional RAG chunks for deeper context


class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []


SYSTEM_PROMPT = """You are ResearchLens, an AI research assistant. Use the provided paper context to answer user questions accurately.

Guidelines:
- Reference specific papers by title when answering
- Use bullet points for clarity
- Use tables for comparisons
- If information isn't in the context, say "I don't have that information from the papers"
- Be concise but thorough

Always base your answers on the provided paper context."""


def _extract_text(response) -> str:
    """Extract text from Anthropic response - handles thinking blocks."""
    try:
        if response is None:
            logger.error("Response is None")
            return ""
        if not hasattr(response, 'content'):
            logger.error(f"Response has no content attribute: {type(response)}")
            return ""
        if response.content is None:
            logger.error("Response.content is None")
            return ""
        if not isinstance(response.content, list) or len(response.content) == 0:
            logger.error(f"Response.content is empty or invalid: {response.content}")
            return ""

        # Find the text block (skip thinking blocks)
        for block in response.content:
            # Check block type
            block_type = getattr(block, 'type', None)
            if block_type == 'text':
                # This is the actual text response
                text = getattr(block, 'text', None)
                if text:
                    return text
            elif block_type == 'thinking':
                # Skip thinking blocks - they have text=None
                continue

        # Fallback: try first block if it has text
        block = response.content[0]
        text = getattr(block, 'text', None)
        if text:
            return text

        logger.error("No text found in response blocks")
        return ""
    except Exception as e:
        logger.error(f"Error extracting response: {e}", exc_info=True)
        return ""


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Handle chat about retrieved papers using RAG context."""
    try:
        # Build comprehensive RAG context
        context = _build_rag_context(req.context_papers, req.context_summaries, req.chunks)

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"RAG CONTEXT:\n{context}"},
        ]

        # Add conversation history (last 10 exchanges max)
        history = req.history[-20:] if req.history else []
        for msg in history:
            if msg.role in ["user", "assistant"]:
                messages.append({"role": msg.role, "content": msg.content})

        # Add current message
        messages.append({"role": "user", "content": req.message})

        logger.info(f"Chat request with {len(req.context_papers)} papers, history: {len(history)} messages")
        logger.info(f"Calling Claude Haiku with model: {LLM_MODEL}")

        # Call Claude Haiku
        response = _client.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            temperature=0.3,
            messages=messages,
        )

        logger.info(f"Claude response type: {type(response)}")
        logger.info(f"Claude response attributes: {dir(response)}")
        logger.info(f"Claude response: {response}")

        # Log the full response structure for debugging
        try:
            import json
            logger.info(f"Claude response dict: {json.dumps(response, default=str, indent=2)[:2000]}")
        except:
            pass

        response_text = _extract_text(response)

        logger.info(f"Extracted text length: {len(response_text)}")

        if not response_text:
            response_text = "I couldn't generate a response. Please try again."

        # Extract referenced papers
        sources = _extract_sources(response_text, req.context_papers)

        logger.info(f"Chat response generated, sources: {len(sources)}")

        return ChatResponse(
            response=response_text,
            sources=sources,
        )

    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        return ChatResponse(
            response="I encountered an error. Please try again.",
            sources=[],
        )


def _build_rag_context(papers: List[Paper], summaries: List[PaperSummary], chunks: List[dict] = None) -> str:
    """
    Build comprehensive RAG context from papers, summaries, and optional chunks.
    This provides the LLM with all retrieved information for accurate answers.
    """
    summary_map = {s.paper_id: s for s in summaries}
    parts = []

    for i, paper in enumerate(papers):
        summary = summary_map.get(paper.paper_id)
        paper_id = paper.paper_id or f"Paper {i+1}"

        # Paper header
        part = f"=== {paper_id} ===\n"
        part += f"Title: {paper.title}\n"

        # Authors and metadata
        authors = paper.authors[:3] if paper.authors else ["Unknown"]
        author_str = ", ".join(authors)
        if len(paper.authors) > 3:
            author_str += " et al."
        part += f"Authors: {author_str}\n"
        part += f"Year: {paper.year} | Source: {paper.source}\n"

        if paper.venue and paper.venue != 'arXiv':
            part += f"Venue: {paper.venue}\n"

        if paper.citation_count > 0:
            part += f"Citations: {paper.citation_count}\n"

        # AI-generated summary and analysis
        if summary:
            part += f"\n--- AI Analysis ---\n"
            part += f"Summary: {summary.summary}\n"
            part += f"Methodology: {summary.key_methodology}\n"
            part += f"Key Findings: {summary.key_findings}\n"
            part += f"Relevance: {summary.relevance_score:.0%} - {summary.relevance_reason}\n"

        # Include relevant chunks if available
        if chunks:
            paper_chunks = [c for c in chunks if c.get('paper_id') == paper.paper_id]
            if paper_chunks:
                part += f"\n--- Relevant Sections ---\n"
                for chunk in paper_chunks[:2]:  # Max 2 chunks per paper
                    part += f"• {chunk.get('text', '')[:300]}...\n"

        # Include abstract for additional context
        if paper.abstract:
            part += f"\n--- Abstract ---\n"
            part += f"{paper.abstract[:500]}...\n"

        part += "\n"
        parts.append(part)

    return "\n".join(parts)


def _extract_sources(response: str, papers: List[Paper]) -> List[str]:
    """Extract paper titles referenced in the response."""
    referenced = []
    response_lower = response.lower()

    for paper in papers:
        # Check if paper title words appear in response
        title_words = paper.title.lower().split()[:4]
        matches = sum(1 for word in title_words if word in response_lower and len(word) > 3)

        if matches >= 2:  # At least 2 significant words match
            if paper.title not in referenced:
                referenced.append(paper.title)

    return referenced[:5]  # Limit to 5 sources