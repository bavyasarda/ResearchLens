"""
Query Expander Service - Uses LLM to expand user queries with academic vocabulary.
"""
import json
import re
import logging
import anthropic

from backend.config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, LLM_MODEL
from backend.models.schemas import QueryExpansion

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    base_url=ANTHROPIC_BASE_URL,
)

QUERY_EXPANSION_PROMPT = """Convert this search query for academic paper search:

"{raw_query}"

Return ONLY JSON (no text):
{{"expanded_query":"search terms","key_concepts":["term1","term2"],"related_terms":["term3"],"exclusion_terms":[]}}"""


def _extract_text_from_response(response) -> str:
    """Extract text from Anthropic API response - handles thinking blocks."""
    try:
        if response is None or not hasattr(response, 'content') or response.content is None:
            return ""
        content = response.content
        if isinstance(content, list) and len(content) > 0:
            # Find the text block (skip thinking blocks)
            for block in content:
                block_type = getattr(block, 'type', None)
                if block_type == 'text':
                    text = getattr(block, 'text', None)
                    if text:
                        return text
            # Fallback: try first block
            block = content[0]
            text = getattr(block, 'text', None)
            if text:
                return text
            return ""
        elif isinstance(content, str):
            return content
        return ""
    except Exception as e:
        logger.error(f"Error: {e}")
        return ""


def _parse_json_response(text: str) -> dict:
    """Parse JSON from response."""
    if not text:
        raise ValueError("Empty response")

    text = text.strip()
    start = text.find('{')
    end = text.rfind('}') + 1
    if start >= 0 and end > start:
        text = text[start:end]
    else:
        text = text.replace('```json', '').replace('```', '').strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r',(\s*[}]])', r'\1', text)
        try:
            return json.loads(text)
        except:
            raise ValueError("Failed to parse JSON")


async def expand_query(raw_query: str) -> QueryExpansion:
    """Use LLM to expand and optimize the user's query."""
    logger.info(f"Expanding query: '{raw_query}'")

    try:
        response = _client.messages.create(
            model=LLM_MODEL,
            max_tokens=512,
            temperature=0.3,
            messages=[{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(raw_query=raw_query)}]
        )

        response_text = _extract_text_from_response(response)

        if not response_text:
            logger.warning("Empty LLM response, using original query")
            return _smart_expand(raw_query)

        result = _parse_json_response(response_text)

        expansion = QueryExpansion(
            expanded_query=result.get("expanded_query", raw_query),
            key_concepts=result.get("key_concepts", []),
            related_terms=result.get("related_terms", []),
            exclusion_terms=result.get("exclusion_terms", []),
        )

        logger.info(f"Query expanded to: '{expansion.expanded_query}'")
        return expansion

    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return _smart_expand(raw_query)


def _smart_expand(query: str) -> QueryExpansion:
    """
    Smart fallback expansion based on query analysis.
    """
    query_lower = query.lower()

    # Common expansions for technical terms
    expansions = {
        "transformer": ["attention mechanism", "self-attention", "BERT", "GPT", "neural network"],
        "neural network": ["deep learning", "machine learning", "neural", "ANN"],
        "machine learning": ["ML", "deep learning", "algorithm", "training"],
        "deep learning": ["neural network", "CNN", "RNN", "deep neural"],
        "nlp": ["natural language processing", "text", "language", "NLP"],
        "computer vision": ["image", "visual", "object detection", "CNN"],
        "classification": ["supervised learning", "categorization", "class labels"],
        "prediction": ["forecasting", "regression", "estimation"],
    }

    # Build expansion
    concepts = [query]
    related = []

    for term, related_terms in expansions.items():
        if term in query_lower:
            concepts.append(term)
            related.extend(related_terms[:2])

    # Remove duplicates while preserving order
    seen = set()
    unique_concepts = []
    for c in concepts:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique_concepts.append(c)

    unique_related = []
    for r in related:
        if r.lower() not in seen:
            seen.add(r.lower())
            unique_related.append(r)

    return QueryExpansion(
        expanded_query=query,
        key_concepts=unique_concepts[:8],
        related_terms=unique_related[:5],
        exclusion_terms=[],
    )