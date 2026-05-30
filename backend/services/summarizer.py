"""
Summarizer Service - LLM-based per-paper summarization with smart fallback.
"""
import json
import re
import logging
import asyncio
from typing import List, Optional, Dict, Tuple
import anthropic

from backend.config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, LLM_MODEL
from backend.models.schemas import Paper, PaperSummary

logger = logging.getLogger(__name__)

# Initialize Anthropic client
_client: anthropic.Anthropic = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    base_url=ANTHROPIC_BASE_URL,
)

# Detailed prompt for comprehensive summaries
SUMMARY_PROMPT = """You are an expert research analyst. Analyze this paper and provide detailed, comprehensive information.

Title: {title}
Year: {year}
Authors: {authors}
Abstract: {abstract}

User's Research Query: {query}

Provide a detailed JSON response with complete information:

{{
  "summary": "A comprehensive 4-5 sentence summary explaining the paper's main contribution, approach, and significance. Include what problem it solves and why it matters.",
  "key_methodology": "Describe the FULL methodology in detail - specific algorithms, architectures, techniques, training procedures, loss functions, optimization methods, or experimental setups used.",
  "key_findings": "List the main results with specific numbers, metrics, accuracy scores, or performance improvements when available. Include what the paper demonstrates or proves.",
  "relevance_score": 0.0-1.0,
  "relevance_reason": "Explain in detail why this paper is or isn't relevant to the user's query, citing specific aspects of the work."
}}

IMPORTANT: Return ONLY valid JSON. No markdown, no text before or after the JSON object. The JSON must be parseable with json.loads()."""


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
        logger.error(f"Error extracting text: {e}")
        return ""


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response."""
    if not text:
        raise ValueError("Empty response")

    text = text.strip()

    # Find JSON bounds
    start = text.find('{')
    end = text.rfind('}') + 1
    if start >= 0 and end > start:
        text = text[start:end]
    else:
        # Try removing markdown code blocks
        text = text.replace('```json', '').replace('```', '').strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fix common JSON issues
        text = re.sub(r',(\s*[}]])', r'\1', text)  # trailing commas
        text = re.sub(r"(\w+): '([^']*)'", r'"\1": "\2"', text)  # single quotes
        text = re.sub(r"(\w+): \"([^\"]*)\"", r'"\1": "\2"', text)
        try:
            return json.loads(text)
        except Exception as e:
            logger.error(f"JSON parse failed: {e}, text: {text[:200]}")
            raise ValueError(f"Failed to parse JSON: {text[:200]}")


async def summarize_paper(paper: Paper, user_query: str) -> PaperSummary:
    """Generate an LLM summary for a single paper."""
    logger.info(f"Summarizing: {paper.title[:50]}...")

    authors_str = ", ".join(paper.authors[:3])
    if len(paper.authors) > 3:
        authors_str += " et al."
    elif len(paper.authors) == 0:
        authors_str = "Unknown authors"

    # Truncate abstract to avoid token limits
    abstract = paper.abstract[:2000] if paper.abstract else "No abstract available."

    prompt = SUMMARY_PROMPT.format(
        title=paper.title[:500],
        authors=authors_str,
        year=paper.year,
        abstract=abstract,
        query=user_query,
    )

    try:
        response = _client.messages.create(
            model=LLM_MODEL,
            max_tokens=2048,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = _extract_text_from_response(response)

        if not response_text:
            logger.warning(f"Empty LLM response for {paper.paper_id}")
            return _generate_smart_fallback(paper, user_query)

        logger.info(f"LLM response length: {len(response_text)} chars")

        result = _parse_json_response(response_text)

        # Ensure all fields have content
        summary = result.get("summary", "") or _generate_smart_fallback(paper, user_query).summary
        methodology = result.get("key_methodology", "") or _extract_methodology_from_text(paper.abstract or "")
        findings = result.get("key_findings", "") or _extract_findings_from_text(paper.abstract or "")
        reason = result.get("relevance_reason", "") or "Relevant to your search query"

        return PaperSummary(
            paper_id=paper.paper_id,
            summary=summary,
            key_methodology=methodology,
            key_findings=findings,
            relevance_score=min(1.0, max(0.0, float(result.get("relevance_score", 0.5)))),
            relevance_reason=reason,
        )

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return _generate_smart_fallback(paper, user_query)


async def summarize_all(papers: List[Paper], user_query: str) -> List[PaperSummary]:
    """Summarize all papers concurrently."""
    logger.info(f"Summarizing {len(papers)} papers")
    tasks = [summarize_paper(paper, user_query) for paper in papers]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    summaries = []
    for paper, result in zip(papers, results):
        if isinstance(result, Exception):
            logger.error(f"Failed: {paper.paper_id}: {result}")
            summaries.append(_generate_smart_fallback(paper, user_query))
        else:
            summaries.append(result)
    return summaries


def _extract_methodology_from_text(text: str) -> str:
    """Extract methodology keywords from abstract."""
    text_lower = text.lower()
    methods = []

    # Neural network / deep learning methods
    if 'convolutional' in text_lower or 'cnn' in text_lower:
        methods.append("Convolutional Neural Network (CNN)")
    if 'recurrent' in text_lower or 'rnn' in text_lower:
        methods.append("Recurrent Neural Network (RNN)")
    if 'lstm' in text_lower:
        methods.append("Long Short-Term Memory (LSTM)")
    if 'transformer' in text_lower:
        methods.append("Transformer architecture")
    if 'attention' in text_lower:
        methods.append("Attention mechanism")
    if 'bert' in text_lower:
        methods.append("BERT model")
    if 'gpt' in text_lower:
        methods.append("GPT model")
    if 'gan' in text_lower:
        methods.append("Generative Adversarial Network")
    if 'vae' in text_lower:
        methods.append("Variational Autoencoder")

    # Traditional ML
    if 'random forest' in text_lower:
        methods.append("Random Forest")
    if 'support vector' in text_lower or 'svm' in text_lower:
        methods.append("Support Vector Machine")
    if 'decision tree' in text_lower:
        methods.append("Decision Tree")
    if 'xgboost' in text_lower or 'gradient boosting' in text_lower:
        methods.append("Gradient Boosting / XGBoost")
    if 'naive bayes' in text_lower:
        methods.append("Naive Bayes")
    if 'k-nearest' in text_lower or 'knn' in text_lower:
        methods.append("K-Nearest Neighbors")

    # Statistical methods
    if 'logistic regression' in text_lower:
        methods.append("Logistic Regression")
    if 'linear regression' in text_lower:
        methods.append("Linear Regression")
    if 'statistical' in text_lower and 'analysis' in text_lower:
        methods.append("Statistical analysis")

    # Optimization
    if 'genetic algorithm' in text_lower:
        methods.append("Genetic Algorithm")
    if 'particle swarm' in text_lower:
        methods.append("Particle Swarm Optimization")
    if 'reinforcement learning' in text_lower or 'rl ' in text_lower:
        methods.append("Reinforcement Learning")

    if not methods:
        # Generic fallback
        if 'neural' in text_lower or 'network' in text_lower:
            return "Neural network approach"
        if 'learning' in text_lower:
            return "Machine learning approach"
        if 'algorithm' in text_lower:
            return "Algorithmic approach"
        return "See paper for methodology details"

    return "; ".join(methods[:2])


def _extract_findings_from_text(text: str) -> str:
    """Extract key findings from abstract."""
    text_lower = text.lower()
    findings = []

    # Performance metrics
    if 'accuracy' in text_lower:
        findings.append("Reports accuracy metrics")
    if 'auc' in text_lower or 'roc' in text_lower:
        findings.append("Reports AUC/ROC performance")
    if 'f1' in text_lower:
        findings.append("Reports F1-score")
    if 'precision' in text_lower and 'recall' in text_lower:
        findings.append("Reports precision and recall")

    # Improvement claims
    if any(x in text_lower for x in ['improve', 'outperform', 'better than', 'state-of-the-art', 'surpass', 'exceed']):
        findings.append("Claims improved performance over baselines")
    if 'reduction' in text_lower or 'reduce' in text_lower:
        findings.append("Aims to reduce target metric")

    # Task type
    if 'detect' in text_lower or 'detection' in text_lower:
        findings.append("Addresses detection task")
    if 'classif' in text_lower:
        findings.append("Addresses classification task")
    if 'segment' in text_lower:
        findings.append("Addresses segmentation task")
    if 'generat' in text_lower:
        findings.append("Addresses generation task")
    if 'predict' in text_lower or 'forecast' in text_lower:
        findings.append("Addresses prediction task")
    if 'recognit' in text_lower:
        findings.append("Addresses recognition task")

    if not findings:
        return "Reports experimental results and findings"
    return "; ".join(findings[:2])


def _extract_domain_from_text(text: str, title: str) -> str:
    """Extract domain/application area from text."""
    text_lower = (text + " " + title).lower()

    domain_keywords = {
        "NLP": ['nlp', 'natural language', 'text', 'translation', 'sentiment', 'nlu', 'word embedding', 'language model'],
        "Computer Vision": ['image', 'vision', 'object detection', 'segmentation', 'imagenet', 'video', 'face'],
        "Healthcare": ['medical', 'health', 'diagnosis', 'patient', 'disease', 'clinical', 'drug', 'cancer'],
        "Finance": ['financial', 'stock', 'trading', 'fraud', 'risk', 'banking', 'market'],
        "Speech/Audio": ['speech', 'audio', 'voice', 'asr', 'tts', 'sound'],
        "Time Series": ['time series', 'temporal', 'forecasting', 'sensor', 'signal'],
        "Robotics": ['robot', 'autonomous', 'control', 'motion'],
        "Security": ['security', 'attack', 'malware', 'intrusion', 'cyber', 'fraud detection'],
        "Recommendation": ['recommend', 'collaborative', 'content-based', 'user preference'],
        "Power Systems": ['power', 'grid', 'transmission', 'energy', 'electrical'],
    }

    for domain, keywords in domain_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return domain

    return "Research application"


def _generate_smart_fallback(paper: Paper, user_query: str) -> PaperSummary:
    """
    Generate an informative fallback summary by analyzing the abstract.
    """
    abstract = paper.abstract or ""
    title = paper.title or ""

    # Extract methodology
    methodology = _extract_methodology_from_text(abstract)

    # Extract findings
    findings = _extract_findings_from_text(abstract)

    # Extract domain
    domain = _extract_domain_from_text(abstract, title)

    # Generate summary from abstract
    if abstract and len(abstract) > 50:
        sentences = re.split(r'[.!?]+', abstract)
        sentences = [s.strip() for s in sentences if s.strip()]
        summary = ". ".join(sentences[:3])
        if len(summary) > 400:
            summary = summary[:400] + "..."
    else:
        summary = f"Research paper titled '{title}' published in {paper.year}."

    # Calculate relevance
    query_terms = set(user_query.lower().split())
    title_terms = set(title.lower().split())
    abstract_terms = set(abstract.lower().split())

    overlap_title = len(query_terms & title_terms)
    overlap_abstract = len(query_terms & abstract_terms) / max(1, len(abstract_terms))
    relevance_score = min(1.0, 0.3 + (overlap_title * 0.2) + (overlap_abstract * 0.3))

    # Generate relevance reason
    if relevance_score > 0.6:
        reason = f"This paper is highly relevant to '{user_query}' based on title and abstract keywords."
    elif relevance_score > 0.3:
        reason = f"This paper partially relates to your query with some relevant concepts."
    else:
        reason = "This paper matched your search parameters."

    return PaperSummary(
        paper_id=paper.paper_id,
        summary=summary,
        key_methodology=methodology,
        key_findings=findings,
        relevance_score=relevance_score,
        relevance_reason=reason,
    )