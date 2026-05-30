"""
Comparator Service - Generate detailed comparative methodology tables.
"""
import json
import re
import logging
from typing import List
import anthropic

from backend.config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, LLM_MODEL
from backend.models.schemas import Paper, PaperSummary, ComparisonTable

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    base_url=ANTHROPIC_BASE_URL,
)

# Detailed comparison prompt
COMPARE_PROMPT = """You are an expert academic researcher. Create a detailed comparative analysis of these research papers for: "{user_query}"

Papers data:
{papers_json}

For each paper, provide DETAILED information based on the paper's abstract and summary:

1. **Methodology**: Extract the specific algorithms, architectures, techniques, or methods used. Be precise (e.g., "Convolutional Neural Network with 8 layers, batch normalization, ReLU activation" not just "Neural Network").

2. **Dataset/Domain**: Identify the specific dataset(s) used (e.g., "ImageNet-1K", "COCO", "SQuAD 2.0", "PubMed") or research domain application area.

3. **Key Innovation**: Describe the novel contribution or breakthrough of this paper.

4. **Limitations**: Identify stated or clearly implied limitations of the approach.

Return ONLY valid JSON (no markdown fences):
{{
  "headers": ["Paper", "Year", "Methodology", "Dataset/Domain", "Key Innovation", "Limitations", "Relevance"],
  "rows": [
    {{
      "Paper": "Short descriptive title (max 50 chars)",
      "Year": "YYYY",
      "Methodology": "Detailed methodology description (50-100 words)",
      "Dataset/Domain": "Specific dataset(s) used or domain application",
      "Key Innovation": "Novel contribution or breakthrough (30-60 words)",
      "Limitations": "Stated or inferred limitations (20-40 words)",
      "Relevance": "High/Medium/Low - brief justification"
    }}
  ],
  "alignment_analysis": "2-3 paragraph synthesis: (1) How these papers collectively address your query, (2) Which approaches are most suitable for implementation and why, (3) Key gaps or emerging research directions."
}}"""


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


async def generate_comparison(
    query: str,
    papers: List[Paper],
    summaries: List[PaperSummary],
) -> ComparisonTable:
    """Generate a detailed comparative table."""
    logger.info(f"Comparing {len(papers)} papers for: '{query[:50]}...'")

    # Build comprehensive paper data
    papers_data = []
    for paper, summary in zip(papers, summaries):
        papers_data.append({
            "title": paper.title,
            "year": paper.year,
            "authors": paper.authors[:3] if paper.authors else [],
            "venue": paper.venue or "Conference/Journal",
            "abstract": paper.abstract[:800] if paper.abstract else "",
            "summary": summary.summary[:300] if summary.summary else "",
            "methodology": summary.key_methodology,
            "findings": summary.key_findings,
            "relevance_score": summary.relevance_score,
        })

    papers_json = json.dumps(papers_data, indent=2)

    prompt = COMPARE_PROMPT.format(user_query=query, papers_json=papers_json)

    try:
        response = _client.messages.create(
            model=LLM_MODEL,
            max_tokens=4096,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = _extract_text_from_response(response)

        if not response_text:
            logger.warning("Empty LLM response")
            return _smart_comparison(papers, summaries, query)

        result = _parse_json_response(response_text)

        return ComparisonTable(
            headers=result.get("headers", []),
            rows=result.get("rows", []),
            alignment_analysis=result.get("alignment_analysis", ""),
        )

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return _smart_comparison(papers, summaries, query)


def _smart_comparison(
    papers: List[Paper],
    summaries: List[PaperSummary],
    query: str,
) -> ComparisonTable:
    """
    Generate comparison table by intelligently analyzing paper content.
    """
    headers = ["Paper", "Year", "Methodology", "Dataset", "Innovation", "Limitations", "Relevance"]
    rows = []

    for paper, summary in zip(papers, summaries):
        # Title (truncated)
        title = paper.title[:60] + "..." if len(paper.title) > 60 else paper.title

        # Relevance level
        if summary.relevance_score >= 0.7:
            relevance = "High"
            rel_desc = f"Strong match ({summary.relevance_score:.0%})"
        elif summary.relevance_score >= 0.4:
            relevance = "Medium"
            rel_desc = f"Partial match ({summary.relevance_score:.0%})"
        else:
            relevance = "Low"
            rel_desc = f"Weak match ({summary.relevance_score:.0%})"

        # Extract methodology - always use extracted from abstract if summary is generic
        methodology = summary.key_methodology
        generic_phrases = ["methodology described in paper", "methodology details extracted", "methodology not extracted", "paper"]
        is_generic = any(phrase in methodology.lower() for phrase in generic_phrases) if methodology else True
        if is_generic:
            methodology = _extract_methodology(paper.abstract or "")
            if "described in paper" in methodology.lower():
                methodology = "Research methodology (see abstract)"

        # Extract innovation
        innovation = _extract_innovation(paper.abstract or "", methodology)

        # Extract domain
        domain = paper.venue or _extract_domain(paper.abstract or "", paper.title)

        # Extract limitations
        limitations = _extract_limitations(paper.abstract or "")

        row = {
            "Paper": title,
            "Year": str(paper.year),
            "Methodology": methodology,
            "Dataset": domain,
            "Innovation": innovation,
            "Limitations": limitations,
            "Relevance": f"{relevance} — {rel_desc}",
        }
        rows.append(row)

    # Generate analysis
    high_rel = [s for s in summaries if s.relevance_score >= 0.6]
    analysis = _generate_analysis(query, papers, summaries, len(high_rel))

    return ComparisonTable(
        headers=headers,
        rows=rows,
        alignment_analysis=analysis,
    )


def _extract_methodology(abstract: str) -> str:
    """Extract methodology from abstract."""
    text_lower = abstract.lower()

    methods = []

    # Deep learning
    if 'transformer' in text_lower:
        methods.append("Transformer architecture")
    if 'attention' in text_lower and 'transformer' not in text_lower:
        methods.append("Attention mechanism")
    if any(x in text_lower for x in ['bert', 'gpt', 'llm', 'language model']):
        methods.append("Pre-trained Language Model")
    if any(x in text_lower for x in ['cnn', 'convolutional']):
        methods.append("Convolutional Neural Network")
    if any(x in text_lower for x in ['lstm', 'rnn', 'recurrent']):
        methods.append("Recurrent Neural Network")
    if any(x in text_lower for x in ['gan', 'generative']):
        methods.append("Generative Adversarial Network")
    if 'neural network' in text_lower and not methods:
        methods.append("Deep Neural Network")

    # ML algorithms
    if any(x in text_lower for x in ['random forest']):
        methods.append("Random Forest")
    if any(x in text_lower for x in ['support vector', 'svm']):
        methods.append("Support Vector Machine")
    if any(x in text_lower for x in ['gradient boosting', 'xgboost', 'lightgbm']):
        methods.append("Gradient Boosting")
    if any(x in text_lower for x in ['k-nearest', 'knn']):
        methods.append("K-Nearest Neighbors")
    if any(x in text_lower for x in ['logistic', 'linear regression']):
        methods.append("Regression analysis")
    if any(x in text_lower for x in ['clustering', 'k-means']):
        methods.append("Clustering")
    if any(x in text_lower for x in ['decision tree']):
        methods.append("Decision Tree")

    # Optimization
    if any(x in text_lower for x in ['genetic algorithm', 'ga']):
        methods.append("Genetic Algorithm")
    if any(x in text_lower for x in ['pso', 'particle swarm']):
        methods.append("Particle Swarm Optimization")

    # Signal processing
    if any(x in text_lower for x in ['fft', 'wavelet', 'filter']):
        methods.append("Signal processing")

    # Statistical
    if any(x in text_lower for x in ['statistical', 'hypothesis', 'p-value']):
        methods.append("Statistical analysis")

    if not methods:
        return "Methodology described in paper"
    return ", ".join(methods[:2])


def _extract_innovation(abstract: str, methodology: str) -> str:
    """Extract key innovation from abstract."""
    text_lower = abstract.lower()

    innovations = []

    if any(x in text_lower for x in ['novel', 'new approach', 'new method', 'introduce', 'propose']):
        innovations.append("Novel approach or framework")

    if any(x in text_lower for x in ['improve', 'improve upon', 'outperform', 'surpass', 'state-of-the-art', 'exceed']):
        innovations.append("Improved performance over baselines")

    if any(x in text_lower for x in ['first', 'initially', 'pioneer']):
        innovations.append("First-of-its-kind application")

    if 'efficient' in text_lower or 'faster' in text_lower:
        innovations.append("Efficiency improvements")

    if 'real-time' in text_lower or 'online' in text_lower:
        innovations.append("Real-time processing capability")

    if 'transfer learning' in text_lower or 'pre-trained' in text_lower:
        innovations.append("Transfer learning approach")

    if not innovations:
        # Extract from methodology if available
        if methodology and "described in paper" not in methodology.lower():
            return f"Uses {methodology.split(',')[0]}"
        return "See paper for key innovations"

    return innovations[0]


def _extract_domain(abstract: str, title: str) -> str:
    """Extract domain/application from abstract."""
    text = (abstract + " " + title).lower()

    domains = {
        "NLP / Text": ['nlp', 'text', 'language', 'translation', 'sentiment', 'document', 'corpus'],
        "Computer Vision": ['image', 'video', 'object detection', 'segmentation', 'recognition'],
        "Healthcare": ['medical', 'health', 'patient', 'disease', 'diagnosis', 'clinical'],
        "Finance": ['financial', 'stock', 'trading', 'fraud', 'risk', 'economic'],
        "Power Systems": ['power', 'transformer', 'generator', 'grid', 'electrical'],
        "Cybersecurity": ['security', 'attack', 'malware', 'intrusion', 'threat'],
        "Speech/Audio": ['speech', 'audio', 'voice', 'sound'],
        "Time Series": ['time series', 'temporal', 'forecasting', 'sensor'],
        "Recommendation": ['recommend', 'user preference', 'collaborative'],
        "IoT": ['sensor', 'iot', 'embedded', 'device'],
    }

    for domain, keywords in domains.items():
        if sum(1 for kw in keywords if kw in text) >= 1:
            return domain

    return "Research / Academic"


def _extract_limitations(abstract: str) -> str:
    """Extract or infer limitations from abstract."""
    text_lower = abstract.lower()

    limitations = []

    if 'limited dataset' in text_lower or 'small dataset' in text_lower:
        limitations.append("Limited dataset size")

    if 'computational' in text_lower or 'expensive' in text_lower:
        limitations.append("High computational cost")

    if 'generalization' in text_lower:
        limitations.append("Generalization concerns")

    if 'assumes' in text_lower or 'assumption' in text_lower:
        limitations.append("Relies on assumptions")

    if 'specific' in text_lower and 'domain' in text_lower:
        limitations.append("Domain-specific")

    if not limitations:
        return "See paper for limitations"

    return "; ".join(limitations[:2])


def _generate_analysis(query: str, papers: List[Paper], summaries: List[PaperSummary], num_high: int) -> str:
    """Generate a comprehensive alignment analysis."""
    methodologies = [s.key_methodology for s in summaries if s.key_methodology]
    innovations = [_extract_innovation(p.abstract or "", s.key_methodology) for p, s in zip(papers, summaries)]

    # Find common approaches
    all_methods = " ".join(methodologies).lower()
    common_approaches = []
    if 'transformer' in all_methods:
        common_approaches.append("transformer-based approaches")
    if 'neural network' in all_methods or 'deep learning' in all_methods:
        common_approaches.append("deep learning methods")
    if 'machine learning' in all_methods:
        common_approaches.append("machine learning techniques")

    analysis = []

    # Summary of the papers
    analysis.append(f"This comparison examines {len(papers)} papers related to '{query}'. ")

    # Methodological trends
    if common_approaches:
        analysis.append(f"The papers primarily use {', '.join(common_approaches[:-1])}" +
                       (f" and {common_approaches[-1]}" if len(common_approaches) > 1 else "") + ".")

    # High relevance papers
    if num_high > 0:
        analysis.append(f"{num_high} paper(s) show high relevance to your query and are strong candidates for further investigation.")

    # Recommendations
    analysis.append("For implementation, consider starting with papers showing High relevance as they most closely match your research goals.")

    return " ".join(analysis)