"""
Paper Fetcher Service - Fetches research papers from arXiv API.
"""
import asyncio
import logging
from typing import List, Optional
from difflib import SequenceMatcher
import httpx
import feedparser

from backend.config import (
    SEMANTIC_SCHOLAR_API_KEY,
    SEMANTIC_SCHOLAR_BASE_URL,
    ARXIV_BASE_URL,
)
from backend.models.schemas import Paper

logger = logging.getLogger(__name__)


class PaperFetcher:
    def __init__(self):
        # Follow redirects (arXiv returns 301)
        self.http_client = httpx.AsyncClient(timeout=60.0, follow_redirects=True)

    async def close(self):
        await self.http_client.aclose()

    async def fetch_papers(
        self,
        query: str,
        num_papers: int,
        preference: str = "balanced",
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> List[Paper]:
        logger.info(f"Fetching papers for query: '{query}', target: {num_papers}")

        all_papers = []

        # Try arXiv
        try:
            arxiv_papers = await self._fetch_arxiv(query, num_papers, preference, year_from, year_to)
            if arxiv_papers:
                all_papers.extend(arxiv_papers)
                logger.info(f"arXiv returned {len(arxiv_papers)} papers")
            else:
                logger.warning("arXiv returned no papers")
        except Exception as e:
            logger.error(f"arXiv error: {e}", exc_info=True)

        # Try Semantic Scholar if API key available
        if SEMANTIC_SCHOLAR_API_KEY:
            try:
                ss_papers = await self._fetch_semantic_scholar(
                    query, num_papers, preference, year_from, year_to
                )
                if ss_papers:
                    all_papers.extend(ss_papers)
                    logger.info(f"Semantic Scholar returned {len(ss_papers)} papers")
            except Exception as e:
                logger.error(f"Semantic Scholar error: {e}")
        else:
            logger.info("No Semantic Scholar API key - using arXiv only")

        # Deduplicate
        deduplicated = self._deduplicate_papers(all_papers)

        if year_from or year_to:
            deduplicated = self._filter_by_year(deduplicated, year_from, year_to)

        logger.info(f"Total papers: {len(all_papers)}, after dedup: {len(deduplicated)}")

        return deduplicated[:num_papers]

    async def _fetch_arxiv(
        self,
        query: str,
        limit: int,
        preference: str,
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> List[Paper]:
        """Fetch papers from arXiv API with retry logic."""
        logger.info(f"Calling arXiv API with query: '{query}'")

        # Build URL - arXiv uses 'all' for full-text search
        search_query = query.replace(" ", "+")
        sort_by = "relevance" if preference == "most_cited" else "submittedDate"

        url = f"{ARXIV_BASE_URL}?search_query=all:{search_query}&max_results={limit}&sortBy={sort_by}"
        logger.info(f"arXiv URL: {url}")

        # Recreate client if closed
        if self.http_client.is_closed:
            self.http_client = httpx.AsyncClient(timeout=60.0)

        # Retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Small delay before request (longer on retry)
                if attempt > 0:
                    await asyncio.sleep(2 ** attempt)  # 2, 4 seconds

                response = await self.http_client.get(url, follow_redirects=True)
                logger.info(f"arXiv response status: {response.status_code}")

                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt * 3
                        logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error("arXiv rate limit exceeded")
                        return []
                    break

                if response.status_code not in [200, 301]:
                    logger.error(f"arXiv returned status {response.status_code}")
                    return []

                # Success - parse feed
                feed = feedparser.parse(response.text)
                logger.info(f"arXiv feed entries: {len(feed.entries)}")

                if not feed.entries:
                    logger.warning("No entries in arXiv feed")
                    return []

                papers = []
                for i, entry in enumerate(feed.entries[:limit]):
                    try:
                        # Extract arXiv ID from entry ID
                        entry_id = entry.get("id", "")
                        arxiv_id = entry_id.split("/")[-1] if entry_id else f"unknown_{i}"

                        # Get PDF URL
                        pdf_url = None
                        for link in entry.get("links", []):
                            href = link.get("href", "")
                            if "pdf" in href:
                                pdf_url = href
                                break

                        # Get authors
                        authors = []
                        for author in entry.get("authors", []):
                            name = author.get("name", "")
                            if name:
                                authors.append(name)

                        # Get year from published date
                        published = entry.get("published", "")
                        year = 2024
                        if published:
                            try:
                                year = int(published[:4])
                            except Exception:
                                pass

                        # Get title and clean it
                        title = entry.get("title", "Untitled")
                        title = " ".join(title.split())  # Normalize whitespace

                        # Get abstract/summary
                        abstract = entry.get("summary", "")
                        abstract = " ".join(abstract.split())  # Normalize whitespace

                        paper = Paper(
                            paper_id=f"arxiv:{arxiv_id}",
                            title=title,
                            authors=authors,
                            year=year,
                            abstract=abstract,
                            citation_count=0,
                            url=entry_id,
                            pdf_url=pdf_url,
                            venue="arXiv",
                            source="arxiv",
                        )
                        papers.append(paper)
                        logger.info(f"  Parsed paper: {title[:50]}...")
                    except Exception as e:
                        logger.warning(f"Error parsing entry {i}: {e}")
                        continue

                return papers

            except httpx.RequestError as e:
                logger.error(f"arXiv request failed: {e}")
                return []
            except Exception as e:
                logger.error(f"arXiv fetch error: {e}", exc_info=True)
                return []

    async def _fetch_semantic_scholar(
        self,
        query: str,
        limit: int,
        preference: str,
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> List[Paper]:
        """Fetch papers from Semantic Scholar Graph API."""
        if not SEMANTIC_SCHOLAR_API_KEY:
            return []

        headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
        sort_param = "citationCount" if preference == "most_cited" else "publicationDate"

        params = {
            "query": query,
            "limit": min(limit, 100),
            "sort": sort_param,
            "fields": "paperId,title,authors,year,abstract,citationCount,externalIds,venue,openAccessPdf",
        }

        if self.http_client.is_closed:
            self.http_client = httpx.AsyncClient(timeout=30.0)

        response = await self.http_client.get(
            f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search",
            headers=headers,
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        papers = []
        for item in data.get("data", []):
            try:
                authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
                year = item.get("year")
                if year and year_from and year < year_from:
                    continue
                if year and year_to and year > year_to:
                    continue

                pdf_url = None
                if item.get("openAccessPdf"):
                    pdf_url = item.get("openAccessPdf", {}).get("url")

                doi = item.get("externalIds", {}).get("DOI", "")
                paper_url = f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}"
                if doi:
                    paper_url = f"https://doi.org/{doi}"

                paper = Paper(
                    paper_id=item.get("paperId", ""),
                    title=item.get("title", "Untitled"),
                    authors=authors,
                    year=year or 2024,
                    abstract=item.get("abstract", "") or "",
                    citation_count=item.get("citationCount", 0) or 0,
                    url=paper_url,
                    pdf_url=pdf_url,
                    venue=item.get("venue", ""),
                    source="semantic_scholar",
                )
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Error parsing paper: {e}")
                continue

        return papers

    def _deduplicate_papers(self, papers: List[Paper], threshold: float = 0.85) -> List[Paper]:
        """Remove duplicate papers based on title similarity."""
        if not papers:
            return []

        unique_papers = []
        seen_titles = []

        for paper in papers:
            normalized_title = paper.title.lower().strip()

            is_duplicate = False
            for seen_title in seen_titles:
                similarity = SequenceMatcher(None, normalized_title, seen_title).ratio()
                if similarity >= threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_papers.append(paper)
                seen_titles.append(normalized_title)

        return unique_papers

    def _filter_by_year(
        self,
        papers: List[Paper],
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> List[Paper]:
        """Filter papers by year range."""
        filtered = []
        for paper in papers:
            if year_from and paper.year < year_from:
                continue
            if year_to and paper.year > year_to:
                continue
            filtered.append(paper)
        return filtered


# Singleton instance
_fetcher: Optional[PaperFetcher] = None


def get_paper_fetcher() -> PaperFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = PaperFetcher()
    return _fetcher


async def close_paper_fetcher():
    global _fetcher
    if _fetcher:
        await _fetcher.close()
        _fetcher = None