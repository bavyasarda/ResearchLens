"""
Search API - Main search endpoint orchestrating the full RAG pipeline.
"""
import time
import logging
from typing import List
from fastapi import APIRouter, HTTPException, Response

from backend.models.schemas import SearchRequest, SearchResponse, Paper, PaperSummary, ComparisonTable
from backend.services.paper_fetcher import get_paper_fetcher
from backend.services.query_expander import expand_query
from backend.services.chunker import CoreChunker
from backend.services.embedder import get_embedder
from backend.services.sparse_retriever import BM25Retriever
from backend.services.hybrid_retriever import HybridRetriever
from backend.services.reranker import get_reranker
from backend.services.summarizer import summarize_all
from backend.services.comparator import generate_comparison
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    Main search endpoint - orchestrates the full RAG pipeline.
    """
    start_time = time.time()

    try:
        # 1. Expand query with LLM
        logger.info(f"Step 1: Expanding query '{req.query}'")
        try:
            expansion = await expand_query(req.query)
            expanded_query = expansion.expanded_query
            logger.info(f"Query expanded to: '{expanded_query}'")
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original")
            expanded_query = req.query
            from backend.models.schemas import QueryExpansion
            expansion = QueryExpansion(
                expanded_query=req.query,
                key_concepts=req.query.split(),
                related_terms=[],
                exclusion_terms=[]
            )

        # 2. Fetch papers from APIs
        logger.info("Step 2: Fetching papers from APIs")
        fetcher = get_paper_fetcher()
        papers = await fetcher.fetch_papers(
            query=expanded_query,
            num_papers=req.num_papers,
            preference=req.preference,
            year_from=req.year_from,
            year_to=req.year_to,
        )

        if not papers:
            logger.warning("No papers found")
            return SearchResponse(
                papers=[],
                summaries=[],
                comparison=ComparisonTable(
                    headers=["Paper", "Year", "Methodology", "Dataset", "Innovation", "Limitations", "Relevance"],
                    rows=[],
                    alignment_analysis="No papers found matching your query. Try a different search term.",
                ),
                expanded_query=expanded_query,
                total_fetched=0,
                retrieval_method="hybrid_rag",
            )

        total_fetched = len(papers)
        logger.info(f"Step 2 complete: Fetched {total_fetched} papers")

        # 3. Chunk all papers
        logger.info("Step 3: Chunking papers")
        chunker = CoreChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        chunks = chunker.chunk_all_papers(papers)
        logger.info(f"Step 3 complete: Created {len(chunks)} chunks")

        # 4. Embed chunks (dense)
        logger.info("Step 4: Embedding chunks")
        try:
            embedder = get_embedder()
            embeddings = embedder.embed_chunks(chunks)
            logger.info(f"Step 4 complete: Embedded {len(embeddings)} chunks")
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            embeddings = {}

        # 5. Index chunks (BM25 sparse)
        logger.info("Step 5: Building BM25 index")
        try:
            bm25 = BM25Retriever()
            bm25.index(chunks)
            logger.info("Step 5 complete: BM25 index built")
        except Exception as e:
            logger.error(f"BM25 indexing failed: {e}")
            bm25 = None

        # 6. Hybrid retrieval + RRF fusion
        logger.info("Step 6: Performing hybrid retrieval")
        try:
            if embeddings and bm25:
                hybrid = HybridRetriever(embedder=embedder, bm25=bm25)
                retrieval_results = hybrid.retrieve(
                    query=expanded_query,
                    expanded_terms=expansion.related_terms,
                    chunks=chunks,
                    embeddings=embeddings,
                    top_k=len(papers),
                )
            else:
                # Fallback: return papers as-is
                retrieval_results = [(p, 1.0, None) for p in papers]
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            retrieval_results = [(p, 1.0, None) for p in papers]

        # Extract papers from retrieval results
        retrieved_papers = [paper for paper, _, _ in retrieval_results]
        logger.info(f"Step 6 complete: Hybrid retrieval returned {len(retrieved_papers)} papers")

        # 7. Cross-encoder reranking
        logger.info("Step 7: Reranking with cross-encoder")
        try:
            reranker = get_reranker()
            reranked_results = reranker.rerank(
                query=expanded_query,
                candidates=retrieval_results,
                top_k=req.num_papers,
            )

            # Update papers with reranked order and scores
            reranked_papers_map = {paper.paper_id: (paper, score) for paper, score in reranked_results}
            final_papers = []
            for paper in papers:
                if paper.paper_id in reranked_papers_map:
                    final_papers.append(paper)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            final_papers = papers[:req.num_papers]

        # If we don't have enough from reranking, fill in from original
        existing_ids = set(p.paper_id for p in final_papers)
        for paper in papers:
            if paper.paper_id not in existing_ids and len(final_papers) < req.num_papers:
                final_papers.append(paper)

        logger.info(f"Step 7 complete: Using {len(final_papers)} papers")

        # 8. Summarize all papers concurrently
        logger.info("Step 8: Summarizing papers")
        summaries = await summarize_all(final_papers, req.query)
        logger.info(f"Step 8 complete: Generated {len(summaries)} summaries")

        # 9. Generate comparison table
        logger.info("Step 9: Generating comparison table")
        comparison = await generate_comparison(req.query, final_papers, summaries)
        logger.info("Step 9 complete: Comparison table generated")

        processing_time = time.time() - start_time
        logger.info(f"Search completed in {processing_time:.2f}s")

        return SearchResponse(
            papers=final_papers,
            summaries=summaries,
            comparison=comparison,
            expanded_query=expanded_query,
            total_fetched=total_fetched,
            retrieval_method="hybrid_rag",
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")