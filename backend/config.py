"""
Configuration management for ResearchLens.
Loads environment variables from .env file with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# LLM Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "sk-ant-opm-l4svni-qkNGdQmlPTDXN26ABdjSZiWXo")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.opusmax.pro")

# API Keys
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5501,http://127.0.0.1:5501,http://localhost:5500,http://127.0.0.1:5500").split(",")

# Retrieval Configuration
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
MAX_PAPERS = int(os.getenv("MAX_PAPERS", "30"))

# BM25 Parameters
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

# RRF Parameters (Reciprocal Rank Fusion)
RRF_K = int(os.getenv("RRF_K", "60"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.4"))

# Cross-encoder reranking
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))

# Semantic Scholar API
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_RATE_LIMIT = 10 if SEMANTIC_SCHOLAR_API_KEY else 1  # req/sec

# arXiv API
ARXIV_BASE_URL = "http://export.arxiv.org/api/query"

# CORE API (Disabled - rate limited)
# CORE_BASE_URL = "https://api.core.ac.uk/v3/search/works"
# CORE_API_KEY = os.getenv("CORE_API_KEY", "")

# LLM Model
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))