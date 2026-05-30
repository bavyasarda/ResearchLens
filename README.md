# 🔬 ResearchLens - AI Research Paper Search Engine

ResearchLens is a production-grade, full-stack AI research paper search engine that uses Modern Hybrid RAG (Retrieval-Augmented Generation) to find, analyze, and compare academic papers.

![ResearchLens](https://img.shields.io/badge/ResearchLens-v1.0.0-00d4ff)
![Python](https://img.shields.io/badge/Python-3.11+-7c3aed)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-10b981)

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Manual Setup](#-manual-setup)
- [API Reference](#-api-reference)
- [How Hybrid RAG Works](#-how-hybrid-rag-works)
- [Customization](#-customization)
- [Limitations](#-limitations)

---

## ✨ Features

- **🔍 Multi-Source Paper Fetching**: Searches Semantic Scholar, arXiv, and CORE API with automatic fallback
- **🧠 Intelligent Query Expansion**: Uses Claude to expand user queries with academic vocabulary
- **📄 Context-Aware Chunking**: Sentence-aware document chunking with overlapping context windows
- **🔀 Hybrid Retrieval**: Combines dense (embedding-based) and sparse (BM25) search with RRF fusion
- **🎯 Cross-Encoder Reranking**: Re-ranks results for improved relevance
- **📝 AI-Powered Summaries**: Generates paper summaries using Claude
- **📊 Comparative Analysis**: Creates methodology comparison tables
- **💬 Follow-up Chat**: Ask questions about retrieved papers

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ResearchLens Pipeline                          │
└─────────────────────────────────────────────────────────────────────────────┘

    User Query
         │
         ▼
┌─────────────────┐
│  Query Expander │ ◄──── LLM (Claude)
│    (LLM Call)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Paper Fetcher  │────►│  Semantic Scholar│
│   (API Chain)   │     ├─────────────────┤
│                 │     │      arXiv      │ ◄── Fallback chain
│                 │     ├─────────────────┤
│                 │     │      CORE       │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Core Chunker  │ ◄── Sentence-aware splitting
│  (Context Win)  │     with overlap
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌────────┐
│ Dense │ │ Sparse │
│ Embed │ │  BM25  │
│ (BGE) │ │        │
└───┬───┘ └───┬────┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│ Hybrid Retriever│ ◄── RRF (Reciprocal Rank Fusion)
│   (RRF Fusion)  │     k=60, Dense:0.6, Sparse:0.4
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Cross-Encoder  │ ◄── ms-marco-MiniLM-L-6-v2
│    Reranker     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────┐
│Summarize│ │ Compare  │ ◄──── LLM (Claude)
│ (LLM)  │ │  (LLM)   │
└────────┘ └──────────┘
    │         │
    └────┬────┘
         │
         ▼
   SearchResponse
         │
         ▼
     Frontend UI
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- OR Python 3.11+ with pip

### Option 1: Docker (Recommended)

```bash
# Clone or navigate to the project directory
cd researchlens

# Copy environment template
cp .env.example .env

# Start all services
docker-compose up --build

# Access the application
# - Frontend: http://localhost:5500
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Option 2: Manual Setup

```bash
# Navigate to backend directory
cd researchlens/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment
cp ../.env.example .env  # Edit .env with your API keys

# Start the server
uvicorn main:app --reload --port 8000

# In another terminal, serve the frontend
# Option A: Python simple server
cd ../frontend && python -m http.server 5500

# Option B: VS Code Live Server or nginx
```

---

## 📡 API Reference

### `POST /api/search`

Main search endpoint orchestrating the full RAG pipeline.

**Request:**
```json
{
  "query": "transformer attention mechanism",
  "num_papers": 10,
  "preference": "balanced",
  "year_from": 2020,
  "year_to": 2024
}
```

**Response:**
```json
{
  "papers": [...],
  "summaries": [...],
  "comparison": {
    "headers": [...],
    "rows": [...],
    "alignment_analysis": "..."
  },
  "expanded_query": "...",
  "total_fetched": 10,
  "retrieval_method": "hybrid_rag"
}
```

### `POST /api/summarize`

Generate summary for a single paper.

**Request:**
```json
{
  "paper": {...},
  "user_query": "..."
}
```

### `POST /api/compare`

Generate comparative methodology table.

**Request:**
```json
{
  "query": "...",
  "papers": [...],
  "summaries": [...]
}
```

### `POST /api/chat`

Follow-up conversation about retrieved papers.

**Request:**
```json
{
  "message": "Which paper is best for my use case?",
  "context_papers": [...],
  "context_summaries": [...],
  "history": [{"role": "user", "content": "..."}]
}
```

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "embedder_loaded": true,
  "version": "1.0.0"
}
```

---

## 🔬 How Hybrid RAG Works

### 1. Core Chunk Technique

Unlike simple character-based chunking, ResearchLens uses context-aware chunking:

```
Original Abstract: "The transformer architecture uses self-attention... [long text] ..."

Chunk 1: "...uses self-attention mechanism for sequence modeling..."
         [Context: Previous sentence + Current + Next sentence]

Chunk 2: "...sequence modeling which allows parallel computation..."
         [Context: Previous sentence + Current + Next sentence]
```

**Benefits:**
- Never cuts mid-sentence
- Each chunk has surrounding context for better embeddings
- Overlapping windows preserve boundary information

### 2. Hybrid Retrieval with RRF

**Dense Retrieval:**
- Embeddings capture semantic meaning
- Good for synonyms and conceptual matches
- Uses `BAAI/bge-base-en-v1.5` model

**Sparse Retrieval (BM25):**
- Term-frequency based matching
- Good for exact keyword matches
- Handles rare terms well

**Reciprocal Rank Fusion (RRF):**
```
RRF_score = Σ 1/(k + rank)

Where k = 60 (standard constant)
```

Combined score: `0.6 × dense_RRF + 0.4 × sparse_RRF`

### 3. Cross-Encoder Reranking

After initial retrieval, a cross-encoder re-ranks results:
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Scores (query, document) pairs directly
- Final score: `0.5 × cross_score + 0.5 × hybrid_score`

---

## ⚙️ Customization

### Change Embedding Model

Edit `.env`:
```
EMBED_MODEL=BAAI/bge-base-en-v1.5
```

Other options:
- `sentence-transformers/all-MiniLM-L6-v2` (faster, lower quality)
- `BAAI/bge-large-en-v1.5` (better quality, slower)

### Adjust Retrieval Weights

Edit `.env`:
```
DENSE_WEIGHT=0.7
SPARSE_WEIGHT=0.3
RRF_K=60
```

### Change LLM Model

Edit `.env`:
```
LLM_MODEL=claude-opus-4-5
```

Or in `config.py` directly.

### Add New Paper Sources

Edit `services/paper_fetcher.py`:
```python
async def _fetch_new_source(self, query, limit, ...):
    # Implement new API integration
    pass
```

Then add to the fallback chain in `fetch_papers()`.

---

## ⚠️ Limitations

1. **Abstract Only**: The system works with paper abstracts, not full PDFs. PDF parsing would require additional libraries (e.g., PyMuPDF, pdfplumber).

2. **No Vector Database**: Uses in-memory storage for embeddings. Not suitable for large-scale production deployments. For production, consider:
   - Pinecone
   - Weaviate
   - Chroma
   - Qdrant

3. **Rate Limits**: APIs have rate limits:
   - Semantic Scholar: 1 req/sec (free), 10 req/sec (with key)
   - arXiv: 3 req/sec (soft limit)
   - CORE: Varies by plan

4. **API Key Required**: An Anthropic API key is required for LLM calls. The provided key in `.env.example` is a placeholder.

5. **No Streaming**: Chat responses are not streamed yet. For real-time experience, implement Server-Sent Events (SSE).

---

## 📁 Project Structure

```
researchlens/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── config.py            # Configuration
│   ├── requirements.txt     # Dependencies
│   ├── api/
│   │   ├── search.py        # Main search endpoint
│   │   ├── summarize.py     # Paper summarization
│   │   ├── compare.py       # Comparison table
│   │   └── chat.py          # Follow-up chat
│   ├── services/
│   │   ├── paper_fetcher.py # API integration
│   │   ├── query_expander.py# LLM query expansion
│   │   ├── chunker.py       # Core chunking
│   │   ├── embedder.py      # Dense embeddings
│   │   ├── sparse_retriever.py # BM25
│   │   ├── hybrid_retriever.py # RRF fusion
│   │   ├── reranker.py      # Cross-encoder
│   │   ├── summarizer.py    # LLM summaries
│   │   └── comparator.py    # LLM comparison
│   └── models/
│       └── schemas.py        # Pydantic models
├── frontend/
│   ├── index.html           # UI
│   ├── style.css            # Styles
│   ├── app.js               # Frontend logic
│   └── nginx.conf           # Nginx config
├── docker-compose.yml       # Docker setup
├── Dockerfile              # Backend container
├── .env.example            # Environment template
└── README.md               # This file
```

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please read the contribution guidelines first.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

Built with ❤️ using FastAPI, sentence-transformers, and Claude.