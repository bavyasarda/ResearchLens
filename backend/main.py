"""
ResearchLens - FastAPI Backend Application
Main entry point with CORS, router mounting, and health check.
"""
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import CORS_ORIGINS, HOST, PORT
from backend.api.search import router as search_router
from backend.api.summarize import router as summarize_router
from backend.api.compare import router as compare_router
from backend.api.chat import router as chat_router
from backend.services.paper_fetcher import close_paper_fetcher
from backend.services.embedder import preload_embedder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    logger.info("Starting ResearchLens backend...")

    # Preload embedding model
    logger.info("Preloading embedding model...")
    preload_embedder()
    logger.info("Embedding model preloaded")

    yield

    # Shutdown
    logger.info("Shutting down ResearchLens backend...")
    await close_paper_fetcher()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ResearchLens API",
    description="AI-powered research paper search engine with hybrid RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions with structured JSON error."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url),
        },
    )


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint.
    Returns the status of the service and whether the embedder is loaded.
    """
    return {
        "status": "healthy",
        "embedder_loaded": True,
        "version": "1.0.0",
    }


# Mount routers
app.include_router(search_router, prefix="/api")
app.include_router(summarize_router, prefix="/api")
app.include_router(compare_router, prefix="/api")
app.include_router(chat_router, prefix="/api/chat")


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "ResearchLens API",
        "version": "1.0.0",
        "description": "AI-powered research paper search engine with hybrid RAG",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {HOST}:{PORT}")
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info",
    )