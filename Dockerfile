# ResearchLens Backend Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for sentence-transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ .

# Create .env file with defaults if not exists
RUN if [ ! -f .env ]; then \
    echo 'ANTHROPIC_API_KEY=' > .env && \
    echo 'ANTHROPIC_BASE_URL=https://api.opusmax.pro' >> .env && \
    echo 'CORS_ORIGINS=http://localhost:5500,http://127.0.0.1:5500' >> .env; \
    fi

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]