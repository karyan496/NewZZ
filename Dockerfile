FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml .
COPY uv.lock* .

# Install dependencies
RUN uv sync --no-dev

# Copy the rest of the app
COPY . .

# Pre-download the sentence-transformer model so it's baked into the image
# and doesn't download at runtime on every cold start
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create data directory for FAISS index persistence
RUN mkdir -p /app/data

CMD ["uv", "run", "python", "main.py"]