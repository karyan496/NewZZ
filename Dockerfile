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

# Create temp directories
RUN mkdir -p /tmp/faiss /tmp/huggingface

# HuggingFace will download the model here at runtime
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface

CMD ["uv", "run", "python", "main.py"]