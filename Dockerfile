# Dockerfile for Agentic Harness Build
# Uses Python 3.11 and uv for dependency management

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY agents/ ./agents/
COPY schemas/ ./schemas/
COPY harness.py ./
COPY README.md ./

# Install dependencies
RUN uv sync

# Install Playwright browsers
RUN uv run playwright install chromium

# Create necessary directories
RUN mkdir -p workspace data logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_URL=http://host.docker.internal:11434/v1

# Default command
CMD ["uv", "run", "python", "harness.py"]
