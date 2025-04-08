# Use Python 3.10 slim as base image
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install base dependencies first
RUN pip install --no-cache-dir \
    fastapi==0.109.2 \
    uvicorn==0.27.1 \
    python-multipart==0.0.9 \
    jinja2==3.1.3 \
    python-dotenv==1.0.1 \
    langchain==0.1.9 \
    langchain-openai==0.0.8 \
    langchain-community==0.0.27 \
    chromadb==0.4.22 \
    pypdf>=3.17.1 \
    tiktoken>=0.6.0

# Install ML dependencies with optimizations
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    sentence-transformers==2.2.2 \
    tokenizers==0.13.3 \
    transformers==4.30.2 \
    accelerate==0.27.2 \
    bitsandbytes==0.42.0 \
    langchain-huggingface==0.0.1

# Final stage
FROM python:3.10-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/model_cache /app/templates /app/static

# Copy application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 