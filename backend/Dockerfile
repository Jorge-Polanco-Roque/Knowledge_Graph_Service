# ============================================================
# Stage 1: Builder — install dependencies in isolated layer
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building wheels (chromadb, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# ============================================================
# Stage 2: Runtime — minimal image, no build tools
# ============================================================
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd -r neurograph && useradd -r -g neurograph -m -s /bin/false neurograph

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

# App code
WORKDIR /app
COPY app/ app/

# Create mount points with correct ownership
RUN mkdir -p /data /inputs && chown -R neurograph:neurograph /data /inputs /app

# Switch to non-root user
USER neurograph

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8899/health')" || exit 1

EXPOSE 8899

# Uvicorn with production settings
# Single worker: NetworkX graph is in-process memory — multiple workers = isolated graphs
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8899", \
     "--workers", "1", \
     "--timeout-keep-alive", "120", \
     "--log-level", "warning"]
