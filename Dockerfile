FROM python:3.11-slim

# Metadata
LABEL maintainer="saas-audit-env"
LABEL description="SaaSAuditEnv - AI agent SaaS cost audit environment"

# Prevent Python from writing .pyc files and enable stdout/stderr flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HF Spaces uses port 7860 by default
ENV PORT=7860

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Switch to non-root user
USER appuser

# Expose port (HF Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

# Run with Hypercorn — use PORT env var for HF Spaces compatibility
CMD hypercorn app.main:app --bind 0.0.0.0:${PORT} --workers 1
