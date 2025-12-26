# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH=/home/app/.local/bin:$PATH

# Set work directory
WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/app/.local

# Copy project files
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Run the application
CMD ["python", "-m", "mcp_server.sse_server", "--host", "0.0.0.0"]
