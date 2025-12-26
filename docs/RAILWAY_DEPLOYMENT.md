# Railway Deployment Guide

## Overview

This guide explains how to deploy the Financial News RAG MCP Server to Railway. The deployment uses the SSE server mode as a web service, with external services for Qdrant and Redis.

## Architecture

```
Railway Web Service (SSE Server)
    ↓
Qdrant Cloud (Vector Database)
    ↓
Redis (Cache) - Railway Plugin or Redis Cloud
```

## Prerequisites

- Railway account
- Qdrant Cloud account (free tier available)
- Domain name (optional, for custom URL)

## Deployment Steps

### 1. Prepare Your Repository

Ensure your repository contains:
- `railway.json` configuration file
- `Procfile` (alternative to railway.json)
- `Dockerfile` for container build
- `requirements.txt` with all dependencies

### 2. Set Up Qdrant Cloud

1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a new cluster
3. Get the connection details:
   - Host URL (e.g., `xyz.aws.qdrant.cloud`)
   - Port (usually 6333)
   - API Key

### 3. Deploy to Railway

#### Option A: Via GitHub Integration

1. Connect your GitHub repository to Railway
2. Railway will automatically detect the Python application
3. Configure environment variables (see below)
4. Deploy!

#### Option B: Via Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Link to repository
railway link

# Set environment variables
railway variables set QDRANT_HOST=xyz.aws.qdrant.cloud
railway variables set QDRANT_PORT=6333
railway variables set QDRANT_API_KEY=your_api_key

# Deploy
railway up
```

### 4. Configure Environment Variables

Set these in Railway dashboard:

```bash
# Qdrant Configuration
QDRANT_HOST=xyz.aws.qdrant.cloud
QDRANT_PORT=6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_USE_HYBRID=true
QDRANT_HYBRID_ALPHA=0.5

# Redis Configuration (if using Railway Redis plugin)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Alternative: Redis Cloud
# REDIS_HOST=your-redis-cloud-host
# REDIS_PORT=12345
# REDIS_PASSWORD=your_redis_password

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DEVICE=cpu

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Collections
RSS_COLLECTION=financial_rss_news
API_COLLECTION=financial_api_news
```

### 5. Add Redis Plugin (Optional)

1. In Railway project, click "New Service"
2. Select "Redis" plugin
3. It will be available at `localhost:6379`
4. Set `REDIS_HOST=localhost` in environment variables

### 6. Verify Deployment

1. Check deployment logs in Railway dashboard
2. Test health endpoint:
   ```bash
   curl https://your-app.railway.app/health
   ```
3. Test API endpoint:
   ```bash
   curl -X POST https://your-app.railway.app/api/portfolio-news \
     -H "Content-Type: application/json" \
     -d '{"tickers": ["AAPL"], "days_back": 7}'
   ```

## Configuration Files

### railway.json

```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python -m mcp_server.sse_server --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### Procfile (Alternative)

```
web: python -m mcp_server.sse_server --host 0.0.0.0 --port $PORT
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["python", "-m", "mcp_server.sse_server", "--host", "0.0.0.0", "--port", "$PORT"]
```

## Production Considerations

### 1. Model Download Caching

The embedding model downloads on first cold start. To optimize:

```dockerfile
# Pre-download model in Dockerfile
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
```

### 2. Resource Limits

Railway's free tier has limitations:
- CPU: 0.25 vCPU
- Memory: 512MB
- Build time: 5 minutes

For production:
- Use Eco plan or higher
- Monitor resource usage
- Consider smaller models if memory constrained

### 3. Scaling

Railway automatically scales based on load:
- Minimum: 1 instance
- Maximum: Configurable in settings
- Scaling based on CPU/memory usage

### 4. Monitoring

Enable Railway's built-in monitoring:
- Metrics dashboard
- Log aggregation
- Error tracking
- Performance insights

### 5. Custom Domain

1. Go to project settings
2. Click "Custom Domain"
3. Add your domain
4. Update DNS with provided CNAME

## Data Ingestion in Production

### Option 1: Scheduled Jobs

Use Railway's cron jobs or external scheduler:

```bash
# Add to railway.json
{
  "cron": {
    "ingest": {
      "schedule": "0 */6 * * *",
      "command": "python -m financial_rss_feed.rss_fetcher --days-back 1"
    }
  }
}
```

### Option 2: Separate Ingestion Service

Deploy a separate Railway service for ingestion:
- Runs continuously
- Processes feeds at intervals
- Updates vector database

### Option 3: Webhook Triggers

Implement webhook endpoints to trigger ingestion:
```python
@app.post("/ingest/rss")
async def trigger_rss_ingestion():
    # Trigger RSS ingestion
    pass
```

## Security Best Practices

### 1. API Keys

- Store all API keys in Railway environment variables
- Never commit secrets to git
- Rotate keys regularly

### 2. Rate Limiting

Implement rate limiting for API endpoints:
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/portfolio-news")
@limiter.limit("10/minute")
async def api_portfolio_news(request: Request, ...):
    pass
```

### 3. CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)
```

## Troubleshooting

### Common Issues

1. **Build Timeouts**
   - Reduce model size
   - Use Docker layer caching
   - Optimize Dockerfile

2. **Memory Errors**
   - Use smaller embedding model
   - Reduce chunk size
   - Enable streaming

3. **Connection Refused**
   - Check environment variables
   - Verify Qdrant cluster status
   - Check Redis configuration

4. **Cold Start Delays**
   - Pre-download models
   - Use Railway's always-on feature
   - Optimize initialization code

### Debugging

View logs in Railway dashboard or via CLI:

```bash
# View logs
railway logs

# Stream logs
railway logs --follow

# View specific service logs
railway logs --service <service-name>
```

## Cost Optimization

### Free Tier Usage

- Qdrant Cloud: 1GB storage, 10k requests/month
- Railway: 500 hours/month, 100GB bandwidth
- Redis: Free tier available

### Reducing Costs

1. **Model Selection**
   - Use smaller models: `all-MiniLM-L6-v2`
   - Consider local embeddings for high volume

2. **Storage Optimization**
   - Regular cleanup of old articles
   - Compress embeddings
   - Use efficient chunking

3. **Request Optimization**
   - Implement caching
   - Batch requests
   - Use pagination

## Migration from Local

### 1. Export Data

```bash
# Export from local Qdrant
curl -X GET http://localhost:6333/collections/financial_rss_news/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit": 1000}' > local_data.json
```

### 2. Import to Cloud

```bash
# Import to Qdrant Cloud
curl -X PUT https://xyz.aws.qdrant.cloud/collections/financial_rss_news/points \
  -H "api-key: your_key" \
  -H "Content-Type: application/json" \
  -d @local_data.json
```

### 3. Update Configuration

Update environment variables to point to cloud services.

## Performance Tuning

### 1. Database Optimization

```bash
# Qdrant settings for production
QDRANT_SEARCH_HNSW_EF=256
QDRANT_RESCORING_ENABLED=true
QDRANT_RESCORING_OVERSAMPLING=2.5
```

### 2. Caching Strategy

```bash
# Redis TTL settings
REDIS_CACHE_TTL=3600
REDIS_RESULT_CACHE_SIZE=1000
```

### 3. Monitoring Metrics

Track these metrics:
- Response time (p50, p95, p99)
- Error rate
- Memory usage
- CPU utilization
- Request rate

## Backup and Recovery

### 1. Qdrant Backups

Qdrant Cloud provides automatic backups. For manual:

```bash
# Create snapshot
curl -X POST https://xyz.aws.qdrant.cloud/collections/financial_rss_news/snapshots \
  -H "api-key: your_key"
```

### 2. Configuration Backup

Export environment variables:

```bash
railway variables get > production.env
```

### 3. Disaster Recovery

1. Restore Qdrant from snapshot
2. Redeploy Railway service
3. Restore environment variables
4. Verify functionality

## Next Steps

1. Set up monitoring alerts
2. Configure automated testing
3. Implement CI/CD pipeline
4. Add analytics and usage tracking
5. Set up multi-region deployment for HA

## Support

- Railway documentation: https://docs.railway.app/
- Qdrant Cloud: https://cloud.qdrant.io/docs
- Project issues: Create GitHub issue
