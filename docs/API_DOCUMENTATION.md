# Financial News RAG Server - API Documentation

## Overview

The Financial News RAG Server provides HTTP endpoints for:
- Async pipeline management (RSS and API news ingestion)
- News search and insights retrieval
- Job status tracking

## Base URL

```
http://localhost:8000
```

## Authentication

Currently no authentication required. For production, implement API key validation at proxy level.

## Rate Limits

No built-in rate limiting. Recommended limits:
- Search endpoints: 100 requests/minute
- Pipeline endpoints: 10 requests/minute
- Job status: 60 requests/minute

## Available Endpoints

### Pipeline Management (REST)

#### 1. Run RSS Pipeline

Run async RSS news ingestion pipeline with specified filters.

**Endpoint**: `POST /api/pipelines/rss/run`

**Parameters**:
- `regions` (optional): List of regions to filter
  - Type: `array[string]`
  - Valid: `["usa", "spain", "germany", "uk", "china", "france", "italy", "japan", "india", "brazil", "australia", "canada", "asia_pacific", "middle_east", "africa", "eu_general", "global"]`
- `stocks` (optional): List of stock tickers
  - Type: `array[string]`
  - Example: `["AAPL", "MSFT"]`
- `company_names` (optional): List of company names
  - Type: `array[string]`
  - Example: `["Apple Inc.", "Google LLC"]`
- `cleanup_days` (optional): Days to keep old articles
  - Type: `integer`
  - Default: `30`
  - Range: `1-365`

**Response**:
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "type": "rss",
  "status": "pending",
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/api/pipelines/rss/run" \
  -H "Content-Type: application/json" \
  -d '{
    "regions": ["usa", "eu"],
    "stocks": ["AAPL", "MSFT"],
    "company_names": ["Apple Inc."],
    "cleanup_days": 30
  }'
```

#### 2. Run API Pipeline

Run async API news ingestion pipeline for specified tickers.

**Endpoint**: `POST /api/pipelines/api/run`

**Parameters**:
- `tickers` (required): List of stock tickers
  - Type: `array[string]`
  - Example: `["AAPL", "GOOGL", "NVDA"]`
- `cleanup_days` (optional): Days to keep old articles
  - Type: `integer`
  - Default: `30`
  - Range: `1-365`

**Response**: Same as RSS pipeline

**Example**:
```bash
curl -X POST "http://localhost:8000/api/pipelines/api/run" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "GOOGL", "NVDA"],
    "cleanup_days": 30
  }'
```

#### 3. Get Job Status

Check the status and results of a pipeline job.

**Endpoint**: `GET /api/jobs/{job_id}`

**Response**:
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "type": "rss",
  "status": "completed",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:05Z",
  "completed_at": "2024-01-15T10:32:15Z",
  "progress": {
    "stage": "completed",
    "message": "Job completed successfully"
  },
  "result": {
    "pipeline_type": "rss",
    "stats": {
      "articles_processed": 150,
      "chunks_created": 450
    }
  }
}
```

**Status Values**:
- `pending`: Job queued
- `running`: Job executing
- `completed`: Job finished successfully
- `failed`: Job failed with error

### Search Tools

#### 1. get_portfolio_news

Get news strictly related to portfolio tickers using keyword matching.

**Purpose**: Monitor specific holdings in a portfolio with high precision.

**Note**: Searches across all available collections (RSS feeds and API sources) and merges results.

**Parameters**:
- `tickers` (required): List of ticker symbols
  - Type: `array[string]`
  - Example: `["AAPL", "NVDA", "MSFT"]`
- `days_back` (optional): How many days back to search
  - Type: `integer`
  - Default: `7`
  - Range: `1-90`
- `limit_per_ticker` (optional): Maximum articles per ticker
  - Type: `integer`
  - Default: `5`
  - Range: `1-20`

**Response Structure**:
```json
{
  "portfolio": ["AAPL", "NVDA", "MSFT"],
  "days_back": 7,
  "total_articles": 15,
  "by_ticker": {
    "AAPL": {
      "count": 5,
      "articles": [
        {
          "score": 0.8942,
          "title": "Apple Reports Strong Q4 Earnings",
          "url": "https://example.com/apple-earnings",
          "published_date": "2024-01-15T10:30:00",
          "source": "Yahoo Finance",
          "tickers": ["AAPL"],
          "text": "Apple Inc. reported better-than-expected...",
          "collection": "financial_api_news"
        }
      ]
    }
  },
  "recent_headlines": [
    {
      "ticker": "AAPL",
      "title": "Apple Reports Strong Q4 Earnings",
      "date": "2024-01-15"
    }
  ]
}
```

**MCP Tool Call Example**:
```python
# Via MCP protocol
tool_call = {
  "name": "get_portfolio_news",
  "arguments": {
    "tickers": ["AAPL", "NVDA", "MSFT"],
    "days_back": 7,
    "limit_per_ticker": 5
  }
}
```

**REST API Example**:
```bash
curl -X POST "http://localhost:8000/api/portfolio-news" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "NVDA", "MSFT"],
    "days_back": 7,
    "limit_per_ticker": 5
  }'
```

### 2. get_market_insights

Get broader market insights, sector news, risks, and opportunities.

**Purpose**: Discover sector trends, competitor moves, and market-wide risks/opportunities.

**Note**: Searches across all available collections (RSS feeds and API sources) and merges results.

**Parameters**:
- `tickers` (optional): List of tickers for context
  - Type: `array[string]`
  - Example: `["NVDA", "AMD"]`
- `topics` (optional): Topics to focus on
  - Type: `array[string]`
  - Example: `["AI", "semiconductors", "tariffs"]`
- `days_back` (optional): How many days back to search
  - Type: `integer`
  - Default: `7`
  - Range: `1-90`
- `limit` (optional): Maximum total results
  - Type: `integer`
  - Default: `15`
  - Range: `1-50`

**Response Structure**:
```json
{
  "context": {
    "tickers": ["NVDA", "AMD"],
    "topics": ["AI", "semiconductors"],
    "days_back": 7
  },
  "total_articles": 15,
  "categories": {
    "risks": {
      "count": 3,
      "articles": [...]
    },
    "opportunities": {
      "count": 4,
      "articles": [...]
    },
    "sector_news": {
      "count": 5,
      "articles": [...]
    },
    "earnings": {
      "count": 2,
      "articles": [...]
    }
  },
  "all_results": [
    {
      "score": 0.8567,
      "title": "AI Chip Demand Surges as Tech Giants Expand",
      "url": "https://example.com/ai-chip-demand",
      "published_date": "2024-01-15T09:15:00",
      "source": "Reuters",
      "tickers": ["NVDA", "AMD"],
      "text": "The demand for AI chips continues to rise...",
      "collection": "financial_rss_news"
    }
  ]
}
```

**MCP Tool Call Example**:
```python
# Via MCP protocol
tool_call = {
  "name": "get_market_insights",
  "arguments": {
    "tickers": ["NVDA", "AMD"],
    "topics": ["AI", "semiconductors"],
    "days_back": 7,
    "limit": 15
  }
}
```

**REST API Example**:
```bash
curl -X POST "http://localhost:8000/api/market-insights" \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["NVDA", "AMD"],
    "topics": ["AI", "semiconductors"],
    "days_back": 7,
    "limit": 15
  }'
```

## Search Behavior

### Portfolio News (Strict Matching)
- Uses hybrid search with `require_keyword_match=True`
- Only returns articles that explicitly mention the ticker
- Keyword boost: 3.0 for exact ticker matches
- Deduplicates results across tickers

### Market Insights (Semantic Search)
- Uses semantic search with optional keyword hints
- Finds related news even without explicit ticker mentions
- Keyword boost: 1.5 for tickers, 2.0 for topics
- Auto-categorizes results into:
  - Risks (warning, decline, concern, threat, etc.)
  - Opportunities (growth, surge, bullish, partnership, etc.)
  - Sector news (industry, competition, trends)
  - Earnings (revenue, profit, quarter, guidance)
  - General (other news)

## Configuration

The API uses environment variables for configuration:

### Required
```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Optional
```bash
QDRANT_API_KEY=your_api_key
REDIS_PASSWORD=your_password
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DEVICE=cpu
RSS_COLLECTION=financial_rss_news
API_COLLECTION=financial_api_news
```

## Async Job Management

The pipeline endpoints use an async job management system:

1. **Submit Job**: POST to pipeline endpoint returns a `job_id`
2. **Track Progress**: Poll `/api/jobs/{job_id}` for status updates
3. **Get Results**: Once `status` is `completed`, results are in the `result` field

### Job Lifecycle

```bash
# 1. Submit RSS pipeline job
JOB_ID=$(curl -s -X POST "http://localhost:8000/api/pipelines/rss/run" \
  -H "Content-Type: application/json" \
  -d '{"regions": ["usa"]}' | jq -r .job_id)

# 2. Poll for completion
while true; do
  STATUS=$(curl -s "http://localhost:8000/api/jobs/$JOB_ID" | jq -r .status)
  echo "Status: $STATUS"
  if [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]]; then
    break
  fi
  sleep 2
done

# 3. Get results
curl -s "http://localhost:8000/api/jobs/$JOB_ID" | jq .result
```

### Job Progress Updates

Jobs provide progress information during execution:
- `pending`: Job queued
- `initializing`: Setting up pipeline
- `running`: Fetching/processing articles
- `cleanup`: Removing old articles
- `completed`: Job finished
- `failed`: Error occurred (check `error` field)

### Job TTL and Cleanup

- Jobs are stored in Redis with a TTL of 24 hours
- Completed jobs are automatically cleaned up after 7 days
- Failed jobs are kept for 24 hours for debugging
- Use job results promptly to avoid data loss

### Timeouts

- Pipeline jobs timeout after 30 minutes
- Search queries timeout after 10 seconds
- Job status polling should use intervals > 1 second

## Error Handling

### Common Error Responses

**Invalid Ticker Symbol**:
```json
{
  "error": "No articles found for ticker: INVALID"
}
```

**Connection Error**:
```json
{
  "error": "Failed to connect to Qdrant at localhost:6333"
}
```

**Invalid Parameters**:
```json
{
  "error": "days_back must be between 1 and 90"
}
```

## Integration Examples

### Python Client (REST API)
```python
import requests
import json

class FinancialNewsClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def get_portfolio_news(self, tickers, days_back=7, limit_per_ticker=5):
        response = requests.post(
            f"{self.base_url}/api/portfolio-news",
            json={
                "tickers": tickers,
                "days_back": days_back,
                "limit_per_ticker": limit_per_ticker
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_market_insights(self, tickers=None, topics=None, days_back=7, limit=15):
        response = requests.post(
            f"{self.base_url}/api/market-insights",
            json={
                "tickers": tickers,
                "topics": topics,
                "days_back": days_back,
                "limit": limit
            }
        )
        response.raise_for_status()
        return response.json()

# Usage
client = FinancialNewsClient()

# Get portfolio news
portfolio_news = client.get_portfolio_news(["AAPL", "GOOGL", "MSFT"])

# Get market insights
ai_insights = client.get_market_insights(
    tickers=["NVDA", "AMD"],
    topics=["AI", "semiconductors"]
)
```

### JavaScript/TypeScript Client
```typescript
class FinancialNewsClient {
  private baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  async getPortfolioNews(
    tickers: string[],
    daysBack: number = 7,
    limitPerTicker: number = 5
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/portfolio-news`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tickers,
        days_back: daysBack,
        limit_per_ticker: limitPerTicker,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async getMarketInsights(
    tickers?: string[],
    topics?: string[],
    daysBack: number = 7,
    limit: number = 15
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/market-insights`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tickers,
        topics,
        days_back: daysBack,
        limit,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }
}

// Usage
const client = new FinancialNewsClient();

// Get portfolio news
const portfolioNews = await client.getPortfolioNews(["AAPL", "GOOGL", "MSFT"]);

// Get market insights
const aiInsights = await client.getMarketInsights(
  ["NVDA", "AMD"],
  ["AI", "semiconductors"]
);
```

## Performance Considerations

1. **Response Time**: Typically 200-500ms for simple queries
2. **Rate Limiting**: No built-in rate limiting (implement at proxy if needed)
3. **Concurrent Requests**: Supports multiple concurrent requests
4. **Caching**: Redis caching improves repeated query performance

## Data Sources

The system aggregates news from:
- RSS feeds (financial news sources)
- Yahoo Finance API
- Alpha Vantage (optional)
- Additional providers can be added via the provider registry

Data is continuously updated and stored in Qdrant with:
- Semantic embeddings for search
- Full-text search capabilities
- Ticker extraction and tagging
- Deduplication across sources

## Security Notes

1. **API Keys**: Store securely in environment variables
2. **Network**: Use HTTPS in production
3. **Authentication**: Add API key authentication if exposing publicly
4. **Input Validation**: Server validates all input parameters
5. **CORS**: Configure appropriately for web clients

## Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "ok",
  "server": "financial-news-rag",
  "tools": ["get_portfolio_news", "get_market_insights"]
}
```

## Error Handling

### Common Error Responses

**Invalid Ticker Symbol**:
```json
{
  "error": "No articles found for ticker: INVALID"
}
```

**Connection Error**:
```json
{
  "error": "Failed to connect to Qdrant at localhost:6333"
}
```

**Invalid Parameters**:
```json
{
  "error": "days_back must be between 1 and 90"
}
```

**Job Not Found**:
```json
{
  "error": "Job not found: {job_id}"
}
```

**Pipeline Timeout**:
```json
{
  "error": "Pipeline job timed out after 30 minutes"
}
```

## MCP Client Configuration

To use this server as an MCP (Model Context Protocol) provider, add it to your client's configuration:

### Claude Desktop Configuration

#### Option 1: HTTP/SSE Transport (Recommended)

Add to your `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "financial-news-rag": {
      "url": "http://localhost:8000/mcp",
      "transport": "streamable_http"
    }
  }
}
```
