# Financial News RAG System

A unified HTTP interface for financial news ingestion and search, combining REST endpoints for pipeline management with SSE endpoints for MCP protocol communication.

---

## Features

- **Multi-source ingestion**: RSS feeds (Bloomberg, Reuters, WSJ) and APIs (Yahoo Finance)
- **Hybrid search**: Dense (semantic) + sparse (BM25) vectors for optimal retrieval
- **Unified HTTP API**: REST endpoints for pipelines, SSE for MCP protocol
- **Async job management**: Redis-backed job tracking with progress updates
- **Cloud-ready**: Designed for Railway deployment with Qdrant Cloud

---

## Quick Start

```bash
# Start all services (Qdrant, Redis, MCP Server)
docker-compose up -d

# Check services are running
docker-compose ps

# API docs: http://localhost:8000/docs
# Health check: http://localhost:8000/health
```

The Docker Compose automatically starts:
- **Qdrant** - Vector database for embeddings
- **Redis** - Job management and caching
- **MCP Server** - Unified HTTP API (REST + SSE)

---

## API Overview

### Pipeline Management
```bash
# RSS pipeline
POST /api/pipelines/rss/run
{
  "regions": ["usa", "eu"],
  "stocks": ["AAPL", "MSFT"],
  "cleanup_days": 30
}

# API pipeline  
POST /api/pipelines/api/run
{
  "tickers": ["AAPL", "GOOGL"],
  "cleanup_days": 30
}

# Job status
GET /api/jobs/{job_id}
```

### Search Tools
```bash
# Portfolio news (strict matching)
POST /api/portfolio-news
{
  "tickers": ["AAPL", "NVDA"],
  "days_back": 7
}

# Market insights (semantic)
POST /api/market-insights
{
  "tickers": ["NVDA", "AMD"],
  "topics": ["AI", "semiconductors"]
}
```

---

## Documentation

- **[API Reference](docs/README.md)** - Complete API documentation
- **[Local Development](docs/LOCAL_DEVELOPMENT.md)** - Setup guide
- **[Railway Deployment](docs/RAILWAY_DEPLOYMENT.md)** - Production deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FINANCIAL NEWS RAG SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐     ┌─────────────────────┐                       │
│  │  financial_rss_feed │     │ financial_news_api  │                       │
│  │  ─────────────────  │     │  ─────────────────  │                       │
│  │  • RSSFetcher       │     │  • YahooFinance     │                       │
│  │  • Article model    │     │  • ProviderRegistry │                       │
│  │  • Region/Tier      │     │  • BaseProvider     │                       │
│  └──────────┬──────────┘     └──────────┬──────────┘                       │
│             │                           │                                   │
│             └───────────┬───────────────┘                                   │
│                         ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         shared module                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │   Config    │  │  Pipelines  │  │   Search    │  │  Storage   │  │   │
│  │  │  ─────────  │  │  ─────────  │  │  ─────────  │  │  ────────  │  │   │
│  │  │ Env vars    │  │ RSSPipeline │  │SearchClient │  │QdrantStore │  │   │
│  │  │ Defaults    │  │ APIPipeline │  │ Hybrid/     │  │ Singleton  │  │   │
│  │  │ 30+ params  │  │ BasePipeline│  │ Simple      │  │ Pattern    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │  Chunking   │  │   Lexical   │  │ VocabStore  │  │   Logging  │  │   │
│  │  │  ─────────  │  │  ─────────  │  │  ─────────  │  │  ────────  │  │   │
│  │  │TextChunker  │  │LexicalVocab │  │ Redis/JSON  │  │Structured  │  │   │
│  │  │EmbedGen     │  │BM25 weights │  │ abstraction │  │JSON format │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                         │                                                   │
│                         ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Qdrant Vector Database                        │   │
│  │  ┌─────────────────────────┐  ┌─────────────────────────┐           │   │
│  │  │  financial_rss_news     │  │  financial_api_news     │           │   │
│  │  │  • Dense vectors        │  │  • Dense vectors        │           │   │
│  │  │  • Sparse vectors       │  │  • Sparse vectors       │           │   │
│  │  │  • Metadata payloads    │  │  • Metadata payloads    │           │   │
│  │  └─────────────────────────┘  └─────────────────────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                         │                                                   │
│                         ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          MCP Server                                  │   │
│  │  ┌─────────────────────────┐  ┌─────────────────────────┐           │   │
│  │  │  get_portfolio_news     │  │  get_market_insights    │           │   │
│  │  │  (strict ticker match)  │  │  (semantic + keywords)  │           │   │
│  │  └─────────────────────────┘  └─────────────────────────┘           │   │
│  │                    + REST API (/api/portfolio-news, /api/market-insights)│
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
pip install sentence-transformers qdrant-client feedparser yfinance tqdm rank-bm25 mcp fastapi uvicorn pydantic
```

### Basic Usage

```python
from shared import RSSPipeline, APIPipeline, SearchClient

# Run RSS pipeline - vocabularies are automatically saved to the store
RSSPipeline(regions=["usa", "spain"]).run()

# Run API pipeline - vocabularies are automatically saved to the store
APIPipeline(tickers=["AAPL", "NVDA", "MSFT"]).run()

# Search - vocabularies are automatically loaded from the store
client = SearchClient()

# Hybrid search with keyword boosts and date filter
results = client.search(
    query="NVIDIA earnings report",
    mode="hybrid",
    keywords=["NVIDIA", "earnings"],
    keyword_boosts={"NVIDIA": 2.0},
    published_after="2025-01-01",  # Filter by date
)
```

---

## Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE EXECUTION                              │
└──────────────────────────────────────────────────────────────────────────┘

  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
  │  FETCH  │───▶│  CHUNK  │───▶│  EMBED  │───▶│ LEXICAL │───▶│  STORE  │
  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
       │              │              │              │              │
       ▼              ▼              ▼              ▼              ▼
  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
  │ RSS/API │    │ Split   │    │ Generate│    │ Build   │    │ Upsert  │
  │ Sources │    │ into    │    │ dense   │    │ vocab   │    │ to      │
  │         │    │ chunks  │    │ vectors │    │ for BM25│    │ Qdrant  │
  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
                      │
                      ▼
              ┌───────────────┐
              │ Chunk Types:  │
              │ • summary     │
              │ • body        │
              └───────────────┘
```

### Pipeline Steps

| Step | Description | Output |
|------|-------------|--------|
| **Fetch** | Retrieve articles from RSS feeds or API providers | List of Article/NewsArticle objects |
| **Chunk** | Split articles into semantic chunks with overlap | List of Chunk objects with metadata |
| **Embed** | Generate dense vector embeddings | 768-dim vectors (mpnet) |
| **Lexical** | Build BM25 vocabulary for hybrid search | Vocabulary saved to store |
| **Store** | Upsert chunks + embeddings to Qdrant | Indexed vectors with payloads |

---

## Search Modes

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            SEARCH MODES                                   │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐    ┌─────────────────────────────┐
│       SIMPLE SEARCH         │    │       HYBRID SEARCH         │
├─────────────────────────────┤    ├─────────────────────────────┤
│                             │    │                             │
│  Query ──▶ Embed ──▶ Search │    │  Query ──┬──▶ Dense Vector  │
│                      │      │    │          │                  │
│                      ▼      │    │          └──▶ Sparse Vector │
│              ┌───────────┐  │    │                    │        │
│              │  Dense    │  │    │                    ▼        │
│              │  Vector   │  │    │         ┌──────────────────┐│
│              │  Search   │  │    │         │  DBSF Fusion     ││
│              └───────────┘  │    │         │  (score-based)   ││
│                             │    │         └──────────────────┘│
│  Best for:                  │    │                             │
│  • Semantic similarity      │    │  Best for:                  │
│  • Paraphrase matching      │    │  • Exact keyword matching   │
│  • Concept search           │    │  • Ticker symbols ($AAPL)   │
│                             │    │  • Named entities           │
│                             │    │  • Keyword boosting         │
└─────────────────────────────┘    └─────────────────────────────┘
```

### Hybrid Search Features

- **DBSF Fusion**: Uses Distribution-Based Score Fusion (not RRF) so keyword boosts actually affect ranking
- **Auto-loading Vocabularies**: Vocabularies are automatically loaded from the store per collection
- **BM25 Scoring**: Uses `rank_bm25` library for proper Okapi BM25 term weighting
- **Date Filtering**: Filter results by `published_after` date
- **Keyword Match Requirement**: Returns empty results when explicit keywords don't match any documents

### Search Example

```python
from shared import SearchClient

# Vocabularies are automatically loaded from the store - no manual loading needed!
client = SearchClient()

# Simple semantic search
results = client.search("tech company earnings", mode="simple")

# Hybrid search - vocabulary is auto-loaded for the target collection
results = client.search(
    query="NVIDIA GPU demand",
    mode="hybrid",
    keywords=["NVIDIA", "GPU", "demand"],  # Optional: explicit keywords
    keyword_boosts={"NVIDIA": 2.0},        # Optional: boost specific terms
    published_after="2025-01-01",          # Optional: filter by date
)

# Search specific collection (vocabulary auto-loaded per collection)
results = client.search("Federal Reserve", collection="financial_rss_news", mode="hybrid")
results = client.search("Apple iPhone", collection="financial_api_news", mode="hybrid")

# Search with date filter
from datetime import datetime, timedelta
one_week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
results = client.search("market news", published_after=one_week_ago, mode="hybrid")

# Force reload vocabulary from store (e.g., after pipeline update)
client.clear_vocabulary_cache()
```

---

## MCP Server

The system includes an MCP (Model Context Protocol) server for AI agent integration, plus REST API endpoints.

### Running the Server

```bash
# SSE transport (recommended for web clients)
python -m mcp_server.sse_server --host 0.0.0.0 --port 8000

# STDIO transport (for local MCP clients like Claude Desktop)
python -m mcp_server.server
```

### Available Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `get_portfolio_news` | Strict ticker matching - only returns articles explicitly mentioning tickers | Monitoring specific holdings |
| `get_market_insights` | Semantic search with keyword hints - finds related news even without exact matches | Discovering sector trends, risks, opportunities |

### REST API Endpoints

When running the SSE server:

- `POST /api/portfolio-news` - Get news for portfolio tickers
- `POST /api/market-insights` - Get broader market insights
- `GET /health` - Health check
- `GET /docs` - OpenAPI documentation

### Example: Portfolio News

```python
import requests

response = requests.post("http://localhost:8000/api/portfolio-news", json={
    "tickers": ["AAPL", "NVDA", "MSFT"],
    "days_back": 7,
    "limit_per_ticker": 5
})
print(response.json())
```

### Example: Market Insights

```python
response = requests.post("http://localhost:8000/api/market-insights", json={
    "tickers": ["NVDA", "AMD"],
    "topics": ["AI", "semiconductors", "tariffs"],
    "days_back": 7,
    "limit": 15
})
print(response.json())
```

---

## Configuration

All parameters can be set via environment variables with sensible defaults.

### Environment Variables Reference

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION HIERARCHY                            │
└──────────────────────────────────────────────────────────────────────────┘

  Environment Variables  ──▶  Config Class  ──▶  Module Defaults
        (override)              (central)           (fallback)

  export EMBEDDING_MODEL=...
         │
         ▼
  ┌─────────────────┐
  │  Config.py      │
  │  ─────────────  │
  │  Reads env vars │
  │  Provides typed │
  │  access + reload│
  └─────────────────┘
         │
         ▼
  Used by: Pipelines, Search, Chunking, Storage
```

### Embedding Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | Sentence-transformer model name |
| `EMBEDDING_DEVICE` | `cpu` | Device for inference (`cpu`, `cuda`, `mps`) |

**Model Options:**

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | ⚡ Fast | Good | Development, low resources |
| `all-mpnet-base-v2` | 768 | Medium | ⭐ Best | **Production (recommended)** |
| `all-distilroberta-v1` | 768 | Medium | Good | Alternative to mpnet |

### Chunking Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `512` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         CHUNKING STRATEGY                                 │
└──────────────────────────────────────────────────────────────────────────┘

  Article Content:
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5. Sentence 6.│
  └────────────────────────────────────────────────────────────────────────┘

  With CHUNK_SIZE=512, CHUNK_OVERLAP=50:

  Chunk 1: [Sentence 1. Sentence 2. Sentence 3...]
                                    ├──────────┤
                                      overlap
  Chunk 2:              [Sentence 3. Sentence 4. Sentence 5...]
                                                ├──────────┤
                                                  overlap
  Chunk 3:                          [Sentence 5. Sentence 6...]
```

**Tuning Guidelines:**

- **Larger chunks (800-1000)**: Better context, fewer chunks, may miss specific details
- **Smaller chunks (256-400)**: More precise retrieval, more storage, potential context loss
- **More overlap (100-150)**: Better continuity, more redundancy
- **Less overlap (20-30)**: Less redundancy, potential boundary issues

### Qdrant Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_API_KEY` | `None` | API key for Qdrant Cloud |
| `QDRANT_DISTANCE_METRIC` | `cosine` | Distance metric (`cosine`, `dot`, `euclid`) |

**Distance Metrics:**

| Metric | Formula | Best For |
|--------|---------|----------|
| `cosine` | 1 - cos(a,b) | **Recommended** - normalized embeddings |
| `dot` | -a·b | When magnitude matters |
| `euclid` | \|\|a-b\|\| | Absolute distances |

### Search Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_SEARCH_HNSW_EF` | `128` | HNSW search expansion factor |
| `QDRANT_SEARCH_EXACT` | `false` | Use exact search (slower, more accurate) |
| `QDRANT_RESCORING_ENABLED` | `true` | Enable quantization rescoring |
| `QDRANT_RESCORING_OVERSAMPLING` | `2.0` | Oversampling factor for rescoring |
| `SEARCH_LIMIT_DEFAULT` | `10` | Default number of results |
| `SEARCH_DEDUPLICATE` | `true` | Enable result deduplication |
| `SEARCH_SIMILARITY_THRESHOLD` | `0.85` | Text similarity threshold for dedup |

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      HNSW SEARCH PARAMETERS                               │
└──────────────────────────────────────────────────────────────────────────┘

  HNSW_EF (Expansion Factor):
  
  Lower (64)  ──────────────────────────────────────▶  Higher (256)
  ├── Faster search                                    ├── Slower search
  ├── Less accurate                                    ├── More accurate
  └── Lower recall                                     └── Higher recall

  Recommended:
  • Development: 64-128
  • Production: 128-256
  • High accuracy needs: 256-512
```

### Hybrid Search Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_USE_HYBRID` | `true` | Enable hybrid search capability |
| `QDRANT_HYBRID_ALPHA` | `0.5` | Balance between dense and sparse |
| `LEXICAL_K1` | `1.5` | BM25 term frequency saturation |
| `LEXICAL_B` | `0.75` | BM25 length normalization |
| `VOCABULARY_STORE_BACKEND` | `redis` | Storage backend (`redis` or `json`) |
| `VOCABULARY_STORE_PATH` | `artifacts/vocabularies` | Directory for JSON vocabulary storage |

### Redis Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_DB` | `0` | Redis database number |
| `REDIS_PASSWORD` | `None` | Redis password (optional) |
| `REDIS_VOCAB_KEY_PREFIX` | `vocab:` | Key prefix for vocabulary data |

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    REDIS VOCABULARY STORAGE                               │
└──────────────────────────────────────────────────────────────────────────┘

  Each collection uses 3 Redis Hash keys:

  ┌─────────────────────────────────┬─────────────────────────────────────┐
  │ Key                             │ Purpose                             │
  ├─────────────────────────────────┼─────────────────────────────────────┤
  │ vocab:{collection}:meta         │ BM25 metadata                       │
  │                                 │ Fields: doc_count, avg_doc_len,     │
  │                                 │         lowercase, k1, b            │
  ├─────────────────────────────────┼─────────────────────────────────────┤
  │ vocab:{collection}:tokens       │ Token → ID mapping                  │
  │                                 │ Each token is a field,              │
  │                                 │ value is its integer ID             │
  ├─────────────────────────────────┼─────────────────────────────────────┤
  │ vocab:{collection}:idf          │ Token → IDF score                   │
  │                                 │ Each token is a field,              │
  │                                 │ value is its float IDF score        │
  └─────────────────────────────────┴─────────────────────────────────────┘

  Example for collection "financial_api_news":

  vocab:financial_api_news:meta   → {"doc_count": "1000", "avg_doc_len": "45.2", ...}
  vocab:financial_api_news:tokens → {"nvidia": "0", "earnings": "1", "apple": "2", ...}
  vocab:financial_api_news:idf    → {"nvidia": "3.45", "earnings": "1.23", "apple": "2.89", ...}

  Benefits:
  • O(1) single token lookups via HGET (no full vocab load)
  • Incremental updates possible
  • Shared across distributed services
  • Memory-efficient for large vocabularies
```

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       HYBRID ALPHA TUNING                                 │
└──────────────────────────────────────────────────────────────────────────┘

  alpha = 0.0                    alpha = 0.5                    alpha = 1.0
  ┌─────────┐                    ┌─────────┐                    ┌─────────┐
  │ 100%    │                    │  50%    │                    │   0%    │
  │ Sparse  │                    │ Dense + │                    │ Dense   │
  │ (BM25)  │                    │  50%    │                    │  only   │
  │         │                    │ Sparse  │                    │         │
  └─────────┘                    └─────────┘                    └─────────┘
       │                              │                              │
       ▼                              ▼                              ▼
  Best for exact              Best for balanced            Best for semantic
  keyword matching            retrieval (default)          similarity only
```

**BM25 Parameters:**

| Parameter | Effect of Increasing | Typical Range |
|-----------|---------------------|---------------|
| `k1` | More weight to term frequency | 1.2 - 2.0 |
| `b` | More penalty for long documents | 0.5 - 1.0 |

### Pipeline Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_BATCH_SIZE` | `100` | Batch size for Qdrant upserts |
| `PIPELINE_CLEANUP_DAYS` | `90` | Delete documents older than N days |
| `RSS_COLLECTION` | `financial_rss_news` | RSS collection name |
| `API_COLLECTION` | `financial_api_news` | API collection name |

### Logging Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `json` | Log format (`json`, `text`) |

---

## Module Reference

### Pipelines

```python
from shared import RSSPipeline, APIPipeline

# RSS Pipeline
rss = RSSPipeline(
    # Fetch filters
    regions=["usa", "spain"],             # Filter by regions (None = all)
    tiers=["tier1", "tier2"],             # Filter by source tiers (None = all)
    tickers=["AAPL", "NVDA"],             # Generate stock-specific feeds
    company_names=["Apple", "NVIDIA"],    # Generate company-specific feeds
    max_age_hours=24,                     # Max article age
    # Base pipeline options
    collection="my_rss_collection",       # Override collection name
    batch_size=200,                       # Override batch size
    cleanup_days=30,                      # Override cleanup period
    recreate_collection=False,            # Recreate if schema changes
)

result = rss.run(cleanup=True)  # run() only accepts cleanup flag

# API Pipeline
api = APIPipeline(
    # Fetch filters
    tickers=["AAPL", "NVDA", "TSLA"],     # Tickers to fetch
    providers=["yahoo"],                   # Providers to use
    days_back=7,                          # Days of history
    max_articles_per_ticker=10,           # Limit per ticker
    # Base pipeline options
    collection="my_api_collection",       # Override collection name
)

result = api.run()  # cleanup=True by default

# Check available providers
print(APIPipeline.get_available_providers())  # {"yahoo": "yahoo_finance"}
```

### Search Client

```python
from shared import SearchClient

# Vocabularies are automatically loaded from the store
client = SearchClient()

# Basic search - vocabulary auto-loaded for the collection
results = client.search(
    query="NVIDIA earnings",
    collection="financial_api_news",      # Target collection
    mode="hybrid",                        # "simple" or "hybrid"
    limit=20,                             # Max results
    deduplicate=True,                     # Remove duplicates
)

# With keyword boosting
results = client.search(
    query="tech earnings report",
    mode="hybrid",
    keywords=["NVIDIA", "earnings"],      # Explicit keywords
    keyword_boosts={"NVIDIA": 2.0},       # Boost NVIDIA matches
)

# With filters
results = client.search(
    query="market analysis",
    filters={"source_tier": "tier1"},     # Qdrant payload filters
)

# Clear vocabulary cache to force reload from store
client.clear_vocabulary_cache()
client.clear_vocabulary_cache("financial_api_news")  # Clear specific collection
```

### Lexical Vocabulary

```python
from shared import LexicalVocabulary, get_vocabulary_store

# Build from documents
documents = ["Article 1 text...", "Article 2 text..."]
vocab = LexicalVocabulary.build(
    documents=documents,
    lowercase=True,
    min_freq=2,                           # Min document frequency
    k1=1.5,                               # BM25 k1
    b=0.75,                               # BM25 b
)

# Save/Load via VocabularyStore (recommended)
store = get_vocabulary_store()
store.save_vocabulary("my_collection", vocab)
vocab = store.get_vocabulary("my_collection")

# Check if vocabulary exists
if store.exists("my_collection"):
    vocab = store.get_vocabulary("my_collection")

# Generate sparse vector
sparse = vocab.to_sparse_vector(
    keywords=["nvidia", "earnings"],
    boosts={"nvidia": 2.0},
    l2_normalize=True,
)
```

### Vocabulary Store

```python
from shared import get_vocabulary_store, set_vocabulary_store
from shared import JSONVocabularyStore, RedisVocabularyStore

# Get the default store (uses VOCABULARY_STORE_BACKEND config, defaults to Redis)
store = get_vocabulary_store()

# Explicitly request a specific backend
store = get_vocabulary_store(backend="redis")  # Redis backend
store = get_vocabulary_store(backend="json")   # JSON file backend

# Store operations (same interface for both backends)
store.save_vocabulary("collection_name", vocab)  # Save
vocab = store.get_vocabulary("collection_name")  # Load
exists = store.exists("collection_name")          # Check existence
store.delete("collection_name")                   # Delete

# Redis-specific: efficient partial lookups (no full vocab load)
if isinstance(store, RedisVocabularyStore):
    token_id = store.get_token_id("collection_name", "nvidia")
    token_idf = store.get_token_idf("collection_name", "nvidia")
    # Batch lookup for multiple tokens
    batch = store.get_tokens_batch("collection_name", ["nvidia", "earnings"])
    # List all collections
    collections = store.list_collections()

# Use a custom Redis connection
custom_store = RedisVocabularyStore(
    host="redis.example.com",
    port=6379,
    password="secret",
    key_prefix="myapp:vocab:",
)
set_vocabulary_store(custom_store)

# Use JSON file backend with custom path
json_store = JSONVocabularyStore(base_dir="/path/to/vocabs")
set_vocabulary_store(json_store)
```

### Configuration Access

```python
from shared import Config

# Read current values
print(Config.EMBEDDING_MODEL)
print(Config.QDRANT_HOST)

# Reload from environment
Config.reload()

# Get all as dict
print(Config.as_dict())
```

### Adding New API Providers

The system uses a `ProviderRegistry` pattern for extensibility:

```python
# 1. Create a new provider in financial_news_api_feed/providers/
from financial_news_api_feed.base_provider import BaseNewsProvider
from financial_news_api_feed.models import NewsArticle

class AlphaVantageProvider(BaseNewsProvider):
    def get_provider_name(self) -> str:
        return "alpha_vantage"
    
    def fetch_ticker_news(self, ticker: str, max_results=None, days_back=7):
        # Implementation here
        pass
    
    def fetch_multiple_tickers(self, tickers, max_results=None, days_back=7):
        # Implementation here
        pass

# 2. Register in financial_news_api_feed/providers/__init__.py
from ..provider_registry import ProviderRegistry
ProviderRegistry.register('alpha_vantage', AlphaVantageProvider)

# 3. Add mapping in shared/pipelines/api.py _get_provider()
provider_map = {
    "yahoo": "yahoo_finance",
    "alphavantage": "alpha_vantage",  # Add this
}

# 4. Use in pipeline
APIPipeline(providers=["yahoo", "alphavantage"]).run()
```

---

## Example: Complete Workflow

```python
import os
from shared import RSSPipeline, APIPipeline, SearchClient, Config

# 1. Configure via environment (optional)
os.environ["EMBEDDING_MODEL"] = "sentence-transformers/all-mpnet-base-v2"
os.environ["QDRANT_HOST"] = "localhost"
Config.reload()

# 2. Run RSS pipeline - vocabulary automatically saved to store
print("Running RSS pipeline...")
rss_result = RSSPipeline(
    regions=["usa"],
    tiers=["tier1", "tier2"],
    max_age_hours=24,
).run()
print(f"RSS: {rss_result['articles_fetched']} articles, {rss_result['chunks_stored']} chunks")

# 3. Run API pipeline - vocabulary automatically saved to store
print("Running API pipeline...")
api_result = APIPipeline(
    tickers=["AAPL", "NVDA", "MSFT", "GOOGL"],
    days_back=7,
).run()
print(f"API: {api_result['articles_fetched']} articles, {api_result['chunks_stored']} chunks")

# 4. Search - vocabulary automatically loaded from store
print("Searching...")
client = SearchClient()

results = client.search(
    query="NVIDIA AI chip demand",
    mode="hybrid",
    keywords=["NVIDIA", "AI"],
    keyword_boosts={"NVIDIA": 2.0},
    limit=5,
)

for r in results:
    print(f"  [{r['score']:.3f}] {r['metadata'].get('article_title', 'N/A')[:60]}...")
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing dependency | `pip install sentence-transformers qdrant-client` |
| `Connection refused` | Qdrant not running | Start Qdrant: `docker run -p 6333:6333 qdrant/qdrant` |
| `Dimension mismatch` | Changed embedding model | Set `recreate_collection=True` or delete collection |
| `No results` | Empty collection | Run pipeline first |
| `Hybrid search fallback` | No lexical vocab | Run pipeline first to generate vocabulary |
| `Redis connection refused` | Redis not running | Start Redis: `docker run -p 6379:6379 redis` |
| `redis package not found` | Missing dependency | `pip install redis` |

### Performance Tips

1. **Use GPU for embeddings**: Set `EMBEDDING_DEVICE=cuda`
2. **Batch size tuning**: Increase `PIPELINE_BATCH_SIZE` for faster ingestion
3. **HNSW tuning**: Lower `QDRANT_SEARCH_HNSW_EF` for faster (less accurate) search
4. **Disable rescoring**: Set `QDRANT_RESCORING_ENABLED=false` for speed

---

## License

MIT License
