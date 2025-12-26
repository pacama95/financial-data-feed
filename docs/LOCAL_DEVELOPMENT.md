# Local Development Guide

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- Redis (optional, can run with Docker)
- Git

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd financial-data-feed
```

### 2. Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### 4. Start Required Services

#### Option A: Using Docker Compose (Recommended)

```bash
# Start all services (Qdrant, Redis, MCP Server)
docker-compose up -d

# Verify services are running
docker-compose ps

# View logs
docker-compose logs -f mcp-server

# Services will be available at:
# - Qdrant: http://localhost:6333
# - Redis: localhost:6379
# - MCP Server: http://localhost:8000
```

#### Option B: Manual Setup

```bash
# Start Qdrant
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Start Redis (optional, for caching)
docker run -p 6379:6379 redis:latest

# Run MCP server
python -m mcp_server.sse_server
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the template
cp .env.example .env

# Edit with your configuration
nano .env
```

Example `.env` file:
```bash
# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=

# Cache (Redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DEVICE=cpu

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Collections
RSS_COLLECTION=financial_rss_news
API_COLLECTION=financial_api_news
```

### 6. Initialize the Database

```bash
# Run database initialization
python -m shared.qdrant_store --init

# Verify collections exist
curl http://localhost:6333/collections
```

### 7. Ingest Initial Data

```bash
# Ingest RSS news
python -m financial_rss_feed.rss_fetcher --days-back 7

# Ingest API news (requires API keys)
python -m financial_news_api_feed.provider_registry --providers yahoo
```

## Running the MCP Server

### Option 1: Stdio Server (for Claude Desktop)

```bash
# Run the stdio server
python -m mcp_server.server
```

Test with:
```bash
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | python -m mcp_server.server
```

### Option 2: SSE Server (for web clients)

```bash
# Run the SSE server
python -m mcp_server.sse_server

# With custom host/port
python -m mcp_server.sse_server --host 0.0.0.0 --port 8000
```

Access endpoints:
- Health: http://localhost:8000/health
- SSE: http://localhost:8000/sse
- API Docs: http://localhost:8000/docs

## Development Workflow

### 1. Code Structure

```
financial-data-feed/
├── mcp_server/          # MCP server implementation
│   ├── server.py       # Stdio server
│   └── sse_server.py   # SSE/REST server
├── shared/             # Shared utilities
│   ├── config.py       # Configuration management
│   ├── search.py       # Search client
│   ├── qdrant_store.py # Vector database operations
│   └── pipelines/      # Data ingestion pipelines
├── financial_rss_feed/ # RSS news ingestion
├── financial_news_api_feed/ # API news ingestion
├── tests/              # Test suite
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

### 2. Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=shared --cov=mcp_server

# Run specific test file
pytest tests/test_search.py

# Run with verbose output
pytest -v
```

### 3. Code Formatting and Linting

```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Lint with flake8
flake8 .

# Type checking with mypy
mypy .
```

### 4. Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

Now hooks will run automatically on each commit.

## Data Ingestion

### RSS Feed Ingestion

```bash
# Ingest from all configured RSS feeds
python -m financial_rss_feed.rss_fetcher

# Ingest specific number of days
python -m financial_rss_feed.rss_fetcher --days-back 3

# Run continuously
python -m financial_rss_feed.rss_fetcher --continuous --interval 300
```

### API Feed Ingestion

```bash
# List available providers
python -m financial_news_api_feed.provider_registry --list

# Ingest from specific provider
python -m financial_news_api_feed.provider_registry --providers yahoo

# Ingest from multiple providers
python -m financial_news_api_feed.provider_registry --providers yahoo,alpha_vantage

# Configure API keys in .env
YAHOO_FINANCE_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
```

### Custom Data Pipeline

Create a custom pipeline in `shared/pipelines/`:

```python
from shared.pipelines.base import BasePipeline

class MyCustomPipeline(BasePipeline):
    def process_batch(self, batch):
        # Process your data
        articles = [...]
        
        # Store in vector database
        self.store_articles(articles)
        
        return len(articles)
```

## Monitoring and Debugging

### 1. Logs

View logs in JSON format:
```bash
# Tail logs
tail -f logs/financial_news.log | jq '.'

# Filter by level
jq 'select(.level == "ERROR")' logs/financial_news.log
```

### 2. Database Monitoring

```bash
# Check Qdrant collections
curl http://localhost:6333/collections

# Get collection info
curl http://localhost:6333/collections/financial_rss_news

# Search directly in Qdrant
curl -X POST http://localhost:6333/collections/financial_rss_news/points/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "limit": 5}'
```

### 3. Performance Testing

```bash
# Run search benchmarks
python scripts/benchmark_search.py --queries 100 --concurrent 10

# Test ingestion speed
python scripts/benchmark_ingestion.py --articles 1000
```

## Common Development Tasks

### Adding a New News Provider

1. Create provider class in `financial_news_api_feed/providers/`:
```python
from financial_news_api_feed.base_provider import BaseProvider

class MyProvider(BaseProvider):
    def fetch_news(self, tickers=None, days_back=7):
        # Implement fetching logic
        return articles
```

2. Register in `provider_registry.py`:
```python
PROVIDERS = {
    "yahoo": YahooFinanceProvider,
    "my_provider": MyProvider,
}
```

### Modifying Search Behavior

1. Update `shared/search.py` for search logic
2. Adjust `mcp_server/server.py` for tool behavior
3. Add tests in `tests/test_search.py`

### Adding New MCP Tools

1. Implement tool function in `mcp_server/server.py`
2. Add to `list_tools()` decorator
3. Handle in `call_tool()` method
4. Update documentation

## Environment Variables Reference

### Core Configuration
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR
- `LOG_FORMAT`: json or text
- `PYTHONPATH`: Should include project root

### Qdrant Settings
- `QDRANT_HOST`: Database host (default: localhost)
- `QDRANT_PORT`: Database port (default: 6333)
- `QDRANT_API_KEY`: Authentication key (optional)
- `QDRANT_USE_HYBRID`: Enable hybrid search (default: true)
- `QDRANT_HYBRID_ALPHA`: Weight for hybrid search (0-1, default: 0.5)

### Redis Settings
- `REDIS_HOST`: Cache host (default: localhost)
- `REDIS_PORT`: Cache port (default: 6379)
- `REDIS_PASSWORD`: Authentication password (optional)
- `REDIS_DB`: Database number (default: 0)

### Embedding Settings
- `EMBEDDING_MODEL`: Model name (default: sentence-transformers/all-mpnet-base-v2)
- `EMBEDDING_DEVICE`: cpu or cuda (default: cpu)
- `CHUNK_SIZE`: Text chunk size (default: 512)
- `CHUNK_OVERLAP`: Chunk overlap (default: 50)

### Search Settings
- `SEARCH_LIMIT_DEFAULT`: Default result count (default: 10)
- `SEARCH_DEDUPLICATE`: Remove duplicates (default: true)
- `SEARCH_SIMILARITY_THRESHOLD`: Minimum similarity (default: 0.85)

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure Python path is set
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   
   # Install in development mode
   pip install -e .
   ```

2. **Qdrant Connection Failed**
   ```bash
   # Check if Qdrant is running
   curl http://localhost:6333/health
   
   # Restart Docker services
   docker-compose restart
   ```

3. **Redis Connection Failed**
   ```bash
   # Check Redis
   redis-cli ping
   
   # Disable Redis if not needed
   export VOCABULARY_STORE_BACKEND=json
   ```

4. **Embedding Model Download Issues**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/torch/sentence_transformers/
   
   # Use different model
   export EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

5. **MCP Server Not Responding**
   ```bash
   # Check server logs
   python -m mcp_server.server --log-level DEBUG
   
   # Test with simple message
   echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize"}' | python -m mcp_server.server
   ```

### Performance Issues

1. **Slow Search**
   - Reduce `CHUNK_SIZE` for smaller embeddings
   - Enable `QDRANT_SEARCH_EXACT` for precise search
   - Increase `QDRANT_SEARCH_HNSW_EF` for better recall

2. **Memory Issues**
   - Use smaller embedding model
   - Reduce `PIPELINE_BATCH_SIZE`
   - Enable streaming for large datasets

3. **High CPU Usage**
   - Set `EMBEDDING_DEVICE=cuda` if GPU available
   - Use multithreading for ingestion
   - Cache embeddings in Redis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for all public functions
- Keep functions focused and small
- Add meaningful error messages

### Testing

- Write unit tests for new features
- Test edge cases and error conditions
- Mock external dependencies
- Achieve >80% code coverage

## Deployment Preparation

Before deploying to production:

1. Update dependencies:
   ```bash
   pip-compile requirements.in
   pip-compile requirements-dev.in
   ```

2. Build Docker image:
   ```bash
   docker build -t financial-news-rag .
   ```

3. Test production configuration:
   ```bash
   docker run -p 8000:8000 financial-news-rag
   ```

4. Update environment variables for production
5. Set up monitoring and logging
6. Configure backup and recovery

## Additional Resources

- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
