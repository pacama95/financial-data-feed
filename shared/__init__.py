"""Shared utilities for financial news RAG system.

Usage:
    from shared import RSSPipeline, APIPipeline, SearchClient, Config

    # Run pipelines - vocabularies are automatically saved to the store
    RSSPipeline().run(regions=["usa"])
    APIPipeline().run(tickers=["AAPL", "NVDA"])

    # Search - vocabularies are automatically loaded from the store
    client = SearchClient()
    results = client.search("NVIDIA earnings", mode="hybrid")

    # Hybrid search with explicit keywords
    results = client.search(
        query="NVIDIA earnings report",
        mode="hybrid",
        keywords=["NVDA", "earnings"],
        keyword_boosts={"NVDA": 2.0},
    )
"""

from shared.config import Config
from shared.logging import (
    get_logger,
    setup_logging,
    PerformanceTracker,
    log_operation_start,
    log_operation_end,
)
from shared.chunking_embedding import (
    Chunk,
    TextChunker,
    EmbeddingGenerator,
    chunk_and_embed_articles,
)
from shared.qdrant_store import QdrantStore, create_qdrant_store, reset_qdrant_store
from shared.lexical import LexicalVocabulary, build_sparse_vector
from shared.vocabulary_store import (
    VocabularyStore,
    JSONVocabularyStore,
    RedisVocabularyStore,
    get_vocabulary_store,
    set_vocabulary_store,
)
from shared.search import SearchClient, deduplicate_results
from shared.ticker_extractor import TickerExtractor
from shared.pipelines import BasePipeline, RSSPipeline, APIPipeline

__all__ = [
    # Config
    "Config",
    # Logging
    "get_logger",
    "setup_logging",
    "PerformanceTracker",
    "log_operation_start",
    "log_operation_end",
    # Chunking & Embedding
    "Chunk",
    "TextChunker",
    "EmbeddingGenerator",
    "chunk_and_embed_articles",
    # Storage
    "QdrantStore",
    "create_qdrant_store",
    "reset_qdrant_store",
    # Lexical
    "LexicalVocabulary",
    "build_sparse_vector",
    # Vocabulary Store
    "VocabularyStore",
    "JSONVocabularyStore",
    "RedisVocabularyStore",
    "get_vocabulary_store",
    "set_vocabulary_store",
    # Search
    "SearchClient",
    "deduplicate_results",
    # Utilities
    "TickerExtractor",
    # Pipelines
    "BasePipeline",
    "RSSPipeline",
    "APIPipeline",
]
