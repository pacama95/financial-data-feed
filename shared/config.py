"""Centralized configuration from environment variables with defaults."""

from __future__ import annotations

import os
from typing import Optional


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_str(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value else None


class Config:
    """RAG system configuration from environment variables."""

    # Embedding
    EMBEDDING_MODEL: str = _env_str("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    EMBEDDING_DEVICE: str = _env_str("EMBEDDING_DEVICE", "cpu")

    # Chunking
    CHUNK_SIZE: int = _env_int("CHUNK_SIZE", 512)
    CHUNK_OVERLAP: int = _env_int("CHUNK_OVERLAP", 50)

    # Qdrant connection
    QDRANT_HOST: str = _env_str("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = _env_int("QDRANT_PORT", 6333)
    QDRANT_API_KEY: Optional[str] = _env_optional_str("QDRANT_API_KEY")

    # Qdrant search params
    QDRANT_DISTANCE_METRIC: str = _env_str("QDRANT_DISTANCE_METRIC", "cosine")
    QDRANT_SEARCH_HNSW_EF: int = _env_int("QDRANT_SEARCH_HNSW_EF", 128)
    QDRANT_SEARCH_EXACT: bool = _env_bool("QDRANT_SEARCH_EXACT", False)
    QDRANT_RESCORING_ENABLED: bool = _env_bool("QDRANT_RESCORING_ENABLED", True)
    QDRANT_RESCORING_OVERSAMPLING: float = _env_float("QDRANT_RESCORING_OVERSAMPLING", 2.0)

    # Hybrid search
    QDRANT_USE_HYBRID: bool = _env_bool("QDRANT_USE_HYBRID", True)
    QDRANT_HYBRID_ALPHA: float = _env_float("QDRANT_HYBRID_ALPHA", 0.5)

    # Search defaults
    SEARCH_LIMIT_DEFAULT: int = _env_int("SEARCH_LIMIT_DEFAULT", 10)
    SEARCH_DEDUPLICATE: bool = _env_bool("SEARCH_DEDUPLICATE", True)
    SEARCH_SIMILARITY_THRESHOLD: float = _env_float("SEARCH_SIMILARITY_THRESHOLD", 0.85)

    # Lexical / BM25
    LEXICAL_K1: float = _env_float("LEXICAL_K1", 1.5)
    LEXICAL_B: float = _env_float("LEXICAL_B", 0.75)

    # Vocabulary storage
    VOCABULARY_STORE_PATH: str = _env_str("VOCABULARY_STORE_PATH", "artifacts/vocabularies")
    VOCABULARY_STORE_BACKEND: str = _env_str("VOCABULARY_STORE_BACKEND", "redis")  # "json" or "redis"

    # Redis configuration
    REDIS_HOST: str = _env_str("REDIS_HOST", "localhost")
    REDIS_PORT: int = _env_int("REDIS_PORT", 6379)
    REDIS_DB: int = _env_int("REDIS_DB", 0)
    REDIS_PASSWORD: Optional[str] = _env_optional_str("REDIS_PASSWORD")
    REDIS_VOCAB_KEY_PREFIX: str = _env_str("REDIS_VOCAB_KEY_PREFIX", "vocab:")

    # Pipeline
    PIPELINE_BATCH_SIZE: int = _env_int("PIPELINE_BATCH_SIZE", 100)
    PIPELINE_CLEANUP_DAYS: int = _env_int("PIPELINE_CLEANUP_DAYS", 30)

    # Logging
    LOG_LEVEL: str = _env_str("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = _env_str("LOG_FORMAT", "json")

    # Collection names
    RSS_COLLECTION: str = _env_str("RSS_COLLECTION", "financial_rss_news")
    API_COLLECTION: str = _env_str("API_COLLECTION", "financial_api_news")

    @classmethod
    def reload(cls) -> None:
        """Reload configuration from environment variables."""
        cls.EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        cls.EMBEDDING_DEVICE = _env_str("EMBEDDING_DEVICE", "cpu")
        cls.CHUNK_SIZE = _env_int("CHUNK_SIZE", 512)
        cls.CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 50)
        cls.QDRANT_HOST = _env_str("QDRANT_HOST", "localhost")
        cls.QDRANT_PORT = _env_int("QDRANT_PORT", 6333)
        cls.QDRANT_API_KEY = _env_optional_str("QDRANT_API_KEY")
        cls.QDRANT_DISTANCE_METRIC = _env_str("QDRANT_DISTANCE_METRIC", "cosine")
        cls.QDRANT_SEARCH_HNSW_EF = _env_int("QDRANT_SEARCH_HNSW_EF", 128)
        cls.QDRANT_SEARCH_EXACT = _env_bool("QDRANT_SEARCH_EXACT", False)
        cls.QDRANT_RESCORING_ENABLED = _env_bool("QDRANT_RESCORING_ENABLED", True)
        cls.QDRANT_RESCORING_OVERSAMPLING = _env_float("QDRANT_RESCORING_OVERSAMPLING", 2.0)
        cls.QDRANT_USE_HYBRID = _env_bool("QDRANT_USE_HYBRID", True)
        cls.QDRANT_HYBRID_ALPHA = _env_float("QDRANT_HYBRID_ALPHA", 0.5)
        cls.SEARCH_LIMIT_DEFAULT = _env_int("SEARCH_LIMIT_DEFAULT", 10)
        cls.SEARCH_DEDUPLICATE = _env_bool("SEARCH_DEDUPLICATE", True)
        cls.SEARCH_SIMILARITY_THRESHOLD = _env_float("SEARCH_SIMILARITY_THRESHOLD", 0.85)
        cls.LEXICAL_K1 = _env_float("LEXICAL_K1", 1.5)
        cls.LEXICAL_B = _env_float("LEXICAL_B", 0.75)
        cls.VOCABULARY_STORE_PATH = _env_str("VOCABULARY_STORE_PATH", "artifacts/vocabularies")
        cls.VOCABULARY_STORE_BACKEND = _env_str("VOCABULARY_STORE_BACKEND", "redis")
        cls.REDIS_HOST = _env_str("REDIS_HOST", "localhost")
        cls.REDIS_PORT = _env_int("REDIS_PORT", 6379)
        cls.REDIS_DB = _env_int("REDIS_DB", 0)
        cls.REDIS_PASSWORD = _env_optional_str("REDIS_PASSWORD")
        cls.REDIS_VOCAB_KEY_PREFIX = _env_str("REDIS_VOCAB_KEY_PREFIX", "vocab:")
        cls.PIPELINE_BATCH_SIZE = _env_int("PIPELINE_BATCH_SIZE", 100)
        cls.PIPELINE_CLEANUP_DAYS = _env_int("PIPELINE_CLEANUP_DAYS", 90)
        cls.LOG_LEVEL = _env_str("LOG_LEVEL", "INFO")
        cls.LOG_FORMAT = _env_str("LOG_FORMAT", "json")
        cls.RSS_COLLECTION = _env_str("RSS_COLLECTION", "financial_rss_news")
        cls.API_COLLECTION = _env_str("API_COLLECTION", "financial_api_news")

    @classmethod
    def as_dict(cls) -> dict:
        """Return all config values as a dictionary."""
        return {
            "EMBEDDING_MODEL": cls.EMBEDDING_MODEL,
            "EMBEDDING_DEVICE": cls.EMBEDDING_DEVICE,
            "CHUNK_SIZE": cls.CHUNK_SIZE,
            "CHUNK_OVERLAP": cls.CHUNK_OVERLAP,
            "QDRANT_HOST": cls.QDRANT_HOST,
            "QDRANT_PORT": cls.QDRANT_PORT,
            "QDRANT_API_KEY": "***" if cls.QDRANT_API_KEY else None,
            "QDRANT_DISTANCE_METRIC": cls.QDRANT_DISTANCE_METRIC,
            "QDRANT_SEARCH_HNSW_EF": cls.QDRANT_SEARCH_HNSW_EF,
            "QDRANT_SEARCH_EXACT": cls.QDRANT_SEARCH_EXACT,
            "QDRANT_RESCORING_ENABLED": cls.QDRANT_RESCORING_ENABLED,
            "QDRANT_RESCORING_OVERSAMPLING": cls.QDRANT_RESCORING_OVERSAMPLING,
            "QDRANT_USE_HYBRID": cls.QDRANT_USE_HYBRID,
            "QDRANT_HYBRID_ALPHA": cls.QDRANT_HYBRID_ALPHA,
            "SEARCH_LIMIT_DEFAULT": cls.SEARCH_LIMIT_DEFAULT,
            "SEARCH_DEDUPLICATE": cls.SEARCH_DEDUPLICATE,
            "SEARCH_SIMILARITY_THRESHOLD": cls.SEARCH_SIMILARITY_THRESHOLD,
            "LEXICAL_K1": cls.LEXICAL_K1,
            "LEXICAL_B": cls.LEXICAL_B,
            "VOCABULARY_STORE_PATH": cls.VOCABULARY_STORE_PATH,
            "VOCABULARY_STORE_BACKEND": cls.VOCABULARY_STORE_BACKEND,
            "REDIS_HOST": cls.REDIS_HOST,
            "REDIS_PORT": cls.REDIS_PORT,
            "REDIS_DB": cls.REDIS_DB,
            "REDIS_PASSWORD": "***" if cls.REDIS_PASSWORD else None,
            "REDIS_VOCAB_KEY_PREFIX": cls.REDIS_VOCAB_KEY_PREFIX,
            "PIPELINE_BATCH_SIZE": cls.PIPELINE_BATCH_SIZE,
            "PIPELINE_CLEANUP_DAYS": cls.PIPELINE_CLEANUP_DAYS,
            "LOG_LEVEL": cls.LOG_LEVEL,
            "LOG_FORMAT": cls.LOG_FORMAT,
            "RSS_COLLECTION": cls.RSS_COLLECTION,
            "API_COLLECTION": cls.API_COLLECTION,
        }
