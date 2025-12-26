"""Vocabulary storage abstraction for lexical search.

Provides an abstract interface for storing and retrieving lexical vocabularies,
with JSON file and Redis implementations.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from shared.config import Config
from shared.logging import get_logger

if TYPE_CHECKING:
    from shared.lexical import LexicalVocabulary

logger = get_logger("shared.vocabulary_store")


class VocabularyStore(ABC):
    """Abstract base class for vocabulary storage.
    
    Implementations can store vocabularies in files, databases, or other backends.
    """

    @abstractmethod
    def save(self, collection: str, vocab_data: Dict[str, Any]) -> None:
        """Save vocabulary data for a collection.
        
        Args:
            collection: Collection name (e.g., 'financial_api_news')
            vocab_data: Vocabulary data dictionary containing:
                - token_to_id: Dict[str, int]
                - idf_scores: Dict[str, float]
                - doc_count: int
                - avg_doc_len: float
                - lowercase: bool
                - k1: float
                - b: float
        """
        pass

    @abstractmethod
    def load(self, collection: str) -> Optional[Dict[str, Any]]:
        """Load vocabulary data for a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            Vocabulary data dictionary, or None if not found
        """
        pass

    @abstractmethod
    def exists(self, collection: str) -> bool:
        """Check if vocabulary exists for a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            True if vocabulary exists
        """
        pass

    @abstractmethod
    def delete(self, collection: str) -> bool:
        """Delete vocabulary for a collection.
        
        Args:
            collection: Collection name
            
        Returns:
            True if deleted, False if not found
        """
        pass

    def get_vocabulary(self, collection: str) -> Optional["LexicalVocabulary"]:
        """Load and return a LexicalVocabulary instance.
        
        Args:
            collection: Collection name
            
        Returns:
            LexicalVocabulary instance, or None if not found
        """
        from shared.lexical import LexicalVocabulary
        
        data = self.load(collection)
        if data is None:
            return None
        
        return LexicalVocabulary(
            token_to_id=data.get("token_to_id", {}),
            idf_scores=data.get("idf_scores", data.get("doc_freq", {})),  # Backward compat
            doc_count=data.get("doc_count", 0),
            avg_doc_len=data.get("avg_doc_len", 0.0),
            lowercase=data.get("lowercase", True),
            k1=data.get("k1"),
            b=data.get("b"),
        )

    def save_vocabulary(self, collection: str, vocab: "LexicalVocabulary") -> None:
        """Save a LexicalVocabulary instance.
        
        Args:
            collection: Collection name
            vocab: LexicalVocabulary instance to save
        """
        data = {
            "token_to_id": vocab.token_to_id,
            "idf_scores": vocab.idf_scores,
            "doc_count": vocab.doc_count,
            "avg_doc_len": vocab.avg_doc_len,
            "lowercase": vocab.lowercase,
            "k1": vocab.k1,
            "b": vocab.b,
        }
        self.save(collection, data)


class JSONVocabularyStore(VocabularyStore):
    """JSON file-based vocabulary storage.
    
    Stores each collection's vocabulary as a separate JSON file in a directory.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize JSON vocabulary store.
        
        Args:
            base_dir: Base directory for vocabulary files.
                Defaults to Config.VOCABULARY_STORE_PATH or 'artifacts/vocabularies'
        """
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Use absolute path relative to this file's location (shared module)
            # This ensures it works regardless of working directory
            default_path = getattr(Config, 'VOCABULARY_STORE_PATH', 'artifacts/vocabularies')
            if os.path.isabs(default_path):
                self.base_dir = Path(default_path)
            else:
                # Make relative path absolute from the project root (parent of shared/)
                project_root = Path(__file__).parent.parent
                self.base_dir = project_root / default_path
        
        # Don't create directory on init - only when saving (read-only environments)

    def _get_path(self, collection: str) -> Path:
        """Get file path for a collection's vocabulary."""
        # Sanitize collection name for filesystem
        safe_name = collection.replace("/", "_").replace("\\", "_")
        return self.base_dir / f"{safe_name}_vocab.json"

    def save(self, collection: str, vocab_data: Dict[str, Any]) -> None:
        """Save vocabulary data to JSON file."""
        # Create directory only when saving (not on init for read-only environments)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self._get_path(collection)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f)
        logger.info("Saved vocabulary for collection '%s' to %s", collection, path)

    def load(self, collection: str) -> Optional[Dict[str, Any]]:
        """Load vocabulary data from JSON file."""
        path = self._get_path(collection)
        if not path.exists():
            logger.debug("No vocabulary found for collection '%s' at %s", collection, path)
            return None
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug("Loaded vocabulary for collection '%s' from %s", collection, path)
        return data

    def exists(self, collection: str) -> bool:
        """Check if vocabulary file exists."""
        return self._get_path(collection).exists()

    def delete(self, collection: str) -> bool:
        """Delete vocabulary file."""
        path = self._get_path(collection)
        if path.exists():
            path.unlink()
            logger.info("Deleted vocabulary for collection '%s'", collection)
            return True
        return False


class RedisVocabularyStore(VocabularyStore):
    """Redis-based vocabulary storage using Hash structures.
    
    Uses multiple Redis keys per collection for scalability:
    - vocab:{collection}:meta - Hash with metadata (doc_count, avg_doc_len, lowercase, k1, b)
    - vocab:{collection}:tokens - Hash mapping token -> id
    - vocab:{collection}:idf - Hash mapping token -> idf_score
    
    This structure allows:
    - Partial reads (lookup single token without loading entire vocab)
    - Incremental updates (add tokens without rewriting everything)
    - Better memory efficiency for large vocabularies
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        key_prefix: Optional[str] = None,
    ):
        """Initialize Redis vocabulary store.
        
        Args:
            host: Redis host. Defaults to Config.REDIS_HOST
            port: Redis port. Defaults to Config.REDIS_PORT
            db: Redis database number. Defaults to Config.REDIS_DB
            password: Redis password. Defaults to Config.REDIS_PASSWORD
            key_prefix: Prefix for vocabulary keys. Defaults to Config.REDIS_VOCAB_KEY_PREFIX
        """
        self.host = host or Config.REDIS_HOST
        self.port = port or Config.REDIS_PORT
        self.db = db if db is not None else Config.REDIS_DB
        self.password = password or Config.REDIS_PASSWORD
        self.key_prefix = key_prefix or Config.REDIS_VOCAB_KEY_PREFIX
        self._client: Optional["redis.Redis"] = None

    def _get_client(self) -> "redis.Redis":
        """Get or create Redis client (lazy initialization)."""
        if self._client is None:
            try:
                import redis
            except ImportError as e:
                raise ImportError(
                    "redis package not installed. Run: pip install redis"
                ) from e
            
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
            )
            logger.debug(
                "Connected to Redis at %s:%d (db=%d)",
                self.host, self.port, self.db
            )
        return self._client

    def _get_base_key(self, collection: str) -> str:
        """Get base Redis key for a collection."""
        safe_name = collection.replace("/", "_").replace("\\", "_")
        return f"{self.key_prefix}{safe_name}"

    def _get_keys(self, collection: str) -> tuple:
        """Get all Redis keys for a collection's vocabulary.
        
        Returns:
            Tuple of (meta_key, tokens_key, idf_key)
        """
        base = self._get_base_key(collection)
        return (f"{base}:meta", f"{base}:tokens", f"{base}:idf")

    def save(self, collection: str, vocab_data: Dict[str, Any]) -> None:
        """Save vocabulary data to Redis using Hash structures."""
        client = self._get_client()
        meta_key, tokens_key, idf_key = self._get_keys(collection)
        
        # Use pipeline for atomic batch operation
        pipe = client.pipeline()
        
        # Delete existing keys first (atomic replacement)
        pipe.delete(meta_key, tokens_key, idf_key)
        
        # Store metadata
        meta = {
            "doc_count": str(vocab_data.get("doc_count", 0)),
            "avg_doc_len": str(vocab_data.get("avg_doc_len", 0.0)),
            "lowercase": str(vocab_data.get("lowercase", True)),
            "k1": str(vocab_data.get("k1", 1.5)),
            "b": str(vocab_data.get("b", 0.75)),
        }
        pipe.hset(meta_key, mapping=meta)
        
        # Store token_to_id mapping
        token_to_id = vocab_data.get("token_to_id", {})
        if token_to_id:
            # Convert int values to strings for Redis
            token_mapping = {token: str(idx) for token, idx in token_to_id.items()}
            pipe.hset(tokens_key, mapping=token_mapping)
        
        # Store idf_scores mapping
        idf_scores = vocab_data.get("idf_scores", {})
        if idf_scores:
            # Convert float values to strings for Redis
            idf_mapping = {token: str(score) for token, score in idf_scores.items()}
            pipe.hset(idf_key, mapping=idf_mapping)
        
        pipe.execute()
        
        logger.info(
            "Saved vocabulary for collection '%s' to Redis (%d tokens)",
            collection, len(token_to_id)
        )

    def load(self, collection: str) -> Optional[Dict[str, Any]]:
        """Load vocabulary data from Redis Hash structures."""
        client = self._get_client()
        meta_key, tokens_key, idf_key = self._get_keys(collection)
        
        # Check if vocabulary exists
        if not client.exists(meta_key):
            logger.debug("No vocabulary found for collection '%s'", collection)
            return None
        
        # Use pipeline to fetch all data in one round-trip
        pipe = client.pipeline()
        pipe.hgetall(meta_key)
        pipe.hgetall(tokens_key)
        pipe.hgetall(idf_key)
        meta_raw, tokens_raw, idf_raw = pipe.execute()
        
        # Parse metadata
        data = {
            "doc_count": int(meta_raw.get("doc_count", 0)),
            "avg_doc_len": float(meta_raw.get("avg_doc_len", 0.0)),
            "lowercase": meta_raw.get("lowercase", "True").lower() == "true",
            "k1": float(meta_raw.get("k1", 1.5)),
            "b": float(meta_raw.get("b", 0.75)),
        }
        
        # Parse token_to_id (convert string values back to int)
        data["token_to_id"] = {token: int(idx) for token, idx in tokens_raw.items()}
        
        # Parse idf_scores (convert string values back to float)
        data["idf_scores"] = {token: float(score) for token, score in idf_raw.items()}
        
        logger.debug(
            "Loaded vocabulary for collection '%s' from Redis (%d tokens)",
            collection, len(data["token_to_id"])
        )
        return data

    def exists(self, collection: str) -> bool:
        """Check if vocabulary exists in Redis."""
        client = self._get_client()
        meta_key, _, _ = self._get_keys(collection)
        return bool(client.exists(meta_key))

    def delete(self, collection: str) -> bool:
        """Delete vocabulary from Redis."""
        client = self._get_client()
        meta_key, tokens_key, idf_key = self._get_keys(collection)
        deleted = client.delete(meta_key, tokens_key, idf_key)
        if deleted:
            logger.info("Deleted vocabulary for collection '%s' from Redis", collection)
            return True
        return False

    def list_collections(self) -> list:
        """List all collections with stored vocabularies.
        
        Returns:
            List of collection names
        """
        client = self._get_client()
        # Look for meta keys to identify collections
        pattern = f"{self.key_prefix}*:meta"
        keys = client.keys(pattern)
        # Extract collection name from key pattern: vocab:{collection}:meta
        prefix_len = len(self.key_prefix)
        suffix_len = len(":meta")
        return [key[prefix_len:-suffix_len] for key in keys]

    def get_token_id(self, collection: str, token: str) -> Optional[int]:
        """Get the ID for a single token (efficient partial lookup).
        
        Args:
            collection: Collection name
            token: Token to look up
            
        Returns:
            Token ID, or None if not found
        """
        client = self._get_client()
        _, tokens_key, _ = self._get_keys(collection)
        value = client.hget(tokens_key, token)
        return int(value) if value is not None else None

    def get_token_idf(self, collection: str, token: str) -> Optional[float]:
        """Get the IDF score for a single token (efficient partial lookup).
        
        Args:
            collection: Collection name
            token: Token to look up
            
        Returns:
            IDF score, or None if not found
        """
        client = self._get_client()
        _, _, idf_key = self._get_keys(collection)
        value = client.hget(idf_key, token)
        return float(value) if value is not None else None

    def get_tokens_batch(
        self, collection: str, tokens: list
    ) -> Dict[str, tuple]:
        """Get IDs and IDF scores for multiple tokens in one call.
        
        Args:
            collection: Collection name
            tokens: List of tokens to look up
            
        Returns:
            Dict mapping token -> (id, idf_score) for found tokens
        """
        if not tokens:
            return {}
        
        client = self._get_client()
        _, tokens_key, idf_key = self._get_keys(collection)
        
        pipe = client.pipeline()
        for token in tokens:
            pipe.hget(tokens_key, token)
            pipe.hget(idf_key, token)
        results = pipe.execute()
        
        # Parse results (pairs of id, idf for each token)
        output = {}
        for i, token in enumerate(tokens):
            id_val = results[i * 2]
            idf_val = results[i * 2 + 1]
            if id_val is not None:
                output[token] = (int(id_val), float(idf_val) if idf_val else 0.0)
        
        return output


# Singleton instance for global access
_default_store: Optional[VocabularyStore] = None


def get_vocabulary_store(backend: Optional[str] = None) -> VocabularyStore:
    """Get the default vocabulary store instance.
    
    Args:
        backend: Storage backend to use ("json" or "redis").
            Defaults to Config.VOCABULARY_STORE_BACKEND.
    
    Returns:
        VocabularyStore instance (singleton per backend type).
    """
    global _default_store
    
    backend = backend or Config.VOCABULARY_STORE_BACKEND
    
    # If store exists and matches requested backend, return it
    if _default_store is not None:
        if backend == "redis" and isinstance(_default_store, RedisVocabularyStore):
            return _default_store
        if backend == "json" and isinstance(_default_store, JSONVocabularyStore):
            return _default_store
    
    # Create new store based on backend
    if backend == "redis":
        _default_store = RedisVocabularyStore()
        logger.info("Using Redis vocabulary store")
    else:
        _default_store = JSONVocabularyStore()
        logger.info("Using JSON vocabulary store")
    
    return _default_store


def set_vocabulary_store(store: VocabularyStore) -> None:
    """Set the default vocabulary store instance.
    
    Use this to switch to a database-backed store:
    
        from shared.vocabulary_store import set_vocabulary_store, DatabaseVocabularyStore
        set_vocabulary_store(DatabaseVocabularyStore(connection_string))
    """
    global _default_store
    _default_store = store
    logger.info("Set vocabulary store to %s", type(store).__name__)


__all__ = [
    "VocabularyStore",
    "JSONVocabularyStore",
    "RedisVocabularyStore",
    "get_vocabulary_store",
    "set_vocabulary_store",
]
