"""Base pipeline class for financial news ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from shared.config import Config
from shared.logging import get_logger, PerformanceTracker
from shared.chunking_embedding import (
    Chunk,
    EmbeddingGenerator,
    TextChunker,
    chunk_and_embed_articles,
)
from shared.lexical import LexicalVocabulary
from shared.qdrant_store import QdrantStore, create_qdrant_store
from shared.vocabulary_store import VocabularyStore, get_vocabulary_store

if TYPE_CHECKING:
    pass

logger = get_logger("shared.pipelines.base")


class BasePipeline(ABC):
    """
    Abstract base class for financial news pipelines.
    
    Subclasses must implement:
        - fetch(): Retrieve articles from source
        - _build_metadata(article): Build chunk metadata from article
        - _build_summary(article): Build summary text from article
    """

    def __init__(
        self,
        collection: Optional[str] = None,
        store: Optional[QdrantStore] = None,
        embedder: Optional[EmbeddingGenerator] = None,
        vocabulary_store: Optional[VocabularyStore] = None,
        batch_size: Optional[int] = None,
        cleanup_days: Optional[int] = None,
        recreate_collection: bool = False,
    ):
        self.collection = collection or self._default_collection()
        self._store = store
        self._embedder = embedder
        self._vocabulary_store = vocabulary_store
        self.batch_size = batch_size or Config.PIPELINE_BATCH_SIZE
        self.cleanup_days = cleanup_days or Config.PIPELINE_CLEANUP_DAYS
        self.recreate_collection = recreate_collection
        self._lexical_vocab: Optional[LexicalVocabulary] = None

    @property
    def store(self) -> QdrantStore:
        if self._store is None:
            self._store = create_qdrant_store()
        return self._store

    @property
    def embedder(self) -> EmbeddingGenerator:
        if self._embedder is None:
            self._embedder = EmbeddingGenerator()
        return self._embedder

    @property
    def vocabulary_store(self) -> VocabularyStore:
        if self._vocabulary_store is None:
            self._vocabulary_store = get_vocabulary_store()
        return self._vocabulary_store

    @abstractmethod
    def _default_collection(self) -> str:
        """Return default collection name for this pipeline."""
        pass

    @abstractmethod
    def fetch(self) -> List[Any]:
        """Fetch articles from source. Returns list of article objects."""
        pass

    @abstractmethod
    def _build_metadata(self, article: Any) -> Dict[str, Any]:
        """Build chunk metadata from article."""
        pass

    @abstractmethod
    def _build_summary(self, article: Any) -> str:
        """Build summary text from article."""
        pass

    def _get_chunker(self) -> TextChunker:
        """Create chunker with pipeline-specific metadata/summary builders."""
        return TextChunker(
            metadata_builder=self._build_metadata,
            summary_builder=self._build_summary,
        )

    def process(self, articles: List[Any]) -> List[Tuple[Chunk, List[float]]]:
        """Chunk and embed articles."""
        if not articles:
            logger.warning("No articles to process")
            return []

        chunker = self._get_chunker()
        return chunk_and_embed_articles(
            articles=articles,
            chunker=chunker,
            embedder=self.embedder,
        )

    def build_lexical_vocab(self, articles: List[Any]) -> LexicalVocabulary:
        """Build lexical vocabulary from articles for hybrid search.
        
        Includes article text plus relevant metadata (tickers, source names)
        to enable keyword search on these fields.
        """
        documents = []
        for article in articles:
            text_parts = []
            
            # Core content
            title = getattr(article, "title", "")
            content = getattr(article, "content", "")
            if title:
                text_parts.append(title)
            if content:
                text_parts.append(content)
            
            # Tickers - include all ticker-related fields
            for ticker_field in ["tickers", "related_tickers", "mentioned_tickers"]:
                tickers = getattr(article, ticker_field, None) or []
                if tickers:
                    text_parts.extend(tickers)
            
            # Source name for source-based searches
            source_name = getattr(article, "source_name", "")
            if source_name:
                text_parts.append(source_name)
            
            if text_parts:
                documents.append(" ".join(text_parts))

        self._lexical_vocab = LexicalVocabulary.build(documents)
        logger.info("Built lexical vocabulary with %d tokens", len(self._lexical_vocab))
        return self._lexical_vocab

    def save_lexical_vocab(self) -> None:
        """Save lexical vocabulary to the vocabulary store."""
        if self._lexical_vocab is None:
            logger.warning("No lexical vocabulary to save")
            return
        self.vocabulary_store.save_vocabulary(self.collection, self._lexical_vocab)
        logger.info("Saved vocabulary for collection '%s' to store", self.collection)

    def store_chunks(self, chunks_with_embeddings: List[Tuple[Chunk, List[float]]]) -> None:
        """Store chunks in vector database with sparse vectors for hybrid search."""
        if not chunks_with_embeddings:
            logger.warning("No chunks to store")
            return

        # Ensure collection exists
        vector_size = len(chunks_with_embeddings[0][1])
        self.store.create_collection(
            collection_name=self.collection,
            vector_size=vector_size,
            recreate_if_exists=self.recreate_collection,
        )

        # Upsert chunks with sparse vectors if lexical vocab is available
        self.store.upsert(
            collection_name=self.collection,
            chunks_with_embeddings=chunks_with_embeddings,
            batch_size=self.batch_size,
            lexical_vocab=self._lexical_vocab,
        )

    def cleanup_old(self, days: Optional[int] = None) -> None:
        """Delete old documents from collection."""
        days = days or self.cleanup_days
        self.store.delete_old_documents(
            collection_name=self.collection,
            days=days,
        )

    def run(self, cleanup: bool = True) -> Dict[str, Any]:
        """
        Run the full pipeline: fetch -> process -> build lexical -> store -> cleanup.
        
        All fetch parameters should be configured in the constructor.
        
        Args:
            cleanup: Whether to cleanup old documents after storing new ones
            
        Returns:
            Pipeline results summary
        """
        with PerformanceTracker(f"pipeline.{self.collection}"):
            logger.info("Starting pipeline for collection: %s", self.collection)

            # Fetch
            logger.info("Fetching articles...")
            articles = self.fetch()
            logger.info("Fetched %d articles", len(articles))

            if not articles:
                return {
                    "collection": self.collection,
                    "articles_fetched": 0,
                    "chunks_stored": 0,
                    "status": "no_articles",
                }

            # Process (chunk + embed)
            logger.info("Processing articles...")
            chunks_with_embeddings = self.process(articles)
            logger.info("Generated %d chunks", len(chunks_with_embeddings))

            # Build lexical vocabulary
            logger.info("Building lexical vocabulary...")
            self.build_lexical_vocab(articles)
            self.save_lexical_vocab()

            # Store
            logger.info("Storing chunks...")
            self.store_chunks(chunks_with_embeddings)

            # Cleanup
            if cleanup:
                logger.info("Cleaning up old documents...")
                self.cleanup_old()

            logger.info("Pipeline completed for collection: %s", self.collection)

            return {
                "collection": self.collection,
                "articles_fetched": len(articles),
                "chunks_stored": len(chunks_with_embeddings),
                "lexical_vocab_size": len(self._lexical_vocab) if self._lexical_vocab else 0,
                "status": "success",
            }
