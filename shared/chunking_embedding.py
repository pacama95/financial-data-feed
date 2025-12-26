"""Shared chunking and embedding utilities for financial news content."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from shared.config import Config
from shared.logging import get_logger

logger = get_logger("shared.chunking_embedding")

MetadataBuilder = Callable[[Any], Dict[str, Any]]
SummaryBuilder = Callable[[Any], str]


@dataclass
class Chunk:
    """Unified chunk representation with flexible metadata."""

    text: str
    chunk_index: int
    total_chunks: int

    article_id: str
    article_url: str
    article_title: str
    published_date: str
    source_name: str

    provider: str = ""
    source_tier: str = ""
    region: str = ""
    feed_name: str = ""
    feed_type: str = ""

    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    tickers: List[str] = field(default_factory=list)
    related_tickers: List[str] = field(default_factory=list)
    mentioned_tickers: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None

    chunk_type: str = "body"
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to a payload dictionary suitable for vector stores."""
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "article_id": self.article_id,
            "article_url": self.article_url,
            "article_title": self.article_title,
            "published_date": self.published_date,
            "source_name": self.source_name,
            "provider": self.provider,
            "source_tier": self.source_tier,
            "region": self.region,
            "feed_name": self.feed_name,
            "feed_type": self.feed_type,
            "author": self.author or "",
            "tags": self.tags,
            "tickers": self.tickers,
            "related_tickers": self.related_tickers,
            "mentioned_tickers": self.mentioned_tickers,
            "sentiment": self.sentiment,
            "chunk_type": self.chunk_type,
            "extra_metadata": self.extra_metadata,
        }


class TextChunker:
    """Generic text chunker that relies on metadata/summary builders."""

    def __init__(
        self,
        metadata_builder: MetadataBuilder,
        summary_builder: Optional[SummaryBuilder] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_size: int = 120,
        body_chunk_type: str = "body",
    ):
        if metadata_builder is None:
            raise ValueError("metadata_builder must be provided")

        self.metadata_builder = metadata_builder
        self.summary_builder = summary_builder
        self.chunk_size = chunk_size if chunk_size is not None else Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else Config.CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size
        self.body_chunk_type = body_chunk_type

    def chunk_article(self, article: Any) -> List[Chunk]:
        metadata = self.metadata_builder(article)
        summary_chunk = self._build_summary_chunk(article, metadata)
        body_chunks = self._build_body_chunks(article, metadata)

        chunks = ([] if summary_chunk is None else [summary_chunk]) + body_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        return chunks

    def _build_summary_chunk(self, article: Any, metadata: Dict[str, Any]) -> Optional[Chunk]:
        summary_text = self._build_summary_text(article)
        if not summary_text:
            return None
        return self._create_chunk(summary_text, 0, "summary", metadata)

    def _build_summary_text(self, article: Any) -> str:
        if self.summary_builder:
            return self.summary_builder(article)

        title = getattr(article, "title", "")
        summary = getattr(article, "summary", "")
        fallback_content = getattr(article, "content", "")
        summary_text = summary or fallback_content[:240]
        if title and summary_text:
            return f"{title}\n\n{summary_text}"
        return summary_text

    def _build_body_chunks(self, article: Any, metadata: Dict[str, Any]) -> List[Chunk]:
        content = getattr(article, "content", "")
        sentences = self._split_sentences(content)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current: List[str] = []
        current_len = 0
        chunk_index = 1

        for sentence in sentences:
            length = len(sentence)
            if current_len + length > self.chunk_size and current:
                chunk_text = " ".join(current)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(chunk_text, chunk_index, self.body_chunk_type, metadata))
                    chunk_index += 1

                overlap_size = min(self.chunk_overlap, len(current) - 1)
                overlap = current[-overlap_size:] if overlap_size > 0 else []
                current = overlap + [sentence]
                current_len = sum(len(s) for s in current)
            else:
                current.append(sentence)
                current_len += length

        if current:
            chunk_text = " ".join(current)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(chunk_text, chunk_index, self.body_chunk_type, metadata))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunk(self, text: str, index: int, chunk_type: str, metadata: Dict[str, Any]) -> Chunk:
        payload = dict(metadata)
        return Chunk(
            text=text.strip(),
            chunk_index=index,
            total_chunks=0,
            chunk_type=chunk_type,
            **payload,
        )


class EmbeddingGenerator:
    """Shared embedding generator with lazy model loading."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.device = device or Config.EMBEDDING_DEVICE
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as exc:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers") from exc

        logger.info("Loading embedding model: %s on device: %s", self.model_name, self.device)
        
        # Disable meta device to avoid meta tensor errors
        with torch.device('cpu'):
            self.model = SentenceTransformer(self.model_name)
        
        # Move to target device after initialization
        if self.device != 'cpu':
            self.model = self.model.to(self.device)
        else:
            self.model = self.model.to('cpu')

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self.model is None:
            self._load_model()
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self.model is None:
            self._load_model()

        with self._performance_tracker("embedding_generation", len(texts)):
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        return embeddings.tolist()

    def _performance_tracker(self, operation: str, count: int):
        class _Tracker:
            def __enter__(self_inner):
                self_inner.start = time.time()
                logger.info("Starting %s for %s texts", operation, count)

            def __exit__(self_inner, exc_type, exc_value, traceback):
                duration = time.time() - self_inner.start
                logger.info("Completed %s in %.2fs", operation, duration)

        return _Tracker()


def chunk_and_embed_articles(
    articles: List[Any],
    chunker: TextChunker,
    embedder: Optional[EmbeddingGenerator] = None,
    show_progress: bool = True,
) -> List[Tuple[Chunk, List[float]]]:
    """Generic chunk + embed pipeline that works across providers."""
    if not articles:
        logger.warning("No articles supplied for chunking")
        return []

    embedder = embedder or EmbeddingGenerator()

    chunks: List[Chunk] = []
    try:
        from tqdm import tqdm

        iterator = tqdm(articles, desc="Chunking articles", unit="article") if show_progress else articles
    except ImportError:
        iterator = articles

    for article in iterator:
        article_chunks = chunker.chunk_article(article)
        if article_chunks:
            chunks.extend(article_chunks)

    if not chunks:
        logger.warning("No chunks produced from supplied articles")
        return []

    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.generate_embeddings_batch(texts)
    return list(zip(chunks, embeddings))


__all__ = [
    "Chunk",
    "TextChunker",
    "EmbeddingGenerator",
    "chunk_and_embed_articles",
]
