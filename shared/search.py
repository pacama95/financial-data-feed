"""Unified search client supporting simple and hybrid queries across collections."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

from shared.config import Config
from shared.logging import get_logger

if TYPE_CHECKING:
    from shared.lexical import LexicalVocabulary
    from shared.vocabulary_store import VocabularyStore

logger = get_logger("shared.search")


def deduplicate_results(
    results: List[Dict],
    similarity_threshold: Optional[float] = None,
) -> List[Dict]:
    """Remove duplicate results by URL and text similarity."""
    if not results:
        return results

    threshold = similarity_threshold or Config.SEARCH_SIMILARITY_THRESHOLD

    # URL deduplication
    seen_urls = set()
    url_unique = []
    for r in results:
        url = r.get("metadata", {}).get("article_url")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        url_unique.append(r)

    # Text similarity deduplication
    final = []
    seen_texts: List[str] = []
    for r in url_unique:
        text = r.get("text", "")
        if not text:
            final.append(r)
            continue
        is_dup = any(
            SequenceMatcher(None, text.lower(), seen.lower()).ratio() >= threshold
            for seen in seen_texts
        )
        if not is_dup:
            final.append(r)
            seen_texts.append(text)

    logger.debug("Deduplicated %d -> %d results", len(results), len(final))
    return final


class SearchClient:
    """Unified search client for simple and hybrid queries.
    
    Vocabularies are automatically loaded from the vocabulary store when needed
    for hybrid search. No manual loading required.
    """

    def __init__(
        self,
        store=None,
        embedder=None,
        vocabulary_store: Optional["VocabularyStore"] = None,
    ):
        """Initialize search client.
        
        Args:
            store: Qdrant store instance (uses default if None)
            embedder: Embedding generator (uses default if None)
            vocabulary_store: Vocabulary store for lexical search (uses default if None)
        """
        from shared.qdrant_store import create_qdrant_store
        from shared.chunking_embedding import EmbeddingGenerator
        from shared.vocabulary_store import get_vocabulary_store

        self._store = store or create_qdrant_store()
        self._embedder = embedder or EmbeddingGenerator()
        self._vocabulary_store = vocabulary_store or get_vocabulary_store()
        self._lexical_vocabs: Dict[str, "LexicalVocabulary"] = {}  # Cache for loaded vocabularies

    def _get_lexical_vocab(self, collection: str) -> Optional["LexicalVocabulary"]:
        """Get lexical vocabulary for a collection, loading from store if needed.
        
        Vocabularies are cached after first load for performance.
        
        Args:
            collection: Collection name
            
        Returns:
            LexicalVocabulary instance, or None if not available
        """
        # Check cache first
        if collection in self._lexical_vocabs:
            return self._lexical_vocabs[collection]
        
        # Try to load from store
        vocab = self._vocabulary_store.get_vocabulary(collection)
        if vocab is not None:
            self._lexical_vocabs[collection] = vocab
            logger.info("Loaded vocabulary for collection '%s' from store", collection)
        
        return vocab

    def clear_vocabulary_cache(self, collection: Optional[str] = None) -> None:
        """Clear cached vocabularies to force reload from store.
        
        Args:
            collection: Collection to clear, or None to clear all
        """
        if collection:
            self._lexical_vocabs.pop(collection, None)
        else:
            self._lexical_vocabs.clear()
        logger.debug("Cleared vocabulary cache for %s", collection or "all collections")

    def search(
        self,
        query: str,
        collection: Optional[str] = None,
        mode: str = "simple",
        keywords: Optional[Sequence[str]] = None,
        keyword_boosts: Optional[Dict[str, float]] = None,
        limit: Optional[int] = None,
        filters: Optional[Dict] = None,
        deduplicate: Optional[bool] = None,
        published_after: Optional[str] = None,
        require_keyword_match: bool = False,
    ) -> List[Dict]:
        """
        Perform search on a collection.

        Args:
            query: Search query text
            collection: Collection name (defaults to Config.API_COLLECTION)
            mode: "simple" or "hybrid"
            keywords: Keywords for hybrid search (uses query tokens if None)
            keyword_boosts: Boost multipliers for keywords
            limit: Max results (defaults to Config.SEARCH_LIMIT_DEFAULT)
            filters: Qdrant payload filters
            deduplicate: Whether to deduplicate results
            published_after: ISO date string (e.g., "2025-01-01") to filter results newer than this date
            require_keyword_match: If True and explicit keywords are provided in hybrid mode,
                return empty results when no documents match the keywords (default: True)

        Returns:
            List of search results
        """
        collection = collection or Config.API_COLLECTION
        limit = limit if limit is not None else Config.SEARCH_LIMIT_DEFAULT
        deduplicate = deduplicate if deduplicate is not None else Config.SEARCH_DEDUPLICATE

        # Build filters with date constraint
        filters = dict(filters) if filters else {}
        if published_after:
            filters["_published_after"] = published_after

        # Generate query embedding
        query_vector = self._embedder.generate_embedding(query)

        # Hybrid search
        if mode == "hybrid":
            lexical_vocab = self._get_lexical_vocab(collection)
            if lexical_vocab is None:
                logger.warning(
                    "No lexical vocabulary available for collection '%s'; falling back to simple search",
                    collection,
                )
            else:
                kw = keywords or query.split()
                sparse_vector = lexical_vocab.to_sparse_vector(kw, boosts=keyword_boosts)
                if sparse_vector.get("indices"):
                    search_limit = limit * 2 if deduplicate else limit
                    results = self._store.search(
                        collection_name=collection,
                        query_vector=query_vector,
                        query_sparse_vector=sparse_vector,
                        limit=search_limit,
                        filters=filters,
                        require_keyword_match=require_keyword_match and keywords is not None,
                    )
                    if deduplicate:
                        results = deduplicate_results(results)[:limit]
                    return results
                else:
                    # Keywords not in vocabulary - no documents can match
                    missing = [k for k in kw if (k.lower() if lexical_vocab.lowercase else k) not in lexical_vocab.token_to_id]
                    if require_keyword_match and keywords is not None:
                        logger.info(
                            "Keywords not in vocabulary for collection '%s': %s; returning empty results. "
                            "These terms do not exist in indexed documents.",
                            collection,
                            missing,
                        )
                        return []
                    logger.warning(
                        "Keywords not in vocabulary for collection '%s': %s; falling back to simple search. "
                        "These terms may not exist in indexed documents.",
                        collection,
                        missing,
                    )

        # Simple search
        search_limit = limit * 2 if deduplicate else limit
        results = self._store.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=search_limit,
            filters=filters,
        )
        if deduplicate:
            results = deduplicate_results(results)[:limit]
        return results

    def search_rss(self, query: str, **kwargs) -> List[Dict]:
        """Search RSS collection."""
        return self.search(query, collection=Config.RSS_COLLECTION, **kwargs)

    def search_api(self, query: str, **kwargs) -> List[Dict]:
        """Search API collection."""
        return self.search(query, collection=Config.API_COLLECTION, **kwargs)

    def search_all(
        self,
        query: str,
        limit: Optional[int] = None,
        deduplicate: bool = True,
        **kwargs,
    ) -> List[Dict]:
        """Search across all collections and merge results."""
        limit = limit or Config.SEARCH_LIMIT_DEFAULT
        per_collection_limit = limit

        rss_results = self.search(query, collection=Config.RSS_COLLECTION, limit=per_collection_limit, deduplicate=False, **kwargs)
        api_results = self.search(query, collection=Config.API_COLLECTION, limit=per_collection_limit, deduplicate=False, **kwargs)

        for r in rss_results:
            r["collection"] = Config.RSS_COLLECTION
        for r in api_results:
            r["collection"] = Config.API_COLLECTION

        combined = rss_results + api_results
        combined.sort(key=lambda x: x.get("score", 0), reverse=True)

        if deduplicate:
            combined = deduplicate_results(combined)

        return combined[:limit]


__all__ = [
    "SearchClient",
    "deduplicate_results",
]
