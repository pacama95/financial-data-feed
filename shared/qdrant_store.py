"""Shared Qdrant vector store helpers used by RSS and API pipelines."""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from shared.config import Config
from shared.logging import (
    PerformanceTracker,
    get_logger,
    log_data_processing,
    log_operation_end,
    log_operation_start,
)

logger = get_logger("shared.qdrant_store")


class QdrantStore:
    """Thin wrapper around qdrant-client with hybrid-search tuning knobs."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance,
                PointStruct,
                QuantizationSearchParams,
                SearchParams,
                SparseVector,
                VectorParams,
            )
        except ImportError as exc:
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client") from exc

        self.QdrantClient = QdrantClient
        self.Distance = Distance
        self.VectorParams = VectorParams
        self.PointStruct = PointStruct
        self.SearchParams = SearchParams
        self.QuantizationSearchParams = QuantizationSearchParams
        self.SparseVector = SparseVector

        host = host or Config.QDRANT_HOST
        port = port or Config.QDRANT_PORT
        api_key = api_key or Config.QDRANT_API_KEY

        if api_key:
            self.client = QdrantClient(url=host, api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port)

        self.distance_metric = Config.QDRANT_DISTANCE_METRIC
        self.search_hnsw_ef = Config.QDRANT_SEARCH_HNSW_EF
        self.search_exact = Config.QDRANT_SEARCH_EXACT
        self.hybrid_enabled = Config.QDRANT_USE_HYBRID
        self.hybrid_alpha_default = Config.QDRANT_HYBRID_ALPHA
        self.rescore_enabled = Config.QDRANT_RESCORING_ENABLED
        self.rescore_oversampling = Config.QDRANT_RESCORING_OVERSAMPLING
        self.with_vectors = False
        self.with_payload = True

        logger.info("✅ Connected to Qdrant at %s:%s", host, port)

    def _ensure_payload_indexes(self, collection_name: str) -> None:
        """Ensure required payload indexes exist on the collection."""
        from qdrant_client.models import PayloadSchemaType
        
        required_indexes = {
            "published_date": PayloadSchemaType.DATETIME,
            "tickers": PayloadSchemaType.KEYWORD,
            "source_name": PayloadSchemaType.KEYWORD,
        }
        
        # Get existing collection info
        collection_info = self.client.get_collection(collection_name)
        existing_indexes = collection_info.payload_schema or {}
        
        # Create missing indexes
        for field_name, field_schema in required_indexes.items():
            if field_name not in existing_indexes:
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_schema,
                    )
                    logger.info(f"✅ Created payload index for '{field_name}' in collection '{collection_name}'")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to create index for '{field_name}': {e}")

    def create_collection(self, collection_name: str, vector_size: int, recreate_if_exists: bool = False) -> None:
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name in collection_names:
            collection_info = self.client.get_collection(collection_name)

            has_named_vectors = isinstance(collection_info.config.params.vectors, dict)

            if has_named_vectors:
                existing_size = collection_info.config.params.vectors.get("dense").size
                has_sparse = (
                    "sparse" in collection_info.config.params.sparse_vectors
                    if collection_info.config.params.sparse_vectors
                    else False
                )
            else:
                existing_size = collection_info.config.params.vectors.size
                has_sparse = False

            needs_recreation = False
            reason = None

            if existing_size != vector_size:
                needs_recreation = True
                reason = f"dimension mismatch ({existing_size} vs {vector_size})"
            elif not has_named_vectors:
                needs_recreation = True
                reason = "old schema (needs named vectors for hybrid search)"
            elif not has_sparse:
                needs_recreation = True
                reason = "missing sparse vector support"

            if needs_recreation:
                logger.warning(
                    "⚠️  Collection '%s' needs recreation: %s",
                    collection_name,
                    reason,
                )

                if recreate_if_exists:
                    self.client.delete_collection(collection_name)
                    logger.info("✅ Deleted old collection '%s'", collection_name)
                else:
                    raise ValueError(
                        f"Collection '{collection_name}' needs recreation ({reason}). "
                        "Set recreate_on_dimension_mismatch=True or delete it manually."
                    )
            else:
                # Check if required payload indexes exist
                self._ensure_payload_indexes(collection_name)
                logger.info(
                    "Collection '%s' already exists with correct configuration (dense: %s dims, sparse: enabled)",
                    collection_name,
                    vector_size,
                )
                return

        from qdrant_client.models import SparseVectorParams, PayloadSchemaType

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": self.VectorParams(size=vector_size, distance=self._resolve_distance_metric()),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(),
            },
        )
        
        # Create payload indexes for filtering
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="published_date",
            field_schema=PayloadSchemaType.DATETIME,
        )
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="tickers",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="source_name",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        
        logger.info(
            "Created collection '%s' (dense vector size: %s, sparse vectors: enabled, payload indexes: published_date, tickers, source_name)",
            collection_name,
            vector_size,
        )

    def upsert(
        self,
        collection_name: str,
        chunks_with_embeddings: List[Tuple[Chunk, List[float]]],
        batch_size: int = 100,
        lexical_vocab=None,
    ) -> None:
        start_time = time.time()
        total_chunks = len(chunks_with_embeddings)

        with PerformanceTracker(f"qdrant.upsert.{collection_name}"):
            log_operation_start(
                logger,
                "qdrant_upsert",
                collection_name=collection_name,
                total_chunks=total_chunks,
                batch_size=batch_size,
            )

            original_batch_size = batch_size
            if batch_size > 1000:
                logger.warning(
                    "Large batch size may cause memory issues",
                    extra={
                        "collection_name": collection_name,
                        "original_batch_size": batch_size,
                        "adjusted_batch_size": 500,
                        "warning": "memory_optimization",
                    },
                )
                batch_size = min(batch_size, 500)

            processed_chunks = 0
            failed_batches = 0

            try:
                from tqdm import tqdm

                iterator = tqdm(range(0, total_chunks, batch_size), desc="Upserting to Qdrant", unit="batch")
            except ImportError:
                iterator = range(0, total_chunks, batch_size)

            for i in iterator:
                batch = chunks_with_embeddings[i : i + batch_size]
                points = []

                for chunk, embedding in batch:
                    point_id = hash(f"{chunk.article_id}_{chunk.chunk_index}") & 0x7FFFFFFF
                    payload = chunk.to_dict()
                    vector = self._prepare_vector(embedding, point_id)
                    if vector is None:
                        continue

                    # Build vector dict with dense, optionally sparse
                    vectors = {"dense": vector}
                    
                    if lexical_vocab is not None:
                        # Generate sparse vector from chunk text + relevant metadata
                        # Include metadata fields to enable keyword search on tickers, sources, etc.
                        text_parts = [chunk.text or ""]
                        
                        # Add tickers (e.g., ASML, NVDA, AAPL)
                        if chunk.tickers:
                            text_parts.extend(chunk.tickers)
                        if chunk.related_tickers:
                            text_parts.extend(chunk.related_tickers)
                        if chunk.mentioned_tickers:
                            text_parts.extend(chunk.mentioned_tickers)
                        
                        # Add source and title for better keyword matching
                        if chunk.source_name:
                            text_parts.append(chunk.source_name)
                        if chunk.article_title:
                            text_parts.append(chunk.article_title)
                        
                        combined_text = " ".join(text_parts)
                        tokens = combined_text.lower().split() if combined_text else []
                        if tokens:
                            sparse = lexical_vocab.to_sparse_vector(tokens, l2_normalize=True)
                            if sparse.get("indices"):
                                from qdrant_client.models import SparseVector
                                vectors["sparse"] = SparseVector(
                                    indices=sparse["indices"],
                                    values=sparse["values"],
                                )

                    points.append(
                        self.PointStruct(
                            id=point_id,
                            vector=vectors,
                            payload=payload,
                        )
                    )

                del batch

                if points:
                    try:
                        batch_start = time.time()
                        self.client.upsert(collection_name=collection_name, points=points)
                        batch_duration = time.time() - batch_start

                        processed_chunks += len(points)

                        logger.debug(
                            "Upserted batch to %s",
                            collection_name,
                            extra={
                                "collection_name": collection_name,
                                "batch_start": i,
                                "batch_size": len(points),
                                "batch_duration_seconds": batch_duration,
                                "processed_chunks": processed_chunks,
                                "total_chunks": total_chunks,
                            },
                        )

                    except Exception as exc:
                        failed_batches += 1
                        logger.error(
                            "Failed to upsert batch to %s",
                            collection_name,
                            extra={
                                "collection_name": collection_name,
                                "batch_start": i,
                                "batch_size": len(points),
                                "error": str(exc),
                                "failed_batches": failed_batches,
                            },
                        )
                        raise

                del points

            total_duration = time.time() - start_time

            log_data_processing(
                logger,
                "qdrant_upsert",
                input_count=total_chunks,
                output_count=processed_chunks,
                duration=total_duration,
            )

            log_operation_end(
                logger,
                "qdrant_upsert",
                success=failed_batches == 0,
                collection_name=collection_name,
                total_chunks=total_chunks,
                processed_chunks=processed_chunks,
                failed_batches=failed_batches,
                original_batch_size=original_batch_size,
                final_batch_size=batch_size,
                duration_seconds=total_duration,
                throughput_chunks_per_second=processed_chunks / total_duration if total_duration > 0 else 0,
            )

    def delete_old_documents(self, collection_name: str, days: int) -> None:
        from qdrant_client.models import DatetimeRange, FieldCondition, Filter

        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_iso = cutoff_date.isoformat()

        self.client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="published_date",
                        range=DatetimeRange(lt=cutoff_iso),
                    )
                ]
            ),
        )
        logger.info("Deleted documents older than %s days from '%s'", days, collection_name)

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict] = None,
        query_sparse_vector: Optional[Dict[str, List]] = None,
        require_keyword_match: bool = False,
    ) -> List[Dict]:
        """
        Search collection with dense vector, optionally with hybrid (dense + sparse).
        
        Args:
            collection_name: Name of the collection
            query_vector: Dense embedding vector
            limit: Max results to return
            filters: Payload filters
            query_sparse_vector: Sparse vector dict with 'indices' and 'values' keys
            require_keyword_match: If True, return empty results when sparse search finds no matches
        """
        from qdrant_client.models import (
            Filter,
            FieldCondition,
            MatchAny,
            MatchValue,
            DatetimeRange,
            Prefetch,
            FusionQuery,
            Fusion,
            SparseVector,
        )

        search_params = self._build_search_params()
        
        # Build payload filter
        query_filter = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                # Special handling for date range filter
                if key == "_published_after":
                    must_conditions.append(
                        FieldCondition(
                            key="published_date",
                            range=DatetimeRange(gte=value),
                        )
                    )
                elif isinstance(value, list):
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                else:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            query_filter = Filter(must=must_conditions)

        # Hybrid search with prefetch + fusion
        if query_sparse_vector and query_sparse_vector.get("indices"):
            sparse_vec = SparseVector(
                indices=query_sparse_vector["indices"],
                values=query_sparse_vector["values"],
            )
            
            # If require_keyword_match is True, first check if sparse search has any results
            if require_keyword_match:
                sparse_only_result = self.client.query_points(
                    collection_name=collection_name,
                    query=sparse_vec,
                    using="sparse",
                    limit=1,
                    query_filter=query_filter,
                    with_payload=False,
                    with_vectors=False,
                )
                sparse_hits = sparse_only_result.points if hasattr(sparse_only_result, 'points') else []
                if not sparse_hits:
                    logger.info(
                        "No documents match the specified keywords in collection '%s'; returning empty results",
                        collection_name,
                    )
                    return []
            
            # Prefetch from both dense and sparse, then fuse with DBSF
            # DBSF (Distribution-Based Score Fusion) uses actual scores, not just ranks
            # This allows keyword boosts to have real impact on final ranking
            prefetch = [
                Prefetch(query=query_vector, using="dense", limit=limit * 2),
                Prefetch(query=sparse_vec, using="sparse", limit=limit * 2),
            ]
            
            result = self.client.query_points(
                collection_name=collection_name,
                prefetch=prefetch,
                query=FusionQuery(fusion=Fusion.DBSF),
                limit=limit,
                query_filter=query_filter,
                search_params=search_params,
                with_payload=self.with_payload,
                with_vectors=self.with_vectors,
            )
        else:
            # Simple dense search
            result = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using="dense",
                limit=limit,
                query_filter=query_filter,
                search_params=search_params,
                with_payload=self.with_payload,
                with_vectors=self.with_vectors,
            )

        hits = result.points if hasattr(result, 'points') else []
        if not hits:
            return []

        formatted_results = []
        for hit in hits:
            payload = hit.payload or {}
            formatted_results.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "collection": collection_name,
                    "metadata": payload,
                    "text": payload.get("text"),
                }
            )

        return formatted_results

    def _prepare_vector(self, embedding, point_id: int) -> Optional[List[float]]:
        if hasattr(embedding, "tolist"):
            vector = embedding.tolist()
        else:
            vector = list(embedding)

        cleaned = []
        for idx, value in enumerate(vector):
            try:
                fv = float(value)
            except (TypeError, ValueError):
                logger.warning("Non-numeric value at %s for point %s; skipping chunk", idx, point_id)
                return None
            if not math.isfinite(fv):
                logger.warning("Non-finite value at %s for point %s; skipping chunk", idx, point_id)
                return None
            cleaned.append(fv)
        return cleaned

    def _resolve_distance_metric(self):
        mapping = {
            "cosine": self.Distance.COSINE,
            "dot": self.Distance.DOT,
            "euclid": self.Distance.EUCLID,
            "l2": self.Distance.EUCLID,
        }
        metric = mapping.get(self.distance_metric)
        if not metric:
            logger.warning("Unknown Qdrant distance metric '%s', defaulting to COSINE.", self.distance_metric)
            return self.Distance.COSINE
        return metric

    def _build_search_params(self):
        quantization_params = None
        if self.rescore_enabled or self.rescore_oversampling:
            quantization_params = self.QuantizationSearchParams(
                rescore=self.rescore_enabled,
                oversampling=self.rescore_oversampling,
            )
        return self.SearchParams(
            hnsw_ef=self.search_hnsw_ef,
            exact=self.search_exact,
            quantization=quantization_params,
        )

    def _build_sparse_vector(self, sparse_data: Optional[Dict[str, List[float]]]):
        if not sparse_data:
            return None
        indices = sparse_data.get("indices", [])
        values = sparse_data.get("values", [])
        if not indices or not values:
            return None
        if len(indices) != len(values):
            return None
        return self.SparseVector(indices=indices, values=values)


# Singleton instance for global access
_default_store: Optional[QdrantStore] = None


def create_qdrant_store(config: Optional[Dict] = None) -> QdrantStore:
    """Get or create a singleton QdrantStore instance.
    
    Uses Config defaults or provided overrides. When called without config,
    returns the cached singleton instance for connection reuse.
    
    Args:
        config: Optional config overrides. If provided, creates a new instance
                (useful for testing or connecting to different Qdrant instances).
    
    Returns:
        QdrantStore instance (singleton when no config provided)
    """
    global _default_store
    
    # If custom config provided, create a new instance (don't cache)
    if config:
        return QdrantStore(
            host=config.get("host"),
            port=config.get("port"),
            api_key=config.get("api_key"),
        )
    
    # Return singleton for default config
    if _default_store is None:
        _default_store = QdrantStore()
    
    return _default_store


def reset_qdrant_store() -> None:
    """Reset the singleton QdrantStore instance.
    
    Useful for testing or when configuration changes.
    """
    global _default_store
    _default_store = None


# Import Chunk here to avoid circular import at module level
def _get_chunk_class():
    from shared.chunking_embedding import Chunk
    return Chunk
