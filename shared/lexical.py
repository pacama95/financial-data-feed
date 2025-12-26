"""Lexical vocabulary and sparse vector utilities for hybrid search.

Uses rank_bm25 for proper BM25 scoring (Okapi BM25).
"""

from __future__ import annotations

import json
import math
import re
from typing import Dict, List, Optional, Sequence

from shared.config import Config
from shared.logging import get_logger

logger = get_logger("shared.lexical")


class LexicalVocabulary:
    """
    BM25-based lexical vocabulary for hybrid search.
    
    Uses rank_bm25 library for proper Okapi BM25 scoring with
    term frequency saturation and document length normalization.
    """

    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9$%\.]+")

    def __init__(
        self,
        token_to_id: Dict[str, int],
        idf_scores: Dict[str, float],
        doc_count: int,
        avg_doc_len: float,
        lowercase: bool = True,
        k1: Optional[float] = None,
        b: Optional[float] = None,
    ):
        self.token_to_id = token_to_id
        self.idf_scores = idf_scores
        self.doc_count = doc_count
        self.avg_doc_len = avg_doc_len
        self.lowercase = lowercase
        self.k1 = k1 if k1 is not None else Config.LEXICAL_K1
        self.b = b if b is not None else Config.LEXICAL_B

    @classmethod
    def build(
        cls,
        documents: Sequence[str],
        lowercase: bool = True,
        min_freq: int = 1,
        k1: Optional[float] = None,
        b: Optional[float] = None,
    ) -> "LexicalVocabulary":
        """Build BM25 vocabulary from corpus using rank_bm25."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("rank_bm25 not installed. Run: pip install rank-bm25") from e

        k1 = k1 if k1 is not None else Config.LEXICAL_K1
        b = b if b is not None else Config.LEXICAL_B

        # Tokenize all documents
        tokenized_docs = []
        for text in documents:
            if not text:
                continue
            tokens = cls._tokenize(text, lowercase)
            if tokens:
                tokenized_docs.append(tokens)

        if not tokenized_docs:
            return cls({}, {}, 0, 0.0, lowercase=lowercase, k1=k1, b=b)

        # Build BM25 model
        bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)

        # Extract vocabulary and IDF scores from BM25
        # Aggregate document frequencies across all docs
        from collections import Counter
        global_doc_freq: Counter = Counter()
        for doc_freqs in bm25.doc_freqs:
            for token in doc_freqs:
                global_doc_freq[token] += 1

        # Build token_to_id and idf_scores, filtering by min_freq
        token_to_id = {}
        idf_scores = {}
        idx = 0

        for token, df in global_doc_freq.items():
            if df >= min_freq:
                token_to_id[token] = idx
                # BM25 IDF from rank_bm25
                idf_scores[token] = bm25.idf.get(token, 0.0)
                idx += 1

        avg_doc_len = bm25.avgdl
        doc_count = len(tokenized_docs)

        logger.info(
            "Built BM25 vocabulary: %d tokens from %d documents (k1=%.2f, b=%.2f)",
            len(token_to_id),
            doc_count,
            k1,
            b,
        )

        return cls(
            token_to_id=token_to_id,
            idf_scores=idf_scores,
            doc_count=doc_count,
            avg_doc_len=avg_doc_len,
            lowercase=lowercase,
            k1=k1,
            b=b,
        )

    @staticmethod
    def _tokenize(text: str, lowercase: bool) -> List[str]:
        if lowercase:
            text = text.lower()
        return LexicalVocabulary.TOKEN_PATTERN.findall(text)

    def _normalize_token(self, token: str) -> str:
        token = token.strip()
        return token.lower() if self.lowercase else token

    def _bm25_weight(self, token: str, tf: int, doc_len: int) -> float:
        """Compute full BM25 weight for a token."""
        idf = self.idf_scores.get(token, 0.0)
        if idf == 0:
            return 0.0

        # BM25 term frequency component with saturation
        tf_component = (tf * (self.k1 + 1)) / (
            tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
        )
        return idf * tf_component

    def to_sparse_vector(
        self,
        keywords: Sequence[str],
        boosts: Optional[Dict[str, float]] = None,
        l2_normalize: bool = True,
    ) -> Dict[str, List]:
        """Convert keywords to sparse vector with full BM25 weighting."""
        if not keywords or not self.token_to_id:
            return {"indices": [], "values": []}

        boosts = boosts or {}
        normalized = [self._normalize_token(k) for k in keywords if k and k.strip()]
        
        # Treat keywords as a "query document"
        doc_len = len(normalized)
        from collections import Counter
        counts = Counter(normalized)

        indices: List[int] = []
        values: List[float] = []

        for token, tf in counts.items():
            if token not in self.token_to_id:
                continue
            
            # Full BM25 weight
            weight = self._bm25_weight(token, tf, doc_len)
            
            # Apply boost if specified
            if token in boosts:
                weight *= boosts[token]
            
            if weight > 0:
                indices.append(self.token_to_id[token])
                values.append(weight)

        if l2_normalize and values:
            norm = math.sqrt(sum(v * v for v in values))
            if norm > 0:
                values = [v / norm for v in values]

        return {"indices": indices, "values": values}

    def save(self, path: str) -> None:
        """Save vocabulary to JSON file."""
        data = {
            "token_to_id": self.token_to_id,
            "idf_scores": self.idf_scores,
            "doc_count": self.doc_count,
            "avg_doc_len": self.avg_doc_len,
            "lowercase": self.lowercase,
            "k1": self.k1,
            "b": self.b,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("Saved BM25 vocabulary to %s", path)

    @classmethod
    def load(cls, path: str) -> "LexicalVocabulary":
        """Load vocabulary from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded BM25 vocabulary from %s", path)
        return cls(
            token_to_id=data.get("token_to_id", {}),
            idf_scores=data.get("idf_scores", data.get("doc_freq", {})),  # Backward compat
            doc_count=data.get("doc_count", 0),
            avg_doc_len=data.get("avg_doc_len", 0.0),
            lowercase=data.get("lowercase", True),
            k1=data.get("k1"),
            b=data.get("b"),
        )

    def __len__(self) -> int:
        return len(self.token_to_id)

    def merge(self, other: "LexicalVocabulary") -> "LexicalVocabulary":
        """
        Merge this vocabulary with another, combining tokens and IDF scores.
        
        For overlapping tokens, IDF scores are averaged weighted by doc count.
        
        Args:
            other: Another LexicalVocabulary to merge with
            
        Returns:
            New merged LexicalVocabulary
        """
        # Combine all tokens
        all_tokens = set(self.token_to_id.keys()) | set(other.token_to_id.keys())
        
        # Build new token_to_id mapping
        merged_token_to_id = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        
        # Merge IDF scores (weighted average for overlapping tokens)
        merged_idf = {}
        total_docs = self.doc_count + other.doc_count
        
        for token in all_tokens:
            idf_self = self.idf_scores.get(token, 0.0)
            idf_other = other.idf_scores.get(token, 0.0)
            
            if token in self.idf_scores and token in other.idf_scores:
                # Weighted average by doc count
                merged_idf[token] = (
                    idf_self * self.doc_count + idf_other * other.doc_count
                ) / total_docs
            elif token in self.idf_scores:
                merged_idf[token] = idf_self
            else:
                merged_idf[token] = idf_other
        
        # Weighted average for avg_doc_len
        merged_avg_doc_len = (
            self.avg_doc_len * self.doc_count + other.avg_doc_len * other.doc_count
        ) / total_docs if total_docs > 0 else 0.0
        
        logger.info(
            "Merged vocabularies: %d + %d tokens -> %d tokens, %d + %d docs -> %d docs",
            len(self), len(other), len(merged_token_to_id),
            self.doc_count, other.doc_count, total_docs,
        )
        
        return LexicalVocabulary(
            token_to_id=merged_token_to_id,
            idf_scores=merged_idf,
            doc_count=total_docs,
            avg_doc_len=merged_avg_doc_len,
            lowercase=self.lowercase,
            k1=self.k1,
            b=self.b,
        )


def build_sparse_vector(
    vocabulary: LexicalVocabulary,
    keywords: Sequence[str],
    boosts: Optional[Dict[str, float]] = None,
    l2_normalize: bool = True,
) -> Dict[str, List]:
    """Convenience wrapper for LexicalVocabulary.to_sparse_vector."""
    return vocabulary.to_sparse_vector(keywords, boosts=boosts, l2_normalize=l2_normalize)
