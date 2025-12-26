"""Abstract base class for financial news providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time

from .models import NewsArticle
from shared import get_logger, PerformanceTracker, log_operation_start, log_operation_end

logger = get_logger("financial_news_api_feed.base_provider")


class BaseNewsProvider(ABC):
    """Abstract base class defining the contract for news providers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.rate_limit_seconds: float = float(self.config.get("rate_limit", 1.0))
        self.max_results: int = int(self.config.get("max_results", 50))
        self.api_key: Optional[str] = self.config.get("api_key")

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return a unique provider identifier (e.g., 'yahoo_finance')."""

    @abstractmethod
    def fetch_ticker_news(
        self,
        ticker: str,
        max_results: Optional[int] = None,
    ) -> List[NewsArticle]:
        """Fetch news for a single ticker."""

    @abstractmethod
    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        max_results: Optional[int] = None,
    ) -> List[NewsArticle]:
        """Fetch news for multiple tickers."""

    def enrich_article(self, article: NewsArticle) -> NewsArticle:
        """Optional enrichment hook - can be overridden by subclasses."""
        return article

    def deduplicate(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        with PerformanceTracker(f"{self.get_provider_name()}.deduplicate"):
            seen_hashes = set()
            unique_articles = []
            for article in articles:
                if article.content_hash not in seen_hashes:
                    seen_hashes.add(article.content_hash)
                    unique_articles.append(article)
            
            duplicates_removed = len(articles) - len(unique_articles)
            if duplicates_removed > 0:
                logger.info(
                    "Removed duplicate articles",
                    extra={
                        'provider': self.get_provider_name(),
                        'duplicates_removed': duplicates_removed,
                        'original_count': len(articles),
                        'final_count': len(unique_articles),
                        'operation': 'deduplication'
                    }
                )
            
            return unique_articles

    def _respect_rate_limit(self):
        if self.rate_limit_seconds > 0:
            time.sleep(self.rate_limit_seconds)

    def _log_fetch_start(self, ticker: str):
        log_operation_start(
            logger, 
            f"{self.get_provider_name()}.fetch_news",
            ticker=ticker,
            provider=self.get_provider_name()
        )

    def _log_fetch_end(self, ticker: str, count: int):
        log_operation_end(
            logger,
            f"{self.get_provider_name()}.fetch_news",
            success=True,
            ticker=ticker,
            provider=self.get_provider_name(),
            articles_retrieved=count
        )
