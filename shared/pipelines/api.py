"""API pipeline for financial news ingestion from provider APIs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from shared.config import Config
from shared.logging import get_logger
from shared.pipelines.base import BasePipeline

logger = get_logger("shared.pipelines.api")


class APIPipeline(BasePipeline):
    """Pipeline for ingesting financial news from API providers (Yahoo Finance, etc.)."""

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        days_back: int = 7,
        max_articles_per_ticker: int = 10,
        # BasePipeline args
        collection: Optional[str] = None,
        batch_size: Optional[int] = None,
        cleanup_days: Optional[int] = None,
        recreate_collection: bool = False,
        **kwargs,  # Ignore unknown args
    ):
        """Initialize API pipeline.
        
        Args:
            tickers: List of ticker symbols to fetch news for.
                     Default: ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
            providers: List of provider names to use.
                       Available: see APIPipeline.get_available_providers()
                       Default: ["yahoo"]
            days_back: Number of days back to fetch (default: 7)
            max_articles_per_ticker: Max articles per ticker (default: 10)
            collection: Override collection name
            batch_size: Override batch size for Qdrant upserts
            cleanup_days: Override cleanup period
            recreate_collection: Recreate collection if schema changes
        """
        super().__init__(
            collection=collection,
            batch_size=batch_size,
            cleanup_days=cleanup_days,
            recreate_collection=recreate_collection,
        )
        self.tickers = tickers or ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
        self.providers = providers or ["yahoo"]
        self.days_back = days_back
        self.max_articles_per_ticker = max_articles_per_ticker

    def _default_collection(self) -> str:
        return Config.API_COLLECTION

    def fetch(self) -> List[Any]:
        """Fetch articles from API providers using configured filters."""
        all_articles = []

        for provider_name in self.providers:
            provider = self._get_provider(provider_name)
            if provider is None:
                continue

            for ticker in self.tickers:
                try:
                    articles = provider.fetch_ticker_news(
                        ticker=ticker,
                        max_results=self.max_articles_per_ticker,
                        days_back=self.days_back,
                    )
                    all_articles.extend(articles)
                    logger.debug(
                        "Fetched %d articles for %s from %s",
                        len(articles),
                        ticker,
                        provider_name,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to fetch news for %s from %s: %s",
                        ticker,
                        provider_name,
                        e,
                    )

        logger.info(
            "Fetched %d total articles from API providers (tickers=%s, providers=%s)",
            len(all_articles),
            self.tickers,
            self.providers,
        )
        return all_articles

    @staticmethod
    def get_available_providers() -> dict[str, str]:
        """Get mapping of short names to registry names for available providers.
        
        Returns:
            Dict mapping short name -> registry name (e.g., {"yahoo": "yahoo_finance"})
        """
        return {
            "yahoo": "yahoo_finance",
            # Add new providers here:
            # "alphavantage": "alpha_vantage",
        }

    def _get_provider(self, provider_name: str):
        """Get provider instance by name using the ProviderRegistry."""
        try:
            # Import providers module to trigger registration
            import financial_news_api_feed.providers  # noqa: F401
            from financial_news_api_feed.provider_registry import ProviderRegistry
            
            # Map short names to registry names
            provider_map = self.get_available_providers()
            registry_name = provider_map.get(provider_name, provider_name)
            
            return ProviderRegistry.create(registry_name)
        except ValueError as e:
            available = list(self.get_available_providers().keys())
            logger.warning("Unknown provider: %s. Available: %s", provider_name, available)
            return None
        except ImportError as e:
            logger.error("Failed to import provider %s: %s", provider_name, e)
            return None

    def _build_metadata(self, article: Any) -> Dict[str, Any]:
        """Build chunk metadata from NewsArticle."""
        return {
            "article_id": getattr(article, "article_id", ""),
            "article_url": getattr(article, "url", ""),
            "article_title": getattr(article, "title", ""),
            "published_date": (
                article.published_date.isoformat()
                if hasattr(article, "published_date") and article.published_date
                else ""
            ),
            "source_name": getattr(article, "source_name", ""),
            "provider": getattr(article, "provider", "api"),
            "source_tier": "",
            "region": "",
            "feed_name": "",
            "feed_type": "api",
            "author": getattr(article, "author", None),
            "tags": getattr(article, "tags", []) or [],
            "tickers": getattr(article, "tickers", []) or [],
            "related_tickers": getattr(article, "related_tickers", []) or [],
            "mentioned_tickers": [],
            "sentiment": getattr(article, "sentiment", None),
        }

    def _build_summary(self, article: Any) -> str:
        """Build summary text from NewsArticle."""
        title = getattr(article, "title", "")
        summary = getattr(article, "summary", "")
        if title and summary:
            return f"{title}\n\n{summary}"
        return summary or title
