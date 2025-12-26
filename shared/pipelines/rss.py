"""RSS pipeline for financial news ingestion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from shared.config import Config
from shared.logging import get_logger
from shared.pipelines.base import BasePipeline

logger = get_logger("shared.pipelines.rss")


class RSSPipeline(BasePipeline):
    """Pipeline for ingesting financial news from RSS feeds."""

    def __init__(
        self,
        regions: Optional[List[str]] = None,
        tiers: Optional[List[str]] = None,
        tickers: Optional[List[str]] = None,
        company_names: Optional[List[str]] = None,
        max_age_hours: int = 24,
        # BasePipeline args
        collection: Optional[str] = None,
        batch_size: Optional[int] = None,
        cleanup_days: Optional[int] = None,
        recreate_collection: bool = False,
        **kwargs,  # Ignore unknown args
    ):
        """Initialize RSS pipeline.
        
        Args:
            regions: List of region codes to fetch (None = all regions).
                     Available: "usa", "spain", "global"
            tiers: List of source tiers to fetch (None = all tiers).
                   Available: "tier1", "tier2", "tier3"
            tickers: Optional list of stock tickers for stock-specific feeds
            company_names: Optional list of company names for company-specific feeds
            max_age_hours: Only fetch articles newer than this (default: 24)
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
        self.regions = regions
        self.tiers = tiers
        self.tickers = tickers
        self.company_names = company_names
        self.max_age_hours = max_age_hours

    def _default_collection(self) -> str:
        return Config.RSS_COLLECTION

    def fetch(self) -> List[Any]:
        """Fetch articles from RSS feeds using configured filters."""
        try:
            from financial_rss_feed.rss_fetcher import RSSFetcher, Region, SourceTier
        except ImportError as e:
            logger.error("Failed to import RSS fetcher: %s", e)
            raise

        fetcher = RSSFetcher(
            tickers=self.tickers,
            company_names=self.company_names,
            max_age_hours=self.max_age_hours,
        )

        # Convert region strings to enums if provided
        region_enums = None
        if self.regions:
            region_enums = [Region(r) if isinstance(r, str) else r for r in self.regions]

        # Convert tier strings to enums if provided
        tier_enums = None
        if self.tiers:
            tier_enums = [SourceTier(t) if isinstance(t, str) else t for t in self.tiers]

        articles = fetcher.fetch_all(
            regions=region_enums,
            tiers=tier_enums,
        )

        logger.info(
            "Fetched %d articles from RSS feeds (regions=%s, tiers=%s, tickers=%s)",
            len(articles),
            self.regions,
            self.tiers,
            self.tickers,
        )
        return articles

    def _build_metadata(self, article: Any) -> Dict[str, Any]:
        """Build chunk metadata from RSS Article."""
        return {
            "article_id": getattr(article, "guid", ""),
            "article_url": getattr(article, "url", ""),
            "article_title": getattr(article, "title", ""),
            "published_date": (
                article.published_date.isoformat()
                if hasattr(article, "published_date") and article.published_date
                else ""
            ),
            "source_name": getattr(article, "source_name", ""),
            "source_tier": getattr(article, "source_tier", ""),
            "region": getattr(article, "region", ""),
            "feed_name": getattr(article, "feed_name", ""),
            "feed_type": getattr(article, "feed_type", "rss"),
            "provider": "rss",
            "author": getattr(article, "author", None),
            "tags": getattr(article, "tags", []) or [],
            "tickers": getattr(article, "mentioned_tickers", []) or [],
            "related_tickers": [],
            "mentioned_tickers": getattr(article, "mentioned_tickers", []) or [],
        }

    def _build_summary(self, article: Any) -> str:
        """Build summary text from RSS Article."""
        title = getattr(article, "title", "")
        summary = getattr(article, "summary", "")
        if title and summary:
            return f"{title}\n\n{summary}"
        return summary or title
