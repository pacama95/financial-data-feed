"""Financial RSS feed module - fetch and structure RSS news data.

This module provides RSS fetching capabilities. For storage and search,
use the shared module:

    from shared import RSSPipeline, SearchClient
"""

from financial_rss_feed.rss_fetcher import (
    RSSFetcher,
    Article,
    Region,
    SourceTier,
    FeedType,
)

__all__ = [
    "RSSFetcher",
    "Article",
    "Region",
    "SourceTier",
    "FeedType",
]
