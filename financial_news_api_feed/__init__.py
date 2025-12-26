"""
Financial News API feed module - fetch and structure API news data.

This module provides API fetching capabilities. For storage and search,
use the shared module:

    from shared import APIPipeline, SearchClient
"""

from financial_news_api_feed.models import NewsArticle
from financial_news_api_feed.base_provider import BaseNewsProvider
from financial_news_api_feed.provider_registry import ProviderRegistry
from financial_news_api_feed.providers.yahoo_finance import YahooFinanceProvider

__all__ = [
    "NewsArticle",
    "BaseNewsProvider",
    "ProviderRegistry",
    "YahooFinanceProvider",
]
