"""Yahoo Finance provider implementation using yfinance."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import time

from ..base_provider import BaseNewsProvider
from ..models import NewsArticle
from shared import TickerExtractor, get_logger, PerformanceTracker

logger = get_logger("financial_news_api_feed.providers.yahoo_finance")


class YahooFinanceProvider(BaseNewsProvider):
    """Yahoo Finance news provider using yfinance library."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError as exc:
            raise ImportError("yfinance not installed. Run: pip install yfinance") from exc
        
        self.ticker_extractor = TickerExtractor()

    def get_provider_name(self) -> str:
        """Return provider identifier."""
        return "yahoo_finance"

    def fetch_ticker_news(
        self,
        ticker: str,
        max_results: Optional[int] = None,
        days_back: int = 7,
    ) -> List[NewsArticle]:
        """Fetch news for a single ticker from Yahoo Finance."""
        max_results = max_results or self.max_results
        start_time = time.time()
        
        try:
            with PerformanceTracker(f"yahoo_finance.fetch_news.{ticker}"):
                # Get ticker object
                ticker_obj = self.yf.Ticker(ticker.upper())
                
                # Fetch news with timing
                api_start = time.time()
                news_items = ticker_obj.news
                api_duration = time.time() - api_start
                
                                
                if not news_items:
                    logger.info("No news found for ticker %s (%.2fs)", ticker, api_duration)
                    return []

                # Convert to NewsArticle objects
                articles = []
                cutoff_date = datetime.now() - timedelta(days=days_back)
                conversion_errors = 0
                
                for item in news_items[:max_results]:
                    try:
                        article = self._convert_news_item(item, ticker, cutoff_date)
                        if article:
                            articles.append(article)
                    except Exception as e:
                        conversion_errors += 1
                        logger.warning(f"Failed to convert news item for {ticker}: {e}", extra={
                            'ticker': ticker,
                            'error_type': 'conversion_error',
                            'conversion_errors': conversion_errors
                        })
                        continue

                # Enrich with additional ticker extraction
                with PerformanceTracker("yahoo_finance.enrich_articles"):
                    enriched_articles = []
                    for article in articles:
                        enriched = self.enrich_article(article)
                        enriched_articles.append(enriched)

                # Deduplicate
                unique_articles = self.deduplicate(enriched_articles)
                
                total_duration = time.time() - start_time
                logger.info(
                    "Fetched %d articles for %s (%.2fs)",
                    len(unique_articles),
                    ticker,
                    total_duration,
                )
                
                return unique_articles

        except Exception as e:
            total_duration = time.time() - start_time
            logger.error("Failed to fetch news for %s: %s (%.2fs)", ticker, e, total_duration)
            return []

    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        max_results: Optional[int] = None,
        days_back: int = 7,
    ) -> List[NewsArticle]:
        """Fetch news for multiple tickers with rate limiting."""
        all_articles = []
        
        for ticker in tickers:
            self._log_fetch_start(ticker)
            articles = self.fetch_ticker_news(ticker, max_results, days_back)
            all_articles.extend(articles)
            self._log_fetch_end(ticker, len(articles))
            
            # Respect rate limit
            self._respect_rate_limit()

        # Global deduplication across all tickers
        unique_articles = self.deduplicate(all_articles)
        logger.info(f"Total unique articles across all tickers: {len(unique_articles)}")
        
        return unique_articles

    def enrich_article(self, article: NewsArticle) -> NewsArticle:
        """Extract additional tickers from article content."""
        # Extract tickers from title and content
        title_tickers = self.ticker_extractor.extract_tickers(article.title)
        content_tickers = self.ticker_extractor.extract_tickers(article.content)
        
        # Combine with existing tickers, remove duplicates
        all_tickers = set(article.tickers + article.related_tickers + title_tickers + content_tickers)
        
        # Keep primary tickers separate from related ones
        primary_tickers = set(article.tickers)
        related_tickers = all_tickers - primary_tickers
        
        article.tickers = sorted(list(primary_tickers))
        article.related_tickers = sorted(list(related_tickers))
        
        return article

    def _convert_news_item(
        self,
        news_item: Dict[str, Any],
        ticker: str,
        cutoff_date: datetime,
    ) -> Optional[NewsArticle]:
        """Convert Yahoo Finance news item to NewsArticle."""
        try:
            # Extract the nested content from the yfinance structure
            content_data = news_item.get('content', {})
            if not isinstance(content_data, dict):
                return None
            
            # Extract publish date from the structure
            pub_date_str = content_data.get('pubDate')
            if pub_date_str:
                try:
                    # Parse ISO format date: "2025-11-22T12:14:03Z"
                    pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                except ValueError:
                    # Fallback to current date if parsing fails
                    pub_date = datetime.now()
            else:
                pub_date = datetime.now()
            
            # Make both dates comparable by removing timezone from pub_date
            if pub_date.tzinfo is not None:
                pub_date = pub_date.replace(tzinfo=None)
            
            # Skip if too old
            if pub_date < cutoff_date:
                return None
            
            # Extract content from the nested structure
            title = content_data.get('title', '')
            summary = content_data.get('summary', '')
            description = content_data.get('description', '')
            
            # Use summary or description as content, fallback to title
            content = summary or description or title
            
            if not title.strip():
                return None

            # Extract additional metadata from the new structure
            provider_info = content_data.get('provider', {})
            canonical_url = content_data.get('canonicalUrl', {})
            thumbnail = content_data.get('thumbnail', {})
            
            # Create article
            article = NewsArticle(
                title=title.strip(),
                content=content.strip(),  # Use title if no content
                url=canonical_url.get('url', content_data.get('previewUrl', '')),
                published_date=pub_date,
                fetched_date=datetime.now(),
                source_name="Yahoo Finance",
                provider="yahoo_finance",
                tickers=[ticker.upper()],  # Primary ticker
                author='',  # Not available in new structure
                summary=summary.strip(),
                tags=[],  # Yahoo doesn't provide tags
                sentiment=None,  # Could be added later with sentiment analysis
                article_id='',
                content_hash='',
                extra_metadata={
                    'publisher': provider_info.get('displayName', ''),
                    'publisher_url': provider_info.get('url', ''),
                    'content_type': content_data.get('contentType', ''),
                    'thumbnail': thumbnail,
                    'is_hosted': content_data.get('isHosted', False),
                    'preview_url': content_data.get('previewUrl', ''),
                    'content_id': content_data.get('id', ''),
                    'original_data': news_item
                }
            )
            
            return article

        except Exception as e:
            logger.warning(f"Failed to convert news item: {e}")
            return None
