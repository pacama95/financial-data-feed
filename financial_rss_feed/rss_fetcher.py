"""
RSS Fetcher for Financial News
Fetches, enriches, chunks, and stores articles in vector DB
"""

import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import hashlib
import time
from dataclasses import dataclass, asdict
from enum import Enum
import re
from shared.ticker_extractor import TickerExtractor
from shared.logging import get_logger


logger = get_logger("financial_rss_feed.fetcher")


class Region(Enum):
    """Geographic regions for news classification"""
    USA = "usa"
    SPAIN = "spain"
    GERMANY = "germany"
    UK = "uk"
    CHINA = "china"
    FRANCE = "france"
    ITALY = "italy"
    JAPAN = "japan"
    INDIA = "india"
    BRAZIL = "brazil"
    AUSTRALIA = "australia"
    CANADA = "canada"
    ASIA_PACIFIC = "asia_pacific"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"
    EU_GENERAL = "eu_general"
    GLOBAL = "global"


class SourceTier(Enum):
    """Source credibility tiers for weighting in RAG"""
    TIER1_PREMIUM = "tier1"      # WSJ, FT, Bloomberg, Reuters
    TIER2_STANDARD = "tier2"     # MarketWatch, CNBC, Seeking Alpha
    TIER3_ANALYSIS = "tier3"     # Analysis sites, blogs
    OFFICIAL = "official"        # Fed, SEC, Treasury
    SOCIAL = "social"            # Reddit, Twitter


class FeedType(Enum):
    """High-level categorization of feed origins"""
    NEWSROOM = "newsroom"              # Traditional editorial outlets
    OFFICIAL = "official"              # Regulators, central banks, etc.
    SOCIAL = "social"                  # Reddit, social sentiment
    STOCK_TICKER = "stock_ticker"      # Searches generated from ticker symbols
    COMPANY_QUERY = "company_query"    # Searches generated from company names


@dataclass
class Article:
    """Structured article data with rich metadata for RAG"""
    
    # Core content
    title: str
    content: str
    url: str
    
    # Metadata for RAG filtering and retrieval
    published_date: datetime
    fetched_date: datetime
    source_name: str
    source_tier: str
    region: str
    feed_name: str
    feed_type: str
    
    # Additional metadata
    author: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = None
    mentioned_tickers: List[str] = None
    
    # RAG-specific fields
    guid: str = ""
    content_hash: str = ""
    language: str = "en"
    
    def __post_init__(self):
        """Generate hash and validate data"""
        if not self.guid:
            self.guid = self._generate_guid()
        if not self.content_hash:
            self.content_hash = self._generate_content_hash()
        if self.tags is None:
            self.tags = []
        if self.mentioned_tickers is None:
            self.mentioned_tickers = []
    
    def _generate_guid(self) -> str:
        """Generate unique ID from URL"""
        return hashlib.md5(self.url.encode()).hexdigest()
    
    def _generate_content_hash(self) -> str:
        """Generate hash for deduplication"""
        content = f"{self.title}{self.content}".lower().strip()
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Convert datetime objects to ISO format
        data['published_date'] = self.published_date.isoformat()
        data['fetched_date'] = self.fetched_date.isoformat()
        return data
    
    def to_text_for_embedding(self) -> str:
        """
        Create optimized text for embedding generation
        Combines title and content with proper formatting
        """
        parts = [
            f"Title: {self.title}",
            f"Content: {self.content}",
        ]
        if self.summary:
            parts.insert(1, f"Summary: {self.summary}")
        
        return "\n\n".join(parts)


class RSSFeedConfig:
    """Configuration for all RSS feeds with metadata"""
    
    # US Financial News
    USA_FEEDS = {
        'bloomberg_markets': {
            'url': 'https://feeds.bloomberg.com/markets/news.rss',
            'tier': SourceTier.TIER1_PREMIUM,
            'region': Region.USA,
            'feed_type': FeedType.NEWSROOM,
        },
        'reuters_business': {
            'url': 'https://news.google.com/rss/search?q=site%3Areuters.com&hl=en-US&gl=US&ceid=US%3Aen',
            'tier': SourceTier.TIER1_PREMIUM,
            'region': Region.USA,
            'feed_type': FeedType.NEWSROOM,
        },
        'wsj_markets': {
            'url': 'https://feeds.content.dowjones.io/public/rss/RSSWorldNews',
            'tier': SourceTier.TIER1_PREMIUM,
            'region': Region.GLOBAL,
            'feed_type': FeedType.NEWSROOM,
        },
        'wsj_economy': {
            'url': 'https://feeds.content.dowjones.io/public/rss/socialeconomyfeed',
            'tier': SourceTier.TIER1_PREMIUM,
            'region': Region.USA,
            'feed_type': FeedType.NEWSROOM,
        },
        'wsj_world': {
            'url': 'https://feeds.content.dowjones.io/public/rss/RSSWorldNews',
            'tier': SourceTier.TIER1_PREMIUM,
            'region': Region.GLOBAL,
            'feed_type': FeedType.NEWSROOM,
        },
        'wsj_technology': {
            'url': 'https://feeds.content.dowjones.io/public/rss/RSSWSJD',
            'tier': SourceTier.TIER1_PREMIUM,
            'region': Region.GLOBAL,
            'feed_type': FeedType.NEWSROOM,
        },
        'marketwatch_topstories': {
            'url': 'http://feeds.marketwatch.com/marketwatch/topstories/',
            'tier': SourceTier.TIER2_STANDARD,
            'region': Region.USA,
            'feed_type': FeedType.NEWSROOM,
        },
        'cnbc_topnews': {
            'url': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147',
            'tier': SourceTier.TIER2_STANDARD,
            'region': Region.USA,
            'feed_type': FeedType.NEWSROOM,
        },
        'seeking_alpha': {
            'url': 'https://seekingalpha.com/market_currents.xml',
            'tier': SourceTier.TIER2_STANDARD,
            'region': Region.USA,
            'feed_type': FeedType.NEWSROOM,
        },
    }
    
    # Spain Financial News
    SPAIN_FEEDS = {
        'expansion': {
            'url': 'https://e00-expansion.uecdn.es/rss/portada.xml',
            'tier': SourceTier.TIER2_STANDARD,
            'region': Region.SPAIN,
            'feed_type': FeedType.NEWSROOM,
        },
        'expansion_mercados': {
            'url': 'https://e01-expansion.uecdn.es/rss/mercados.xml',
            'tier': SourceTier.TIER2_STANDARD,
            'region': Region.GLOBAL,
            'feed_type': FeedType.NEWSROOM,
        },
    }
    
    # UK Financial News
    UK_FEEDS = {
        'financial_times': {
            'url': 'https://www.ft.com/rss/home',
            'tier': SourceTier.TIER1_PREMIUM,
            'region': Region.UK,
            'feed_type': FeedType.NEWSROOM,
        },
        'bbc_business': {
            'url': 'http://feeds.bbci.co.uk/news/business/rss.xml',
            'tier': SourceTier.TIER2_STANDARD,
            'region': Region.UK,
            'feed_type': FeedType.NEWSROOM,
        },
        'guardian_business': {
            'url': 'https://www.theguardian.com/uk/business/rss',
            'tier': SourceTier.TIER2_STANDARD,
            'region': Region.UK,
            'feed_type': FeedType.NEWSROOM,
        },
    }
    
    # Official Sources
    OFFICIAL_FEEDS = {
        'fed_press': {
            'url': 'https://www.federalreserve.gov/feeds/press_all.xml',
            'tier': SourceTier.OFFICIAL,
            'region': Region.USA,
            'feed_type': FeedType.OFFICIAL,
        },
        'sec_news': {
            'url': 'https://www.sec.gov/news/pressreleases.rss',
            'tier': SourceTier.OFFICIAL,
            'region': Region.USA,
            'feed_type': FeedType.OFFICIAL,
        },
    }
    
    # Reddit (Social/Sentiment)
    REDDIT_FEEDS = {
        'wsb': {
            'url': 'https://www.reddit.com/r/wallstreetbets/.rss',
            'tier': SourceTier.SOCIAL,
            'region': Region.USA,
            'feed_type': FeedType.SOCIAL,
        },
        'stocks': {
            'url': 'https://www.reddit.com/r/stocks/.rss',
            'tier': SourceTier.SOCIAL,
            'region': Region.USA,
            'feed_type': FeedType.SOCIAL,
        },
        'investing': {
            'url': 'https://www.reddit.com/r/investing/.rss',
            'tier': SourceTier.SOCIAL,
            'region': Region.USA,
            'feed_type': FeedType.SOCIAL,
        },
    }
    
    # Stock-Specific Feed Templates
    STOCK_FEED_TEMPLATES = {
        'yahoo_finance': {
            'url_template': 'https://finance.yahoo.com/rss/headline?s={ticker}',
            'tier': SourceTier.TIER1_PREMIUM,
            'region': Region.GLOBAL,
            'feed_type': FeedType.STOCK_TICKER,
        },
        'google_news': {
            'url_template': 'https://news.google.com/rss/search?q={ticker}',
            'tier': SourceTier.TIER1_PREMIUM,
            'region': Region.GLOBAL,
            'feed_type': FeedType.STOCK_TICKER,
        },
        'seeking_alpha_news': {
            'url_template': 'https://seekingalpha.com/api/sa/combined/{ticker}.xml',
            'tier': SourceTier.TIER1_PREMIUM,
            'region': Region.USA,
            'feed_type': FeedType.STOCK_TICKER,
        },
    }
    
    # Company Name Feed Templates
    COMPANY_FEED_TEMPLATES = {
        'google_news_company': {
            'url_template': 'https://news.google.com/rss/search?q={company_name}',
            'tier': SourceTier.TIER1_PREMIUM,
            'region': Region.GLOBAL,
            'feed_type': FeedType.COMPANY_QUERY,
        },
    }
    
    @classmethod
    def get_all_feeds(cls) -> Dict[str, Dict]:
        """Combine all feeds into one dictionary"""
        all_feeds = {}
        all_feeds.update(cls.USA_FEEDS)
        all_feeds.update(cls.SPAIN_FEEDS)
        all_feeds.update(cls.UK_FEEDS)
        all_feeds.update(cls.OFFICIAL_FEEDS)
        all_feeds.update(cls.REDDIT_FEEDS)
        return all_feeds
    
    @classmethod
    def get_feeds_by_region(cls, regions: List[Region]) -> Dict[str, Dict]:
        """Get feeds filtered by specific regions"""
        all_feeds = cls.get_all_feeds()
        return {
            name: config for name, config in all_feeds.items()
            if config['region'] in regions
        }
    
    @classmethod
    def get_feeds_by_tier(cls, tiers: List[SourceTier]) -> Dict[str, Dict]:
        """Get feeds filtered by source tier"""
        all_feeds = cls.get_all_feeds()
        return {
            name: config for name, config in all_feeds.items()
            if config['tier'] in tiers
        }
    
    @classmethod
    def generate_stock_feeds(cls, tickers: List[str]) -> Dict[str, Dict]:
        """
        Generate stock-specific feeds from ticker list
        
        Args:
            tickers: List of stock tickers (e.g., ['AAPL', 'MSFT', 'GOOGL'])
            
        Returns:
            Dictionary of feed configurations for each ticker
        """
        stock_feeds = {}
        
        for ticker in tickers:
            ticker_upper = ticker.upper()
            
            # Generate feeds for each template
            for template_name, template_config in cls.STOCK_FEED_TEMPLATES.items():
                feed_name = f"{template_name}_{ticker_upper}"
                stock_feeds[feed_name] = {
                    'url': template_config['url_template'].format(ticker=ticker_upper),
                    'tier': template_config['tier'],
                    'region': template_config['region'],
                    'feed_type': template_config['feed_type'],
                }
        
        return stock_feeds
    
    @classmethod
    def generate_company_feeds(cls, company_names: List[str]) -> Dict[str, Dict]:
        """
        Generate company-specific feeds from company name list
        
        Args:
            company_names: List of company names (e.g., ['Apple Inc', 'Microsoft', 'Tesla'])
            
        Returns:
            Dictionary of feed configurations for each company
        """
        company_feeds = {}
        
        for company_name in company_names:
            # URL encode the company name for use in URLs
            import urllib.parse
            encoded_name = urllib.parse.quote(company_name)
            # Create a safe key name (replace spaces with underscores, lowercase)
            safe_name = company_name.replace(' ', '_').lower()
            
            # Generate feeds for each template
            for template_name, template_config in cls.COMPANY_FEED_TEMPLATES.items():
                feed_name = f"{template_name}_{safe_name}"
                company_feeds[feed_name] = {
                    'url': template_config['url_template'].format(company_name=encoded_name),
                    'tier': template_config['tier'],
                    'region': template_config['region'],
                    'feed_type': template_config['feed_type'],
                }
        
        return company_feeds

class ArticleEnricher:
    """Enrich articles with additional metadata"""
    
    @staticmethod
    def extract_tags(article: Article) -> List[str]:
        """
        Extract tags/topics from article content
        Basic keyword extraction - could be enhanced with NLP
        """
        tags = []
        text = f"{article.title} {article.content}".lower()
        
        # Financial keywords
        keywords = {
            'earnings': ['earnings', 'quarterly results', 'financial results'],
            'merger': ['merger', 'acquisition', 'M&A', 'takeover'],
            'ipo': ['ipo', 'initial public offering', 'going public'],
            'dividend': ['dividend', 'shareholder return'],
            'regulation': ['regulation', 'regulatory', 'compliance'],
            'bankruptcy': ['bankruptcy', 'chapter 11', 'insolvency'],
            'guidance': ['guidance', 'forecast', 'outlook'],
            'fed': ['federal reserve', 'fed', 'interest rate', 'monetary policy'],
            'inflation': ['inflation', 'cpi', 'consumer price'],
            'recession': ['recession', 'economic downturn'],
        }
        
        for tag, terms in keywords.items():
            if any(term in text for term in terms):
                tags.append(tag)
        
        return tags
    
    @staticmethod
    def enrich(article: Article) -> Article:
        """Add all enrichments to article"""
        # Extract tickers
        text = f"{article.title} {article.content}"
        article.mentioned_tickers = TickerExtractor.extract_tickers(text)
        
        # Extract tags
        article.tags = ArticleEnricher.extract_tags(article)
        
        return article


class RSSFetcher:
    """Main RSS fetcher class"""
    
    def __init__(self, max_age_hours: int = 24, rate_limit_seconds: float = 1.0, 
                 tickers: Optional[List[str]] = None, company_names: Optional[List[str]] = None):
        """
        Initialize RSS fetcher
        
        Args:
            max_age_hours: Only fetch articles newer than this
            rate_limit_seconds: Delay between feed requests
            tickers: Optional list of stock tickers to generate stock-specific feeds
            company_names: Optional list of company names to generate company-specific feeds
        """
        self.max_age_hours = max_age_hours
        self.rate_limit_seconds = rate_limit_seconds
        self.feed_config = RSSFeedConfig()
        self.enricher = ArticleEnricher()
        
        # Generate and store stock-specific feeds if tickers provided
        self.stock_feeds = {}
        if tickers:
            self.stock_feeds = RSSFeedConfig.generate_stock_feeds(tickers)
            logger.info(f"Generated {len(self.stock_feeds)} stock-specific feeds for {len(tickers)} tickers")
        
        # Generate and store company-specific feeds if company names provided
        self.company_feeds = {}
        if company_names:
            self.company_feeds = RSSFeedConfig.generate_company_feeds(company_names)
            logger.info(f"Generated {len(self.company_feeds)} company-specific feeds for {len(company_names)} companies")
    
    def fetch_feed(self, feed_url: str, source_name: str, 
                   source_tier: SourceTier, region: Region,
                   feed_type: FeedType) -> List[Article]:
        """
        Fetch and parse a single RSS feed
        
        Args:
            feed_url: RSS feed URL
            source_name: Name of the source/feed
            source_tier: Credibility tier
            region: Geographic region
            feed_type: Feed origin category
            
        Returns:
            List of Article objects
        """
        articles = []
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)
        
        try:
            logger.info(f"Fetching {source_name} ({feed_type.value}) from {region.value}...")
            
            # Parse feed with sanitization
            feed = feedparser.parse(feed_url, sanitize_html=True)
            
            # Check for parsing errors
            if feed.bozo:
                error_type = type(feed.bozo_exception).__name__
                logger.warning(f"  ⚠️  {source_name} has parsing issues ({error_type})")
                
                # If no entries were parsed, skip this feed
                if not feed.entries:
                    logger.error(f"  ❌ {source_name} returned no entries, skipping")
                    return articles
            
            # Process each entry
            for entry in feed.entries:
                try:
                    # Parse publication date
                    pub_date = self._parse_date(entry)
                    
                    # Filter by age
                    if pub_date and pub_date < cutoff_time:
                        continue
                    
                    # Extract content
                    content = self._extract_content(entry)
                    
                    # Create article
                    article = Article(
                        title=entry.get('title', 'Untitled'),
                        content=content,
                        url=entry.get('link', ''),
                        published_date=pub_date or datetime.now(),
                        fetched_date=datetime.now(),
                        source_name=source_name,
                        source_tier=source_tier.value,
                        region=region.value,
                        feed_name=source_name,
                        feed_type=feed_type.value,
                        author=entry.get('author', None),
                        summary=entry.get('summary', None),
                        guid=entry.get('id', entry.get('link', '')),
                    )
                    
                    # Enrich with metadata
                    article = self.enricher.enrich(article)
                    
                    articles.append(article)
                    
                except Exception as e:
                    logger.error(f"  ❌ Error processing entry: {e}")
                    continue
            
            logger.info(f"  ✓ {source_name}: {len(articles)} articles")
            
        except Exception as e:
            logger.error(f"  ❌ Error fetching {source_name}: {e}")
        
        return articles
    
    def _parse_date(self, entry) -> Optional[datetime]:
        """Parse publication date from entry"""
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            return datetime(*entry.published_parsed[:6])
        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            return datetime(*entry.updated_parsed[:6])
        return None
    
    def _extract_content(self, entry) -> str:
        """
        Extract the best available content from entry
        Priority: content > summary > description
        """
        # Try content field first (most detailed)
        if hasattr(entry, 'content') and entry.content:
            content = entry.content[0].get('value', '')
        # Fall back to summary
        elif hasattr(entry, 'summary'):
            content = entry.summary
        # Or description
        elif hasattr(entry, 'description'):
            content = entry.description
        else:
            content = entry.get('title', '')
        
        # Clean HTML tags
        content = self._clean_html(content)
        
        return content
    
    def _clean_html(self, html: str) -> str:
        """Remove HTML tags from text"""
        import html as html_lib
        # Remove HTML tags
        text = re.sub('<[^<]+?>', '', html)
        # Unescape HTML entities
        text = html_lib.unescape(text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def fetch_all(self, regions: Optional[List[Region]] = None,
                  tiers: Optional[List[SourceTier]] = None,
                  include_stock_feeds: bool = True,
                  include_company_feeds: bool = True) -> List[Article]:
        """
        Fetch articles from all configured feeds
        
        Args:
            regions: Optional list of regions to filter by
            tiers: Optional list of source tiers to filter by
            include_stock_feeds: Whether to include dynamically generated stock feeds
            include_company_feeds: Whether to include dynamically generated company feeds
            
        Returns:
            List of all articles
        """
        # Get feeds based on filters
        if regions:
            feeds = self.feed_config.get_feeds_by_region(regions)
        elif tiers:
            feeds = self.feed_config.get_feeds_by_tier(tiers)
        else:
            feeds = self.feed_config.get_all_feeds()
        
        # Add stock-specific feeds if requested
        dynamic_count = 0
        if include_stock_feeds and self.stock_feeds:
            feeds = {**feeds, **self.stock_feeds}
            dynamic_count += len(self.stock_feeds)
        
        # Add company-specific feeds if requested
        if include_company_feeds and self.company_feeds:
            feeds = {**feeds, **self.company_feeds}
            dynamic_count += len(self.company_feeds)
        
        if dynamic_count > 0:
            logger.info(f"Fetching from {len(feeds)} sources ({dynamic_count} dynamic feeds)...")
        else:
            logger.info(f"Fetching from {len(feeds)} sources...")
        
        all_articles = []
        
        for source_name, config in feeds.items():
            articles = self.fetch_feed(
                feed_url=config['url'],
                source_name=source_name,
                source_tier=config['tier'],
                region=config['region'],
                feed_type=config['feed_type']
            )
            all_articles.extend(articles)
            
            # Rate limiting
            time.sleep(self.rate_limit_seconds)
        
        logger.info(f"✅ Total articles fetched: {len(all_articles)}")
        
        return all_articles
    
    def deduplicate(self, articles: List[Article]) -> List[Article]:
        """Remove duplicate articles based on content hash"""
        seen_hashes = set()
        unique_articles = []
        
        for article in articles:
            if article.content_hash not in seen_hashes:
                seen_hashes.add(article.content_hash)
                unique_articles.append(article)
        
        removed = len(articles) - len(unique_articles)
        if removed > 0:
            logger.info(f"🔍 Removed {removed} duplicate articles")
        
        return unique_articles