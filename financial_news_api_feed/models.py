"""
Common data models for financial news articles.

Provides a unified NewsArticle model that works across all providers.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib


@dataclass
class NewsArticle:
    """
    Universal news article model for all providers.
    
    This standardized model ensures consistent data structure regardless
    of the underlying news provider (Yahoo Finance, Alpha Vantage, etc.).
    """
    
    # Core content
    title: str
    content: str
    url: str
    
    # Temporal metadata
    published_date: datetime
    fetched_date: datetime
    
    # Source metadata
    source_name: str              # e.g., "Yahoo Finance", "Alpha Vantage"
    provider: str                 # e.g., "yahoo", "alphavantage"
    
    # Financial metadata
    tickers: List[str] = field(default_factory=list)           # Primary tickers
    related_tickers: List[str] = field(default_factory=list)   # Mentioned tickers
    
    # Additional metadata
    author: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None      # positive/negative/neutral
    
    # Identifiers
    article_id: str = ""
    content_hash: str = ""
    
    # Provider-specific metadata (flexible)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate IDs and validate data after initialization"""
        if not self.article_id:
            self.article_id = self._generate_article_id()
        if not self.content_hash:
            self.content_hash = self._generate_content_hash()
    
    def _generate_article_id(self) -> str:
        """Generate unique ID from URL and provider"""
        unique_string = f"{self.provider}:{self.url}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def _generate_content_hash(self) -> str:
        """Generate hash for deduplication based on content"""
        content = f"{self.title}{self.content}".lower().strip()
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert article to dictionary for storage.
        
        Returns:
            Dictionary with all article fields, datetime objects converted to ISO format
        """
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        data['published_date'] = self.published_date.isoformat()
        data['fetched_date'] = self.fetched_date.isoformat()
        return data
    
    def to_text_for_embedding(self) -> str:
        """
        Create optimized text representation for embedding generation.
        
        Combines title, summary, and content with proper formatting
        to create a rich text representation for semantic search.
        
        Returns:
            Formatted text string suitable for embedding
        """
        parts = [f"Title: {self.title}"]
        
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        
        parts.append(f"Content: {self.content}")
        
        # Add ticker context if available
        if self.tickers:
            parts.append(f"Tickers: {', '.join(self.tickers)}")
        
        return "\n\n".join(parts)
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"NewsArticle(title='{self.title[:50]}...', "
            f"provider='{self.provider}', "
            f"tickers={self.tickers}, "
            f"published={self.published_date.strftime('%Y-%m-%d %H:%M')})"
        )
