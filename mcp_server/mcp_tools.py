"""MCP tools for financial news search."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from mcp.server import Server
from mcp.types import Tool, TextContent
import json

from shared import SearchClient


def format_results(results: List[dict], max_text_length: int = 500) -> List[dict]:
    """Format search results for agent consumption."""
    formatted = []
    for r in results:
        metadata = r.get("metadata", {})
        text = r.get("text", "")
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        
        formatted.append({
            "score": round(r.get("score", 0), 4),
            "title": metadata.get("article_title", ""),
            "url": metadata.get("article_url", ""),
            "published_date": metadata.get("published_date", ""),
            "source": metadata.get("source_name", ""),
            "tickers": metadata.get("tickers", []),
            "text": text,
            "collection": r.get("collection"),
        })
    
    return formatted


def get_portfolio_news(
    tickers: List[str],
    days_back: int = 7,
    limit_per_ticker: int = 5,
) -> Dict[str, Any]:
    """Get news strictly related to portfolio tickers."""
    client = SearchClient()
    cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    portfolio_news = {}
    all_results = []
    
    for ticker in tickers:
        results = client.search_all(
            query=f"{ticker} stock news",
            mode="hybrid",
            published_after=cutoff_date,
            keywords=[ticker],
            keyword_boosts={ticker: 3.0},
            require_keyword_match=True,
            limit=limit_per_ticker,
            deduplicate=True,
        )
        
        formatted = format_results(results)
        portfolio_news[ticker] = {
            "count": len(formatted),
            "articles": formatted,
        }
        
        for article in formatted:
            article["matched_ticker"] = ticker
        all_results.extend(formatted)
    
    all_results.sort(
        key=lambda x: x.get("published_date", ""),
        reverse=True,
    )
    
    return {
        "portfolio": tickers,
        "days_back": days_back,
        "total_articles": len(all_results),
        "by_ticker": portfolio_news,
        "recent_headlines": [
            {
                "ticker": r.get("matched_ticker", ""),
                "title": r["title"],
                "date": r["published_date"],
            }
            for r in all_results[:10]
        ],
    }


def get_market_insights(
    tickers: Optional[List[str]] = None,
    topics: Optional[List[str]] = None,
    days_back: int = 7,
    limit: int = 15,
) -> Dict[str, Any]:
    """Get broader market insights, sector news, risks and opportunities."""
    client = SearchClient()
    cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    query_parts = []
    if tickers:
        query_parts.extend(tickers)
    if topics:
        query_parts.extend(topics)
    
    if not query_parts:
        query_parts = ["market", "stocks", "financial news"]
    
    query = " ".join(query_parts) + " sector industry market trends risks opportunities"
    
    keywords = []
    keyword_boosts = {}
    
    if tickers:
        keywords.extend(tickers)
        for ticker in tickers:
            keyword_boosts[ticker] = 1.5
    
    if topics:
        keywords.extend(topics)
        for topic in topics:
            keyword_boosts[topic] = 2.0
    
    results = client.search_all(
        query=query,
        mode="hybrid" if keywords else "simple",
        published_after=cutoff_date,
        keywords=keywords if keywords else None,
        keyword_boosts=keyword_boosts if keyword_boosts else None,
        require_keyword_match=False,
        limit=limit,
        deduplicate=True,
    )
    
    formatted = format_results(results, max_text_length=400)
    
    categories = {
        "risks": [],
        "opportunities": [],
        "sector_news": [],
        "earnings": [],
        "general": [],
    }
    
    risk_keywords = ["risk", "warning", "decline", "fall", "drop", "concern", "threat", "tariff", "regulation", "lawsuit"]
    opportunity_keywords = ["opportunity", "growth", "rise", "surge", "bullish", "upgrade", "expansion", "partnership", "deal"]
    earnings_keywords = ["earnings", "revenue", "profit", "quarter", "q1", "q2", "q3", "q4", "guidance", "forecast"]
    sector_keywords = ["sector", "industry", "market", "competition", "competitor", "trend"]
    
    for article in formatted:
        text_lower = (article.get("text", "") + article.get("title", "")).lower()
        
        if any(kw in text_lower for kw in risk_keywords):
            categories["risks"].append(article)
        elif any(kw in text_lower for kw in opportunity_keywords):
            categories["opportunities"].append(article)
        elif any(kw in text_lower for kw in earnings_keywords):
            categories["earnings"].append(article)
        elif any(kw in text_lower for kw in sector_keywords):
            categories["sector_news"].append(article)
        else:
            categories["general"].append(article)
    
    return {
        "context": {
            "tickers": tickers or [],
            "topics": topics or [],
            "days_back": days_back,
        },
        "total_articles": len(formatted),
        "categories": {
            k: {"count": len(v), "articles": v}
            for k, v in categories.items()
            if v
        },
        "all_results": formatted,
    }


def create_mcp_server() -> Server:
    """Create and configure the MCP server with search tools."""
    server = Server("financial-news-rag")
    
    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="get_portfolio_news",
                description=(
                    "Get news strictly related to your portfolio tickers. "
                    "Uses strict keyword matching - only returns articles that explicitly "
                    "mention the ticker symbols. Best for monitoring specific holdings. "
                    "Example: Get news for AAPL, NVDA, MSFT positions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of ticker symbols in your portfolio (e.g., ['AAPL', 'NVDA', 'MSFT'])",
                        },
                        "days_back": {
                            "type": "integer",
                            "default": 7,
                            "description": "How many days back to search (default: 7)",
                        },
                        "limit_per_ticker": {
                            "type": "integer",
                            "default": 5,
                            "description": "Maximum articles per ticker (default: 5)",
                        },
                    },
                    "required": ["tickers"],
                },
            ),
            Tool(
                name="get_market_insights",
                description=(
                    "Get broader market insights, sector news, risks and opportunities. "
                    "Uses semantic search - finds related news even if it doesn't explicitly "
                    "mention the tickers. Best for discovering sector trends, competitor moves, "
                    "and market-wide risks/opportunities. "
                    "Example: Find AI sector risks for NVDA and AMD holdings."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tickers for context (e.g., ['NVDA', 'AMD'])",
                        },
                        "topics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Topics to focus on (e.g., ['AI', 'semiconductors', 'tariffs', 'regulation'])",
                        },
                        "days_back": {
                            "type": "integer",
                            "default": 7,
                            "description": "How many days back to search (default: 7)",
                        },
                        "limit": {
                            "type": "integer",
                            "default": 15,
                            "description": "Maximum total results (default: 15)",
                        },
                    },
                    "required": [],
                },
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]):
        try:
            if name == "get_portfolio_news":
                result = get_portfolio_news(
                    tickers=arguments["tickers"],
                    days_back=arguments.get("days_back", 7),
                    limit_per_ticker=arguments.get("limit_per_ticker", 5),
                )
            elif name == "get_market_insights":
                result = get_market_insights(
                    tickers=arguments.get("tickers"),
                    topics=arguments.get("topics"),
                    days_back=arguments.get("days_back", 7),
                    limit=arguments.get("limit", 15),
                )
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    return server
