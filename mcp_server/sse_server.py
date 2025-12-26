"""SSE-based MCP Server for Financial News RAG using FastAPI.

Run with: python -m mcp_server.sse_server
Access at: http://localhost:8000/sse
"""

from __future__ import annotations

import sys
import os
from contextlib import asynccontextmanager

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from mcp_server.models import RSSPipelineRequest, APIPipelineRequest, JobResponse
from mcp_server.job_manager import get_job_manager, JobStatus
from mcp_server.pipeline_runners import get_pipeline_runner
from mcp_server.mcp_tools import get_portfolio_news, get_market_insights, create_mcp_server


# =============================================================================
# Request Models
# =============================================================================

class PortfolioNewsRequest(BaseModel):
    """Request model for portfolio news endpoint."""
    tickers: list[str] = Field(..., description="List of ticker symbols", examples=[["AAPL", "NVDA", "MSFT"]])
    days_back: int = Field(default=7, description="Days to search back", ge=1, le=90)
    limit_per_ticker: int = Field(default=5, description="Max articles per ticker", ge=1, le=20)


class MarketInsightsRequest(BaseModel):
    """Request model for market insights endpoint."""
    tickers: Optional[list[str]] = Field(default=None, description="Optional tickers for context", examples=[["NVDA", "AMD"]])
    topics: Optional[list[str]] = Field(default=None, description="Topics to focus on", examples=[["AI", "semiconductors", "tariffs"]])
    days_back: int = Field(default=7, description="Days to search back", ge=1, le=90)
    limit: int = Field(default=15, description="Max total results", ge=1, le=50)


# Create MCP server
mcp_server = create_mcp_server()
session_manager = None  # Will be initialized in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Create session manager inside lifespan
    global session_manager
    session_manager = StreamableHTTPSessionManager(mcp_server)
    
    print("Financial News RAG MCP Server starting...")
    async with session_manager.run():
        yield
    print("Financial News RAG MCP Server shutting down...")


app = FastAPI(
    title="Financial News RAG MCP Server",
    description="MCP server exposing portfolio news and market insights tools",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({
        "status": "ok",
        "server": "financial-news-rag",
        "tools": ["get_portfolio_news", "get_market_insights"],
    })


@app.get("/mcp")
@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """MCP endpoint for streamable HTTP transport."""
    await session_manager.handle_request(
        request.scope, request.receive, request._send
    )


@app.post("/api/pipelines/rss/run", tags=["Pipeline Management"], response_model=JobResponse)
async def run_rss_pipeline(request: RSSPipelineRequest):
    """
    Run RSS pipeline with specified configuration.
    
    The pipeline will fetch news from RSS feeds based on the provided filters
    and store them in the vector database. This is an async operation that
    returns a job ID for tracking.
    """
    job_manager = get_job_manager()
    pipeline_runner = get_pipeline_runner()
    
    # Create job
    job = job_manager.create_job("rss", request.dict())
    
    # Start pipeline in background
    import asyncio
    asyncio.create_task(
        pipeline_runner.run_pipeline(job.job_id, "rss", request.dict())
    )
    
    return JobResponse(
        job_id=job.job_id,
        type=job.type,
        status=job.status,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        progress=job.progress,
        result=job.result,
        error=job.error
    )


@app.post("/api/pipelines/api/run", tags=["Pipeline Management"], response_model=JobResponse)
async def run_api_pipeline(request: APIPipelineRequest):
    """
    Run API pipeline with specified configuration.
    
    The pipeline will fetch news from API providers (Yahoo Finance, etc.)
    for the specified tickers and store them in the vector database.
    This is an async operation that returns a job ID for tracking.
    """
    job_manager = get_job_manager()
    pipeline_runner = get_pipeline_runner()
    
    # Create job
    job = job_manager.create_job("api", request.dict())
    
    # Start pipeline in background
    import asyncio
    asyncio.create_task(
        pipeline_runner.run_pipeline(job.job_id, "api", request.dict())
    )
    
    return JobResponse(
        job_id=job.job_id,
        type=job.type,
        status=job.status,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        progress=job.progress,
        result=job.result,
        error=job.error
    )


@app.get("/api/jobs/{job_id}", tags=["Pipeline Management"], response_model=JobResponse)
async def get_job_status(job_id: str):
    """
    Get the status and details of a pipeline job.
    
    Returns the current status, progress, and results if completed.
    """
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(
        job_id=job.job_id,
        type=job.type,
        status=job.status,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        progress=job.progress,
        result=job.result,
        error=job.error
    )


@app.get("/api/jobs", tags=["Pipeline Management"])
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    type: Optional[str] = Query(None, description="Filter by job type (rss or api)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip")
):
    """
    List pipeline jobs with optional filtering.
    
    Returns a paginated list of jobs with their current status.
    """
    # Note: This is a simplified implementation
    # In production, you might want to add proper pagination to Redis
    job_manager = get_job_manager()
    
    # For now, return empty list - implement based on your Redis structure
    return {
        "jobs": [],
        "total": 0,
        "limit": limit,
        "offset": offset
    }


# =============================================================================
# Direct REST API Endpoints
# =============================================================================

@app.post("/api/portfolio-news", tags=["REST API"])
async def api_portfolio_news(request: PortfolioNewsRequest):
    """
    Get news strictly related to portfolio tickers.
    
    Uses strict keyword matching - only returns articles that explicitly
    mention the ticker symbols. Best for monitoring specific holdings.
    """
    return get_portfolio_news(
        tickers=request.tickers,
        days_back=request.days_back,
        limit_per_ticker=request.limit_per_ticker,
    )


@app.post("/api/market-insights", tags=["REST API"])
async def api_market_insights(request: MarketInsightsRequest):
    """
    Get broader market insights, sector news, risks and opportunities.
    
    Uses semantic search - finds related news even if it doesn't explicitly
    mention the tickers. Best for discovering sector trends and risks.
    """
    return get_market_insights(
        tickers=request.tickers,
        topics=request.topics,
        days_back=request.days_back,
        limit=request.limit,
    )


def run_sse_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the MCP server with SSE transport."""
    import uvicorn
    
    print(f"Starting SSE MCP server at http://{host}:{port}")
    print(f"SSE endpoint: http://{host}:{port}/sse")
    print(f"Health check: http://{host}:{port}/health")
    print(f"API docs: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Financial News RAG MCP server with SSE")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    args = parser.parse_args()
    
    port = args.port if args.port is not None else int(os.getenv("PORT", "8000"))
    
    run_sse_server(host=args.host, port=port)
