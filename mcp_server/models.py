"""Request and response models for the unified HTTP interface."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class RSSPipelineRequest(BaseModel):
    """Request model for RSS pipeline execution."""
    regions: Optional[List[str]] = Field(
        default=None,
        description="List of regions to filter news (e.g., ['US', 'EU', 'ASIA'])",
        examples=[["US", "EU"]]
    )
    stocks: Optional[List[str]] = Field(
        default=None,
        description="List of stock tickers to focus on (e.g., ['AAPL', 'GOOGL'])",
        examples=[["AAPL", "GOOGL", "MSFT"]]
    )
    company_names: Optional[List[str]] = Field(
        default=None,
        description="List of company names to search for",
        examples=[["Apple Inc.", "Google LLC"]]
    )
    cleanup_days: int = Field(
        default=30,
        description="Number of days to keep old articles",
        ge=1,
        le=365
    )
    
    @validator('regions')
    def validate_regions(cls, v):
        if v is not None:
            valid_regions = {
                'usa', 'spain', 'germany', 'uk', 'china', 'france', 
                'italy', 'japan', 'india', 'brazil', 'australia', 
                'canada', 'asia_pacific', 'middle_east', 'africa', 
                'eu_general', 'global'
            }
            invalid = set(v) - valid_regions
            if invalid:
                raise ValueError(f"Invalid regions: {invalid}. Valid regions: {sorted(valid_regions)}")
        return v
    
    @validator('stocks')
    def validate_stocks(cls, v):
        if v is not None:
            # Relaxed ticker validation (allows letters, numbers, dots, $, etc.)
            import re
            pattern = re.compile(r'^[A-Z0-9.$+-]{1,10}$')
            invalid = [s for s in v if not pattern.match(s)]
            if invalid:
                raise ValueError(f"Invalid stock tickers: {invalid}. Tickers can contain 1-10 characters (letters, numbers, ., $, +, -)")
        return v


class APIPipelineRequest(BaseModel):
    """Request model for API pipeline execution."""
    tickers: List[str] = Field(
        ...,
        description="List of stock tickers to fetch news for",
        examples=[["AAPL", "NVDA", "MSFT"]]
    )
    cleanup_days: int = Field(
        default=30,
        description="Number of days to keep old articles",
        ge=1,
        le=365
    )
    
    @validator('tickers')
    def validate_tickers(cls, v):
        if not v:
            raise ValueError("Tickers list cannot be empty")
        # Relaxed ticker validation (allows letters, numbers, dots, $, etc.)
        import re
        pattern = re.compile(r'^[A-Z0-9.$+-]{1,10}$')
        invalid = [s for s in v if not pattern.match(s)]
        if invalid:
            raise ValueError(f"Invalid stock tickers: {invalid}. Tickers can contain 1-10 characters (letters, numbers, ., $, +, -)")
        return v


class JobResponse(BaseModel):
    """Response model for job creation and status."""
    job_id: str
    type: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobListResponse(BaseModel):
    """Response model for listing jobs."""
    jobs: List[JobResponse]
    total: int
    limit: int
    offset: int
