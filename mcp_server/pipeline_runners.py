"""Async pipeline runners for RSS and API pipelines."""

from __future__ import annotations

import asyncio
from typing import Dict, Any, Optional
import concurrent.futures

from shared.logging import get_logger
from shared.pipelines.rss import RSSPipeline
from shared.pipelines.api import APIPipeline
from .job_manager import get_job_manager, JobStatus

logger = get_logger("mcp_server.pipeline_runners")


class PipelineRunner:
    """Base class for running pipelines asynchronously."""
    
    def __init__(self):
        self.job_manager = get_job_manager()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
    async def run_pipeline(self, job_id: str, pipeline_type: str, config: Dict[str, Any]):
        """Run a pipeline in background thread."""
        try:
            # Mark job as started
            self.job_manager.start_job(job_id)
            
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_pipeline_sync,
                job_id,
                pipeline_type,
                config
            )
            
            # Mark job as completed
            self.job_manager.complete_job(job_id, result)
            
        except Exception as e:
            logger.error("Pipeline failed for job %s: %s", job_id, e)
            self.job_manager.fail_job(job_id, str(e))
    
    def _run_pipeline_sync(self, job_id: str, pipeline_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run pipeline synchronously in thread."""
        if pipeline_type == "rss":
            return self._run_rss_pipeline(job_id, config)
        elif pipeline_type == "api":
            return self._run_api_pipeline(job_id, config)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    def _run_rss_pipeline(self, job_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run RSS pipeline with given configuration."""
        self.job_manager.update_progress(
            job_id,
            "initializing",
            "Initializing RSS pipeline..."
        )
        
        pipeline = RSSPipeline(
            regions=config.get("regions"),
            tickers=config.get("stocks"),  # Map stocks to tickers
            company_names=config.get("company_names"),
            cleanup_days=config.get("cleanup_days", 30),
            max_age_hours=24,  # Default to 24 hours
        )
        
        self.job_manager.update_progress(
            job_id,
            "running",
            "Running RSS pipeline (fetch, process, store)..."
        )
        
        # Run the complete pipeline (fetch, process, store, cleanup)
        stats = pipeline.run(cleanup=True)
        
        return {
            "pipeline_type": "rss",
            "stats": stats,
            "config": {
                "regions": config.get("regions"),
                "stocks": config.get("stocks"),
                "company_names": config.get("company_names"),
                "cleanup_days": config.get("cleanup_days", 30)
            }
        }
    
    def _run_api_pipeline(self, job_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run API pipeline with given configuration."""
        self.job_manager.update_progress(
            job_id,
            "initializing",
            "Initializing API pipeline..."
        )
        
        pipeline = APIPipeline(
            tickers=config.get("tickers", []),
            cleanup_days=config.get("cleanup_days", 30),
            days_back=7,  # Default to 7 days
            max_articles_per_ticker=10,
        )
        
        self.job_manager.update_progress(
            job_id,
            "running",
            f"Running API pipeline for {len(config.get('tickers', []))} tickers..."
        )
        
        # Run the complete pipeline (fetch, process, store, cleanup)
        stats = pipeline.run(cleanup=True)
        
        return {
            "pipeline_type": "api",
            "stats": stats,
            "config": {
                "tickers": config.get("tickers", []),
                "cleanup_days": config.get("cleanup_days", 30)
            }
        }


# Global pipeline runner instance
_pipeline_runner: Optional[PipelineRunner] = None


def get_pipeline_runner() -> PipelineRunner:
    """Get or create pipeline runner singleton."""
    global _pipeline_runner
    if _pipeline_runner is None:
        _pipeline_runner = PipelineRunner()
    return _pipeline_runner
