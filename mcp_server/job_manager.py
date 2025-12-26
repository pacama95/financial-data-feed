"""Job management system for async pipeline execution."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

import redis
from pydantic import BaseModel

from shared.config import Config


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    job_id: str
    type: str  # "rss" or "api"
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobManager:
    """Manages async jobs using Redis for persistence."""
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            password=Config.REDIS_PASSWORD,
            decode_responses=True
        )
        self.job_ttl = 86400  # 24 hours
    
    def create_job(self, job_type: str, config: Dict[str, Any]) -> Job:
        """Create a new job."""
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,  # Use job_id instead of id
            type=job_type,
            status=JobStatus.PENDING,
            created_at=datetime.utcnow(),
            progress={"stage": "queued", "message": "Job queued for execution"}
        )
        
        # Store job in Redis
        self._store_job(job)
        
        # Store config separately
        self.redis_client.setex(
            f"job:{job_id}:config",
            self.job_ttl,
            json.dumps(config)
        )
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        data = self.redis_client.get(f"job:{job_id}")
        if not data:
            return None
        
        job_dict = json.loads(data)
        # Convert datetime strings back to datetime objects
        for field in ["created_at", "started_at", "completed_at"]:
            if job_dict.get(field):
                job_dict[field] = datetime.fromisoformat(job_dict[field])
        
        return Job(**job_dict)
    
    def update_job(self, job: Job):
        """Update job status and metadata."""
        self._store_job(job)
    
    def complete_job(self, job_id: str, result: Dict[str, Any]):
        """Mark job as completed with result."""
        job = self.get_job(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = result
            job.progress = {"stage": "completed", "message": "Job completed successfully"}
            self._store_job(job)
    
    def fail_job(self, job_id: str, error: str):
        """Mark job as failed with error message."""
        job = self.get_job(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error = error
            job.progress = {"stage": "failed", "message": f"Job failed: {error}"}
            self._store_job(job)
    
    def start_job(self, job_id: str):
        """Mark job as started."""
        job = self.get_job(job_id)
        if job:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()
            job.progress = {"stage": "running", "message": "Job is running"}
            self._store_job(job)
    
    def update_progress(self, job_id: str, stage: str, message: str, **kwargs):
        """Update job progress."""
        job = self.get_job(job_id)
        if job:
            job.progress = {"stage": stage, "message": message, **kwargs}
            self._store_job(job)
    
    def get_job_config(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job configuration."""
        data = self.redis_client.get(f"job:{job_id}:config")
        if not data:
            return None
        return json.loads(data)
    
    def _store_job(self, job: Job):
        """Store job in Redis with TTL."""
        job_dict = job.dict()
        # Convert datetime objects to ISO strings
        for field in ["created_at", "started_at", "completed_at"]:
            if job_dict.get(field):
                job_dict[field] = job_dict[field].isoformat()
        
        self.redis_client.setex(
            f"job:{job.job_id}",
            self.job_ttl,
            json.dumps(job_dict)
        )
    
    def cleanup_old_jobs(self, days: int = 7):
        """Clean up jobs older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        pattern = "job:*"
        
        for key in self.redis_client.scan_iter(match=pattern):
            if not key.endswith(":config"):
                data = self.redis_client.get(key)
                if data:
                    job_dict = json.loads(data)
                    created_at = datetime.fromisoformat(job_dict["created_at"])
                    if created_at < cutoff:
                        # Delete job and its config
                        job_id = key.split(":")[1]
                        self.redis_client.delete(key)
                        self.redis_client.delete(f"job:{job_id}:config")


# Global job manager instance
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get or create job manager singleton."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
