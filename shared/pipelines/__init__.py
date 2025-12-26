"""Shared pipelines for financial news ingestion."""

from shared.pipelines.base import BasePipeline
from shared.pipelines.rss import RSSPipeline
from shared.pipelines.api import APIPipeline

__all__ = [
    "BasePipeline",
    "RSSPipeline",
    "APIPipeline",
]
