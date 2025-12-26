"""Concrete provider implementations for financial news APIs."""

from .yahoo_finance import YahooFinanceProvider

# Register all available providers
from ..provider_registry import ProviderRegistry

ProviderRegistry.register('yahoo_finance', YahooFinanceProvider)

__all__ = ['YahooFinanceProvider']
