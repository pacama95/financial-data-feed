"""Provider registry and factory utilities."""

from __future__ import annotations

from typing import Dict, Type, List

from .base_provider import BaseNewsProvider


class ProviderRegistry:
    """Simple registry/factory for provider implementations."""

    _providers: Dict[str, Type[BaseNewsProvider]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: Type[BaseNewsProvider]) -> None:
        """Register a provider class under a unique name."""
        if name in cls._providers:
            raise ValueError(f"Provider '{name}' already registered")
        cls._providers[name] = provider_cls

    @classmethod
    def override(cls, name: str, provider_cls: Type[BaseNewsProvider]) -> None:
        """Override an existing provider registration."""
        cls._providers[name] = provider_cls

    @classmethod
    def create(cls, name: str, config: Dict | None = None) -> BaseNewsProvider:
        """Instantiate a provider by name."""
        try:
            provider_cls = cls._providers[name]
        except KeyError as exc:
            raise ValueError(f"Unknown provider '{name}'. Registered: {cls.list_providers()}") from exc
        return provider_cls(config or {})

    @classmethod
    def list_providers(cls) -> List[str]:
        """Return registered provider names."""
        return sorted(cls._providers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear registry (useful for tests)."""
        cls._providers.clear()
