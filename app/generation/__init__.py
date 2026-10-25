from .base import GenerationError, GenerationProvider
from .ollama_provider import OllamaProvider


def get_provider(settings) -> GenerationProvider:
    """Factory — the only place that maps a config string to a concrete
    provider class. Adding a hosting-compatible provider later means
    adding one branch here, not touching any caller."""
    if settings.generation_provider == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_s=settings.generation_timeout_s,
            api_key=settings.generation_api_key,
        )
    raise ValueError(
        f"Unknown generation_provider {settings.generation_provider!r}. "
        f"Only 'ollama' is implemented in this milestone."
    )


__all__ = ["GenerationError", "GenerationProvider", "OllamaProvider", "get_provider"]
