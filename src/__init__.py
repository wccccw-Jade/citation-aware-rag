"""Citation-aware RAG package."""

__all__ = ["CitationAwareRAG"]


def __getattr__(name: str):
    if name == "CitationAwareRAG":
        from .pipeline import CitationAwareRAG

        return CitationAwareRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
