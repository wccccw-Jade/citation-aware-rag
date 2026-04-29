from src.config import Settings
from src.llm import create_llm_provider


def test_create_llm_provider_returns_none_when_disabled() -> None:
    settings = Settings(use_llm_generation=False, openai_api_key="test-key")

    assert create_llm_provider(settings) is None


def test_create_llm_provider_returns_none_without_openai_key() -> None:
    settings = Settings(use_llm_generation=True, llm_provider="openai", openai_api_key="")

    assert create_llm_provider(settings) is None

