from __future__ import annotations

from typing import Protocol

from .config import Settings


class LLMProvider(Protocol):
    provider_name: str
    model_name: str

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate an answer from prompts."""


class OpenAILLMProvider:
    provider_name = "openai"

    def __init__(self, api_key: str, model_name: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI LLM generation.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional production dependency
            raise RuntimeError("openai is not installed. Install requirements-prod.txt first.") from exc

        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            instructions=system_prompt,
            input=user_prompt,
        )
        return response.output_text.strip()


def create_llm_provider(settings: Settings) -> LLMProvider | None:
    if not settings.use_llm_generation:
        return None

    provider = settings.llm_provider.lower().strip()
    if provider == "openai":
        if not settings.openai_api_key:
            return None
        return OpenAILLMProvider(settings.openai_api_key, settings.llm_model_name)

    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

