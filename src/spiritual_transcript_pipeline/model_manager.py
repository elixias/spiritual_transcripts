from __future__ import annotations

import os
import re
from typing import Any

from .config import StageLLMConfig


class ModelManager:
    """Return the correct LangChain chat model implementation from a model name."""

    def __init__(self, stage_cfg: StageLLMConfig) -> None:
        self.stage_cfg = stage_cfg

    def get_chat_model(self, *, temperature: float = 0.0) -> Any:
        model_name = (self.stage_cfg.model or "").strip()
        if not model_name:
            raise ValueError("Model name is not set for this stage.")

        provider = self._resolve_provider(model_name)
        if provider == "openai":
            return self._openai_chat(model_name, temperature=temperature)
        if provider == "anthropic":
            return self._anthropic_chat(model_name, temperature=temperature)
        if provider == "google":
            return self._google_chat(model_name, temperature=temperature)

        raise ValueError(
            f"Unsupported model/provider for '{model_name}'. "
            "Set SEGMENT_PROVIDER explicitly or use a supported model prefix."
        )

    def _resolve_provider(self, model_name: str) -> str:
        explicit = (self.stage_cfg.provider or "").strip().lower()
        if explicit:
            aliases = {
                "openai": "openai",
                "anthropic": "anthropic",
                "claude": "anthropic",
                "google": "google",
                "gemini": "google",
            }
            if explicit in aliases:
                return aliases[explicit]
            raise ValueError(f"Unsupported provider override: {self.stage_cfg.provider}")

        lower = model_name.lower()
        if lower.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
            return "openai"
        if lower.startswith("claude"):
            return "anthropic"
        if lower.startswith("gemini"):
            return "google"
        return "openai"

    def _openai_chat(self, model_name: str, *, temperature: float) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("langchain-openai is not installed.") from exc

        kwargs: dict[str, Any] = {"model": model_name, "temperature": temperature}
        api_key = self.stage_cfg.api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        if self._supports_openai_reasoning_controls(model_name):
            kwargs["reasoning"] = {"effort": "low", "summary": None}
            kwargs["output_version"] = "responses/v1"
        return ChatOpenAI(**kwargs)

    def _supports_openai_reasoning_controls(self, model_name: str) -> bool:
        lower = model_name.lower()
        if lower.startswith(("o1", "o3", "o4")):
            return True
        return re.match(r"^gpt-5(?:$|[.\-].*)", lower) is not None

    def _anthropic_chat(self, model_name: str, *, temperature: float) -> Any:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("langchain-anthropic is not installed.") from exc

        kwargs: dict[str, Any] = {"model": model_name, "temperature": temperature}
        api_key = self.stage_cfg.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            kwargs["anthropic_api_key"] = api_key
        return ChatAnthropic(**kwargs)

    def _google_chat(self, model_name: str, *, temperature: float) -> Any:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("langchain-google-genai is not installed.") from exc

        kwargs: dict[str, Any] = {"model": model_name, "temperature": temperature}
        api_key = self.stage_cfg.api_key or os.getenv("GOOGLE_API_KEY")
        if api_key:
            kwargs["google_api_key"] = api_key
        return ChatGoogleGenerativeAI(**kwargs)
