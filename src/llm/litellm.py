"""Unified LLM backend via the litellm library.

Supports any model string that litellm recognizes.  The provider is encoded
in the model-string prefix — no separate platform flag is needed:

    watsonx/meta-llama/llama-3-3-70b-instruct   → IBM WatsonX
    litellm_proxy/GCP/claude-4-sonnet            → LiteLLM proxy
    gemini/gemini-2.5-flash                      → Google Gemini

Credentials are resolved from environment variables based on the prefix:

    watsonx/*  :  WATSONX_APIKEY, WATSONX_PROJECT_ID, WATSONX_URL (optional)
    gemini/*   :  GEMINI_API_KEY
    otherwise  :  LITELLM_API_KEY, LITELLM_BASE_URL
"""

from __future__ import annotations

import os

from .base import LLMBackend


class LiteLLMBackend(LLMBackend):
    """LLM backend using the litellm library.

    Args:
        model_id: litellm model string with provider prefix, e.g.:
                  ``"watsonx/meta-llama/llama-3-3-70b-instruct"``
                  ``"litellm_proxy/GCP/claude-4-sonnet"``
    """

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._call_log: list[dict] = []

    @property
    def call_log(self) -> list[dict]:
        """Token usage log: list of {input_tokens, output_tokens, total_tokens} per call."""
        return self._call_log

    def reset_call_log(self) -> None:
        self._call_log.clear()

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        import litellm

        kwargs: dict = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 4096,
        }

        if self._model_id.startswith("gemini/"):
            kwargs["api_key"] = os.environ["GEMINI_API_KEY"]
        elif self._model_id.startswith("watsonx/"):
            kwargs["api_key"] = os.environ["WATSONX_APIKEY"]
            kwargs["project_id"] = os.environ["WATSONX_PROJECT_ID"]
            if url := os.environ.get("WATSONX_URL"):
                kwargs["api_base"] = url
        else:
            kwargs["api_key"] = os.environ["LITELLM_API_KEY"]
            kwargs["api_base"] = os.environ["LITELLM_BASE_URL"]

        response = litellm.completion(**kwargs)

        usage = response.usage
        self._call_log.append({
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "prompt_chars": len(prompt),
        })

        return response.choices[0].message.content
