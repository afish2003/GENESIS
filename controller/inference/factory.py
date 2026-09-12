"""Backend selection — one place that turns config into an InferenceBackend."""

from __future__ import annotations

import logging

from controller.config import Backend, RunConfig
from controller.inference.backend import InferenceBackend

logger = logging.getLogger(__name__)


def create_backend(config: RunConfig) -> InferenceBackend:
    """Construct the inference backend named by config.inference_backend."""
    kind = config.inference_backend

    if kind == Backend.OLLAMA:
        from controller.inference.ollama_backend import OllamaBackend

        logger.info("Inference: Ollama at %s (model=%s)", config.ollama_host, config.model_name)
        return OllamaBackend(
            host=config.ollama_host,
            model=config.model_name,
            timeout=config.request_timeout,
        )

    if kind == Backend.OPENAI:
        from controller.inference.openai_backend import OpenAICompatBackend

        logger.info(
            "Inference: OpenAI-compatible at %s (model=%s, key=%s)",
            config.api_base_url,
            config.model_name,
            "set" if config.api_key else "none",
        )
        return OpenAICompatBackend(
            base_url=config.api_base_url,
            model=config.model_name,
            api_key=config.api_key,
            timeout=config.request_timeout,
            json_mode=config.api_json_mode,
        )

    if kind == Backend.MOCK:
        from controller.inference.mock_backend import MockBackend

        logger.warning("Inference: MOCK backend — no model is being called")
        return MockBackend(model=config.model_name)

    raise ValueError(f"Unknown inference backend: {kind}")


def describe_backend(config: RunConfig) -> str:
    """One-line human-readable summary for the startup banner. Never logs the key."""
    kind = config.inference_backend
    if kind == Backend.OLLAMA:
        return f"ollama @ {config.ollama_host}"
    if kind == Backend.OPENAI:
        auth = "authenticated" if config.api_key else "no key"
        return f"openai-compat @ {config.api_base_url} ({auth})"
    return "mock (no model)"
