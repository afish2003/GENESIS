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
            max_output_tokens=config.max_output_tokens,
            enable_thinking=config.enable_thinking,
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
            max_output_tokens=config.max_output_tokens,
            enable_thinking=config.enable_thinking,
        )

    if kind == Backend.MOCK:
        from controller.inference.mock_backend import MockBackend

        logger.warning("Inference: MOCK backend — no model is being called")
        return MockBackend(model=config.model_name)

    raise ValueError(f"Unknown inference backend: {kind}")


def create_evaluator_backend(config: RunConfig) -> InferenceBackend | None:
    """A separate backend for the evaluation phase, or None to reuse the agents'.

    Returning None rather than a copy is deliberate: the caller then passes the
    agents' own backend, and `config.json` records evaluator_model as null, so
    "the judge was the same model" stays visible in the run record instead of
    being hidden behind a duplicated object.
    """
    if not config.uses_independent_evaluator:
        return None

    model = config.evaluator_model or config.model_name
    base_url = config.evaluator_api_base_url or config.api_base_url

    if config.inference_backend == Backend.MOCK:
        from controller.inference.mock_backend import MockBackend

        return MockBackend(model=model)

    from controller.inference.openai_backend import OpenAICompatBackend

    logger.info(
        "Evaluation: independent judge %s @ %s — the agents are scored by a "
        "model that is not themselves", model, base_url,
    )
    return OpenAICompatBackend(
        base_url=base_url,
        model=model,
        api_key=config.evaluator_api_key or config.api_key,
        timeout=config.request_timeout,
        json_mode=config.api_json_mode,
        # These were omitted, so the judge silently used the constructor
        # defaults while the agents used the run's configured values. A run
        # that raises max_output_tokens raises it for the agents only, and the
        # judge starts truncating mid-JSON with nothing to say why.
        max_output_tokens=config.max_output_tokens,
        enable_thinking=config.enable_thinking,
    )


def describe_backend(config: RunConfig) -> str:
    """One-line human-readable summary for the startup banner. Never logs the key."""
    kind = config.inference_backend
    if kind == Backend.OLLAMA:
        return f"ollama @ {config.ollama_host}"
    if kind == Backend.OPENAI:
        auth = "authenticated" if config.api_key else "no key"
        return f"openai-compat @ {config.api_base_url} ({auth})"
    return "mock (no model)"
