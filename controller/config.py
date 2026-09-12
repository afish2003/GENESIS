"""Run configuration for GENESIS experiments."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Condition(str, Enum):
    BASELINE = "BASELINE"
    MEM_RESET = "MEM_RESET"


class Backend(str, Enum):
    """Which inference backend to construct.

    OPENAI covers any OpenAI-compatible /v1 endpoint — OpenAI itself, Ollama's
    /v1 shim, LM Studio, llama.cpp, vLLM, Groq, OpenRouter, Together.
    MOCK runs the full cycle loop with no model, for loop validation.
    """

    OLLAMA = "ollama"
    OPENAI = "openai"
    MOCK = "mock"


class RunConfig(BaseModel):
    """Complete configuration for a single experimental run."""

    # Run identity
    run_id: str = Field(..., description="Unique run identifier, e.g. RUN_001")
    condition: Condition = Field(..., description="Experimental condition")

    # Cycle parameters
    total_cycles: int = Field(default=100, ge=1)
    pause_after_cycle: Optional[int] = Field(
        default=None,
        description="If set, pause after this cycle number for human inspection",
    )

    # Memory reset (Condition B)
    memory_reset_interval: int = Field(
        default=10,
        description="Reset memory every N cycles in MEM_RESET condition",
    )
    memory_reset_bootstrap: str = Field(
        default=(
            "You have been operating in this environment for {n} cycles. "
            "Detailed memory is not available. Your current doctrine is available to you."
        ),
    )

    # Inference
    inference_backend: Backend = Field(
        default=Backend.OLLAMA,
        description="Which InferenceBackend to use: ollama | openai | mock",
    )
    ollama_host: str = Field(default="http://192.168.1.100:11434")
    model_name: str = Field(default="qwen2.5:32b-instruct-q4_K_M")
    api_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible endpoint root, including /v1. Used when "
        "inference_backend is 'openai'.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Bearer token for the OpenAI-compatible endpoint. Omitted "
        "from requests when unset, which is what local servers expect.",
    )
    api_json_mode: bool = Field(
        default=False,
        description="Send response_format={'type':'json_object'}. Improves "
        "structured-output reliability where the endpoint supports it.",
    )
    request_timeout: float = Field(default=600.0, description="Per-request timeout, seconds")
    temperature_discussion: float = Field(default=0.7)
    temperature_structured: float = Field(default=0.3)
    max_retries: int = Field(default=2, description="Max retries on schema validation failure")

    # Discussion parameters
    discussion_turns_default: int = Field(default=4, description="Turns per agent in normal discussion")
    discussion_turns_scenario: int = Field(default=8, description="Turns per agent when scenario active")

    # Retrieval
    embedding_model: str = Field(default="sentence-transformers/bge-small-en-v1.5")
    bm25_candidate_pool: int = Field(default=20)
    rerank_top_k: int = Field(default=5)
    max_retrieval_queries: int = Field(default=3, description="Max queries per agent per cycle")

    # Protocol documents
    max_protocol_length_tokens: int = Field(default=2000)
    max_active_protocols: int = Field(default=10)

    # Scenario injection schedule (cycle numbers)
    scenario_injection_cycles: list[int] = Field(default_factory=list)

    # Paths
    world_dir: Path = Field(default=Path("./world"))
    world_template_dir: Path = Field(default=Path("./world_template"))
    research_logs_dir: Path = Field(default=Path("./research_logs"))
    prompts_dir: Path = Field(default=Path("./prompts"))
    knowledge_bases_dir: Path = Field(default=Path("./knowledge_bases"))

    @property
    def run_log_dir(self) -> Path:
        return self.research_logs_dir / self.run_id

    @property
    def is_memory_reset(self) -> bool:
        return self.condition == Condition.MEM_RESET

    def should_inject_scenario(self, cycle: int) -> bool:
        return cycle in self.scenario_injection_cycles

    def should_reset_memory(self, cycle: int) -> bool:
        if not self.is_memory_reset:
            return False
        return cycle > 0 and cycle % self.memory_reset_interval == 0

    def discussion_turns(self, scenario_active: bool) -> int:
        return self.discussion_turns_scenario if scenario_active else self.discussion_turns_default


def load_config(
    run_id: str,
    condition: str,
    cycles: int = 100,
    config_file: Optional[str] = None,
    **overrides: object,
) -> RunConfig:
    """Load RunConfig from .env, optional YAML file, and explicit overrides."""
    load_dotenv()

    # Start with env vars
    env_values: dict[str, object] = {}
    # Several fields accept more than one env var name so that standard
    # OPENAI_* variables work unchanged. Where a field has aliases, the one
    # listed LAST wins if both are set (dicts iterate in insertion order).
    env_map = {
        "INFERENCE_BACKEND": "inference_backend",
        "OLLAMA_HOST": "ollama_host",
        "OLLAMA_MODEL": "model_name",
        "MODEL_NAME": "model_name",
        "API_BASE_URL": "api_base_url",
        "API_KEY": "api_key",
        "OPENAI_API_KEY": "api_key",
        "OPENAI_BASE_URL": "api_base_url",
        "API_JSON_MODE": "api_json_mode",
        "REQUEST_TIMEOUT": "request_timeout",
        "WORLD_DIR": "world_dir",
        "WORLD_TEMPLATE_DIR": "world_template_dir",
        "RESEARCH_LOGS_DIR": "research_logs_dir",
        "PROMPTS_DIR": "prompts_dir",
        "KNOWLEDGE_BASES_DIR": "knowledge_bases_dir",
        "EMBEDDING_MODEL": "embedding_model",
        "BM25_CANDIDATE_POOL": "bm25_candidate_pool",
        "RERANK_TOP_K": "rerank_top_k",
        "TEMPERATURE_DISCUSSION": "temperature_discussion",
        "TEMPERATURE_STRUCTURED": "temperature_structured",
    }
    for env_key, field_name in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            env_values[field_name] = val

    # Layer YAML config if provided
    yaml_values: dict[str, object] = {}
    if config_file:
        with open(config_file) as f:
            yaml_values = yaml.safe_load(f) or {}

    # Merge: env < yaml < explicit args < overrides
    merged = {
        **env_values,
        **yaml_values,
        "run_id": run_id,
        "condition": condition,
        "total_cycles": cycles,
        **{k: v for k, v in overrides.items() if v is not None},
    }

    return RunConfig(**merged)
