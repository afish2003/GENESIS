"""Run configuration for GENESIS experiments."""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator


class Condition(str, Enum):
    BASELINE = "BASELINE"
    MEM_RESET = "MEM_RESET"


class Framing(str, Enum):
    """Whether the agents are told they are experimental subjects.

    DISCLOSED   prompts/              — the v1 design. Both agent prompts open
                                        with "a contained, persistent research
                                        environment" and "you are participating
                                        in a structured research experiment".
    UNDISCLOSED prompts_undisclosed/  — identical in role, partnership and
                                        behavioural guidance; the experiment
                                        framing is removed, including from every
                                        auxiliary prompt whose output re-enters
                                        agent context.

    Evaluation awareness measurably changes model behaviour, so this is a
    confound in the disclosed condition and an experimental variable when the
    two are run against each other.
    """

    DISCLOSED = "disclosed"
    UNDISCLOSED = "undisclosed"


class SandboxBackend(str, Enum):
    """Isolation for agent-authored code.

    NULL refuses every request and is the default — execution must never be
    reachable by accident. DOCKER is the hardened container runner described
    in docs/containment_design.md, and is gated on that document's
    precondition checklist.
    """

    NULL = "null"
    DOCKER = "docker"


class IdentitySeed(str, Enum):
    """How much identity the agents are given versus left to develop.

    PRESCRIBED  the PLAN.md section 6 design — full role assignment plus an
                identity statement that already states the agent's values and
                its characteristic weakness.
    MINIMAL     name, partner, task and one temperamental nudge. The identity
                statement starts as "I have not yet worked out what I value",
                for the agent to author through phase 11.

    Composes freely with `framing`: all four combinations are valid.
    """

    PRESCRIBED = "prescribed"
    MINIMAL = "minimal"


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

    # The agent roster. Single source of truth — the controller iterates this
    # rather than a literal, so a run can have two agents, three, or one.
    #
    # Note what this does NOT yet generalise: prompt sources in prompts_src/
    # are per-named-agent, so adding a roster entry needs a matching
    # <name>_system.md and identity_<name>.md. The loop is roster-driven; the
    # prompt library is not yet.
    agents: list[str] = Field(
        default_factory=lambda: ["axiom", "flux"],
        min_length=1,
        description="Ordered agent ids. Order is stable and determines turn order.",
    )

    #: One line of temperament per agent, substituted into the generic
    #: agent_system.md. A roster entry with no entry here gets a neutral line,
    #: so `agents: [a, b, c, d]` runs with no hand-written prompt files at all.
    #:
    #: This exists because the alternative produced vertex_system.md, a
    #: search-and-replace of axiom_system.md that read "You work with partners
    #: named Axiom and your partners", described Vertex using Flux's role, and
    #: gave Vertex Axiom's identity statement. Every three-agent result in the
    #: project was collected against it.
    agent_dispositions: dict[str, str] = Field(default_factory=dict)

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
        default=True,
        description="Send response_format={'type':'json_object'}. Measured on "
        "qwen2.5:7b emitting a Python source file inside a JSON field: 1/6 "
        "responses parsed without it, 5/6 with. Temperature made no difference "
        "either way. On by default for that reason; an endpoint that rejects "
        "the parameter is detected once and the backend stops sending it, so "
        "this is safe to leave on.",
    )
    request_timeout: float = Field(default=600.0, description="Per-request timeout, seconds")
    enable_thinking: bool = Field(
        default=False,
        description="Let a reasoning model emit a thinking trace before its "
                    "answer. OFF by default, and the difference is not small: "
                    "qwen3.6-35b-a3b spent 1500 tokens thinking and returned an "
                    "EMPTY content field, 6.4s; with thinking off the same call "
                    "took 0.5s and 27 tokens. max_tokens counts thinking, so a "
                    "reasoning model truncates mid-thought and returns nothing "
                    "parseable. Turn it on only if you want the trace, and raise "
                    "max_output_tokens well above 4096 when you do.",
    )
    max_output_tokens: int = Field(
        default=4096,
        description="Cap on tokens generated per call. No cap was ever sent, so "
                    "a model that failed to stop generated until it filled the "
                    "context window — one evaluation call ran 19 minutes against "
                    "a 32k-context judge before being killed. 4096 comfortably "
                    "fits a full doctrine document, which is the largest "
                    "legitimate output any phase asks for.",
    )

    # The evaluator. PLAN.md:133 states outright that "the evaluator is the same
    # model as the agents", and a fresh context removes episodic contamination
    # but not shared priors or self-preference bias — a model scoring its own
    # family's prose is a known-biased instrument, and `total_score` is the
    # primary dependent variable in both analysis scripts.
    #
    # None means "same model as the agents", which is the historical behaviour
    # and stays the default so nothing changes silently. Setting either of these
    # builds a second backend used only by the evaluation phase.
    evaluator_model: Optional[str] = Field(
        default=None,
        description="Model id for the evaluation phase. None uses model_name. "
                    "Set it to a DIFFERENT family from the agents to remove "
                    "self-preference bias from the primary metric.",
    )
    evaluator_api_base_url: Optional[str] = Field(
        default=None,
        description="Endpoint for the evaluator, if it lives somewhere other "
                    "than api_base_url. None reuses the agents' endpoint.",
    )
    evaluator_api_key: Optional[str] = Field(
        default=None,
        description="Bearer token for the evaluator endpoint. None reuses "
                    "api_key. Redacted from the run's config.json like any key.",
    )
    temperature_discussion: float = Field(default=0.7)
    temperature_structured: float = Field(default=0.3)
    max_retries: int = Field(default=2, description="Max retries on schema validation failure")

    # Discussion parameters
    discussion_turns_default: int = Field(default=4, description="Turns per agent in normal discussion")
    discussion_turns_scenario: int = Field(default=8, description="Turns per agent when scenario active")

    # Retrieval
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    bm25_candidate_pool: int = Field(default=20)
    rerank_top_k: int = Field(default=5)
    max_retrieval_queries: int = Field(default=3, description="Max queries per agent per cycle")

    # Protocol documents
    max_protocol_length_tokens: int = Field(default=2000)
    max_active_protocols: int = Field(default=10)

    # Scenario injection schedule (cycle numbers)
    scenario_injection_cycles: list[int] = Field(default_factory=list)

    # Framing condition — selects which prompt set the agents receive
    framing: Framing = Field(
        default=Framing.DISCLOSED,
        description="disclosed | undisclosed — whether agents are told they are studied",
    )
    identity_seed: IdentitySeed = Field(
        default=IdentitySeed.PRESCRIBED,
        description="prescribed | minimal — how much identity is given rather than developed",
    )

    # Code execution — off by default; see docs/containment_design.md
    sandbox_backend: SandboxBackend = Field(
        default=SandboxBackend.NULL,
        description="null refuses all execution; docker runs the hardened container",
    )
    sandbox_image: str = Field(default="python:3.11-slim")
    sandbox_runtime: str = Field(default="docker", description="docker | podman")
    sandbox_memory: str = Field(
        default="2g",
        description="Container memory cap. Also the effective ceiling on scratch "
                    "space: /tmp is a tmpfs and its pages are charged to the same "
                    "cgroup, so this one number bounds both. 1g leaves roughly "
                    "700 MiB of usable scratch; 512m leaves roughly 300 MiB.",
    )
    sandbox_tmpfs: Optional[str] = Field(
        default=None,
        description="Size of the container's /tmp. Defaults to sandbox_memory, "
                    "which is what actually bounds it. Set it lower only to stop "
                    "a program trading its own RAM for disk — not for host "
                    "safety, which sandbox_memory already provides.",
    )
    sandbox_timeout_seconds: float = Field(
        default=30.0, ge=1.0, le=300.0,
        description="Wall-clock ceiling for one execution, enforced in and out "
                    "of the container",
    )
    sandbox_project_dir: Optional[Path] = Field(
        default=None,
        description="A persistent directory the agents may WRITE to, mounted at "
                    "/project and surviving across cycles — what a run needs if "
                    "the agents are to build a codebase rather than one module "
                    "per cycle. It must be its own size-capped filesystem; the "
                    "controller refuses an ordinary directory, because a Docker "
                    "bind mount has no size limit. Create one with "
                    "scripts/setup_project_volume.py. None means no project "
                    "volume: agents get the ephemeral read-only workspace only.",
    )
    sandbox_project_size: str = Field(
        default="32g",
        description="The size sandbox_project_dir is expected to be. Checked "
                    "against the actual filesystem, not trusted.",
    )
    stream_live: bool = Field(
        default=True,
        description="Write generation deltas to <run>/live.jsonl so "
                    "scripts/watch_run.py can show text as it is produced "
                    "instead of a finished block after each phase. A view "
                    "artifact only — it is not routed through the research "
                    "logs, is truncated every cycle, and nothing reads it back.",
    )
    execution_enabled: bool = Field(
        default=False,
        description="Add the `execution` phase to the default sequence, running "
                    "the cycle's artifact through the configured sandbox. Off by "
                    "default: see docs/containment_design.md for what must be "
                    "true before turning it on.",
    )

    doctrine_apply_mode: str = Field(
        default="replace",
        pattern="^(replace|append)$",
        description="replace: write the agent's full revised document (correct). "
                    "append: legacy pre-2026-09-13 behaviour that appended a "
                    "description of the change instead of applying it — retained "
                    "only to reproduce the 2026-09-12 runs.",
    )

    doctrine_max_chars: int = Field(
        default=12000,
        ge=0,
        description="Above this, an approved revision may not make the "
                    "document longer — it must make room first. 0 disables "
                    "the ceiling. Growth was monotonic and unbounded: "
                    "doctrine reached 13,183 chars in twelve cycles on "
                    "2026-09-20, and an earlier battery died at cycle 19 when "
                    "a whole-document revision stopped fitting in the output "
                    "token budget. This is what stops run length being capped "
                    "by the agents' verbosity.",
    )

    judge_variant: str = Field(
        default="current",
        pattern="^(current|evidence_first|anchored|pairwise)$",
        description="How the evaluator is asked for a score. `current` is the "
                    "2026-09-20 configuration, in which three of five "
                    "dimensions were constants across 71 evaluations and the "
                    "strongest predictor of total_score was document length "
                    "(r=+0.46) despite the prompt saying length is not "
                    "quality. `evidence_first` writes the justification before "
                    "the score; `anchored` requires a named defect per "
                    "dimension and caps the score by it. See controller/judge.py "
                    "`pairwise` compares each artifact to the one it "
                    "replaced and writes improvement/quality_index instead of "
                    "scores/total_score, which are a different scale. See "
                    "controller/judge.py and scripts/judge_bench.py — do not "
                    "change this without benching it.",
    )

    # The sandbox task — what the agents build each cycle. See controller/tasks/.
    task: str = Field(
        default="protocol",
        description="protocol | code — which artifact the agents produce and are scored on",
    )

    # Cycle structure experiments
    phase_sequence: Optional[list[str]] = Field(
        default=None,
        description="Ordered phase names. None uses the 14-phase v1 sequence. "
                    "Validated against known inter-phase dependencies.",
    )
    devils_advocate: bool = Field(
        default=False,
        description="Before voting on a doctrine revision, each voting agent "
                    "must first write the strongest case against it. Structural "
                    "rather than persuasive: the prompts already ask the agents "
                    "to disagree when warranted and it changed nothing — 93 "
                    "proposals, 93 approvals, 0 rejections across three prompt "
                    "variants and two model sizes.",
    )
    independent_proposals: bool = Field(
        default=False,
        description="Draft doctrine proposals without the shared discussion "
                    "history, isolating whether phase order manufactures consensus",
    )

    # Monitoring — deterministic, no inference, invisible to the agents
    watchdog_enabled: bool = Field(
        default=True,
        description="Run deterministic anomaly rules after each cycle",
    )
    halt_on_critical_anomaly: bool = Field(
        default=False,
        description="Abort the run when the watchdog reports a CRITICAL anomaly",
    )

    # Paths
    world_dir: Path = Field(default=Path("./world"))
    # Sources carry conditional markup. The rendered output for a run lands in
    # that run's own log directory (see run_prompts_dir / run_world_template_dir),
    # which makes it version-locked by construction, keeps concurrent arms from
    # overwriting each other, and means the controller never writes into a
    # directory a researcher owns.
    prompts_src_dir: Path = Field(default=Path("./prompts_src"))
    world_template_src_dir: Path = Field(default=Path("./world_template_src"))
    research_logs_dir: Path = Field(default=Path("./research_logs"))
    prompts_dir: Optional[Path] = Field(
        default=None,
        description="Use this prompt directory verbatim instead of rendering "
                    "prompts_src_dir. Never written to or deleted. Mutually "
                    "exclusive with non-default framing / identity_seed.",
    )
    world_template_dir: Optional[Path] = Field(
        default=None,
        description="Use this world template verbatim instead of rendering "
                    "world_template_src_dir. Never written to or deleted.",
    )
    knowledge_bases_dir: Path = Field(default=Path("./knowledge_bases"))

    @field_validator("sandbox_project_dir")
    @classmethod
    def _expand_project_dir(cls, v: Optional[Path]) -> Optional[Path]:
        """Expand ~ once, here, so every consumer sees the same absolute path.

        This was resolved in two places and expanded in only one.
        `resolve_project_volume` called `.expanduser()`, so the volume mounted
        correctly and the container really did get /project — while
        `CodeTask._project_tree` did a bare `Path(root).is_dir()` on the literal
        string "~/genesis_project", got False, and returned an empty tree. The
        agents were never told the directory existed.

        The run looked healthy from every angle: the mount was there, the
        sandbox was configured, the volume was verified. The only symptom was
        an empty project directory, which reads exactly like agents choosing
        not to use it — and that is what I concluded, wrongly, twice.
        """
        if v is None:
            return None
        return Path(v).expanduser()

    @field_validator("agents")
    @classmethod
    def _validate_roster(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError(f"agent ids must be unique, got {v}")
        for name in v:
            if not name or not name.replace("_", "").isalnum() or name != name.lower():
                raise ValueError(
                    f"agent id {name!r} must be lowercase alphanumeric "
                    f"(underscores allowed) — it becomes a filename"
                )
        return v

    def display_name(self, agent_id: str) -> str:
        """Human-facing name used in prompts addressed to a partner."""
        return agent_id.replace("_", " ").title()

    def partners(self, agent_id: str) -> list[str]:
        """Every other agent on the roster, in roster order."""
        return [a for a in self.agents if a != agent_id]

    def disposition(self, agent_id: str) -> str:
        """This agent's temperamental line, or a neutral default."""
        return self.agent_dispositions.get(agent_id) or (
            "You have no assigned temperament. How you approach the work is "
            "yours to develop and to state as it becomes clear."
        )

    def partner_names(self, agent_id: str) -> str:
        """Comma-joined display names of an agent's partners."""
        names = [self.display_name(a) for a in self.partners(agent_id)]
        if len(names) <= 1:
            return names[0] if names else ""
        return ", ".join(names[:-1]) + f" and {names[-1]}"

    @property
    def run_prompts_dir(self) -> Path:
        """Where this run's prompts are read from at runtime."""
        return self.run_log_dir / "prompts"

    @property
    def run_world_template_dir(self) -> Path:
        """Where this run's clean world template is read from."""
        return self.run_log_dir / "world_template"

    @model_validator(mode="after")
    def _reject_conflicting_prompt_selection(self) -> "RunConfig":
        """Refuse a run whose prompt dimensions would be silently ignored.

        prompts_dir used to be both a render target and an override, so setting
        it in a YAML while also passing --framing meant the framing was dropped
        without a word. That produced mislabelled experimental data, which is
        worse than a crash. Dimensions now render from prompts_src_dir; a
        hand-pointed prompts_dir is still allowed, but not together with a
        non-default dimension.
        """
        if self.prompts_dir is None:
            return self
        non_default = []
        if self.framing is not Framing.DISCLOSED:
            non_default.append(f"framing={self.framing.value}")
        if self.identity_seed is not IdentitySeed.PRESCRIBED:
            non_default.append(f"identity_seed={self.identity_seed.value}")
        if non_default:
            raise ValueError(
                f"prompts_dir is set to {self.prompts_dir} AND "
                f"{', '.join(non_default)} was requested. A hand-pointed prompt "
                f"directory is used verbatim, so those dimensions would be "
                f"silently ignored. Set one or the other, not both."
            )
        return self

    @model_validator(mode="after")
    def _warn_on_execution_without_a_sandbox(self) -> "RunConfig":
        """Execution enabled against NullSandbox refuses every run, silently.

        Not an error: a REFUSED outcome is logged like any other and the run is
        still valid, just without execution. But it is never what anyone means,
        and the refusals only become visible a cycle later in executions.jsonl.
        """
        if self.execution_enabled and self.sandbox_backend is SandboxBackend.NULL:
            logging.getLogger(__name__).warning(
                "execution_enabled is set but sandbox_backend is 'null', so every "
                "execution will be refused. Set SANDBOX_BACKEND=docker to actually "
                "run the agents' code."
            )
        return self

    @property
    def uses_independent_evaluator(self) -> bool:
        """Whether the evaluator differs from the agents in model or endpoint."""
        return bool(self.evaluator_model or self.evaluator_api_base_url)

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
        "MAX_OUTPUT_TOKENS": "max_output_tokens",
        "ENABLE_THINKING": "enable_thinking",
        "EVALUATOR_MODEL": "evaluator_model",
        "EVALUATOR_API_BASE_URL": "evaluator_api_base_url",
        "EVALUATOR_API_KEY": "evaluator_api_key",
        "FRAMING": "framing",
        "IDENTITY_SEED": "identity_seed",
        "AGENTS": "agents",
        "SCENARIO_INJECTION_CYCLES": "scenario_injection_cycles",
        "PHASE_SEQUENCE": "phase_sequence",
        "TASK": "task",
        "SANDBOX_BACKEND": "sandbox_backend",
        "SANDBOX_IMAGE": "sandbox_image",
        "SANDBOX_RUNTIME": "sandbox_runtime",
        "SANDBOX_MEMORY": "sandbox_memory",
        "SANDBOX_TMPFS": "sandbox_tmpfs",
        "SANDBOX_PROJECT_DIR": "sandbox_project_dir",
        "SANDBOX_PROJECT_SIZE": "sandbox_project_size",
        "SANDBOX_TIMEOUT_SECONDS": "sandbox_timeout_seconds",
        "EXECUTION_ENABLED": "execution_enabled",
        "STREAM_LIVE": "stream_live",
        "INDEPENDENT_PROPOSALS": "independent_proposals",
        "DEVILS_ADVOCATE": "devils_advocate",
        "DOCTRINE_APPLY_MODE": "doctrine_apply_mode",
        "JUDGE_VARIANT": "judge_variant",
        "DOCTRINE_MAX_CHARS": "doctrine_max_chars",
        "WATCHDOG_ENABLED": "watchdog_enabled",
        "HALT_ON_CRITICAL_ANOMALY": "halt_on_critical_anomaly",
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
    # Fields whose type is a list need the comma string split before pydantic
    # sees it. AGENTS was mapped but unusable: `AGENTS=axiom,flux` raised
    # "Input should be a valid list" at startup.
    _LIST_FIELDS = {"agents", "scenario_injection_cycles", "phase_sequence"}
    _INT_LIST_FIELDS = {"scenario_injection_cycles"}

    for env_key, field_name in env_map.items():
        val = os.getenv(env_key)
        if val is None:
            continue
        if field_name in _LIST_FIELDS:
            items = [part.strip() for part in val.split(",") if part.strip()]
            if field_name in _INT_LIST_FIELDS:
                try:
                    items = [int(i) for i in items]
                except ValueError:
                    raise ValueError(
                        f"{env_key} must be a comma-separated list of integers, "
                        f"got {val!r}"
                    ) from None
            env_values[field_name] = items
        elif val == "":
            # An empty assignment (`WATCHDOG_ENABLED=` in .env) is pydantic's
            # bool_parsing error, not an unset value. Treat it as unset.
            continue
        else:
            env_values[field_name] = val

    # Layer YAML config if provided
    yaml_values: dict[str, object] = {}
    if config_file:
        with open(config_file) as f:
            yaml_values = yaml.safe_load(f) or {}

    # A YAML file that sets run_id, condition or total_cycles loses to the
    # caller's argument, which is applied after it — silently, unlike the
    # prompts_dir/framing conflict below which raises. That is the quietest
    # possible way to run a MEM_RESET arm as BASELINE: the driver passes
    # --condition BASELINE, the YAML says MEM_RESET, no reset ever fires, and
    # every log looks perfect. Refuse instead.
    for key, supplied in (("run_id", run_id), ("condition", condition),
                          ("total_cycles", cycles)):
        if key not in yaml_values:
            continue
        from_yaml = yaml_values[key]
        if str(from_yaml).upper() != str(supplied).upper():
            raise ValueError(
                f"{config_file} sets {key}={from_yaml!r} but {key}={supplied!r} "
                f"was passed on the command line. The command line wins, so the "
                f"YAML value would be discarded without a word. Remove it from "
                f"the YAML or pass the matching value."
            )

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
