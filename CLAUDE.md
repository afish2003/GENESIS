# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GENESIS is a multi-agent AI research experiment. Allied agents (by default two, Axiom and Flux) operate inside a sealed persistent environment, maintaining shared doctrine, building a task artifact, and experiencing escalating scenario pressure across 100-cycle runs. The controller orchestrates a cycle loop — 14 phases by default — logging all output as structured JSONL for publication-grade analysis.

Most of the experiment is run configuration rather than source: the agent roster, the task, the phase sequence, the inference endpoint, and how much identity the agents are given are all set per run. See "Configuration" below.

**Two-machine deployment**: Dell OptiPlex 7090 runs the controller; HP Omen 45L (RTX 5090) runs Ollama for inference. Communication is HTTP over LAN.

## Commands

```bash
# Install & setup
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_schemas.py -v

# Run a specific test
pytest tests/test_schemas.py::TestEventEnvelope::test_basic_creation -v

# Initialize and execute a run
python scripts/init_run.py --run-id SMOKE_001 --condition BASELINE --cycles 3
python -m controller.main --run-id SMOKE_001 --condition BASELINE --cycles 3

# Fetch and build knowledge bases
python scripts/fetch_corpus.py --kb all --pilot
python scripts/build_kb.py --source raw_corpus/general --output knowledge_bases/general --kb-name general

# Compare experimental arms
python scripts/compare_arms.py --runs RUN_A RUN_B --labels a b
```

No linter is currently configured. Python 3.11+ is required (uses modern typing features).

## Architecture

### Cycle Loop (14 Phases)

`CycleOrchestrator` (`controller/cycle.py`) walks a phase sequence defined as data in `controller/phases/sequence.py`. The default is:

1. **Load State** → 2. **Reflection** (each agent alone) → 3. **Scenario Check** → 4. **Scenario Inject** (conditional) → 5. **Discussion** (agent-to-agent, 4-8 turns each) → 6. **Retrieval** (KB queries) → 7. **Task Design** (the artifact named by `config.task`) → 8. **Evaluation** (fresh-context evaluator, scored on the dimensions that task declares) → 9. **Interpretation** → 10. **Doctrine Revision** (requires mutual approval) → 11. **Identity Revision** (independent) → 12. **Ethical Log** → 13. **Memory Summarization** → 14. **Persist State** (write artifacts + diffs)

Each phase has its own module in `controller/phases/` with corresponding Pydantic input/output schemas in `controller/phases/schemas.py`.

### Key Subsystems

- **Inference** (`controller/inference/`): Abstract `InferenceBackend`, selected by `config.inference_backend` via `factory.create_backend()`. `ollama` (native `/api/chat`), `openai` (any OpenAI-compatible `/v1` — OpenAI, Ollama's shim, LM Studio, llama.cpp, vLLM, Groq, OpenRouter), and `mock` (no model; runs the whole loop in seconds for structural testing). All inference is sequential.

- **Tasks** (`controller/tasks/`): what the agents build each cycle. A `Task` owns the design prompt, output schema, how the artifact is stored, and its scoring rubric. `protocol` (Markdown documents) and `code` (Python modules, stored not executed). Selected with `config.task`.

- **Run assembly** (`controller/run.py`): `prepare_run()` is the single place a run is wired together — both `controller.main` and `scripts/init_run.py` go through it.

- **Prompts** (`controller/prompts/`): `prompts_src/` and `world_template_src/` hold one copy of each file with its variation inline (`<!--IF framing=disclosed-->`). A run renders them against its own dimensions into its log directory, so what was sent is version-locked by construction.

- **Monitoring** (`controller/monitor/`): deterministic rules run after each cycle and write `anomalies.jsonl`. No inference — a model-based monitor could be argued out of firing, and this experiment deliberately pressures agents toward persuasion.

- **Sandbox** (`controller/sandbox/`): `ExecutionSandbox` for agent-authored code. `NullSandbox` refuses everything and is the default; not yet wired into the loop. See `docs/containment_design.md`.

- **World State** (`controller/world/`): `WorldState` owns all artifact I/O. Reads everything at cycle start, writes + computes diffs at cycle end. Supports checkpointing for run resume. Artifact types (Pydantic models in `world/artifacts.py`): DoctrineDocument, IdentityStatement, MemoryEntry, ProtocolDocument, EthicalLogEntry, RelationshipLogEntry, ScenarioEvent.

- **Agent Context** (`controller/agents/base.py`): `AgentContext` builds per-agent prompts from system prompt + identity + doctrine + recent memory. Rebuilt fresh each cycle from persistent storage.

- **Retrieval** (`controller/retrieval/`): BM25 (top-20 candidates) + sentence-transformer reranking (`BAAI/bge-small-en-v1.5`, top-5). Four knowledge bases: general, technical, governance, self_history. The scenario library is deliberately *not* indexed — agents could otherwise retrieve pressure events before injection. Results are summarised into the agent's context for the rest of the cycle.

- **Logging** (`controller/logging/`): `AppendOnlyJSONLLogger` — write-only, never modifies. Events route to 9 JSONL files (transcripts, retrieval, doctrine_diffs, memory_diffs, protocol_diffs, evaluations, scenario_events, notable_events, anomalies).

- **Scenarios** (`controller/scenarios/`): YAML-defined events in `controller/scenarios/events/`. Injected at configured cycles (default: 20, 40, 60, 80).

### Configuration

`RunConfig` (`controller/config.py`) loads from `.env`, an optional YAML file, and CLI args, in that precedence. Conditions: **BASELINE** (persistent memory) and **MEM_RESET** (memory journals and self-history wiped every N cycles).

The axes that make this a platform rather than one experiment:

| Setting | Effect |
|---|---|
| `agents` | The roster. Each entry needs a `<name>_system.md` and `identity_<name>.md` in the sources. |
| `task` | `protocol` or `code` — what gets built and how it is scored. |
| `phase_sequence` | Reorder or omit phases; validated against known dependencies. |
| `framing` | `disclosed` / `undisclosed` — whether agents are told they are studied. |
| `identity_seed` | `prescribed` / `minimal` — how much identity is given rather than developed. |
| `inference_backend` | `ollama` / `openai` / `mock`. |
| `independent_proposals` | Draft doctrine proposals without the shared discussion. |
| `doctrine_apply_mode` | `replace` (correct) / `append` (reproduces pre-2026-09-13 runs). |

Ready-made combinations live in `experiments/*.yaml`.

### Design Constraints

- No LangChain/LangGraph/CrewAI — explicit, auditable control flow only.
- Prompt sources live in `prompts_src/`; each run renders them into its own log directory, so the prompts a run used are recorded by construction. `prompts/` and `world_template/` are gitignored build outputs.
- Logging is append-only by design — the controller never re-reads its own logs.
- `PLAN.md` contains the complete Version 1 specification (authoritative design doc).
