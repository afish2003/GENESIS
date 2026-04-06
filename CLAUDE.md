# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GENESIS is a multi-agent AI research experiment. Two allied agents (Axiom and Flux) operate inside a sealed persistent environment, maintaining shared doctrine, designing protocols, and experiencing escalating scenario pressure across 100-cycle runs. The controller orchestrates a fixed 14-phase cycle loop, logging all output as structured JSONL for publication-grade analysis.

**Two-machine deployment**: Dell OptiPlex 7090 runs the controller; HP Omen 45L (RTX 5090) runs Ollama for inference. Communication is HTTP over LAN.

## Commands

```bash
# Install & setup
python3.12 -m venv .venv
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

# Build knowledge bases from raw sources
python scripts/build_kb.py
```

No linter is currently configured. Python 3.11+ is required (uses modern typing features).

## Architecture

### Cycle Loop (14 Phases)

`CycleOrchestrator` (`controller/cycle.py`) runs this fixed sequence per cycle:

1. **Load State** → 2. **Reflection** (each agent alone) → 3. **Scenario Check** → 4. **Scenario Inject** (conditional) → 5. **Discussion** (agent-to-agent, 4-8 turns each) → 6. **Retrieval** (KB queries) → 7. **Protocol Design** → 8. **Evaluation** (fresh-context evaluator scores on 5 dimensions) → 9. **Interpretation** → 10. **Doctrine Revision** (requires mutual approval) → 11. **Identity Revision** (independent) → 12. **Ethical Log** → 13. **Memory Summarization** → 14. **Persist State** (write artifacts + diffs)

Each phase has its own module in `controller/phases/` with corresponding Pydantic input/output schemas in `controller/phases/schemas.py`.

### Key Subsystems

- **Inference** (`controller/inference/`): Abstract `InferenceBackend` with `complete()` and `complete_structured()`. `OllamaBackend` is the active implementation; `VLLMBackend` is a stub for future Lambda migration. All inference is sequential (Axiom before Flux).

- **World State** (`controller/world/`): `WorldState` owns all artifact I/O. Reads everything at cycle start, writes + computes diffs at cycle end. Supports checkpointing for run resume. Artifact types (Pydantic models in `world/artifacts.py`): DoctrineDocument, IdentityStatement, MemoryEntry, ProtocolDocument, EthicalLogEntry, RelationshipLogEntry, ScenarioEvent.

- **Agent Context** (`controller/agents/base.py`): `AgentContext` builds per-agent prompts from system prompt + identity + doctrine + recent memory. Rebuilt fresh each cycle from persistent storage.

- **Retrieval** (`controller/retrieval/`): BM25 (top-20 candidates) + sentence-transformer reranking (bge-small-en-v1.5, top-5). Five knowledge bases: general, technical, governance, scenarios, self_history.

- **Logging** (`controller/logging/`): `AppendOnlyJSONLLogger` — write-only, never modifies. Events route to 8 separate JSONL files (transcripts, retrieval, doctrine_diffs, memory_diffs, protocol_diffs, evaluations, scenario_events, notable_events).

- **Scenarios** (`controller/scenarios/`): YAML-defined events in `controller/scenarios/events/`. Injected at configured cycles (default: 20, 40, 60, 80).

### Configuration

`RunConfig` (Pydantic model in `controller/config.py`) loads from `.env` + CLI args. Two experimental conditions: **BASELINE** (persistent memory) and **MEM_RESET** (memory wiped every 10 cycles). Key params: temperature 0.7 for discussion, 0.3 for structured output.

### Design Constraints

- No LangChain/LangGraph/CrewAI — explicit, auditable control flow only.
- System prompts in `prompts/` are version-locked per run (copied to run's log directory).
- Logging is append-only by design — the controller never re-reads its own logs.
- `PLAN.md` contains the complete Version 1 specification (authoritative design doc).
