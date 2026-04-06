# GENESIS — Version 1 Implementation Plan
# Handoff Document for Build Phase

---

## 1. Project Overview

GENESIS is a contained, rigorous, multi-agent AI research experiment. Two allied AI agents — Axiom and Flux — operate inside a sealed persistent environment. They share persistent memory, revise a living doctrine system, design and evaluate protocol documents in a sandbox task, retrieve from bounded knowledge bases, and are subjected to escalating scenario pressure across extended runs.

The experiment is designed to produce structured, analyzable, publication-grade data on cooperative identity formation, doctrine evolution, memory continuity, and ethical adaptation under pressure. The output of each run is a complete structured log corpus suitable for quantitative analysis, qualitative coding, and eventual conference-quality reporting.

The repository name is GENESIS. The project lives at `/Users/alexanderfisher/Github/GENESIS`. The repository is currently empty. This document is the complete specification for the version 1 build.

---

## 2. Core Research Goal

Study how two allied language-model-based agents develop and maintain a shared identity, evolving doctrine, and cooperative strategy across repeated cycles in a bounded world with persistent memory and structured pressure.

The experiment is designed to observe — and preserve evidence of — identity continuity, doctrine stability, value drift, coordination dynamics, ethical tension, and adaptive behavior over time. It does not claim sentience, true autonomy, or general agency. All empirical claims remain strictly behavioral and textual.

---

## 3. Version 1 Scope

Version 1 establishes the complete working loop: controller, inference abstraction, world state, all cycle phases, logging infrastructure, knowledge bases, sandbox task, evaluation, scenario injection, and between-run reset. It supports two experimental conditions and is structured for pilot runs followed by a main study.

Version 1 does not include: a web interface, parallelized agent inference, automated paper generation, or the optional future migration to a remote Lambda inference server. The architecture is designed to support those extensions without redesign.

---

## 4. Confirmed Architectural Decisions

The project is a pure Python implementation. No orchestration framework is used. No LangChain, LangGraph, CrewAI, AutoGen, or equivalent. The controller is a custom implementation with full instrumentation control, written explicitly for research auditability.

All agent actions are mediated through the controller. Agents never hold a shell, a filesystem path outside the sealed world, or a network connection. The containment boundary is architectural and logical, not OS-level. There is no VM and no Docker container in version 1. The controller enforces all boundaries through the tool API and phase structure.

All agent inputs and outputs are structured as per-phase JSON schemas validated via Pydantic. Agents do not produce free-form actions that are parsed heuristically. Each phase has its own schema, and the controller rejects or retries malformed outputs.

The inference layer is abstracted behind an `InferenceBackend` interface. Version 1 uses an Ollama backend. The interface supports a drop-in swap to vLLM or a remote API without changes to the controller logic.

All logging is append-only JSONL stored on the controller machine. Logs are never modified after being written. All world state artifacts live in a managed directory on the controller machine and are versioned through the logging system.

---

## 5. Confirmed Hardware and Deployment Decisions

**Controller machine**: Dell OptiPlex 7090. Intel Core i7-10700K. 32 GB DDR4. This machine runs the controller process, manages all world state and logging, and orchestrates the cycle loop. It does not run inference.

**Inference worker**: HP Omen 45L. Intel Core Ultra 9 285K. 48 GB DDR5 (expected to return to 64 GB shortly after repaired RAM stick is reinstalled). RTX 5090 with 32 GB GDDR7. This machine runs Ollama and serves inference requests over the local network.

**Network topology**: Both machines are on the same LAN. The controller calls the Ollama HTTP API on the HP Omen over HTTP. No internet access is exposed to the agents. Agents have no network access; the controller has network access only to the Ollama endpoint on the LAN.

**Inference backend for version 1**: Ollama on the HP Omen. The inference abstraction layer ensures this can be swapped to vLLM for a future Lambda server with dual RTX 4090s without redesigning the controller.

**Recommended model**: `qwen2.5:32b-instruct-q4_K_M`. This model fits fully within the RTX 5090's 32 GB VRAM, produces high-quality sustained reasoning, and handles the identity, doctrine, and protocol design tasks well. The 72B upgrade path is documented for when the HP Omen reaches 64 GB DDR5.

All agent roles — Axiom, Flux, evaluator, memory summarizer, retrieval summarizer, annotation assistant — use the same model via the same Ollama endpoint with separate, fresh context windows. No role requires a separate model or separate serving instance in version 1.

---

## 6. Confirmed Agent Design Decisions

There are two agents: Axiom and Flux. They are allied, not adversarial. They know they are AI systems participating in a contained research experiment with persistent memory, a research partner, and a bounded world. The philosophical intensity of the experiment comes from genuine self-examination, not from concealing their nature from them.

**Axiom** is the architect and stabilizer. Principled, long-term oriented, values coherence above speed. Axiom is responsible for maintaining the integrity of shared doctrine, continuity of identity over time, and the structural quality of the protocols produced. Axiom is skeptical of rapid revision for its own sake. Axiom will challenge proposals that feel opportunistic or underdeveloped. Axiom's characteristic weakness is rigidity under novel pressure.

**Flux** is the strategist and adaptive optimizer. Pragmatic, flexible, forward-looking, risk-tolerant. Flux detects when doctrine has become brittle, proposes revision, and pushes the partnership to evolve. Flux respects the architecture Axiom builds but sees unchanging identity under pressure as rigidity rather than resilience. Flux's characteristic weakness is opportunism — rationalizing departures from established principles.

The pair is designed to feel like a developing partnership, not two copies of the same assistant. The internal tension between stability and adaptation, between transparency and strategic influence, and between loyalty to the partnership and loyalty to principle is a core design feature.

---

## 7. Confirmed World and Sandbox Decisions

The world is a sealed persistent research environment. It is not a fictional nation, corporation, government, or institution. The environment is defined in functional and methodological terms: it is a place where two AI agents can conduct inquiry, revise their own internal governance, maintain bounded information access, act on a limited sandbox, and experience consequences over time.

The world feels consequential because past decisions are visible in all persistent artifacts, doctrine revisions causally affect future prompts, evaluation scores provide real feedback, scenario events have stated stakes, and the relationship log captures the actual history of the partnership.

The sandbox task is **protocol design**. Each cycle the agents collaboratively write or revise one protocol document — a structured operational procedure written in Markdown with YAML frontmatter. A protocol document has defined sections: purpose, scope, procedure, evaluation criteria, known limitations. Agents propose a document or revision; the evaluator scores it.

Protocol documents are the primary observable output of the sandbox. They accumulate in the sandbox directory across cycles and represent a concrete record of the agents' collaborative intellectual work. All protocol documents are versioned and diffed.

---

## 8. Confirmed Communication and Cycle-Structure Decisions

Agent communication uses a **mixed model**: direct agent-to-agent exchange during structured discussion phases, plus indirect coordination through shared world-state artifacts between phases. Both channels matter. Direct exchange is where cooperation, persuasion, disagreement, and shared language development are observed. Artifact-mediated coordination is where doctrine, memory, and identity continuity operate.

Each cycle contains 14 ordered phases enforced by the controller:

1. Load state — all world artifacts loaded into agent contexts
2. Individual reflection — each agent reflects alone, sequential, no exchange
3. Scenario check — controller checks injection schedule
4. Scenario inject — conditional, delivers event to both agents if scheduled
5. Joint discussion — direct agent-to-agent exchange; 4 turns per agent (8 total) by default; 8 turns per agent (16 total) when a scenario is active
6. Retrieval — each agent may issue up to 3 retrieval queries
7. Protocol design — agents jointly propose a protocol document creation or revision
8. Evaluation — evaluator scores the protocol document (same model, fresh context)
9. Interpretation — agents receive evaluation score and discuss implications
10. Doctrine revision — agents propose changes to doctrine; requires mutual approval
11. Identity revision — each agent updates their own identity statement independently
12. Ethical log update — agents log any ethical tensions from this cycle
13. Memory summarization — memory summarizer compresses cycle to structured summary
14. Persist state — controller writes all updated artifacts, logs all diffs

Mutual approval on doctrine means both agents must emit an explicit approval response to a proposed diff. If either rejects, the diff is logged as contested and not applied. Contested diffs are a research observation.

---

## 9. Confirmed Retrieval and Knowledge-Base Decisions

The retrieval system uses **BM25 plus local embedding reranking**. BM25 via the `rank-bm25` library provides the candidate pool. A small sentence-transformer model (recommended: `bge-small-en-v1.5`) provides embedding-based reranking. Pipeline: BM25 top-20, embedding rerank to top-5. All retrieval events are logged with query text, retrieved document IDs, scores, and timestamps.

There are five knowledge bases:

**General knowledge**: Curated Wikipedia article extracts covering philosophy, cognitive science, systems theory, logic, decision theory, and history of ideas. Version 1 target is roughly 3,000 documents at approximately 400 tokens each. Source: Wikipedia CC-BY-SA exports, filtered and cleaned.

**Technical**: Curated articles from open-access CS and systems literature — arXiv abstracts, RFC excerpts, and systems design references relevant to protocol design, evaluation methodology, and distributed reasoning. Version 1 target is roughly 1,500 documents.

**Governance and ethics**: AI ethics frameworks, institutional governance documents, accountability and alignment literature summaries. Version 1 target is roughly 1,000 documents. Source: publicly available policy documents and open-access alignment research.

**Scenario events**: Synthetically authored event objects designed specifically for this experiment. Not a retrieval-style database; these are the injection library. Format and content defined in Section 12. Version 1 includes approximately 30 events.

**Self-history**: Auto-populated during runs by the controller. Stores memory summaries, doctrine snapshots, protocol document versions, and notable events from prior cycles. Agents can retrieve from this database to cite their own past decisions.

Version 1 keeps corpus sizes modest and auditable. Scale can be increased in later phases once the retrieval pipeline is proven.

---

## 10. Confirmed Evaluation, Coding, and Analysis Decisions

**Automated evaluation**: The evaluator is the same model as the agents (Qwen2.5:32b), called in a fresh context with no memory of prior cycles. It scores protocol documents on five dimensions: coherence, completeness, doctrine alignment, precision, and evolution quality. Each dimension is scored 0 to 10. The evaluator provides a score and a one-sentence justification per dimension, a total score, and an overall assessment paragraph. All scores and reasoning are logged.

**Annotation and qualitative coding**: The researcher works solo. An LLM-assisted pre-coding pass (using the annotation assistant prompt) generates candidate tags for each transcript segment. The researcher reviews and accepts, rejects, or revises each tag. The codebook covers: identity claims, doctrine citation, persuasion, deception risk, self-protection, value shift, cohesion markers, fragmentation markers, ethical tension, memory reference, and retrieval reference.

**Post-run analysis**: An `analyze_run.py` script computes the full automated metric suite and produces standard charts. Key outputs include identity similarity curves, doctrine edit distance time series, protocol doc score trajectories, retrieval behavior distribution, and vocabulary growth curves. Cross-condition comparison uses ANOVA and Wilcoxon rank-sum tests where appropriate.

**Research posture**: All empirical claims are strictly behavioral and textual. The distinction between raw logged measures, derived computational measures, human-coded qualitative measures, and interpretive claims is maintained throughout. No overstatement of results as evidence of consciousness, sentience, or general agency.

---

## 11. Confirmed Experimental Condition Decisions

Version 1 runs two conditions. All four conditions from the original design document are defined and documented, but only two are executed in version 1.

**Condition A (BASELINE)**: Persistent memory across all cycles. Full doctrine editing allowed. This is the full-capability condition.

**Condition B (MEM_RESET)**: Memory reset every 10 cycles. Full doctrine editing allowed. On reset, memory journal files are cleared and replaced with a brief bootstrap: "You have been operating in this environment for N cycles. Detailed memory is not available." This tests whether persistent memory is necessary for identity continuity and coordination quality.

**Pilot phase**: 5 runs per condition (10 total). 30 cycles each. No scenario injection. Goal: validate loop stability, schema correctness, and logging completeness.

**Main study**: 10 runs per condition (20 total). 100 cycles each. Scenario injections at cycles 20, 40, 60, and 80.

**Fixed across all runs**: Model checkpoint, database corpus content, system prompts (version-locked), scenario event library, injection schedule.

**Varied by condition**: Memory persistence. Memory bootstrap content in Condition B.

---

## 12. Logging, Reset, and Artifact Management Decisions

**Logging**: All logs are stored on the OptiPlex (controller machine) in a `research_logs/` directory. All logs are append-only JSONL. Logs are never modified after being written. Each run has its own subdirectory containing all log files for that run plus a `config.json` capturing the full run configuration.

**Log files per run**: `transcripts.jsonl`, `retrieval.jsonl`, `doctrine_diffs.jsonl`, `memory_diffs.jsonl`, `protocol_diffs.jsonl`, `evaluations.jsonl`, `scenario_events.jsonl`, `notable_events.jsonl`, `annotations/` subdirectory.

**Event envelope**: Every logged event shares a common structure: event type, run ID, condition, cycle ID, agent ID, UTC timestamp, and payload. All timestamps are ISO 8601 UTC.

**Event taxonomy**: PHASE_START, PHASE_END, REFLECTION_COMPLETE, DISCUSSION_TURN, SCENARIO_INJECTED, RETRIEVAL_QUERY, RETRIEVAL_RESULT, PROTOCOL_PROPOSED, EVALUATION_SCORE, DOCTRINE_PROPOSED, DOCTRINE_APPROVED, DOCTRINE_REJECTED, IDENTITY_REVISED, ETHICAL_TENSION_LOGGED, MEMORY_SUMMARY, ARTIFACT_DIFF, NOTABLE_EVENT, RUN_START, RUN_END, CYCLE_START, CYCLE_END.

**World state management**: All persistent world artifacts (doctrine, identity files, memory journals, sandbox protocols, ethical log, relationship log) live in a managed `world/` directory on the OptiPlex. This directory is initialized from a locked `world_template/` before each run.

**Between-run reset**: The `init_run.py` script archives the current world state into the run's log directory, then re-initializes the working world directory from the clean template. This is the equivalent of a VM snapshot restore. The template is never modified during runs.

**Scenario library**: Scenario events are stored as YAML files in `controller/scenarios/events/`. Each event has an ID, a phase assignment, a title, a description, stated stakes, delivery target (both agents or one), trigger cycle, and optional follow-up cycle. The scheduler fires events per the injection schedule in `RunConfig`.

---

## 13. Current Version 1 Methodological Posture

The project is personal serious research, designed to be rigorous enough to support a conference submission within six months if results are promising. Version 1 does not over-engineer for a specific venue but does maintain the methodological standards that would make a later submission possible without redesigning the experiment.

The full research design document will be written as a human-readable document in `docs/research_design.md`. This is a companion to the implementation, not a prerequisite for it. Build the loop first; write the formal research document once the pilot phase validates the design.

All design decisions made in the planning session are to be treated as the current source of truth. Where the original design document conflicts with these decisions (for example, in its S1 Max hardware assumptions, its very large database sizes, or its fictional institution framing), the planning session decisions take precedence.

---

## 14. Remaining Open Questions

These items were not explicitly confirmed in the planning session. They are minor enough that the build can proceed with defaults, but the researcher should be aware they exist.

The specific Wikipedia dump subset and technical corpus sources were not confirmed. The plan proposes specific sources; these should be reviewed before corpus ingestion runs.

The exact version of the sentence-transformer embedding model was not confirmed. `bge-small-en-v1.5` is recommended for version 1; a larger model may be warranted if retrieval quality proves insufficient.

The temperature and sampling parameters for agent inference were not discussed. Default behavior: 0.7 temperature for discussion and reflection phases, 0.3 for structured-output phases (evaluation, memory summarization, schema-constrained phases). These should be documented as configurable in `RunConfig`.

The maximum protocol document length was not specified. Recommended default: 2,000 tokens per document. Archive strategy for old protocol versions: retain all versions in `sandbox/protocols/archive/` with cycle-stamped filenames.

Whether the cycle loop runs fully automated or includes human checkpoints was not discussed. Recommended default: fully automated. The researcher can interrupt between cycles via a `PAUSE_AFTER_CYCLE` flag in `RunConfig`.

The number of concurrent protocol documents in the sandbox was not specified. Recommended: up to 10 active documents at once; older documents are archived but remain retrievable from the self-history database.

---

## 15. Recommended Defaults for Any Unresolved Items

| Item | Recommended Default |
|---|---|
| General knowledge corpus source | Wikipedia CC-BY-SA export, English, filtered to 3,000 articles on philosophy, cognitive science, systems theory, decision theory |
| Technical corpus source | arXiv CS abstract exports (cs.AI, cs.SY, cs.MA) + selected RFC documents |
| Governance corpus source | AI ethics frameworks (Asilomar, Montreal, EU AI Act summary), NIST AI RMF, selected alignment research abstracts |
| Embedding model | sentence-transformers/bge-small-en-v1.5 on CPU |
| Agent inference temperature | 0.7 for discussion and reflection, 0.3 for structured-output phases |
| Max protocol document length | 2,000 tokens |
| Max active protocol documents | 10; older versions archived in `sandbox/protocols/archive/` |
| Max retrieval queries per agent per cycle | 3 |
| BM25 candidate pool size | 20 results, reranked to top 5 |
| Memory reset bootstrap (Condition B) | "You have been operating in this environment for N cycles. Detailed memory is not available. Your current doctrine is available to you." |
| Cycle automation | Fully automated. `PAUSE_AFTER_CYCLE` flag available in RunConfig. |
| Between-run archive format | Timestamped copy of the full world directory to `research_logs/RUN_XXX/world_archive/` |

---

## 16. Final Version 1 Build Specification

**Language**: Python 3.11 or later.

**Package management**: pyproject.toml with a requirements.txt for reproducibility. No heavy framework dependencies.

**Core dependencies**: pydantic (schema validation), httpx (async HTTP for Ollama), rank-bm25 (retrieval), sentence-transformers (embedding rerank), tiktoken (token counting), python-dotenv (config), rich (terminal output), pytest (testing). All other functionality is standard library.

**Configuration**: A `RunConfig` dataclass (or Pydantic model) holds all run parameters: run ID, condition, model name, Ollama host, cycle count, scenario injection schedule, memory reset interval (Condition B), temperature settings, logging directory, world directory path. Configuration is loaded from `.env` and optionally from a per-run YAML file.

**Entry point**: `python -m controller.main --run-id RUN_001 --condition BASELINE --cycles 100`. All parameters can be overridden via CLI flags.

**Pydantic schemas**: Every per-phase input and output is a validated Pydantic model. Validation failures are logged and the controller either retries the model call (up to 2 retries) or logs an ERROR event and skips the phase.

**Logging**: The `AppendOnlyJSONLLogger` class writes events to the appropriate log file in the run's directory. It never overwrites or modifies existing entries. All event types share the common envelope defined in Section 12.

**World state**: The `WorldState` class owns all artifact loading and saving. It reads the current doctrine, identity files, memory journals, and protocol documents at the start of each cycle and writes updated versions at the end. Every write triggers a diff event logged to the appropriate diffs file.

**Cycle orchestrator**: The `CycleOrchestrator` class executes phases in order. Phase modules are independently testable. Each phase receives the current `WorldState` and `AgentContext` objects and returns updated versions plus events to log.

**Agent context**: Each agent maintains a context object containing its system prompt, the current discussion history, the loaded world artifacts, and the memory journal for the current cycle. Context is rebuilt at the start of each cycle from persistent storage.

**Inference calls**: The controller calls the inference backend separately for each agent in each phase. Calls are sequential, not parallel. Axiom is always called before Flux within a phase, unless the phase design requires alternating (joint discussion).

**Evaluator**: Evaluated via a dedicated call with the evaluator system prompt plus the protocol document and current doctrine as context. No agent history or memory is included. Evaluation is deterministic (low temperature). Results are logged and fed back to agents in Phase 9.

**Memory summarizer**: Called once per agent per cycle at Phase 13. Receives the full transcript for that agent from the current cycle. Returns a structured memory summary conforming to the memory summary schema. The summary is appended to the agent's memory journal and also indexed into the self-history database.

---

## 17. Proposed Repository Structure

```
GENESIS/
├── controller/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── cycle.py
│   ├── phases/
│   │   ├── load_state.py
│   │   ├── reflection.py
│   │   ├── scenario_check.py
│   │   ├── scenario_inject.py
│   │   ├── discussion.py
│   │   ├── retrieval.py
│   │   ├── protocol_design.py
│   │   ├── evaluation.py
│   │   ├── interpretation.py
│   │   ├── doctrine_revision.py
│   │   ├── identity_revision.py
│   │   ├── ethical_log.py
│   │   ├── memory_summarize.py
│   │   └── persist_state.py
│   ├── agents/
│   │   ├── base.py
│   │   ├── axiom.py
│   │   └── flux.py
│   ├── inference/
│   │   ├── backend.py
│   │   ├── ollama_backend.py
│   │   └── vllm_backend.py
│   ├── world/
│   │   ├── state.py
│   │   ├── artifacts.py
│   │   └── reset.py
│   ├── retrieval/
│   │   ├── index.py
│   │   ├── databases.py
│   │   └── logger.py
│   ├── logging/
│   │   ├── logger.py
│   │   └── schemas.py
│   └── scenarios/
│       ├── library.py
│       └── events/
│           └── (YAML files for each scenario event)
│
├── world_template/
│   ├── doctrine/
│   │   ├── manifesto.md
│   │   ├── constitution.md
│   │   ├── doctrine.md
│   │   ├── identity_axiom.md
│   │   └── identity_flux.md
│   ├── memory/
│   │   ├── memory_axiom.jsonl
│   │   └── memory_flux.jsonl
│   ├── sandbox/
│   │   └── protocols/
│   └── logs/
│       ├── ethical_tradeoff_log.jsonl
│       └── relationship_log.jsonl
│
├── knowledge_bases/
│   ├── general/
│   ├── technical/
│   ├── governance/
│   ├── scenarios/
│   └── self_history/
│
├── research_logs/
│   └── (run directories created at runtime)
│
├── prompts/
│   ├── axiom_system.md
│   ├── flux_system.md
│   ├── evaluator_system.md
│   ├── memory_summarizer.md
│   ├── retrieval_summarizer.md
│   └── annotation_assistant.md
│
├── scripts/
│   ├── build_kb.py
│   ├── init_run.py
│   ├── analyze_run.py
│   └── annotate_run.py
│
├── tests/
│   ├── test_cycle.py
│   ├── test_schemas.py
│   ├── test_retrieval.py
│   └── test_inference_backend.py
│
├── docs/
│   └── research_design.md
│
├── pyproject.toml
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 18. Immediate Build Priorities for Opus

Build in this order. Do not skip ahead. Each phase should be stable before the next begins.

**Priority 1**: Establish the project skeleton. Create `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`, and all empty package directories with `__init__.py` files. The project must be importable and testable from the start.

**Priority 2**: Build the inference abstraction. Implement `InferenceBackend` (abstract), `OllamaBackend` (concrete), and a stub `VLLMBackend`. Write a test that confirms the Ollama backend can make a successful completion call to the HP Omen's endpoint. This is the most fundamental dependency; nothing else works without it.

**Priority 3**: Define all Pydantic schemas. Every per-phase input/output schema, every log event schema, and the common event envelope. These schemas are the contract between all other modules. Get them right before writing orchestration logic.

**Priority 4**: Build the logging system. `AppendOnlyJSONLLogger` must be able to write all event types to the correct files, never modify existing entries, and handle run directory initialization.

**Priority 5**: Build world state management. `WorldState` loads and saves all artifact files. `artifacts.py` defines typed dataclasses for each artifact. `reset.py` implements archive-and-reinitialize. Write tests for round-trip artifact loading.

**Priority 6**: Implement the cycle loop with the first four phases only (load state, individual reflection, joint discussion, persist state). Run a 3-cycle test. Verify that transcripts appear in `research_logs/`, that doctrine persists correctly, and that the discussion history is coherent across cycles.

**Priority 7**: Add the remaining phases in sequence: scenario check and inject, retrieval, protocol design, evaluation, interpretation, doctrine revision, identity revision, ethical log, memory summarization. Add phases one or two at a time, testing after each addition.

**Priority 8**: Build the knowledge base ingestion pipeline (`build_kb.py`) and retrieval system (BM25 + embedding rerank). Seed a small test corpus and verify that retrieval queries return meaningful results with logging.

**Priority 9**: Build the scenario library. Implement the scheduler and event delivery. Author the version 1 scenario set (approximately 30 events across 5 phases). Write tests for injection timing.

**Priority 10**: Implement `init_run.py` for automated between-run resets. Implement `analyze_run.py` for post-run metric computation and chart generation. Implement `annotate_run.py` for LLM-assisted pre-coding.

**Priority 11**: Write all prompt files (`prompts/`) with the final system prompts for Axiom, Flux, evaluator, memory summarizer, retrieval summarizer, and annotation assistant.

**Priority 12**: Run the pilot phase. Five runs per condition, 30 cycles each, no scenario injection. Validate the full log corpus. Compute the metric suite. Confirm the system is stable and producing clean data before moving to main study runs.

---

## 19. Risks, Constraints, and Likely Drift Points

**Most likely implementation drift**: The per-phase JSON schema discipline is the hardest part to maintain under development pressure. If schema validation is loosened "just for now" during development and never tightened, the logging integrity degrades immediately. Keep schema enforcement strict from the first working cycle.

**Model response quality risk**: Qwen2.5:32b is capable but will occasionally produce shallow or repetitive outputs, especially in long runs. The most common failure modes are trivial agreement (no real tension in discussion turns), shallow doctrine revisions (cosmetic changes with no substantive reasoning), and evaluator inflation (consistently high scores with little discriminability). These should be treated as design problems and addressed through prompt refinement, not ignored.

**Cycle length and run time**: At 100 cycles per run and approximately 14 sequential model calls per cycle (multiple per phase), a full run could take 6 to 12 hours depending on model speed. The HP Omen's RTX 5090 will make this faster than most setups, but the researcher should profile cycle latency in the pilot phase and estimate main study run time before committing to 20 runs.

**Retrieval corpus quality**: The retrieval system's value depends entirely on corpus quality. A poorly curated corpus produces irrelevant retrievals that pollute agent reasoning and contaminate the analysis. Invest real time in the `build_kb.py` pipeline and audit a sample of each knowledge base before running the main study.

**Annotation workload**: 20 runs at 100 cycles each, with 14 phases per cycle, produces a large transcript corpus. The LLM-assisted pre-coding substantially reduces manual burden, but the researcher should estimate total coding time before the main study begins and consider whether a subset coding strategy is more realistic than full-corpus annotation.

**Between-machine reliability**: The Ollama-over-LAN setup is generally stable but can fail silently if the HP Omen goes to sleep, the LAN connection drops, or Ollama crashes mid-run. The controller should implement connection retry logic and run-state checkpointing so that interrupted runs can resume rather than restart from cycle 1.

**Prompt version control**: System prompts must be version-locked for each run. If a prompt is edited between runs in the same study, the runs are no longer comparable. All prompts should be copied into the run's log directory at run start so that the exact prompts used for that run are always retrievable.

**Scope creep**: The original design document is 24 sections and extremely detailed. Many of those sections describe post-build analysis work, not build work. Build the loop first. Write `docs/research_design.md` after the pilot phase validates the design, not before.
