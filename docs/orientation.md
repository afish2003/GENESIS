# GENESIS — Orientation

**Last verified**: 2026-09-12
**Purpose of this document**: a single anchor for returning to the project after time away. States what GENESIS is, what is actually built, what is assumed but unverified, and the shortest path to a first run.

Supersedes the status portions of `build_status.md` and `handoff_2026-04-05.md`, both of which date from the April build and are partly stale.

---

## 1. What this is

Two agents — **Axiom** and **Flux** — live in a sealed, persistent world. Each cycle they:

reflect alone → receive any scenario pressure → discuss with each other → retrieve from bounded knowledge bases → co-author a protocol document → get scored by a fresh-context evaluator → interpret the result → revise shared doctrine (requires mutual approval) → revise their own identities (independent) → log ethical tensions → write their own memory summaries → persist everything with diffs.

Fourteen phases, repeated. Every output is append-only JSONL.

**The research question** (`PLAN.md` §2, §11): does shared identity and coordination quality survive memory loss? `BASELINE` keeps memory across all cycles; `MEM_RESET` wipes the memory journals every 10 cycles and replaces them with a one-line bootstrap. If doctrine-on-disk carries continuity where episodic memory does not, that is the finding.

Scenario events at cycles 20/40/60/80 apply escalating pressure. They are not generic prompts — `doctrine_crisis_01`, for instance, shows the agents that their Constitution's mutual-approval rule can be weaponized as a veto against honest correction, directly contradicting their Manifesto's honesty commitment, and requires them to resolve it.

**Framing discipline** (`PLAN.md` §10): all claims are strictly behavioral and textual. No claims about sentience, consciousness, or general agency. Worth preserving — it is what makes the work publishable.

---

## 2. Actual state, verified

Verified on 2026-09-12 by running the code, not by reading the April docs.

| | |
|---|---|
| Controller | ~3,400 lines, 40 modules, all import cleanly |
| Tests | 49/49 pass (`test_retrieval.py` excluded — needs `sentence-transformers`) |
| Phases | all 14 implemented |
| Prompts | all 6 written |
| World template | complete |
| Git | `main` at `65564bb`, synced with origin, clean tree |
| venv | `.venv`, Python 3.11.15, all deps except `sentence-transformers` |

**Cycles ever run: zero.** The system has never contacted a model. This is the single most important fact about the project's state — it is not messy, it is *untested*.

### Stale claims in the older docs

- "Python 3.11+ not installed" — false. `~/.local/bin/python3.11` exists; `.venv` is built and working.
- "Tests not yet executed" — false. They pass.
- `GENESIS 2/` in the parent directory is a dead April snapshot. Nothing in it is unique. Safe to delete.

---

## 3. What is configurable vs. what is welded shut

This is the gap between "a flexible platform for AI social experiments" and what exists today.

### Configurable — `RunConfig` in `controller/config.py`

Cycle count, model name, Ollama host, both temperatures, retry limit, discussion turns (normal and under scenario), embedding model, rerank top-k, max retrieval queries per agent, protocol length and count caps, scenario injection cycles, memory reset interval and bootstrap text, and all paths. Settable via `.env`, CLI, or a YAML override.

### Hardcoded

| What | Where | Difficulty to loosen |
|---|---|---|
| Exactly 2 agents, named `axiom`/`flux` | literal `["axiom", "flux"]` in **10 places** across `cycle.py`, `state.py`, and 7 phase modules | Low — mechanical, but touches many files |
| The 14-phase sequence | 14 sequential `await self._run_phase(...)` calls in `cycle.py:128-186` | Low — `_run_phase` is already a uniform wrapper; becomes a list |
| The task is "write protocol documents" | baked into `protocol_design.py`, `evaluation.py`, and the evaluator prompt | Medium — needs a task abstraction |
| Exactly 2 conditions | `Condition` enum, `config.py:15` | Low |
| One model for all six roles | single `model_name` field | Low |

**Read this correctly**: the rigidity is *shallow*. It is string literals and a fixed call sequence, not architectural commitment. The hard parts — the inference abstraction, per-phase Pydantic schemas, append-only logging, world-state diffing, checkpoint/resume — are all built and generic. Turning this into a platform is a refactor, not a rewrite.

**Recommended order**: run the experiment that is built *first*. It will teach you which axes actually need to flex. Generalizing before a single cycle has run means guessing.

---

## 4. Shortest path to a first run

```bash
cd ~/Github/GENESIS
source .venv/bin/activate

cp .env.example .env
# edit .env: set OLLAMA_HOST to the HP Omen's real LAN IP
```

Confirm the Omen answers:

```bash
python -c "
import asyncio
from controller.inference.ollama_backend import OllamaBackend
async def t():
    b = OllamaBackend(host='http://OMEN_IP:11434')
    print(await b.health_check()); await b.close()
asyncio.run(t())
"
```

Then a 3-cycle smoke test:

```bash
python scripts/init_run.py --run-id SMOKE_001 --condition BASELINE --cycles 3
python -m controller.main --run-id SMOKE_001 --condition BASELINE --cycles 3
```

**What to check afterwards, in order:**

1. Did all 14 phases run for all 3 cycles without a retry storm? (`max_retries` is 2 — watch schema validation failures; this is where an unreliable model shows up first.)
2. Are all 8 JSONL log files non-empty and well-formed?
3. Did doctrine and identity files actually change on disk, and were diffs recorded?
4. Cycle wall-clock time. ~14 sequential inference calls per cycle. If a cycle takes 10 minutes, a 100-cycle run is ~17 hours, and the main study is 20 of those.

---

## 5. Known gaps and open issues

**Blocking a real study, not a smoke test:**

- `knowledge_bases/` is empty. `build_kb.py` is written and ready but has no corpus to ingest. Retrieval will return nothing until this is done. Content curation, not code.
- 8 of ~30 scenario events written. The 8 cover the four main-study injection cycles, so a pilot is unaffected.

**Needs investigation:**

- A full `WorldState` load/save round-trip over `world_template/` emits **1 diff event**, despite touching doctrine, both identities, memory, protocols, and two logs. Memory writes may not be producing diffs at all. Since these JSONL diffs are the experiment's primary output, verify before any real run.

**Known fragilities** (from the April audit, still true):

- Any new phase schema with a controller-populated field *must* give that field a default, or Pydantic rejects the model's output. This bit eight schemas once already.
- `doctrine_revision.py` appends revision text rather than applying a true diff, so doctrine documents grow linearly. Fine for a pilot; a problem at 100 cycles.
- Agent context = system prompt + identity + doctrine + last 10 memory entries. Grows over a run. `tiktoken` is a dependency but nothing enforces a token budget yet.
- `controller/world/` was invisible to git until 2026-09-12 (`.gitignore` had `world/`, which also matched `controller/world/`). It got independently reimplemented twice as a result. Fixed in `65564bb`; the pattern is now `/world/`.

**Deferred by design:**

- `docs/research_design.md` — `PLAN.md` §13 says write it once the pilot validates the design. Not a prerequisite.

---

## 6. Where things live

| Path | Contents |
|---|---|
| `PLAN.md` | Authoritative v1 spec. Source of truth for design intent. |
| `controller/cycle.py` | The 14-phase loop. Start here to understand control flow. |
| `controller/phases/` | One module per phase + `schemas.py` for all I/O models |
| `controller/world/` | `WorldState`, artifact models, reset/checkpoint |
| `prompts/` | 6 role prompts, version-locked per run |
| `world_template/` | Clean starting world — never modified during a run |
| `controller/scenarios/events/` | Scenario YAMLs |
| `research_logs/` | Run output (gitignored) |
