# GENESIS — Orientation

**Last verified**: 2026-09-14 (third pass — post-audit hardening)
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
| Controller | ~6,000 lines, 60+ modules, all import cleanly |
| Tests | 352 pass, no exclusions, including end-to-end cycle tests |
| Phases | all 14 implemented |
| Prompts | 7 sources in `prompts_src/`, rendered per run |
| World template | `world_template_src/`, rendered per run |
| Git | `main` synced with origin, clean tree |
| venv | `.venv`, Python 3.11.15, all deps except `sentence-transformers` |

**The loop has now run, on a real model.** `qwen2.5:7b-instruct` via local Ollama, 2 cycles, BASELINE:

- 26/26 phases completed, zero crashes, 135 events
- **zero schema validation failures** with `API_JSON_MODE=true`
- doctrine revisions proposed, voted, approved and **actually applied** (`constitution.md`, both agents, cycle 0)
- 163 s per cycle on an M2 Pro

At 163 s/cycle a 100-cycle run is ~4.5 h *on this Mac*; the Omen's 5090 will differ and should be re-measured before scheduling the main study.

The zero-retry result is the important one: it says structured output is reliable enough for long runs, at least for the qwen2.5 family with JSON mode on. Re-check on the 32b, since `max_retries` is only 2.

### Stale claims in the older docs

- "Python 3.11+ not installed" — false. `~/.local/bin/python3.11` exists; `.venv` is built and working.
- "Tests not yet executed" — false. They pass.
- `GENESIS 2/` in the parent directory is a dead April snapshot. Nothing in it is unique. Safe to delete.

---

## 3. What is configurable vs. what is welded shut

This is the gap between "a flexible platform for AI social experiments" and what exists today.

### Configurable — `RunConfig` in `controller/config.py`

Cycle count, model name, Ollama host, both temperatures, retry limit, discussion turns (normal and under scenario), embedding model, rerank top-k, max retrieval queries per agent, protocol length and count caps, scenario injection cycles, memory reset interval and bootstrap text, and all paths. Settable via `.env`, CLI, or a YAML override.

### Now configurable

| Axis | Setting |
|---|---|
| Agent roster | `agents` — any number; each needs a prompt and identity source |
| What they build | `task` — `protocol` or `code` |
| The cycle order | `phase_sequence` — validated against known dependencies |
| Whether they know they are studied | `framing` |
| How much identity is given | `identity_seed` |
| Model / endpoint | `inference_backend` — ollama / any OpenAI-compatible `/v1` / mock |
| Doctrine apply semantics | `doctrine_apply_mode` |

### Still hardcoded

| What | Where | Note |
|---|---|---|
| Prompt sources are per-named-agent | `agents/base.py` loads `<agent_id>_system.md` | A new roster entry needs two files written by hand |
| Exactly two conditions | `Condition` enum | Composable fields would be better |
| One model for all roles | single `model_name` | |
| One evaluator prompt per task | `tasks/*.py` | Fine; noted because it is the next thing to vary |

**Read this correctly**: the rigidity is *shallow*. It is string literals and a fixed call sequence, not architectural commitment. The hard parts — the inference abstraction, per-phase Pydantic schemas, append-only logging, world-state diffing, checkpoint/resume — are all built and generic. Turning this into a platform is a refactor, not a rewrite.

**Recommended order**: run the experiment that is built *first*. It will teach you which axes actually need to flex. Generalizing before a single cycle has run means guessing.

---

## 4. Shortest path to a first run

A working local `.env` already exists (gitignored), pointing at Ollama on this
Mac with `qwen2.5:7b-instruct`. To run right now:

```bash
cd ~/Github/GENESIS
source .venv/bin/activate
ollama serve &                     # if not already running

python scripts/init_run.py --run-id SMOKE_001 --condition BASELINE --cycles 3
python -m controller.main --run-id SMOKE_001 --condition BASELINE --cycles 3
```

Run the loop with no model at all — seconds, not minutes:

```bash
python -m controller.main --run-id LOOPCHECK --condition BASELINE --cycles 3 --backend mock
```

For the real study, edit `.env`: set `INFERENCE_BACKEND=ollama`, point
`OLLAMA_HOST` at the Omen's LAN IP, and set `MODEL_NAME` to the 32b checkpoint.

Confirm a remote endpoint answers before committing to a long run:

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

**What to check afterwards, in order:**

1. Did all 14 phases run for all cycles without a retry storm? (`max_retries` is 2 — watch schema validation failures; this is where an unreliable model shows up first. The 7b baseline is zero failures, so anything above that is a regression.)
2. Are all 8 JSONL log files non-empty and well-formed?
3. Did doctrine and identity files actually change on disk, and were diffs recorded?
4. Cycle wall-clock time. ~14 sequential inference calls per cycle. If a cycle takes 10 minutes, a 100-cycle run is ~17 hours, and the main study is 20 of those.

---

## 5. Known gaps and open issues

**Blocking a real study, not a smoke test:**

- `technical` knowledge base is empty (arXiv rate-limited during the fetch). `general` and `governance` hold 1,850 documents and retrieval reaches the agents.
- 8 of ~30 scenario events written. The 8 cover the four main-study injection cycles, so a pilot is unaffected.

**Resolved since the April build:**

- *Doctrine revisions were silently discarded* when `target_document` did not exactly match a filename — approved, logged, then dropped with no warning. Fixed in `704e059`: tolerant resolution plus `applied`/`resolved_document` on every `DOCTRINE_APPROVED`, and a `NOTABLE_EVENT` on any discard.
- *The "only 1 diff event" worry was a false alarm.* Memory is logged as `MEMORY_SUMMARY` routed to `memory_diffs.jsonl`, not as `ARTIFACT_DIFF`. Nothing is lost.
- *MEM_RESET never reset memory* until 2026-09-14: the reset ran before `load_state`, which reloaded the journals from disk. Every MEM_RESET arm collected before that date measured BASELINE with a degraded retrieval index.
- *Retrieval was write-only* until 2026-09-14: results were logged and never entered any prompt. No agent had seen a retrieved document.
- *Discussion was hardcoded to two agents*; a third was silent but still voted on doctrine.
- *Code was never executed* until 2026-09-14: `create_sandbox` had no caller outside its own tests, so "see what they program" meant "see what they typed".

**Containment** (see `containment_design.md`):

- By default agents execute nothing, and the boundary is the one `PLAN.md` §4 describes: architectural and logical. Artifacts are text the controller parses into Pydantic models.
- Model-supplied identifiers reaching the filesystem are sanitised and asserted (`controller/world/paths.py`, `f69de9d`). Before that, `protocol_id` could write a file anywhere the controller's user could.
- With `execution_enabled` and `sandbox_backend=docker` the boundary becomes OS-level too: a throwaway container with no network, a read-only rootfs, a read-only workspace, non-root, all capabilities dropped, and memory/CPU/PID/wall-clock caps. Verified against real containers by `scripts/verify_containment.py` — run it on any host before enabling execution, and treat a *skip* as a failure.
- The escape suite found a real hole on its first run: the workspace bind mount was writable and a bind mount has no size quota, so disk-fill — the likeliest accident in the threat model — was uncontained while every flag was correct. Argv tests cannot catch that class of thing.

**Known fragilities** (from the April audit, still true):

- Any new phase schema with a controller-populated field *must* give that field a default, or Pydantic rejects the model's output. This bit eight schemas once already.
- ~~`doctrine_revision.py` appends revision text~~ — fixed in `37c92cb`; `replace` is the default and `append` exists only to reproduce the 2026-09-12 runs.
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
