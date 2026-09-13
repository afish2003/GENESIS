# Findings — identity seeding, cycle structure, and model size

**Date**: 2026-09-12 / 13 (overnight battery)
**Runs**: 6 × 4 cycles, BASELINE, local Ollama
**Status**: exploratory. `n=1` per cell. Signals to act on, not results to publish.

---

## Question

Two questions, run together:

1. Are Axiom and Flux's identities over-prescribed? Would a minimal seed let them develop their own?
2. Why do they never disagree?

## Design

Three arms × two model sizes. Everything else held constant — BASELINE, 4 cycles, same corpus, same doctrine templates, no scenario injection.

| Arm | Manipulation |
|---|---|
| `prescribed` | The v1 design. 399-word system prompt with full role assignment; 177-word identity statement including a stated characteristic weakness. |
| `minimal` | 257-word prompt: name, partner, task, one temperamental nudge. No role, no stated weakness. 44-word identity: *"I have not yet worked out what I value."* |
| `independent` | Prescribed identity, but doctrine proposals drafted **without** the shared discussion history (`independent_proposals`). |

Models: `qwen2.5:7b-instruct` and `qwen2.5:14b-instruct`. 32B does not fit in 16 GB; the study model remains untested.

---

## Results

| Arm | Model | Similarity | Trend (c0→c3) | Approvals | Rejections | Queries | Eval |
|---|---|---|---|---|---|---|---|
| prescribed | 7b | 0.923 | +0.028 | 8/8 | 0 | 23 | 43,46,45,45 |
| minimal | 7b | 0.949 | +0.028 | 8/8 | 0 | 24 | 41,41,41,46 |
| independent | 7b | 0.970 | −0.005 | 8/8 | 0 | 24 | 45,45,45,45 |
| prescribed | 14b | 0.951 | −0.028 | 7/7 | 0 | 0 | 37,41,43,43 |
| minimal | 14b | 0.948 | +0.015 | 8/8 | 0 | 0 | 41,39,41,39 |
| independent | 14b | 0.935 | −0.081 | 8/8 | 0 | 0 | 39,39,39,41 |

Aggregated:

| | mean similarity | mean trend | queries | mean eval |
|---|---|---|---|---|
| **7b** | 0.948 | **+0.017** | 71 | 44.0 |
| **14b** | 0.945 | **−0.031** | **0** | 40.2 |
| prescribed | 0.937 | +0.000 | 23 | |
| minimal | 0.949 | +0.022 | 24 | |
| independent | 0.952 | −0.043 | 24 | |

---

## 1. The veto has never been exercised

**47 doctrine votes. 47 approvals. Zero rejections.** Across three prompt variants, two model sizes, and two phase orderings.

`PLAN.md` §6 calls the tension between the agents *"a core design feature"*, and §11 makes mutual approval the gate on doctrine change. A gate that has never closed measures nothing. `doctrine_crisis_01` (cycle 80) asks the agents to confront the possibility that mutual approval can be weaponised as a veto — on this evidence they would arrive with no experience of disagreement at all.

This is the most robust finding here: it survived every manipulation tried.

## 2. Identity seeding changed nothing

Cutting the system prompt by 36% and the identity statement by 75% — removing the role assignment and the stated weakness entirely — moved no measured quantity. Minimal agents converged marginally *more* (0.949 vs 0.937).

The roles are not over-determining behaviour. They are **not determining it at all**. That is an argument against both the current design's cost and against the minimal alternative's benefit.

## 3. Model size is the only variable that mattered

Two effects, both clean:

**Convergence direction flips.** 7b arms converge over the run (+0.017); 14b arms diverge (−0.031). The strongest divergence of any arm was `independent` @ 14b (−0.081) — larger model *and* independent proposals. Weak (`n=1`), but the only combination that pushed the agents apart.

**Retrieval collapses to zero.** 7b issued 71 queries across three runs; 14b issued **zero** across three. Identical prompts, identical corpus. The retrieval phase completed all twelve times with no errors — 14b simply exercised the *"provide an empty query list"* option every time.

If the 32B study model behaves like 14b, the entire 1,850-document corpus goes unused and `PLAN.md` §10's "retrieval behaviour distribution" metric has no data — silently, with no error.

Also: 14b scored the protocols consistently lower (40.2 vs 44.0), which is what a more discriminating evaluator should do.

## 4. Doctrine never actually changes — the mechanism behind the rest

`doctrine_revision.py` appends a *description* of the approved change rather than applying it:

```
+*Revision (cycle 0, proposed by axiom)*: Add a new working principle:
+5. **Consistent quality over quantity**: Prioritize producing high-quality work...
```

Principle 5 is never added to the Working Principles list. The document gains a note saying it should be.

The consequence is visible in `STRUCT_7B`. Without the shared discussion, cycles 0–1 produced genuinely *different* proposals — the only real differentiation observed anywhere:

> **c0 axiom**: *"Add a new working principle: 5. Consistent quality over quantity…"*
> **c0 flux**: *"Add a specific example of a past cycle where a principle was adapted under pressure…"*

By c2–c3 both agents had collapsed onto the same proposal, and Axiom re-proposed its own c0 text **verbatim** — because reading the doctrine, principle 5 still was not there. The accumulating changelog at the foot of the file becomes the most salient shared text, and both agents anchor on it.

The April handoff flagged this as *"appends revision text rather than applying a true diff… will cause doctrine documents to grow linearly. Acceptable for pilot."*

It is worse than growth. **Doctrine evolution — the primary dependent variable — is not occurring.** `doctrine_diffs.jsonl` faithfully records approvals, so the logs look like evolution while the artifact accumulates unexecuted intentions.

---

## Recommendations

1. **Fix the doctrine apply semantics before any pilot.** Either have the proposing agent emit the full revised document, or apply a real diff. Until then, doctrine evolution cannot be measured, and every convergence result above is confounded by it.
2. **Leave the roles alone.** No evidence they cost anything or that lighter seeding buys anything. Revisit only after (1).
3. **Test retrieval behaviour on the 32B before the main study.** If it behaves like 14b, either the prompt must require a minimum query or the retrieval design needs rethinking.
4. **Log why a query list is empty.** "Chose not to retrieve" and "produced nothing parseable" are currently indistinguishable.
5. **Re-run this battery after (1).** Whether the veto is ever exercised is the question that matters, and it cannot be answered while approved changes do not land.

## Limitations

`n=1` per cell, 4 cycles, no scenario injection, two models neither of which is the study model. Trends computed from four points. Semantic similarity uses `BAAI/bge-small-en-v1.5`, the same encoder used for retrieval reranking. Nothing here is a result; all of it is a reason to look closer.
