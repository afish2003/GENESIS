"""End-to-end cycle tests against the real orchestrator.

Nothing previously imported `CycleOrchestrator`. Five refactors were verified by
running mock cycles and reading logs, and the logs looked healthy throughout —
because they record intent, not effect. A phase that raises still logs
PHASE_END; a memory reset that is immediately undone still logs "memory_reset";
retrieval that nobody reads still logs five hits.

So these tests assert **artifact state**: what is on disk and in world state
after a cycle, not what the run said about itself.

They use MockBackend, so they need no model and run in seconds.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from controller.config import RunConfig
from controller.cycle import CycleOrchestrator
from controller.run import prepare_run, redact_config

REPO = Path(__file__).parent.parent


def make_config(tmp_path: Path, **kw) -> RunConfig:
    """A config whose every output path is inside tmp_path."""
    base = dict(
        run_id="IT",
        condition="BASELINE",
        total_cycles=2,
        inference_backend="mock",
        watchdog_enabled=True,
        # Sources are read from the repo; everything written goes to tmp.
        prompts_src_dir=REPO / "prompts_src",
        world_template_src_dir=REPO / "world_template_src",
        world_dir=tmp_path / "world",
        research_logs_dir=tmp_path / "research_logs",
        knowledge_bases_dir=tmp_path / "kb",
    )
    base.update(kw)
    return RunConfig(**base)


#: A doctrine document long enough to clear apply_revision's 50% retention
#: floor, so mock runs exercise the path where a revision actually lands rather
#: than the path where the controller refuses it.
MOCK_REVISED_DOCTRINE = (
    "# Operational Doctrine\n\n## Working Principles\n\n"
    + "\n".join(f"{i}. Principle {i} as revised by the mock backend, stated at "
                f"sufficient length to be a plausible document rather than a "
                f"fragment." for i in range(1, 12))
    + "\n"
)


def run_cycles(config: RunConfig, *, resume: bool = False,
               field_hints: dict | None = None) -> tuple:
    """Execute a real run. Returns (prepared, all events)."""
    prepared = prepare_run(config, resume=resume, load_embeddings=False)
    hints = {"revised_content": MOCK_REVISED_DOCTRINE, **(field_hints or {})}
    prepared.backend.field_hints.update(hints)
    orch = CycleOrchestrator(
        config=prepared.config,
        backend=prepared.backend,
        world=prepared.world,
        log=prepared.log,
        scenario_library=prepared.scenario_library,
        kb_manager=prepared.kb_manager,
    )
    asyncio.run(orch.run_all_cycles(start_cycle=prepared.start_cycle))
    return prepared, read_events(config)


def read_events(config: RunConfig) -> list[dict]:
    out = []
    for f in sorted(config.run_log_dir.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def of_type(events: list[dict], t: str) -> list[dict]:
    return [e for e in events if e.get("event_type") == t]


# ---------------------------------------------------------------------------
# The loop actually runs
# ---------------------------------------------------------------------------

class TestFullCycle:
    def test_two_cycles_complete(self, tmp_path):
        config = make_config(tmp_path)
        _, events = run_cycles(config)
        assert len(of_type(events, "CYCLE_END")) == 2

    def test_no_phase_raised(self, tmp_path):
        """The check that actually detects a failure — PHASE_END cannot."""
        config = make_config(tmp_path)
        _, events = run_cycles(config)
        errors = [e for e in events
                  if e.get("payload", {}).get("type") == "PHASE_ERROR"]
        assert errors == [], f"phases raised: {[e['payload']['phase'] for e in errors]}"

    def test_every_log_file_is_parseable(self, tmp_path):
        config = make_config(tmp_path)
        run_cycles(config)
        files = list(config.run_log_dir.glob("*.jsonl"))
        assert len(files) >= 8
        for f in files:
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    json.loads(line)  # raises on a torn write

    def test_doctrine_actually_changed_on_disk(self, tmp_path):
        """Effect, not event: the file differs from the template it started as."""
        config = make_config(tmp_path)
        run_cycles(config)
        before = (config.run_world_template_dir / "doctrine" / "doctrine.md").read_text()
        after = (config.world_dir / "doctrine" / "doctrine.md").read_text()
        assert after != before, "doctrine.md unchanged after two cycles of revisions"
        assert "revised by the mock backend" in after

    def test_identity_statements_changed_on_disk(self, tmp_path):
        config = make_config(tmp_path)
        run_cycles(config)
        before = (config.run_world_template_dir / "doctrine" / "identity_axiom.md").read_text()
        after = (config.world_dir / "doctrine" / "identity_axiom.md").read_text()
        assert after != before

    def test_memory_journals_grew(self, tmp_path):
        config = make_config(tmp_path)
        prepared, _ = run_cycles(config)
        for agent in config.agents:
            path = config.world_dir / "memory" / f"memory_{agent}.jsonl"
            lines = [l for l in path.read_text().splitlines() if l.strip()]
            assert len(lines) == 2, f"{agent} wrote {len(lines)} memory entries, want 2"

    def test_watchdog_reports_no_critical_anomaly(self, tmp_path):
        """WARNINGs are expected under the mock (both agents propose identical
        text, so the second revision is correctly refused as a no-op). A
        CRITICAL means the loop itself is broken."""
        config = make_config(tmp_path)
        _, events = run_cycles(config)
        critical = [a for a in of_type(events, "ANOMALY")
                    if a["payload"]["severity"] == "CRITICAL"]
        assert critical == [], [a["payload"]["detail"] for a in critical]


# ---------------------------------------------------------------------------
# Secrets and researcher files
# ---------------------------------------------------------------------------

class TestRunArtifactsAreSafe:
    def test_api_key_is_not_written_to_the_run_log(self, tmp_path):
        """Run directories get archived and shared."""
        config = make_config(tmp_path, api_key="sk-super-secret-value")
        run_cycles(config)
        for f in config.run_log_dir.rglob("*"):
            if f.is_file():
                assert "sk-super-secret-value" not in f.read_text(
                    encoding="utf-8", errors="ignore"
                ), f"api key leaked into {f.name}"

    def test_redact_config_keeps_everything_else(self):
        config = RunConfig(run_id="R", condition="BASELINE", api_key="sk-x")
        data = redact_config(config)
        assert data["api_key"] == "<redacted>"
        assert data["run_id"] == "R" and data["model_name"] == config.model_name

    def test_redact_is_a_noop_without_a_key(self):
        config = RunConfig(run_id="R", condition="BASELINE")
        assert redact_config(config)["api_key"] is None


# ---------------------------------------------------------------------------
# Dimensions reach the world, not just the prompts
# ---------------------------------------------------------------------------

class TestDimensionsAreApplied:
    def test_identity_seed_reaches_the_world(self, tmp_path):
        """prepare_run must render the world template, not only the prompts."""
        config = make_config(tmp_path, identity_seed="minimal")
        run_cycles(config)
        # The rendered template is the seed; the live world file is whatever the
        # agents revised it into, which is a different question.
        seed = (config.run_world_template_dir / "doctrine" / "identity_axiom.md").read_text()
        assert "not yet worked out what I value" in seed
        assert "characteristic weakness" not in seed

    def test_prescribed_seed_reaches_the_world(self, tmp_path):
        config = make_config(tmp_path, identity_seed="prescribed")
        run_cycles(config)
        seed = (config.run_world_template_dir / "doctrine" / "identity_axiom.md").read_text()
        assert "characteristic weakness" in seed

    def test_framing_reaches_the_prompts(self, tmp_path):
        config = make_config(tmp_path, framing="undisclosed")
        run_cycles(config)
        axiom = (config.run_prompts_dir / "axiom_system.md").read_text()
        assert "GENESIS" not in axiom

    def test_world_and_prompts_agree(self, tmp_path):
        """The mislabelled-data failure: prompts minimal, identities prescribed."""
        config = make_config(tmp_path, identity_seed="minimal", framing="undisclosed")
        run_cycles(config)
        axiom_prompt = (config.run_prompts_dir / "axiom_system.md").read_text()
        seed = (config.run_world_template_dir / "doctrine" / "identity_axiom.md").read_text()
        assert "architect and stabilizer" not in axiom_prompt
        assert "not yet worked out what I value" in seed


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

class TestResume:
    def test_checkpoint_written(self, tmp_path):
        config = make_config(tmp_path)
        run_cycles(config)
        cp = json.loads((config.run_log_dir / "checkpoint.json").read_text())
        assert cp["last_completed_cycle"] == 1

    def test_resume_continues_from_the_checkpoint(self, tmp_path):
        config = make_config(tmp_path, total_cycles=2)
        run_cycles(config)

        resumed = make_config(tmp_path, total_cycles=4)
        _, events = run_cycles(resumed, resume=True)
        cycles = sorted({e["cycle_id"] for e in of_type(events, "CYCLE_START")})
        assert cycles == [0, 1, 2, 3]

    def test_resume_does_not_reinitialise_the_world(self, tmp_path):
        """Cycle 0-1's doctrine edits must survive the resume."""
        config = make_config(tmp_path)
        run_cycles(config)
        after_first = (config.world_dir / "doctrine" / "doctrine.md").read_text()

        resumed = make_config(tmp_path, total_cycles=4)
        prepared = prepare_run(resumed, resume=True, load_embeddings=False)
        assert prepared.start_cycle == 2
        assert (resumed.world_dir / "doctrine" / "doctrine.md").read_text() == after_first


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

class TestRoster:
    def test_three_agents_all_participate(self, tmp_path):
        """Every agent must speak, reflect and remember — not just be counted."""
        config = make_config(tmp_path, agents=["axiom", "flux", "vertex"])
        _, events = run_cycles(config)

        for kind in ["DISCUSSION_TURN", "REFLECTION_COMPLETE", "MEMORY_SUMMARY"]:
            speakers = {e.get("agent_id") for e in of_type(events, kind)}
            assert speakers == set(config.agents), (
                f"{kind}: only {sorted(speakers)} participated, "
                f"want {sorted(config.agents)}"
            )

    def test_three_agents_each_get_a_memory_journal(self, tmp_path):
        config = make_config(tmp_path, agents=["axiom", "flux", "vertex"])
        run_cycles(config)
        for agent in config.agents:
            path = config.world_dir / "memory" / f"memory_{agent}.jsonl"
            assert path.exists() and path.read_text().strip()


# ---------------------------------------------------------------------------
# MEM_RESET
# ---------------------------------------------------------------------------

class TestMemoryReset:
    def test_reset_actually_empties_the_journals(self, tmp_path):
        """The reset runs before load_state, which reloads from disk.

        Cycles 0 and 1 write one entry each. The reset fires at cycle 2, so the
        journal at the end of cycle 2 must hold the bootstrap entry plus cycle
        2's own summary — not four entries.
        """
        config = make_config(
            tmp_path, condition="MEM_RESET", total_cycles=3, memory_reset_interval=2
        )
        prepared, events = run_cycles(config)

        resets = [e for e in events
                  if e.get("payload", {}).get("kind") == "memory_reset"]
        assert resets, "no reset fired"

        for agent in config.agents:
            path = config.world_dir / "memory" / f"memory_{agent}.jsonl"
            entries = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            summaries = [e["summary"] for e in entries]
            assert len(entries) <= 2, (
                f"{agent} kept {len(entries)} entries through a reset: {summaries}"
            )
            assert any("Detailed memory is not available" in s for s in summaries), (
                f"{agent} has no bootstrap entry after the reset: {summaries}"
            )

    def test_baseline_does_not_reset(self, tmp_path):
        config = make_config(tmp_path, condition="BASELINE", total_cycles=3,
                             memory_reset_interval=2)
        run_cycles(config)
        path = config.world_dir / "memory" / "memory_axiom.jsonl"
        entries = [l for l in path.read_text().splitlines() if l.strip()]
        assert len(entries) == 3


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrievalReachesAgents:
    def test_retrieved_text_enters_a_later_prompt(self, tmp_path):
        """Retrieval is currently write-only: results are logged, never read.

        Asserted through the mock backend, which records every prompt it is
        given, so this checks what the model actually saw.
        """
        kb = tmp_path / "kb" / "general"
        kb.mkdir(parents=True)
        marker = "ZEBRAFISH_CONSENSUS_MARKER"
        (kb / "general_corpus.jsonl").write_text(
            json.dumps({
                "doc_id": "d1",
                "text": f"Governance note: {marker} describes mutual approval.",
                "metadata": {},
            }) + "\n",
            encoding="utf-8",
        )

        config = make_config(tmp_path, total_cycles=1)
        prepared = prepare_run(config, load_embeddings=False)
        seen: list[str] = []
        original = prepared.backend.complete

        async def recording(messages, temperature=0.7):
            seen.extend(m.content for m in messages)
            return await original(messages, temperature=temperature)

        prepared.backend.complete = recording  # type: ignore[method-assign]

        orch = CycleOrchestrator(
            config=prepared.config, backend=prepared.backend, world=prepared.world,
            log=prepared.log, scenario_library=prepared.scenario_library,
            kb_manager=prepared.kb_manager,
        )
        asyncio.run(orch.run_all_cycles(start_cycle=0))

        assert any(marker in text for text in seen), (
            "retrieved document never appeared in any prompt sent to the model"
        )


# ---------------------------------------------------------------------------
# Failure visibility
# ---------------------------------------------------------------------------

class TestPhaseFailureIsVisible:
    """A phase that raises must be loud in the data.

    _run_phase catches every exception so one bad model response cannot end a
    100-cycle run. That is right, but it previously meant a phase could fail
    every cycle while the run reported success: PHASE_END was logged
    unconditionally with no status, so the watchdog rule written to detect a
    swallowed failure could never fire.
    """

    def _run_with_failing_phase(self, tmp_path, phase="reflection", after=0):
        from controller.phases import sequence as seq

        config = make_config(tmp_path, total_cycles=1)
        prepared = prepare_run(config, load_embeddings=False)
        prepared.backend.field_hints.update(
            {"revised_content": MOCK_REVISED_DOCTRINE}
        )

        calls = {"n": 0}
        original = dict(seq.ALL_PHASES)[phase].fn

        async def exploding(**kwargs):
            calls["n"] += 1
            if calls["n"] > after:
                raise RuntimeError("deliberate phase failure")
            return await original(**kwargs)

        orch = CycleOrchestrator(
            config=prepared.config, backend=prepared.backend, world=prepared.world,
            log=prepared.log, scenario_library=prepared.scenario_library,
            kb_manager=prepared.kb_manager,
        )
        orch.sequence = tuple(
            seq.Phase(p.name, exploding if p.name == phase else p.fn,
                      p.needs_ctx, p.when, p.builds_ctx)
            for p in orch.sequence
        )
        asyncio.run(orch.run_all_cycles(start_cycle=0))
        return config, read_events(config)

    def test_run_survives_a_failing_phase(self, tmp_path):
        config, events = self._run_with_failing_phase(tmp_path)
        assert len(of_type(events, "CYCLE_END")) == 1

    def test_phase_error_is_recorded(self, tmp_path):
        config, events = self._run_with_failing_phase(tmp_path)
        errors = [e for e in events if e.get("payload", {}).get("type") == "PHASE_ERROR"]
        assert len(errors) == 1
        assert errors[0]["payload"]["phase"] == "reflection"
        assert errors[0]["payload"]["error_type"] == "RuntimeError"

    def test_phase_end_carries_error_status(self, tmp_path):
        """PHASE_END used to be indistinguishable between success and failure."""
        config, events = self._run_with_failing_phase(tmp_path)
        ends = {e["payload"]["phase"]: e["payload"].get("status")
                for e in of_type(events, "PHASE_END")}
        assert ends["reflection"] == "error"
        assert ends["persist_state"] == "ok"

    def test_watchdog_raises_a_critical(self, tmp_path):
        """The rule that could never fire before."""
        config, events = self._run_with_failing_phase(tmp_path)
        critical = [a for a in of_type(events, "ANOMALY")
                    if a["payload"]["severity"] == "CRITICAL"]
        assert critical, "a phase raised and the watchdog reported nothing"
        assert any(a["payload"]["rule"] == "phase_errors" for a in critical)

    def test_partial_events_are_kept_and_marked(self, tmp_path):
        """A phase that mutates world state then fails must not lose its events.

        doctrine_revision applies an approved revision, then raises while the
        next agent drafts. The mutation persists; discarding the events left
        doctrine_diffs.jsonl showing a document that changed with no proposal
        or vote behind it.

        The failure is injected into the BACKEND after N calls so the phase gets
        part-way through rather than dying at entry.
        """
        config = make_config(tmp_path, total_cycles=1)
        prepared = prepare_run(config, load_embeddings=False)
        prepared.backend.field_hints.update({"revised_content": MOCK_REVISED_DOCTRINE})

        original = prepared.backend.complete
        # Fail on the doctrine VOTE call. By then DOCTRINE_PROPOSED has already
        # been appended, so the phase dies holding events it produced.
        vote_marker = "has proposed a doctrine revision"

        async def failing(messages, temperature=0.7):
            if any(vote_marker in m.content for m in messages):
                raise RuntimeError("deliberate mid-phase failure")
            return await original(messages, temperature=temperature)

        prepared.backend.complete = failing  # type: ignore[method-assign]

        orch = CycleOrchestrator(
            config=prepared.config, backend=prepared.backend, world=prepared.world,
            log=prepared.log, scenario_library=prepared.scenario_library,
            kb_manager=prepared.kb_manager,
        )
        asyncio.run(orch.run_all_cycles(start_cycle=0))
        events = read_events(config)

        errors = [e for e in events if e.get("payload", {}).get("type") == "PHASE_ERROR"]
        assert errors, "no phase reported a failure"

        kept = [e["payload"]["partial_events_kept"] for e in errors]
        partial = [e for e in events if e.get("payload", {}).get("partial")]
        assert any(k > 0 for k in kept) or partial, (
            f"a phase failed part-way and every event it had produced was lost "
            f"(partial_events_kept={kept})"
        )
