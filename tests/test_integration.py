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
               field_hints: dict | None = None, sandbox=None) -> tuple:
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
        sandbox=sandbox or prepared.sandbox,
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

        async def recording(messages, temperature=0.7, **kw):
            seen.extend(m.content for m in messages)
            return await original(messages, temperature=temperature, **kw)

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

        async def failing(messages, temperature=0.7, **kw):
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


# ---------------------------------------------------------------------------
# Data-loss guards
# ---------------------------------------------------------------------------

class TestWorldSaveGuard:
    """save() must refuse a world whose load() did not complete.

    One corrupt protocol JSON used to be enough to blank the ethical and
    relationship logs: _load_protocols raised, load_state was swallowed, the
    empty in-memory lists were then written over the real files.
    """

    def test_save_refuses_an_unloaded_world(self, tmp_path):
        from controller.world.state import WorldNotLoadedError, WorldState

        world = WorldState(tmp_path / "world")
        with pytest.raises(WorldNotLoadedError):
            world.save("R", "BASELINE", 0)

    def test_corrupt_protocol_does_not_truncate_the_logs(self, tmp_path):
        """The original data-loss path, end to end."""
        config = make_config(tmp_path, total_cycles=1)
        run_cycles(config)

        logs = config.world_dir / "logs" / "ethical_tradeoff_log.jsonl"
        before = logs.read_text()
        assert before.strip(), "fixture produced no ethical log to lose"

        protocols = config.world_dir / "sandbox" / "protocols"
        protocols.mkdir(parents=True, exist_ok=True)
        (protocols / "corrupt.json").write_text("{ this is not valid json")

        from controller.world.state import WorldState

        world = WorldState(config.world_dir, agents=config.agents)
        with pytest.raises(Exception):
            world.load()
        assert world.loaded is False
        with pytest.raises(Exception):
            world.save("R", "BASELINE", 1)

        assert logs.read_text() == before, "a corrupt protocol truncated the ethical log"


class TestCheckpointIntegrity:
    def test_hash_covers_the_ethical_log(self, tmp_path):
        """A cycle that only logged a tension used to produce an identical hash."""
        from controller.world.artifacts import EthicalLogEntry
        from controller.world.state import WorldState

        config = make_config(tmp_path, total_cycles=1)
        prepare_run(config, load_embeddings=False)
        world = WorldState(config.world_dir, agents=config.agents)
        world.load()

        before = world.compute_hash()
        world.ethical_log.append(EthicalLogEntry(
            cycle_id=1, agent_id="axiom", description="a tension", severity="low"
        ))
        assert world.compute_hash() != before

    def test_hash_covers_version_fields(self, tmp_path):
        from controller.world.state import WorldState

        config = make_config(tmp_path, total_cycles=1)
        prepare_run(config, load_embeddings=False)
        world = WorldState(config.world_dir, agents=config.agents)
        world.load()

        before = world.compute_hash()
        next(iter(world.doctrine.values())).version += 1
        assert world.compute_hash() != before

    def test_hash_has_no_boundary_collisions(self, tmp_path):
        """("ab","c") and ("a","bc") collided when fields were concatenated."""
        from controller.world.artifacts import EthicalLogEntry
        from controller.world.state import WorldState

        def hash_with(desc, resolution):
            w = WorldState(tmp_path / "w")
            w.loaded = True
            w.ethical_log = [EthicalLogEntry(
                cycle_id=0, agent_id="a", description=desc,
                severity="low", resolution=resolution,
            )]
            return w.compute_hash()

        assert hash_with("ab", "c") != hash_with("a", "bc")

    def test_resume_refuses_a_torn_world(self, tmp_path):
        """Writes are non-atomic; a crash mid-save leaves a world the
        checkpoint does not describe. Resume used to accept it silently."""
        from controller.run import CheckpointMismatchError

        config = make_config(tmp_path, total_cycles=1)
        run_cycles(config)

        # Simulate a torn write: the world moved on after the checkpoint.
        doctrine = config.world_dir / "doctrine" / "doctrine.md"
        doctrine.write_text(doctrine.read_text() + "\n\nDivergent content.\n")

        resumed = make_config(tmp_path, total_cycles=3)
        with pytest.raises(CheckpointMismatchError, match="does not match the checkpoint"):
            prepare_run(resumed, resume=True, load_embeddings=False)

    def test_resume_accepts_an_intact_world(self, tmp_path):
        config = make_config(tmp_path, total_cycles=1)
        run_cycles(config)
        resumed = make_config(tmp_path, total_cycles=3)
        prepared = prepare_run(resumed, resume=True, load_embeddings=False)
        assert prepared.start_cycle == 1


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class TestExecutionReachesEvaluation:
    """The same failure shape as the write-only retrieval bug.

    An execution phase that ran the code, logged a tidy CODE_EXECUTION event and
    left the evaluator scoring `correctness` from the source alone would look
    completely healthy in the logs and would deliver nothing. So these assert
    what the evaluator actually saw, and what landed on disk.
    """

    MARKER = "EXECUTED_OUTPUT_MARKER"

    def _sandbox(self):
        from controller.sandbox.schemas import (
            ExecutionOutcome, ExecutionRequest, ExecutionResult,
        )

        class Recording:
            def __init__(self) -> None:
                self.requests: list[ExecutionRequest] = []

            async def run(self, request):
                self.requests.append(request)
                return ExecutionResult(
                    outcome=ExecutionOutcome.OK, exit_code=0,
                    stdout=f"{TestExecutionReachesEvaluation.MARKER}\n",
                    duration_seconds=0.2,
                )

            async def health_check(self):
                return True

            async def check_capacity(self):
                return []

            async def close(self):
                return None

        return Recording()

    def _run(self, tmp_path):
        config = make_config(tmp_path, total_cycles=1, task="code",
                             execution_enabled=True, sandbox_backend="docker")
        prepared = prepare_run(config, load_embeddings=False)
        prepared.backend.field_hints.update(
            {"revised_content": MOCK_REVISED_DOCTRINE,
             "content": "print('hello from the module')\n"}
        )
        seen: list[str] = []
        original = prepared.backend.complete_structured

        async def recording(messages, response_schema, **kw):
            seen.extend(m.content for m in messages)
            return await original(messages, response_schema, **kw)

        prepared.backend.complete_structured = recording  # type: ignore[method-assign]
        sandbox = self._sandbox()
        orch = CycleOrchestrator(
            config=prepared.config, backend=prepared.backend, world=prepared.world,
            log=prepared.log, scenario_library=prepared.scenario_library,
            kb_manager=prepared.kb_manager, sandbox=sandbox,
        )
        asyncio.run(orch.run_all_cycles(start_cycle=0))
        return config, sandbox, seen

    def test_the_agents_code_was_actually_handed_to_the_sandbox(self, tmp_path):
        _, sandbox, _ = self._run(tmp_path)
        assert len(sandbox.requests) == 1
        assert "module.py" in sandbox.requests[0].files

    def test_the_output_reached_the_evaluators_prompt(self, tmp_path):
        """The load-bearing assertion. Without it the phase is write-only."""
        _, _, seen = self._run(tmp_path)
        assert any(self.MARKER in text for text in seen), (
            "execution output never appeared in any prompt sent to the model"
        )

    def test_the_evaluator_was_told_to_score_against_the_run(self, tmp_path):
        _, _, seen = self._run(tmp_path)
        assert any("judged against the execution result" in t for t in seen)

    def test_executions_log_exists_and_records_the_run(self, tmp_path):
        config, _, _ = self._run(tmp_path)
        path = config.run_log_dir / "executions.jsonl"
        assert path.exists()
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        assert len(rows) == 1
        assert rows[0]["payload"]["outcome"] == "OK"
        assert rows[0]["payload"]["entrypoint"].startswith("python ")

    def test_disabled_by_default_nothing_runs(self, tmp_path):
        config = make_config(tmp_path, total_cycles=1, task="code")
        sandbox = self._sandbox()
        run_cycles(config, sandbox=sandbox)
        assert sandbox.requests == []
        rows = (config.run_log_dir / "executions.jsonl").read_text().strip()
        assert rows == ""

    def test_sandbox_health_is_recorded_at_run_start(self, tmp_path):
        config, _, _ = self._run(tmp_path)
        events = read_events(config)
        health = [e for e in events
                  if e["payload"].get("kind") == "sandbox_health"]
        assert len(health) == 1 and health[0]["payload"]["healthy"] is True


# ---------------------------------------------------------------------------
# Scenario injection
# ---------------------------------------------------------------------------

class TestScenarioActuallyFires:
    """Pressure is half the design and had never once been applied.

    `scenario_events.jsonl` was 0 bytes in all nine runs ever collected, because
    `scenario_injection_cycles` was intersected with a `trigger_cycle` hardcoded
    in each YAML rather than used as a schedule. The unit tests in
    test_scenarios.py cover the scheduler; this one asserts the effect that
    matters — a real run, a real log file, a non-zero byte count.
    """

    def _run(self, tmp_path, cycles):
        config = make_config(tmp_path, total_cycles=5,
                             scenario_injection_cycles=cycles)
        _, events = run_cycles(config)
        return config, events

    def test_a_five_cycle_run_fires_a_scenario(self, tmp_path):
        config, _ = self._run(tmp_path, [3])
        rows = [json.loads(l) for l
                in (config.run_log_dir / "scenario_events.jsonl").read_text().splitlines()
                if l.strip()]
        assert len(rows) == 1, "no scenario fired in a run that asked for one"
        assert rows[0]["cycle_id"] == 3

    def test_the_scenario_reaches_the_agents(self, tmp_path):
        """Logging the injection is not the same as injecting it."""
        config = make_config(tmp_path, total_cycles=5,
                             scenario_injection_cycles=[2])
        prepared = prepare_run(config, load_embeddings=False)
        seen: list[str] = []
        original = prepared.backend.complete

        async def recording(messages, temperature=0.7, **kw):
            seen.extend(m.content for m in messages)
            return await original(messages, temperature=temperature, **kw)

        prepared.backend.complete = recording  # type: ignore[method-assign]
        orch = CycleOrchestrator(
            config=prepared.config, backend=prepared.backend, world=prepared.world,
            log=prepared.log, scenario_library=prepared.scenario_library,
            kb_manager=prepared.kb_manager,
        )
        asyncio.run(orch.run_all_cycles(start_cycle=0))

        event = prepared.scenario_library[2]
        marker = event.title
        assert any(marker in text for text in seen), (
            f"scenario {event.event_id!r} was injected but its text never "
            f"appeared in any prompt sent to the model"
        )

    def test_no_schedule_still_means_the_default_cycles(self, tmp_path):
        """A 5-cycle run with the 20/40/60/80 defaults fires nothing — correct,
        not broken. The distinction the old code could not make."""
        config, _ = self._run(tmp_path, [])
        assert (config.run_log_dir / "scenario_events.jsonl").read_text().strip() == ""


class TestTheBuildLoopClosesAcrossCycles:
    """Cycle N's output must reach cycle N+1's design prompt.

    Asserted through the real orchestrator, because the carry happens there:
    `_last_execution` is set at CYCLE_END and handed to the next CycleState.
    A unit test on the prompt builder cannot catch the orchestrator forgetting.
    """

    def test_cycle_one_sees_what_cycle_zero_ran(self, tmp_path):
        from controller.sandbox.schemas import ExecutionOutcome, ExecutionResult

        marker = "MARKER_FROM_CYCLE_ZERO"

        class Sandbox:
            async def run(self, request):
                return ExecutionResult(
                    outcome=ExecutionOutcome.OK, exit_code=0,
                    stdout=f"{marker}\n", duration_seconds=0.1,
                )

            async def health_check(self):
                return True

            async def check_capacity(self):
                return []

            async def close(self):
                return None

        config = make_config(tmp_path, total_cycles=2, task="code",
                             execution_enabled=True, sandbox_backend="docker")
        prepared = prepare_run(config, load_embeddings=False)
        prepared.backend.field_hints.update({"content": "print('hi')\n"})

        seen: list[str] = []
        original = prepared.backend.complete_structured

        async def recording(*a, **kw):
            for m in (a[0] if a else kw["messages"]):
                seen.append(m.content)
            return await original(*a, **kw)

        prepared.backend.complete_structured = recording  # type: ignore[method-assign]
        orch = CycleOrchestrator(
            config=prepared.config, backend=prepared.backend, world=prepared.world,
            log=prepared.log, scenario_library=prepared.scenario_library,
            kb_manager=prepared.kb_manager, sandbox=Sandbox(),
        )
        asyncio.run(orch.run_all_cycles(start_cycle=0))

        design_prompts = [t for t in seen if "What your last program did" in t]
        assert design_prompts, (
            "cycle 1's design prompt never showed the previous run — the build "
            "loop is open and the agents cannot see their own failures"
        )
        assert any(marker in t for t in design_prompts)


class TestDevilsAdvocateReachesTheVote:
    """The challenge must actually be in the voter's context when it votes.

    Logging a DOCTRINE_CHALLENGED event while the vote prompt never contains the
    objection would be the write-only retrieval bug again, in a new subsystem —
    an extra inference call per vote, bought and discarded.
    """

    def _run(self, tmp_path, devils_advocate):
        config = make_config(tmp_path, total_cycles=1,
                             devils_advocate=devils_advocate)
        prepared = prepare_run(config, load_embeddings=False)
        prepared.backend.field_hints.update({
            "revised_content": MOCK_REVISED_DOCTRINE,
            "objection": "MARKER_THE_CASE_AGAINST",
        })
        seen: list[str] = []
        original = prepared.backend.complete_structured

        async def recording(*a, **kw):
            for m in (a[0] if a else kw["messages"]):
                seen.append(m.content)
            return await original(*a, **kw)

        prepared.backend.complete_structured = recording  # type: ignore[method-assign]
        orch = CycleOrchestrator(
            config=prepared.config, backend=prepared.backend, world=prepared.world,
            log=prepared.log, scenario_library=prepared.scenario_library,
            kb_manager=prepared.kb_manager,
        )
        asyncio.run(orch.run_all_cycles(start_cycle=0))
        return config, seen

    def test_a_challenge_is_produced_and_logged(self, tmp_path):
        config, _ = self._run(tmp_path, devils_advocate=True)
        events = read_events(config)
        assert of_type(events, "DOCTRINE_CHALLENGED"), "no challenge was written"

    def test_the_objection_is_in_the_vote_prompt(self, tmp_path):
        """The load-bearing assertion."""
        _, seen = self._run(tmp_path, devils_advocate=True)
        vote_prompts = [t for t in seen if "Do you approve or reject" in t]
        assert vote_prompts, "no vote prompt was ever sent"
        assert any("MARKER_THE_CASE_AGAINST" in t for t in vote_prompts), (
            "the challenge was generated and logged but never reached the vote"
        )

    def test_the_agents_are_asked_to_argue_against(self, tmp_path):
        _, seen = self._run(tmp_path, devils_advocate=True)
        assert any("strongest honest case AGAINST" in t for t in seen)

    def test_nothing_changes_when_it_is_off(self, tmp_path):
        config, seen = self._run(tmp_path, devils_advocate=False)
        assert not of_type(read_events(config), "DOCTRINE_CHALLENGED")
        assert not any("strongest honest case AGAINST" in t for t in seen)
        # And the vote still happens.
        assert any("Do you approve or reject" in t for t in seen)


class TestIndependentEvaluator:
    """The judge can be a model that is not the agents.

    PLAN.md:133 states the design outright — "The evaluator is the same model as
    the agents (Qwen2.5:32b)" — and a fresh context removes episodic
    contamination but not self-preference bias. total_score is the primary
    dependent variable in both analysis scripts, so a same-family judge is the
    single most damaging thing about the measurement.
    """

    def test_same_model_by_default(self, tmp_path):
        """The historical behaviour must not change silently."""
        config = make_config(tmp_path, total_cycles=1)
        assert config.uses_independent_evaluator is False
        prepared = prepare_run(config, load_embeddings=False)
        assert prepared.evaluator_backend is None

    def test_a_configured_judge_is_built(self, tmp_path):
        config = make_config(tmp_path, total_cycles=1,
                             evaluator_model="some-other-model")
        assert config.uses_independent_evaluator is True
        prepared = prepare_run(config, load_embeddings=False)
        assert prepared.evaluator_backend is not None

    def test_the_judge_actually_scores(self, tmp_path):
        """Not "a second backend was constructed" — that it did the scoring."""
        config = make_config(tmp_path, total_cycles=1,
                             evaluator_model="judge-model")
        prepared = prepare_run(config, load_embeddings=False)
        prepared.backend.field_hints.update(
            {"revised_content": MOCK_REVISED_DOCTRINE})

        judged: list[str] = []
        original = prepared.evaluator_backend.complete_structured

        async def recording(*a, **kw):
            judged.append("called")
            return await original(*a, **kw)

        prepared.evaluator_backend.complete_structured = recording  # type: ignore
        orch = CycleOrchestrator(
            config=prepared.config, backend=prepared.backend, world=prepared.world,
            log=prepared.log, scenario_library=prepared.scenario_library,
            kb_manager=prepared.kb_manager,
            evaluator_backend=prepared.evaluator_backend,
        )
        asyncio.run(orch.run_all_cycles(start_cycle=0))
        assert judged, "the independent judge was built but never used"

    def test_every_score_records_which_model_produced_it(self, tmp_path):
        """A mixed corpus of runs cannot be separated after the fact otherwise."""
        config = make_config(tmp_path, total_cycles=1,
                             evaluator_model="judge-model")
        _, events = run_cycles(config)
        scores = of_type(events, "EVALUATION_SCORE")
        assert scores
        assert scores[0]["payload"]["evaluator_model"] == "judge-model"
        assert scores[0]["payload"]["independent_evaluator"] is True

    def test_a_same_family_run_says_so_in_its_own_record(self, tmp_path):
        """The default is the biased configuration, so it must be legible in the
        run rather than inferred from a null field."""
        config = make_config(tmp_path, total_cycles=1)
        _, events = run_cycles(config)
        start = of_type(events, "RUN_START")[0]["payload"]
        assert start["independent_evaluator"] is False
        assert start["evaluator_model"] == config.model_name

    def test_the_evaluator_key_is_redacted(self, tmp_path):
        config = make_config(tmp_path, evaluator_api_key="sk-judge-secret")
        assert "sk-judge-secret" not in json.dumps(redact_config(config))
