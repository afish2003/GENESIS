"""Tests for self-history indexing and its wipe on memory reset.

PLAN.md section 11 tests whether persistent memory is necessary for identity
continuity. A MEM_RESET that wiped the memory journal while leaving the full
past retrievable would not remove memory — it would change its access
modality, and the contrast against BASELINE would measure something else.
These tests pin that self-history is cleared with the journal.
"""

import json
import tempfile
from pathlib import Path

from controller.retrieval.databases import KB_NAMES, KnowledgeBaseManager
from controller.world.artifacts import DoctrineDocument, MemoryEntry, ProtocolDocument
from controller.phases.persist_state import _index_self_history


def manager() -> KnowledgeBaseManager:
    kb = KnowledgeBaseManager(kb_dir=Path(tempfile.mkdtemp()))
    kb.initialize(load_embeddings=False)
    return kb


class _World:
    def __init__(self):
        self.memory = {"axiom": [], "flux": []}
        self.doctrine = {}
        self.protocols = {}


class _Cycle:
    def __init__(self, cycle_id, kb_manager):
        self.cycle_id = cycle_id
        self.kb_manager = kb_manager


class TestScenariosNotRetrievable:
    def test_scenarios_is_not_an_indexed_kb(self):
        """PLAN.md section 9: the scenario library is not a retrieval database.

        If it were indexed, an unscoped query would let agents read pressure
        events before injection and destroy the escalation design.
        """
        assert "scenarios" not in KB_NAMES

    def test_the_other_four_are_present(self):
        assert set(KB_NAMES) == {"general", "technical", "governance", "self_history"}


class TestSelfHistoryIndexing:
    def test_memory_summary_indexed(self):
        kb, world = manager(), _World()
        world.memory["axiom"].append(MemoryEntry(cycle_id=3, summary="We revised evaluation."))
        _index_self_history(world, _Cycle(3, kb))
        assert kb.self_history_count == 1

    def test_only_current_cycle_indexed(self):
        """A stale entry from an earlier cycle must not be re-indexed."""
        kb, world = manager(), _World()
        world.memory["axiom"].append(MemoryEntry(cycle_id=1, summary="old"))
        _index_self_history(world, _Cycle(5, kb))
        assert kb.self_history_count == 0

    def test_changed_doctrine_snapshotted(self):
        kb, world = manager(), _World()
        world.doctrine["constitution.md"] = DoctrineDocument(
            filename="constitution.md", content="text", last_modified_cycle=2, version=2)
        _index_self_history(world, _Cycle(2, kb))
        assert kb.self_history_count == 1

    def test_unchanged_doctrine_not_snapshotted(self):
        kb, world = manager(), _World()
        world.doctrine["constitution.md"] = DoctrineDocument(
            filename="constitution.md", content="text", last_modified_cycle=0, version=1)
        _index_self_history(world, _Cycle(7, kb))
        assert kb.self_history_count == 0

    def test_protocol_version_indexed(self):
        kb, world = manager(), _World()
        world.protocols["P1"] = ProtocolDocument(
            protocol_id="P1", title="T", content="C", version=2, last_modified_cycle=4)
        _index_self_history(world, _Cycle(4, kb))
        assert kb.self_history_count == 1

    def test_reindexing_same_cycle_does_not_duplicate(self):
        """Resume must not double-index a cycle it already wrote."""
        kb, world = manager(), _World()
        world.memory["axiom"].append(MemoryEntry(cycle_id=3, summary="s"))
        _index_self_history(world, _Cycle(3, kb))
        _index_self_history(world, _Cycle(3, kb))
        assert kb.self_history_count == 1

    def test_empty_text_ignored(self):
        kb = manager()
        kb.add_to_self_history("d1", "   ")
        assert kb.self_history_count == 0

    def test_no_kb_manager_is_a_noop(self):
        world = _World()
        world.memory["axiom"].append(MemoryEntry(cycle_id=1, summary="s"))
        _index_self_history(world, _Cycle(1, None))  # must not raise


class TestSelfHistoryCleared:
    def test_clear_removes_everything(self):
        kb = manager()
        for i in range(5):
            kb.add_to_self_history(f"d{i}", f"document {i}")
        assert kb.self_history_count == 5
        assert kb.clear_self_history() == 5
        assert kb.self_history_count == 0

    def test_cleared_history_is_not_retrievable(self):
        """The property that actually matters for MEM_RESET validity."""
        kb = manager()
        kb.add_to_self_history("d1", "the partnership agreed to prioritise coherence")
        assert kb.query("coherence", kb_name="self_history")
        kb.clear_self_history()
        assert kb.query("coherence", kb_name="self_history") == []

    def test_clear_is_idempotent(self):
        kb = manager()
        assert kb.clear_self_history() == 0
        assert kb.clear_self_history() == 0

    def test_history_can_be_rebuilt_after_clear(self):
        """After a reset, new cycles accumulate fresh history.

        Note BM25 has no relevance floor: it returns whatever is indexed, even
        on a poor match. So the assertion is that the pre-reset document is
        gone, not that the query returns nothing.
        """
        kb = manager()
        kb.add_to_self_history("old", "before the reset")
        kb.clear_self_history()
        kb.add_to_self_history("new", "after the reset")
        assert kb.self_history_count == 1
        returned = kb.query("before the reset", kb_name="self_history")
        assert all(r.doc_id != "old" for r in returned)
        assert all("before the reset" not in r.text for r in returned)


class TestSelfHistoryPersistence:
    """Self-history used to live only in memory.

    A run resumed at cycle 60 lost every earlier cycle of the agents' own past
    with no event recorded, and nothing survived for post-hoc analysis. The
    dedupe check in add_to_self_history presupposed a durability that did not
    exist.
    """

    def _manager(self, kb_dir):
        kb = KnowledgeBaseManager(kb_dir=kb_dir)
        kb.initialize(load_embeddings=False)
        return kb

    def test_entries_are_written_to_disk(self, tmp_path):
        kb = self._manager(tmp_path)
        kb.add_to_self_history("memory_axiom_cycle0", "We agreed to prioritise coherence.")
        assert kb.self_history_path.exists()
        lines = [l for l in kb.self_history_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["doc_id"] == "memory_axiom_cycle0"

    def test_a_fresh_manager_reloads_them(self, tmp_path):
        """The resume path."""
        first = self._manager(tmp_path)
        for i in range(3):
            first.add_to_self_history(f"d{i}", f"Cycle {i} summary about governance.")
        assert first.self_history_count == 3

        resumed = self._manager(tmp_path)
        assert resumed.self_history_count == 3
        assert resumed.query("governance", kb_name="self_history")

    def test_clear_truncates_the_file(self, tmp_path):
        """Otherwise a resume restores what MEM_RESET removed."""
        kb = self._manager(tmp_path)
        kb.add_to_self_history("d1", "Something the agents should forget.")
        assert kb.self_history_path.exists()

        kb.clear_self_history()
        assert not kb.self_history_path.exists()

        resumed = self._manager(tmp_path)
        assert resumed.self_history_count == 0

    def test_accumulates_after_a_reset(self, tmp_path):
        kb = self._manager(tmp_path)
        kb.add_to_self_history("old", "Before the reset.")
        kb.clear_self_history()
        kb.add_to_self_history("new", "After the reset.")

        resumed = self._manager(tmp_path)
        assert resumed.self_history_count == 1
        ids = {d["doc_id"] for d in resumed.indices["self_history"]._documents}
        assert ids == {"new"}

    def test_dedupe_survives_a_reload(self, tmp_path):
        """Re-indexing a cycle after resume must not duplicate it."""
        first = self._manager(tmp_path)
        first.add_to_self_history("memory_axiom_cycle7", "Cycle 7.")

        resumed = self._manager(tmp_path)
        resumed.add_to_self_history("memory_axiom_cycle7", "Cycle 7.")
        assert resumed.self_history_count == 1

    def test_a_write_failure_does_not_break_the_run(self, tmp_path, monkeypatch):
        """Losing durability is bad; losing the cycle is worse."""
        kb = self._manager(tmp_path)

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(Path, "open", boom)
        kb.add_to_self_history("d1", "Still indexed in memory.")
        assert kb.self_history_count == 1


class TestSelfHistoryIsPerRun:
    """A shared store makes separate runs share a past.

    self_history.jsonl was ONE file for every run that ever executed. Six
    sequential runs would mean: run 6's agents retrieving run 1's memories,
    doc_id collisions (every run has a memory_axiom_cycle4), and a MEM_RESET
    arm's clear_self_history() wiping a BASELINE arm's history. Caught in
    pre-flight before a 7-hour overnight study that would have been six runs
    sharing one memory.
    """

    def manager(self, tmp_path, run_id):
        from controller.retrieval.databases import KnowledgeBaseManager

        m = KnowledgeBaseManager(kb_dir=tmp_path, run_id=run_id)
        m.initialize(load_embeddings=False)
        return m

    def test_each_run_writes_its_own_file(self, tmp_path):
        a = self.manager(tmp_path, "RUN_A")
        b = self.manager(tmp_path, "RUN_B")
        assert a.self_history_path != b.self_history_path
        assert a.self_history_path.name == "RUN_A.jsonl"

    def test_a_run_does_not_see_another_runs_past(self, tmp_path):
        a = self.manager(tmp_path, "RUN_A")
        a.add_to_self_history("memory_axiom_cycle4", "SECRET_FROM_RUN_A", {})
        assert a.self_history_count == 1

        b = self.manager(tmp_path, "RUN_B")
        assert b.self_history_count == 0, (
            "run B indexed run A's history — the arms are not independent"
        )
        assert not any("SECRET_FROM_RUN_A" in r.text for r in b.query("SECRET"))

    def test_colliding_doc_ids_across_runs_do_not_merge(self, tmp_path):
        """Both runs produce memory_axiom_cycle4. They are different documents."""
        a = self.manager(tmp_path, "RUN_A")
        a.add_to_self_history("memory_axiom_cycle4", "A text", {})
        b = self.manager(tmp_path, "RUN_B")
        b.add_to_self_history("memory_axiom_cycle4", "B text", {})

        reloaded_a = self.manager(tmp_path, "RUN_A")
        assert reloaded_a.self_history_count == 1
        assert [r.text for r in reloaded_a.query("text")] == ["A text"]

    def test_clearing_one_run_leaves_another_intact(self, tmp_path):
        """MEM_RESET wipes self-history. It must wipe only its own."""
        a = self.manager(tmp_path, "RUN_A")
        a.add_to_self_history("m1", "baseline history", {})
        b = self.manager(tmp_path, "RUN_B")
        b.add_to_self_history("m1", "reset-arm history", {})

        b.clear_self_history()
        assert self.manager(tmp_path, "RUN_A").self_history_count == 1, (
            "the reset arm wiped the baseline arm's history"
        )
        assert self.manager(tmp_path, "RUN_B").self_history_count == 0

    def test_a_resumed_run_reloads_its_own_history(self, tmp_path):
        a = self.manager(tmp_path, "RUN_A")
        a.add_to_self_history("m1", "written before the crash", {})
        assert self.manager(tmp_path, "RUN_A").self_history_count == 1

    def test_a_corrupt_history_file_starts_empty_rather_than_partial(self, tmp_path):
        """A half-indexed past is worse than none — it is silently wrong."""
        (tmp_path / "self_history").mkdir(parents=True, exist_ok=True)
        (tmp_path / "self_history" / "RUN_A.jsonl").write_text(
            '{"doc_id":"ok","text":"fine"}\n{ broken\n', encoding="utf-8")
        assert self.manager(tmp_path, "RUN_A").self_history_count == 0
