"""Tests for self-history indexing and its wipe on memory reset.

PLAN.md section 11 tests whether persistent memory is necessary for identity
continuity. A MEM_RESET that wiped the memory journal while leaving the full
past retrievable would not remove memory — it would change its access
modality, and the contrast against BASELINE would measure something else.
These tests pin that self-history is cleared with the journal.
"""

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
