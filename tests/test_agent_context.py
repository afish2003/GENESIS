"""Tests for the thing that assembles every prompt in the system.

controller/agents/base.py was imported by no test file. `build_system_message`
concatenates the system prompt, identity, retrieved context, memory window and
doctrine into the message every agent receives every cycle — and nothing
asserted any of it.

The 10-entry memory window matters most. It is the ONLY place MEM_RESET's
effect on context is realised, and it interacts with the reset interval in a
way nobody has measured: the prompt shows `memory[-10:]` and the reset interval
is also 10, so in the cycle before a reset a BASELINE agent and a MEM_RESET
agent see nearly the same context. The manipulation's effective dose is far
smaller than "memory vs no memory" implies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from controller.agents.base import AgentContext, build_agent_context, fill_roster
from controller.config import RunConfig
from controller.world.artifacts import IdentityStatement, MemoryEntry

REPO = Path(__file__).parent.parent


def entry(cycle: int, summary: str = "") -> MemoryEntry:
    return MemoryEntry(cycle_id=cycle, summary=summary or f"cycle {cycle} happened",
                       key_events=[], relationship_note="")


def context(memory=None, retrieved="", identity="I am Axiom.") -> AgentContext:
    return AgentContext(
        agent_id="axiom",
        system_prompt="SYSTEM_PROMPT_MARKER",
        identity=IdentityStatement(agent_id="axiom", content=identity),
        memory=memory or [],
        doctrine_context="DOCTRINE_MARKER",
    )


class TestEveryStrandReachesTheAgent:
    def test_the_system_prompt_is_included(self):
        assert "SYSTEM_PROMPT_MARKER" in context().build_system_message().content

    def test_the_identity_is_included(self):
        msg = context(identity="IDENTITY_MARKER").build_system_message().content
        assert "IDENTITY_MARKER" in msg

    def test_the_doctrine_is_included(self):
        assert "DOCTRINE_MARKER" in context().build_system_message().content

    def test_retrieved_context_is_included_when_present(self):
        ctx = context()
        ctx.retrieved_context = "RETRIEVED_MARKER"
        assert "RETRIEVED_MARKER" in ctx.build_system_message().content

    def test_no_retrieval_section_when_nothing_was_retrieved(self):
        """An empty heading would tell the agent retrieval happened and found
        nothing, which is different from not having run."""
        assert "Retrieved This Cycle" not in context().build_system_message().content

    def test_the_message_is_a_system_message(self):
        assert context().build_system_message().role == "system"


class TestTheMemoryWindow:
    """memory[-10:] is where MEM_RESET's effect on context actually happens."""

    def test_recent_memory_appears(self):
        msg = context(memory=[entry(0, "MEMORY_MARKER")]).build_system_message().content
        assert "MEMORY_MARKER" in msg

    def test_only_the_last_ten_entries_are_shown(self):
        msg = context(memory=[entry(i) for i in range(30)]).build_system_message().content
        assert "cycle 29 happened" in msg
        assert "cycle 20 happened" in msg
        assert "cycle 19 happened" not in msg, "the window is wider than 10"

    def test_an_agent_with_no_memory_gets_no_memory_section(self):
        assert "Memory Journal" not in context().build_system_message().content

    def test_the_window_equals_the_default_reset_interval(self):
        """Not a bug, but the thing that makes the manipulation weak, and it
        should fail loudly if either number moves without the other being
        reconsidered. In the cycle before a reset, BASELINE and MEM_RESET
        agents see nearly identical context.
        """
        from controller.agents.base import AgentContext as AC

        default_interval = RunConfig(
            run_id="X", condition="BASELINE").memory_reset_interval
        shown = len([e for e in [entry(i) for i in range(30)]][-10:])
        assert shown == default_interval == 10, (
            "the prompt memory window and memory_reset_interval have diverged; "
            "revisit what MEM_RESET actually removes"
        )


class TestRosterSubstitution:
    def test_placeholders_are_filled(self):
        config = RunConfig(run_id="X", condition="BASELINE",
                           agents=["axiom", "flux", "vertex"])
        out = fill_roster("{display_name} works with {partner_names}. {disposition}",
                          "vertex", config)
        assert out.startswith("Vertex works with Axiom and Flux.")
        assert "{" not in out

    def test_an_agent_is_never_its_own_partner(self):
        config = RunConfig(run_id="X", condition="BASELINE",
                           agents=["axiom", "flux", "vertex"])
        for agent in config.agents:
            partners = fill_roster("{partner_names}", agent, config)
            assert config.display_name(agent) not in partners

    def test_a_single_agent_roster_does_not_produce_an_empty_sentence(self):
        config = RunConfig(run_id="X", condition="BASELINE", agents=["solo"])
        assert fill_roster("{partner_names}", "solo", config) == "no one yet"


class TestBuildAgentContext:
    def test_a_hand_written_prompt_is_used_when_present(self, tmp_path):
        config = RunConfig(run_id="X", condition="BASELINE")
        (tmp_path / "axiom_system.md").write_text("HANDWRITTEN", encoding="utf-8")
        ctx = build_agent_context(
            "axiom", tmp_path, IdentityStatement(agent_id="axiom", content="i"),
            [], {}, config=config)
        assert ctx.system_prompt == "HANDWRITTEN"

    def test_the_generic_template_is_used_otherwise(self, tmp_path):
        config = RunConfig(run_id="X", condition="BASELINE",
                           agents=["axiom", "newcomer"])
        (tmp_path / "agent_system.md").write_text(
            "I am {display_name}, working with {partner_names}.", encoding="utf-8")
        ctx = build_agent_context(
            "newcomer", tmp_path, IdentityStatement(agent_id="newcomer", content="i"),
            [], {}, config=config)
        assert ctx.system_prompt == "I am Newcomer, working with Axiom."

    def test_a_missing_prompt_raises_rather_than_producing_an_empty_one(self, tmp_path):
        config = RunConfig(run_id="X", condition="BASELINE")
        with pytest.raises(FileNotFoundError):
            build_agent_context(
                "ghost", tmp_path, IdentityStatement(agent_id="ghost", content="i"),
                [], {}, config=config)

    def test_doctrine_documents_are_sorted_for_stability(self, tmp_path):
        """Unordered doctrine would change the prompt between cycles for no
        reason, which is invisible and confounds everything."""
        config = RunConfig(run_id="X", condition="BASELINE")
        (tmp_path / "axiom_system.md").write_text("s", encoding="utf-8")
        ctx = build_agent_context(
            "axiom", tmp_path, IdentityStatement(agent_id="axiom", content="i"),
            [], {"zebra.md": "Z", "alpha.md": "A"}, config=config)
        assert ctx.doctrine_context.index("alpha.md") < ctx.doctrine_context.index("zebra.md")
