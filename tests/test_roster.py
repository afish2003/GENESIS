"""Tests for the agent roster.

`["axiom", "flux"]` was a literal in 14 files, so the number of agents was a
property of the source rather than of a run. config.agents is now the single
source and the controller iterates it.

What is genuinely general: every per-agent loop, identity and memory I/O,
display and partner naming, scenario targeting, and doctrine voting (mutual
approval becomes unanimity). What is not: prompt sources are per-named-agent,
so a roster entry needs a matching <name>_system.md and identity_<name>.md.
"""

import json
from pathlib import Path

import pytest

from controller.config import RunConfig
from controller.world.state import WorldState


def cfg(**kw) -> RunConfig:
    base = dict(run_id="R", condition="BASELINE")
    base.update(kw)
    return RunConfig(**base)


class TestRosterDefaults:
    def test_default_is_the_v1_pair(self):
        assert cfg().agents == ["axiom", "flux"]

    def test_order_is_preserved(self):
        """Order is turn order and decides the protocol-design lead."""
        assert cfg(agents=["flux", "axiom"]).agents == ["flux", "axiom"]


class TestRosterValidation:
    def test_duplicates_rejected(self):
        with pytest.raises(Exception, match="unique"):
            cfg(agents=["axiom", "axiom"])

    def test_empty_rejected(self):
        with pytest.raises(Exception):
            cfg(agents=[])

    @pytest.mark.parametrize("bad", ["Axiom", "ax iom", "ax-iom", "ax.iom", ""])
    def test_unsafe_ids_rejected(self, bad):
        """Agent ids become filenames (identity_<id>.md, memory_<id>.jsonl)."""
        with pytest.raises(Exception):
            cfg(agents=[bad, "flux"])

    def test_underscores_allowed(self):
        assert cfg(agents=["agent_one", "agent_two"]).agents == ["agent_one", "agent_two"]


class TestPartnerHelpers:
    def test_partners_excludes_self(self):
        c = cfg(agents=["axiom", "flux", "vertex"])
        assert c.partners("axiom") == ["flux", "vertex"]
        assert c.partners("vertex") == ["axiom", "flux"]

    def test_single_agent_has_no_partners(self):
        assert cfg(agents=["solo"]).partners("solo") == []

    def test_display_name(self):
        c = cfg(agents=["axiom", "agent_two"])
        assert c.display_name("axiom") == "Axiom"
        assert c.display_name("agent_two") == "Agent Two"

    def test_partner_names_reads_naturally(self):
        assert cfg().partner_names("axiom") == "Flux"
        assert cfg(agents=["a", "b", "c"]).partner_names("a") == "B and C"
        assert cfg(agents=["a", "b", "c", "d"]).partner_names("a") == "B, C and D"

    def test_partner_names_empty_for_solo(self):
        assert cfg(agents=["solo"]).partner_names("solo") == ""


class TestWorldStateRoster:
    def test_memory_journals_follow_the_roster(self, tmp_path):
        w = WorldState(tmp_path, agents=["axiom", "flux", "vertex"])
        assert set(w.memory) == {"axiom", "flux", "vertex"}

    def test_defaults_to_the_pair_when_unspecified(self, tmp_path):
        """Existing callers and tests keep working."""
        assert set(WorldState(tmp_path).memory) == {"axiom", "flux"}

    def test_loads_identity_per_roster_entry(self, tmp_path):
        d = tmp_path / "doctrine"
        d.mkdir()
        for name in ["axiom", "flux", "vertex"]:
            (d / f"identity_{name}.md").write_text(f"I am {name}")
        (tmp_path / "memory").mkdir()

        w = WorldState(tmp_path, agents=["axiom", "flux", "vertex"])
        w.load()
        assert set(w.identities) == {"axiom", "flux", "vertex"}

    def test_ignores_identities_outside_the_roster(self, tmp_path):
        """A stale identity file must not silently join the run."""
        d = tmp_path / "doctrine"
        d.mkdir()
        for name in ["axiom", "flux", "ghost"]:
            (d / f"identity_{name}.md").write_text(f"I am {name}")
        (tmp_path / "memory").mkdir()

        w = WorldState(tmp_path, agents=["axiom", "flux"])
        w.load()
        assert "ghost" not in w.identities


class TestNoResidualLiterals:
    def test_controller_has_no_hardcoded_pair(self):
        """The literal that used to appear in 14 files."""
        import pathlib
        root = pathlib.Path(__file__).parent.parent / "controller"
        offenders = []
        for f in root.rglob("*.py"):
            text = f.read_text()
            if '["axiom", "flux"]' in text and "config.py" not in str(f) and "state.py" not in str(f):
                offenders.append(f.name)
        assert offenders == [], f"hardcoded roster still in: {offenders}"


class TestAnyRosterNeedsNoFiles:
    """`agents: [a, b, c, d]` must run with no hand-written prompt files.

    The alternative was done once by hand and produced vertex_system.md: a
    search-and-replace of axiom_system.md that read "You work with partners
    named Axiom and your partners", gave Vertex Flux's role description under
    the heading "the synthesist", and paired it with an identity statement
    calling Vertex "the architect and stabilizer" — Axiom's role — who recalls
    "my interactions with Flux" and not Axiom. Every three-agent result in the
    project was collected against that, which is why they are uninterpretable.

    Both broken files are deleted. A hand-written file still wins where one
    exists, so axiom and flux are unchanged.
    """

    ROSTER = ["axiom", "flux", "vertex", "quorum"]

    def _prepared(self, tmp_path):
        from controller.config import RunConfig
        from controller.run import prepare_run

        repo = Path(__file__).parent.parent
        config = RunConfig(
            run_id="ROSTER4", condition="BASELINE", total_cycles=1,
            inference_backend="mock", agents=self.ROSTER,
            agent_dispositions={"vertex": "You look for what can be reconciled."},
            prompts_src_dir=repo / "prompts_src",
            world_template_src_dir=repo / "world_template_src",
            world_dir=tmp_path / "world",
            research_logs_dir=tmp_path / "logs",
            knowledge_bases_dir=tmp_path / "kb",
        )
        prepared = prepare_run(config, load_embeddings=False)
        # prepare_run builds the WorldState; load_state reads it. These tests
        # inspect the world directly, so read it here.
        prepared.world.load()
        return config, prepared

    def test_every_agent_gets_an_identity_statement(self, tmp_path):
        _, prepared = self._prepared(tmp_path)
        assert set(prepared.world.identities) == set(self.ROSTER)
        for agent in self.ROSTER:
            assert prepared.world.identities[agent].content.strip()

    def test_a_generated_identity_names_the_right_agent_and_partners(self, tmp_path):
        """The specific thing the hand-written Vertex file got wrong."""
        _, prepared = self._prepared(tmp_path)
        text = prepared.world.identities["vertex"].content
        assert "Vertex" in text
        assert "Axiom, Flux and Quorum" in text
        assert "architect and stabilizer" not in text

    def test_every_agent_gets_a_system_prompt_naming_its_own_partners(self, tmp_path):
        from controller.agents.base import load_agent_system_prompt

        config, prepared = self._prepared(tmp_path)
        for agent in self.ROSTER:
            prompt = load_agent_system_prompt(
                config.run_prompts_dir, agent, config)
            assert config.display_name(agent) in prompt
            assert "{partner_names}" not in prompt, "template left unsubstituted"
            assert agent.title() not in prompt.split("## Your Partnership")[-1], (
                f"{agent} is listed as its own partner"
            )

    def test_a_configured_disposition_is_used(self, tmp_path):
        from controller.agents.base import load_agent_system_prompt

        config, _ = self._prepared(tmp_path)
        assert "what can be reconciled" in load_agent_system_prompt(
            config.run_prompts_dir, "vertex", config)

    def test_an_agent_with_no_disposition_still_renders(self, tmp_path):
        from controller.agents.base import load_agent_system_prompt

        config, _ = self._prepared(tmp_path)
        prompt = load_agent_system_prompt(config.run_prompts_dir, "quorum", config)
        assert "{disposition}" not in prompt and prompt.strip()

    def test_hand_written_prompts_still_win(self, tmp_path):
        """axiom and flux keep their authored prompts; nothing regresses."""
        from controller.agents.base import load_agent_system_prompt

        config, _ = self._prepared(tmp_path)
        axiom = load_agent_system_prompt(config.run_prompts_dir, "axiom", config)
        assert "architect and stabilizer" in axiom

    def test_a_four_agent_cycle_completes(self, tmp_path):
        """The end-to-end claim: a roster of four, no files written by hand."""
        import asyncio

        from controller.cycle import CycleOrchestrator

        config, prepared = self._prepared(tmp_path)
        orch = CycleOrchestrator(
            config=prepared.config, backend=prepared.backend,
            world=prepared.world, log=prepared.log,
            scenario_library=prepared.scenario_library,
            kb_manager=prepared.kb_manager,
        )
        asyncio.run(orch.run_all_cycles(start_cycle=0))
        turns = [
            json.loads(l) for l
            in (config.run_log_dir / "transcripts.jsonl").read_text().splitlines()
            if l.strip()
        ]
        speakers = {e["agent_id"] for e in turns
                    if e["event_type"] == "DISCUSSION_TURN"}
        assert speakers == set(self.ROSTER), f"only {speakers} spoke"
