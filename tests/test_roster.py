"""Tests for the agent roster.

`["axiom", "flux"]` was a literal in 14 files, so the number of agents was a
property of the source rather than of a run. config.agents is now the single
source and the controller iterates it.

What is genuinely general: every per-agent loop, identity and memory I/O,
display and partner naming, scenario targeting, and doctrine voting (mutual
approval becomes unanimity). What is not: prompt sources are per-named-agent,
so a roster entry needs a matching <name>_system.md and identity_<name>.md.
"""

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
