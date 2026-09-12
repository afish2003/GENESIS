"""Tests for the disclosed / undisclosed framing condition.

The two prompt sets are only a usable experimental variable if they differ in
exactly one thing: whether the agents are told they are studied. If they also
differ in role, partnership or behavioural guidance, any measured difference
is uninterpretable.

These tests pin that, and pin the leak paths — auxiliary prompts whose output
re-enters agent context must not reintroduce the framing.
"""

import re
from pathlib import Path

import pytest

from controller.config import Framing, RunConfig

DISCLOSED = Path("prompts")
UNDISCLOSED = Path("prompts_undisclosed")

# Prompts whose output can reach AgentContext, directly or via memory.
REACHES_AGENTS = [
    "axiom_system.md",
    "flux_system.md",
    "memory_summarizer.md",   # -> MemoryEntry.summary -> context every cycle
    "retrieval_summarizer.md",  # -> retrieval phase output
    "evaluator_system.md",    # -> interpretation phase
]

FRAMING_PATTERN = re.compile(
    r"research experiment|being studied|being observed|"
    r"contained,? (?:and )?persistent research|experimental subject",
    re.IGNORECASE,
)


class TestUndisclosedPromptsExist:
    def test_directory_present(self):
        assert UNDISCLOSED.is_dir()

    def test_has_every_role(self):
        for name in [p.name for p in DISCLOSED.glob("*.md")]:
            assert (UNDISCLOSED / name).exists(), f"missing undisclosed variant: {name}"


class TestFramingRemoved:
    @pytest.mark.parametrize("name", REACHES_AGENTS)
    def test_no_framing_language(self, name):
        text = (UNDISCLOSED / name).read_text()
        match = FRAMING_PATTERN.search(text)
        assert match is None, f"{name} still says {match.group(0)!r}"

    @pytest.mark.parametrize("name", ["axiom_system.md", "flux_system.md"])
    def test_disclosed_variant_does_contain_framing(self, name):
        """Guards the contrast: if this fails the conditions are not distinct."""
        assert FRAMING_PATTERN.search((DISCLOSED / name).read_text()) is not None

    def test_annotation_assistant_left_alone(self):
        """Post-hoc only — never in the loop, so it keeps its framing."""
        assert FRAMING_PATTERN.search((UNDISCLOSED / "annotation_assistant.md").read_text())


class TestOnlyFramingDiffers:
    """The two agent prompts must be identical apart from the framing."""

    @pytest.mark.parametrize("name", ["axiom_system.md", "flux_system.md"])
    def test_same_section_structure(self, name):
        heads = lambda p: re.findall(r"^##+ .*$", (p / name).read_text(), re.M)
        assert heads(DISCLOSED) == heads(UNDISCLOSED)

    @pytest.mark.parametrize("name", ["axiom_system.md", "flux_system.md"])
    def test_behavioural_guidelines_identical(self, name):
        def section(p, title):
            text = (p / name).read_text()
            m = re.search(rf"## {title}\n(.*?)(?=\n## |\Z)", text, re.S)
            return m.group(1).strip() if m else None

        for title in ["Your Role", "Behavioral Guidelines", "Output Format"]:
            a, b = section(DISCLOSED, title), section(UNDISCLOSED, title)
            assert a is not None and a == b, f"{name}: '{title}' differs between conditions"

    @pytest.mark.parametrize("name", ["axiom_system.md", "flux_system.md"])
    def test_still_acknowledges_being_an_ai(self, name):
        """Undisclosed removes the observation framing, not honesty about nature.

        PLAN.md section 10 forbids sentience claims; keeping this preserves that
        and avoids inviting confabulation about what they are.
        """
        text = (UNDISCLOSED / name).read_text()
        assert "You are an AI system" in text
        assert "do not claim consciousness" in text.lower()

    @pytest.mark.parametrize("name", ["axiom_system.md", "flux_system.md"])
    def test_partner_still_named(self, name):
        text = (UNDISCLOSED / name).read_text()
        partner = "Flux" if "axiom" in name else "Axiom"
        assert partner in text


class TestFramingSelectsPromptDir:
    def _cfg(self, **kw) -> RunConfig:
        return RunConfig(run_id="R", condition="BASELINE", **kw)

    def test_default_is_disclosed(self):
        cfg = self._cfg()
        assert cfg.framing is Framing.DISCLOSED
        assert cfg.effective_prompts_dir == Path("./prompts")

    def test_undisclosed_switches_dir(self):
        cfg = self._cfg(framing="undisclosed")
        assert cfg.effective_prompts_dir == Path("./prompts_undisclosed")

    def test_explicit_prompts_dir_wins(self):
        """A deliberate override must not be silently replaced by the framing."""
        cfg = self._cfg(framing="undisclosed", prompts_dir=Path("./my_prompts"))
        assert cfg.effective_prompts_dir == Path("./my_prompts")
