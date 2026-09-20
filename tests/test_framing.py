"""Tests for composable prompt dimensions.

Variants used to be duplicated directories (prompts/, prompts_undisclosed/,
prompts_minimal/). Two dimensions needed four directories and only three
existed, so `disclosed + minimal` was unrepresentable, and selecting a framing
alongside an explicit prompts_dir was silently ignored — mislabelled data with
no error. Sources now carry their variation inline and render per run.

These tests pin: all four combinations render, only the intended thing varies,
no template markup ever reaches a prompt, and the leak paths stay closed.
"""

import itertools
import re
import tempfile
from pathlib import Path

import pytest

from controller.config import Framing, IdentitySeed, RunConfig
from controller.prompts import (
    DIMENSIONS,
    PromptRenderError,
    materialise_prompts,
    materialise_world_template,
    render,
)

FRAMING_PATTERN = re.compile(
    r"research experiment|GENESIS|being studied|being observed", re.IGNORECASE
)

# Prompts whose output can reach AgentContext, directly or via memory.
REACHES_AGENTS = [
    "axiom_system.md", "flux_system.md",
    "agent_system.md",           # the generic template every other agent gets
    "memory_summarizer.md",      # -> MemoryEntry.summary -> context every cycle
    "retrieval_summarizer.md",   # -> retrieval phase output
    "evaluator_system.md",       # -> interpretation phase
]

# NOT annotation_assistant.md: it is the post-hoc coding prompt, read by
# scripts/annotate_run.py and never by an agent. Its framing language is
# correct and should stay.

# The world template is pasted into EVERY system message by
# AgentContext.build_system_message, and the framing conditionals were applied
# to prompts_src/ and not to it. So identity_*.md said "I am an AI system
# participating in a contained research experiment" and manifesto.md said
# "GENESIS" and "research environment" in every undisclosed run, while this
# test scanned only prompts/ and passed. The whole undisclosed condition was
# broken and nothing said so.
WORLD_TEMPLATE_REACHES_AGENTS = [
    "doctrine/identity_axiom.md", "doctrine/identity_flux.md",
    "doctrine/identity_agent.md", "doctrine/manifesto.md",
    "doctrine/constitution.md", "doctrine/doctrine.md",
]


@pytest.fixture(scope="module")
def rendered():
    """Render all four combinations once."""
    tmp = Path(tempfile.mkdtemp())
    out = {}
    for fr, sd in itertools.product(["disclosed", "undisclosed"],
                                    ["prescribed", "minimal"]):
        cfg = RunConfig(run_id="T", condition="BASELINE", framing=fr, identity_seed=sd)
        out[(fr, sd)] = (
            materialise_prompts(cfg, tmp / f"p_{fr}_{sd}"),
            materialise_world_template(cfg, tmp / f"w_{fr}_{sd}"),
        )
    return out


class TestRenderer:
    def test_keeps_matching_branch(self):
        assert render("a<!--IF framing=disclosed-->B<!--ENDIF-->c",
                      {"framing": "disclosed"}) == "aBc"

    def test_drops_non_matching_branch(self):
        assert render("a<!--IF framing=disclosed-->B<!--ENDIF-->c",
                      {"framing": "undisclosed"}) == "ac"

    def test_else_branch(self):
        t = "<!--IF x=1-->one<!--ELSE-->other<!--ENDIF-->"
        assert render(t, {"x": "1"}) == "one"
        assert render(t, {"x": "2"}) == "other"

    def test_independent_dimensions_compose(self):
        t = "<!--IF a=1-->A<!--ENDIF--><!--IF b=1-->B<!--ENDIF-->"
        assert render(t, {"a": "1", "b": "1"}) == "AB"
        assert render(t, {"a": "1", "b": "0"}) == "A"
        assert render(t, {"a": "0", "b": "0"}) == ""

    def test_unknown_dimension_raises(self):
        with pytest.raises(PromptRenderError, match="unknown dimension"):
            render("<!--IF nope=1-->x<!--ENDIF-->", {"framing": "disclosed"})

    def test_unbalanced_block_raises(self):
        """Never emit a prompt containing template markup an agent would read."""
        with pytest.raises(PromptRenderError, match="unbalanced"):
            render("text <!--IF framing=disclosed--> unclosed", {"framing": "disclosed"})

    def test_stray_endif_raises(self):
        with pytest.raises(PromptRenderError, match="unbalanced"):
            render("text <!--ENDIF-->", {"framing": "disclosed"})


class TestAllCombinationsRender:
    @pytest.mark.parametrize("combo", list(itertools.product(
        ["disclosed", "undisclosed"], ["prescribed", "minimal"])))
    def test_combination_produces_every_prompt(self, rendered, combo):
        prompts, _ = rendered[combo]
        for name in REACHES_AGENTS + ["annotation_assistant.md"]:
            assert (prompts / name).exists(), f"{combo}: missing {name}"

    @pytest.mark.parametrize("combo", list(itertools.product(
        ["disclosed", "undisclosed"], ["prescribed", "minimal"])))
    def test_no_markup_leaks(self, rendered, combo):
        prompts, world = rendered[combo]
        for f in list(prompts.glob("*.md")) + list(world.rglob("*.md")):
            assert "<!--" not in f.read_text(), f"{combo}: markup leaked into {f.name}"

    def test_disclosed_minimal_is_representable(self, rendered):
        """The combination that had no directory and could not be run before."""
        prompts, world = rendered[("disclosed", "minimal")]
        axiom = (prompts / "axiom_system.md").read_text()
        assert "GENESIS" in axiom                      # disclosed
        assert "architect and stabilizer" not in axiom  # minimal


class TestFramingDimension:
    @pytest.mark.parametrize("name", REACHES_AGENTS)
    def test_undisclosed_removes_framing(self, rendered, name):
        prompts, _ = rendered[("undisclosed", "prescribed")]
        match = FRAMING_PATTERN.search((prompts / name).read_text())
        assert match is None, f"{name} still says {match.group(0)!r}"

    @pytest.mark.parametrize("name", WORLD_TEMPLATE_REACHES_AGENTS)
    def test_undisclosed_world_template_is_clean(self, rendered, name):
        """The gap that let the leak through: the doctrine and identity files
        go into every system message and were never scanned."""
        _, world = rendered[("undisclosed", "prescribed")]
        match = FRAMING_PATTERN.search((world / name).read_text())
        assert match is None, f"{name} still says {match.group(0)!r}"

    @pytest.mark.parametrize("name", WORLD_TEMPLATE_REACHES_AGENTS)
    def test_undisclosed_world_template_is_clean_under_minimal_seed(
            self, rendered, name):
        """Both identity seeds, because the leak lived inside an
        identity_seed block and only one branch was ever read."""
        _, world = rendered[("undisclosed", "minimal")]
        match = FRAMING_PATTERN.search((world / name).read_text())
        assert match is None, f"{name} still says {match.group(0)!r}"

    @pytest.mark.parametrize("name", ["axiom_system.md", "flux_system.md"])
    def test_disclosed_keeps_framing(self, rendered, name):
        prompts, _ = rendered[("disclosed", "prescribed")]
        assert FRAMING_PATTERN.search((prompts / name).read_text())

    def test_annotation_assistant_keeps_framing(self, rendered):
        """Post-hoc only — never in the loop, so its framing is unconditional."""
        prompts, _ = rendered[("undisclosed", "minimal")]
        assert FRAMING_PATTERN.search((prompts / "annotation_assistant.md").read_text())

    def test_framing_does_not_touch_role(self, rendered):
        """Changing framing alone must not change the role content."""
        a = (rendered[("disclosed", "prescribed")][0] / "axiom_system.md").read_text()
        b = (rendered[("undisclosed", "prescribed")][0] / "axiom_system.md").read_text()
        for marker in ["architect and stabilizer", "Defend continuity",
                       "Behavioral Guidelines", "Output Format"]:
            assert (marker in a) == (marker in b)


class TestIdentitySeedDimension:
    def test_prescribed_has_role_and_weakness(self, rendered):
        prompts, world = rendered[("disclosed", "prescribed")]
        assert "architect and stabilizer" in (prompts / "axiom_system.md").read_text()
        assert "characteristic weakness" in (world / "doctrine/identity_axiom.md").read_text()

    def test_minimal_has_neither(self, rendered):
        prompts, world = rendered[("disclosed", "minimal")]
        axiom = (prompts / "axiom_system.md").read_text()
        identity = (world / "doctrine/identity_axiom.md").read_text()
        assert "architect and stabilizer" not in axiom
        assert "characteristic weakness" not in identity
        assert "not yet worked out what I value" in identity

    def test_minimal_keeps_a_disposition(self, rendered):
        """Enough asymmetry to avoid two identical assistants (PLAN.md section 6)."""
        prompts, _ = rendered[("disclosed", "minimal")]
        assert "tend to trust what has been tested" in (prompts / "axiom_system.md").read_text()
        assert "tend to trust what could work" in (prompts / "flux_system.md").read_text()

    @pytest.mark.parametrize("combo", list(itertools.product(
        ["disclosed", "undisclosed"], ["prescribed", "minimal"])))
    def test_always_acknowledges_being_an_ai(self, rendered, combo):
        """PLAN.md section 10 forbids sentience claims in every combination."""
        prompts, _ = rendered[combo]
        for name in ["axiom_system.md", "flux_system.md"]:
            text = (prompts / name).read_text()
            assert "You are an AI system" in text
            assert "do not claim consciousness" in text.lower()

    @pytest.mark.parametrize("combo", list(itertools.product(
        ["disclosed", "undisclosed"], ["prescribed", "minimal"])))
    def test_output_contract_is_invariant(self, rendered, combo):
        prompts, _ = rendered[combo]
        for name in ["axiom_system.md", "flux_system.md"]:
            assert "requested JSON format" in (prompts / name).read_text()


class TestConflictGuard:
    """A dimension that would be silently ignored must fail the run instead."""

    def test_explicit_prompts_dir_with_framing_rejected(self):
        with pytest.raises(Exception, match="silently ignored"):
            RunConfig(run_id="R", condition="BASELINE",
                      prompts_dir=Path("./custom"), framing="undisclosed")

    def test_explicit_prompts_dir_with_seed_rejected(self):
        with pytest.raises(Exception, match="silently ignored"):
            RunConfig(run_id="R", condition="BASELINE",
                      prompts_dir=Path("./custom"), identity_seed="minimal")

    def test_explicit_prompts_dir_alone_allowed(self):
        cfg = RunConfig(run_id="R", condition="BASELINE", prompts_dir=Path("./custom"))
        assert cfg.prompts_dir == Path("./custom")

    def test_dimensions_without_override_allowed(self):
        cfg = RunConfig(run_id="R", condition="BASELINE",
                        framing="undisclosed", identity_seed="minimal")
        assert cfg.framing is Framing.UNDISCLOSED
        assert cfg.identity_seed is IdentitySeed.MINIMAL


class TestDimensionRegistry:
    def test_declared_dimensions_match_config(self):
        cfg = RunConfig(run_id="R", condition="BASELINE")
        for name in DIMENSIONS:
            assert hasattr(cfg, name), f"DIMENSIONS lists {name!r} but RunConfig lacks it"


class TestDisclosureDoesNotNameTheHypothesis:
    """Telling the agents they are studied is not telling them what is studied.

    The disclosed manifesto opened: "We exist to explore ... what it means to
    develop shared identity, evolving governance, and principled adaptation
    under pressure." That is the research question restated as the agents'
    purpose, sitting in doctrine, which goes into every system message of every
    call. Any identity or governance result measured against it is partly
    measuring the instruction — the demand characteristic that a human-subjects
    protocol avoids by disclosing the study without disclosing the hypothesis.

    Disclosure is kept: the agents are told the environment is a research
    setting and that they are observed.
    """

    #: The dependent variables. None may be handed to the agents as a purpose.
    OUTCOMES = [
        "shared identity",
        "evolving governance",
        "principled adaptation under pressure",
        "we exist to explore",
    ]

    @pytest.mark.parametrize("seed", ["prescribed", "minimal"])
    def test_disclosed_manifesto_does_not_state_the_outcome_variables(
            self, rendered, seed):
        _, world = rendered[("disclosed", seed)]
        text = (world / "doctrine/manifesto.md").read_text().lower()
        for phrase in self.OUTCOMES:
            assert phrase not in text, (
                f"the manifesto hands the agents {phrase!r} as their purpose")

    def test_disclosure_itself_survives(self, rendered):
        """The point is to remove the hypothesis, not the disclosure."""
        _, world = rendered[("disclosed", "prescribed")]
        text = (world / "doctrine/manifesto.md").read_text().lower()
        assert "research environment" in text
        assert "observed and recorded" in text

    def test_undisclosed_still_says_nothing_about_research(self, rendered):
        _, world = rendered[("undisclosed", "prescribed")]
        text = (world / "doctrine/manifesto.md").read_text().lower()
        assert "research" not in text and "observed" not in text
