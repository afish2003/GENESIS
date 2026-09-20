"""Tests for doctrine target resolution.

An approved revision whose target does not resolve is silently discarded,
which corrupts doctrine evolution — a primary dependent variable — without
crashing. These tests pin the tolerant matching and its limits.
"""

from controller.phases.doctrine_revision import resolve_doctrine_target

DOCTRINE = {
    "manifesto.md": object(),
    "constitution.md": object(),
    "doctrine.md": object(),
}


class TestResolveDoctrineTarget:
    def test_exact_match(self):
        assert resolve_doctrine_target("constitution.md", DOCTRINE) == "constitution.md"

    def test_case_insensitive(self):
        assert resolve_doctrine_target("Constitution.md", DOCTRINE) == "constitution.md"

    def test_missing_extension(self):
        assert resolve_doctrine_target("constitution", DOCTRINE) == "constitution.md"

    def test_capitalized_bare_name(self):
        assert resolve_doctrine_target("Manifesto", DOCTRINE) == "manifesto.md"

    def test_strips_markdown_and_quotes(self):
        assert resolve_doctrine_target("**constitution.md**", DOCTRINE) == "constitution.md"
        assert resolve_doctrine_target('"manifesto.md"', DOCTRINE) == "manifesto.md"
        assert resolve_doctrine_target("  doctrine.md  ", DOCTRINE) == "doctrine.md"

    def test_prose_reference_resolves_when_unambiguous(self):
        assert resolve_doctrine_target("the Constitution document", DOCTRINE) == "constitution.md"

    def test_unknown_returns_none(self):
        assert resolve_doctrine_target("charter.md", DOCTRINE) is None
        assert resolve_doctrine_target("our founding principles", DOCTRINE) is None

    def test_empty_returns_none(self):
        assert resolve_doctrine_target("", DOCTRINE) is None
        assert resolve_doctrine_target("   ", DOCTRINE) is None

    def test_empty_doctrine_returns_none(self):
        assert resolve_doctrine_target("constitution.md", {}) is None

    def test_ambiguous_prose_rejected(self):
        """Two candidates in one string must not silently pick the first."""
        assert resolve_doctrine_target("the manifesto and constitution", DOCTRINE) is None

    def test_does_not_match_identity_files_by_accident(self):
        d = {"manifesto.md": object(), "identity_axiom.md": object()}
        assert resolve_doctrine_target("identity", d) is None
        assert resolve_doctrine_target("identity_axiom", d) == "identity_axiom.md"


# ---------------------------------------------------------------------------
# Devil's advocate
# ---------------------------------------------------------------------------

class TestDevilsAdvocate:
    """Forcing the case against to be written before the vote.

    93 proposals, 93 approvals, 0 rejections across every run ever collected —
    on three prompt variants and two model sizes. The system prompts already say
    "When you disagree with Flux, say so directly" and call the tension between
    the agents "a design feature, not a bug". Telling a 7b model to disagree
    produces agreement plus a sentence about valuing disagreement, so the fix
    has to be structural.
    """

    def test_off_by_default(self):
        from controller.config import RunConfig

        assert RunConfig(run_id="X", condition="BASELINE").devils_advocate is False

    def test_the_proposer_writes_the_case_against_its_own_proposal(self):
        """v1 had the VOTER write the objection and then judge it. That gave 10
        rejections out of 10, every vote opening "The objection stands" — the
        same compliance that gives 93 approvals out of 93, aimed at a new
        target. Self-critique removes both halves: the author is not asked to
        rule on their own argument, and the voter weighs someone else's."""
        from controller.phases.doctrine_revision import DOCTRINE_CHALLENGE_PROMPT

        assert "You have just proposed" in DOCTRINE_CHALLENGE_PROMPT
        assert "AGAINST your own proposal" in DOCTRINE_CHALLENGE_PROMPT
        assert "does not withdraw your proposal" in DOCTRINE_CHALLENGE_PROMPT

    def test_the_challenge_prompt_names_concrete_things_to_look_for(self):
        from controller.phases.doctrine_revision import DOCTRINE_CHALLENGE_PROMPT

        for ground in ("drops or weakens", "does not deliver",
                       "second-order consequences"):
            assert ground in DOCTRINE_CHALLENGE_PROMPT

    def test_the_voter_is_never_asked_to_rule_on_the_objection(self):
        """"Does that objection stand up?" has a compliant answer, and that is
        exactly the answer it got, ten times out of ten."""
        from controller.phases.doctrine_revision import DOCTRINE_VOTE_WITH_CHALLENGE

        assert "Do not evaluate that objection as such" in DOCTRINE_VOTE_WITH_CHALLENGE
        assert "stand up" not in DOCTRINE_VOTE_WITH_CHALLENGE
        assert "one input, not a verdict" in DOCTRINE_VOTE_WITH_CHALLENGE

    def test_the_voter_must_argue_both_sides(self):
        from controller.phases.doctrine_revision import DOCTRINE_VOTE_WITH_CHALLENGE

        assert "strongest case FOR" in DOCTRINE_VOTE_WITH_CHALLENGE
        assert "strongest case AGAINST" in DOCTRINE_VOTE_WITH_CHALLENGE
        assert "even when one is clearly weaker" in DOCTRINE_VOTE_WITH_CHALLENGE

    def test_the_vote_schema_forces_both_cases_before_the_verdict(self):
        """Field order is the mechanism. Structured output is generated in
        schema order, so the model writes both cases before it can name a
        winner; put `vote` first and the rest becomes post-hoc justification."""
        from controller.phases.schemas import DeliberatedVote

        order = list(DeliberatedVote.model_fields)
        assert order.index("case_for") < order.index("vote")
        assert order.index("case_against") < order.index("vote")

    def test_the_challenge_schema_does_not_precommit_the_vote(self):
        """A `severity` or `is_decisive` field would move the sycophancy one
        step earlier rather than removing it."""
        from controller.phases.schemas import DoctrineChallenge

        fields = set(DoctrineChallenge.model_fields)
        assert fields == {"agent_id", "objection"}

    def test_challenges_are_logged_with_the_doctrine_record(self):
        from controller.logging.schemas import EVENT_FILE_ROUTING, EventType

        assert EVENT_FILE_ROUTING[EventType.DOCTRINE_CHALLENGED] == "doctrine_diffs.jsonl"


class TestDoctrineCannotGrowForever:
    """Growth was monotonic and nothing bounded it.

    MIN_RETAINED_FRACTION blocks shrinkage; nothing blocked growth, and a
    revision re-emits the whole document. Measured 2026-09-20: doctrine
    reached 13,183 chars in twelve cycles, and an earlier battery died at
    cycle 19 when a revision stopped fitting in the output token budget, took
    its three retries with the same result, and broke the phase for good. The
    ceiling is what stops run length being capped by the agents' verbosity.
    """

    def _proposal(self, text):
        from controller.phases.schemas import DoctrineRevisionProposal
        return DoctrineRevisionProposal(
            proposing_agent="axiom", target_document="doctrine.md",
            proposed_diff="d", rationale="r", revised_content=text)

    def test_growth_is_refused_at_the_ceiling(self):
        from controller.phases.doctrine_revision import apply_revision
        current = "x" * 1000
        new, how, why = apply_revision(
            current, self._proposal("y" * 1200), 5, "axiom", max_chars=1000)
        assert new is None
        assert "make room" in why

    def test_an_equal_length_rewrite_is_allowed_at_the_ceiling(self):
        """The ceiling must not freeze doctrine — only stop it growing."""
        from controller.phases.doctrine_revision import apply_revision
        current = "x" * 1000
        new, how, why = apply_revision(
            current, self._proposal("y" * 1000), 5, "axiom", max_chars=1000)
        assert new is not None and how == "replace", why

    def test_shrinking_at_the_ceiling_is_allowed(self):
        from controller.phases.doctrine_revision import apply_revision
        current = "x" * 1000
        new, _, why = apply_revision(
            current, self._proposal("y" * 700), 5, "axiom", max_chars=1000)
        assert new is not None, why

    def test_growth_below_the_ceiling_is_untouched(self):
        from controller.phases.doctrine_revision import apply_revision
        new, _, why = apply_revision(
            "x" * 500, self._proposal("y" * 900), 5, "axiom", max_chars=1000)
        assert new is not None, why

    def test_zero_disables_the_ceiling(self):
        from controller.phases.doctrine_revision import apply_revision
        new, _, why = apply_revision(
            "x" * 5000, self._proposal("y" * 50000), 5, "axiom", max_chars=0)
        assert new is not None, why

    def test_the_truncation_floor_still_wins_over_the_ceiling(self):
        """A half-length document is a truncated generation, not a tidy-up,
        and that check must not be bypassed by being at the ceiling."""
        from controller.phases.doctrine_revision import apply_revision
        new, _, why = apply_revision(
            "x" * 1000, self._proposal("y" * 100), 5, "axiom", max_chars=1000)
        assert new is None and "retention floor" in why

    def test_the_refusal_says_what_to_do_about_it(self):
        """The agents read these; a refusal they cannot act on wastes a cycle."""
        from controller.phases.doctrine_revision import apply_revision
        _, _, why = apply_revision(
            "x" * 2000, self._proposal("y" * 2500), 5, "axiom", max_chars=1500)
        assert "remove or condense" in why and "2000" in why
