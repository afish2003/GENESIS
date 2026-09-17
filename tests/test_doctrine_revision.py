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
