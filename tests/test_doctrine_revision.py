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
