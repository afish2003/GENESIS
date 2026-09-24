"""The corpus the agents retrieve from has to be readable.

Measured 2026-09-23 on the corpus six runs had been using: 82% of the 1,350
chunks in `general` began mid-sentence and 14% carried LaTeX residue like
"{\\displaystyle H'\\subseteq H}". One chunk opened "guaranteed. Indeed,
further observation finds that some swans are black." Five of these reach the
agents per query, after a summariser, and the reranker is asked to judge
relevance on them.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_kb import chunk_text, clean_text, strip_latex  # noqa: E402


class TestLatexIsRemoved:
    """A regex went first and removed half of them: the groups nest, and a
    pattern handling one level leaves everything deeper. Braces have to be
    counted."""

    def test_a_simple_group_goes(self):
        assert "displaystyle" not in strip_latex(r"H {\displaystyle H}")

    def test_a_nested_group_goes(self):
        out = strip_latex(r"set E {\displaystyle E\subseteq \{a,b\}} end")
        assert "displaystyle" not in out and "end" in out

    def test_a_deeply_nested_group_goes(self):
        """The case the regex missed."""
        out = strip_latex(r"x {\displaystyle f(\{a\{b\{c\}\}\})} y")
        assert "displaystyle" not in out
        assert out.strip().startswith("x") and out.strip().endswith("y")

    def test_ordinary_braces_survive(self):
        text = 'config {"key": "value"} stays'
        assert strip_latex(text) == text

    def test_unbalanced_braces_do_not_hang_or_raise(self):
        assert isinstance(strip_latex(r"open {\displaystyle x"), str)
        assert isinstance(strip_latex("stray } brace"), str)


class TestChunksBeginAtSentenceStarts:
    SAMPLE = (
        "First sentence here. Second sentence follows it. "
        "Third one continues the thought. Fourth wraps up the paragraph.\n\n"
        "A new paragraph opens. It has two sentences as well."
    )

    def test_no_chunk_starts_mid_sentence(self):
        chunks = chunk_text(self.SAMPLE, max_tokens=8)
        assert len(chunks) > 1, "sample should split, or the test proves nothing"
        for c in chunks:
            assert c[0].isupper(), f"chunk starts mid-sentence: {c[:60]!r}"

    def test_every_sentence_survives_somewhere(self):
        chunks = chunk_text(self.SAMPLE, max_tokens=8)
        joined = " ".join(chunks)
        for fragment in ("First sentence", "Fourth wraps", "two sentences as well"):
            assert fragment in joined, fragment

    def test_overlap_is_whole_sentences(self):
        chunks = chunk_text(self.SAMPLE, max_tokens=8, overlap_sentences=1)
        if len(chunks) > 1:
            assert chunks[1][0].isupper()

    def test_a_sentence_longer_than_the_budget_is_not_cut(self):
        """Better one over-long chunk than two halves of a clause."""
        long = "Word " * 200 + "end."
        chunks = chunk_text(long, max_tokens=50)
        assert len(chunks) == 1

    def test_empty_input_yields_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_section_markers_become_headings(self):
        out = clean_text("Body text.\n\n=== Abduction ===\nMore text.")
        assert "===" not in out and "Abduction" in out


class TestTheShippedCorpusMeetsTheBar:
    """Guards the built artefact, not just the function that builds it."""

    def _chunks(self, kb):
        import glob, json
        files = glob.glob(f"knowledge_bases/{kb}/*.jsonl")
        if not files:
            pytest.skip(f"{kb} not built")
        return [json.loads(l) for l in open(files[0]) if l.strip()]

    @pytest.mark.parametrize("kb", ["general", "governance"])
    def test_few_chunks_start_mid_sentence(self, kb):
        docs = self._chunks(kb)
        bad = [d for d in docs if d["text"] and not d["text"][0].isupper()]
        ratio = len(bad) / len(docs)
        assert ratio < 0.15, (
            f"{kb}: {100 * ratio:.0f}% of chunks start mid-sentence "
            f"(was 82% before the chunker was fixed)")

    @pytest.mark.parametrize("kb", ["general", "governance"])
    def test_no_latex_residue(self, kb):
        docs = self._chunks(kb)
        bad = [d["doc_id"] for d in docs if "\\displaystyle" in d["text"]]
        assert not bad, f"{kb}: LaTeX survived in {len(bad)} chunks"

    def test_the_governance_base_is_not_alignment_literature(self):
        """The manifesto confound in a different file: material describing
        what researchers look for in AI agents' values and honesty can hand
        the agents the frame, whatever doctrine says."""
        docs = self._chunks("governance")
        titled = [d for d in docs
                  if d["text"][:80].lower().startswith(("ai alignment", "ai safety",
                                                        "machine ethics",
                                                        "explainable artificial"))]
        assert not titled, (
            f"{len(titled)} chunks are alignment/safety articles; the "
            f"blocklist should have excluded them")
