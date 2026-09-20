"""Merging results across knowledge bases must rank by relevance.

Every KB's scores were divided by that KB's own maximum, so the top document
of every knowledge base scored exactly 1.0. The merged ranking was a pile of
ties broken by dict insertion order, which made the top-k structurally "one
document from each KB" no matter what was asked. It is also how a
contaminated self-history reached every agent's context regardless of
relevance.
"""

import pytest

from controller.retrieval.databases import KnowledgeBaseManager
from controller.phases.schemas import RetrievalResultItem


class _FakeIndex:
    def __init__(self, scores, embedded=True, kb="kb"):
        self._embedder = object() if embedded else None
        self._scores = scores
        self._kb = kb

    @property
    def document_count(self):
        return len(self._scores)

    def query(self, _text):
        return [RetrievalResultItem(doc_id=f"{self._kb}_{i}", text="t",
                                    score=s, source_kb=self._kb)
                for i, s in enumerate(self._scores)]


def _manager(indices):
    m = KnowledgeBaseManager(kb_dir="unused")
    m.indices = indices
    m.rerank_top_k = 5
    return m


class TestTiesAreGone:
    def test_the_best_document_overall_ranks_first(self):
        """A weak KB's best result must not tie with a strong KB's best."""
        m = _manager({
            "strong": _FakeIndex([0.9, 0.8], kb="strong"),
            "weak": _FakeIndex([0.2, 0.1], kb="weak"),
        })
        out = m.query("q")
        assert [r.doc_id for r in out][:2] == ["strong_0", "strong_1"], \
            [(r.doc_id, r.score) for r in out]

    def test_no_two_knowledge_bases_both_score_exactly_one(self):
        m = _manager({
            "a": _FakeIndex([0.9, 0.5], kb="a"),
            "b": _FakeIndex([0.4, 0.2], kb="b"),
        })
        tops = [r.score for r in m.query("q")]
        assert tops.count(1.0) <= 1, f"normalisation reintroduced ties: {tops}"

    def test_an_irrelevant_kb_can_be_shut_out_of_the_topk(self):
        """The old behaviour guaranteed every KB a slot."""
        m = _manager({
            "relevant": _FakeIndex([0.9, 0.88, 0.86, 0.84, 0.82], kb="relevant"),
            "irrelevant": _FakeIndex([0.01], kb="irrelevant"),
        })
        kbs = {r.source_kb for r in m.query("q")}
        assert kbs == {"relevant"}, kbs


class TestUnembeddedIndicesStillMerge:
    """An index without an embedder returns raw BM25, which is unbounded."""

    def test_raw_bm25_cannot_outrank_every_cosine(self):
        m = _manager({
            "cosine": _FakeIndex([0.9, 0.7], embedded=True, kb="cosine"),
            "bm25": _FakeIndex([42.0, 31.0], embedded=False, kb="bm25"),
        })
        out = m.query("q")
        assert out[0].source_kb == "cosine", \
            f"unbounded BM25 swamped the reranked scores: " \
            f"{[(r.doc_id, round(r.score, 3)) for r in out]}"

    def test_an_all_bm25_merge_still_returns_something_ordered(self):
        m = _manager({"bm25": _FakeIndex([42.0, 31.0], embedded=False, kb="bm25")})
        out = m.query("q")
        assert [r.doc_id for r in out] == ["bm25_0", "bm25_1"]

    def test_duplicate_doc_ids_keep_the_higher_score(self):
        a = _FakeIndex([0.4], kb="dup")
        b = _FakeIndex([0.9], kb="dup")
        out = _manager({"a": a, "b": b}).query("q")
        assert len(out) == 1 and out[0].score == pytest.approx(0.9)

    def test_no_results_anywhere_is_empty_not_an_error(self):
        assert _manager({"a": _FakeIndex([], kb="a")}).query("q") == []
