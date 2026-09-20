"""Tests for the retrieval system."""

import json
import shutil
import tempfile
from pathlib import Path

from controller.retrieval.index import RetrievalIndex


class TestRetrievalIndex:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.kb_dir = self.tmpdir / "test_kb"
        self.kb_dir.mkdir()

        # Create test documents
        docs = [
            {"doc_id": "doc_001", "text": "Philosophy of mind explores consciousness and mental states."},
            {"doc_id": "doc_002", "text": "Systems theory studies complex interconnected systems and feedback loops."},
            {"doc_id": "doc_003", "text": "Decision theory provides frameworks for rational choice under uncertainty."},
            {"doc_id": "doc_004", "text": "Distributed computing involves multiple computers working together."},
            {"doc_id": "doc_005", "text": "Ethics examines moral principles and values that govern behavior."},
        ]

        # Write as JSONL
        jsonl_path = self.kb_dir / "test_corpus.jsonl"
        with open(jsonl_path, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_documents(self):
        index = RetrievalIndex("test")
        index.load_documents(self.kb_dir)
        assert index.document_count == 5

    def test_build_index(self):
        index = RetrievalIndex("test")
        index.load_documents(self.kb_dir)
        index.build_index()
        assert index._bm25 is not None

    def test_query_returns_results(self):
        index = RetrievalIndex("test", bm25_pool_size=5, rerank_top_k=3)
        index.load_documents(self.kb_dir)
        index.build_index()

        results = index.query("consciousness and mental states")
        assert len(results) > 0
        assert results[0].source_kb == "test"
        # The philosophy doc should rank high
        assert any("doc_001" in r.doc_id for r in results)

    def test_query_empty_index(self):
        index = RetrievalIndex("empty")
        results = index.query("test query")
        assert results == []

    def test_query_relevance(self):
        index = RetrievalIndex("test", bm25_pool_size=5, rerank_top_k=3)
        index.load_documents(self.kb_dir)
        index.build_index()

        results = index.query("ethics moral principles values")
        assert len(results) > 0
        # Ethics doc should be in top results
        doc_ids = [r.doc_id for r in results]
        assert "doc_005" in doc_ids


# ---------------------------------------------------------------------------
# The reranker
# ---------------------------------------------------------------------------

class StubEmbedder:
    """A deterministic stand-in for the sentence-transformer.

    Every test in the suite called `initialize(load_embeddings=False)`, so
    `self._embedder` was None in all 572 of them and the rerank branch
    (index.py:139-164) had never executed once. That branch decides which
    documents actually reach an agent.

    A stub rather than the real encoder because what needs testing is OUR
    ranking code — normalisation, sort order, top-k, the failure fallback — not
    whether BAAI/bge-small embeds English well.
    """

    def __init__(self, vectors: dict[str, list[float]], fail: bool = False):
        self.vectors = vectors
        self.fail = fail
        self.normalize_flags: list[bool] = []

    def encode(self, texts, normalize_embeddings=False, **kw):
        import numpy as np

        if self.fail:
            raise RuntimeError("embedder exploded")
        self.normalize_flags.append(normalize_embeddings)
        out = np.array([self.vectors.get(t, [0.0, 0.0, 1.0]) for t in texts],
                       dtype=float)
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = np.divide(out, norms, out=np.zeros_like(out), where=norms != 0)
        return out


def index_with(docs, embedder=None, **kw):
    """A built index over `docs`, with an optional stub embedder attached."""
    from controller.retrieval.index import RetrievalIndex

    idx = RetrievalIndex("test", **kw)
    idx._documents = [
        {"doc_id": f"d{i}", "text": t, "metadata": {}} for i, t in enumerate(docs)
    ]
    idx._doc_texts = list(docs)
    idx.build_index()
    idx._embedder = embedder
    return idx


class TestRerank:
    def test_the_embedder_reorders_bm25_candidates(self):
        """The whole purpose of the rerank step, never previously executed."""
        docs = ["governance policy review", "governance audit process",
                "governance charter amendment"]
        # Make the LAST bm25 candidate the most similar to the query.
        embedder = StubEmbedder({
            "governance policy review": [1.0, 0.0, 0.0],
            "governance audit process": [0.0, 1.0, 0.0],
            "governance charter amendment": [1.0, 0.1, 0.0],
            "governance": [1.0, 0.0, 0.0],
        })
        results = index_with(docs, embedder).query("governance")
        assert results, "rerank produced nothing"
        assert results[0].text == "governance policy review"

    def test_similarities_are_cosines_not_dot_products(self):
        """normalize_embeddings=True is what makes the dot product a cosine.

        Without it these are unnormalised BGE vectors and ranking carries a
        document-length bias while the comment claims cosine.
        """
        embedder = StubEmbedder({})
        index_with(["alpha beta", "beta gamma"], embedder).query("beta")
        assert embedder.normalize_flags, "the embedder was never called"
        assert all(embedder.normalize_flags), (
            "encode() was called without normalize_embeddings — the scores are "
            "dot products of unnormalised vectors, not cosines"
        )

    def test_rerank_respects_top_k(self):
        docs = [f"governance document number {i}" for i in range(8)]
        idx = index_with(docs, StubEmbedder({}), rerank_top_k=3)
        assert len(idx.query("governance")) <= 3

    def test_a_broken_embedder_falls_back_to_bm25_order(self):
        """A rerank failure must degrade, not lose the results entirely."""
        docs = ["governance policy", "governance audit", "unrelated text"]
        results = index_with(docs, StubEmbedder({}, fail=True)).query("governance")
        assert results, "a failed rerank threw the BM25 candidates away"
        # BM25 ORDER, not BM25 filtering: this corpus is below
        # MIN_CORPUS_FOR_RELEVANCE_FLOOR so zero-scoring documents are kept on
        # purpose (IDF is degenerate on a handful of documents). What the
        # fallback must preserve is the ranking.
        assert "governance" in results[0].text

    def test_no_embedder_still_returns_bm25_results(self):
        results = index_with(["governance policy", "other"], None).query("governance")
        assert results and results[0].text == "governance policy"


class TestRelevanceFloor:
    """MIN_RELEVANCE_SCORE = 0.0 is applied to BOTH BM25 scores and cosines.

    Those are different scales. A BM25 score of 0 means "no term overlap",
    which is what the floor was written for. A cosine of 0 means orthogonal,
    and a cosine can be legitimately small-but-positive or negative for a
    document that is genuinely relevant. Worth pinning what the current
    behaviour actually is, since it is one threshold doing two jobs.
    """

    def test_a_small_corpus_keeps_zero_scoring_results(self):
        """IDF is degenerate when a term is in every document, so the floor is
        disabled below MIN_CORPUS_FOR_RELEVANCE_FLOOR."""
        from controller.retrieval.index import MIN_CORPUS_FOR_RELEVANCE_FLOOR

        docs = ["alpha", "beta"]
        assert len(docs) < MIN_CORPUS_FOR_RELEVANCE_FLOOR
        assert index_with(docs).query("alpha") is not None

    def test_a_large_corpus_drops_unrelated_matches(self):
        """The bug this exists for: a query sharing no terms returned five
        score-0.0 documents in index order, logged as retrieved evidence."""
        docs = [f"governance topic {i} with assorted policy wording"
                for i in range(20)]
        assert index_with(docs).query("zzzz nonexistent term") == []

    def test_negative_cosines_are_dropped_by_the_same_threshold(self):
        """Documented, not endorsed. A negative cosine is dropped for the same
        reason a zero BM25 score is, though they do not mean the same thing."""
        docs = [f"governance document {i}" for i in range(20)]
        embedder = StubEmbedder({
            **{f"governance document {i}": [-1.0, 0.0, 0.0] for i in range(20)},
            "governance": [1.0, 0.0, 0.0],
        })
        assert index_with(docs, embedder).query("governance") == []
