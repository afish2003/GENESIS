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
