"""Knowledge base management — load and index all retrieval databases."""

from __future__ import annotations

import logging
from pathlib import Path

from controller.phases.schemas import RetrievalResultItem
from controller.retrieval.index import RetrievalIndex

logger = logging.getLogger(__name__)

# Knowledge base names matching the plan
KB_NAMES = ["general", "technical", "governance", "scenarios", "self_history"]


class KnowledgeBaseManager:
    """Manages all five knowledge bases and routes queries."""

    def __init__(
        self,
        kb_dir: Path,
        bm25_pool_size: int = 20,
        rerank_top_k: int = 5,
        embedding_model: str = "sentence-transformers/bge-small-en-v1.5",
    ) -> None:
        self.kb_dir = kb_dir
        self.bm25_pool_size = bm25_pool_size
        self.rerank_top_k = rerank_top_k
        self.embedding_model = embedding_model
        self.indices: dict[str, RetrievalIndex] = {}

    def initialize(self, load_embeddings: bool = True) -> None:
        """Load all knowledge bases and build indices."""
        for name in KB_NAMES:
            index = RetrievalIndex(
                name=name,
                bm25_pool_size=self.bm25_pool_size,
                rerank_top_k=self.rerank_top_k,
            )
            kb_path = self.kb_dir / name
            index.load_documents(kb_path)

            if index.document_count > 0:
                index.build_index()
                if load_embeddings:
                    index.load_embedder(self.embedding_model)

            self.indices[name] = index

        total = sum(idx.document_count for idx in self.indices.values())
        logger.info("Knowledge bases initialized: %d total documents across %d KBs",
                     total, len(self.indices))

    def query(self, query_text: str, kb_name: str | None = None) -> list[RetrievalResultItem]:
        """Query a specific KB or all KBs.

        If kb_name is None, queries all KBs and merges/deduplicates results.
        """
        if kb_name and kb_name in self.indices:
            return self.indices[kb_name].query(query_text)

        # Query all KBs and merge
        all_results: list[RetrievalResultItem] = []
        for name, index in self.indices.items():
            results = index.query(query_text)
            all_results.extend(results)

        # Sort by score descending and take top_k
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:self.rerank_top_k]

    def add_to_self_history(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        """Add a document to the self-history knowledge base.

        Called during runs to index memory summaries, doctrine snapshots, etc.
        """
        doc = {
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata or {},
        }
        index = self.indices.get("self_history")
        if index is not None:
            index._documents.append(doc)
            index._doc_texts.append(text)
            # Rebuild index to include new document
            index.build_index()
