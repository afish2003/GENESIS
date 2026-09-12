"""Knowledge base management — load and index all retrieval databases."""

from __future__ import annotations

import logging
from pathlib import Path

from controller.phases.schemas import RetrievalResultItem
from controller.retrieval.index import RetrievalIndex

logger = logging.getLogger(__name__)

# Retrievable knowledge bases.
#
# "scenarios" is deliberately NOT here. PLAN.md section 9 specifies the scenario
# library as "not a retrieval-style database" — it is the injection library. If
# it were indexed, query() with no kb_name would search it, letting agents
# retrieve pressure events before they are injected (reading cycle 80's doctrine
# crisis at cycle 12) and silently destroying the escalation design.
KB_NAMES = ["general", "technical", "governance", "self_history"]


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
        """Index one artifact of the agents' own past.

        Called each cycle with memory summaries, doctrine snapshots and protocol
        versions, so agents can cite their own history.
        """
        if not text or not text.strip():
            return
        index = self.indices.get("self_history")
        if index is None:
            return
        if any(d.get("doc_id") == doc_id for d in index._documents):
            return  # already indexed; resume must not duplicate
        index._documents.append({
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata or {},
        })
        index._doc_texts.append(text)
        index.build_index()

    def clear_self_history(self) -> int:
        """Wipe self-history. Returns how many documents were removed.

        Called on memory reset in the MEM_RESET condition. PLAN.md section 11
        tests "whether persistent memory is necessary for identity continuity",
        so a reset that wiped the memory journal while leaving the full past
        retrievable would not remove memory — it would only change its access
        modality, confounding the comparison against BASELINE.
        """
        index = self.indices.get("self_history")
        if index is None:
            return 0
        removed = len(index._documents)
        index._documents.clear()
        index._doc_texts.clear()
        index.build_index()
        if removed:
            logger.info("Cleared %d self-history documents on memory reset", removed)
        return removed

    @property
    def self_history_count(self) -> int:
        index = self.indices.get("self_history")
        return len(index._documents) if index is not None else 0
