"""Knowledge base management — load and index all retrieval databases."""

from __future__ import annotations

import json
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
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        run_id: str = "",
    ) -> None:
        self.kb_dir = kb_dir
        #: Scopes self-history to this run. Without it every run appends to one
        #: shared self_history.jsonl, so a later run retrieves an earlier run's
        #: memories, doc_ids collide (every run has a memory_axiom_cycle4), and
        #: a MEM_RESET arm's clear_self_history() wipes a BASELINE arm's past.
        #: Six sequential runs sharing one store is not six runs.
        self.run_id = run_id
        self.bm25_pool_size = bm25_pool_size
        self.rerank_top_k = rerank_top_k
        self.embedding_model = embedding_model
        self.indices: dict[str, RetrievalIndex] = {}

    def initialize(self, load_embeddings: bool = True) -> None:
        """Load all knowledge bases and build indices."""
        self._embeddings_enabled = load_embeddings
        for name in KB_NAMES:
            index = RetrievalIndex(
                name=name,
                bm25_pool_size=self.bm25_pool_size,
                rerank_top_k=self.rerank_top_k,
            )
            if name == "self_history":
                # Only THIS run's past. load_documents globs the directory, so
                # without this a run would index every previous run's file that
                # happens to sit beside its own — the agents of run 6 retrieving
                # the memories of run 1, with colliding doc_ids.
                self._load_own_history(index)
            else:
                index.load_documents(self.kb_dir / name)

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

        # Merge across KBs. Two problems the docstring claimed were handled:
        #
        # 1. There was no deduplication at all, so the same doc_id could occupy
        #    several of the top-k slots.
        # 2. Scores are not comparable between indices — an index with an
        #    embedder returns a normalised cosine, one without returns raw BM25,
        #    which is unbounded. Sorting them together ranked by scale rather
        #    than relevance. Normalising per-KB before the merge makes the
        #    comparison meaningful.
        merged: dict[str, RetrievalResultItem] = {}
        for name, index in self.indices.items():
            results = index.query(query_text)
            if not results:
                continue
            top = max(r.score for r in results) or 1.0
            for r in results:
                normalised = r.model_copy(update={"score": r.score / top})
                previous = merged.get(r.doc_id)
                if previous is None or normalised.score > previous.score:
                    merged[r.doc_id] = normalised

        ranked = sorted(merged.values(), key=lambda r: r.score, reverse=True)
        return ranked[:self.rerank_top_k]

    def _load_own_history(self, index) -> None:
        """Populate the self-history index from this run's file alone."""
        index._documents = []
        index._doc_texts = []
        path = self.self_history_path
        if not path.exists():
            return
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                doc = json.loads(line)
                index._documents.append(doc)
                index._doc_texts.append(doc.get("text", ""))
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Could not read self-history %s: %s — starting empty "
                         "rather than indexing a partial file", path, e)
            index._documents = []
            index._doc_texts = []

    @property
    def self_history_path(self) -> Path:
        """Per run. A shared file makes separate runs share a past."""
        name = f"{self.run_id}.jsonl" if self.run_id else "self_history.jsonl"
        return self.kb_dir / "self_history" / name

    def add_to_self_history(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        """Index one artifact of the agents' own past, and persist it.

        Called each cycle with memory summaries, doctrine snapshots and protocol
        versions, so agents can cite their own history.

        Persistence matters: this used to append to the in-memory index only,
        so a run resumed at cycle 60 lost every earlier cycle of self-history
        with no event recorded, and nothing was left for post-hoc analysis. The
        dedupe check below presupposed a durability that did not exist.
        """
        if not text or not text.strip():
            return
        index = self.indices.get("self_history")
        if index is None:
            return
        if any(d.get("doc_id") == doc_id for d in index._documents):
            return  # already indexed; resume must not duplicate
        doc = {"doc_id": doc_id, "text": text, "metadata": metadata or {}}
        index._documents.append(doc)
        index._doc_texts.append(text)
        index.build_index()

        # An index without an embedder returns raw BM25 scores while the other
        # KBs return reranked similarities, and query() sorts them together —
        # so self_history either swamped the corpora or was swamped by them,
        # regardless of relevance. It starts empty, so this is the first chance
        # to attach one.
        if self._embeddings_enabled and index._embedder is None:
            index.load_embedder(self.embedding_model)

        try:
            path = self.self_history_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error("Could not persist self-history entry %s: %s", doc_id, e)

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

        # Truncate on disk too, or a resume would restore what the reset removed.
        try:
            path = self.self_history_path
            if path.exists():
                path.unlink()
        except OSError as e:
            logger.error("Could not clear persisted self-history: %s", e)

        if removed:
            logger.info("Cleared %d self-history documents on memory reset", removed)
        return removed

    @property
    def self_history_count(self) -> int:
        index = self.indices.get("self_history")
        return len(index._documents) if index is not None else 0
