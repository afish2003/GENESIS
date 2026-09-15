"""Retrieval index — BM25 + embedding rerank pipeline.

Pipeline: BM25 top-20 candidates → embedding rerank to top-5.
All retrieval events are logged with query text, document IDs, scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from controller.phases.schemas import RetrievalResultItem

logger = logging.getLogger(__name__)

#: Results at or below this score are dropped rather than returned as padding.
#: BM25 gives 0.0 to a document sharing no query terms.
MIN_RELEVANCE_SCORE = 0.0

#: The floor is only applied above this corpus size. BM25's IDF term is
#: degenerate on a tiny index — a word appearing in every document scores 0.0
#: however relevant it is — and self_history is tiny for the first cycles of
#: every run. Filtering there would hide the agents' own past from them exactly
#: when there is least of it.
MIN_CORPUS_FOR_RELEVANCE_FLOOR = 10


class RetrievalIndex:
    """BM25 + embedding rerank retrieval over a knowledge base.

    Each knowledge base gets its own RetrievalIndex instance.
    Documents are stored as JSON with doc_id, text, and metadata.
    """

    def __init__(
        self,
        name: str,
        bm25_pool_size: int = 20,
        rerank_top_k: int = 5,
    ) -> None:
        self.name = name
        self.bm25_pool_size = bm25_pool_size
        self.rerank_top_k = rerank_top_k
        self._documents: list[dict] = []
        self._doc_texts: list[str] = []
        self._bm25 = None
        self._embedder = None

    def load_documents(self, docs_dir: Path) -> None:
        """Load documents from a directory of JSON files."""
        self._documents = []
        self._doc_texts = []

        if not docs_dir.exists():
            logger.warning("Knowledge base directory not found: %s", docs_dir)
            return

        for filepath in sorted(docs_dir.glob("*.json")):
            try:
                with open(filepath, encoding="utf-8") as f:
                    doc = json.load(f)
                self._documents.append(doc)
                self._doc_texts.append(doc.get("text", ""))
            except Exception as e:
                logger.error("Failed to load document %s: %s", filepath, e)

        # Always load JSONL too. This used to be gated on `not self._documents`,
        # so a directory holding both per-document .json files and a
        # build_kb.py-produced <kb>_corpus.jsonl silently dropped the corpus.
        for filepath in sorted(docs_dir.glob("*.jsonl")):
        
                try:
                    with open(filepath, encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                doc = json.loads(line)
                                self._documents.append(doc)
                                self._doc_texts.append(doc.get("text", ""))
                except Exception as e:
                    logger.error("Failed to load JSONL %s: %s", filepath, e)

        logger.info("Loaded %d documents into %s index", len(self._documents), self.name)

    def build_index(self) -> None:
        """Build the BM25 index from loaded documents."""
        if not self._doc_texts:
            logger.warning("No documents loaded for %s, skipping index build", self.name)
            return

        try:
            from rank_bm25 import BM25Okapi

            tokenized = [doc.lower().split() for doc in self._doc_texts]
            self._bm25 = BM25Okapi(tokenized)
            logger.info("Built BM25 index for %s (%d docs)", self.name, len(self._documents))
        except ImportError:
            logger.error("rank-bm25 not installed. Retrieval will not work.")

    def load_embedder(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        """Load the sentence transformer for reranking."""
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(model_name)
            logger.info("Loaded embedding model: %s", model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed. Reranking disabled.")
        except Exception as e:  # noqa: BLE001 - a bad repo id or no network
            # Only ImportError was caught, so a wrong model id or an offline
            # host killed KnowledgeBaseManager.initialize() — and with it the
            # whole run — with a raw traceback instead of degrading to BM25.
            # The repo id was in fact wrong once (commit ffb0768).
            logger.error(
                "Could not load embedding model %r (%s: %s). Falling back to "
                "BM25 only; retrieval will be less precise but the run continues.",
                model_name, type(e).__name__, e,
            )
            self._embedder = None

    def query(self, query_text: str) -> list[RetrievalResultItem]:
        """Run BM25 retrieval with optional embedding rerank.

        Returns top-k results as RetrievalResultItem objects.
        """
        if not self._documents or self._bm25 is None:
            return []

        # BM25 scoring
        tokenized_query = query_text.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)

        # Get top candidates
        scored = list(enumerate(bm25_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        candidates = scored[:self.bm25_pool_size]

        # Embedding rerank if available
        if self._embedder is not None and candidates:
            candidate_texts = [self._doc_texts[i] for i, _ in candidates]
            candidate_indices = [i for i, _ in candidates]

            try:
                # normalize_embeddings=True makes the dot product an actual
                # cosine. Without it these were unnormalised BGE vectors, so
                # ranking carried a document-length bias while the comment
                # claimed cosine. compare_arms.py already did this correctly.
                query_embedding = self._embedder.encode(
                    [query_text], normalize_embeddings=True
                )
                doc_embeddings = self._embedder.encode(
                    candidate_texts, normalize_embeddings=True
                )

                import numpy as np

                similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
                reranked = list(zip(candidate_indices, similarities))
                reranked.sort(key=lambda x: x[1], reverse=True)
                candidates = [(i, float(s)) for i, s in reranked[:self.rerank_top_k]]
            except Exception as e:
                logger.warning("Embedding rerank failed, using BM25 order: %s", e)
                candidates = candidates[:self.rerank_top_k]
        else:
            candidates = candidates[:self.rerank_top_k]

        # A query sharing no terms with any document still produced
        # rerank_top_k results at score 0.0, in index order, logged as
        # retrieved evidence. Analysis could not distinguish "found five
        # relevant documents" from "found nothing".
        if self.document_count >= MIN_CORPUS_FOR_RELEVANCE_FLOOR:
            kept = [(i, sc) for i, sc in candidates if sc > MIN_RELEVANCE_SCORE]
            if len(kept) < len(candidates):
                logger.debug(
                    "%s: dropped %d zero-relevance result(s) for %r",
                    self.name, len(candidates) - len(kept), query_text[:60],
                )
            candidates = kept

        # Build results
        results = []
        for idx, score in candidates:
            doc = self._documents[idx]
            results.append(RetrievalResultItem(
                doc_id=doc.get("doc_id", f"doc_{idx}"),
                text=doc.get("text", ""),
                score=score,
                source_kb=self.name,
            ))

        return results

    @property
    def document_count(self) -> int:
        return len(self._documents)
