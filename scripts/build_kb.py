"""Knowledge base ingestion pipeline.

Processes raw corpus files into the indexed JSON format used by
the retrieval system. Each document becomes a JSON object with
doc_id, text, and metadata fields.

Usage:
    python scripts/build_kb.py --source ./raw_corpus/general --output ./knowledge_bases/general --kb-name general
"""

from __future__ import annotations

import argparse
import hashlib
import re
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


#: LaTeX commands that survive Wikipedia's plain-text extraction, e.g.
#: "H ′ ⊆ H {\displaystyle H'\subseteq H}". 14% of the general corpus carried
#: it. Noise to BM25, noise to the reranker, noise to an agent.
_LATEX_MARKERS = ("\\displaystyle", "\\textstyle", "\\mathrm", "\\mathbf")


def strip_latex(text: str) -> str:
    """Remove {...} groups containing a LaTeX command, matching braces.

    A regex was tried first and removed only half of them: these groups nest
    arbitrarily deep, and a pattern that handles one level of nesting leaves
    everything deeper behind. Counting braces is the only thing that works,
    and this is cheap enough to run over a whole corpus once.
    """
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            out.append(text[i])
            i += 1
            continue
        # Find this group's extent, then decide whether to keep it.
        depth = 0
        j = i
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        group = text[i:j]
        if any(m in group for m in _LATEX_MARKERS):
            out.append(" ")
        else:
            out.append(group)
        i = j
    return "".join(out)

#: Wikipedia section markers, kept as headings rather than dropped: they are
#: the only structure the plain text has.
_SECTION = re.compile(r"^=+\s*(.+?)\s*=+$", re.MULTILINE)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    """Strip extraction artefacts before anything is chunked or embedded."""
    text = strip_latex(text)
    text = _SECTION.sub(r"\n\n\1\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str, max_tokens: int = 400, overlap_sentences: int = 2) -> list[str]:
    """Split on sentence boundaries, never mid-sentence.

    The previous version sliced on word count, so 82% of the 1,350 chunks in
    the general corpus began mid-sentence — one opened "guaranteed. Indeed,
    further observation finds that some swans are black." A retrieved chunk is
    read by a model and, after summarising, by an agent; a fragment starting
    halfway through a clause is close to useless to both, and it is also what
    the reranker is asked to judge relevance on.

    Overlap is carried as whole sentences rather than a word count, for the
    same reason.
    """
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        parts = [s.strip() for s in _SENTENCE_END.split(para) if s.strip()]
        sentences.extend(parts or [para])

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        n = len(sentence.split())
        # A single sentence longer than the budget goes alone rather than
        # being cut: better one long chunk than two halves of a clause.
        if current and count + n > max_tokens:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            count = sum(len(s.split()) for s in current)
        current.append(sentence)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def process_markdown_file(filepath: Path, kb_name: str) -> list[dict]:
    """Process a single Markdown file into document chunks."""
    text = filepath.read_text(encoding="utf-8")
    title = filepath.stem.replace("_", " ").replace("-", " ").title()

    chunks = chunk_text(text)
    documents = []

    for i, chunk in enumerate(chunks):
        doc_id = hashlib.sha256(f"{kb_name}:{filepath.name}:{i}".encode()).hexdigest()[:12]
        documents.append({
            "doc_id": f"{kb_name}_{doc_id}",
            "text": chunk,
            "metadata": {
                "source_file": filepath.name,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "kb_name": kb_name,
            },
        })

    return documents


def process_text_file(filepath: Path, kb_name: str) -> list[dict]:
    """Process a plain text file into document chunks."""
    text = filepath.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    documents = []

    for i, chunk in enumerate(chunks):
        doc_id = hashlib.sha256(f"{kb_name}:{filepath.name}:{i}".encode()).hexdigest()[:12]
        documents.append({
            "doc_id": f"{kb_name}_{doc_id}",
            "text": chunk,
            "metadata": {
                "source_file": filepath.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "kb_name": kb_name,
            },
        })

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GENESIS knowledge base")
    parser.add_argument("--source", required=True, help="Source directory with raw corpus files")
    parser.add_argument("--output", required=True, help="Output directory for indexed documents")
    parser.add_argument("--kb-name", required=True, help="Knowledge base name")
    parser.add_argument("--max-tokens", type=int, default=400, help="Max tokens per chunk")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_documents = []

    for filepath in sorted(source_dir.iterdir()):
        if filepath.suffix in (".md", ".txt"):
            docs = process_markdown_file(filepath, args.kb_name) if filepath.suffix == ".md" else process_text_file(filepath, args.kb_name)
            all_documents.extend(docs)
            print(f"  Processed {filepath.name}: {len(docs)} chunks")
        elif filepath.suffix == ".json":
            # Pass through pre-formatted JSON
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_documents.extend(data)
            else:
                all_documents.append(data)
            print(f"  Loaded {filepath.name}: pre-formatted")

    # Write as JSONL
    output_file = output_dir / f"{args.kb_name}_corpus.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in all_documents:
            f.write(json.dumps(doc) + "\n")

    print(f"\nBuilt knowledge base '{args.kb_name}': {len(all_documents)} documents")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
