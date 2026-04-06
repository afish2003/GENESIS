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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> list[str]:
    """Split text into chunks of approximately max_tokens words.

    Uses simple word-based splitting with overlap.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

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
