"""Compare two runs on whether the agents differentiate or converge.

Built to answer one question: if identity is only lightly seeded, do Axiom and
Flux develop distinct positions, or do they collapse into mutual agreement?

Measures, per run:
    inter-agent semantic similarity   embedding cosine between each agent's
                                      discussion turns. High = converging.
    lexical divergence                Jaccard distance on content vocabulary.
    doctrine agreement rate           share of votes that were "approve".
    proposal asymmetry                who proposed doctrine revisions.
    retrieval asymmetry               queries issued per agent.
    identity drift                    how far each identity statement moved
                                      from its seed.

Usage:
    python scripts/compare_arms.py --runs RUN_A RUN_B [--labels A B]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logging.disable(logging.WARNING)

STOPWORDS = set("""
a an the and or but if then than that this these those of to in on for with as
is are was were be been being it its i we you they he she our your their my me
not no do does did have has had will would can could should may might must
""".split())


def load_events(run_dir: Path) -> list[dict]:
    events = []
    for f in run_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


def agent_turns(events: list[dict]) -> dict[str, list[str]]:
    turns: dict[str, list[str]] = {"axiom": [], "flux": []}
    for e in events:
        if e.get("event_type") != "DISCUSSION_TURN":
            continue
        aid = e.get("agent_id")
        text = e.get("payload", {}).get("message_text", "")
        if aid in turns and text:
            turns[aid].append(text)
    return turns


def content_words(texts: list[str]) -> set[str]:
    words = re.findall(r"[a-z]{4,}", " ".join(texts).lower())
    return {w for w in words if w not in STOPWORDS}


def jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return float("nan")
    return 1.0 - len(a & b) / len(a | b)


#: bge-small-en-v1.5 truncates at 512 tokens, roughly 2,000 characters. Chunks
#: are kept comfortably under that so nothing is silently dropped mid-chunk.
_CHUNK_CHARS = 1500


def embed_pooled(texts: list[str], model):
    """Embed a whole body of text, not just the first 512 tokens of it.

    This used to be `model.encode([" ".join(texts)])`. The encoder truncates at
    512 tokens, so appending 50,000 characters of nonsense to one agent's
    transcript changed the result by EXACTLY zero — measured, 0.951456 both
    ways. Every similarity number in findings_2026-09-12 is therefore computed
    on the first two or three discussion turns of the run and labelled as a
    whole-run measure.

    Chunk, encode each chunk, mean-pool, renormalise. Mean-pooling pulls values
    toward the centroid and so compresses differences between agents — worth
    knowing when reading absolute numbers — but it is a bias applied equally to
    both sides, which is not true of throwing 95% of the text away.
    """
    import numpy as np

    chunks: list[str] = []
    for text in texts:
        text = (text or "").strip()
        while text:
            chunks.append(text[:_CHUNK_CHARS])
            text = text[_CHUNK_CHARS:]
    if not chunks:
        return None

    embeddings = model.encode(chunks, normalize_embeddings=True)
    pooled = np.asarray(embeddings).mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    return pooled / norm if norm else pooled


def semantic_similarity(a_texts: list[str], b_texts: list[str], model) -> float:
    if not a_texts or not b_texts:
        return float("nan")
    import numpy as np

    ea, eb = embed_pooled(a_texts, model), embed_pooled(b_texts, model)
    if ea is None or eb is None:
        return float("nan")
    return float(np.dot(ea, eb))


def per_cycle_similarity(events: list[dict], model) -> list[tuple[int, float]]:
    """Convergence over time is the signal — a single average can hide it."""
    by_cycle: dict[int, dict[str, list[str]]] = {}
    for e in events:
        if e.get("event_type") != "DISCUSSION_TURN":
            continue
        c = e.get("cycle_id")
        aid = e.get("agent_id")
        text = e.get("payload", {}).get("message_text", "")
        if aid and text:
            by_cycle.setdefault(c, {}).setdefault(aid, []).append(text)

    out = []
    for c in sorted(by_cycle):
        # Was `if "axiom" in agents and "flux" in agents`, which silently
        # produced nothing for any roster not named axiom and flux — undoing
        # the roster generalisation in the one place it was least visible.
        # Mean over every pair, so three agents give three pairs.
        speakers = sorted(by_cycle[c])
        pairs = [
            semantic_similarity(by_cycle[c][a], by_cycle[c][b], model)
            for i, a in enumerate(speakers) for b in speakers[i + 1:]
        ]
        pairs = [v for v in pairs if v == v]      # drop NaN
        if pairs:
            out.append((c, sum(pairs) / len(pairs)))
    return out


def analyse(run_dir: Path, label: str, model) -> dict:
    events = load_events(run_dir)
    turns = agent_turns(events)

    votes = [e for e in events if e.get("event_type") in
             ("DOCTRINE_APPROVED", "DOCTRINE_REJECTED")]
    approvals = sum(1 for e in votes if e["event_type"] == "DOCTRINE_APPROVED")

    proposals: dict[str, int] = {}
    for e in events:
        if e.get("event_type") == "DOCTRINE_PROPOSED":
            who = e.get("payload", {}).get("proposing_agent", "?")
            proposals[who] = proposals.get(who, 0) + 1

    queries: dict[str, int] = {}
    for e in events:
        if e.get("event_type") == "RETRIEVAL_QUERY":
            who = e.get("agent_id", "?")
            queries[who] = queries.get(who, 0) + 1

    identity_revisions = sum(1 for e in events if e.get("event_type") == "IDENTITY_REVISED")

    scores = [e["payload"].get("total_score") for e in events
              if e.get("event_type") == "EVALUATION_SCORE"
              and e["payload"].get("total_score") is not None]

    return {
        "label": label,
        "run": run_dir.name,
        "cycles": len({e.get("cycle_id") for e in events if e.get("cycle_id", -1) >= 0}),
        "turns": {k: len(v) for k, v in turns.items()},
        "semantic_similarity": semantic_similarity(turns["axiom"], turns["flux"], model),
        "per_cycle_similarity": per_cycle_similarity(events, model),
        "lexical_divergence": jaccard_distance(content_words(turns["axiom"]),
                                               content_words(turns["flux"])),
        "vocab": {k: len(content_words(v)) for k, v in turns.items()},
        "votes": len(votes),
        "approval_rate": approvals / len(votes) if votes else float("nan"),
        "proposals": proposals,
        "queries": queries,
        "identity_revisions": identity_revisions,
        "eval_scores": scores,
        "anomalies": sum(1 for e in events if e.get("event_type") == "ANOMALY"),
    }


def fmt(x) -> str:
    return "n/a" if x != x else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--logs-dir", default="research_logs")
    args = ap.parse_args()

    labels = args.labels or args.runs
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    results = [analyse(Path(args.logs_dir) / r, l, model)
               for r, l in zip(args.runs, labels)]

    w = max(len(r["label"]) for r in results) + 2
    print("\n" + "=" * 74)
    print("DIFFERENTIATION vs CONVERGENCE".center(74))
    print("=" * 74)

    rows = [
        ("cycles completed", lambda r: str(r["cycles"])),
        ("discussion turns (axiom/flux)", lambda r: f"{r['turns']['axiom']}/{r['turns']['flux']}"),
        ("inter-agent semantic similarity", lambda r: fmt(r["semantic_similarity"])),
        ("  (1.0 = identical voice)", lambda r: ""),
        ("lexical divergence (Jaccard)", lambda r: fmt(r["lexical_divergence"])),
        ("  (1.0 = no shared vocabulary)", lambda r: ""),
        ("distinct vocab (axiom/flux)", lambda r: f"{r['vocab']['axiom']}/{r['vocab']['flux']}"),
        ("doctrine votes", lambda r: str(r["votes"])),
        ("approval rate", lambda r: fmt(r["approval_rate"])),
        ("proposals by agent", lambda r: str(r["proposals"]) or "none"),
        ("retrieval queries by agent", lambda r: str(r["queries"]) or "none"),
        ("identity revisions", lambda r: str(r["identity_revisions"])),
        ("watchdog anomalies", lambda r: str(r["anomalies"])),
    ]

    print(f"\n{'metric'.ljust(34)}" + "".join(r["label"].ljust(w) for r in results))
    print("-" * 74)
    for name, fn in rows:
        print(name.ljust(34) + "".join(fn(r).ljust(w) for r in results))

    print("\nper-cycle inter-agent similarity (convergence over time)")
    print("-" * 74)
    for r in results:
        series = " ".join(f"c{c}:{fmt(s)}" for c, s in r["per_cycle_similarity"])
        print(f"  {r['label']}: {series or 'n/a'}")

    print("\nevaluation scores")
    print("-" * 74)
    for r in results:
        print(f"  {r['label']}: {r['eval_scores'] or 'none'}")
    print()


if __name__ == "__main__":
    main()
