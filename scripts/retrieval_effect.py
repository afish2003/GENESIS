#!/usr/bin/env python3
"""Does retrieval change what the agents build?

Nobody had asked. Retrieval is a whole subsystem — BM25, a reranker, four
corpora, a summariser, a cap on how much enters the prompt — and the question
of whether any of it reaches the artifact had never been measured. It should
be asked before any effort goes into growing the corpora.

Method: lexical overlap (Jaccard over distinctive words) between a cycle's
artifact and the text retrieved for that cycle, against a control of the same
cycle's retrieved text versus a DIFFERENT cycle's artifact FROM THE SAME RUN.

The control has to be within-run. A first version of this drew the control
from other runs, reported 68% against a 50% baseline, and was wrong: cycles
in one run share vocabulary and topic drift, so a cross-run control is less
similar for reasons having nothing to do with retrieval, which manufactures
an effect. Within-run, the same data gives 44% and p=0.34.

Caveat worth keeping in view: lexical overlap is a crude proxy. The retrieval
summariser condenses documents before they reach the agents, so wording is
lost by design, and an idea could influence a document without sharing words
with it. A null here means "no detectable lexical trace", not "the agents were
unaffected".
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

STOP = set("""the a an and or of to in for on with that this is are be as by from it its
their our we you not can may should must will would each other than then these those
protocol document doctrine agent agents cycle section purpose scope procedure""".split())


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{5,}", (text or "").lower()) if w not in STOP}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def rows_for(run: Path) -> list[tuple[int, set[str], set[str]]]:
    retrieved: dict[int, set[str]] = {}
    for e in load(run / "retrieval.jsonl"):
        for r in (e.get("payload", {}).get("results") or []):
            retrieved.setdefault(e["cycle_id"], set()).update(words(r.get("text")))
    artifacts = {e["cycle_id"]: words(e["payload"].get("content"))
                 for e in load(run / "protocol_diffs.jsonl")
                 if e["event_type"] == "PROTOCOL_PROPOSED"}
    return [(c, retrieved[c], artifacts[c]) for c in sorted(retrieved)
            if c in artifacts and retrieved[c] and artifacts[c]]


def sign_test(wins: int, n: int) -> float:
    """Two-sided, exact. No scipy dependency for one binomial."""
    if n == 0:
        return 1.0
    k = min(wins, n - wins)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="MEM")
    ap.add_argument("--logs-dir", default="research_logs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    matched: list[float] = []
    control: list[float] = []
    per_run = []

    for run in sorted((REPO / args.logs_dir).iterdir()):
        if not (run.is_dir() and run.name.startswith(args.prefix)):
            continue
        rows = rows_for(run)
        if len(rows) < 2:
            continue
        m0, c0 = len(matched), len(control)
        for i, (_, retrieved, artifact) in enumerate(rows):
            matched.append(jaccard(retrieved, artifact))
            j = rng.choice([k for k in range(len(rows)) if k != i])
            control.append(jaccard(retrieved, rows[j][2]))
        per_run.append((run.name, len(rows),
                        statistics.fmean(matched[m0:]),
                        statistics.fmean(control[c0:])))

    if not matched:
        print("no runs with both retrieval results and artifacts")
        return 1

    print(f"\n{len(matched)} cycles across {len(per_run)} runs\n")
    for name, n, m, c in per_run:
        print(f"  {name:10} n={n:>3}  own {m:.4f}  control {c:.4f}  {m - c:+.4f}")

    wins = sum(1 for m, c in zip(matched, control) if m > c)
    n = len(matched)
    print(f"\n  own cycle's retrieval   : {statistics.fmean(matched):.4f}")
    print(f"  another cycle, same run : {statistics.fmean(control):.4f}")
    print(f"  difference              : "
          f"{statistics.fmean(matched) - statistics.fmean(control):+.4f}")
    print(f"  own higher in {wins}/{n} ({100 * wins / n:.0f}%; 50% is no effect)")
    print(f"  sign test p = {sign_test(wins, n):.4f}")
    print("\n  Lexical overlap only. The summariser condenses before the agents\n"
          "  see anything, so a null is 'no detectable lexical trace', not\n"
          "  'the agents were unaffected'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
