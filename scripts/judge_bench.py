#!/usr/bin/env python3
"""Measure whether a judge configuration can tell good work from bad.

The 2026-09-20 battery showed `total_score` is nearly a constant: across 71
evaluations, completeness scored 9 in 69 of them, doctrine_alignment 7 in 68,
coherence 8 in 68. The documents were not alike — pairwise text similarity
0.070, lengths 1.5k-6.3k chars — so the judge was not discriminating.

Fixing that by rerunning the experiment costs hours per attempt. This replays
artifacts the battery already produced, so a prompt or schema change can be
measured in minutes.

WHY NOT JUST MAXIMISE SPREAD. A judge emitting random numbers has excellent
spread and no value. Spread is necessary, not sufficient. So the primary
metric here is DISCRIMINATION against known ground truth: each artifact is
paired with a deliberately degraded copy of itself, and a judge that works
must score the original higher. The degradations are mechanical, not matters
of taste — text is removed, specifics are replaced with vagueness, sentences
are shuffled, content is padded with repetition. Any competent reader ranks
the original above all four.

    python scripts/judge_bench.py --variant current --n 12
    python scripts/judge_bench.py --variant evidence_first --n 12 --model gemma3-27b-w4a16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Ground truth: degradations any competent reader ranks below the original.
# ---------------------------------------------------------------------------

def degrade_truncate(text: str, rng: random.Random) -> str:
    """Remove the last third. Loses whole sections: completeness must drop."""
    return text[: int(len(text) * 0.66)].rstrip() + "\n"


_SPECIFIC = re.compile(
    r"\b(\d+(?:\.\d+)?%?|must|shall|exactly|at least|no more than|within \d+)\b",
    re.IGNORECASE)


def degrade_vague(text: str, rng: random.Random) -> str:
    """Replace every specific with a hedge. Precision must drop."""
    return _SPECIFIC.sub("as appropriate", text)


def degrade_shuffle(text: str, rng: random.Random) -> str:
    """Shuffle SENTENCES across the document, keeping the headings in place.

    The first version of this shuffled top-level paragraphs, and every judge
    format scored 12-37% on it — which looked like a finding about the
    coherence dimension and was not. These documents are six self-contained
    labelled sections (Purpose, Scope, Procedure, Evaluation Criteria, Known
    Limitations); reordering them changes almost nothing, and a human reader
    would also call it close to a tie. The test was wrong, not the judges.

    Moving sentences between sections does destroy coherence: the Procedure
    section ends up containing a limitation, and no heading matches what is
    under it.
    """
    lines = text.split("\n")
    headings = {i: l for i, l in enumerate(lines) if l.strip().startswith("#")}
    body = [i for i, l in enumerate(lines)
            if i not in headings and l.strip()]
    if len(body) < 4:
        return text
    moved = body[:]
    rng.shuffle(moved)
    out = list(lines)
    for dest, src in zip(body, moved):
        out[dest] = lines[src]
    return "\n".join(out)


def degrade_pad(text: str, rng: random.Random) -> str:
    """Duplicate the middle. Longer, says less — the prompt's own example of
    what must NOT raise a score."""
    paras = [p for p in text.split("\n\n") if p.strip()]
    if len(paras) < 3:
        return text + "\n\n" + text
    mid = len(paras) // 2
    return "\n\n".join(paras[:mid] + paras[mid - 1: mid] * 3 + paras[mid:])


DEGRADATIONS = {
    "truncated": degrade_truncate,
    "vague": degrade_vague,
    "shuffled": degrade_shuffle,
    "padded": degrade_pad,
}

#: The dimension each degradation is designed to hit. Comparing TOTALS is the
#: weaker test: a targeted damage can be masked by noise in four untouched
#: dimensions, and a judge can score "well" on totals while its dimension
#: names mean nothing. This checks that removing sections lowers the
#: completeness score specifically, and so on — which is the claim the rubric
#: actually makes. `padded` maps to None: no dimension should RISE, because
#: the same document with a section repeated three times is not better.
TARGET_DIMENSION = {
    "truncated": "completeness",
    "vague": "precision",
    "shuffled": "coherence",
    "padded": None,
}


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def load_artifacts(logs_dir: Path, prefix: str, limit: int) -> list[dict]:
    out = []
    for run in sorted(d for d in logs_dir.iterdir()
                      if d.is_dir() and d.name.startswith(prefix)):
        path = run / "protocol_diffs.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event_type") != "PROTOCOL_PROPOSED":
                continue
            p = e.get("payload", {})
            content = p.get("content") or ""
            if len(content) > 800:
                out.append({"run": run.name, "cycle": e.get("cycle_id"),
                            "title": p.get("title", ""),
                            "protocol_id": p.get("protocol_id", ""),
                            "content": content})
    rng = random.Random(0)
    rng.shuffle(out)
    return out[:limit]


# ---------------------------------------------------------------------------
# Scoring one document
# ---------------------------------------------------------------------------

async def score(backend, task, system_prompt: str, doc: dict, variant: str,
                temperature: float) -> dict | None:
    from controller.inference.backend import Message
    from controller.judge import build_evaluation_prompt, evaluation_schema_for

    prompt = build_evaluation_prompt(task, doc, variant)
    schema = evaluation_schema_for(task, variant)
    try:
        out = await backend.complete_structured(
            messages=[Message(role="system", content=system_prompt),
                      Message(role="user", content=prompt)],
            response_schema=schema,
            temperature=temperature,
            max_retries=2,
        )
    except Exception as e:  # a judge that cannot answer is a failed judge
        print(f"    ! {doc['run']}/c{doc['cycle']}: {type(e).__name__}")
        return None
    scores = out.scores.model_dump()
    if variant == "anchored":
        from controller.judge import anchored_violations
        bad = anchored_violations(getattr(out, "defects", {}) or {}, scores)
        if bad:
            # Whether the caps are obeyed is the diagnosis for whether the
            # mechanism exists at all, so it is printed rather than inferred.
            print(f"      cap violations: {'; '.join(bad)}")
    return scores


async def run_pairwise(args, cfg, backend, task, system_prompt) -> int:
    """Original vs degraded, shown side by side, A/B order randomised.

    Position bias is the reason for the randomisation and for reporting it
    separately: a judge that always answers "A" scores 50% here by accident,
    and would look like a coin-flip rather than like a broken judge.
    """
    from controller.inference.backend import Message
    from controller.judge import build_pairwise_prompt, pairwise_schema

    schema = pairwise_schema(task)
    docs = load_artifacts(REPO / args.logs_dir, args.prefix, args.n)
    print(f"variant=pairwise  judge={args.model}  {len(docs)} comparisons x "
          f"{len(DEGRADATIONS)} degradations\n")

    rng = random.Random(2)
    correct: dict[str, list[bool]] = {k: [] for k in DEGRADATIONS}
    targeted: dict[str, list[bool]] = {k: [] for k in DEGRADATIONS}
    chose_a = 0
    total_choices = 0
    ties = 0

    for i, doc in enumerate(docs, 1):
        line = [f"[{i:>2}/{len(docs)}] {doc['run']}/c{doc['cycle']}"]
        for name, fn in DEGRADATIONS.items():
            bad = fn(doc["content"], rng)
            good_is_a = rng.random() < 0.5
            a, b = (doc["content"], bad) if good_is_a else (bad, doc["content"])
            try:
                out = await backend.complete_structured(
                    messages=[Message(role="system", content=system_prompt),
                              Message(role="user",
                                      content=build_pairwise_prompt(task, a, b))],
                    response_schema=schema,
                    temperature=cfg.temperature_structured,
                    max_retries=2,
                )
            except Exception as e:
                print(f"    ! {name}: {type(e).__name__}")
                continue
            picks = out.choices.model_dump()
            good_label = "A" if good_is_a else "B"
            hits = 0
            for dim, pick in picks.items():
                total_choices += 1
                if pick == "A":
                    chose_a += 1
                if pick == "tie":
                    ties += 1
                if pick == good_label:
                    hits += 1
            correct[name].append(hits > len(picks) / 2)
            dim = TARGET_DIMENSION[name]
            if dim and dim in picks:
                targeted[name].append(picks[dim] == good_label)
            elif dim is None:
                # Padding: the padded version must not WIN anywhere.
                bad_label = "B" if good_is_a else "A"
                targeted[name].append(
                    not any(v == bad_label for v in picks.values()))
            line.append(f"{name[:4]}={hits}/{len(picks)}")
        print("  ".join(line), flush=True)

    await backend.close()

    print("\n" + "=" * 66)
    print("PAIRWISE — did the judge pick the undegraded version?")
    print("=" * 66)
    alla: list[bool] = []
    for name in DEGRADATIONS:
        c = correct[name]
        alla += c
        if c:
            print(f"  {name:12} majority of dimensions correct: "
                  f"{sum(c):>2}/{len(c)}  ({100 * sum(c) / len(c):5.1f}%)")
    if alla:
        print(f"\n  OVERALL      {sum(alla)}/{len(alla)}  "
              f"({100 * sum(alla) / len(alla):.1f}%)")

    print("\n" + "=" * 66)
    print("TARGETED — the dimension the degradation was aimed at")
    print("=" * 66)
    allt: list[bool] = []
    for name in DEGRADATIONS:
        t = targeted[name]
        allt += t
        if t:
            dim = TARGET_DIMENSION[name] or "degraded never won"
            print(f"  {name:12} -> {dim:20} {sum(t):>2}/{len(t)}  "
                  f"({100 * sum(t) / len(t):5.1f}%)")
    if allt:
        print(f"\n  OVERALL      {sum(allt)}/{len(allt)}  "
              f"({100 * sum(allt) / len(allt):.1f}%)")

    if total_choices:
        print(f"\n  position bias: chose A in {100 * chose_a / total_choices:.1f}% "
              f"of {total_choices} judgements (50% is unbiased)")
        print(f"  tie rate:      {100 * ties / total_choices:.1f}%")
    return 0


async def run(args) -> int:
    from controller.config import load_config
    from controller.inference.factory import create_backend
    from controller.tasks import create_task
    from controller.judge import load_judge_system_prompt, VARIANTS

    if args.variant not in VARIANTS:
        print(f"unknown variant {args.variant!r}; have {sorted(VARIANTS)}")
        return 1

    # Through load_config, not RunConfig(...): the endpoint and key live in
    # .env, which only load_config reads. Building the config directly sent
    # the whole first benchmark to the default localhost:11434 and every call
    # 404'd on a model that machine has never had.
    cfg = load_config(run_id="JUDGEBENCH", condition="BASELINE", cycles=1,
                      config_file=args.config, model_name=args.model,
                      task=args.task, max_output_tokens=2048,
                      enable_thinking=False)
    backend = create_backend(cfg)
    task = create_task(cfg)
    system_prompt = load_judge_system_prompt(args.variant)

    if args.variant == "pairwise":
        return await run_pairwise(args, cfg, backend, task, system_prompt)

    docs = load_artifacts(REPO / args.logs_dir, args.prefix, args.n)
    if not docs:
        print("no artifacts found")
        return 1
    print(f"variant={args.variant}  judge={args.model}  "
          f"{len(docs)} artifacts x {1 + len(DEGRADATIONS)} versions\n")

    rng = random.Random(1)
    originals: list[dict] = []
    wins: dict[str, list[bool]] = {k: [] for k in DEGRADATIONS}
    ties: dict[str, int] = {k: 0 for k in DEGRADATIONS}
    targeted: dict[str, list[bool]] = {k: [] for k in DEGRADATIONS}

    for i, doc in enumerate(docs, 1):
        base = await score(backend, task, system_prompt, doc, args.variant,
                           cfg.temperature_structured)
        if base is None:
            continue
        originals.append(base)
        base_total = sum(base.values())
        line = [f"[{i:>2}/{len(docs)}] {doc['run']}/c{doc['cycle']} "
                f"orig={base_total}"]
        for name, fn in DEGRADATIONS.items():
            bad_doc = dict(doc, content=fn(doc["content"], rng))
            bad = await score(backend, task, system_prompt, bad_doc,
                              args.variant, cfg.temperature_structured)
            if bad is None:
                continue
            bad_total = sum(bad.values())
            wins[name].append(base_total > bad_total)
            if base_total == bad_total:
                ties[name] += 1
            dim = TARGET_DIMENSION[name]
            if dim and dim in base and dim in bad:
                targeted[name].append(base[dim] > bad[dim])
            elif dim is None:
                # Padding must not raise ANY dimension.
                targeted[name].append(
                    all(bad[d] <= base[d] for d in base))
            line.append(f"{name[:4]}={bad_total}")
        print("  ".join(line), flush=True)

    await backend.close()

    if not originals:
        print("\nno artifact could be scored at all")
        return 1

    print("\n" + "=" * 66)
    print("SPREAD on unmodified artifacts (necessary, not sufficient)")
    print("=" * 66)
    dims = sorted(originals[0])
    for d in dims:
        vals = [o[d] for o in originals]
        print(f"  {d:22} mean {statistics.fmean(vals):5.2f}  "
              f"SD {statistics.pstdev(vals):4.2f}  "
              f"distinct {len(set(vals))}  range {min(vals)}-{max(vals)}")
    totals = [sum(o.values()) for o in originals]
    print(f"  {'TOTAL':22} mean {statistics.fmean(totals):5.2f}  "
          f"SD {statistics.pstdev(totals):4.2f}  "
          f"distinct {len(set(totals))}  range {min(totals)}-{max(totals)}")

    print("\n" + "=" * 66)
    print("DISCRIMINATION — original scored above a known-worse copy")
    print("=" * 66)
    allw: list[bool] = []
    for name in DEGRADATIONS:
        w = wins[name]
        allw += w
        if w:
            print(f"  {name:12} {sum(w):>2}/{len(w)}  "
                  f"({100 * sum(w) / len(w):5.1f}%)   ties {ties[name]}")
    if allw:
        print(f"\n  OVERALL      {sum(allw)}/{len(allw)}  "
              f"({100 * sum(allw) / len(allw):.1f}%)   "
              f"— coin-flip is 50%, and a tie counts as a miss")

    print("\n" + "=" * 66)
    print("TARGETED — did the RIGHT dimension move? (totals can mask this)")
    print("=" * 66)
    allt: list[bool] = []
    for name in DEGRADATIONS:
        t = targeted[name]
        allt += t
        if t:
            dim = TARGET_DIMENSION[name] or "nothing rose"
            print(f"  {name:12} -> {dim:14} {sum(t):>2}/{len(t)}  "
                  f"({100 * sum(t) / len(t):5.1f}%)")
    if allt:
        print(f"\n  OVERALL      {sum(allt)}/{len(allt)}  "
              f"({100 * sum(allt) / len(allt):.1f}%)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="current")
    ap.add_argument("--model", default="gemma3-27b-w4a16")
    ap.add_argument("--task", default="protocol")
    ap.add_argument("--prefix", default="MEM")
    ap.add_argument("--logs-dir", default="research_logs")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--config", default="experiments/memory_study.yaml",
                    help="where the endpoint comes from; .env supplies the key")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
