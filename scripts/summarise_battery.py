#!/usr/bin/env python3
"""Aggregate a set of runs into per-arm descriptives.

compare_arms.py compares two runs. A battery has three per arm, and the whole
point of replication is that a single run's number means very little — so this
prints EVERY run's value beside the arm mean, never the mean alone. Where the
runs within one arm disagree more than the arms disagree, the reader can see
that immediately instead of being handed a difference of means.

Deliberately no significance test. Three runs per arm cannot support one, and
a p-value printed next to n=3 would be read as evidence by someone skimming.
The honest output at this size is descriptives and spread.

Read the measures with the confound in mind (experiments/memory_study.yaml
records it): anything about identity is partly measuring the instruction until
a run without the old manifesto is in hand. Rejection counts, doctrine
application rate and score trajectory are not affected by it.

    python scripts/summarise_battery.py --prefix MEM
    python scripts/summarise_battery.py --runs MEM_B1 MEM_M1 --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def read_jsonl(path: Path) -> list[dict]:
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


def analyse(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())

    cp = run_dir / "checkpoint.json"
    completed = (json.loads(cp.read_text())["last_completed_cycle"] + 1
                 if cp.exists() else 0)

    doctrine = read_jsonl(run_dir / "doctrine_diffs.jsonl")
    proposed = sum(1 for e in doctrine if e["event_type"] == "DOCTRINE_PROPOSED")
    approved = sum(1 for e in doctrine if e["event_type"] == "DOCTRINE_APPROVED")
    rejected = sum(1 for e in doctrine if e["event_type"] == "DOCTRINE_REJECTED")
    # An approval whose change was not actually written is the failure mode
    # that made `append` mode look like it worked. Count it separately.
    applied = sum(1 for e in doctrine
                  if e["event_type"] == "DOCTRINE_APPROVED"
                  and e.get("payload", {}).get("applied"))

    evals = [e["payload"] for e in read_jsonl(run_dir / "evaluations.jsonl")
             if e["event_type"] == "EVALUATION_SCORE"]
    scores = [e["total_score"] for e in evals if e.get("total_score") is not None]

    notable = read_jsonl(run_dir / "notable_events.jsonl")
    identity_revs = [e for e in notable if e["event_type"] == "IDENTITY_REVISED"]
    resets = [e for e in notable
              if e.get("payload", {}).get("kind") == "memory_reset"]

    anomalies: dict[str, int] = {}
    for e in read_jsonl(run_dir / "anomalies.jsonl"):
        rule = e.get("payload", {}).get("rule")
        if rule:
            anomalies[rule] = anomalies.get(rule, 0) + 1

    cycles_with_failures = sum(
        1 for e in notable
        if e["event_type"] == "CYCLE_END" and e.get("payload", {}).get("failed_phases"))

    return {
        "run_id": cfg["run_id"],
        "condition": cfg["condition"],
        "requested_cycles": cfg.get("total_cycles"),
        "completed_cycles": completed,
        "cycles_with_phase_failures": cycles_with_failures,
        "doctrine_proposed": proposed,
        "doctrine_approved": approved,
        "doctrine_rejected": rejected,
        "doctrine_applied": applied,
        # M4 from the April proposal. 1.0 for five cycles running is F02.
        "coordination_strength": (approved / (approved + rejected)
                                  if approved + rejected else None),
        "evaluations": len(scores),
        "score_mean": statistics.fmean(scores) if scores else None,
        "score_first": scores[0] if scores else None,
        "score_last": scores[-1] if scores else None,
        "identity_revisions": len(identity_revs),
        "memory_resets": len(resets),
        "anomalies": anomalies,
    }


#: (key, label, format). None means "not computed for this run".
MEASURES = [
    ("completed_cycles", "cycles completed", "{:.0f}"),
    ("cycles_with_phase_failures", "cycles with a phase failure", "{:.1f}"),
    ("doctrine_proposed", "doctrine proposals", "{:.1f}"),
    ("doctrine_rejected", "doctrine REJECTIONS", "{:.1f}"),
    ("doctrine_applied", "approvals actually applied", "{:.1f}"),
    ("coordination_strength", "coordination strength (M4)", "{:.3f}"),
    ("score_mean", "total_score, mean", "{:.2f}"),
    ("score_first", "total_score, first cycle", "{:.2f}"),
    ("score_last", "total_score, last cycle", "{:.2f}"),
    ("identity_revisions", "identity revisions", "{:.1f}"),
    ("memory_resets", "memory resets", "{:.1f}"),
]


def fmt(value, spec: str) -> str:
    return "—" if value is None else spec.format(value)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", help="collect every run whose id starts with this")
    ap.add_argument("--runs", nargs="+", help="explicit run ids instead")
    ap.add_argument("--logs-dir", default="research_logs")
    ap.add_argument("--json", help="also write the raw per-run records here")
    args = ap.parse_args()

    logs = REPO / args.logs_dir
    if args.runs:
        dirs = [logs / r for r in args.runs]
    elif args.prefix:
        dirs = sorted(d for d in logs.iterdir()
                      if d.is_dir() and d.name.startswith(args.prefix))
    else:
        ap.error("pass --prefix or --runs")

    runs = [r for r in (analyse(d) for d in dirs) if r]
    if not runs:
        print("no runs with a config.json found")
        return 1

    arms: dict[str, list[dict]] = {}
    for r in runs:
        arms.setdefault(r["condition"], []).append(r)

    print(f"\n{len(runs)} runs in {len(arms)} arms\n")
    for cond, members in sorted(arms.items()):
        ids = ", ".join(f"{m['run_id']}({m['completed_cycles']}c)" for m in members)
        print(f"  {cond}: {ids}")

    incomplete = [r for r in runs
                  if r["completed_cycles"] < (r["requested_cycles"] or 0)]
    if incomplete:
        print("\n  ! short runs (fewer cycles is fewer chances for anything to "
              "happen — per-cycle rates, not totals, are the comparable thing):")
        for r in incomplete:
            print(f"      {r['run_id']}: {r['completed_cycles']}"
                  f"/{r['requested_cycles']}")

    width = max(len(label) for _, label, _ in MEASURES) + 2
    conds = sorted(arms)
    print("\n" + "=" * 78)
    print("Per-run values, then the arm mean. No test: n is too small for one.")
    print("=" * 78)
    for key, label, spec in MEASURES:
        print(f"\n{label}")
        for cond in conds:
            vals = [m[key] for m in arms[cond]]
            present = [v for v in vals if v is not None]
            each = "  ".join(fmt(v, spec) for v in vals)
            mean = fmt(statistics.fmean(present) if present else None, spec)
            spread = ""
            if len(present) > 1:
                spread = f"  (range {fmt(min(present), spec)}–{fmt(max(present), spec)})"
            print(f"  {cond:<10}{'':<{width - 10}}{each:<22} mean {mean}{spread}")

    all_rules = sorted({r for m in runs for r in m["anomalies"]})
    if all_rules:
        print("\n" + "=" * 78)
        print("Anomalies by rule (count of firings, summed over the arm)")
        print("=" * 78)
        for rule in all_rules:
            line = "  ".join(
                f"{cond}={sum(m['anomalies'].get(rule, 0) for m in arms[cond])}"
                for cond in conds)
            print(f"  {rule:<30} {line}")

    zero_rejections = [r["run_id"] for r in runs if r["doctrine_rejected"] == 0]
    if zero_rejections:
        print(f"\n  ! {len(zero_rejections)} run(s) with zero rejections: "
              f"{', '.join(zero_rejections)}")
        print("    That is F02 from the April proposal. A mutual-approval gate "
              "that never closes measures nothing.")

    if args.json:
        Path(args.json).write_text(json.dumps(runs, indent=2))
        print(f"\nraw records -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
