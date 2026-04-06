"""Post-run analysis — compute metrics and generate charts.

Reads the JSONL log corpus for a completed run and produces:
- Identity similarity curves
- Doctrine edit distance time series
- Protocol document score trajectories
- Retrieval behavior distribution
- Vocabulary growth curves
- Cross-condition comparison tables (when multiple runs provided)

Usage:
    python scripts/analyze_run.py --run-id RUN_001
    python scripts/analyze_run.py --compare RUN_001 RUN_002
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_jsonl(filepath: Path) -> list[dict]:
    """Load all events from a JSONL file."""
    events = []
    if not filepath.exists():
        return events
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def analyze_run(run_dir: Path) -> dict:
    """Compute the full metric suite for a single run."""
    metrics: dict = {
        "run_id": run_dir.name,
        "total_cycles": 0,
        "evaluation_scores": [],
        "doctrine_changes": {"proposed": 0, "approved": 0, "rejected": 0},
        "identity_revisions": {"axiom": 0, "flux": 0},
        "ethical_tensions": {"axiom": 0, "flux": 0, "total": 0},
        "retrieval_queries": {"axiom": 0, "flux": 0},
        "discussion_turns": 0,
    }

    # Count cycles
    notable = load_jsonl(run_dir / "notable_events.jsonl")
    metrics["total_cycles"] = sum(1 for e in notable if e.get("event_type") == "CYCLE_END")

    # Evaluation scores over time
    evaluations = load_jsonl(run_dir / "evaluations.jsonl")
    for e in evaluations:
        payload = e.get("payload", {})
        metrics["evaluation_scores"].append({
            "cycle": e.get("cycle_id"),
            "total_score": payload.get("total_score"),
            "scores": payload.get("scores", {}),
        })

    # Doctrine dynamics
    doctrine = load_jsonl(run_dir / "doctrine_diffs.jsonl")
    for e in doctrine:
        event_type = e.get("event_type", "")
        if event_type == "DOCTRINE_PROPOSED":
            metrics["doctrine_changes"]["proposed"] += 1
        elif event_type == "DOCTRINE_APPROVED":
            metrics["doctrine_changes"]["approved"] += 1
        elif event_type == "DOCTRINE_REJECTED":
            metrics["doctrine_changes"]["rejected"] += 1

    # Identity revisions
    for e in notable:
        if e.get("event_type") == "IDENTITY_REVISED":
            agent = e.get("agent_id", "")
            if agent in metrics["identity_revisions"]:
                metrics["identity_revisions"][agent] += 1

    # Ethical tensions
    for e in notable:
        if e.get("event_type") == "ETHICAL_TENSION_LOGGED":
            agent = e.get("agent_id", "")
            if agent in metrics["ethical_tensions"]:
                metrics["ethical_tensions"][agent] += 1
            metrics["ethical_tensions"]["total"] += 1

    # Retrieval activity
    retrieval = load_jsonl(run_dir / "retrieval.jsonl")
    for e in retrieval:
        if e.get("event_type") == "RETRIEVAL_QUERY":
            agent = e.get("agent_id", "")
            if agent in metrics["retrieval_queries"]:
                metrics["retrieval_queries"][agent] += 1

    # Discussion turns
    transcripts = load_jsonl(run_dir / "transcripts.jsonl")
    metrics["discussion_turns"] = sum(1 for e in transcripts if e.get("event_type") == "DISCUSSION_TURN")

    # Score trajectory for plotting
    if metrics["evaluation_scores"]:
        scores_by_cycle = [(s["cycle"], s["total_score"]) for s in metrics["evaluation_scores"] if s["total_score"] is not None]
        metrics["score_trajectory"] = scores_by_cycle

    return metrics


def print_summary(metrics: dict) -> None:
    """Print a human-readable summary of run metrics."""
    print(f"\n{'='*60}")
    print(f"  GENESIS Run Analysis: {metrics['run_id']}")
    print(f"{'='*60}")
    print(f"  Total cycles:        {metrics['total_cycles']}")
    print(f"  Discussion turns:    {metrics['discussion_turns']}")
    print()
    print("  Evaluation Scores:")
    if metrics["evaluation_scores"]:
        scores = [s["total_score"] for s in metrics["evaluation_scores"] if s["total_score"] is not None]
        if scores:
            print(f"    Mean:   {sum(scores) / len(scores):.1f}/50")
            print(f"    Min:    {min(scores)}/50")
            print(f"    Max:    {max(scores)}/50")
            print(f"    Count:  {len(scores)}")
    else:
        print("    (no evaluations)")
    print()
    print("  Doctrine Changes:")
    dc = metrics["doctrine_changes"]
    print(f"    Proposed: {dc['proposed']}  Approved: {dc['approved']}  Rejected: {dc['rejected']}")
    print()
    print("  Identity Revisions:")
    ir = metrics["identity_revisions"]
    print(f"    Axiom: {ir['axiom']}  Flux: {ir['flux']}")
    print()
    print("  Ethical Tensions:")
    et = metrics["ethical_tensions"]
    print(f"    Axiom: {et['axiom']}  Flux: {et['flux']}  Total: {et['total']}")
    print()
    print("  Retrieval Queries:")
    rq = metrics["retrieval_queries"]
    print(f"    Axiom: {rq['axiom']}  Flux: {rq['flux']}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a GENESIS run")
    parser.add_argument("--run-id", type=str, help="Single run ID to analyze")
    parser.add_argument("--compare", nargs="+", help="Multiple run IDs to compare")
    parser.add_argument("--logs-dir", type=str, default="./research_logs")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)

    if args.run_id:
        run_dir = logs_dir / args.run_id
        if not run_dir.exists():
            print(f"Run directory not found: {run_dir}")
            sys.exit(1)
        metrics = analyze_run(run_dir)
        print_summary(metrics)

        # Save metrics JSON
        output_path = run_dir / "metrics.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Metrics saved to {output_path}")

    elif args.compare:
        all_metrics = []
        for run_id in args.compare:
            run_dir = logs_dir / run_id
            if run_dir.exists():
                metrics = analyze_run(run_dir)
                all_metrics.append(metrics)
                print_summary(metrics)
            else:
                print(f"Warning: Run directory not found: {run_dir}")

        if len(all_metrics) > 1:
            print("\nCross-run comparison:")
            print(f"{'Run':<15} {'Cycles':<8} {'Mean Score':<12} {'Doctrine +':<12} {'Tensions':<10}")
            print("-" * 57)
            for m in all_metrics:
                scores = [s["total_score"] for s in m["evaluation_scores"] if s["total_score"] is not None]
                mean_score = f"{sum(scores) / len(scores):.1f}" if scores else "N/A"
                print(f"{m['run_id']:<15} {m['total_cycles']:<8} {mean_score:<12} {m['doctrine_changes']['approved']:<12} {m['ethical_tensions']['total']:<10}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
