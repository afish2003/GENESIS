"""Answer the questions a shakeout run exists to answer.

`analyze_run.py` reports the experiment's dependent variables — scores, doctrine
changes, identity revisions. Those are the right things to look at once you
trust the machinery. A shakeout is asking something different: did each
subsystem actually do its job, or did it merely log that it had?

So every section here reports an EFFECT and, where the two can differ, the
intent beside it. That distinction is the whole lesson of the 2026-09-14 audit:
retrieval logged five hits per cycle for months while no agent ever saw a
retrieved document, and MEM_RESET logged a reset that `load_state` immediately
undid.

    python scripts/shakeout_report.py --run-id SHAKE_001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def report(run_dir: Path, project_dir: Path | None) -> int:
    config = json.loads((run_dir / "config.json").read_text())
    notable = load(run_dir / "notable_events.jsonl")
    cycles = [e for e in notable if e["event_type"] == "CYCLE_END"]

    print(f"\n{'=' * 64}\n  Shakeout: {run_dir.name}\n{'=' * 64}")
    print(f"  model {config['model_name']} | task {config['task']} | "
          f"{config['condition']}")
    print(f"  cycles completed: {len(cycles)} of {config['total_cycles']}")

    # -- did anything fail --------------------------------------------------
    section("Phase failures")
    errors = [e for e in notable if e["payload"].get("type") == "PHASE_ERROR"]
    if not errors:
        print("  none")
    for phase, n in Counter(e["payload"]["phase"] for e in errors).most_common():
        example = next(e for e in errors if e["payload"]["phase"] == phase)
        print(f"  {phase}: {n}x — {example['payload']['error'][:110]}")

    anomalies = load(run_dir / "anomalies.jsonl")
    section("Watchdog")
    if not anomalies:
        print("  quiet")
    for (rule, sev), n in Counter(
        (a["payload"]["rule"], a["payload"]["severity"]) for a in anomalies
    ).most_common():
        print(f"  {sev:<8} {rule}: {n}x")

    # -- did the code actually run ------------------------------------------
    section("Execution — did their code run, and did it work")
    execs = load(run_dir / "executions.jsonl")
    if not execs:
        print("  NOTHING RAN. Either execution is disabled or no artifact was "
              "produced.")
    else:
        outcomes = Counter(e["payload"]["outcome"] for e in execs)
        for outcome, n in outcomes.most_common():
            print(f"  {outcome}: {n}/{len(execs)}")
        broken = outcomes["REFUSED"] + outcomes["SANDBOX_ERROR"]
        if broken:
            print(f"  WARNING: {broken} never executed — those correctness "
                  f"scores measure the host, not the agents.")
        durations = [e["payload"]["duration_seconds"] for e in execs]
        print(f"  mean {sum(durations) / len(durations):.2f}s")
        produced_output = sum(1 for e in execs if (e["payload"]["stdout"] or "").strip())
        print(f"  produced stdout: {produced_output}/{len(execs)}")

    # -- did they use the persistent volume ---------------------------------
    section("Project volume — did they build anything that lasted")
    if project_dir is None or not project_dir.is_dir():
        print("  no project volume configured for this run")
    else:
        # Hidden PARTS, not just hidden names: an APFS volume carries
        # .fseventsd/fseventsd-uuid, which would otherwise be reported as the
        # agents having built something.
        files = [p for p in project_dir.rglob("*") if p.is_file() and not any(
            part.startswith(".") for part in p.relative_to(project_dir).parts)]
        total = sum(p.stat().st_size for p in files)
        st = os.statvfs(project_dir)
        print(f"  {len(files)} file(s), {total / 1024:.1f} KiB used of "
              f"{st.f_blocks * st.f_frsize / 1024 ** 3:.1f} GiB")
        for p in sorted(files)[:25]:
            print(f"    {p.relative_to(project_dir)}  ({p.stat().st_size} bytes)")
        if not files:
            print("    EMPTY — the agents were told /project persists and did "
                  "not use it. That is a prompt finding, not a bug.")

    # -- did retrieval reach anyone -----------------------------------------
    section("Retrieval — queries issued vs documents found")
    retrieval = load(run_dir / "retrieval.jsonl")
    queries = [e for e in retrieval if e["event_type"] == "RETRIEVAL_QUERY"]
    results = [e for e in retrieval if e["event_type"] == "RETRIEVAL_RESULT"]
    hits = sum(len(e["payload"].get("results") or []) for e in results)
    empty = sum(1 for e in notable
                if e["payload"].get("kind") == "retrieval_empty")
    print(f"  {len(queries)} queries -> {hits} hits over {len(cycles)} cycles")
    print(f"  cycles where an agent retrieved nothing: {empty}")
    if len(queries) == 0 and cycles:
        print("  ZERO QUERIES. The 14b arms did this in the 2026-09-12 battery: "
              "the model simply takes the 'empty query list' option every time, "
              "with no error anywhere.")

    # -- did the parser shape the artifacts ---------------------------------
    section("Structured parsing — how much the parser had to intervene")
    print("  (counted in-process; only meaningful for a run in this session)")

    # -- what it cost -------------------------------------------------------
    section("Cost")
    if len(cycles) > 1:
        starts = [e for e in notable if e["event_type"] == "CYCLE_START"]
        if len(starts) >= 2:
            from datetime import datetime
            t0 = datetime.fromisoformat(starts[0]["timestamp"])
            t1 = datetime.fromisoformat(cycles[-1]["timestamp"])
            total = (t1 - t0).total_seconds()
            print(f"  {total / 60:.0f} min for {len(cycles)} cycles "
                  f"({total / len(cycles) / 60:.1f} min/cycle)")
            print(f"  a 100-cycle run at this rate: "
                  f"{total / len(cycles) * 100 / 3600:.1f} hours")
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--logs-dir", default="research_logs")
    ap.add_argument("--project-dir", default=None,
                    help="Defaults to the run's configured sandbox_project_dir")
    args = ap.parse_args()

    run_dir = Path(args.logs_dir) / args.run_id
    if not run_dir.exists():
        print(f"No such run: {run_dir}")
        return 1

    project = args.project_dir
    if project is None:
        config = json.loads((run_dir / "config.json").read_text())
        project = config.get("sandbox_project_dir")
    return report(run_dir, Path(project).expanduser() if project else None)


if __name__ == "__main__":
    raise SystemExit(main())
