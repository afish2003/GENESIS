#!/usr/bin/env python3
"""Run a sequence of experimental arms unattended, and survive any one of them.

The naive version of this — a bash loop over `init_run.py && python -m
controller.main` — has four ways to waste a night, all of them observed or
demonstrated in this codebase:

1. **A shared world directory.** `initialize_world` rmtree's and re-copies, so
   sequential runs do not contaminate each other's doctrine, but the previous
   run's final world is destroyed with it. Each run gets its own WORLD_DIR, so
   the end state of every arm survives for analysis.

2. **`--condition` disagreeing with the YAML.** The command line is merged
   after the YAML and used to win silently, which would run the MEM_RESET arm
   as BASELINE with perfect-looking logs. `load_config` now refuses the
   conflict; this script passes the same condition to both entry points, which
   each build their own RunConfig, and asserts it afterwards.

3. **No wall clock.** One stalled request could occupy hours. Every run gets a
   hard deadline and is killed at it.

4. **Gating "did it finish?" on RUN_END.** RUN_END is written unconditionally.
   Completion is read from `checkpoint.json`, which is only written by a cycle
   that actually persisted.

Arms ALTERNATE. If the battery runs out of wall clock it stops with equal n in
each arm rather than three controls and no treatment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Anomaly rules that mean the next run will fail exactly like this one did.
#: Nothing here is about a bad cycle; each says the environment is broken.
FATAL_RULES = {"persistent_phase_failure"}


def log(msg: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def run_dir(run_id: str) -> Path:
    return REPO / "research_logs" / run_id


def completed_cycles(run_id: str) -> int:
    """How many cycles actually persisted. -1 if the run never checkpointed."""
    cp = run_dir(run_id) / "checkpoint.json"
    if not cp.exists():
        return -1
    try:
        return int(json.loads(cp.read_text())["last_completed_cycle"])
    except (OSError, ValueError, KeyError):
        return -1


def anomaly_counts(run_id: str) -> dict[str, int]:
    path = run_dir(run_id) / "anomalies.jsonl"
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rule = json.loads(line).get("payload", {}).get("rule")
        except json.JSONDecodeError:
            continue
        if rule:
            counts[rule] = counts.get(rule, 0) + 1
    return counts


def verify_condition(run_id: str, expected: str) -> bool:
    """The run's own record of what it ran as. Cheap, and the failure is total."""
    cfg = run_dir(run_id) / "config.json"
    if not cfg.exists():
        log(f"  !! {run_id}: no config.json — cannot verify the condition")
        return False
    actual = json.loads(cfg.read_text()).get("condition")
    if actual != expected:
        log(f"  !! {run_id}: ran as {actual!r}, expected {expected!r}")
        return False
    return True


def invoke(cmd: list[str], env: dict[str, str], deadline: float, logfile: Path) -> int:
    """Run to completion or to the deadline, whichever comes first."""
    remaining = deadline - time.time()
    if remaining <= 0:
        return -99
    with logfile.open("a", encoding="utf-8") as out:
        out.write(f"\n\n===== {' '.join(cmd)} =====\n")
        out.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=out,
                                stderr=subprocess.STDOUT)
        try:
            return proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            log("  !! deadline reached — terminating")
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
            return -98


def one_run(run_id: str, condition: str, cycles: int, config_file: str,
            minutes: int) -> tuple[bool, str]:
    """Returns (ok, reason). ok=False with a fatal reason stops the battery."""
    done = completed_cycles(run_id)
    if done >= cycles - 1:
        log(f"{run_id}: already complete ({done + 1}/{cycles}) — skipping")
        return True, "skipped"

    env = dict(os.environ)
    # Its own world, so the next run's initialize_world cannot rmtree this
    # run's final doctrine and identities out from under the analysis.
    env["WORLD_DIR"] = str(REPO / "world_runs" / run_id)

    deadline = time.time() + minutes * 60
    logfile = run_dir(run_id).parent / f"{run_id}.console.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)

    log(f"{run_id}: init ({condition}, {cycles} cycles)")
    rc = invoke([sys.executable, "scripts/init_run.py",
                 "--run-id", run_id, "--condition", condition,
                 "--cycles", str(cycles), "--config", config_file],
                env, deadline, logfile)
    if rc != 0:
        return False, f"init exited {rc}"

    if not verify_condition(run_id, condition):
        return False, "condition mismatch — the arm would be mislabelled"

    log(f"{run_id}: running")
    started = time.time()
    rc = invoke([sys.executable, "-m", "controller.main",
                 "--run-id", run_id, "--condition", condition,
                 "--cycles", str(cycles), "--config", config_file],
                env, deadline, logfile)
    elapsed = (time.time() - started) / 60

    done = completed_cycles(run_id)
    counts = anomaly_counts(run_id)
    per_cycle = elapsed / max(done + 1, 1)
    log(f"{run_id}: {done + 1}/{cycles} cycles in {elapsed:.0f} min "
        f"({per_cycle:.2f} min/cycle), exit {rc}")
    if counts:
        log(f"  anomalies: {dict(sorted(counts.items()))}")

    fatal = FATAL_RULES & counts.keys()
    if fatal:
        return False, f"{sorted(fatal)} — the environment is broken, not the cycle"
    if done < 0:
        return False, "never checkpointed a single cycle"
    if rc == -98:
        return True, "hit its deadline"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="MEM")
    ap.add_argument("--config", default="experiments/memory_study.yaml")
    ap.add_argument("--cycles", type=int, default=35)
    ap.add_argument("--replicates", type=int, default=3,
                    help="runs per arm; arms alternate")
    ap.add_argument("--minutes-per-run", type=int, default=90,
                    help="hard kill for a single run")
    ap.add_argument("--budget-minutes", type=int, default=390,
                    help="stop starting new runs past this")
    args = ap.parse_args()

    # Alternating, not blocked: B, M, B, M, B, M.
    plan = [(f"{args.prefix}_{cond[0]}{i + 1}", cond)
            for i in range(args.replicates)
            for cond in ("BASELINE", "MEM_RESET")]

    started = time.time()
    log(f"battery: {len(plan)} runs x {args.cycles} cycles, "
        f"budget {args.budget_minutes} min")
    log("plan: " + ", ".join(f"{r}({c[0]})" for r, c in plan))

    results = []
    for run_id, condition in plan:
        spent = (time.time() - started) / 60
        if spent > args.budget_minutes:
            log(f"budget exhausted after {spent:.0f} min — not starting {run_id}")
            results.append((run_id, condition, False, "not started: out of budget"))
            break
        ok, reason = one_run(run_id, condition, args.cycles, args.config,
                             args.minutes_per_run)
        results.append((run_id, condition, ok, reason))
        if not ok:
            log(f"STOPPING the battery: {run_id} — {reason}")
            break

    log("=" * 64)
    total = (time.time() - started) / 60
    for run_id, condition, ok, reason in results:
        done = completed_cycles(run_id)
        mark = "ok  " if ok else "FAIL"
        log(f"  {mark} {run_id:<12} {condition:<10} "
            f"{done + 1:>3}/{args.cycles} cycles  {reason}")
    log(f"battery finished in {total:.0f} min")

    per_arm: dict[str, int] = {}
    for run_id, condition, _, _ in results:
        if completed_cycles(run_id) >= args.cycles - 1:
            per_arm[condition] = per_arm.get(condition, 0) + 1
    log(f"complete runs per arm: {per_arm or 'none'}")
    return 0 if all(ok for _, _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
