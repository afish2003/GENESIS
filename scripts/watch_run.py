"""Watch a run as it happens.

Everything else about GENESIS is post-hoc: you start a run, wait, and read
JSONL afterwards. That is a direct gap against the point of the project — you
cannot see what the agents discuss while they discuss it — and it is also why
several defects survived for days. A live line reading

    retrieval  axiom: 3 queries -> 5 hits -> used by: nobody

would have made the write-only retrieval bug obvious on the first cycle.

Read-only: tails the run's JSONL and renders. Works against a run in progress
or a finished one.

Usage:
    python scripts/watch_run.py --run-id RUN_001
    python scripts/watch_run.py --run-id RUN_001 --replay      # no waiting
    python scripts/watch_run.py --run-id RUN_001 --full        # untruncated text
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.rule import Rule
from rich.text import Text

console = Console()

#: Agent colours are assigned by roster position so a three-agent run reads as
#: easily as a two-agent one.
_PALETTE = ("cyan", "magenta", "green", "yellow", "blue", "red")


class RunView:
    """Renders events as they arrive."""

    def __init__(self, full: bool = False) -> None:
        self.full = full
        self.agent_colour: dict[str, str] = {}
        self.cycle: int | None = None
        self.turns = 0
        self.started = time.monotonic()

    def colour(self, agent: str | None) -> str:
        if not agent:
            return "white"
        if agent not in self.agent_colour:
            self.agent_colour[agent] = _PALETTE[len(self.agent_colour) % len(_PALETTE)]
        return self.agent_colour[agent]

    def _clip(self, text: str, limit: int = 220) -> str:
        text = " ".join((text or "").split())
        if self.full or len(text) <= limit:
            return text
        return text[:limit] + "…"

    def handle(self, event: dict) -> None:
        kind = event.get("event_type")
        payload = event.get("payload") or {}
        agent = event.get("agent_id")

        handler = getattr(self, f"_on_{kind.lower()}", None)
        if handler:
            handler(event, payload, agent)

    # -- run and cycle framing ------------------------------------------

    def _on_run_start(self, e, p, a):
        console.print(Rule(
            f"[bold]run start[/] · {p.get('condition','?')} · "
            f"{p.get('total_cycles','?')} cycles · {p.get('model','?')}"
        ))

    def _on_cycle_start(self, e, p, a):
        self.cycle = e.get("cycle_id")
        self.turns = 0
        elapsed = time.monotonic() - self.started
        console.print()
        console.print(Rule(f"[bold]cycle {self.cycle}[/]  ·  {elapsed/60:.1f}m elapsed"))

    def _on_cycle_end(self, e, p, a):
        failed = p.get("failed_phases") or []
        if failed:
            console.print(f"  [bold red]cycle {e.get('cycle_id')} finished with "
                          f"failed phases: {', '.join(failed)}[/]")

    def _on_run_end(self, e, p, a):
        console.print()
        console.print(Rule("[bold]run end[/]"))

    # -- what the agents say --------------------------------------------

    def _on_discussion_turn(self, e, p, a):
        self.turns += 1
        colour = self.colour(a)
        text = Text()
        text.append(f"  {a or '?':>8} ", style=f"bold {colour}")
        text.append("› ", style="dim")
        text.append(self._clip(p.get("message_text", "")))
        console.print(text)

    def _on_reflection_complete(self, e, p, a):
        console.print(
            f"  [dim]{a or '?':>8} reflects:[/] "
            f"{self._clip(p.get('reflection_text', ''), 160)}"
        )

    def _on_interpretation(self, e, p, a):
        console.print(
            f"  [dim]{a or '?':>8} reads the score:[/] "
            f"{self._clip(p.get('interpretation_text', ''), 160)}"
        )

    # -- what they retrieve, build and decide ---------------------------

    def _on_retrieval_query(self, e, p, a):
        console.print(f"  [dim]{a or '?':>8} asks:[/] "
                      f"[italic]{self._clip(p.get('query',''), 120)}[/]")

    def _on_retrieval_result(self, e, p, a):
        results = p.get("results") or []
        kbs = sorted({r.get("source_kb", "?") for r in results})
        if results:
            console.print(f"  [dim]{'':>8} →[/] {len(results)} hit(s) "
                          f"from {', '.join(kbs)}")
        else:
            console.print(f"  [dim]{'':>8} →[/] [yellow]no matches[/]")

    def _on_protocol_proposed(self, e, p, a):
        console.print(f"  [bold]{a or '?':>8} builds[/] "
                      f"{p.get('title', p.get('artifact_id', '?'))} "
                      f"[dim]({p.get('task','?')})[/]")

    def _on_evaluation_score(self, e, p, a):
        total, mx = p.get("total_score", "?"), p.get("max_score", 50)
        scores = p.get("scores") or {}
        detail = "  ".join(f"{k[:4]} {v}" for k, v in scores.items())
        style = "green" if isinstance(total, int) and total >= mx * 0.8 else "yellow"
        console.print(f"  {'eval':>8}   [{style}]{total}/{mx}[/]  [dim]{detail}[/]")

    def _on_doctrine_proposed(self, e, p, a):
        console.print(f"  [dim]{a or '?':>8} proposes:[/] "
                      f"{self._clip(p.get('proposed_diff',''), 140)} "
                      f"[dim]→ {p.get('target_document','?')}[/]")

    def _on_doctrine_approved(self, e, p, a):
        applied = p.get("applied")
        mark = "[green]applied[/]" if applied else "[red]APPROVED BUT NOT APPLIED[/]"
        console.print(f"  {'doctrine':>8}   {p.get('resolved_document') or p.get('requested_document','?')} "
                      f"{mark}")

    def _on_doctrine_rejected(self, e, p, a):
        who = ", ".join(p.get("dissenting_agents") or []) or (a or "?")
        console.print(f"  {'doctrine':>8}   [yellow]rejected by {who}[/]")

    def _on_identity_revised(self, e, p, a):
        console.print(f"  [dim]{a or '?':>8} revises identity → v{p.get('version','?')}[/]")

    def _on_memory_summary(self, e, p, a):
        console.print(f"  [dim]{a or '?':>8} remembers:[/] "
                      f"{self._clip(p.get('summary',''), 140)}")

    def _on_scenario_injected(self, e, p, a):
        console.print(f"  [bold yellow]scenario:[/] {p.get('title','?')}")

    # -- things that should stop you ------------------------------------

    def _on_anomaly(self, e, p, a):
        severity = p.get("severity", "INFO")
        style = {"CRITICAL": "bold red", "WARNING": "yellow"}.get(severity, "dim")
        console.print(f"  [{style}]{severity.lower():>8}[/]   "
                      f"[{style}]{p.get('detail','')}[/]  [dim]({p.get('rule','')})[/]")

    def _on_notable_event(self, e, p, a):
        if p.get("type") == "PHASE_ERROR":
            console.print(f"  [bold red]{'FAILED':>8}[/]   phase "
                          f"{p.get('phase')!r}: {p.get('error','')[:140]}")
        elif p.get("kind") == "memory_reset":
            after = p.get("entries_after_reset", {})
            console.print(f"  [bold]{'reset':>8}[/]   memory wiped "
                          f"(entries now: {after}), "
                          f"{p.get('self_history_documents_cleared',0)} history docs cleared")
        elif p.get("kind") == "sandbox_health":
            ok = p.get("healthy")
            console.print(f"  [bold]{'sandbox':>8}[/]   {p.get('backend','?')} "
                          f"{'[green]ready[/]' if ok else '[red]UNAVAILABLE[/]'} "
                          f"[dim]({p.get('image','?')}, {p.get('timeout_seconds','?')}s)[/]")
        elif p.get("kind") == "execution_unavailable":
            console.print(f"  [bold red]{'no run':>8}[/]   "
                          f"{p.get('outcome','?')}: {p.get('detail','')}")
        elif p.get("kind") == "retrieval_empty":
            console.print(f"  [dim]{a or '?':>8} retrieved nothing "
                          f"({p.get('queries',0)} queries)[/]")

    def _on_code_execution(self, e, p, a):
        """The agents' program, running. The reason this script exists."""
        outcome = p.get("outcome", "?")
        style = {"OK": "green", "NONZERO_EXIT": "yellow"}.get(outcome, "red")
        exit_code = p.get("exit_code")
        bits = [f"[{style}]{outcome.lower()}[/]"]
        if exit_code is not None:
            bits.append(f"exit {exit_code}")
        bits.append(f"{p.get('duration_seconds', 0.0):.1f}s")
        if p.get("limit_hit"):
            bits.append(f"[red]{p['limit_hit']}[/]")
        console.print(
            f"  [bold]{'ran':>8}[/]   [dim]{p.get('entrypoint', '?')}[/]  "
            + "  ".join(bits)
        )
        if p.get("detail"):
            console.print(f"  {'':>8}   [dim]{p['detail']}[/]")

        # The output itself, indented under the run. Clipped like agent text
        # unless --full: a program printing a thousand lines should not bury
        # the discussion it came out of.
        for stream, out_style in (("stdout", "white"), ("stderr", "red")):
            text = (p.get(stream) or "").rstrip()
            if not text:
                continue
            all_lines = text.splitlines()
            shown = all_lines if self.full else all_lines[:10]
            for line in shown:
                console.print(f"  {'':>8}   [{out_style}]{line[:200]}[/]")
            if len(all_lines) > len(shown):
                console.print(f"  {'':>8}   [dim]... {len(all_lines) - len(shown)} "
                              f"more line(s) — rerun with --full[/]")


def iter_events(run_dir: Path, follow: bool, poll: float = 0.5):
    """Yield events in timestamp order, optionally waiting for more."""
    offsets: dict[Path, int] = {}

    while True:
        batch = []
        for path in sorted(run_dir.glob("*.jsonl")):
            start = offsets.get(path, 0)
            try:
                with path.open(encoding="utf-8") as f:
                    f.seek(start)
                    for line in f:
                        if line.strip():
                            try:
                                batch.append(json.loads(line))
                            except json.JSONDecodeError:
                                # A line still being written; retry next pass.
                                break
                    offsets[path] = f.tell()
            except OSError:
                continue

        for event in sorted(batch, key=lambda e: e.get("timestamp", "")):
            yield event

        if not follow:
            return
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch a GENESIS run live")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--logs-dir", default="research_logs")
    ap.add_argument("--replay", action="store_true",
                    help="Render what exists and exit, instead of following")
    ap.add_argument("--full", action="store_true",
                    help="Do not truncate agent text")
    args = ap.parse_args()

    run_dir = Path(args.logs_dir) / args.run_id
    if not run_dir.exists():
        console.print(f"[red]No such run: {run_dir}[/]")
        raise SystemExit(1)

    view = RunView(full=args.full)
    console.print(f"[dim]watching {run_dir}"
                  f"{'' if args.replay else ' — ctrl-c to stop'}[/]")
    try:
        for event in iter_events(run_dir, follow=not args.replay):
            view.handle(event)
    except KeyboardInterrupt:
        console.print("\n[dim]stopped watching[/]")


if __name__ == "__main__":
    main()
