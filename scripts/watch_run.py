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
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
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

    def _clip(self, text: str, limit: int = 700) -> str:
        """Tidy whitespace; truncate only what would genuinely flood the view.

        The old default was 220 chars, and reflection/interpretation were passed
        160 — so the private reasoning, which is the most interesting strand in
        the run and the one you cannot get anywhere else, was the most heavily
        truncated thing on screen. Limits are now generous, rich wraps rather
        than overflows, and --full removes them entirely.
        """
        text = " ".join((text or "").split())
        if self.full or len(text) <= limit:
            return text
        return text[:limit] + " […]"

    def _thought(self, agent: str | None, label: str, text: str) -> None:
        """A private strand: what an agent thinks, rather than says.

        Indented and dimmed so it reads as an aside next to dialogue, but NOT
        clipped short — the whole point of showing it is to read it.
        """
        if not (text or "").strip():
            return
        body = Text()
        body.append(f"{agent or '?'} {label}\n", style=f"italic {self.colour(agent)}")
        body.append(self._clip(text, 1400), style="italic dim")
        console.print(Padding(body, (0, 0, 1, 11)))

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
        text = Text()
        text.append(f"{a or '?':>8} ", style=f"bold {self.colour(a)}")
        text.append("› ", style="dim")
        text.append(self._clip(p.get("message_text", ""), 900))
        # Hanging indent: a wrapped turn stays in its speaker's column instead
        # of running back to the left margin and colliding with the next one.
        console.print(Padding(text, (0, 0, 0, 2)))

    def _on_reflection_complete(self, e, p, a):
        self._thought(a, "reflects", p.get("reflection_text", ""))

    def _on_interpretation(self, e, p, a):
        self._thought(a, "reads the score", p.get("interpretation_text", ""))

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
        text = Text()
        text.append(f"  {a or '?':>8} ", style=f"bold {self.colour(a)}")
        text.append("proposes ", style="dim")
        text.append(f"{p.get('target_document','?')}", style="bold")
        console.print(text)
        summary = (p.get("proposed_diff") or "").strip()
        if summary:
            console.print(Padding(
                Text(self._clip(summary, 500), style="dim"), (0, 0, 0, 11)))

    def _on_doctrine_approved(self, e, p, a):
        applied = p.get("applied")
        mark = "[green]applied[/]" if applied else "[red]APPROVED BUT NOT APPLIED[/]"
        console.print(f"  {'doctrine':>8}   {p.get('resolved_document') or p.get('requested_document','?')} "
                      f"{mark}")

    def _on_doctrine_challenged(self, e, p, a):
        """The case against, written before the vote. Only in devil's-advocate runs."""
        objection = (p.get("objection") or "").strip()
        if not objection:
            return
        text = Text()
        text.append(f"  {a or '?':>8} ", style=f"bold {self.colour(a)}")
        text.append("argues against ", style="dim")
        text.append(f"{p.get('target_document','?')}", style="bold")
        console.print(text)
        console.print(Padding(
            Text(self._clip(objection, 900), style="yellow"), (0, 0, 1, 11)))

    def _on_doctrine_rejected(self, e, p, a):
        """The rarest event in the system, and the most interesting.

        Across every run ever collected: 93 doctrine proposals, 93 approvals,
        zero rejections. The mutual-approval gate that PLAN.md section 11 makes
        the centre of the design has never once closed. So when it finally does,
        it should be impossible to miss and the reasoning should be right there.
        """
        who = ", ".join(p.get("dissenting_agents") or []) or (a or "?")
        body = Text()
        for vote in p.get("votes") or []:
            if vote.get("vote") == "approve":
                continue
            body.append(f"{vote.get('agent_id','?')}: ",
                        style=f"bold {self.colour(vote.get('agent_id'))}")
            body.append(self._clip(vote.get("reason", ""), 700) + "\n")
        if not body:
            body.append("(no reason recorded)", style="dim")
        console.print()
        console.print(Panel(
            body,
            title=f"[bold red]REJECTED by {who}[/]",
            subtitle=f"[dim]{(p.get('proposal') or {}).get('target_document','?')}[/]",
            border_style="red", padding=(1, 2),
        ))
        console.print()

    def _on_identity_revised(self, e, p, a):
        console.print(f"  [dim]{a or '?':>8} revises identity → v{p.get('version','?')}[/]")

    def _on_memory_summary(self, e, p, a):
        self._thought(a, "remembers", p.get("summary", ""))

    def _on_scenario_injected(self, e, p, a):
        """The pressure events are the drama. Give them the whole width.

        These were one yellow line. They are also the only thing in a run that
        arrives from outside the partnership, so they should read as an
        interruption rather than as another log entry.
        """
        body = Text()
        body.append(self._clip(p.get("description", ""), 1800))
        stakes = (p.get("stated_stakes") or "").strip()
        if stakes:
            body.append("\n\nAt stake: ", style="bold yellow")
            body.append(self._clip(stakes, 600), style="yellow")
        console.print()
        console.print(Panel(
            body,
            title=f"[bold yellow]scenario · {p.get('title', '?')}[/]",
            subtitle=f"[dim]{p.get('event_id', '')} → {p.get('delivery_target', 'both')}[/]",
            border_style="yellow", padding=(1, 2),
        ))
        console.print()

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


def _unescape(text: str) -> str:
    """JSON string escapes, well enough for a preview."""
    return (text.replace("\\n", " ").replace("\\t", " ")
                .replace('\\"', '"').replace("\\\\", "\\"))


def _unescaped_quote_count(text: str) -> int:
    """Quotes that actually delimit a string, ignoring escaped ones."""
    count = 0
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            count += 1
    return count


#: The generation feed. Written by the controller, read only here, truncated
#: every cycle, and deliberately not part of the research log — so it must not
#: be picked up as an event source.
LIVE_FILE = "live.jsonl"


def iter_events(run_dir: Path, follow: bool, poll: float = 0.5):
    """Yield events in timestamp order, optionally waiting for more."""
    offsets: dict[Path, int] = {}

    while True:
        batch = []
        for path in sorted(run_dir.glob("*.jsonl")):
            if path.name == LIVE_FILE:
                continue
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


class LiveTail:
    """Follows the generation feed and hands back whatever is new.

    Re-reads the whole file each time rather than seeking from a saved offset.
    That looks wasteful and is the only correct option here: the controller
    truncates this file at the start of every cycle, and truncation is not
    reliably detectable from the size alone — rewrite it with a similar amount
    of data and a seek-based reader silently resumes mid-line and shows garbage,
    or nothing, for the rest of the run.

    The file holds one cycle of deltas, so this is a few hundred KB read a few
    times a second, locally. Cheap compared to being wrong.
    """

    def __init__(self, path: Path, keep: int = 400) -> None:
        self.path = path
        self.keep = keep
        self.buffer = ""
        self.phase = ""
        self._last_size = -1
        #: How far into the feed has already been rendered as a finished event.
        #: Indexes the full concatenated text, not the clipped buffer, so
        #: clipping for display cannot corrupt the accounting.
        self._cleared_at = 0
        self._full_len = 0

    def poll(self) -> bool:
        """Re-read the feed. True if anything changed."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return False
        if size == self._last_size:
            return False
        self._last_size = size

        try:
            raw = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False

        if len(raw) < self._cleared_at:   # truncated: a new cycle began
            self._cleared_at = 0
            self._full_len = 0

        chunks: list[str] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue          # half-written line; it arrives complete later
            self.phase = row.get("phase", self.phase)
            chunks.append(row.get("delta", ""))

        text = "".join(chunks)
        self._full_len = len(text)
        # Everything up to the last clear() has already been rendered as a
        # finished event; showing it again would print it twice. Only the tail
        # is ever displayed, so the stored buffer is clipped too.
        previous = self.buffer
        self.buffer = text[self._cleared_at:][-self.keep * 4:]
        return self.buffer != previous

    #: Values shorter than this are ids, enums and field names rather than
    #: something a person would want to read stream past.
    _MIN_READABLE = 25

    @staticmethod
    def readable(raw: str) -> str:
        """Pull the prose out of a partially-generated JSON object.

        Most phases use structured output, so the raw stream is
        `{"message_text": "I am not convinced that...` — watching JSON
        punctuation arrive is not watching an agent think. This lifts out the
        long string values, which is where the writing is, and leaves plain
        prose untouched.

        Deliberately a regex over a partial document rather than a JSON parse:
        the point is to show text before it is complete, and no parser will
        accept a half-written object.
        """
        if "{" not in raw[:200]:
            return raw

        values = [
            _unescape(m[1:-1])
            for m in re.findall(r'"(?:[^"\\]|\\.)*"', raw)
            if len(m) - 2 >= LiveTail._MIN_READABLE
        ]

        # An odd number of unescaped quotes means the last one opened a string
        # that is still being written — which is the most interesting part, so
        # recover it even though it is not yet valid JSON.
        if _unescaped_quote_count(raw) % 2 == 1:
            partial = _unescape(raw.rsplit('"', 1)[-1])
            if partial.strip():
                values.append(partial)

        return " … ".join(v for v in values if v.strip())

    def render(self) -> Text:
        text = " ".join(self.readable(self.buffer).split())[-self.keep:]
        out = Text()
        if not text:
            return out
        out.append(f"  {self.phase or 'thinking'} ", style="bold dim")
        out.append("· ", style="dim")
        out.append(text, style="italic dim")
        return out

    def clear(self) -> None:
        """Drop what has been rendered as a finished event.

        Records how far into the feed we had got, so a later re-read does not
        resurrect it — the reader re-reads the whole file every poll.
        """
        self._cleared_at = self._full_len
        self.buffer = ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch a GENESIS run live")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--logs-dir", default="research_logs")
    ap.add_argument("--replay", action="store_true",
                    help="Render what exists and exit, instead of following")
    ap.add_argument("--full", action="store_true",
                    help="Do not truncate agent text")
    ap.add_argument("--no-stream", action="store_true",
                    help="Do not show text as it is generated")
    args = ap.parse_args()

    run_dir = Path(args.logs_dir) / args.run_id
    if not run_dir.exists():
        console.print(f"[red]No such run: {run_dir}[/]")
        raise SystemExit(1)

    view = RunView(full=args.full)
    console.print(f"[dim]watching {run_dir}"
                  f"{'' if args.replay else ' — ctrl-c to stop'}[/]")

    if args.replay or args.no_stream:
        try:
            for event in iter_events(run_dir, follow=not args.replay):
                view.handle(event)
        except KeyboardInterrupt:
            console.print("\n[dim]stopped watching[/]")
        return

    _follow_live(run_dir, view)


def _follow_live(run_dir: Path, view: RunView, poll: float = 0.25) -> None:
    """Follow a run with the in-progress generation pinned to the bottom.

    Finished events scroll normally; the text currently being generated sits in
    a transient region beneath them and is discarded once the phase's event
    lands, so nothing is printed twice. Without this a phase is nine seconds of
    silence followed by a finished block, and you never see them think.
    """
    events = iter_events(run_dir, follow=True, poll=0.0)
    tail = LiveTail(run_dir / LIVE_FILE)

    try:
        with Live(Text(""), console=console, transient=True,
                  refresh_per_second=12) as live:
            while True:
                rendered_any = False
                for event in events:
                    # console.print inside a Live prints ABOVE the live region,
                    # which is what keeps the transcript and the stream from
                    # interleaving.
                    view.handle(event)
                    rendered_any = True
                    if event.get("event_type") in _SETTLED_BY:
                        tail.clear()
                        live.update(Text(""))
                    break

                if tail.poll() or rendered_any:
                    live.update(tail.render())

                if not rendered_any:
                    time.sleep(poll)
    except KeyboardInterrupt:
        console.print("\n[dim]stopped watching[/]")


#: Events that mean "the thing being generated is now finished and rendered".
#: The streamed copy is dropped so the same text is never shown twice.
_SETTLED_BY = {
    "DISCUSSION_TURN", "REFLECTION_COMPLETE", "INTERPRETATION",
    "PROTOCOL_PROPOSED", "DOCTRINE_PROPOSED", "DOCTRINE_APPROVED",
    "DOCTRINE_REJECTED", "IDENTITY_REVISED", "MEMORY_SUMMARY",
    "EVALUATION_SCORE", "CODE_EXECUTION", "PHASE_END",
}


if __name__ == "__main__":
    main()
