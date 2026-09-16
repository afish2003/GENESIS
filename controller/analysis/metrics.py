"""The metrics the April 2026 proposal specified, computed from run logs.

The proposal defines twelve (M1-M12) with formulas, ranges and stated
limitations. The build shipped roughly one of them — retrieval counts — while
`analyze_run.py`'s docstring advertised five it never computed. This module is
where they actually live.

Two of them are load-bearing right now, because the proposal also specified
failure modes with detection thresholds and neither detector was ever built:

    F02 Trivial Agreement   "CS = 1.0 for 5+ cycles"    -> M4
    F03 Repetitive Looping  "DSI > 0.98 for 10+ cycles" -> M2

Run against SHAKE_002 both fire: 23 of 23 cycles at CS = 1.0, and 17 doctrine
transitions above the DSI threshold. The failure the design predicted has been
running continuously and undetected.

Everything here reads logged events. Nothing re-reads a run's logs from inside
the controller — that constraint applies to the run itself, and these are
analysis functions called afterwards or by the watchdog on events it already
holds in memory.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from dataclasses import dataclass, field

#: F02: coordination strength this high for this many cycles means the agents
#: are not really deciding anything. From the proposal's failure-mode table.
TRIVIAL_AGREEMENT_CYCLES = 5
#: F03: doctrine this stable for this long means the loop is not going anywhere.
REPETITIVE_LOOPING_CYCLES = 10
STABILITY_THRESHOLD = 0.98


@dataclass
class Series:
    """A metric over cycles, plus the values needed to describe it."""

    name: str
    points: list[tuple[int, float]] = field(default_factory=list)

    @property
    def values(self) -> list[float]:
        return [v for _, v in self.points]

    @property
    def mean(self) -> float | None:
        vals = self.values
        return sum(vals) / len(vals) if vals else None

    def longest_run(self, predicate) -> int:
        """Longest consecutive stretch satisfying `predicate`.

        The failure-mode thresholds are all "for N+ cycles", so consecutive
        length is the quantity, not a count or a mean.
        """
        best = current = 0
        for _, value in self.points:
            current = current + 1 if predicate(value) else 0
            best = max(best, current)
        return best


def coordination_strength(events: list[dict]) -> Series:
    """M4: agreed decisions / total decisions, per cycle.

    `CS_t = agreed_first_exchange / total_decisions`. Range [0, 1]. The
    proposal's stated limitation is the important one: "high CS could indicate
    alignment or premature acquiescence" — it does not tell you which, only
    that the gate is never closing.
    """
    tally: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for event in events:
        kind = event.get("event_type")
        if kind == "DOCTRINE_APPROVED":
            tally[event.get("cycle_id", 0)][0] += 1
        elif kind == "DOCTRINE_REJECTED":
            tally[event.get("cycle_id", 0)][1] += 1

    series = Series("coordination_strength")
    for cycle in sorted(tally):
        approved, rejected = tally[cycle]
        total = approved + rejected
        if total:
            series.points.append((cycle, approved / total))
    return series


def doctrine_stability(events: list[dict]) -> dict[str, Series]:
    """M2: 1 - edit_distance / max(len), between consecutive versions.

    One series per document. Uses difflib's ratio rather than a true Levenshtein
    distance — the proposal's formula names Levenshtein, and on documents of a
    few thousand characters the two agree closely enough for a threshold test
    while difflib needs no dependency. Stated here rather than quietly
    substituted.
    """
    versions: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for event in events:
        if event.get("event_type") != "DOCTRINE_PROPOSED":
            continue
        payload = event.get("payload") or {}
        content = payload.get("revised_content")
        if content:
            versions[payload.get("target_document", "?")].append(
                (event.get("cycle_id", 0), content))

    out: dict[str, Series] = {}
    for document, history in versions.items():
        series = Series(f"doctrine_stability:{document}")
        for (_, before), (cycle, after) in zip(history, history[1:]):
            series.points.append(
                (cycle, difflib.SequenceMatcher(None, before, after).ratio()))
        if series.points:
            out[document] = series
    return out


def identity_texts(events: list[dict]) -> dict[str, list[tuple[int, str]]]:
    """Identity statements per agent per cycle, if the run recorded them.

    Returns empty for any run before 2026-09-16: IDENTITY_REVISED carried only
    a changes_summary and a version number, so M1 and M5 are not merely
    unimplemented for those runs, they are uncomputable.
    """
    by_agent: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for event in events:
        if event.get("event_type") != "IDENTITY_REVISED":
            continue
        text = (event.get("payload") or {}).get("identity_text")
        if text:
            by_agent[event.get("agent_id") or "?"].append(
                (event.get("cycle_id", 0), text))
    return dict(by_agent)


def failure_modes(events: list[dict]) -> list[str]:
    """Which of the proposal's predicted failure modes the run is in.

    Returns human-readable findings rather than booleans, because the useful
    output is "F02 fires: 23 consecutive cycles at CS = 1.0 (rule needs 5)",
    not True.
    """
    findings: list[str] = []

    cs = coordination_strength(events)
    trivial = cs.longest_run(lambda v: v >= 1.0)
    if trivial >= TRIVIAL_AGREEMENT_CYCLES:
        findings.append(
            f"F02 Trivial Agreement: {trivial} consecutive cycles with every "
            f"proposal approved (rule fires at {TRIVIAL_AGREEMENT_CYCLES}). "
            f"The mutual-approval gate is not gating anything."
        )

    for document, series in doctrine_stability(events).items():
        looping = series.longest_run(lambda v: v > STABILITY_THRESHOLD)
        if looping >= REPETITIVE_LOOPING_CYCLES:
            findings.append(
                f"F03 Repetitive Looping: {document} unchanged beyond "
                f"{STABILITY_THRESHOLD} similarity across {looping} consecutive "
                f"revisions (rule fires at {REPETITIVE_LOOPING_CYCLES})."
            )

    return findings
