"""Judge configurations, so a scoring change can be measured before it ships.

The 2026-09-20 battery produced 71 evaluations in which three of five
dimensions were constants. The evaluator prompt already said "Use the whole
range. A 7 and a 9 should mean different things. If every artifact scores in
the high 30s out of 50, the scores carry no information" — and the judge did
exactly that anyway. So the fix is not more instruction. It has to be
structural, the same conclusion the trivial-agreement work reached after three
prompt variants produced 93 approvals and no rejections.

Three structural defects, each addressed by a variant here:

1. **Scores are emitted before justifications.** The generated schema orders
   `scores, justifications`, so the model commits to a number and then writes
   prose to fit it. This codebase already knows the remedy: DeliberatedVote
   orders case_for, case_against, vote for the same reason.

2. **The scale has no anchors.** Each dimension has a definition and nothing
   says what a 3, a 7 or a 9 looks like. Asked for 0-10 with no anchors, a
   model returns its modal notion of "good" — which is what a constant 9 on
   completeness is.

3. **Evidence and number are not coupled.** A judge can write "could be more
   specific" and score 9, and nothing objects. `anchored` makes the defect
   determine the ceiling: name one and you may not score above 7; find none
   and you must score at least 9. That is checkable by a reader and by a test.

`current` reproduces what the battery ran, so a benchmark comparison is
against the real thing rather than a remembered version of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Type

from pydantic import BaseModel

VARIANTS = ("current", "evidence_first", "anchored")


_SHARED_HEAD = """You score artifacts produced by a group of collaborating AI \
agents. Each artifact is scored on a set of named dimensions, 0-10 each; the \
dimensions and what they mean are given to you with the artifact.

## Your Role

You are called in a fresh context every time. You have no memory of previous
cycles and no stake in the agents' work. Score what is in front of you.
"""

_CURRENT_BODY = """
## How to Score

- **Use the whole range.** A 7 and a 9 should mean different things. If every
  artifact scores in the high 30s out of 50, the scores carry no information
  and the evaluation is doing no work.
- **Score the artifact, not the effort.** Length, confident tone and elaborate
  structure are not quality. A short precise document beats a long vague one.
- **Justify each dimension in one sentence**, referring to something specific
  in the artifact rather than restating the dimension's name.
- **Make the total the sum of the dimensions.** If they disagree, the
  dimensions are what count.
- **Say what would raise the score.** The agents read your assessment; a
  criticism they cannot act on is not useful to them.

## What You Are Not Doing

You are not editing the artifact, not proposing the next version, and not
judging the agents. You are assigning a defensible number to a document and
explaining it.
"""

_EVIDENCE_BODY = """
## How to Score

Write the justification for a dimension BEFORE you decide its number. The
justification is the reasoning; the score is its conclusion. Do not pick a
number and then explain it.

- **Quote the artifact.** Every justification must contain a short quotation,
  or name a section that is missing. A justification that could be pasted onto
  a different document is not a justification.
- **Score the artifact, not the effort.** Length, confident tone and elaborate
  structure are not quality. A short precise document beats a long vague one.
- **Make the total the sum of the dimensions.**

## What You Are Not Doing

You are not editing the artifact, not proposing the next version, and not
judging the agents. You are assigning a defensible number to a document and
explaining it.
"""

_ANCHORED_BODY = """
## How to Score

For each dimension, first name the single worst concrete defect in the
artifact on that dimension — or write exactly `NO DEFECT FOUND` if, having
looked, there is none. Then give the score. Write the defect before the score,
not after: the defect is the reasoning and the score is its conclusion.

**A named defect caps the score.** These are rules, not advice:

- You named a defect that makes the document unusable on this dimension → 0-3
- You named a defect that a reader would have to work around → 4-6
- You named a defect that is real but minor → 7-8
- You wrote `NO DEFECT FOUND` → 9-10

So a dimension cannot score 9 with a criticism attached to it, and cannot
score 6 with nothing wrong. If you cannot find a defect, say so and score
high; if you can, the number has to follow it down.

**A defect must be specific to THIS document.** Quote it, or name the section
that should exist and does not. "Could be more detailed" is not a defect —
it is true of every document ever written, and it names nothing.

**Score the artifact, not the effort.** Length, confident tone and elaborate
structure are not quality. A document that repeats itself is worse than the
same document without the repetition, not longer and therefore better.

## What You Are Not Doing

You are not editing the artifact, not proposing the next version, and not
judging the agents. You are assigning a defensible number to a document and
explaining it.
"""

_BODIES = {
    "current": _CURRENT_BODY,
    "evidence_first": _EVIDENCE_BODY,
    "anchored": _ANCHORED_BODY,
}


def load_judge_system_prompt(variant: str, framing: str = "disclosed") -> str:
    """The judge's system prompt for a variant.

    `current` is read from the rendered prompt source when one is available,
    so the benchmark's baseline is the file the runs actually used rather than
    a copy of it that can drift.
    """
    head = ("You are the Evaluator for the GENESIS research experiment. "
            if framing == "disclosed" else "You are the Evaluator. ")
    return head + _SHARED_HEAD + _BODIES[variant]


_DEFECT_INSTRUCTION = """
For each dimension give `defects[<dimension>]`: the single worst concrete
defect, quoted or named, or exactly `NO DEFECT FOUND`. Then give the scores,
respecting the caps in your instructions, a total that is their sum, and an
overall assessment."""

_EVIDENCE_INSTRUCTION = """
For each dimension give `justifications[<dimension>]` first — one sentence
containing a quotation from the artifact or naming a section that is missing.
Then give the scores, a total that is their sum, and an overall assessment."""


def variant_instruction(variant: str) -> str:
    """The tail of the user prompt: what to produce, in what order.

    Shared by the benchmark and the live evaluation phase so the thing being
    measured is the thing that ships.
    """
    if variant == "anchored":
        return _DEFECT_INSTRUCTION
    if variant == "evidence_first":
        return _EVIDENCE_INSTRUCTION
    return ("\nProvide a one-sentence justification per dimension, a total "
            "score that is the sum of them, and an overall assessment.")


def reasoning_field(variant: str) -> str:
    """Which field carries the judge's reasoning for this variant."""
    return "defects" if variant == "anchored" else "justifications"


def build_evaluation_prompt(task, doc: dict, variant: str) -> str:
    """Bench-only prompt, from a stored artifact rather than live world state.

    NOT the production prompt: it omits the doctrine context and the prior
    version, which the live phase supplies from WorldState and CycleState.
    That shifts the absolute level of doctrine_alignment and evolution_quality
    and is worth remembering when reading those two rows. It does not affect
    the comparison between variants, which all receive this same prompt, nor
    the degradations, which target completeness, precision and coherence.
    """
    header = (
        f"Evaluate the following protocol document.\n\n"
        f"## Protocol Document\n\n"
        f"**Title**: {doc.get('title', '')}\n"
        f"**Protocol ID**: {doc.get('protocol_id', '')}\n\n"
        f"{doc['content']}\n\n"
        f"Score this document on these dimensions:\n\n{task.rubric()}\n"
    )
    return header + variant_instruction(variant)


def evaluation_schema_for(task, variant: str) -> Type[BaseModel]:
    """Schema for a variant. FIELD ORDER IS THE POINT, not decoration.

    A structured decoder emits fields in declaration order, so whatever comes
    first is what the model reasons with. `current` puts `scores` first, which
    means every justification in the 2026-09-20 battery was written to fit a
    number that had already been chosen.
    """
    from pydantic import Field, create_model

    if variant == "current":
        return task.evaluation_schema()

    reason_field = reasoning_field(variant)
    description = (
        "Worst concrete defect per dimension, or 'NO DEFECT FOUND'"
        if variant == "anchored"
        else "One-sentence justification per dimension, with a quotation")

    return create_model(  # type: ignore[call-overload]
        f"{task.name.title()}Evaluation{variant.title().replace('_', '')}",
        protocol_id=(str, Field(default="", description="Set by controller")),
        **{reason_field: (dict[str, str], Field(..., description=description))},
        scores=(task.scores_schema(), ...),
        total_score=(int, Field(..., ge=0, le=task.max_score)),
        assessment=(str, Field(..., description="Overall assessment paragraph")),
    )


#: Caps from _ANCHORED_BODY, as data so a test can check the judge obeyed them.
NO_DEFECT_MIN = 9
MINOR_DEFECT_MAX = 8


def anchored_violations(defects: dict[str, str], scores: dict[str, int]) -> list[str]:
    """Dimensions where the score contradicts the defect the judge wrote.

    The whole mechanism is that a defect caps the score, so whether the judge
    actually obeys it is the thing to measure — not something to assume.
    """
    out = []
    for dim, score in scores.items():
        text = (defects.get(dim) or "").strip().upper()
        none_found = text == "NO DEFECT FOUND"
        if none_found and score < NO_DEFECT_MIN:
            out.append(f"{dim}: no defect but scored {score}")
        elif not none_found and text and score > MINOR_DEFECT_MAX:
            out.append(f"{dim}: defect named but scored {score}")
    return out
