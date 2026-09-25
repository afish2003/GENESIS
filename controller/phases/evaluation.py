"""Phase 8: Evaluation — independent evaluator scores the protocol document."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from controller.agents.base import load_system_prompt
from controller.inference.backend import Message
from controller.logging.schemas import EventEnvelope, EventType
from controller.judge import (
    anchored_violations,
    build_pairwise_prompt,
    evaluation_schema_for,
    improvement_from_choices,
    load_judge_system_prompt,
    pairwise_schema,
    reasoning_field,
    variant_instruction,
)
from controller.tasks import create_task

if TYPE_CHECKING:
    from controller.agents.base import AgentContext
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.world.state import WorldState

_logger = logging.getLogger(__name__)



async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    contexts: dict[str, AgentContext],
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Score this cycle's task artifact with a fresh-context evaluator."""
    events = cycle.pending_events
    task = create_task(config)

    if cycle.proposed_protocol is None:
        _logger.warning("No %s produced in cycle %d, skipping evaluation",
                        task.artifact_noun, cycle.cycle_id)
        return events

    proposal = cycle.proposed_protocol

    # Build doctrine context for evaluator
    doctrine_parts = []
    for name in sorted(world.doctrine):
        doctrine_parts.append(f"### {name}\n\n{world.doctrine[name].content}")
    doctrine_context = "\n\n".join(doctrine_parts)

    # Load evaluator system prompt
    evaluator_prompt = load_system_prompt(config.run_prompts_dir, "evaluator_system.md")

    proto = world.protocols.get(proposal["protocol_id"])

    # The task owns its rubric and its prompt; the judge variant owns the tail
    # that says what to output and in what order, and the schema it is parsed
    # against. Both come from the same place so the prompt cannot ask for one
    # shape while the parser expects another.
    variant = config.judge_variant

    if variant == "pairwise":
        return await _evaluate_pairwise(
            config, backend, world, cycle, contexts, logger,
            task, proposal, evaluator_prompt, events)

    prompt = task.evaluation_prompt(
        world, cycle, doctrine_context, instruction=variant_instruction(variant))
    if prompt is None:
        _logger.warning("Task %s produced nothing to evaluate in cycle %d",
                        task.name, cycle.cycle_id)
        return events

    messages = [
        Message(role="system", content=evaluator_prompt),
        Message(role="user", content=prompt),
    ]

    # An independent judge when one is configured, otherwise the agents' own
    # backend. PLAN.md:133 fixes the latter as the design — "the evaluator is
    # the same model as the agents" — and a fresh context removes episodic
    # contamination but not self-preference bias, which matters because
    # total_score is the primary dependent variable in both analysis scripts.
    judge = cycle.evaluator_backend or backend
    output = await judge.complete_structured(
        messages=messages,
        response_schema=evaluation_schema_for(task, variant),
        temperature=config.temperature_structured,
        max_retries=config.max_retries,
        # Not an agent. Labelled so the live feed distinguishes the judge's
        # text from the agents' rather than attributing it to whoever spoke
        # last, which is worse than leaving it blank.
        speaker="evaluator",
    )
    output.protocol_id = proposal["protocol_id"]

    # total_score is the primary dependent variable and is read directly by
    # analyze_run.py and compare_arms.py. It was only range-checked, never
    # compared to the parts, so a model returning five 8s with total_score 20
    # corrupted every downstream number in silence.
    dimension_total = sum(output.scores.model_dump().values())
    if output.total_score != dimension_total:
        _logger.warning(
            "Cycle %d: evaluator reported total_score %d but its dimensions sum "
            "to %d; using the sum.",
            cycle.cycle_id, output.total_score, dimension_total,
        )
        events.append(EventEnvelope(
            event_type=EventType.NOTABLE_EVENT,
            run_id=config.run_id,
            condition=config.condition.value,
            cycle_id=cycle.cycle_id,
            payload={
                "kind": "evaluation_total_mismatch",
                "reported_total": output.total_score,
                "dimension_sum": dimension_total,
                "scores": output.scores.model_dump(),
                "detail": "Reported total disagreed with the dimensions; the "
                          "sum was used so the primary metric stays internally "
                          "consistent.",
            },
        ))
        output.total_score = dimension_total

    # `anchored` claims a named defect caps the score. Whether the judge obeys
    # is the thing to measure, not to assume — a variant that is ignored looks
    # exactly like one that works until someone checks the numbers against the
    # text. Logged rather than corrected: rewriting the judge's score would
    # make the metric something the controller decided.
    reasons = getattr(output, reasoning_field(variant), {}) or {}
    if variant == "anchored":
        violations = anchored_violations(reasons, output.scores.model_dump())
        if violations:
            _logger.warning("Cycle %d: judge broke its own score caps: %s",
                            cycle.cycle_id, "; ".join(violations))
            events.append(EventEnvelope(
                event_type=EventType.NOTABLE_EVENT,
                run_id=config.run_id,
                condition=config.condition.value,
                cycle_id=cycle.cycle_id,
                payload={
                    "kind": "judge_cap_violation",
                    "violations": violations,
                    "defects": reasons,
                    "scores": output.scores.model_dump(),
                    "detail": "The judge's score contradicts the defect it "
                              "wrote for that dimension. The anchored variant "
                              "is only worth having if this stays rare.",
                },
            ))

    # Justifications keyed to something other than the rubric render as blanks
    # in the interpretation phase.
    missing = set(task.dimensions) - set(reasons)
    if missing:
        _logger.info("Cycle %d: evaluator gave no justification for %s",
                     cycle.cycle_id, sorted(missing))

    # Store in cycle state for interpretation phase
    cycle.evaluation_result = output.model_dump()

    # And put it where the memory summariser can see it. Memory is the only
    # channel by which a score can reach a later cycle, and the summariser was
    # never shown one: across 30 memory entries in DA_CONTROL not a single
    # score appears, while the doctrine tells the agents that "low scores in
    # specific dimensions should inform targeted revisions".
    summary_line = (
        f"This cycle's {task.artifact_noun} scored {output.total_score}/"
        f"{task.max_score} (" + ", ".join(
            f"{dim} {val}" for dim, val in output.scores.model_dump().items()
        ) + ")."
    )
    for ctx in contexts.values():
        ctx.cycle_events.append(summary_line)

    # Store in protocol's evaluation history
    if proto:
        proto.evaluation_history.append({
            "cycle": cycle.cycle_id,
            "total_score": output.total_score,
            "scores": output.scores.model_dump(),
        })

    events.append(EventEnvelope(
        event_type=EventType.EVALUATION_SCORE,
        run_id=config.run_id,
        condition=config.condition.value,
        cycle_id=cycle.cycle_id,
        payload={
            "task": task.name,
            "max_score": task.max_score,
            # Which model produced the score, on every score. Without it a
            # mixed corpus of runs cannot be separated after the fact.
            "evaluator_model": config.evaluator_model or config.model_name,
            "independent_evaluator": config.uses_independent_evaluator,
            **output.model_dump(),
        },
    ))

    return events


async def _evaluate_pairwise(config, backend, world, cycle, contexts, logger,
                             task, proposal, evaluator_prompt, events):
    """Score this cycle's artifact against the one it replaced.

    Chosen over absolute grading on 2026-09-20 after benchmarking: the same
    judge ranks at 96.9% against known damage and grades at 87.5%, and three
    of five dimensions were constants across 71 real evaluations. "Is this
    better than what came before" is also nearer the question the experiment
    asks than "what is this worth".

    Writes improvement / improvement_net / quality_index, NOT scores and
    total_score. Those are a different scale, eleven files read total_score,
    and putting a [-5, +5] delta in a field everything treats as a [0, 50]
    level would corrupt every comparison without erroring once.
    """
    import random

    previous = (getattr(cycle, "previous_artifact_content", "") or "").strip()
    current = (proposal.get("content") or "").strip()

    if not previous:
        # Cycle 0, or a genuinely new artifact. There is nothing to compare
        # against and inventing a baseline would make the first cycle of every
        # run an arbitrary number.
        cycle.evaluation_result = {
            "protocol_id": proposal["protocol_id"],
            "comparison": "none",
            "detail": "no prior version to compare against",
            "quality_index": 0,
        }
        events.append(EventEnvelope(
            event_type=EventType.EVALUATION_SCORE,
            run_id=config.run_id,
            condition=config.condition.value,
            cycle_id=cycle.cycle_id,
            payload=cycle.evaluation_result,
        ))
        return events

    # Which slot the NEW artifact takes is randomised per comparison. A judge
    # with a position preference would otherwise read as steady improvement or
    # steady decline for a whole run. Recorded so the bias is measurable
    # after the fact rather than assumed away.
    new_is_a = random.random() < 0.5
    a, b = (current, previous) if new_is_a else (previous, current)

    # What each version DID, when the sandbox ran it. Without this the judge
    # compares two listings by eye: the 2026-09-24 run had 9 of 12 executions
    # fail and pairwise still reported +2.00 per cycle, because nothing told it
    # the code did not run. The absolute path has always included the execution
    # report via the task's own prompt; the pairwise path dropped it, and
    # `correctness` is the dimension that most depends on it.
    evidence = _execution_evidence(cycle, new_is_a)

    judge = cycle.evaluator_backend or backend
    output = await judge.complete_structured(
        messages=[
            Message(role="system", content=load_judge_system_prompt("pairwise")),
            Message(role="user",
                    content=build_pairwise_prompt(task, a, b, evidence)),
        ],
        response_schema=pairwise_schema(task),
        temperature=config.temperature_structured,
        max_retries=config.max_retries,
        speaker="evaluator",
    )

    choices = output.choices.model_dump()
    improvement = improvement_from_choices(choices, "A" if new_is_a else "B")
    net = sum(improvement.values())
    index = int(getattr(cycle, "previous_quality_index", 0) or 0) + net

    cycle.evaluation_result = {
        "protocol_id": proposal["protocol_id"],
        "comparison": "pairwise",
        "improvement": improvement,
        "improvement_net": net,
        "quality_index": index,
        "new_version_slot": "A" if new_is_a else "B",
        "choices": choices,
        "justifications": output.justifications,
        "assessment": output.assessment,
    }
    events.append(EventEnvelope(
        event_type=EventType.EVALUATION_SCORE,
        run_id=config.run_id,
        condition=config.condition.value,
        cycle_id=cycle.cycle_id,
        payload=cycle.evaluation_result,
    ))

    better = [d for d, v in improvement.items() if v > 0]
    worse = [d for d, v in improvement.items() if v < 0]
    summary_line = (
        f"This cycle's {task.artifact_noun} was judged against the previous "
        f"version: net {net:+d} "
        f"(better: {', '.join(better) or 'none'}; "
        f"worse: {', '.join(worse) or 'none'})."
    )
    for ctx in contexts.values():
        ctx.cycle_events.append(summary_line)

    return events


def _execution_evidence(cycle, new_is_a: bool) -> str:
    """How each version behaved when run, labelled by its A/B slot.

    Empty for a task with no execution phase, which is the common case. When
    there is one, this is the only objective signal in the comparison, and
    leaving it out let a judge call a version that crashed an improvement.
    """
    def describe(result: dict | None) -> str | None:
        if not result:
            return None
        outcome = result.get("outcome", "?")
        code = result.get("exit_code", "?")
        text = ((result.get("stderr") or "").strip()
                or (result.get("stdout") or "").strip() or "(no output)")
        if len(text) > 600:
            text = text[:600] + "\n...[clipped]"
        return f"outcome {outcome}, exit code {code}\n```\n{text}\n```"

    new_run = describe(getattr(cycle, "execution_result", None))
    old_run = describe(getattr(cycle, "previous_execution", None))
    if not (new_run or old_run):
        return ""

    a_run, b_run = (new_run, old_run) if new_is_a else (old_run, new_run)
    lines = ["## What happened when each version was run\n"]
    lines.append(f"**Version A**: {a_run or '(not run)'}\n")
    lines.append(f"**Version B**: {b_run or '(not run)'}\n")
    lines.append("A version that does not run is not better than one that "
                 "does, whatever it looks like on the page. Weigh this for "
                 "correctness above all, and do not let it decide the other "
                 "dimensions by itself.\n\n")
    return "\n".join(lines)
