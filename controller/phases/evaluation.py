"""Phase 8: Evaluation — independent evaluator scores the protocol document."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from controller.agents.base import load_system_prompt
from controller.inference.backend import Message
from controller.logging.schemas import EventEnvelope, EventType
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

    # The task owns its rubric and its prompt.
    prompt = task.evaluation_prompt(world, cycle, doctrine_context)
    if prompt is None:
        _logger.warning("Task %s produced nothing to evaluate in cycle %d",
                        task.name, cycle.cycle_id)
        return events

    messages = [
        Message(role="system", content=evaluator_prompt),
        Message(role="user", content=prompt),
    ]

    output = await backend.complete_structured(
        messages=messages,
        response_schema=task.evaluation_schema(),
        temperature=config.temperature_structured,
        max_retries=config.max_retries,
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

    # Justifications keyed to something other than the rubric render as blanks
    # in the interpretation phase.
    missing = set(task.dimensions) - set(output.justifications)
    if missing:
        _logger.info("Cycle %d: evaluator gave no justification for %s",
                     cycle.cycle_id, sorted(missing))

    # Store in cycle state for interpretation phase
    cycle.evaluation_result = output.model_dump()

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
        payload={"task": task.name, "max_score": task.max_score, **output.model_dump()},
    ))

    return events
