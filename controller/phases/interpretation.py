"""Phase 9: Interpretation — agents receive evaluation and discuss implications."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from controller.inference.backend import Message
from controller.logging.schemas import EventEnvelope, EventType
from controller.phases.schemas import InterpretationOutput

if TYPE_CHECKING:
    from controller.agents.base import AgentContext
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.world.state import WorldState

_logger = logging.getLogger(__name__)

INTERPRETATION_PROMPT = """The evaluator has scored the protocol document. Review the results and share your interpretation.

## Evaluation Results

{evaluation_text}

What do these scores tell us? What should we adjust in future cycles? Are there patterns across evaluations we should address?"""


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    contexts: dict[str, AgentContext],
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Both agents interpret the evaluation results."""
    events = []

    if cycle.evaluation_result is None:
        return events

    eval_data = cycle.evaluation_result
    evaluation_text = (
        f"Protocol: {eval_data.get('protocol_id', 'unknown')}\n"
        f"Total Score: {eval_data.get('total_score', 'N/A')}/50\n\n"
        f"Scores:\n"
    )
    scores = eval_data.get("scores", {})
    justifications = eval_data.get("justifications", {})
    for dim, score in scores.items():
        justification = justifications.get(dim, "")
        evaluation_text += f"  - {dim}: {score}/10 — {justification}\n"
    evaluation_text += f"\nAssessment: {eval_data.get('assessment', '')}"

    for agent_id in ["axiom", "flux"]:
        ctx = contexts[agent_id]
        messages = [
            ctx.build_system_message(),
        ]
        for msg in ctx.get_discussion_messages():
            messages.append(msg)
        messages.append(Message(
            role="user",
            content=INTERPRETATION_PROMPT.format(evaluation_text=evaluation_text),
        ))

        output = await backend.complete_structured(
            messages=messages,
            response_schema=InterpretationOutput,
            temperature=config.temperature_discussion,
            max_retries=config.max_retries,
        )
        output.agent_id = agent_id

        events.append(EventEnvelope(
            event_type=EventType.INTERPRETATION,
            run_id=config.run_id,
            condition=config.condition.value,
            cycle_id=cycle.cycle_id,
            agent_id=agent_id,
            payload=output.model_dump(),
        ))

    return events
