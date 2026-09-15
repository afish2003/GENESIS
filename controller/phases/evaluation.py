"""Phase 8: Evaluation — independent evaluator scores the protocol document."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from controller.agents.base import load_system_prompt
from controller.inference.backend import Message
from controller.logging.schemas import EventEnvelope, EventType
from controller.phases.schemas import EvaluationOutput

if TYPE_CHECKING:
    from controller.agents.base import AgentContext
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.world.state import WorldState

_logger = logging.getLogger(__name__)

EVALUATION_PROMPT = """Evaluate the following protocol document.

## Current Doctrine Context

{doctrine_context}

## Protocol Document

**Title**: {title}
**Protocol ID**: {protocol_id}
**Version**: {version}
**Action**: {action}

{content}

Score this document on the five dimensions (coherence, completeness, doctrine_alignment, precision, evolution_quality), each 0-10. Provide justifications, a total score, and an overall assessment."""


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    contexts: dict[str, AgentContext],
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Evaluate the proposed protocol using the evaluator (fresh context, no memory)."""
    events = []

    if cycle.proposed_protocol is None:
        _logger.warning("No protocol proposed in cycle %d, skipping evaluation", cycle.cycle_id)
        return events

    proposal = cycle.proposed_protocol

    # Build doctrine context for evaluator
    doctrine_parts = []
    for name in sorted(world.doctrine):
        doctrine_parts.append(f"### {name}\n\n{world.doctrine[name].content}")
    doctrine_context = "\n\n".join(doctrine_parts)

    # Load evaluator system prompt
    evaluator_prompt = load_system_prompt(config.prompts_dir, "evaluator_system.md")

    # Get protocol version for context
    proto = world.protocols.get(proposal["protocol_id"])
    version = proto.version if proto else 1

    messages = [
        Message(role="system", content=evaluator_prompt),
        Message(role="user", content=EVALUATION_PROMPT.format(
            doctrine_context=doctrine_context,
            title=proposal["title"],
            protocol_id=proposal["protocol_id"],
            version=version,
            action=proposal["action"],
            content=proposal["content"],
        )),
    ]

    output = await backend.complete_structured(
        messages=messages,
        response_schema=EvaluationOutput,
        temperature=config.temperature_structured,
        max_retries=config.max_retries,
    )
    output.protocol_id = proposal["protocol_id"]

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
        payload=output.model_dump(),
    ))

    return events
