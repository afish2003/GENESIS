"""Phase 13: Memory Summarization — compress cycle to structured summary."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from controller.agents.base import load_system_prompt
from controller.inference.backend import Message
from controller.logging.schemas import EventEnvelope, EventType
from controller.phases.schemas import MemorySummaryOutput
from controller.world.artifacts import MemoryEntry

if TYPE_CHECKING:
    from controller.agents.base import AgentContext
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.world.state import WorldState

_logger = logging.getLogger(__name__)

MEMORY_PROMPT = """Summarize this cycle's experience for {agent_name}'s memory journal.

## Cycle {cycle_id} Transcript

{transcript}

Produce a structured memory summary capturing the essential information {agent_name} would need in future cycles."""


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    contexts: dict[str, AgentContext],
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Summarize the cycle for each agent's memory journal."""
    events = []

    # Load memory summarizer prompt
    summarizer_prompt = load_system_prompt(config.prompts_dir, "memory_summarizer.md")

    agent_names = {"axiom": "Axiom", "flux": "Flux"}

    for agent_id in ["axiom", "flux"]:
        ctx = contexts[agent_id]

        # Build transcript from discussion history
        transcript_lines = []
        for msg in ctx.get_discussion_messages():
            transcript_lines.append(f"[{msg.role}]: {msg.content}")
        transcript = "\n\n".join(transcript_lines) if transcript_lines else "(No discussion this cycle)"

        messages = [
            Message(role="system", content=summarizer_prompt),
            Message(role="user", content=MEMORY_PROMPT.format(
                agent_name=agent_names[agent_id],
                cycle_id=cycle.cycle_id,
                transcript=transcript,
            )),
        ]

        output = await backend.complete_structured(
            messages=messages,
            response_schema=MemorySummaryOutput,
            temperature=config.temperature_structured,
            max_retries=config.max_retries,
        )
        output.agent_id = agent_id
        output.cycle_id = cycle.cycle_id

        # Append to agent's memory journal
        world.memory[agent_id].append(MemoryEntry(
            cycle_id=cycle.cycle_id,
            summary=output.summary,
            key_events=output.key_events,
            relationship_note=output.relationship_note,
            doctrine_changes=output.doctrine_changes,
        ))

        events.append(EventEnvelope(
            event_type=EventType.MEMORY_SUMMARY,
            run_id=config.run_id,
            condition=config.condition.value,
            cycle_id=cycle.cycle_id,
            agent_id=agent_id,
            payload=output.model_dump(),
        ))

    return events
