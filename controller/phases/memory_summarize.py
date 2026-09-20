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
    events = cycle.pending_events

    # Load memory summarizer prompt
    summarizer_prompt = load_system_prompt(config.run_prompts_dir, "memory_summarizer.md")

    agent_names = {a: config.display_name(a) for a in config.agents}

    for agent_id in config.agents:
        ctx = contexts[agent_id]

        # Build the transcript the summariser prompt actually promises.
        #
        # This was discussion turns alone, labelled by role. The prompt tells
        # the summariser it receives "reflections, discussion turns, proposals,
        # evaluation feedback, doctrine decisions, and any scenario events" and
        # asks it to record "scores received, doctrine changes" — none of which
        # were there. It filled them in anyway: in DA_CONTROL cycle 2, two
        # doctrine revisions were approved and both agents' memory reads "No
        # doctrine changes were proposed, approved, or rejected this cycle."
        # Across 30 entries, not one contains a score.
        #
        # That is not a prompt defect. Memory is the only channel carrying
        # anything between cycles, so every downstream cycle was reasoning from
        # a record that was confidently wrong about the events the experiment
        # exists to measure — and BASELINE vs MEM_RESET was comparing two
        # versions of a partly-invented history.
        transcript_lines = []
        for msg in ctx.get_discussion_messages():
            # Named, not "[user]". The summariser could not tell who was
            # speaking and guessed: one DA_CONTROL entry opens "Axiom and the
            # user agreed", where "the user" is Flux.
            speaker = (config.display_name(agent_id) if msg.role == "assistant"
                       else "Partner")
            transcript_lines.append(f"[{speaker}]: {msg.content}")
        transcript = ("\n\n".join(transcript_lines) if transcript_lines
                      else "(No discussion this cycle)")

        if ctx.cycle_events:
            transcript += (
                "\n\n---\n\nWhat else happened to you this cycle:\n"
                + "\n".join(f"- {e}" for e in ctx.cycle_events)
            )

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
