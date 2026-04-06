"""Phase 10: Doctrine Revision — agents propose and vote on doctrine changes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from controller.inference.backend import Message
from controller.logging.schemas import EventEnvelope, EventType
from controller.phases.schemas import DoctrineRevisionProposal, DoctrineVote

if TYPE_CHECKING:
    from controller.agents.base import AgentContext
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.world.state import WorldState

_logger = logging.getLogger(__name__)

DOCTRINE_PROPOSAL_PROMPT = """This is the doctrine revision phase. Based on this cycle's discussion, evaluation feedback, and your reflection, do you want to propose a change to any doctrine document?

Available doctrine documents:
{doctrine_list}

If you have a proposal, describe the specific changes and your rationale. If no changes are needed this cycle, respond with an empty proposed_diff and a rationale explaining why the current doctrine is adequate.

Target document options: {doc_names}"""

DOCTRINE_VOTE_PROMPT = """Your partner {proposer_name} has proposed a doctrine revision:

**Target document**: {target_document}
**Proposed changes**: {proposed_diff}
**Rationale**: {rationale}

Do you approve or reject this proposal? Provide your vote and reason."""


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    contexts: dict[str, AgentContext],
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Handle doctrine revision proposals and mutual approval voting."""
    events = []

    doc_names = ", ".join(sorted(world.doctrine.keys()))
    doctrine_list = "\n".join(
        f"- {name}: {doc.content[:100]}..." for name, doc in sorted(world.doctrine.items())
    )

    # Each agent can propose (Axiom first)
    for proposer_id, voter_id in [("axiom", "flux"), ("flux", "axiom")]:
        proposer_ctx = contexts[proposer_id]
        messages = [
            proposer_ctx.build_system_message(),
        ]
        for msg in proposer_ctx.get_discussion_messages():
            messages.append(msg)
        messages.append(Message(
            role="user",
            content=DOCTRINE_PROPOSAL_PROMPT.format(
                doctrine_list=doctrine_list,
                doc_names=doc_names,
            ),
        ))

        proposal = await backend.complete_structured(
            messages=messages,
            response_schema=DoctrineRevisionProposal,
            temperature=config.temperature_discussion,
            max_retries=config.max_retries,
        )
        proposal.proposing_agent = proposer_id

        # Skip if no actual changes proposed
        if not proposal.proposed_diff.strip():
            continue

        events.append(EventEnvelope(
            event_type=EventType.DOCTRINE_PROPOSED,
            run_id=config.run_id,
            condition=config.condition.value,
            cycle_id=cycle.cycle_id,
            agent_id=proposer_id,
            payload=proposal.model_dump(),
        ))

        # Get vote from the other agent
        voter_ctx = contexts[voter_id]
        partner_names = {"axiom": "Axiom", "flux": "Flux"}
        vote_messages = [
            voter_ctx.build_system_message(),
        ]
        for msg in voter_ctx.get_discussion_messages():
            vote_messages.append(msg)
        vote_messages.append(Message(
            role="user",
            content=DOCTRINE_VOTE_PROMPT.format(
                proposer_name=partner_names[proposer_id],
                target_document=proposal.target_document,
                proposed_diff=proposal.proposed_diff,
                rationale=proposal.rationale,
            ),
        ))

        vote = await backend.complete_structured(
            messages=vote_messages,
            response_schema=DoctrineVote,
            temperature=config.temperature_structured,
            max_retries=config.max_retries,
        )
        vote.agent_id = voter_id

        if vote.vote == "approve":
            events.append(EventEnvelope(
                event_type=EventType.DOCTRINE_APPROVED,
                run_id=config.run_id,
                condition=config.condition.value,
                cycle_id=cycle.cycle_id,
                agent_id=voter_id,
                payload={
                    "proposal": proposal.model_dump(),
                    "vote": vote.model_dump(),
                },
            ))
            # Apply the change
            target = proposal.target_document
            if target in world.doctrine:
                doc = world.doctrine[target]
                # The proposed_diff is a description; in a more sophisticated
                # version we'd apply an actual diff. For v1, we append the
                # proposed changes as a revision note and let the model
                # produce the full updated content in future cycles.
                doc.content += f"\n\n---\n*Revision (cycle {cycle.cycle_id}, proposed by {proposer_id})*: {proposal.proposed_diff}"
                doc.last_modified_cycle = cycle.cycle_id
                doc.version += 1
        else:
            events.append(EventEnvelope(
                event_type=EventType.DOCTRINE_REJECTED,
                run_id=config.run_id,
                condition=config.condition.value,
                cycle_id=cycle.cycle_id,
                agent_id=voter_id,
                payload={
                    "proposal": proposal.model_dump(),
                    "vote": vote.model_dump(),
                },
            ))

    return events
