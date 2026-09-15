"""The v1 sandbox task: author and revise protocol documents.

Behaviourally identical to the hardcoded version it replaces — same prompt,
same schema, same five dimensions — so runs before and after the Task
extraction remain comparable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

from pydantic import BaseModel

from controller.phases.schemas import ProtocolProposalOutput
from controller.tasks.base import Task
from controller.world.artifacts import ProtocolDocument
from controller.world.paths import safe_artifact_id

if TYPE_CHECKING:
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.world.state import WorldState

DESIGN_PROMPT = """This is the protocol design phase. You and {partner_name} should collaboratively propose either a new protocol document or a revision to an existing one.

Current active protocols: {protocol_list}

A protocol document should include:
- Purpose: What problem or need does this protocol address?
- Scope: What does this protocol cover and not cover?
- Procedure: Step-by-step operational procedure
- Evaluation criteria: How to assess whether the protocol is being followed effectively
- Known limitations: Honest assessment of gaps or weaknesses

Based on the discussion this cycle, the evaluation feedback from previous cycles, and your priorities, propose a protocol action (create or revise).

If revising, specify which protocol and what changes. If creating, propose a complete new document."""

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


class ProtocolTask(Task):
    name = "protocol"
    artifact_noun = "protocol document"
    dimensions = (
        "coherence",
        "completeness",
        "doctrine_alignment",
        "precision",
        "evolution_quality",
    )

    def design_prompt(self, config: RunConfig, world: WorldState, cycle: CycleState) -> str:
        active = [f"- {pid}: {p.title}" for pid, p in world.protocols.items() if not p.archived]
        return DESIGN_PROMPT.format(
            partner_name=config.partner_names(config.agents[0]),
            protocol_list="\n".join(active) if active else "(none yet)",
        )

    def output_schema(self) -> Type[BaseModel]:
        return ProtocolProposalOutput

    def apply(self, world: WorldState, output: BaseModel, cycle: CycleState) -> str:
        assert isinstance(output, ProtocolProposalOutput)
        # Model-supplied and becomes a filename — sanitise at ingress so world
        # state, logs and evaluation all carry the same safe identifier.
        output.protocol_id = safe_artifact_id(
            output.protocol_id, fallback=f"protocol_cycle{cycle.cycle_id}"
        )
        pid = output.protocol_id

        if output.action.value == "create":
            world.protocols[pid] = ProtocolDocument(
                protocol_id=pid,
                title=output.title,
                content=output.content,
                version=1,
                created_cycle=cycle.cycle_id,
                last_modified_cycle=cycle.cycle_id,
            )
        elif pid in world.protocols:
            proto = world.protocols[pid]
            proto.content = output.content
            proto.title = output.title
            proto.version += 1
            proto.last_modified_cycle = cycle.cycle_id
        return pid

    def evaluation_prompt(
        self, world: WorldState, cycle: CycleState, doctrine_context: str
    ) -> str | None:
        proposal = cycle.proposed_protocol
        if not proposal:
            return None
        proto = world.protocols.get(proposal["protocol_id"])
        return EVALUATION_PROMPT.format(
            doctrine_context=doctrine_context,
            title=proposal.get("title", ""),
            protocol_id=proposal.get("protocol_id", ""),
            version=proto.version if proto else 1,
            action=proposal.get("action", ""),
            content=proposal.get("content", ""),
        )
