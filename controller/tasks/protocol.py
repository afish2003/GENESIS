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
{prior_version}
Score this document on the five dimensions (coherence, completeness, doctrine_alignment, precision, evolution_quality), each 0-10. Provide justifications, a total score, and an overall assessment."""


#: The version being replaced, shown only when there is one. `evolution_quality`
#: asks whether a revision improves on the prior version and could never see it,
#: so its "if you do not have the prior version" fallback fired every time.
PRIOR_VERSION_SECTION = """
## The version this replaces

```
{previous}
```

Compare the two directly when scoring evolution_quality: what did the revision
add, remove, or reword, and is the document better for it? A change that is
merely longer, or that restates what was already there, is not an improvement.
"""


def prior_version_section(cycle, limit: int = 4000) -> str:
    """The replaced text, or empty for a genuinely new artifact."""
    previous = getattr(cycle, "previous_artifact_content", None) if cycle else None
    if not (previous or "").strip():
        return ""
    if len(previous) > limit:
        previous = previous[:limit] + "\n...[truncated for the evaluator]"
    return PRIOR_VERSION_SECTION.format(previous=previous)


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

    def apply(self, world: WorldState, output: BaseModel, cycle: CycleState,
              max_tokens: int | None = None) -> str:
        assert isinstance(output, ProtocolProposalOutput)
        # Model-supplied and becomes a filename — sanitise at ingress so world
        # state, logs and evaluation all carry the same safe identifier.
        output.protocol_id = safe_artifact_id(
            output.protocol_id, fallback=f"protocol_cycle{cycle.cycle_id}"
        )
        pid = output.protocol_id

        if max_tokens:
            output.content, truncated = self.enforce_length(output.content, max_tokens)
            if truncated:
                import logging
                logging.getLogger(__name__).warning(
                    "Cycle %d: %s truncated at the configured length limit",
                    cycle.cycle_id, pid,
                )

        existing = world.protocols.get(pid)

        # A "create" naming an existing id used to replace the entry outright,
        # losing its version, created_cycle and entire evaluation_history. A
        # "revise" naming an id that does not exist matched neither branch, so
        # the artifact never reached world state while PROTOCOL_PROPOSED and
        # EVALUATION_SCORE were still logged for it. Treat the action as a hint
        # and let the id decide, as CodeTask already did.
        if existing is None:
            world.protocols[pid] = ProtocolDocument(
                protocol_id=pid,
                title=output.title,
                content=output.content,
                version=1,
                created_cycle=cycle.cycle_id,
                last_modified_cycle=cycle.cycle_id,
            )
        else:
            # Capture before overwriting — this is the only moment the prior
            # text exists alongside the new one.
            cycle.previous_artifact_content = existing.content
            existing.content = output.content
            existing.title = output.title
            existing.version += 1
            existing.last_modified_cycle = cycle.cycle_id
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
            prior_version=prior_version_section(cycle),
        )
