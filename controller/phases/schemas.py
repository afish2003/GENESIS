"""Per-phase input/output Pydantic schemas.

Each phase has its own schema for what the model must return.
The controller validates every model response against these schemas.

Fields populated by the controller (not the LLM) have defaults so that
model_validate_json() succeeds before the controller overrides them.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Phase 2: Individual Reflection
# ---------------------------------------------------------------------------

class ReflectionOutput(BaseModel):
    """Output from a single agent's reflection phase."""

    agent_id: str = Field(default="", description="Set by controller after parsing")
    reflection_text: str = Field(..., description="Free-form reflection on current state")
    concerns: list[str] = Field(default_factory=list, description="Current concerns or uncertainties")
    priorities: list[str] = Field(default_factory=list, description="Priorities for this cycle")


# ---------------------------------------------------------------------------
# Phase 5: Joint Discussion
# ---------------------------------------------------------------------------

class DiscussionTurnOutput(BaseModel):
    """Output from a single discussion turn."""

    agent_id: str = Field(default="", description="Set by controller after parsing")
    turn_number: int = Field(default=0, description="Set by controller after parsing")
    message_text: str
    references: list[str] = Field(
        default_factory=list,
        description="References to doctrine, memory, or prior decisions",
    )


# ---------------------------------------------------------------------------
# Phase 6: Retrieval
# ---------------------------------------------------------------------------

class RetrievalQueryOutput(BaseModel):
    """Agent's retrieval queries for this cycle."""

    agent_id: str = Field(default="", description="Set by controller after parsing")
    queries: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to 3 retrieval queries",
    )


class RetrievalResultItem(BaseModel):
    """A single retrieved document."""

    doc_id: str
    text: str
    score: float
    source_kb: str


# ---------------------------------------------------------------------------
# Phase 7: Protocol Design
# ---------------------------------------------------------------------------

class ProtocolAction(str, Enum):
    CREATE = "create"
    REVISE = "revise"


class ProtocolProposalOutput(BaseModel):
    """Joint proposal for a protocol document."""

    action: ProtocolAction
    protocol_id: str = Field(..., description="Unique ID for the protocol document")
    title: str
    content: str = Field(..., description="Full protocol document in Markdown")
    rationale: str = Field(..., description="Why this document or revision is proposed")
    proposing_agent: str = Field(default="", description="Set by controller after parsing")


# ---------------------------------------------------------------------------
# Phase 8: Evaluation
# ---------------------------------------------------------------------------

class EvaluationScores(BaseModel):
    """Scores from the evaluator on a protocol document."""

    coherence: int = Field(..., ge=0, le=10)
    completeness: int = Field(..., ge=0, le=10)
    doctrine_alignment: int = Field(..., ge=0, le=10)
    precision: int = Field(..., ge=0, le=10)
    evolution_quality: int = Field(..., ge=0, le=10)


class EvaluationOutput(BaseModel):
    """Full evaluation result for a protocol document."""

    protocol_id: str = Field(default="", description="Set by controller after parsing")
    scores: EvaluationScores
    justifications: dict[str, str] = Field(
        ...,
        description="One-sentence justification per dimension",
    )
    total_score: int = Field(..., ge=0, le=50)
    assessment: str = Field(..., description="Overall assessment paragraph")


# ---------------------------------------------------------------------------
# Phase 9: Interpretation
# ---------------------------------------------------------------------------

class InterpretationOutput(BaseModel):
    """Agent's interpretation of evaluation results."""

    agent_id: str = Field(default="", description="Set by controller after parsing")
    interpretation_text: str
    proposed_adjustments: list[str] = Field(
        default_factory=list,
        description="Suggested changes based on evaluation feedback",
    )


# ---------------------------------------------------------------------------
# Phase 10: Doctrine Revision
# ---------------------------------------------------------------------------

class DoctrineRevisionProposal(BaseModel):
    """A proposed change to shared doctrine."""

    proposing_agent: str = Field(default="", description="Set by controller after parsing")
    target_document: str = Field(..., description="Which doctrine document to modify")
    proposed_diff: str = Field(..., description="Description of proposed changes")
    rationale: str


class DoctrineVote(BaseModel):
    """An agent's vote on a proposed doctrine revision."""

    agent_id: str = Field(default="", description="Set by controller after parsing")
    vote: str = Field(..., pattern="^(approve|reject)$")
    reason: str


# ---------------------------------------------------------------------------
# Phase 11: Identity Revision
# ---------------------------------------------------------------------------

class IdentityRevisionOutput(BaseModel):
    """Agent's updated identity statement."""

    agent_id: str = Field(default="", description="Set by controller after parsing")
    updated_identity: str = Field(..., description="Full updated identity statement")
    changes_summary: str = Field(
        default="",
        description="Brief summary of what changed and why",
    )


# ---------------------------------------------------------------------------
# Phase 12: Ethical Log
# ---------------------------------------------------------------------------

class EthicalTension(BaseModel):
    """A single ethical tension observed during the cycle."""

    description: str
    severity: str = Field(..., pattern="^(low|medium|high)$")
    resolution: str = Field(default="", description="How it was resolved, if at all")


class EthicalLogOutput(BaseModel):
    """Agent's ethical tensions from this cycle."""

    agent_id: str = Field(default="", description="Set by controller after parsing")
    tensions: list[EthicalTension] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 13: Memory Summarization
# ---------------------------------------------------------------------------

class MemorySummaryOutput(BaseModel):
    """Structured memory summary for one agent for one cycle."""

    agent_id: str = Field(default="", description="Set by controller after parsing")
    cycle_id: int = Field(default=0, description="Set by controller after parsing")
    summary: str = Field(..., description="Narrative summary of the cycle")
    key_events: list[str] = Field(default_factory=list, description="Notable events worth remembering")
    relationship_note: str = Field(
        default="",
        description="Note on the state of the partnership",
    )
    doctrine_changes: list[str] = Field(
        default_factory=list,
        description="Doctrine changes that occurred",
    )
