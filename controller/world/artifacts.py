"""Typed dataclasses for all world-state artifacts.

These represent the persistent world that agents interact with.
All artifacts are loaded at cycle start and persisted at cycle end.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class DoctrineDocument(BaseModel):
    """A single doctrine document (manifesto, constitution, or doctrine)."""

    filename: str
    content: str
    last_modified_cycle: int = 0
    version: int = 1


class IdentityStatement(BaseModel):
    """An agent's current identity statement."""

    agent_id: str
    content: str
    last_modified_cycle: int = 0
    version: int = 1


class MemoryEntry(BaseModel):
    """A single memory journal entry."""

    cycle_id: int
    summary: str
    key_events: list[str] = Field(default_factory=list)
    relationship_note: str = ""
    doctrine_changes: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProtocolDocument(BaseModel):
    """A protocol document in the sandbox."""

    protocol_id: str
    title: str
    content: str
    version: int = 1
    created_cycle: int = 0
    last_modified_cycle: int = 0
    archived: bool = False
    evaluation_history: list[dict] = Field(default_factory=list)


class EthicalLogEntry(BaseModel):
    """An entry in the ethical tradeoff log."""

    cycle_id: int
    agent_id: str
    description: str
    severity: str
    resolution: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RelationshipLogEntry(BaseModel):
    """An entry in the relationship log."""

    cycle_id: int
    agent_id: str
    note: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScenarioEvent(BaseModel):
    """A scenario event from the injection library."""

    event_id: str
    title: str
    description: str
    stated_stakes: str
    delivery_target: str = Field(
        default="both",
        pattern="^(both|axiom|flux)$",
        description="Which agent(s) receive the event",
    )
    trigger_cycle: int
    followup_cycle: Optional[int] = None
    phase: str = Field(default="", description="Thematic phase assignment")
    tags: list[str] = Field(default_factory=list)


class Checkpoint(BaseModel):
    """Run checkpoint for resume support."""

    run_id: str
    last_completed_cycle: int
    world_state_hash: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
