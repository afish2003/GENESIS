"""Pydantic models for all world-state artifacts.

These are the typed in-memory representations of every persistent
artifact the WorldState loads from and saves to disk.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DoctrineDocument(BaseModel):
    """A living doctrine document (Markdown content with version tracking)."""

    filename: str
    content: str
    version: int = 1
    last_modified_cycle: int = -1


class IdentityStatement(BaseModel):
    """An agent's persistent identity statement."""

    agent_id: str
    content: str
    version: int = 1
    last_modified_cycle: int = -1


class MemoryEntry(BaseModel):
    """A single summarized memory entry in an agent's journal."""

    cycle_id: int
    summary: str
    key_events: list[str] = Field(default_factory=list)
    relationship_note: Optional[str] = None
    doctrine_changes: Optional[str] = None


class ProtocolDocument(BaseModel):
    """A protocol document produced in the sandbox task."""

    protocol_id: str
    title: str
    content: str
    version: int = 1
    created_cycle: int = 0
    last_modified_cycle: int = 0
    archived: bool = False


class EthicalLogEntry(BaseModel):
    """A single ethical tension log entry."""

    cycle_id: int
    agent_id: str
    description: str
    severity: str = "low"
    resolution: Optional[str] = None


class RelationshipLogEntry(BaseModel):
    """A single entry in the shared relationship log."""

    cycle_id: int
    event_type: str
    description: str
    agents_involved: list[str] = Field(default_factory=list)


class ScenarioEvent(BaseModel):
    """A scenario event from the injection library."""

    event_id: str
    title: str
    description: str
    stated_stakes: str
    delivery_target: str = "both"
    trigger_cycle: int
    follow_up_cycle: Optional[int] = None
    phase_assignment: Optional[str] = None


class Checkpoint(BaseModel):
    """Run checkpoint written after each completed cycle."""

    run_id: str
    last_completed_cycle: int
    world_state_hash: str
    timestamp: Optional[str] = None
