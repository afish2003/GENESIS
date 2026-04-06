"""Event envelope and all log event schemas.

Every logged event shares the EventEnvelope structure. The payload
is discriminated by event_type. All timestamps are ISO 8601 UTC.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    # Lifecycle
    RUN_START = "RUN_START"
    RUN_END = "RUN_END"
    CYCLE_START = "CYCLE_START"
    CYCLE_END = "CYCLE_END"
    PHASE_START = "PHASE_START"
    PHASE_END = "PHASE_END"

    # Phase outputs
    REFLECTION_COMPLETE = "REFLECTION_COMPLETE"
    DISCUSSION_TURN = "DISCUSSION_TURN"
    SCENARIO_INJECTED = "SCENARIO_INJECTED"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    RETRIEVAL_RESULT = "RETRIEVAL_RESULT"
    PROTOCOL_PROPOSED = "PROTOCOL_PROPOSED"
    EVALUATION_SCORE = "EVALUATION_SCORE"
    DOCTRINE_PROPOSED = "DOCTRINE_PROPOSED"
    DOCTRINE_APPROVED = "DOCTRINE_APPROVED"
    DOCTRINE_REJECTED = "DOCTRINE_REJECTED"
    IDENTITY_REVISED = "IDENTITY_REVISED"
    ETHICAL_TENSION_LOGGED = "ETHICAL_TENSION_LOGGED"
    MEMORY_SUMMARY = "MEMORY_SUMMARY"

    # Artifact tracking
    ARTIFACT_DIFF = "ARTIFACT_DIFF"

    # Misc
    NOTABLE_EVENT = "NOTABLE_EVENT"
    INTERPRETATION = "INTERPRETATION"


class EventEnvelope(BaseModel):
    """Common wrapper for every logged event."""

    event_type: EventType
    run_id: str
    condition: str
    cycle_id: int
    agent_id: Optional[str] = None  # None for system-level events
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any] = Field(default_factory=dict)

    def model_dump_jsonl(self) -> str:
        """Serialize to a single JSON line (for JSONL writing)."""
        return self.model_dump_json()


# ---------------------------------------------------------------------------
# Event routing: which event types go to which log files
# ---------------------------------------------------------------------------

EVENT_FILE_ROUTING: dict[EventType, str] = {
    # transcripts.jsonl
    EventType.REFLECTION_COMPLETE: "transcripts.jsonl",
    EventType.DISCUSSION_TURN: "transcripts.jsonl",
    EventType.INTERPRETATION: "transcripts.jsonl",
    # retrieval.jsonl
    EventType.RETRIEVAL_QUERY: "retrieval.jsonl",
    EventType.RETRIEVAL_RESULT: "retrieval.jsonl",
    # doctrine_diffs.jsonl
    EventType.DOCTRINE_PROPOSED: "doctrine_diffs.jsonl",
    EventType.DOCTRINE_APPROVED: "doctrine_diffs.jsonl",
    EventType.DOCTRINE_REJECTED: "doctrine_diffs.jsonl",
    # memory_diffs.jsonl
    EventType.MEMORY_SUMMARY: "memory_diffs.jsonl",
    # protocol_diffs.jsonl
    EventType.PROTOCOL_PROPOSED: "protocol_diffs.jsonl",
    EventType.ARTIFACT_DIFF: "protocol_diffs.jsonl",
    # evaluations.jsonl
    EventType.EVALUATION_SCORE: "evaluations.jsonl",
    # scenario_events.jsonl
    EventType.SCENARIO_INJECTED: "scenario_events.jsonl",
    # notable_events.jsonl — everything else
    EventType.RUN_START: "notable_events.jsonl",
    EventType.RUN_END: "notable_events.jsonl",
    EventType.CYCLE_START: "notable_events.jsonl",
    EventType.CYCLE_END: "notable_events.jsonl",
    EventType.PHASE_START: "notable_events.jsonl",
    EventType.PHASE_END: "notable_events.jsonl",
    EventType.IDENTITY_REVISED: "notable_events.jsonl",
    EventType.ETHICAL_TENSION_LOGGED: "notable_events.jsonl",
    EventType.NOTABLE_EVENT: "notable_events.jsonl",
}

# All log files that should be created per run
LOG_FILES = sorted(set(EVENT_FILE_ROUTING.values()))
