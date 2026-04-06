"""Retrieval logging utilities."""

from __future__ import annotations

from controller.logging.schemas import EventEnvelope, EventType
from controller.phases.schemas import RetrievalResultItem


def build_retrieval_result_event(
    run_id: str,
    condition: str,
    cycle_id: int,
    agent_id: str,
    query: str,
    results: list[RetrievalResultItem],
) -> EventEnvelope:
    """Build a RETRIEVAL_RESULT event from query results."""
    return EventEnvelope(
        event_type=EventType.RETRIEVAL_RESULT,
        run_id=run_id,
        condition=condition,
        cycle_id=cycle_id,
        agent_id=agent_id,
        payload={
            "query": query,
            "results": [r.model_dump() for r in results],
            "result_count": len(results),
        },
    )
