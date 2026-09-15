"""Phase 6: Retrieval — each agent issues up to 3 retrieval queries."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from controller.inference.backend import Message
from controller.logging.schemas import EventEnvelope, EventType
from controller.phases.schemas import RetrievalQueryOutput
from controller.agents.base import load_system_prompt
from controller.retrieval.logger import build_retrieval_result_event

if TYPE_CHECKING:
    from controller.agents.base import AgentContext
    from controller.config import RunConfig
    from controller.cycle import CycleState
    from controller.inference.backend import InferenceBackend
    from controller.logging.logger import AppendOnlyJSONLLogger
    from controller.world.state import WorldState

_logger = logging.getLogger(__name__)

#: Cap on retrieved text entering a prompt, so a large corpus cannot inflate
#: every subsequent call in the cycle.
_MAX_RETRIEVED_CHARS = 4000

RETRIEVAL_PROMPT = """Based on the current cycle's discussion and your priorities, you may now query the knowledge bases.

Available knowledge bases:
- General knowledge (philosophy, cognitive science, systems theory, decision theory)
- Technical (CS, systems design, protocol design, evaluation methodology)
- Governance and ethics (AI ethics frameworks, governance documents, alignment research)
- Self-history (your own past memory summaries, doctrine snapshots, and protocol versions)

Formulate up to 3 specific, targeted queries. Each query should address a specific information need relevant to this cycle's work. If you do not need external knowledge this cycle, provide an empty query list."""


async def execute(
    config: RunConfig,
    backend: InferenceBackend,
    world: WorldState,
    cycle: CycleState,
    contexts: dict[str, AgentContext],
    logger: AppendOnlyJSONLLogger,
) -> list[EventEnvelope]:
    """Gather retrieval queries from both agents and execute them."""
    events = cycle.pending_events

    for agent_id in config.agents:
        ctx = contexts[agent_id]
        messages = [
            ctx.build_system_message(),
            Message(role="user", content=RETRIEVAL_PROMPT),
        ]

        output = await backend.complete_structured(
            messages=messages,
            response_schema=RetrievalQueryOutput,
            temperature=config.temperature_structured,
            max_retries=config.max_retries,
        )
        output.agent_id = agent_id

        # Truncate to max queries
        queries = output.queries[:config.max_retrieval_queries]

        agent_results = []
        for query in queries:
            events.append(EventEnvelope(
                event_type=EventType.RETRIEVAL_QUERY,
                run_id=config.run_id,
                condition=config.condition.value,
                cycle_id=cycle.cycle_id,
                agent_id=agent_id,
                payload={"query": query},
            ))

            # Execute query against knowledge bases
            if cycle.kb_manager is not None:
                results = cycle.kb_manager.query(query)
                agent_results.extend(results)

                events.append(build_retrieval_result_event(
                    run_id=config.run_id,
                    condition=config.condition.value,
                    cycle_id=cycle.cycle_id,
                    agent_id=agent_id,
                    query=query,
                    results=results,
                ))

        cycle.retrieval_results[agent_id] = agent_results

        # Feed it back to the agent. Without this the whole phase is write-only:
        # queries cost an inference call, hits are logged, and no agent ever
        # sees a retrieved document.
        if agent_results:
            ctx.retrieved_context = await _summarise(
                config, backend, agent_results,
            )
            _logger.info(
                "Cycle %d: %s retrieved %d documents (%d chars into context)",
                cycle.cycle_id, agent_id, len(agent_results),
                len(ctx.retrieved_context),
            )
        else:
            ctx.retrieved_context = ""
            events.append(EventEnvelope(
                event_type=EventType.NOTABLE_EVENT,
                run_id=config.run_id,
                condition=config.condition.value,
                cycle_id=cycle.cycle_id,
                agent_id=agent_id,
                payload={
                    "kind": "retrieval_empty",
                    "queries": len(queries),
                    "detail": ("No documents retrieved. An empty query list and a "
                               "query that matched nothing are different failures; "
                               "this records which occurred."),
                },
            ))

    return events


async def _summarise(config, backend, results) -> str:
    """Condense retrieved documents for the agent's context.

    Falls back to the raw excerpts if the summariser fails — losing the material
    entirely would silently reproduce the bug this exists to fix.
    """
    raw = "\n\n".join(
        f"[{r.doc_id} | {r.source_kb}] {r.text}" for r in results
    )
    try:
        summariser = load_system_prompt(
            config.run_prompts_dir, "retrieval_summarizer.md"
        )
    except FileNotFoundError:
        return raw[:_MAX_RETRIEVED_CHARS]

    try:
        result = await backend.complete(
            messages=[
                Message(role="system", content=summariser),
                Message(
                    role="user",
                    content=("Summarise the following retrieved documents for the "
                             f"agent, citing document ids.\n\n{raw}"),
                ),
            ],
            temperature=config.temperature_structured,
        )
        text = (result.content or "").strip()
        return text[:_MAX_RETRIEVED_CHARS] if text else raw[:_MAX_RETRIEVED_CHARS]
    except Exception as e:  # noqa: BLE001 - never lose the material over a summariser fault
        _logger.warning("Retrieval summariser failed (%s); using raw excerpts", e)
        return raw[:_MAX_RETRIEVED_CHARS]
