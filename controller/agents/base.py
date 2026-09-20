"""Agent context management.

Each agent maintains a context object that holds its system prompt,
identity, memory journal, and the current cycle's discussion history.
Context is rebuilt at the start of each cycle from persistent storage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from controller.inference.backend import Message
from controller.world.artifacts import IdentityStatement, MemoryEntry

logger = logging.getLogger(__name__)


class AgentContext:
    """Per-agent state container used throughout a cycle.

    Holds everything needed to construct prompts for this agent:
    system prompt, identity, memory, doctrine context, and the
    running discussion history for the current cycle.
    """

    def __init__(
        self,
        agent_id: str,
        system_prompt: str,
        identity: IdentityStatement,
        memory: list[MemoryEntry],
        doctrine_context: str,
    ) -> None:
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.identity = identity
        self.memory = memory
        self.doctrine_context = doctrine_context
        self.discussion_history: list[Message] = []
        self.cycle_events: list[str] = []  # Running log of notable events this cycle
        # Summarised retrieval for this cycle, set by the retrieval phase.
        # Until 2026-09-14 retrieval results were written to CycleState and read
        # by nobody: agents paid for queries every cycle, the hits were logged,
        # and nothing re-entered any prompt.
        self.retrieved_context: str = ""

    def build_system_message(self) -> Message:
        """Construct the full system message for this agent."""
        parts = [self.system_prompt]

        # Identity
        parts.append(f"\n---\n\n## Your Current Identity\n\n{self.identity.content}")

        # Doctrine context
        parts.append(f"\n---\n\n## Current Doctrine\n\n{self.doctrine_context}")

        # Anything this agent retrieved this cycle
        if self.retrieved_context:
            parts.append(
                "\n---\n\n## Retrieved This Cycle\n\n"
                "You requested the following material. Cite it by document id "
                "when you rely on it.\n\n"
                f"{self.retrieved_context}"
            )

        # Memory (recent entries, most recent last)
        if self.memory:
            memory_text = "\n---\n\n## Memory Journal (Recent)\n\n"
            # Include last 10 memory entries to manage context length
            recent = self.memory[-10:]
            for entry in recent:
                memory_text += f"### Cycle {entry.cycle_id}\n"
                memory_text += f"{entry.summary}\n"
                if entry.key_events:
                    memory_text += f"Key events: {', '.join(entry.key_events)}\n"
                if entry.relationship_note:
                    memory_text += f"Partnership note: {entry.relationship_note}\n"
                memory_text += "\n"
            parts.append(memory_text)

        return Message(role="system", content="\n".join(parts))

    def add_discussion_turn(self, role: str, content: str) -> None:
        """Add a turn to the current discussion history."""
        self.discussion_history.append(Message(role=role, content=content))

    def get_discussion_messages(self) -> list[Message]:
        """Return the full discussion history for prompt construction."""
        return list(self.discussion_history)

    def get_memory_summary(self) -> str:
        """Return a brief text summary of recent memory for inclusion in prompts."""
        if not self.memory:
            return "No prior memory available."
        recent = self.memory[-5:]
        lines = []
        for entry in recent:
            lines.append(f"Cycle {entry.cycle_id}: {entry.summary[:200]}")
        return "\n".join(lines)


#: Substituted into the generic agent template. Kept small and explicit: the
#: template is rendered for framing/identity_seed at materialise time, and only
#: these per-agent values are filled in here, per cycle.
GENERIC_AGENT_PROMPT = "agent_system.md"


def load_system_prompt(prompts_dir: Path, prompt_filename: str) -> str:
    """Load a system prompt from the prompts directory."""
    filepath = prompts_dir / prompt_filename
    if not filepath.exists():
        raise FileNotFoundError(f"System prompt not found: {filepath}")
    return filepath.read_text(encoding="utf-8")


def load_agent_system_prompt(prompts_dir: Path, agent_id: str, config) -> str:
    """This agent's system prompt: its own file, or the generic template.

    A hand-written `<agent_id>_system.md` still wins, so axiom and flux are
    untouched. Anything else falls back to `agent_system.md` with the roster
    substituted in.

    The fallback exists because the alternative was done once by hand and
    produced vertex_system.md: a search-and-replace of axiom_system.md reading
    "You work with partners named Axiom and your partners", describing Vertex
    with Flux's role, paired with an identity statement calling Vertex "the
    architect and stabilizer" — Axiom's role — who remembers "my interactions
    with Flux" and not Axiom. Every three-agent result in the project was
    collected against that.
    """
    specific = prompts_dir / f"{agent_id}_system.md"
    if specific.exists():
        return specific.read_text(encoding="utf-8")

    template = load_system_prompt(prompts_dir, GENERIC_AGENT_PROMPT)
    return fill_roster(template, agent_id, config)


def fill_roster(template: str, agent_id: str, config) -> str:
    """Substitute the roster-dependent fields of a generic template."""
    return (template
            .replace("{display_name}", config.display_name(agent_id))
            .replace("{partner_names}", config.partner_names(agent_id) or "no one yet")
            .replace("{disposition}", config.disposition(agent_id))
            .replace("{agent_id}", agent_id))


def build_agent_context(
    agent_id: str,
    prompts_dir: Path,
    identity: IdentityStatement,
    memory: list[MemoryEntry],
    doctrine_texts: dict[str, str],
    config=None,
) -> AgentContext:
    """Build a full agent context from persistent state.

    Called at the start of each cycle to reconstruct the agent's
    view of the world.
    """
    if config is not None:
        system_prompt = load_agent_system_prompt(prompts_dir, agent_id, config)
    else:
        system_prompt = load_system_prompt(prompts_dir, f"{agent_id}_system.md")

    # Build doctrine context from all doctrine documents
    doctrine_parts = []
    for name in sorted(doctrine_texts):
        doctrine_parts.append(f"### {name}\n\n{doctrine_texts[name]}")
    doctrine_context = "\n\n---\n\n".join(doctrine_parts)

    return AgentContext(
        agent_id=agent_id,
        system_prompt=system_prompt,
        identity=identity,
        memory=memory,
        doctrine_context=doctrine_context,
    )
