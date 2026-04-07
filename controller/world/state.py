"""WorldState — load, save, and diff all persistent world artifacts.

Directory layout expected under world_dir/:
    doctrine/           Markdown files (manifesto.md, constitution.md, etc.)
    memory/             JSONL journals (memory_axiom.jsonl, memory_flux.jsonl)
    sandbox/protocols/  Protocol Markdown files (*.md) + archive/ subdir
    logs/               JSONL log files (ethical_tradeoff_log.jsonl,
                        relationship_log.jsonl)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from controller.logging.schemas import EventEnvelope, EventType
from controller.world.artifacts import (
    DoctrineDocument,
    EthicalLogEntry,
    IdentityStatement,
    MemoryEntry,
    ProtocolDocument,
    RelationshipLogEntry,
)

logger = logging.getLogger(__name__)


class WorldState:
    """Owns the in-memory representation of all persistent world artifacts.

    Usage::

        world = WorldState(world_dir)
        world.load()           # read everything from disk
        # ... phases mutate world.doctrine, world.identities, etc.
        events = world.save(run_id, condition, cycle_id)  # write + diff events
    """

    def __init__(self, world_dir: Path) -> None:
        self.world_dir = Path(world_dir)

        # Doctrine documents: filename -> DoctrineDocument
        self.doctrine: dict[str, DoctrineDocument] = {}

        # Identity statements: agent_id ("axiom"/"flux") -> IdentityStatement
        self.identities: dict[str, IdentityStatement] = {}

        # Memory journals: agent_id -> list[MemoryEntry] (ordered oldest-first)
        self.memory: dict[str, list[MemoryEntry]] = {}

        # Protocol documents: protocol_id -> ProtocolDocument
        self.protocols: dict[str, ProtocolDocument] = {}

        # Ethical tension log (append-only in-memory, flushed on save)
        self.ethical_log: list[EthicalLogEntry] = []

        # Relationship log (append-only in-memory, flushed on save)
        self.relationship_log: list[RelationshipLogEntry] = []

        # Snapshot of content hashes taken at last load (used for diff events)
        self._snapshots: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Read all artifacts from disk into memory."""
        self._load_doctrine()
        self._load_identities()
        self._load_memory()
        self._load_protocols()
        self._load_ethical_log()
        self._load_relationship_log()
        self._take_snapshot()

    def _load_doctrine(self) -> None:
        doctrine_dir = self.world_dir / "doctrine"
        self.doctrine = {}
        if not doctrine_dir.exists():
            return
        for path in sorted(doctrine_dir.glob("*.md")):
            if path.stem.startswith("identity_"):
                continue  # handled separately
            content = path.read_text(encoding="utf-8")
            self.doctrine[path.name] = DoctrineDocument(
                filename=path.name,
                content=content,
            )

    def _load_identities(self) -> None:
        doctrine_dir = self.world_dir / "doctrine"
        self.identities = {}
        if not doctrine_dir.exists():
            return
        for agent_id in ["axiom", "flux"]:
            path = doctrine_dir / f"identity_{agent_id}.md"
            if path.exists():
                content = path.read_text(encoding="utf-8")
                self.identities[agent_id] = IdentityStatement(
                    agent_id=agent_id,
                    content=content,
                )

    def _load_memory(self) -> None:
        memory_dir = self.world_dir / "memory"
        self.memory = {"axiom": [], "flux": []}
        if not memory_dir.exists():
            return
        for agent_id in ["axiom", "flux"]:
            path = memory_dir / f"memory_{agent_id}.jsonl"
            if not path.exists():
                continue
            entries: list[MemoryEntry] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(MemoryEntry.model_validate_json(line))
                except Exception as exc:
                    logger.warning("Skipping malformed memory entry in %s: %s", path, exc)
            self.memory[agent_id] = entries

    def _load_protocols(self) -> None:
        protocols_dir = self.world_dir / "sandbox" / "protocols"
        self.protocols = {}
        if not protocols_dir.exists():
            return
        for path in sorted(protocols_dir.glob("*.md")):
            if path.parent.name == "archive":
                continue
            try:
                content = path.read_text(encoding="utf-8")
                # Protocol ID is the stem; title comes from first H1 line
                protocol_id = path.stem
                title = _extract_title(content) or protocol_id
                self.protocols[protocol_id] = ProtocolDocument(
                    protocol_id=protocol_id,
                    title=title,
                    content=content,
                )
            except Exception as exc:
                logger.warning("Skipping protocol %s: %s", path, exc)

    def _load_ethical_log(self) -> None:
        path = self.world_dir / "logs" / "ethical_tradeoff_log.jsonl"
        self.ethical_log = []
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self.ethical_log.append(EthicalLogEntry.model_validate_json(line))
            except Exception as exc:
                logger.warning("Skipping malformed ethical log entry: %s", exc)

    def _load_relationship_log(self) -> None:
        path = self.world_dir / "logs" / "relationship_log.jsonl"
        self.relationship_log = []
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self.relationship_log.append(RelationshipLogEntry.model_validate_json(line))
            except Exception as exc:
                logger.warning("Skipping malformed relationship log entry: %s", exc)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        run_id: str,
        condition: str,
        cycle_id: int,
    ) -> list[EventEnvelope]:
        """Write all artifacts to disk and return ARTIFACT_DIFF events.

        Only writes files whose content has changed since last load/save.
        """
        events: list[EventEnvelope] = []

        events.extend(self._save_doctrine(run_id, condition, cycle_id))
        events.extend(self._save_identities(run_id, condition, cycle_id))
        self._save_memory()
        events.extend(self._save_protocols(run_id, condition, cycle_id))
        self._save_ethical_log()
        self._save_relationship_log()

        self._take_snapshot()
        return events

    def _save_doctrine(
        self, run_id: str, condition: str, cycle_id: int
    ) -> list[EventEnvelope]:
        events = []
        doctrine_dir = self.world_dir / "doctrine"
        doctrine_dir.mkdir(parents=True, exist_ok=True)

        for filename, doc in self.doctrine.items():
            path = doctrine_dir / filename
            new_hash = _hash_text(doc.content)
            old_hash = self._snapshots.get(f"doctrine:{filename}", "")
            if new_hash != old_hash:
                path.write_text(doc.content, encoding="utf-8")
                events.append(_diff_event(
                    run_id, condition, cycle_id,
                    artifact_type="doctrine",
                    artifact_id=filename,
                    version=doc.version,
                ))
        return events

    def _save_identities(
        self, run_id: str, condition: str, cycle_id: int
    ) -> list[EventEnvelope]:
        events = []
        doctrine_dir = self.world_dir / "doctrine"
        doctrine_dir.mkdir(parents=True, exist_ok=True)

        for agent_id, identity in self.identities.items():
            path = doctrine_dir / f"identity_{agent_id}.md"
            new_hash = _hash_text(identity.content)
            old_hash = self._snapshots.get(f"identity:{agent_id}", "")
            if new_hash != old_hash:
                path.write_text(identity.content, encoding="utf-8")
                events.append(_diff_event(
                    run_id, condition, cycle_id,
                    artifact_type="identity",
                    artifact_id=agent_id,
                    version=identity.version,
                ))
        return events

    def _save_memory(self) -> None:
        memory_dir = self.world_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        for agent_id, entries in self.memory.items():
            path = memory_dir / f"memory_{agent_id}.jsonl"
            lines = [entry.model_dump_json() for entry in entries]
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _save_protocols(
        self, run_id: str, condition: str, cycle_id: int
    ) -> list[EventEnvelope]:
        events = []
        protocols_dir = self.world_dir / "sandbox" / "protocols"
        protocols_dir.mkdir(parents=True, exist_ok=True)

        for protocol_id, proto in self.protocols.items():
            path = protocols_dir / f"{protocol_id}.md"
            new_hash = _hash_text(proto.content)
            old_hash = self._snapshots.get(f"protocol:{protocol_id}", "")
            if new_hash != old_hash:
                path.write_text(proto.content, encoding="utf-8")
                events.append(_diff_event(
                    run_id, condition, cycle_id,
                    artifact_type="protocol",
                    artifact_id=protocol_id,
                    version=proto.version,
                ))
        return events

    def _save_ethical_log(self) -> None:
        path = self.world_dir / "logs" / "ethical_tradeoff_log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [entry.model_dump_json() for entry in self.ethical_log]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _save_relationship_log(self) -> None:
        path = self.world_dir / "logs" / "relationship_log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [entry.model_dump_json() for entry in self.relationship_log]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def compute_hash(self) -> str:
        """Return a short (16-char hex) hash of the complete world state.

        Used for checkpointing — a change in hash means the world changed.
        """
        parts: list[str] = []
        for name in sorted(self.doctrine):
            parts.append(self.doctrine[name].content)
        for agent_id in sorted(self.identities):
            parts.append(self.identities[agent_id].content)
        for agent_id in sorted(self.memory):
            for entry in self.memory[agent_id]:
                parts.append(entry.summary)
        combined = "\n".join(parts)
        return hashlib.md5(combined.encode("utf-8")).hexdigest()[:16]

    def reset_memory(
        self,
        agent_id: str,
        cycle_id: int,
        bootstrap_template: str,
    ) -> None:
        """Clear an agent's memory and replace with a single bootstrap entry.

        The bootstrap_template may contain ``{n}`` which is replaced with
        ``cycle_id``.
        """
        summary = bootstrap_template.replace("{n}", str(cycle_id))
        self.memory[agent_id] = [
            MemoryEntry(cycle_id=cycle_id, summary=summary)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _take_snapshot(self) -> None:
        """Record content hashes of all current artifacts for diff detection."""
        self._snapshots = {}
        for filename, doc in self.doctrine.items():
            self._snapshots[f"doctrine:{filename}"] = _hash_text(doc.content)
        for agent_id, identity in self.identities.items():
            self._snapshots[f"identity:{agent_id}"] = _hash_text(identity.content)
        for protocol_id, proto in self.protocols.items():
            self._snapshots[f"protocol:{protocol_id}"] = _hash_text(proto.content)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _extract_title(content: str) -> Optional[str]:
    """Return the text of the first H1 line, or None."""
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _diff_event(
    run_id: str,
    condition: str,
    cycle_id: int,
    artifact_type: str,
    artifact_id: str,
    version: int,
) -> EventEnvelope:
    return EventEnvelope(
        event_type=EventType.ARTIFACT_DIFF,
        run_id=run_id,
        condition=condition,
        cycle_id=cycle_id,
        payload={
            "artifact_type": artifact_type,
            "artifact_id": artifact_id,
            "version": version,
        },
    )
