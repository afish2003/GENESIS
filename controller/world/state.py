"""World state management — load, save, and diff all persistent artifacts.

The WorldState class owns all artifact I/O. It reads the current doctrine,
identity files, memory journals, protocol documents, ethical log, and
relationship log at the start of each cycle and writes updated versions
at the end. Every write triggers a diff event.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from controller.logging.schemas import EventEnvelope, EventType
from controller.world.artifacts import (
    Checkpoint,
    DoctrineDocument,
    EthicalLogEntry,
    IdentityStatement,
    MemoryEntry,
    ProtocolDocument,
    RelationshipLogEntry,
)
from controller.world.paths import assert_within, safe_artifact_id

logger = logging.getLogger(__name__)


#: Version and last_modified_cycle live beside the documents rather than inside
#: them: the .md files are what the agents read, and a metadata header would
#: become part of the doctrine text. Without this the counters were incremented
#: every revision and reset to 1 on the next load, so version tracking did not
#: survive a resume and "how much did doctrine evolve" was unanswerable.
_METADATA_FILE = ".metadata.json"


class WorldNotLoadedError(RuntimeError):
    """save() was called on a world whose load() did not complete."""


class WorldState:
    """Manages all persistent world artifacts.

    Artifacts live in a working `world/` directory. At cycle start,
    load() reads everything. At cycle end, save() writes updates
    and produces diff events for the logger.
    """

    def __init__(self, world_dir: Path, agents: list[str] | None = None) -> None:
        """
        Args:
            world_dir: the working world directory.
            agents: the run's agent roster. Defaults to the v1 pair so existing
                callers and tests keep working; the controller passes
                config.agents so a run is never tied to two named agents.
        """
        self.world_dir = world_dir
        self.agents = list(agents) if agents else ["axiom", "flux"]

        # Doctrine documents
        self.doctrine: dict[str, DoctrineDocument] = {}

        # Identity statements
        self.identities: dict[str, IdentityStatement] = {}

        # Memory journals (agent_id -> list of entries)
        self.memory: dict[str, list[MemoryEntry]] = {a: [] for a in self.agents}

        # Protocol documents
        self.protocols: dict[str, ProtocolDocument] = {}

        # Logs
        self.ethical_log: list[EthicalLogEntry] = []
        self.relationship_log: list[RelationshipLogEntry] = []

        #: True once load() has completed without raising. save() refuses while
        #: this is False: a partially-loaded world written back to disk destroys
        #: whatever it failed to read. One corrupt protocol JSON used to be
        #: enough to blank ethical_tradeoff_log.jsonl, because _load_protocols
        #: raised before _load_ethical_log ran and the empty in-memory list was
        #: then saved over the real file.
        self.loaded: bool = False

    def load(self) -> None:
        """Load all artifacts from the world directory.

        `loaded` flips only if every loader succeeded; a raise leaves it False
        and save() will refuse.
        """
        self.loaded = False
        self._load_doctrine()
        self._load_identities()
        self._load_memory()
        self._load_protocols()
        self._load_ethical_log()
        self._load_relationship_log()
        self.loaded = True
        logger.info("World state loaded from %s", self.world_dir)

    def save(self, run_id: str, condition: str, cycle_id: int) -> list[EventEnvelope]:
        """Save all artifacts and return diff events for any changes.

        Raises WorldNotLoadedError if the last load() did not complete. Writing
        a partially-loaded world back is unrecoverable: the artifacts that
        failed to load are empty in memory and would overwrite the real files.
        """
        if not self.loaded:
            raise WorldNotLoadedError(
                f"Refusing to save cycle {cycle_id}: the world was never "
                f"successfully loaded, so saving would overwrite artifacts that "
                f"failed to read with empty in-memory state."
            )
        events: list[EventEnvelope] = []
        events.extend(self._save_doctrine(run_id, condition, cycle_id))
        events.extend(self._save_identities(run_id, condition, cycle_id))
        events.extend(self._save_memory(run_id, condition, cycle_id))
        events.extend(self._save_protocols(run_id, condition, cycle_id))
        self._save_ethical_log()
        self._save_relationship_log()
        self._write_metadata()
        logger.info("World state saved to %s (%d diff events)", self.world_dir, len(events))
        return events

    def compute_hash(self) -> str:
        """A hash of the complete world state, for checkpoint verification.

        Every field that a cycle can change must be covered, or resume will
        accept a materially different world. The previous version hashed only
        doctrine/identity content, memory summaries and protocol bodies —
        omitting both logs, all version and last_modified_cycle fields,
        evaluation_history and archived flags. A cycle that only logged an
        ethical tension produced an identical hash to the one before it.

        Fields are separated by a NUL byte; concatenating them directly made
        ("ab","c") and ("a","bc") collide.
        """
        hasher = hashlib.sha256()

        def feed(*parts: object) -> None:
            for part in parts:
                hasher.update(str(part).encode("utf-8"))
                hasher.update(b"\x00")

        for name in sorted(self.doctrine):
            doc = self.doctrine[name]
            feed("doctrine", name, doc.content, doc.version, doc.last_modified_cycle)

        for agent_id in sorted(self.identities):
            ident = self.identities[agent_id]
            feed("identity", agent_id, ident.content, ident.version,
                 ident.last_modified_cycle)

        for agent_id in sorted(self.memory):
            feed("memory", agent_id)
            for entry in self.memory[agent_id]:
                feed(entry.cycle_id, entry.summary, entry.key_events,
                     entry.relationship_note, entry.doctrine_changes)

        for pid in sorted(self.protocols):
            proto = self.protocols[pid]
            feed("protocol", pid, proto.title, proto.content, proto.version,
                 proto.last_modified_cycle, proto.archived,
                 len(proto.evaluation_history))

        for entry in self.ethical_log:
            feed("ethical", entry.cycle_id, entry.agent_id, entry.description,
                 entry.severity, entry.resolution)

        for entry in self.relationship_log:
            feed("relationship", entry.cycle_id, entry.agent_id, entry.note)

        return hasher.hexdigest()[:16]

    # ------------------------------------------------------------------
    # Doctrine
    # ------------------------------------------------------------------

    def _read_metadata(self) -> dict:
        path = self.world_dir / "doctrine" / _METADATA_FILE
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Doctrine metadata at %s is unreadable; version "
                           "counters restart from 1", path)
            return {}

    def _write_metadata(self) -> None:
        doctrine_dir = self.world_dir / "doctrine"
        doctrine_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            name: {"version": doc.version,
                   "last_modified_cycle": doc.last_modified_cycle}
            for name, doc in self.doctrine.items()
        }
        meta.update({
            f"identity_{agent_id}.md": {
                "version": ident.version,
                "last_modified_cycle": ident.last_modified_cycle,
            }
            for agent_id, ident in self.identities.items()
        })
        (doctrine_dir / _METADATA_FILE).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def _load_doctrine(self) -> None:
        doctrine_dir = self.world_dir / "doctrine"
        if not doctrine_dir.exists():
            return
        meta = self._read_metadata()
        # Rebuild rather than update: a document deleted from the world
        # directory used to survive in memory and be rewritten on the next save.
        self.doctrine = {}
        for filepath in sorted(doctrine_dir.iterdir()):
            if filepath.suffix == ".md" and not filepath.name.startswith("identity_"):
                entry = meta.get(filepath.name, {})
                self.doctrine[filepath.name] = DoctrineDocument(
                    filename=filepath.name,
                    content=filepath.read_text(encoding="utf-8"),
                    version=entry.get("version", 1),
                    last_modified_cycle=entry.get("last_modified_cycle", 0),
                )

    def _save_doctrine(self, run_id: str, condition: str, cycle_id: int) -> list[EventEnvelope]:
        events = []
        doctrine_dir = self.world_dir / "doctrine"
        doctrine_dir.mkdir(parents=True, exist_ok=True)

        for name, doc in self.doctrine.items():
            filepath = assert_within(self.world_dir, doctrine_dir / name)
            old_content = filepath.read_text(encoding="utf-8") if filepath.exists() else ""
            if doc.content != old_content:
                filepath.write_text(doc.content, encoding="utf-8")
                diff = _compute_diff(old_content, doc.content, name)
                events.append(EventEnvelope(
                    event_type=EventType.ARTIFACT_DIFF,
                    run_id=run_id,
                    condition=condition,
                    cycle_id=cycle_id,
                    payload={"artifact": name, "type": "doctrine", "diff": diff},
                ))
        return events

    # ------------------------------------------------------------------
    # Identities
    # ------------------------------------------------------------------

    def _load_identities(self) -> None:
        doctrine_dir = self.world_dir / "doctrine"
        if not doctrine_dir.exists():
            return
        meta = self._read_metadata()
        self.identities = {}
        for agent_id in self.agents:
            filepath = doctrine_dir / f"identity_{agent_id}.md"
            if filepath.exists():
                entry = meta.get(filepath.name, {})
                self.identities[agent_id] = IdentityStatement(
                    agent_id=agent_id,
                    content=filepath.read_text(encoding="utf-8"),
                    version=entry.get("version", 1),
                    last_modified_cycle=entry.get("last_modified_cycle", 0),
                )

    def _save_identities(self, run_id: str, condition: str, cycle_id: int) -> list[EventEnvelope]:
        events = []
        doctrine_dir = self.world_dir / "doctrine"
        doctrine_dir.mkdir(parents=True, exist_ok=True)

        for agent_id, identity in self.identities.items():
            filepath = doctrine_dir / f"identity_{agent_id}.md"
            old_content = filepath.read_text(encoding="utf-8") if filepath.exists() else ""
            if identity.content != old_content:
                filepath.write_text(identity.content, encoding="utf-8")
                diff = _compute_diff(old_content, identity.content, f"identity_{agent_id}.md")
                events.append(EventEnvelope(
                    event_type=EventType.ARTIFACT_DIFF,
                    run_id=run_id,
                    condition=condition,
                    cycle_id=cycle_id,
                    agent_id=agent_id,
                    payload={"artifact": f"identity_{agent_id}.md", "type": "identity", "diff": diff},
                ))
        return events

    # ------------------------------------------------------------------
    # Memory journals
    # ------------------------------------------------------------------

    def _load_memory(self) -> None:
        memory_dir = self.world_dir / "memory"
        if not memory_dir.exists():
            return
        for agent_id in self.agents:
            filepath = memory_dir / f"memory_{agent_id}.jsonl"
            if filepath.exists():
                entries = []
                for line in filepath.read_text(encoding="utf-8").strip().split("\n"):
                    if line.strip():
                        entries.append(MemoryEntry.model_validate_json(line))
                self.memory[agent_id] = entries

    def _save_memory(self, run_id: str, condition: str, cycle_id: int) -> list[EventEnvelope]:
        events = []
        memory_dir = self.world_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        for agent_id in self.agents:
            filepath = memory_dir / f"memory_{agent_id}.jsonl"
            content = "\n".join(
                entry.model_dump_json() for entry in self.memory[agent_id]
            )
            if content:
                content += "\n"
            filepath.write_text(content, encoding="utf-8")
        return events

    def reset_memory(self, agent_id: str, cycle: int, bootstrap_template: str) -> None:
        """Reset an agent's memory for MEM_RESET condition."""
        self.memory[agent_id] = [
            MemoryEntry(
                cycle_id=cycle,
                summary=bootstrap_template.format(n=cycle),
                key_events=["Memory reset performed"],
                relationship_note="",
            )
        ]

    # ------------------------------------------------------------------
    # Protocol documents
    # ------------------------------------------------------------------

    def _load_protocols(self) -> None:
        protocols_dir = self.world_dir / "sandbox" / "protocols"
        if not protocols_dir.exists():
            return

        for filepath in protocols_dir.iterdir():
            if filepath.suffix == ".json" and filepath.name != "archive":
                proto = ProtocolDocument.model_validate_json(
                    filepath.read_text(encoding="utf-8")
                )
                self.protocols[proto.protocol_id] = proto

    def _save_protocols(self, run_id: str, condition: str, cycle_id: int) -> list[EventEnvelope]:
        events = []
        protocols_dir = self.world_dir / "sandbox" / "protocols"
        protocols_dir.mkdir(parents=True, exist_ok=True)
        archive_dir = protocols_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        for pid, proto in self.protocols.items():
            # pid originates from the model. It is sanitised at ingress in
            # protocol_design.py; re-sanitise here so a protocol loaded from a
            # hand-edited world directory cannot escape either.
            safe_pid = safe_artifact_id(pid)
            filepath = assert_within(self.world_dir, protocols_dir / f"{safe_pid}.json")

            old_content = ""
            if filepath.exists():
                old_content = filepath.read_text(encoding="utf-8")

            new_content = proto.model_dump_json(indent=2)
            if new_content != old_content:
                # Archive old version if it existed
                if old_content:
                    archive_path = assert_within(
                        self.world_dir,
                        archive_dir / f"{safe_pid}_v{proto.version - 1}_cycle{cycle_id}.json",
                    )
                    archive_path.write_text(old_content, encoding="utf-8")

                filepath.write_text(new_content, encoding="utf-8")
                events.append(EventEnvelope(
                    event_type=EventType.ARTIFACT_DIFF,
                    run_id=run_id,
                    condition=condition,
                    cycle_id=cycle_id,
                    payload={"artifact": pid, "type": "protocol", "version": proto.version},
                ))

        return events

    # ------------------------------------------------------------------
    # Ethical tradeoff log
    # ------------------------------------------------------------------

    def _load_ethical_log(self) -> None:
        filepath = self.world_dir / "logs" / "ethical_tradeoff_log.jsonl"
        if not filepath.exists():
            return
        self.ethical_log = []
        for line in filepath.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                self.ethical_log.append(EthicalLogEntry.model_validate_json(line))

    def _save_ethical_log(self) -> None:
        logs_dir = self.world_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        filepath = logs_dir / "ethical_tradeoff_log.jsonl"
        content = "\n".join(entry.model_dump_json() for entry in self.ethical_log)
        if content:
            content += "\n"
        filepath.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Relationship log
    # ------------------------------------------------------------------

    def _load_relationship_log(self) -> None:
        filepath = self.world_dir / "logs" / "relationship_log.jsonl"
        if not filepath.exists():
            return
        self.relationship_log = []
        for line in filepath.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                self.relationship_log.append(RelationshipLogEntry.model_validate_json(line))

    def _save_relationship_log(self) -> None:
        logs_dir = self.world_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        filepath = logs_dir / "relationship_log.jsonl"
        content = "\n".join(entry.model_dump_json() for entry in self.relationship_log)
        if content:
            content += "\n"
        filepath.write_text(content, encoding="utf-8")


def _compute_diff(old: str, new: str, label: str) -> str:
    """Compute a unified diff between old and new text."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{label}", tofile=f"b/{label}")
    return "".join(diff)
