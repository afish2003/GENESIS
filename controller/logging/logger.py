"""Append-only JSONL logger for GENESIS research logs.

All events are written through this logger. Files are never modified
after being written. The logger routes events to the correct log file
based on the event type.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from controller.logging.schemas import (
    EVENT_FILE_ROUTING,
    LOG_FILES,
    EventEnvelope,
    EventType,
)

logger = logging.getLogger(__name__)


class AppendOnlyJSONLLogger:
    """Write-only JSONL logger that routes events to per-category files.

    Each run has its own directory under research_logs/. Log files are
    created on first use and only appended to — never read, modified,
    or truncated by this class.
    """

    def __init__(self, run_log_dir: Path) -> None:
        self._run_log_dir = run_log_dir
        self._file_handles: dict[str, object] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Create the run log directory and all log files."""
        self._run_log_dir.mkdir(parents=True, exist_ok=True)

        # Create annotations subdirectory
        (self._run_log_dir / "annotations").mkdir(exist_ok=True)

        # Create all log files (empty if they don't exist)
        for filename in LOG_FILES:
            filepath = self._run_log_dir / filename
            if not filepath.exists():
                filepath.touch()

        self._initialized = True
        logger.info("Initialized log directory: %s", self._run_log_dir)

    def log_event(self, event: EventEnvelope) -> None:
        """Write a single event to the appropriate log file.

        Events are serialized as a single JSON line and appended.
        The file is flushed after each write to ensure durability.
        """
        if not self._initialized:
            raise RuntimeError("Logger not initialized. Call initialize() first.")

        filename = EVENT_FILE_ROUTING.get(event.event_type)
        if filename is None:
            logger.warning("No routing for event type %s, writing to notable_events.jsonl", event.event_type)
            filename = "notable_events.jsonl"

        filepath = self._run_log_dir / filename
        line = event.model_dump_jsonl()

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def log_events(self, events: list[EventEnvelope]) -> None:
        """Write multiple events. Each goes to its routed file."""
        for event in events:
            self.log_event(event)

    def write_config(self, config_data: dict) -> None:
        """Write the run configuration as config.json (not JSONL)."""
        if not self._initialized:
            raise RuntimeError("Logger not initialized. Call initialize() first.")

        config_path = self._run_log_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, default=str)

    def copy_prompts(self, prompts_dir: Path) -> None:
        """Copy all prompt files into the run log directory for version-locking."""
        if not self._initialized:
            raise RuntimeError("Logger not initialized. Call initialize() first.")

        dest = self._run_log_dir / "prompts"
        dest.mkdir(exist_ok=True)

        if prompts_dir.exists():
            for prompt_file in prompts_dir.iterdir():
                if prompt_file.is_file():
                    target = dest / prompt_file.name
                    target.write_text(prompt_file.read_text(encoding="utf-8"), encoding="utf-8")
                    logger.info("Copied prompt: %s", prompt_file.name)

    @property
    def run_log_dir(self) -> Path:
        return self._run_log_dir
