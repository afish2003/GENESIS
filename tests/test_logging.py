"""Tests for the append-only JSONL logger."""

import json
import shutil
import tempfile
from pathlib import Path

from controller.logging.logger import AppendOnlyJSONLLogger
from controller.logging.schemas import EVENT_FILE_ROUTING, LOG_FILES, EventEnvelope, EventType


class TestAppendOnlyJSONLLogger:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.run_dir = self.tmpdir / "RUN_TEST"

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_initialize(self):
        log = AppendOnlyJSONLLogger(self.run_dir)
        log.initialize()
        assert self.run_dir.exists()
        assert (self.run_dir / "annotations").exists()
        for filename in LOG_FILES:
            assert (self.run_dir / filename).exists()

    def test_log_event(self):
        log = AppendOnlyJSONLLogger(self.run_dir)
        log.initialize()

        event = EventEnvelope(
            event_type=EventType.CYCLE_START,
            run_id="RUN_TEST",
            condition="BASELINE",
            cycle_id=0,
        )
        log.log_event(event)

        filepath = self.run_dir / "notable_events.jsonl"
        lines = filepath.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["event_type"] == "CYCLE_START"

    def test_append_only(self):
        log = AppendOnlyJSONLLogger(self.run_dir)
        log.initialize()

        for i in range(3):
            event = EventEnvelope(
                event_type=EventType.DISCUSSION_TURN,
                run_id="RUN_TEST",
                condition="BASELINE",
                cycle_id=0,
                agent_id="axiom",
                payload={"turn": i},
            )
            log.log_event(event)

        filepath = self.run_dir / "transcripts.jsonl"
        lines = filepath.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_event_routing(self):
        log = AppendOnlyJSONLLogger(self.run_dir)
        log.initialize()

        # Log events of different types and verify they go to correct files
        test_cases = [
            (EventType.DISCUSSION_TURN, "transcripts.jsonl"),
            (EventType.RETRIEVAL_QUERY, "retrieval.jsonl"),
            (EventType.DOCTRINE_PROPOSED, "doctrine_diffs.jsonl"),
            (EventType.EVALUATION_SCORE, "evaluations.jsonl"),
            (EventType.SCENARIO_INJECTED, "scenario_events.jsonl"),
            (EventType.CYCLE_START, "notable_events.jsonl"),
        ]

        for event_type, expected_file in test_cases:
            event = EventEnvelope(
                event_type=event_type,
                run_id="RUN_TEST",
                condition="BASELINE",
                cycle_id=0,
            )
            log.log_event(event)

            filepath = self.run_dir / expected_file
            content = filepath.read_text().strip()
            assert content, f"No content in {expected_file} for {event_type}"

    def test_write_config(self):
        log = AppendOnlyJSONLLogger(self.run_dir)
        log.initialize()

        config_data = {"run_id": "RUN_TEST", "condition": "BASELINE"}
        log.write_config(config_data)

        config_path = self.run_dir / "config.json"
        assert config_path.exists()
        parsed = json.loads(config_path.read_text())
        assert parsed["run_id"] == "RUN_TEST"

    def test_uninitialized_raises(self):
        log = AppendOnlyJSONLLogger(self.run_dir)
        event = EventEnvelope(
            event_type=EventType.CYCLE_START,
            run_id="RUN_TEST",
            condition="BASELINE",
            cycle_id=0,
        )
        try:
            log.log_event(event)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass
