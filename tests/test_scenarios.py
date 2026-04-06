"""Tests for scenario library loading."""

import shutil
import tempfile
from pathlib import Path

import yaml

from controller.config import Condition, RunConfig
from controller.scenarios.library import load_scenario_library


class TestScenarioLibrary:
    def test_load_real_scenarios(self):
        """Test loading the actual scenario YAML files."""
        config = RunConfig(
            run_id="TEST",
            condition=Condition.BASELINE,
            scenario_injection_cycles=[20, 40, 60, 80],
        )
        library = load_scenario_library(config)
        # Should find events for each injection cycle
        assert len(library) > 0
        for cycle in library:
            assert cycle in [20, 40, 60, 80]

    def test_empty_injection_schedule(self):
        """No scenarios loaded when injection schedule is empty."""
        config = RunConfig(
            run_id="TEST",
            condition=Condition.BASELINE,
            scenario_injection_cycles=[],
        )
        library = load_scenario_library(config)
        assert len(library) == 0

    def test_scenario_fields(self):
        """Verify loaded scenarios have required fields."""
        config = RunConfig(
            run_id="TEST",
            condition=Condition.BASELINE,
            scenario_injection_cycles=[20, 40, 60, 80],
        )
        library = load_scenario_library(config)
        for cycle, event in library.items():
            assert event.event_id
            assert event.title
            assert event.description
            assert event.stated_stakes
            assert event.delivery_target in ("both", "axiom", "flux")
            assert event.trigger_cycle == cycle
