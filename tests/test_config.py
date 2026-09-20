"""Tests for RunConfig."""

from controller.config import Condition, RunConfig


class TestRunConfig:
    def test_basic_creation(self):
        config = RunConfig(run_id="RUN_001", condition=Condition.BASELINE)
        assert config.run_id == "RUN_001"
        assert config.condition == Condition.BASELINE
        assert config.total_cycles == 100

    def test_run_log_dir(self):
        config = RunConfig(run_id="RUN_001", condition=Condition.BASELINE)
        assert str(config.run_log_dir).endswith("research_logs/RUN_001")

    def test_memory_reset_condition(self):
        config = RunConfig(run_id="RUN_001", condition=Condition.MEM_RESET)
        assert config.is_memory_reset
        assert config.should_reset_memory(10)
        assert not config.should_reset_memory(5)
        assert config.should_reset_memory(20)
        assert not config.should_reset_memory(0)

    def test_baseline_no_memory_reset(self):
        config = RunConfig(run_id="RUN_001", condition=Condition.BASELINE)
        assert not config.is_memory_reset
        assert not config.should_reset_memory(10)

    def test_scenario_injection(self):
        config = RunConfig(
            run_id="RUN_001",
            condition=Condition.BASELINE,
            scenario_injection_cycles=[20, 40, 60, 80],
        )
        assert config.should_inject_scenario(20)
        assert config.should_inject_scenario(60)
        assert not config.should_inject_scenario(30)

    def test_discussion_turns(self):
        config = RunConfig(run_id="RUN_001", condition=Condition.BASELINE)
        assert config.discussion_turns(False) == 4
        assert config.discussion_turns(True) == 8


class TestYamlAndCliMustAgree:
    """A YAML value the command line overrides must not vanish quietly.

    `condition` is applied after `yaml_values` in the merge, so a YAML saying
    MEM_RESET and a driver passing --condition BASELINE produced a BASELINE
    run with no warning: no reset fires, `should_reset_memory()` returns False
    every cycle, and the logs are indistinguishable from an intended control.
    Three mislabelled runs is the whole experiment.
    """

    def _yaml(self, tmp_path, body: str) -> str:
        p = tmp_path / "c.yaml"
        p.write_text(body)
        return str(p)

    def test_conflicting_condition_raises(self, tmp_path):
        import pytest
        from controller.config import load_config
        f = self._yaml(tmp_path, "condition: MEM_RESET\n")
        with pytest.raises(ValueError, match="discarded without a word"):
            load_config(run_id="R", condition="BASELINE", cycles=3, config_file=f)

    def test_matching_condition_is_fine(self, tmp_path):
        from controller.config import load_config
        f = self._yaml(tmp_path, "condition: MEM_RESET\n")
        cfg = load_config(run_id="R", condition="MEM_RESET", cycles=3, config_file=f)
        assert cfg.condition.value == "MEM_RESET"

    def test_conflicting_cycle_count_raises(self, tmp_path):
        import pytest
        from controller.config import load_config
        f = self._yaml(tmp_path, "total_cycles: 40\n")
        with pytest.raises(ValueError, match="total_cycles"):
            load_config(run_id="R", condition="BASELINE", cycles=3, config_file=f)

    def test_the_real_study_yaml_pins_nothing_the_cli_sets(self):
        """memory_study.yaml must stay loadable for both arms."""
        from controller.config import load_config
        for cond in ("BASELINE", "MEM_RESET"):
            cfg = load_config(run_id=f"T_{cond}", condition=cond, cycles=4,
                              config_file="experiments/memory_study.yaml")
            assert cfg.condition.value == cond
            assert cfg.total_cycles == 4
