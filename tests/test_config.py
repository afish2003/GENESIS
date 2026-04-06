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
