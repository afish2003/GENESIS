"""Who is speaking has to reach the live feed, or watching them argue is a
wall of anonymous text.

The feed carried phase and cycle_id and no agent_id, so during a two-agent
discussion — the thing most worth watching — you could not tell Axiom from
Flux mid-generation. The structured logs were attributed the whole time; only
the token stream was not.
"""

import ast
import inspect
from pathlib import Path

import pytest

from controller.inference.backend import InferenceBackend, Message
from controller.inference.mock_backend import MockBackend


class TestTheSinkReceivesASpeaker:
    def test_write_delta_records_it(self, tmp_path):
        from controller.cycle import CycleOrchestrator

        captured = []
        # Exercise the sink directly: it is one small method and standing up a
        # whole orchestrator to reach it would test the harness, not the sink.
        class _Stub:
            _live_path = tmp_path / "live.jsonl"
            _live_phase = "discussion"
            _live_cycle = 3
            _write_delta = CycleOrchestrator._write_delta

        _Stub()._write_delta("hello", "flux")
        import json
        rec = json.loads((tmp_path / "live.jsonl").read_text().strip())
        assert rec["agent_id"] == "flux"
        assert rec["delta"] == "hello"
        assert rec["phase"] == "discussion"

    def test_an_unattributed_delta_still_writes(self, tmp_path):
        """Nothing may break because a caller had no agent to name."""
        from controller.cycle import CycleOrchestrator
        import json

        class _Stub:
            _live_path = tmp_path / "live.jsonl"
            _live_phase = "retrieval"
            _live_cycle = 0
            _write_delta = CycleOrchestrator._write_delta

        _Stub()._write_delta("x")
        assert json.loads((tmp_path / "live.jsonl").read_text().strip())["agent_id"] is None


class TestEveryBackendAcceptsASpeaker:
    """A backend that does not is a TypeError at the first streamed token."""

    @pytest.mark.parametrize("name", ["openai_backend", "ollama_backend", "mock_backend"])
    def test_complete_takes_speaker(self, name):
        import importlib
        mod = importlib.import_module(f"controller.inference.{name}")
        cls = next(v for v in vars(mod).values()
                   if isinstance(v, type) and issubclass(v, InferenceBackend)
                   and v is not InferenceBackend)
        assert "speaker" in inspect.signature(cls.complete).parameters, name

    def test_the_abstract_interface_declares_it(self):
        for method in (InferenceBackend.complete, InferenceBackend.complete_structured):
            assert "speaker" in inspect.signature(method).parameters

    @pytest.mark.asyncio
    async def test_passing_a_speaker_does_not_disturb_generation(self):
        b = MockBackend(model="m")
        out = await b.complete([Message(role="user", content="hi")], speaker="axiom")
        assert out.content


class TestEveryAgentCallSiteNamesItsSpeaker:
    """The thing that actually went wrong was not the plumbing but that no
    caller supplied a value. A new phase that forgets is the same bug again."""

    PHASES = Path("controller/phases")

    #: Calls with genuinely no speaker. Empty on purpose — if one is added,
    #: it should be a deliberate entry here, not a silent omission.
    EXEMPT: set[tuple[str, int]] = set()

    def _model_calls(self, path: Path):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not isinstance(fn, ast.Attribute):
                continue
            if fn.attr not in ("complete", "complete_structured"):
                continue
            yield node

    def test_no_phase_calls_the_model_without_naming_a_speaker(self):
        missing = []
        for path in sorted(self.PHASES.glob("*.py")):
            for node in self._model_calls(path):
                if (path.name, node.lineno) in self.EXEMPT:
                    continue
                if not any(k.arg == "speaker" for k in node.keywords):
                    missing.append(f"{path.name}:{node.lineno}")
        assert not missing, (
            "these model calls will stream unattributed text into the live "
            f"feed: {missing}")

    def test_the_check_would_notice_a_missing_one(self):
        """A guard that cannot fail is the failure mode this repo keeps
        producing, so prove this one can."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad_phase.py"
            p.write_text("async def f(backend):\n"
                         "    return await backend.complete_structured(messages=[])\n")
            calls = list(self._model_calls(p))
            assert calls and not any(k.arg == "speaker" for k in calls[0].keywords)
