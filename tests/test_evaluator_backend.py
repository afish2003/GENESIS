"""The judge is built by its own factory path, which drifted from the agents'."""

class TestEvaluatorBackendInheritsRunSettings:
    """The judge must be built with the run's settings, not the defaults.

    create_evaluator_backend passed base_url, model, key, timeout and
    json_mode, and silently dropped max_output_tokens and enable_thinking.
    They happened to match the constructor defaults, so nothing failed — until
    a run raised the cap for the agents and the judge kept truncating its JSON
    at the old one with no error that said so.
    """

    def _cfg(self, **kw):
        from controller.config import RunConfig
        base = dict(run_id="R", condition="BASELINE", inference_backend="openai",
                    api_base_url="http://x/v1", model_name="agent-model",
                    evaluator_model="judge-model", max_output_tokens=9001,
                    enable_thinking=True)
        base.update(kw)
        return RunConfig(**base)

    def test_token_cap_and_thinking_flag_reach_the_judge(self):
        from controller.inference.factory import create_evaluator_backend
        b = create_evaluator_backend(self._cfg())
        assert b is not None
        assert b.max_output_tokens == 9001
        assert b.enable_thinking is True

    def test_judge_and_agents_agree_on_both(self):
        from controller.inference.factory import create_backend, create_evaluator_backend
        cfg = self._cfg(max_output_tokens=2048, enable_thinking=False)
        agents = create_backend(cfg)
        judge = create_evaluator_backend(cfg)
        assert judge.max_output_tokens == agents.max_output_tokens
        assert judge.enable_thinking == agents.enable_thinking
        assert judge.model != agents.model  # still an independent judge


class TestEveryBackendTheFactoryCanBuildActuallyBuilds:
    """The default backend crashed at prepare_run and no test noticed.

    create_backend passes enable_thinking to every branch; OllamaBackend had no
    such parameter, so `inference_backend: ollama` — the DEFAULT — raised
    TypeError before cycle 0. The factory was only ever tested on the openai
    and mock paths.
    """

    import pytest

    @pytest.mark.parametrize("backend,host", [
        ("ollama", {"ollama_host": "http://h:11434"}),
        ("openai", {"api_base_url": "http://h/v1"}),
        ("mock", {}),
    ])
    def test_factory_constructs_it(self, backend, host):
        from controller.config import RunConfig
        from controller.inference.factory import create_backend
        cfg = RunConfig(run_id="R", condition="BASELINE",
                        inference_backend=backend, max_output_tokens=1234, **host)
        b = create_backend(cfg)  # the bug was here: TypeError, no assert needed
        if backend != "mock":  # mock generates nothing, so it has no cap
            assert b.max_output_tokens == 1234, f"{backend} dropped the token cap"


class TestOllamaSendsItsTokenCap:
    """Storing the cap is not sending it."""

    def test_num_predict_is_in_the_payload(self):
        import asyncio, json
        from controller.inference.ollama_backend import OllamaBackend
        from controller.inference.backend import Message

        sent = {}

        class _Resp:
            def json(self):
                return {"message": {"content": "hi"}}

        b = OllamaBackend(model="m", max_output_tokens=777, enable_thinking=False)

        async def fake(payload):
            sent.update(payload)
            return _Resp()

        b._request_with_retry = fake
        asyncio.run(b.complete([Message(role="user", content="x")]))
        assert sent["options"]["num_predict"] == 777
        assert sent["think"] is False
