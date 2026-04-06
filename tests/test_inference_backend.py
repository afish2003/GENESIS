"""Tests for inference backend abstraction."""

import pytest

from controller.inference.backend import InferenceBackend, InferenceResult, Message
from controller.inference.vllm_backend import VLLMBackend


class TestMessage:
    def test_creation(self):
        msg = Message(role="system", content="You are a test agent.")
        assert msg.role == "system"
        assert msg.content == "You are a test agent."


class TestInferenceResult:
    def test_creation(self):
        result = InferenceResult(
            content="Hello",
            model="test-model",
            total_duration_ms=100,
            prompt_tokens=10,
            completion_tokens=5,
        )
        assert result.content == "Hello"
        assert result.model == "test-model"

    def test_optional_fields(self):
        result = InferenceResult(content="Hello", model="test-model")
        assert result.total_duration_ms is None
        assert result.prompt_tokens is None


class TestVLLMBackendStub:
    @pytest.mark.asyncio
    async def test_complete_raises(self):
        backend = VLLMBackend()
        with pytest.raises(NotImplementedError):
            await backend.complete([Message(role="user", content="test")])

    @pytest.mark.asyncio
    async def test_health_check_raises(self):
        backend = VLLMBackend()
        with pytest.raises(NotImplementedError):
            await backend.health_check()

    @pytest.mark.asyncio
    async def test_close_succeeds(self):
        backend = VLLMBackend()
        await backend.close()  # Should not raise
