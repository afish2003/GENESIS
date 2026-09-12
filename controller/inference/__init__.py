"""Inference backends.

InferenceBackend is the abstraction; concrete backends are selected at runtime
by controller.inference.factory.create_backend() from RunConfig.
"""

from controller.inference.backend import InferenceBackend, InferenceResult, Message
from controller.inference.factory import create_backend, describe_backend

__all__ = [
    "InferenceBackend",
    "InferenceResult",
    "Message",
    "create_backend",
    "describe_backend",
]
