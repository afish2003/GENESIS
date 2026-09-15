"""Sandbox tasks — what the agents build each cycle.

The cycle loop is task-agnostic. Selecting a task changes what gets built and
how it is scored; it does not touch orchestration, doctrine, identity, memory,
retrieval or logging.
"""

from controller.tasks.base import Task
from controller.tasks.code import CodeTask
from controller.tasks.protocol import ProtocolTask

TASKS: dict[str, type[Task]] = {
    ProtocolTask.name: ProtocolTask,
    CodeTask.name: CodeTask,
}

__all__ = ["Task", "ProtocolTask", "CodeTask", "TASKS", "create_task"]


def create_task(config) -> Task:
    """Build the task named by config.task."""
    try:
        return TASKS[config.task]()
    except KeyError:
        raise ValueError(
            f"Unknown task {config.task!r}. Available: {sorted(TASKS)}"
        ) from None
