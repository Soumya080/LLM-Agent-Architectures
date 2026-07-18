from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from schemas.plan_schema import PlanSchema
from schemas.task_schema import TaskSchema


class ExecutionContext:
    """
    Represents the runtime execution state of a plan.

    Responsibilities
    ----------------
    - Track execution progress
    - Track the current task
    - Store execution history
    - Track execution status

    This class NEVER:
    - Plans
    - Reasons
    - Executes tools
    - Evaluates
    - Reflects
    """

    VALID_STATUS = {
        "PENDING",
        "RUNNING",
        "PAUSED",
        "COMPLETED",
        "FAILED",
    }

    def __init__(
        self,
        execution_id: str,
        plan: PlanSchema,
        current_task_index: int = 0,
        completed_tasks: Optional[List[TaskSchema]] = None,
        failed_tasks: Optional[List[TaskSchema]] = None,
        skipped_tasks: Optional[List[TaskSchema]] = None,
        execution_status: str = "PENDING",
        iteration: int = 0,
        runtime_history: Optional[List[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        started_at: Optional[str] = None,
    ):

        if execution_status not in self.VALID_STATUS:
            raise ValueError(
                f"Invalid execution status: {execution_status}"
            )

        self.execution_id = execution_id
        self.plan = plan

        self.current_task_index = current_task_index

        self.completed_tasks = completed_tasks or []
        self.failed_tasks = failed_tasks or []
        self.skipped_tasks = skipped_tasks or []

        self.execution_status = execution_status

        self.iteration = iteration

        self.runtime_history = runtime_history or []

        self.metadata = metadata or {}

        self.started_at = (
            started_at
            if started_at is not None
            else datetime.now(timezone.utc).isoformat()
        )

    # =====================================================
    # Properties
    # =====================================================

    @property
    def current_task(self) -> Optional[TaskSchema]:
        """
        Return the currently active task.
        """

        if (
            self.current_task_index >= len(self.plan.tasks)
            or self.current_task_index < 0
        ):
            return None

        return self.plan.tasks[self.current_task_index]

    # =====================================================
    # Execution Helpers
    # =====================================================

    def has_next_task(self) -> bool:
        return self.current_task_index + 1 < len(self.plan.tasks)

    def is_finished(self) -> bool:
        return self.current_task_index >= len(self.plan.tasks)

    def advance(self) -> Optional[TaskSchema]:
        """
        Move execution to the next task.

        Returns:
            Next TaskSchema or None.
        """

        if self.has_next_task():
            self.current_task_index += 1
            return self.current_task

        self.current_task_index = len(self.plan.tasks)
        return None

    def complete_current_task(self):
        """
        Mark the current task as completed.
        """

        if self.current_task is not None:
            self.completed_tasks.append(
                self.current_task
            )

    def fail_current_task(self):
        """
        Mark the current task as failed.
        """

        if self.current_task is not None:
            self.failed_tasks.append(
                self.current_task
            )

    def skip_current_task(self):
        """
        Skip the current task.
        """

        if self.current_task is not None:
            self.skipped_tasks.append(
                self.current_task
            )

    def increment_iteration(self):
        self.iteration += 1

    def add_runtime_event(self, event: Any):
        self.runtime_history.append(event)

    def mark_running(self):
        self.execution_status = "RUNNING"

    def mark_completed(self):
        self.execution_status = "COMPLETED"

    def mark_failed(self):
        self.execution_status = "FAILED"

    def mark_paused(self):
        self.execution_status = "PAUSED"

    # =====================================================
    # Serialization
    # =====================================================

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "plan": self.plan.to_dict(),
            "current_task_index": self.current_task_index,
            "completed_tasks": [
                task.to_dict()
                for task in self.completed_tasks
            ],
            "failed_tasks": [
                task.to_dict()
                for task in self.failed_tasks
            ],
            "skipped_tasks": [
                task.to_dict()
                for task in self.skipped_tasks
            ],
            "execution_status": self.execution_status,
            "iteration": self.iteration,
            "runtime_history": self.runtime_history,
            "metadata": self.metadata,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "ExecutionContext":

        return cls(
            execution_id=data["execution_id"],
            plan=PlanSchema.from_dict(
                data["plan"]
            ),
            current_task_index=data.get(
                "current_task_index",
                0,
            ),
            completed_tasks=[
                TaskSchema.from_dict(task)
                for task in data.get(
                    "completed_tasks",
                    [],
                )
            ],
            failed_tasks=[
                TaskSchema.from_dict(task)
                for task in data.get(
                    "failed_tasks",
                    [],
                )
            ],
            skipped_tasks=[
                TaskSchema.from_dict(task)
                for task in data.get(
                    "skipped_tasks",
                    [],
                )
            ],
            execution_status=data.get(
                "execution_status",
                "PENDING",
            ),
            iteration=data.get(
                "iteration",
                0,
            ),
            runtime_history=data.get(
                "runtime_history",
                [],
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
            started_at=data.get(
                "started_at",
            ),
        )

    def __repr__(self):
        return (
            f"ExecutionContext("
            f"execution_id='{self.execution_id}', "
            f"status='{self.execution_status}', "
            f"current_task={self.current_task_index}, "
            f"completed={len(self.completed_tasks)}, "
            f"failed={len(self.failed_tasks)}, "
            f"iteration={self.iteration}"
            f")"
        )

    __str__ = __repr__