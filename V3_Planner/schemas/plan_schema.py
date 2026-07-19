from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from schemas.task_schema import TaskSchema


class PlanSchema:
    """
    Represents the complete execution plan generated
    by the Planner.

    A Plan consists of one or more TaskSchema objects.

    NOTE:
    PlanSchema contains NO execution logic.
    """

    VALID_STATUS = {
        "PENDING",
        "RUNNING",
        "COMPLETED",
        "FAILED",
        "ABANDONED",
    }

    def __init__(
        self,
        plan_id: str,
        goal: str,
        tasks: List[TaskSchema],
        overall_success_criteria: str,
        planner_notes: str = "",
        status: str = "PENDING",
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
    ):

        if not goal:
            raise ValueError("Plan goal cannot be empty.")

        if not tasks:
            raise ValueError("Plan must contain at least one task.")

        if status not in self.VALID_STATUS:
            raise ValueError(
                f"Invalid plan status: {status}"
            )

        self.plan_id = plan_id
        self.goal = goal
        self.tasks = tasks
        self.overall_success_criteria = overall_success_criteria
        self.planner_notes = planner_notes
        self.status = status
        self.metadata = metadata or {}

        self.created_at = (
            created_at
            if created_at is not None
            else datetime.utcnow().isoformat()
        )

    def mark_running(self):
        self.status = "RUNNING"

    def mark_completed(self):
        self.status = "COMPLETED"

    def mark_failed(self):
        self.status = "FAILED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "tasks": [
                task.to_dict()
                for task in self.tasks
            ],
            "overall_success_criteria": self.overall_success_criteria,
            "planner_notes": self.planner_notes,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any]
    ) -> "PlanSchema":

        tasks = [
            TaskSchema.from_dict(task)
            for task in data.get("tasks", [])
        ]

        return cls(
            plan_id=data.get("plan_id"),
            goal=data.get("goal", ""),
            tasks=tasks,
            overall_success_criteria=data.get(
                "overall_success_criteria",
                "",
            ),
            planner_notes=data.get(
                "planner_notes",
                "",
            ),
            status=data.get(
                "status",
                "PENDING",
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
            created_at=data.get(
                "created_at",
            ),
        )

    def __repr__(self):
        return (
            f"PlanSchema("
            f"plan_id='{self.plan_id}', "
            f"goal='{self.goal}', "
            f"tasks={len(self.tasks)}, "
            f"status='{self.status}')"
        )

    __str__ = __repr__