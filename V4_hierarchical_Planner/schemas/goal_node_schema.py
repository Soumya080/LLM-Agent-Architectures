from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional


class GoalStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DECOMPOSED = "decomposed"
    ATOMIC = "atomic"


class GoalType(str, Enum):
    ROOT = "ROOT"
    INTERMEDIATE = "INTERMEDIATE"
    ATOMIC = "ATOMIC"


class GoalNode:
    """
    Represents a single planning node inside the GoalTree.

    GoalNode contains ONLY planning information.
    It contains NO execution logic.
    """

    def __init__(
        self,
        goal_id: str,
        goal: str,
        goal_type: GoalType = GoalType.INTERMEDIATE,
        parent_id: Optional[str] = None,
        children_ids: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        depth: int = 0,
        status: GoalStatus = GoalStatus.PENDING,
        priority: int = 0,
        completion_criteria: str = "",
        estimated_cost: float = 0.0,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        goalnode: str = "",
        **kwargs
    ):

        if not goal_id:
            raise ValueError("goal_id cannot be empty.")

        if not goal:
            raise ValueError("goal cannot be empty.")

        if depth < 0:
            raise ValueError("depth cannot be negative.")

        if priority < 0:
            raise ValueError("priority cannot be negative.")

        if estimated_cost < 0:
            raise ValueError("estimated_cost cannot be negative.")

        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence must lie between 0 and 1.")

        self.goal_id = goal_id
        self.goal = goal
        self.goalnode = goalnode or goal_id

        self.goal_type = goal_type

        self.parent_id = parent_id
        self.children_ids = children_ids or []
        self.dependencies = dependencies or []

        self.depth = depth

        # Ensure status is an instance of GoalStatus or a compatible string
        if isinstance(status, str):
            try:
                self.status = GoalStatus(status.lower())
            except ValueError:
                self.status = status
        else:
            self.status = status

        self.priority = priority

        self.completion_criteria = completion_criteria

        self.estimated_cost = estimated_cost

        self.confidence = confidence

        self.metadata = metadata or {}

    @property
    def children(self) -> List[str]:
        return self.children_ids

    @property
    def parent(self) -> Optional[str]:
        return self.parent_id

    @parent.setter
    def parent(self, value: Optional[str]) -> None:
        self.parent_id = value

    # -------------------------------------------------------
    # Relationship Helpers
    # -------------------------------------------------------

    def add_child(self, child_id: str) -> None:
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def remove_child(self, child_id: str) -> None:
        if child_id in self.children_ids:
            self.children_ids.remove(child_id)

    def add_dependency(self, goal_id: str) -> None:
        if goal_id not in self.dependencies:
            self.dependencies.append(goal_id)

    # -------------------------------------------------------
    # State Helpers
    # -------------------------------------------------------

    def mark_running(self) -> None:
        self.status = GoalStatus.RUNNING

    def mark_completed(self) -> None:
        self.status = GoalStatus.COMPLETED

    def mark_failed(self) -> None:
        self.status = GoalStatus.FAILED

    def mark_atomic(self) -> None:
        self.goal_type = GoalType.ATOMIC
        self.status = GoalStatus.ATOMIC

    # -------------------------------------------------------
    # Query Helpers
    # -------------------------------------------------------

    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0

    def is_root(self) -> bool:
        return self.parent_id is None

    def is_atomic(self) -> bool:
        return self.goal_type == GoalType.ATOMIC

    # -------------------------------------------------------
    # Serialization
    # -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "goal": self.goal,
            "goal_type": self.goal_type.value,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "dependencies": self.dependencies,
            "depth": self.depth,
            "status": self.status.value if isinstance(self.status, GoalStatus) else str(self.status),
            "priority": self.priority,
            "completion_criteria": self.completion_criteria,
            "estimated_cost": self.estimated_cost,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalNode":
        status_val = data.get("status", "pending")
        if isinstance(status_val, str):
            status_val = status_val.lower()
        try:
            status = GoalStatus(status_val)
        except ValueError:
            status = status_val

        return cls(
            goal_id=data["goal_id"],
            goal=data["goal"],
            goal_type=GoalType(data.get("goal_type", "INTERMEDIATE")),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            dependencies=data.get("dependencies", []),
            depth=data.get("depth", 0),
            status=status,
            priority=data.get("priority", 0),
            completion_criteria=data.get("completion_criteria", ""),
            estimated_cost=data.get("estimated_cost", 0.0),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )

    # -------------------------------------------------------
    # Debugging
    # -------------------------------------------------------

    def __repr__(self) -> str:
        status_str = self.status.value if isinstance(self.status, GoalStatus) else str(self.status)
        return (
            f"GoalNode("
            f"id='{self.goal_id}', "
            f"goal='{self.goal}', "
            f"type={self.goal_type.value}, "
            f"depth={self.depth}, "
            f"status={status_str})"
        )

    __str__ = __repr__