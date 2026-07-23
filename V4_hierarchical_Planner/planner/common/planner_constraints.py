from __future__ import annotations

from typing import Any, Dict, Optional


class PlannerConstraints:
    """
    Immutable configuration object defining the limits under which the
    Hierarchical Planner is allowed to operate.

    Responsibilities
    ----------------
    - Store planning constraints
    - Validate runtime state against constraints
    - Serialize / Deserialize

    Non Responsibilities
    --------------------
    - Planning
    - Decomposition
    - Decision Making
    - LLM Calls
    """

    def __init__(
        self,
        max_depth: int = 8,
        max_nodes: int = 200,
        max_recursive_calls: int = 200,
        max_llm_calls: int = 100,
        max_planning_time: float = 60.0,
        max_tokens: int = 50000,
        min_confidence: float = 0.60,
        min_atomic_confidence: float = 0.75,
        max_retry: int = 2,
        max_child_nodes: int = 10,
        max_branching: int = 10,
        max_dependencies: int = 20,
    ) -> None:

        # ------------------------
        # Validation
        # ------------------------

        if max_depth <= 0:
            raise ValueError("max_depth must be > 0")

        if max_nodes <= 0:
            raise ValueError("max_nodes must be > 0")

        if max_recursive_calls <= 0:
            raise ValueError("max_recursive_calls must be > 0")

        if max_llm_calls <= 0:
            raise ValueError("max_llm_calls must be > 0")

        if max_planning_time <= 0:
            raise ValueError("max_planning_time must be > 0")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")

        if max_retry < 0:
            raise ValueError("max_retry cannot be negative")

        if max_child_nodes <= 0:
            raise ValueError("max_child_nodes must be > 0")

        if max_branching <= 0:
            raise ValueError("max_branching must be > 0")

        if max_dependencies < 0:
            raise ValueError("max_dependencies cannot be negative")

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

        if not 0.0 <= min_atomic_confidence <= 1.0:
            raise ValueError("min_atomic_confidence must be between 0 and 1")

        # ------------------------
        # Search Constraints
        # ------------------------

        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_recursive_calls = max_recursive_calls

        # ------------------------
        # Budget Constraints
        # ------------------------

        self.max_llm_calls = max_llm_calls
        self.max_planning_time = max_planning_time
        self.max_tokens = max_tokens

        # ------------------------
        # Quality Constraints
        # ------------------------

        self.min_confidence = min_confidence
        self.min_atomic_confidence = min_atomic_confidence
        self.max_retry = max_retry

        # ------------------------
        # Safety Constraints
        # ------------------------

        self.max_child_nodes = max_child_nodes
        self.max_branching = max_branching
        self.max_dependencies = max_dependencies

    # --------------------------------------------------
    # Runtime Validation
    # --------------------------------------------------

    def validate(self, state: Dict[str, Any]) -> Dict[str, Any]:

        violations = []

        if state.get("depth", 0) > self.max_depth:
            violations.append("Maximum tree depth exceeded.")

        if state.get("nodes", 0) > self.max_nodes:
            violations.append("Maximum node count exceeded.")

        if state.get("recursive_calls", 0) > self.max_recursive_calls:
            violations.append("Maximum recursive calls exceeded.")

        if state.get("llm_calls", 0) > self.max_llm_calls:
            violations.append("Maximum LLM calls exceeded.")

        if state.get("planning_time", 0.0) > self.max_planning_time:
            violations.append("Planning time budget exceeded.")

        if state.get("tokens", 0) > self.max_tokens:
            violations.append("Token budget exceeded.")

        return {
            "valid": len(violations) == 0,
            "violations": violations,
        }

    # --------------------------------------------------
    # Serialization
    # --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:

        return {
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "max_recursive_calls": self.max_recursive_calls,
            "max_llm_calls": self.max_llm_calls,
            "max_planning_time": self.max_planning_time,
            "max_tokens": self.max_tokens,
            "min_confidence": self.min_confidence,
            "min_atomic_confidence": self.min_atomic_confidence,
            "max_retry": self.max_retry,
            "max_child_nodes": self.max_child_nodes,
            "max_branching": self.max_branching,
            "max_dependencies": self.max_dependencies,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "PlannerConstraints":

        return cls(**data)

    # --------------------------------------------------
    # Representation
    # --------------------------------------------------

    def __repr__(self) -> str:

        return (
            "PlannerConstraints("
            f"depth={self.max_depth}, "
            f"nodes={self.max_nodes}, "
            f"llm_calls={self.max_llm_calls}, "
            f"tokens={self.max_tokens})"
        )

    __str__ = __repr__