import json
import re
from typing import Any, Dict


class RecursivePlannerParser:
    """
    Parses the output produced by the RecursivePlannerPrompt.

    Responsibility:
        Raw LLM Response
                ↓
        Clean Response
                ↓
        JSON
                ↓
        Validate
                ↓
        Normalize
                ↓
        Return Python Dictionary

    This class DOES NOT:
        - Create GoalNodes
        - Modify GoalTree
        - Execute Planning
    """

    def parse(self, response: str) -> Dict[str, Any]:
        cleaned = self._clean_response(response)

        data = self._load_json(cleaned)

        self._validate_response(data)

        data = self._normalize(data)

        return data

    # --------------------------------------------------
    # Internal Helpers
    # --------------------------------------------------

    def _clean_response(self, response: str) -> str:
        """
        Removes markdown code fences from LLM output.
        """

        response = response.strip()

        match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            response,
            flags=re.DOTALL,
        )

        if match:
            return match.group(1).strip()

        return response

    def _load_json(self, response: str) -> Dict[str, Any]:
        """
        Converts JSON string into Python dictionary.
        """

        try:
            return json.loads(response)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON returned by planner.\n\n{response}"
            ) from exc

    def _validate_response(self, data: Dict[str, Any]) -> None:
        """
        Validate top-level planner response.
        """

        if not isinstance(data, dict):
            raise ValueError("Planner response must be a JSON object.")

        data.setdefault("reasoning", "Decomposition decision.")
        data.setdefault("children", [])

        required_fields = [
            "decompose",
            "reasoning",
            "children",
        ]

        for field in required_fields:
            if field not in data:
                raise ValueError(
                    f"Missing required field '{field}'."
                )

        if not isinstance(data["decompose"], bool):
            raise ValueError("'decompose' must be boolean.")

        if not isinstance(data["reasoning"], str):
            raise ValueError("'reasoning' must be string.")

        if not isinstance(data["children"], list):
            raise ValueError("'children' must be list.")

        for child in data["children"]:
            self._validate_child(child)

    def _validate_child(self, child: Dict[str, Any]) -> None:
        """
        Validate one child goal.
        """

        if not isinstance(child, dict):
            raise ValueError(
                "Every child must be a JSON object."
            )

        required_fields = [
            "goal",
            "priority",
            "completion_criteria",
        ]

        for field in required_fields:
            if field not in child:
                raise ValueError(
                    f"Child missing field '{field}'."
                )

        if not isinstance(child["goal"], str):
            raise ValueError("'goal' must be string.")

        if not isinstance(child["priority"], int):
            raise ValueError("'priority' must be integer.")

        if not isinstance(child["completion_criteria"], str):
            raise ValueError(
                "'completion_criteria' must be string."
            )

    def _normalize(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fill optional fields with defaults.
        """

        for child in data["children"]:

            child.setdefault("confidence", 1.0)

            child.setdefault("metadata", {})

        return data

    def __repr__(self) -> str:
        return "RecursivePlannerParser()"

    __str__ = __repr__