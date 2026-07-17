import json
import re
from typing import Dict, Any

from schemas.plan_schema import PlanSchema


class PlannerParser:
    """
    Converts the Planner LLM response into a validated PlanSchema.

    Responsibilities
    ----------------
    1. Remove markdown formatting.
    2. Parse JSON.
    3. Validate required fields.
    4. Construct PlanSchema.

    This class NEVER:
    - Calls the LLM
    - Executes plans
    - Knows about tools
    - Knows about the Reasoning Engine
    """

    REQUIRED_PLAN_FIELDS = {
        "goal",
        "tasks",
        "overall_success_criteria",
    }

    REQUIRED_TASK_FIELDS = {
        "task_id",
        "description",
        "capability_required",
        "expected_output",
        "completion_criteria",
        "priority",
    }

    def parse(self, response: str) -> PlanSchema:
        """
        Parse an LLM response into a PlanSchema.

        Args:
            response: Raw LLM response.

        Returns:
            PlanSchema

        Raises:
            ValueError:
                If the planner output is invalid.
        """

        cleaned = self._clean_response(response)

        try:
            plan_data = json.loads(cleaned)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Planner returned invalid JSON.\n\n{cleaned}"
            ) from exc

        self._validate_plan(plan_data)

        return PlanSchema.from_dict(plan_data)

    def _clean_response(self, response: str) -> str:
        """
        Remove markdown code fences from the LLM response.
        """

        response = response.strip()

        match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            response,
            re.DOTALL,
        )

        if match:
            return match.group(1).strip()

        return response

    def _validate_plan(self, plan_data: Dict[str, Any]) -> None:
        """
        Validate planner JSON before constructing PlanSchema.
        """

        if not isinstance(plan_data, dict):
            raise ValueError(
                "Planner output must be a JSON object."
            )

        missing = self.REQUIRED_PLAN_FIELDS - plan_data.keys()

        if missing:
            raise ValueError(
                f"Planner output missing fields: {missing}"
            )

        tasks = plan_data["tasks"]

        if not isinstance(tasks, list):
            raise ValueError(
                "'tasks' must be a list."
            )

        if len(tasks) == 0:
            raise ValueError(
                "Planner generated an empty task list."
            )

        for idx, task in enumerate(tasks):

            if not isinstance(task, dict):
                raise ValueError(
                    f"Task {idx} is not a JSON object."
                )

            missing = (
                self.REQUIRED_TASK_FIELDS
                - task.keys()
            )

            if missing:
                raise ValueError(
                    f"Task {idx} missing fields: {missing}"
                )

    def __repr__(self):
        return "PlannerParser()"

    __str__ = __repr__