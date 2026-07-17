import json
import re
from typing import Optional

from schemas.action_schema import ActionSchema
from schemas.execution_context import ExecutionContext
from agent.llm_client import LLMClient
from tools.tool_registry import ToolRegistry
from agent.reasoning_prompt import ReasoningPrompt


class ReasoningEngine:
    """
    Decide the next action for the current task in the execution plan.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        prompt_builder: Optional[ReasoningPrompt] = None,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.prompt_builder = prompt_builder or ReasoningPrompt(tool_registry)

    def build_prompt(
        self,
        execution_context: ExecutionContext,
        relevant_memory: Optional[str] = None,
    ) -> str:
        """
        Build a task-focused prompt from the active execution context.
        """

        return self.prompt_builder.build_prompt(
            execution_context=execution_context,
            relevant_memory=relevant_memory,
        )

    def parse_response(self, response: str) -> ActionSchema:
        """
        Convert LLM JSON response into ActionSchema.
        """

        try:
            cleaned = response.strip()
            fence_match = re.search(
                r"```(?:json)?\s*\n?(.*?)\n?\s*```",
                cleaned,
                re.DOTALL,
            )
            if fence_match:
                cleaned = fence_match.group(1).strip()

            try:
                action_data = json.loads(cleaned)
            except json.JSONDecodeError:
                pattern = r'("([^"\\]|\\.)*")|(\b\d+(?:\.\d+)?(?:\s*(?:\*\*|[\+\-\*\/])\s*\d+(?:\.\d+)?)+\b)'

                def replacer(match):
                    if match.group(1):
                        return match.group(1)

                    expr = match.group(3)
                    try:
                        if not re.search(r"[a-zA-Z_]", expr):
                            return str(eval(expr))
                    except Exception:
                        pass
                    return expr

                fixed = re.sub(pattern, replacer, cleaned)
                action_data = json.loads(fixed)

            return ActionSchema.from_dict(action_data)

        except Exception as exc:
            raise ValueError(
                f"Failed to parse LLM response: {exc}\nRaw response: {response}"
            ) from exc

    def _is_duplicate_action(
        self,
        action: ActionSchema,
        execution_context: ExecutionContext,
    ) -> bool:
        """
        Check if the action was already performed.
        """

        if not execution_context.runtime_history:
            return False

        for event in execution_context.runtime_history:
            if isinstance(event, dict) and "action" in event:
                previous_action = event["action"]
                if (
                    previous_action.get("tool_name") == action.tool_name
                    and previous_action.get("parameters") == action.parameters
                ):
                    return True

        return False

    def think(
        self,
        execution_context: ExecutionContext,
        relevant_memory: Optional[str] = None,
    ) -> ActionSchema:
        """
        Generate the next action for the active task using the LLM.
        Retries once if the action is a duplicate.
        """

        prompt = self.build_prompt(execution_context, relevant_memory)

        response = self.llm_client.generate(prompt)
        action = self.parse_response(response)

        if self._is_duplicate_action(action, execution_context):
            retry_hint = (
                "\n\nYou already performed this action for the current task. "
                "Choose a different step or call finish if the task is complete. "
                "Return ONLY valid JSON:"
            )

            response = self.llm_client.generate(prompt + retry_hint)
            action = self.parse_response(response)

        return action

    def __str__(self):
        return (
            f"ReasoningEngine("
            f"llm={self.llm_client.model_name}, "
            f"tools={self.tool_registry.list_tools()}"
            f")"
        )

    def __repr__(self):
        return self.__str__()