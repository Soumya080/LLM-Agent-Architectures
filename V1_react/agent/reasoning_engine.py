import json
import re

from schemas.action_schema import ActionSchema
from schemas.context_schema import ContextSchema
from agent.llm_client import LLMClient
from tools.tool_registry import ToolRegistry


class ReasoningEngine:
    """
    Responsible for converting the current agent state
    into the next action using the LLM.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry

    def format_history(
        self,
        context: ContextSchema
    ) -> str:
        """
        Format action/observation history into clean,
        numbered text the LLM can read and reason over.
        """

        if not context.action_history:
            return "None yet."

        lines = []

        for i, action in enumerate(
            context.action_history
        ):
            step_num = i + 1
            lines.append(
                f"Step {step_num}:"
            )
            lines.append(
                f"  Action: {action.tool_name}"
                f"({action.parameters})"
            )

            # Pair with observation if available
            if i < len(context.observation_history):
                obs = context.observation_history[i]

                if obs.success:
                    # Truncate long results
                    result_str = str(obs.result)
                    if len(result_str) > 500:
                        result_str = (
                            result_str[:500] + "..."
                        )
                    lines.append(
                        f"  Result: {result_str}"
                    )
                else:
                    lines.append(
                        f"  Error: {obs.error}"
                    )

            lines.append("")

        return "\n".join(lines)

    def build_prompt(
        self,
        context: ContextSchema
    ) -> str:
        """
        Build prompt for the LLM using current context.
        """

        available_tools = ""

        for tool_name in self.tool_registry.list_tools():
            tool = self.tool_registry.get_tool(tool_name)

            available_tools += (
                f"- {tool.name}: "
                f"{tool.description}\n"
            )

        history = self.format_history(context)

        prompt = f"""You are a ReAct agent. You solve tasks step-by-step by choosing one tool at a time.

Available Tools:
{available_tools}

TOOL PARAMETER FORMATS:
- search: {{"query": "your search query"}}
- calculator: {{"operation": "add|subtract|multiply|divide|sqrt|power", "operands": [numbers]}}
- finish: {{"answer": "your final answer"}}

EXAMPLE of multi-step reasoning:

Query: "How many seconds are in the number of days in a leap year?"

Step 1: I search for the number of days.
{{"tool_name": "search", "parameters": {{"query": "how many days in a leap year"}}, "reason": "I need to find the number of days in a leap year."}}
Result: A leap year has 366 days.

Step 2: Now I know it is 366 days. I calculate 366 * 24 * 3600.
{{"tool_name": "calculator", "parameters": {{"operation": "multiply", "operands": [366, 24, 3600]}}, "reason": "I learned 366 days. Now I calculate seconds."}}
Result: 31622400

Step 3: I have the answer.
{{"tool_name": "finish", "parameters": {{"answer": "There are 31,622,400 seconds in 366 days."}}, "reason": "Calculation complete."}}

RULES:
1. NEVER repeat a previous action with the same parameters.
2. If a search already returned useful information, USE it with calculator or finish.
3. If you have enough information to answer, call finish immediately.

USER QUERY:
{context.query}

YOUR PREVIOUS STEPS:
{history}

Based on the steps above, what is your NEXT action? Return ONLY valid JSON:
"""

        return prompt

    def parse_response(
        self,
        response: str
    ) -> ActionSchema:
        """
        Convert LLM JSON response into ActionSchema.
        """

        try:
            # Strip markdown code fences if present
            cleaned = response.strip()
            fence_match = re.search(
                r'```(?:json)?\s*\n?(.*?)\n?\s*```',
                cleaned,
                re.DOTALL
            )
            if fence_match:
                cleaned = fence_match.group(1).strip()

            try:
                action_data = json.loads(cleaned)
            except json.JSONDecodeError:
                # If json loading fails, try resolving mathematical expressions in operands
                # E.g., [4500 * 3, 8000] -> [13500, 8000]
                # Pattern matches double-quoted strings (to skip/keep them as is) OR math expressions.
                pattern = r'("([^"\\]|\\.)*")|(\b\d+(?:\.\d+)?(?:\s*(?:\*\*|[\+\-\*\/])\s*\d+(?:\.\d+)?)+\b)'
                
                def replacer(match):
                    if match.group(1):
                        return match.group(1)
                    else:
                        expr = match.group(3)
                        try:
                            # Evaluate expression safely (no letters, only digits/operators)
                            if not re.search(r'[a-zA-Z_]', expr):
                                return str(eval(expr))
                        except Exception:
                            pass
                        return expr

                fixed = re.sub(pattern, replacer, cleaned)
                action_data = json.loads(fixed)

            return ActionSchema.from_dict(
                action_data
            )

        except Exception as e:
            raise ValueError(
                f"Failed to parse LLM response: {e}\nRaw response: {response}"
            )

    def _is_duplicate_action(
        self,
        action: ActionSchema,
        context: ContextSchema
    ) -> bool:
        """
        Check if the action was already performed.
        """

        for prev in context.action_history:
            if (
                prev.tool_name == action.tool_name
                and prev.parameters == action.parameters
            ):
                return True

        return False

    def think(
        self,
        context: ContextSchema
    ) -> ActionSchema:
        """
        Generate the next action using the LLM.
        Retries once if the action is a duplicate.
        """

        prompt = self.build_prompt(context)

        response = self.llm_client.generate(
            prompt
        )

        action = self.parse_response(
            response
        )

        # Safety net: retry once if duplicate
        if self._is_duplicate_action(
            action, context
        ):
            retry_hint = (
                "\n\nYou already performed: "
                f"{action.tool_name}({action.parameters}). "
                "Do NOT repeat it. Based on the result "
                "you already received, choose a DIFFERENT "
                "tool or call finish with your answer. "
                "Return ONLY valid JSON:"
            )

            response = self.llm_client.generate(
                prompt + retry_hint
            )

            action = self.parse_response(
                response
            )

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