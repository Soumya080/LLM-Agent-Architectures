from __future__ import annotations

from typing import Optional

from schemas.execution_context import ExecutionContext
from tools.tool_registry import ToolRegistry


class ReasoningPrompt:
    """
    Build task-focused prompts for the reasoning engine.
    """

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.tool_registry = tool_registry

    def build_prompt(
        self,
        execution_context: ExecutionContext,
        relevant_memory: Optional[str] = None,
    ) -> str:
        current_task = execution_context.current_task
        plan = execution_context.plan

        if current_task is None:
            raise ValueError("No current task available in execution context.")

        available_tools = self._format_tools()
        history = self._format_runtime_history(execution_context)
        memory_block = relevant_memory or "None available."

        return f"""You are the reasoning component of a Planner -> Reason -> Act -> Evaluate -> Reflect agent.

Your job is to decide the NEXT ACTION for the CURRENT TASK only.
Do NOT reason about the entire original user query unless it is directly relevant to the current task.

PLAN GOAL:
{plan.goal}

CURRENT TASK:
- Task ID: {current_task.task_id}
- Description: {current_task.description}
- Capability Required: {current_task.capability_required}
- Expected Output: {current_task.expected_output}
- Completion Criteria: {current_task.completion_criteria}
- Dependencies: {', '.join(map(str, current_task.dependencies)) if current_task.dependencies else 'None'}
- Priority: {current_task.priority}
- Status: {current_task.status}

PLANNER NOTES:
{plan.planner_notes or 'None'}

RELEVANT RUNTIME HISTORY:
{history}

RELEVANT MEMORY:
{memory_block}

AVAILABLE TOOLS:
{available_tools}

RETURN ONLY VALID JSON with this shape:
{{"tool_name": "tool_name", "parameters": {{...}}, "reason": "why this action is appropriate for the current task"}}

Rules:
1. Focus only on the current task.
2. Use the current task's expected output and completion criteria to decide the next action.
3. Prefer tools that advance the current task specifically.
4. If the task is complete, call finish with a concise answer.
5. Do not repeat the same action with the same parameters.
"""

    def _format_tools(self) -> str:
        if self.tool_registry is None:
            return "- search: query a source of information\n- calculator: perform arithmetic\n- finish: provide the final answer"

        tool_lines = []
        for tool_name in self.tool_registry.list_tools():
            tool = self.tool_registry.get_tool(tool_name)
            tool_lines.append(f"- {tool.name}: {tool.description}")

        return "\n".join(tool_lines) if tool_lines else "None"

    def _format_runtime_history(
        self,
        execution_context: ExecutionContext,
    ) -> str:
        if not execution_context.runtime_history:
            return "None yet."

        lines = []
        for index, event in enumerate(execution_context.runtime_history, start=1):
            lines.append(f"{index}. {event}")

        return "\n".join(lines)
