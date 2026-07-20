from agent.llm_client import LLMClient
from tools.tool_registry import ToolRegistry
from tools.calculator_tool import CalculatorTool
from tools.search_tool import SearchTool
from agent.reasoning_engine import ReasoningEngine
from schemas.execution_context import ExecutionContext
from schemas.plan_schema import PlanSchema
from schemas.task_schema import TaskSchema


def main():

    registry = ToolRegistry()

    registry.register_tool(
        CalculatorTool()
    )

    registry.register_tool(
        SearchTool()
    )

    llm = LLMClient(
        model_name="qwen2.5-coder:7b"
    )

    reasoning_engine = ReasoningEngine(
        llm_client=llm,
        tool_registry=registry
    )

    task = TaskSchema(
        task_id=1,
        description="Compute 25 + 37",
        capability_required="calculator",
        expected_output="the sum",
        completion_criteria="return the arithmetic result",
    )

    plan = PlanSchema(
        plan_id="plan-1",
        goal="Solve simple arithmetic",
        tasks=[task],
        overall_success_criteria="correct answer",
        planner_notes="Use calculator for arithmetic.",
    )

    execution_context = ExecutionContext(
        execution_id="exec-1",
        plan=plan,
        execution_status="RUNNING",
    )

    action = reasoning_engine.think(execution_context)

    print("\nGenerated Action:")
    print(action)


if __name__ == "__main__":
    main()