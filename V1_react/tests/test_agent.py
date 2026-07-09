from agent.llm_client import LLMClient

from tools.tool_registry import ToolRegistry
from tools.calculator_tool import CalculatorTool
from tools.search_tool import SearchTool

from agent.reasoning_engine import ReasoningEngine

from tools.tool_executor import ToolExecutor
from agent.action_validator import ActionValidator

from agent.termination_checker import TerminationChecker
from agent.agent_loop import AgentLoop


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

    validator = ActionValidator(
        registry
    )

    executor = ToolExecutor(
        registry
    )

    termination_checker = (
        TerminationChecker(
            max_iterations=5
        )
    )

    agent = AgentLoop(
        reasoning_engine=reasoning_engine,
        action_validator=validator,
        tool_executor=executor,
        termination_checker=termination_checker
    )

    query = input(
        "Enter your query: "
    )

    context = agent.run(
        query
    )

    print("\nFinal Context:")
    print(context)


if __name__ == "__main__":
    main()