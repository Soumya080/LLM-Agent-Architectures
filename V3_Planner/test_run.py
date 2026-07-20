"""
Non-interactive test: runs the agent on a fixed query.
"""

from agent.llm_client import LLMClient
from tools.tool_registry import ToolRegistry
from tools.calculator_tool import CalculatorTool
from tools.search_tool import SearchTool
from tools.finish_tool import FinishTool
from agent.reasoning_engine import ReasoningEngine
from tools.tool_executor import ToolExecutor
from agent.action_validator import ActionValidator
from agent.termination_checker import TerminationChecker
from agent.agent_loop import AgentLoop


def main():
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_tool(SearchTool())
    registry.register_tool(FinishTool())

    llm = LLMClient(model_name="qwen2.5-coder:7b")

    if not llm.health_check():
        print("ERROR: Ollama not reachable.")
        return

    reasoning_engine = ReasoningEngine(llm_client=llm, tool_registry=registry)
    validator = ActionValidator(registry)
    executor = ToolExecutor(registry)
    termination_checker = TerminationChecker(max_iterations=5)

    agent = AgentLoop(
        reasoning_engine=reasoning_engine,
        action_validator=validator,
        tool_executor=executor,
        termination_checker=termination_checker
    )

    query = "What is the square root of the year Python was created?"
    print(f"Query: {query}\n")

    context = agent.run(query)

    print("\n" + "-" * 40)
    print("Agent Trajectory:")
    print(f"  Iterations: {context.iteration_count}")
    print(f"  Done: {context.done}")

    for i, (action, obs) in enumerate(
        zip(context.action_history, context.observation_history)
    ):
        print(f"\n  Step {i + 1}:")
        print(f"    Tool: {action.tool_name}")
        print(f"    Params: {action.parameters}")
        print(f"    Reason: {action.reason}")
        print(f"    Success: {obs.success}")
        result_str = str(obs.result)
        if len(result_str) > 300:
            result_str = result_str[:300] + "..."
        print(f"    Result: {result_str}")

    print("-" * 40)


if __name__ == "__main__":
    main()
