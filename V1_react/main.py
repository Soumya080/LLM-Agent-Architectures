"""
ReAct V1 Agent - Main Entry Point

Usage:
    python main.py

Runs the full ReAct agent loop interactively.
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

    # ---- Tool Registry ----
    registry = ToolRegistry()

    registry.register_tool(
        CalculatorTool()
    )

    registry.register_tool(
        SearchTool()
    )
    registry.register_tool(
        FinishTool()
    )

    print(f"Registered tools: {registry.list_tools()}")

    # ---- LLM Client ----
    llm = LLMClient(
        model_name="qwen2.5-coder:7b"
    )

    if not llm.health_check():
        print("ERROR: Ollama is not reachable. Start it with: ollama serve")
        return

    print(f"LLM connected: {llm}")

    # ---- Agent Components ----
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

    termination_checker = TerminationChecker(
        max_iterations=5
    )

    # ---- Agent Loop ----
    agent = AgentLoop(
        reasoning_engine=reasoning_engine,
        action_validator=validator,
        tool_executor=executor,
        termination_checker=termination_checker
    )

    print("\n" + "=" * 50)
    print("  ReAct V1 Agent Ready")
    print("=" * 50)
    print("\nType your query and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    # ---- Interactive Loop ----
    while True:
        query = input("You: ").strip()

        if not query:
            print("Please enter a query.\n")
            continue

        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        print(f"\nProcessing: \"{query}\"\n")

        try:
            context = agent.run(query)

            # Show results
            print("\n" + "-" * 40)
            print("Agent Trajectory:")
            print(f"  Iterations: {context.iteration_count}")
            print(f"  Actions taken: {len(context.action_history)}")

            for i, (action, obs) in enumerate(
                zip(
                    context.action_history,
                    context.observation_history
                )
            ):
                print(f"\n  Step {i + 1}:")
                print(f"    Tool: {action.tool_name}")
                print(f"    Params: {action.parameters}")
                print(f"    Reason: {action.reason}")
                print(f"    Success: {obs.success}")
                print(f"    Result: {obs.result}")

            print("-" * 40 + "\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
