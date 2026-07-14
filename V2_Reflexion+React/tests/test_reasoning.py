from agent.llm_client import LLMClient
from tools.tool_registry import ToolRegistry
from tools.calculator_tool import CalculatorTool
from tools.search_tool import SearchTool
from agent.reasoning_engine import ReasoningEngine
from schemas.context_schema import ContextSchema


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

    context = ContextSchema(
        query="What is 25 + 37?"
    )

    action = reasoning_engine.think(
        context
    )

    print("\nGenerated Action:")
    print(action)


if __name__ == "__main__":
    main()