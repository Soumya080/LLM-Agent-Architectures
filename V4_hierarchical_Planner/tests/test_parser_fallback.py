import sys
import os
import re

# Resolve project root so imports work
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from agent.reasoning_engine import ReasoningEngine
from agent.llm_client import LLMClient
from tools.tool_registry import ToolRegistry

def test_parser_fallback():
    # Setup dummy objects
    llm = LLMClient()
    registry = ToolRegistry()
    engine = ReasoningEngine(llm, registry)

    # Test cases
    # Case 1: Standard valid JSON
    res1 = engine.parse_response('{"tool_name": "search", "parameters": {"query": "test"}, "reason": "why"}')
    assert res1.tool_name == "search"
    assert res1.parameters == {"query": "test"}

    # Case 2: Math expression 4500 * 3
    res2 = engine.parse_response('{"tool_name": "calculator", "parameters": {"operation": "subtract", "operands": [4500 * 3, 8000]}, "reason": "Calculate remainder"}')
    assert res2.tool_name == "calculator"
    assert res2.parameters == {"operation": "subtract", "operands": [13500, 8000]}

    # Case 3: Math expression 2 * 17.5, 2 * 23.8 with newlines
    res3 = engine.parse_response('''{
  "tool_name": "calculator",
  "parameters": {
    "operation": "add",
    "operands": [2 * 17.5, 2 * 23.8]
  },
  "reason": "Calculate perimeter"
}''')
    assert res3.tool_name == "calculator"
    assert res3.parameters == {"operation": "add", "operands": [35.0, 47.6]}

    # Case 4: MD code block wrapping
    res4 = engine.parse_response('''```json
{
  "tool_name": "calculator",
  "parameters": {
    "operation": "multiply",
    "operands": [10 + 20, 2 ** 3]
  },
  "reason": "Calculate exponent"
}
```''')
    assert res4.tool_name == "calculator"
    assert res4.parameters == {"operation": "multiply", "operands": [30, 8]}

    print("All parser fallback tests passed successfully!")

if __name__ == "__main__":
    test_parser_fallback()
