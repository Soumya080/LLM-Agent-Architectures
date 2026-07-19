from tools.base_tool import BaseTool


class ToolRegistry:

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: BaseTool):
        if tool.name in self.tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered."
            )

        self.tools[tool.name] = tool

    def get_tool(self, tool_name: str) -> BaseTool:
        if tool_name not in self.tools:
            raise ValueError(
                f"Tool '{tool_name}' is not registered."
            )

        return self.tools[tool_name]

    def list_tools(self):
        return list(self.tools.keys())

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.tools

    def __str__(self):
        return (
            f"ToolRegistry("
            f"tools={list(self.tools.keys())})"
        )