class ActionSchema:
    """
    Represents a single action that the agent wants to perform.

    Example:
        Tool: search
        Parameters: {"query": "python creation year"}
        Reason: "Need the year before calculating square root"
    """

    def __init__(self, tool_name: str, parameters: dict, reason: str):
        self.tool_name = tool_name
        self.parameters = parameters
        self.reason = reason

    def __str__(self):
        return (
            f"ActionSchema("
            f"tool_name='{self.tool_name}', "
            f"parameters={self.parameters}, "
            f"reason='{self.reason}')"
        )

    def to_dict(self):
        """
        Convert ActionSchema object into dictionary.
        Useful for logging, JSON conversion, APIs, etc.
        """
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "reason": self.reason
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create ActionSchema object from dictionary.
        """
        return cls(
            tool_name=data.get("tool_name"),
            parameters=data.get("parameters", {}),
            reason=data.get("reason", "")
        )