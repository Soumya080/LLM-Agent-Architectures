class ObservationSchema:
    """
    Represents the response received from a tool execution.
    """

    def __init__(
        self,
        tool_name: str,
        success: bool,
        result=None,
        error=None
    ):
        self.tool_name = tool_name
        self.success = success
        self.result = result
        self.error = error

    def __str__(self):
        return (
            f"ObservationSchema("
            f"tool_name='{self.tool_name}', "
            f"success={self.success}, "
            f"result={self.result}, "
            f"error={self.error})"
        )

    def to_dict(self):
        """
        Convert ObservationSchema object into dictionary.
        Useful for logging, JSON conversion, APIs, etc.
        """
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create ObservationSchema object from dictionary.
        """
        return cls(
            tool_name=data.get("tool_name", ""),
            success=data.get("success", False),
            result=data.get("result", None),
            error=data.get("error", None)
        )
        