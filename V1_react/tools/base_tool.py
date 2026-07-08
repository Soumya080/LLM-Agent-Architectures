class BaseTool:
    """
    Base class for all agent tools.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, parameters: dict):
        """
        Execute the tool with given parameters.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclass must implement execute() method."
        )

    def __str__(self):
        return (
            f"BaseTool("
            f"name='{self.name}', "
            f"description='{self.description}')"
        )