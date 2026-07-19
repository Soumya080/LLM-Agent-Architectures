from tools.base_tool import BaseTool


class FinishTool(BaseTool):
    """
    Tool for finishing the task and returning the final answer.
    """
    def __init__(self):
        super().__init__(
            name="finish",
            description="Use this tool when the user's task is fully solved and a final answer should be returned."
        )

    def execute(self, parameters: dict):
        """
        Finish the task and return the final answer.
        """

        answer = parameters.get("answer")

        if not answer:
            raise ValueError(
                "FinishTool requires an 'answer' parameter."
            )

        return answer