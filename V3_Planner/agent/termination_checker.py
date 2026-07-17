class TerminationChecker:
    """
    Responsible for deciding whether the agent
    should continue execution.
    """

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations

    def should_continue(self, context) -> bool:
        """
        Returns:
            True  -> Continue execution
            False -> Stop execution
        """

        # Agent explicitly finished
        if context.done:
            return False

        # Maximum iterations reached
        if context.iteration_count >= self.max_iterations:
            return False

        return True