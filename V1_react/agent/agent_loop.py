from schemas.context_schema import ContextSchema


class AgentLoop:

    def __init__(
        self,
        reasoning_engine,
        action_validator,
        tool_executor,
        termination_checker
    ):
        self.reasoning_engine = reasoning_engine
        self.action_validator = action_validator
        self.tool_executor = tool_executor
        self.termination_checker = termination_checker

    def initialize_context(
        self,
        query: str
    ) -> ContextSchema:

        return ContextSchema(
            query=query
        )

    def run(
        self,
        query: str
    ):
        """
        Main ReAct loop.
        """

        context = self.initialize_context(
            query
        )

        while self.termination_checker.should_continue(
            context
        ):

            # THINK
            action = self.reasoning_engine.think(
                context
            )

            # STORE ACTION
            context.add_action(
                action
            )

            # VALIDATE
            self.action_validator.validate(
                action
            )

            # ACT
            observation = self.tool_executor.execute(
                action
            )
            
            # OBSERVE
            context.add_observation(
                observation
            )

            # UPDATE ITERATION
            context.increment_iteration()

            # V1 termination rule
            if action.tool_name == "finish":
                context.mark_done()
    
        return context
    
    
    def handle_observation(self, context: ContextSchema, observation):
        """
        Handle the observation after executing an action.
        This method can be extended to include more complex logic
        based on the observation's success or failure.
        """

        if observation.success:
            print(f"Action '{observation.tool_name}' executed successfully.")
            print(f"Result: {observation.result}")
        else:
            print(f"Action '{observation.tool_name}' failed.")
            print(f"Error: {observation.error}")
            
        return context
    
    
    