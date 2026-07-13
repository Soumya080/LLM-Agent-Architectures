from schemas.context_schema import ContextSchema


class AgentLoop:

    def __init__(
        self,
        reasoning_engine,
        action_validator,
        tool_executor,
        termination_checker,
        evaluator=None,
        reflection_engine=None
    ):
        self.reasoning_engine = reasoning_engine
        self.action_validator = action_validator
        self.tool_executor = tool_executor
        self.termination_checker = termination_checker
        self.evaluator = evaluator
        self.reflection_engine = reflection_engine

    def initialize_context(
        self,
        query: str
    ) -> ContextSchema:
        """
        Initialize fresh context for a new query.
        """

        return ContextSchema(
            query=query
        )

    def run(
        self,
        query: str
    ) -> ContextSchema:
        """
        Main Agent Execution Loop.

        Flow:

        Think
        ↓
        Act
        ↓
        Observe
        ↓
        Evaluate
        ↓
        Reflect (optional)
        ↓
        Repeat
        """

        context = self.initialize_context(
            query=query
        )

        while self.termination_checker.should_continue(
            context
        ):

            try:

                # ======================================
                # THINK
                # ======================================
                action = self.reasoning_engine.think(
                    context
                )

                context.add_action(
                    action
                )

                # ======================================
                # VALIDATE
                # ======================================
                self.action_validator.validate(
                    action
                )

                # ======================================
                # ACT
                # ======================================
                observation = self.tool_executor.execute(
                    action
                )

                context.add_observation(
                    observation
                )

                # ======================================
                # EVALUATE
                # ======================================
                evaluation = None
                if self.evaluator is not None:
                    evaluation = self.evaluator.evaluate(
                        query=context.query,
                        action=action,
                        observation=observation,
                        context=context
                    )

                    context.add_evaluation(
                        evaluation
                    )

                # ======================================
                # REFLECT
                # ======================================
                if evaluation is not None and evaluation.should_reflect and self.reflection_engine is not None:

                    reflection = (
                        self.reflection_engine.reflect(
                            query=context.query,
                            action=action,
                            observation=observation,
                            evaluation=evaluation,
                            context=context
                        )
                    )

                    context.add_reflection(
                        reflection
                    )

                # ======================================
                # TERMINATION
                # ======================================
                if (
                    action.tool_name.lower()
                    == "finish"
                    and (evaluation is None or evaluation.success)
                ):
                    context.mark_done()

                # ======================================
                # UPDATE ITERATION
                # ======================================
                context.increment_iteration()

            except Exception as e:

                print(
                    f"[AGENT LOOP ERROR]: {e}"
                )

                break

        return context

    def print_step(
        self,
        action,
        observation,
        evaluation,
        reflection=None
    ):
        """
        Optional debugging utility.
        """

        print("\n" + "=" * 60)

        print("\nACTION:")
        print(action)

        print("\nOBSERVATION:")
        print(observation)

        print("\nEVALUATION:")
        print(evaluation)

        if reflection is not None:
            print("\nREFLECTION:")
            print(reflection)

        print("\n" + "=" * 60)

    def __str__(self):
        return (
            f"AgentLoop("
            f"reasoning_engine={self.reasoning_engine}, "
            f"tool_executor={self.tool_executor}, "
            f"evaluator={self.evaluator}, "
            f"reflection_engine={self.reflection_engine}"
            f")"
        )

    def __repr__(self):
        return self.__str__()