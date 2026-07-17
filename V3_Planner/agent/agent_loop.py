from typing import Optional

from schemas.context_schema import ContextSchema
from schemas.execution_context import ExecutionContext
from schemas.plan_schema import PlanSchema
from schemas.task_schema import TaskSchema
from planner.planner import Planner


class AgentLoop:

    def __init__(
        self,
        reasoning_engine,
        action_validator,
        tool_executor,
        termination_checker,
        evaluator=None,
        reflection_engine=None,
        plan: Optional[PlanSchema] = None,
        planner: Optional[Planner] = None,
    ):
        self.reasoning_engine = reasoning_engine
        self.action_validator = action_validator
        self.tool_executor = tool_executor
        self.termination_checker = termination_checker
        self.evaluator = evaluator
        self.reflection_engine = reflection_engine
        self.plan = plan
        self.planner = planner
        self.execution_context = None

    def initialize_context(
        self,
        query: str,
    ) -> ContextSchema:
        """
        Initialize fresh context for a new query.
        """

        return ContextSchema(query=query)

    def initialize_execution_context(
        self,
        query: str,
        context: ContextSchema,
    ) -> ExecutionContext:
        """
        Generate a plan and initialize execution context from it.
        """

        plan = self.plan
        if plan is None and self.planner is not None:
            plan = self.planner.plan(query=query, context=context)

        if plan is None:
            plan = self._build_default_plan(query)

        self.plan = plan

        execution_context = ExecutionContext(
            execution_id=f"exec-{len(query)}",
            plan=plan,
            execution_status="RUNNING",
        )

        execution_context.mark_running()
        self.execution_context = execution_context
        return execution_context

    def _build_default_plan(self, query: str) -> PlanSchema:
        task = TaskSchema(
            task_id=1,
            description=query,
            capability_required="general",
            expected_output="answer",
            completion_criteria="resolve the query",
        )

        return PlanSchema(
            plan_id="default-plan",
            goal=query,
            tasks=[task],
            overall_success_criteria="answer the user query",
        )

    def run(
        self,
        query: str,
    ) -> ContextSchema:
        """
        Main Agent Execution Loop.

        Flow:

        Generate Plan
        ↓
        Initialize ExecutionContext
        ↓
        Execute Current Task
        ↓
        Update ExecutionContext
        ↓
        Advance()
        ↓
        Repeat
        """

        context = self.initialize_context(query=query)
        execution_context = self.initialize_execution_context(
            query=query,
            context=context,
        )

        while (
            self.termination_checker.should_continue(context)
            and not execution_context.is_finished()
        ):

            try:
                # ======================================
                # THINK
                # ======================================
                action = self.reasoning_engine.think(execution_context)

                context.add_action(action)

                # ======================================
                # VALIDATE
                # ======================================
                self.action_validator.validate(action)

                # ======================================
                # ACT
                # ======================================
                observation = self.tool_executor.execute(action)

                context.add_observation(observation)

                # ======================================
                # EVALUATE
                # ======================================
                evaluation = None
                if self.evaluator is not None:
                    evaluation = self.evaluator.evaluate(
                        query=context.query,
                        action=action,
                        observation=observation,
                        context=context,
                    )

                    context.add_evaluation(evaluation)

                # ======================================
                # REFLECT
                # ======================================
                if (
                    evaluation is not None
                    and evaluation.should_reflect
                    and self.reflection_engine is not None
                ):
                    reflection = self.reflection_engine.reflect(
                        query=context.query,
                        action=action,
                        observation=observation,
                        evaluation=evaluation,
                        context=context,
                    )

                    context.add_reflection(reflection)

                # ======================================
                # TERMINATION
                # ======================================
                if (
                    action.tool_name.lower() == "finish"
                    and (evaluation is None or evaluation.success)
                ):
                    context.mark_done()
                    execution_context.mark_completed()
                    break

                if observation.success:
                    execution_context.complete_current_task()
                else:
                    execution_context.fail_current_task()

                execution_context.add_runtime_event(
                    {
                        "action": action.to_dict(),
                        "observation": observation.to_dict(),
                    }
                )

                execution_context.advance()

                # ======================================
                # UPDATE ITERATION
                # ======================================
                context.increment_iteration()
                execution_context.increment_iteration()

            except Exception as e:
                execution_context.mark_failed()
                execution_context.add_runtime_event({"error": str(e)})
                print(f"[AGENT LOOP ERROR]: {e}")
                break

        return context

    def print_step(
        self,
        action,
        observation,
        evaluation,
        reflection=None,
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