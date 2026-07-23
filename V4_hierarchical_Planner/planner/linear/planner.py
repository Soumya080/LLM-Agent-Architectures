from typing import Optional

from V4_hierarchical_Planner.planner.linear.planner_prompt import PlannerPrompt
from V4_hierarchical_Planner.planner.linear.planner_parser import PlannerParser

from agent.llm_client import LLMClient

from schemas.context_schema import ContextSchema
from schemas.plan_schema import PlanSchema


class Planner:
    """
    Planner is responsible for generating an execution plan
    from the user's query.

    Responsibilities:
        - Build planning prompt
        - Call the LLM
        - Parse LLM response
        - Return PlanSchema

    Planner NEVER:
        - Executes tools
        - Performs reasoning
        - Evaluates actions
        - Reflects
    """

    def __init__(
        self,
        llm_client: LLMClient,
        planner_prompt: PlannerPrompt,
        planner_parser: PlannerParser,
    ):
        self.llm_client = llm_client
        self.planner_prompt = planner_prompt
        self.planner_parser = planner_parser

    def plan(
        self,
        query: str,
        context: Optional[ContextSchema] = None,
        relevant_memory: Optional[str] = None,
    ) -> PlanSchema:
        """
        Generate a structured execution plan.

        Args:
            query: User query
            context: Current execution context
            relevant_memory: Retrieved memory relevant
                             to planning

        Returns:
            PlanSchema
        """

        # Build planning prompt
        prompt = self.planner_prompt.build_prompt(
            query=query,
            context=context,
            relevant_memory=relevant_memory,
        )

        # LLM generates plan
        response = self.llm_client.generate(
            prompt
        )

        # Parse into PlanSchema
        plan = self.planner_parser.parse(
            response
        )

        return plan

    def __repr__(self):
        return (
            f"Planner("
            f"llm='{self.llm_client.model_name}', "
            f"prompt_builder={self.planner_prompt.__class__.__name__}, "
            f"parser={self.planner_parser.__class__.__name__}"
            f")"
        )

    __str__ = __repr__