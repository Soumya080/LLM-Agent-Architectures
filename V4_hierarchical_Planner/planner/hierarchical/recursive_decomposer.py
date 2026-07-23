from __future__ import annotations

from typing import List, Optional

try:
    from V4_hierarchical_Planner.schemas.goal_node_schema import GoalNode
    from V4_hierarchical_Planner.schemas.context_schema import ContextSchema
    from V4_hierarchical_Planner.planner.common.planner_constraints import PlannerConstraints
except ImportError:
    from schemas.goal_node_schema import GoalNode
    from schemas.context_schema import ContextSchema
    from planner.common.planner_constraints import PlannerConstraints


class RecursiveDecomposer:
    """
    Coordinates ONE decomposition step.

    Flow:
        GoalNode
            ↓
        Planner Prompt
            ↓
        LLM
            ↓
        Planner Parser
            ↓
        GoalNode Objects
            ↓
        Return List[GoalNode]

    NOTE:
        This class DOES NOT

        • recurse
        • modify GoalTree
        • execute planning
        • decide whether decomposition should happen

        Those responsibilities belong elsewhere.
    """

    def __init__(
        self,
        llm_client,
        planner_prompt,
        planner_parser,
        planner_constraints: Optional[PlannerConstraints] = None,
    ):
        self.llm_client = llm_client
        self.planner_prompt = planner_prompt
        self.planner_parser = planner_parser
        self.planner_constraints = planner_constraints

    # ==========================================================
    # Public API
    # ==========================================================

    def decompose(
        self,
        goal_node: GoalNode,
        goal_tree=None,
        planning_context: Optional[ContextSchema] = None,
        context: Optional[ContextSchema] = None,
        memory: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
    ) -> List[GoalNode]:
        """
        Performs ONE decomposition step.

        Returns
        -------
        List[GoalNode]
        """

        if context is None and planning_context is not None:
            context = planning_context

        prompt = self._build_prompt(
            goal_node=goal_node,
            context=context,
            memory=memory,
            capabilities=capabilities,
        )

        response = self._call_llm(prompt)

        planner_output = self._parse_response(response)

        if not planner_output.get("decompose", False):
            return []

        return self._build_goal_nodes(
            planner_output=planner_output,
            parent_node=goal_node,
        )

    # ==========================================================
    # Internal Pipeline
    # ==========================================================

    def _build_prompt(
        self,
        goal_node: GoalNode,
        context: Optional[ContextSchema],
        memory: Optional[str],
        capabilities: Optional[List[str]],
    ) -> str:

        return self.planner_prompt.build(
            goal_node=goal_node,
            context=context,
            memory=memory,
            capabilities=capabilities,
            constraints=self.planner_constraints,
        )

    def _call_llm(self, prompt: str) -> str:
        """
        Send prompt to LLM.
        """

        return self.llm_client.generate(prompt)

    def _parse_response(self, response: str):
        """
        Parse LLM response.
        """

        return self.planner_parser.parse(response)

    def _build_goal_nodes(
        self,
        planner_output,
        parent_node: GoalNode,
    ) -> List[GoalNode]:
        """
        Convert parsed planner output into GoalNodes.
        """

        goal_nodes: List[GoalNode] = []

        children = planner_output.get("children", [])

        for index, child in enumerate(children, start=1):

            node = GoalNode(
                goal_id=f"{parent_node.goal_id}.{index}",
                goal=child["goal"],
                parent=parent_node.goal_id,
                depth=parent_node.depth + 1,
                priority=child.get("priority", 0),
                completion_criteria=child.get(
                    "completion_criteria",
                    "",
                ),
                confidence=child.get(
                    "confidence",
                    1.0,
                ),
                metadata=child.get(
                    "metadata",
                    {},
                ),
                status="PENDING",
            )

            goal_nodes.append(node)

        return goal_nodes

    def __repr__(self) -> str:
        return (
            f"RecursiveDecomposer("
            f"llm={type(self.llm_client).__name__})"
        )

    __str__ = __repr__