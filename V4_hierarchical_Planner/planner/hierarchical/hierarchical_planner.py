from __future__ import annotations

from typing import Optional

try:
    from V4_hierarchical_Planner.schemas.goal_node_schema import GoalNode, GoalStatus
    from V4_hierarchical_Planner.schemas.goal_tree_schema import GoalTree
    from V4_hierarchical_Planner.planner.common.planner_constraints import PlannerConstraints
    from V4_hierarchical_Planner.planner.hierarchical.decision_policy import DecisionPolicy
    from V4_hierarchical_Planner.planner.hierarchical.recursive_decomposer import RecursiveDecomposer
except ImportError:
    from schemas.goal_node_schema import GoalNode, GoalStatus
    from schemas.goal_tree_schema import GoalTree
    from planner.common.planner_constraints import PlannerConstraints
    from planner.hierarchical.decision_policy import DecisionPolicy
    from planner.hierarchical.recursive_decomposer import RecursiveDecomposer


class HierarchicalPlanner:
    """
    Main Orchestrator of Hierarchical Planning.

    Responsibilities
    ----------------
    1. Create root goal
    2. Build GoalTree
    3. Expand recursively
    4. Validate tree
    5. Return GoalTree

    This class NEVER performs decomposition itself.
    It NEVER decides planning policy.
    It NEVER executes tasks.

    It only orchestrates.
    """

    def __init__(
        self,
        decision_policy: Optional[DecisionPolicy] = None,
        recursive_decomposer: Optional[RecursiveDecomposer] = None,
        planner_constraints: Optional[PlannerConstraints] = None,
        **kwargs,
    ):

        self.decision_policy = decision_policy or kwargs.get("decisionpolicy")
        self.recursive_decomposer = recursive_decomposer or kwargs.get("recursivedecomposer")
        self.constraints = planner_constraints or kwargs.get("plannerconstraints")
        self.traversal_policy = kwargs.get("traversalpolicy")

    # -------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------

    def plan(
        self,
        query: str,
        context=None,
        memory=None,
    ) -> GoalTree:

        tree = self._initialize()

        root = self._create_root(query)

        root_id = tree.add_node(root)

        tree.set_root(root_id)

        self._expand(
            node=root,
            tree=tree,
            context=context,
            memory=memory,
        )

        self._validate(tree)

        return tree

    # -------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------

    def _initialize(self) -> GoalTree:

        return GoalTree()

    def _create_root(
        self,
        query: str,
    ) -> GoalNode:

        return GoalNode(
            goal_id="root",
            goal=query,
            depth=0,
            priority=1,
            status="PENDING",
            parent=None,
        )

    # -------------------------------------------------------
    # CORE RECURSIVE ALGORITHM
    # -------------------------------------------------------

    def _expand(
        self,
        node: GoalNode,
        tree: GoalTree,
        context=None,
        memory=None,
    ) -> None:

        decision = self.decision_policy.should_decompose(
            goal_node=node,
            goal_tree=tree,
            planning_context=context,
            memory=memory,
        )

        if not decision["decompose"]:
            node.mark_atomic()
            return

        children = self.recursive_decomposer.decompose(
            goal_node=node,
            goal_tree=tree,
            planning_context=context,
            memory=memory,
        )

        if not children:
            node.mark_atomic()
            return

        for child in children:

            child_id = tree.add_node(child)

            tree.connect(
                parent_id=node.goal_id,
                child_id=child_id,
            )

            self._expand(
                node=child,
                tree=tree,
                context=context,
                memory=memory,
            )

        node.status = GoalStatus.DECOMPOSED

    # -------------------------------------------------------
    # VALIDATION
    # -------------------------------------------------------

    def _validate(
        self,
        tree: GoalTree,
    ) -> None:

        if not tree.validate_tree():
            raise RuntimeError(
                "GoalTree validation failed."
            )

    # -------------------------------------------------------

    def __repr__(self):

        return (
            "HierarchicalPlanner("
            f"policy={type(self.decision_policy).__name__}, "
            f"decomposer={type(self.recursive_decomposer).__name__}"
            ")"
        )