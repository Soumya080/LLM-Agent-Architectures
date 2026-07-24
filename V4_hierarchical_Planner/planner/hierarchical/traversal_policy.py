from typing import List, Optional

try:
    from V4_hierarchical_Planner.schemas.goal_node_schema import GoalNode
    from V4_hierarchical_Planner.schemas.goal_tree_schema import GoalTree
except ImportError:
    from schemas.goal_node_schema import GoalNode
    from schemas.goal_tree_schema import GoalTree

class TraversalPolicy:
    """
    TraversalPolicy determines the order in which GoalNodes in the GoalTree
    are traversed and selected for expansion.
    """
    def __init__(self, strategy: str = "dfs"):
        self.strategy = strategy.lower()
        if self.strategy not in ("dfs", "bfs", "priority"):
            raise ValueError(f"Unknown traversal strategy: {strategy}")

    def get_next_nodes(self, goal_tree: GoalTree) -> List[GoalNode]:
        """
        Returns a list of GoalNodes that are currently 'pending' expansion,
        ordered according to the traversal strategy.
        """
        # Find all pending nodes
        pending_nodes = goal_tree.get_pending_nodes()
        if not pending_nodes:
            return []

        return self.get_next_nodes_from_list(pending_nodes)

    def get_next_nodes_from_list(self, nodes: List[GoalNode]) -> List[GoalNode]:
        """
        Sorts and returns a list of GoalNodes according to the traversal strategy.
        """
        nodes_copy = list(nodes)
        if self.strategy == "bfs":
            # Sort by depth ascending (closer to root first)
            nodes_copy.sort(key=lambda n: (n.depth, n.goal_id))
        elif self.strategy == "dfs":
            # Sort by depth descending (deepest first, i.e., child nodes first)
            nodes_copy.sort(key=lambda n: (-n.depth, n.goal_id))
        elif self.strategy == "priority":
            # Sort by priority descending (higher priority first)
            nodes_copy.sort(key=lambda n: (-n.priority, n.goal_id))
        return nodes_copy

    def __repr__(self) -> str:
        return f"TraversalPolicy(strategy='{self.strategy}')"
