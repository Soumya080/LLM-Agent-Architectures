from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from schemas.goal_node_schema import GoalNode, GoalStatus


@dataclass
class GoalTree:
    """
    Stores and manages the hierarchical planning structure.

    Responsibilities
    ----------------
    - Store GoalNodes
    - Manage parent-child relationships
    - Query tree structure
    - Validate structural integrity

    Non Responsibilities
    --------------------
    - Planning
    - Decomposition
    - Traversal Algorithms
    - Execution
    """

    root_id: Optional[str] = None
    nodes: Dict[str, GoalNode] = field(default_factory=dict)

    # --------------------------------------------------
    # Basic Operations
    # --------------------------------------------------

    def add_node(self, node: GoalNode, goal_id: Optional[str] = None) -> str:
        """
        Insert a GoalNode into the tree.
        """

        node_id = goal_id or node.goal_id

        if node_id in self.nodes:
            raise ValueError(
                f"GoalNode '{node_id}' already exists."
            )

        self.nodes[node_id] = node
        return node_id

    def remove_node(self, goal_id: str) -> GoalNode:
        """
        Remove a node from the tree.

        Only leaf nodes can be removed.
        """

        node = self.get_node(goal_id)

        if node is None:
            raise ValueError(
                f"GoalNode '{goal_id}' does not exist."
            )

        if node.children_ids:
            raise ValueError(
                "Cannot remove a node that still has children."
            )

        if node.parent_id is not None:
            parent = self.get_node(node.parent_id)
            if parent is not None:
                parent.remove_child(goal_id)

        return self.nodes.pop(goal_id)

    def contains(self, goal_id: str) -> bool:
        return goal_id in self.nodes

    def size(self) -> int:
        return len(self.nodes)

    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    # --------------------------------------------------
    # Root
    # --------------------------------------------------

    def set_root(self, goal_id: str) -> None:

        if goal_id not in self.nodes:
            raise ValueError(
                f"Goal '{goal_id}' not found."
            )

        self.root_id = goal_id

    def get_root(self) -> Optional[GoalNode]:

        if self.root_id is None:
            return None

        return self.nodes.get(self.root_id)

    # --------------------------------------------------
    # Lookup
    # --------------------------------------------------

    def get_node(
        self,
        goal_id: str,
    ) -> Optional[GoalNode]:

        return self.nodes.get(goal_id)

    def get_parent(
        self,
        goal_id: str,
    ) -> Optional[GoalNode]:

        node = self.get_node(goal_id)

        if node is None:
            return None

        if node.parent_id is None:
            return None

        return self.get_node(node.parent_id)

    def get_children(
        self,
        goal_id: str,
    ) -> List[GoalNode]:

        node = self.get_node(goal_id)

        if node is None:
            return []

        children = []

        for child_id in node.children_ids:

            child = self.get_node(child_id)

            if child is not None:
                children.append(child)

        return children

    # --------------------------------------------------
    # Relationships
    # --------------------------------------------------

    def connect(
        self,
        parent_id: str,
        child_id: str,
    ) -> None:

        if parent_id == child_id:
            raise ValueError(
                "A node cannot be its own parent."
            )

        parent = self.get_node(parent_id)
        child = self.get_node(child_id)

        if parent is None:
            raise ValueError(
                f"Parent '{parent_id}' not found."
            )

        if child is None:
            raise ValueError(
                f"Child '{child_id}' not found."
            )

        parent.add_child(child.goal_id)

        child.parent_id = parent.goal_id

        child.depth = parent.depth + 1

    def disconnect(
        self,
        parent_id: str,
        child_id: str,
    ) -> None:

        parent = self.get_node(parent_id)
        child = self.get_node(child_id)

        if parent is None or child is None:
            return

        parent.remove_child(child.goal_id)

        child.parent_id = None

    # --------------------------------------------------
    # Queries
    # --------------------------------------------------

    def get_leaf_nodes(self) -> List[GoalNode]:

        return [
            node
            for node in self.nodes.values()
            if node.is_leaf()
        ]

    def get_atomic_nodes(self) -> List[GoalNode]:

        return [
            node
            for node in self.nodes.values()
            if node.is_atomic()
        ]

    def get_pending_nodes(self) -> List[GoalNode]:

        return [
            node
            for node in self.nodes.values()
            if (
                (isinstance(node.status, GoalStatus) and node.status == GoalStatus.PENDING)
                or str(node.status).lower() == "pending"
            )
        ]

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validate_tree(self) -> bool:
        """
        Validate tree structure:
        1. Root must exist and be in self.nodes.
        2. Parent-child links must be consistent.
        3. No cycles must exist.
        """

        if self.root_id is None:
            return False

        if self.root_id not in self.nodes:
            return False

        for node in self.nodes.values():
            if node.parent_id is not None:
                if node.parent_id not in self.nodes:
                    return False
            for child_id in node.children_ids:
                if child_id not in self.nodes:
                    return False

        # DFS path cycle detection
        visited = set()
        path = set()

        def dfs(node_id):
            if node_id in path:
                return False
            if node_id in visited:
                return True

            node = self.get_node(node_id)
            if node is None:
                return False

            path.add(node_id)
            for child_id in node.children_ids:
                # Optional: assert child has node as parent
                # child_node = self.get_node(child_id)
                # if child_node and child_node.parent_id != node_id:
                #     return False
                if not dfs(child_id):
                    return False
            path.remove(node_id)
            visited.add(node_id)
            return True

        return dfs(self.root_id)

    # --------------------------------------------------
    # Serialization
    # --------------------------------------------------

    def to_dict(self):

        return {
            "root_id": self.root_id,
            "nodes": {
                node_id: node.to_dict()
                for node_id, node
                in self.nodes.items()
            },
        }

    @classmethod
    def from_dict(cls, data):

        tree = cls(
            root_id=data.get("root_id")
        )

        for node_id, node_data in data.get(
            "nodes",
            {},
        ).items():

            tree.nodes[node_id] = GoalNode.from_dict(
                node_data
            )

        return tree

    # --------------------------------------------------
    # Debugging
    # --------------------------------------------------

    def __repr__(self):

        return (
            f"GoalTree("
            f"root='{self.root_id}', "
            f"nodes={len(self.nodes)})"
        )

    __str__ = __repr__