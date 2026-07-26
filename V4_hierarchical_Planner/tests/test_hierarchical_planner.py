import pytest
import os
import sys

# Resolve project root so imports work
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from schemas.goal_node_schema import GoalNode
from schemas.goal_tree_schema import GoalTree
from planner.common.planner_constraints import PlannerConstraints
from planner.hierarchical.decision_policy import DecisionPolicy
from planner.hierarchical.recursive_decomposer import RecursiveDecomposer
from planner.hierarchical.traversal_policy import TraversalPolicy
from planner.hierarchical.RecursivePlannerPrompt import RecursivePlannerPrompt
from planner.hierarchical.recursive_planner_parser import RecursivePlannerParser
from planner.hierarchical.hierarchical_planner import HierarchicalPlanner


class DummyLLM:
    def __init__(self):
        self.generate_calls = 0

    def generate(self, prompt: str) -> str:
        self.generate_calls += 1
        if "YES or NO" in prompt or "Should the node" in prompt:
            # Use single quotes to match exact node names
            if "'root'" in prompt or "'child1'" in prompt:
                return "YES"
            return "NO"

        # Match the actual prompt format from RecursivePlannerPrompt
        # which contains "Goal:\n{goal_node.goal}"
        if "Goal:\nroot" in prompt:
            return """
            {
              "decompose": true,
              "children": [
                {
                  "goal": "child1",
                  "priority": 1,
                  "completion_criteria": "done child1"
                },
                {
                  "goal": "child2",
                  "priority": 2,
                  "completion_criteria": "done child2"
                }
              ]
            }
            """
        elif "Goal:\nchild1" in prompt:
            return """
            {
              "decompose": true,
              "children": [
                {
                  "goal": "grandchild1",
                  "priority": 1,
                  "completion_criteria": "done grandchild1"
                }
              ]
            }
            """
        return '{"decompose": false}'


def test_goal_node():
    node = GoalNode(goalnode="test", goal_id="123", goal="detailed goal")
    assert node.goalnode == "test"
    assert node.goal_id == "123"
    assert node.goal == "detailed goal"
    assert node.status == "pending"
    assert node.is_leaf() is True
    assert node.is_root() is True

    node.add_child("child1")
    assert node.is_leaf() is False
    assert "child1" in node.children

    node.remove_child("child1")
    assert node.is_leaf() is True


def test_goal_tree():
    tree = GoalTree()
    root = GoalNode(goalnode="root", goal_id="root", goal="root goal")
    child = GoalNode(goalnode="child", goal_id="child", goal="child goal")

    tree.add_node(root, root.goal_id)
    tree.add_node(child, child.goal_id)
    tree.set_root(root.goal_id)

    assert tree.root_id == "root"
    assert tree.contains("child") is True

    tree.connect("root", "child")
    assert child.parent == "root"
    assert "child" in root.children

    assert tree.get_parent("child") == root
    assert tree.get_children("root") == [child]

    assert tree.validate_tree() is True

    # Test cycle detection
    root.add_child("root")
    assert tree.validate_tree() is False


def test_traversal_policy():
    tree = GoalTree()
    n1 = GoalNode(goalnode="n1", goal_id="n1", goal="g1", depth=1, priority=10, status="pending")
    n2 = GoalNode(goalnode="n2", goal_id="n2", goal="g2", depth=2, priority=5, status="pending")
    n3 = GoalNode(goalnode="n3", goal_id="n3", goal="g3", depth=0, priority=20, status="pending")

    tree.add_node(n1, "n1")
    tree.add_node(n2, "n2")
    tree.add_node(n3, "n3")

    policy_dfs = TraversalPolicy(strategy="dfs")
    policy_bfs = TraversalPolicy(strategy="bfs")
    policy_priority = TraversalPolicy(strategy="priority")

    # DFS: Deepest nodes first (n2, n1, n3)
    dfs_nodes = policy_dfs.get_next_nodes(tree)
    assert dfs_nodes[0].goal_id == "n2"

    # BFS: Shallowest nodes first (n3, n1, n2)
    bfs_nodes = policy_bfs.get_next_nodes(tree)
    assert bfs_nodes[0].goal_id == "n3"

    # Priority: Higher priority first (n3, n1, n2)
    pri_nodes = policy_priority.get_next_nodes(tree)
    assert pri_nodes[0].goal_id == "n3"
    assert pri_nodes[1].goal_id == "n1"


def test_decision_policy_constraints():
    constraints = PlannerConstraints(max_depth=2, max_nodes=5)
    tree = GoalTree()
    
    # Root node (depth 0)
    root = GoalNode(goalnode="root", goal_id="root", goal="g", depth=0, status="pending")
    tree.add_node(root, "root")
    
    policy = DecisionPolicy(
        GoalNode=root,
        GoalTree=tree,
        PlannerConsraints=constraints,
        PlanningContext=None,
        Memory=None,
        PlanningState={"depth": 0, "nodes": 1}
    )
    res = policy.should_decompose()
    assert res["decompose"] is True

    # Create node at depth 3 (exceeding max depth 2)
    deep_node = GoalNode(goalnode="deep", goal_id="deep", goal="g", depth=3, status="pending")
    tree.add_node(deep_node, "deep")
    
    policy_fail = DecisionPolicy(
        GoalNode=deep_node,
        GoalTree=tree,
        PlannerConsraints=constraints,
        PlanningContext=None,
        Memory=None,
        PlanningState={"depth": 3, "nodes": 2}
    )
    res_fail = policy_fail.should_decompose()
    assert res_fail["decompose"] is False
    assert "Constraint violation detected" in res_fail["reasoning"]


def test_hierarchical_planner_end_to_end():
    llm = DummyLLM()
    prompt_builder = RecursivePlannerPrompt()
    parser_impl = RecursivePlannerParser()
    constraints = PlannerConstraints(max_depth=4, max_nodes=20)
    
    decomposer = RecursiveDecomposer(
        llm_client=llm,
        planner_prompt=prompt_builder,
        planner_parser=parser_impl
    )
    
    traversal_policy = TraversalPolicy(strategy="dfs")
    
    decision_policy_template = DecisionPolicy(
        GoalNode=None,
        GoalTree=None,
        PlannerConsraints=constraints,
        PlanningContext=None,
        Memory=None,
        PlanningState=None
    )

    planner = HierarchicalPlanner(
        decisionpolicy=decision_policy_template,
        recursivedecomposer=decomposer,
        plannerconstraints=constraints,
        traversalpolicy=traversal_policy
    )

    goaltree = planner.plan("root")
    assert goaltree.validate_tree() is True
    
    # Total nodes expected: root, child1, child2, grandchild1
    assert len(goaltree.nodes) == 4
    
    root_node = goaltree.get_node(goaltree.root_id)
    assert root_node.status == "decomposed"
    assert len(root_node.children) == 2
    
    # grandchild1 should be atomic
    gc = [node for node in goaltree.nodes.values() if node.goal == "grandchild1"][0]
    assert gc.status == "atomic"
