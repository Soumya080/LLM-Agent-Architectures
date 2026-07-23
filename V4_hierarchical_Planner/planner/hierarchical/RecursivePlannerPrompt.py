from typing import List, Optional

try:
    from V4_hierarchical_Planner.schemas.goal_node_schema import GoalNode
    from V4_hierarchical_Planner.schemas.context_schema import ContextSchema
    from V4_hierarchical_Planner.planner.common.planner_constraints import PlannerConstraints
except ImportError:
    from schemas.goal_node_schema import GoalNode
    from schemas.context_schema import ContextSchema
    from planner.common.planner_constraints import PlannerConstraints


class RecursivePlannerPrompt:
    """
    Builds the prompt used by the RecursiveDecomposer.

    Responsibility:
        Build ONE decomposition prompt.

    It DOES NOT:
        - call the LLM
        - parse responses
        - create GoalNodes
        - modify the GoalTree
    """

    def build(
        self,
        goal_node: GoalNode,
        context: Optional[ContextSchema] = None,
        memory: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        constraints: Optional[PlannerConstraints] = None,
    ) -> str:

        capability_text = "\n".join(
            f"- {cap}"
            for cap in (capabilities or [])
        )

        prompt = f"""
You are an expert Hierarchical Planning Engine.

Your ONLY responsibility is to decompose ONE goal into smaller child goals.

You are NOT an executor.

You are NOT an agent.

You are NOT a tool user.

You ONLY perform hierarchical task decomposition.

=====================================================
CURRENT GOAL
=====================================================

Goal:
{goal_node.goal}

Depth:
{goal_node.depth}

Completion Criteria:
{goal_node.completion_criteria}

Priority:
{goal_node.priority}

=====================================================
CURRENT CONTEXT
=====================================================

{context if context else "No context available."}

=====================================================
RELEVANT MEMORY
=====================================================

{memory if memory else "No relevant memory."}

=====================================================
AVAILABLE CAPABILITIES
=====================================================

{capability_text if capability_text else "No capabilities provided."}

=====================================================
PLANNER CONSTRAINTS
=====================================================

{constraints.to_dict() if constraints else "No planner constraints provided."}

=====================================================
YOUR RESPONSIBILITIES
=====================================================

1. Determine whether this goal should be decomposed.

2. If this goal is already atomic,
return an empty child list.

3. Otherwise,
generate the MINIMUM number of meaningful child goals.

4. Every child goal must contribute directly toward
completing the parent goal.

5. Every child goal should represent ONE logical objective.

6. Child goals should remain high-level enough that
they can themselves be decomposed later if necessary.

7. Do NOT generate implementation steps.

8. Do NOT mention programming languages.

9. Do NOT mention tools.

10. Think in goals, not actions.

=====================================================
DECOMPOSITION RULES
=====================================================

✔ Preserve parent intent

✔ Avoid redundant children

✔ Minimize branching factor

✔ Maintain logical ordering

✔ Produce independent goals whenever possible

✔ Do NOT over-decompose

✔ Stop before implementation-level details

=====================================================
OUTPUT FORMAT
=====================================================

Return ONLY valid JSON.

{{
    "decompose": true,

    "reasoning": "...",

    "children":
    [
        {{
            "goal": "...",
            "priority": 1,
            "completion_criteria": "...",
            "confidence": 0.95,
            "metadata": {{}}
        }}
    ]
}}

If the goal is already atomic, return:

{{
    "decompose": false,
    "reasoning": "Goal is atomic.",
    "children": []
}}

Return ONLY JSON.
"""

        return prompt

    def __repr__(self):
        return "RecursivePlannerPrompt()"

    __str__ = __repr__