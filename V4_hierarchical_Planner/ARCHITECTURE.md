# Architecture — Hierarchical Planner Agent V4

This document details the internal architecture, data flow, and design decisions of the Hierarchical Planner Agent.

---

## System Overview

The agent operates in two distinct phases:

1. **Planning Phase** — Decomposes a user query into a hierarchical `GoalTree`
2. **Execution Phase** — Flattens the tree into sequential tasks and runs the agent loop

```
┌──────────────────────────────────────────────────────────────┐
│                     PLANNING PHASE                          │
│                                                              │
│  User Query ──► HierarchicalPlanner ──► GoalTree            │
│                       │                                      │
│            ┌──────────┼──────────┐                           │
│            ▼          ▼          ▼                           │
│      DecisionPolicy  Decomposer  TraversalPolicy            │
│      (Should I       (LLM call   (Which node                │
│       decompose?)     + parse)    to expand next?)           │
│                                                              │
│  Constraints: PlannerConstraints (depth, nodes, budget)      │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                  BRIDGE (GoalTree → PlanSchema)              │
│                                                              │
│  GoalTree.leaf_nodes() ──► TaskSchema[] ──► PlanSchema       │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    EXECUTION PHASE                           │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  AgentLoop                                          │     │
│  │                                                     │     │
│  │  1. ReasoningEngine.think()    ──► ActionSchema      │     │
│  │  2. ActionValidator.validate() ──► Pass/Fail         │     │
│  │  3. ToolExecutor.execute()     ──► ObservationSchema │     │
│  │  4. Evaluator.evaluate()       ──► EvaluationSchema  │     │
│  │  5. ReflectionEngine.reflect() ──► ReflectionSchema  │     │
│  │  6. ExecutionContext.advance() ──► Next Task         │     │
│  │                                                     │     │
│  └─────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

---

## Component Interaction Matrix

| Component | Depends On | Produces | Never Does |
|-----------|-----------|----------|------------|
| `HierarchicalPlanner` | DecisionPolicy, RecursiveDecomposer | GoalTree | Decomposition, execution |
| `DecisionPolicy` | PlannerConstraints, GoalNode state | Decision dict `{decompose: bool}` | Tree modification |
| `RecursiveDecomposer` | LLMClient, PlannerPrompt, PlannerParser | `List[GoalNode]` | Recursion, tree mutation |
| `RecursivePlannerPrompt` | GoalNode, ContextSchema | Prompt string | LLM calls, parsing |
| `RecursivePlannerParser` | Raw LLM response string | Validated Python dict | GoalNode creation |
| `TraversalPolicy` | GoalTree | Sorted node list | Tree modification |
| `PlannerConstraints` | Configuration parameters | Validation results | Decisions |
| `GoalTree` | GoalNode instances | Tree queries | Planning, decomposition |
| `GoalNode` | None (data only) | Serialization | Any logic |
| `AgentLoop` | All execution components | ContextSchema | Planning |

---

## Data Flow: Planning Phase

```
1. User Query: "What is (15 * 3) + (42 / 6)?"
   │
2. HierarchicalPlanner._create_root()
   │  Creates GoalNode(id="root", goal="What is (15*3)+(42/6)?")
   │
3. HierarchicalPlanner._expand(root)
   │
   ├─ DecisionPolicy.should_decompose(root)
   │  └─ Returns {decompose: True}
   │
   ├─ RecursiveDecomposer.decompose(root)
   │  ├─ RecursivePlannerPrompt.build(root) → prompt string
   │  ├─ LLMClient.generate(prompt) → raw JSON
   │  ├─ RecursivePlannerParser.parse(raw) → validated dict
   │  └─ Returns [GoalNode("root.1", "Calculate 15*3"),
   │              GoalNode("root.2", "Calculate 42/6"),
   │              GoalNode("root.3", "Add results")]
   │
   ├─ For each child: GoalTree.add_node() + GoalTree.connect()
   │
   ├─ _expand(child1) → DecisionPolicy says NO → mark_atomic()
   ├─ _expand(child2) → DecisionPolicy says NO → mark_atomic()
   └─ _expand(child3) → DecisionPolicy says NO → mark_atomic()
   │
   └─ root.status = DECOMPOSED
   │
4. HierarchicalPlanner._validate(tree) → passes
   │
5. Returns GoalTree
```

---

## Data Flow: Bridge (GoalTree → PlanSchema)

```
GoalTree:
  root [DECOMPOSED]
  ├── root.1 "Calculate 15*3" [ATOMIC]
  ├── root.2 "Calculate 42/6" [ATOMIC]
  └── root.3 "Add results"    [ATOMIC]

          │
          ▼ DFS traversal of leaf/atomic nodes
          │
PlanSchema:
  Task 1: "Calculate 15*3"    [PENDING]
  Task 2: "Calculate 42/6"    [PENDING]
  Task 3: "Add results"       [PENDING]
```

---

## GoalNode State Machine

```
         ┌──────────┐
         │ PENDING   │ ← initial state
         └─────┬─────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌────────────┐  ┌────────────┐
│ DECOMPOSED │  │  ATOMIC    │ ← leaf / no children
└────────────┘  └──────┬─────┘
                       │
                       ▼
                ┌────────────┐
                │  RUNNING   │
                └──────┬─────┘
                       │
               ┌───────┴───────┐
               ▼               ▼
        ┌────────────┐  ┌────────────┐
        │ COMPLETED  │  │  FAILED    │
        └────────────┘  └────────────┘
```

---

## Constraint Enforcement

The `PlannerConstraints` object is checked at two points:

1. **Before decomposition** — `DecisionPolicy` checks depth, node count, and budget
2. **During validation** — `PlannerConstraints.validate()` checks all limits

| Constraint | Default | Checked By |
|-----------|---------|------------|
| `max_depth` | 8 | DecisionPolicy |
| `max_nodes` | 200 | DecisionPolicy |
| `max_recursive_calls` | 200 | PlannerConstraints |
| `max_llm_calls` | 100 | PlannerConstraints |
| `max_planning_time` | 60s | PlannerConstraints |
| `max_tokens` | 50,000 | PlannerConstraints |
| `min_confidence` | 0.60 | PlannerConstraints |
| `max_branching` | 10 | PlannerConstraints |

---

## Key Design Decisions

### 1. GoalTree → PlanSchema Bridge (Not Direct Execution)

**Decision**: Convert GoalTree to flat PlanSchema rather than executing the tree directly.

**Rationale**: The execution pipeline (`ExecutionContext`, `AgentLoop`, `ReasoningEngine`) is proven stable. Rewriting execution for tree-based plans would introduce risk without proportional benefit. The bridge pattern preserves hierarchical planning benefits while reusing the battle-tested execution path.

### 2. Single Responsibility Per Class

Every class in the planner has exactly one job:
- `RecursivePlannerPrompt` builds prompts — it never calls the LLM
- `RecursivePlannerParser` parses JSON — it never creates GoalNodes
- `RecursiveDecomposer` coordinates one decomposition step — it never recurses
- `HierarchicalPlanner` orchestrates — it never implements policy or decomposition

### 3. Stateless Decomposer, Stateful Policy

The `RecursiveDecomposer` is stateless — every call is independent. The `DecisionPolicy` maintains references to the current node and tree state, allowing it to make context-aware decisions across the planning session.

### 4. Try/Except Import Fallback

All modules use:
```python
try:
    from V4_hierarchical_Planner.schemas.goal_node_schema import GoalNode
except ImportError:
    from schemas.goal_node_schema import GoalNode
```

This allows the code to work both when run from the project root and when imported as a sub-package.
