<p align="center">
  <h1 align="center">🧠 Hierarchical Planner Agent</h1>
  <p align="center">
    <em>A research-grade AI agent framework that decomposes complex goals into executable sub-tasks using recursive hierarchical planning.</em>
  </p>
  <p align="center">
    <a href="#architecture">Architecture</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#components">Components</a> •
    <a href="#evaluation">Evaluation</a> •
    <a href="#tests">Tests</a>
  </p>
</p>

---

## ✨ Highlights

| Feature | Description |
|---------|-------------|
| 🌳 **Goal Tree Decomposition** | Recursively breaks high-level goals into a structured tree of sub-goals using LLM-powered decomposition |
| 🧭 **Pluggable Traversal** | Supports DFS, BFS, and Priority-based tree traversal strategies |
| 🛡️ **Constraint-Aware Planning** | Enforces depth, node count, token budget, and time limits during planning |
| 🔄 **Reflexion Loop** | Built-in self-evaluation and reflection for error correction and learning |
| 📊 **Benchmark Evaluation** | Automated evaluation framework with 5 benchmark categories and detailed reporting |
| 🔌 **Local LLM via Ollama** | Runs entirely on your machine — no API keys, no cloud dependency |

---

## 🏗️ Architecture

The system follows a **Plan → Reason → Act → Evaluate → Reflect** loop, with the planner generating a hierarchical goal tree that is flattened into executable tasks.

```
                    ┌─────────────────────┐
                    │   User Query        │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ HierarchicalPlanner  │
                    │  (Orchestrator)      │
                    └─────────┬───────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
        ┌────────────┐ ┌────────────┐ ┌────────────┐
        │ Decision   │ │ Recursive  │ │ Traversal  │
        │ Policy     │ │ Decomposer │ │ Policy     │
        │            │ │            │ │            │
        │ Should     │ │ LLM Call → │ │ DFS / BFS /│
        │ decompose? │ │ Parse →    │ │ Priority   │
        │            │ │ GoalNodes  │ │            │
        └────────────┘ └────────────┘ └────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │     GoalTree        │
                    │  (Planning Output)  │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  GoalTree → Plan    │
                    │  Schema Converter   │
                    └─────────┬───────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │        Agent Loop            │
               │                              │
               │  Think → Validate → Act →    │
               │  Evaluate → Reflect → Loop   │
               └──────────────────────────────┘
```

### Goal Tree Example

```
root: "Solve multi-step math problem"
├── child1: "Parse the problem statement"      [DECOMPOSED]
│   └── grandchild1: "Identify quantities"     [ATOMIC]
├── child2: "Set up equations"                 [ATOMIC]
└── child3: "Compute and verify answer"        [ATOMIC]
```

---

## 📁 Project Structure

```
V4_hierarchical_Planner/
│
├── agent/                          # Core agent loop & reasoning
│   ├── agent_loop.py               # Main execution loop (Plan→Reason→Act→Evaluate→Reflect)
│   ├── reasoning_engine.py         # LLM-powered reasoning & action selection
│   ├── llm_client.py               # Ollama HTTP client wrapper
│   ├── action_validator.py         # Validates proposed actions against tool registry
│   └── termination_checker.py      # Controls loop termination conditions
│
├── planner/                        # Hierarchical planning system
│   ├── hierarchical/               # ← The core innovation
│   │   ├── hierarchical_planner.py # Orchestrates recursive goal decomposition
│   │   ├── decision_policy.py      # Decides whether to decompose a goal node
│   │   ├── recursive_decomposer.py # Performs one decomposition step via LLM
│   │   ├── RecursivePlannerPrompt.py # Builds decomposition prompts
│   │   ├── recursive_planner_parser.py # Parses LLM JSON responses
│   │   └── traversal_policy.py     # DFS / BFS / Priority traversal strategies
│   ├── common/
│   │   └── planner_constraints.py  # Depth, node, token, and time budget limits
│   └── linear/                     # Legacy linear planner (V3 baseline)
│
├── schemas/                        # Data structures (no logic)
│   ├── goal_node_schema.py         # GoalNode — single node in the planning tree
│   ├── goal_tree_schema.py         # GoalTree — manages the full tree structure
│   ├── plan_schema.py              # PlanSchema — flat execution plan
│   ├── task_schema.py              # TaskSchema — atomic executable task
│   ├── context_schema.py           # ContextSchema — agent conversation state
│   ├── execution_context.py        # ExecutionContext — runtime tracking
│   ├── action_schema.py            # ActionSchema — tool invocation record
│   └── observation_schema.py       # ObservationSchema — tool result record
│
├── reflexion/                      # Self-improvement via reflection
│   ├── evaluator.py                # Evaluates action-observation pairs
│   ├── reflection_engine.py        # Generates reflective insights from failures
│   ├── evaluation_schema.py        # Evaluation data structure
│   └── reflection_schema.py        # Reflection data structure
│
├── tools/                          # Agent tool implementations
│   ├── base_tool.py                # Abstract tool interface
│   ├── calculator_tool.py          # Mathematical expression evaluator
│   ├── search_tool.py              # Knowledge search tool
│   ├── finish_tool.py              # Terminal action — returns final answer
│   ├── tool_registry.py            # Tool discovery & registration
│   └── tool_executor.py            # Safe tool execution wrapper
│
├── evaluation/                     # Benchmark evaluation framework
│   ├── run_evaluation.py           # Full evaluation pipeline
│   ├── run_academic_eval.py        # Academic benchmark runner
│   ├── academic_loader.py          # Dataset loader for academic benchmarks
│   ├── answer_evaluator.py         # Answer correctness evaluator
│   └── benchmarks/                 # Benchmark datasets (JSON)
│       ├── arithmetic_tasks.json
│       ├── search_tasks.json
│       ├── multistep_tasks.json
│       ├── failure_cases.json
│       └── ambiguous_queries.json
│
├── tests/                          # Unit & integration tests
│   ├── test_hierarchical_planner.py # End-to-end planner tests
│   ├── test_agent.py               # Agent loop tests
│   ├── test_reasoning.py           # Reasoning engine tests
│   └── test_parser_fallback.py     # Parser robustness tests
│
├── main.py                         # Interactive terminal entry point
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
└── README.md                       # ← You are here
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.ai)** installed and running locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/hierarchical-planner-agent.git
cd hierarchical-planner-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama and pull a model

```bash
ollama serve
ollama pull qwen2.5-coder:7b
```

### 4. Run the agent

```bash
python main.py
```

### 5. Run the evaluation benchmark

```bash
python evaluation/run_evaluation.py --category arithmetic --max-iterations 5
```

---

## 🧩 Components

### Hierarchical Planner

The `HierarchicalPlanner` is the core orchestrator. It:

1. Creates a **root GoalNode** from the user's query
2. Builds a **GoalTree** data structure
3. Recursively asks the **DecisionPolicy** whether each node should be decomposed
4. Uses the **RecursiveDecomposer** to split nodes into children via LLM calls
5. Validates the final tree structure (no cycles, consistent parent-child links)
6. Returns the complete GoalTree

```python
from planner.hierarchical import HierarchicalPlanner, DecisionPolicy, RecursiveDecomposer
from planner.common import PlannerConstraints

planner = HierarchicalPlanner(
    decision_policy=decision_policy,
    recursive_decomposer=decomposer,
    planner_constraints=PlannerConstraints(max_depth=4, max_nodes=20),
)

goal_tree = planner.plan("Solve: What is (15 * 3) + (42 / 6)?")
```

### Decision Policy

Evaluates four dimensions before deciding to decompose:
- **Constraints** — depth/node limits not exceeded
- **Atomicity** — goal isn't already atomic
- **Budget** — token/time budget available
- **Complexity** — estimated task complexity score

### GoalTree → PlanSchema Bridge

The `AgentLoop` automatically converts the hierarchical `GoalTree` into a flat `PlanSchema` (ordered list of `TaskSchema` objects) for execution. This means:
- The **planner** thinks hierarchically
- The **executor** operates sequentially
- No changes needed to the proven execution pipeline

### Reflexion Engine

After each action, the system:
1. **Evaluates** the action-observation pair for correctness
2. **Reflects** on failures to generate improvement insights
3. Feeds reflections back into future reasoning

---

## 📊 Evaluation

Run the full benchmark suite across 5 categories:

```bash
# All categories
python evaluation/run_evaluation.py

# Single category
python evaluation/run_evaluation.py --category arithmetic

# Custom model
python evaluation/run_evaluation.py --model llama3:8b

# Adjust iterations
python evaluation/run_evaluation.py --max-iterations 10
```

### Benchmark Categories

| Category | Description | Cases |
|----------|-------------|-------|
| `arithmetic` | Single and multi-step math problems | 10 |
| `search` | Knowledge retrieval queries | 10 |
| `multistep` | Multi-tool reasoning chains | 10 |
| `failure` | Edge cases and error handling | 10 |
| `ambiguous` | Underspecified or ambiguous queries | 10 |

### Sample Report Output

```
============================================================
  PLANNER AGENT V4 — EVALUATION REPORT
============================================================
  Model     : qwen2.5-coder:7b
  Total     : 50 cases
============================================================

  ┌─ ARITHMETIC (10 cases)
  │  Tool Accuracy    :  80.0%  (8/10)
  │  Task Success     :  90.0%  (9/10)
  │  Finish Accuracy  :  90.0%  (9/10)
  │  Average Steps    :   2.3
  └─────────────────────────────────────────────────────
```

---

## 🧪 Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run hierarchical planner tests only
python -m pytest tests/test_hierarchical_planner.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing
```

### Test Coverage

| Test File | What It Tests |
|-----------|---------------|
| `test_hierarchical_planner.py` | GoalNode creation, GoalTree operations, traversal strategies, constraint enforcement, end-to-end planning |
| `test_agent.py` | Agent loop initialization and execution |
| `test_reasoning.py` | Reasoning engine action selection |
| `test_parser_fallback.py` | JSON parser robustness with malformed LLM output |

---

## 🔧 Configuration

All configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `MAX_ITERATIONS` | `10` | Max agent loop iterations |
| `CONFIDENCE_THRESHOLD` | `0.6` | Evaluator confidence threshold |
| `DEBUG` | `false` | Enable verbose logging |

---

## 🧬 Design Principles

1. **Separation of Concerns** — Each class has exactly ONE responsibility
2. **No God Objects** — Orchestrators delegate, never implement logic
3. **Schema = Data Only** — Schema classes contain zero execution logic
4. **Pluggable Components** — Swap policies, decomposers, or LLM backends without touching core logic
5. **Fail-Safe Imports** — All imports use try/except fallback for flexible module resolution

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with 🔬 as part of <a href="https://github.com/YOUR_USERNAME">LLM Research Lab</a></sub>
</p>