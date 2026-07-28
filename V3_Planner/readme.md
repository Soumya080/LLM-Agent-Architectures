<p align="center">
  <h1 align="center">Planner Agent (V3)</h1>
  <p align="center">
    <em>An LLM agent that decomposes complex goals into structured execution plans before acting, combining planning with reflexion for robust task completion.</em>
  </p>
  <p align="center">
    <a href="#architecture">Architecture</a> &bull;
    <a href="#quick-start">Quick Start</a> &bull;
    <a href="#components">Components</a> &bull;
    <a href="#evaluation">Evaluation</a> &bull;
    <a href="#tests">Tests</a>
  </p>
</p>

---

## Highlights

| Feature | Description |
|---------|-------------|
| **Plan-Before-Act** | LLM generates a structured `PlanSchema` with ordered tasks before any tool execution begins |
| **Task Dependencies** | Each task defines preconditions and postconditions, enabling dependency-aware sequencing |
| **Plan-Guided Execution** | Agent loop follows the plan step-by-step, grounding each reasoning step in the current task |
| **Reflexion Integration** | Inherits V2's self-evaluation and reflection for per-step error correction |
| **Experience Memory** | Plan-contextualized retrieval surfaces relevant past reflections scoped to similar tasks |

---

## Architecture

V3 introduces a **Plan phase** that sits between the user query and the ReAct execution loop. The planner generates a structured task graph, and the agent loop executes each task sequentially:

```
                    +---------------------+
                    |    User Query       |
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    |      Planner        |
                    |                     |
                    |  Prompt -> LLM ->   |
                    |  Parse -> PlanSchema|
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    |     PlanSchema      |
                    |                     |
                    |  Task 1 -> Task 2   |
                    |  -> Task 3 -> ...   |
                    +----------+----------+
                               |
                               v
               +-------------------------------+
               |     Agent Loop (per task)     |
               |                               |
               |   For each task in plan:      |
               |   Think -> Validate -> Act    |
               |   -> Observe -> Evaluate      |
               |   -> Reflect (on failure)     |
               +-------------------------------+
                               |
                               v
                    +---------------------+
                    |   Final Answer      |
                    +---------------------+
```

---

## Project Structure

```
V3_Planner/
|
+-- agent/                          # Core agent loop & reasoning
|   +-- agent_loop.py               # Plan-guided execution loop with per-task stepping
|   +-- reasoning_engine.py         # Plan-aware reasoning with dynamic prompt construction
|   +-- reasoning_prompt.py         # Prompt templates incorporating current plan context
|   +-- llm_client.py               # Ollama HTTP client wrapper
|   +-- action_validator.py         # Validates proposed actions against tool schemas
|   +-- termination_checker.py      # Controls loop termination conditions
|   +-- planner_loop.py             # Orchestrates plan generation and sequential execution
|
+-- planner/                        # Task planning subsystem (V3 innovation)
|   +-- planner.py                  # Core planner: prompt -> LLM -> parse -> PlanSchema
|   +-- planner_prompt.py           # System prompt with decomposition constraints
|   +-- planner_parser.py           # Parses LLM-generated plans into structured PlanSchema
|
+-- reflexion/                      # Self-improvement via reflection (from V2)
|   +-- evaluator.py                # Base evaluator interface for trial scoring
|   +-- rule_evaluator.py           # Heuristic-based evaluation with configurable rules
|   +-- llm_evaluator.py            # LLM-based semantic correctness scoring
|   +-- reflection_engine.py        # Generates structured lessons from failures
|   +-- experience_memory.py        # Plan-contextualized episode retrieval
|   +-- evaluation_schema.py        # Evaluation result data structure
|   +-- reflection_schema.py        # Reflection data structure with failure taxonomy
|
+-- schemas/                        # Pure data classes (no execution logic)
|   +-- plan_schema.py              # PlanSchema: ordered collection of TaskSchema objects
|   +-- task_schema.py              # TaskSchema: atomic task with pre/postconditions
|   +-- execution_context.py        # Runtime tracking for plan execution state
|   +-- context_schema.py           # Session state with trajectory history
|   +-- action_schema.py            # Structured representation of a proposed action
|   +-- observation_schema.py       # Tool execution result representation
|
+-- tools/                          # Agent tool implementations
|   +-- base_tool.py                # Abstract tool interface
|   +-- calculator_tool.py          # Mathematical expression evaluator
|   +-- search_tool.py              # Knowledge retrieval tool
|   +-- finish_tool.py              # Terminal action for returning final answers
|   +-- tool_registry.py            # Tool discovery & registration
|   +-- tool_executor.py            # Safe tool execution wrapper
|
+-- evaluation/                     # Benchmark evaluation framework
|   +-- run_evaluation.py           # Full evaluation pipeline with plan analytics
|   +-- run_academic_eval.py        # Academic benchmark runner
|   +-- academic_loader.py          # Dataset loader for academic benchmarks
|   +-- answer_evaluator.py         # Answer correctness evaluator with fuzzy matching
|   +-- benchmarks/                 # Benchmark datasets (JSON)
|   +-- results/                    # Evaluation results and reports
|
+-- tests/                          # Unit & integration tests
|   +-- test_agent.py               # Agent loop tests
|   +-- test_reasoning.py           # Reasoning engine tests
|   +-- test_parser_fallback.py     # Parser robustness tests
|   +-- test_ollama.py              # Ollama integration tests
|
+-- main.py                         # Interactive terminal entry point
+-- test_run.py                     # Programmatic test runner
+-- PlanSchema.md                   # PlanSchema specification document
+-- README.md                       # This file
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.ai)** installed and running locally

### 1. Install dependencies

```bash
cd V3_Planner
pip install -r ../V4_hierarchical_Planner/requirements.txt
```

### 2. Start Ollama

```bash
ollama serve
ollama pull qwen2.5-coder:7b
```

### 3. Run the agent

```bash
python main.py
```

### 4. Run the evaluation benchmark

```bash
python evaluation/run_evaluation.py --category multistep
```

---

## Components

### Planner

The `Planner` class is a pure planning module with strict separation of concerns:

```python
from planner.planner import Planner

planner = Planner(
    llm_client=llm_client,
    planner_prompt=planner_prompt,
    planner_parser=planner_parser,
)

plan = planner.plan("What is (15 * 3) + (42 / 6)?")
# Returns PlanSchema with ordered TaskSchema objects
```

**Design constraint**: The Planner *never* executes tools, performs reasoning, evaluates actions, or reflects. It only generates plans.

### PlanSchema & TaskSchema

```python
PlanSchema(
    plan_id="plan-001",
    goal="Solve multi-step math problem",
    tasks=[
        TaskSchema(task_id="t1", description="Calculate 15 * 3", ...),
        TaskSchema(task_id="t2", description="Calculate 42 / 6", ...),
        TaskSchema(task_id="t3", description="Add results", depends_on=["t1", "t2"], ...),
    ],
    overall_success_criteria="Return the final numeric answer",
    status="PENDING"
)
```

Each `TaskSchema` includes:
- **Preconditions**: What must be true before execution
- **Postconditions**: What should be true after execution
- **Expected tool**: Which tool the planner expects the agent to use
- **Dependencies**: Other task IDs that must complete first

### Plan-Guided Agent Loop

The V3 agent loop differs from V1/V2 by grounding each reasoning step in the current plan task:

1. **Select** the next pending task from the plan
2. **Inject** task context into the reasoning prompt
3. **Execute** the standard Think-Act-Observe cycle
4. **Evaluate** against the task's postconditions
5. **Advance** to the next task or reflect on failure

---

## Evaluation

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

---

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing
```

---

## Evolution from V2

| Aspect | V2 (Reflexion+ReAct) | V3 (Planner) |
|--------|----------------------|--------------|
| Approach | React first, reflect after | Plan first, then execute |
| Decomposition | Implicit (LLM decides on the fly) | Explicit (structured PlanSchema) |
| Task ordering | Sequential by LLM intuition | Dependency-aware with pre/postconditions |
| Reasoning scope | Full query context each step | Scoped to current plan task |
| New schemas | ReflectionSchema, EvaluationSchema | PlanSchema, TaskSchema, ExecutionContext |

---

<p align="center">
  <sub>Part of the <a href="https://github.com/Soumya080/LLM-Agent-Architectures">LLM Agent Architectures</a> research project</sub>
</p>