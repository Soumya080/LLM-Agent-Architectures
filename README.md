<p align="center">
  <h1 align="center">LLM Agent Architectures</h1>
  <p align="center">
    <em>A from-scratch research exploration of four progressive LLM agent architectures &mdash; from simple ReAct loops to recursive hierarchical planners.</em>
  </p>
  <p align="center">
    <a href="#architectures">Architectures</a> &bull;
    <a href="#progression">Progression</a> &bull;
    <a href="#quick-start">Quick Start</a> &bull;
    <a href="#evaluation">Evaluation</a> &bull;
    <a href="#project-structure">Project Structure</a>
  </p>
</p>

---

## Overview

This repository contains four autonomous agent systems built entirely from scratch in Python. Each version introduces a new cognitive capability, progressing from a simple reasoning loop to a recursive hierarchical planner with self-improvement.

Every architecture is:
- **Self-contained** &mdash; each version is a standalone, runnable agent
- **Locally executable** &mdash; runs on Ollama with no cloud API dependencies
- **Benchmarked** &mdash; includes a 50-case evaluation suite across 5 task categories
- **Tested** &mdash; unit and integration tests for all core components

---

## Architectures

### V1 &mdash; ReAct Agent

> *Think &rarr; Act &rarr; Observe*

The foundational agent implementing the [ReAct](https://arxiv.org/abs/2210.03629) pattern. The LLM reasons about the problem, selects a tool, observes the result, and repeats until the task is solved.

**[&rarr; V1 Documentation](V1_react/readme.md)**

---

### V2 &mdash; Reflexion + ReAct

> *Think &rarr; Act &rarr; Observe &rarr; Evaluate &rarr; Reflect &rarr; Retry*

Wraps V1 in a [Reflexion](https://arxiv.org/abs/2303.11366) outer loop. After each trial, the agent evaluates its own output, generates structured lessons from failures, and stores them in persistent experience memory for future retrieval.

**Key additions:** `ReflectionEngine`, `ExperienceMemory`, dual evaluation (rule-based + LLM-based)

**[&rarr; V2 Documentation](V2_Reflexion+React/readme.md)**

---

### V3 &mdash; Planner Agent

> *Plan &rarr; Think &rarr; Act &rarr; Observe &rarr; Evaluate &rarr; Reflect*

Adds a **planning phase** before execution. The LLM generates a structured `PlanSchema` with ordered tasks, dependency constraints, and pre/postconditions. The agent loop then executes each task sequentially, grounding its reasoning in the current plan step.

**Key additions:** `Planner`, `PlanSchema`, `TaskSchema`, `ExecutionContext`

**[&rarr; V3 Documentation](V3_Planner/readme.md)**

---

### V4 &mdash; Hierarchical Planner

> *Decompose &rarr; Plan &rarr; Think &rarr; Act &rarr; Observe &rarr; Evaluate &rarr; Reflect*

The most advanced architecture. Replaces the linear planner with a **recursive hierarchical decomposer** that builds a goal tree. A decision policy determines whether each node should be further decomposed or executed directly. Traversal strategies (DFS, BFS, Priority) control execution order.

**Key additions:** `HierarchicalPlanner`, `RecursiveDecomposer`, `DecisionPolicy`, `TraversalPolicy`, `GoalTree`, `GoalNode`

**[&rarr; V4 Documentation](V4_hierarchical_Planner/README.md)**

---

## Progression

The following table summarizes the architectural evolution across all four versions:

| Capability | V1 | V2 | V3 | V4 |
|:-----------|:--:|:--:|:--:|:--:|
| ReAct loop (Think-Act-Observe) | ✓ | ✓ | ✓ | ✓ |
| Tool registry & executor | ✓ | ✓ | ✓ | ✓ |
| Self-evaluation | | ✓ | ✓ | ✓ |
| Reflection & experience memory | | ✓ | ✓ | ✓ |
| Task planning (PlanSchema) | | | ✓ | ✓ |
| Task dependencies & pre/postconditions | | | ✓ | ✓ |
| Recursive goal decomposition | | | | ✓ |
| Goal tree with traversal policies | | | | ✓ |
| Decision policy (decompose vs execute) | | | | ✓ |
| Constraint-aware planning (depth, budget) | | | | ✓ |

```
V1 ReAct            V2 Reflexion         V3 Planner           V4 Hierarchical
+-----------+       +-----------+        +-----------+        +-----------+
|  Think    |       |  Think    |        |  Plan     |        | Decompose |
|  Act      | --->  |  Act      | --->   |  Think    | --->   |  Plan     |
|  Observe  |       |  Observe  |        |  Act      |        |  Think    |
+-----------+       |  Evaluate |        |  Observe  |        |  Act      |
                    |  Reflect  |        |  Evaluate |        |  Observe  |
                    +-----------+        |  Reflect  |        |  Evaluate |
                                         +-----------+        |  Reflect  |
                                                              +-----------+
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.ai)** installed and running

### Setup

```bash
# Clone
git clone https://github.com/Soumya080/LLM-Agent-Architectures.git
cd LLM-Agent-Architectures

# Install dependencies
pip install -r V4_hierarchical_Planner/requirements.txt

# Start Ollama and pull a model
ollama serve
ollama pull qwen2.5-coder:7b
```

### Run any version

```bash
# V1 - ReAct
python V1_react/main.py

# V2 - Reflexion+ReAct
python V2_Reflexion+React/main.py

# V3 - Planner
python V3_Planner/main.py

# V4 - Hierarchical Planner
python V4_hierarchical_Planner/main.py
```

---

## Evaluation

Each version ships with the same 50-case benchmark suite across 5 categories:

| Category | Description |
|----------|-------------|
| `arithmetic` | Single and multi-step math problems |
| `search` | Knowledge retrieval queries |
| `multistep` | Multi-tool reasoning chains |
| `failure` | Edge cases and error handling |
| `ambiguous` | Underspecified or ambiguous queries |

Run evaluation for any version:

```bash
python V4_hierarchical_Planner/evaluation/run_evaluation.py
python V4_hierarchical_Planner/evaluation/run_evaluation.py --category arithmetic
```

---

## Project Structure

```
LLM-Agent-Architectures/
|
+-- V1_react/                       # ReAct agent (Think-Act-Observe)
|   +-- agent/                      # Core loop, reasoning engine, LLM client
|   +-- tools/                      # Calculator, search, finish tools
|   +-- schemas/                    # Action, context, observation schemas
|   +-- tests/                      # Unit tests
|   +-- evaluation/                 # Benchmarks & results
|   +-- main.py                     # Entry point
|
+-- V2_Reflexion+React/             # Reflexion + ReAct (self-improvement)
|   +-- agent/                      # Reflexion-augmented agent loop
|   +-- reflexion/                  # Evaluators, reflection engine, experience memory
|   +-- tools/                      # Tool implementations
|   +-- schemas/                    # Data schemas
|   +-- tests/                      # Unit tests
|   +-- evaluation/                 # Benchmarks & results
|   +-- main.py                     # Entry point
|
+-- V3_Planner/                     # Plan-then-execute agent
|   +-- agent/                      # Plan-guided agent loop
|   +-- planner/                    # Planner, parser, prompt templates
|   +-- reflexion/                  # Self-evaluation & reflection
|   +-- tools/                      # Tool implementations
|   +-- schemas/                    # Plan, task, execution context schemas
|   +-- tests/                      # Unit tests
|   +-- evaluation/                 # Benchmarks & results
|   +-- main.py                     # Entry point
|
+-- V4_hierarchical_Planner/        # Recursive hierarchical planner
|   +-- agent/                      # Hierarchical agent loop
|   +-- planner/                    # Hierarchical, linear, common planners
|   |   +-- hierarchical/           # Recursive decomposer, decision & traversal policies
|   |   +-- linear/                 # Linear planner (V3 baseline)
|   |   +-- common/                 # Goal tree, constraints
|   +-- reflexion/                  # Self-evaluation & reflection
|   +-- tools/                      # Tool implementations
|   +-- schemas/                    # Goal tree, goal node, plan schemas
|   +-- tests/                      # Unit tests (incl. hierarchical planner tests)
|   +-- evaluation/                 # Benchmarks & results
|   +-- main.py                     # Entry point
|
+-- .gitignore
+-- README.md                       # This file
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| LLM Backend | [Ollama](https://ollama.ai) (local inference) |
| Default Model | `qwen2.5-coder:7b` |
| Testing | `pytest` |
| Dependencies | Minimal &mdash; standard library + `requests` |

---

## License

This project is licensed under the MIT License. See [LICENSE](V4_hierarchical_Planner/LICENSE) for details.

---

<p align="center">
  <sub>Built from scratch as part of independent LLM agent architecture research</sub>
</p>
