<p align="center">
  <h1 align="center">Reflexion+ReAct Agent (V2)</h1>
  <p align="center">
    <em>A self-improving AI agent that learns from its mistakes through structured reflection and experience memory.</em>
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
| **Self-Correction Loop** | Multi-trial execution where the agent evaluates its own output and retries with reflective insights |
| **Experience Memory** | Persistent episode store that surfaces relevant past failures to prevent repeated mistakes |
| **Dual Evaluation** | Rule-based heuristic scoring + LLM-based semantic correctness evaluation |
| **Reflection Engine** | Generates structured lessons from failed trajectories with failure category taxonomy |
| **Backward-Compatible** | Extends V1 ReAct without modifying the core Think-Act-Observe loop |

---

## Architecture

V2 wraps the V1 ReAct loop inside a **Reflexion outer loop** that enables multi-trial self-improvement:

```
                    +---------------------+
                    |    User Query       |
                    +----------+----------+
                               |
                               v
                    +---------------------+
                    |  Experience Memory  |
                    |  (retrieve relevant |
                    |   past reflections) |
                    +----------+----------+
                               |
                               v
               +-------------------------------+
               |        ReAct Agent Loop       |
               |                               |
               |   Think -> Validate -> Act    |
               |          -> Observe           |
               +-------------------------------+
                               |
                               v
                    +---------------------+
                    |     Evaluator       |
                    |  (Rule + LLM)       |
                    +----------+----------+
                               |
                    +----------+----------+
                    |  Pass?   | Fail?    |
                    +----+-----+----+-----+
                         |          |
                         v          v
                   +---------+  +------------------+
                   | Return  |  | Reflection       |
                   | Answer  |  | Engine           |
                   +---------+  +--------+---------+
                                         |
                                         v
                                +------------------+
                                | Store reflection |
                                | in Experience    |
                                | Memory           |
                                +--------+---------+
                                         |
                                         v
                                   Retry (Trial N+1)
```

---

## Project Structure

```
V2_Reflexion+React/
|
+-- agent/                          # Core agent loop & reasoning
|   +-- agent_loop.py               # Reflexion-augmented execution loop with multi-trial retry
|   +-- reasoning_engine.py         # LLM-powered reasoning with reflection-aware prompting
|   +-- llm_client.py               # Ollama HTTP client wrapper
|   +-- action_validator.py         # Validates proposed actions against tool schemas
|   +-- termination_checker.py      # Controls loop termination conditions
|   +-- planner_loop.py             # Planner integration stub (expanded in V3)
|
+-- reflexion/                      # Self-improvement subsystem (V2 innovation)
|   +-- evaluator.py                # Base evaluator interface for trial scoring
|   +-- rule_evaluator.py           # Heuristic-based evaluation with configurable rules
|   +-- llm_evaluator.py            # LLM-based semantic correctness scoring
|   +-- reflection_engine.py        # Generates structured lessons from failed trajectories
|   +-- experience_memory.py        # Persistent episode store with similarity retrieval
|   +-- evaluation_schema.py        # Evaluation result data structure
|   +-- reflection_schema.py        # Reflection data structure with failure taxonomy
|
+-- schemas/                        # Pure data classes (no execution logic)
|   +-- action_schema.py            # Structured representation of a proposed action
|   +-- context_schema.py           # Session state with trajectory history
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
|   +-- run_evaluation.py           # Full evaluation pipeline with per-trial tracking
|   +-- run_academic_eval.py        # Academic benchmark runner with multi-trial comparison
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
+-- README.md                       # This file
```

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.ai)** installed and running locally

### 1. Install dependencies

```bash
cd V2_Reflexion+React
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
python evaluation/run_evaluation.py --category arithmetic
```

---

## Components

### Reflexion Loop

The core innovation of V2. After each trial, the agent:

1. **Evaluates** its output using dual evaluation (rule-based + LLM-based)
2. **Reflects** on failures to extract structured lessons
3. **Stores** reflections in experience memory
4. **Retrieves** relevant past reflections before the next trial

```python
from reflexion.reflection_engine import ReflectionEngine
from reflexion.experience_memory import ExperienceMemory

# After a failed trial, generate a reflection
reflection = reflection_engine.reflect(
    query=query,
    action=last_action,
    observation=observation,
    evaluation=eval_result,
    context=context
)

# Store for future retrieval
experience_memory.store(reflection)
```

### Dual Evaluation Strategy

| Evaluator | Method | Use Case |
|-----------|--------|----------|
| `RuleEvaluator` | Heuristic scoring with configurable rules | Fast, deterministic checks (tool usage, format) |
| `LLMEvaluator` | Semantic correctness via LLM judgment | Nuanced answer quality assessment |

### Reflection Schema

Each reflection captures:
- **Failure category** (wrong tool, incorrect reasoning, hallucination, etc.)
- **Root cause analysis** from the LLM
- **Corrective strategy** for future attempts
- **Confidence score** for the reflection itself

---

## Evaluation

```bash
# All categories
python evaluation/run_evaluation.py

# Single category
python evaluation/run_evaluation.py --category arithmetic

# Custom model
python evaluation/run_evaluation.py --model llama3:8b
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

## Key Differences from V1

| Aspect | V1 (ReAct) | V2 (Reflexion+ReAct) |
|--------|------------|----------------------|
| Trials | Single attempt | Multi-trial with reflection |
| Evaluation | None (manual) | Dual: rule-based + LLM-based |
| Memory | Stateless | Persistent experience memory |
| Self-improvement | None | Structured reflection loop |
| Error handling | Fail and stop | Reflect, learn, and retry |

---

<p align="center">
  <sub>Part of the <a href="https://github.com/Soumya080/LLM-Agent-Architectures">LLM Agent Architectures</a> research project</sub>
</p>