"""
Planner Agent V3 - Evaluation Framework

Loads benchmark datasets, runs the full Planner agent on each query,
collects metrics, and generates a detailed evaluation report.

Usage:
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --category arithmetic
    python evaluation/run_evaluation.py --max-iterations 10
    python evaluation/run_evaluation.py --model qwen2.5-coder:7b
"""

import json
import os
import sys
import time
import argparse
import traceback
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─── Resolve project root so imports work ───
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from agent.llm_client import LLMClient
from agent.reasoning_engine import ReasoningEngine
from agent.action_validator import ActionValidator
from agent.agent_loop import AgentLoop
from agent.termination_checker import TerminationChecker
from tools.tool_registry import ToolRegistry
from tools.tool_executor import ToolExecutor
from tools.calculator_tool import CalculatorTool
from tools.search_tool import SearchTool
from tools.finish_tool import FinishTool

from planner.planner import Planner
from planner.planner_prompt import PlannerPrompt
from planner.planner_parser import PlannerParser

from reflexion.evaluator import Evaluator
from reflexion.reflection_engine import ReflectionEngine


# ═══════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════

@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    id: int
    query: str
    expected_tools: list
    expected_behavior: str
    difficulty: str
    category: str = ""


@dataclass
class RunResult:
    """Result of running a single benchmark case."""
    case_id: int
    query: str
    category: str
    difficulty: str

    # Expected
    expected_tools: list = field(default_factory=list)
    expected_behavior: str = ""

    # Actual
    actual_tools: list = field(default_factory=list)
    final_answer: Optional[str] = None
    iteration_count: int = 0
    agent_finished: bool = False

    # Metrics
    tool_sequence_match: bool = False
    used_finish: bool = False
    had_error: bool = False
    error_message: str = ""
    looped: bool = False
    elapsed_seconds: float = 0.0

    # Reflexion Metrics
    reflection_count: int = 0
    reflection_triggered: bool = False
    reflection_categories: dict = field(default_factory=dict)


@dataclass
class CategoryReport:
    """Aggregated metrics for a single category."""
    category: str
    total: int = 0
    tool_sequence_matches: int = 0
    finish_used: int = 0
    agent_completed: int = 0
    errors: int = 0
    loops: int = 0
    total_steps: int = 0
    
    # Reflexion Metrics
    total_reflections: int = 0
    queries_with_reflections: int = 0
    correct_with_reflections: int = 0
    reflection_category_counts: dict = field(default_factory=dict)

    @property
    def tool_accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.tool_sequence_matches / self.total) * 100

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.agent_completed / self.total) * 100

    @property
    def finish_accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.finish_used / self.total) * 100

    @property
    def average_steps(self) -> float:
        if self.total == 0:
            return 0.0
        return self.total_steps / self.total

    @property
    def loop_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.loops / self.total) * 100

    @property
    def error_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.errors / self.total) * 100

    @property
    def reflection_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.queries_with_reflections / self.total) * 100

    @property
    def reflections_per_query(self) -> float:
        if self.total == 0:
            return 0.0
        return self.total_reflections / self.total

    @property
    def reflection_success_rate(self) -> float:
        if self.queries_with_reflections == 0:
            return 0.0
        return (self.correct_with_reflections / self.queries_with_reflections) * 100
        
    @property
    def most_common_reflection_type(self) -> str:
        if not self.reflection_category_counts:
            return "none"
        return max(self.reflection_category_counts.items(), key=lambda x: x[1])[0]


# ═══════════════════════════════════════════
# Benchmark Loader
# ═══════════════════════════════════════════

class BenchmarkLoader:
    """
    Loads benchmark JSON files from a directory.
    Each file name (minus extension) becomes the category.
    """

    CATEGORY_FILE_MAP = {
        "arithmetic": "arithmetic_tasks.json",
        "search": "search_tasks.json",
        "multistep": "multistep_tasks.json",
        "failure": "failure_cases.json",
        "ambiguous": "ambiguous_queries.json",
    }

    def __init__(self, benchmark_dir: str):
        self.benchmark_dir = benchmark_dir

    def load_category(
        self, category: str
    ) -> list[BenchmarkCase]:
        """Load a single category of benchmarks."""

        filename = self.CATEGORY_FILE_MAP.get(category)

        if filename is None:
            raise ValueError(
                f"Unknown category: '{category}'. "
                f"Available: {list(self.CATEGORY_FILE_MAP.keys())}"
            )

        filepath = os.path.join(
            self.benchmark_dir, filename
        )

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Benchmark file not found: {filepath}"
            )

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        cases = []
        for item in data:
            case = BenchmarkCase(
                id=item["id"],
                query=item["query"],
                expected_tools=item["expected_tools"],
                expected_behavior=item["expected_behavior"],
                difficulty=item["difficulty"],
                category=category,
            )
            cases.append(case)

        return cases

    def load_all(self) -> dict[str, list[BenchmarkCase]]:
        """Load all benchmark categories."""

        all_benchmarks = {}

        for category in self.CATEGORY_FILE_MAP:
            try:
                all_benchmarks[category] = (
                    self.load_category(category)
                )
            except FileNotFoundError as e:
                print(f"  [WARN] Skipping {category}: {e}")

        return all_benchmarks


# ═══════════════════════════════════════════
# Agent Factory
# ═══════════════════════════════════════════

class AgentFactory:
    """
    Builds a fresh AgentLoop instance with the full
    Planner-based architecture.

    Creates a new instance per run to avoid state leakage.

    Initialises:
        Planner  (PlannerPrompt + PlannerParser)
        ReasoningEngine
        ActionValidator
        ToolExecutor
        TerminationChecker
        Evaluator
        ReflectionEngine
        AgentLoop  (top-level orchestrator)
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        max_iterations: int = 5,
        confidence_threshold: float = 0.6,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

    def build(self) -> AgentLoop:
        """
        Create a fresh agent instance matching the
        production Planner architecture.
        """

        # ── Tool Registry ────────────────────────
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())
        registry.register_tool(SearchTool())
        registry.register_tool(FinishTool())

        # ── LLM Client ──────────────────────────
        llm = LLMClient(
            model_name=self.model_name,
            base_url=self.base_url,
        )

        # ── Reasoning Engine ────────────────────
        reasoning_engine = ReasoningEngine(
            llm_client=llm,
            tool_registry=registry,
        )

        # ── Validation & Execution ──────────────
        validator = ActionValidator(registry)
        executor = ToolExecutor(registry)

        # ── Termination ─────────────────────────
        termination_checker = TerminationChecker(
            max_iterations=self.max_iterations
        )

        # ── Planner ─────────────────────────────
        planner = Planner(
            llm_client=llm,
            planner_prompt=PlannerPrompt(),
            planner_parser=PlannerParser(),
        )

        # ── Evaluator ───────────────────────────
        evaluator = Evaluator(
            llm_client=llm,
            confidence_threshold=self.confidence_threshold,
        )

        # ── Reflection Engine ───────────────────
        reflection_engine = ReflectionEngine(
            llm_client=llm,
        )

        # ── Agent Loop (full Planner pipeline) ──
        return AgentLoop(
            reasoning_engine=reasoning_engine,
            action_validator=validator,
            tool_executor=executor,
            termination_checker=termination_checker,
            evaluator=evaluator,
            reflection_engine=reflection_engine,
            planner=planner,
        )

    def health_check(self) -> bool:
        """Verify the Ollama server is reachable."""

        llm = LLMClient(
            model_name=self.model_name,
            base_url=self.base_url,
        )
        return llm.health_check()


# ═══════════════════════════════════════════
# Result Analyzer
# ═══════════════════════════════════════════

class ResultAnalyzer:
    """
    Compares expected vs actual tool sequences
    and computes per-result metrics.
    """

    @staticmethod
    def extract_tool_sequence(context) -> list[str]:
        """
        Extract the ordered list of tool names
        from the agent's action history.
        """
        return [
            action.tool_name
            for action in context.action_history
        ]

    @staticmethod
    def extract_final_answer(context) -> Optional[str]:
        """
        Extract the final answer from the last
        finish tool observation, if any.
        """

        for action, obs in zip(
            reversed(context.action_history),
            reversed(context.observation_history),
        ):
            if action.tool_name == "finish" and obs.success:
                return str(obs.result)

        return None

    @staticmethod
    def check_tool_sequence_match(
        expected: list[str],
        actual: list[str],
    ) -> bool:
        """
        Check if the actual tool sequence matches
        the expected sequence exactly.
        """
        return expected == actual

    @staticmethod
    def detect_loop(context) -> bool:
        """
        Detect if the agent looped:
        - hit max iterations without finishing
        - repeated the same tool+params consecutively
        """

        if not context.done:
            return True

        # Check for consecutive duplicate actions
        actions = context.action_history

        for i in range(1, len(actions)):
            prev = actions[i - 1]
            curr = actions[i]

            if (
                prev.tool_name == curr.tool_name
                and prev.parameters == curr.parameters
            ):
                return True

        return False

    @staticmethod
    def classify_reflection(reflection) -> str:
        """Classify reflection into predefined categories."""
        content = str(reflection).lower()
        if "broad_search_query" in content or "broad search" in content:
            return "broad_search_query"
        if "repeated_tool_usage" in content or "repeated tool" in content or "same tool" in content:
            return "repeated_tool_usage"
        if "premature_finish" in content or "premature finish" in content:
            return "premature_finish"
        if "wrong_calculation" in content or "wrong calculation" in content or "math error" in content:
            return "wrong_calculation"
        if "missing_information" in content or "missing information" in content or "insufficient information" in content:
            return "missing_information"
        if "loop_behavior" in content or "looping" in content or "stuck in loop" in content:
            return "loop_behavior"
        return "generic_failure"

    def analyze(
        self,
        case: BenchmarkCase,
        context,
        elapsed: float,
        error: Optional[str] = None,
    ) -> RunResult:
        """Analyze a single run and produce a RunResult."""

        result = RunResult(
            case_id=case.id,
            query=case.query,
            category=case.category,
            difficulty=case.difficulty,
            expected_tools=case.expected_tools,
            expected_behavior=case.expected_behavior,
            elapsed_seconds=elapsed,
        )

        if error is not None:
            result.had_error = True
            result.error_message = error
            return result

        result.actual_tools = self.extract_tool_sequence(
            context
        )
        result.final_answer = self.extract_final_answer(
            context
        )
        result.iteration_count = context.iteration_count
        result.agent_finished = context.done
        result.used_finish = "finish" in result.actual_tools
        result.looped = self.detect_loop(context)

        # Reflection Metrics Extraction
        if hasattr(context, "reflection_histroy") and context.reflection_histroy:
            result.reflection_count = len(context.reflection_histroy)
            result.reflection_triggered = result.reflection_count > 0
            for refl in context.reflection_histroy:
                cat = self.classify_reflection(refl)
                result.reflection_categories[cat] = result.reflection_categories.get(cat, 0) + 1

        result.tool_sequence_match = (
            self.check_tool_sequence_match(
                case.expected_tools,
                result.actual_tools,
            )
        )

        return result


# ═══════════════════════════════════════════
# Evaluation Runner
# ═══════════════════════════════════════════

class EvaluationRunner:
    """
    Orchestrates loading benchmarks, running agent,
    collecting results, and generating reports.
    """

    def __init__(
        self,
        agent_factory: AgentFactory,
        benchmark_dir: str,
        output_dir: str,
    ):
        self.agent_factory = agent_factory
        self.loader = BenchmarkLoader(benchmark_dir)
        self.analyzer = ResultAnalyzer()
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    # ─── Run a single case ───

    def run_single(
        self, case: BenchmarkCase
    ) -> RunResult:
        """Run the agent on a single benchmark case."""

        agent = self.agent_factory.build()

        start = time.time()
        error = None
        context = None

        try:
            context = agent.run(case.query)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        elapsed = time.time() - start

        return self.analyzer.analyze(
            case, context, elapsed, error
        )

    # ─── Run a full category ───

    def run_category(
        self,
        category: str,
        cases: list[BenchmarkCase],
    ) -> list[RunResult]:
        """Run all cases in a category."""

        print(f"\n{'─' * 50}")
        print(f"  Category: {category.upper()}")
        print(f"  Cases: {len(cases)}")
        print(f"{'─' * 50}")

        results = []

        for i, case in enumerate(cases):
            label = (
                f"  [{i + 1}/{len(cases)}] "
                f"ID={case.id} | "
                f"difficulty={case.difficulty}"
            )
            print(label)
            print(f"    Query: {case.query[:80]}...")

            result = self.run_single(case)

            status = (
                "✓ MATCH"
                if result.tool_sequence_match
                else "✗ MISMATCH"
            )
            print(
                f"    Expected: {case.expected_tools}"
            )
            print(
                f"    Actual:   {result.actual_tools}"
            )
            print(
                f"    Status:   {status} | "
                f"Steps={result.iteration_count} | "
                f"Time={result.elapsed_seconds:.1f}s"
            )

            if result.had_error:
                print(
                    f"    ERROR:    {result.error_message}"
                )

            results.append(result)

        return results

    # ─── Run everything ───

    def run_all(
        self,
        categories: Optional[list[str]] = None,
    ) -> dict[str, list[RunResult]]:
        """
        Run evaluation across all (or selected)
        categories.
        """

        all_benchmarks = self.loader.load_all()

        if categories:
            all_benchmarks = {
                k: v
                for k, v in all_benchmarks.items()
                if k in categories
            }

        all_results = {}

        for category, cases in all_benchmarks.items():
            results = self.run_category(
                category, cases
            )
            all_results[category] = results

        return all_results

    # ─── Aggregate metrics ───

    @staticmethod
    def aggregate(
        results: list[RunResult], category: str
    ) -> CategoryReport:
        """Aggregate results into a CategoryReport."""

        report = CategoryReport(
            category=category,
            total=len(results),
        )

        for r in results:
            if r.tool_sequence_match:
                report.tool_sequence_matches += 1
            if r.used_finish:
                report.finish_used += 1
            if r.agent_finished:
                report.agent_completed += 1
            if r.had_error:
                report.errors += 1
            if r.looped:
                report.loops += 1

            # Aggregate Reflexion Metrics
            report.total_reflections += r.reflection_count
            if r.reflection_triggered:
                report.queries_with_reflections += 1
                if r.tool_sequence_match:
                    report.correct_with_reflections += 1
            for cat, count in r.reflection_categories.items():
                report.reflection_category_counts[cat] = report.reflection_category_counts.get(cat, 0) + count

            report.total_steps += r.iteration_count

        return report

    # ─── Report generation ───

    def generate_report(
        self,
        all_results: dict[str, list[RunResult]],
    ) -> str:
        """Generate the final evaluation report."""

        reports: dict[str, CategoryReport] = {}
        for category, results in all_results.items():
            reports[category] = self.aggregate(
                results, category
            )

        # ── Global aggregates ──
        total_cases = sum(
            r.total for r in reports.values()
        )
        total_matches = sum(
            r.tool_sequence_matches
            for r in reports.values()
        )
        total_finished = sum(
            r.finish_used
            for r in reports.values()
        )
        total_steps = sum(
            r.total_steps
            for r in reports.values()
        )
        total_loops = sum(
            r.loops for r in reports.values()
        )
        total_errors = sum(
            r.errors for r in reports.values()
        )

        global_tool_acc = (
            (total_matches / total_cases * 100)
            if total_cases > 0
            else 0.0
        )
        global_finish_acc = (
            (total_finished / total_cases * 100)
            if total_cases > 0
            else 0.0
        )
        global_avg_steps = (
            (total_steps / total_cases)
            if total_cases > 0
            else 0.0
        )
        global_loop_rate = (
            (total_loops / total_cases * 100)
            if total_cases > 0
            else 0.0
        )
        global_error_rate = (
            (total_errors / total_cases * 100)
            if total_cases > 0
            else 0.0
        )

        total_queries_with_refl = sum(r.queries_with_reflections for r in reports.values())
        total_correct_with_refl = sum(r.correct_with_reflections for r in reports.values())
        total_refl_all = sum(r.total_reflections for r in reports.values())
        
        global_reflection_rate = (total_queries_with_refl / total_cases * 100) if total_cases > 0 else 0.0
        global_reflection_success_rate = (total_correct_with_refl / total_queries_with_refl * 100) if total_queries_with_refl > 0 else 0.0
        global_reflections_per_query = (total_refl_all / total_cases) if total_cases > 0 else 0.0
        
        global_refl_counts = {}
        for r in reports.values():
            for k, v in r.reflection_category_counts.items():
                global_refl_counts[k] = global_refl_counts.get(k, 0) + v
        global_most_common_refl = max(global_refl_counts.items(), key=lambda x: x[1])[0] if global_refl_counts else "none"

        # ── Build report string ──
        lines = []
        w = 60

        lines.append("")
        lines.append("=" * w)
        lines.append(
            "  PLANNER AGENT V3 — EVALUATION REPORT"
        )
        lines.append("=" * w)
        lines.append(
            f"  Timestamp : {datetime.now().isoformat()}"
        )
        lines.append(
            f"  Model     : {self.agent_factory.model_name}"
        )
        lines.append(
            f"  Max Iter  : {self.agent_factory.max_iterations}"
        )
        lines.append(
            f"  Total     : {total_cases} cases"
        )
        lines.append("=" * w)

        # ── Per-category breakdown ──
        lines.append("")
        lines.append("─" * w)
        lines.append("  PER-CATEGORY BREAKDOWN")
        lines.append("─" * w)

        for cat_name in [
            "arithmetic",
            "search",
            "multistep",
            "failure",
            "ambiguous",
        ]:
            if cat_name not in reports:
                continue

            r = reports[cat_name]

            lines.append("")
            lines.append(
                f"  ┌─ {cat_name.upper()} "
                f"({r.total} cases)"
            )
            lines.append(
                f"  │  Tool Accuracy    : "
                f"{r.tool_accuracy:5.1f}%  "
                f"({r.tool_sequence_matches}/{r.total})"
            )
            lines.append(
                f"  │  Task Success     : "
                f"{r.success_rate:5.1f}%  "
                f"({r.agent_completed}/{r.total})"
            )
            lines.append(
                f"  │  Finish Accuracy  : "
                f"{r.finish_accuracy:5.1f}%  "
                f"({r.finish_used}/{r.total})"
            )
            lines.append(
                f"  │  Average Steps    : "
                f"{r.average_steps:5.1f}"
            )
            lines.append(
                f"  │  Loop Rate        : "
                f"{r.loop_rate:5.1f}%  "
                f"({r.loops}/{r.total})"
            )
            lines.append(
                f"  │  Error Rate       : "
                f"{r.error_rate:5.1f}%  "
                f"({r.errors}/{r.total})"
            )
            lines.append(f"  └{'─' * (w - 3)}")

        # ── Global summary ──
        lines.append("")
        lines.append("=" * w)
        lines.append("  GLOBAL SUMMARY")
        lines.append("=" * w)
        lines.append(
            f"  Tool Accuracy    : "
            f"{global_tool_acc:5.1f}%  "
            f"({total_matches}/{total_cases})"
        )
        lines.append(
            f"  Finish Accuracy  : "
            f"{global_finish_acc:5.1f}%  "
            f"({total_finished}/{total_cases})"
        )
        lines.append(
            f"  Average Steps    : "
            f"{global_avg_steps:5.1f}"
        )
        lines.append(
            f"  Loop Rate        : "
            f"{global_loop_rate:5.1f}%  "
            f"({total_loops}/{total_cases})"
        )
        lines.append(
            f"  Error Rate       : "
            f"{global_error_rate:5.1f}%  "
            f"({total_errors}/{total_cases})"
        )
        lines.append("=" * w)

        lines.append("")
        lines.append("=" * w)
        lines.append("  REFLEXION METRICS")
        lines.append("=" * w)
        lines.append(f"  Reflection Rate:               {global_reflection_rate:5.1f}%")
        lines.append(f"  Reflection Success Rate:       {global_reflection_success_rate:5.1f}%")
        lines.append(f"  Average Reflections per Query: {global_reflections_per_query:5.1f}")
        lines.append(f"  Most Common Reflection Type:   {global_most_common_refl}")
        lines.append("=" * w)

        # ── Difficulty breakdown ──
        lines.append("")
        lines.append("─" * w)
        lines.append("  DIFFICULTY BREAKDOWN")
        lines.append("─" * w)

        all_flat = [
            r
            for results in all_results.values()
            for r in results
        ]

        for diff in ["easy", "medium", "hard"]:
            subset = [
                r for r in all_flat
                if r.difficulty == diff
            ]
            if not subset:
                continue

            n = len(subset)
            matches = sum(
                1 for r in subset
                if r.tool_sequence_match
            )
            loops = sum(
                1 for r in subset if r.looped
            )

            lines.append(
                f"  {diff.upper():8s}  "
                f"n={n:3d}  "
                f"tool_acc={matches / n * 100:5.1f}%  "
                f"loop_rate={loops / n * 100:5.1f}%"
            )

        lines.append("=" * w)

        # ── Failed cases detail ──
        failed = [
            r for r in all_flat
            if not r.tool_sequence_match
        ]

        if failed:
            lines.append("")
            lines.append("─" * w)
            lines.append(
                f"  MISMATCHED CASES ({len(failed)})"
            )
            lines.append("─" * w)

            for r in failed[:20]:
                lines.append(
                    f"  [{r.category}] ID={r.case_id} "
                    f"({r.difficulty})"
                )
                lines.append(
                    f"    Query:    "
                    f"{r.query[:60]}"
                )
                lines.append(
                    f"    Expected: "
                    f"{r.expected_tools}"
                )
                lines.append(
                    f"    Actual:   "
                    f"{r.actual_tools}"
                )
                if r.had_error:
                    lines.append(
                        f"    Error:    "
                        f"{r.error_message[:60]}"
                    )
                lines.append("")

            if len(failed) > 20:
                lines.append(
                    f"  ... and {len(failed) - 20} more"
                )

        lines.append("")

        return "\n".join(lines)

    # ─── Save results to JSON ───

    def save_results(
        self,
        all_results: dict[str, list[RunResult]],
        filename: str = "evaluation_results.json",
    ):
        """Save raw results to JSON."""

        output_path = os.path.join(
            self.output_dir, filename
        )

        serializable = {}
        for category, results in all_results.items():
            serializable[category] = [
                asdict(r) for r in results
            ]

        with open(
            output_path, "w", encoding="utf-8"
        ) as f:
            json.dump(serializable, f, indent=2)

        print(f"\n  Results saved to: {output_path}")

    def save_report(
        self,
        report: str,
        filename: str = "evaluation_report.txt",
    ):
        """Save the text report to a file."""

        output_path = os.path.join(
            self.output_dir, filename
        )

        with open(
            output_path, "w", encoding="utf-8"
        ) as f:
            f.write(report)

        print(f"  Report saved to:  {output_path}")


# ═══════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Planner Agent V3 — Evaluation Framework"
    )

    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=[
            "arithmetic",
            "search",
            "multistep",
            "failure",
            "ambiguous",
        ],
        help=(
            "Run only a specific category. "
            "Default: run all."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-coder:7b",
        help="Ollama model name.",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL.",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Max agent iterations per query.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help=(
            "Evaluator confidence threshold. "
            "Default: 0.6."
        ),
    )

    parser.add_argument(
        "--reflexion",
        action="store_true",
        help=(
            "[DEPRECATED] Ignored. Evaluation and reflection "
            "are always enabled in the Planner architecture."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.reflexion:
        print(
            "  [WARN] --reflexion is deprecated and ignored. "
            "Evaluation and reflection are always enabled "
            "in the Planner architecture."
        )

    eval_dir = os.path.dirname(
        os.path.abspath(__file__)
    )
    benchmark_dir = os.path.join(
        eval_dir, "benchmarks"
    )
    output_dir = os.path.join(
        eval_dir, "results"
    )

    # ── Build agent factory ──
    factory = AgentFactory(
        model_name=args.model,
        base_url=args.base_url,
        max_iterations=args.max_iterations,
        confidence_threshold=args.confidence_threshold,
    )

    # ── Health check ──
    print("\n  Checking Ollama server...")
    if not factory.health_check():
        print(
            "  ERROR: Ollama is not reachable. "
            "Start it with: ollama serve"
        )
        sys.exit(1)
    print("  ✓ Ollama is reachable.\n")

    # ── Build runner ──
    runner = EvaluationRunner(
        agent_factory=factory,
        benchmark_dir=benchmark_dir,
        output_dir=output_dir,
    )

    # ── Select categories ──
    categories = (
        [args.category] if args.category else None
    )

    # ── Run ──
    all_results = runner.run_all(
        categories=categories
    )

    # ── Report ──
    report = runner.generate_report(all_results)
    print(report)

    # ── Save ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner.save_results(
        all_results,
        f"eval_results_{timestamp}.json",
    )
    runner.save_report(
        report,
        f"eval_report_{timestamp}.txt",
    )


if __name__ == "__main__":
    main()
