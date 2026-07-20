"""
Planner Agent V3 — Academic Benchmark Evaluation Runner

Runs the full Planner agent against GSM8K, HotpotQA, and StrategyQA,
evaluates answer correctness and architecture health,
and generates a combined report.

Usage:
    python evaluation/run_academic_eval.py
    python evaluation/run_academic_eval.py --benchmark gsm8k
    python evaluation/run_academic_eval.py --sample 100
    python evaluation/run_academic_eval.py --model qwen2.5-coder:7b --max-iterations 8
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

from evaluation.academic_loader import (
    AcademicCase,
    AcademicBenchmarkLoader,
)
from evaluation.answer_evaluator import (
    EvalScore,
    get_evaluator,
)
from evaluation.run_evaluation import (
    AgentFactory,
    ResultAnalyzer,
)


# ═══════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════

@dataclass
class AcademicRunResult:
    """Result of running a single academic case."""

    # Identity
    case_id: int
    benchmark: str
    query: str
    difficulty: str

    # Ground truth
    ground_truth: str

    # Agent output
    final_answer: Optional[str] = None
    actual_tools: list = field(default_factory=list)
    iteration_count: int = 0
    agent_finished: bool = False

    # Answer evaluation
    answer_correct: bool = False
    answer_score: float = 0.0
    eval_method: str = ""
    normalized_predicted: str = ""
    normalized_ground_truth: str = ""

    # Architecture metrics
    used_finish: bool = False
    used_search: bool = False
    used_calculator: bool = False
    looped: bool = False
    had_error: bool = False
    error_message: str = ""
    elapsed_seconds: float = 0.0

    # Reflexion Metrics
    reflection_count: int = 0
    reflection_triggered: bool = False
    reflection_categories: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Aggregated metrics for a single benchmark."""

    benchmark: str
    total: int = 0

    # Answer quality
    correct_answers: int = 0
    total_f1: float = 0.0

    # Architecture health
    finished: int = 0
    used_finish_tool: int = 0
    used_search_tool: int = 0
    used_calculator_tool: int = 0
    loops: int = 0
    errors: int = 0
    total_steps: int = 0
    total_elapsed: float = 0.0

    # Reflexion Metrics
    total_reflections: int = 0
    queries_with_reflections: int = 0
    correct_with_reflections: int = 0
    reflection_category_counts: dict = field(default_factory=dict)

    # Difficulty breakdown
    easy_total: int = 0
    easy_correct: int = 0
    medium_total: int = 0
    medium_correct: int = 0
    hard_total: int = 0
    hard_correct: int = 0

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.correct_answers / self.total) * 100

    @property
    def avg_f1(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.total_f1 / self.total) * 100

    @property
    def finish_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.finished / self.total) * 100

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
    def avg_steps(self) -> float:
        if self.total == 0:
            return 0.0
        return self.total_steps / self.total

    @property
    def avg_time(self) -> float:
        if self.total == 0:
            return 0.0
        return self.total_elapsed / self.total

    @property
    def search_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.used_search_tool / self.total) * 100

    @property
    def calculator_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (
            self.used_calculator_tool / self.total
        ) * 100

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
# Academic Evaluation Runner
# ═══════════════════════════════════════════

class AcademicEvaluationRunner:
    """
    Runs the ReAct agent against academic benchmarks,
    evaluates answers, and generates reports.

    Reuses AgentFactory and ResultAnalyzer from
    the custom evaluation framework.
    """

    BENCHMARK_LABELS = {
        "gsm8k": "GSM8K (Grade School Math)",
        "hotpotqa": "HotpotQA (Multi-Hop QA)",
        "strategyqa": "StrategyQA (Yes/No Reasoning)",
    }

    def __init__(
        self,
        agent_factory: AgentFactory,
        output_dir: str,
    ):
        self.agent_factory = agent_factory
        self.analyzer = ResultAnalyzer()
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

    # ─── Run a single case ───

    def run_single(
        self, case: AcademicCase
    ) -> AcademicRunResult:
        """Run the agent on a single academic case."""

        result = AcademicRunResult(
            case_id=case.id,
            benchmark=case.benchmark,
            query=case.query,
            difficulty=case.difficulty,
            ground_truth=case.ground_truth,
        )

        agent = self.agent_factory.build()

        start = time.time()
        context = None

        try:
            context = agent.run(case.query)
        except Exception as e:
            result.had_error = True
            result.error_message = (
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()

        result.elapsed_seconds = time.time() - start

        # ── Extract agent outputs ──

        if context is not None:
            result.actual_tools = (
                self.analyzer.extract_tool_sequence(
                    context
                )
            )
            result.final_answer = (
                self.analyzer.extract_final_answer(
                    context
                )
            )
            result.iteration_count = (
                context.iteration_count
            )
            result.agent_finished = context.done
            result.used_finish = (
                "finish" in result.actual_tools
            )
            result.used_search = (
                "search" in result.actual_tools
            )
            result.used_calculator = (
                "calculator" in result.actual_tools
            )
            result.looped = (
                self.analyzer.detect_loop(context)
            )

        # Reflection Metrics Extraction
        if hasattr(context, "reflection_histroy") and context.reflection_histroy:
            result.reflection_count = len(context.reflection_histroy)
            result.reflection_triggered = result.reflection_count > 0
            for refl in context.reflection_histroy:
                cat = self.analyzer.classify_reflection(refl)
                result.reflection_categories[cat] = result.reflection_categories.get(cat, 0) + 1

        # ── Evaluate answer ──

        evaluator = get_evaluator(case.benchmark)
        eval_score: EvalScore = evaluator.evaluate(
            result.final_answer,
            case.ground_truth,
        )

        result.answer_correct = eval_score.correct
        result.answer_score = eval_score.score
        result.eval_method = eval_score.method
        result.normalized_predicted = (
            eval_score.predicted
        )
        result.normalized_ground_truth = (
            eval_score.ground_truth
        )

        return result

    # ─── Run a full benchmark ───

    def run_benchmark(
        self,
        benchmark: str,
        cases: list[AcademicCase],
    ) -> list[AcademicRunResult]:
        """Run all cases in a benchmark."""

        label = self.BENCHMARK_LABELS.get(
            benchmark, benchmark.upper()
        )

        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"  Cases: {len(cases)}")
        print(f"{'─' * 60}")

        results = []

        for i, case in enumerate(cases):
            progress = (
                f"  [{i + 1}/{len(cases)}] "
                f"ID={case.id}"
            )
            print(progress)

            # Truncate query for display
            query_preview = case.query[:70]
            if len(case.query) > 70:
                query_preview += "..."
            print(f"    Q: {query_preview}")

            result = self.run_single(case)

            # Status display
            if result.had_error:
                status = "✗ ERROR"
            elif result.answer_correct:
                status = "✓ CORRECT"
            else:
                status = "✗ WRONG"

            print(
                f"    {status} | "
                f"tools={result.actual_tools} | "
                f"steps={result.iteration_count} | "
                f"time={result.elapsed_seconds:.1f}s"
            )

            if result.had_error:
                print(
                    f"    Error: "
                    f"{result.error_message[:60]}"
                )
            else:
                print(
                    f"    Pred: "
                    f"{result.normalized_predicted[:50]}"
                )
                print(
                    f"    Gold: "
                    f"{result.normalized_ground_truth[:50]}"
                )

            results.append(result)

        return results

    # ─── Run everything ───

    def run_all(
        self,
        all_cases: dict[str, list[AcademicCase]],
    ) -> dict[str, list[AcademicRunResult]]:
        """Run evaluation across all benchmarks."""

        all_results = {}

        for benchmark, cases in all_cases.items():
            results = self.run_benchmark(
                benchmark, cases
            )
            all_results[benchmark] = results

        return all_results

    # ─── Aggregate metrics ───

    @staticmethod
    def aggregate(
        results: list[AcademicRunResult],
        benchmark: str,
    ) -> BenchmarkReport:
        """Aggregate results into a BenchmarkReport."""

        report = BenchmarkReport(
            benchmark=benchmark,
            total=len(results),
        )

        for r in results:
            # Answer quality
            if r.answer_correct:
                report.correct_answers += 1
            report.total_f1 += r.answer_score

            # Architecture
            if r.agent_finished:
                report.finished += 1
            if r.used_finish:
                report.used_finish_tool += 1
            if r.used_search:
                report.used_search_tool += 1
            if r.used_calculator:
                report.used_calculator_tool += 1
            if r.looped:
                report.loops += 1
            if r.had_error:
                report.errors += 1

            # Aggregate Reflexion Metrics
            report.total_reflections += r.reflection_count
            if r.reflection_triggered:
                report.queries_with_reflections += 1
                if r.answer_correct:
                    report.correct_with_reflections += 1
            for cat, count in r.reflection_categories.items():
                report.reflection_category_counts[cat] = report.reflection_category_counts.get(cat, 0) + count

            report.total_steps += r.iteration_count
            report.total_elapsed += r.elapsed_seconds

            # Difficulty
            if r.difficulty == "easy":
                report.easy_total += 1
                if r.answer_correct:
                    report.easy_correct += 1
            elif r.difficulty == "medium":
                report.medium_total += 1
                if r.answer_correct:
                    report.medium_correct += 1
            elif r.difficulty == "hard":
                report.hard_total += 1
                if r.answer_correct:
                    report.hard_correct += 1

        return report

    # ─── Report generation ───

    def generate_report(
        self,
        all_results: dict[str, list[AcademicRunResult]],
    ) -> str:
        """Generate the academic evaluation report."""

        reports: dict[str, BenchmarkReport] = {}
        for bench, results in all_results.items():
            reports[bench] = self.aggregate(
                results, bench
            )

        w = 60
        lines = []

        # ── Header ──
        lines.append("")
        lines.append("=" * w)
        lines.append(
            "  PLANNER AGENT V3 — "
            "ACADEMIC BENCHMARK REPORT"
        )
        lines.append("=" * w)
        lines.append(
            f"  Timestamp  : "
            f"{datetime.now().isoformat()}"
        )
        lines.append(
            f"  Model      : "
            f"{self.agent_factory.model_name}"
        )
        lines.append(
            f"  Max Iter   : "
            f"{self.agent_factory.max_iterations}"
        )

        total_cases = sum(
            r.total for r in reports.values()
        )
        lines.append(
            f"  Total      : {total_cases} cases"
        )
        lines.append("=" * w)

        # ── Per-benchmark breakdown ──

        for bench_key in ["gsm8k", "hotpotqa", "strategyqa"]:
            if bench_key not in reports:
                continue

            r = reports[bench_key]
            label = self.BENCHMARK_LABELS.get(
                bench_key, bench_key
            )

            lines.append("")
            lines.append("─" * w)
            lines.append(
                f"  {label} — {r.total} samples"
            )
            lines.append("─" * w)

            # Answer quality (benchmark-specific)
            if bench_key == "gsm8k":
                lines.append(
                    f"  Answer Accuracy (EM) : "
                    f"{r.accuracy:5.1f}%  "
                    f"({r.correct_answers}/{r.total})"
                )
            elif bench_key == "hotpotqa":
                lines.append(
                    f"  Exact Match          : "
                    f"{r.accuracy:5.1f}%  "
                    f"({r.correct_answers}/{r.total})"
                )
                lines.append(
                    f"  F1 Score             : "
                    f"{r.avg_f1:5.1f}%"
                )
            elif bench_key == "strategyqa":
                lines.append(
                    f"  Boolean Accuracy     : "
                    f"{r.accuracy:5.1f}%  "
                    f"({r.correct_answers}/{r.total})"
                )

            # Architecture metrics
            lines.append(
                f"  Finish Rate          : "
                f"{r.finish_rate:5.1f}%  "
                f"({r.finished}/{r.total})"
            )

            if bench_key in ("hotpotqa", "strategyqa"):
                lines.append(
                    f"  Used Search          : "
                    f"{r.search_rate:5.1f}%  "
                    f"({r.used_search_tool}/{r.total})"
                )

            if bench_key == "gsm8k":
                lines.append(
                    f"  Used Calculator      : "
                    f"{r.calculator_rate:5.1f}%  "
                    f"({r.used_calculator_tool}"
                    f"/{r.total})"
                )

            lines.append(
                f"  Average Steps        : "
                f"{r.avg_steps:5.1f}"
            )
            lines.append(
                f"  Loop Rate            : "
                f"{r.loop_rate:5.1f}%  "
                f"({r.loops}/{r.total})"
            )
            lines.append(
                f"  Error Rate           : "
                f"{r.error_rate:5.1f}%  "
                f"({r.errors}/{r.total})"
            )
            lines.append(
                f"  Avg Time/Query       : "
                f"{r.avg_time:5.1f}s"
            )

            # Difficulty breakdown
            if (
                r.easy_total > 0
                or r.medium_total > 0
                or r.hard_total > 0
            ):
                lines.append("")
                lines.append(
                    f"  Difficulty Breakdown:"
                )
                for diff, dt, dc in [
                    ("EASY", r.easy_total, r.easy_correct),
                    ("MEDIUM", r.medium_total, r.medium_correct),
                    ("HARD", r.hard_total, r.hard_correct),
                ]:
                    if dt > 0:
                        acc = dc / dt * 100
                        lines.append(
                            f"    {diff:8s}  "
                            f"n={dt:3d}  "
                            f"acc={acc:5.1f}%"
                        )

        # ── Architecture Health ──

        lines.append("")
        lines.append("=" * w)
        lines.append(
            "  ARCHITECTURE HEALTH "
            "(aggregated across all benchmarks)"
        )
        lines.append("=" * w)

        total_finished = sum(
            r.finished for r in reports.values()
        )
        total_loops = sum(
            r.loops for r in reports.values()
        )
        total_errors = sum(
            r.errors for r in reports.values()
        )
        total_steps = sum(
            r.total_steps for r in reports.values()
        )
        total_elapsed = sum(
            r.total_elapsed for r in reports.values()
        )

        if total_cases > 0:
            lines.append(
                f"  Overall Finish Rate  : "
                f"{total_finished / total_cases * 100:5.1f}%"
            )
            lines.append(
                f"  Overall Loop Rate    : "
                f"{total_loops / total_cases * 100:5.1f}%"
            )
            lines.append(
                f"  Overall Avg Steps    : "
                f"{total_steps / total_cases:5.1f}"
            )
            lines.append(
                f"  Overall Error Rate   : "
                f"{total_errors / total_cases * 100:5.1f}%"
            )
            lines.append(
                f"  Total Runtime        : "
                f"{total_elapsed:5.1f}s"
            )

        lines.append("=" * w)

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

        lines.append("")
        lines.append("=" * w)
        lines.append("  REFLEXION METRICS")
        lines.append("=" * w)
        lines.append(f"  Reflection Rate:               {global_reflection_rate:5.1f}%")
        lines.append(f"  Reflection Success Rate:       {global_reflection_success_rate:5.1f}%")
        lines.append(f"  Average Reflections per Query: {global_reflections_per_query:5.1f}")
        lines.append(f"  Most Common Reflection Type:   {global_most_common_refl}")
        lines.append("=" * w)

        # ── Incorrect samples ──

        all_flat = [
            r
            for results in all_results.values()
            for r in results
        ]
        incorrect = [
            r for r in all_flat
            if not r.answer_correct
            and not r.had_error
        ]

        if incorrect:
            lines.append("")
            lines.append("─" * w)
            lines.append(
                f"  SAMPLE INCORRECT ANSWERS "
                f"(first 15 of {len(incorrect)})"
            )
            lines.append("─" * w)

            for r in incorrect[:15]:
                lines.append(
                    f"  [{r.benchmark}] "
                    f"ID={r.case_id} "
                    f"({r.difficulty})"
                )
                lines.append(
                    f"    Q:    {r.query[:55]}"
                )
                lines.append(
                    f"    Pred: "
                    f"{r.normalized_predicted[:45]}"
                )
                lines.append(
                    f"    Gold: "
                    f"{r.normalized_ground_truth[:45]}"
                )
                lines.append("")

        lines.append("")

        return "\n".join(lines)

    # ─── Save ───

    def save_results(
        self,
        all_results: dict[str, list[AcademicRunResult]],
        filename: str,
    ):
        """Save raw results to JSON."""

        path = os.path.join(self.output_dir, filename)

        serializable = {}
        for bench, results in all_results.items():
            serializable[bench] = [
                asdict(r) for r in results
            ]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        print(f"\n  Results saved to: {path}")

    def save_report(
        self,
        report: str,
        filename: str,
    ):
        """Save the text report to a file."""

        path = os.path.join(self.output_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"  Report saved to:  {path}")


# ═══════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Planner Agent V3 — "
            "Academic Benchmark Evaluation"
        )
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        choices=["gsm8k", "hotpotqa", "strategyqa"],
        help=(
            "Run only a specific benchmark. "
            "Default: run all three."
        ),
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help=(
            "Number of examples to sample per "
            "benchmark. Default: 50."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling. Default: 42.",
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
        default=8,
        help=(
            "Max agent iterations per query. "
            "Default: 8 (higher than custom eval "
            "to accommodate multi-step reasoning)."
        ),
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

    # ── Load datasets ──
    print("  Loading academic benchmarks...\n")

    loader = AcademicBenchmarkLoader()

    benchmarks = (
        [args.benchmark] if args.benchmark else None
    )

    all_cases = loader.load(
        benchmarks=benchmarks,
        sample=args.sample,
        seed=args.seed,
    )

    total = sum(len(c) for c in all_cases.values())
    print(f"\n  Total cases to evaluate: {total}")

    # ── Run evaluation ──
    runner = AcademicEvaluationRunner(
        agent_factory=factory,
        output_dir=output_dir,
    )

    all_results = runner.run_all(all_cases)

    # ── Generate report ──
    report = runner.generate_report(all_results)
    print(report)

    # ── Save ──
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    runner.save_results(
        all_results,
        f"academic_results_{timestamp}.json",
    )
    runner.save_report(
        report,
        f"academic_report_{timestamp}.txt",
    )


if __name__ == "__main__":
    main()
