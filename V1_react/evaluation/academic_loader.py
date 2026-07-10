"""
Academic Benchmark Dataset Loaders

Loads GSM8K, HotpotQA, and StrategyQA from HuggingFace
and converts them into a standard format compatible
with the agent evaluation runner.

Requirements:
    pip install datasets
"""

import re
import random
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════
# Data Class
# ═══════════════════════════════════════════

@dataclass
class AcademicCase:
    """
    A single academic benchmark test case.

    Fields:
        id:             Unique identifier within the benchmark
        benchmark:      "gsm8k" | "hotpotqa" | "strategyqa"
        query:          The question string sent to the agent
        ground_truth:   The correct answer (cleaned string)
        raw_answer:     The original answer from the dataset
        difficulty:     "easy" | "medium" | "hard"
        metadata:       Extra fields (question type, etc.)
    """
    id: int
    benchmark: str
    query: str
    ground_truth: str
    raw_answer: str
    difficulty: str = "medium"
    metadata: Optional[dict] = None


# ═══════════════════════════════════════════
# GSM8K Loader
# ═══════════════════════════════════════════

class GSM8KLoader:
    """
    Loads the GSM8K (Grade School Math 8K) dataset.

    What it tests:
        Multi-step arithmetic reasoning.
        The agent must decompose a word problem into
        a sequence of calculator operations.

    Expected tool chain:
        calculator → ... → finish

    HuggingFace:
        openai/gsm8k, split="test" (1,319 examples)

    Answer format:
        "Step-by-step reasoning\\n#### <final_number>"
        We extract the number after "####".
    """

    DATASET_NAME = "openai/gsm8k"
    DATASET_CONFIG = "main"
    SPLIT = "test"

    @staticmethod
    def extract_numeric_answer(raw_answer: str) -> str:
        """
        Extract the final numeric answer from
        GSM8K's "#### <number>" format.

        Examples:
            "...#### 18"       -> "18"
            "...#### 1,234"    -> "1234"
            "...#### -5.5"     -> "-5.5"
        """

        match = re.search(
            r'####\s*(.+?)$',
            raw_answer,
            re.MULTILINE
        )

        if match:
            # Remove commas from numbers like "1,234"
            return match.group(1).strip().replace(",", "")

        return raw_answer.strip()

    @staticmethod
    def classify_difficulty(answer: str) -> str:
        """
        Heuristic difficulty based on the magnitude
        and complexity of the answer.
        """
        try:
            num = float(answer)

            if abs(num) <= 100 and num == int(num):
                return "easy"
            elif abs(num) <= 10000:
                return "medium"
            else:
                return "hard"
        except ValueError:
            return "medium"

    def load(
        self,
        sample: Optional[int] = None,
        seed: int = 42,
    ) -> list[AcademicCase]:
        """
        Load GSM8K test set.

        Args:
            sample: Number of examples to sample.
                    None = load all.
            seed:   Random seed for reproducibility.

        Returns:
            List of AcademicCase objects.
        """
        from datasets import load_dataset

        print(
            f"  Loading GSM8K ({self.SPLIT})..."
        )

        ds = load_dataset(
            self.DATASET_NAME,
            self.DATASET_CONFIG,
            split=self.SPLIT,
            trust_remote_code=True,
        )

        indices = list(range(len(ds)))

        if sample is not None and sample < len(ds):
            rng = random.Random(seed)
            indices = rng.sample(indices, sample)

        cases = []

        for idx in indices:
            example = ds[idx]

            raw_answer = example["answer"]
            ground_truth = self.extract_numeric_answer(
                raw_answer
            )

            difficulty = self.classify_difficulty(
                ground_truth
            )

            case = AcademicCase(
                id=idx,
                benchmark="gsm8k",
                query=example["question"],
                ground_truth=ground_truth,
                raw_answer=raw_answer,
                difficulty=difficulty,
                metadata={
                    "reasoning_steps": raw_answer.count("\n"),
                },
            )
            cases.append(case)

        print(
            f"  ✓ GSM8K loaded: {len(cases)} cases"
        )

        return cases


# ═══════════════════════════════════════════
# HotpotQA Loader
# ═══════════════════════════════════════════

class HotpotQALoader:
    """
    Loads the HotpotQA dataset.

    What it tests:
        Multi-hop factual retrieval.
        The agent must search for information across
        multiple sources and synthesize an answer.

    Expected tool chain:
        search → ... → finish

    HuggingFace:
        hotpotqa/hotpot_qa (distractor), split="validation"
        (7,405 examples)

    Answer format:
        Short free-text answer string.
    """

    DATASET_NAME = "hotpotqa/hotpot_qa"
    DATASET_CONFIG = "distractor"
    SPLIT = "validation"

    def load(
        self,
        sample: Optional[int] = None,
        seed: int = 42,
    ) -> list[AcademicCase]:
        """
        Load HotpotQA validation set.

        Args:
            sample: Number of examples to sample.
            seed:   Random seed for reproducibility.

        Returns:
            List of AcademicCase objects.
        """
        from datasets import load_dataset

        print(
            f"  Loading HotpotQA ({self.SPLIT})..."
        )

        ds = load_dataset(
            self.DATASET_NAME,
            self.DATASET_CONFIG,
            split=self.SPLIT,
            trust_remote_code=True,
        )

        indices = list(range(len(ds)))

        if sample is not None and sample < len(ds):
            rng = random.Random(seed)
            indices = rng.sample(indices, sample)

        cases = []

        for idx in indices:
            example = ds[idx]

            # Map HotpotQA levels to our difficulty
            level = example.get("level", "medium")
            difficulty = level.lower()
            if difficulty not in (
                "easy", "medium", "hard"
            ):
                difficulty = "medium"

            question_type = example.get("type", "unknown")

            case = AcademicCase(
                id=idx,
                benchmark="hotpotqa",
                query=example["question"],
                ground_truth=example["answer"],
                raw_answer=example["answer"],
                difficulty=difficulty,
                metadata={
                    "type": question_type,
                    "level": level,
                },
            )
            cases.append(case)

        print(
            f"  ✓ HotpotQA loaded: {len(cases)} cases"
        )

        return cases


# ═══════════════════════════════════════════
# StrategyQA Loader
# ═══════════════════════════════════════════

class StrategyQALoader:
    """
    Loads the StrategyQA dataset.

    What it tests:
        Implicit multi-step yes/no reasoning.
        The agent must decompose a question that
        requires world knowledge into sub-questions.

    Expected tool chain:
        search → finish

    HuggingFace:
        ChilleD/StrategyQA, split="test"
        (2,290 examples)

    Answer format:
        Boolean (True/False) → converted to "yes"/"no".
    """

    DATASET_NAME = "ChilleD/StrategyQA"
    SPLIT = "test"

    @staticmethod
    def bool_to_yesno(answer) -> str:
        """Convert boolean answer to yes/no string."""
        if isinstance(answer, bool):
            return "yes" if answer else "no"
        if isinstance(answer, str):
            return answer.strip().lower()

        # Fallback: truthy check
        return "yes" if answer else "no"

    @staticmethod
    def classify_difficulty(question: str) -> str:
        """
        Heuristic difficulty based on question
        complexity (word count).
        """

        word_count = len(question.split())

        if word_count <= 8:
            return "easy"
        elif word_count <= 15:
            return "medium"
        else:
            return "hard"

    def load(
        self,
        sample: Optional[int] = None,
        seed: int = 42,
    ) -> list[AcademicCase]:
        """
        Load StrategyQA test set.

        Args:
            sample: Number of examples to sample.
            seed:   Random seed for reproducibility.

        Returns:
            List of AcademicCase objects.
        """
        from datasets import load_dataset

        print(
            f"  Loading StrategyQA ({self.SPLIT})..."
        )

        ds = load_dataset(
            self.DATASET_NAME,
            split=self.SPLIT,
            trust_remote_code=True,
        )

        indices = list(range(len(ds)))

        if sample is not None and sample < len(ds):
            rng = random.Random(seed)
            indices = rng.sample(indices, sample)

        cases = []

        for idx in indices:
            example = ds[idx]

            raw = example.get("answer", False)
            ground_truth = self.bool_to_yesno(raw)

            question = example["question"]
            difficulty = self.classify_difficulty(
                question
            )

            case = AcademicCase(
                id=idx,
                benchmark="strategyqa",
                query=question,
                ground_truth=ground_truth,
                raw_answer=str(raw),
                difficulty=difficulty,
                metadata={},
            )
            cases.append(case)

        print(
            f"  ✓ StrategyQA loaded: {len(cases)} cases"
        )

        return cases


# ═══════════════════════════════════════════
# Unified Loader
# ═══════════════════════════════════════════

class AcademicBenchmarkLoader:
    """
    Unified interface to load any combination
    of academic benchmarks.
    """

    LOADERS = {
        "gsm8k": GSM8KLoader,
        "hotpotqa": HotpotQALoader,
        "strategyqa": StrategyQALoader,
    }

    def load(
        self,
        benchmarks: Optional[list[str]] = None,
        sample: Optional[int] = None,
        seed: int = 42,
    ) -> dict[str, list[AcademicCase]]:
        """
        Load one or more academic benchmarks.

        Args:
            benchmarks: List of benchmark names.
                        None = load all.
            sample:     Per-benchmark sample size.
            seed:       Random seed.

        Returns:
            Dict mapping benchmark name → cases.
        """

        if benchmarks is None:
            benchmarks = list(self.LOADERS.keys())

        results = {}

        for name in benchmarks:
            if name not in self.LOADERS:
                print(
                    f"  [WARN] Unknown benchmark: "
                    f"'{name}'. Skipping."
                )
                continue

            loader_cls = self.LOADERS[name]
            loader = loader_cls()
            results[name] = loader.load(
                sample=sample, seed=seed
            )

        return results
