"""
Answer Evaluators for Academic Benchmarks

Provides three evaluation strategies matching
academic standards:

1. NumericMatcher  — GSM8K (exact numeric match)
2. F1Evaluator     — HotpotQA (token F1 + EM)
3. BooleanMatcher  — StrategyQA (yes/no accuracy)

Reference:
    F1/EM implementation follows the SQuAD 2.0
    evaluation script conventions.
"""

import re
import string
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════
# Score Container
# ═══════════════════════════════════════════

@dataclass
class EvalScore:
    """
    Container for evaluation scores.

    Fields:
        correct:        Whether the answer is correct
        score:          Float score (0.0 to 1.0)
        predicted:      Normalized predicted answer
        ground_truth:   Normalized ground truth answer
        method:         Evaluation method used
    """
    correct: bool
    score: float
    predicted: str
    ground_truth: str
    method: str


# ═══════════════════════════════════════════
# 1. Numeric Exact Match (GSM8K)
# ═══════════════════════════════════════════

class NumericMatcher:
    """
    Evaluates numeric answers for GSM8K.

    Strategy:
        1. Extract all numbers from the agent's
           final answer.
        2. Compare against the ground truth number.
        3. Match if any extracted number equals
           ground truth within tolerance.

    Handles:
        - Commas in numbers (1,234 → 1234)
        - Dollar signs ($50 → 50)
        - Percentage signs (25% → 25)
        - Negative numbers (-5.5)
        - Decimal numbers (3.14)
    """

    TOLERANCE = 1e-6

    @staticmethod
    def extract_numbers(text: str) -> list[float]:
        """
        Extract all numeric values from a text string.

        Examples:
            "The answer is 18"         → [18.0]
            "I got $1,234.56"          → [1234.56]
            "Steps: 3, result: -42"    → [3.0, -42.0]
        """

        if text is None:
            return []

        # Remove commas inside numbers (1,234 → 1234)
        cleaned = re.sub(
            r'(\d),(\d)', r'\1\2', text
        )

        # Remove $ and % signs adjacent to numbers
        cleaned = cleaned.replace("$", "")
        cleaned = cleaned.replace("%", "")

        # Find all numbers (int or float, optional sign)
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, cleaned)

        numbers = []

        for m in matches:
            try:
                numbers.append(float(m))
            except ValueError:
                continue

        return numbers

    @classmethod
    def evaluate(
        cls,
        predicted: Optional[str],
        ground_truth: str,
    ) -> EvalScore:
        """
        Evaluate a predicted answer against
        GSM8K ground truth.

        Args:
            predicted:    Agent's final answer string
            ground_truth: The number after "####"

        Returns:
            EvalScore with match result.
        """

        if predicted is None:
            return EvalScore(
                correct=False,
                score=0.0,
                predicted="<no answer>",
                ground_truth=ground_truth,
                method="numeric_exact_match",
            )

        # Parse ground truth
        try:
            gt_num = float(
                ground_truth.replace(",", "")
            )
        except ValueError:
            return EvalScore(
                correct=False,
                score=0.0,
                predicted=predicted,
                ground_truth=ground_truth,
                method="numeric_exact_match",
            )

        # Extract numbers from prediction
        pred_numbers = cls.extract_numbers(predicted)

        if not pred_numbers:
            return EvalScore(
                correct=False,
                score=0.0,
                predicted=predicted,
                ground_truth=ground_truth,
                method="numeric_exact_match",
            )

        # Check if any extracted number matches
        for num in pred_numbers:
            if abs(num - gt_num) < cls.TOLERANCE:
                return EvalScore(
                    correct=True,
                    score=1.0,
                    predicted=str(num),
                    ground_truth=ground_truth,
                    method="numeric_exact_match",
                )

        # Check last number (often the final answer)
        last_num = pred_numbers[-1]

        return EvalScore(
            correct=False,
            score=0.0,
            predicted=str(last_num),
            ground_truth=ground_truth,
            method="numeric_exact_match",
        )


# ═══════════════════════════════════════════
# 2. Token F1 + Exact Match (HotpotQA)
# ═══════════════════════════════════════════

class F1Evaluator:
    """
    Evaluates free-text answers for HotpotQA.

    Implements the standard SQuAD-style metrics:
        - Exact Match (EM): 1.0 if normalized
          strings are identical, else 0.0
        - Token F1: precision/recall over word tokens

    Normalization:
        1. Lowercase
        2. Remove punctuation
        3. Remove articles (a, an, the)
        4. Collapse whitespace
    """

    ARTICLES = {"a", "an", "the"}

    @classmethod
    def normalize(cls, text: str) -> str:
        """
        Normalize text for comparison.

        Steps:
            1. Lowercase
            2. Remove punctuation
            3. Remove articles
            4. Collapse whitespace
        """

        if text is None:
            return ""

        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )

        # Remove articles
        tokens = text.split()
        tokens = [
            t for t in tokens
            if t not in cls.ARTICLES
        ]

        # Collapse whitespace
        return " ".join(tokens).strip()

    @classmethod
    def compute_f1(
        cls,
        predicted: str,
        ground_truth: str,
    ) -> float:
        """
        Compute token-level F1 score.

        Returns:
            Float between 0.0 and 1.0.
        """

        pred_tokens = predicted.split()
        gt_tokens = ground_truth.split()

        if not pred_tokens and not gt_tokens:
            return 1.0

        if not pred_tokens or not gt_tokens:
            return 0.0

        # Common tokens
        common = set(pred_tokens) & set(gt_tokens)
        num_common = sum(
            min(
                pred_tokens.count(t),
                gt_tokens.count(t),
            )
            for t in common
        )

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

        return f1

    @classmethod
    def compute_em(
        cls,
        predicted: str,
        ground_truth: str,
    ) -> float:
        """
        Compute Exact Match score.

        Returns:
            1.0 if normalized strings match, else 0.0.
        """
        return 1.0 if predicted == ground_truth else 0.0

    @classmethod
    def evaluate(
        cls,
        predicted: Optional[str],
        ground_truth: str,
    ) -> EvalScore:
        """
        Evaluate a predicted answer against
        HotpotQA ground truth.

        Args:
            predicted:    Agent's final answer string
            ground_truth: Expected answer string

        Returns:
            EvalScore with F1 as score, EM as correct.
        """

        if predicted is None:
            return EvalScore(
                correct=False,
                score=0.0,
                predicted="<no answer>",
                ground_truth=ground_truth,
                method="f1_em",
            )

        norm_pred = cls.normalize(predicted)
        norm_gt = cls.normalize(ground_truth)

        em = cls.compute_em(norm_pred, norm_gt)
        f1 = cls.compute_f1(norm_pred, norm_gt)

        return EvalScore(
            correct=(em == 1.0),
            score=f1,
            predicted=norm_pred,
            ground_truth=norm_gt,
            method="f1_em",
        )


# ═══════════════════════════════════════════
# 3. Boolean Accuracy (StrategyQA)
# ═══════════════════════════════════════════

class BooleanMatcher:
    """
    Evaluates yes/no answers for StrategyQA.

    Strategy:
        1. Normalize agent answer to "yes" or "no".
        2. Binary match against ground truth.

    Recognized patterns:
        YES: "yes", "true", "correct", "right",
             "affirmative", "definitely"
        NO:  "no", "false", "incorrect", "wrong",
             "negative", "not"
    """

    YES_PATTERNS = {
        "yes", "true", "correct", "right",
        "affirmative", "definitely", "absolutely",
        "indeed", "certainly",
    }

    NO_PATTERNS = {
        "no", "false", "incorrect", "wrong",
        "negative", "not", "nope", "neither",
    }

    @classmethod
    def normalize_to_bool(
        cls, text: str
    ) -> Optional[str]:
        """
        Normalize a text answer to "yes" or "no".

        Returns:
            "yes", "no", or None if unrecognizable.
        """

        if text is None:
            return None

        cleaned = text.lower().strip()

        # Remove punctuation
        cleaned = cleaned.translate(
            str.maketrans("", "", string.punctuation)
        )

        # Check first word
        first_word = cleaned.split()[0] if cleaned else ""

        if first_word in cls.YES_PATTERNS:
            return "yes"
        if first_word in cls.NO_PATTERNS:
            return "no"

        # Check if any yes/no keyword appears
        words = set(cleaned.split())

        yes_hits = words & cls.YES_PATTERNS
        no_hits = words & cls.NO_PATTERNS

        if yes_hits and not no_hits:
            return "yes"
        if no_hits and not yes_hits:
            return "no"

        # Ambiguous: check for "yes" or "no" substring
        if "yes" in cleaned:
            return "yes"
        if "no" in cleaned:
            return "no"

        return None

    @classmethod
    def evaluate(
        cls,
        predicted: Optional[str],
        ground_truth: str,
    ) -> EvalScore:
        """
        Evaluate a predicted yes/no answer.

        Args:
            predicted:    Agent's final answer string
            ground_truth: "yes" or "no"

        Returns:
            EvalScore with binary match.
        """

        if predicted is None:
            return EvalScore(
                correct=False,
                score=0.0,
                predicted="<no answer>",
                ground_truth=ground_truth,
                method="boolean_match",
            )

        norm_pred = cls.normalize_to_bool(predicted)

        if norm_pred is None:
            return EvalScore(
                correct=False,
                score=0.0,
                predicted=predicted[:50],
                ground_truth=ground_truth,
                method="boolean_match",
            )

        correct = (norm_pred == ground_truth)

        return EvalScore(
            correct=correct,
            score=1.0 if correct else 0.0,
            predicted=norm_pred,
            ground_truth=ground_truth,
            method="boolean_match",
        )


# ═══════════════════════════════════════════
# Evaluator Registry
# ═══════════════════════════════════════════

EVALUATORS = {
    "gsm8k": NumericMatcher,
    "hotpotqa": F1Evaluator,
    "strategyqa": BooleanMatcher,
}


def get_evaluator(benchmark: str):
    """
    Get the appropriate evaluator class
    for a given benchmark.

    Args:
        benchmark: "gsm8k" | "hotpotqa" | "strategyqa"

    Returns:
        Evaluator class with an .evaluate() method.

    Raises:
        ValueError if benchmark is unknown.
    """

    if benchmark not in EVALUATORS:
        raise ValueError(
            f"Unknown benchmark: '{benchmark}'. "
            f"Available: {list(EVALUATORS.keys())}"
        )

    return EVALUATORS[benchmark]
