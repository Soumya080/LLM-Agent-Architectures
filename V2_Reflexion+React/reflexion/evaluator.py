from reflexion.rule_evaluator import RuleEvaluator
from reflexion.llm_evaluator import LLMEvaluator
from reflexion.evaluation_schema import EvaluationSchema


class Evaluator:
    """
    Hybrid evaluator using:

    1. Rule-based evaluation
    2. LLM-based evaluation (fallback)
    """

    def __init__(
        self,
        llm_client=None,
        confidence_threshold: float = 0.6
    ):
        self.rule_evaluator = RuleEvaluator()

        self.llm_evaluator = (
            LLMEvaluator(llm_client)
            if llm_client is not None
            else None
        )

        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        query,
        action,
        observation,
        context
    ) -> EvaluationSchema:

        # -----------------------------------------
        # First pass: rule evaluator
        # -----------------------------------------
        evaluation = self.rule_evaluator.evaluate(
            query=query,
            action=action,
            observation=observation,
            context=context
        )

        # -----------------------------------------
        # Fallback to LLM evaluator if confidence low
        # -----------------------------------------
        if (
            evaluation.confidence
            < self.confidence_threshold
            and self.llm_evaluator is not None
        ):

            evaluation = self.llm_evaluator.evaluate(
                query=query,
                action=action,
                observation=observation,
                context=context
            )

        return evaluation

    def __str__(self):
        return (
            f"Evaluator("
            f"rule_evaluator=True, "
            f"llm_evaluator={self.llm_evaluator is not None}, "
            f"threshold={self.confidence_threshold}"
            f")"
        )

    def __repr__(self):
        return self.__str__()