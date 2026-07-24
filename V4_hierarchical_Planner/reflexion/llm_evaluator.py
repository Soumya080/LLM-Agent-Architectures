import json
import re

from reflexion.evaluation_schema import EvaluationSchema


class LLMEvaluator:

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def build_prompt(
        self,
        query,
        action,
        observation,
        context
    ):

        return f"""
You are an evaluator for an autonomous AI agent.

User Query:
{query}

Action Taken:
{action}

Observation Received:
{observation}

Context:
{context}

Evaluate whether the agent made progress.

Return ONLY valid JSON:

{{
    "success": true,
    "progress_score": 0.75,
    "feedback": "The observation moved the agent closer to solving the task.",
    "should_reflect": false,
    "confidence": 0.85
}}
"""

    def parse_response(
        self,
        response: str
    ) -> EvaluationSchema:

        cleaned = response.strip()

        fence_match = re.search(
            r'```(?:json)?\s*(.*?)\s*```',
            cleaned,
            re.DOTALL
        )

        if fence_match:
            cleaned = fence_match.group(1).strip()

        data = json.loads(cleaned)

        return EvaluationSchema(
            success=data.get("success", False),
            progress_score=data.get("progress_score", 0.0),
            feedback=data.get("feedback", ""),
            should_reflect=data.get(
                "should_reflect",
                False
            ),
            confidence=data.get(
                "confidence",
                0.5
            )
        )

    def evaluate(
        self,
        query,
        action,
        observation,
        context
    ) -> EvaluationSchema:

        prompt = self.build_prompt(
            query,
            action,
            observation,
            context
        )

        response = self.llm_client.generate(
            prompt
        )

        return self.parse_response(
            response
        )