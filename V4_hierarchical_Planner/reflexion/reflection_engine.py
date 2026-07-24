import json
import re
import uuid

from reflexion.reflection_schema import ReflectionSchema


class ReflectionEngine:
    """
    Generates reflections from failed trajectories.

    Input:
        Query
        Action
        Observation
        Evaluation
        Context

    Output:
        ReflectionSchema
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def build_prompt(
        self,
        query,
        action,
        observation,
        evaluation,
        context
    ) -> str:

        prompt = f"""
You are an expert reflection engine for an autonomous AI agent.

Your task is to analyze failures and generate lessons.

================ USER QUERY ================
{query}

================ ACTION TAKEN ================
{action}

================ OBSERVATION ================
{observation}

================ EVALUATION ================
Progress Score:
{evaluation.progress_score}

Feedback:
{evaluation.feedback}

================ CONTEXT ================
{context}

Analyze:

1. What mistake did the agent make?
2. What lesson should be learned?
3. What should the agent do differently next time?

Return ONLY valid JSON.

Example:

{{
    "mistake": "Search query was too broad.",
    "lesson": "Broad queries often return generic information.",
    "recommendation": "Use entity specific searches."
}}
"""

        return prompt

    def parse_response(
        self,
        response: str
    ) -> ReflectionSchema:

        cleaned = response.strip()

        fence_match = re.search(
            r'```(?:json)?\s*(.*?)\s*```',
            cleaned,
            re.DOTALL
        )

        if fence_match:
            cleaned = fence_match.group(1).strip()

        data = json.loads(cleaned)

        return ReflectionSchema(
            mistake=data.get(
                "mistake",
                "Unknown mistake"
            ),
            lesson=data.get(
                "lesson",
                "No lesson generated."
            ),
            recommendation=data.get(
                "recommendation",
                "No recommendation generated."
            ),
            reflection_id=str(uuid.uuid4())
        )

    def reflect(
        self,
        query,
        action,
        observation,
        evaluation,
        context
    ) -> ReflectionSchema:

        prompt = self.build_prompt(
            query=query,
            action=action,
            observation=observation,
            evaluation=evaluation,
            context=context
        )

        response = self.llm_client.generate(
            prompt
        )

        reflection = self.parse_response(
            response
        )

        return reflection

    def __str__(self):
        return (
            f"ReflectionEngine("
            f"llm={self.llm_client.model_name}"
            f")"
        )

    def __repr__(self):
        return self.__str__()