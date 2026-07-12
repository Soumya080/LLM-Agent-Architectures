from reflexion.evaluation_schema import EvaluationSchema


class RuleEvaluator:
    """
    Simple rule-based evaluator for Reflexion V1.

    Responsible for answering:

    - Did the previous action help?
    - Did the agent make progress?
    - Should reflection be triggered?
    """

    def evaluate(
        self,
        query: str,
        action,
        observation,
        context
    ) -> EvaluationSchema:

        score = 0.0
        feedback = ""
        should_reflect = False

        # --------------------------------------------------
        # Rule 1 : Tool execution failed
        # --------------------------------------------------
        if not observation.success:

            score = 0.0
            should_reflect = True
            feedback = (
                "Tool execution failed. "
                "Reflection recommended."
            )

        # --------------------------------------------------
        # Rule 2 : Finish tool called
        # --------------------------------------------------
        elif action.tool_name.lower() == "finish":

            score = 1.0
            should_reflect = False
            feedback = (
                "Task marked as complete."
            )

        # --------------------------------------------------
        # Rule 3 : Empty result returned
        # --------------------------------------------------
        elif (
            observation.result is None
            or observation.result == ""
        ):

            score = 0.2
            should_reflect = True
            feedback = (
                "Observation returned no useful information."
            )

        # --------------------------------------------------
        # Rule 4 : Same tool repeatedly used
        # --------------------------------------------------
        elif len(context.action_history) >= 2:

            previous_action = context.action_history[-2]

            if (
                previous_action.tool_name
                == action.tool_name
            ):

                score = 0.3
                should_reflect = True
                feedback = (
                    "Repeated tool usage detected. "
                    "Possible loop behavior."
                )

            else:

                score = 0.7
                should_reflect = False
                feedback = (
                    "Action produced useful progress."
                )

        # --------------------------------------------------
        # Rule 5 : Default successful progress
        # --------------------------------------------------
        else:

            score = 0.7
            should_reflect = False
            feedback = (
                "Action produced useful progress."
            )

        return EvaluationSchema(
            success=(score >= 0.5),
            progress_score=score,
            feedback=feedback,
            should_reflect=should_reflect
        )
        
        