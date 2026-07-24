class EvaluationSchema:
    """
    Stores evaluation information for a single agent step.
    """

    def __init__(
        self,
        success: bool,
        progress_score: float,
        feedback: str,
        should_reflect: bool,
        confidence: float = 1.0
    ):
        self.success = success
        self.progress_score = progress_score
        self.feedback = feedback
        self.should_reflect = should_reflect
        self.confidence = confidence

    def to_dict(self):
        return {
            "success": self.success,
            "progress_score": self.progress_score,
            "feedback": self.feedback,
            "should_reflect": self.should_reflect,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            success=data.get("success", False),
            progress_score=data.get("progress_score", 0.0),
            feedback=data.get("feedback", ""),
            should_reflect=data.get("should_reflect", False),
            confidence=data.get("confidence", 1.0)
        )

    def __str__(self):
        return (
            f"EvaluationSchema("
            f"success={self.success}, "
            f"progress_score={self.progress_score}, "
            f"should_reflect={self.should_reflect}, "
            f"confidence={self.confidence}, "
            f"feedback='{self.feedback}'"
            f")"
        )

    def __repr__(self):
        return self.__str__()