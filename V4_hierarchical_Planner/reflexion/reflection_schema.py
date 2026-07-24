class ReflectionSchema:

    def __init__(
        self,
        mistake: str,
        lesson: str,
        recommendation: str,
        reflection_id: str
    ):
        self.mistake = mistake
        self.lesson = lesson
        self.recommendation = recommendation
        self.reflection_id = reflection_id

    def to_dict(self):
        return {
            "mistake": self.mistake,
            "lesson": self.lesson,
            "recommendation": self.recommendation,
            "reflection_id": self.reflection_id
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            mistake=data.get("mistake", ""),
            lesson=data.get("lesson", ""),
            recommendation=data.get("recommendation", ""),
            reflection_id=data.get("reflection_id", "")
        )

    def __str__(self):
        return (
            f"ReflectionSchema("
            f"mistake='{self.mistake}', "
            f"lesson='{self.lesson}', "
            f"recommendation='{self.recommendation}', "
            f"reflection_id='{self.reflection_id}'"
            f")"
        )

    def __repr__(self):
        return self.__str__()