class ContextSchema:
    """
    Represents the complete state of the agent at a given timestep.
    """

    def __init__(self, query: str):
        self.query = query

        self.thought_history = []
        self.action_history = []
        self.observation_history = []

        self.iteration_count = 0
        self.done = False

    # ------------------------
    # State Update Methods
    # ------------------------

    def add_thought(self, thought: str):
        self.thought_history.append(thought)

    def add_action(self, action):
        self.action_history.append(action)

    def add_observation(self, observation):
        self.observation_history.append(observation)

    def increment_iteration(self):
        self.iteration_count += 1

    def mark_done(self):
        self.done = True

    # ------------------------
    # Helper Methods
    # ------------------------

    def get_latest_thought(self):
        if not self.thought_history:
            return None

        return self.thought_history[-1]

    def get_latest_action(self):
        if not self.action_history:
            return None

        return self.action_history[-1]

    def get_latest_observation(self):
        if not self.observation_history:
            return None

        return self.observation_history[-1]

    # ------------------------
    # Serialization
    # ------------------------

    def to_dict(self):
        return {
            "query": self.query,
            "thought_history": self.thought_history,
            "action_history": self.action_history,
            "observation_history": self.observation_history,
            "iteration_count": self.iteration_count,
            "done": self.done
        }

    @classmethod
    def from_dict(cls, data: dict):
        context = cls(
            query=data.get("query", "")
        )

        context.thought_history = data.get(
            "thought_history",
            []
        )

        context.action_history = data.get(
            "action_history",
            []
        )

        context.observation_history = data.get(
            "observation_history",
            []
        )

        context.iteration_count = data.get(
            "iteration_count",
            0
        )

        context.done = data.get(
            "done",
            False
        )

        return context

    # ------------------------
    # Pretty Print
    # ------------------------

    def __str__(self):
        return (
            f"ContextSchema("
            f"query='{self.query}', "
            f"thoughts={len(self.thought_history)}, "
            f"actions={len(self.action_history)}, "
            f"observations={len(self.observation_history)}, "
            f"iteration_count={self.iteration_count}, "
            f"done={self.done}"
            f")"
        )
        