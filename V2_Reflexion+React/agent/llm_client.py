import requests


class LLMClient:
    """
    Handles communication with the local Ollama server.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str) -> str:
        """
        Send prompt to Ollama and return model response.
        """

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload
        )

        response.raise_for_status()

        result = response.json()

        return result["response"]

    def health_check(self) -> bool:
        """
        Check whether Ollama server is reachable.
        """

        try:
            response = requests.get(
                f"{self.base_url}/api/tags"
            )

            return response.status_code == 200

        except Exception:
            return False

    def __str__(self):
        return (
            f"LLMClient("
            f"model='{self.model_name}', "
            f"base_url='{self.base_url}')"
        )
        


if __name__ == "__main__":
    llm = LLMClient(
        model_name="qwen2.5-coder:7b"
    )


    response = llm.generate(
        "What is the capital of Japan?"
    )

    print(response)