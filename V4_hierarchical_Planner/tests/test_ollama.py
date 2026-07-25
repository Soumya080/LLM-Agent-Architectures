from agent.llm_client import LLMClient


def main():

    llm = LLMClient(
        model_name="qwen2.5-coder:7b"
    )

    print("Health Check:", llm.health_check())

    prompt = """
Return ONLY the following JSON exactly.

{
    "tool_name": "calculator",
    "parameters": {
        "operation": "add",
        "operands": [5, 10]
    },
    "reason": "User requested addition."
}
"""

    response = llm.generate(prompt)

    print("\nMODEL OUTPUT:\n")
    print(response)


if __name__ == "__main__":
    main()