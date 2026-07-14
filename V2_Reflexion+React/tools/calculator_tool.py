import math

from tools.base_tool import BaseTool


class CalculatorTool(BaseTool):
    """
    Tool for performing basic arithmetic operations.
    Supported operations:
    - add
    - subtract
    - multiply
    - divide
    """

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Performs arithmetic: add, subtract, multiply, divide, sqrt, power."
        )

    def execute(self, parameters: dict):
        """
        Expected Parameters:

        {
            "operation": "add",
            "operands": [1, 2, 3]
        }
        """

        operation = parameters.get("operation")
        operands = parameters.get("operands")

        # Basic validation
        if operation is None:
            raise ValueError(
                "Missing required parameter: 'operation'"
            )

        if operands is None:
            raise ValueError(
                "Missing required parameter: 'operands'"
            )

        if not isinstance(operands, list):
            raise ValueError(
                "'operands' must be a list."
            )

        if len(operands) == 0:
            raise ValueError(
                "'operands' cannot be empty."
            )

        # Perform operation
        if operation == "add":
            return sum(operands)

        elif operation == "subtract":
            result = operands[0]

            for num in operands[1:]:
                result -= num

            return result

        elif operation == "multiply":
            result = 1

            for num in operands:
                result *= num

            return result

        elif operation == "divide":
            result = operands[0]

            for num in operands[1:]:
                if num == 0:
                    raise ValueError(
                        "Division by zero is not allowed."
                    )

                result /= num

            return result

        elif operation == "sqrt":
            return math.sqrt(operands[0])

        elif operation == "power":
            return operands[0] ** operands[1]

        else:
            raise ValueError(
                f"Unsupported operation '{operation}'."
            )