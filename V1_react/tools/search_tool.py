from tools.base_tool import BaseTool
from ddgs import DDGS


class SearchTool(BaseTool):

    def __init__(self):
        super().__init__(
            name="search",
            description="Searches for external information."
        )

    def execute(self, parameters: dict):

        query = parameters.get(
            "query"
        )

        validated_query = self.validate_query(
            query
        )

        results = self.perform_search(
            validated_query
        )

        return results

    def validate_query(
        self,
        query: str
    ) -> str:

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if not isinstance(
            query,
            str
        ):
            raise ValueError(
                "Query must be a string."
            )

        return query.strip()

    def perform_search(
        self,
        query: str
    ):

        with DDGS() as ddgs:
            results = list(
                ddgs.text(query, max_results=3)
            )

            if not results:
                return "No results found."

            formatted = []

            for r in results:
                title = r.get("title", "N/A")
                body = r.get("body", "N/A")

                # Truncate long snippets
                if len(body) > 200:
                    body = body[:200] + "..."

                formatted.append(
                    f"- {title}: {body}"
                )

            return "\n".join(formatted)