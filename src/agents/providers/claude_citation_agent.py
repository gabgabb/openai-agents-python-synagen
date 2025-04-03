from src.agents.agent import Agent
from src.agents.providers.anthropic_provider import ask_with_citations

class ClaudeCitationAgent(Agent):
    def __init__(self, model: str = "claude-3-5-sonnet-latest"):
        self.model = model

    def run(self, task: str, context: dict, with_raw_response: bool = False) -> str:
        documents = context.get("documents", [])

        # Each document should have "title" and "data" fields
        try:
            return ask_with_citations(
                question=task,
                documents=documents,
                model=self.model,
                with_raw_response=with_raw_response
            )
        except Exception as e:
            return f"Claude agent failed: {e}"
