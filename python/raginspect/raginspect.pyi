from typing import Any, Dict

def profile(
    pipeline: Dict[str, Any],
    query: str = "What is RAG?",
    inspect_mode: str = "full",
    architecture: str = "naive",
) -> Dict[str, Any]: ...

def classify(pipeline: Dict[str, Any]) -> Dict[str, Any]: ...
