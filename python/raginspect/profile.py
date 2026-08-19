"""High-level API for raginspect pipeline inspection and classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from raginspect.raginspect import classify as _raw_classify
from raginspect.raginspect import profile as _raw_profile


@dataclass
class InspectionReport:
    """RAG pipeline inspection report containing metrics across all pipeline stages.

    Attributes:
        query: Target query evaluated during inspection.
        config_name: Name of the pipeline configuration.
        model_name: Name of the LLM or embedding model evaluated.
        architecture: RAG architecture type evaluated (e.g. 'naive', 'advanced').
        inspect_mode: Granularity mode of inspection ('full', 'retrieval', etc.).
        timestamp: ISO 8601 timestamp when inspection was performed.
        overall_score: Overall RAG health score (0.0 to 100.0).
        retrieval: Metrics dictionary for the vector retrieval stage.
        context: Metrics dictionary for the context construction stage.
        generation: Metrics dictionary for the generation and grounding stage.
        recommendations: List of actionable diagnostic recommendations.
    """

    query: str
    config_name: str
    model_name: str
    architecture: str
    inspect_mode: str
    timestamp: str
    overall_score: float
    retrieval: Dict[str, Any]
    context: Dict[str, Any]
    generation: Dict[str, Any]
    recommendations: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InspectionReport:
        """Construct an InspectionReport from a raw dictionary.

        Args:
            data: Raw dictionary returned by the extension profile() function.

        Returns:
            InspectionReport instance with typed fields.
        """
        return cls(
            query=data.get("query", ""),
            config_name=data.get("config_name", ""),
            model_name=data.get("model_name", ""),
            architecture=str(data.get("architecture", "")),
            inspect_mode=str(data.get("inspect_mode", "")),
            timestamp=data.get("timestamp", ""),
            overall_score=float(data.get("overall_score", 0.0)),
            retrieval=data.get("retrieval", {}),
            context=data.get("context", {}),
            generation=data.get("generation", {}),
            recommendations=data.get("recommendations", []),
        )

    def summary(self) -> str:
        """Return a one-line summary string of the inspection report.

        Returns:
            Formatted summary string containing overall score, architecture, and recommendation count.

        Example:
            >>> report.summary()
            'score=72.5 architecture=naive recommendations=5'
        """
        return f"score={self.overall_score} architecture={self.architecture} recommendations={len(self.recommendations)}"


@dataclass
class ClassificationResult:
    """Result of RAG pipeline architecture classification.

    Attributes:
        architecture: Classified architecture name (e.g. 'naive', 'advanced', 'agentic', 'hyde').
        confidence: Classification confidence score between 0.0 and 1.0.
        reason: Diagnostic explanation for the architecture classification.
        scores: List of component or architecture confidence scores.
    """

    architecture: str
    confidence: float
    reason: str
    scores: List[Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClassificationResult:
        """Construct a ClassificationResult from a raw dictionary.

        Args:
            data: Raw dictionary returned by the extension classify() function.

        Returns:
            ClassificationResult instance with typed fields.
        """
        arch = str(data.get("architecture", "naive"))
        conf = float(data.get("confidence", 1.0))
        reason = data.get("reason")
        if reason is None:
            reason = f"Classified pipeline as '{arch}' architecture with confidence {conf:.2f}."
        scores = data.get("scores", [])
        return cls(
            architecture=arch,
            confidence=conf,
            reason=str(reason),
            scores=scores if isinstance(scores, list) else [],
        )


def inspect_pipeline(
    pipeline: Dict[str, Any],
    query: str = "What is RAG?",
    mode: str = "full",
    architecture: Optional[str] = None,
) -> InspectionReport:
    """Inspect a RAG pipeline configuration and return a detailed diagnostic report.

    Args:
        pipeline: RAG pipeline configuration dictionary.
        query: Query string to evaluate against the pipeline. Defaults to "What is RAG?".
        mode: Granularity mode for inspection ('full', 'retrieval', 'context', 'quick'). Defaults to "full".
        architecture: Optional target architecture override ('naive', 'advanced', etc.). Defaults to None ("naive").

    Returns:
        InspectionReport wrapping the diagnostic results.

    Raises:
        ValueError: If pipeline configuration or inspection parameters are invalid.

    Example:
        >>> report = inspect_pipeline({}, query="What is RAG?", mode="full")
        >>> print(report.overall_score)
        88.5
        >>> print(report.summary())
        score=88.5 architecture=naive recommendations=1
    """
    arch_arg = architecture if architecture is not None else "naive"
    raw_res = _raw_profile(
        pipeline=pipeline,
        query=query,
        inspect_mode=mode,
        architecture=arch_arg,
    )
    return InspectionReport.from_dict(raw_res)


def classify_pipeline(pipeline: Dict[str, Any]) -> ClassificationResult:
    """Classify the architecture type of a RAG pipeline configuration.

    Args:
        pipeline: RAG pipeline configuration dictionary.

    Returns:
        ClassificationResult containing predicted architecture, confidence, reason, and scores.

    Raises:
        ValueError: If pipeline configuration is invalid.

    Example:
        >>> result = classify_pipeline({"hyde": {"enabled": True}})
        >>> print(result.architecture, result.confidence)
        hyde 0.9
    """
    raw_res = _raw_classify(pipeline)
    return ClassificationResult.from_dict(raw_res)
