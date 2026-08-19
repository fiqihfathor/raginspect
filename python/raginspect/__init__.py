"""raginspect — RAG Inspection & Profiling Engine Python Package."""

from __future__ import annotations

from raginspect.profile import (
    ClassificationResult,
    InspectionReport,
    classify_pipeline,
    inspect_pipeline,
)
from raginspect.raginspect import classify, profile

__version__ = "0.1.0"

__all__ = [
    "profile",
    "classify",
    "inspect_pipeline",
    "classify_pipeline",
    "InspectionReport",
    "ClassificationResult",
    "__version__",
]
