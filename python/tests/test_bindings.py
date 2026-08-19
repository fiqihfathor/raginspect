import pytest

raginspect = pytest.importorskip("raginspect")
from raginspect import (
    ClassificationResult,
    InspectionReport,
    classify,
    classify_pipeline,
    inspect_pipeline,
    profile,
)


def test_profile_default():
    res = profile({})
    assert isinstance(res, dict)
    expected_keys = {
        "query",
        "overall_score",
        "retrieval",
        "context",
        "generation",
        "recommendations",
    }
    assert expected_keys.issubset(res.keys())
    assert res["query"] == "What is RAG?"


def test_classify_default():
    res = classify({})
    assert isinstance(res, dict)
    assert res["architecture"] == "naive"
    assert "confidence" in res
    assert isinstance(res["confidence"], (int, float))
    assert res["confidence"] >= 0.9


def test_classify_hyde():
    res = classify({"hyde": {"enabled": True}})
    assert isinstance(res, dict)
    assert res["architecture"] == "hyde"


def test_inspect_pipeline():
    report = inspect_pipeline({})
    assert isinstance(report, InspectionReport)
    assert 0.0 <= report.overall_score <= 100.0
    assert report.query == "What is RAG?"
    assert report.architecture == "naive"
    assert report.inspect_mode == "full"
    summary_str = report.summary()
    assert isinstance(summary_str, str)
    assert "score=" in summary_str
    assert "architecture=" in summary_str
    assert "recommendations=" in summary_str


def test_classify_pipeline():
    res = classify_pipeline({"hyde": {"enabled": True}})
    assert isinstance(res, ClassificationResult)
    assert res.architecture == "hyde"
    assert isinstance(res.confidence, float)
    assert res.confidence >= 0.8
    assert isinstance(res.reason, str)
    assert isinstance(res.scores, list)
