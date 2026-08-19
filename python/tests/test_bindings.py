import pytest

raginspect = pytest.importorskip("raginspect")


def test_profile_default():
    res = raginspect.profile({})
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
    res = raginspect.classify({})
    assert isinstance(res, dict)
    assert res["architecture"] == "naive"
    assert "confidence" in res
    assert isinstance(res["confidence"], (int, float))
    assert res["confidence"] >= 0.9


def test_classify_hyde():
    res = raginspect.classify({"hyde": {"enabled": True}})
    assert isinstance(res, dict)
    assert res["architecture"] == "hyde"

