"""CI QUALITY GATE: the build fails if retrieval quality regresses below these thresholds.
Thresholds are for the lite embedder; raise them when CI runs with the clip backend."""
from app.evaluation.retrieval_eval import evaluate


def test_retrieval_quality_gate(engine):
    report = evaluate(systems=["hybrid", "agent"])["systems"]
    assert report["hybrid"]["nDCG@10"] >= 0.70, report
    assert report["agent"]["nDCG@10"] >= 0.80, report
    assert report["agent"]["MRR"] >= 0.90, report


def test_image_search_quality_gate(engine):
    """Indexing every photo (MaxSim) must beat one photo per boot on held-out shopper photos."""
    from app.evaluation.image_eval import evaluate as evaluate_images
    shopper = evaluate_images(protocols=("shopper",))["protocols"]["shopper"]["strategies"]
    assert shopper["all_max"]["product@1"] >= 0.70, shopper
    assert shopper["all_max"]["MRR"] > shopper["primary_only"]["MRR"], shopper
