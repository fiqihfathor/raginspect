//! Integration tests for raginspect

use raginspect::{
    Inspector, InspectMode, PipelineConfig, RagArchitecture,
};

#[test]
fn test_naive_rag_inspection() {
    let mut inspector = Inspector::new(PipelineConfig::default(), None);
    inspector.set_architecture(RagArchitecture::Naive);

    let report = inspector
        .inspect("What is RAG?", InspectMode::Full)
        .expect("inspection should succeed");

    assert!(!report.query.is_empty());
    assert!(report.overall_score >= 0.0 && report.overall_score <= 100.0);
    assert!(!report.recommendations.is_empty());
}

#[test]
fn test_all_architectures_produce_recommendations() {
    let architectures = [
        RagArchitecture::Naive,
        RagArchitecture::Advanced,
        RagArchitecture::Modular,
        RagArchitecture::Agentic,
        RagArchitecture::Graph,
        RagArchitecture::Hyde,
        RagArchitecture::Multimodal,
    ];

    for arch in architectures {
        let mut inspector = Inspector::new(PipelineConfig::default(), None);
        inspector.set_architecture(arch);

        let report = inspector
            .inspect("test query about Rust async runtime", InspectMode::Quick)
            .expect("inspection should succeed");

        assert!(
            !report.recommendations.is_empty(),
            "Architecture {:?} should produce recommendations",
            arch
        );
    }
}

#[test]
fn test_inspect_modes() {
    for mode in [InspectMode::Full, InspectMode::Retrieval, InspectMode::Context, InspectMode::Quick] {
        let mut inspector = Inspector::new(PipelineConfig::default(), None);
        inspector.set_architecture(RagArchitecture::Naive);

        let report = inspector
            .inspect("test query", mode)
            .expect("inspection should succeed");

        assert_eq!(report.query, "test query");
    }
}

#[test]
fn test_token_efficiency_calculation() {
    let mut inspector = Inspector::new(PipelineConfig::default(), None);
    inspector.set_architecture(RagArchitecture::Naive);

    let report = inspector
        .inspect("Tokio task memory overhead", InspectMode::Full)
        .expect("inspection should succeed");

    let ctx = &report.context;
    assert!(ctx.total_tokens > 0);
    assert!(ctx.useful_tokens + ctx.wasted_tokens == ctx.total_tokens);
    assert!(ctx.efficiency_ratio >= 0.0 && ctx.efficiency_ratio <= 1.0);
}

#[test]
fn test_hallucination_detection() {
    let mut inspector = Inspector::new(PipelineConfig::default(), None);
    inspector.set_architecture(RagArchitecture::Naive);

    let report = inspector
        .inspect("Rust Tokio memory", InspectMode::Full)
        .expect("inspection should succeed");

    assert!(report.generation.hallucination_score >= 0.0);
    assert!(report.generation.hallucination_score <= 1.0);
    assert!(report.generation.source_attribution_pct >= 0.0);
    assert!(report.generation.source_attribution_pct <= 1.0);
}
