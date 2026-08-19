//! Common test helpers and fixtures shared across integration tests.

use raginspect::{InspectMode, InspectionReport, Inspector, PipelineConfig};

/// Load the sample pipeline config from the examples directory.
pub fn load_sample_config() -> PipelineConfig {
    PipelineConfig::load_from_file("examples/configs/sample.toml").expect("sample.toml must load")
}

/// Load the test fixture config from the tests/fixtures directory.
pub fn load_fixture_config() -> PipelineConfig {
    PipelineConfig::load_from_file("tests/fixtures/mock_pipeline.toml")
        .expect("fixture mock_pipeline.toml must load")
}

/// Create a default inspector with Naive architecture.
pub fn create_inspector() -> Inspector {
    let mut inspector = Inspector::new(PipelineConfig::default(), None);
    inspector.set_architecture(raginspect::RagArchitecture::Naive);
    inspector
}

/// Run a full inspection with the default config and return the report.
pub fn inspect_default(query: &str) -> InspectionReport {
    let inspector = create_inspector();
    inspector
        .inspect(query, InspectMode::Full)
        .expect("inspection should succeed")
}

/// Run an inspection with a specific config and mode.
pub fn inspect_query(config: PipelineConfig, query: &str, mode: InspectMode) -> InspectionReport {
    let mut inspector = Inspector::new(config, None);
    inspector.set_architecture(raginspect::RagArchitecture::Naive);
    inspector
        .inspect(query, mode)
        .expect("inspection should succeed")
}

/// Assert that a score is within [0.0, 100.0].
pub fn assert_valid_score(score: f64, label: &str) {
    assert!(
        (0.0..=100.0).contains(&score),
        "{label} must be in [0.0, 100.0], got {score}"
    );
}

/// Assert that a ratio is within [0.0, 1.0].
pub fn assert_valid_ratio(ratio: f64, label: &str) {
    assert!(
        (0.0..=1.0).contains(&ratio),
        "{label} must be in [0.0, 1.0], got {ratio}"
    );
}
