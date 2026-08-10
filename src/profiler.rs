//! Pipeline stage model — profiling primitives for RAG pipelines.
//!
//! Defines [`Stage`] and [`PipelineProfile`] for measuring and recording
//! per-stage performance metrics in a RAG pipeline.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

/// A single stage in a RAG pipeline execution (e.g. embedding, retrieval, generation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    /// Stage identifier (e.g. `"query_embedding"`, `"vector_search"`)
    pub name: String,
    /// Execution duration in milliseconds
    pub duration_ms: u128,
    /// Number of tokens consumed or produced by this stage
    pub token_count: usize,
    /// Estimated cost in USD for this stage
    pub cost: f64,
    /// Extensible key-value metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Stage {
    /// Create a new stage with the given name and zero initial metrics.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            duration_ms: 0,
            token_count: 0,
            cost: 0.0,
            metadata: HashMap::new(),
        }
    }

    /// Set the duration in milliseconds.
    pub fn with_duration(mut self, ms: u128) -> Self {
        self.duration_ms = ms;
        self
    }

    /// Set the token count.
    pub fn with_tokens(mut self, tokens: usize) -> Self {
        self.token_count = tokens;
        self
    }

    /// Set the estimated cost.
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = cost;
        self
    }

    /// Insert a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}ms, {} tokens, ${:.4})",
            self.name, self.duration_ms, self.token_count, self.cost
        )
    }
}

/// A timing guard that measures elapsed time between creation and consumption.
///
/// ```rust,ignore
/// let timer = StageTimer::start("vector_search");
/// // ... do work ...
/// let stage = timer.finish(1500, 0.0003); // tokens, cost
/// ```
pub struct StageTimer {
    name: String,
    start: Instant,
    metadata: HashMap<String, String>,
}

impl StageTimer {
    /// Begin timing a stage.
    pub fn start(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    /// Attach metadata before finishing.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Complete timing and produce a [`Stage`] with token count and cost.
    pub fn finish(self, token_count: usize, cost: f64) -> Stage {
        Stage {
            name: self.name,
            duration_ms: self.start.elapsed().as_millis(),
            token_count,
            cost,
            metadata: self.metadata,
        }
    }
}

/// Aggregated profile of a complete RAG pipeline execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineProfile {
    /// Ordered list of stages in execution order
    pub stages: Vec<Stage>,
    /// Total duration across all stages (ms)
    pub total_duration_ms: u128,
    /// Total tokens across all stages
    pub total_tokens: usize,
    /// Total estimated cost across all stages (USD)
    pub total_cost: f64,
}

impl PipelineProfile {
    /// Create an empty pipeline profile.
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            total_duration_ms: 0,
            total_tokens: 0,
            total_cost: 0.0,
        }
    }

    /// Add a completed stage and recompute aggregate stats.
    pub fn add_stage(&mut self, stage: Stage) {
        self.stages.push(stage);
        self.aggregate_stats();
    }

    /// Recompute total_duration_ms, total_tokens, and total_cost from stages.
    pub fn aggregate_stats(&mut self) {
        self.total_duration_ms = self.stages.iter().map(|s| s.duration_ms).sum();
        self.total_tokens = self.stages.iter().map(|s| s.token_count).sum();
        self.total_cost = self.stages.iter().map(|s| s.cost).sum();
    }

    /// Number of stages in the profile.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Find a stage by name.
    pub fn find(&self, name: &str) -> Option<&Stage> {
        self.stages.iter().find(|s| s.name == name)
    }
}

impl Default for PipelineProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PipelineProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pipeline Profile ({} stages):", self.stages.len())?;
        for stage in &self.stages {
            writeln!(f, "  • {}", stage)?;
        }
        writeln!(
            f,
            "Total: {}ms, {} tokens, ${:.4}",
            self.total_duration_ms, self.total_tokens, self.total_cost
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_serialization() {
        let stage = Stage::new("vector_search")
            .with_duration(42)
            .with_tokens(1500)
            .with_cost(0.0003)
            .with_metadata("backend", "qdrant");

        let json = serde_json::to_string(&stage).expect("serialize");
        let deserialized: Stage = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.name, "vector_search");
        assert_eq!(deserialized.duration_ms, 42);
        assert_eq!(deserialized.token_count, 1500);
        assert!((deserialized.cost - 0.0003).abs() < f64::EPSILON);
        assert_eq!(
            deserialized.metadata.get("backend"),
            Some(&"qdrant".to_string())
        );
    }

    #[test]
    fn test_pipeline_profile_aggregation() {
        let mut profile = PipelineProfile::new();

        profile.add_stage(
            Stage::new("query_embedding")
                .with_duration(12)
                .with_tokens(8)
                .with_cost(0.0001),
        );
        profile.add_stage(
            Stage::new("vector_search")
                .with_duration(35)
                .with_tokens(0)
                .with_cost(0.0),
        );
        profile.add_stage(
            Stage::new("generation")
                .with_duration(850)
                .with_tokens(450)
                .with_cost(0.012),
        );

        assert_eq!(profile.stage_count(), 3);
        assert_eq!(profile.total_duration_ms, 897);
        assert_eq!(profile.total_tokens, 458);
        assert!((profile.total_cost - 0.0121).abs() < 0.0001);
    }

    #[test]
    fn test_pipeline_profile_serialization() {
        let mut profile = PipelineProfile::new();
        profile.add_stage(
            Stage::new("embedding")
                .with_duration(10)
                .with_tokens(5)
                .with_cost(0.001),
        );

        let json = serde_json::to_string(&profile).expect("serialize");
        let deserialized: PipelineProfile = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.stage_count(), 1);
        assert_eq!(deserialized.total_duration_ms, 10);
        assert_eq!(deserialized.total_tokens, 5);
        assert!((deserialized.total_cost - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_pipeline() {
        let profile = PipelineProfile::new();

        assert_eq!(profile.stage_count(), 0);
        assert_eq!(profile.total_duration_ms, 0);
        assert_eq!(profile.total_tokens, 0);
        assert!((profile.total_cost - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stage_timer() {
        let timer = StageTimer::start("test_stage");
        // Simulate some work
        std::thread::sleep(std::time::Duration::from_millis(2));
        let stage = timer.finish(100, 0.01);

        assert_eq!(stage.name, "test_stage");
        assert!(stage.duration_ms >= 2, "duration should be >= 2ms");
        assert_eq!(stage.token_count, 100);
        assert!((stage.cost - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_stage_by_name() {
        let mut profile = PipelineProfile::new();
        profile.add_stage(Stage::new("retrieval").with_duration(50));
        profile.add_stage(Stage::new("generation").with_duration(200));

        assert!(profile.find("retrieval").is_some());
        assert!(profile.find("generation").is_some());
        assert!(profile.find("nonexistent").is_none());
    }
}
