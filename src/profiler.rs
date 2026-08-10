//! Pipeline stage model — profiling primitives for RAG pipelines.
//!
//! Defines [`Stage`] and [`PipelineProfile`] for measuring and recording
//! per-stage performance metrics in a RAG pipeline.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

#[cfg(feature = "memory-tracking")]
use sysinfo::{Pid, System};

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

/// Percentile statistics for a single stage across multiple pipeline runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageStats {
    /// Stage name these stats refer to
    pub name: String,
    /// Number of runs sampled
    pub runs: usize,
    /// Median duration in milliseconds (p50)
    pub p50_ms: f64,
    /// 99th percentile duration in milliseconds
    pub p99_ms: f64,
    /// Minimum duration observed (ms)
    pub min_ms: f64,
    /// Maximum duration observed (ms)
    pub max_ms: f64,
    /// Mean duration (ms)
    pub mean_ms: f64,
    /// Median token count
    pub p50_tokens: f64,
    /// Median cost (USD)
    pub p50_cost: f64,
}

impl StageStats {
    /// Compute percentile statistics from a list of durations (ms).
    fn from_durations(
        name: &str,
        durations_ms: &mut [f64],
        tokens: &mut [usize],
        costs: &mut [f64],
    ) -> Self {
        let runs = durations_ms.len();
        if runs == 0 {
            return Self {
                name: name.to_string(),
                runs: 0,
                p50_ms: 0.0,
                p99_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                mean_ms: 0.0,
                p50_tokens: 0.0,
                p50_cost: 0.0,
            };
        }

        durations_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        tokens.sort();
        costs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        Self {
            name: name.to_string(),
            runs,
            p50_ms: percentile(durations_ms, 50),
            p99_ms: percentile(durations_ms, 99),
            min_ms: durations_ms[0],
            max_ms: durations_ms[runs - 1],
            mean_ms: durations_ms.iter().sum::<f64>() / runs as f64,
            p50_tokens: percentile_usize(tokens, 50),
            p50_cost: percentile(costs, 50),
        }
    }
}

impl fmt::Display for StageStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (p50={:.1}ms, p99={:.1}ms, mean={:.1}ms, n={})",
            self.name, self.p50_ms, self.p99_ms, self.mean_ms, self.runs
        )
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

/// Compute percentile from a sorted slice of f64 values.
/// `pct` is 0-100.
fn percentile(sorted: &[f64], pct: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((pct as f64 / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Compute percentile from a sorted slice of usize values.
fn percentile_usize(sorted: &[usize], pct: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((pct as f64 / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)] as f64
}

/// Profiler that runs a pipeline multiple times and computes percentile stats.
///
/// ```no_run
/// use raginspect::MultiRunProfiler;
/// use raginspect::Stage;
///
/// let mut profiler = MultiRunProfiler::new(10); // 10 runs
/// for _ in 0..10 {
///     profiler.run(|profile| {
///         profile.add_stage(Stage::new("embedding").with_duration(5).with_tokens(10));
///         profile.add_stage(Stage::new("search").with_duration(20).with_tokens(0));
///     });
/// }
/// let stats = profiler.compute_stats();
/// println!("p50 search: {:.1}ms", stats[1].p50_ms);
/// ```
pub struct MultiRunProfiler {
    /// Number of runs to execute
    pub runs: usize,
    /// All completed profiles
    pub profiles: Vec<PipelineProfile>,
    #[cfg(feature = "memory-tracking")]
    sys: System,
    #[cfg(feature = "memory-tracking")]
    pid: Pid,
}

impl MultiRunProfiler {
    /// Create a new multi-run profiler.
    pub fn new(runs: usize) -> Self {
        Self {
            runs,
            profiles: Vec::with_capacity(runs),
            #[cfg(feature = "memory-tracking")]
            sys: System::new_all(),
            #[cfg(feature = "memory-tracking")]
            pid: Pid::from(std::process::id() as usize),
        }
    }

    /// Execute a single run. The closure receives an empty [`PipelineProfile`] to populate.
    pub fn run<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut PipelineProfile),
    {
        let mut profile = PipelineProfile::new();
        f(&mut profile);
        self.profiles.push(profile);
    }

    /// Compute per-stage percentile statistics across all runs.
    pub fn compute_stats(&self) -> Vec<StageStats> {
        // Collect stage names from first profile
        if self.profiles.is_empty() {
            return Vec::new();
        }
        let stage_names: Vec<&str> = self.profiles[0]
            .stages
            .iter()
            .map(|s| s.name.as_str())
            .collect();

        stage_names
            .iter()
            .map(|&name| {
                let mut durations: Vec<f64> = Vec::new();
                let mut tokens: Vec<usize> = Vec::new();
                let mut costs: Vec<f64> = Vec::new();

                for profile in &self.profiles {
                    if let Some(stage) = profile.find(name) {
                        durations.push(stage.duration_ms as f64);
                        tokens.push(stage.token_count);
                        costs.push(stage.cost);
                    }
                }

                StageStats::from_durations(name, &mut durations, &mut tokens, &mut costs)
            })
            .collect()
    }

    /// Get the memory usage in bytes for the current process (requires `memory-tracking` feature).
    #[cfg(feature = "memory-tracking")]
    pub fn memory_usage_bytes(&mut self) -> u64 {
        self.sys.refresh_process(self.pid);
        self.sys.process(self.pid).map(|p| p.memory()).unwrap_or(0)
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

    #[test]
    fn test_percentile_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        // p50 of 10 elements: index = round(0.5 * 9) = 5 → data[5] = 6.0
        assert!((percentile(&data, 50) - 6.0).abs() < f64::EPSILON);
        // p99: index = round(0.99 * 9) = 9 → data[9] = 10.0
        assert!((percentile(&data, 99) - 10.0).abs() < f64::EPSILON);
        // p0: index 0
        assert!((percentile(&data, 0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentile_empty() {
        let data: Vec<f64> = vec![];
        assert_eq!(percentile(&data, 50), 0.0);
    }

    #[test]
    fn test_multi_run_profiler_stats() {
        let mut profiler = MultiRunProfiler::new(5);

        // Simulate 5 runs with varying durations
        let durations = [10, 12, 8, 15, 11];
        for &d in &durations {
            profiler.run(|profile| {
                profile.add_stage(
                    Stage::new("embedding")
                        .with_duration(d)
                        .with_tokens(8)
                        .with_cost(0.0001),
                );
                profile.add_stage(
                    Stage::new("search")
                        .with_duration(d * 3)
                        .with_tokens(0)
                        .with_cost(0.0),
                );
            });
        }

        assert_eq!(profiler.profiles.len(), 5);

        let stats = profiler.compute_stats();
        assert_eq!(stats.len(), 2); // 2 stages

        // Embedding stats
        let emb = &stats[0];
        assert_eq!(emb.name, "embedding");
        assert_eq!(emb.runs, 5);
        assert!(emb.p50_ms >= 8.0 && emb.p50_ms <= 15.0);
        assert!(emb.p99_ms >= emb.p50_ms);
        assert!(emb.min_ms <= emb.p50_ms);
        assert!(emb.max_ms >= emb.p50_ms);

        // Search stats
        let search = &stats[1];
        assert_eq!(search.name, "search");
        assert_eq!(search.runs, 5);
        assert!(search.p50_ms > emb.p50_ms); // search takes longer
    }

    #[test]
    fn test_multi_run_profiler_empty() {
        let profiler = MultiRunProfiler::new(3);
        let stats = profiler.compute_stats();
        assert!(stats.is_empty());
    }

    #[test]
    fn test_stage_stats_display() {
        let mut durations = vec![10.0, 20.0, 30.0];
        let mut tokens = vec![5, 10, 15];
        let mut costs = vec![0.01, 0.02, 0.03];
        let stats = StageStats::from_durations("test", &mut durations, &mut tokens, &mut costs);

        let display = format!("{}", stats);
        assert!(display.contains("test"));
        assert!(display.contains("p50="));
        assert!(display.contains("n=3"));
    }
}
