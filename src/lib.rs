//! 🔍 raginspect — RAG Inspection & Profiling Engine
//!
//! X-ray for Retrieval Augmented Generation pipelines.
//! Supports 7 architectures: Naive, Advanced, Modular, Agentic, Graph, HyDE, Multimodal.
//!
//! ## Quick Start
//!
//! ```no_run
//! use raginspect::{Inspector, InspectMode, PipelineConfig, RagArchitecture};
//!
//! fn main() -> anyhow::Result<()> {
//!     let mut inspector = Inspector::new(PipelineConfig::default(), None);
//!     inspector.set_architecture(RagArchitecture::Naive);
//!     let report = inspector.inspect("What is RAG?", InspectMode::Full)?;
//!     println!("Health Score: {:.1}", report.overall_score);
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod inspector;
pub mod profiler;
pub mod report;
pub mod types;

// Re-export primary types for convenience
pub use config::PipelineConfig;
pub use inspector::Inspector;
pub use profiler::{PipelineProfile, Stage, StageTimer};
pub use report::ReportRenderer;
pub use types::{
    ChunkInfo, ContextStage, GenerationResult, InspectMode, InspectionReport, RagArchitecture,
    RetrievalResult, Verdict,
};
