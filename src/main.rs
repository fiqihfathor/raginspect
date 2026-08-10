use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use raginspect::{InspectMode, Inspector, PipelineConfig, RagArchitecture, ReportRenderer};

/// 🔍 raginspect — X-ray diagnostic engine for RAG pipelines
#[derive(Parser, Debug)]
#[command(
    name = "raginspect",
    author = "fiqihfathor",
    version = "0.1.0",
    about = "🔍 RAG Inspection & Profiling Engine — X-ray for RAG pipelines",
    long_about = "raginspect profiles each layer of a RAG application — retrieval relevance, \
                  context token efficiency, and LLM generation grounding — to deliver actionable \
                  diagnostics and quality verdicts across 7 RAG architectures."
)]
struct CliArgs {
    /// Target query string to inspect through the RAG pipeline
    #[arg(
        short = 'q',
        long = "query",
        default_value = "What is the memory overhead of Tokio tasks in Rust?"
    )]
    query: String,

    /// Path to the pipeline configuration file (TOML format)
    #[arg(
        short = 'c',
        long = "pipeline-config",
        default_value = "examples/configs/sample.toml"
    )]
    pipeline_config: PathBuf,

    /// Override the LLM model name specified in configuration
    #[arg(short = 'm', long = "model")]
    model: Option<String>,

    /// Inspection mode (full, retrieval, context, quick)
    #[arg(short = 'i', long = "inspect-mode", value_enum, default_value_t = InspectMode::Full)]
    inspect_mode: InspectMode,

    /// Output full inspection report as formatted JSON
    #[arg(long = "json")]
    json: bool,

    /// RAG architecture type to inspect
    #[arg(short = 'a', long = "architecture", value_enum, default_value_t = RagArchitecture::Naive)]
    architecture: RagArchitecture,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = CliArgs::parse();

    let config = match PipelineConfig::load_from_file(&args.pipeline_config) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!(
                "⚠️  Could not load config from {:?}: {}. Using defaults.",
                args.pipeline_config, err
            );
            PipelineConfig::default()
        }
    };

    let mut inspector = Inspector::new(config, args.model);
    inspector.set_architecture(args.architecture);

    let report = inspector.inspect(&args.query, args.inspect_mode)?;

    if args.json {
        ReportRenderer::print_json_report(&report)?;
    } else {
        ReportRenderer::print_terminal_report(&report)?;
    }

    Ok(())
}
