use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

mod reporter;

use raginspect::{
    InspectMode, Inspector, MultiRunProfiler, PipelineConfig, PipelineProfile, RagArchitecture,
    ReportRenderer, Stage,
};

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
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

/// Output format for profiling results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum OutputFormat {
    /// Render results as a formatted table (default)
    Table,
    /// Render results as pretty-printed JSON
    Json,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Inspect a RAG pipeline with diagnostic analysis
    Inspect {
        /// Target query string to inspect
        #[arg(
            short = 'q',
            long = "query",
            default_value = "What is the memory overhead of Tokio tasks in Rust?"
        )]
        query: String,

        /// Path to the pipeline configuration file (TOML)
        #[arg(
            short = 'c',
            long = "pipeline-config",
            default_value = "examples/configs/sample.toml"
        )]
        pipeline_config: PathBuf,

        /// Override the LLM model name
        #[arg(short = 'm', long = "model")]
        model: Option<String>,

        /// Inspection mode
        #[arg(short = 'i', long = "inspect-mode", value_enum, default_value_t = InspectMode::Full)]
        inspect_mode: InspectMode,

        /// RAG architecture type
        #[arg(short = 'a', long = "architecture", value_enum, default_value_t = RagArchitecture::Naive)]
        architecture: RagArchitecture,

        /// Output as JSON
        #[arg(long = "json")]
        json: bool,
    },

    /// Profile a RAG pipeline and measure per-stage timing
    Profile {
        /// Path to the pipeline configuration file (TOML)
        #[arg(
            short = 'c',
            long = "pipeline-config",
            default_value = "examples/configs/sample.toml"
        )]
        pipeline_config: PathBuf,

        /// Number of profiling runs (for p50/p99 stats)
        #[arg(short = 'n', long = "runs", default_value_t = 1)]
        runs: usize,

        /// Output format
        #[arg(short = 'f', long = "format", value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,

        /// Query to profile (uses mock pipeline by default)
        #[arg(short = 'q', long = "query", default_value = "What is RAG?")]
        query: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Inspect {
            query,
            pipeline_config,
            model,
            inspect_mode,
            architecture,
            json,
        }) => {
            let config = match PipelineConfig::load_from_file(&pipeline_config) {
                Ok(cfg) => cfg,
                Err(err) => {
                    eprintln!(
                        "⚠️  Could not load config from {:?}: {}. Using defaults.",
                        pipeline_config, err
                    );
                    PipelineConfig::default()
                }
            };

            let mut inspector = Inspector::new(config, model);
            inspector.set_architecture(architecture);

            let report = inspector.inspect(&query, inspect_mode)?;

            if json {
                ReportRenderer::print_json_report(&report)?;
            } else {
                ReportRenderer::print_terminal_report(&report)?;
            }
        }

        Some(Commands::Profile {
            pipeline_config,
            runs,
            format,
            query,
        }) => {
            // Validate --runs > 0
            if runs == 0 {
                eprintln!("error: --runs must be at least 1");
                return Err(anyhow::anyhow!("--runs must be at least 1"));
            }

            // Validate the config file exists.
            // TODO: Replace mock data below with real profiling that uses this config.
            match PipelineConfig::load_from_file(&pipeline_config) {
                Ok(_cfg) => {
                    // Config loaded successfully; real profiling would use it here.
                }
                Err(err) => {
                    eprintln!(
                        "⚠️  Warning: Could not load config from {:?}: {}. Proceeding with mock data.",
                        pipeline_config, err
                    );
                }
            }

            if runs == 1 {
                // TODO: This is temporary mock data — replace with real pipeline profiling.
                let mut profile = PipelineProfile::new();
                profile.add_stage(
                    Stage::new("query_embedding")
                        .with_duration(8)
                        .with_tokens(6),
                );
                profile.add_stage(Stage::new("vector_search").with_duration(25));
                profile.add_stage(
                    Stage::new("context_assembly")
                        .with_duration(3)
                        .with_tokens(1200),
                );
                profile.add_stage(
                    Stage::new("generation")
                        .with_duration(450)
                        .with_tokens(320)
                        .with_cost(0.0048),
                );

                print_profile(&profile, format)?;
            } else {
                // TODO: This is temporary mock data — replace with real pipeline profiling.
                let mut profiler = MultiRunProfiler::new(runs);
                let durations_emb = [6, 8, 7, 9, 8, 7, 10, 8, 7, 9];
                let durations_search = [20, 25, 22, 28, 24, 21, 30, 26, 23, 25];
                let durations_gen = [400, 450, 420, 480, 440, 410, 500, 460, 430, 450];

                for i in 0..runs {
                    let di = i % durations_emb.len();
                    profiler.run(|profile| {
                        profile.add_stage(
                            Stage::new("query_embedding")
                                .with_duration(durations_emb[di])
                                .with_tokens(6),
                        );
                        profile.add_stage(
                            Stage::new("vector_search").with_duration(durations_search[di]),
                        );
                        profile.add_stage(
                            Stage::new("context_assembly")
                                .with_duration(3)
                                .with_tokens(1200),
                        );
                        profile.add_stage(
                            Stage::new("generation")
                                .with_duration(durations_gen[di])
                                .with_tokens(320)
                                .with_cost(0.0048),
                        );
                    });
                }

                let stats = profiler.compute_stats();
                print_stats(&stats, format, &query)?;
            }
        }

        None => {
            eprintln!("🔍 raginspect — RAG Inspection & Profiling Engine\n");
            eprintln!("Usage: raginspect <COMMAND>\n");
            eprintln!("Commands:");
            eprintln!("  inspect   Inspect a RAG pipeline with diagnostic analysis");
            eprintln!("  profile   Profile a RAG pipeline and measure per-stage timing");
            eprintln!("  help      Print this message or the help of the given subcommand(s)\n");
            eprintln!("Run 'raginspect <command> --help' for more information.");
        }
    }

    Ok(())
}

fn print_profile(profile: &PipelineProfile, format: OutputFormat) -> Result<()> {
    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(profile)?;
            println!("{}", json);
        }
        OutputFormat::Table => {
            reporter::render_profile_table(profile, &reporter::Thresholds::default());
        }
    }
    Ok(())
}

fn print_stats(stats: &[raginspect::StageStats], format: OutputFormat, query: &str) -> Result<()> {
    match format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(stats)?;
            println!("{}", json);
        }
        OutputFormat::Table => {
            reporter::render_stats_table(stats, query, &reporter::Thresholds::default());
        }
    }
    Ok(())
}
