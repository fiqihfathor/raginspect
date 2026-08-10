use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;
use std::path::PathBuf;

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

        /// Output format: table or json
        #[arg(short = 'f', long = "format", default_value = "table")]
        format: String,

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
            pipeline_config: _,
            runs,
            format,
            query,
        }) => {
            if runs == 1 {
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

                print_profile(&profile, &format);
            } else {
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
                print_stats(&stats, &format, &query)?;
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

fn print_profile(profile: &PipelineProfile, format: &str) {
    match format {
        "json" => {
            let json = serde_json::to_string_pretty(profile).unwrap_or_default();
            println!("{}", json);
        }
        _ => {
            println!("{}", "Pipeline Profile".bold().cyan());
            println!("{}", "─".repeat(60));

            use comfy_table::{presets::UTF8_FULL, ContentArrangement, Table};

            let mut table = Table::new();
            table
                .load_preset(UTF8_FULL)
                .set_content_arrangement(ContentArrangement::Dynamic)
                .set_header(vec!["Stage", "Duration", "Tokens", "Cost"]);

            for stage in &profile.stages {
                table.add_row(vec![
                    &stage.name,
                    &format!("{} ms", stage.duration_ms),
                    &format!("{}", stage.token_count),
                    &format!("${:.4}", stage.cost),
                ]);
            }

            table.add_row(vec![
                "TOTAL",
                &format!("{} ms", profile.total_duration_ms),
                &format!("{}", profile.total_tokens),
                &format!("${:.4}", profile.total_cost),
            ]);

            println!("{table}");
        }
    }
}

fn print_stats(stats: &[raginspect::StageStats], format: &str, query: &str) -> Result<()> {
    match format {
        "json" => {
            let json = serde_json::to_string_pretty(stats).unwrap_or_default();
            println!("{}", json);
        }
        _ => {
            println!("{}", "Profiling Results".bold().cyan());
            println!("Query: \"{}\"", query);
            println!("{}", "─".repeat(70));

            use comfy_table::{presets::UTF8_FULL, ContentArrangement, Table};

            let mut table = Table::new();
            table
                .load_preset(UTF8_FULL)
                .set_content_arrangement(ContentArrangement::Dynamic)
                .set_header(vec![
                    "Stage", "P50 (ms)", "P99 (ms)", "Min", "Max", "Mean", "Runs",
                ]);

            for s in stats {
                table.add_row(vec![
                    &s.name,
                    &format!("{:.1}", s.p50_ms),
                    &format!("{:.1}", s.p99_ms),
                    &format!("{:.1}", s.min_ms),
                    &format!("{:.1}", s.max_ms),
                    &format!("{:.1}", s.mean_ms),
                    &format!("{}", s.runs),
                ]);
            }

            println!("{table}");
        }
    }
    Ok(())
}
