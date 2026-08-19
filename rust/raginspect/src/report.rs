use anyhow::Result;
use colored::*;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Attribute, Cell, Color, ContentArrangement, Table};

use crate::types::{InspectMode, InspectionReport, Verdict};

/// Report renderer that formats inspection results as terminal tables or JSON.
pub struct ReportRenderer;

impl ReportRenderer {
    /// Render inspection report to terminal standard output.
    pub fn print_terminal_report(report: &InspectionReport) -> Result<()> {
        println!();
        Self::print_banner();
        Self::print_summary_card(report);
        Self::print_architecture_profile(report);

        match report.inspect_mode {
            InspectMode::Full => {
                Self::print_retrieval_layer(report);
                Self::print_context_layer(report);
                Self::print_generation_layer(report);
            }
            InspectMode::Retrieval => {
                Self::print_retrieval_layer(report);
            }
            InspectMode::Context => {
                Self::print_retrieval_layer(report);
                Self::print_context_layer(report);
            }
            InspectMode::Quick => {
                Self::print_quick_summary(report);
            }
        }

        Self::print_recommendations(report);
        println!();

        Ok(())
    }

    /// Output report formatted as pretty JSON.
    pub fn print_json_report(report: &InspectionReport) -> Result<()> {
        let json_str = serde_json::to_string_pretty(report)?;
        println!("{}", json_str);
        Ok(())
    }

    fn print_banner() {
        println!("{}", "═".repeat(80).bright_blue());
        println!(
            " {} {} {}",
            "🔍".bold(),
            "RAGINSPECT".bold().bright_cyan(),
            "— Retrieval Augmented Generation Inspection & Profiling Engine".dimmed()
        );
        println!("{}", "═".repeat(80).bright_blue());
    }

    fn print_summary_card(report: &InspectionReport) {
        println!();
        println!(
            "{}",
            "📋 INSPECTION METADATA & HEALTH SCORE".bold().underline()
        );

        let score_colored = if report.overall_score >= 85.0 {
            format!("{:.1}/100 [EXCELLENT]", report.overall_score)
                .bold()
                .green()
        } else if report.overall_score >= 70.0 {
            format!("{:.1}/100 [GOOD]", report.overall_score)
                .bold()
                .yellow()
        } else {
            format!("{:.1}/100 [NEEDS ATTENTION]", report.overall_score)
                .bold()
                .red()
        };

        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_content_arrangement(ContentArrangement::Dynamic);

        table.add_row(vec![
            Cell::new("Target Query").add_attribute(Attribute::Bold),
            Cell::new(&report.query).fg(Color::Cyan),
        ]);
        table.add_row(vec![
            Cell::new("Pipeline Config").add_attribute(Attribute::Bold),
            Cell::new(&report.config_name),
        ]);
        table.add_row(vec![
            Cell::new("LLM Model").add_attribute(Attribute::Bold),
            Cell::new(&report.model_name),
        ]);
        table.add_row(vec![
            Cell::new("RAG Architecture").add_attribute(Attribute::Bold),
            Cell::new(format!("{}", report.architecture)).fg(Color::Magenta),
        ]);
        table.add_row(vec![
            Cell::new("Inspection Mode").add_attribute(Attribute::Bold),
            Cell::new(format!("{}", report.inspect_mode)),
        ]);
        table.add_row(vec![
            Cell::new("RAG Health Score").add_attribute(Attribute::Bold),
            Cell::new(score_colored.to_string()),
        ]);

        println!("{table}");
    }

    fn print_architecture_profile(report: &InspectionReport) {
        println!();
        println!("{}", "🏗️ ARCHITECTURE PROFILE".bold().bright_blue());

        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_content_arrangement(ContentArrangement::Dynamic);

        table.add_row(vec![
            Cell::new("Architecture Type").add_attribute(Attribute::Bold),
            Cell::new(format!("{}", report.architecture)).fg(Color::Magenta),
        ]);
        table.add_row(vec![
            Cell::new("Focus Area").add_attribute(Attribute::Bold),
            Cell::new(report.architecture.diagnostic_focus()),
        ]);

        let inspection_points = report.architecture.diagnostic_layers();
        let points_str = inspection_points
            .iter()
            .map(|p| format!("  • {}", p))
            .collect::<Vec<_>>()
            .join("\n");
        table.add_row(vec![
            Cell::new("Inspection Layers").add_attribute(Attribute::Bold),
            Cell::new(points_str),
        ]);

        println!("{table}");
    }

    fn print_retrieval_layer(report: &InspectionReport) {
        println!();
        println!(
            "{} (Top-K: {}, Retrieved: {}, Latency: {}ms, Avg Similarity: {:.2})",
            "⚡ LAYER 1: VECTOR RETRIEVAL ANALYSIS"
                .bold()
                .bright_yellow(),
            report.retrieval.top_k,
            report.retrieval.chunks_retrieved,
            report.retrieval.latency_ms,
            report.retrieval.avg_similarity
        );

        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_content_arrangement(ContentArrangement::Dynamic);

        table.set_header(vec![
            Cell::new("ID").add_attribute(Attribute::Bold),
            Cell::new("Source Document").add_attribute(Attribute::Bold),
            Cell::new("Score").add_attribute(Attribute::Bold),
            Cell::new("Tokens").add_attribute(Attribute::Bold),
            Cell::new("Verdict").add_attribute(Attribute::Bold),
            Cell::new("Diagnostic Rationale").add_attribute(Attribute::Bold),
        ]);

        for chunk in &report.retrieval.chunks {
            let verdict_cell = match chunk.verdict {
                Verdict::Relevant => Cell::new(chunk.verdict.badge()).fg(Color::Green),
                Verdict::PartiallyRelevant => Cell::new(chunk.verdict.badge()).fg(Color::Yellow),
                Verdict::Irrelevant => Cell::new(chunk.verdict.badge()).fg(Color::Red),
                Verdict::Duplicate => Cell::new(chunk.verdict.badge()).fg(Color::Magenta),
                Verdict::LowConfidence => Cell::new(chunk.verdict.badge()).fg(Color::Cyan),
            };

            let score_str = format!("{:.3}", chunk.score);
            let score_cell = if chunk.score >= 0.80 {
                Cell::new(score_str).fg(Color::Green)
            } else if chunk.score >= 0.60 {
                Cell::new(score_str).fg(Color::Yellow)
            } else {
                Cell::new(score_str).fg(Color::Red)
            };

            table.add_row(vec![
                Cell::new(&chunk.id).add_attribute(Attribute::Bold),
                Cell::new(&chunk.source_doc).fg(Color::Blue),
                score_cell,
                Cell::new(chunk.token_count.to_string()),
                verdict_cell,
                Cell::new(&chunk.verdict_reason),
            ]);
        }

        println!("{table}");
    }

    fn print_context_layer(report: &InspectionReport) {
        println!();
        println!(
            "{}",
            "📊 LAYER 2: CONTEXT CONSTRUCTION & TOKEN EFFICIENCY"
                .bold()
                .bright_magenta()
        );

        let ctx = &report.context;
        let eff_pct = ctx.efficiency_ratio * 100.0;

        // Visual Progress Bar for Token Efficiency
        let bar_width = 30;
        let filled = ((ctx.efficiency_ratio * bar_width as f64).round() as usize).min(bar_width);
        let empty = bar_width - filled;
        let bar = format!(
            "[{}{}] {:.1}%",
            "█".repeat(filled),
            "░".repeat(empty),
            eff_pct
        );

        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_content_arrangement(ContentArrangement::Dynamic);

        table.add_row(vec![
            Cell::new("Token Efficiency Gauge").add_attribute(Attribute::Bold),
            Cell::new(bar),
        ]);
        table.add_row(vec![
            Cell::new("Total Context Tokens").add_attribute(Attribute::Bold),
            Cell::new(format!("{} tokens", ctx.total_tokens)),
        ]);
        table.add_row(vec![
            Cell::new("Useful Context Tokens").add_attribute(Attribute::Bold),
            Cell::new(format!("{} tokens", ctx.useful_tokens)).fg(Color::Green),
        ]);
        table.add_row(vec![
            Cell::new("Wasted Context Tokens").add_attribute(Attribute::Bold),
            Cell::new(format!("{} tokens (Duplicates/Noise)", ctx.wasted_tokens)).fg(Color::Red),
        ]);
        table.add_row(vec![
            Cell::new("Chunks Deduplicated / Pruned").add_attribute(Attribute::Bold),
            Cell::new(format!(
                "{} duplicates, {} low-relevance",
                ctx.deduplicated_chunks, ctx.irrelevant_chunks_pruned
            )),
        ]);
        table.add_row(vec![
            Cell::new("Context Window Utilization").add_attribute(Attribute::Bold),
            Cell::new(format!(
                "{} / {} max window tokens ({:.1}%)",
                ctx.total_tokens,
                ctx.context_window_limit,
                (ctx.total_tokens as f64 / ctx.context_window_limit as f64) * 100.0
            )),
        ]);

        println!("{table}");
    }

    fn print_generation_layer(report: &InspectionReport) {
        println!();
        println!(
            "{} (Inference Latency: {}ms, Prompt: {} tok, Completion: {} tok)",
            "🧠 LAYER 3: GENERATION & GROUNDING ANALYSIS"
                .bold()
                .bright_cyan(),
            report.generation.latency_ms,
            report.generation.prompt_tokens,
            report.generation.completion_tokens
        );

        println!();
        println!("{}", "Generated Response Preview:".bold());
        println!(
            "  {}",
            format!("\"{}\"", report.generation.generated_text)
                .italic()
                .dimmed()
        );
        println!();

        let mut table = Table::new();
        table.load_preset(UTF8_FULL);
        table.set_content_arrangement(ContentArrangement::Dynamic);

        let hallucination_risk_str =
            format!("{:.1}%", report.generation.hallucination_score * 100.0);
        let hallucination_cell = if report.generation.hallucination_score <= 0.05 {
            Cell::new(format!("{} (LOW)", hallucination_risk_str)).fg(Color::Green)
        } else if report.generation.hallucination_score <= 0.15 {
            Cell::new(format!("{} (MODERATE)", hallucination_risk_str)).fg(Color::Yellow)
        } else {
            Cell::new(format!("{} (HIGH)", hallucination_risk_str)).fg(Color::Red)
        };

        table.add_row(vec![
            Cell::new("Grounding Attribution").add_attribute(Attribute::Bold),
            Cell::new(format!(
                "{:.1}%",
                report.generation.source_attribution_pct * 100.0
            ))
            .fg(Color::Green),
        ]);
        table.add_row(vec![
            Cell::new("Hallucination Risk Score").add_attribute(Attribute::Bold),
            hallucination_cell,
        ]);
        table.add_row(vec![
            Cell::new("Cited Context Chunks").add_attribute(Attribute::Bold),
            Cell::new(report.generation.cited_chunk_ids.join(", ")).fg(Color::Blue),
        ]);

        println!("{table}");

        if !report.generation.uncited_claims.is_empty() {
            println!();
            println!("{}", "⚠️ UNATTRIBUTED CLAIMS DETECTED:".bold().red());
            for claim in &report.generation.uncited_claims {
                println!("  • {}", claim.yellow());
            }
        }
    }

    fn print_quick_summary(report: &InspectionReport) {
        println!();
        println!("{}", "⚡ QUICK DIAGNOSTIC SUMMARY".bold().yellow());
        println!(
            "  • Retrieved Chunks: {} (Avg Sim: {:.2})",
            report.retrieval.chunks_retrieved, report.retrieval.avg_similarity
        );
        println!(
            "  • Token Efficiency: {:.1}% useful ({} tokens wasted)",
            report.context.efficiency_ratio * 100.0,
            report.context.wasted_tokens
        );
        println!(
            "  • Grounding Score: {:.1}% (Hallucination Index: {:.2})",
            report.generation.source_attribution_pct * 100.0,
            report.generation.hallucination_score
        );
    }

    fn print_recommendations(report: &InspectionReport) {
        println!();
        println!("{}", "🛠️ ACTIONABLE X-RAY RECOMMENDATIONS".bold().green());
        for (i, rec) in report.recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, rec.bright_white());
        }
    }
}
