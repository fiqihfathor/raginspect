//! Terminal table reporter — color-coded profiling output.
//!
//! Provides pretty-printed tables with green/yellow/red color coding
//! based on configurable performance thresholds.

use colored::Colorize;
use raginspect::{PipelineProfile, StageStats};

/// Threshold configuration for color coding.
#[derive(Debug, Clone)]
pub struct Thresholds {
    /// Stages faster than this (ms) are green
    pub ok_ms: u128,
    /// Stages slower than this (ms) are red
    pub slow_ms: u128,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            ok_ms: 50,
            slow_ms: 500,
        }
    }
}

/// Duration status based on thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurationStatus {
    Ok,
    Warn,
    Slow,
}

impl DurationStatus {
    /// Classify a duration (ms) into Ok/Warn/Slow.
    pub fn from_ms(ms: u128, thresholds: &Thresholds) -> Self {
        if ms < thresholds.ok_ms {
            Self::Ok
        } else if ms < thresholds.slow_ms {
            Self::Warn
        } else {
            Self::Slow
        }
    }

    /// Colored label for terminal output.
    pub fn label(&self) -> String {
        match self {
            Self::Ok => "OK".green().to_string(),
            Self::Warn => "WARN".yellow().to_string(),
            Self::Slow => "SLOW".red().to_string(),
        }
    }

    /// Apply color to a duration string.
    pub fn colorize(&self, text: &str) -> String {
        match self {
            Self::Ok => text.green().to_string(),
            Self::Warn => text.yellow().to_string(),
            Self::Slow => text.red().to_string(),
        }
    }
}

/// Render a single-run pipeline profile as a color-coded table.
pub fn render_profile_table(profile: &PipelineProfile, thresholds: &Thresholds) {
    println!("{}", "Pipeline Profile".bold().cyan());
    println!("{}", "─".repeat(60));

    use comfy_table::{presets::UTF8_FULL, ContentArrangement, Table};

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Stage", "Duration", "Tokens", "Cost", "Status"]);

    for stage in &profile.stages {
        let status = DurationStatus::from_ms(stage.duration_ms, thresholds);
        let dur_str = format!("{} ms", stage.duration_ms);

        table.add_row(vec![
            &stage.name,
            &status.colorize(&dur_str),
            &format!("{}", stage.token_count),
            &format!("${:.4}", stage.cost),
            &status.label(),
        ]);
    }

    let total_status = DurationStatus::from_ms(profile.total_duration_ms, thresholds);
    let total_dur = format!("{} ms", profile.total_duration_ms);

    table.add_row(vec![
        "TOTAL",
        &total_status.colorize(&total_dur),
        &format!("{}", profile.total_tokens),
        &format!("${:.4}", profile.total_cost),
        &total_status.label(),
    ]);

    println!("{table}");
}

/// Render multi-run profiling stats as a color-coded table.
pub fn render_stats_table(stats: &[StageStats], query: &str, thresholds: &Thresholds) {
    println!("{}", "Profiling Results".bold().cyan());
    println!("Query: \"{}\"", query);
    println!("{}", "─".repeat(70));

    use comfy_table::{presets::UTF8_FULL, ContentArrangement, Table};

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            "Stage", "P50 (ms)", "P99 (ms)", "Min", "Max", "Mean", "Runs", "Status",
        ]);

    for s in stats {
        let status = DurationStatus::from_ms(s.p50_ms as u128, thresholds);
        let p50_str = format!("{:.1}", s.p50_ms);
        let p99_str = format!("{:.1}", s.p99_ms);

        table.add_row(vec![
            &s.name,
            &status.colorize(&p50_str),
            &status.colorize(&p99_str),
            &format!("{:.1}", s.min_ms),
            &format!("{:.1}", s.max_ms),
            &format!("{:.1}", s.mean_ms),
            &format!("{}", s.runs),
            &status.label(),
        ]);
    }

    println!("{table}");
}
