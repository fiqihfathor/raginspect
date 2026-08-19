//! Example: Inspect a Naive RAG pipeline using raginspect
//!
//! Run with: `cargo run --example naive_rag_inspect`

use raginspect::{InspectMode, Inspector, PipelineConfig, RagArchitecture, ReportRenderer};

fn main() -> anyhow::Result<()> {
    // Load pipeline config (or use defaults)
    let config = PipelineConfig::default();

    // Create inspector
    let mut inspector = Inspector::new(config, None);
    inspector.set_architecture(RagArchitecture::Naive);

    // Run inspection
    let report = inspector.inspect(
        "What is the memory overhead of Tokio tasks in Rust?",
        InspectMode::Full,
    )?;

    // Print terminal report
    ReportRenderer::print_terminal_report(&report)?;

    // You can also get JSON
    let json = serde_json::to_string_pretty(&report)?;
    println!("\n--- JSON excerpt (first 200 chars) ---");
    println!("{}", &json[..200.min(json.len())]);

    Ok(())
}
