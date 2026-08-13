# Quickstart Guide

Get up and running with `raginspect` in under 5 minutes.

## Prerequisites

- **Rust** stable (1.70+) — [install via rustup](https://rustup.rs/)
- No API keys required for basic inspection (uses mock data)

## 1. Build

```bash
git clone https://github.com/fiqihfathor/raginspect.git
cd raginspect
cargo build --release
```

The binary will be at `target/release/raginspect`.

## 2. Run Your First Inspection

Inspect a Naive RAG pipeline with the default mock query:

```bash
cargo run --release -- inspect
```

You'll see a diagnostic report with:
- Retrieval analysis (relevance, redundancy)
- Context window efficiency (token waste %)
- Generation grounding (hallucination index)
- RAG Health Score (0–100)
- Actionable recommendations

## 3. Profile a Pipeline

Measure per-stage timing:

```bash
# Single run — show timing table
cargo run --release -- profile

# Multi-run with p50/p99 stats (10 runs)
cargo run --release -- profile --runs 10

# JSON output
cargo run --release -- profile --format json
```

Output includes color-coded status (🟢 OK / 🟡 WARN / 🔴 SLOW) based on thresholds:
- Green: < 50ms
- Yellow: 50–499ms
- Red: ≥ 500ms

## 4. Use as a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
raginspect = "0.1"
```

### Inspect a Pipeline

```rust
use raginspect::{
    Inspector, PipelineConfig, RagArchitecture,
    InspectMode, ReportRenderer,
};

fn main() -> anyhow::Result<()> {
    let mut inspector = Inspector::new(PipelineConfig::default(), None);
    inspector.set_architecture(RagArchitecture::Naive);

    let report = inspector.inspect("What is RAG?", InspectMode::Full)?;

    // Pretty-print to terminal
    ReportRenderer::print_terminal_report(&report)?;

    // Or serialize to JSON
    let json = serde_json::to_string_pretty(&report)?;
    println!("{}", json);

    Ok(())
}
```

### Profile a Mock Pipeline

```rust
use raginspect::naive_pipeline::NaivePipeline;

fn main() {
    let pipeline = NaivePipeline::new();
    let profile = pipeline.run("What is RAG?", 5);

    println!("{}", profile);
    println!("Total: {}ms, {} tokens, ${:.4}",
        profile.total_duration_ms,
        profile.total_tokens,
        profile.total_cost);
}
```

### Multi-Run Profiling with Percentiles

```rust
use raginspect::{MultiRunProfiler, Stage};

fn main() {
    let mut profiler = MultiRunProfiler::new(10);

    for _ in 0..10 {
        profiler.run(|profile| {
            profile.add_stage(
                Stage::new("embedding")
                    .with_duration(8)
                    .with_tokens(6),
            );
            profile.add_stage(
                Stage::new("search")
                    .with_duration(25),
            );
        });
    }

    let stats = profiler.compute_stats();
    for s in &stats {
        println!("{}: p50={:.1}ms, p99={:.1}ms (n={})",
            s.name, s.p50_ms, s.p99_ms, s.runs);
    }
}
```

## 5. Custom Configuration

Create a TOML config file (see `examples/configs/sample.toml` for full reference):

```toml
name = "My-Pipeline"

[embedding]
model = "text-embedding-3-small"
dimension = 1536
distance_metric = "cosine"

[vector_store]
provider = "qdrant"
collection = "docs"
top_k = 5
similarity_threshold = 0.65

[llm]
provider = "openai"
model = "gpt-4o-mini"
max_tokens = 1024

[context]
max_context_tokens = 4096
deduplicate_threshold = 0.85
prune_irrelevant = true
```

Use it with:

```bash
cargo run --release -- inspect -c examples/configs/sample.toml -q "Your query here"
cargo run --release -- profile -c examples/configs/sample.toml --runs 5
```

## 6. CLI Reference

```
raginspect inspect   # Run diagnostic inspection
raginspect profile   # Measure per-stage timing

# Inspect options
  -q, --query <QUERY>              Query string (default: RAG question)
  -c, --pipeline-config <PATH>     Config file (default: examples/configs/sample.toml)
  -m, --model <MODEL>              Override LLM model name
  -i, --inspect-mode <MODE>        full | retrieval | context | quick
  -a, --architecture <ARCH>        naive | advanced | modular | agentic | graph | hyde | multimodal
      --json                        Output as JSON

# Profile options
  -c, --pipeline-config <PATH>     Config file
  -n, --runs <N>                   Number of runs for p50/p99 (default: 1)
  -f, --format <FORMAT>            table | json
  -q, --query <QUERY>              Query to profile
```

## 7. Run the Example

```bash
cargo run --example naive_rag_inspect
```

This demonstrates the full inspection flow: load config → inspect → print report → JSON output.

## 8. Run Tests

```bash
# All tests
cargo test

# Only unit tests
cargo test --lib

# Only integration tests
cargo test --test integration_test

# With memory-tracking feature
cargo test --all-features
```

## Supported Architectures

| Architecture | Description |
|-------------|-------------|
| **Naive** | Single retrieve → generate |
| **Advanced** | Pre/post retrieval processing, reranking |
| **Modular** | Routing, fusion, context compression |
| **Agentic** | Tool-calling, multi-step retrieval |
| **Graph** | Knowledge graph traversal |
| **HyDE** | Hypothetical document generation |
| **Multimodal** | Multi-modal retrieval (text, image, audio) |

## Next Steps

- Read [Architecture Overview](architecture.md) for the 3-layer inspection model
- Browse the [API documentation](https://github.com/fiqihfathor/raginspect#readme)
- Check the [Roadmap](../README.md#-roadmap) for upcoming features
