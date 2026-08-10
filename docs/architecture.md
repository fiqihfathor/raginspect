# raginspect — Architecture

> This document describes the internal architecture of raginspect, including module breakdown, data flow, extension points, and the Python binding strategy.

---

## Overview

raginspect is a Rust workspace composed of four crates, a CLI binary, and optional Python bindings. The core design principle: **the profiler should understand RAG pipelines structurally, not just time them.**

```
┌──────────────────────────────────────────────────────────┐
│                      raginspect CLI / Python API          │
│                          (user-facing)                    │
├──────────────┬───────────────┬────────────┬──────────────┤
│   profiler    │  classifier   │  metrics   │  reporter    │
│  (timing,    │  (architecture │ (quality   │  (table,     │
│   resources) │   detection)   │  scoring)  │   JSON, TUI) │
├──────────────┴───────────────┴────────────┴──────────────┤
│                    raginspect-core                        │
│           (pipeline model, types, traits)                 │
└──────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

### 1. `raginspect-core` — Core Engine

The foundation. Defines the pipeline model, timing infrastructure, and shared types.

| Module | Responsibility |
|---|---|
| `pipeline` | `Pipeline`, `Stage`, `StageType` data structures. A pipeline is a directed acyclic graph (DAG) of stages. |
| `profiler` | Timing infrastructure (`StageTimer`, `PipelineProfiler`). Wraps each stage execution with instrumentation. Records wall-clock latency, allocation tracking, and token counters. |
| `config` | Configuration types (`ProfilerConfig`, `OutputFormat`). |
| `error` | Error types via `thiserror`. No `unwrap()` or `panic!` in library code. |
| `result` | `ProfileResult`, `StageResult` — serializable output structures (serde). |

**Key types:**

```rust
pub struct Pipeline {
    pub stages: Vec<Stage>,
    pub architecture: Option<ArchitectureType>,
}

pub struct Stage {
    pub name: String,
    pub stage_type: StageType,
    pub latency: LatencyBreakdown,
    pub tokens: TokenCount,
    pub cost_usd: Option<f64>,
    pub quality_score: Option<f64>,
}

pub enum StageType {
    QueryTransformation,
    Retrieval,
    Reranking,
    Fusion,
    Compression,
    Generation,
    ToolCall,
    GraphTraversal,
}
```

### 2. `raginspect-arch` — Architecture Detection

Classifies pipelines into one of 7 architecture types and applies architecture-appropriate metric defaults.

| Module | Responsibility |
|---|---|
| `classifier` | Pattern matching on pipeline topology (stage sequence, branching, tool calls). Returns `ArchitectureType` with confidence score. |
| `archetypes` | Definitions for each of the 7 architectures. Each archetype contains: topology signature, default metric set, expected stage patterns. |
| `registry` | Registry of all architecture types. New architectures can be registered via the `Architecture` trait. |

**Architecture types:**

```rust
pub enum ArchitectureType {
    Naive,        // query → retrieve → generate
    Advanced,     // + pre/post retrieval processing
    Modular,      // routing, fusion, compression paths
    Agentic,      // tool-calling, multi-step reasoning
    Graph,        // knowledge graph traversal
    HyDE,         // hypothetical document embeddings
    Multimodal,   // text + image + audio retrieval
}
```

**Detection strategy:**

Each architecture has a **topology signature** — a pattern of stage types and their connections. The classifier walks the pipeline DAG and matches against known signatures:

| Signal | Detected Architecture |
|---|---|
| query → retrieve → generate (3 stages, linear) | Naive |
| + reranking or query expansion | Advanced |
| + routing/fusion/compression branching | Modular |
| + tool_call stages or cyclic references | Agentic |
| + graph_traversal stages | Graph |
| + generation before retrieval (hypothetical doc) | HyDE |
| + multimodal input/output types | Multimodal |

### 3. `raginspect-metrics` — Quality Metrics

Rust implementations of RAG quality metrics. These are computational (not LLM-judge-based) where possible, falling back to LLM evaluation for subjective metrics.

| Module | Metric | Method |
|---|---|---|
| `faithfulness` | Is the answer grounded in retrieved context? | Statement decomposition + context entailment |
| `context_relevance` | Is the retrieved context relevant to the query? | Token overlap + embedding similarity |
| `answer_relevancy` | Does the answer address the query? | Query reconstruction + cosine similarity |
| `context_precision` | Are relevant chunks ranked higher? | NDCG@k computation |
| `cost` | Token cost per stage | Provider pricing tables |

Each metric implements the `Metric` trait:

```rust
pub trait Metric: Send + Sync {
    fn name(&self) -> &str;
    fn compute(&self, input: &MetricInput) -> Result<MetricOutput>;
    fn requires_llm(&self) -> bool;
}
```

### 4. `raginspect-bindings` — Python Bindings

pyo3-based Python bindings. See the [Python Binding Strategy](#python-binding-strategy) section below.

### 5. `src/bin/raginspect.rs` — CLI

The CLI entry point. Uses `clap` for argument parsing.

**Commands:**

```
raginspect profile <pipeline>     Profile a RAG pipeline, output results
raginspect classify <pipeline>    Detect and display architecture type
raginspect compare <a> <b>        Compare two pipelines (A/B)
raginspect flamegraph <pipeline>  Generate flamegraph visualization
```

**Output formats:** table (default), JSON, CSV, flamegraph SVG.

---

## Data Flow

```
                    ┌──────────────┐
                    │  Pipeline    │
                    │  Definition  │
                    │ (JSON / Dict)│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Profiler   │
                    │              │
                    │ For each     │
                    │ stage:       │
                    │  - time it   │
                    │  - count tok │
                    │  - track mem │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Classifier  │
                    │              │
                    │ Match topo   │
                    │ to 7 archs   │
                    │ + confidence │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Metrics    │
                    │              │
                    │ Apply arch-  │
                    │ specific     │
                    │ metric suite │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Reporter   │
                    │              │
                    │ Format:      │
                    │ table/JSON/  │
                    │ flamegraph   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Output     │
                    │              │
                    │ stdout /     │
                    │ file / OTel  │
                    └──────────────┘
```

### Step-by-Step

1. **Input**: A pipeline definition — either a JSON file (CLI) or a Python dict/object (bindings). The pipeline describes stages, their types, connections, and input/output data.

2. **Profiler**: Wraps each stage execution with timing (`Instant::now()`), token counting, and optional memory tracking (`jemalloc` stats). Produces a `ProfileResult` with per-stage `StageResult` entries.

3. **Classifier**: Takes the pipeline topology (ignoring timing data) and walks the DAG. Matches against the 7 architecture signatures. Returns `ArchitectureType` + confidence score (0.0–1.0). If confidence < 0.5, falls back to "generic" metric set.

4. **Metrics**: Based on the detected architecture, selects the appropriate metric suite. Computes quality scores for stages where applicable (retrieval → context precision, generation → faithfulness, etc.).

5. **Reporter**: Formats the combined result. Default: rich terminal table (using `comfy-table` or `tabled`). Also supports JSON (via `serde_json`), CSV, and flamegraph SVG export.

6. **Output**: Writes to stdout (CLI), returns as dict (Python), or exports via OpenTelemetry to external backends.

---

## Extension Points

### Adding a New RAG Architecture

Implement the `Architecture` trait and register it:

```rust
// In raginspect-arch/src/archetypes/my_arch.rs

use raginspect_arch::{Architecture, ArchitectureType, MetricSet};

pub struct MyArchRag;

impl Architecture for MyArchRag {
    fn arch_type(&self) -> ArchitectureType {
        ArchitectureType::Custom("MyArch".into())
    }

    fn topology_signature(&self) -> TopologySignature {
        // Define the stage pattern to match on
        TopologySignature::builder()
            .stage(StageType::QueryTransformation)
            .stage(StageType::Retrieval)
            .stage(StageType::Custom("MyStage".into()))
            .stage(StageType::Generation)
            .build()
    }

    fn default_metrics(&self) -> MetricSet {
        MetricSet::builder()
            .metric(Faithfulness::default())
            .metric(MyCustomMetric::default())
            .build()
    }
}

// Register at startup
raginspect_arch::register(MyArchRag);
```

### Adding a New Metric

Implement the `Metric` trait:

```rust
// In raginspect-metrics/src/my_metric.rs

use raginspect_metrics::{Metric, MetricInput, MetricOutput};

pub struct RetrievalLatencyBudget;

impl Metric for RetrievalLatencyBudget {
    fn name(&self) -> &str {
        "retrieval_latency_budget"
    }

    fn compute(&self, input: &MetricInput) -> Result<MetricOutput> {
        let retrieval_ms = input.stage("retrieval")?.latency_p50;
        let total_ms = input.pipeline_latency_p50;

        let ratio = retrieval_ms / total_ms;
        Ok(MetricOutput::Score(ratio))
    }

    fn requires_llm(&self) -> bool {
        false
    }
}
```

### Adding a New Output Format

Implement the `Reporter` trait:

```rust
// In raginspect-core/src/reporter/

pub trait Reporter {
    fn format(&self, result: &ProfileResult) -> Result<String>;
}

pub struct HtmlReporter;

impl Reporter for HtmlReporter {
    fn format(&self, result: &ProfileResult) -> Result<String> {
        // Generate self-contained HTML report
        Ok(html_string)
    }
}
```

---

## Python Binding Strategy

### Stack: pyo3 + maturin

| Component | Choice | Rationale |
|---|---|---|
| **Bindings** | [pyo3](https://pyo3.rs) v0.22+ | Mature Rust↔Python FFI. Active development, excellent community. |
| **Build backend** | [maturin](https://maturin.rs) | Purpose-built for Rust→Python wheels. Handles abi3, manylinux, musllinux. |
| **Type stubs** | Hand-written `.pyi` files | Full IDE autocomplete without runtime cost. |
| **Distribution** | PyPI (pip) + crates.io (cargo) | Dual ecosystem reach. |

### Architecture

```
┌─────────────────────────────────────────────┐
│           Python User Code                   │
│  raginspect.profile(pipeline) → dict         │
├─────────────────────────────────────────────┤
│         raginspect (Python package)          │
│  Thin wrapper, type hints, ergonomic API     │
├─────────────────────────────────────────────┤
│         pyo3 FFI Boundary                     │
│  Zero-copy where possible (PyBytes, buffers) │
├─────────────────────────────────────────────┤
│         raginspect-bindings crate             │
│  #[pyfunction] wrappers, type conversions    │
├─────────────────────────────────────────────┤
│    raginspect-core / arch / metrics           │
│         (pure Rust, no Python deps)           │
└─────────────────────────────────────────────┘
```

### Design Principles

1. **Ergonomic Python API** — Users shouldn't know Rust is involved. Follow Python conventions (snake_case, dict returns, context managers where natural).
2. **Zero-copy interop** — Use `PyBytes` and buffer protocol for large payloads (embeddings, token arrays). Avoid serializing through JSON internally.
3. **GIL release** — All computation-heavy functions release the GIL via `py.allow_threads()`. This is the entire point of using Rust.
4. **Error mapping** — Rust `Result<T, E>` → Python exceptions. `raginspect::Error::Profiler` → `raginspect.ProfilerError`, etc.
5. **Async-friendly** — Provide both sync and async APIs. Async uses PyO3's `pyo3/async` for coroutine support.

### Wheel Targets

| Platform | Target Triple | Wheel Tag |
|---|---|---|
| Linux x64 | `x86_64-unknown-linux-gnu` | `cp38-abi3-manylinux_2_17_x86_64` |
| Linux ARM64 | `aarch64-unknown-linux-gnu` | `cp38-abi3-manylinux_2_17_aarch64` |
| macOS Intel | `x86_64-apple-darwin` | `cp38-abi3-macosx_10_9_x86_64` |
| macOS ARM | `aarch64-apple-darwin` | `cp38-abi3-macosx_11_0_arm64` |
| Windows x64 | `x86_64-pc-windows-msvc` | `cp38-abi3-win_amd64` |

Using abi3 (stable Python ABI) means one wheel per platform supports Python 3.8+.

### Build & Publish

```bash
# Local development
maturin develop --release

# Build wheels for PyPI
maturin build --release
# CI uses maturin-action for cross-platform builds

# Publish
maturin publish
```

---

## Dependencies (Key)

| Crate | Version | Purpose |
|---|---|---|
| `serde` / `serde_json` | 1.0 | Serialization everywhere |
| `clap` | 4.5 | CLI argument parsing |
| `thiserror` | 1.0 | Ergonomic error types |
| `comfy-table` | 7.0 | Terminal table output |
| `tracing` | 0.1 | Structured logging |
| `pyo3` | 0.22 | Python bindings (bindings crate only) |
| `maturin` | 1.7 | Build system (dev dependency) |
| `criterion` | 0.5 | Benchmarks |

---

*Last updated: 2026-08-10 — Sprint 0*
