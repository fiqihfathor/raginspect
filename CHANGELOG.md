# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-21

First public release. 🔍 X-ray for RAG pipelines — Rust core, CLI, and Python bindings.

### Core Inspection Engine
- 3-layer inspection: retrieval, context construction, generation grounding
- 7 RAG architectures supported: Naive, Advanced, Modular, Agentic, Graph, HyDE, Multimodal
- RAG Health Score (0–100) with **architecture-specific metric weights** (Naive=efficiency-heavy, Agentic=grounding-heavy, etc.) and `[metrics]` config override
- Architecture auto-classification: `TopologyAnalyzer` extracts pipeline components → `ArchitectureClassifier` pattern-matches with confidence score + reason
- Actionable diagnostics (e.g. "Lower Top-K from 5 to 4", "Enable dedup threshold > 0.85")

### CLI (`raginspect` binary)
- `inspect` subcommand: 4 modes (full, retrieval, context, quick), JSON output, config override
- `profile` subcommand: per-stage timing, multi-run p50/p99 stats, color-coded table/JSON output, config validation, `--runs` guard, `OutputFormat` enum (invalid values rejected at parse time)

### Profiler Library
- `Stage` / `StageTimer` / `PipelineProfile` — stage-level timing, tokens, cost
- `MultiRunProfiler` — percentile stats (p50/p99/min/max/mean) across runs
- `NaivePipeline` — fully offline mock RAG pipeline (4 stages, deterministic embeddings, custom corpus/latencies)
- Optional `memory-tracking` feature (sysinfo)

### Python Bindings
- pyo3 0.22 + maturin, abi3 (Python 3.8+), manylinux wheels via CI
- High-level typed API: `inspect_pipeline()` → `InspectionReport` dataclass (with `.summary()`), `classify_pipeline()` → `ClassificationResult`
- Raw `profile()` / `classify()` dict API also exported
- Type stubs (`.pyi`) + `py.typed` (PEP 561) for IDE support
- CI/CD: `wheels.yml` builds + smoke-tests wheels on `v*` tags, publishes to PyPI

### Pipeline Configuration (TOML)
- Core sections: `[embedding]`, `[vector_store]`, `[llm]`, `[context]`
- Optional architecture components: `[reranking]`, `[fusion]`, `[routing]`, `[tools]`, `[graph]`, `[hyde]`, `[multimodal]`
- Metric weight overrides: `[metrics]`
- Partial configs merge onto sensible defaults

### Quality
- **124 tests** (unit + integration + bindings + doctests), CI-enforced fmt/clippy/test/build
- NaN-safe float sorting (`total_cmp`) throughout the profiler
- Post-merge code review fixes: CI config validation, error propagation, dead-field removal

### Docs
- `docs/quickstart.md` — step-by-step tutorial (build, CLI, library, config reference)
- `docs/architecture.md` — module breakdown
- README with feature comparison and install paths (crates.io, PyPI, source)

## [0.1.0-rc] - 2026-08-09

Initial internal release: core inspection engine, terminal reporter, JSON mode, TOML config, tiktoken token counting.
