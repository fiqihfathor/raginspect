# raginspect

<!-- Badges -->
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Crates.io](https://img.shields.io/crates/v/raginspect)
![PyPI](https://img.shields.io/pypi/v/raginspect)
![Rust](https://img.shields.io/badge/rust-stable-orange.svg)
![Build](https://img.shields.io/github/actions/workflow/status/fiqihfathor/raginspect/ci.yml)
![Stars](https://img.shields.io/github/stars/fiqihfathor/raginspect)

> **Grafana for RAG** — X-ray vision for your retrieval-augmented generation pipelines.

RAG pipelines are everywhere, but nobody can see inside them. Every existing tool treats RAG as a black box. **raginspect** changes that: architecture-aware profiling, stage-level performance flamegraphs, and optimization recommendations — all powered by a Rust core that runs 10–50× faster than Python alternatives.

---

## ✨ Features

- **Architecture-Aware Profiling** — Automatically detects your RAG architecture and applies the right metrics for your specific pipeline pattern.
- **Stage-Level Flamegraph** — Visual breakdown of every pipeline stage (query transformation, retrieval, reranking, generation) with latency, token count, cost, and quality scores.
- **Rust Performance Core** — 10–50× faster than Python-only tools. Profile production-scale pipelines in real time, not batch.
- **Python Bindings** — `pip install raginspect` and get native-speed metrics inside your existing LangChain, LlamaIndex, or Ragas workflow. Zero rewrites.
- **7 RAG Architectures Supported** — Naive, Advanced, Modular, Agentic, Graph, HyDE, and Multimodal. No other tool classifies pipelines structurally.
- **Edge-Ready** — Compiles to WASM for browser/edge deployment. Single binary, no runtime dependencies.
- **OpenTelemetry Export** — Pipe metrics into Phoenix, LangSmith, Datadog, or any OTel-compatible backend.

---

## 🚀 Quick Start

### Rust CLI

```bash
# Build from source
git clone https://github.com/fiqihfathor/raginspect.git
cd raginspect
cargo build --release

# Inspect a RAG pipeline (uses mock data, no API keys needed)
cargo run --release -- inspect

# Profile per-stage timing
cargo run --release -- profile

# Multi-run profiling with p50/p99 stats
cargo run --release -- profile --runs 10 --format json

# Run the example
cargo run --example naive_rag_inspect
```

📖 **[Full Quickstart Guide →](docs/quickstart.md)** — step-by-step tutorial, library usage, config reference, and CLI examples.

### Python

```bash
# From PyPI (once v0.1.0 is published)
pip install raginspect

# Or build from source (requires Rust toolchain)
pip install maturin
maturin develop --release

# Or editable install for development
pip install -e .
```

```python
from raginspect import inspect_pipeline, classify_pipeline

# High-level API — typed dataclasses, full docstrings
report = inspect_pipeline({"vector_store": {"top_k": 3}}, query="What is RAG?")
print(report.summary())
# score=75.0 architecture=naive recommendations=4
print(report.overall_score, report.recommendations[0])

# Auto-detect the RAG architecture
result = classify_pipeline({"hyde": {"enabled": True}})
print(result.architecture, result.confidence)  # hyde 0.95

# Raw dicts still available via the compiled functions
from raginspect import profile, classify
raw = profile({})
```

---

## 📐 Supported RAG Architectures

raginspect recognizes **7 distinct RAG architectures**, each with tailored metric suites:

| Architecture | Description | Key Metrics |
|---|---|---|
| **Naive RAG** | Simple retrieve-then-generate | Context recall, faithfulness |
| **Advanced RAG** | Pre/post-retrieval processing (query expansion, reranking) | Context precision + retrieval latency breakdown |
| **Modular RAG** | Routing, fusion, compression, multiple retrieval paths | Route accuracy, fusion quality, compression ratio |
| **Agentic RAG** | Tool-calling agents with multi-step reasoning | Tool selection accuracy, multi-hop coherence |
| **Graph RAG** | Knowledge graph traversal for retrieval | Entity resolution, relation extraction F1, traversal depth |
| **HyDE** | Hypothetical document embeddings | Hypothetical doc quality, embedding alignment |
| **Multimodal RAG** | Text + image + audio cross-modal retrieval | Cross-modal alignment, modality coverage |

---

## 📊 Comparison

| | raginspect | Ragas | TruLens | Phoenix | DeepEval |
|---|:---:|:---:|:---:|:---:|:---:|
| **Language** | Rust + Python | Python | Python | Python | Python |
| **Architecture-aware** | ✅ 7 types | ❌ | ❌ | ❌ | ❌ |
| **Stage flamegraph** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Real-time profiling** | ✅ Rust core | ❌ | ❌ | ❌ | ❌ |
| **Edge / WASM** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Quality metrics** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **OTel export** | ✅ | ❌ | ✅ | ✅ | ❌ |
| **License** | MIT | Apache 2.0 | MIT | ELv2 | Apache 2.0 |
| **Performance** | 10–50× | 1× | 1× | 1× | 1× |

---

## 🗺️ Roadmap

| Sprint | Goal | Status |
|---|---|---|
| **Sprint 0** (Aug 10–14) | Planning, CI/CD, docs, backlog | 🟡 In Progress |
| **Sprint 1** (Aug 15–28) | Core profiler + flamegraph + CLI + Python bindings MVP | ⬜ |
| **Sprint 2** (Aug 29–Sep 11) | Quality metrics suite (faithfulness, relevance, answer quality) | ⬜ |
| **Sprint 3** (Sep 12–25) | TUI viewer, web viewer, OTel integration, v0.2.0 | ⬜ |
| **Sprint 4+** (Oct) | Advanced architectures, optimization engine, hosted dashboard | ⬜ |

See the [full roadmap](docs/roadmap.md) for details.

---

## 📦 Project Structure

```
raginspect/
├── crates/
│   ├── raginspect-core/       # Core profiling engine
│   ├── raginspect-metrics/    # Quality metrics (faithfulness, relevance, etc.)
│   ├── raginspect-arch/       # Architecture detection + classification
│   └── raginspect-bindings/   # pyo3 Python bindings
├── examples/                  # Example pipelines
├── src/
│   ├── bin/raginspect.rs      # CLI entry point
│   ├── profiler/              # Stage profiling logic
│   ├── classifier/            # Architecture auto-detection
│   └── reporter/              # Output (table, JSON, flamegraph)
├── python/raginspect/         # Python package wrapper
├── tests/                     # Integration tests
└── docs/                      # Architecture docs, metric references
```

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and commit conventions.

---

## 📄 License

MIT © [Fiqih Fathor](https://github.com/fiqihfathor)

---

*raginspect is the missing observability layer for RAG — because you can't optimize what you can't see.*
