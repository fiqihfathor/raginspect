# 🔍 `raginspect` — A RAG Inspection & Profiling Engine

[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![RAG Diagnostics](https://img.shields.io/badge/diagnostics-X--Ray-orange.svg)]()

> **An X-ray diagnostic and profiling tool for Retrieval-Augmented Generation (RAG) pipelines.**  
> Inspect each layer of your RAG architecture—retrieval relevance, context token efficiency, and LLM output grounding—to catch context waste, hallucinations, and retrieval degradation before production.

---

## ⚡ Key Features

- **🔍 Multi-Layer Pipeline Inspection**:
  - **Layer 1: Retrieval Analysis**: Inspects top-K vector search results, similarity scores, token counts, and relevance verdicts (`RELEVANT`, `PARTIAL`, `IRRELEVANT`, `DUPLICATE`).
  - **Layer 2: Context Window & Token Efficiency**: Measures total tokens fed into context vs. tokens wasted on duplicate or off-topic context chunks.
  - **Layer 3: Generation & Grounding**: Evaluates LLM completion latency, source attribution percentage, and flags hallucinated / uncited claims.
- **📊 RAG Health Score (0–100)**: Calculates an aggregate quality score combining retrieval precision, context packing efficiency, and response grounding.
- **🛠️ Actionable Diagnostics**: Delivers exact recommendations (e.g. "Lower top-K from 5 to 4", "Enable deduplication threshold > 0.85").
- **💻 CLI & JSON Output Modes**: Full rich terminal tabular reporting or machine-readable JSON output for CI/CD pipeline automated regression testing.

---

## 🏗️ Architecture Overview

```
                          ┌────────────────────────┐
                          │   Input User Query     │
                          └───────────┬────────────┘
                                      │
                                      ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  LAYER 1: VECTOR RETRIEVAL                                       │
    │  - Top-K Similarity Scoring (Cosine / Dot Product)              │
    │  - Relevance & Redundancy Verdicts                               │
    └─────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  LAYER 2: CONTEXT CONSTRUCTION                                   │
    │  - Tiktoken BPE Token Counting                                   │
    │  - Useful vs. Wasted Token Ratio Gauge                           │
    └─────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  LAYER 3: GENERATION & GROUNDING                                 │
    │  - Hallucination Index & Citation Matching                        │
    │  - Source Attribution & Uncited Claim Flagging                   │
    └─────────────────────────────┬────────────────────────────────────┘
                                  │
                                  ▼
                          ┌────────────────────────┐
                          │  RAG X-Ray Report      │
                          │  & Health Score        │
                          └────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Build from Source

```bash
cargo build --release
```

### 2. Run Diagnostic Inspection

```bash
# Run full inspection with default query
cargo run -- -q "What is the memory overhead of Tokio tasks in Rust?"

# Run inspection with custom pipeline configuration
cargo run -- --query "Explain vector search indexing in Qdrant" --pipeline-config configs/sample.toml

# Inspect vector retrieval layer only
cargo run -- -q "How does RAG context deduplication work?" --inspect-mode retrieval

# Export diagnostic report as JSON for CI/CD checking
cargo run -- -q "Async runtime task allocation overhead" --json > report.json
```

---

## ⚙️ Configuration Reference

Pipeline behavior is configured using TOML (`configs/sample.toml`):

```toml
name = "Production-Search-Pipeline"
description = "Production RAG configuration evaluating vector retrieval, token usage, and LLM grounding"

[embedding]
model = "text-embedding-3-small"
dimension = 1536
distance_metric = "cosine"

[vector_store]
provider = "qdrant"
collection = "technical_docs_v2"
top_k = 5
similarity_threshold = 0.65

[llm]
provider = "openai"
model = "gpt-4o-mini"
endpoint = "https://api.openai.com/v1"
max_tokens = 1024
temperature = 0.2

[context]
max_context_tokens = 4096
deduplicate_threshold = 0.85
prune_irrelevant = true
```

---

## 📋 Sample Terminal Output

```
================================================================================
 🔍 RAGINSPECT — Retrieval Augmented Generation Inspection & Profiling Engine
================================================================================

📋 INSPECTION METADATA & HEALTH SCORE
┌───────────────────────┬─────────────────────────────────────────────────────┐
│ Target Query          │ What is the memory overhead of Tokio tasks in Rust? │
│ Pipeline Config       │ Production-Search-Pipeline                          │
│ LLM Model             │ gpt-4o-mini                                         │
│ Inspection Mode       │ Full Pipeline                                       │
│ RAG Health Score      │ 87.2/100 [EXCELLENT]                                │
└───────────────────────┴─────────────────────────────────────────────────────┘

⚡ LAYER 1: VECTOR RETRIEVAL ANALYSIS (Top-K: 5, Retrieved: 5, Latency: 42ms)
┌──────────┬─────────────────────────────────────┬───────┬────────┬─────────────┬─────────────────────────────────────────┐
│ ID       │ Source Document                     │ Score │ Tokens │ Verdict     │ Diagnostic Rationale                    │
├──────────┼─────────────────────────────────────┼───────┼────────┼─────────────┼─────────────────────────────────────────┤
│ chunk_01 │ docs/tokio_architecture.md#chunk_01 │ 0.940 │ 32     │ ✅ RELEVANT │ Direct exact match for memory overhead  │
│ chunk_02 │ docs/tokio_runtime.md#chunk_04      │ 0.860 │ 28     │ ⚠️ PARTIAL  │ Relevant architectural context          │
│ chunk_03 │ docs/rust_memory_model.md#chunk_02  │ 0.840 │ 29     │ 🔄 DUPLICATE│ 91% semantic overlap with chunk_01      │
│ chunk_04 │ docs/async_std_comparison.md        │ 0.580 │ 22     │ ⚠️ PARTIAL  │ Peripheral comparison context           │
│ chunk_05 │ docs/postgres_connector.md         │ 0.380 │ 21     │ ❌ IRRELEVANT│ Low similarity score (0.38 < 0.65)       │
└──────────┴─────────────────────────────────────┴───────┴────────┴─────────────┴─────────────────────────────────────────┘

📊 LAYER 2: CONTEXT CONSTRUCTION & TOKEN EFFICIENCY
┌──────────────────────────────┬──────────────────────────────────────────────┐
│ Token Efficiency Gauge       │ [██████████████████████░░░░░░░] 60.0%        │
│ Useful Context Tokens        │ 60 tokens                                    │
│ Wasted Context Tokens        │ 50 tokens (Duplicates/Noise)                 │
└──────────────────────────────┴──────────────────────────────────────────────┘

🛠️ ACTIONABLE X-RAY RECOMMENDATIONS
  1. Token Waste Detected: 50 out of 110 tokens (45.5%) in context were wasted on duplicate or irrelevant chunks.
  2. Redundant Retrieval: Chunk 'chunk_03' is a near-duplicate of a higher ranked chunk. Lower Top-K from 5 to 4.
  3. Hallucination Risk: Detected 1 uncited claim in LLM generation. Stricter prompt instruction recommended.
```

---

## 🛣️ Roadmap & Future Phases

- [x] **Phase 1**: CLI one-shot inspection with simulated pipeline layers, token counting, and rich terminal reporting.
- [ ] **Phase 2**: Live vector store adapters (Qdrant, Pinecone, Chroma) & LLM API integration (OpenAI, Ollama, Anthropic).
- [ ] **Phase 3**: Interactive TUI dashboard built with `ratatui` for real-time stream inspection.

---

## 📄 License

Dual-licensed under MIT License.
