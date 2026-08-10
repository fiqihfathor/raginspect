use anyhow::Result;
use chrono::Utc;
use tiktoken_rs::cl100k_base;

use crate::config::PipelineConfig;
use crate::types::{
    ChunkInfo, ContextStage, GenerationResult, InspectMode, InspectionReport, RagArchitecture,
    RetrievalResult, Verdict,
};

/// Core inspection engine that runs a query through a RAG pipeline and captures diagnostic metrics.
pub struct Inspector {
    config: PipelineConfig,
    model_override: Option<String>,
    architecture: RagArchitecture,
}

impl Inspector {
    /// Create a new Inspector instance with a given pipeline config.
    pub fn new(config: PipelineConfig, model_override: Option<String>) -> Self {
        Self {
            config,
            model_override,
            architecture: RagArchitecture::Naive,
        }
    }

    /// Set the RAG architecture type for architecture-specific diagnostics.
    pub fn set_architecture(&mut self, architecture: RagArchitecture) {
        self.architecture = architecture;
    }

    /// Run inspection on the input query.
    pub fn inspect(&self, query: &str, mode: InspectMode) -> Result<InspectionReport> {
        let active_model = self
            .model_override
            .as_deref()
            .unwrap_or(&self.config.llm.model)
            .to_string();

        let start_time = std::time::Instant::now();

        // 1. Simulate Vector Retrieval Stage
        let retrieval = self.simulate_retrieval(query)?;

        // 2. Simulate Context Construction & Token Efficiency Stage
        let context = self.analyze_context_construction(&retrieval)?;

        // 3. Simulate LLM Generation & Grounding Stage
        let generation = self.analyze_generation(query, &retrieval)?;

        // 4. Calculate Overall RAG Health Score
        let overall_score = self.calculate_health_score(&retrieval, &context, &generation);

        // 5. Generate Diagnostic Recommendations
        let mut recommendations = self.generate_recommendations(&retrieval, &context, &generation);

        // 6. Architecture-specific recommendations
        recommendations.extend(self.architecture_recommendations());

        let _elapsed = start_time.elapsed().as_millis();

        Ok(InspectionReport {
            query: query.to_string(),
            config_name: self.config.name.clone(),
            model_name: active_model,
            architecture: self.architecture,
            inspect_mode: mode,
            timestamp: Utc::now().to_rfc3339(),
            retrieval,
            context,
            generation,
            overall_score,
            recommendations,
        })
    }

    /// Accurately count tokens using BPE tokenizer with character fallback.
    fn count_tokens(&self, text: &str) -> usize {
        if let Ok(bpe) = cl100k_base() {
            bpe.encode_with_special_tokens(text).len()
        } else {
            // Fallback estimation (~4 chars per token for English text)
            (text.len() as f64 / 4.0).ceil() as usize
        }
    }

    /// Simulate vector database retrieval with realistic similarity scoring and chunk metrics.
    fn simulate_retrieval(&self, query: &str) -> Result<RetrievalResult> {
        let top_k = self.config.vector_store.top_k;
        let mut chunks = Vec::new();

        // Domain-aware mock data generation based on query keywords
        let is_rust_query = query.to_lowercase().contains("rust")
            || query.to_lowercase().contains("tokio")
            || query.to_lowercase().contains("async")
            || query.to_lowercase().contains("memory");

        let mock_templates = if is_rust_query {
            vec![
                (
                    "docs/tokio_architecture.md#chunk_01",
                    "Tokio task allocation memory overhead is minimal. Each async task requires ~64 bytes for allocation headers, plus stack frame state sized at compile-time by the future generator.",
                    0.94,
                    Verdict::Relevant,
                    true,
                    "Direct exact match for task memory overhead details",
                ),
                (
                    "docs/tokio_runtime.md#chunk_04",
                    "The Tokio multi-threaded runtime maintains worker queues. Tasks scheduled across workers incur zero-copy channel passing when using atomic ring buffers.",
                    0.86,
                    Verdict::PartiallyRelevant,
                    true,
                    "Relevant architectural context regarding worker task queues",
                ),
                (
                    "docs/rust_memory_model.md#chunk_02",
                    "Tokio async task memory overhead consists of 64-byte allocation metadata and compiler-generated state machine structs.",
                    0.84,
                    Verdict::Duplicate,
                    false,
                    "91% semantic overlap with chunk_01 (redundant context)",
                ),
                (
                    "docs/async_std_comparison.md#chunk_08",
                    "Async-std uses task budgets similar to Tokio to prevent cooperative starvation in busy event loops.",
                    0.58,
                    Verdict::PartiallyRelevant,
                    false,
                    "Peripheral comparison context; not directly addressing memory overhead",
                ),
                (
                    "docs/postgres_connector.md#chunk_12",
                    "Database connection pooling in Rust requires connection timeout configuration under heavy TCP loads.",
                    0.38,
                    Verdict::Irrelevant,
                    false,
                    "Off-topic vector match (low similarity 0.38 < threshold 0.65)",
                ),
            ]
        } else {
            vec![
                (
                    "docs/rag_overview.md#chunk_01",
                    "Retrieval-Augmented Generation (RAG) combines dense vector search with large language models to ground responses in external knowledge stores.",
                    0.92,
                    Verdict::Relevant,
                    true,
                    "Direct definition and primary concept match",
                ),
                (
                    "docs/vector_databases.md#chunk_03",
                    "Vector indexes like HNSW and IVF-PQ provide fast approximate nearest neighbor search over high-dimensional embedding vectors.",
                    0.83,
                    Verdict::PartiallyRelevant,
                    true,
                    "Provides foundational vector search mechanics",
                ),
                (
                    "docs/rag_architecture.md#chunk_02",
                    "RAG pipelines retrieve relevant documents from vector indices to inject into LLM prompts for grounded text generation.",
                    0.81,
                    Verdict::Duplicate,
                    false,
                    "88% semantic overlap with chunk_01 (redundant definition)",
                ),
                (
                    "docs/prompt_engineering.md#chunk_05",
                    "Few-shot prompt formatting improves instruction following across generic LLM benchmarks.",
                    0.61,
                    Verdict::PartiallyRelevant,
                    false,
                    "General prompt context; weak alignment to specific retrieval query",
                ),
                (
                    "docs/deployment_kubernetes.md#chunk_10",
                    "Deploying microservices on Kubernetes clusters requires proper readiness probe timeouts and resource limits.",
                    0.35,
                    Verdict::Irrelevant,
                    false,
                    "Unrelated infrastructure chunk retrieved due to sparse index noise",
                ),
            ]
        };

        let mut total_tokens = 0;
        let mut sum_similarity = 0.0;

        for (i, (source, content, score, verdict, is_used, reason)) in
            mock_templates.into_iter().enumerate()
        {
            if i >= top_k {
                break;
            }
            let token_count = self.count_tokens(content);
            total_tokens += token_count;
            sum_similarity += score;

            chunks.push(ChunkInfo {
                id: format!("chunk_{:02}", i + 1),
                source_doc: source.to_string(),
                content: content.to_string(),
                score,
                token_count,
                verdict,
                is_used_in_gen: is_used,
                verdict_reason: reason.to_string(),
            });
        }

        let chunks_len = chunks.len();
        let avg_similarity = if chunks_len > 0 {
            sum_similarity / chunks_len as f64
        } else {
            0.0
        };

        Ok(RetrievalResult {
            top_k,
            chunks_retrieved: chunks_len,
            total_retrieved_tokens: total_tokens,
            latency_ms: 42,
            chunks,
            avg_similarity,
        })
    }

    /// Analyze context window usage, identifying useful vs wasted tokens.
    fn analyze_context_construction(&self, retrieval: &RetrievalResult) -> Result<ContextStage> {
        let mut useful_tokens = 0;
        let mut wasted_tokens = 0;
        let mut deduplicated_chunks = 0;
        let mut irrelevant_chunks_pruned = 0;

        for chunk in &retrieval.chunks {
            match chunk.verdict {
                Verdict::Relevant | Verdict::PartiallyRelevant => {
                    useful_tokens += chunk.token_count;
                }
                Verdict::Duplicate => {
                    wasted_tokens += chunk.token_count;
                    deduplicated_chunks += 1;
                }
                Verdict::Irrelevant | Verdict::LowConfidence => {
                    wasted_tokens += chunk.token_count;
                    irrelevant_chunks_pruned += 1;
                }
            }
        }

        let total_tokens = useful_tokens + wasted_tokens;
        let efficiency_ratio = if total_tokens > 0 {
            useful_tokens as f64 / total_tokens as f64
        } else {
            1.0
        };

        Ok(ContextStage {
            total_tokens,
            useful_tokens,
            wasted_tokens,
            efficiency_ratio,
            deduplicated_chunks,
            irrelevant_chunks_pruned,
            context_window_limit: self.config.context.max_context_tokens,
        })
    }

    /// Analyze generation output, checking hallucination risk and citation attributions.
    fn analyze_generation(
        &self,
        query: &str,
        retrieval: &RetrievalResult,
    ) -> Result<GenerationResult> {
        let is_rust =
            query.to_lowercase().contains("rust") || query.to_lowercase().contains("tokio");

        let generated_text = if is_rust {
            "Tokio async task memory overhead is lightweight, requiring approximately 64 bytes for task allocation headers alongside compiler-generated state machine frames [chunk_01]. Worker threads schedule these tasks efficiently using lock-free atomic ring buffers [chunk_02]. Note that Tokio automatically caps default task stack size at 2MB per thread."
        } else {
            "Retrieval-Augmented Generation (RAG) is an AI architecture that enhances LLMs by retrieving relevant context chunks from vector databases [chunk_01]. Vector databases index document embeddings using algorithms like HNSW for low-latency similarity search [chunk_02]."
        }.to_string();

        let prompt_tokens = retrieval.total_retrieved_tokens + self.count_tokens(query) + 50;
        let completion_tokens = self.count_tokens(&generated_text);

        let cited_chunk_ids = vec!["chunk_01".to_string(), "chunk_02".to_string()];

        let uncited_claims = if is_rust {
            vec!["Claim 'Tokio caps default task stack size at 2MB per thread' is not supported by retrieved context chunks.".to_string()]
        } else {
            vec![]
        };

        let hallucination_score = if uncited_claims.is_empty() {
            0.04
        } else {
            0.18
        };

        let source_attribution_pct = 1.0 - hallucination_score;

        Ok(GenerationResult {
            generated_text,
            prompt_tokens,
            completion_tokens,
            latency_ms: 385,
            hallucination_score,
            source_attribution_pct,
            cited_chunk_ids,
            uncited_claims,
        })
    }

    /// Calculate aggregate RAG health score (0-100).
    fn calculate_health_score(
        &self,
        retrieval: &RetrievalResult,
        context: &ContextStage,
        generation: &GenerationResult,
    ) -> f64 {
        // Relevance Component (0-100)
        let relevance_score = (retrieval.avg_similarity * 100.0).min(100.0);

        // Efficiency Component (0-100)
        let efficiency_score = context.efficiency_ratio * 100.0;

        // Grounding Component (0-100)
        let grounding_score = generation.source_attribution_pct * 100.0;

        // Weighted Average: 35% Relevance, 35% Efficiency, 30% Grounding
        (relevance_score * 0.35 + efficiency_score * 0.35 + grounding_score * 0.30).round()
    }

    /// Generate actionable recommendations based on diagnostic findings.
    fn generate_recommendations(
        &self,
        retrieval: &RetrievalResult,
        context: &ContextStage,
        generation: &GenerationResult,
    ) -> Vec<String> {
        let mut recs = Vec::new();

        if context.wasted_tokens > 0 {
            recs.push(format!(
                "Token Waste Detected: {} out of {} tokens ({:.1}%) in context were wasted on duplicate or irrelevant chunks. Enable similarity deduplication threshold > {:.2}.",
                context.wasted_tokens,
                context.total_tokens,
                (1.0 - context.efficiency_ratio) * 100.0,
                self.config.context.deduplicate_threshold
            ));
        }

        for chunk in &retrieval.chunks {
            if chunk.verdict == Verdict::Duplicate {
                recs.push(format!(
                    "Redundant Retrieval: Chunk '{}' is a near-duplicate of a higher ranked chunk. Lower Top-K from {} to {} or adjust chunking stride.",
                    chunk.id, retrieval.top_k, retrieval.top_k.saturating_sub(1)
                ));
            }
            if chunk.verdict == Verdict::Irrelevant {
                recs.push(format!(
                    "Low Quality Match: Chunk '{}' had similarity score {:.2} (below threshold {:.2}). Increase vector similarity threshold.",
                    chunk.id, chunk.score, self.config.vector_store.similarity_threshold
                ));
            }
        }

        if !generation.uncited_claims.is_empty() {
            recs.push(format!(
                "Hallucination Risk: Detected {} uncited claim(s) in LLM generation. Consider adding stricter system instructions for source attribution.",
                generation.uncited_claims.len()
            ));
        }

        if recs.is_empty() {
            recs.push("Pipeline Health Nominal: High retrieval relevance, efficient context packing, and grounded generation.".to_string());
        }

        recs
    }

    /// Generate architecture-specific diagnostic recommendations.
    fn architecture_recommendations(&self) -> Vec<String> {
        let mut recs = Vec::new();

        match self.architecture {
            RagArchitecture::Naive => {
                recs.push(
                    "Naive Architecture: Consider upgrading to Advanced for re-ranking and hybrid search to improve retrieval precision.".to_string(),
                );
                if self.config.vector_store.top_k > 5 {
                    recs.push(format!(
                        "Naive Architecture: Top-K={} is high for a linear pipeline — unused chunks waste context tokens. Consider K=3–4.",
                        self.config.vector_store.top_k
                    ));
                }
            }
            RagArchitecture::Advanced => {
                recs.push(
                    "Advanced Architecture: Verify cross-encoder re-ranker model is warmed up — cold-start re-ranking degrades precision by ~12%.".to_string(),
                );
                recs.push(
                    "Advanced Architecture: Monitor hybrid search fusion weights (α) — drift toward BM25 or dense-only reduces robustness.".to_string(),
                );
            }
            RagArchitecture::Modular => {
                recs.push(
                    "Modular Architecture: Validate inter-module schema contracts — loose serialization between stages causes silent data loss.".to_string(),
                );
                recs.push(
                    "Modular Architecture: Instrument per-stage latency to identify pipeline bottlenecks in the DAG.".to_string(),
                );
            }
            RagArchitecture::Agentic => {
                recs.push(
                    "Agentic Architecture: Cap retrieval iteration depth to prevent token explosion from unbounded self-correction loops.".to_string(),
                );
                recs.push(
                    "Agentic Architecture: Audit LLM routing decisions — misrouted queries to wrong tools waste budget and degrade answer quality.".to_string(),
                );
                recs.push(
                    "Agentic Architecture: Log per-hop token usage to detect runaway agent loops early.".to_string(),
                );
            }
            RagArchitecture::Graph => {
                recs.push(
                    "Graph Architecture: Verify entity extraction coverage — missing entities create orphan nodes invisible to traversal.".to_string(),
                );
                recs.push(
                    "Graph Architecture: Tune community detection resolution — coarse clusters dilute relevance, fine clusters fragment context.".to_string(),
                );
                recs.push(format!(
                    "Graph Architecture: Current similarity threshold {:.2} may prune valid cross-entity edges — consider graph-aware thresholding.",
                    self.config.vector_store.similarity_threshold
                ));
            }
            RagArchitecture::Hyde => {
                recs.push(
                    "HyDE Architecture: Monitor hypothetical document quality — poor generations embed closer to wrong clusters, hurting precision.".to_string(),
                );
                recs.push(
                    "HyDE Architecture: Track embedding drift between hypothetical and real document distributions — drift > 0.15 cosine indicates model mismatch.".to_string(),
                );
            }
            RagArchitecture::Multimodal => {
                recs.push(
                    "Multimodal Architecture: Audit modality weight distribution — dominance of one modality signals fusion layer under-tuning.".to_string(),
                );
                recs.push(
                    "Multimodal Architecture: Measure modality gap score — large gaps indicate misaligned cross-modal embedding spaces.".to_string(),
                );
                recs.push(
                    "Multimodal Architecture: Verify that image and text embeddings share a comparable norm distribution before fusion.".to_string(),
                );
            }
        }

        recs
    }
}
