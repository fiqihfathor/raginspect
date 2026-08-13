//! Naive RAG Reference Pipeline — offline mock pipeline for demo and testing.
//!
//! This module provides a simple, self-contained Naive RAG pipeline that requires
//! no external API keys. It demonstrates the profiling flow end-to-end:
//!
//! 1. Query embedding (mock)
//! 2. Vector search (mock corpus)
//! 3. Context assembly (token budgeting)
//! 4. Generation (mock LLM)
//!
//! ## Usage
//!
//! ```no_run
//! use raginspect::naive_pipeline::NaivePipeline;
//! use raginspect::StageTimer;
//!
//! let mut pipeline = NaivePipeline::new();
//! let profile = pipeline.run("What is Rust?", 5);
//! println!("{}", profile);
//! ```

use crate::profiler::{PipelineProfile, StageTimer};
use serde::{Deserialize, Serialize};

/// A mock document chunk for the naive RAG corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockChunk {
    /// Chunk identifier
    pub id: String,
    /// Source document name
    pub source: String,
    /// Text content
    pub content: String,
    /// Mock embedding vector (simplified as a single f64 "relevance score")
    pub mock_embedding: f64,
}

/// Mock LLM response simulating generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockGeneration {
    /// Generated answer text
    pub answer: String,
    /// Number of input (prompt) tokens
    pub prompt_tokens: usize,
    /// Number of output (completion) tokens
    pub completion_tokens: usize,
}

/// Naive RAG pipeline with mock components — runs fully offline.
pub struct NaivePipeline {
    /// Mock document corpus
    corpus: Vec<MockChunk>,
    /// Simulated embedding latency (ms)
    embedding_latency_ms: u128,
    /// Simulated search latency (ms)
    search_latency_ms: u128,
    /// Simulated generation latency (ms)
    generation_latency_ms: u128,
}

impl NaivePipeline {
    /// Create a new naive pipeline with a pre-built mock corpus.
    pub fn new() -> Self {
        Self {
            corpus: default_corpus(),
            embedding_latency_ms: 8,
            search_latency_ms: 25,
            generation_latency_ms: 450,
        }
    }

    /// Create a pipeline with a custom corpus.
    pub fn with_corpus(corpus: Vec<MockChunk>) -> Self {
        Self {
            corpus,
            embedding_latency_ms: 8,
            search_latency_ms: 25,
            generation_latency_ms: 450,
        }
    }

    /// Override simulated latencies (for testing).
    pub fn with_latencies(
        mut self,
        embedding_ms: u128,
        search_ms: u128,
        generation_ms: u128,
    ) -> Self {
        self.embedding_latency_ms = embedding_ms;
        self.search_latency_ms = search_ms;
        self.generation_latency_ms = generation_ms;
        self
    }

    /// Run the naive RAG pipeline for a query.
    ///
    /// Returns a [`PipelineProfile`] with per-stage timing.
    /// `top_k` controls how many chunks are retrieved.
    pub fn run(&self, query: &str, top_k: usize) -> PipelineProfile {
        let mut profile = PipelineProfile::new();

        // Stage 1: Query embedding
        let query_tokens = estimate_tokens(query);
        let timer = StageTimer::start("query_embedding");
        std::thread::sleep(std::time::Duration::from_millis(
            self.embedding_latency_ms as u64,
        ));
        let query_emb = mock_embed(query);
        profile.add_stage(
            timer
                .finish(query_tokens, 0.0001)
                .with_metadata("method", "mock_bpe"),
        );

        // Stage 2: Vector search
        let timer = StageTimer::start("vector_search");
        std::thread::sleep(std::time::Duration::from_millis(
            self.search_latency_ms as u64,
        ));
        let retrieved = self.search(&query_emb, top_k);
        let retrieved_tokens: usize = retrieved.iter().map(|c| estimate_tokens(&c.content)).sum();
        profile.add_stage(
            timer
                .finish(retrieved_tokens, 0.0)
                .with_metadata("top_k", top_k.to_string())
                .with_metadata("corpus_size", self.corpus.len().to_string()),
        );

        // Stage 3: Context assembly
        let timer = StageTimer::start("context_assembly");
        let context = self.assemble_context(&retrieved);
        let context_tokens = estimate_tokens(&context);
        profile.add_stage(
            timer
                .finish(context_tokens, 0.0)
                .with_metadata("chunks_used", retrieved.len().to_string()),
        );

        // Stage 4: Generation
        let timer = StageTimer::start("generation");
        std::thread::sleep(std::time::Duration::from_millis(
            self.generation_latency_ms as u64,
        ));
        let gen = self.mock_generate(query, &context);
        profile.add_stage(
            timer
                .finish(gen.completion_tokens, 0.0048)
                .with_metadata("model", "mock-llm")
                .with_metadata("prompt_tokens", gen.prompt_tokens.to_string()),
        );

        profile
    }

    /// Compute a mock relevance score for a query embedding.
    fn search(&self, query_emb: &f64, top_k: usize) -> Vec<&MockChunk> {
        let mut scored: Vec<(f64, &MockChunk)> = self
            .corpus
            .iter()
            .map(|chunk| {
                let similarity = 1.0 - (chunk.mock_embedding - query_emb).abs();
                (similarity, chunk)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored.into_iter().take(top_k).map(|(_, c)| c).collect()
    }

    /// Assemble context text from retrieved chunks.
    fn assemble_context(&self, chunks: &[&MockChunk]) -> String {
        chunks
            .iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n---\n\n")
    }

    /// Generate a mock LLM response.
    fn mock_generate(&self, query: &str, context: &str) -> MockGeneration {
        let answer = format!(
            "Based on the retrieved context, here is what I found about \"{}\":\n\n\
             [Mock LLM Response]\n\
             The context contains {} characters of supporting information. \
             This is a simulated generation for demonstration purposes.",
            query,
            context.len()
        );

        MockGeneration {
            prompt_tokens: estimate_tokens(context) + estimate_tokens(query),
            completion_tokens: estimate_tokens(&answer),
            answer,
        }
    }
}

impl Default for NaivePipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a deterministic mock embedding (0.0–1.0) from text.
fn mock_embed(text: &str) -> f64 {
    let hash: u64 = text
        .chars()
        .map(|c| c as u64)
        .fold(0u64, |acc, c| acc.wrapping_mul(31).wrapping_add(c));
    (hash % 1000) as f64 / 1000.0
}

/// Rough token estimate: ~4 chars per token.
fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

/// Default mock corpus covering common RAG topics.
fn default_corpus() -> Vec<MockChunk> {
    vec![
        MockChunk {
            id: "rust_async_01".to_string(),
            source: "rust-async-guide.md".to_string(),
            content: "Tokio is an asynchronous runtime for the Rust programming language. It provides an I/O driver, task scheduler, and timer. Tokio tasks are lightweight, requiring only a few kilobytes of memory each, making them much cheaper than OS threads.".to_string(),
            mock_embedding: 0.72,
        },
        MockChunk {
            id: "rust_async_02".to_string(),
            source: "rust-async-guide.md".to_string(),
            content: "The async/await syntax in Rust allows writing asynchronous code that looks like synchronous code. The .await keyword yields control back to the runtime, allowing other tasks to execute.".to_string(),
            mock_embedding: 0.68,
        },
        MockChunk {
            id: "rag_overview_01".to_string(),
            source: "rag-architecture.md".to_string(),
            content: "Retrieval Augmented Generation (RAG) combines a retrieval system with a generative LLM. The retrieval component finds relevant documents, which are then passed as context to the LLM for generating grounded responses.".to_string(),
            mock_embedding: 0.85,
        },
        MockChunk {
            id: "rag_overview_02".to_string(),
            source: "rag-architecture.md".to_string(),
            content: "A naive RAG pipeline follows a simple linear flow: embed the query, search a vector database, assemble context from top-K results, and generate an answer. This architecture is easy to implement but lacks advanced features like reranking or query expansion.".to_string(),
            mock_embedding: 0.90,
        },
        MockChunk {
            id: "vector_db_01".to_string(),
            source: "vector-databases.md".to_string(),
            content: "Vector databases store high-dimensional embeddings of text chunks. Popular options include Qdrant, Weaviate, Pinecone, and Milvus. They support approximate nearest neighbor (ANN) search for fast retrieval at scale.".to_string(),
            mock_embedding: 0.65,
        },
        MockChunk {
            id: "embedding_01".to_string(),
            source: "embedding-models.md".to_string(),
            content: "Embedding models like text-embedding-3-small (OpenAI) and BGE (BAAI) convert text into dense vector representations. These vectors capture semantic meaning, enabling similarity-based search.".to_string(),
            mock_embedding: 0.60,
        },
        MockChunk {
            id: "hallucination_01".to_string(),
            source: "rag-evaluation.md".to_string(),
            content: "Hallucination in RAG occurs when the LLM generates claims not supported by the retrieved context. Metrics like faithfulness and source attribution percentage help quantify grounding quality.".to_string(),
            mock_embedding: 0.55,
        },
        MockChunk {
            id: "reranking_01".to_string(),
            source: "advanced-rag.md".to_string(),
            content: "Re-ranking improves retrieval precision by applying a cross-encoder model to re-score initial retrieval results. This is common in Advanced RAG architectures but adds latency.".to_string(),
            mock_embedding: 0.45,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_pipeline_basic_run() {
        let pipeline = NaivePipeline::new();
        let profile = pipeline.run("What is RAG?", 3);

        assert_eq!(profile.stage_count(), 4);
        assert!(profile.total_duration_ms > 0);
        assert!(profile.total_tokens > 0);
    }

    #[test]
    fn test_naive_pipeline_stage_names() {
        let pipeline = NaivePipeline::new();
        let profile = pipeline.run("test query", 5);

        let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "query_embedding",
                "vector_search",
                "context_assembly",
                "generation"
            ]
        );
    }

    #[test]
    fn test_naive_pipeline_top_k() {
        let pipeline = NaivePipeline::new();
        let profile = pipeline.run("vector database", 2);

        // Context assembly should have metadata about chunks used
        let ctx_stage = profile
            .find("context_assembly")
            .expect("context stage exists");
        assert_eq!(
            ctx_stage.metadata.get("chunks_used"),
            Some(&"2".to_string())
        );
    }

    #[test]
    fn test_mock_embed_deterministic() {
        let emb1 = mock_embed("hello world");
        let emb2 = mock_embed("hello world");
        assert_eq!(emb1, emb2);

        let emb3 = mock_embed("different text");
        assert_ne!(emb1, emb3);
    }

    #[test]
    fn test_default_corpus_not_empty() {
        let corpus = default_corpus();
        assert!(!corpus.is_empty());
        for chunk in &corpus {
            assert!(!chunk.id.is_empty());
            assert!(!chunk.content.is_empty());
            assert!(chunk.mock_embedding >= 0.0 && chunk.mock_embedding <= 1.0);
        }
    }

    #[test]
    fn test_custom_corpus() {
        let custom = vec![MockChunk {
            id: "custom_01".to_string(),
            source: "custom.md".to_string(),
            content: "Custom content".to_string(),
            mock_embedding: 0.5,
        }];
        let pipeline = NaivePipeline::with_corpus(custom);
        let profile = pipeline.run("custom query", 1);

        let search_stage = profile.find("vector_search").expect("search stage exists");
        assert_eq!(
            search_stage.metadata.get("corpus_size"),
            Some(&"1".to_string())
        );
    }

    #[test]
    fn test_fast_latencies() {
        let pipeline = NaivePipeline::new().with_latencies(1, 1, 1);
        let profile = pipeline.run("fast query", 3);

        // Stages with sleep should be at least 1ms
        assert!(profile.find("query_embedding").unwrap().duration_ms >= 1);
        assert!(profile.find("vector_search").unwrap().duration_ms >= 1);
        assert!(profile.find("generation").unwrap().duration_ms >= 1);
        // context_assembly has no sleep, so just verify the stage exists
        assert!(profile.find("context_assembly").is_some());
    }

    #[test]
    fn test_token_estimation() {
        assert_eq!(estimate_tokens("hello"), 1);
        assert_eq!(estimate_tokens("hello world"), 2); // 11 chars / 4 = 2.75 → 2
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
    }
}
