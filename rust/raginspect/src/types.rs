use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::fmt;

/// RAG architecture type under inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum RagArchitecture {
    /// Simple linear pipeline: embed → search → stuff → generate
    Naive,
    /// Adds re-ranking, query expansion, and hybrid search
    Advanced,
    /// Multi-stage pipeline with independent configurable modules
    Modular,
    /// LLM-driven routing, tool-use, and iterative retrieval
    Agentic,
    /// Graph-based retrieval with entity relationships (e.g. GraphRAG)
    Graph,
    /// Hypothetical Document Embeddings — generate hypothetical answers before retrieval
    Hyde,
    /// Retrieves and fuses text, images, and structured data
    Multimodal,
}

impl fmt::Display for RagArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RagArchitecture::Naive => write!(f, "Naive (Linear)"),
            RagArchitecture::Advanced => write!(f, "Advanced (Re-ranked + Hybrid)"),
            RagArchitecture::Modular => write!(f, "Modular (Multi-stage)"),
            RagArchitecture::Agentic => write!(f, "Agentic (LLM-driven Routing)"),
            RagArchitecture::Graph => write!(f, "Graph (Entity-relationship)"),
            RagArchitecture::Hyde => write!(f, "HyDE (Hypothetical Document)"),
            RagArchitecture::Multimodal => write!(f, "Multimodal (Cross-modal Fusion)"),
        }
    }
}

impl RagArchitecture {
    /// Return a short diagnostic description of what this architecture adds.
    pub fn diagnostic_focus(&self) -> &'static str {
        match self {
            RagArchitecture::Naive => "Single-pass dense retrieval with direct context stuffing",
            RagArchitecture::Advanced => {
                "Re-ranking quality, hybrid search fusion, and query expansion effectiveness"
            }
            RagArchitecture::Modular => {
                "Inter-module contract validation, stage isolation, and pipeline orchestration"
            }
            RagArchitecture::Agentic => {
                "Tool-use decision accuracy, retrieval iteration depth, and routing correctness"
            }
            RagArchitecture::Graph => {
                "Entity extraction coverage, relationship traversal depth, and community detection"
            }
            RagArchitecture::Hyde => {
                "Hypothetical document quality, embedding drift, and retrieval precision lift"
            }
            RagArchitecture::Multimodal => {
                "Cross-modal alignment, modality weighting, and fusion strategy effectiveness"
            }
        }
    }

    /// Architecture-specific diagnostic layers to inspect.
    pub fn diagnostic_layers(&self) -> Vec<&'static str> {
        match self {
            RagArchitecture::Naive => vec![
                "Embedding quality",
                "Top-K selection",
                "Context stuffing ratio",
            ],
            RagArchitecture::Advanced => vec![
                "Query expansion coverage",
                "Hybrid search fusion (BM25 + dense)",
                "Re-ranker score delta",
                "Cross-encoder precision",
            ],
            RagArchitecture::Modular => vec![
                "Module interface contracts",
                "Stage latency breakdown",
                "Pipeline DAG validation",
                "Fallback chain integrity",
            ],
            RagArchitecture::Agentic => vec![
                "LLM routing accuracy",
                "Tool selection relevance",
                "Retrieval iteration depth",
                "Self-correction loops",
                "Token budget per hop",
            ],
            RagArchitecture::Graph => vec![
                "Entity extraction coverage",
                "Relationship type distribution",
                "Community cluster quality",
                "Graph traversal depth",
                "Subgraph relevance",
            ],
            RagArchitecture::Hyde => vec![
                "Hypothetical document quality",
                "Embedding drift score",
                "Retrieval precision lift vs naive",
                "Generation-retrieval alignment",
            ],
            RagArchitecture::Multimodal => vec![
                "Cross-modal embedding alignment",
                "Modality weight distribution",
                "Fusion strategy (concat / attention)",
                "Modality gap score",
                "Multimodal context packing",
            ],
        }
    }
}

/// Inspection mode specifying the granularity of the RAG pipeline analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum InspectMode {
    /// Full inspection across retrieval, context construction, and LLM generation layers
    Full,
    /// Inspect only the retrieval and vector search layer
    Retrieval,
    /// Inspect retrieval and context assembly / deduplication efficiency
    Context,
    /// High-level summary inspection
    Quick,
}

impl fmt::Display for InspectMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InspectMode::Full => write!(f, "Full Pipeline"),
            InspectMode::Retrieval => write!(f, "Retrieval Layer Only"),
            InspectMode::Context => write!(f, "Context Construction Only"),
            InspectMode::Quick => write!(f, "Quick Diagnostic"),
        }
    }
}

/// Diagnostic verdict for an individual retrieved chunk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Verdict {
    /// High relevance score; chunk directly addresses the query
    Relevant,
    /// Moderate relevance; provides supporting context
    PartiallyRelevant,
    /// Low similarity score or off-topic context
    Irrelevant,
    /// Semantically redundant with a higher-ranked chunk
    Duplicate,
    /// Low confidence / ambiguous retrieval score
    LowConfidence,
}

impl Verdict {
    pub fn badge(&self) -> &'static str {
        match self {
            Verdict::Relevant => "✅ RELEVANT",
            Verdict::PartiallyRelevant => "⚠️ PARTIAL",
            Verdict::Irrelevant => "❌ IRRELEVANT",
            Verdict::Duplicate => "🔄 DUPLICATE",
            Verdict::LowConfidence => "❓ LOW CONF",
        }
    }

    #[allow(dead_code)]
    pub fn color_code(&self) -> &'static str {
        match self {
            Verdict::Relevant => "green",
            Verdict::PartiallyRelevant => "yellow",
            Verdict::Irrelevant => "red",
            Verdict::Duplicate => "magenta",
            Verdict::LowConfidence => "cyan",
        }
    }
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.badge())
    }
}

/// Detailed information about a single retrieved document chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkInfo {
    /// Identifier for the chunk (e.g. "doc_architecture#chunk_02")
    pub id: String,
    /// Source document file path or URI
    pub source_doc: String,
    /// Raw textual content of the chunk
    pub content: String,
    /// Vector similarity score (0.0 to 1.0)
    pub score: f64,
    /// Token count estimated or calculated by tiktoken
    pub token_count: usize,
    /// Diagnostic verdict
    pub verdict: Verdict,
    /// Whether key concepts from this chunk were detected in the generated answer
    pub is_used_in_gen: bool,
    /// Explanatory rationale for the verdict
    pub verdict_reason: String,
}

/// Detailed metrics for the Vector Retrieval layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// Configured Top-K parameter
    pub top_k: usize,
    /// Number of chunks retrieved from the vector store
    pub chunks_retrieved: usize,
    /// Total token count across all retrieved chunks
    pub total_retrieved_tokens: usize,
    /// Retrieval latency in milliseconds
    pub latency_ms: u64,
    /// Individual chunk details
    pub chunks: Vec<ChunkInfo>,
    /// Average vector similarity score
    pub avg_similarity: f64,
}

/// Detailed metrics for Context Construction and Token Efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextStage {
    /// Total prompt tokens fed into the LLM context
    pub total_tokens: usize,
    /// Tokens contributing to relevant context
    pub useful_tokens: usize,
    /// Tokens wasted on irrelevant or duplicate context
    pub wasted_tokens: usize,
    /// Token efficiency ratio (useful / total)
    pub efficiency_ratio: f64,
    /// Count of duplicate chunks pruned
    pub deduplicated_chunks: usize,
    /// Count of low-relevance chunks pruned
    pub irrelevant_chunks_pruned: usize,
    /// Maximum context window limit of the model
    pub context_window_limit: usize,
}

/// Detailed metrics for the Generation and Grounding analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// Response text generated by the LLM
    pub generated_text: String,
    /// Prompt tokens passed to LLM
    pub prompt_tokens: usize,
    /// Completion tokens produced by LLM
    pub completion_tokens: usize,
    /// LLM inference latency in milliseconds
    pub latency_ms: u64,
    /// Hallucination index (0.0 = fully grounded, 1.0 = completely hallucinated)
    pub hallucination_score: f64,
    /// Percentage of statements directly attributed to retrieved context
    pub source_attribution_pct: f64,
    /// IDs of chunks cited in the response
    pub cited_chunk_ids: Vec<String>,
    /// Statements or claims in the generation that lack source context support
    pub uncited_claims: Vec<String>,
}

/// Overall RAG Inspection Report containing findings across all pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InspectionReport {
    /// User query inspected
    pub query: String,
    /// Name of the pipeline configuration used
    pub config_name: String,
    /// Name of the LLM / embedding model evaluated
    pub model_name: String,
    /// RAG architecture type inspected
    pub architecture: RagArchitecture,
    /// Inspection mode executed
    pub inspect_mode: InspectMode,
    /// ISO 8601 timestamp of inspection execution
    pub timestamp: String,
    /// Vector retrieval findings
    pub retrieval: RetrievalResult,
    /// Context window & token efficiency findings
    pub context: ContextStage,
    /// LLM generation and grounding findings
    pub generation: GenerationResult,
    /// Overall RAG Health Score (0 to 100)
    pub overall_score: f64,
    /// Actionable diagnostic recommendations
    pub recommendations: Vec<String>,
}
