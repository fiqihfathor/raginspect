use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model: String,
    pub dimension: usize,
    pub distance_metric: String,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "text-embedding-3-small".to_string(),
            dimension: 1536,
            distance_metric: "cosine".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    pub provider: String,
    pub collection: String,
    pub top_k: usize,
    pub similarity_threshold: f64,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            provider: "qdrant".to_string(),
            collection: "knowledge_base".to_string(),
            top_k: 5,
            similarity_threshold: 0.65,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String,
    pub model: String,
    pub endpoint: String,
    pub max_tokens: usize,
    pub temperature: f64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            model: "gpt-4o-mini".to_string(),
            endpoint: "https://api.openai.com/v1".to_string(),
            max_tokens: 1024,
            temperature: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    pub max_context_tokens: usize,
    pub deduplicate_threshold: f64,
    pub prune_irrelevant: bool,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 4096,
            deduplicate_threshold: 0.85,
            prune_irrelevant: true,
        }
    }
}

// ── Optional pipeline components (all default to disabled) ──

/// Re-ranking configuration (Advanced RAG).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RerankingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub model: String,
    #[serde(default = "default_top_n_rerank")]
    pub top_n: usize,
}

fn default_top_n_rerank() -> usize {
    3
}

/// Hybrid search fusion configuration (Advanced/Modular RAG).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FusionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub method: String,
}

/// LLM-driven routing configuration (Modular/Agentic RAG).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoutingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub routes: Vec<String>,
}

/// Tool-calling configuration (Agentic RAG).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub tools: Vec<String>,
}

/// Knowledge graph configuration (Graph RAG).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub entity_types: Vec<String>,
}

/// HyDE configuration (Hypothetical Document Embeddings).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HydeConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub generation_model: String,
}

/// Multimodal retrieval configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultimodalConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub modalities: Vec<String>,
}

/// Metric weight override configuration.
///
/// When `override_weights` is true, the provided weight values are used
/// instead of the architecture-specific defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    #[serde(default)]
    pub override_weights: bool,
    #[serde(default = "default_relevance_weight")]
    pub relevance_weight: f64,
    #[serde(default = "default_efficiency_weight")]
    pub efficiency_weight: f64,
    #[serde(default = "default_grounding_weight")]
    pub grounding_weight: f64,
}

fn default_relevance_weight() -> f64 {
    0.35
}
fn default_efficiency_weight() -> f64 {
    0.35
}
fn default_grounding_weight() -> f64 {
    0.30
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            override_weights: false,
            relevance_weight: 0.35,
            efficiency_weight: 0.35,
            grounding_weight: 0.30,
        }
    }
}

/// Master pipeline configuration structure parsed from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub vector_store: VectorStoreConfig,
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub context: ContextConfig,
    #[serde(default)]
    pub reranking: RerankingConfig,
    #[serde(default)]
    pub fusion: FusionConfig,
    #[serde(default)]
    pub routing: RoutingConfig,
    #[serde(default)]
    pub tools: ToolConfig,
    #[serde(default)]
    pub graph: GraphConfig,
    #[serde(default)]
    pub hyde: HydeConfig,
    #[serde(default)]
    pub multimodal: MultimodalConfig,
    #[serde(default)]
    pub metrics: MetricsConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            name: "Default-RAG-Pipeline".to_string(),
            description: "Standard RAG pipeline configuration".to_string(),
            embedding: EmbeddingConfig::default(),
            vector_store: VectorStoreConfig::default(),
            llm: LlmConfig::default(),
            context: ContextConfig::default(),
            reranking: RerankingConfig::default(),
            fusion: FusionConfig::default(),
            routing: RoutingConfig::default(),
            tools: ToolConfig::default(),
            graph: GraphConfig::default(),
            hyde: HydeConfig::default(),
            multimodal: MultimodalConfig::default(),
            metrics: MetricsConfig::default(),
        }
    }
}

impl PipelineConfig {
    /// Load pipeline configuration from a TOML file path.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        if !path_ref.exists() {
            anyhow::bail!("Pipeline config file not found at path: {:?}", path_ref);
        }

        let content = fs::read_to_string(path_ref)
            .with_context(|| format!("Failed to read config file {:?}", path_ref))?;

        let config: PipelineConfig = toml::from_str(&content)
            .with_context(|| format!("Failed to parse TOML configuration from {:?}", path_ref))?;

        Ok(config)
    }
}
