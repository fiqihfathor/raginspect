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
