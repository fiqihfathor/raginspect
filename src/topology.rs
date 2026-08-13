//! Pipeline topology analyzer — extract component structure from a pipeline config.
//!
//! Given a [`PipelineConfig`], identifies which RAG components are present
//! (retrieval method, reranking, fusion, routing, tool-calling, etc.)
//! and produces a [`PipelineTopology`] snapshot that the architecture
//! classifier can pattern-match on.
//!
//! ## Example
//!
//! ```
//! use raginspect::PipelineConfig;
//! use raginspect::topology::TopologyAnalyzer;
//!
//! let config = PipelineConfig::default();
//! let analyzer = TopologyAnalyzer::new();
//! let topology = analyzer.analyze(&config);
//! println!("Components: {:?}", topology.detected_components());
//! ```

use crate::config::{
    FusionConfig, GraphConfig, HydeConfig, MultimodalConfig, PipelineConfig, RerankingConfig,
    RoutingConfig, ToolConfig,
};

/// A single RAG pipeline component detected during topology analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineComponent {
    /// Dense vector embedding + similarity search (always present)
    DenseRetrieval,
    /// Sparse / lexical retrieval (BM25, TF-IDF)
    SparseRetrieval,
    /// Cross-encoder or LLM-based reranking of initial results
    Reranking,
    /// Query expansion, rewriting, or multi-query generation
    QueryExpansion,
    /// Hybrid search fusion (combining sparse + dense results)
    Fusion,
    /// LLM-driven routing to different retrieval strategies
    Routing,
    /// Tool-calling / function-calling for external data
    ToolCalling,
    /// Knowledge graph traversal
    GraphTraversal,
    /// Hypothetical document generation (HyDE)
    HyDE,
    /// Multi-modal retrieval (text + image + audio)
    Multimodal,
    /// Context compression or summarization
    ContextCompression,
}

impl PipelineComponent {
    /// Human-readable label for display.
    pub fn label(&self) -> &'static str {
        match self {
            Self::DenseRetrieval => "Dense Vector Retrieval",
            Self::SparseRetrieval => "Sparse/Lexical Retrieval",
            Self::Reranking => "Re-ranking",
            Self::QueryExpansion => "Query Expansion",
            Self::Fusion => "Hybrid Search Fusion",
            Self::Routing => "LLM Routing",
            Self::ToolCalling => "Tool Calling",
            Self::GraphTraversal => "Graph Traversal",
            Self::HyDE => "HyDE (Hypothetical Document)",
            Self::Multimodal => "Multimodal Retrieval",
            Self::ContextCompression => "Context Compression",
        }
    }
}

/// Snapshot of a pipeline's component structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineTopology {
    /// All components detected in the pipeline
    pub components: Vec<PipelineComponent>,
    /// Number of retrieval stages (1 = single-pass, 2+ = multi-stage)
    pub retrieval_stages: usize,
    /// Whether the pipeline has iterative/looped retrieval
    pub is_iterative: bool,
    /// Configured top_k
    pub top_k: usize,
    /// Embedding dimension
    pub embedding_dimension: usize,
    /// Distance metric used
    pub distance_metric: String,
    /// Whether reranking is enabled
    pub has_reranking: bool,
    /// Whether fusion is enabled
    pub has_fusion: bool,
    /// Whether routing is enabled
    pub has_routing: bool,
    /// Whether tool-calling is enabled
    pub has_tool_calling: bool,
    /// Whether graph traversal is enabled
    pub has_graph: bool,
    /// Whether HyDE is enabled
    pub has_hyde: bool,
    /// Whether multimodal retrieval is enabled
    pub has_multimodal: bool,
    /// Whether context compression is enabled
    pub has_context_compression: bool,
}

impl PipelineTopology {
    /// Return the list of detected component labels for display.
    pub fn detected_components(&self) -> Vec<&str> {
        self.components.iter().map(|c| c.label()).collect()
    }

    /// Number of distinct components detected.
    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// True if this is a simple single-pass pipeline (only dense retrieval).
    /// Context pruning/compression is not counted as an advanced component.
    pub fn is_naive(&self) -> bool {
        let advanced_components: Vec<&PipelineComponent> = self
            .components
            .iter()
            .filter(|c| {
                !matches!(
                    c,
                    PipelineComponent::DenseRetrieval | PipelineComponent::ContextCompression
                )
            })
            .collect();
        advanced_components.is_empty()
    }
}

/// Analyzer that inspects a [`PipelineConfig`] and produces a [`PipelineTopology`].
pub struct TopologyAnalyzer;

impl TopologyAnalyzer {
    /// Create a new topology analyzer.
    pub fn new() -> Self {
        Self
    }

    /// Analyze a pipeline config and extract its topology.
    pub fn analyze(&self, config: &PipelineConfig) -> PipelineTopology {
        let mut components = vec![PipelineComponent::DenseRetrieval];

        let has_reranking = Self::is_reranking_enabled(&config.reranking);
        let has_fusion = Self::is_fusion_enabled(&config.fusion);
        let has_routing = Self::is_routing_enabled(&config.routing);
        let has_tool_calling = Self::is_tool_calling_enabled(&config.tools);
        let has_graph = Self::is_graph_enabled(&config.graph);
        let has_hyde = Self::is_hyde_enabled(&config.hyde);
        let has_multimodal = Self::is_multimodal_enabled(&config.multimodal);
        let has_context_compression = config.context.prune_irrelevant;

        // Sparse retrieval is implied by fusion (hybrid = sparse + dense)
        let has_sparse = has_fusion;

        if has_sparse {
            components.push(PipelineComponent::SparseRetrieval);
        }
        if has_reranking {
            components.push(PipelineComponent::Reranking);
        }
        if has_fusion {
            components.push(PipelineComponent::Fusion);
        }
        if has_routing {
            components.push(PipelineComponent::Routing);
        }
        if has_tool_calling {
            components.push(PipelineComponent::ToolCalling);
        }
        if has_graph {
            components.push(PipelineComponent::GraphTraversal);
        }
        if has_hyde {
            components.push(PipelineComponent::HyDE);
        }
        if has_multimodal {
            components.push(PipelineComponent::Multimodal);
        }
        if has_context_compression {
            components.push(PipelineComponent::ContextCompression);
        }

        // Count retrieval stages
        let retrieval_stages = 1 + has_fusion as usize + has_graph as usize;

        // Iterative if routing or tool-calling (enables multi-hop retrieval)
        let is_iterative = has_routing || has_tool_calling;

        PipelineTopology {
            components,
            retrieval_stages,
            is_iterative,
            top_k: config.vector_store.top_k,
            embedding_dimension: config.embedding.dimension,
            distance_metric: config.embedding.distance_metric.clone(),
            has_reranking,
            has_fusion,
            has_routing,
            has_tool_calling,
            has_graph,
            has_hyde,
            has_multimodal,
            has_context_compression,
        }
    }

    fn is_reranking_enabled(cfg: &RerankingConfig) -> bool {
        cfg.enabled && !cfg.model.is_empty()
    }

    fn is_fusion_enabled(cfg: &FusionConfig) -> bool {
        cfg.enabled
    }

    fn is_routing_enabled(cfg: &RoutingConfig) -> bool {
        cfg.enabled
    }

    fn is_tool_calling_enabled(cfg: &ToolConfig) -> bool {
        cfg.enabled && !cfg.tools.is_empty()
    }

    fn is_graph_enabled(cfg: &GraphConfig) -> bool {
        cfg.enabled
    }

    fn is_hyde_enabled(cfg: &HydeConfig) -> bool {
        cfg.enabled
    }

    fn is_multimodal_enabled(cfg: &MultimodalConfig) -> bool {
        cfg.enabled && cfg.modalities.len() > 1
    }
}

impl Default for TopologyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PipelineConfig {
        PipelineConfig::default()
    }

    #[test]
    fn test_naive_topology() {
        let analyzer = TopologyAnalyzer::new();
        let topo = analyzer.analyze(&default_config());

        assert!(topo.is_naive());
        // Default config has prune_irrelevant=true, which adds ContextCompression
        assert!(topo.components.contains(&PipelineComponent::DenseRetrieval));
        assert!(!topo.has_reranking);
        assert!(!topo.has_fusion);
    }

    #[test]
    fn test_reranking_detected() {
        let mut config = default_config();
        config.reranking.enabled = true;
        config.reranking.model = "cross-encoder/ms-marco-minilm-v6".to_string();

        let topo = TopologyAnalyzer::new().analyze(&config);
        assert!(topo.has_reranking);
        assert!(topo.components.contains(&PipelineComponent::Reranking));
    }

    #[test]
    fn test_fusion_adds_sparse_retrieval() {
        let mut config = default_config();
        config.fusion.enabled = true;

        let topo = TopologyAnalyzer::new().analyze(&config);
        assert!(topo.has_fusion);
        assert!(topo.components.contains(&PipelineComponent::Fusion));
        assert!(topo
            .components
            .contains(&PipelineComponent::SparseRetrieval));
    }

    #[test]
    fn test_graph_detected() {
        let mut config = default_config();
        config.graph.enabled = true;

        let topo = TopologyAnalyzer::new().analyze(&config);
        assert!(topo.has_graph);
        assert!(topo.components.contains(&PipelineComponent::GraphTraversal));
    }

    #[test]
    fn test_hyde_detected() {
        let mut config = default_config();
        config.hyde.enabled = true;

        let topo = TopologyAnalyzer::new().analyze(&config);
        assert!(topo.has_hyde);
        assert!(topo.components.contains(&PipelineComponent::HyDE));
    }

    #[test]
    fn test_multimodal_requires_multiple_modalities() {
        let mut config = default_config();
        config.multimodal.enabled = true;
        config.multimodal.modalities = vec!["text".to_string()]; // only 1

        let topo = TopologyAnalyzer::new().analyze(&config);
        assert!(
            !topo.has_multimodal,
            "single modality should not count as multimodal"
        );

        config.multimodal.modalities = vec!["text".to_string(), "image".to_string()];
        let topo = TopologyAnalyzer::new().analyze(&config);
        assert!(topo.has_multimodal);
    }

    #[test]
    fn test_tool_calling_makes_iterative() {
        let mut config = default_config();
        config.tools.enabled = true;
        config.tools.tools = vec!["search".to_string(), "calculator".to_string()];

        let topo = TopologyAnalyzer::new().analyze(&config);
        assert!(topo.has_tool_calling);
        assert!(
            topo.is_iterative,
            "tool-calling should make pipeline iterative"
        );
    }

    #[test]
    fn test_routing_makes_iterative() {
        let mut config = default_config();
        config.routing.enabled = true;

        let topo = TopologyAnalyzer::new().analyze(&config);
        assert!(topo.has_routing);
        assert!(topo.is_iterative);
    }

    #[test]
    fn test_component_labels() {
        assert_eq!(
            PipelineComponent::DenseRetrieval.label(),
            "Dense Vector Retrieval"
        );
        assert_eq!(PipelineComponent::Reranking.label(), "Re-ranking");
        assert!(!PipelineComponent::Fusion.label().is_empty());
    }

    #[test]
    fn test_detected_components_returns_labels() {
        let mut config = default_config();
        config.reranking.enabled = true;
        config.reranking.model = "cross-encoder".to_string();

        let topo = TopologyAnalyzer::new().analyze(&config);
        let labels = topo.detected_components();
        assert!(labels.contains(&"Dense Vector Retrieval"));
        assert!(labels.contains(&"Re-ranking"));
    }

    #[test]
    fn test_retrieval_stages_count() {
        let analyzer = TopologyAnalyzer::new();

        // Naive: 1 stage
        assert_eq!(analyzer.analyze(&default_config()).retrieval_stages, 1);

        // With fusion: 2 stages
        let mut config = default_config();
        config.fusion.enabled = true;
        assert_eq!(analyzer.analyze(&config).retrieval_stages, 2);

        // With fusion + graph: 3 stages
        config.graph.enabled = true;
        assert_eq!(analyzer.analyze(&config).retrieval_stages, 3);
    }

    #[test]
    fn test_reranking_without_model_not_detected() {
        let mut config = default_config();
        config.reranking.enabled = true;
        // model is empty (default)
        let topo = TopologyAnalyzer::new().analyze(&config);
        assert!(
            !topo.has_reranking,
            "reranking without model should not be detected"
        );
    }
}
