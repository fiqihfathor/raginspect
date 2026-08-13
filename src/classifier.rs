//! Architecture classifier — pattern-match pipeline topology to a RAG architecture.
//!
//! Given a [`PipelineTopology`], applies heuristic rules to determine
//! which of the 7 supported RAG architectures best describes the pipeline,
//! along with a confidence score (0.0–1.0).
//!
//! ## Sprint 1 Scope
//!
//! Full detection for all 7 architectures:
//! - **Naive**: single retrieve → generate
//! - **Advanced**: re-ranking, hybrid search, query expansion
//! - **Modular**: routing, fusion, multi-stage
//! - **Agentic**: tool-calling, iterative retrieval
//! - **Graph**: knowledge graph traversal
//! - **HyDE**: hypothetical document generation
//! - **Multimodal**: multi-modal retrieval
//!
//! ## Example
//!
//! ```
//! use raginspect::PipelineConfig;
//! use raginspect::topology::TopologyAnalyzer;
//! use raginspect::classifier::ArchitectureClassifier;
//!
//! let config = PipelineConfig::default();
//! let topology = TopologyAnalyzer::new().analyze(&config);
//! let result = ArchitectureClassifier::new().classify(&topology);
//! println!("Detected: {} (confidence: {:.2})", result.architecture, result.confidence);
//! ```

use crate::topology::PipelineTopology;
use crate::types::RagArchitecture;
use serde::{Deserialize, Serialize};

/// Result of architecture classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// The best-matching architecture
    pub architecture: RagArchitecture,
    /// Confidence score (0.0–1.0)
    pub confidence: f64,
    /// Human-readable reason for the classification
    pub reason: String,
    /// All architectures considered, with their individual scores
    pub scores: Vec<(RagArchitecture, f64)>,
}

/// Pattern-matching classifier for RAG architectures.
pub struct ArchitectureClassifier;

impl ArchitectureClassifier {
    /// Create a new classifier.
    pub fn new() -> Self {
        Self
    }

    /// Classify a pipeline topology into a RAG architecture.
    ///
    /// Scores each architecture against the topology using heuristic rules,
    /// then returns the highest-scoring match with a confidence score.
    pub fn classify(&self, topology: &PipelineTopology) -> ClassificationResult {
        let scores = self.score_all(topology);

        // Find the best match
        let (best_arch, best_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or((RagArchitecture::Naive, 0.0));

        let reason = self.explain(&best_arch, topology);

        ClassificationResult {
            architecture: best_arch,
            confidence: best_score,
            reason,
            scores,
        }
    }

    /// Score all 7 architectures against the topology.
    fn score_all(&self, topo: &PipelineTopology) -> Vec<(RagArchitecture, f64)> {
        vec![
            (RagArchitecture::Multimodal, self.score_multimodal(topo)),
            (RagArchitecture::Hyde, self.score_hyde(topo)),
            (RagArchitecture::Graph, self.score_graph(topo)),
            (RagArchitecture::Agentic, self.score_agentic(topo)),
            (RagArchitecture::Modular, self.score_modular(topo)),
            (RagArchitecture::Advanced, self.score_advanced(topo)),
            (RagArchitecture::Naive, self.score_naive(topo)),
        ]
    }

    /// Naive: only dense retrieval, no enhancements.
    fn score_naive(&self, topo: &PipelineTopology) -> f64 {
        if topo.is_naive() {
            1.0
        } else if topo.component_count() <= 2 && !topo.has_fusion && !topo.has_routing {
            0.5 // might still be naive with just pruning
        } else {
            0.1
        }
    }

    /// Advanced: re-ranking and/or fusion, but no routing/tools/graph.
    fn score_advanced(&self, topo: &PipelineTopology) -> f64 {
        let has_enhancements = topo.has_reranking || topo.has_fusion;
        let has_complex = topo.has_routing
            || topo.has_tool_calling
            || topo.has_graph
            || topo.has_hyde
            || topo.has_multimodal;

        if has_enhancements && !has_complex {
            let mut score: f64 = 0.7;
            if topo.has_reranking {
                score += 0.15;
            }
            if topo.has_fusion {
                score += 0.15;
            }
            score.min(1.0_f64)
        } else if has_enhancements {
            0.3 // has enhancements but also complex components
        } else {
            0.0
        }
    }

    /// Modular: routing + multi-stage, but no tools/graph/hyde/multimodal.
    fn score_modular(&self, topo: &PipelineTopology) -> f64 {
        let has_modular_traits = topo.has_routing || topo.retrieval_stages >= 2;
        let has_complex =
            topo.has_tool_calling || topo.has_graph || topo.has_hyde || topo.has_multimodal;

        if has_modular_traits && !has_complex {
            let mut score: f64 = 0.6;
            if topo.has_routing {
                score += 0.2;
            }
            if topo.retrieval_stages >= 2 {
                score += 0.1;
            }
            if topo.has_fusion {
                score += 0.1;
            }
            score.min(1.0_f64)
        } else {
            0.1
        }
    }

    /// Agentic: tool-calling and/or iterative retrieval.
    fn score_agentic(&self, topo: &PipelineTopology) -> f64 {
        if topo.has_tool_calling {
            let mut score: f64 = 0.8;
            if topo.is_iterative {
                score += 0.1;
            }
            if topo.has_routing {
                score += 0.1;
            }
            score.min(1.0_f64)
        } else if topo.has_routing && topo.is_iterative && !topo.has_graph {
            0.5 // routing without tools is borderline agentic
        } else {
            0.0
        }
    }

    /// Graph: knowledge graph traversal enabled.
    fn score_graph(&self, topo: &PipelineTopology) -> f64 {
        if topo.has_graph {
            let mut score: f64 = 0.85;
            if topo.retrieval_stages > 1 {
                score += 0.1;
            }
            score.min(1.0_f64)
        } else {
            0.0
        }
    }

    /// HyDE: hypothetical document generation.
    fn score_hyde(&self, topo: &PipelineTopology) -> f64 {
        if topo.has_hyde {
            0.95 // HyDE is a very specific signal
        } else {
            0.0
        }
    }

    /// Multimodal: multiple modalities.
    fn score_multimodal(&self, topo: &PipelineTopology) -> f64 {
        if topo.has_multimodal {
            0.95 // Multimodal is a very specific signal
        } else {
            0.0
        }
    }

    /// Generate a human-readable explanation for a classification.
    fn explain(&self, arch: &RagArchitecture, topo: &PipelineTopology) -> String {
        match arch {
            RagArchitecture::Naive => {
                "Pipeline has only dense vector retrieval with no advanced components. \
                 Classified as Naive RAG."
                    .to_string()
            }
            RagArchitecture::Advanced => {
                let mut parts = vec![];
                if topo.has_reranking {
                    parts.push("re-ranking");
                }
                if topo.has_fusion {
                    parts.push("hybrid search fusion");
                }
                format!(
                    "Pipeline includes {} but no routing, tools, or graph traversal. \
                     Classified as Advanced RAG.",
                    parts.join(" + ")
                )
            }
            RagArchitecture::Modular => {
                let mut parts = vec!["routing".to_string()];
                if topo.retrieval_stages >= 2 {
                    parts.push(format!("{} retrieval stages", topo.retrieval_stages));
                }
                format!(
                    "Pipeline has {} indicating a modular multi-stage design. \
                     Classified as Modular RAG.",
                    parts.join(" + ")
                )
            }
            RagArchitecture::Agentic => {
                let mut parts = vec!["tool-calling".to_string()];
                if topo.is_iterative {
                    parts.push("iterative retrieval".to_string());
                }
                format!(
                    "Pipeline features {} indicating LLM-driven autonomous retrieval. \
                     Classified as Agentic RAG.",
                    parts.join(" + ")
                )
            }
            RagArchitecture::Graph => {
                "Pipeline includes knowledge graph traversal. Classified as Graph RAG.".to_string()
            }
            RagArchitecture::Hyde => "Pipeline generates hypothetical documents before retrieval. \
                 Classified as HyDE RAG."
                .to_string(),
            RagArchitecture::Multimodal => "Pipeline retrieves across multiple modalities. \
                 Classified as Multimodal RAG."
                .to_string(),
        }
    }
}

impl Default for ArchitectureClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PipelineConfig;
    use crate::topology::TopologyAnalyzer;

    fn classify(config: &PipelineConfig) -> ClassificationResult {
        let topo = TopologyAnalyzer::new().analyze(config);
        ArchitectureClassifier::new().classify(&topo)
    }

    fn default_config() -> PipelineConfig {
        PipelineConfig::default()
    }

    // ── Naive detection ──

    #[test]
    fn test_classify_naive() {
        let result = classify(&default_config());
        assert_eq!(result.architecture, RagArchitecture::Naive);
        assert!(result.confidence >= 0.9);
        assert!(!result.reason.is_empty());
    }

    // ── Advanced detection ──

    #[test]
    fn test_classify_advanced_with_reranking() {
        let mut config = default_config();
        config.reranking.enabled = true;
        config.reranking.model = "cross-encoder".to_string();

        let result = classify(&config);
        assert_eq!(result.architecture, RagArchitecture::Advanced);
        assert!(result.confidence >= 0.7);
    }

    #[test]
    fn test_classify_advanced_with_fusion() {
        let mut config = default_config();
        config.fusion.enabled = true;

        let result = classify(&config);
        assert_eq!(result.architecture, RagArchitecture::Advanced);
        assert!(result.confidence >= 0.7);
    }

    #[test]
    fn test_classify_advanced_with_reranking_and_fusion() {
        let mut config = default_config();
        config.reranking.enabled = true;
        config.reranking.model = "cross-encoder".to_string();
        config.fusion.enabled = true;

        let result = classify(&config);
        assert_eq!(result.architecture, RagArchitecture::Advanced);
        assert!(result.confidence >= 0.9);
    }

    // ── Modular detection ──

    #[test]
    fn test_classify_modular_with_routing() {
        let mut config = default_config();
        config.routing.enabled = true;
        config.routing.routes = vec!["semantic".to_string(), "keyword".to_string()];

        let result = classify(&config);
        assert_eq!(result.architecture, RagArchitecture::Modular);
        assert!(result.confidence >= 0.6);
    }

    // ── Agentic detection ──

    #[test]
    fn test_classify_agentic_with_tools() {
        let mut config = default_config();
        config.tools.enabled = true;
        config.tools.tools = vec!["search".to_string(), "calculator".to_string()];

        let result = classify(&config);
        assert_eq!(result.architecture, RagArchitecture::Agentic);
        assert!(result.confidence >= 0.8);
    }

    // ── Graph detection ──

    #[test]
    fn test_classify_graph() {
        let mut config = default_config();
        config.graph.enabled = true;

        let result = classify(&config);
        assert_eq!(result.architecture, RagArchitecture::Graph);
        assert!(result.confidence >= 0.85);
    }

    // ── HyDE detection ──

    #[test]
    fn test_classify_hyde() {
        let mut config = default_config();
        config.hyde.enabled = true;

        let result = classify(&config);
        assert_eq!(result.architecture, RagArchitecture::Hyde);
        assert!(result.confidence >= 0.9);
    }

    // ── Multimodal detection ──

    #[test]
    fn test_classify_multimodal() {
        let mut config = default_config();
        config.multimodal.enabled = true;
        config.multimodal.modalities = vec!["text".to_string(), "image".to_string()];

        let result = classify(&config);
        assert_eq!(result.architecture, RagArchitecture::Multimodal);
        assert!(result.confidence >= 0.9);
    }

    // ── Confidence score range ──

    #[test]
    fn test_confidence_is_in_valid_range() {
        let result = classify(&default_config());
        assert!(
            (0.0..=1.0).contains(&result.confidence),
            "confidence must be in [0.0, 1.0], got {}",
            result.confidence
        );

        // Check all scores are in range
        for (_, score) in &result.scores {
            assert!(
                (0.0..=1.0).contains(score),
                "individual score must be in [0.0, 1.0]"
            );
        }
    }

    // ── Reason is always non-empty ──

    #[test]
    fn test_reason_is_non_empty() {
        let result = classify(&default_config());
        assert!(!result.reason.is_empty());
        assert!(result.reason.len() > 10, "reason should be descriptive");
    }

    // ── All 7 architectures are scored ──

    #[test]
    fn test_all_architectures_scored() {
        let result = classify(&default_config());
        assert_eq!(result.scores.len(), 7, "must score all 7 architectures");
    }

    // ── Best score is the classified architecture ──

    #[test]
    fn test_best_score_matches_classification() {
        let mut config = default_config();
        config.graph.enabled = true;

        let result = classify(&config);
        let (_, best_score) = result
            .scores
            .iter()
            .find(|(a, _)| *a == result.architecture)
            .unwrap();

        // The classified architecture should have the highest or tied-highest score
        let max_score = result.scores.iter().map(|(_, s)| *s).fold(0.0f64, f64::max);
        assert!(
            (best_score - max_score).abs() < 1e-9,
            "classified arch score {} should equal max score {}",
            best_score,
            max_score
        );
    }
}
