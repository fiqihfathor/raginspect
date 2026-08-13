//! Architecture-specific metric defaults.
//!
//! Each RAG architecture emphasizes different quality dimensions. This module
//! defines the default metric weight sets per architecture and provides a
//! configuration override mechanism.
//!
//! ## Example
//!
//! ```
//! use raginspect::metrics::MetricWeights;
//! use raginspect::RagArchitecture;
//!
//! let weights = MetricWeights::for_architecture(&RagArchitecture::Advanced);
//! println!("Relevance weight: {:.2}", weights.relevance);
//!
//! // Override via config
//! let custom = MetricWeights {
//!     relevance: 0.50,
//!     efficiency: 0.30,
//!     grounding: 0.20,
//! };
//! ```

use crate::config::MetricsConfig;
use crate::types::RagArchitecture;
use serde::{Deserialize, Serialize};

/// Metric weights for health score calculation.
///
/// Each weight should be in [0.0, 1.0] and the three weights should sum to 1.0.
/// If they don't, they will be normalized automatically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricWeights {
    /// Weight for retrieval relevance (similarity scores, top-k precision)
    pub relevance: f64,
    /// Weight for context efficiency (token waste, packing ratio)
    pub efficiency: f64,
    /// Weight for generation grounding (hallucination index, source attribution)
    pub grounding: f64,
}

impl MetricWeights {
    /// Default weights for a given RAG architecture.
    ///
    /// Each architecture emphasizes different quality dimensions:
    /// - **Naive**: balanced, slight emphasis on efficiency (context recall + faithfulness)
    /// - **Advanced**: relevance-heavy (re-ranking precision, hybrid search quality)
    /// - **Modular**: balanced (inter-module contract validation)
    /// - **Agentic**: grounding-heavy (tool-use correctness, routing accuracy)
    /// - **Graph**: relevance-heavy (entity coverage, traversal depth)
    /// - **HyDE**: relevance + efficiency (embedding drift, retrieval precision lift)
    /// - **Multimodal**: grounding-heavy (cross-modal alignment quality)
    pub fn for_architecture(arch: &RagArchitecture) -> Self {
        match arch {
            RagArchitecture::Naive => Self {
                // Context recall + faithfulness are the core metrics
                relevance: 0.30,
                efficiency: 0.40,
                grounding: 0.30,
            },
            RagArchitecture::Advanced => Self {
                // Re-ranking precision and hybrid search quality dominate
                relevance: 0.45,
                efficiency: 0.30,
                grounding: 0.25,
            },
            RagArchitecture::Modular => Self {
                // Balanced — all stages are independently validated
                relevance: 0.35,
                efficiency: 0.35,
                grounding: 0.30,
            },
            RagArchitecture::Agentic => Self {
                // Tool-use accuracy and routing correctness (grounding)
                relevance: 0.25,
                efficiency: 0.30,
                grounding: 0.45,
            },
            RagArchitecture::Graph => Self {
                // Entity coverage and relationship traversal (relevance)
                relevance: 0.45,
                efficiency: 0.30,
                grounding: 0.25,
            },
            RagArchitecture::Hyde => Self {
                // Embedding drift + retrieval precision lift
                relevance: 0.40,
                efficiency: 0.35,
                grounding: 0.25,
            },
            RagArchitecture::Multimodal => Self {
                // Cross-modal alignment quality (grounding)
                relevance: 0.30,
                efficiency: 0.30,
                grounding: 0.40,
            },
        }
    }

    /// Resolve weights: use config override if provided, otherwise architecture defaults.
    pub fn resolve(arch: &RagArchitecture, config: &MetricsConfig) -> Self {
        let defaults = Self::for_architecture(arch);
        if config.override_weights {
            Self {
                relevance: config.relevance_weight,
                efficiency: config.efficiency_weight,
                grounding: config.grounding_weight,
            }
        } else {
            defaults
        }
    }

    /// Normalize weights so they sum to 1.0.
    pub fn normalized(self) -> Self {
        let sum = self.relevance + self.efficiency + self.grounding;
        if sum <= 0.0 {
            return Self {
                relevance: 0.33,
                efficiency: 0.34,
                grounding: 0.33,
            };
        }
        Self {
            relevance: self.relevance / sum,
            efficiency: self.efficiency / sum,
            grounding: self.grounding / sum,
        }
    }

    /// Calculate the weighted health score from three component scores (each 0-100).
    pub fn calculate_score(&self, relevance: f64, efficiency: f64, grounding: f64) -> f64 {
        let n = self.clone().normalized();
        (relevance * n.relevance + efficiency * n.efficiency + grounding * n.grounding).round()
    }

    /// Return the list of metric names that are emphasized (weight > 0.33) for this architecture.
    pub fn emphasized_metrics(&self) -> Vec<&'static str> {
        let n = self.clone().normalized();
        let mut emphasized = vec![];
        if n.relevance > 0.33 {
            emphasized.push("relevance");
        }
        if n.efficiency > 0.33 {
            emphasized.push("efficiency");
        }
        if n.grounding > 0.33 {
            emphasized.push("grounding");
        }
        emphasized
    }
}

impl Default for MetricWeights {
    fn default() -> Self {
        Self::for_architecture(&RagArchitecture::Naive)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Architecture-specific defaults ──

    #[test]
    fn test_naive_weights() {
        let w = MetricWeights::for_architecture(&RagArchitecture::Naive);
        assert_eq!(w.relevance, 0.30);
        assert_eq!(w.efficiency, 0.40);
        assert_eq!(w.grounding, 0.30);
        // Emphasizes efficiency (context recall)
        assert!(w.emphasized_metrics().contains(&"efficiency"));
    }

    #[test]
    fn test_advanced_weights() {
        let w = MetricWeights::for_architecture(&RagArchitecture::Advanced);
        assert!(w.relevance > w.efficiency);
        assert!(w.relevance > w.grounding);
        assert!(w.emphasized_metrics().contains(&"relevance"));
    }

    #[test]
    fn test_agentic_weights() {
        let w = MetricWeights::for_architecture(&RagArchitecture::Agentic);
        assert!(w.grounding > w.relevance);
        assert!(w.grounding > w.efficiency);
        assert!(w.emphasized_metrics().contains(&"grounding"));
    }

    #[test]
    fn test_graph_weights() {
        let w = MetricWeights::for_architecture(&RagArchitecture::Graph);
        assert!(w.relevance >= 0.40);
    }

    #[test]
    fn test_hyde_weights() {
        let w = MetricWeights::for_architecture(&RagArchitecture::Hyde);
        assert!(w.relevance > 0.33);
    }

    #[test]
    fn test_multimodal_weights() {
        let w = MetricWeights::for_architecture(&RagArchitecture::Multimodal);
        assert!(w.grounding > 0.33);
    }

    #[test]
    fn test_modular_weights_balanced() {
        let w = MetricWeights::for_architecture(&RagArchitecture::Modular);
        // Modular should be relatively balanced
        let max = w.relevance.max(w.efficiency).max(w.grounding);
        let min = w.relevance.min(w.efficiency).min(w.grounding);
        assert!(max - min < 0.15, "modular weights should be balanced");
    }

    // ── Normalization ──

    #[test]
    fn test_weights_sum_to_one() {
        for arch in [
            RagArchitecture::Naive,
            RagArchitecture::Advanced,
            RagArchitecture::Modular,
            RagArchitecture::Agentic,
            RagArchitecture::Graph,
            RagArchitecture::Hyde,
            RagArchitecture::Multimodal,
        ] {
            let w = MetricWeights::for_architecture(&arch);
            let sum = w.relevance + w.efficiency + w.grounding;
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "Weights for {:?} should sum to 1.0, got {}",
                arch,
                sum
            );
        }
    }

    #[test]
    fn test_normalization() {
        let w = MetricWeights {
            relevance: 2.0,
            efficiency: 3.0,
            grounding: 5.0,
        };
        let n = w.normalized();
        assert!((n.relevance - 0.2).abs() < 1e-9);
        assert!((n.efficiency - 0.3).abs() < 1e-9);
        assert!((n.grounding - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_normalization_zero_sum_fallback() {
        let w = MetricWeights {
            relevance: 0.0,
            efficiency: 0.0,
            grounding: 0.0,
        };
        let n = w.normalized();
        let sum = n.relevance + n.efficiency + n.grounding;
        assert!((sum - 1.0).abs() < 1e-9);
    }

    // ── Score calculation ──

    #[test]
    fn test_calculate_score() {
        let w = MetricWeights {
            relevance: 0.4,
            efficiency: 0.3,
            grounding: 0.3,
        };
        // 80*0.4 + 90*0.3 + 70*0.3 = 32 + 27 + 21 = 80
        let score = w.calculate_score(80.0, 90.0, 70.0);
        assert_eq!(score, 80.0);
    }

    #[test]
    fn test_calculate_score_normalizes() {
        let w = MetricWeights {
            relevance: 4.0,
            efficiency: 3.0,
            grounding: 3.0,
        };
        // Normalized to 0.4, 0.3, 0.3 → same as above
        let score = w.calculate_score(80.0, 90.0, 70.0);
        assert_eq!(score, 80.0);
    }

    // ── Config override ──

    #[test]
    fn test_resolve_uses_defaults_when_no_override() {
        let config = MetricsConfig::default();
        let w = MetricWeights::resolve(&RagArchitecture::Advanced, &config);
        let defaults = MetricWeights::for_architecture(&RagArchitecture::Advanced);
        assert_eq!(w.relevance, defaults.relevance);
    }

    #[test]
    fn test_resolve_uses_override_when_set() {
        let config = MetricsConfig {
            override_weights: true,
            relevance_weight: 0.50,
            efficiency_weight: 0.30,
            grounding_weight: 0.20,
        };

        let w = MetricWeights::resolve(&RagArchitecture::Naive, &config);
        assert_eq!(w.relevance, 0.50);
        assert_eq!(w.efficiency, 0.30);
        assert_eq!(w.grounding, 0.20);
    }

    // ── Emphasized metrics ──

    #[test]
    fn test_emphasized_metrics_naive() {
        let w = MetricWeights::for_architecture(&RagArchitecture::Naive);
        let em = w.emphasized_metrics();
        assert!(em.contains(&"efficiency"));
        assert!(!em.contains(&"relevance"));
    }

    #[test]
    fn test_emphasized_metrics_balanced_returns_empty() {
        let w = MetricWeights {
            relevance: 0.33,
            efficiency: 0.34,
            grounding: 0.33,
        };
        let em = w.emphasized_metrics();
        // Only efficiency is > 0.33
        assert!(em.contains(&"efficiency"));
    }
}
