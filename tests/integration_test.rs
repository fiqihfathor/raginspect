//! Comprehensive E2E integration test suite for `raginspect`.
//!
//! Test groups:
//!   - `config_loading`   — PipelineConfig loading from TOML and defaults
//!   - `inspector_e2e`    — Full Inspector runs across modes and architectures
//!   - `naive_pipeline`   — NaivePipeline stages, profiles, and serde round-trips
//!   - `profiler_e2e`     — MultiRunProfiler statistical invariants and edge cases
//!   - `arch_detection`   — RagArchitecture metadata, uniqueness, and coverage

mod common;

// ════════════════════════════════════════════════════════════════════════════════
// 1. Config Loading Tests
// ════════════════════════════════════════════════════════════════════════════════

mod config_loading {
    use crate::common;
    use raginspect::PipelineConfig;

    #[test]
    fn load_valid_config_from_sample_toml() {
        let cfg = common::load_sample_config();
        assert_eq!(cfg.name, "Production-Search-Pipeline");
        assert_eq!(cfg.embedding.model, "text-embedding-3-small");
        assert_eq!(cfg.embedding.dimension, 1536);
        assert_eq!(cfg.embedding.distance_metric, "cosine");
        assert_eq!(cfg.vector_store.provider, "qdrant");
        assert_eq!(cfg.vector_store.top_k, 5);
        assert!((cfg.vector_store.similarity_threshold - 0.65).abs() < 1e-9);
        assert_eq!(cfg.llm.model, "gpt-4o-mini");
        assert_eq!(cfg.llm.max_tokens, 1024);
        assert_eq!(cfg.context.max_context_tokens, 4096);
        assert!((cfg.context.deduplicate_threshold - 0.85).abs() < 1e-9);
        assert!(cfg.context.prune_irrelevant);
    }

    #[test]
    fn load_valid_config_from_fixture_toml() {
        let cfg = common::load_fixture_config();
        assert_eq!(cfg.name, "Test-Pipeline");
        assert_eq!(cfg.vector_store.top_k, 3);
        assert_eq!(cfg.embedding.dimension, 768);
        assert_eq!(cfg.llm.max_tokens, 512);
        assert_eq!(cfg.context.max_context_tokens, 2048);
    }

    #[test]
    fn load_config_from_nonexistent_path_returns_error() {
        let result = PipelineConfig::load_from_file("/tmp/raginspect_nonexistent_xzy_123.toml");
        assert!(
            result.is_err(),
            "Loading from a non-existent path must return Err"
        );
    }

    #[test]
    fn default_config_has_expected_defaults() {
        let cfg = PipelineConfig::default();
        assert!(
            !cfg.name.is_empty(),
            "default config name must not be empty"
        );
        assert_eq!(cfg.vector_store.top_k, 5);
        assert_eq!(cfg.llm.model, "gpt-4o-mini");
        assert_eq!(cfg.context.max_context_tokens, 4096);
    }

    #[test]
    fn config_round_trip_json_serialization() {
        let original = common::load_sample_config();
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: PipelineConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.name, original.name);
        assert_eq!(restored.embedding.model, original.embedding.model);
        assert_eq!(restored.vector_store.top_k, original.vector_store.top_k);
        assert_eq!(restored.llm.model, original.llm.model);
        assert_eq!(
            restored.context.max_context_tokens,
            original.context.max_context_tokens
        );
    }

    #[test]
    fn config_round_trip_toml_serialization() {
        let original = PipelineConfig::default();
        let toml_str = toml::to_string(&original).expect("serialize to TOML");
        let restored: PipelineConfig = toml::from_str(&toml_str).expect("deserialize from TOML");

        assert_eq!(restored.name, original.name);
        assert_eq!(restored.llm.endpoint, original.llm.endpoint);
        assert_eq!(
            restored.embedding.distance_metric,
            original.embedding.distance_metric
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// 2. Inspector E2E Tests
// ════════════════════════════════════════════════════════════════════════════════

mod inspector_e2e {
    use crate::common;
    use raginspect::{
        InspectMode, InspectionReport, Inspector, PipelineConfig, RagArchitecture, Verdict,
    };

    const RUST_QUERY: &str = "What is the memory overhead of Tokio async tasks?";
    const GENERIC_QUERY: &str = "How does RAG retrieval grounding work?";

    #[test]
    fn full_inspection_produces_valid_report() {
        let report = common::inspect_default(RUST_QUERY);
        assert_eq!(report.query, RUST_QUERY);
        assert!(!report.config_name.is_empty());
        assert!(!report.model_name.is_empty());
        common::assert_valid_score(report.overall_score, "overall_score");
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn all_inspect_modes_produce_reports_with_correct_mode() {
        let modes = [
            InspectMode::Full,
            InspectMode::Retrieval,
            InspectMode::Context,
            InspectMode::Quick,
        ];
        for mode in modes {
            let report = common::inspect_query(PipelineConfig::default(), GENERIC_QUERY, mode);
            assert_eq!(report.inspect_mode, mode);
        }
    }

    #[test]
    fn inspection_score_is_within_valid_range() {
        for query in [RUST_QUERY, GENERIC_QUERY] {
            let report = common::inspect_default(query);
            assert!(
                report.overall_score >= 0.0 && report.overall_score <= 100.0,
                "score {} out of range for '{}'",
                report.overall_score,
                query
            );
        }
    }

    #[test]
    fn all_architectures_produce_recommendations() {
        let architectures = [
            RagArchitecture::Naive,
            RagArchitecture::Advanced,
            RagArchitecture::Modular,
            RagArchitecture::Agentic,
            RagArchitecture::Graph,
            RagArchitecture::Hyde,
            RagArchitecture::Multimodal,
        ];
        for arch in architectures {
            let mut inspector = Inspector::new(PipelineConfig::default(), None);
            inspector.set_architecture(arch);
            let report = inspector
                .inspect("test query about RAG", InspectMode::Quick)
                .expect("inspection should succeed");
            assert!(
                !report.recommendations.is_empty(),
                "Architecture {:?} should produce recommendations",
                arch
            );
        }
    }

    #[test]
    fn inspection_report_query_is_preserved() {
        let query = "What are the latency characteristics of HNSW vector index?";
        let report = common::inspect_default(query);
        assert_eq!(report.query, query);
    }

    #[test]
    fn verdict_badges_are_non_empty_strings() {
        let verdicts = [
            Verdict::Relevant,
            Verdict::PartiallyRelevant,
            Verdict::Irrelevant,
            Verdict::Duplicate,
            Verdict::LowConfidence,
        ];
        for verdict in &verdicts {
            let badge = verdict.badge();
            assert!(
                !badge.is_empty(),
                "badge must not be empty for {:?}",
                verdict
            );
        }
    }

    #[test]
    fn verdict_color_codes_are_valid() {
        let verdicts = [
            Verdict::Relevant,
            Verdict::PartiallyRelevant,
            Verdict::Irrelevant,
            Verdict::Duplicate,
            Verdict::LowConfidence,
        ];
        for verdict in &verdicts {
            let color = verdict.color_code();
            assert!(
                !color.is_empty(),
                "color_code must not be empty for {:?}",
                verdict
            );
        }
    }

    #[test]
    fn retrieval_respects_top_k_config() {
        let mut cfg = PipelineConfig::default();
        cfg.vector_store.top_k = 3;
        let report = common::inspect_query(cfg, RUST_QUERY, InspectMode::Full);
        assert_eq!(report.retrieval.top_k, 3);
        assert!(
            report.retrieval.chunks_retrieved <= 3,
            "chunks_retrieved ({}) must not exceed top_k (3)",
            report.retrieval.chunks_retrieved
        );
    }

    #[test]
    fn context_efficiency_ratio_is_valid() {
        let report = common::inspect_default(RUST_QUERY);
        common::assert_valid_ratio(report.context.efficiency_ratio, "efficiency_ratio");
    }

    #[test]
    fn hallucination_score_is_valid() {
        let report = common::inspect_default(RUST_QUERY);
        common::assert_valid_ratio(report.generation.hallucination_score, "hallucination_score");
    }

    #[test]
    fn useful_tokens_le_total_tokens() {
        let report = common::inspect_default(RUST_QUERY);
        assert!(
            report.context.useful_tokens <= report.context.total_tokens,
            "useful_tokens ({}) must not exceed total_tokens ({})",
            report.context.useful_tokens,
            report.context.total_tokens
        );
    }

    #[test]
    fn model_override_is_reflected_in_report() {
        let override_model = "gpt-4-turbo";
        let inspector = Inspector::new(PipelineConfig::default(), Some(override_model.to_string()));
        let report = inspector.inspect(RUST_QUERY, InspectMode::Full).unwrap();
        assert_eq!(report.model_name, override_model);
    }

    #[test]
    fn inspection_report_serializes_to_json() {
        let report = common::inspect_default(GENERIC_QUERY);
        let json =
            serde_json::to_string_pretty(&report).expect("InspectionReport must serialize to JSON");
        let restored: InspectionReport =
            serde_json::from_str(&json).expect("JSON must deserialize back to InspectionReport");
        assert_eq!(restored.query, report.query);
        assert!((restored.overall_score - report.overall_score).abs() < 1e-9);
    }

    #[test]
    fn retrieval_chunk_ids_are_non_empty() {
        let report = common::inspect_default(RUST_QUERY);
        for chunk in &report.retrieval.chunks {
            assert!(!chunk.id.is_empty(), "chunk.id must not be empty");
            assert!(
                !chunk.source_doc.is_empty(),
                "chunk.source_doc must not be empty"
            );
        }
    }

    #[test]
    fn total_retrieved_tokens_equals_sum_of_chunks() {
        let report = common::inspect_default(RUST_QUERY);
        let sum: usize = report.retrieval.chunks.iter().map(|c| c.token_count).sum();
        assert_eq!(report.retrieval.total_retrieved_tokens, sum);
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// 3. NaivePipeline E2E Tests
// ════════════════════════════════════════════════════════════════════════════════

mod naive_pipeline_tests {
    use raginspect::naive_pipeline::{MockChunk, NaivePipeline};
    use raginspect::PipelineProfile;

    #[test]
    fn default_pipeline_produces_four_stages() {
        let pipeline = NaivePipeline::new();
        let profile = pipeline.run("What is RAG?", 5);
        assert_eq!(profile.stage_count(), 4);

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
    fn custom_corpus_top_k_one() {
        let custom = vec![
            MockChunk {
                id: "c1".to_string(),
                source: "doc1.md".to_string(),
                content: "RAG combines retrieval with generation.".to_string(),
                mock_embedding: 0.8,
            },
            MockChunk {
                id: "c2".to_string(),
                source: "doc2.md".to_string(),
                content: "Vector search uses embeddings.".to_string(),
                mock_embedding: 0.6,
            },
        ];
        let pipeline = NaivePipeline::with_corpus(custom);
        let profile = pipeline.run("What is RAG?", 1);

        let search_stage = profile.find("vector_search").expect("search stage exists");
        assert_eq!(search_stage.metadata.get("top_k"), Some(&"1".to_string()));
    }

    #[test]
    fn custom_corpus_top_k_three() {
        let pipeline = NaivePipeline::new();
        let profile = pipeline.run("vector databases", 3);
        let search_stage = profile.find("vector_search").unwrap();
        assert_eq!(search_stage.metadata.get("top_k"), Some(&"3".to_string()));
    }

    #[test]
    fn pipeline_profile_json_round_trip() {
        let pipeline = NaivePipeline::new();
        let profile = pipeline.run("hallucination in RAG", 3);

        let json = serde_json::to_string(&profile).expect("serialize");
        let restored: PipelineProfile = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.stage_count(), profile.stage_count());
        assert_eq!(restored.total_duration_ms, profile.total_duration_ms);
        assert_eq!(restored.total_tokens, profile.total_tokens);
        assert!((restored.total_cost - profile.total_cost).abs() < 1e-12);
    }

    #[test]
    fn total_cost_equals_sum_of_stage_costs() {
        let pipeline = NaivePipeline::new();
        let profile = pipeline.run("test query", 3);
        let sum: f64 = profile.stages.iter().map(|s| s.cost).sum();
        assert!(
            (profile.total_cost - sum).abs() < 1e-12,
            "total_cost {} should equal sum {}",
            profile.total_cost,
            sum
        );
    }

    #[test]
    fn total_duration_equals_sum_of_stage_durations() {
        let pipeline = NaivePipeline::new();
        let profile = pipeline.run("test query", 3);
        let sum: u128 = profile.stages.iter().map(|s| s.duration_ms).sum();
        assert_eq!(profile.total_duration_ms, sum);
    }

    #[test]
    fn total_tokens_equals_sum_of_stage_tokens() {
        let pipeline = NaivePipeline::new();
        let profile = pipeline.run("test query", 3);
        let sum: usize = profile.stages.iter().map(|s| s.token_count).sum();
        assert_eq!(profile.total_tokens, sum);
    }

    #[test]
    fn fast_latencies_produce_valid_profile() {
        let pipeline = NaivePipeline::new().with_latencies(1, 1, 1);
        let profile = pipeline.run("fast", 2);
        assert!(profile.total_duration_ms > 0);
        assert!(profile.stage_count() == 4);
    }

    #[test]
    fn empty_query_produces_valid_profile() {
        let pipeline = NaivePipeline::new();
        let profile = pipeline.run("", 3);
        assert_eq!(profile.stage_count(), 4);
        // Empty query should have 0 tokens in embedding stage
        let emb = profile.find("query_embedding").unwrap();
        assert_eq!(emb.token_count, 0);
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// 4. Profiler E2E Tests
// ════════════════════════════════════════════════════════════════════════════════

mod profiler_e2e {
    use raginspect::{MultiRunProfiler, Stage};

    #[test]
    fn multi_run_produces_correct_run_counts() {
        let mut profiler = MultiRunProfiler::new(5);
        let durations = [10, 12, 8, 15, 11];
        for &d in &durations {
            profiler.run(|profile| {
                profile.add_stage(
                    Stage::new("embedding")
                        .with_duration(d)
                        .with_tokens(8)
                        .with_cost(0.0001),
                );
            });
        }
        let stats = profiler.compute_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].runs, 5);
    }

    #[test]
    fn p50_le_p99_for_all_stages() {
        let mut profiler = MultiRunProfiler::new(10);
        let durs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        for &d in &durs {
            profiler.run(|profile| {
                profile.add_stage(Stage::new("stage_a").with_duration(d));
                profile.add_stage(Stage::new("stage_b").with_duration(d * 2));
            });
        }
        let stats = profiler.compute_stats();
        for s in &stats {
            assert!(
                s.p50_ms <= s.p99_ms,
                "p50 ({}) must be <= p99 ({}) for {}",
                s.p50_ms,
                s.p99_ms,
                s.name
            );
        }
    }

    #[test]
    fn min_le_p50_le_max() {
        let mut profiler = MultiRunProfiler::new(5);
        let durs = [10, 30, 20, 50, 40];
        for &d in &durs {
            profiler.run(|profile| {
                profile.add_stage(Stage::new("stage").with_duration(d));
            });
        }
        let stats = profiler.compute_stats();
        for s in &stats {
            assert!(
                s.min_ms <= s.p50_ms,
                "min ({}) must be <= p50 ({})",
                s.min_ms,
                s.p50_ms
            );
            assert!(
                s.p50_ms <= s.max_ms,
                "p50 ({}) must be <= max ({})",
                s.p50_ms,
                s.max_ms
            );
        }
    }

    #[test]
    fn empty_profiler_produces_empty_stats() {
        let profiler = MultiRunProfiler::new(3);
        let stats = profiler.compute_stats();
        assert!(stats.is_empty());
    }

    #[test]
    fn single_run_stats_have_p50_eq_p99() {
        let mut profiler = MultiRunProfiler::new(1);
        profiler.run(|profile| {
            profile.add_stage(Stage::new("only_stage").with_duration(42));
        });
        let stats = profiler.compute_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].runs, 1);
        assert_eq!(stats[0].p50_ms, stats[0].p99_ms);
        assert_eq!(stats[0].min_ms, stats[0].max_ms);
    }

    #[test]
    fn stats_stage_names_are_preserved() {
        let mut profiler = MultiRunProfiler::new(3);
        for _ in 0..3 {
            profiler.run(|profile| {
                profile.add_stage(Stage::new("alpha"));
                profile.add_stage(Stage::new("beta"));
                profile.add_stage(Stage::new("gamma"));
            });
        }
        let stats = profiler.compute_stats();
        let names: Vec<&str> = stats.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// 5. Architecture Detection Tests
// ════════════════════════════════════════════════════════════════════════════════

mod arch_detection {
    use raginspect::RagArchitecture;

    #[test]
    fn all_architectures_have_unique_diagnostic_focus() {
        let architectures = [
            RagArchitecture::Naive,
            RagArchitecture::Advanced,
            RagArchitecture::Modular,
            RagArchitecture::Agentic,
            RagArchitecture::Graph,
            RagArchitecture::Hyde,
            RagArchitecture::Multimodal,
        ];
        let focuses: Vec<&str> = architectures.iter().map(|a| a.diagnostic_focus()).collect();
        // All should be unique
        let unique: std::collections::HashSet<&str> = focuses.iter().copied().collect();
        assert_eq!(
            unique.len(),
            architectures.len(),
            "All architectures must have unique diagnostic_focus strings"
        );
    }

    #[test]
    fn all_architectures_have_non_empty_diagnostic_layers() {
        let architectures = [
            RagArchitecture::Naive,
            RagArchitecture::Advanced,
            RagArchitecture::Modular,
            RagArchitecture::Agentic,
            RagArchitecture::Graph,
            RagArchitecture::Hyde,
            RagArchitecture::Multimodal,
        ];
        for arch in &architectures {
            let layers = arch.diagnostic_layers();
            assert!(
                !layers.is_empty(),
                "Architecture {:?} must have non-empty diagnostic_layers",
                arch
            );
        }
    }

    #[test]
    fn all_seven_architectures_are_covered() {
        let variants = [
            RagArchitecture::Naive,
            RagArchitecture::Advanced,
            RagArchitecture::Modular,
            RagArchitecture::Agentic,
            RagArchitecture::Graph,
            RagArchitecture::Hyde,
            RagArchitecture::Multimodal,
        ];
        assert_eq!(variants.len(), 7, "Must have exactly 7 architectures");
    }

    #[test]
    fn architecture_display_is_descriptive() {
        let arch = RagArchitecture::Naive;
        let display = format!("{}", arch);
        assert!(!display.is_empty());
        assert!(display.contains("Naive"));
    }

    #[test]
    fn naive_focus_mentions_retrieval() {
        let focus = RagArchitecture::Naive.diagnostic_focus();
        assert!(
            focus.to_lowercase().contains("retrieval") || focus.to_lowercase().contains("dense"),
            "Naive focus should mention retrieval: got '{}'",
            focus
        );
    }
}
