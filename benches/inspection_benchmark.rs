//! Benchmark: raginspect pipeline inspection performance
//!
//! Run with: `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use raginspect::{InspectMode, Inspector, PipelineConfig, RagArchitecture};

fn bench_naive_inspection(c: &mut Criterion) {
    let mut inspector = Inspector::new(PipelineConfig::default(), None);
    inspector.set_architecture(RagArchitecture::Naive);

    c.bench_function("naive_full_inspection", |b| {
        b.iter(|| {
            black_box(
                inspector
                    .inspect("What is the memory overhead of Tokio tasks?", InspectMode::Full)
                    .unwrap(),
            );
        });
    });
}

fn bench_architecture_recommendations(c: &mut Criterion) {
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

        let label = format!("arch_recommendations_{:?}", arch);
        c.bench_function(&label, |b| {
            b.iter(|| {
                black_box(
                    inspector
                        .inspect("benchmark query", InspectMode::Quick)
                        .unwrap(),
                );
            });
        });
    }
}

criterion_group!(benches, bench_naive_inspection, bench_architecture_recommendations);
criterion_main!(benches);
