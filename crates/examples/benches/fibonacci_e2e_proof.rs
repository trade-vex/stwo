//! End-to-end STARK proof generation benchmark: Metal GPU vs SIMD CPU
//!
//! This benchmark measures COMPLETE proof generation time, including:
//! - Twiddle precomputation
//! - Trace generation
//! - Trace commitment (FFT)
//! - Constraint evaluation
//! - Quotient computation (IFFT + FFT)
//! - FRI proving
//! - Merkle tree construction
//!
//! Run with: cargo bench --features metal_prover --bench fibonacci_e2e_proof

use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use num_traits::{One, Zero};
use stwo::core::channel::Blake2sM31Channel;
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::pcs::PcsConfig;
use stwo::core::poly::circle::CanonicCoset;
use stwo::core::vcs::blake2_merkle::Blake2sM31MerkleChannel;
#[cfg(all(target_os = "macos", feature = "metal_prover"))]
use stwo::prover::backend::metal::{
    MetalBackend, MetalBlake2sM31Channel, MetalBlake2sM31MerkleChannel,
};
use stwo::prover::backend::simd::m31::{PackedBaseField, LOG_N_LANES};
use stwo::prover::backend::simd::SimdBackend;
#[cfg(all(target_os = "macos", feature = "metal_prover"))]
use stwo::prover::backend::{Col, Column};
use stwo::prover::poly::circle::{CircleEvaluation, PolyOps};
use stwo::prover::poly::BitReversedOrder;
use stwo::prover::{prove, CommitmentSchemeProver};
use stwo_constraint_framework::TraceLocationAllocator;
use stwo_examples::wide_fibonacci::{
    generate_trace, FibInput, WideFibonacciComponent, WideFibonacciEval,
};

const FIB_SEQUENCE_LENGTH: usize = 100;

// Helper function to generate test trace (copied from tests module)
fn generate_test_trace(
    log_n_instances: u32,
) -> stwo::core::ColumnVec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> {
    if log_n_instances < LOG_N_LANES {
        let n_instances = 1 << log_n_instances;
        let inputs = vec![FibInput {
            a: PackedBaseField::from_array(std::array::from_fn(|j| {
                if j < n_instances {
                    BaseField::one()
                } else {
                    BaseField::zero()
                }
            })),
            b: PackedBaseField::from_array(std::array::from_fn(|j| {
                if j < n_instances {
                    BaseField::from_u32_unchecked((j) as u32)
                } else {
                    BaseField::zero()
                }
            })),
        }];
        return generate_trace::<FIB_SEQUENCE_LENGTH>(log_n_instances, &inputs);
    }
    let inputs = (0..(1 << (log_n_instances - LOG_N_LANES)))
        .map(|i| FibInput {
            a: PackedBaseField::one(),
            b: PackedBaseField::from_array(std::array::from_fn(|j| {
                BaseField::from_u32_unchecked((i * 16 + j) as u32)
            })),
        })
        .collect_vec();
    generate_trace::<FIB_SEQUENCE_LENGTH>(log_n_instances, &inputs)
}

/// Benchmark SIMD end-to-end proof generation
fn bench_simd_e2e_proof(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_proof_simd");
    group.sample_size(10);

    // Test realistic problem sizes
    // log_n_instances controls the number of Fibonacci instances, not trace size
    // The actual trace has ~100 columns
    for log_n_instances in [6, 8, 10, 12, 14, 15, 16, 17, 18] {
        let description = format!(
            "log_n={} (~{}K trace)",
            log_n_instances,
            (1 << log_n_instances) / 1024
        );

        group.bench_function(BenchmarkId::new("simd", &description), |b| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;

                for _ in 0..iters {
                    let config = PcsConfig::default();

                    let start = Instant::now();

                    // Precompute twiddles
                    let twiddles = SimdBackend::precompute_twiddles(
                        CanonicCoset::new(
                            log_n_instances + 1 + config.fri_config.log_blowup_factor,
                        )
                        .circle_domain()
                        .half_coset,
                    );

                    // Setup protocol
                    let prover_channel = &mut Blake2sM31Channel::default();
                    let mut commitment_scheme = CommitmentSchemeProver::<
                        SimdBackend,
                        Blake2sM31MerkleChannel,
                    >::new(config, &twiddles);

                    // Preprocessed trace (empty)
                    let mut tree_builder = commitment_scheme.tree_builder();
                    tree_builder.extend_evals([]);
                    tree_builder.commit(prover_channel);

                    // Generate trace
                    let trace = generate_test_trace(log_n_instances);
                    let mut tree_builder = commitment_scheme.tree_builder();
                    tree_builder.extend_evals(trace);
                    tree_builder.commit(prover_channel);

                    // Prove constraints
                    let component = WideFibonacciComponent::new(
                        &mut TraceLocationAllocator::default(),
                        WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                            log_n_rows: log_n_instances,
                        },
                        SecureField::zero(),
                    );

                    let _proof = prove::<SimdBackend, Blake2sM31MerkleChannel>(
                        &[&component],
                        prover_channel,
                        commitment_scheme,
                    )
                    .expect("SIMD proof generation failed");

                    total_time += start.elapsed();
                }

                total_time
            });
        });
    }

    group.finish();
}

#[cfg(all(target_os = "macos", feature = "metal_prover"))]
fn bench_metal_e2e_proof(c: &mut Criterion) {
    // Initialize profiling if METAL_PROFILE env var is set
    stwo::prover::backend::metal::profiling::init_profiling();

    let mut group = c.benchmark_group("e2e_proof_metal");
    group.sample_size(20);

    for log_n_instances in [6, 8, 10, 12, 14, 15, 16, 17, 18] {
        let description = format!(
            "log_n={} (~{}K trace)",
            log_n_instances,
            (1 << log_n_instances) / 1024
        );

        group.bench_function(BenchmarkId::new("metal", &description), |b| {
            b.iter_custom(|iters| {
                let mut total_time = std::time::Duration::ZERO;

                for _ in 0..iters {
                    let config = PcsConfig::default();

                    let start = Instant::now();

                    // Precompute twiddles
                    let twiddles = MetalBackend::precompute_twiddles(
                        CanonicCoset::new(
                            log_n_instances + 1 + config.fri_config.log_blowup_factor,
                        )
                        .circle_domain()
                        .half_coset,
                    );

                    // Setup protocol with GPU-resident channel state
                    let prover_channel = &mut MetalBlake2sM31Channel::default();
                    let mut commitment_scheme = CommitmentSchemeProver::<
                        MetalBackend,
                        MetalBlake2sM31MerkleChannel,
                    >::new(config, &twiddles);

                    // Preprocessed trace (empty)
                    let mut tree_builder = commitment_scheme.tree_builder();
                    tree_builder.extend_evals([]);
                    tree_builder.commit(prover_channel);

                    // Generate trace using SIMD, then convert to Metal
                    let simd_trace = generate_test_trace(log_n_instances);
                    let metal_trace: Vec<
                        CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>,
                    > = simd_trace
                        .iter()
                        .map(|simd_eval| {
                            let cpu_values = simd_eval.values.to_cpu();
                            let metal_col: Col<MetalBackend, BaseField> =
                                cpu_values.into_iter().collect();
                            CircleEvaluation::<MetalBackend, _, BitReversedOrder>::new(
                                simd_eval.domain,
                                metal_col,
                            )
                        })
                        .collect();

                    let mut tree_builder = commitment_scheme.tree_builder();
                    tree_builder.extend_evals(metal_trace);
                    tree_builder.commit(prover_channel);

                    // Prove constraints
                    let component = WideFibonacciComponent::new(
                        &mut TraceLocationAllocator::default(),
                        WideFibonacciEval::<FIB_SEQUENCE_LENGTH> {
                            log_n_rows: log_n_instances,
                        },
                        SecureField::zero(),
                    );

                    let _proof = prove::<MetalBackend, MetalBlake2sM31MerkleChannel>(
                        &[&component],
                        prover_channel,
                        commitment_scheme,
                    )
                    .expect("Metal proof generation failed");

                    total_time += start.elapsed();
                }

                total_time
            });
        });
    }

    group.finish();
}

#[cfg(all(target_os = "macos", feature = "metal_prover"))]
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_simd_e2e_proof, bench_metal_e2e_proof
);

#[cfg(not(all(target_os = "macos", feature = "metal_prover")))]
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_simd_e2e_proof
);

criterion_main!(benches);
