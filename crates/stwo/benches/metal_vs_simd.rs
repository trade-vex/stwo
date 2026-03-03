//! Benchmark comparing Metal GPU backend vs SIMD CPU backend.
//!
//! This benchmark will compare the performance of:
//! - FFT/IFFT operations
//! - FRI folding
//! - Quotient accumulation
//! - Merkle tree construction
//!
//! Run with: cargo bench --features metal_prover --bench metal_vs_simd

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use stwo::core::fields::m31::BaseField;
use stwo::core::poly::circle::CanonicCoset;
#[cfg(target_os = "macos")]
use stwo::prover::backend::metal::MetalBackend;
use stwo::prover::backend::simd::column::BaseColumn as SimdColumn;
use stwo::prover::backend::simd::SimdBackend;
use stwo::prover::backend::ColumnOps;
use stwo::prover::poly::circle::{CircleCoefficients, PolyOps};

pub fn column_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("column_allocation");

    for log_size in [16, 20, 24] {
        let size = 1 << log_size;

        group.throughput(Throughput::Bytes((size * 4) as u64));

        // SIMD column allocation
        group.bench_function(BenchmarkId::new("simd", log_size), |b| {
            b.iter(|| {
                let _col: SimdColumn = (0..size).map(BaseField::from).collect();
            })
        });

        // TODO: Add Metal column allocation benchmark when implemented
        // group.bench_function(BenchmarkId::new("metal", log_size), |b| {
        //     b.iter(|| {
        //         let _col: <MetalBackend as ColumnOps<BaseField>>::Column =
        //             (0..size).map(BaseField::from).collect();
        //     })
        // });
    }

    group.finish();
}

/// Benchmark FFT operations comparing Metal GPU vs SIMD CPU implementations.
/// Tests log_sizes where non_vecwise_layers is divisible by 3.
/// For non_vecwise = (num_fft_layers - 5) to be divisible by 3:
/// num_fft_layers = 5 + 3k, so log_size = 6 + 3k
///
/// NOTE: Phase 1 Metal implementation currently falls back to SIMD because
/// vecwise layers (bottom 5) are not yet implemented on GPU. This benchmark
/// shows the current fallback performance and provides a baseline for Phase 2.
#[cfg(target_os = "macos")]
pub fn fft_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");
    group.sample_size(10);

    // Test sizes where non_vecwise_layers is divisible by 3
    // log_size 12: num_fft_layers = 11, non_vecwise = 6 (6 % 3 = 0) ✓
    // log_size 15: num_fft_layers = 14, non_vecwise = 9 (9 % 3 = 0) ✓
    // log_size 18: num_fft_layers = 17, non_vecwise = 12 (12 % 3 = 0) ✓
    // log_size 21: num_fft_layers = 20, non_vecwise = 15 (15 % 3 = 0) ✓
    // log_size 24: num_fft_layers = 23, non_vecwise = 18 (18 % 3 = 0) ✓
    for log_size in [12, 15, 18, 21, 24] {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;
        group.throughput(Throughput::Bytes((size * 4) as u64));

        // SIMD FFT benchmark
        group.bench_function(BenchmarkId::new("simd_fft", log_size), |b| {
            b.iter_batched(
                || {
                    let coeffs: SimdColumn = (0..size).map(BaseField::from).collect();
                    let poly = CircleCoefficients::new(coeffs);
                    let twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
                    (poly, twiddles)
                },
                |(poly, twiddles)| SimdBackend::evaluate(&poly, domain, &twiddles),
                BatchSize::LargeInput,
            )
        });

        // Metal FFT benchmark
        group.bench_function(BenchmarkId::new("metal_fft", log_size), |b| {
            b.iter_batched(
                || {
                    let coeffs: <MetalBackend as ColumnOps<BaseField>>::Column =
                        (0..size).map(BaseField::from).collect();
                    let poly = CircleCoefficients::new(coeffs);
                    let twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
                    (poly, twiddles)
                },
                |(poly, twiddles)| MetalBackend::evaluate(&poly, domain, &twiddles),
                BatchSize::LargeInput,
            )
        });
    }

    group.finish();
}

/// Benchmark IFFT operations comparing Metal GPU vs SIMD CPU implementations.
/// Same size constraints as FFT benchmark.
#[cfg(target_os = "macos")]
pub fn ifft_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifft");
    group.sample_size(10);

    for log_size in [12, 15, 18, 21, 24] {
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;
        group.throughput(Throughput::Bytes((size * 4) as u64));

        // SIMD IFFT benchmark
        group.bench_function(BenchmarkId::new("simd_ifft", log_size), |b| {
            b.iter_batched(
                || {
                    let coeffs: SimdColumn = (0..size).map(BaseField::from).collect();
                    let poly = CircleCoefficients::new(coeffs);
                    let twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
                    let eval = SimdBackend::evaluate(&poly, domain, &twiddles);
                    (eval, twiddles)
                },
                |(eval, twiddles)| SimdBackend::interpolate(eval, &twiddles),
                BatchSize::LargeInput,
            )
        });

        // Metal IFFT benchmark
        group.bench_function(BenchmarkId::new("metal_ifft", log_size), |b| {
            b.iter_batched(
                || {
                    let coeffs: <MetalBackend as ColumnOps<BaseField>>::Column =
                        (0..size).map(BaseField::from).collect();
                    let poly = CircleCoefficients::new(coeffs);
                    let twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
                    let eval = MetalBackend::evaluate(&poly, domain, &twiddles);
                    (eval, twiddles)
                },
                |(eval, twiddles)| MetalBackend::interpolate(eval, &twiddles),
                BatchSize::LargeInput,
            )
        });
    }

    group.finish();
}

#[cfg(target_os = "macos")]
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = column_allocation, fft_comparison, ifft_comparison
);

#[cfg(not(target_os = "macos"))]
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = column_allocation
);

criterion_main!(benches);
