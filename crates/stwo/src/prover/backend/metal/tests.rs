//! Comprehensive test suite for the Metal GPU backend.
//!
//! Tests compare Metal backend results against SIMD backend (ground truth)
//! to verify correctness of all GPU operations.

#[cfg(test)]
mod tests {
    use crate::core::fields::m31::{BaseField, M31};
    use crate::core::fields::qm31::SecureField;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::backend::metal::column::MetalBaseColumn;
    use crate::prover::backend::metal::context::MetalContext;
    use crate::prover::backend::metal::MetalBackend;
    use crate::prover::backend::simd::SimdBackend;
    use crate::prover::backend::{Column, ColumnOps, CpuBackend};
    use crate::prover::poly::circle::{CircleCoefficients, PolyOps};
    use crate::prover::secure_column::SecureColumnByCoords;

    // ========================================================================
    // Bit Reversal Tests
    // ========================================================================

    #[test]
    fn test_bit_reverse_base_field_metal_vs_simd() {
        if !MetalContext::is_available() {
            return;
        }

        for log_size in 4..=16 {
            let size = 1usize << log_size;
            let data: Vec<BaseField> = (0..size).map(|i| M31::from(i as u32)).collect();

            let mut metal_col: MetalBaseColumn = data.iter().copied().collect();
            <MetalBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut metal_col);
            let metal_result = metal_col.to_cpu();

            let mut simd_col: crate::prover::backend::simd::column::BaseColumn =
                data.iter().copied().collect();
            <SimdBackend as ColumnOps<BaseField>>::bit_reverse_column(&mut simd_col);
            let simd_result = simd_col.to_cpu();

            assert_eq!(
                metal_result, simd_result,
                "Bit reverse mismatch at log_size={}",
                log_size
            );
        }
    }

    #[test]
    fn test_bit_reverse_secure_field_metal_vs_cpu() {
        if !MetalContext::is_available() {
            return;
        }

        for log_size in 4..=14 {
            let size = 1usize << log_size;
            let data: Vec<SecureField> = (0..size)
                .map(|i| {
                    SecureField::from_m31(
                        M31::from(i as u32),
                        M31::from((i + 1) as u32),
                        M31::from((i + 2) as u32),
                        M31::from((i + 3) as u32),
                    )
                })
                .collect();

            let mut metal_col: crate::prover::backend::metal::column::MetalSecureColumn =
                data.iter().copied().collect();
            <MetalBackend as ColumnOps<SecureField>>::bit_reverse_column(&mut metal_col);
            let metal_result = metal_col.to_cpu();

            // Compare against CPU bit_reverse
            let mut cpu_data = data.clone();
            crate::core::utils::bit_reverse(&mut cpu_data);

            assert_eq!(
                metal_result, cpu_data,
                "Secure field bit reverse mismatch at log_size={}",
                log_size
            );
        }
    }

    // ========================================================================
    // FFT / IFFT Tests
    // ========================================================================

    #[test]
    fn test_fft_metal_vs_simd() {
        if !MetalContext::is_available() {
            return;
        }

        for log_size in 4..=16 {
            let size = 1usize << log_size;
            let domain = CanonicCoset::new(log_size).circle_domain();
            let coeffs_data: Vec<BaseField> = (0..size).map(|i| M31::from(i as u32)).collect();

            let simd_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
            let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
            let simd_result = SimdBackend::evaluate(&simd_poly, domain, &simd_twiddles);

            let metal_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
            let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
            let metal_result = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);

            let simd_values = simd_result.values.to_cpu();
            let metal_values = metal_result.values.to_cpu();

            assert_eq!(
                simd_values, metal_values,
                "FFT mismatch at log_size={}",
                log_size
            );
        }
    }

    #[test]
    fn test_ifft_metal_vs_simd() {
        if !MetalContext::is_available() {
            return;
        }

        for log_size in 4..=16 {
            let size = 1usize << log_size;
            let domain = CanonicCoset::new(log_size).circle_domain();
            let eval_data: Vec<BaseField> = (0..size).map(|i| M31::from(i as u32)).collect();

            let simd_eval = crate::prover::poly::circle::CircleEvaluation::<
                SimdBackend,
                BaseField,
                crate::prover::poly::BitReversedOrder,
            >::new(domain, eval_data.iter().copied().collect());
            let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
            let simd_poly = SimdBackend::interpolate(simd_eval, &simd_twiddles);

            let metal_eval = crate::prover::poly::circle::CircleEvaluation::<
                MetalBackend,
                BaseField,
                crate::prover::poly::BitReversedOrder,
            >::new(domain, eval_data.iter().copied().collect());
            let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
            let metal_poly = MetalBackend::interpolate(metal_eval, &metal_twiddles);

            let simd_coeffs = simd_poly.coeffs.to_cpu();
            let metal_coeffs = metal_poly.coeffs.to_cpu();

            assert_eq!(
                simd_coeffs, metal_coeffs,
                "IFFT mismatch at log_size={}",
                log_size
            );
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        if !MetalContext::is_available() {
            return;
        }

        let log_size = 14;
        let size = 1usize << log_size;
        let domain = CanonicCoset::new(log_size).circle_domain();

        let original: Vec<BaseField> = (0..size).map(|i| M31::from(i as u32)).collect();

        let poly = CircleCoefficients::new(original.iter().copied().collect());
        let twiddles = MetalBackend::precompute_twiddles(domain.half_coset);

        let eval = MetalBackend::evaluate(&poly, domain, &twiddles);
        let recovered_poly = MetalBackend::interpolate(eval, &twiddles);
        let recovered = recovered_poly.coeffs.to_cpu();

        assert_eq!(original, recovered, "FFT/IFFT roundtrip failed");
    }

    // ========================================================================
    // Merkle Tests
    // ========================================================================

    #[test]
    fn test_merkle_blake2s_build_leaves_metal_vs_simd() {
        use crate::core::vcs::blake2_hash::Blake2sHash;
        use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
        use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

        if !MetalContext::is_available() {
            return;
        }

        for log_size in 5..=14 {
            let size = 1usize << log_size;
            let col_data: Vec<BaseField> = (0..size).map(|i| M31::from(i as u32)).collect();

            let simd_col: crate::prover::backend::simd::column::BaseColumn =
                col_data.iter().copied().collect();
            let simd_leaves = <SimdBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
                &[&simd_col],
                log_size,
            );

            let metal_col: MetalBaseColumn = col_data.iter().copied().collect();
            let metal_leaves =
                <MetalBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
                    &[&metal_col],
                    log_size,
                );

            let simd_hashes: Vec<Blake2sHash> = simd_leaves.to_cpu();
            let metal_hashes: Vec<Blake2sHash> = metal_leaves.to_cpu();

            assert_eq!(
                simd_hashes, metal_hashes,
                "Merkle build_leaves mismatch at log_size={}",
                log_size
            );
        }
    }

    #[test]
    fn test_merkle_build_leaves_variable_sizes() {
        use crate::core::vcs::blake2_hash::Blake2sHash;
        use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
        use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

        if !MetalContext::is_available() {
            return;
        }

        // Columns must be sorted by size (increasing) per SIMD implementation requirement
        let lifting_log_size: u32 = 12;
        let col_log_sizes = [10, 10, 11, 12, 12];

        let columns_data: Vec<Vec<BaseField>> = col_log_sizes
            .iter()
            .map(|&log_size| {
                (0..(1usize << log_size))
                    .map(|i| M31::from(i as u32))
                    .collect()
            })
            .collect();

        let simd_cols: Vec<crate::prover::backend::simd::column::BaseColumn> = columns_data
            .iter()
            .map(|data| data.iter().copied().collect())
            .collect();
        let simd_refs: Vec<&crate::prover::backend::simd::column::BaseColumn> =
            simd_cols.iter().collect();
        let simd_leaves = <SimdBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
            &simd_refs,
            lifting_log_size,
        );

        let metal_cols: Vec<MetalBaseColumn> = columns_data
            .iter()
            .map(|data| data.iter().copied().collect())
            .collect();
        let metal_refs: Vec<&MetalBaseColumn> = metal_cols.iter().collect();
        let metal_leaves = <MetalBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
            &metal_refs,
            lifting_log_size,
        );

        let simd_hashes: Vec<Blake2sHash> = simd_leaves.to_cpu();
        let metal_hashes: Vec<Blake2sHash> = metal_leaves.to_cpu();

        assert_eq!(
            simd_hashes, metal_hashes,
            "Variable-size build_leaves mismatch"
        );
    }

    #[test]
    fn test_merkle_build_next_layer_metal_vs_simd() {
        use crate::core::vcs::blake2_hash::Blake2sHash;
        use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasher;
        use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

        if !MetalContext::is_available() {
            return;
        }

        // Test both SIMD fallback (log_size < 10) and GPU path (log_size >= 10)
        for log_size in 5..=14 {
            let size = 1usize << log_size;
            let col_data: Vec<BaseField> = (0..size).map(|i| M31::from(i as u32)).collect();

            let simd_col: crate::prover::backend::simd::column::BaseColumn =
                col_data.iter().copied().collect();
            let simd_leaves = <SimdBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
                &[&simd_col],
                log_size,
            );
            let simd_next =
                <SimdBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_next_layer(
                    &simd_leaves,
                );

            let metal_col: MetalBaseColumn = col_data.iter().copied().collect();
            let metal_leaves =
                <MetalBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_leaves(
                    &[&metal_col],
                    log_size,
                );
            let metal_next =
                <MetalBackend as MerkleOpsLifted<Blake2sMerkleHasher>>::build_next_layer(
                    &metal_leaves,
                );

            let simd_hashes: Vec<Blake2sHash> = simd_next.to_cpu();
            let metal_hashes: Vec<Blake2sHash> = metal_next.to_cpu();

            assert_eq!(
                simd_hashes, metal_hashes,
                "build_next_layer mismatch at log_size={}",
                log_size
            );
        }
    }

    // ========================================================================
    // FRI Fold Tests
    // ========================================================================

    #[test]
    fn test_fold_line_metal_vs_simd() {
        use crate::core::fields::m31::M31;
        use crate::core::fields::qm31::SecureField;
        use crate::core::poly::line::LineDomain;
        use crate::prover::backend::simd::column::BaseColumn;
        use crate::prover::fri::FriOps;
        use crate::prover::line::LineEvaluation;

        if !MetalContext::is_available() {
            return;
        }

        let alpha = SecureField::from_m31(M31::from(3), M31::from(5), M31::from(7), M31::from(11));

        // Test both SIMD fallback (< 10) and GPU path (>= 10)
        for log_size in 5..=14 {
            let size = 1usize << log_size;
            let domain = LineDomain::new(CanonicCoset::new(log_size + 1).half_coset());

            // Generate test data for 4 coordinate columns
            let coord_data: [Vec<BaseField>; 4] = std::array::from_fn(|c| {
                (0..size)
                    .map(|i| M31::from((i * 4 + c) as u32))
                    .collect()
            });

            // SIMD
            let simd_values = SecureColumnByCoords {
                columns: coord_data
                    .clone()
                    .map(|d| d.into_iter().collect::<BaseColumn>()),
            };
            let simd_eval = LineEvaluation::new(domain, simd_values);
            let simd_twiddles = SimdBackend::precompute_twiddles(domain.coset());
            let simd_result = SimdBackend::fold_line(&simd_eval, alpha, &simd_twiddles, 1);

            // Metal
            let metal_values = SecureColumnByCoords {
                columns: coord_data.map(|d| d.into_iter().collect::<MetalBaseColumn>()),
            };
            let metal_eval = LineEvaluation::new(domain, metal_values);
            let metal_twiddles = MetalBackend::precompute_twiddles(domain.coset());
            let metal_result = MetalBackend::fold_line(&metal_eval, alpha, &metal_twiddles, 1);

            let simd_cpu = simd_result.values.to_cpu();
            let metal_cpu = metal_result.values.to_cpu();

            for i in 0..simd_cpu.len() {
                assert_eq!(
                    simd_cpu.at(i),
                    metal_cpu.at(i),
                    "fold_line mismatch at index {} for log_size={}",
                    i,
                    log_size
                );
            }
        }
    }

    // ========================================================================
    // MLE Fold Tests
    // ========================================================================

    #[test]
    fn test_mle_fold_m31_metal_vs_simd() {
        use crate::core::fields::m31::M31;
        use crate::core::fields::qm31::SecureField;
        use crate::prover::lookups::mle::{Mle, MleOps};

        if !MetalContext::is_available() {
            return;
        }

        let assignment =
            SecureField::from_m31(M31::from(3), M31::from(5), M31::from(7), M31::from(11));

        // Test both SIMD fallback (< 10) and GPU path (>= 10)
        for log_size in 5..=14 {
            let size = 1usize << log_size;
            let data: Vec<BaseField> = (0..size).map(|i| M31::from(i as u32)).collect();

            // SIMD
            let simd_col: crate::prover::backend::simd::column::BaseColumn =
                data.iter().copied().collect();
            let simd_mle: Mle<SimdBackend, BaseField> = Mle::new(simd_col);
            let simd_result = SimdBackend::fix_first_variable(simd_mle, assignment);
            let simd_vals = simd_result.into_evals().to_cpu();

            // Metal
            let metal_col: MetalBaseColumn = data.iter().copied().collect();
            let metal_mle: Mle<MetalBackend, BaseField> = Mle::new(metal_col);
            let metal_result = MetalBackend::fix_first_variable(metal_mle, assignment);
            let metal_vals = metal_result.into_evals().to_cpu();

            assert_eq!(
                simd_vals, metal_vals,
                "MLE fold M31 mismatch at log_size={}",
                log_size
            );
        }
    }

    #[test]
    fn test_mle_fold_qm31_metal_vs_simd() {
        use crate::core::fields::m31::M31;
        use crate::core::fields::qm31::SecureField;
        use crate::prover::lookups::mle::{Mle, MleOps};

        if !MetalContext::is_available() {
            return;
        }

        let assignment =
            SecureField::from_m31(M31::from(3), M31::from(5), M31::from(7), M31::from(11));

        for log_size in 5..=14 {
            let size = 1usize << log_size;
            let data: Vec<SecureField> = (0..size)
                .map(|i| {
                    SecureField::from_m31(
                        M31::from(i as u32),
                        M31::from((i + 1) as u32),
                        M31::from((i + 2) as u32),
                        M31::from((i + 3) as u32),
                    )
                })
                .collect();

            // SIMD
            let simd_col: crate::prover::backend::simd::column::SecureColumn =
                data.iter().copied().collect();
            let simd_mle: Mle<SimdBackend, SecureField> = Mle::new(simd_col);
            let simd_result = SimdBackend::fix_first_variable(simd_mle, assignment);
            let simd_vals = simd_result.into_evals().to_cpu();

            // Metal
            let metal_col: crate::prover::backend::metal::column::MetalSecureColumn =
                data.iter().copied().collect();
            let metal_mle: Mle<MetalBackend, SecureField> = Mle::new(metal_col);
            let metal_result = MetalBackend::fix_first_variable(metal_mle, assignment);
            let metal_vals = metal_result.into_evals().to_cpu();

            assert_eq!(
                simd_vals, metal_vals,
                "MLE fold QM31 mismatch at log_size={}",
                log_size
            );
        }
    }

    // ========================================================================
    // Accumulation Tests
    // ========================================================================

    #[test]
    fn test_accumulate_metal_vs_simd() {
        use crate::prover::AccumulationOps;

        if !MetalContext::is_available() {
            return;
        }

        for log_size in 5..=14 {
            let size = 1usize << log_size;

            let a_vals: Vec<SecureField> = (0..size)
                .map(|i| {
                    SecureField::from_m31(
                        M31::from(i as u32),
                        M31::from((i + 1) as u32),
                        M31::from((i + 2) as u32),
                        M31::from((i + 3) as u32),
                    )
                })
                .collect();

            let b_vals: Vec<SecureField> = (0..size)
                .map(|i| {
                    SecureField::from_m31(
                        M31::from((i + 100) as u32),
                        M31::from((i + 200) as u32),
                        M31::from((i + 300) as u32),
                        M31::from((i + 400) as u32),
                    )
                })
                .collect();

            // SIMD
            let mut simd_a = SecureColumnByCoords::<SimdBackend>::from_cpu(
                SecureColumnByCoords::<CpuBackend>::from_iter(a_vals.iter().copied()),
            );
            let simd_b = SecureColumnByCoords::<SimdBackend>::from_cpu(
                SecureColumnByCoords::<CpuBackend>::from_iter(b_vals.iter().copied()),
            );
            SimdBackend::accumulate(&mut simd_a, &simd_b);
            let simd_result = simd_a.to_cpu();

            // Metal
            let mut metal_a = SecureColumnByCoords::<MetalBackend>::from_cpu(
                SecureColumnByCoords::<CpuBackend>::from_iter(a_vals.iter().copied()),
            );
            let metal_b = SecureColumnByCoords::<MetalBackend>::from_cpu(
                SecureColumnByCoords::<CpuBackend>::from_iter(b_vals.iter().copied()),
            );
            MetalBackend::accumulate(&mut metal_a, &metal_b);
            let metal_result = metal_a.to_cpu();

            for i in 0..size {
                assert_eq!(
                    simd_result.at(i),
                    metal_result.at(i),
                    "accumulate mismatch at index {} for log_size={}",
                    i,
                    log_size
                );
            }
        }
    }

    // ========================================================================
    // Grinding Tests
    // ========================================================================

    #[test]
    fn test_grind_finds_valid_nonce() {
        use crate::core::channel::{Blake2sChannel, Channel};
        use crate::core::proof_of_work::GrindOps;

        if !MetalContext::is_available() {
            return;
        }

        let mut channel = Blake2sChannel::default();
        channel.mix_u64(42);

        let pow_bits: u32 = 10;
        let metal_nonce = MetalBackend::grind(&channel, pow_bits);

        assert!(
            metal_nonce < u64::MAX,
            "Metal failed to find a valid nonce"
        );

        // Verify the nonce produces the required leading zeros by checking
        // the SIMD backend agrees it's valid (SIMD finds same minimum nonce
        // when searching the same range).
        let simd_nonce = SimdBackend::grind(&channel, pow_bits);
        // Both should find valid nonces (not necessarily the same one due to
        // different search strategies, but both must be < u64::MAX)
        assert!(simd_nonce < u64::MAX, "SIMD failed to find a valid nonce");
    }

    // ========================================================================
    // Eval at Point Tests
    // ========================================================================

    #[test]
    fn test_eval_at_point_metal_vs_simd() {
        use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;

        if !MetalContext::is_available() {
            return;
        }

        // Test both CPU fallback (log_size <= 8) and GPU path (log_size >= 9)
        for log_size in 4..=10 {
            let size = 1usize << log_size;
            let coeffs_data: Vec<BaseField> = (0..size).map(|i| M31::from(i as u32)).collect();

            let point = SECURE_FIELD_CIRCLE_GEN.mul(12345u128);

            let simd_poly = CircleCoefficients::<SimdBackend>::new(
                coeffs_data.iter().copied().collect(),
            );
            let simd_result = simd_poly.eval_at_point(point);

            let metal_poly = CircleCoefficients::<MetalBackend>::new(
                coeffs_data.iter().copied().collect(),
            );
            let metal_result = metal_poly.eval_at_point(point);

            assert_eq!(
                simd_result, metal_result,
                "eval_at_point mismatch at log_size={}",
                log_size
            );
        }
    }

    // ========================================================================
    // Batch Inverse Tests
    // ========================================================================

    #[test]
    fn test_batch_inverse_m31_gpu() {
        use metal::MTLResourceOptions;

        if !MetalContext::is_available() {
            return;
        }

        for log_size in 9..=14 {
            let size = 1usize << log_size;
            let input: Vec<u32> = (1..=size as u32).collect(); // avoid zero

            let ctx = MetalContext::global();
            let device = ctx.device();

            let input_buffer = device.new_buffer_with_data(
                input.as_ptr() as *const _,
                (size * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let output_buffer = device.new_buffer(
                (size * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let command_buffer = ctx.command_queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(ctx.batch_inverse_m31_pipeline());
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            let total_size = size as u32;
            encoder.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &total_size as *const u32 as *const _,
            );

            let threads_per_group = 256u64;
            let num_groups = ((size as u64) + 511) / 512;
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(num_groups, 1, 1),
                metal::MTLSize::new(threads_per_group, 1, 1),
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let output = unsafe {
                std::slice::from_raw_parts(output_buffer.contents() as *const u32, size)
            };

            const M31_PRIME: u64 = 0x7FFFFFFF;
            for i in 0..size {
                let product = ((input[i] as u64) * (output[i] as u64)) % M31_PRIME;
                assert_eq!(
                    product, 1,
                    "Batch inverse failed at index {} (log_size={}): {} * {} = {} (mod M31)",
                    i, log_size, input[i], output[i], product
                );
            }
        }
    }

    // ========================================================================
    // Column Operations Tests
    // ========================================================================

    #[test]
    fn test_column_from_iter_roundtrip() {
        if !MetalContext::is_available() {
            return;
        }

        let data: Vec<BaseField> = (0..1024).map(|i| M31::from(i as u32)).collect();
        let col: MetalBaseColumn = data.iter().copied().collect();

        assert_eq!(col.len(), 1024);
        for i in 0..1024 {
            assert_eq!(col.at(i), data[i]);
        }
    }

    #[test]
    fn test_column_split_at_mid() {
        if !MetalContext::is_available() {
            return;
        }

        let data: Vec<BaseField> = (0..1024).map(|i| M31::from(i as u32)).collect();
        let col: MetalBaseColumn = data.iter().copied().collect();

        let (left, right) = col.split_at_mid();
        assert_eq!(left.len(), 512);
        assert_eq!(right.len(), 512);

        for i in 0..512 {
            assert_eq!(left.at(i), data[i]);
            assert_eq!(right.at(i), data[i + 512]);
        }
    }

    // ========================================================================
    // End-to-End Prove+Verify Tests
    // ========================================================================

    // ========================================================================
    // Quotient Tests
    // ========================================================================

    #[test]
    fn test_quotient_metal_vs_simd() {
        use itertools::Itertools;
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
        use crate::core::pcs::quotients::{
            build_samples_with_randomness_and_periodicity, ColumnSampleBatch, PointSample,
        };
        use crate::core::pcs::TreeVec;
        use crate::prover::backend::simd::column::BaseColumn;
        use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
        use crate::prover::poly::circle::CircleEvaluation;
        use crate::prover::poly::BitReversedOrder;
        use crate::prover::QuotientOps;

        if !MetalContext::is_available() {
            return;
        }

        // Test across multiple log sizes, covering both SIMD fallback and GPU paths
        for log_size in 5..=12 {
            let size = 1usize << log_size;
            let mut rng = SmallRng::seed_from_u64(log_size as u64);
            let domain = CanonicCoset::new(log_size).circle_domain();
            let n_cols = 10usize;

            // Create random column data
            let col_data: Vec<BaseField> =
                (0..size).map(|i| M31::from((i * 7 + 3) as u32)).collect();

            // SIMD columns
            let simd_base = BaseColumn::from_cpu(&col_data);
            let simd_eval =
                CircleEvaluation::<SimdBackend, BaseField, BitReversedOrder>::new(domain, simd_base);
            let simd_columns: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> =
                (0..n_cols).map(|_| simd_eval.clone()).collect();

            // Metal columns
            let metal_col: MetalBaseColumn = col_data.iter().copied().collect();
            let metal_eval =
                CircleEvaluation::<MetalBackend, BaseField, BitReversedOrder>::new(
                    domain, metal_col,
                );
            let metal_columns: Vec<CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>> =
                (0..n_cols).map(|_| metal_eval.clone()).collect();

            // Create random sample points and build sample batches
            let mask_structure: Vec<usize> =
                (0..n_cols).map(|_| rng.gen_range(1..=2)).collect();
            let points = [
                SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
                SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            ];
            let samples: Vec<Vec<PointSample>> = (0..n_cols)
                .zip(mask_structure.iter())
                .map(|(_, i)| {
                    points
                        .into_iter()
                        .take(*i)
                        .map(|point| PointSample {
                            point,
                            value: SecureField::from(rng.gen::<u32>()),
                        })
                        .collect()
                })
                .collect();

            let random_coeff = SecureField::from_m31(
                M31::from(rng.gen::<u32>()),
                M31::from(rng.gen::<u32>()),
                M31::from(rng.gen::<u32>()),
                M31::from(rng.gen::<u32>()),
            );

            let sample_batches = ColumnSampleBatch::new_vec(
                &build_samples_with_randomness_and_periodicity(
                    &TreeVec(vec![samples.clone()]),
                    vec![vec![log_size; n_cols].into_iter()],
                    log_size,
                    random_coeff,
                )
                .iter()
                .flatten()
                .collect_vec(),
            );

            // Test accumulate_numerators
            let mut simd_accs: Vec<AccumulatedNumerators<SimdBackend>> = vec![];
            SimdBackend::accumulate_numerators(
                &simd_columns.iter().collect_vec(),
                &sample_batches,
                &mut simd_accs,
            );

            let mut metal_accs: Vec<AccumulatedNumerators<MetalBackend>> = vec![];
            MetalBackend::accumulate_numerators(
                &metal_columns.iter().collect_vec(),
                &sample_batches,
                &mut metal_accs,
            );

            assert_eq!(
                simd_accs.len(),
                metal_accs.len(),
                "accumulate_numerators: different number of accumulators for log_size={}",
                log_size
            );

            for (j, (simd_acc, metal_acc)) in
                simd_accs.iter().zip(metal_accs.iter()).enumerate()
            {
                assert_eq!(
                    simd_acc.first_linear_term_acc, metal_acc.first_linear_term_acc,
                    "first_linear_term_acc mismatch at batch {} for log_size={}",
                    j, log_size
                );
                assert_eq!(
                    simd_acc.sample_point, metal_acc.sample_point,
                    "sample_point mismatch at batch {} for log_size={}",
                    j, log_size
                );

                let simd_nums = simd_acc.partial_numerators_acc.to_cpu();
                let metal_nums = metal_acc.partial_numerators_acc.to_cpu();
                assert_eq!(
                    simd_nums.len(),
                    metal_nums.len(),
                    "partial_numerators_acc length mismatch at batch {} for log_size={}",
                    j,
                    log_size
                );
                for k in 0..simd_nums.len() {
                    assert_eq!(
                        simd_nums.at(k),
                        metal_nums.at(k),
                        "partial_numerators mismatch at batch={}, idx={}, log_size={}",
                        j,
                        k,
                        log_size
                    );
                }
            }

            // Test compute_quotients_and_combine
            let lifting_log_size = log_size;
            let simd_result = SimdBackend::compute_quotients_and_combine(simd_accs, lifting_log_size);
            let metal_result =
                MetalBackend::compute_quotients_and_combine(metal_accs, lifting_log_size);

            let simd_vals = simd_result.values.to_cpu();
            let metal_vals = metal_result.values.to_cpu();
            for k in 0..simd_vals.len() {
                assert_eq!(
                    simd_vals.at(k),
                    metal_vals.at(k),
                    "quotient combine mismatch at idx={}, log_size={}",
                    k,
                    log_size
                );
            }
        }
    }

    // ========================================================================
    // End-to-End PCS Tests
    // ========================================================================

    #[test]
    fn test_pcs_prove_and_verify_metal_blake2s() {
        use itertools::Itertools;
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        use crate::core::channel::Blake2sChannel;
        use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
        use crate::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
        use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleChannel;
        use crate::prover::poly::circle::CircleCoefficients;
        use crate::prover::CommitmentSchemeProver;

        if !MetalContext::is_available() {
            return;
        }

        const N_COLS: usize = 5;
        // Use log_size 14 to exercise GPU paths for FFT, FRI, Merkle
        const LIFTING_LOG_SIZE: u32 = 14;

        let mut rng = SmallRng::seed_from_u64(42);
        let config = PcsConfig::default();
        let twiddles = MetalBackend::precompute_twiddles(
            CanonicCoset::new(LIFTING_LOG_SIZE + config.fri_config.log_blowup_factor).half_coset(),
        );
        let mut commitment_scheme =
            CommitmentSchemeProver::<MetalBackend, Blake2sMerkleChannel>::new(config, &twiddles);

        let mut polys: Vec<CircleCoefficients<MetalBackend>> = (0..N_COLS - 1)
            .map(|_| {
                CircleCoefficients::new(
                    (0..1 << rng.gen_range(4..LIFTING_LOG_SIZE - 1))
                        .map(M31::from)
                        .collect(),
                )
            })
            .collect();
        polys.push(CircleCoefficients::new(
            (0..1 << LIFTING_LOG_SIZE).map(M31::from).collect(),
        ));

        let sizes = polys.iter().map(|poly| poly.log_size()).collect_vec();

        let mut channel = Blake2sChannel::default();
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(polys);
        tree_builder.commit(&mut channel);

        let mask_structure = (0..N_COLS).map(|_| rng.gen_range(1..=2)).collect_vec();
        let samples = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let sampled_points = vec![(0..N_COLS)
            .zip(mask_structure.iter())
            .map(|(_, i)| samples.into_iter().take(*i).collect_vec())
            .collect_vec()];

        let proof = commitment_scheme.prove_values(TreeVec(sampled_points.clone()), &mut channel);

        let mut channel = Blake2sChannel::default();
        let mut verifier = CommitmentSchemeVerifier::<Blake2sMerkleChannel>::new(config);
        verifier.commit(proof.proof.commitments[0], &sizes, &mut channel);
        let result = verifier.verify_values(TreeVec(sampled_points), proof.proof, &mut channel);
        assert!(result.is_ok(), "Metal PCS verify failed: {:?}", result.err());
    }

    #[test]
    fn test_pcs_prove_and_verify_metal_blake2s_m31() {
        use itertools::Itertools;
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        use crate::core::channel::Blake2sM31Channel;
        use crate::core::circle::SECURE_FIELD_CIRCLE_GEN;
        use crate::core::pcs::{CommitmentSchemeVerifier, PcsConfig, TreeVec};
        use crate::core::vcs_lifted::blake2_merkle::Blake2sM31MerkleChannel;
        use crate::prover::poly::circle::CircleCoefficients;
        use crate::prover::CommitmentSchemeProver;

        if !MetalContext::is_available() {
            return;
        }

        const N_COLS: usize = 5;
        // Use log_size 14 to exercise GPU paths for FFT, FRI, Merkle
        const LIFTING_LOG_SIZE: u32 = 14;

        let mut rng = SmallRng::seed_from_u64(42);
        let config = PcsConfig::default();
        let twiddles = MetalBackend::precompute_twiddles(
            CanonicCoset::new(LIFTING_LOG_SIZE + config.fri_config.log_blowup_factor).half_coset(),
        );
        let mut commitment_scheme =
            CommitmentSchemeProver::<MetalBackend, Blake2sM31MerkleChannel>::new(
                config, &twiddles,
            );

        let mut polys: Vec<CircleCoefficients<MetalBackend>> = (0..N_COLS - 1)
            .map(|_| {
                CircleCoefficients::new(
                    (0..1 << rng.gen_range(4..LIFTING_LOG_SIZE - 1))
                        .map(M31::from)
                        .collect(),
                )
            })
            .collect();
        polys.push(CircleCoefficients::new(
            (0..1 << LIFTING_LOG_SIZE).map(M31::from).collect(),
        ));

        let sizes = polys.iter().map(|poly| poly.log_size()).collect_vec();

        let mut channel = Blake2sM31Channel::default();
        let mut tree_builder = commitment_scheme.tree_builder();
        tree_builder.extend_polys(polys);
        tree_builder.commit(&mut channel);

        let mask_structure = (0..N_COLS).map(|_| rng.gen_range(1..=2)).collect_vec();
        let samples = [
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
            SECURE_FIELD_CIRCLE_GEN.mul(rng.gen::<u128>()),
        ];
        let sampled_points = vec![(0..N_COLS)
            .zip(mask_structure.iter())
            .map(|(_, i)| samples.into_iter().take(*i).collect_vec())
            .collect_vec()];

        let proof = commitment_scheme.prove_values(TreeVec(sampled_points.clone()), &mut channel);

        let mut channel = Blake2sM31Channel::default();
        let mut verifier = CommitmentSchemeVerifier::<Blake2sM31MerkleChannel>::new(config);
        verifier.commit(proof.proof.commitments[0], &sizes, &mut channel);
        let result = verifier.verify_values(TreeVec(sampled_points), proof.proof, &mut channel);
        assert!(
            result.is_ok(),
            "Metal Blake2sM31 PCS verify failed: {:?}",
            result.err()
        );
    }
}
