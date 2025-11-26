//! Metal quotient operations.

use metal::MTLResourceOptions;

use super::context::MetalContext;
use super::thresholds::MIN_QUOTIENT_LOG_SIZE;
use super::MetalBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{column_line_coeffs, ColumnSampleBatch};
use crate::core::poly::circle::CircleDomain;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Column;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::QuotientOps;

impl QuotientOps for MetalBackend {
    fn accumulate_quotients(
        domain: CircleDomain,
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        random_coeff: SecureField,
        sample_batches: &[ColumnSampleBatch],
        _log_blowup_factor: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        let _timer = crate::metal_profile_fn!(
            "quotient",
            "GPU",
            log_size = domain.log_size(),
            num_columns = columns.len()
        );

        // Fall back to SIMD for small domains
        if domain.log_size() < MIN_QUOTIENT_LOG_SIZE || !MetalContext::is_available() {
            use crate::prover::backend::simd::column::BaseColumn;

            // Convert Metal columns to SIMD
            let simd_columns_owned: Vec<
                CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>,
            > = columns
                .iter()
                .map(|col| {
                    let cpu_vals = col.values.to_cpu();
                    let simd_col: BaseColumn = cpu_vals.into_iter().collect();
                    CircleEvaluation::new(col.domain, simd_col)
                })
                .collect();
            let simd_columns_refs: Vec<
                &CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>,
            > = simd_columns_owned.iter().collect();

            let simd_result = SimdBackend::accumulate_quotients(
                domain,
                &simd_columns_refs,
                random_coeff,
                sample_batches,
                _log_blowup_factor,
            );

            // Convert result back to Metal
            let metal_values = SecureColumnByCoords::from_simd(simd_result.values);
            return SecureEvaluation::new(simd_result.domain, metal_values);
        }
        let domain_size = domain.size();
        let num_columns = columns.len();

        // Precompute line coefficients (a, b, c) for each sample
        let line_coeffs = column_line_coeffs(sample_batches, random_coeff);

        // Get cached domain points (x, y) in bit-reversed order
        let ctx = MetalContext::global();
        let (domain_x_buffer, domain_y_buffer) = ctx.get_or_create_domain_xy_buffers(domain);

        // Get device and command queue for GPU operations
        let device = ctx.device();
        let command_queue = ctx.command_queue();

        // Prepare column buffer for flattening
        let columns_buffer_size = (num_columns * domain_size * std::mem::size_of::<u32>()) as u64;
        let columns_pooled = ctx.checkout_shared_buffer(columns_buffer_size);
        let columns_buffer = columns_pooled.buffer();

        // Flatten line coefficients and build metadata
        let mut line_coeffs_flat = Vec::new();
        let mut column_indices = Vec::new();
        let mut batch_sizes = Vec::new();

        for (_batch_idx, (batch, coeffs)) in sample_batches.iter().zip(&line_coeffs).enumerate() {
            batch_sizes.push(batch.columns_and_values.len() as u32);

            for (_col_offset, ((col_idx, _), (a, b, c))) in batch
                .columns_and_values
                .iter()
                .zip(coeffs.iter())
                .enumerate()
            {
                column_indices.push(*col_idx as u32);

                // Store (a, b, c) as QM31 values (4 components each)
                // QM31(CM31, CM31) and CM31(M31, M31) - both tuple structs
                // Access pattern: QM31.0 or .1 -> CM31.0 or .1 -> M31.0 -> u32
                line_coeffs_flat.push(a.0 .0 .0); // a.0.0 (first CM31's first M31)
                line_coeffs_flat.push(a.0 .1 .0); // a.0.1 (first CM31's second M31)
                line_coeffs_flat.push(a.1 .0 .0); // a.1.0 (second CM31's first M31)
                line_coeffs_flat.push(a.1 .1 .0); // a.1.1 (second CM31's second M31)

                line_coeffs_flat.push(b.0 .0 .0); // b.0.0
                line_coeffs_flat.push(b.0 .1 .0); // b.0.1
                line_coeffs_flat.push(b.1 .0 .0); // b.1.0
                line_coeffs_flat.push(b.1 .1 .0); // b.1.1

                line_coeffs_flat.push(c.0 .0 .0); // c.0.0
                line_coeffs_flat.push(c.0 .1 .0); // c.0.1
                line_coeffs_flat.push(c.1 .0 .0); // c.1.0
                line_coeffs_flat.push(c.1 .1 .0); // c.1.1
            }
        }

        // Extract sample points (x, y)
        let mut sample_points_x = Vec::new();
        let mut sample_points_y = Vec::new();
        for batch in sample_batches {
            // sample_batch.point is CirclePoint<SecureField>
            // SecureField = QM31(CM31, CM31), CM31(M31, M31)
            sample_points_x.push(batch.point.x.0 .0 .0); // x.0.0
            sample_points_x.push(batch.point.x.0 .1 .0); // x.0.1
            sample_points_x.push(batch.point.x.1 .0 .0); // x.1.0
            sample_points_x.push(batch.point.x.1 .1 .0); // x.1.1

            sample_points_y.push(batch.point.y.0 .0 .0); // y.0.0
            sample_points_y.push(batch.point.y.0 .1 .0); // y.0.1
            sample_points_y.push(batch.point.y.1 .0 .0); // y.1.0
            sample_points_y.push(batch.point.y.1 .1 .0); // y.1.1
        }

        // Dispatch to Metal GPU

        let column_indices_buffer = device.new_buffer_with_data(
            column_indices.as_ptr() as *const _,
            (column_indices.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let line_coeffs_buffer = device.new_buffer_with_data(
            line_coeffs_flat.as_ptr() as *const _,
            (line_coeffs_flat.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let sample_x_buffer = device.new_buffer_with_data(
            sample_points_x.as_ptr() as *const _,
            (sample_points_x.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let sample_y_buffer = device.new_buffer_with_data(
            sample_points_y.as_ptr() as *const _,
            (sample_points_y.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let batch_sizes_buffer = device.new_buffer_with_data(
            batch_sizes.as_ptr() as *const _,
            (batch_sizes.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Output buffer (4 u32s per QM31, domain_size elements) using buffer pool
        let output_size = (domain_size * 4 * std::mem::size_of::<u32>()) as u64;
        let output_pooled = ctx.checkout_shared_buffer(output_size);
        let output_buffer = output_pooled.buffer();

        // Fuse blit + quotient kernel into single command buffer
        let command_buffer = command_queue.new_command_buffer();

        // Step 1: Blit encoder to flatten columns on GPU
        let blit_encoder = command_buffer.new_blit_command_encoder();
        let mut offset = 0u64;
        let col_size = (domain_size * std::mem::size_of::<u32>()) as u64;
        for col in columns {
            blit_encoder.copy_from_buffer(
                col.values.buffer(),
                0,
                &columns_buffer,
                offset,
                col_size,
            );
            offset += col_size;
        }
        blit_encoder.end_encoding();

        // Step 2: Quotient kernel
        let encoder = command_buffer.new_compute_command_encoder();

        let pipeline = ctx.quotient_pipeline();
        encoder.set_compute_pipeline_state(pipeline);

        encoder.set_buffer(0, Some(&domain_x_buffer), 0);
        encoder.set_buffer(1, Some(&domain_y_buffer), 0);
        encoder.set_buffer(2, Some(&columns_buffer), 0);
        let num_columns_u32 = num_columns as u32;
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &num_columns_u32 as *const u32 as *const _,
        );
        let domain_size_u32 = domain_size as u32;
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &domain_size_u32 as *const u32 as *const _,
        );
        encoder.set_buffer(5, Some(&column_indices_buffer), 0);
        encoder.set_buffer(6, Some(&line_coeffs_buffer), 0);
        encoder.set_buffer(7, Some(&sample_x_buffer), 0);
        encoder.set_buffer(8, Some(&sample_y_buffer), 0);
        encoder.set_buffer(9, Some(&batch_sizes_buffer), 0);
        let num_batches = sample_batches.len() as u32;
        encoder.set_bytes(
            10,
            std::mem::size_of::<u32>() as u64,
            &num_batches as *const u32 as *const _,
        );
        encoder.set_buffer(11, Some(&output_buffer), 0);

        let threadgroup_size = 256u64.min(domain_size as u64);
        let num_threadgroups =
            ((domain_size as u64 + threadgroup_size - 1) / threadgroup_size).max(1);

        encoder.dispatch_thread_groups(
            metal::MTLSize::new(num_threadgroups, 1, 1),
            metal::MTLSize::new(threadgroup_size, 1, 1),
        );

        encoder.end_encoding();
        command_buffer.commit();

        let _gpu_timer = std::time::Instant::now();
        command_buffer.wait_until_completed();
        if std::env::var("METAL_PROFILE").is_ok() {
            eprintln!(
                "[CPU_PROFILE] quotient_gpu_wait | time={:.3}ms",
                _gpu_timer.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Convert output buffer to SecureColumnByCoords
        let values = unsafe {
            SecureColumnByCoords::from_qm31_interleaved_buffer(&output_buffer, domain_size)
        };

        // Keep pooled buffers alive until after GPU completes and data is read
        drop(columns_pooled);
        drop(output_pooled);

        SecureEvaluation::new(domain, values)
    }
}
