//! Metal FRI operations.

use metal::MTLResourceOptions;

use super::context::MetalContext;
use super::thresholds::MIN_FRI_LOG_SIZE;
use super::MetalBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::utils::domain_line_twiddles_from_tree;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::fri::FriOps;
use crate::prover::line::LineEvaluation;
use crate::prover::poly::circle::SecureEvaluation;
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;

impl FriOps for MetalBackend {
    fn fold_line(
        eval: &LineEvaluation<Self>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
        fold_step: u32,
    ) -> LineEvaluation<Self> {
        let log_size = eval.len().ilog2();
        let _timer = crate::metal_profile_fn!("fri_fold_line", "GPU", log_size = log_size);

        // Fall back to SIMD for small sizes or multi-step folding
        if log_size < MIN_FRI_LOG_SIZE || fold_step > 1 {
            use crate::prover::backend::simd::column::BaseColumn;
            use crate::prover::backend::Column;
            use crate::prover::secure_column::SecureColumnByCoords;

            // Convert Metal eval to SIMD
            let simd_columns = eval.values.columns.clone().map(|col| {
                let cpu_vals = col.to_cpu();
                let simd_col: BaseColumn = cpu_vals.into_iter().collect();
                simd_col
            });
            let simd_values = SecureColumnByCoords {
                columns: simd_columns,
            };
            let simd_eval = LineEvaluation::new(eval.domain(), simd_values);

            let simd_twiddles: &TwiddleTree<SimdBackend> =
                unsafe { &*(twiddles as *const _ as *const _) };
            let simd_result = SimdBackend::fold_line(&simd_eval, alpha, simd_twiddles, fold_step);

            // Convert result back to Metal
            let domain = simd_result.domain();
            let metal_values = SecureColumnByCoords::from_simd(simd_result.values);
            return LineEvaluation::new(domain, metal_values);
        }

        let ctx = MetalContext::global();
        let device = ctx.device();

        let domain = eval.domain();
        let itwiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles)[0];

        let output_len = 1 << (log_size - 1);

        // Allocate output coordinate buffers
        let coord_size = (output_len * std::mem::size_of::<u32>()) as u64;
        let out_cols = [
            device.new_buffer(coord_size, MTLResourceOptions::StorageModeShared),
            device.new_buffer(coord_size, MTLResourceOptions::StorageModeShared),
            device.new_buffer(coord_size, MTLResourceOptions::StorageModeShared),
            device.new_buffer(coord_size, MTLResourceOptions::StorageModeShared),
        ];

        // Twiddles and alpha
        let twiddle_buffer = ctx.get_or_create_twiddle_buffer(itwiddles);
        let alpha_data = alpha.to_m31_array().map(|m| m.0);
        let alpha_buffer = device.new_buffer_with_data(
            alpha_data.as_ptr() as *const _,
            (4 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Fused fold kernel (coords -> coords, no intermediate QM31 buffer)
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(ctx.fri_fold_line_coords_pipeline());
        encoder.set_buffer(0, Some(eval.values.columns[0].buffer()), 0);
        encoder.set_buffer(1, Some(eval.values.columns[1].buffer()), 0);
        encoder.set_buffer(2, Some(eval.values.columns[2].buffer()), 0);
        encoder.set_buffer(3, Some(eval.values.columns[3].buffer()), 0);
        encoder.set_buffer(4, Some(&out_cols[0]), 0);
        encoder.set_buffer(5, Some(&out_cols[1]), 0);
        encoder.set_buffer(6, Some(&out_cols[2]), 0);
        encoder.set_buffer(7, Some(&out_cols[3]), 0);
        encoder.set_buffer(8, Some(&twiddle_buffer), 0);
        encoder.set_buffer(9, Some(&alpha_buffer), 0);
        encoder.set_bytes(
            10,
            std::mem::size_of::<u32>() as u64,
            &log_size as *const u32 as *const _,
        );

        let num_threads = output_len as u64;
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: threadgroups,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: threadgroup_size,
                height: 1,
                depth: 1,
            },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Convert output buffers to Metal columns
        use crate::prover::backend::metal::column::MetalBaseColumn;
        let folded_values = SecureColumnByCoords {
            columns: out_cols.map(|buf| MetalBaseColumn::from_buffer(buf, output_len)),
        };

        LineEvaluation::new(domain.double(), folded_values)
    }

    fn fold_circle_into_line(
        dst: &mut LineEvaluation<Self>,
        src: &SecureEvaluation<Self, BitReversedOrder>,
        alpha: SecureField,
        twiddles: &TwiddleTree<Self>,
    ) {
        let log_size = src.len().ilog2();
        let _timer = crate::metal_profile_fn!("fri_fold_circle", "GPU", log_size = log_size);

        // Fall back to SIMD for small sizes
        if log_size < MIN_FRI_LOG_SIZE {
            use crate::prover::backend::simd::column::BaseColumn;
            use crate::prover::backend::Column;

            // Convert dst and src from Metal to SIMD
            let dst_domain = dst.domain();
            let simd_dst_columns = dst.values.columns.clone().map(|col| {
                let cpu_vals = col.to_cpu();
                let simd_col: BaseColumn = cpu_vals.into_iter().collect();
                simd_col
            });
            let simd_dst_values = SecureColumnByCoords {
                columns: simd_dst_columns,
            };
            let mut simd_dst = LineEvaluation::new(dst_domain, simd_dst_values);

            let simd_src_columns = src.values.columns.clone().map(|col| {
                let cpu_vals = col.to_cpu();
                let simd_col: BaseColumn = cpu_vals.into_iter().collect();
                simd_col
            });
            let simd_src_values = SecureColumnByCoords {
                columns: simd_src_columns,
            };
            let simd_src = SecureEvaluation::new(src.domain, simd_src_values);

            let simd_twiddles: &TwiddleTree<SimdBackend> =
                unsafe { &*(twiddles as *const _ as *const _) };

            SimdBackend::fold_circle_into_line(&mut simd_dst, &simd_src, alpha, simd_twiddles);

            // Write result back to dst
            dst.values = SecureColumnByCoords::from_simd(simd_dst.values);
            return;
        }

        // Metal GPU path
        let ctx = MetalContext::global();
        let device = ctx.device();

        let domain = src.domain;
        let alpha_sq = alpha * alpha;
        let itwiddles = domain_line_twiddles_from_tree(domain, &twiddles.itwiddles)[0];

        let output_len = 1 << (log_size - 1);

        // Twiddles and alpha
        let twiddle_buffer = ctx.get_or_create_twiddle_buffer(itwiddles);
        let alpha_data = alpha.to_m31_array().map(|m| m.0);
        let alpha_buffer = device.new_buffer_with_data(
            alpha_data.as_ptr() as *const _,
            (4 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let alpha_sq_data = alpha_sq.to_m31_array().map(|m| m.0);
        let alpha_sq_buffer = device.new_buffer_with_data(
            alpha_sq_data.as_ptr() as *const _,
            (4 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Fused fold kernel (coords → coords, no intermediate QM31 buffers)
        // Reads from src and dst (for accumulation), writes to dst (in-place)
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(ctx.fri_fold_circle_into_line_coords_pipeline());
        encoder.set_buffer(0, Some(src.values.columns[0].buffer()), 0);
        encoder.set_buffer(1, Some(src.values.columns[1].buffer()), 0);
        encoder.set_buffer(2, Some(src.values.columns[2].buffer()), 0);
        encoder.set_buffer(3, Some(src.values.columns[3].buffer()), 0);
        encoder.set_buffer(4, Some(dst.values.columns[0].buffer()), 0);
        encoder.set_buffer(5, Some(dst.values.columns[1].buffer()), 0);
        encoder.set_buffer(6, Some(dst.values.columns[2].buffer()), 0);
        encoder.set_buffer(7, Some(dst.values.columns[3].buffer()), 0);
        encoder.set_buffer(8, Some(&twiddle_buffer), 0);
        encoder.set_buffer(9, Some(&alpha_buffer), 0);
        encoder.set_buffer(10, Some(&alpha_sq_buffer), 0);
        encoder.set_bytes(
            11,
            std::mem::size_of::<u32>() as u64,
            &log_size as *const u32 as *const _,
        );

        let num_threads = output_len as u64;
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: threadgroups,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: threadgroup_size,
                height: 1,
                depth: 1,
            },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    fn decompose(
        eval: &SecureEvaluation<Self, BitReversedOrder>,
    ) -> (SecureEvaluation<Self, BitReversedOrder>, SecureField) {
        let domain_size = eval.len();
        let _timer = crate::metal_profile_fn!("fri_decompose", "GPU", domain_size = domain_size);
        let half_size = domain_size / 2;

        let lambda = {
            let ctx = MetalContext::global();
            let device = ctx.device();

            // Pack input to QM31 for GPU
            let input_qm31_size = (domain_size * 4 * std::mem::size_of::<u32>()) as u64;
            let input_pooled = ctx.checkout_shared_buffer(input_qm31_size);
            let input_buffer = input_pooled.buffer();

            // Allocate buffer for partial sums (2 QM31 per threadgroup)
            let num_threadgroups = 256usize; // Use 256 threadgroups
            let partial_sums_size = (num_threadgroups * 2 * 4 * std::mem::size_of::<u32>()) as u64;
            let partial_sums_buffer =
                device.new_buffer(partial_sums_size, MTLResourceOptions::StorageModeShared);

            let command_buffer = ctx.command_queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // Pack coordinates to QM31
            ctx.pack_coords_to_qm31_batched(
                &encoder,
                &eval.values.columns[0],
                &eval.values.columns[1],
                &eval.values.columns[2],
                &eval.values.columns[3],
                &input_buffer,
            );

            // Dispatch parallel reduction kernel
            encoder.set_compute_pipeline_state(ctx.fri_decompose_sum_pipeline());
            encoder.set_buffer(0, Some(&input_buffer), 0);
            encoder.set_buffer(1, Some(&partial_sums_buffer), 0);
            let half_size_u32 = half_size as u32;
            encoder.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &half_size_u32 as *const u32 as *const _,
            );
            let grid_size = num_threadgroups * 256;
            let grid_size_u32 = grid_size as u32;
            encoder.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &grid_size_u32 as *const u32 as *const _,
            );

            encoder.dispatch_thread_groups(
                metal::MTLSize {
                    width: num_threadgroups as u64,
                    height: 1,
                    depth: 1,
                },
                metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Sum partial results on CPU
            let partial_sums_ptr = partial_sums_buffer.contents() as *const [u32; 4];
            let mut a_sum = SecureField::from(BaseField::from(0));
            let mut b_sum = SecureField::from(BaseField::from(0));
            for i in 0..num_threadgroups {
                let a_qm31 = unsafe { &*partial_sums_ptr.offset((i * 2) as isize) };
                let b_qm31 = unsafe { &*partial_sums_ptr.offset((i * 2 + 1) as isize) };
                a_sum += SecureField::from_m31_array(std::array::from_fn(|j| {
                    BaseField::from(a_qm31[j])
                }));
                b_sum += SecureField::from_m31_array(std::array::from_fn(|j| {
                    BaseField::from(b_qm31[j])
                }));
            }

            drop(input_pooled);

            (b_sum - a_sum) / SecureField::from(BaseField::from(2 * domain_size as u32))
        };

        let ctx = MetalContext::global();
        let device = ctx.device();

        // Pack input to QM31
        let input_qm31_size = (domain_size * 4 * std::mem::size_of::<u32>()) as u64;
        let input_pooled = ctx.checkout_shared_buffer(input_qm31_size);
        let input_buffer = input_pooled.buffer();

        // Pack output buffer
        let output_pooled = ctx.checkout_shared_buffer(input_qm31_size);
        let output_buffer = output_pooled.buffer();

        // Lambda buffer
        let lambda_data = lambda.to_m31_array().map(|m| m.0);
        let lambda_buffer = device.new_buffer_with_data(
            lambda_data.as_ptr() as *const _,
            (4 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Unpack output to coordinate buffers
        let coord_size = (domain_size * std::mem::size_of::<u32>()) as u64;
        let out_cols = [
            device.new_buffer(coord_size, MTLResourceOptions::StorageModeShared),
            device.new_buffer(coord_size, MTLResourceOptions::StorageModeShared),
            device.new_buffer(coord_size, MTLResourceOptions::StorageModeShared),
            device.new_buffer(coord_size, MTLResourceOptions::StorageModeShared),
        ];

        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Pack input coordinates to QM31
        ctx.pack_coords_to_qm31_batched(
            &encoder,
            &eval.values.columns[0],
            &eval.values.columns[1],
            &eval.values.columns[2],
            &eval.values.columns[3],
            &input_buffer,
        );

        // FRI decompose kernel: output[i] = input[i] ± lambda
        encoder.set_compute_pipeline_state(ctx.fri_decompose_pipeline());
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&lambda_buffer), 0);
        let half_size_u32 = half_size as u32;
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &half_size_u32 as *const u32 as *const _,
        );

        let num_threads = domain_size as u64;
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize {
                width: threadgroups,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: threadgroup_size,
                height: 1,
                depth: 1,
            },
        );

        // Unpack QM31 output to coordinates
        ctx.unpack_qm31_to_coords_batched(&encoder, &output_buffer, &out_cols, domain_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        use crate::prover::backend::metal::column::MetalBaseColumn;
        let g_values = SecureColumnByCoords {
            columns: out_cols.map(|buf| MetalBaseColumn::from_buffer(buf, domain_size)),
        };

        // Keep pooled buffers alive until after GPU completes
        drop(input_pooled);
        drop(output_pooled);

        (SecureEvaluation::new(eval.domain, g_values), lambda)
    }
}
