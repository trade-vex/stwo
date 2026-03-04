//! Metal quotient operations.
//!
//! GPU-accelerated for domain sizes >= MIN_QUOTIENT_LOG_SIZE,
//! with SIMD fallback for smaller sizes.

use std::iter::zip;

use metal::MTLResourceOptions;

use super::column::MetalBaseColumn;
use super::context::MetalContext;
use super::thresholds::MIN_QUOTIENT_LOG_SIZE;
use super::MetalBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::quotients::{quotient_constants, ColumnSampleBatch, NumeratorData};
use crate::core::poly::circle::CanonicCoset;
use crate::prover::backend::simd::column::BaseColumn;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::pcs::quotient_ops::AccumulatedNumerators;
use crate::prover::poly::circle::{CircleEvaluation, SecureEvaluation};
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::QuotientOps;

impl QuotientOps for MetalBackend {
    fn accumulate_numerators(
        columns: &[&CircleEvaluation<Self, BaseField, BitReversedOrder>],
        sample_batches: &[ColumnSampleBatch],
        accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<Self>>,
    ) {
        let size = columns[0].domain.size();
        let log_size = columns[0].domain.log_size();

        if log_size < MIN_QUOTIENT_LOG_SIZE {
            return simd_accumulate_numerators(columns, sample_batches, accumulated_numerators_vec);
        }

        // GPU path
        let ctx = MetalContext::global();
        let device = ctx.device();
        let domain_size = size as u32;
        let quotient_consts = quotient_constants(sample_batches);

        // Pack all columns into one contiguous GPU buffer.
        // All columns have the same domain size within a single call.
        let num_columns = columns.len();
        let total_elements = num_columns * size;
        let mut packed_data: Vec<u32> = Vec::with_capacity(total_elements);
        for col in columns {
            let slice = col.values.as_slice();
            let u32_slice = unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const u32, slice.len())
            };
            packed_data.extend_from_slice(u32_slice);
        }
        let columns_buffer = device.new_buffer_with_data(
            packed_data.as_ptr() as *const _,
            (total_elements * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        for (batch, coeffs) in zip(sample_batches, quotient_consts.line_coeffs) {
            let batch_size = batch.cols_vals_randpows.len() as u32;

            // Build col_indices buffer from NumeratorData
            let col_indices: Vec<u32> = batch
                .cols_vals_randpows
                .iter()
                .map(|NumeratorData { column_index, .. }| *column_index as u32)
                .collect();
            let col_indices_buffer = device.new_buffer_with_data(
                col_indices.as_ptr() as *const _,
                (col_indices.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Build bc_coeffs buffer: flattened (b, c) pairs as QM31
            // Each QM31 = 4 u32s, so each pair = 8 u32s
            let mut bc_data: Vec<u32> = Vec::with_capacity(coeffs.len() * 8);
            for (_, b, c) in &coeffs {
                let b_arr = b.to_m31_array();
                bc_data.push(b_arr[0].0);
                bc_data.push(b_arr[1].0);
                bc_data.push(b_arr[2].0);
                bc_data.push(b_arr[3].0);
                let c_arr = c.to_m31_array();
                bc_data.push(c_arr[0].0);
                bc_data.push(c_arr[1].0);
                bc_data.push(c_arr[2].0);
                bc_data.push(c_arr[3].0);
            }
            let bc_coeffs_buffer = device.new_buffer_with_data(
                bc_data.as_ptr() as *const _,
                (bc_data.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Allocate 4 output buffers (SecureColumnByCoords layout)
            let out_size = (size * 4) as u64;
            let out0 = device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);
            let out1 = device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);
            let out2 = device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);
            let out3 = device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            // Dispatch GPU kernel
            let command_buffer = ctx.command_queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(ctx.quotient_partial_numerator_pipeline());

            encoder.set_buffer(0, Some(&columns_buffer), 0);
            encoder.set_buffer(1, Some(&col_indices_buffer), 0);
            encoder.set_buffer(2, Some(&bc_coeffs_buffer), 0);
            encoder.set_bytes(3, 4, &domain_size as *const u32 as *const _);
            encoder.set_bytes(4, 4, &batch_size as *const u32 as *const _);
            encoder.set_buffer(5, Some(&out0), 0);
            encoder.set_buffer(6, Some(&out1), 0);
            encoder.set_buffer(7, Some(&out2), 0);
            encoder.set_buffer(8, Some(&out3), 0);

            let thread_count = metal::MTLSize::new(size as u64, 1, 1);
            let threadgroup_size = metal::MTLSize::new(
                ctx.quotient_partial_numerator_pipeline()
                    .max_total_threads_per_threadgroup()
                    .min(size as u64),
                1,
                1,
            );
            encoder.dispatch_threads(thread_count, threadgroup_size);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Wrap output buffers as SecureColumnByCoords<MetalBackend>
            let partial_numerators_acc = SecureColumnByCoords {
                columns: [
                    MetalBaseColumn::from_buffer(out0, size),
                    MetalBaseColumn::from_buffer(out1, size),
                    MetalBaseColumn::from_buffer(out2, size),
                    MetalBaseColumn::from_buffer(out3, size),
                ],
            };

            let first_linear_term_acc: SecureField = coeffs.iter().map(|(a, ..)| a).sum();
            accumulated_numerators_vec.push(AccumulatedNumerators {
                sample_point: batch.point,
                partial_numerators_acc,
                first_linear_term_acc,
            });
        }
    }

    fn compute_quotients_and_combine(
        accs: Vec<AccumulatedNumerators<Self>>,
        lifting_log_size: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        if lifting_log_size < MIN_QUOTIENT_LOG_SIZE {
            return simd_compute_quotients_and_combine(accs, lifting_log_size);
        }

        let ctx = MetalContext::global();
        let device = ctx.device();
        let domain = CanonicCoset::new(lifting_log_size).circle_domain();
        let lifting_size = 1u32 << lifting_log_size;
        let num_accumulations = accs.len() as u32;

        // Get cached domain point buffers
        let (domain_x_buf, domain_y_buf) = ctx.get_or_create_domain_xy_buffers(domain);

        // Pack all accumulator partial numerator coords into one buffer.
        // Layout: for each accumulation, 4 coord arrays of acc_size elements each.
        let mut acc_offsets: Vec<u32> = Vec::with_capacity(accs.len());
        let mut acc_log_sizes: Vec<u32> = Vec::with_capacity(accs.len());
        let mut total_elements = 0usize;
        for acc in &accs {
            acc_offsets.push(total_elements as u32);
            let acc_size = acc.partial_numerators_acc.len();
            acc_log_sizes.push(acc_size.ilog2());
            total_elements += acc_size * 4; // 4 coords per element
        }

        let mut packed_acc_data: Vec<u32> = Vec::with_capacity(total_elements);
        for acc in &accs {
            // Pack 4 coord columns sequentially: [coord0, coord1, coord2, coord3]
            for coord_idx in 0..4 {
                let col_slice = acc.partial_numerators_acc.columns[coord_idx].as_slice();
                let u32_slice = unsafe {
                    std::slice::from_raw_parts(col_slice.as_ptr() as *const u32, col_slice.len())
                };
                packed_acc_data.extend_from_slice(u32_slice);
            }
        }
        let acc_data_buffer = device.new_buffer_with_data(
            packed_acc_data.as_ptr() as *const _,
            (packed_acc_data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let acc_offsets_buffer = device.new_buffer_with_data(
            acc_offsets.as_ptr() as *const _,
            (acc_offsets.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let acc_log_sizes_buffer = device.new_buffer_with_data(
            acc_log_sizes.as_ptr() as *const _,
            (acc_log_sizes.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Build first_linear_terms buffer (QM31 per accumulation)
        let mut linear_terms_data: Vec<u32> = Vec::with_capacity(accs.len() * 4);
        for acc in &accs {
            let arr = acc.first_linear_term_acc.to_m31_array();
            linear_terms_data.push(arr[0].0);
            linear_terms_data.push(arr[1].0);
            linear_terms_data.push(arr[2].0);
            linear_terms_data.push(arr[3].0);
        }
        let linear_terms_buffer = device.new_buffer_with_data(
            linear_terms_data.as_ptr() as *const _,
            (linear_terms_data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Build sample_points_x/y buffers (QM31 per accumulation)
        let mut sample_x_data: Vec<u32> = Vec::with_capacity(accs.len() * 4);
        let mut sample_y_data: Vec<u32> = Vec::with_capacity(accs.len() * 4);
        for acc in &accs {
            let x_arr = acc.sample_point.x.to_m31_array();
            sample_x_data.push(x_arr[0].0);
            sample_x_data.push(x_arr[1].0);
            sample_x_data.push(x_arr[2].0);
            sample_x_data.push(x_arr[3].0);
            let y_arr = acc.sample_point.y.to_m31_array();
            sample_y_data.push(y_arr[0].0);
            sample_y_data.push(y_arr[1].0);
            sample_y_data.push(y_arr[2].0);
            sample_y_data.push(y_arr[3].0);
        }
        let sample_x_buffer = device.new_buffer_with_data(
            sample_x_data.as_ptr() as *const _,
            (sample_x_data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let sample_y_buffer = device.new_buffer_with_data(
            sample_y_data.as_ptr() as *const _,
            (sample_y_data.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Allocate 4 output buffers
        let out_byte_size = (lifting_size as u64) * 4;
        let out0 = device.new_buffer(out_byte_size, MTLResourceOptions::StorageModeShared);
        let out1 = device.new_buffer(out_byte_size, MTLResourceOptions::StorageModeShared);
        let out2 = device.new_buffer(out_byte_size, MTLResourceOptions::StorageModeShared);
        let out3 = device.new_buffer(out_byte_size, MTLResourceOptions::StorageModeShared);

        // Dispatch GPU kernel
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.quotient_combine_pipeline());

        encoder.set_buffer(0, Some(&domain_x_buf), 0);
        encoder.set_buffer(1, Some(&domain_y_buf), 0);
        encoder.set_buffer(2, Some(&acc_data_buffer), 0);
        encoder.set_buffer(3, Some(&acc_offsets_buffer), 0);
        encoder.set_buffer(4, Some(&acc_log_sizes_buffer), 0);
        encoder.set_buffer(5, Some(&linear_terms_buffer), 0);
        encoder.set_buffer(6, Some(&sample_x_buffer), 0);
        encoder.set_buffer(7, Some(&sample_y_buffer), 0);
        encoder.set_bytes(8, 4, &lifting_log_size as *const u32 as *const _);
        encoder.set_bytes(9, 4, &num_accumulations as *const u32 as *const _);
        encoder.set_buffer(10, Some(&out0), 0);
        encoder.set_buffer(11, Some(&out1), 0);
        encoder.set_buffer(12, Some(&out2), 0);
        encoder.set_buffer(13, Some(&out3), 0);

        let thread_count = metal::MTLSize::new(lifting_size as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(
            ctx.quotient_combine_pipeline()
                .max_total_threads_per_threadgroup()
                .min(lifting_size as u64),
            1,
            1,
        );
        encoder.dispatch_threads(thread_count, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let quotients = SecureColumnByCoords {
            columns: [
                MetalBaseColumn::from_buffer(out0, lifting_size as usize),
                MetalBaseColumn::from_buffer(out1, lifting_size as usize),
                MetalBaseColumn::from_buffer(out2, lifting_size as usize),
                MetalBaseColumn::from_buffer(out3, lifting_size as usize),
            ],
        };
        SecureEvaluation::new(domain, quotients)
    }
}

/// SIMD fallback for accumulate_numerators (small domain sizes).
fn simd_accumulate_numerators(
    columns: &[&CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>],
    sample_batches: &[ColumnSampleBatch],
    accumulated_numerators_vec: &mut Vec<AccumulatedNumerators<MetalBackend>>,
) {
    let simd_columns_owned: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> =
        columns
            .iter()
            .map(|col| {
                let simd_col = BaseColumn::from_cpu(col.values.as_slice());
                CircleEvaluation::new(col.domain, simd_col)
            })
            .collect();
    let simd_columns_refs: Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> =
        simd_columns_owned.iter().collect();

    let mut simd_accs: Vec<AccumulatedNumerators<SimdBackend>> = Vec::new();
    SimdBackend::accumulate_numerators(&simd_columns_refs, sample_batches, &mut simd_accs);

    for acc in simd_accs {
        accumulated_numerators_vec.push(AccumulatedNumerators {
            sample_point: acc.sample_point,
            partial_numerators_acc: SecureColumnByCoords::from_simd(acc.partial_numerators_acc),
            first_linear_term_acc: acc.first_linear_term_acc,
        });
    }
}

/// SIMD fallback for compute_quotients_and_combine (small domain sizes).
fn simd_compute_quotients_and_combine(
    accs: Vec<AccumulatedNumerators<MetalBackend>>,
    lifting_log_size: u32,
) -> SecureEvaluation<MetalBackend, BitReversedOrder> {
    let simd_accs: Vec<AccumulatedNumerators<SimdBackend>> = accs
        .into_iter()
        .map(|acc| {
            let simd_col = SecureColumnByCoords::<SimdBackend> {
                columns: acc
                    .partial_numerators_acc
                    .columns
                    .map(|col| BaseColumn::from_cpu(col.as_slice())),
            };
            AccumulatedNumerators {
                sample_point: acc.sample_point,
                partial_numerators_acc: simd_col,
                first_linear_term_acc: acc.first_linear_term_acc,
            }
        })
        .collect();

    let simd_result = SimdBackend::compute_quotients_and_combine(simd_accs, lifting_log_size);

    let metal_values = SecureColumnByCoords::from_simd(simd_result.values);
    SecureEvaluation::new(simd_result.domain, metal_values)
}
