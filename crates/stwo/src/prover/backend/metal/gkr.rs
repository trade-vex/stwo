//! Metal GKR and MLE operations.

use metal::MTLResourceOptions;

use super::context::MetalContext;
use super::thresholds::MIN_MLE_LOG_SIZE;
use super::MetalBackend;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Column;
use crate::prover::lookups::gkr_prover::{GkrMultivariatePolyOracle, GkrOps, Layer};
use crate::prover::lookups::mle::{Mle, MleOps};
use crate::prover::lookups::utils::UnivariatePoly;

// MleOps implementations are required for GkrOps
impl MleOps<BaseField> for MetalBackend {
    fn fix_first_variable(
        mle: Mle<Self, BaseField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        let log_size = mle.len().ilog2();

        // Fall back to SIMD for small sizes
        if log_size < MIN_MLE_LOG_SIZE {
            use crate::prover::backend::simd::column::BaseColumn;

            // Convert Metal MLE to SIMD
            let cpu_vals = mle.into_evals().to_cpu();
            let simd_col: BaseColumn = cpu_vals.into_iter().collect();
            let simd_mle: Mle<SimdBackend, BaseField> = Mle::new(simd_col);

            let simd_result = SimdBackend::fix_first_variable(simd_mle, assignment);

            // Convert result back to Metal
            let result_cpu = simd_result.into_evals().to_cpu();
            let metal_col = result_cpu.into_iter().collect();
            return Mle::new(metal_col);
        }

        // Metal GPU path
        let ctx = MetalContext::global();
        let device = ctx.device();

        // Zero-copy access to input data (M31 values) via Deref
        let input_data: Vec<u32> = mle.as_slice().iter().map(|f| f.0).collect();
        let output_len = input_data.len() / 2;

        // Create Metal buffers
        let input_buffer = device.new_buffer_with_data(
            input_data.as_ptr() as *const _,
            (input_data.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Allocate output buffer directly (GPU will write all values, no need to zero-initialize on
        // CPU)
        let output_size = (output_len * 4 * std::mem::size_of::<u32>()) as u64;
        let output_pooled = ctx.checkout_shared_buffer(output_size);
        let output_buffer = output_pooled.buffer();

        // Create assignment buffer (QM31)
        let assignment_data = assignment.to_m31_array().map(|m| m.0);
        let assignment_buffer = device.new_buffer_with_data(
            assignment_data.as_ptr() as *const _,
            (4 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Dispatch kernel
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.mle_fold_m31_pipeline());
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&assignment_buffer), 0);
        encoder.set_bytes(
            3,
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

        // Read back results (QM31 format)
        let output_data = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const u32, output_len * 4)
        };

        // Convert QM31 buffer to Vec<SecureField>
        let result_values: Vec<SecureField> = (0..output_len)
            .map(|i| {
                let a = BaseField::from(output_data[i * 4]);
                let b = BaseField::from(output_data[i * 4 + 1]);
                let c = BaseField::from(output_data[i * 4 + 2]);
                let d = BaseField::from(output_data[i * 4 + 3]);
                SecureField::from_m31(a, b, c, d)
            })
            .collect();

        Mle::new(result_values.into_iter().collect())
    }
}

impl MleOps<SecureField> for MetalBackend {
    fn fix_first_variable(
        mle: Mle<Self, SecureField>,
        assignment: SecureField,
    ) -> Mle<Self, SecureField> {
        let log_size = mle.len().ilog2();

        // Fall back to SIMD for small sizes
        if log_size < MIN_MLE_LOG_SIZE {
            use crate::prover::backend::simd::column::SecureColumn;

            // Convert Metal MLE to SIMD
            let cpu_vals = mle.into_evals().to_cpu();
            let simd_col: SecureColumn = cpu_vals.into_iter().collect();
            let simd_mle: Mle<SimdBackend, SecureField> = Mle::new(simd_col);

            let simd_result = SimdBackend::fix_first_variable(simd_mle, assignment);

            // Convert result back to Metal
            let result_cpu = simd_result.into_evals().to_cpu();
            let metal_col = result_cpu.into_iter().collect();
            return Mle::new(metal_col);
        }

        // Metal GPU path
        let ctx = MetalContext::global();
        let device = ctx.device();

        // Zero-copy access to input data and convert to interleaved QM31 layout via Deref
        let input_values = mle.as_slice();
        let mut input_data = Vec::with_capacity(input_values.len() * 4);
        for val in input_values {
            let [a, b, c, d] = val.to_m31_array();
            input_data.push(a.0);
            input_data.push(b.0);
            input_data.push(c.0);
            input_data.push(d.0);
        }

        let output_len = input_values.len() / 2;

        // Create Metal buffers
        let input_buffer = device.new_buffer_with_data(
            input_data.as_ptr() as *const _,
            (input_data.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Allocate output buffer directly (GPU will write all values, no need to zero-initialize on
        // CPU)
        let output_size = (output_len * 4 * std::mem::size_of::<u32>()) as u64;
        let output_pooled = ctx.checkout_shared_buffer(output_size);
        let output_buffer = output_pooled.buffer();

        // Create assignment buffer (QM31)
        let assignment_data = assignment.to_m31_array().map(|m| m.0);
        let assignment_buffer = device.new_buffer_with_data(
            assignment_data.as_ptr() as *const _,
            (4 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Dispatch kernel
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.mle_fold_qm31_pipeline());
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_buffer(2, Some(&assignment_buffer), 0);
        encoder.set_bytes(
            3,
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

        // Read back results (QM31 format)
        let output_data = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const u32, output_len * 4)
        };

        // Convert QM31 buffer to Vec<SecureField>
        let result_values: Vec<SecureField> = (0..output_len)
            .map(|i| {
                let a = BaseField::from(output_data[i * 4]);
                let b = BaseField::from(output_data[i * 4 + 1]);
                let c = BaseField::from(output_data[i * 4 + 2]);
                let d = BaseField::from(output_data[i * 4 + 3]);
                SecureField::from_m31(a, b, c, d)
            })
            .collect();

        // Keep pooled buffer alive until after GPU completes and data is read
        drop(output_pooled);

        Mle::new(result_values.into_iter().collect())
    }
}

impl GkrOps for MetalBackend {
    fn gen_eq_evals(y: &[SecureField], v: SecureField) -> Mle<Self, SecureField> {
        let simd_result = SimdBackend::gen_eq_evals(y, v);

        // Convert result from SIMD to Metal
        let result_cpu = simd_result.into_evals().to_cpu();
        let metal_col = result_cpu.into_iter().collect();
        Mle::new(metal_col)
    }

    fn next_layer(layer: &Layer<Self>) -> Layer<Self> {
        use crate::prover::backend::cpu::CpuBackend;

        // Convert Metal layer to CPU, process, convert back
        let cpu_layer = layer.to_cpu();
        let cpu_result = CpuBackend::next_layer(&cpu_layer);

        // Convert result back to Metal using from_cpu pattern
        match cpu_result {
            Layer::GrandProduct(mle) => {
                let metal_col = mle.into_evals().into_iter().collect();
                Layer::GrandProduct(Mle::new(metal_col))
            }
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => {
                let metal_num = numerators.into_evals().into_iter().collect();
                let metal_den = denominators.into_evals().into_iter().collect();
                Layer::LogUpGeneric {
                    numerators: Mle::new(metal_num),
                    denominators: Mle::new(metal_den),
                }
            }
            Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => {
                let metal_num = numerators.into_evals().into_iter().collect();
                let metal_den = denominators.into_evals().into_iter().collect();
                Layer::LogUpMultiplicities {
                    numerators: Mle::new(metal_num),
                    denominators: Mle::new(metal_den),
                }
            }
            Layer::LogUpSingles { denominators } => {
                let metal_den = denominators.into_evals().into_iter().collect();
                Layer::LogUpSingles {
                    denominators: Mle::new(metal_den),
                }
            }
        }
    }

    fn sum_as_poly_in_first_variable(
        h: &GkrMultivariatePolyOracle<'_, Self>,
        claim: SecureField,
    ) -> UnivariatePoly<SecureField> {
        use crate::prover::backend::CpuBackend;

        // Convert to CPU backend and compute there.
        // The previous implementation used an unsafe transmute to SimdBackend which was UB
        // since Metal and SIMD column types have different layouts.
        let cpu_h = h.to_cpu();
        CpuBackend::sum_as_poly_in_first_variable(&cpu_h, claim)
    }
}
