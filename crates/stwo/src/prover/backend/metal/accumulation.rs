//! Metal accumulation operations.

use super::context::MetalContext;
use super::MetalBackend;
use crate::core::fields::qm31::SecureField;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::secure_column::SecureColumnByCoords;
use crate::prover::AccumulationOps;

impl AccumulationOps for MetalBackend {
    fn accumulate(column: &mut SecureColumnByCoords<Self>, other: &SecureColumnByCoords<Self>) {
        let len = column.len();

        // Use GPU for large columns (above threshold)
        const MIN_GPU_ACCUMULATE_SIZE: usize = 1024;

        if len >= MIN_GPU_ACCUMULATE_SIZE {
            let _timer = crate::metal_profile_fn!("accumulate", "GPU", size = len);

            let ctx = MetalContext::global();
            let command_buffer = ctx.command_queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(ctx.accumulate_m31_pipeline());

            // Accumulate each of the 4 coordinate columns separately
            for coord_idx in 0..4 {
                let dst_col = &column.columns[coord_idx];
                let src_col = &other.columns[coord_idx];

                encoder.set_buffer(0, Some(dst_col.buffer()), 0);
                encoder.set_buffer(1, Some(src_col.buffer()), 0);
                let count = len as u32;
                encoder.set_bytes(
                    2,
                    std::mem::size_of::<u32>() as u64,
                    &count as *const u32 as *const _,
                );

                let threads_per_grid = metal::MTLSize::new(len as u64, 1, 1);
                let threads_per_threadgroup = metal::MTLSize::new(256.min(len as u64), 1, 1);
                encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            }

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        } else {
            // Small columns: use SIMD fallback
            let simd_column: &mut SecureColumnByCoords<SimdBackend> =
                unsafe { &mut *(column as *mut _ as *mut _) };
            let simd_other: &SecureColumnByCoords<SimdBackend> =
                unsafe { &*(other as *const _ as *const _) };
            SimdBackend::accumulate(simd_column, simd_other)
        }
    }

    fn generate_secure_powers(felt: SecureField, n_powers: usize) -> Vec<SecureField> {
        SimdBackend::generate_secure_powers(felt, n_powers)
    }
}
