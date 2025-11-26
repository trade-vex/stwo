//! Metal proof-of-work grinding operations.

use bytemuck::cast_slice;
use metal::MTLResourceOptions;

use super::channel::MetalBlake2sChannelGeneric;
use super::context::MetalContext;
use super::MetalBackend;
use crate::core::channel::Blake2sChannelGeneric;
use crate::core::proof_of_work::GrindOps;
use crate::core::vcs::blake2_hash::Blake2sHasherGeneric;
use crate::prover::backend::simd::SimdBackend;

// Batch size per GPU thread (each thread tries this many sequential nonces)
const BATCH_SIZE: u32 = 1024;

// Number of GPU threads to launch per batch
const THREADS_PER_BATCH: u64 = 16384;

impl<const IS_M31_OUTPUT: bool> GrindOps<Blake2sChannelGeneric<IS_M31_OUTPUT>> for MetalBackend {
    fn grind(channel: &Blake2sChannelGeneric<IS_M31_OUTPUT>, pow_bits: u32) -> u64 {
        // For small pow_bits or when Metal is not available, use SIMD
        if pow_bits < 10 || !MetalContext::is_available() {
            return SimdBackend::grind(channel, pow_bits);
        }

        // TODO: support more than 32 bits
        assert!(pow_bits <= 32, "pow_bits > 32 is not supported");

        let digest = channel.digest();

        // Compute the prefix digest H(POW_PREFIX, [0_u8; 12], digest, n_bits)
        let mut hasher = Blake2sHasherGeneric::<IS_M31_OUTPUT>::default();
        hasher.update(&Blake2sChannelGeneric::<IS_M31_OUTPUT>::POW_PREFIX.to_le_bytes());
        hasher.update(&[0_u8; 12]);
        hasher.update(&digest.0[..]);
        hasher.update(&pow_bits.to_le_bytes());
        let prefixed_digest = hasher.finalize();
        let prefixed_digest_u32: &[u32] = cast_slice(&prefixed_digest.0[..]);

        // GPU grinding in batches
        let ctx = MetalContext::global();
        let device = ctx.device();

        // Create digest buffer
        let digest_buffer = device.new_buffer_with_data(
            prefixed_digest_u32.as_ptr() as *const _,
            (8 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Search in batches until we find a solution
        let mut batch_id: u64 = 0;
        loop {
            let start_nonce = batch_id * THREADS_PER_BATCH * (BATCH_SIZE as u64);

            // Create result buffer initialized to UINT64_MAX
            let result_init = u64::MAX;
            let result_buffer = device.new_buffer_with_data(
                &result_init as *const u64 as *const _,
                std::mem::size_of::<u64>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Dispatch kernel
            let command_buffer = ctx.command_queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(ctx.grind_pipeline());
            encoder.set_buffer(0, Some(&digest_buffer), 0);
            encoder.set_buffer(1, Some(&result_buffer), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &pow_bits as *const u32 as *const _,
            );
            encoder.set_bytes(
                3,
                std::mem::size_of::<u64>() as u64,
                &start_nonce as *const u64 as *const _,
            );
            encoder.set_bytes(
                4,
                std::mem::size_of::<u32>() as u64,
                &BATCH_SIZE as *const u32 as *const _,
            );
            encoder.set_bytes(
                5,
                std::mem::size_of::<bool>() as u64,
                &IS_M31_OUTPUT as *const bool as *const _,
            );

            let threadgroup_size = 256.min(THREADS_PER_BATCH);
            let threadgroups = (THREADS_PER_BATCH + threadgroup_size - 1) / threadgroup_size;

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

            // Check if we found a solution
            let result = unsafe { *(result_buffer.contents() as *const u64) };

            if result != u64::MAX {
                return result;
            }

            batch_id += 1;

            // Safety check: prevent infinite loop
            if batch_id > 1000 {
                panic!(
                    "Grinding failed to find solution after {} batches",
                    batch_id
                );
            }
        }
    }
}

// GrindOps implementation for GPU-resident channel
// Uses the same GPU grinding logic as CPU channel since digest format is compatible
impl<const IS_M31_OUTPUT: bool> GrindOps<MetalBlake2sChannelGeneric<IS_M31_OUTPUT>>
    for MetalBackend
{
    fn grind(channel: &MetalBlake2sChannelGeneric<IS_M31_OUTPUT>, pow_bits: u32) -> u64 {
        assert!(pow_bits <= 32, "pow_bits > 32 is not supported");

        let digest = channel.digest();

        // Compute the prefix digest H(POW_PREFIX, [0_u8; 12], digest, n_bits)
        let mut hasher = Blake2sHasherGeneric::<IS_M31_OUTPUT>::default();
        hasher.update(&Blake2sChannelGeneric::<IS_M31_OUTPUT>::POW_PREFIX.to_le_bytes());
        hasher.update(&[0_u8; 12]);
        hasher.update(&digest.0[..]);
        hasher.update(&pow_bits.to_le_bytes());
        let prefixed_digest = hasher.finalize();
        let prefixed_digest_u32: &[u32] = cast_slice(&prefixed_digest.0[..]);

        // GPU grinding in batches
        let ctx = MetalContext::global();
        let device = ctx.device();

        // Create digest buffer
        let digest_buffer = device.new_buffer_with_data(
            prefixed_digest_u32.as_ptr() as *const _,
            (8 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Search in batches until we find a solution
        let mut batch_id: u64 = 0;
        loop {
            let start_nonce = batch_id * THREADS_PER_BATCH * (BATCH_SIZE as u64);

            // Create result buffer initialized to UINT64_MAX
            let result_init = u64::MAX;
            let result_buffer = device.new_buffer_with_data(
                &result_init as *const u64 as *const _,
                std::mem::size_of::<u64>() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Dispatch kernel
            let command_buffer = ctx.command_queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(ctx.grind_pipeline());
            encoder.set_buffer(0, Some(&digest_buffer), 0);
            encoder.set_buffer(1, Some(&result_buffer), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &pow_bits as *const u32 as *const _,
            );
            encoder.set_bytes(
                3,
                std::mem::size_of::<u64>() as u64,
                &start_nonce as *const u64 as *const _,
            );
            encoder.set_bytes(
                4,
                std::mem::size_of::<u32>() as u64,
                &BATCH_SIZE as *const u32 as *const _,
            );
            encoder.set_bytes(
                5,
                std::mem::size_of::<bool>() as u64,
                &IS_M31_OUTPUT as *const bool as *const _,
            );

            let threadgroup_size = 256.min(THREADS_PER_BATCH);
            let threadgroups = (THREADS_PER_BATCH + threadgroup_size - 1) / threadgroup_size;

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

            // Check if we found a solution
            let result = unsafe { *(result_buffer.contents() as *const u64) };

            if result != u64::MAX {
                return result;
            }

            batch_id += 1;

            // Safety check: prevent infinite loop
            if batch_id > 1000 {
                panic!(
                    "Grinding failed to find solution after {} batches",
                    batch_id
                );
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub mod poseidon252 {
    use super::MetalBackend;
    use crate::core::channel::Poseidon252Channel;
    use crate::core::proof_of_work::GrindOps;
    use crate::prover::backend::simd::SimdBackend;

    impl GrindOps<Poseidon252Channel> for MetalBackend {
        fn grind(channel: &Poseidon252Channel, pow_bits: u32) -> u64 {
            SimdBackend::grind(channel, pow_bits)
        }
    }
}
