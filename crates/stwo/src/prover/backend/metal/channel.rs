//! GPU-resident Blake2s channel for Metal backend.
//!
//! This module provides MetalBlake2sChannel which keeps the Fiat-Shamir
//! channel state (digest) on the GPU.
//!

use metal::{Buffer, MTLResourceOptions};
use std::sync::Arc;
use std::iter;

use crate::core::channel::{Blake2sChannelGeneric, Channel};
use crate::core::fields::m31::{BaseField, P};
use crate::core::fields::qm31::{SecureField, SECURE_EXTENSION_DEGREE};
use crate::core::vcs::blake2_hash::{Blake2sHash, Blake2sHasherGeneric};

use super::context::MetalContext;

/// Constants from blake2s module
const BLAKE_BYTES_PER_HASH: usize = 32;
const FELTS_PER_HASH: usize = 8;

/// GPU-resident Blake2s channel with Metal compute kernels.
pub struct MetalBlake2sChannelGeneric<const IS_M31_OUTPUT: bool> {
    /// Metal context for kernel dispatch.
    context: Arc<MetalContext>,

    /// GPU-resident digest buffer (8 u32s = 32 bytes).
    /// Uses StorageModeShared for zero-copy CPU reads.
    digest_buffer: Buffer,

    /// Draw counter (CPU-side, passed to kernels).
    n_draws: u32,
}

impl<const IS_M31_OUTPUT: bool> MetalBlake2sChannelGeneric<IS_M31_OUTPUT> {
    /// Create a new GPU channel with default (zero) digest.
    pub fn new() -> Self {
        let context = MetalContext::global();

        // Create GPU digest buffer (8 u32s = 32 bytes)
        let digest_buffer = context.device().new_buffer(
            32,
            MTLResourceOptions::StorageModeShared,
        );

        // Initialize digest to zero (matches CPU channel default)
        {
            let ptr = digest_buffer.contents() as *mut u32;
            unsafe {
                for i in 0..8 {
                    ptr.add(i).write(0);
                }
            }
        }

        Self {
            context,
            digest_buffer,
            n_draws: 0,
        }
    }

    /// Read current digest from GPU (zero-copy on Apple Silicon).
    pub fn digest(&self) -> Blake2sHash {
        let ptr = self.digest_buffer.contents() as *const u32;
        let mut hash_bytes = [0u8; 32];
        unsafe {
            for i in 0..8 {
                let u32_val = ptr.add(i).read();
                let bytes = u32_val.to_le_bytes();
                hash_bytes[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
            }
        }
        Blake2sHash(hash_bytes)
    }

    /// Update digest buffer on GPU.
    pub fn update_digest(&mut self, new_digest: Blake2sHash) {
        let ptr = self.digest_buffer.contents() as *mut u32;
        unsafe {
            for i in 0..8 {
                let bytes: [u8; 4] = new_digest.0[i * 4..(i + 1) * 4].try_into().unwrap();
                ptr.add(i).write(u32::from_le_bytes(bytes));
            }
        }
        self.n_draws = 0;
    }

    /// Mix data into channel state using GPU kernel.
    pub fn gpu_mix_root(&mut self, root: &Blake2sHash) {
        let _timer = if super::profiling::is_profiling_enabled() {
            Some(super::profiling::ScopedTimer::new(
                "blake2s_channel_mix",
                String::new(),
                "GPU",
            ))
        } else {
            None
        };
        // Create temporary buffer for root hash
        let root_buffer = self.context.device().new_buffer(
            32,
            MTLResourceOptions::StorageModeShared,
        );

        // Copy root hash to GPU buffer
        {
            let ptr = root_buffer.contents() as *mut u32;
            unsafe {
                for i in 0..8 {
                    let bytes: [u8; 4] = root.0[i * 4..(i + 1) * 4].try_into().unwrap();
                    ptr.add(i).write(u32::from_le_bytes(bytes));
                }
            }
        }

        // Create output buffer for new digest
        let new_digest_buffer = self.context.device().new_buffer(
            32,
            MTLResourceOptions::StorageModeShared,
        );

        // Dispatch GPU kernel
        let command_buffer = self.context.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(self.context.blake2s_channel_mix_pipeline());
        encoder.set_buffer(0, Some(&self.digest_buffer), 0);
        encoder.set_buffer(1, Some(&root_buffer), 0);
        let is_m31 = IS_M31_OUTPUT;
        encoder.set_bytes(
            2,
            std::mem::size_of::<bool>() as u64,
            &is_m31 as *const bool as *const _,
        );
        encoder.set_buffer(3, Some(&new_digest_buffer), 0);

        // Single thread execution (channel ops are serial)
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(1, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Replace old digest buffer with new one
        self.digest_buffer = new_digest_buffer;
        self.n_draws = 0;
    }

    /// Draw random u32s using GPU kernel.
    fn gpu_draw_u32s(&mut self) -> Vec<u32> {
        let _timer = if super::profiling::is_profiling_enabled() {
            Some(super::profiling::ScopedTimer::new(
                "blake2s_channel_draw",
                String::new(),
                "GPU",
            ))
        } else {
            None
        };
        // Create output buffer
        let output_buffer = self.context.device().new_buffer(
            32,
            MTLResourceOptions::StorageModeShared,
        );

        // Dispatch GPU kernel
        let command_buffer = self.context.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(self.context.blake2s_channel_draw_pipeline());
        encoder.set_buffer(0, Some(&self.digest_buffer), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &self.n_draws as *const u32 as *const _,
        );
        let domain_sep: u8 = 0;
        encoder.set_bytes(
            2,
            std::mem::size_of::<u8>() as u64,
            &domain_sep as *const u8 as *const _,
        );
        let is_m31 = IS_M31_OUTPUT;
        encoder.set_bytes(
            3,
            std::mem::size_of::<bool>() as u64,
            &is_m31 as *const bool as *const _,
        );
        encoder.set_buffer(4, Some(&output_buffer), 0);

        // Single thread execution
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(1, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Increment counter
        self.n_draws += 1;

        // Read output from GPU
        let ptr = output_buffer.contents() as *const u32;
        let mut result = Vec::with_capacity(8);
        unsafe {
            for i in 0..8 {
                result.push(ptr.add(i).read());
            }
        }
        result
    }

    /// Draw base field elements with rejection sampling.
    fn draw_base_felts(&mut self) -> [BaseField; FELTS_PER_HASH] {
        loop {
            let u32s: [u32; FELTS_PER_HASH] = self.gpu_draw_u32s().try_into().unwrap();

            // Retry if not all u32 are in range [0, 2P)
            if u32s.iter().all(|x| *x < 2 * P) {
                return u32s
                    .into_iter()
                    .map(|x| BaseField::reduce(x as u64))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
            }
        }
    }

    /// Mix felts using GPU kernel (for large arrays).
    fn gpu_mix_felts(&mut self, felts: &[SecureField]) {
        let _timer = if super::profiling::is_profiling_enabled() {
            Some(super::profiling::ScopedTimer::new(
                "blake2s_channel_mix_felts",
                format!("n_felts={}", felts.len()),
                "GPU",
            ))
        } else {
            None
        };

        // Serialize QM31 elements to u32 array (4 u32s per QM31)
        let felts_u32: Vec<u32> = felts
            .iter()
            .flat_map(|qm31| {
                let m31_array = qm31.to_m31_array();
                [m31_array[0].0, m31_array[1].0, m31_array[2].0, m31_array[3].0]
            })
            .collect();

        // Create GPU buffer for felts
        let felts_buffer = self.context.device().new_buffer_with_data(
            felts_u32.as_ptr() as *const _,
            (felts_u32.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create output buffer for new digest
        let new_digest_buffer = self.context.device().new_buffer(
            32,
            MTLResourceOptions::StorageModeShared,
        );

        // Dispatch GPU kernel
        let command_buffer = self.context.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(self.context.blake2s_channel_mix_felts_pipeline());
        encoder.set_buffer(0, Some(&self.digest_buffer), 0);
        encoder.set_buffer(1, Some(&felts_buffer), 0);
        let num_felts = felts.len() as u32;
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &num_felts as *const u32 as *const _,
        );
        let is_m31 = IS_M31_OUTPUT;
        encoder.set_bytes(
            3,
            std::mem::size_of::<bool>() as u64,
            &is_m31 as *const bool as *const _,
        );
        encoder.set_buffer(4, Some(&new_digest_buffer), 0);

        // Single thread execution (channel ops are serial)
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(1, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Replace old digest buffer with new one
        self.digest_buffer = new_digest_buffer;
        self.n_draws = 0;
    }

    /// Mix felts using CPU fallback (for small arrays).
    fn cpu_mix_felts(&mut self, felts: &[SecureField]) {
        let _timer = if super::profiling::is_profiling_enabled() {
            Some(super::profiling::ScopedTimer::new(
                "blake2s_channel_mix_felts",
                format!("n_felts={}", felts.len()),
                "CPU",
            ))
        } else {
            None
        };

        let felts_bytes: Vec<u8> = felts
            .iter()
            .flat_map(|qm31| qm31.to_m31_array())
            .flat_map(|m31| m31.0.to_le_bytes())
            .collect();

        let mut hasher = Blake2sHasherGeneric::<IS_M31_OUTPUT>::new();
        hasher.update(&self.digest().0);
        hasher.update(&felts_bytes);

        self.update_digest(hasher.finalize());
    }
}

impl<const IS_M31_OUTPUT: bool> Default for MetalBlake2sChannelGeneric<IS_M31_OUTPUT> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const IS_M31_OUTPUT: bool> Clone for MetalBlake2sChannelGeneric<IS_M31_OUTPUT> {
    fn clone(&self) -> Self {
        let context = self.context.clone();

        // Create new digest buffer
        let digest_buffer = context.device().new_buffer(
            32,
            MTLResourceOptions::StorageModeShared,
        );

        // Copy digest content
        {
            let src_ptr = self.digest_buffer.contents() as *const u32;
            let dst_ptr = digest_buffer.contents() as *mut u32;
            unsafe {
                for i in 0..8 {
                    dst_ptr.add(i).write(src_ptr.add(i).read());
                }
            }
        }

        Self {
            context,
            digest_buffer,
            n_draws: self.n_draws,
        }
    }
}

impl<const IS_M31_OUTPUT: bool> std::fmt::Debug for MetalBlake2sChannelGeneric<IS_M31_OUTPUT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBlake2sChannelGeneric")
            .field("digest", &self.digest())
            .field("n_draws", &self.n_draws)
            .field("is_m31_output", &IS_M31_OUTPUT)
            .finish()
    }
}

impl<const IS_M31_OUTPUT: bool> Channel for MetalBlake2sChannelGeneric<IS_M31_OUTPUT> {
    const BYTES_PER_HASH: usize = BLAKE_BYTES_PER_HASH;

    fn mix_felts(&mut self, felts: &[SecureField]) {
        // Use GPU for large arrays (>= 10 elements) to amortize kernel launch overhead
        // Use CPU for small arrays to avoid GPU round-trip latency
        const GPU_THRESHOLD: usize = 10;

        if felts.len() >= GPU_THRESHOLD {
            self.gpu_mix_felts(felts);
        } else {
            self.cpu_mix_felts(felts);
        }
    }

    fn mix_u32s(&mut self, data: &[u32]) {
        // Fall back to CPU for arbitrary u32 mixing
        let mut hasher = Blake2sHasherGeneric::<IS_M31_OUTPUT>::new();
        hasher.update(&self.digest().0);
        for word in data {
            hasher.update(&word.to_le_bytes());
        }

        self.update_digest(hasher.finalize());
    }

    fn mix_u64(&mut self, value: u64) {
        self.mix_u32s(&[value as u32, (value >> 32) as u32])
    }

    fn draw_secure_felt(&mut self) -> SecureField {
        let felts: [BaseField; FELTS_PER_HASH] = self.draw_base_felts();
        SecureField::from_m31_array(felts[..SECURE_EXTENSION_DEGREE].try_into().unwrap())
    }

    fn draw_secure_felts(&mut self, n_felts: usize) -> Vec<SecureField> {
        let mut felts = iter::from_fn(|| Some(self.draw_base_felts())).flatten();
        let secure_felts = iter::from_fn(|| {
            Some(SecureField::from_m31_array([
                felts.next()?,
                felts.next()?,
                felts.next()?,
                felts.next()?,
            ]))
        });
        secure_felts.take(n_felts).collect()
    }

    fn draw_u32s(&mut self) -> Vec<u32> {
        self.gpu_draw_u32s()
    }

    fn verify_pow_nonce(&self, n_bits: u32, nonce: u64) -> bool {
        let digest = self.digest();
        // Compute H(POW_PREFIX, [0_u8; 12], digest, n_bits).
        let mut hasher = Blake2sHasherGeneric::<IS_M31_OUTPUT>::default();
        hasher.update(&Blake2sChannelGeneric::<IS_M31_OUTPUT>::POW_PREFIX.to_le_bytes());
        hasher.update(&[0_u8; 12]);
        hasher.update(&digest.0[..]);
        hasher.update(&n_bits.to_le_bytes());
        let prefixed_digest = hasher.finalize();
        // Compute `H(prefixed_digest, nonce)`.
        let mut hasher = Blake2sHasherGeneric::<IS_M31_OUTPUT>::default();
        hasher.update(prefixed_digest.as_ref());
        hasher.update(&nonce.to_le_bytes());
        let res = hasher.finalize();
        let n_zeros = u128::from_le_bytes(core::array::from_fn(|i| res.0[i])).trailing_zeros();
        n_zeros >= n_bits
    }
}

/// Type alias for GPU channel with standard Blake2s output.
pub type MetalBlake2sChannel = MetalBlake2sChannelGeneric<false>;

/// Type alias for GPU channel with M31-reduced output.
pub type MetalBlake2sM31Channel = MetalBlake2sChannelGeneric<true>;

use crate::core::channel::MerkleChannel;
use crate::core::vcs::blake2_merkle::{Blake2sMerkleHasherGeneric};
use crate::core::vcs::MerkleHasher;

/// GPU-accelerated Merkle channel using Metal compute kernels.
#[derive(Default)]
pub struct MetalBlake2sMerkleChannelGeneric<const IS_M31_OUTPUT: bool>;

impl<const IS_M31_OUTPUT: bool> MerkleChannel for MetalBlake2sMerkleChannelGeneric<IS_M31_OUTPUT> {
    type C = MetalBlake2sChannelGeneric<IS_M31_OUTPUT>;
    type H = Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>;

    fn mix_root(channel: &mut Self::C, root: <Self::H as MerkleHasher>::Hash) {
        channel.gpu_mix_root(&root);
    }
}

/// Type alias for GPU Merkle channel with standard Blake2s output.
pub type MetalBlake2sMerkleChannel = MetalBlake2sMerkleChannelGeneric<false>;

/// Type alias for GPU Merkle channel with M31-reduced output.
pub type MetalBlake2sM31MerkleChannel = MetalBlake2sMerkleChannelGeneric<true>;
