//! Metal GPU-accelerated backend for Apple Silicon.
//!
//! This backend leverages Metal compute shaders to accelerate:
//! - FFT/IFFT operations (circle polynomials)
//! - FRI folding (line and circle)
//! - Quotient accumulation
//! - Merkle tree construction
//! - Lookup operations (MLE/GKR)
//! - Proof-of-work grinding
//! - Blake2s channel operations (GPU-resident state)
//!
//! # Memory Model
//!
//! Uses `MTLStorageModeShared` for unified memory access between CPU and GPU,
//! eliminating explicit copies on Apple Silicon.
//!
//! # Pipelining Strategy
//!
//! - Command buffers can be enqueued while previous ones execute
//! - Threadgroup (shared) memory used for FFT tiles and reductions
//! - Falls back to SIMD/CPU for small workloads (below threshold)
//!
//! # GPU-Resident Channel State
//!
//! The backend provides `MetalBlake2sM31MerkleChannel` which keeps the Fiat-Shamir
//! channel digest on the GPU, eliminating CPU-GPU round-trips during proof generation.
//! This is particularly beneficial for FRI operations where the channel is frequently
//! updated with Merkle roots.
//!
//! ## Usage Example
//! See `channel.rs` module documentation for detailed analysis.
//!
//! ```rust,ignore
//! use stwo::prover::backend::metal::{MetalBackend, MetalBlake2sM31Channel, MetalBlake2sM31MerkleChannel};
//! use stwo::prover::CommitmentSchemeProver;
//!
//! let prover_channel = &mut MetalBlake2sM31Channel::default();
//! let mut commitment_scheme = CommitmentSchemeProver::<MetalBackend, MetalBlake2sM31MerkleChannel>::new(
//!     config, &twiddles,
//! );
//! ```
//!
//! See `crates/examples/examples/test_gpu_channel.rs` for a complete example.

#[cfg(target_os = "macos")]
mod buffer_pool;
#[cfg(target_os = "macos")]
mod channel;
#[cfg(target_os = "macos")]
mod column;
#[cfg(target_os = "macos")]
pub mod constraint_eval_gpu;
#[cfg(target_os = "macos")]
mod context;
#[cfg(target_os = "macos")]
pub mod profiling;
#[cfg(target_os = "macos")]
mod shaders;
#[cfg(target_os = "macos")]
mod twiddle_manager;

// Export Metal context (actively used)
// Export GPU channel types (for GPU-resident Fiat-Shamir state)
#[cfg(target_os = "macos")]
pub use channel::{
    MetalBlake2sChannel, MetalBlake2sM31Channel, MetalBlake2sM31MerkleChannel,
    MetalBlake2sMerkleChannel,
};
// Export Metal column types (now actively used)
#[cfg(target_os = "macos")]
pub use column::{GpuSlice, MetalBaseColumn, MetalSecureColumn};
#[cfg(target_os = "macos")]
pub use context::{MetalContext, MetalContextHandle};
#[cfg(target_os = "macos")]
use serde::{Deserialize, Serialize};

#[cfg(target_os = "macos")]
use crate::core::fields::m31::BaseField;
#[cfg(target_os = "macos")]
use crate::core::fields::qm31::SecureField;
#[cfg(target_os = "macos")]
use crate::core::vcs::blake2_merkle::{Blake2sM31MerkleChannel, Blake2sMerkleChannel};
#[cfg(target_os = "macos")]
use crate::prover::backend::{Backend, BackendForChannel, ColumnOps};

/// Size thresholds for GPU vs CPU dispatch.
///
/// These thresholds are tuned based on benchmarks to ensure GPU is only used
/// when it provides a performance benefit over SIMD. Below these thresholds,
/// GPU dispatch overhead dominates and SIMD is faster.
#[cfg(target_os = "macos")]
pub mod thresholds {
    /// Minimum log size for GPU FFT.
    /// Lowered to 12 for batched operations and improved kernels.
    pub const MIN_FFT_LOG_SIZE: u32 = 12;

    /// Minimum log size for GPU FRI folding.
    /// Lowered to 10 for better GPU utilization in batched contexts.
    pub const MIN_FRI_LOG_SIZE: u32 = 10;

    /// Minimum log size for GPU Merkle operations.
    pub const MIN_MERKLE_LOG_SIZE: u32 = 10;

    /// Minimum log size for GPU quotient accumulation.
    /// Lowered to 10 for better GPU coverage.
    pub const MIN_QUOTIENT_LOG_SIZE: u32 = 10;

    /// Minimum log size for GPU MLE operations.
    /// Lowered to 10 for better GPU coverage.
    pub const MIN_MLE_LOG_SIZE: u32 = 10;
}

// Trait implementations
#[cfg(target_os = "macos")]
mod accumulation;
#[cfg(target_os = "macos")]
mod fri;
#[cfg(target_os = "macos")]
mod gkr;
#[cfg(target_os = "macos")]
mod grind;
#[cfg(target_os = "macos")]
mod merkle;
#[cfg(target_os = "macos")]
mod poly;
#[cfg(target_os = "macos")]
mod quotients;

/// Metal GPU-accelerated backend.
///
/// This backend uses Metal GPU compute shaders for large workloads and falls back
/// to SIMD/CPU for small workloads below size thresholds.
#[cfg(target_os = "macos")]
#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
pub struct MetalBackend;

#[cfg(target_os = "macos")]
impl Backend for MetalBackend {}

#[cfg(target_os = "macos")]
impl BackendForChannel<Blake2sMerkleChannel> for MetalBackend {}

#[cfg(target_os = "macos")]
impl BackendForChannel<Blake2sM31MerkleChannel> for MetalBackend {}

#[cfg(target_os = "macos")]
impl BackendForChannel<MetalBlake2sMerkleChannel> for MetalBackend {}

#[cfg(target_os = "macos")]
impl BackendForChannel<MetalBlake2sM31MerkleChannel> for MetalBackend {}

#[cfg(target_os = "macos")]
impl ColumnOps<BaseField> for MetalBackend {
    type Column = MetalBaseColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        use crate::core::utils::bit_reverse;
        let slice = column.as_mut_slice();
        bit_reverse(slice);
    }
}

#[cfg(target_os = "macos")]
impl ColumnOps<SecureField> for MetalBackend {
    type Column = MetalSecureColumn;

    fn bit_reverse_column(column: &mut Self::Column) {
        use crate::core::utils::bit_reverse;
        let slice = column.as_mut_slice();
        bit_reverse(slice);
    }
}

// Placeholder implementations when not on macOS
#[cfg(not(target_os = "macos"))]
pub struct MetalBackend;

#[cfg(not(target_os = "macos"))]
impl MetalBackend {
    pub fn is_available() -> bool {
        false
    }
}
