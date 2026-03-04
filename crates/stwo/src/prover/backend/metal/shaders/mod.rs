//! Metal shader management and utilities.
//!
//! This module handles shader compilation and provides access to
//! Metal kernel source code.
//!
//! Shaders are split into separate files by operation for readability
//! and auditability (matching CUDA's per-operation file structure),
//! then concatenated at compile time since Metal requires a single
//! compilation unit.

/// Metal shader source code for all compute kernels.
///
/// This includes implementations for:
/// - Field arithmetic (M31, CM31, QM31)
/// - FFT/IFFT (circle domain, radix-2/8)
/// - FRI folding (circle and line)
/// - Quotient accumulation
/// - Merkle tree hashing (BLAKE2s)
/// - Blake2s channel operations
/// - MLE folding for GKR lookups
/// - Proof-of-work grinding
/// - Circle polynomial evaluation at points
/// - Constraint evaluation VM
/// - Bit reversal permutation
/// - Batch inverse (Montgomery's trick)
pub const KERNEL_SOURCE: &str = concat!(
    include_str!("fields.metal"),
    include_str!("blake2s.metal"),
    include_str!("blake2s_channel.metal"),
    include_str!("utils.metal"),
    include_str!("fft.metal"),
    include_str!("fri.metal"),
    include_str!("quotients.metal"),
    include_str!("merkle.metal"),
    include_str!("grind.metal"),
    include_str!("mle.metal"),
    include_str!("eval_at_point.metal"),
    include_str!("constraint_eval.metal"),
    include_str!("bit_reverse.metal"),
    include_str!("batch_inverse.metal"),
);
