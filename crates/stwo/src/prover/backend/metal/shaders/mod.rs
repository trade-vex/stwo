//! Metal shader management and utilities.
//!
//! This module handles shader compilation and provides access to
//! Metal kernel source code.

/// Metal shader source code for all compute kernels.
///
/// This includes implementations for:
/// - FFT/IFFT (circle domain)
/// - FRI folding (circle and line)
/// - Quotient accumulation
/// - Merkle tree hashing (BLAKE2s)
/// - MLE folding for GKR lookups
pub const KERNEL_SOURCE: &str = include_str!("kernels.metal");
