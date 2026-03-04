//! Poseidon252 Merkle support for the Metal backend.
//!
//! Delegates to SIMD/CPU since Poseidon252 operates on 252-bit field elements
//! which are too complex for Metal shader implementation. The SIMD backend itself
//! also delegates to CPU for Poseidon hashing (no vectorized Poseidon yet).

use starknet_ff::FieldElement as FieldElement252;

use super::MetalBackend;
use crate::core::fields::m31::BaseField;
use crate::core::vcs_lifted::poseidon252_merkle::Poseidon252MerkleHasher;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::{Col, Column, ColumnOps};
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

impl ColumnOps<FieldElement252> for MetalBackend {
    type Column = Vec<FieldElement252>;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOpsLifted<Poseidon252MerkleHasher> for MetalBackend {
    fn build_leaves(
        columns: &[&Col<Self, BaseField>],
        lifting_log_size: u32,
    ) -> Col<Self, FieldElement252> {
        // Convert Metal columns to SIMD columns, delegate to SIMD
        use crate::prover::backend::simd::column::BaseColumn;

        let simd_columns_owned: Vec<BaseColumn> = columns
            .iter()
            .map(|col| {
                let cpu_vals = col.to_cpu();
                cpu_vals.into_iter().collect()
            })
            .collect();
        let simd_columns_refs: Vec<&BaseColumn> = simd_columns_owned.iter().collect();

        <SimdBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_leaves(
            &simd_columns_refs,
            lifting_log_size,
        )
    }

    fn build_next_layer(prev_layer: &Col<Self, FieldElement252>) -> Col<Self, FieldElement252> {
        <SimdBackend as MerkleOpsLifted<Poseidon252MerkleHasher>>::build_next_layer(prev_layer)
    }
}
