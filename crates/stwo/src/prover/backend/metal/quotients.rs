//! Metal quotient operations.

use super::MetalBackend;
use crate::core::fields::m31::BaseField;
use crate::core::pcs::quotients::ColumnSampleBatch;
use crate::prover::backend::simd::column::BaseColumn;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Column;
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
        // Convert Metal columns to SIMD
        let simd_columns_owned: Vec<CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> =
            columns
                .iter()
                .map(|col| {
                    let cpu_vals = col.values.to_cpu();
                    let simd_col: BaseColumn = cpu_vals.into_iter().collect();
                    CircleEvaluation::new(col.domain, simd_col)
                })
                .collect();
        let simd_columns_refs: Vec<&CircleEvaluation<SimdBackend, BaseField, BitReversedOrder>> =
            simd_columns_owned.iter().collect();

        // Delegate to SIMD backend
        let mut simd_accs: Vec<AccumulatedNumerators<SimdBackend>> = Vec::new();
        SimdBackend::accumulate_numerators(&simd_columns_refs, sample_batches, &mut simd_accs);

        // Convert results back to Metal
        for acc in simd_accs {
            accumulated_numerators_vec.push(AccumulatedNumerators {
                sample_point: acc.sample_point,
                partial_numerators_acc: SecureColumnByCoords::from_simd(acc.partial_numerators_acc),
                first_linear_term_acc: acc.first_linear_term_acc,
            });
        }
    }

    fn compute_quotients_and_combine(
        accs: Vec<AccumulatedNumerators<Self>>,
        lifting_log_size: u32,
    ) -> SecureEvaluation<Self, BitReversedOrder> {
        // Convert Metal AccumulatedNumerators to SIMD
        let simd_accs: Vec<AccumulatedNumerators<SimdBackend>> = accs
            .into_iter()
            .map(|acc| {
                let cpu_col = acc.partial_numerators_acc.to_cpu();
                let simd_col = SecureColumnByCoords::<SimdBackend>::from_cpu(cpu_col);
                AccumulatedNumerators {
                    sample_point: acc.sample_point,
                    partial_numerators_acc: simd_col,
                    first_linear_term_acc: acc.first_linear_term_acc,
                }
            })
            .collect();

        let simd_result = SimdBackend::compute_quotients_and_combine(simd_accs, lifting_log_size);

        // Convert result back to Metal
        let metal_values = SecureColumnByCoords::from_simd(simd_result.values);
        SecureEvaluation::new(simd_result.domain, metal_values)
    }
}
