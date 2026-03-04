//! GPU constraint evaluation dispatcher.

use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use metal::{Buffer, MTLResourceOptions};

use super::context::MetalContext;
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::pcs::TreeVec;
use crate::core::poly::circle::CanonicCoset;
use crate::core::utils::bit_reverse;
use crate::prover::backend::metal::MetalBackend;
use crate::prover::poly::circle::CircleEvaluation;
use crate::prover::poly::BitReversedOrder;
use crate::prover::secure_column::SecureColumnByCoords;

struct ReshapedTraceCache {
    key: Option<(u64, usize)>,
    trace_buffer: Option<Buffer>,
    column_offsets: Vec<u32>,
}

impl ReshapedTraceCache {
    fn new() -> Self {
        Self {
            key: None,
            trace_buffer: None,
            column_offsets: Vec::new(),
        }
    }

    fn is_valid(&self, trace_hash: u64, eval_domain_size: usize) -> bool {
        self.key == Some((trace_hash, eval_domain_size)) && self.trace_buffer.is_some()
    }

    fn store(
        &mut self,
        trace_hash: u64,
        eval_domain_size: usize,
        trace_buffer: Buffer,
        column_offsets: Vec<u32>,
    ) {
        self.key = Some((trace_hash, eval_domain_size));
        self.trace_buffer = Some(trace_buffer);
        self.column_offsets = column_offsets;
    }

    fn get(&self) -> (Buffer, Vec<u32>) {
        (
            self.trace_buffer.as_ref().unwrap().clone(),
            self.column_offsets.clone(),
        )
    }

    #[allow(dead_code)]
    fn clear(&mut self) {
        self.key = None;
        self.trace_buffer = None;
        self.column_offsets.clear();
    }
}

thread_local! {
    static RESHAPED_TRACE_CACHE: RefCell<ReshapedTraceCache> = RefCell::new(ReshapedTraceCache::new());
}

fn hash_trace(
    trace: &TreeVec<Vec<&CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>>>,
) -> u64 {
    let mut hasher = DefaultHasher::new();

    trace.len().hash(&mut hasher);
    for tree_cols in trace.iter() {
        tree_cols.len().hash(&mut hasher);
        for col in tree_cols.iter() {
            let ptr = col.values.buffer().contents() as usize;
            ptr.hash(&mut hasher);
        }
    }

    hasher.finish()
}

pub struct GpuConstraintEvalResult {
    pub output: SecureColumnByCoords<MetalBackend>,
}

#[allow(dead_code)]
pub fn clear_trace_cache() {
    RESHAPED_TRACE_CACHE.with(|cache_cell| {
        cache_cell.borrow_mut().clear();
    });
}

fn reshape_trace_columns_gpu(
    trace: &TreeVec<Vec<&CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>>>,
    eval_domain_size: usize,
) -> (metal::Buffer, Vec<u32>) {
    let trace_hash = hash_trace(trace);

    let cached = RESHAPED_TRACE_CACHE.with(|cache_cell| {
        let cache = cache_cell.borrow();
        if cache.is_valid(trace_hash, eval_domain_size) {
            Some(cache.get())
        } else {
            None
        }
    });

    if let Some((buffer, offsets)) = cached {
        return (buffer, offsets);
    }

    let ctx = MetalContext::global();
    let device = ctx.device();

    let mut total_cols = 0usize;
    let mut column_offsets = vec![0u32];
    for tree_cols in trace.iter() {
        total_cols += tree_cols.len();
        column_offsets.push(total_cols as u32);
    }

    let output_buffer = device.new_buffer(
        (total_cols * eval_domain_size * std::mem::size_of::<u32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(ctx.trace_reshape_column_pipeline());

    let mut col_idx = 0u32;
    for tree_cols in trace.iter() {
        for col in tree_cols.iter() {
            encoder.set_buffer(0, Some(col.values.buffer()), 0);
            encoder.set_buffer(1, Some(&output_buffer), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &col_idx as *const u32 as *const _,
            );
            encoder.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &(eval_domain_size as u32) as *const u32 as *const _,
            );

            let grid_size = metal::MTLSize::new(eval_domain_size as u64, 1, 1);
            let threadgroup_size = metal::MTLSize::new(256.min(eval_domain_size as u64), 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);

            col_idx += 1;
        }
    }

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    RESHAPED_TRACE_CACHE.with(|cache_cell| {
        cache_cell.borrow_mut().store(
            trace_hash,
            eval_domain_size,
            output_buffer.clone(),
            column_offsets.clone(),
        );
    });

    (output_buffer, column_offsets)
}

pub fn evaluate_constraints_gpu(
    bytecode_bytes: &[u8],
    trace: &TreeVec<Vec<&CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>>>,
    random_coeffs: &[SecureField],
    trace_domain: &CanonicCoset,
    eval_domain_log_size: u32,
) -> GpuConstraintEvalResult {
    let ctx = MetalContext::global();
    let device = ctx.device();

    let eval_domain_size = 1usize << eval_domain_log_size;

    let (trace_buffer, column_offsets) = reshape_trace_columns_gpu(trace, eval_domain_size);

    let bytecode_buffer = device.new_buffer_with_data(
        bytecode_bytes.as_ptr() as *const _,
        bytecode_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let random_coeffs_u32: Vec<u32> = random_coeffs
        .iter()
        .flat_map(|qm31| {
            let m31_array = qm31.to_m31_array();
            [
                m31_array[0].0,
                m31_array[1].0,
                m31_array[2].0,
                m31_array[3].0,
            ]
        })
        .collect();

    let random_coeffs_buffer = device.new_buffer_with_data(
        random_coeffs_u32.as_ptr() as *const _,
        (random_coeffs_u32.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let column_offsets_buffer = device.new_buffer_with_data(
        column_offsets.as_ptr() as *const _,
        (column_offsets.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let eval_domain = CanonicCoset::new(eval_domain_log_size).circle_domain();
    let log_expand = eval_domain_log_size - trace_domain.log_size();
    let mut denom_invs_m31: Vec<BaseField> = (0..1 << log_expand)
        .map(|i| {
            use crate::core::constraints::coset_vanishing;
            coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse()
        })
        .collect();
    bit_reverse(&mut denom_invs_m31);

    let denom_invs: Vec<SecureField> = denom_invs_m31
        .iter()
        .map(|&m| SecureField::from(m))
        .collect();

    let denom_invs_full: Vec<u32> = (0..eval_domain_size)
        .flat_map(|row_idx| {
            let denom_inv = denom_invs[row_idx >> trace_domain.log_size()];
            let m31_array = denom_inv.to_m31_array();
            [
                m31_array[0].0,
                m31_array[1].0,
                m31_array[2].0,
                m31_array[3].0,
            ]
        })
        .collect();

    let denom_invs_buffer = device.new_buffer_with_data(
        denom_invs_full.as_ptr() as *const _,
        (denom_invs_full.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.new_buffer(
        (eval_domain_size * 4 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(ctx.constraint_eval_vm_pipeline());

    let bytecode_len = bytecode_bytes.len() as u32;
    let bytecode_len_buffer = device.new_buffer_with_data(
        &bytecode_len as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let n_columns = column_offsets[column_offsets.len() - 1];
    let n_columns_buffer = device.new_buffer_with_data(
        &n_columns as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let n_random_coeffs = random_coeffs.len() as u32;
    let n_random_coeffs_buffer = device.new_buffer_with_data(
        &n_random_coeffs as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let trace_log_size = trace_domain.log_size();
    let trace_log_size_buffer = device.new_buffer_with_data(
        &trace_log_size as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let eval_log_size_buffer = device.new_buffer_with_data(
        &eval_domain_log_size as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    encoder.set_buffer(0, Some(&bytecode_buffer), 0);
    encoder.set_buffer(1, Some(&bytecode_len_buffer), 0);
    encoder.set_buffer(2, Some(&trace_buffer), 0);
    encoder.set_buffer(3, Some(&column_offsets_buffer), 0);
    encoder.set_buffer(4, Some(&n_columns_buffer), 0);
    encoder.set_buffer(5, Some(&random_coeffs_buffer), 0);
    encoder.set_buffer(6, Some(&n_random_coeffs_buffer), 0);
    encoder.set_buffer(7, Some(&denom_invs_buffer), 0);
    encoder.set_buffer(8, Some(&trace_log_size_buffer), 0);
    encoder.set_buffer(9, Some(&eval_log_size_buffer), 0);
    encoder.set_buffer(10, Some(&output_buffer), 0);

    let grid_size = metal::MTLSize::new(eval_domain_size as u64, 1, 1);
    let threadgroup_size = metal::MTLSize::new(256, 1, 1);

    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let ctx = MetalContext::global();
    let unpack_result = ctx.unpack_qm31_to_coords(&output_buffer, eval_domain_size);

    let output = SecureColumnByCoords::<MetalBackend> {
        columns: unpack_result,
    };

    GpuConstraintEvalResult { output }
}

/// Evaluate multiple constraint programs on the same trace.
///
/// Each program runs independently on GPU with its own bytecode and random coefficients,
/// sharing the same reshaped trace data (cached on first call).
pub fn evaluate_constraints_batched_gpu(
    bytecode_programs: &[&[u8]],
    random_coeffs_per_program: &[&[SecureField]],
    trace: &TreeVec<Vec<&CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>>>,
    trace_domain: &CanonicCoset,
    eval_domain_log_size: u32,
) -> Vec<GpuConstraintEvalResult> {
    assert_eq!(bytecode_programs.len(), random_coeffs_per_program.len());

    // Pre-warm the trace reshape cache (shared across all programs)
    let eval_domain_size = 1usize << eval_domain_log_size;
    let _ = reshape_trace_columns_gpu(trace, eval_domain_size);

    // Run each program independently, reusing the cached reshaped trace
    bytecode_programs
        .iter()
        .zip(random_coeffs_per_program.iter())
        .map(|(bytecode, coeffs)| {
            evaluate_constraints_gpu(
                bytecode,
                trace,
                coeffs,
                trace_domain,
                eval_domain_log_size,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_constraint_eval_smoke() {
        assert!(MetalContext::is_available());
    }
}
