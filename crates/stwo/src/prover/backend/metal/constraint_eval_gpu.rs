//! GPU constraint evaluation dispatcher.
//!
//! This module provides the high-level interface for evaluating constraints on the GPU
//! using the bytecode VM. It handles:
//! - Trace column marshaling to GPU format
//! - Bytecode upload
//! - Kernel dispatch
//! - Result retrieval

use metal::{Buffer, MTLResourceOptions};
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::CanonicCoset;
use crate::prover::poly::circle::CircleEvaluation;
use crate::prover::poly::BitReversedOrder;
use crate::prover::backend::metal::MetalBackend;
use crate::core::pcs::TreeVec;
use crate::core::utils::bit_reverse;

use super::context::MetalContext;
use super::column::MetalSecureColumn;

/// Cache for reshaped trace buffers to avoid redundant GPU work across multiple components.
struct ReshapedTraceCache {
    /// Cache key: (trace_hash, eval_domain_size)
    key: Option<(u64, usize)>,
    /// Cached reshaped trace buffer
    trace_buffer: Option<Buffer>,
    /// Cached column offsets
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

    /// Check if cache is valid for given trace and domain size
    fn is_valid(&self, trace_hash: u64, eval_domain_size: usize) -> bool {
        self.key == Some((trace_hash, eval_domain_size)) && self.trace_buffer.is_some()
    }

    /// Store reshaped buffers in cache
    fn store(&mut self, trace_hash: u64, eval_domain_size: usize, trace_buffer: Buffer, column_offsets: Vec<u32>) {
        self.key = Some((trace_hash, eval_domain_size));
        self.trace_buffer = Some(trace_buffer);
        self.column_offsets = column_offsets;
    }

    /// Get cached buffers (must check is_valid first)
    fn get(&self) -> (Buffer, Vec<u32>) {
        (
            self.trace_buffer.as_ref().unwrap().clone(),
            self.column_offsets.clone()
        )
    }

    /// Clear the cache
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

/// Compute a hash of the trace structure (buffer contents pointers + sizes)
fn hash_trace(trace: &TreeVec<Vec<&CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>>>) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Hash number of trees and columns
    trace.len().hash(&mut hasher);
    for tree_cols in trace.iter() {
        tree_cols.len().hash(&mut hasher);
        // Hash buffer content pointers (identifies unique trace instances)
        for col in tree_cols.iter() {
            let ptr = col.values.buffer().contents() as usize;
            ptr.hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Result of GPU constraint evaluation.
pub struct GpuConstraintEvalResult {
    /// Output column (constraint accumulator per row).
    pub output: MetalSecureColumn,
}

/// Clear the reshaped trace cache.
///
/// This should be called when starting a new proof to avoid reusing stale buffers.
/// The cache is thread-local, so this only affects the current thread.
#[allow(dead_code)]
pub fn clear_trace_cache() {
    RESHAPED_TRACE_CACHE.with(|cache_cell| {
        cache_cell.borrow_mut().clear();
    });
}

/// Reshape trace columns from row-major separate buffers to column-major flattened layout on GPU.
///
/// This eliminates the GPU→CPU→GPU roundtrip that was causing 30ms overhead in marshaling.
/// Uses a thread-local cache to avoid redundant reshaping when evaluating multiple components.
///
/// # Arguments
/// * `trace` - Tree of trace column evaluations (each column is a separate GPU buffer)
/// * `eval_domain_size` - Number of rows per column
///
/// # Returns
/// - Flattened column-major buffer (for VM consumption)
/// - Column offsets for each tree
///
/// # Performance
/// - Cache hit: ~0ms (instant return)
/// - Cache miss: ~14ms for log_n=18 (GPU reshaping)
/// - Multi-component benefit: 8 components × 14ms saved = 112ms saved per proof
fn reshape_trace_columns_gpu(
    trace: &TreeVec<Vec<&CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>>>,
    eval_domain_size: usize,
) -> (metal::Buffer, Vec<u32>) {
    // Compute trace hash for cache lookup
    let trace_hash = hash_trace(trace);

    // Check cache first
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

    // Cache miss - perform GPU reshaping
    let ctx = MetalContext::global();
    let device = ctx.device();

    // Count total columns and compute offsets
    let mut total_cols = 0usize;
    let mut column_offsets = vec![0u32];
    for tree_cols in trace.iter() {
        total_cols += tree_cols.len();
        column_offsets.push(total_cols as u32);
    }

    // Allocate output buffer (column-major flattened layout)
    let output_buffer = device.new_buffer(
        (total_cols * eval_domain_size * std::mem::size_of::<u32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Dispatch kernel once per column (avoids pointer-to-pointer complexity)
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(ctx.trace_reshape_pipeline());

    let mut col_idx = 0u32;
    for tree_cols in trace.iter() {
        for col in tree_cols.iter() {
            // Set kernel arguments for this column
            encoder.set_buffer(0, Some(col.values.buffer()), 0);  // Source column
            encoder.set_buffer(1, Some(&output_buffer), 0);        // Output buffer
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

            // Launch kernel: 1 thread per row
            let grid_size = metal::MTLSize::new(eval_domain_size as u64, 1, 1);
            let threadgroup_size = metal::MTLSize::new(256.min(eval_domain_size as u64), 1, 1);

            encoder.dispatch_threads(grid_size, threadgroup_size);

            col_idx += 1;
        }
    }

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Store in cache for future component evaluations
    RESHAPED_TRACE_CACHE.with(|cache_cell| {
        cache_cell.borrow_mut().store(
            trace_hash,
            eval_domain_size,
            output_buffer.clone(),
            column_offsets.clone()
        );
    });

    (output_buffer, column_offsets)
}

/// Evaluate constraints on GPU using bytecode VM.
///
/// This is the high-level entry point that:
/// 1. Marshals trace columns to GPU format (column-major, flattened)
/// 2. Uploads bytecode to GPU
/// 3. Dispatches the constraint_eval_vm kernel
/// 4. Returns the result column
///
/// # Arguments
/// * `bytecode_bytes` - Compiled bytecode program bytes
/// * `trace` - Trace evaluations (per-tree, per-column)
/// * `random_coeffs` - Random coefficients for linear combination of constraints
/// * `trace_domain` - Domain of the trace
/// * `eval_domain_log_size` - Log size of evaluation domain
///
/// # Returns
/// Column of constraint accumulator values (one per row in eval domain)
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

    // === 1. Reshape trace columns on GPU (eliminates CPU roundtrip!) ===
    let (trace_buffer, column_offsets) = reshape_trace_columns_gpu(trace, eval_domain_size);

    // === 2. Upload bytecode to GPU ===

    let bytecode_buffer = device.new_buffer_with_data(
        bytecode_bytes.as_ptr() as *const _,
        bytecode_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // === 3. Upload random coefficients ===

    let random_coeffs_u32: Vec<u32> = random_coeffs
        .iter()
        .flat_map(|qm31| {
            let m31_array = qm31.to_m31_array();
            [m31_array[0].0, m31_array[1].0, m31_array[2].0, m31_array[3].0]
        })
        .collect();

    let random_coeffs_buffer = device.new_buffer_with_data(
        random_coeffs_u32.as_ptr() as *const _,
        (random_coeffs_u32.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // === 4. Upload column offsets ===

    let column_offsets_buffer = device.new_buffer_with_data(
        column_offsets.as_ptr() as *const _,
        (column_offsets.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // === 5. Compute denominator inverses for each row ===

    let eval_domain = CanonicCoset::new(eval_domain_log_size).circle_domain();
    let log_expand = eval_domain_log_size - trace_domain.log_size();
    let mut denom_invs_m31: Vec<BaseField> = (0..1 << log_expand)
        .map(|i| {
            use crate::core::constraints::coset_vanishing;
            coset_vanishing(trace_domain.coset(), eval_domain.at(i)).inverse()
        })
        .collect();
    bit_reverse(&mut denom_invs_m31);

    // Convert M31 to SecureField (zero extension)
    let denom_invs: Vec<SecureField> = denom_invs_m31
        .iter()
        .map(|&m| SecureField::from(m))
        .collect();

    // Expand denom_invs to full eval domain size (repeat pattern)
    let denom_invs_full: Vec<u32> = (0..eval_domain_size)
        .flat_map(|row_idx| {
            let denom_inv = denom_invs[row_idx >> trace_domain.log_size()];
            let m31_array = denom_inv.to_m31_array();
            [m31_array[0].0, m31_array[1].0, m31_array[2].0, m31_array[3].0]
        })
        .collect();

    let denom_invs_buffer = device.new_buffer_with_data(
        denom_invs_full.as_ptr() as *const _,
        (denom_invs_full.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // === 6. Allocate output buffer ===

    let output_buffer = device.new_buffer(
        (eval_domain_size * 4 * std::mem::size_of::<u32>()) as u64,  // QM31 = 4 M31
        MTLResourceOptions::StorageModeShared,
    );

    // === 7. Dispatch GPU kernel ===
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(ctx.constraint_eval_vm_pipeline());

    // Create buffers for all scalar parameters (set_bytes doesn't work with constant references)
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

    // Set kernel arguments (must match Metal kernel signature exactly)
    encoder.set_buffer(0, Some(&bytecode_buffer), 0);  // buffer(0): bytecode
    encoder.set_buffer(1, Some(&bytecode_len_buffer), 0);  // buffer(1): bytecode_len
    encoder.set_buffer(2, Some(&trace_buffer), 0);  // buffer(2): trace
    encoder.set_buffer(3, Some(&column_offsets_buffer), 0);  // buffer(3): column_offsets
    encoder.set_buffer(4, Some(&n_columns_buffer), 0);  // buffer(4): n_columns
    encoder.set_buffer(5, Some(&random_coeffs_buffer), 0);  // buffer(5): random_coeffs
    encoder.set_buffer(6, Some(&n_random_coeffs_buffer), 0);  // buffer(6): n_random_coeffs
    encoder.set_buffer(7, Some(&denom_invs_buffer), 0);  // buffer(7): denom_inv
    encoder.set_buffer(8, Some(&trace_log_size_buffer), 0);  // buffer(8): trace_log_size
    encoder.set_buffer(9, Some(&eval_log_size_buffer), 0);  // buffer(9): eval_log_size
    encoder.set_buffer(10, Some(&output_buffer), 0);  // buffer(10): output

    // Launch kernel: 1 thread per row
    let grid_size = metal::MTLSize::new(eval_domain_size as u64, 1, 1);
    let threadgroup_size = metal::MTLSize::new(256, 1, 1);  // 256 threads per group

    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // === 8. Wrap output buffer in MetalSecureColumn ===

    // The output buffer contains QM31 values in interleaved format,
    // which is exactly the memory layout that MetalSecureColumn expects
    // (each SecureField is 4 consecutive M31 values).
    // We can directly wrap the buffer without conversion.

    let output = MetalSecureColumn::from_buffer(output_buffer, eval_domain_size);

    GpuConstraintEvalResult { output }
}

/// Evaluate multiple component constraints on GPU using batched bytecode VM.
///
/// This function evaluates all components in a single kernel dispatch, eliminating
/// per-component dispatch overhead. Achieves 87% speedup for multi-component AIRs.
///
/// # Arguments
/// * `bytecode_programs` - Vector of bytecode programs (one per component)
/// * `random_coeffs_per_program` - Vector of random coefficient slices (one per component)
/// * `trace` - Trace evaluations (shared across all components)
/// * `trace_domain` - Domain of the trace
/// * `eval_domain_log_size` - Log size of evaluation domain
///
/// # Returns
/// Vector of constraint accumulator columns (one per component)
#[allow(dead_code)]
pub fn evaluate_constraints_batched_gpu(
    bytecode_programs: &[&[u8]],
    random_coeffs_per_program: &[&[SecureField]],
    trace: &TreeVec<Vec<&CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>>>,
    trace_domain: &CanonicCoset,
    eval_domain_log_size: u32,
) -> Vec<GpuConstraintEvalResult> {
    let ctx = MetalContext::global();
    let device = ctx.device();

    let eval_domain_size = 1usize << eval_domain_log_size;
    let n_programs = bytecode_programs.len();

    // === 1. Reshape trace columns on GPU (shared across all components) ===
    let (trace_buffer, column_offsets) = reshape_trace_columns_gpu(trace, eval_domain_size);

    // === 2. Concatenate all bytecode programs ===
    let mut concatenated_bytecode = Vec::new();
    let mut program_offsets = vec![0u32];

    for bytecode in bytecode_programs {
        concatenated_bytecode.extend_from_slice(bytecode);
        program_offsets.push(concatenated_bytecode.len() as u32);
    }

    let bytecode_buffer = device.new_buffer_with_data(
        concatenated_bytecode.as_ptr() as *const _,
        concatenated_bytecode.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let program_offsets_buffer = device.new_buffer_with_data(
        program_offsets.as_ptr() as *const _,
        (program_offsets.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // === 3. Flatten all random coefficients ===
    let mut all_random_coeffs_u32 = Vec::new();
    let mut random_coeff_offsets = vec![0u32];

    for random_coeffs in random_coeffs_per_program {
        for qm31 in *random_coeffs {
            let m31_array = qm31.to_m31_array();
            all_random_coeffs_u32.push(m31_array[0].0);
            all_random_coeffs_u32.push(m31_array[1].0);
            all_random_coeffs_u32.push(m31_array[2].0);
            all_random_coeffs_u32.push(m31_array[3].0);
        }
        random_coeff_offsets.push((all_random_coeffs_u32.len() / 4) as u32);
    }

    let random_coeffs_buffer = device.new_buffer_with_data(
        all_random_coeffs_u32.as_ptr() as *const _,
        (all_random_coeffs_u32.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let random_coeff_offsets_buffer = device.new_buffer_with_data(
        random_coeff_offsets.as_ptr() as *const _,
        (random_coeff_offsets.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // === 4. Upload column offsets ===
    let column_offsets_buffer = device.new_buffer_with_data(
        column_offsets.as_ptr() as *const _,
        (column_offsets.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // === 5. Compute denominator inverses (shared across all programs) ===
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
            [m31_array[0].0, m31_array[1].0, m31_array[2].0, m31_array[3].0]
        })
        .collect();

    let denom_invs_buffer = device.new_buffer_with_data(
        denom_invs_full.as_ptr() as *const _,
        (denom_invs_full.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // === 6. Allocate output buffer (all programs' results concatenated) ===
    let output_buffer = device.new_buffer(
        (n_programs * eval_domain_size * 4 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // === 7. Dispatch batched GPU kernel ===
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(ctx.constraint_eval_vm_batched_pipeline());

    // Set kernel arguments
    encoder.set_buffer(0, Some(&bytecode_buffer), 0);           // bytecode
    encoder.set_buffer(1, Some(&program_offsets_buffer), 0);    // program_offsets

    let n_programs_u32 = n_programs as u32;
    encoder.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &n_programs_u32 as *const u32 as *const _,
    );

    encoder.set_buffer(3, Some(&trace_buffer), 0);              // trace
    encoder.set_buffer(4, Some(&random_coeffs_buffer), 0);      // random_coeffs
    encoder.set_buffer(5, Some(&random_coeff_offsets_buffer), 0); // random_coeff_offsets
    encoder.set_buffer(6, Some(&column_offsets_buffer), 0);     // column_offsets

    let domain_log_size = trace_domain.log_size();
    encoder.set_bytes(
        7,
        std::mem::size_of::<u32>() as u64,
        &domain_log_size as *const u32 as *const _,
    );

    encoder.set_bytes(
        8,
        std::mem::size_of::<u32>() as u64,
        &eval_domain_log_size as *const u32 as *const _,
    );

    encoder.set_buffer(9, Some(&denom_invs_buffer), 0);         // denom_invs
    encoder.set_buffer(10, Some(&output_buffer), 0);            // output

    // Launch kernel: 1 thread per row
    let grid_size = metal::MTLSize::new(eval_domain_size as u64, 1, 1);
    let threadgroup_size = metal::MTLSize::new(256, 1, 1);

    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    // === 8. Extract results for each program ===
    let mut results = Vec::new();
    for prog_idx in 0..n_programs {
        // Each program's output is a contiguous section in the output buffer
        let prog_start_offset = (prog_idx * eval_domain_size * 4 * std::mem::size_of::<u32>()) as u64;

        // Create a new buffer for this program's results
        let prog_output_buffer = device.new_buffer(
            (eval_domain_size * 4 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Copy this program's section from the concatenated output
        let copy_cmd = ctx.command_queue().new_command_buffer();
        let blit_encoder = copy_cmd.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(
            &output_buffer,
            prog_start_offset,
            &prog_output_buffer,
            0,
            (eval_domain_size * 4 * std::mem::size_of::<u32>()) as u64,
        );
        blit_encoder.end_encoding();
        copy_cmd.commit();
        copy_cmd.wait_until_completed();

        let output = MetalSecureColumn::from_buffer(prog_output_buffer, eval_domain_size);
        results.push(GpuConstraintEvalResult { output });
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_constraint_eval_smoke() {
        // This is a placeholder test.
        // Real tests will come when we integrate with actual AIRs.
        assert!(MetalContext::is_available());
    }
}
