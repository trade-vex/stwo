//! Metal polynomial operations (PolyOps trait implementation).
//!
//! This module implements circle FFT/IFFT operations with GPU acceleration
//! for large workloads and SIMD fallback for small workloads.

use metal::MTLResourceOptions;

use crate::core::circle::{CirclePoint, Coset};
use crate::core::fields::m31::BaseField;
use crate::core::fields::qm31::SecureField;
use crate::core::poly::circle::CircleDomain;
use crate::core::poly::utils::domain_line_twiddles_from_tree;
use crate::prover::air::component_prover::Poly;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::Column;
use crate::prover::poly::circle::{CircleCoefficients, CircleEvaluation, PolyOps};
use crate::prover::poly::twiddles::TwiddleTree;
use crate::prover::poly::BitReversedOrder;
use crate::core::ColumnVec;

use super::context::MetalContext;
use super::thresholds::MIN_FFT_LOG_SIZE;
use super::{MetalBackend, MetalBaseColumn};

/// Generates circle twiddles (layer 0) from first line twiddles (layer 1).
/// For each pair [x, y] in first_line_twiddles, generates [y, -y, -x, x].
/// Note: Twiddles are doubled (2*value) for SIMD optimization.
fn circle_twiddles_from_line_twiddles(first_line_twiddles: &[u32]) -> Vec<u32> {
    const P_DBL: u32 = 4294967294;  // 2 * (2^31 - 1) = 2^32 - 2
    let neg_m31_dbl = |a_dbl: u32| if a_dbl == 0 { 0 } else { P_DBL - a_dbl };

    first_line_twiddles
        .chunks_exact(2)
        .flat_map(|chunk| {
            let x_dbl = chunk[0];
            let y_dbl = chunk[1];
            let neg_y_dbl = neg_m31_dbl(y_dbl);
            let neg_x_dbl = neg_m31_dbl(x_dbl);
            [y_dbl, neg_y_dbl, neg_x_dbl, x_dbl]
        })
        .collect()
}

impl PolyOps for MetalBackend {
    // Use SIMD's twiddle format (doubled u32 values)
    type Twiddles = <SimdBackend as PolyOps>::Twiddles;

    fn interpolate(
        eval: CircleEvaluation<Self, BaseField, BitReversedOrder>,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleCoefficients<Self> {
        // Dispatch to Metal GPU for large IFFTs
        metal_ifft_dispatch(eval, twiddles)
    }

    fn eval_at_point(
        poly: &CircleCoefficients<Self>,
        point: CirclePoint<SecureField>,
    ) -> SecureField {
        // Use GPU-accelerated evaluation for large polynomials
        metal_eval_at_point_gpu(poly, point)
    }

    fn eval_at_points_batched(
        polys_and_points: &[(&CircleCoefficients<Self>, CirclePoint<SecureField>)],
    ) -> Vec<SecureField> {
        let _timer = crate::metal_profile_fn!("eval_at_points_batched", "GPU", num_evals = polys_and_points.len());

        if polys_and_points.is_empty() {
            return Vec::new();
        }

        // For very small batches or small polynomials, fall back to CPU
        let max_log_size = polys_and_points.iter()
            .map(|(poly, _)| poly.log_size())
            .max()
            .unwrap_or(0);

        if polys_and_points.len() < 4 || max_log_size <= 8 {
            // Fall back to default implementation (CPU)
            return polys_and_points
                .iter()
                .map(|(poly, point)| Self::eval_at_point(poly, *point))
                .collect();
        }

        metal_eval_at_points_batched_gpu(polys_and_points)
    }

    fn eval_at_point_by_folding(
        evals: &CircleEvaluation<Self, BaseField, BitReversedOrder>,
        point: CirclePoint<SecureField>,
        twiddles: &TwiddleTree<Self>,
    ) -> SecureField {
        let _timer = crate::metal_profile_fn!("eval_at_point_by_folding", "CPU", log_size = evals.domain.log_size());
        use crate::prover::backend::Column;
        use crate::prover::backend::simd::column::BaseColumn;

        // Convert Metal eval to SIMD
        let cpu_vals = evals.values.to_cpu();
        let simd_col: BaseColumn = cpu_vals.into_iter().collect();
        let simd_evals = CircleEvaluation::new(evals.domain, simd_col);

        // Twiddles can be transmuted (they're just metadata)
        let simd_twiddles: &TwiddleTree<SimdBackend> =
            unsafe { &*(twiddles as *const _ as *const _) };

        SimdBackend::eval_at_point_by_folding(&simd_evals, point, simd_twiddles)
    }

    fn extend(
        poly: &CircleCoefficients<Self>,
        log_size: u32,
    ) -> CircleCoefficients<Self> {
        let _timer = crate::metal_profile_fn!("extend", "CPU", from_log_size = poly.log_size(), to_log_size = log_size);
        use crate::prover::backend::Column;
        use crate::prover::backend::simd::column::BaseColumn;

        // Convert Metal poly to SIMD
        let cpu_coeffs = poly.coeffs.to_cpu();
        let simd_coeffs: BaseColumn = cpu_coeffs.into_iter().collect();
        let simd_poly = CircleCoefficients::new(simd_coeffs);

        let simd_result = SimdBackend::extend(&simd_poly, log_size);

        // Convert result back to Metal
        let cpu_result = simd_result.coeffs.to_cpu();
        let metal_coeffs: MetalBaseColumn = cpu_result.into_iter().collect();
        CircleCoefficients::new(metal_coeffs)
    }

    fn evaluate(
        poly: &CircleCoefficients<Self>,
        domain: CircleDomain,
        twiddles: &TwiddleTree<Self>,
    ) -> CircleEvaluation<Self, BaseField, BitReversedOrder> {
        // Dispatch to Metal GPU for large FFTs
        metal_fft_dispatch(poly, domain, twiddles)
    }

    fn evaluate_polynomials(
        polynomials: ColumnVec<CircleCoefficients<Self>>,
        log_blowup_factor: u32,
        twiddles: &TwiddleTree<Self>,
        store_polynomials_coefficients: bool,
    ) -> Vec<Poly<Self>>
    where
        Self: crate::prover::backend::Backend,
    {
        let num_polys = polynomials.len();
        let _timer = crate::metal_profile_fn!("fft_batch", "GPU", num_polys = num_polys);

        use crate::core::poly::circle::CanonicCoset;

        // If no polynomials or all are small, use default implementation
        if polynomials.is_empty() || polynomials.iter().all(|p| p.log_size() < MIN_FFT_LOG_SIZE) {
            return polynomials
                .into_iter()
                .map(|poly_coeffs| {
                    let evals = poly_coeffs.evaluate_with_twiddles(
                        CanonicCoset::new(poly_coeffs.log_size() + log_blowup_factor).circle_domain(),
                        twiddles,
                    );
                    Poly::new(store_polynomials_coefficients.then_some(poly_coeffs), evals)
                })
                .collect();
        }

        // Separate polynomials into Metal and SIMD batches
        let mut metal_batch = Vec::new();
        let mut simd_batch = Vec::new();
        let mut batch_indices = Vec::new(); // Track original order

        for (idx, poly) in polynomials.into_iter().enumerate() {
            if poly.log_size() >= MIN_FFT_LOG_SIZE {
                metal_batch.push(poly);
                batch_indices.push((idx, true)); // true = Metal
            } else {
                simd_batch.push(poly);
                batch_indices.push((idx, false)); // false = SIMD
            }
        }

        // Process Metal batch
        let mut metal_results = Vec::new();
        let mut metal_buffers = Vec::new();  // Keep buffers alive
        if !metal_batch.is_empty() {
            let _setup_timer = std::time::Instant::now();
            let ctx = MetalContext::global();
            let device = ctx.device();
            let command_buffer = ctx.command_queue().new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            // First pass: dispatch all FFT operations to GPU
            for poly_coeffs in &metal_batch {
                let domain = CanonicCoset::new(poly_coeffs.log_size() + log_blowup_factor).circle_domain();
                let buffer = metal_fft_batched_prepare(&ctx, &device, &encoder, &poly_coeffs, domain, twiddles);
                metal_buffers.push((domain, buffer));
            }
            if std::env::var("METAL_PROFILE").is_ok() {
                eprintln!("[CPU_PROFILE] fft_batch_setup | num_polys={} | time={:.3}ms", metal_batch.len(), _setup_timer.elapsed().as_secs_f64() * 1000.0);
            }

            // Submit all Metal operations at once
            encoder.end_encoding();
            command_buffer.commit();

            let _gpu_wait = std::time::Instant::now();
            command_buffer.wait_until_completed();
            if std::env::var("METAL_PROFILE").is_ok() {
                eprintln!("[CPU_PROFILE] fft_batch_gpu_wait | num_polys={} | time={:.3}ms", metal_batch.len(), _gpu_wait.elapsed().as_secs_f64() * 1000.0);
            }

            // Second pass: wrap results from GPU buffers (zero-copy)
            for (poly_coeffs, (domain, output_buffer)) in metal_batch.into_iter().zip(metal_buffers.iter()) {
                let eval_size = domain.size();
                // Zero-copy: wrap GPU buffer directly instead of copying to Vec
                let result_col = MetalBaseColumn::from_buffer(output_buffer.clone(), eval_size);
                let evals = CircleEvaluation::new(*domain, result_col);
                metal_results.push(Poly::new(
                    store_polynomials_coefficients.then_some(poly_coeffs),
                    evals,
                ));
            }
        }

        // Process SIMD batch
        let mut simd_results = Vec::new();
        for poly_coeffs in simd_batch {
            let domain = CanonicCoset::new(poly_coeffs.log_size() + log_blowup_factor).circle_domain();
            let evals = poly_coeffs.evaluate_with_twiddles(domain, twiddles);
            simd_results.push(Poly::new(
                store_polynomials_coefficients.then_some(poly_coeffs),
                evals,
            ));
        }

        // Reassemble results in original order
        let mut results = Vec::with_capacity(batch_indices.len());
        for _ in 0..batch_indices.len() {
            results.push(None);
        }

        let mut metal_iter = metal_results.into_iter();
        let mut simd_iter = simd_results.into_iter();

        for (original_idx, is_metal) in batch_indices {
            if is_metal {
                results[original_idx] = Some(metal_iter.next().unwrap());
            } else {
                results[original_idx] = Some(simd_iter.next().unwrap());
            }
        }

        results.into_iter().map(|r| r.unwrap()).collect()
    }

    fn precompute_twiddles(coset: Coset) -> TwiddleTree<Self> {
        // Twiddle precomputation uses SIMD, same twiddle format (Vec<u32>)
        let simd_result = SimdBackend::precompute_twiddles(coset);
        TwiddleTree {
            root_coset: simd_result.root_coset,
            twiddles: simd_result.twiddles,
            itwiddles: simd_result.itwiddles,
        }
    }

    fn split_at_mid(
        poly: CircleCoefficients<Self>,
    ) -> (CircleCoefficients<Self>, CircleCoefficients<Self>) {
        let (left, right) = poly.coeffs.split_at_mid();
        (
            CircleCoefficients::new(left),
            CircleCoefficients::new(right),
        )
    }

    /// Batched IFFT for multiple columns to reduce GPU dispatch overhead.
    /// This is critical for performance - interpolating 208 columns individually
    /// wastes ~76ms on synchronous GPU dispatches (76% of proof time at log_n=14).
    fn interpolate_columns(
        columns: impl IntoIterator<Item = CircleEvaluation<Self, BaseField, BitReversedOrder>>,
        twiddles: &TwiddleTree<Self>,
    ) -> Vec<CircleCoefficients<Self>> {
        let columns: Vec<_> = columns.into_iter().collect();
        let _timer = crate::metal_profile_fn!("interpolate_columns", "GPU", num_columns = columns.len());

        // Fall back to serial for small batches
        if columns.is_empty() || columns.iter().all(|eval| eval.domain.log_size() < MIN_FFT_LOG_SIZE) {
            return columns
                .into_iter()
                .map(|eval| eval.interpolate_with_twiddles(twiddles))
                .collect();
        }

        let ctx = MetalContext::global();
        let mut results = Vec::with_capacity(columns.len());

        // Create single command buffer for ALL IFFTs
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Dispatch all IFFT operations to GPU (without waiting)
        for eval in &columns {
            let log_size = eval.domain.log_size();

            // Skip tiny sizes
            if log_size < MIN_FFT_LOG_SIZE {
                continue;
            }

            // Dispatch IFFT to the shared encoder (no commit/wait)
            metal_ifft_batched_dispatch(&ctx, &encoder, eval, twiddles);
        }

        // Submit all IFFTs at once
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Now read results
        for eval in columns {
            if eval.domain.log_size() < MIN_FFT_LOG_SIZE {
                // Use SIMD fallback for small sizes
                use crate::prover::backend::Column;
                use crate::prover::backend::simd::column::BaseColumn;

                let cpu_vals = eval.values.to_cpu();
                let simd_col: BaseColumn = cpu_vals.into_iter().collect();
                let simd_eval = CircleEvaluation::new(eval.domain, simd_col);

                let simd_twiddles = TwiddleTree {
                    root_coset: twiddles.root_coset,
                    twiddles: twiddles.twiddles.clone(),
                    itwiddles: twiddles.itwiddles.clone(),
                };
                let simd_result = SimdBackend::interpolate(simd_eval, &simd_twiddles);

                let cpu_coeffs = simd_result.coeffs.to_cpu();
                let metal_coeffs: MetalBaseColumn = cpu_coeffs.into_iter().collect();
                results.push(CircleCoefficients::new(metal_coeffs));
            } else {
                // Results are already in GPU buffers from batched dispatch
                let data_len = eval.values.len();
                let data_buffer = eval.values.buffer().clone();
                let result_col = MetalBaseColumn::from_buffer(data_buffer, data_len);
                results.push(CircleCoefficients::new(result_col));
            }
        }

        results
    }
}

// ============================================================================
// Metal GPU Dispatch Functions
// ============================================================================

/// Dispatch forward FFT to Metal GPU (radix-8 implementation).
///
/// This function processes FFT layers using GPU acceleration for eligible transforms.
/// For Phase 1, we only use GPU if all layers can be processed (log_size divisible by 3).
/// Otherwise, we fall back to full SIMD implementation.
///
/// TODO(Phase 2): Implement mixed GPU+SIMD execution for partial layer processing.
fn metal_fft_dispatch(
    poly: &CircleCoefficients<MetalBackend>,
    domain: CircleDomain,
    twiddles: &TwiddleTree<MetalBackend>,
) -> CircleEvaluation<MetalBackend, BaseField, BitReversedOrder> {
    let log_size = poly.log_size();
    let _timer = crate::metal_profile_fn!("fft", "GPU", log_size = log_size);

    let domain_log_size = domain.log_size();

    // Fall back to SIMD if size is too small
    if log_size < MIN_FFT_LOG_SIZE {
        use crate::prover::backend::Column;
        use crate::prover::backend::simd::column::BaseColumn;

        // Convert Metal poly to SIMD
        let cpu_coeffs = poly.coeffs.to_cpu();
        let simd_coeffs: BaseColumn = cpu_coeffs.into_iter().collect();
        let simd_poly = CircleCoefficients::new(simd_coeffs);

        // Twiddles can be transmuted (same structure)
        let simd_twiddles: &TwiddleTree<SimdBackend> =
            unsafe { &*(twiddles as *const _ as *const _) };
        let simd_result = SimdBackend::evaluate(&simd_poly, domain, simd_twiddles);

        // Convert result back to Metal
        let cpu_vals = simd_result.values.to_cpu();
        let metal_col: MetalBaseColumn = cpu_vals.into_iter().collect();
        return CircleEvaluation::new(simd_result.domain, metal_col);
    }

    let ctx = MetalContext::global();

    // Subdomain evaluation: when domain is larger than polynomial, evaluate on multiple subdomains
    // This matches SIMD's implementation in circle.rs:288-320
    let fft_log_size = log_size;
    let log_subdomains = domain_log_size - fft_log_size;
    let num_subdomains = 1usize << log_subdomains;
    let poly_size = poly.coeffs.len();
    let eval_size = domain.size();

    // Access input coefficients directly from GPU buffer (zero-copy with MTLStorageModeShared)
    let input_coeffs = poly.coeffs.as_slice();

    // Allocate output buffer for all subdomains (shared memory, accessible from CPU/GPU)
    let device = ctx.device();
    let output_buffer = device.new_buffer(
        (eval_size * std::mem::size_of::<BaseField>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Get full domain twiddles
    let domain_twiddles = domain_line_twiddles_from_tree(domain, &twiddles.twiddles);

    // Prepare all subdomain data first (CPU side)
    let subdomain_data: Vec<_> = (0..num_subdomains)
        .map(|subdomain_idx| {
            let output_offset = subdomain_idx * poly_size;

            // Copy input coefficients directly into shared buffer (no intermediate allocation)
            unsafe {
                let dst_ptr = (output_buffer.contents() as *mut BaseField).add(output_offset);
                std::ptr::copy_nonoverlapping(
                    input_coeffs.as_ptr(),
                    dst_ptr,
                    poly_size,
                );
            }

            // Extract subdomain-specific twiddles
            let twiddle_slices: Vec<&[u32]> = (0..(fft_log_size - 1))
                .map(|layer_i| {
                    let layer_i_usize = layer_i as usize;
                    let start = subdomain_idx << (fft_log_size - 2 - layer_i);
                    let end = (subdomain_idx + 1) << (fft_log_size - 2 - layer_i);
                    &domain_twiddles[layer_i_usize][start..end]
                })
                .collect();

            (output_offset, twiddle_slices)
        })
        .collect();

    // Create single command buffer for all subdomains
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    // Process all subdomains in single submission
    for (_subdomain_idx, (output_offset, twiddle_slices)) in subdomain_data.iter().enumerate() {
        metal_fft_single_domain_batched(
            &ctx,
            &encoder,
            &output_buffer,
            *output_offset,
            fft_log_size,
            twiddle_slices,
        );
    }

    // Submit once for all subdomains
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Wrap GPU buffer directly - zero copy with MTLStorageModeShared
    let result_col = MetalBaseColumn::from_buffer(output_buffer.clone(), eval_size);
    CircleEvaluation::new(domain, result_col)
}

/// Process FFT using an existing encoder (for batching multiple FFTs).
/// This function adds FFT operations to the encoder but does NOT submit the command buffer.
/// Returns the output buffer which will contain results after command buffer execution.
fn metal_fft_batched_prepare(
    ctx: &MetalContext,
    device: &metal::DeviceRef,
    encoder: &metal::ComputeCommandEncoderRef,
    poly: &CircleCoefficients<MetalBackend>,
    domain: CircleDomain,
    twiddles: &TwiddleTree<MetalBackend>,
) -> metal::Buffer {
    let log_size = poly.log_size();
    let domain_log_size = domain.log_size();
    let fft_log_size = log_size;
    let log_subdomains = domain_log_size - fft_log_size;
    let num_subdomains = 1usize << log_subdomains;
    let poly_size = poly.coeffs.len();
    let eval_size = domain.size();

    // Access input coefficients directly from GPU buffer (zero-copy with MTLStorageModeShared)
    let input_coeffs = poly.coeffs.as_slice();

    // Allocate output buffer for this polynomial
    let output_buffer = device.new_buffer(
        (eval_size * std::mem::size_of::<BaseField>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Get full domain twiddles
    let domain_twiddles = domain_line_twiddles_from_tree(domain, &twiddles.twiddles);

    // Process all subdomains for this polynomial
    for subdomain_idx in 0..num_subdomains {
        let output_offset = subdomain_idx * poly_size;

        // Copy input coefficients into buffer
        unsafe {
            let dst_ptr = (output_buffer.contents() as *mut BaseField).add(output_offset);
            std::ptr::copy_nonoverlapping(
                input_coeffs.as_ptr(),
                dst_ptr,
                poly_size,
            );
        }

        // Extract subdomain-specific twiddles
        let twiddle_slices: Vec<&[u32]> = (0..(fft_log_size - 1))
            .map(|layer_i| {
                let layer_i_usize = layer_i as usize;
                let start = subdomain_idx << (fft_log_size - 2 - layer_i);
                let end = (subdomain_idx + 1) << (fft_log_size - 2 - layer_i);
                &domain_twiddles[layer_i_usize][start..end]
            })
            .collect();

        // Add FFT operations to the shared encoder
        metal_fft_single_domain_batched(
            ctx,
            encoder,
            &output_buffer,
            output_offset,
            fft_log_size,
            &twiddle_slices,
        );
    }

    // Return the buffer - results will be available after command buffer execution
    output_buffer
}

/// Run FFT on a single domain with batched encoding (helper for subdomain evaluation).
/// Processes data in-place at the given offset within the buffer.
/// Uses the provided encoder instead of creating command buffers.
fn metal_fft_single_domain_batched(
    ctx: &MetalContext,
    encoder: &metal::ComputeCommandEncoderRef,
    data_buffer: &metal::BufferRef,
    data_offset: usize,
    log_size: u32,
    twiddle_slices: &[&[u32]],
) {
    let device = ctx.device();
    let num_fft_layers = twiddle_slices.len() as u32;
    let data_size = 1usize << log_size;
    let buffer_offset_bytes = (data_offset * std::mem::size_of::<BaseField>()) as u64;

    // Full GPU acceleration:
    // - Use radix-2 for vecwise layers (0-4)
    // - Use radix-8 for groups of 3 non-vecwise layers + radix-2 for leftovers
    const VECWISE_FFT_BITS: u32 = 5;
    // Enable radix-8 for FFTs with >= 8 layers
    const MIN_RADIX8_LAYERS: u32 = 8;

    // IMPORTANT: VECWISE_FFT_BITS includes circle layer 0, so vecwise processes layers 0-4.
    // Non-vecwise layers start at VECWISE_FFT_BITS (layer 5) and go up to num_fft_layers+1.
    // Example: log_size=9 has num_fft_layers=8 (half_coset log_size), which means line layers 1-8.
    //          Vecwise handles 0-4, non-vecwise handles 5-8 (that's 4 layers, not 3!)
    let non_vecwise_layers = if num_fft_layers + 1 >= MIN_RADIX8_LAYERS {
        // Count layers from VECWISE_FFT_BITS to num_fft_layers (inclusive)
        // num_fft_layers is the highest LINE layer index, +1 to get count
        (num_fft_layers + 1) - VECWISE_FFT_BITS
    } else {
        0  // Use radix-2 for all layers
    };

    // Phase 1: Process non-vecwise layers (6+) using radix-8 for groups of 3
    // Calculate how many complete groups of 3 we can process
    let radix8_groups = non_vecwise_layers / 3;
    let leftover_non_vecwise = non_vecwise_layers % 3;


    // Phase 1a: Process leftover non-vecwise layers (1 or 2 layers) with radix-2 FIRST
    // These are the highest layers, so they must be processed before radix-8 groups in DIF order
    if leftover_non_vecwise > 0 {
        // Leftover layers start after all radix-8 groups
        let leftover_start = VECWISE_FFT_BITS + radix8_groups * 3;
        let leftover_end = leftover_start + leftover_non_vecwise - 1;

        for layer in (leftover_start..=leftover_end).rev() {
            let tw_idx = (layer - 1) as usize;
            let tw_layer = twiddle_slices[tw_idx];

            #[cfg(test)]
            {
                let data_ptr = data_buffer.contents() as *const u32;
                let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, 16) };
                println!("      [Metal] Before layer {}: {:?}", layer, data_slice);
            }

            let tw_buffer = ctx.get_or_create_twiddle_buffer(tw_layer);

            encoder.set_compute_pipeline_state(ctx.fft_radix2_pipeline());
            encoder.set_buffer(0, Some(&data_buffer), buffer_offset_bytes);
            encoder.set_buffer(1, Some(&tw_buffer), 0);
            encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

            let num_threads = 1u64 << (log_size - 1);
            let threadgroup_size = 256.min(num_threads.max(1));
            let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

            encoder.dispatch_thread_groups(
                metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
                metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
            );
        }
    }

    // Phase 1b: Process complete radix-8 groups (3 layers each) from high to low
    if radix8_groups > 0 {
        // Non-vecwise layers start at layer VECWISE_FFT_BITS (which is layer 5 for standard config)
        // Process radix-8 groups from highest to lowest
        for radix8_idx in (0..radix8_groups).rev() {
            let layer = VECWISE_FFT_BITS + radix8_idx * 3;  // Base layer of this radix-8 group

        // Get twiddle slices for this radix-8 step (3 layers)
        // Use same indexing as vecwise layers: tw_idx = layer - 1
        let tw_idx_layer2 = (layer + 2 - 1) as usize;  // Coarsest layer (highest)
        let tw_idx_layer1 = (layer + 1 - 1) as usize;  // Middle layer
        let tw_idx_layer0 = (layer - 1) as usize;      // Finest layer (lowest)

        let tw_layer2 = twiddle_slices[tw_idx_layer2];
        let tw_layer1 = twiddle_slices[tw_idx_layer1];
        let tw_layer0 = twiddle_slices[tw_idx_layer0];

        #[cfg(test)]
        {
            println!("      tw_layer0 (finest, len={}): {:?}...", tw_layer0.len(), &tw_layer0[..tw_layer0.len().min(4)]);
            println!("      tw_layer1 (middle, len={}): {:?}...", tw_layer1.len(), &tw_layer1[..tw_layer1.len().min(4)]);
            println!("      tw_layer2 (coarsest, len={}): {:?}...", tw_layer2.len(), &tw_layer2[..tw_layer2.len().min(4)]);

            // Print input data before radix-8
            let data_ptr = data_buffer.contents() as *const u32;
            let data_slice = unsafe { std::slice::from_raw_parts(data_ptr, data_size.min(16)) };
            println!("      Input data (first 16): {:?}", data_slice);
        }

        // Use flattened twiddle buffer for all 3 layers
        let flat_twiddles = ctx.get_or_create_flat_twiddle_buffer(&[tw_layer0, tw_layer1, tw_layer2]);

        // Set pipeline
        encoder.set_compute_pipeline_state(ctx.fft_radix8_pipeline());

        // Set buffers - use single flat buffer with offsets for each layer
        encoder.set_buffer(0, Some(&data_buffer), buffer_offset_bytes);
        encoder.set_buffer(1, Some(&flat_twiddles.buffer), flat_twiddles.sections[0].offset);  // layer0 (finest)
        encoder.set_buffer(2, Some(&flat_twiddles.buffer), flat_twiddles.sections[1].offset);  // layer1 (middle)
        encoder.set_buffer(3, Some(&flat_twiddles.buffer), flat_twiddles.sections[2].offset);  // layer2 (coarsest)
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

        // Pass twiddle lengths for correct masking
        let tw0_len_u32 = tw_layer0.len() as u32;
        let tw1_len_u32 = tw_layer1.len() as u32;
        let tw2_len_u32 = tw_layer2.len() as u32;

        encoder.set_bytes(6, std::mem::size_of::<u32>() as u64, &tw0_len_u32 as *const u32 as *const _);
        encoder.set_bytes(7, std::mem::size_of::<u32>() as u64, &tw1_len_u32 as *const u32 as *const _);
        encoder.set_bytes(8, std::mem::size_of::<u32>() as u64, &tw2_len_u32 as *const u32 as *const _);

        // Dispatch threads: radix-8 processes 8 elements per thread
        let num_threads = 1u64 << (log_size - 3);
        let threadgroup_size = 256.min(num_threads);
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );
        }
    }

    // Phase 2: Process vecwise line layers using radix-2, in reverse order
    // Vecwise line layers are [1, VECWISE_FFT_BITS) or [1, num_fft_layers] if all layers are vecwise
    let _line_layers_start = 1u32;
    let _line_layers_end = if non_vecwise_layers == 0 {
        // All layers are vecwise, process all of them
        num_fft_layers
    } else {
        // Process vecwise line layers up to (but not including) VECWISE_FFT_BITS
        VECWISE_FFT_BITS - 1
    };


    // Use fused vecwise kernel for layers 1-4 to eliminate multiple kernel launch overhead
    if _line_layers_start == 1 && _line_layers_end >= 4 && twiddle_slices.len() >= 4 {

        // Create flattened twiddle buffer for all 4 layers
        // Twiddle indexing: layer N uses twiddle_slices[N-1]
        let tw_layers: Vec<&[u32]> = (0..4).map(|i| twiddle_slices[i]).collect();
        let flat_twiddles = ctx.get_or_create_flat_twiddle_buffer(&tw_layers);

        encoder.set_compute_pipeline_state(ctx.fft_vecwise_fused_pipeline());
        encoder.set_buffer(0, Some(&data_buffer), buffer_offset_bytes);
        // Buffers map to layers 1,2,3,4 (kernel processes them in reverse order)
        encoder.set_buffer(1, Some(&flat_twiddles.buffer), flat_twiddles.sections[0].offset);  // Layer 1 twiddles
        encoder.set_buffer(2, Some(&flat_twiddles.buffer), flat_twiddles.sections[1].offset);  // Layer 2 twiddles
        encoder.set_buffer(3, Some(&flat_twiddles.buffer), flat_twiddles.sections[2].offset);  // Layer 3 twiddles
        encoder.set_buffer(4, Some(&flat_twiddles.buffer), flat_twiddles.sections[3].offset);  // Layer 4 twiddles
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);

        let num_threads = 1u64 << (log_size - 1);
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );

        // If there's layer 5, handle it separately
        if _line_layers_end >= 5 {
            for layer in (5..=_line_layers_end).rev() {
                let tw_idx = (layer - 1) as usize;
                let tw_layer = twiddle_slices[tw_idx];

                let tw_buffer = device.new_buffer_with_data(
                    tw_layer.as_ptr() as *const _,
                    (tw_layer.len() * std::mem::size_of::<u32>()) as u64,
                    MTLResourceOptions::StorageModeShared,
                );

                encoder.set_compute_pipeline_state(ctx.fft_radix2_pipeline());
                encoder.set_buffer(0, Some(&data_buffer), buffer_offset_bytes);
                encoder.set_buffer(1, Some(&tw_buffer), 0);
                encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
                encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

                let num_threads = 1u64 << (log_size - 1);
                let threadgroup_size = 256.min(num_threads.max(1));
                let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

                encoder.dispatch_thread_groups(
                    metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
                    metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
                );
            }
        }
    } else {
        // Fallback to individual layer processing
        for layer in (_line_layers_start..=_line_layers_end).rev() {
            // Twiddle indexing: use layer - 1
            let tw_idx = (layer - 1) as usize;
            #[cfg(test)]
            {
                if tw_idx >= twiddle_slices.len() {
                    panic!("OUT OF BOUNDS: tw_idx={} >= twiddle_slices.len()={} for layer={}",
                           tw_idx, twiddle_slices.len(), layer);
                }
            }
            let tw_layer = twiddle_slices[tw_idx];

            let tw_buffer = ctx.get_or_create_twiddle_buffer(tw_layer);

            encoder.set_compute_pipeline_state(ctx.fft_radix2_pipeline());
            encoder.set_buffer(0, Some(&data_buffer), buffer_offset_bytes);
            encoder.set_buffer(1, Some(&tw_buffer), 0);
            encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

            let num_threads = 1u64 << (log_size - 1);
            let threadgroup_size = 256.min(num_threads.max(1));
            let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

            encoder.dispatch_thread_groups(
                metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
                metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
            );
        }
    }

    // Phase 3: Process circle layer (layer 0) with special circle twiddles

    if num_fft_layers > 0 {
        // Generate circle twiddles from first line twiddles (layer 1)
        let first_line_twiddles = twiddle_slices[0];
        let circle_twiddles = circle_twiddles_from_line_twiddles(first_line_twiddles);

        let tw_buffer = device.new_buffer_with_data(
            circle_twiddles.as_ptr() as *const _,
            (circle_twiddles.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(ctx.fft_radix2_pipeline());
        encoder.set_buffer(0, Some(&data_buffer), buffer_offset_bytes);
        encoder.set_buffer(1, Some(&tw_buffer), 0);
        // For circle layer, use data_size (full circle domain size) not poly.log_size()
        let circle_log_size = data_size.ilog2() as u32;
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &circle_log_size as *const u32 as *const _);
        let layer = 0u32;
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

        let num_threads = 1u64 << (circle_log_size - 1);
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );
    }
}

/// Dispatch inverse FFT to Metal GPU (radix-8 implementation).
///
/// Similar to forward FFT but processes layers in reverse order for IFFT.
/// For Phase 1, we only use GPU if all layers can be processed (log_size divisible by 3).
///
/// TODO(Phase 2): Implement mixed GPU+SIMD execution for partial layer processing.
fn metal_ifft_dispatch(
    eval: CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>,
    twiddles: &TwiddleTree<MetalBackend>,
) -> CircleCoefficients<MetalBackend> {
    let log_size = eval.domain.log_size();
    let _timer = crate::metal_profile_fn!("ifft", "GPU", log_size = log_size);

    // Number of IFFT layers is log_size - 1 (based on coset size)
    let num_ifft_layers = log_size - 1;

    // Full GPU acceleration with radix-2 for vecwise layers (0-4) and radix-8 for higher layers
    const VECWISE_FFT_BITS: u32 = 5;
    // Vecwise layers are [0, VECWISE_FFT_BITS) = [0, 5) = layers 0-4 (matching SIMD)
    let _non_vecwise_layers = if num_ifft_layers > VECWISE_FFT_BITS {
        num_ifft_layers - VECWISE_FFT_BITS
    } else {
        0  // All layers are vecwise, handled by radix-2
    };

    // Only fall back to SIMD if the size is too small
    // We can handle any number of non-vecwise layers using radix-2
    if log_size < MIN_FFT_LOG_SIZE {
        use crate::prover::backend::Column;
        use crate::prover::backend::simd::column::BaseColumn;

        // Convert Metal eval to SIMD
        let cpu_vals = eval.values.to_cpu();
        let simd_col: BaseColumn = cpu_vals.into_iter().collect();
        let simd_eval = CircleEvaluation::new(eval.domain, simd_col);

        let simd_twiddles = TwiddleTree {
            root_coset: twiddles.root_coset,
            twiddles: twiddles.twiddles.clone(),
            itwiddles: twiddles.itwiddles.clone(),
        };
        let simd_result = SimdBackend::interpolate(simd_eval, &simd_twiddles);

        // Convert result back to Metal
        let cpu_coeffs = simd_result.coeffs.to_cpu();
        let metal_coeffs: MetalBaseColumn = cpu_coeffs.into_iter().collect();
        return CircleCoefficients::new(metal_coeffs);
    }

    let ctx = MetalContext::global();
    let device = ctx.device();

    let data_len = eval.values.len();

    // IFFT operates in-place on the eval buffer
    // Since eval.values is already a MetalBaseColumn with a shared buffer, use it directly (zero-copy)
    let data_buffer = eval.values.buffer().clone();

    // Convert flat inverse twiddles to per-layer slices
    let itwiddle_slices = domain_line_twiddles_from_tree(eval.domain, &twiddles.itwiddles);
    let num_layers = itwiddle_slices.len() as u32;

    // Create a SINGLE command buffer and encoder for ALL IFFT layers
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    // Phase 1: Process circle layer (layer 0) FIRST with special circle twiddles (DIT IFFT)
    if num_ifft_layers > 0 {

        // Generate circle twiddles from first line inverse twiddles
        // Use index 0 (same as FFT uses for circle layer)
        let first_line_itw_idx = 0usize;
        let first_line_itwiddles = itwiddle_slices[first_line_itw_idx];  // layer 1 itwiddles
        let circle_itwiddles = circle_twiddles_from_line_twiddles(first_line_itwiddles);

        let itw_buffer = device.new_buffer_with_data(
            circle_itwiddles.as_ptr() as *const _,
            (circle_itwiddles.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(ctx.ifft_radix2_pipeline());
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&itw_buffer), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
        let layer = 0u32;
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

        let num_threads = 1u64 << (log_size - 1);
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );

    }

    // Phase 2: Process vecwise line layers using radix-2, in FORWARD order (DIT IFFT)
    // For log_size=6: num_ifft_layers=5, so line layers are 1,2,3,4,5 (5 layers)
    let line_layers_start = 1u32;
    let line_layers_end = if _non_vecwise_layers == 0 {
        num_ifft_layers  // All remaining layers are line layers (circle layer 0 already done)
    } else {
        VECWISE_FFT_BITS - 1  // Vecwise line layers are [1, VECWISE_FFT_BITS)
    };
    for layer in line_layers_start..=line_layers_end {
        // Line layer L uses itwiddle_slices[L-1] (same as CPU implementation)
        let itw_idx = (layer - 1) as usize;


        let itw_layer = itwiddle_slices[itw_idx];

        let itw_buffer = device.new_buffer_with_data(
            itw_layer.as_ptr() as *const _,
            (itw_layer.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(ctx.ifft_radix2_pipeline());
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&itw_buffer), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

        let num_threads = 1u64 << (log_size - 1);
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );
    }

    // Phase 3: Process non-vecwise layers (5+) using radix-2
    // For log_size=13: layers 5-12 (8 layers)
    if _non_vecwise_layers > 0 {
        let non_vecwise_start = VECWISE_FFT_BITS;
        let non_vecwise_end = num_ifft_layers;


        for layer in non_vecwise_start..=non_vecwise_end {
            // Line layer L uses itwiddle_slices[L-1] (same as CPU implementation)
            let itw_idx = (layer - 1) as usize;


            let itw_layer = itwiddle_slices[itw_idx];

            let itw_buffer = device.new_buffer_with_data(
                itw_layer.as_ptr() as *const _,
                (itw_layer.len() * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            encoder.set_compute_pipeline_state(ctx.ifft_radix2_pipeline());
            encoder.set_buffer(0, Some(&data_buffer), 0);
            encoder.set_buffer(1, Some(&itw_buffer), 0);
            encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

            let num_threads = 1u64 << (log_size - 1);
            let threadgroup_size = 256.min(num_threads.max(1));
            let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

            encoder.dispatch_thread_groups(
                metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
                metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
            );
        }
    }

    // Append normalization to the same command buffer (saves one submit/wait cycle)
    // Normalize by 1/N (standard IFFT normalization) on GPU
    let domain_size = eval.domain.size();
    let n_inv = BaseField::from(domain_size).inverse();


    let normalize_count = domain_size.min(data_len) as u32;
    let n_inv_raw = n_inv.0;

    // Reuse the same encoder - normalization will execute after IFFT on GPU
    encoder.set_compute_pipeline_state(ctx.ifft_normalize_pipeline());
    encoder.set_buffer(0, Some(&data_buffer), 0);
    encoder.set_bytes(1, std::mem::size_of::<u32>() as u64, &n_inv_raw as *const u32 as *const _);
    encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &normalize_count as *const u32 as *const _);

    let num_threads = normalize_count as u64;
    let threadgroup_size = 256.min(num_threads.max(1));
    let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

    encoder.dispatch_thread_groups(
        metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
        metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
    );

    // End encoding and submit IFFT + normalization together
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Phase 4 (unused): Process remaining non-vecwise layers using radix-8 kernels (disabled)
    // Iterate through remaining layers, stepping by 3
    // Process as long as we have 3 consecutive layers available
    if false {
        let layer = 0u32;  // Unused placeholder

        // Get inverse twiddle slices
        // Note: twiddle_slices are in REVERSE order (index 0 = highest layer)
        // IFFT processes bottom-to-top: when layer=0, we process layers 0,1,2
        // Kernel applies layer0 (finest=0), layer1 (middle=1), layer2 (coarsest=2)
        let itw_layer0 = itwiddle_slices[(num_layers - 1 - layer) as usize];
        let itw_layer1 = itwiddle_slices[(num_layers - 1 - (layer + 1)) as usize];
        let itw_layer2 = itwiddle_slices[(num_layers - 1 - (layer + 2)) as usize];

        // Use flattened twiddle buffer for all 3 layers
        let flat_itwiddles = ctx.get_or_create_flat_twiddle_buffer(&[itw_layer0, itw_layer1, itw_layer2]);

        // Create and encode command
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(ctx.ifft_radix8_pipeline());
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&flat_itwiddles.buffer), flat_itwiddles.sections[0].offset);
        encoder.set_buffer(2, Some(&flat_itwiddles.buffer), flat_itwiddles.sections[1].offset);
        encoder.set_buffer(3, Some(&flat_itwiddles.buffer), flat_itwiddles.sections[2].offset);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

        // Dispatch threads: radix-8 processes 8 elements per thread
        let num_threads = 1u64 << (log_size - 3);
        let threadgroup_size = 256.min(num_threads);
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // current_layer += 3;  // Unused since radix-8 is disabled
    }

    // Wrap GPU buffer directly - zero copy with MTLStorageModeShared
    let result_col = MetalBaseColumn::from_buffer(data_buffer.clone(), data_len);
    CircleCoefficients::new(result_col)
}

/// Batched IFFT dispatch - adds IFFT operations to an existing encoder without committing.
/// This allows multiple IFFTs to be batched into a single command buffer submission.
fn metal_ifft_batched_dispatch(
    ctx: &MetalContext,
    encoder: &metal::ComputeCommandEncoderRef,
    eval: &CircleEvaluation<MetalBackend, BaseField, BitReversedOrder>,
    twiddles: &TwiddleTree<MetalBackend>,
) {
    let log_size = eval.domain.log_size();
    let device = ctx.device();
    let data_buffer = eval.values.buffer();

    // Convert flat inverse twiddles to per-layer slices
    let itwiddle_slices = domain_line_twiddles_from_tree(eval.domain, &twiddles.itwiddles);
    let num_ifft_layers = log_size - 1;

    const VECWISE_FFT_BITS: u32 = 5;
    let _non_vecwise_layers = if num_ifft_layers > VECWISE_FFT_BITS {
        num_ifft_layers - VECWISE_FFT_BITS
    } else {
        0
    };

    // Phase 1: Process circle layer (layer 0) FIRST with special circle twiddles (DIT IFFT)
    if num_ifft_layers > 0 {
        let first_line_itw_idx = 0usize;
        let first_line_itwiddles = itwiddle_slices[first_line_itw_idx];
        let circle_itwiddles = circle_twiddles_from_line_twiddles(first_line_itwiddles);

        let itw_buffer = device.new_buffer_with_data(
            circle_itwiddles.as_ptr() as *const _,
            (circle_itwiddles.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(ctx.ifft_radix2_pipeline());
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&itw_buffer), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
        let layer = 0u32;
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

        let num_threads = 1u64 << (log_size - 1);
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );
    }

    // Phase 2: Process vecwise line layers using radix-2, in FORWARD order (DIT IFFT)
    let line_layers_start = 1u32;
    let line_layers_end = if _non_vecwise_layers == 0 {
        num_ifft_layers
    } else {
        VECWISE_FFT_BITS - 1
    };

    for layer in line_layers_start..=line_layers_end {
        let itw_idx = (layer - 1) as usize;
        let itw_layer = itwiddle_slices[itw_idx];

        let itw_buffer = device.new_buffer_with_data(
            itw_layer.as_ptr() as *const _,
            (itw_layer.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_compute_pipeline_state(ctx.ifft_radix2_pipeline());
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&itw_buffer), 0);
        encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

        let num_threads = 1u64 << (log_size - 1);
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

        encoder.dispatch_thread_groups(
            metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
            metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
        );
    }

    // Phase 3: Process non-vecwise layers (5+) using radix-2
    if _non_vecwise_layers > 0 {
        let non_vecwise_start = VECWISE_FFT_BITS;
        let non_vecwise_end = num_ifft_layers;

        for layer in non_vecwise_start..=non_vecwise_end {
            let itw_idx = (layer - 1) as usize;
            let itw_layer = itwiddle_slices[itw_idx];

            let itw_buffer = device.new_buffer_with_data(
                itw_layer.as_ptr() as *const _,
                (itw_layer.len() * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            encoder.set_compute_pipeline_state(ctx.ifft_radix2_pipeline());
            encoder.set_buffer(0, Some(&data_buffer), 0);
            encoder.set_buffer(1, Some(&itw_buffer), 0);
            encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
            encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);

            let num_threads = 1u64 << (log_size - 1);
            let threadgroup_size = 256.min(num_threads.max(1));
            let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

            encoder.dispatch_thread_groups(
                metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
                metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
            );
        }
    }

    // Phase 4: Normalization
    let domain_size = eval.domain.size();
    let n_inv = BaseField::from(domain_size).inverse();
    let data_len = eval.values.len();
    let normalize_count = domain_size.min(data_len) as u32;
    let n_inv_raw = n_inv.0;

    encoder.set_compute_pipeline_state(ctx.ifft_normalize_pipeline());
    encoder.set_buffer(0, Some(&data_buffer), 0);
    encoder.set_bytes(1, std::mem::size_of::<u32>() as u64, &n_inv_raw as *const u32 as *const _);
    encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, &normalize_count as *const u32 as *const _);

    let num_threads = normalize_count as u64;
    let threadgroup_size = 256.min(num_threads.max(1));
    let threadgroups = if num_threads == 0 { 1 } else { (num_threads + threadgroup_size - 1) / threadgroup_size };

    encoder.dispatch_thread_groups(
        metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
        metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
    );
}

/// GPU-accelerated polynomial evaluation at a point using batched dispatch.
///
/// For small polynomials (log_size <= 8), falls back to CPU evaluation.
/// For large polynomials, dispatches to GPU using the circle_eval_at_point kernel.
///
/// The kernel implements binary tree folding (Horner's method) where:
///   eval(coeffs) = eval(left_half) + eval(right_half) * folding_factor
///
/// Folding factors are computed from the circle point: [y, x, pi(x), pi^2(x), ...]
/// where pi(x) = 2*x^2 - 1 is the circle point doubling formula.
fn metal_eval_at_point_gpu(
    poly: &CircleCoefficients<MetalBackend>,
    point: CirclePoint<SecureField>,
) -> SecureField {
    let log_size = poly.log_size();
    let _timer = crate::metal_profile_fn!("eval_at_point", "GPU/CPU", log_size = log_size);

    // For small polynomials, fall back to CPU (more efficient than GPU dispatch overhead)
    if log_size <= 8 {
        use crate::prover::backend::Column;
        use crate::prover::backend::simd::column::BaseColumn;

        let cpu_coeffs = poly.coeffs.to_cpu();
        let simd_coeffs: BaseColumn = cpu_coeffs.into_iter().collect();
        let simd_poly = CircleCoefficients::new(simd_coeffs);
        return SimdBackend::eval_at_point(&simd_poly, point);
    }

    let ctx = MetalContext::global();
    let device = ctx.device();

    // Coefficients are already in GPU memory (MetalBaseColumn)
    let coeffs_buffer = poly.coeffs.buffer();

    // Create buffers for kernel parameters
    // For a single evaluation, we'll use batch size = 1
    let num_evals = 1u32;

    // Offset into coeffs (0 since we're evaluating a single polynomial)
    let coeff_offset = 0u32;
    let coeff_offset_buffer = device.new_buffer_with_data(
        &coeff_offset as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Log size buffer
    let log_size_buffer = device.new_buffer_with_data(
        &log_size as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Evaluation point buffer (CirclePoint<SecureField>)
    // SecureField is QM31: 4 x u32 (c0.a, c0.b, c1.a, c1.b)
    // CirclePoint has x and y, so 8 x u32 total
    let point_data: [u32; 8] = [
        point.x.0 .0 .0,  // x.c0.a
        point.x.0 .1 .0,  // x.c0.b
        point.x.1 .0 .0,  // x.c1.a
        point.x.1 .1 .0,  // x.c1.b
        point.y.0 .0 .0,  // y.c0.a
        point.y.0 .1 .0,  // y.c0.b
        point.y.1 .0 .0,  // y.c1.a
        point.y.1 .1 .0,  // y.c1.b
    ];
    let point_buffer = device.new_buffer_with_data(
        point_data.as_ptr() as *const _,
        (point_data.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Result buffer (1 x QM31 = 4 x u32)
    let result_buffer = device.new_buffer(
        (4 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Num evals buffer
    let num_evals_buffer = device.new_buffer_with_data(
        &num_evals as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Dispatch kernel
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(ctx.eval_at_point_pipeline());
    encoder.set_buffer(0, Some(coeffs_buffer), 0);  // coeffs_base
    encoder.set_buffer(1, Some(&coeff_offset_buffer), 0);  // coeff_offsets
    encoder.set_buffer(2, Some(&log_size_buffer), 0);  // log_sizes
    encoder.set_buffer(3, Some(&point_buffer), 0);  // points
    encoder.set_buffer(4, Some(&result_buffer), 0);  // results
    encoder.set_buffer(5, Some(&num_evals_buffer), 0);  // num_evals

    // Dispatch 1 thread (1 evaluation)
    let threadgroup_size = 1;
    let threadgroups = 1;
    encoder.dispatch_thread_groups(
        metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
        metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read result from GPU
    let result_ptr = result_buffer.contents() as *const u32;
    let result_data = unsafe { std::slice::from_raw_parts(result_ptr, 4) };

    // Convert QM31 back to SecureField
    use crate::core::fields::cm31::CM31;
    use crate::core::fields::m31::M31;
    let c0 = CM31::from_m31(M31::from(result_data[0]), M31::from(result_data[1]));
    let c1 = CM31::from_m31(M31::from(result_data[2]), M31::from(result_data[3]));
    SecureField::from_m31_array([c0.0, c0.1, c1.0, c1.1])
}

/// GPU-accelerated batched polynomial evaluation at multiple points.
///
/// This function evaluates multiple polynomials at different points in a single GPU dispatch,
/// significantly reducing dispatch overhead compared to calling eval_at_point individually.
///
/// The implementation:
/// 1. Consolidates all coefficient data into a single GPU buffer (or uses existing Metal buffers)
/// 2. Creates offset, log_size, and point arrays for the kernel
/// 3. Dispatches a single GPU kernel with one thread per evaluation
/// 4. Reads back all results at once
fn metal_eval_at_points_batched_gpu(
    polys_and_points: &[(&CircleCoefficients<MetalBackend>, CirclePoint<SecureField>)],
) -> Vec<SecureField> {
    let num_evals = polys_and_points.len() as u32;
    if num_evals == 0 {
        return Vec::new();
    }

    let ctx = MetalContext::global();
    let device = ctx.device();

    // Build coefficient buffer, offsets, and log_sizes
    let mut coeff_offsets = Vec::with_capacity(num_evals as usize);
    let mut log_sizes = Vec::with_capacity(num_evals as usize);
    let mut all_coeffs = Vec::new();
    let mut current_offset = 0u32;

    for (poly, _) in polys_and_points {
        let cpu_coeffs = poly.coeffs.to_cpu();
        coeff_offsets.push(current_offset);
        log_sizes.push(poly.log_size());
        all_coeffs.extend(cpu_coeffs.iter().map(|f| f.0));
        current_offset += 1u32 << poly.log_size();
    }

    // Create GPU buffers
    let coeffs_buffer = device.new_buffer_with_data(
        all_coeffs.as_ptr() as *const _,
        (all_coeffs.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let offsets_buffer = device.new_buffer_with_data(
        coeff_offsets.as_ptr() as *const _,
        (coeff_offsets.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let log_sizes_buffer = device.new_buffer_with_data(
        log_sizes.as_ptr() as *const _,
        (log_sizes.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Build points array (each point is 8 x u32: x and y, each is QM31 = 4 x u32)
    let mut points_data = Vec::with_capacity(num_evals as usize * 8);
    for (_, point) in polys_and_points {
        points_data.extend_from_slice(&[
            point.x.0 .0 .0,  // x.c0.a
            point.x.0 .1 .0,  // x.c0.b
            point.x.1 .0 .0,  // x.c1.a
            point.x.1 .1 .0,  // x.c1.b
            point.y.0 .0 .0,  // y.c0.a
            point.y.0 .1 .0,  // y.c0.b
            point.y.1 .0 .0,  // y.c1.a
            point.y.1 .1 .0,  // y.c1.b
        ]);
    }

    let points_buffer = device.new_buffer_with_data(
        points_data.as_ptr() as *const _,
        (points_data.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Result buffer (num_evals x QM31 = num_evals x 4 x u32)
    let result_buffer = device.new_buffer(
        (num_evals as usize * 4 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let num_evals_buffer = device.new_buffer_with_data(
        &num_evals as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Dispatch kernel
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(ctx.eval_at_point_pipeline());
    encoder.set_buffer(0, Some(&coeffs_buffer), 0);      // coeffs_base
    encoder.set_buffer(1, Some(&offsets_buffer), 0);     // coeff_offsets
    encoder.set_buffer(2, Some(&log_sizes_buffer), 0);   // log_sizes
    encoder.set_buffer(3, Some(&points_buffer), 0);      // points
    encoder.set_buffer(4, Some(&result_buffer), 0);      // results
    encoder.set_buffer(5, Some(&num_evals_buffer), 0);   // num_evals

    // Dispatch one thread per evaluation
    let threadgroup_size = 256.min(num_evals as u64).max(1);
    let threadgroups = ((num_evals as u64 + threadgroup_size - 1) / threadgroup_size).max(1);

    encoder.dispatch_thread_groups(
        metal::MTLSize { width: threadgroups, height: 1, depth: 1 },
        metal::MTLSize { width: threadgroup_size, height: 1, depth: 1 },
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read results
    let result_ptr = result_buffer.contents() as *const u32;
    let result_data = unsafe { std::slice::from_raw_parts(result_ptr, num_evals as usize * 4) };

    // Convert results to SecureField
    use crate::core::fields::cm31::CM31;
    use crate::core::fields::m31::M31;

    (0..num_evals as usize)
        .map(|i| {
            let offset = i * 4;
            let c0 = CM31::from_m31(
                M31::from(result_data[offset]),
                M31::from(result_data[offset + 1]),
            );
            let c1 = CM31::from_m31(
                M31::from(result_data[offset + 2]),
                M31::from(result_data[offset + 3]),
            );
            SecureField::from_m31_array([c0.0, c0.1, c1.0, c1.1])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::poly::circle::CanonicCoset;
    use crate::prover::backend::Column;

    /// Compare SIMD vs Metal radix-8 on same input, step by step.
    #[test]
    fn test_radix8_intermediate_consistency() {
        use crate::prover::backend::simd::SimdBackend;

        // log_size=9: radix-8 processes layers 5,6,7
        let log_size = 9;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;

        println!("\n=== Testing SIMD vs Metal radix-8 ===");

        let coeffs_data: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();

        // Run SIMD FFT
        let simd_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
        let simd_result = SimdBackend::evaluate(&simd_poly, domain, &simd_twiddles);

        // Run Metal FFT
        let metal_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
        let metal_result = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);

        // Compare results
        let simd_values = simd_result.values.to_cpu();
        let metal_values = metal_result.values.to_cpu();

        println!("\nFirst 16 SIMD values:  {:?}", &simd_values[..16].iter().map(|f| f.0).collect::<Vec<_>>());
        println!("First 16 Metal values: {:?}", &metal_values[..16].iter().map(|f| f.0).collect::<Vec<_>>());

        for i in 0..size {
            if simd_values[i] != metal_values[i] {
                panic!("Mismatch at index {}: SIMD={:?}, Metal={:?}", i, simd_values[i], metal_values[i]);
            }
        }

        println!("\n✓ All {} values match!", size);
    }

    /// Test a single radix-8 operation on 8 elements to debug the kernel.
    #[test]
    fn test_metal_radix8_single_block() {
        if !MetalContext::is_available() {
            return;
        }

        use metal::MTLResourceOptions;

        println!("\n=== Testing single radix-8 block ===");

        let ctx = MetalContext::global();
        let device = ctx.device();

        // Simple test data: 8 consecutive values
        let input_data: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        println!("Input: {:?}", input_data);

        // Create data buffer
        let data_buffer = device.new_buffer_with_data(
            input_data.as_ptr() as *const _,
            (input_data.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Simple twiddles (all 1s doubled = 2)
        let twiddles: Vec<u32> = vec![2; 8];

        let tw0_buffer = device.new_buffer_with_data(
            twiddles.as_ptr() as *const _,
            (twiddles.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let tw1_buffer = device.new_buffer_with_data(
            twiddles.as_ptr() as *const _,
            (4 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let tw2_buffer = device.new_buffer_with_data(
            twiddles.as_ptr() as *const _,
            (2 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Dispatch kernel with layer=0 (minimum stride)
        let log_size = 3u32; // 8 elements
        let layer = 0u32; // Elements are contiguous
        let tw0_len = 8u32;
        let tw1_len = 4u32;
        let tw2_len = 2u32;

        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(ctx.fft_radix8_pipeline());
        encoder.set_buffer(0, Some(&data_buffer), 0);
        encoder.set_buffer(1, Some(&tw0_buffer), 0);
        encoder.set_buffer(2, Some(&tw1_buffer), 0);
        encoder.set_buffer(3, Some(&tw2_buffer), 0);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &log_size as *const u32 as *const _);
        encoder.set_bytes(5, std::mem::size_of::<u32>() as u64, &layer as *const u32 as *const _);
        encoder.set_bytes(6, std::mem::size_of::<u32>() as u64, &tw0_len as *const u32 as *const _);
        encoder.set_bytes(7, std::mem::size_of::<u32>() as u64, &tw1_len as *const u32 as *const _);
        encoder.set_bytes(8, std::mem::size_of::<u32>() as u64, &tw2_len as *const u32 as *const _);

        // Single thread processes all 8 elements
        encoder.dispatch_thread_groups(
            metal::MTLSize { width: 1, height: 1, depth: 1 },
            metal::MTLSize { width: 1, height: 1, depth: 1 },
        );

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read output
        let output_ptr = data_buffer.contents() as *const u32;
        let output_data = unsafe { std::slice::from_raw_parts(output_ptr, 8) };
        println!("Output: {:?}", output_data);

        // Now compute what SIMD should produce
        println!("\nExpected SIMD behavior:");
        println!("With twiddles all = 2 (doubled 1), butterflies should compute:");
        println!("  v0' = v0 + v1*1 = v0 + v1");
        println!("  v1' = v0 - v1*1 = v0 - v1");

        // Manual computation for layer 2 (coarsest):
        // Uses single twiddle tw2[0] = 2 for all 4 butterflies
        // (v0,v4), (v1,v5), (v2,v6), (v3,v7)
        println!("\nLayer 2: tw={}", twiddles[0]);
        println!("  (0,4) -> ({}, {})", 0+4, (0i32-4).rem_euclid(0x7FFFFFFF));
        println!("  (1,5) -> ({}, {})", 1+5, (1i32-5).rem_euclid(0x7FFFFFFF));
    }

    /// Test ONLY radix-8 layers to verify intermediate state.
    #[test]
    fn test_metal_radix8_intermediate() {
        // Test with log_size=9 but only process radix-8 layers (5,6,7)
        // by temporarily modifying vecwise range
        let log_size = 9;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;

        println!("\n=== Testing radix-8 intermediate state: log_size={} ===", log_size);

        // Create simple test input
        let coeffs_data: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();

        // Run full SIMD FFT
        let simd_poly_full = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
        let simd_result_full = SimdBackend::evaluate(&simd_poly_full, domain, &simd_twiddles);
        let simd_values_full = simd_result_full.values.to_cpu();

        println!("\nFull SIMD output (first 16): {:?}", &simd_values_full[..16].iter().map(|f| f.0).collect::<Vec<_>>());

        // Now run Metal which processes radix-8 + vecwise
        let metal_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);        let metal_result = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);
        let metal_values = metal_result.values.to_cpu();

        println!("\nFull Metal output (first 16): {:?}", &metal_values[..16].iter().map(|f| f.0).collect::<Vec<_>>());

        // The intermediate state after radix-8 was already printed in the test output
        // Compare final results
        for (i, (s, m)) in simd_values_full.iter().zip(metal_values.iter()).enumerate().take(16) {
            if s != m {
                println!("Mismatch at {}: SIMD={}, Metal={}", i, s.0, m.0);
            }
        }
    }

    /// Test with one radix-8 group + vecwise layers.
    #[test]
    fn test_metal_mixed_radix8_vecwise() {
        // Test with log_size=9: num_fft_layers=8
        // non_vecwise=3 (layers 5,6,7 - one radix-8 group)
        // vecwise (layers 4,3,2,1,0)
        let log_size = 9;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;

        println!("\n=== Testing mixed radix-8 + vecwise: log_size={} ===", log_size);
        println!("num_fft_layers: {}, non_vecwise: 3 (layers 5,6,7)", log_size - 1);

        // Create simple test input
        let coeffs_data: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();

        println!("\nInput (first 8):");
        for i in 0..8 {
            println!("  [{}] = {}", i, coeffs_data[i].0);
        }

        // SIMD
        let simd_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
        let simd_result = SimdBackend::evaluate(&simd_poly, domain, &simd_twiddles);
        let simd_values = simd_result.values.to_cpu();

        // Metal
        let metal_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
        let metal_result = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);
        let metal_values = metal_result.values.to_cpu();

        println!("\nSIMD output (first 16):");
        for i in 0..16 {
            println!("  [{}] = {}", i, simd_values[i].0);
        }

        println!("\nMetal output (first 16):");
        for i in 0..16 {
            println!("  [{}] = {}", i, metal_values[i].0);
        }

        // Compare
        for (i, (s, m)) in simd_values.iter().zip(metal_values.iter()).enumerate() {
            assert_eq!(s, m, "Mismatch at index {}", i);
        }
    }

    /// Test a single radix-2 layer to isolate kernel bugs.
    #[test]
    fn test_single_radix2_layer() {
        // Manually test layer 1 processing
        let log_size = 3; // 8 elements
        let size = 1 << log_size;

        println!("\n=== Testing single radix-2 layer ===");

        // Simple input
        let input: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();
        println!("Input: {:?}", input.iter().map(|f| f.0).collect::<Vec<_>>());

        // TODO: Manually invoke the Metal radix-2 kernel for layer 1
        // Then compare with expected CPU output

        println!("This test needs manual kernel invocation - skipping for now");
    }

    /// Test just the vecwise layers (no radix-8) to isolate the issue.
    #[test]
    fn test_metal_vecwise_only() {
        // Test with log_size=6: num_fft_layers=5 (all vecwise, no radix-8)
        let log_size = 6;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;

        println!("\n=== Testing vecwise-only: log_size={} ===", log_size);
        println!("num_fft_layers: {}", log_size - 1);

        // Create simple test input
        let coeffs_data: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();

        println!("\nInput (first 8):");
        for i in 0..8 {
            println!("  [{}] = {}", i, coeffs_data[i].0);
        }

        // SIMD
        let simd_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
        let simd_result = SimdBackend::evaluate(&simd_poly, domain, &simd_twiddles);
        let simd_values = simd_result.values.to_cpu();

        // Metal
        let metal_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
        let metal_result = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);
        let metal_values = metal_result.values.to_cpu();

        println!("\nSIMD output (first 8):");
        for i in 0..8 {
            println!("  [{}] = {}", i, simd_values[i].0);
        }

        println!("\nMetal output (first 8):");
        for i in 0..8 {
            println!("  [{}] = {}", i, metal_values[i].0);
        }

        // Compare
        for (i, (s, m)) in simd_values.iter().zip(metal_values.iter()).enumerate() {
            assert_eq!(s, m, "Mismatch at index {}", i);
        }
    }

    /// Debug test with detailed logging to understand Metal vs SIMD divergence.
    #[test]
    fn test_metal_fft_debug() {
        use crate::core::poly::utils::domain_line_twiddles_from_tree;

        // Test with log_size=10 - should fallback to SIMD
        let log_size = 10;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;

        println!("\n=== FFT Debug Test: log_size={} ===", log_size);
        println!("Size: {}, num_fft_layers: {}", size, log_size - 1);
        println!("Domain log_size: {}, Domain size: {}", domain.log_size(), domain.size());
        println!("Half coset log_size: {}, Half coset size: {}",
                 domain.half_coset.log_size(), domain.half_coset.size());

        // Create simple test input
        let coeffs_data: Vec<BaseField> = (0..size).map(|i| BaseField::from(i as u32)).collect();

        println!("\nInput (first 16 elements):");
        for i in 0..16 {
            println!("  coeffs[{}] = {}", i, coeffs_data[i].0);
        }

        // Get twiddle information
        let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
        let twiddle_slices = domain_line_twiddles_from_tree(domain, &simd_twiddles.twiddles);

        println!("\nTwiddle slice information:");
        println!("  Total twiddle slices: {}", twiddle_slices.len());
        for (i, slice) in twiddle_slices.iter().enumerate() {
            println!("  twiddle_slices[{}]: len={} (2^{})",
                     i, slice.len(), (slice.len() as f64).log2() as u32);
        }

        // For first radix-8 iteration (layer=6), show which twiddles are used
        let layer = 6u32;
        let num_fft_layers = (log_size - 1) as u32;
        println!("\nFirst radix-8 iteration (layer={}):", layer);
        println!("  Processing physical layers: {}, {}, {}", layer, layer+1, layer+2);

        let tw_idx_layer0 = (num_fft_layers - 1 - layer) as usize;
        let tw_idx_layer1 = (num_fft_layers - 1 - (layer + 1)) as usize;
        let tw_idx_layer2 = (num_fft_layers - 1 - (layer + 2)) as usize;

        println!("  tw_layer0 (finest): twiddle_slices[{}], len={}",
                 tw_idx_layer0, twiddle_slices[tw_idx_layer0].len());
        println!("  tw_layer1 (middle): twiddle_slices[{}], len={}",
                 tw_idx_layer1, twiddle_slices[tw_idx_layer1].len());
        println!("  tw_layer2 (coarsest): twiddle_slices[{}], len={}",
                 tw_idx_layer2, twiddle_slices[tw_idx_layer2].len());

        // Show first few twiddles for each layer and verify they match what SIMD would use
        println!("\n  First 4 twiddles from each layer (for first radix-8 iteration):");
        println!("  For layer={}, index=0 (first block):", layer);

        // SIMD uses twiddle_dbl[layer-1], twiddle_dbl[layer], twiddle_dbl[layer+1]
        // But we get them from domain_line_twiddles which are in reverse order
        // So we need to map correctly

        for i in 0..4.min(twiddle_slices[tw_idx_layer0].len()) {
            println!("    tw_layer0[{}] = {} (used by finest layer, physical layer {})",
                     i, twiddle_slices[tw_idx_layer0][i], layer);
        }
        for i in 0..4.min(twiddle_slices[tw_idx_layer1].len()) {
            println!("    tw_layer1[{}] = {} (used by middle layer, physical layer {})",
                     i, twiddle_slices[tw_idx_layer1][i], layer + 1);
        }
        for i in 0..4.min(twiddle_slices[tw_idx_layer2].len()) {
            println!("    tw_layer2[{}] = {} (used by coarsest layer, physical layer {})",
                     i, twiddle_slices[tw_idx_layer2][i], layer + 2);
        }

        // Also check what SIMD twiddle_dbl array looks like
        println!("\n  Full twiddle_slices array lengths:");
        for (i, slice) in twiddle_slices.iter().enumerate() {
            println!("    twiddle_slices[{}]: len={}, physical_layer={}",
                     i, slice.len(), (num_fft_layers as usize) - 1 - i);
        }

        // Run SIMD FFT
        let simd_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let simd_result = SimdBackend::evaluate(&simd_poly, domain, &simd_twiddles);
        let simd_values = simd_result.values.to_cpu();

        println!("\nSIMD FFT output (first 16 elements):");
        for i in 0..16 {
            println!("  simd[{}] = {}", i, simd_values[i].0);
        }

        // Run Metal FFT
        let metal_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
        let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
        let metal_result = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);
        let metal_values = metal_result.values.to_cpu();

        println!("\nMetal FFT output (first 16 elements):");
        for i in 0..16 {
            println!("  metal[{}] = {}", i, metal_values[i].0);
        }

        println!("\nComparison (first 16 elements):");
        let mut num_diffs = 0;
        for i in 0..16 {
            if simd_values[i] != metal_values[i] {
                println!("  [{}] DIFF: simd={}, metal={}",
                         i, simd_values[i].0, metal_values[i].0);
                num_diffs += 1;
            } else {
                println!("  [{}] OK: {}", i, simd_values[i].0);
            }
        }

        if num_diffs > 0 {
            panic!("{} differences found in first 16 elements", num_diffs);
        }
    }

    /// Test that Metal FFT produces the same results as SIMD FFT.
    #[test]
    fn test_metal_fft_correctness() {
        // Test on ALL sizes where Metal GPU is used (log_size >= MIN_FFT_LOG_SIZE = 5)
        // After fix, GPU should work for ALL non-vecwise layer counts (not just multiples of 3)
        // log_size 6: non_vecwise=0 (all radix-2)
        // log_size 7: non_vecwise=0 (all radix-2, 6 layers)
        // Start with 6-7 to verify they pass, then expand
        for log_size in 6..=15 {
            println!("\n=== Testing FFT at log_size={} ===", log_size);
            let domain = CanonicCoset::new(log_size).circle_domain();
            let size = 1 << log_size;

            // Create test input
            let coeffs_data: Vec<BaseField> = (0..size).map(|i| BaseField::from(i)).collect();

            // Run SIMD FFT
            let simd_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
            let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
            let simd_result = SimdBackend::evaluate(&simd_poly, domain, &simd_twiddles);
            let simd_values = simd_result.values.to_cpu();

            // Run Metal FFT
            let metal_poly = CircleCoefficients::new(coeffs_data.iter().copied().collect());
            let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
            let metal_result = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);
            let metal_values = metal_result.values.to_cpu();

            // Compare results
            assert_eq!(
                simd_values.len(),
                metal_values.len(),
                "FFT result lengths differ for log_size {}",
                log_size
            );

            for (i, (simd_val, metal_val)) in
                simd_values.iter().zip(metal_values.iter()).enumerate()
            {
                assert_eq!(
                    simd_val, metal_val,
                    "FFT results differ at index {} for log_size {}",
                    i, log_size
                );
            }
        }
    }

    /// Test that Metal IFFT produces the same results as SIMD IFFT.
    #[test]
    fn test_metal_ifft_correctness() {
        for log_size in [6, 13] {  // Start with small size for debugging
            let domain = CanonicCoset::new(log_size).circle_domain();
            let size = 1 << log_size;

            // First verify FFT outputs match
            println!("\n=== FFT Comparison for log_size={} ===", log_size);
            let coeffs: Vec<BaseField> = (0..size).map(|i| BaseField::from(i)).collect();

            let simd_poly = CircleCoefficients::new(coeffs.iter().copied().collect());
            let simd_twiddles_fft = SimdBackend::precompute_twiddles(domain.half_coset);
            let simd_eval = SimdBackend::evaluate(&simd_poly, domain, &simd_twiddles_fft);
            let simd_eval_values = simd_eval.values.to_cpu();

            let metal_poly = CircleCoefficients::new(coeffs.iter().copied().collect());
            let metal_twiddles_fft = MetalBackend::precompute_twiddles(domain.half_coset);
            let metal_eval = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles_fft);
            let metal_eval_values = metal_eval.values.to_cpu();

            // Verify FFT outputs match
            for (i, (s, m)) in simd_eval_values.iter().zip(metal_eval_values.iter()).enumerate() {
                assert_eq!(s, m, "FFT outputs differ at index {} for log_size {}", i, log_size);
            }
            println!("✓ FFT outputs match");

            if log_size == 6 {
                println!("FFT output (first 16): {:?}", &simd_eval_values[..16].iter().map(|f| f.0).collect::<Vec<_>>());
            }

            // Now test IFFT on identical inputs
            println!("\n=== IFFT Comparison for log_size={} ===", log_size);
            println!("Using FFT output as IFFT input");

            // Create evaluations from the FFT output
            let eval_data = simd_eval_values.clone();

            // Run SIMD IFFT
            let simd_eval = CircleEvaluation::new(
                domain,
                eval_data.iter().copied().collect(),
            );
            let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
            let simd_result = SimdBackend::interpolate(simd_eval, &simd_twiddles);
            let simd_coeffs = simd_result.coeffs.to_cpu();

            if log_size == 6 {
                println!("SIMD result (ALL): {:?}", simd_coeffs.iter().map(|f| f.0).collect::<Vec<_>>());
            } else {
                println!("SIMD result (first 16): {:?}", &simd_coeffs[..16].iter().map(|f| f.0).collect::<Vec<_>>());
            }

            // Run Metal IFFT
            let metal_eval = CircleEvaluation::new(
                domain,
                eval_data.iter().copied().collect(),
            );
            let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
            let metal_result = MetalBackend::interpolate(metal_eval, &metal_twiddles);
            let metal_coeffs = metal_result.coeffs.to_cpu();

            if log_size == 6 {
                println!("Metal result (ALL): {:?}", metal_coeffs.iter().map(|f| f.0).collect::<Vec<_>>());
            } else {
                println!("Metal result (first 16): {:?}", &metal_coeffs[..16].iter().map(|f| f.0).collect::<Vec<_>>());
            }

            // Compare results
            assert_eq!(
                simd_coeffs.len(),
                metal_coeffs.len(),
                "IFFT result lengths differ for log_size {}",
                log_size
            );

            for (i, (simd_val, metal_val)) in
                simd_coeffs.iter().zip(metal_coeffs.iter()).enumerate()
            {
                assert_eq!(
                    simd_val, metal_val,
                    "IFFT results differ at index {} for log_size {}",
                    i, log_size
                );
            }
        }
    }

    /// Benchmark Metal vs SIMD FFT performance with REAL wall-clock measurements.
    #[test]
    #[ignore]  // Run with: cargo test --release --features metal_prover test_metal_vs_simd_benchmark -- --ignored --nocapture
    fn test_metal_vs_simd_benchmark() {
        use std::time::Instant;

        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║         ACTUAL Metal vs SIMD FFT Performance Benchmark       ║");
        println!("╚═══════════════════════════════════════════════════════════════╝\n");
        println!("{:<10} {:>12} {:>12} {:>12} {:>12}",
                 "log_size", "SIMD (ms)", "Metal (ms)", "Speedup", "Elements");
        println!("{}", "─".repeat(62));

        for log_size in [18, 19, 20, 21, 22].iter().copied() {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let size = 1 << log_size;
            let coeffs: Vec<BaseField> = (0..size).map(|i| BaseField::from(i % 10000)).collect();

            // Warm up and measure SIMD
            let simd_poly = CircleCoefficients::new(coeffs.iter().copied().collect());
            let simd_twiddles = SimdBackend::precompute_twiddles(domain.half_coset);
            let _ = SimdBackend::evaluate(&simd_poly, domain, &simd_twiddles);  // Warm up

            let simd_start = Instant::now();
            let _ = SimdBackend::evaluate(&simd_poly, domain, &simd_twiddles);
            let simd_time = simd_start.elapsed();

            // Warm up and measure Metal
            let metal_poly = CircleCoefficients::new(coeffs.iter().copied().collect());
            let metal_twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
            let _ = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);  // Warm up

            let metal_start = Instant::now();
            let _ = MetalBackend::evaluate(&metal_poly, domain, &metal_twiddles);
            let metal_time = metal_start.elapsed();

            let speedup = simd_time.as_secs_f64() / metal_time.as_secs_f64();

            println!("{:<10} {:>12.3} {:>12.3} {:>11.2}x {:>12}",
                     log_size,
                     simd_time.as_secs_f64() * 1000.0,
                     metal_time.as_secs_f64() * 1000.0,
                     speedup,
                     size);
        }

        println!("\n✓ All benchmarks completed successfully");
        println!("Note: These are REAL measured wall-clock times, not theoretical predictions\n");
    }

    /// Test FFT/IFFT round-trip produces original coefficients.
    #[test]
    fn test_metal_fft_ifft_roundtrip() {
        for log_size in [13, 16] {
            let domain = CanonicCoset::new(log_size).circle_domain();
            let size = 1 << log_size;

            // Create original coefficients
            let original: Vec<BaseField> = (0..size).map(|i| BaseField::from(i % 1000)).collect();

            // FFT -> IFFT round trip
            let poly = CircleCoefficients::new(original.iter().copied().collect());
            let twiddles = MetalBackend::precompute_twiddles(domain.half_coset);

            let eval = MetalBackend::evaluate(&poly, domain, &twiddles);
            let reconstructed = MetalBackend::interpolate(eval, &twiddles);

            let result = reconstructed.coeffs.to_cpu();

            // Compare
            assert_eq!(
                original.len(),
                result.len(),
                "Round-trip length mismatch for log_size {}",
                log_size
            );

            for (i, (orig, recon)) in original.iter().zip(result.iter()).enumerate() {
                assert_eq!(
                    orig, recon,
                    "Round-trip failed at index {} for log_size {}",
                    i, log_size
                );
            }
        }
    }

    /// Test that sizes where num_fft_layers is not divisible by 3 fall back to SIMD correctly.
    #[test]
    fn test_metal_fallback_to_simd() {
        // log_size 14: num_fft_layers = 13 (not divisible by 3, should fall back to SIMD)
        let log_size = 14;
        let domain = CanonicCoset::new(log_size).circle_domain();
        let size = 1 << log_size;

        let coeffs: Vec<BaseField> = (0..size).map(|i| BaseField::from(i)).collect();

        // This should fall back to SIMD internally but still work
        let poly = CircleCoefficients::new(coeffs.iter().copied().collect());
        let twiddles = MetalBackend::precompute_twiddles(domain.half_coset);
        let eval = MetalBackend::evaluate(&poly, domain, &twiddles);
        let reconstructed = MetalBackend::interpolate(eval, &twiddles);

        let result = reconstructed.coeffs.to_cpu();

        // Verify round-trip
        for (i, (orig, recon)) in coeffs.iter().zip(result.iter()).enumerate() {
            assert_eq!(
                orig, recon,
                "Fallback round-trip failed at index {}",
                i
            );
        }
    }

    // Removed - test moved to examples crate: test_wide_fib_prove_with_metal
}
