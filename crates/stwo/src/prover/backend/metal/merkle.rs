//! Metal Merkle operations.

use std::sync::{Arc, Mutex};

use metal::{Buffer, MTLResourceOptions};

use super::context::MetalContext;
use super::thresholds::MIN_MERKLE_LOG_SIZE;
use super::MetalBackend;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs::blake2_merkle::{Blake2sM31MerkleHasher, Blake2sMerkleHasher};
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::{Col, Column, ColumnOps};
use crate::prover::vcs::ops::MerkleOps;

/// Lazy GPU-backed column for Blake2s hashes.
#[derive(Clone)]
pub struct MetalBlake2sColumn {
    buffer: Buffer,
    len: usize,
    /// Shared state for lazy synchronization
    sync_state: Arc<Mutex<SyncState>>,
}

#[derive(Clone)]
enum SyncState {
    /// GPU work pending, need to wait before reading
    Pending(Arc<metal::CommandBuffer>),
    /// GPU work completed, data ready
    Synced,
}

impl MetalBlake2sColumn {
    /// Create column from GPU buffer with pending command buffer
    fn from_gpu_pending(buffer: Buffer, len: usize, cmd_buf: Arc<metal::CommandBuffer>) -> Self {
        Self {
            buffer,
            len,
            sync_state: Arc::new(Mutex::new(SyncState::Pending(cmd_buf))),
        }
    }

    /// Create column from CPU data (already synced)
    fn from_cpu(hashes: Vec<Blake2sHash>) -> Self {
        let ctx = MetalContext::global();
        let len = hashes.len();

        // Flatten to u32 array
        let mut data = Vec::with_capacity(len * 8);
        for hash in &hashes {
            let hash_u32s: &[u32; 8] = bytemuck::cast_ref(&hash.0);
            data.extend_from_slice(hash_u32s);
        }

        let buffer = ctx.device().new_buffer_with_data(
            data.as_ptr() as *const _,
            (data.len() * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            buffer,
            len,
            sync_state: Arc::new(Mutex::new(SyncState::Synced)),
        }
    }

    /// Ensure GPU work is complete before reading
    fn ensure_synced(&self) {
        let mut state = self.sync_state.lock().unwrap();
        if let SyncState::Pending(cmd_buf) = &*state {
            cmd_buf.wait_until_completed();
            *state = SyncState::Synced;
        }
    }

    /// Read a single hash at index
    fn read_hash(&self, index: usize) -> Blake2sHash {
        self.ensure_synced();

        let output_data = unsafe {
            std::slice::from_raw_parts(self.buffer.contents() as *const u32, self.len * 8)
        };

        let hash_u32s: [u32; 8] = [
            output_data[index * 8],
            output_data[index * 8 + 1],
            output_data[index * 8 + 2],
            output_data[index * 8 + 3],
            output_data[index * 8 + 4],
            output_data[index * 8 + 5],
            output_data[index * 8 + 6],
            output_data[index * 8 + 7],
        ];
        let hash_bytes: [u8; 32] = bytemuck::cast(hash_u32s);
        Blake2sHash(hash_bytes)
    }

    /// Get buffer for GPU operations (no sync required)
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl std::fmt::Debug for MetalBlake2sColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBlake2sColumn")
            .field("len", &self.len)
            .field(
                "synced",
                &matches!(*self.sync_state.lock().unwrap(), SyncState::Synced),
            )
            .finish()
    }
}

impl Column<Blake2sHash> for MetalBlake2sColumn {
    fn zeros(_len: usize) -> Self {
        unimplemented!("Blake2s columns are not zero-initialized")
    }

    unsafe fn uninitialized(_len: usize) -> Self {
        unimplemented!("Blake2s columns are created from GPU kernels")
    }

    fn to_cpu(&self) -> Vec<Blake2sHash> {
        self.ensure_synced();

        let output_data = unsafe {
            std::slice::from_raw_parts(self.buffer.contents() as *const u32, self.len * 8)
        };

        let mut result = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let hash_u32s: [u32; 8] = [
                output_data[i * 8],
                output_data[i * 8 + 1],
                output_data[i * 8 + 2],
                output_data[i * 8 + 3],
                output_data[i * 8 + 4],
                output_data[i * 8 + 5],
                output_data[i * 8 + 6],
                output_data[i * 8 + 7],
            ];
            let hash_bytes: [u8; 32] = bytemuck::cast(hash_u32s);
            result.push(Blake2sHash(hash_bytes));
        }
        result
    }

    fn len(&self) -> usize {
        self.len
    }

    fn at(&self, index: usize) -> Blake2sHash {
        self.read_hash(index)
    }

    fn set(&mut self, _index: usize, _value: Blake2sHash) {
        unimplemented!("Blake2s columns are read-only")
    }

    fn split_at_mid(self) -> (Self, Self) {
        unimplemented!("Blake2s columns don't support splitting")
    }
}

impl FromIterator<Blake2sHash> for MetalBlake2sColumn {
    fn from_iter<T: IntoIterator<Item = Blake2sHash>>(iter: T) -> Self {
        Self::from_cpu(iter.into_iter().collect())
    }
}

/// Build complete Merkle tree in a single GPU submission.
#[allow(dead_code)]
fn commit_tree_batched<H: Into<bool>>(
    ctx: &MetalContext,
    initial_layer: Vec<Blake2sHash>,
    num_layers: u32,
    is_m31: H,
) -> Vec<Vec<Blake2sHash>> {
    if num_layers == 0 || initial_layer.is_empty() {
        return vec![];
    }

    let is_m31_output = is_m31.into();
    let device = ctx.device();

    // Pre-allocate all buffers for the entire tree
    let mut layer_buffers = Vec::new();
    let mut current_size = initial_layer.len();

    // First layer is input
    let mut initial_data = Vec::with_capacity(initial_layer.len() * 8);
    for hash in &initial_layer {
        let hash_u32s: &[u32; 8] = bytemuck::cast_ref(&hash.0);
        initial_data.extend_from_slice(hash_u32s);
    }

    let initial_buffer = device.new_buffer_with_data(
        initial_data.as_ptr() as *const _,
        (initial_data.len() * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    layer_buffers.push((current_size, initial_buffer));

    // Pre-allocate buffers for all subsequent layers
    for _ in 0..num_layers {
        current_size /= 2;
        let buffer = device.new_buffer(
            (current_size * 8 * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        layer_buffers.push((current_size, buffer));
    }

    // Create a single command buffer for all layers
    let command_buffer = ctx.command_queue().new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    // Process each layer
    for i in 0..num_layers as usize {
        let (_, input_buffer) = &layer_buffers[i];
        let (num_parents, output_buffer) = &layer_buffers[i + 1];
        let size_param = *num_parents as u32;

        encoder.set_compute_pipeline_state(ctx.merkle_pipeline());
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(output_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<bool>() as u64,
            &is_m31_output as *const bool as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &size_param as *const u32 as *const _,
        );

        let num_threads = *num_parents as u64;
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

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
    }

    // Submit all operations at once
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back all layer results
    let mut results = Vec::new();
    for i in 1..=num_layers as usize {
        let (num_elems, buffer) = &layer_buffers[i];
        let output_data =
            unsafe { std::slice::from_raw_parts(buffer.contents() as *const u32, num_elems * 8) };

        let mut layer_result = Vec::with_capacity(*num_elems);
        for j in 0..*num_elems {
            let hash_u32s: [u32; 8] = [
                output_data[j * 8],
                output_data[j * 8 + 1],
                output_data[j * 8 + 2],
                output_data[j * 8 + 3],
                output_data[j * 8 + 4],
                output_data[j * 8 + 5],
                output_data[j * 8 + 6],
                output_data[j * 8 + 7],
            ];
            let hash_bytes: [u8; 32] = bytemuck::cast(hash_u32s);
            layer_result.push(Blake2sHash(hash_bytes));
        }
        results.push(layer_result);
    }

    results
}

// Blake2s hash columns use lazy GPU-backed columns
impl ColumnOps<Blake2sHash> for MetalBackend {
    type Column = MetalBlake2sColumn;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl MerkleOps<Blake2sMerkleHasher> for MetalBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&MetalBlake2sColumn>,
        columns: &[&Col<Self, BaseField>],
    ) -> MetalBlake2sColumn {
        let _timer = crate::metal_profile_fn!(
            "merkle_blake2s",
            "CPU/GPU",
            log_size = log_size,
            num_columns = columns.len()
        );

        // Fall back to SIMD for small sizes or when columns are present
        // (column hashing not yet implemented in Metal)
        if log_size < MIN_MERKLE_LOG_SIZE || !columns.is_empty() || prev_layer.is_none() {
            use crate::prover::backend::simd::column::BaseColumn;
            use crate::prover::backend::Column;

            // Convert Metal columns to SIMD
            let simd_columns_owned: Vec<BaseColumn> = columns
                .iter()
                .map(|col| {
                    let cpu_vals = col.to_cpu();
                    cpu_vals.into_iter().collect()
                })
                .collect();
            let simd_columns_refs: Vec<&BaseColumn> = simd_columns_owned.iter().collect();

            let prev_layer_cpu = prev_layer.map(|l| l.to_cpu());
            let simd_result = <SimdBackend as MerkleOps<Blake2sMerkleHasher>>::commit_on_layer(
                log_size,
                prev_layer_cpu.as_ref(),
                &simd_columns_refs,
            );

            return MetalBlake2sColumn::from_cpu(simd_result);
        }

        // Metal GPU path - hash pairs of child nodes (LAZY SYNC)
        let prev_layer = prev_layer.unwrap();
        let num_parents = 1 << log_size;

        let ctx = MetalContext::global();
        let device = ctx.device();

        // Allocate output buffer (GPU will write all values)
        let parents_size = (num_parents * 8 * std::mem::size_of::<u32>()) as u64;
        let parents_buffer = device.new_buffer(parents_size, MTLResourceOptions::StorageModeShared);

        let is_m31_output: bool = false;
        let size_param = num_parents as u32;

        // Dispatch kernel
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.merkle_pipeline());
        encoder.set_buffer(0, Some(prev_layer.buffer()), 0);
        encoder.set_buffer(1, Some(&parents_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<bool>() as u64,
            &is_m31_output as *const bool as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &size_param as *const u32 as *const _,
        );

        let num_threads = num_parents as u64;
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

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

        // NO WAIT - Return lazy column with pending command buffer
        MetalBlake2sColumn::from_gpu_pending(
            parents_buffer,
            num_parents,
            Arc::new(command_buffer.to_owned()),
        )
    }
}

impl MerkleOps<Blake2sM31MerkleHasher> for MetalBackend {
    fn commit_on_layer(
        log_size: u32,
        prev_layer: Option<&MetalBlake2sColumn>,
        columns: &[&Col<Self, BaseField>],
    ) -> MetalBlake2sColumn {
        let _timer = crate::metal_profile_fn!(
            "merkle_m31",
            "CPU/GPU",
            log_size = log_size,
            num_columns = columns.len()
        );

        // Fall back to SIMD for small sizes
        if log_size < MIN_MERKLE_LOG_SIZE {
            use crate::prover::backend::simd::column::BaseColumn;
            use crate::prover::backend::Column;

            // Convert Metal columns to SIMD
            let simd_columns_owned: Vec<BaseColumn> = columns
                .iter()
                .map(|col| {
                    let cpu_vals = col.to_cpu();
                    cpu_vals.into_iter().collect()
                })
                .collect();
            let simd_columns_refs: Vec<&BaseColumn> = simd_columns_owned.iter().collect();

            let prev_layer_cpu = prev_layer.map(|l| l.to_cpu());
            let simd_result = <SimdBackend as MerkleOps<Blake2sM31MerkleHasher>>::commit_on_layer(
                log_size,
                prev_layer_cpu.as_ref(),
                &simd_columns_refs,
            );

            return MetalBlake2sColumn::from_cpu(simd_result);
        }

        let ctx = MetalContext::global();
        let device = ctx.device();
        let domain_size = 1 << log_size;

        // GPU leaf hashing path (when columns are present) - LAZY SYNC
        if !columns.is_empty() {
            assert!(prev_layer.is_none(), "Leaf layer should have no prev_layer");

            let num_columns = columns.len();

            // Allocate persistent buffers (not pooled, owned by column)
            let columns_buffer_size =
                (num_columns * domain_size * std::mem::size_of::<u32>()) as u64;
            let columns_buffer =
                device.new_buffer(columns_buffer_size, MTLResourceOptions::StorageModeShared);

            let output_size = (domain_size * 8 * std::mem::size_of::<u32>()) as u64;
            let output_buffer =
                device.new_buffer(output_size, MTLResourceOptions::StorageModeShared);

            let command_buffer = ctx.command_queue().new_command_buffer();

            // Flatten columns using GPU blit
            let blit_encoder = command_buffer.new_blit_command_encoder();
            let mut offset = 0u64;
            let col_size = (domain_size * std::mem::size_of::<u32>()) as u64;
            for col in columns {
                blit_encoder.copy_from_buffer(col.buffer(), 0, &columns_buffer, offset, col_size);
                offset += col_size;
            }
            blit_encoder.end_encoding();

            // Leaf hash kernel
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(ctx.merkle_leaf_pipeline());
            encoder.set_buffer(0, Some(&columns_buffer), 0);
            let num_columns_u32 = num_columns as u32;
            encoder.set_bytes(
                1,
                std::mem::size_of::<u32>() as u64,
                &num_columns_u32 as *const u32 as *const _,
            );
            let domain_size_u32 = domain_size as u32;
            encoder.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &domain_size_u32 as *const u32 as *const _,
            );
            let is_m31_output = true;
            encoder.set_bytes(
                3,
                std::mem::size_of::<bool>() as u64,
                &is_m31_output as *const bool as *const _,
            );
            encoder.set_buffer(4, Some(&output_buffer), 0);

            let num_threads = domain_size as u64;
            let threadgroup_size = 256.min(num_threads.max(1));
            let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;
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

            // NO WAIT - Return lazy column with pending command buffer
            return MetalBlake2sColumn::from_gpu_pending(
                output_buffer,
                domain_size,
                Arc::new(command_buffer.to_owned()),
            );
        }

        // Metal GPU path with M31 reduction (node hashing) - LAZY SYNC
        let prev_layer = prev_layer.unwrap();
        let num_parents = domain_size;

        // Allocate output buffer (not pooled, owned by column)
        let parents_size = (num_parents * 8 * std::mem::size_of::<u32>()) as u64;
        let parents_buffer = device.new_buffer(parents_size, MTLResourceOptions::StorageModeShared);

        let is_m31_output: bool = true; // M31 reduction enabled
        let size_param = num_parents as u32;

        // Dispatch kernel
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.merkle_pipeline());
        encoder.set_buffer(0, Some(prev_layer.buffer()), 0);
        encoder.set_buffer(1, Some(&parents_buffer), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<bool>() as u64,
            &is_m31_output as *const bool as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &size_param as *const u32 as *const _,
        );

        let num_threads = num_parents as u64;
        let threadgroup_size = 256.min(num_threads.max(1));
        let threadgroups = (num_threads + threadgroup_size - 1) / threadgroup_size;

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

        // NO WAIT - Return lazy column with pending command buffer
        MetalBlake2sColumn::from_gpu_pending(
            parents_buffer,
            num_parents,
            Arc::new(command_buffer.to_owned()),
        )
    }

    /// Batched commit for multiple consecutive node-only layers.
    fn commit_node_layers_batched(
        initial_layer: &Col<Self, Blake2sHash>,
        num_layers: u32,
    ) -> Vec<Col<Self, Blake2sHash>> {
        let _timer = crate::metal_profile_fn!(
            "merkle_m31_batched",
            "GPU",
            num_layers = num_layers,
            initial_size = initial_layer.len()
        );

        // Use layer-by-layer with lazy sync
        Self::commit_node_layers_batched_default(initial_layer, num_layers)
    }
}

// Default implementation helper (extracted to avoid code duplication)
impl MetalBackend {
    fn commit_node_layers_batched_default(
        initial_layer: &MetalBlake2sColumn,
        num_layers: u32,
    ) -> Vec<MetalBlake2sColumn> {
        let mut layers = Vec::new();
        let mut prev_layer = initial_layer;
        let mut log_size = initial_layer.len().ilog2() - 1;

        for _ in 0..num_layers {
            let layer = <Self as MerkleOps<Blake2sM31MerkleHasher>>::commit_on_layer(
                log_size,
                Some(prev_layer),
                &[],
            );
            layers.push(layer);
            prev_layer = layers.last().unwrap();
            log_size = log_size.saturating_sub(1);
        }

        layers
    }
}
