//! Metal Merkle operations.

use std::sync::{Arc, Mutex};

use metal::{Buffer, MTLResourceOptions};

use super::context::MetalContext;
use super::thresholds::MIN_MERKLE_LOG_SIZE;
use super::MetalBackend;
use crate::core::fields::m31::BaseField;
use crate::core::vcs::blake2_hash::Blake2sHash;
use crate::core::vcs_lifted::blake2_merkle::Blake2sMerkleHasherGeneric;
use crate::prover::backend::simd::SimdBackend;
use crate::prover::backend::{Col, Column, ColumnOps};
use crate::prover::vcs_lifted::ops::MerkleOpsLifted;

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

// Blake2s hash columns use lazy GPU-backed columns
impl ColumnOps<Blake2sHash> for MetalBackend {
    type Column = MetalBlake2sColumn;

    fn bit_reverse_column(_column: &mut Self::Column) {
        unimplemented!()
    }
}

impl<const IS_M31_OUTPUT: bool> MerkleOpsLifted<Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>>
    for MetalBackend
{
    fn build_leaves(
        columns: &[&Col<Self, BaseField>],
        lifting_log_size: u32,
    ) -> Col<Self, Blake2sHash> {
        // SIMD fallback for small sizes or empty columns
        if lifting_log_size < MIN_MERKLE_LOG_SIZE || columns.is_empty() {
            let _timer = crate::metal_profile_fn!(
                "merkle_build_leaves",
                "SIMD",
                lifting_log_size = lifting_log_size,
                num_columns = columns.len()
            );

            use crate::prover::backend::simd::column::BaseColumn;

            let simd_columns_owned: Vec<BaseColumn> = columns
                .iter()
                .map(|col| BaseColumn::from_cpu(col.as_slice()))
                .collect();
            let simd_columns_refs: Vec<&BaseColumn> = simd_columns_owned.iter().collect();

            let simd_result = <SimdBackend as MerkleOpsLifted<
                Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>,
            >>::build_leaves(&simd_columns_refs, lifting_log_size);

            return MetalBlake2sColumn::from_cpu(simd_result);
        }

        // GPU path: pack column data and dispatch leaf hashing kernel
        let _timer = crate::metal_profile_fn!(
            "merkle_build_leaves",
            "GPU",
            lifting_log_size = lifting_log_size,
            num_columns = columns.len()
        );

        let n_leaves = 1usize << lifting_log_size;
        let n_columns = columns.len();
        let ctx = MetalContext::global();
        let device = ctx.device();

        // Compute element offsets and pack column data into one contiguous buffer.
        // M31 is repr(transparent) over u32, so as_slice() data is already u32 layout.
        let mut col_offsets_vec: Vec<u32> = Vec::with_capacity(n_columns);
        let mut col_log_sizes_vec: Vec<u32> = Vec::with_capacity(n_columns);
        let mut total_elements = 0usize;

        for col in columns {
            col_offsets_vec.push(total_elements as u32);
            col_log_sizes_vec.push(col.len().ilog2());
            total_elements += col.len();
        }

        // Pack all column data into a single contiguous buffer.
        // Reads from Metal shared memory buffers (zero-copy on Apple Silicon).
        let mut packed_data: Vec<u32> = Vec::with_capacity(total_elements);
        for col in columns {
            let slice = col.as_slice();
            // Safety: M31 is #[repr(transparent)] over u32
            let u32_slice = unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const u32, slice.len())
            };
            packed_data.extend_from_slice(u32_slice);
        }

        // Create GPU buffers
        let packed_buffer = device.new_buffer_with_data(
            packed_data.as_ptr() as *const _,
            (total_elements * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let offsets_buffer = device.new_buffer_with_data(
            col_offsets_vec.as_ptr() as *const _,
            (n_columns * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let log_sizes_buffer = device.new_buffer_with_data(
            col_log_sizes_vec.as_ptr() as *const _,
            (n_columns * std::mem::size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Output: n_leaves * 8 u32s (each hash is 32 bytes = 8 u32s)
        let output_size = (n_leaves * 8 * std::mem::size_of::<u32>()) as u64;
        let output_buffer =
            device.new_buffer(output_size, MTLResourceOptions::StorageModeShared);

        // Kernel parameters
        let n_columns_param = n_columns as u32;
        let is_m31_output: bool = IS_M31_OUTPUT;

        // Dispatch GPU kernel
        let command_buffer = ctx.command_queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.merkle_leaf_lifted_pipeline());
        encoder.set_buffer(0, Some(&packed_buffer), 0);
        encoder.set_buffer(1, Some(&offsets_buffer), 0);
        encoder.set_buffer(2, Some(&log_sizes_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &n_columns_param as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &lifting_log_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<bool>() as u64,
            &is_m31_output as *const bool as *const _,
        );

        let num_threads = n_leaves as u64;
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

        // Return lazy column — GPU work completes asynchronously
        MetalBlake2sColumn::from_gpu_pending(
            output_buffer,
            n_leaves,
            Arc::new(command_buffer.to_owned()),
        )
    }

    fn build_next_layer(prev_layer: &Col<Self, Blake2sHash>) -> Col<Self, Blake2sHash> {
        let log_size = prev_layer.len().ilog2() - 1;
        let _timer =
            crate::metal_profile_fn!("merkle_build_next_layer", "CPU/GPU", log_size = log_size);

        // Use GPU path for large layers, SIMD fallback for small
        if log_size < MIN_MERKLE_LOG_SIZE {
            let prev_cpu = prev_layer.to_cpu();
            let simd_result = <SimdBackend as MerkleOpsLifted<
                Blake2sMerkleHasherGeneric<IS_M31_OUTPUT>,
            >>::build_next_layer(&prev_cpu);
            return MetalBlake2sColumn::from_cpu(simd_result);
        }

        // Metal GPU path - hash pairs of child nodes (LAZY SYNC)
        let num_parents = 1 << log_size;

        let ctx = MetalContext::global();
        let device = ctx.device();

        // Allocate output buffer
        let parents_size = (num_parents * 8 * std::mem::size_of::<u32>()) as u64;
        let parents_buffer = device.new_buffer(parents_size, MTLResourceOptions::StorageModeShared);

        let is_m31_output: bool = IS_M31_OUTPUT;
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
