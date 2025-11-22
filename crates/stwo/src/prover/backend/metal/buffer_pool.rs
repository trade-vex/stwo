//! Buffer pool for reusable Metal buffers.
//!
//! This module provides a pool of reusable Metal buffers to reduce allocation overhead.
//! Buffers are organized by size and can be checked out and returned to the pool.

use metal::{Buffer, Device, MTLResourceOptions};
use std::collections::HashMap;
use std::sync::Mutex;

/// Size bucket for buffer pooling (power of 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SizeBucket {
    /// Log2 of the buffer size in bytes.
    log_size: u32,
}

impl SizeBucket {
    /// Create a size bucket for the given size in bytes.
    fn for_size(size: u64) -> Self {
        // Round up to next power of 2
        let log_size = if size == 0 {
            0
        } else {
            64 - (size - 1).leading_zeros()
        };
        Self { log_size }
    }

    /// Get the actual buffer size for this bucket.
    fn size(&self) -> u64 {
        1u64 << self.log_size
    }
}

/// A pool of reusable Metal buffers.
pub struct BufferPool {
    /// Device for creating new buffers.
    device: Device,
    /// Available buffers organized by size bucket.
    available: Mutex<HashMap<SizeBucket, Vec<Buffer>>>,
    /// Resource options for buffer creation.
    resource_options: MTLResourceOptions,
    /// Maximum buffers per size bucket.
    max_per_bucket: usize,
    /// Statistics for monitoring.
    stats: Mutex<PoolStats>,
}

#[derive(Debug, Default)]
struct PoolStats {
    allocations: u64,
    reuses: u64,
    returns: u64,
}

impl BufferPool {
    /// Create a new buffer pool.
    pub fn new(device: Device, resource_options: MTLResourceOptions) -> Self {
        Self {
            device,
            available: Mutex::new(HashMap::new()),
            resource_options,
            max_per_bucket: 16, // Keep up to 16 buffers per size
            stats: Mutex::new(PoolStats::default()),
        }
    }

    /// Check out a buffer of at least the specified size.
    ///
    /// Returns a buffer that is at least `size` bytes. The buffer may be larger
    /// (rounded up to the next power of 2). If no suitable buffer is available
    /// in the pool, a new one is allocated.
    pub fn checkout(&self, size: u64) -> PooledBuffer {
        let bucket = SizeBucket::for_size(size);
        let actual_size = bucket.size();

        // Try to get a buffer from the pool
        let buffer = {
            let mut available = self.available.lock().unwrap();
            available
                .entry(bucket)
                .or_insert_with(Vec::new)
                .pop()
        };

        let buffer = if let Some(buffer) = buffer {
            // Reused from pool
            let mut stats = self.stats.lock().unwrap();
            stats.reuses += 1;
            buffer
        } else {
            // Allocate new buffer
            let mut stats = self.stats.lock().unwrap();
            stats.allocations += 1;
            self.device.new_buffer(actual_size, self.resource_options)
        };

        PooledBuffer {
            buffer,
            bucket,
            pool: self as *const BufferPool,
        }
    }

    /// Return a buffer to the pool.
    ///
    /// Called automatically when a PooledBuffer is dropped.
    fn return_buffer(&self, bucket: SizeBucket, buffer: Buffer) {
        let mut available = self.available.lock().unwrap();
        let buffers = available.entry(bucket).or_insert_with(Vec::new);

        // Only keep up to max_per_bucket buffers
        if buffers.len() < self.max_per_bucket {
            buffers.push(buffer);
            let mut stats = self.stats.lock().unwrap();
            stats.returns += 1;
        }
        // Otherwise, let the buffer be dropped
    }

    /// Clear all buffers from the pool.
    #[allow(dead_code)]
    pub fn clear(&self) {
        let mut available = self.available.lock().unwrap();
        available.clear();
    }

    /// Get pool statistics for monitoring.
    #[allow(dead_code)]
    pub fn stats(&self) -> (u64, u64, u64) {
        let stats = self.stats.lock().unwrap();
        (stats.allocations, stats.reuses, stats.returns)
    }
}

/// A buffer checked out from the pool.
///
/// Automatically returns the buffer to the pool when dropped.
pub struct PooledBuffer {
    buffer: Buffer,
    bucket: SizeBucket,
    pool: *const BufferPool,
}

impl PooledBuffer {
    /// Get the underlying Metal buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the actual size of the buffer (may be larger than requested).
    pub fn size(&self) -> u64 {
        self.bucket.size()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Return buffer to pool
        unsafe {
            let pool = &*self.pool;
            // Take ownership of buffer to return it
            let buffer = std::mem::replace(&mut self.buffer,
                pool.device.new_buffer(1, MTLResourceOptions::empty()));
            pool.return_buffer(self.bucket, buffer);
        }
    }
}

// Safety: PooledBuffer can be sent between threads
unsafe impl Send for PooledBuffer {}
unsafe impl Sync for PooledBuffer {}

/// Global buffer pools for different resource types.
pub struct GlobalPools {
    /// Pool for shared memory buffers.
    pub shared: BufferPool,
    /// Pool for private memory buffers (GPU-only).
    pub private: BufferPool,
}

impl GlobalPools {
    /// Create global buffer pools for the given device.
    pub fn new(device: Device) -> Self {
        Self {
            shared: BufferPool::new(
                device.clone(),
                MTLResourceOptions::StorageModeShared,
            ),
            private: BufferPool::new(
                device,
                MTLResourceOptions::StorageModePrivate,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_bucket() {
        // Test power of 2 rounding
        assert_eq!(SizeBucket::for_size(0).size(), 1);
        assert_eq!(SizeBucket::for_size(1).size(), 1);
        assert_eq!(SizeBucket::for_size(2).size(), 2);
        assert_eq!(SizeBucket::for_size(3).size(), 4);
        assert_eq!(SizeBucket::for_size(1024).size(), 1024);
        assert_eq!(SizeBucket::for_size(1025).size(), 2048);
    }
}