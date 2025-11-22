//! Flattened twiddle buffer management for Metal backend.
//!
//! This module provides efficient management of twiddle factors by storing them
//! in a single large buffer with offset tracking instead of allocating separate
//! buffers for each layer.

use metal::{Buffer, Device, MTLResourceOptions};
use std::collections::HashMap;
use std::sync::Mutex;

/// A handle to a twiddle section within the flat buffer.
#[derive(Debug, Clone, Copy)]
pub struct TwiddleSection {
    /// Byte offset into the flat buffer.
    pub offset: u64,
    /// Number of twiddle elements (u32 values).
    pub len: u32,
}

/// Manages flattened twiddle buffers for improved GPU performance.
///
/// Instead of creating separate Metal buffers for each twiddle layer,
/// this manager packs all twiddles into a single buffer and tracks
/// offsets for each section. This reduces buffer allocation overhead
/// and improves GPU memory locality.
pub struct FlatTwiddleManager {
    /// Cache of flattened twiddle buffers.
    /// Key is a hash of the combined twiddle data.
    cache: Mutex<HashMap<u64, FlatTwiddleBuffer>>,
}

/// A flattened twiddle buffer containing multiple twiddle layers.
pub struct FlatTwiddleBuffer {
    /// The single Metal buffer containing all twiddles.
    pub buffer: Buffer,
    /// Mapping from layer index to section info.
    pub sections: Vec<TwiddleSection>,
    /// Total size in bytes.
    pub total_size: u64,
}

impl FlatTwiddleManager {
    /// Create a new twiddle manager.
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a flattened twiddle buffer from multiple layers.
    ///
    /// This function takes multiple twiddle layers and packs them into a
    /// single Metal buffer, returning the buffer and section information
    /// for each layer.
    pub fn get_or_create_flat_buffer(
        &self,
        device: &Device,
        twiddle_layers: &[&[u32]],
    ) -> FlatTwiddleBuffer {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Compute cache key from twiddle data
        let mut hasher = DefaultHasher::new();
        twiddle_layers.len().hash(&mut hasher);

        // Sample a few values from each layer for hashing
        for layer in twiddle_layers {
            layer.len().hash(&mut hasher);
            if !layer.is_empty() {
                layer[0].hash(&mut hasher);
                if layer.len() > 1 {
                    layer[layer.len() / 2].hash(&mut hasher);
                    layer[layer.len() - 1].hash(&mut hasher);
                }
            }
        }
        let cache_key = hasher.finish();

        // Check cache first
        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                return FlatTwiddleBuffer {
                    buffer: cached.buffer.clone(),
                    sections: cached.sections.clone(),
                    total_size: cached.total_size,
                };
            }
        }

        // Create flat buffer if not cached
        let flat_buffer = self.create_flat_buffer(device, twiddle_layers);

        // Store in cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(cache_key, FlatTwiddleBuffer {
                buffer: flat_buffer.buffer.clone(),
                sections: flat_buffer.sections.clone(),
                total_size: flat_buffer.total_size,
            });
        }

        flat_buffer
    }

    /// Create a new flat buffer from twiddle layers.
    fn create_flat_buffer(
        &self,
        device: &Device,
        twiddle_layers: &[&[u32]],
    ) -> FlatTwiddleBuffer {
        // Calculate total size and sections
        let mut sections = Vec::with_capacity(twiddle_layers.len());
        let mut current_offset = 0u64;
        let mut flat_data = Vec::new();

        for layer in twiddle_layers {
            sections.push(TwiddleSection {
                offset: current_offset,
                len: layer.len() as u32,
            });

            flat_data.extend_from_slice(layer);
            current_offset += (layer.len() * std::mem::size_of::<u32>()) as u64;
        }

        let total_size = current_offset;

        // Create single Metal buffer with all twiddles
        let buffer = if !flat_data.is_empty() {
            device.new_buffer_with_data(
                flat_data.as_ptr() as *const _,
                total_size,
                MTLResourceOptions::StorageModeShared,
            )
        } else {
            // Empty buffer case
            device.new_buffer(
                std::mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };

        FlatTwiddleBuffer {
            buffer,
            sections,
            total_size,
        }
    }

    /// Clear the cache to free memory.
    #[allow(dead_code)]
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }
}

impl Default for FlatTwiddleManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for passing twiddle sections to Metal compute encoders.
#[allow(dead_code)]
pub trait TwiddleEncoderExt {
    /// Set buffer with offset for a specific twiddle section.
    fn set_twiddle_section(
        &self,
        index: usize,
        buffer: &Buffer,
        section: TwiddleSection,
    );
}

impl TwiddleEncoderExt for metal::ComputeCommandEncoderRef {
    fn set_twiddle_section(
        &self,
        index: usize,
        buffer: &Buffer,
        section: TwiddleSection,
    ) {
        // Set the buffer at the given index with the section's offset
        self.set_buffer(index as u64, Some(buffer), section.offset);
    }
}