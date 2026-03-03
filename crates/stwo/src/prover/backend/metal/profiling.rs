//! Profiling and timing instrumentation for Metal backend operations.
//!
//! This module provides macros and utilities to measure wall-clock time
//! for critical operations and log whether GPU or SIMD path was taken.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Global flag to enable/disable profiling.
/// Can be controlled via environment variable METAL_PROFILE=1
static PROFILING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Initialize profiling based on environment variable.
pub fn init_profiling() {
    if std::env::var("METAL_PROFILE").is_ok() {
        PROFILING_ENABLED.store(true, Ordering::Relaxed);
        eprintln!("[METAL] Profiling enabled");
    }
}

/// Check if profiling is enabled.
#[inline]
pub fn is_profiling_enabled() -> bool {
    PROFILING_ENABLED.load(Ordering::Relaxed)
}

/// Scoped timer that logs elapsed time on drop.
pub struct ScopedTimer {
    name: &'static str,
    start: Instant,
    metadata: String,
    backend: &'static str,
}

impl ScopedTimer {
    pub fn new(name: &'static str, metadata: String, backend: &'static str) -> Self {
        Self {
            name,
            start: Instant::now(),
            metadata,
            backend,
        }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        eprintln!(
            "[METAL_PROFILE] {} | backend={} | {} | time={:.3}ms",
            self.name,
            self.backend,
            self.metadata,
            elapsed.as_secs_f64() * 1000.0
        );
    }
}

/// Macro to time a block of code with metadata.
///
/// Usage:
/// ```ignore
/// metal_profile!("evaluate", format!("log_size={}", log_size), "GPU", {
///     // code to profile
/// });
/// ```
#[macro_export]
macro_rules! metal_profile {
    ($name:expr, $metadata:expr, $backend:expr, $body:block) => {{
        let _timer = if $crate::prover::backend::metal::profiling::is_profiling_enabled() {
            Some($crate::prover::backend::metal::profiling::ScopedTimer::new(
                $name, $metadata, $backend,
            ))
        } else {
            None
        };
        $body
    }};
}

/// Macro to time a function call with automatic metadata.
#[macro_export]
macro_rules! metal_profile_fn {
    ($name:expr, $backend:expr, $($key:ident = $value:expr),* $(,)?) => {{
        let metadata = format!(
            concat!($(stringify!($key), "={}, "),*),
            $($value),*
        );
        let metadata = metadata.trim_end_matches(", ").to_string();

        if $crate::prover::backend::metal::profiling::is_profiling_enabled() {
            Some($crate::prover::backend::metal::profiling::ScopedTimer::new(
                $name,
                metadata,
                $backend,
            ))
        } else {
            None
        }
    }};
}
