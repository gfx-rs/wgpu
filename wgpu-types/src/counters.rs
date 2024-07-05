#[cfg(feature = "counters")]
use std::sync::atomic::{AtomicIsize, Ordering};

/// An internal counter for debugging purposes
///
/// Internally represented as an atomic isize if the `counters` feature is enabled,
/// or compiles to nothing otherwise.
pub struct InternalCounter {
    #[cfg(feature = "counters")]
    value: AtomicIsize,
}

impl InternalCounter {
    /// Creates a counter with value 0.
    #[inline]
    pub const fn new() -> Self {
        InternalCounter {
            #[cfg(feature = "counters")]
            value: AtomicIsize::new(0),
        }
    }

    /// Get the counter's value.
    #[cfg(feature = "counters")]
    #[inline]
    pub fn read(&self) -> isize {
        self.value.load(Ordering::Relaxed)
    }

    /// Get the counter's value.
    ///
    /// Always returns 0 if the `counters` feature is not enabled.
    #[cfg(not(feature = "counters"))]
    #[inline]
    pub fn read(&self) -> isize {
        0
    }

    /// Get and reset the counter's value.
    ///
    /// Always returns 0 if the `counters` feature is not enabled.
    #[cfg(feature = "counters")]
    #[inline]
    pub fn take(&self) -> isize {
        self.value.swap(0, Ordering::Relaxed)
    }

    /// Get and reset the counter's value.
    ///
    /// Always returns 0 if the `counters` feature is not enabled.
    #[cfg(not(feature = "counters"))]
    #[inline]
    pub fn take(&self) -> isize {
        0
    }

    /// Increment the counter by the provided amount.
    #[inline]
    pub fn add(&self, _val: isize) {
        #[cfg(feature = "counters")]
        self.value.fetch_add(_val, Ordering::Relaxed);
    }

    /// Decrement the counter by the provided amount.
    #[inline]
    pub fn sub(&self, _val: isize) {
        #[cfg(feature = "counters")]
        self.value.fetch_add(-_val, Ordering::Relaxed);
    }

    /// Sets the counter to the provided value.
    #[inline]
    pub fn set(&self, _val: isize) {
        #[cfg(feature = "counters")]
        self.value.store(_val, Ordering::Relaxed);
    }
}

impl Clone for InternalCounter {
    fn clone(&self) -> Self {
        InternalCounter {
            #[cfg(feature = "counters")]
            value: AtomicIsize::new(self.read()),
        }
    }
}

impl Default for InternalCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for InternalCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.read().fmt(f)
    }
}

/// `wgpu-hal`'s internal counters.
#[allow(missing_docs)]
#[derive(Clone, Default)]
pub struct HalCounters {
    // API objects
    pub buffers: InternalCounter,
    pub textures: InternalCounter,
    pub texture_views: InternalCounter,
    pub bind_groups: InternalCounter,
    pub bind_group_layouts: InternalCounter,
    pub render_pipelines: InternalCounter,
    pub compute_pipelines: InternalCounter,
    pub pipeline_layouts: InternalCounter,
    pub samplers: InternalCounter,
    pub command_encoders: InternalCounter,
    pub shader_modules: InternalCounter,
    pub query_sets: InternalCounter,
    pub fences: InternalCounter,

    // Resources
    /// Amount of allocated gpu memory attributed to buffers, in bytes.
    pub buffer_memory: InternalCounter,
    /// Amount of allocated gpu memory attributed to textures, in bytes.
    pub texture_memory: InternalCounter,
    /// Number of gpu memory allocations.
    pub memory_allocations: InternalCounter,
}

/// `wgpu-core`'s internal counters.
#[derive(Clone, Default)]
pub struct CoreCounters {
    // TODO
}

/// All internal counters, exposed for debugging purposes.
#[derive(Clone, Default)]
pub struct InternalCounters {
    /// `wgpu-core` counters.
    pub core: CoreCounters,
    /// `wgpu-hal` counters.
    pub hal: HalCounters,
}
