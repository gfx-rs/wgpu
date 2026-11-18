// Copyright 2018-2025 the Deno authors. MIT license.

//! Reports GPU allocations to V8 as external memory, so that garbage
//! collection runs when unreferenced buffers and textures hold a lot of GPU
//! memory. Without this, V8 only sees the small JS wrappers, and a long-running
//! isolate can exhaust GPU memory before it collects them.

use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicI64;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use deno_core::v8;

/// Bytes released by finalizers, which cannot call into the isolate. They are
/// reported on the device's next allocation or `destroy()`.
#[derive(Clone, Default)]
pub(crate) struct PendingRelease(Arc<AtomicI64>);

impl PendingRelease {
  fn report(&self, isolate: &mut v8::Isolate, change: i64) {
    let released = self.0.swap(0, Ordering::Relaxed);
    let change = change.saturating_sub(released);
    if change != 0 {
      isolate.adjust_amount_of_external_allocated_memory(change);
    }
  }
}

/// One resource's share of the isolate's external memory.
pub(crate) struct ExternalAllocation {
  size: i64,
  pending: PendingRelease,
  reported: AtomicBool,
}

impl ExternalAllocation {
  pub(crate) fn new(
    isolate: &mut v8::Isolate,
    pending: &PendingRelease,
    size: u64,
  ) -> Self {
    let size = i64::try_from(size).unwrap_or(i64::MAX);
    pending.report(isolate, size);
    Self {
      size,
      pending: pending.clone(),
      reported: AtomicBool::new(true),
    }
  }

  /// Releases the allocation when the resource is destroyed explicitly.
  pub(crate) fn release(&self, isolate: &mut v8::Isolate) {
    if self.reported.swap(false, Ordering::Relaxed) {
      self.pending.report(isolate, -self.size);
    }
  }
}

impl Drop for ExternalAllocation {
  fn drop(&mut self) {
    if *self.reported.get_mut() {
      self.pending.0.fetch_add(self.size, Ordering::Relaxed);
    }
  }
}

/// Approximate size of a texture, without panicking on invalid descriptors,
/// which wgpu-core reports as validation errors.
pub(crate) fn texture_size<L, V>(
  desc: &wgpu_types::TextureDescriptor<L, V>,
) -> u64 {
  let (block_width, block_height) = desc.format.block_dimensions();
  let block_size = u64::from(desc.format.block_copy_size(None).unwrap_or(4));
  (0..desc.mip_level_count.min(32))
    .map(|level| {
      let size = desc.size.mip_level_size(level, desc.dimension);
      u64::from(size.width.div_ceil(block_width))
        .saturating_mul(u64::from(size.height.div_ceil(block_height)))
        .saturating_mul(u64::from(size.depth_or_array_layers))
        .saturating_mul(block_size)
    })
    .fold(0u64, u64::saturating_add)
    .saturating_mul(u64::from(desc.sample_count))
}
