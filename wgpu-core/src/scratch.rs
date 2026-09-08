use alloc::{boxed::Box, sync::Arc};
use core::mem::ManuallyDrop;

use wgt::BufferUses;

use crate::device::{Device, DeviceError};
use crate::{hal_label, resource_log};

/// A scratch buffer parked in [`Device::scratch_buffer_cache`] between
/// acceleration-structure builds, so repeated builds (streaming geometry,
/// per-frame rebuilds) reuse one allocation instead of allocating and freeing
/// a large buffer every build.
///
/// Holds the raw buffer only - deliberately no `Arc<Device>`, so the cache on
/// the device does not create a reference cycle. The device destroys any parked
/// buffer in its own `Drop`.
#[derive(Debug)]
pub(crate) struct CachedScratchBuffer {
    pub(crate) raw: Box<dyn hal::DynBuffer>,
    pub(crate) size: wgt::BufferSize,
}

/// A GPU buffer used as build scratch for acceleration-structure builds.
///
/// Scratch buffers can be large (proportional to the geometry being built), so
/// rather than allocating one per build, [`ScratchBuffer::new`] reuses the
/// device's parked scratch buffer when it is large enough, and `Drop` parks the
/// buffer back in the cache, keeping the larger of the two. A `ScratchBuffer`
/// only reaches `Drop` once the submission using it has retired (it is carried
/// as a [`TempResource`] of that submission) or before it was ever submitted
/// (encode error paths, device teardown), so a parked buffer is always idle and
/// reusing it needs no additional synchronization.
///
/// [`TempResource`]: crate::device::queue::TempResource
#[derive(Debug)]
pub struct ScratchBuffer {
    raw: ManuallyDrop<Box<dyn hal::DynBuffer>>,
    size: wgt::BufferSize,
    device: Arc<Device>,
}

impl ScratchBuffer {
    pub(crate) fn new(device: &Arc<Device>, size: wgt::BufferSize) -> Result<Self, DeviceError> {
        // Reuse the parked scratch buffer if it is large enough. Its actual
        // (possibly larger) size is carried along, so parking it again keeps
        // the largest scratch the device has needed so far.
        let cached = {
            let mut cache = device.scratch_buffer_cache.lock();
            match &*cache {
                Some(c) if c.size >= size => cache.take(),
                _ => None,
            }
        };
        if let Some(cached) = cached {
            return Ok(Self {
                raw: ManuallyDrop::new(cached.raw),
                size: cached.size,
                device: device.clone(),
            });
        }
        let raw = unsafe {
            device
                .raw()
                .create_buffer(&hal::BufferDescriptor {
                    label: hal_label(Some("(wgpu) scratch buffer"), device.instance_flags),
                    size: size.get(),
                    usage: BufferUses::ACCELERATION_STRUCTURE_SCRATCH,
                    memory_flags: hal::MemoryFlags::empty(),
                })
                .map_err(DeviceError::from_hal)?
        };
        Ok(Self {
            raw: ManuallyDrop::new(raw),
            size,
            device: device.clone(),
        })
    }
    pub(crate) fn raw(&self) -> &dyn hal::DynBuffer {
        self.raw.as_ref()
    }
}

impl Drop for ScratchBuffer {
    fn drop(&mut self) {
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        // Park the buffer for the next build; keep the larger of this buffer
        // and any already-parked one, and destroy the other.
        let evicted = {
            let mut cache = self.device.scratch_buffer_cache.lock();
            match &*cache {
                Some(c) if c.size >= self.size => Some(raw),
                _ => cache
                    .replace(CachedScratchBuffer {
                        raw,
                        size: self.size,
                    })
                    .map(|c| c.raw),
            }
        };
        if let Some(buffer) = evicted {
            resource_log!("Destroy raw ScratchBuffer");
            unsafe { self.device.raw().destroy_buffer(buffer) };
        }
    }
}
