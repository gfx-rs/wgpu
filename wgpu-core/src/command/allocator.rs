use crate::resource_log;

use crate::lock::{rank, Mutex};

/// A pool of free [`wgpu_hal::CommandEncoder`]s, owned by a `Device`.
///
/// Each encoder in this list is in the "closed" state.
///
/// Since a raw [`CommandEncoder`][ce] is itself a pool for allocating
/// raw [`CommandBuffer`][cb]s, this is a pool of pools.
///
/// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
/// [ce]: hal::CommandEncoder
/// [cb]: hal::Api::CommandBuffer
pub(crate) struct CommandAllocator {
    free_encoders: Mutex<Vec<Box<dyn hal::DynCommandEncoder>>>,
}

impl CommandAllocator {
    pub(crate) fn new() -> Self {
        Self {
            free_encoders: Mutex::new(rank::COMMAND_ALLOCATOR_FREE_ENCODERS, Vec::new()),
        }
    }

    /// Return a fresh [`wgpu_hal::CommandEncoder`] in the "closed" state.
    ///
    /// If we have free encoders in the pool, take one of those. Otherwise,
    /// create a new one on `device`.
    ///
    /// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
    pub(crate) fn acquire_encoder(
        &self,
        device: &dyn hal::DynDevice,
        queue: &dyn hal::DynQueue,
    ) -> Result<Box<dyn hal::DynCommandEncoder>, hal::DeviceError> {
        let mut free_encoders = self.free_encoders.lock();
        match free_encoders.pop() {
            Some(encoder) => Ok(encoder),
            None => unsafe {
                let hal_desc = hal::CommandEncoderDescriptor { label: None, queue };
                device.create_command_encoder(&hal_desc)
            },
        }
    }

    /// Add `encoder` back to the free pool.
    pub(crate) fn release_encoder(&self, encoder: Box<dyn hal::DynCommandEncoder>) {
        let mut free_encoders = self.free_encoders.lock();
        free_encoders.push(encoder);
    }

    /// Free the pool of command encoders.
    ///
    /// This is only called when the `Device` is dropped.
    pub(crate) fn dispose(&self, device: &dyn hal::DynDevice) {
        let mut free_encoders = self.free_encoders.lock();
        resource_log!("CommandAllocator::dispose encoders {}", free_encoders.len());
        for cmd_encoder in free_encoders.drain(..) {
            unsafe {
                device.destroy_command_encoder(cmd_encoder);
            }
        }
    }
}
