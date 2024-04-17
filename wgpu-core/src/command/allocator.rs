use crate::hal_api::HalApi;
use crate::resource_log;
use hal::Device as _;

use parking_lot::Mutex;

/// A pool of free [`wgpu_hal::CommandEncoder`]s, owned by a `Device`.
///
/// Each encoder in this list is in the "closed" state.
///
/// Since a raw [`CommandEncoder`][ce] is itself a pool for allocating
/// raw [`CommandBuffer`][cb]s, this is a pool of pools.
///
/// [ce]: wgpu_hal::CommandEncoder
/// [cb]: wgpu_hal::Api::CommandBuffer
pub(crate) struct CommandAllocator<A: HalApi> {
    free_encoders: Mutex<Vec<A::CommandEncoder>>,
}

impl<A: HalApi> CommandAllocator<A> {
    pub(crate) fn new() -> Self {
        Self {
            free_encoders: Mutex::new(Vec::new()),
        }
    }

    /// Return a fresh [`wgpu_hal::CommandEncoder`] in the "closed" state.
    ///
    /// If we have free encoders in the pool, take one of those. Otherwise,
    /// create a new one on `device`.
    pub(crate) fn acquire_encoder(
        &self,
        device: &A::Device,
        queue: &A::Queue,
    ) -> Result<A::CommandEncoder, hal::DeviceError> {
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
    pub(crate) fn release_encoder(&self, encoder: A::CommandEncoder) {
        let mut free_encoders = self.free_encoders.lock();
        free_encoders.push(encoder);
    }

    /// Free the pool of command encoders.
    ///
    /// This is only called when the `Device` is dropped.
    pub(crate) fn dispose(&self, device: &A::Device) {
        let mut free_encoders = self.free_encoders.lock();
        resource_log!("CommandAllocator::dispose encoders {}", free_encoders.len());
        for cmd_encoder in free_encoders.drain(..) {
            unsafe {
                device.destroy_command_encoder(cmd_encoder);
            }
        }
    }
}
