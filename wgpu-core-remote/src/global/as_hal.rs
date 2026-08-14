use core::ops::Deref;

use crate::global::Global;
use crate::id::*;

impl Global {
    /// # Safety
    ///
    /// - The raw buffer handle must not be manually destroyed
    pub unsafe fn buffer_as_hal<A: hal::Api>(
        &self,
        id: BufferId,
    ) -> Option<impl Deref<Target = A::Buffer>> {
        let hub = &self.hub;

        let buffer = hub.buffers.get(id);

        unsafe { buffer.as_hal::<A>() }
    }

    /// # Safety
    ///
    /// - The raw texture handle must not be manually destroyed
    pub unsafe fn texture_as_hal<A: hal::Api>(
        &self,
        id: TextureId,
    ) -> Option<impl Deref<Target = A::Texture>> {
        let hub = &self.hub;

        let texture = hub.textures.get(id);

        unsafe { texture.as_hal::<A>() }
    }

    /// # Safety
    ///
    /// - The raw texture view handle must not be manually destroyed
    pub unsafe fn texture_view_as_hal<A: hal::Api>(
        &self,
        id: TextureViewId,
    ) -> Option<impl Deref<Target = A::TextureView>> {
        let hub = &self.hub;

        let view = hub.texture_views.get(id);

        unsafe { view.as_hal::<A>() }
    }

    /// # Safety
    ///
    /// - The raw adapter handle must not be manually destroyed
    pub unsafe fn adapter_as_hal<A: hal::Api>(
        &self,
        id: AdapterId,
    ) -> Option<impl Deref<Target = A::Adapter>> {
        let hub = &self.hub;
        let adapter = hub.adapters.get(id);

        unsafe { adapter.as_hal::<A>() }
    }

    /// # Safety
    ///
    /// - The raw device handle must not be manually destroyed
    pub unsafe fn device_as_hal<A: hal::Api>(
        &self,
        id: DeviceId,
    ) -> Option<impl Deref<Target = A::Device>> {
        let device = self.hub.devices.get(id);

        unsafe { device.as_hal::<A>() }
    }

    /// # Safety
    ///
    /// - The raw fence handle must not be manually destroyed
    pub unsafe fn device_fence_as_hal<A: hal::Api>(
        &self,
        id: DeviceId,
    ) -> Option<impl Deref<Target = A::Fence>> {
        let device = self.hub.devices.get(id);

        unsafe { device.fence_as_hal::<A>() }
    }

    /// Encode commands using the raw HAL command encoder.
    ///
    /// # Panics
    ///
    /// If the command encoder has already been used with the wgpu encoding API.
    ///
    /// # Safety
    ///
    /// - The raw command encoder handle must not be manually destroyed
    pub unsafe fn command_encoder_as_hal_mut<
        A: hal::Api,
        F: FnOnce(Option<&mut A::CommandEncoder>) -> R,
        R,
    >(
        &self,
        id: CommandEncoderId,
        hal_command_encoder_callback: F,
    ) -> R {
        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(id);
        unsafe { cmd_enc.as_hal_mut::<A, F, R>(hal_command_encoder_callback) }
    }

    /// # Safety
    ///
    /// - The raw queue handle must not be manually destroyed
    pub unsafe fn queue_as_hal<A: hal::Api>(
        &self,
        id: QueueId,
    ) -> Option<impl Deref<Target = A::Queue>> {
        let queue = self.hub.queues.get(id);

        unsafe { queue.as_hal::<A>() }
    }
}
