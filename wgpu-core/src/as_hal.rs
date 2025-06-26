use crate::{
    global::Global,
    hal_api::HalApi,
    id::{
        AdapterId, BlasId, BufferId, CommandEncoderId, DeviceId, QueueId, SurfaceId, TextureId,
        TextureViewId, TlasId,
    },
    resource::AccelerationStructure,
};

impl Global {
    /// # Safety
    ///
    /// - The raw buffer handle must not be manually destroyed
    pub unsafe fn buffer_as_hal<A: HalApi, F: FnOnce(Option<&A::Buffer>) -> R, R>(
        &self,
        id: BufferId,
        hal_buffer_callback: F,
    ) -> R {
        profiling::scope!("Buffer::as_hal");

        let hub = &self.hub;

        if let Ok(buffer) = hub.buffers.get(id).get() {
            let snatch_guard = buffer.device.snatchable_lock.read();
            let hal_buffer = buffer
                .raw(&snatch_guard)
                .and_then(|b| b.as_any().downcast_ref());
            hal_buffer_callback(hal_buffer)
        } else {
            hal_buffer_callback(None)
        }
    }

    /// # Safety
    ///
    /// - The raw texture handle must not be manually destroyed
    pub unsafe fn texture_as_hal<A: HalApi, F: FnOnce(Option<&A::Texture>) -> R, R>(
        &self,
        id: TextureId,
        hal_texture_callback: F,
    ) -> R {
        profiling::scope!("Texture::as_hal");

        let hub = &self.hub;

        if let Ok(texture) = hub.textures.get(id).get() {
            let snatch_guard = texture.device.snatchable_lock.read();
            let hal_texture = texture.raw(&snatch_guard);
            let hal_texture = hal_texture
                .as_ref()
                .and_then(|it| it.as_any().downcast_ref());
            hal_texture_callback(hal_texture)
        } else {
            hal_texture_callback(None)
        }
    }

    /// # Safety
    ///
    /// - The raw texture view handle must not be manually destroyed
    pub unsafe fn texture_view_as_hal<A: HalApi, F: FnOnce(Option<&A::TextureView>) -> R, R>(
        &self,
        id: TextureViewId,
        hal_texture_view_callback: F,
    ) -> R {
        profiling::scope!("TextureView::as_hal");

        let hub = &self.hub;

        if let Ok(texture_view) = hub.texture_views.get(id).get() {
            let snatch_guard = texture_view.device.snatchable_lock.read();
            let hal_texture_view = texture_view.raw(&snatch_guard);
            let hal_texture_view = hal_texture_view
                .as_ref()
                .and_then(|it| it.as_any().downcast_ref());
            hal_texture_view_callback(hal_texture_view)
        } else {
            hal_texture_view_callback(None)
        }
    }

    /// # Safety
    ///
    /// - The raw adapter handle must not be manually destroyed
    pub unsafe fn adapter_as_hal<A: HalApi, F: FnOnce(Option<&A::Adapter>) -> R, R>(
        &self,
        id: AdapterId,
        hal_adapter_callback: F,
    ) -> R {
        profiling::scope!("Adapter::as_hal");

        let hub = &self.hub;
        let adapter = hub.adapters.get(id);
        let hal_adapter = adapter.raw.adapter.as_any().downcast_ref();

        hal_adapter_callback(hal_adapter)
    }

    /// # Safety
    ///
    /// - The raw device handle must not be manually destroyed
    pub unsafe fn device_as_hal<A: HalApi, F: FnOnce(Option<&A::Device>) -> R, R>(
        &self,
        id: DeviceId,
        hal_device_callback: F,
    ) -> R {
        profiling::scope!("Device::as_hal");

        let device = self.hub.devices.get(id);
        let hal_device = device.raw().as_any().downcast_ref();

        hal_device_callback(hal_device)
    }

    /// # Safety
    ///
    /// - The raw fence handle must not be manually destroyed
    pub unsafe fn device_fence_as_hal<A: HalApi, F: FnOnce(Option<&A::Fence>) -> R, R>(
        &self,
        id: DeviceId,
        hal_fence_callback: F,
    ) -> R {
        profiling::scope!("Device::fence_as_hal");

        let device = self.hub.devices.get(id);
        let fence = device.fence.read();
        hal_fence_callback(fence.as_any().downcast_ref())
    }

    /// # Safety
    /// - The raw surface handle must not be manually destroyed
    pub unsafe fn surface_as_hal<A: HalApi, F: FnOnce(Option<&A::Surface>) -> R, R>(
        &self,
        id: SurfaceId,
        hal_surface_callback: F,
    ) -> R {
        profiling::scope!("Surface::as_hal");

        let surface = self.surfaces.get(id);
        let hal_surface = surface
            .raw(A::VARIANT)
            .and_then(|surface| surface.as_any().downcast_ref());

        hal_surface_callback(hal_surface)
    }

    /// # Safety
    ///
    /// - The raw command encoder handle must not be manually destroyed
    pub unsafe fn command_encoder_as_hal_mut<
        A: HalApi,
        F: FnOnce(Option<&mut A::CommandEncoder>) -> R,
        R,
    >(
        &self,
        id: CommandEncoderId,
        hal_command_encoder_callback: F,
    ) -> R {
        profiling::scope!("CommandEncoder::as_hal");

        let hub = &self.hub;

        let cmd_buf = hub.command_buffers.get(id.into_command_buffer_id());
        let mut cmd_buf_data = cmd_buf.data.lock();
        cmd_buf_data.record_as_hal_mut(|opt_cmd_buf| -> R {
            hal_command_encoder_callback(opt_cmd_buf.and_then(|cmd_buf| {
                cmd_buf
                    .encoder
                    .open()
                    .ok()
                    .and_then(|encoder| encoder.as_any_mut().downcast_mut())
            }))
        })
    }

    /// # Safety
    ///
    /// - The raw queue handle must not be manually destroyed
    pub unsafe fn queue_as_hal<A: HalApi, F, R>(&self, id: QueueId, hal_queue_callback: F) -> R
    where
        F: FnOnce(Option<&A::Queue>) -> R,
    {
        profiling::scope!("Queue::as_hal");

        let queue = self.hub.queues.get(id);
        let hal_queue = queue.raw().as_any().downcast_ref();

        hal_queue_callback(hal_queue)
    }

    /// # Safety
    ///
    /// - The raw blas handle must not be manually destroyed
    pub unsafe fn blas_as_hal<A: HalApi, F: FnOnce(Option<&A::AccelerationStructure>) -> R, R>(
        &self,
        id: BlasId,
        hal_blas_callback: F,
    ) -> R {
        profiling::scope!("Blas::as_hal");

        let hub = &self.hub;

        if let Ok(blas) = hub.blas_s.get(id).get() {
            let snatch_guard = blas.device.snatchable_lock.read();
            let hal_blas = blas
                .try_raw(&snatch_guard)
                .ok()
                .and_then(|b| b.as_any().downcast_ref());
            hal_blas_callback(hal_blas)
        } else {
            hal_blas_callback(None)
        }
    }

    /// # Safety
    ///
    /// - The raw tlas handle must not be manually destroyed
    pub unsafe fn tlas_as_hal<A: HalApi, F: FnOnce(Option<&A::AccelerationStructure>) -> R, R>(
        &self,
        id: TlasId,
        hal_tlas_callback: F,
    ) -> R {
        profiling::scope!("Blas::as_hal");

        let hub = &self.hub;

        if let Ok(tlas) = hub.tlas_s.get(id).get() {
            let snatch_guard = tlas.device.snatchable_lock.read();
            let hal_tlas = tlas
                .try_raw(&snatch_guard)
                .ok()
                .and_then(|t| t.as_any().downcast_ref());
            hal_tlas_callback(hal_tlas)
        } else {
            hal_tlas_callback(None)
        }
    }
}
