use alloc::sync::Arc;

use wgpu_core::instance::RequestDeviceError;
use wgpu_core_remote_types::DeviceDescriptor;
use wgpu_core_remote_types::RequestAdapterOptions;
use wgt::Backends;

use crate::global::Global;
use crate::hub::Hub;
use crate::id::{AdapterId, DeviceId, QueueId};

impl Global {
    pub fn request_adapter(
        &self,
        desc: &RequestAdapterOptions,
        apply_limit_buckets: bool,
        backends: Backends,
        id_in: AdapterId,
    ) -> Result<AdapterId, wgt::RequestAdapterError> {
        let mut hub = self.hub.borrow_mut();
        let desc = wgt::RequestAdapterOptions {
            power_preference: desc.power_preference,
            force_fallback_adapter: desc.force_fallback_adapter,
            compatible_surface: None,
            apply_limit_buckets,
        };
        let adapter = self.instance.request_adapter(&desc, backends)?;
        let id = hub.adapters.assign(id_in, adapter);
        Ok(id)
    }

    /// Create an adapter from a HAL adapter.
    ///
    /// The HAL adapter may be obtained e.g. by calling `enumerate_adapters` on
    /// the HAL directly.
    ///
    /// If [limit bucketing][lt] is desired, [`wgpu_core::limits::apply_limit_buckets`]
    /// should be called with the HAL adapter before calling this function.
    ///
    /// # Safety
    ///
    /// `hal_adapter` must be created from this global internal instance handle.
    ///
    /// [lt]: wgpu_core::limits#Limit-bucketing
    pub unsafe fn create_adapter_from_hal(
        &self,
        hal_adapter: hal::DynExposedAdapter,
        id_in: AdapterId,
    ) -> AdapterId {
        let mut hub = self.hub.borrow_mut();
        hub.adapters.assign(id_in, unsafe {
            self.instance.create_adapter_from_hal(hal_adapter)
        })
    }

    pub fn adapter_get_info(&self, adapter_id: AdapterId) -> wgt::AdapterInfo {
        let hub = self.hub.borrow();
        let adapter = hub.adapters.get(adapter_id);
        adapter.get_info()
    }

    pub fn adapter_get_texture_format_features(
        &self,
        adapter_id: AdapterId,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        let hub = self.hub.borrow();
        let adapter = hub.adapters.get(adapter_id);
        adapter.get_texture_format_features(format)
    }

    pub fn adapter_features(&self, adapter_id: AdapterId) -> wgt::Features {
        let hub = self.hub.borrow();
        let adapter = hub.adapters.get(adapter_id);
        adapter.features()
    }

    pub fn adapter_limits(&self, adapter_id: AdapterId) -> wgt::Limits {
        let hub = self.hub.borrow();
        let adapter = hub.adapters.get(adapter_id);
        adapter.limits()
    }

    pub fn adapter_downlevel_capabilities(
        &self,
        adapter_id: AdapterId,
    ) -> wgt::DownlevelCapabilities {
        let hub = self.hub.borrow();
        let adapter = hub.adapters.get(adapter_id);
        adapter.downlevel_capabilities()
    }

    pub fn adapter_get_presentation_timestamp(
        &self,
        adapter_id: AdapterId,
    ) -> wgt::PresentationTimestamp {
        let hub = self.hub.borrow();
        let adapter = hub.adapters.get(adapter_id);
        adapter.get_presentation_timestamp()
    }

    pub fn adapter_cooperative_matrix_properties(
        &self,
        adapter_id: AdapterId,
    ) -> Vec<wgt::CooperativeMatrixProperties> {
        let hub = self.hub.borrow();
        let adapter = hub.adapters.get(adapter_id);
        adapter.cooperative_matrix_properties()
    }

    pub fn adapter_remove(&self, adapter_id: AdapterId) -> Arc<wgpu_core::instance::Adapter> {
        let mut hub = self.hub.borrow_mut();
        hub.adapters.remove(adapter_id)
    }
}

impl Global {
    pub fn adapter_request_device(
        &self,
        adapter_id: AdapterId,
        desc: &DeviceDescriptor,
        device_id_in: DeviceId,
        queue_id_in: QueueId,
    ) -> Result<(DeviceId, QueueId), RequestDeviceError> {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            adapters,
            devices,
            queues,
            ..
        } = &mut *hub;

        let adapter = adapters.get(adapter_id);
        let (device, queue) = adapter.request_device(desc)?;

        let device_id = devices.assign(device_id_in, device);

        let queue_id = queues.assign(queue_id_in, queue);

        Ok((device_id, queue_id))
    }

    pub fn adapter_validate_device_descriptor(
        &self,
        adapter_id: AdapterId,
        desc: &mut DeviceDescriptor,
    ) -> Result<(), RequestDeviceError> {
        let hub = self.hub.borrow();
        let adapter = hub.adapters.get(adapter_id);
        adapter.validate_device_descriptor(desc)
    }

    /// # Safety
    ///
    /// - `hal_device` must be created from `adapter_id` or its internal handle.
    /// - `desc` must be a subset of `hal_device` features and limits.
    pub unsafe fn create_device_from_hal(
        &self,
        adapter_id: AdapterId,
        hal_device: hal::DynOpenDevice,
        desc: &DeviceDescriptor,
        device_id_in: DeviceId,
        queue_id_in: QueueId,
    ) -> Result<(DeviceId, QueueId), RequestDeviceError> {
        let mut hub = self.hub.borrow_mut();
        let Hub {
            adapters,
            devices,
            queues,
            ..
        } = &mut *hub;

        let adapter = adapters.get(adapter_id);
        let (device, queue) =
            unsafe { adapter.create_device_and_queue_from_hal(hal_device, desc) }?;

        let device_id = devices.assign(device_id_in, device);

        let queue_id = queues.assign(queue_id_in, queue);

        Ok((device_id, queue_id))
    }
}
