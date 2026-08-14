use wgpu_core::device::DeviceDescriptor;
use wgpu_core::instance::RequestDeviceError;
use wgt::Backends;

use crate::global::Global;
use crate::id::{AdapterId, DeviceId, QueueId};

pub type RequestAdapterOptions = wgt::RequestAdapterOptions<()>;

impl Global {
    pub fn enumerate_adapters(
        &self,
        backends: Backends,
        apply_limit_buckets: bool,
    ) -> Vec<AdapterId> {
        let adapters = self
            .instance
            .enumerate_adapters(backends, apply_limit_buckets);
        adapters
            .into_iter()
            .map(|adapter| self.hub.adapters.prepare(None).assign(adapter))
            .collect()
    }

    pub fn request_adapter(
        &self,
        desc: &RequestAdapterOptions,
        backends: Backends,
        id_in: Option<AdapterId>,
    ) -> Result<AdapterId, wgt::RequestAdapterError> {
        let desc = wgt::RequestAdapterOptions {
            power_preference: desc.power_preference,
            force_fallback_adapter: desc.force_fallback_adapter,
            compatible_surface: None,
            apply_limit_buckets: desc.apply_limit_buckets,
        };
        let adapter = self.instance.request_adapter(&desc, backends)?;
        let id = self.hub.adapters.prepare(id_in).assign(adapter);
        Ok(id)
    }

    /// Create an adapter from a HAL adapter.
    ///
    /// The HAL adapter may be obtained e.g. by calling `enumerate_adapters` on
    /// the HAL directly.
    ///
    /// If [limit bucketing][lt] is desired, [`crate::limits::apply_limit_buckets`]
    /// should be called with the HAL adapter before calling this function.
    ///
    /// # Safety
    ///
    /// `hal_adapter` must be created from this global internal instance handle.
    ///
    /// [lt]: crate::limits#Limit-bucketing
    pub unsafe fn create_adapter_from_hal(
        &self,
        hal_adapter: hal::DynExposedAdapter,
        input: Option<AdapterId>,
    ) -> AdapterId {
        let fid = self.hub.adapters.prepare(input);
        fid.assign(unsafe { self.instance.create_adapter_from_hal(hal_adapter) })
    }

    pub fn adapter_get_info(&self, adapter_id: AdapterId) -> wgt::AdapterInfo {
        let adapter = self.hub.adapters.get(adapter_id);
        adapter.get_info()
    }

    pub fn adapter_get_texture_format_features(
        &self,
        adapter_id: AdapterId,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        let adapter = self.hub.adapters.get(adapter_id);
        adapter.get_texture_format_features(format)
    }

    pub fn adapter_features(&self, adapter_id: AdapterId) -> wgt::Features {
        let adapter = self.hub.adapters.get(adapter_id);
        adapter.features()
    }

    pub fn adapter_limits(&self, adapter_id: AdapterId) -> wgt::Limits {
        let adapter = self.hub.adapters.get(adapter_id);
        adapter.limits()
    }

    pub fn adapter_downlevel_capabilities(
        &self,
        adapter_id: AdapterId,
    ) -> wgt::DownlevelCapabilities {
        let adapter = self.hub.adapters.get(adapter_id);
        adapter.downlevel_capabilities()
    }

    pub fn adapter_get_presentation_timestamp(
        &self,
        adapter_id: AdapterId,
    ) -> wgt::PresentationTimestamp {
        let adapter = self.hub.adapters.get(adapter_id);
        adapter.get_presentation_timestamp()
    }

    pub fn adapter_cooperative_matrix_properties(
        &self,
        adapter_id: AdapterId,
    ) -> Vec<wgt::CooperativeMatrixProperties> {
        let adapter = self.hub.adapters.get(adapter_id);
        adapter.cooperative_matrix_properties()
    }

    pub fn adapter_drop(&self, adapter_id: AdapterId) {
        self.hub.adapters.remove(adapter_id);
    }
}

impl Global {
    pub fn adapter_request_device(
        &self,
        adapter_id: AdapterId,
        desc: &DeviceDescriptor,
        device_id_in: Option<DeviceId>,
        queue_id_in: Option<QueueId>,
    ) -> Result<(DeviceId, QueueId), RequestDeviceError> {
        let device_fid = self.hub.devices.prepare(device_id_in);
        let queue_fid = self.hub.queues.prepare(queue_id_in);

        let adapter = self.hub.adapters.get(adapter_id);
        let (device, queue) = adapter.request_device(desc)?;

        let device_id = device_fid.assign(device);

        let queue_id = queue_fid.assign(queue);

        Ok((device_id, queue_id))
    }

    pub fn adapter_validate_device_descriptor(
        &self,
        adapter_id: AdapterId,
        desc: &mut DeviceDescriptor,
    ) -> Result<(), RequestDeviceError> {
        let adapter = self.hub.adapters.get(adapter_id);
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
        device_id_in: Option<DeviceId>,
        queue_id_in: Option<QueueId>,
    ) -> Result<(DeviceId, QueueId), RequestDeviceError> {
        let devices_fid = self.hub.devices.prepare(device_id_in);
        let queues_fid = self.hub.queues.prepare(queue_id_in);

        let adapter = self.hub.adapters.get(adapter_id);
        let (device, queue) =
            unsafe { adapter.create_device_and_queue_from_hal(hal_device, desc) }?;

        let device_id = devices_fid.assign(device);

        let queue_id = queues_fid.assign(queue);

        Ok((device_id, queue_id))
    }
}
