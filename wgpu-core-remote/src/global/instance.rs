use alloc::{borrow::Cow, format};

use alloc::sync::Arc;

use wgpu_core_remote_types::DeviceDescriptor;
use wgpu_core_remote_types::Label;
use wgpu_core_remote_types::RequestAdapterOptions;
use wgpu_core_remote_types::RequestDeviceError;
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

fn map_request_device_error(err: wgpu_core::instance::RequestDeviceError) -> RequestDeviceError {
    match err {
        e @ wgpu_core::instance::RequestDeviceError::LimitsExceeded(_) => {
            RequestDeviceError::FailedLimit(e.to_string())
        }
        e @ wgpu_core::instance::RequestDeviceError::UnsupportedFeature(_) => {
            RequestDeviceError::UnsupportedFeature(e.to_string())
        }
        e => RequestDeviceError::Other(e.to_string()),
    }
}

/// Accepts a `DeviceDescriptor` from the content process
/// and accepts other parameters from trusted process.
pub fn map_device_descriptor<'a>(
    desc: &'a DeviceDescriptor<'a>,
    trace: wgt::Trace,
    additional_features: wgt::Features,
    instance_flags: wgt::InstanceFlags,
) -> Result<wgt::DeviceDescriptor<Label<'a>>, RequestDeviceError> {
    // Validate requested features against trusted browser policy.
    let mut allowed_features =
        wgt::FeaturesWebGPU::all().difference(wgt::FeaturesWebGPU::EXPERIMENTAL_RAY_QUERY);
    if instance_flags.contains(wgt::InstanceFlags::ALLOW_EXPERIMENTAL_RAY_QUERY) {
        allowed_features |= wgt::FeaturesWebGPU::EXPERIMENTAL_RAY_QUERY;
    }
    let disallowed_features = desc.required_features.difference(allowed_features);
    if !disallowed_features.is_empty() {
        return Err(RequestDeviceError::UnsupportedFeature(format!(
            "Requested features are disabled by browser policy: {disallowed_features:?}"
        )));
    }
    // Determine whether the device explicitly requests experimental ray queries.
    let ray_query = desc
        .required_features
        .contains(wgt::FeaturesWebGPU::EXPERIMENTAL_RAY_QUERY);
    let required_features = wgt::Features::from(desc.required_features).union(additional_features);
    Ok(wgt::DeviceDescriptor {
        label: desc.label.as_ref().map(|l| Cow::Borrowed(l.as_ref())),
        required_features,
        required_limits: desc.required_limits.clone(),
        default_queue: wgt::QueueDescriptor {
            label: desc
                .default_queue
                .label
                .as_ref()
                .map(|l| Cow::Borrowed(l.as_ref())),
        },
        experimental_features: if ray_query {
            // Browser policy above authorizes this experimental device.
            unsafe { wgt::ExperimentalFeatures::enabled() }
        } else {
            wgt::ExperimentalFeatures::disabled()
        },
        memory_hints: wgt::MemoryHints::MemoryUsage,
        trace,
    })
}

impl Global {
    pub fn adapter_request_device(
        &self,
        adapter_id: AdapterId,
        desc: &DeviceDescriptor,
        trace: wgt::Trace,
        additional_features: wgt::Features,
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
        let desc = map_device_descriptor(desc, trace, additional_features, self.instance.flags)?;
        let (device, queue) = adapter
            .request_device(&desc)
            .map_err(map_request_device_error)?;

        let device_id = devices.assign(device_id_in, device);

        let queue_id = queues.assign(queue_id_in, queue);

        Ok((device_id, queue_id))
    }

    pub fn adapter_validate_device_descriptor<'a>(
        &self,
        adapter_id: AdapterId,
        desc: &mut wgt::DeviceDescriptor<Label<'a>>,
    ) -> Result<(), RequestDeviceError> {
        let hub = self.hub.borrow();
        let adapter = hub.adapters.get(adapter_id);
        adapter
            .validate_device_descriptor(desc)
            .map_err(map_request_device_error)
    }

    /// # Safety
    ///
    /// - `hal_device` must be created from `adapter_id` or its internal handle.
    /// - `desc` must be a subset of `hal_device` features and limits.
    pub unsafe fn create_device_from_hal<'a>(
        &self,
        adapter_id: AdapterId,
        hal_device: hal::DynOpenDevice,
        desc: &wgt::DeviceDescriptor<Label<'a>>,
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
        let (device, queue) = unsafe { adapter.create_device_and_queue_from_hal(hal_device, desc) }
            .map_err(map_request_device_error)?;

        let device_id = devices.assign(device_id_in, device);

        let queue_id = queues.assign(queue_id_in, queue);

        Ok((device_id, queue_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_queries_require_browser_policy_and_device_opt_in() {
        let ray_query = wgt::FeaturesWebGPU::EXPERIMENTAL_RAY_QUERY;
        for (required_features, allowed) in [
            (wgt::FeaturesWebGPU::empty(), false),
            (ray_query, false),
            (wgt::FeaturesWebGPU::empty(), true),
            (ray_query, true),
        ] {
            let desc = DeviceDescriptor {
                required_features,
                ..Default::default()
            };
            let mut flags = wgt::InstanceFlags::STRICT_WEBGPU_COMPLIANCE;
            flags.set(wgt::InstanceFlags::ALLOW_EXPERIMENTAL_RAY_QUERY, allowed);
            let result =
                map_device_descriptor(&desc, wgt::Trace::Off, wgt::Features::empty(), flags);
            if !allowed && required_features.contains(ray_query) {
                assert!(matches!(
                    result,
                    Err(RequestDeviceError::UnsupportedFeature(_))
                ));
            } else {
                let mapped = result.unwrap();
                assert_eq!(
                    mapped.required_features,
                    wgt::Features::from(required_features)
                );
                assert_eq!(
                    mapped.experimental_features.is_enabled(),
                    required_features.contains(ray_query)
                );
            }
        }
    }

    #[test]
    fn unknown_feature_bits_are_rejected_even_with_ray_query_policy() {
        let desc = DeviceDescriptor {
            required_features: wgt::FeaturesWebGPU::EXPERIMENTAL_RAY_QUERY
                | wgt::FeaturesWebGPU::from_bits_retain(1 << 63),
            ..Default::default()
        };
        assert!(matches!(
            map_device_descriptor(
                &desc,
                wgt::Trace::Off,
                wgt::Features::empty(),
                wgt::InstanceFlags::ALLOW_EXPERIMENTAL_RAY_QUERY,
            ),
            Err(RequestDeviceError::UnsupportedFeature(_))
        ));
    }
}
