/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    backend,
    device::Device,
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Input, Token},
    id::{AdapterId, DeviceId, SurfaceId},
    power, span, LifeGuard, PrivateFeatures, Stored, MAX_BIND_GROUPS,
};

use wgt::{Backend, BackendBit, DeviceDescriptor, PowerPreference, BIND_BUFFER_ALIGNMENT};

#[cfg(feature = "replay")]
use serde::Deserialize;
#[cfg(feature = "trace")]
use serde::Serialize;

use hal::{
    adapter::{AdapterInfo as HalAdapterInfo, DeviceType as HalDeviceType, PhysicalDevice as _},
    queue::QueueFamily as _,
    window::Surface as _,
    Instance as _,
};
use thiserror::Error;

/// Size that is guaranteed to be available in push constants.
///
/// This is needed because non-vulkan backends might not
/// provide a push-constant size limit.
const MIN_PUSH_CONSTANT_SIZE: u32 = 128;

pub type RequestAdapterOptions = wgt::RequestAdapterOptions<SurfaceId>;

#[derive(Debug)]
pub struct Instance {
    #[cfg(any(
        not(any(target_os = "ios", target_os = "macos")),
        feature = "gfx-backend-vulkan"
    ))]
    pub vulkan: Option<gfx_backend_vulkan::Instance>,
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub metal: Option<gfx_backend_metal::Instance>,
    #[cfg(windows)]
    pub dx12: Option<gfx_backend_dx12::Instance>,
    #[cfg(windows)]
    pub dx11: Option<gfx_backend_dx11::Instance>,
}

impl Instance {
    pub fn new(name: &str, version: u32, backends: BackendBit) -> Self {
        backends_map! {
            let map = |(backend, backend_create)| {
                if backends.contains(backend.into()) {
                    backend_create(name, version).ok()
                } else {
                    None
                }
            };
            Instance {
                #[vulkan]
                vulkan: map((Backend::Vulkan, gfx_backend_vulkan::Instance::create)),
                #[metal]
                metal: map((Backend::Metal, gfx_backend_metal::Instance::create)),
                #[dx12]
                dx12: map((Backend::Dx12, gfx_backend_dx12::Instance::create)),
                #[dx11]
                dx11: map((Backend::Dx11, gfx_backend_dx11::Instance::create)),
            }
        }
    }

    pub(crate) fn destroy_surface(&mut self, surface: Surface) {
        backends_map! {
            let map = |(surface_backend, self_backend)| {
                unsafe {
                    if let Some(suf) = surface_backend {
                        self_backend.as_mut().unwrap().destroy_surface(suf);
                    }
                }
            };

            #[vulkan]
            map((surface.vulkan, &mut self.vulkan)),
            #[metal]
            map((surface.metal, &mut self.metal)),
            #[dx12]
            map((surface.dx12, &mut self.dx12)),
            #[dx11]
            map((surface.dx11, &mut self.dx11)),
        }
    }
}

type GfxSurface<B> = <B as hal::Backend>::Surface;

#[derive(Debug)]
pub struct Surface {
    #[cfg(any(
        not(any(target_os = "ios", target_os = "macos")),
        feature = "gfx-backend-vulkan"
    ))]
    pub vulkan: Option<GfxSurface<backend::Vulkan>>,
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub metal: Option<GfxSurface<backend::Metal>>,
    #[cfg(windows)]
    pub dx12: Option<GfxSurface<backend::Dx12>>,
    #[cfg(windows)]
    pub dx11: Option<GfxSurface<backend::Dx11>>,
}

#[derive(Debug)]
pub struct Adapter<B: hal::Backend> {
    pub(crate) raw: hal::adapter::Adapter<B>,
    features: wgt::Features,
    limits: wgt::Limits,
    life_guard: LifeGuard,
}

impl<B: hal::Backend> Adapter<B> {
    fn new(raw: hal::adapter::Adapter<B>) -> Self {
        span!(_guard, INFO, "Adapter::new");

        let adapter_features = raw.physical_device.features();

        let mut features = wgt::Features::default()
            | wgt::Features::MAPPABLE_PRIMARY_BUFFERS
            | wgt::Features::PUSH_CONSTANTS;
        features.set(
            wgt::Features::SAMPLED_TEXTURE_BINDING_ARRAY,
            adapter_features.contains(hal::Features::TEXTURE_DESCRIPTOR_ARRAY),
        );
        features.set(
            wgt::Features::SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING,
            adapter_features.contains(hal::Features::SHADER_SAMPLED_IMAGE_ARRAY_DYNAMIC_INDEXING),
        );
        features.set(
            wgt::Features::SAMPLED_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
            adapter_features.contains(hal::Features::SAMPLED_TEXTURE_DESCRIPTOR_INDEXING),
        );
        features.set(
            wgt::Features::UNSIZED_BINDING_ARRAY,
            adapter_features.contains(hal::Features::UNSIZED_DESCRIPTOR_ARRAY),
        );
        features.set(
            wgt::Features::MULTI_DRAW_INDIRECT,
            adapter_features.contains(hal::Features::MULTI_DRAW_INDIRECT),
        );
        features.set(
            wgt::Features::MULTI_DRAW_INDIRECT_COUNT,
            adapter_features.contains(hal::Features::DRAW_INDIRECT_COUNT),
        );

        let adapter_limits = raw.physical_device.limits();

        let default_limits = wgt::Limits::default();

        // All these casts to u32 are safe as the underlying vulkan types are u32s.
        // If another backend provides larger limits than u32, we need to clamp them to u32::MAX.
        // TODO: fix all gfx-hal backends to produce limits we care about, and remove .max
        let limits = wgt::Limits {
            max_bind_groups: (adapter_limits.max_bound_descriptor_sets as u32)
                .min(MAX_BIND_GROUPS as u32)
                .max(default_limits.max_bind_groups),
            max_dynamic_uniform_buffers_per_pipeline_layout: (adapter_limits
                .max_descriptor_set_uniform_buffers_dynamic
                as u32)
                .max(default_limits.max_dynamic_uniform_buffers_per_pipeline_layout),
            max_dynamic_storage_buffers_per_pipeline_layout: (adapter_limits
                .max_descriptor_set_storage_buffers_dynamic
                as u32)
                .max(default_limits.max_dynamic_storage_buffers_per_pipeline_layout),
            max_sampled_textures_per_shader_stage: (adapter_limits
                .max_per_stage_descriptor_sampled_images
                as u32)
                .max(default_limits.max_sampled_textures_per_shader_stage),
            max_samplers_per_shader_stage: (adapter_limits.max_per_stage_descriptor_samplers
                as u32)
                .max(default_limits.max_samplers_per_shader_stage),
            max_storage_buffers_per_shader_stage: (adapter_limits
                .max_per_stage_descriptor_storage_buffers
                as u32)
                .max(default_limits.max_storage_buffers_per_shader_stage),
            max_storage_textures_per_shader_stage: (adapter_limits
                .max_per_stage_descriptor_storage_images
                as u32)
                .max(default_limits.max_storage_textures_per_shader_stage),
            max_uniform_buffers_per_shader_stage: (adapter_limits
                .max_per_stage_descriptor_uniform_buffers
                as u32)
                .max(default_limits.max_uniform_buffers_per_shader_stage),
            max_uniform_buffer_binding_size: (adapter_limits.max_uniform_buffer_range as u32)
                .max(default_limits.max_uniform_buffer_binding_size),
            max_push_constant_size: (adapter_limits.max_push_constants_size as u32)
                .max(MIN_PUSH_CONSTANT_SIZE), // As an extension, the default is always 0, so define a separate minimum.
        };

        Adapter {
            raw,
            features,
            limits,
            life_guard: LifeGuard::new(),
        }
    }
}

/// Metadata about a backend adapter.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct AdapterInfo {
    /// Adapter name
    pub name: String,
    /// Vendor PCI id of the adapter
    pub vendor: usize,
    /// PCI id of the adapter
    pub device: usize,
    /// Type of device
    pub device_type: DeviceType,
    /// Backend used for device
    pub backend: Backend,
}

impl AdapterInfo {
    fn from_gfx(adapter_info: HalAdapterInfo, backend: Backend) -> Self {
        let HalAdapterInfo {
            name,
            vendor,
            device,
            device_type,
        } = adapter_info;

        AdapterInfo {
            name,
            vendor,
            device,
            device_type: device_type.into(),
            backend,
        }
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
/// Error when requesting a device from the adaptor
pub enum RequestDeviceError {
    #[error("unsupported features were requested: {0:?}")]
    UnsupportedFeature(wgt::Features),
    #[error("device has no queue supporting graphics")]
    NoGraphicsQueue,
    #[error(transparent)]
    DeviceCreationError(#[from] hal::device::CreationError),
    #[error("some of the requested device limits are not supported")]
    LimitsExceeded,
}

/// Supported physical device types.
#[repr(u8)]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "trace", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum DeviceType {
    /// Other.
    Other,
    /// Integrated GPU with shared CPU/GPU memory.
    IntegratedGpu,
    /// Discrete GPU with separate CPU/GPU memory.
    DiscreteGpu,
    /// Virtual / Hosted.
    VirtualGpu,
    /// Cpu / Software Rendering.
    Cpu,
}

impl From<HalDeviceType> for DeviceType {
    fn from(device_type: HalDeviceType) -> Self {
        match device_type {
            HalDeviceType::Other => Self::Other,
            HalDeviceType::IntegratedGpu => Self::IntegratedGpu,
            HalDeviceType::DiscreteGpu => Self::DiscreteGpu,
            HalDeviceType::VirtualGpu => Self::VirtualGpu,
            HalDeviceType::Cpu => Self::Cpu,
        }
    }
}

pub enum AdapterInputs<'a, I> {
    IdSet(&'a [I], fn(&I) -> Backend),
    Mask(BackendBit, fn(Backend) -> I),
}

impl<I: Clone> AdapterInputs<'_, I> {
    fn find(&self, b: Backend) -> Option<I> {
        match *self {
            AdapterInputs::IdSet(ids, ref fun) => ids.iter().find(|id| fun(id) == b).cloned(),
            AdapterInputs::Mask(bits, ref fun) => {
                if bits.contains(b.into()) {
                    Some(fun(b))
                } else {
                    None
                }
            }
        }
    }
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    #[cfg(feature = "raw-window-handle")]
    pub fn instance_create_surface(
        &self,
        handle: &impl raw_window_handle::HasRawWindowHandle,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        span!(_guard, INFO, "Instance::create_surface");

        let surface = unsafe {
            backends_map! {
                let map = |inst| {
                    inst
                    .as_ref()
                    .and_then(|inst| inst.create_surface(handle).ok())
                };

                Surface {
                    #[vulkan]
                    vulkan: map(&self.instance.vulkan),
                    #[metal]
                    metal: map(&self.instance.metal),
                    #[dx12]
                    dx12: map(&self.instance.dx12),
                    #[dx11]
                    dx11: map(&self.instance.dx11),
                }
            }
        };

        let mut token = Token::root();
        self.surfaces.register_identity(id_in, surface, &mut token)
    }

    pub fn enumerate_adapters(&self, inputs: AdapterInputs<Input<G, AdapterId>>) -> Vec<AdapterId> {
        span!(_guard, INFO, "Instance::enumerate_adapters");

        let instance = &self.instance;
        let mut token = Token::root();
        let mut adapters = Vec::new();

        backends_map! {
            let map = |(instance_field, backend, backend_info, backend_hub)| {
                if let Some(inst) = instance_field {
                    if let Some(id_backend) = inputs.find(backend) {
                        for raw in inst.enumerate_adapters() {
                            let adapter = Adapter::new(raw);
                            log::info!("Adapter {} {:?}", backend_info, adapter.raw.info);
                            adapters.push(backend_hub(self).adapters.register_identity(
                                id_backend.clone(),
                                adapter,
                                &mut token,
                            ));
                        }
                    }
                }
            };

            #[vulkan]
            map((&instance.vulkan, Backend::Vulkan, "Vulkan", backend::Vulkan::hub)),
            #[metal]
            map((&instance.metal, Backend::Metal, "Metal", backend::Metal::hub)),
            #[dx12]
            map((&instance.dx12, Backend::Dx12, "Dx12", backend::Dx12::hub)),
            #[dx11]
            map((&instance.dx11, Backend::Dx11, "Dx11", backend::Dx11::hub)),
        }

        adapters
    }

    pub fn pick_adapter(
        &self,
        desc: &RequestAdapterOptions,
        inputs: AdapterInputs<Input<G, AdapterId>>,
    ) -> Option<AdapterId> {
        span!(_guard, INFO, "Instance::pick_adapter");

        let instance = &self.instance;
        let mut token = Token::root();
        let (surface_guard, mut token) = self.surfaces.read(&mut token);
        let compatible_surface = desc.compatible_surface.map(|id| &surface_guard[id]);
        let mut device_types = Vec::new();

        let mut id_vulkan = inputs.find(Backend::Vulkan);
        let mut id_metal = inputs.find(Backend::Metal);
        let mut id_dx12 = inputs.find(Backend::Dx12);
        let mut id_dx11 = inputs.find(Backend::Dx11);

        backends_map! {
            let map = |(instance_backend, id_backend, surface_backend)| {
                match instance_backend {
                    Some(ref inst) if id_backend.is_some() => {
                        let mut adapters = inst.enumerate_adapters();
                        if let Some(surface_backend) = compatible_surface.and_then(surface_backend) {
                            adapters.retain(|a| {
                                a.queue_families
                                    .iter()
                                    .find(|qf| qf.queue_type().supports_graphics())
                                    .map_or(false, |qf| surface_backend.supports_queue_family(qf))
                            });
                        }
                        device_types.extend(adapters.iter().map(|ad| ad.info.device_type.clone()));
                        adapters
                    }
                    _ => Vec::new(),
                }
            };

            // NB: The internal function definitions are a workaround for Rust
            // being weird with lifetimes for closure literals...
            #[vulkan]
            let adapters_vk = map((&instance.vulkan, &id_vulkan, {
                fn surface_vulkan(surf: &Surface) -> Option<&GfxSurface<backend::Vulkan>> {
                    surf.vulkan.as_ref()
                }
                surface_vulkan
            }));
            #[metal]
            let adapters_mtl = map((&instance.metal, &id_metal, {
                fn surface_metal(surf: &Surface) -> Option<&GfxSurface<backend::Metal>> {
                    surf.metal.as_ref()
                }
                surface_metal
            }));
            #[dx12]
            let adapters_dx12 = map((&instance.dx12, &id_dx12, {
                fn surface_dx12(surf: &Surface) -> Option<&GfxSurface<backend::Dx12>> {
                    surf.dx12.as_ref()
                }
                surface_dx12
            }));
            #[dx11]
            let adapters_dx11 = map((&instance.dx11, &id_dx11, {
                fn surface_dx11(surf: &Surface) -> Option<&GfxSurface<backend::Dx11>> {
                    surf.dx11.as_ref()
                }
                surface_dx11
            }));
        }

        if device_types.is_empty() {
            log::warn!("No adapters are available!");
            return None;
        }

        let (mut integrated, mut discrete, mut virt, mut other) = (None, None, None, None);

        for (i, ty) in device_types.into_iter().enumerate() {
            match ty {
                hal::adapter::DeviceType::IntegratedGpu => {
                    integrated = integrated.or(Some(i));
                }
                hal::adapter::DeviceType::DiscreteGpu => {
                    discrete = discrete.or(Some(i));
                }
                hal::adapter::DeviceType::VirtualGpu => {
                    virt = virt.or(Some(i));
                }
                _ => {
                    other = other.or(Some(i));
                }
            }
        }

        let preferred_gpu = match desc.power_preference {
            PowerPreference::Default => match power::is_battery_discharging() {
                Ok(false) => discrete.or(integrated).or(other).or(virt),
                Ok(true) => integrated.or(discrete).or(other).or(virt),
                Err(err) => {
                    log::debug!(
                        "Power info unavailable, preferring integrated gpu ({})",
                        err
                    );
                    integrated.or(discrete).or(other).or(virt)
                }
            },
            PowerPreference::LowPower => integrated.or(other).or(discrete).or(virt),
            PowerPreference::HighPerformance => discrete.or(other).or(integrated).or(virt),
        };

        let mut selected = preferred_gpu.unwrap_or(0);

        backends_map! {
            let map = |(info_adapter, id_backend, mut adapters_backend, backend_hub)| {
                if selected < adapters_backend.len() {
                    let adapter = Adapter::new(adapters_backend.swap_remove(selected));
                    log::info!("Adapter {} {:?}", info_adapter, adapter.raw.info);
                    let id = backend_hub(self).adapters.register_identity(
                        id_backend.take().unwrap(),
                        adapter,
                        &mut token,
                    );
                    return Some(id);
                }
                selected -= adapters_backend.len();
            };

            #[vulkan]
            map(("Vulkan", &mut id_vulkan, adapters_vk, backend::Vulkan::hub)),
            #[metal]
            map(("Metal", &mut id_metal, adapters_mtl, backend::Metal::hub)),
            #[dx12]
            map(("Dx12", &mut id_dx12, adapters_dx12, backend::Dx12::hub)),
            #[dx11]
            map(("Dx11", &mut id_dx11, adapters_dx11, backend::Dx11::hub)),
        }

        let _ = (
            selected,
            id_vulkan.take(),
            id_metal.take(),
            id_dx12.take(),
            id_dx11.take(),
        );
        log::warn!("Some adapters are present, but enumerating them failed!");
        None
    }

    pub fn adapter_get_info<B: GfxBackend>(&self, adapter_id: AdapterId) -> AdapterInfo {
        span!(_guard, INFO, "Adapter::get_info");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        let adapter = &adapter_guard[adapter_id];
        AdapterInfo::from_gfx(adapter.raw.info.clone(), adapter_id.backend())
    }

    pub fn adapter_features<B: GfxBackend>(&self, adapter_id: AdapterId) -> wgt::Features {
        span!(_guard, INFO, "Adapter::features");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        let adapter = &adapter_guard[adapter_id];

        adapter.features
    }

    pub fn adapter_limits<B: GfxBackend>(&self, adapter_id: AdapterId) -> wgt::Limits {
        span!(_guard, INFO, "Adapter::limits");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        let adapter = &adapter_guard[adapter_id];

        adapter.limits.clone()
    }

    pub fn adapter_destroy<B: GfxBackend>(&self, adapter_id: AdapterId) {
        span!(_guard, INFO, "Adapter::drop");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (mut guard, _) = hub.adapters.write(&mut token);

        if guard[adapter_id]
            .life_guard
            .ref_count
            .take()
            .unwrap()
            .load()
            == 1
        {
            hub.adapters.free_id(adapter_id);
            let _adapter = guard.remove(adapter_id).unwrap();
        }
    }
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn adapter_request_device<B: GfxBackend>(
        &self,
        adapter_id: AdapterId,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
        id_in: Input<G, DeviceId>,
    ) -> Result<DeviceId, RequestDeviceError> {
        span!(_guard, INFO, "Adapter::request_device");

        let hub = B::hub(self);
        let mut token = Token::root();
        let device = {
            let (adapter_guard, _) = hub.adapters.read(&mut token);
            let adapter = &adapter_guard[adapter_id];
            let phd = &adapter.raw.physical_device;

            // Verify all features were exposed by the adapter
            if !adapter.features.contains(desc.features) {
                return Err(RequestDeviceError::UnsupportedFeature(
                    desc.features - adapter.features,
                ));
            }

            // Verify feature preconditions
            if desc
                .features
                .contains(wgt::Features::MAPPABLE_PRIMARY_BUFFERS)
                && adapter.raw.info.device_type == hal::adapter::DeviceType::DiscreteGpu
            {
                log::warn!("Feature MAPPABLE_PRIMARY_BUFFERS enabled on a discrete gpu. This is a massive performance footgun and likely not what you wanted");
            }

            let available_features = adapter.raw.physical_device.features();

            // Check features that are always needed
            let wishful_features = hal::Features::VERTEX_STORES_AND_ATOMICS
                | hal::Features::FRAGMENT_STORES_AND_ATOMICS
                | hal::Features::NDC_Y_UP
                | hal::Features::INDEPENDENT_BLENDING
                | hal::Features::SAMPLER_ANISOTROPY;
            let mut enabled_features = available_features & wishful_features;
            if enabled_features != wishful_features {
                log::warn!(
                    "Missing features: {:?}",
                    wishful_features - enabled_features
                );
            }

            // Features
            enabled_features.set(
                hal::Features::TEXTURE_DESCRIPTOR_ARRAY,
                adapter
                    .features
                    .contains(wgt::Features::SAMPLED_TEXTURE_BINDING_ARRAY),
            );
            enabled_features.set(
                hal::Features::SHADER_SAMPLED_IMAGE_ARRAY_DYNAMIC_INDEXING,
                adapter
                    .features
                    .contains(wgt::Features::SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING),
            );
            enabled_features.set(
                hal::Features::SHADER_SAMPLED_IMAGE_ARRAY_DYNAMIC_INDEXING,
                adapter
                    .features
                    .contains(wgt::Features::SAMPLED_TEXTURE_ARRAY_DYNAMIC_INDEXING),
            );
            enabled_features.set(
                hal::Features::SAMPLED_TEXTURE_DESCRIPTOR_INDEXING,
                adapter
                    .features
                    .contains(wgt::Features::SAMPLED_TEXTURE_ARRAY_NON_UNIFORM_INDEXING),
            );
            enabled_features.set(
                hal::Features::UNSIZED_DESCRIPTOR_ARRAY,
                adapter
                    .features
                    .contains(wgt::Features::UNSIZED_BINDING_ARRAY),
            );
            enabled_features.set(
                hal::Features::MULTI_DRAW_INDIRECT,
                adapter
                    .features
                    .contains(wgt::Features::MULTI_DRAW_INDIRECT),
            );
            enabled_features.set(
                hal::Features::DRAW_INDIRECT_COUNT,
                adapter
                    .features
                    .contains(wgt::Features::MULTI_DRAW_INDIRECT_COUNT),
            );

            let family = adapter
                .raw
                .queue_families
                .iter()
                .find(|family| family.queue_type().supports_graphics())
                .ok_or(RequestDeviceError::NoGraphicsQueue)?;
            let mut gpu = unsafe { phd.open(&[(family, &[1.0])], enabled_features)? };

            let limits = phd.limits();
            assert_eq!(
                0,
                BIND_BUFFER_ALIGNMENT % limits.min_storage_buffer_offset_alignment,
                "Adapter storage buffer offset alignment not compatible with WGPU"
            );
            assert_eq!(
                0,
                BIND_BUFFER_ALIGNMENT % limits.min_uniform_buffer_offset_alignment,
                "Adapter uniform buffer offset alignment not compatible with WGPU"
            );
            if adapter.limits < desc.limits {
                return Err(RequestDeviceError::LimitsExceeded);
            }

            let mem_props = phd.memory_properties();
            if !desc.shader_validation {
                log::warn!("Shader validation is disabled");
            }
            let private_features = PrivateFeatures {
                shader_validation: desc.shader_validation,
                anisotropic_filtering: enabled_features.contains(hal::Features::SAMPLER_ANISOTROPY),
                texture_d24_s8: phd
                    .format_properties(Some(hal::format::Format::D24UnormS8Uint))
                    .optimal_tiling
                    .contains(hal::format::ImageFeature::DEPTH_STENCIL_ATTACHMENT),
            };

            Device::new(
                gpu.device,
                Stored {
                    value: adapter_id,
                    ref_count: adapter.life_guard.add_ref(),
                },
                gpu.queue_groups.swap_remove(0),
                mem_props,
                limits,
                private_features,
                desc,
                trace_path,
            )
        };

        Ok(hub.devices.register_identity(id_in, device, &mut token))
    }
}
