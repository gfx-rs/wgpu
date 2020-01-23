/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    backend,
    binding_model::MAX_BIND_GROUPS,
    device::{Device, BIND_BUFFER_ALIGNMENT},
    hub::{GfxBackend, Global, IdentityFilter, Token},
    id::{AdapterId, DeviceId},
    power,
    Backend,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use hal::adapter::{AdapterInfo, DeviceType};
use hal::{self, adapter::PhysicalDevice as _, queue::QueueFamily as _, Instance as _};


#[derive(Debug)]
pub struct Instance {
    #[cfg(any(
        not(any(target_os = "ios", target_os = "macos")),
        feature = "gfx-backend-vulkan"
    ))]
    pub vulkan: Option<gfx_backend_vulkan::Instance>,
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub metal: gfx_backend_metal::Instance,
    #[cfg(windows)]
    pub dx12: Option<gfx_backend_dx12::Instance>,
    #[cfg(windows)]
    pub dx11: gfx_backend_dx11::Instance,
}

impl Instance {
    pub fn new(name: &str, version: u32) -> Self {
        Instance {
            #[cfg(any(
                not(any(target_os = "ios", target_os = "macos")),
                feature = "gfx-backend-vulkan"
            ))]
            vulkan: gfx_backend_vulkan::Instance::create(name, version).ok(),
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            metal: gfx_backend_metal::Instance::create(name, version).unwrap(),
            #[cfg(windows)]
            dx12: gfx_backend_dx12::Instance::create(name, version).ok(),
            #[cfg(windows)]
            dx11: gfx_backend_dx11::Instance::create(name, version).unwrap(),
        }
    }

    pub(crate) fn destroy_surface(&mut self, surface: Surface) {
        #[cfg(any(
            not(any(target_os = "ios", target_os = "macos")),
            feature = "gfx-backend-vulkan"
        ))]
        unsafe {
            if let Some(suf) = surface.vulkan {
                self.vulkan.as_mut().unwrap().destroy_surface(suf);
            }
        }
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        unsafe {
            self.metal.destroy_surface(surface.metal);
        }
        #[cfg(windows)]
        unsafe {
            if let Some(suf) = surface.dx12 {
                self.dx12.as_mut().unwrap().destroy_surface(suf);
            }
            self.dx11.destroy_surface(surface.dx11);
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
    pub metal: GfxSurface<backend::Metal>,
    #[cfg(windows)]
    pub dx12: Option<GfxSurface<backend::Dx12>>,
    #[cfg(windows)]
    pub dx11: GfxSurface<backend::Dx11>,
}

#[derive(Debug)]
pub struct Adapter<B: hal::Backend> {
    pub(crate) raw: hal::adapter::Adapter<B>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PowerPreference {
    Default = 0,
    LowPower = 1,
    HighPerformance = 2,
}

#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RequestAdapterOptions {
    pub power_preference: PowerPreference,
}

impl Default for RequestAdapterOptions {
    fn default() -> Self {
        RequestAdapterOptions {
            power_preference: PowerPreference::Default,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Extensions {
    pub anisotropic_filtering: bool,
}

#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Limits {
    pub max_bind_groups: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_bind_groups: MAX_BIND_GROUPS as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DeviceDescriptor {
    pub extensions: Extensions,
    pub limits: Limits,
}

bitflags::bitflags! {
    #[repr(transparent)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    pub struct BackendBit: u32 {
        const VULKAN = 1 << Backend::Vulkan as u32;
        const GL = 1 << Backend::Gl as u32;
        const METAL = 1 << Backend::Metal as u32;
        const DX12 = 1 << Backend::Dx12 as u32;
        const DX11 = 1 << Backend::Dx11 as u32;
        /// Vulkan + METAL + DX12
        const PRIMARY = Self::VULKAN.bits | Self::METAL.bits | Self::DX12.bits;
        /// OpenGL + DX11
        const SECONDARY = Self::GL.bits | Self::DX11.bits;
    }
}

impl From<Backend> for BackendBit {
    fn from(backend: Backend) -> Self {
        BackendBit::from_bits(1 << backend as u32).unwrap()
    }
}

pub enum AdapterInputs<'a, I> {
    IdSet(&'a [I], fn(&I) -> Backend),
    Mask(BackendBit, fn() -> I),
}

impl<I: Clone> AdapterInputs<'_, I> {
    fn find(&self, b: Backend) -> Option<I> {
        match *self {
            AdapterInputs::IdSet(ids, ref fun) => ids.iter().find(|id| fun(id) == b).cloned(),
            AdapterInputs::Mask(bits, ref fun) => {
                if bits.contains(b.into()) {
                    Some(fun())
                } else {
                    None
                }
            }
        }
    }
}

impl<F: IdentityFilter<AdapterId>> Global<F> {
    pub fn pick_adapter(
        &self,
        desc: &RequestAdapterOptions,
        inputs: AdapterInputs<F::Input>,
    ) -> Option<AdapterId> {
        let instance = &self.instance;
        let mut device_types = Vec::new();

        let id_vulkan = inputs.find(Backend::Vulkan);
        let id_metal = inputs.find(Backend::Metal);
        let id_dx12 = inputs.find(Backend::Dx12);
        let id_dx11 = inputs.find(Backend::Dx11);

        #[cfg(any(
            not(any(target_os = "ios", target_os = "macos")),
            feature = "gfx-backend-vulkan"
        ))]
        let mut adapters_vk = match instance.vulkan {
            Some(ref inst) if id_vulkan.is_some() => {
                let adapters = inst.enumerate_adapters();
                device_types.extend(adapters.iter().map(|ad| ad.info.device_type.clone()));
                adapters
            }
            _ => Vec::new(),
        };
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        let mut adapters_mtl = if id_metal.is_some() {
            let adapters = instance.metal.enumerate_adapters();
            device_types.extend(adapters.iter().map(|ad| ad.info.device_type.clone()));
            adapters
        } else {
            Vec::new()
        };
        #[cfg(windows)]
        let mut adapters_dx12 = match instance.dx12 {
            Some(ref inst) if id_dx12.is_some() => {
                let adapters = inst.enumerate_adapters();
                device_types.extend(adapters.iter().map(|ad| ad.info.device_type.clone()));
                adapters
            }
            _ => Vec::new(),
        };
        #[cfg(windows)]
        let mut adapters_dx11 = if id_dx11.is_some() {
            let adapters = instance.dx11.enumerate_adapters();
            device_types.extend(adapters.iter().map(|ad| ad.info.device_type.clone()));
            adapters
        } else {
            Vec::new()
        };

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
            PowerPreference::Default => {
                match power::is_battery_discharging() {
                    Ok(false) => discrete.or(integrated).or(other).or(virt),
                    Ok(true) => integrated.or(discrete).or(other).or(virt),
                    Err(err) => {
                        log::debug!("Power info unavailable, preferring integrated gpu ({})", err);
                        integrated.or(discrete).or(other).or(virt)
                    }
                }
            },
            PowerPreference::LowPower => integrated.or(other).or(discrete).or(virt),
            PowerPreference::HighPerformance => discrete.or(other).or(integrated).or(virt),
        };

        let mut token = Token::root();
        let mut selected = preferred_gpu.unwrap_or(0);
        #[cfg(any(
            not(any(target_os = "ios", target_os = "macos")),
            feature = "gfx-backend-vulkan"
        ))]
        {
            if selected < adapters_vk.len() {
                let adapter = Adapter {
                    raw: adapters_vk.swap_remove(selected),
                };
                log::info!("Adapter Vulkan {:?}", adapter.raw.info);
                let id = backend::Vulkan::hub(self).adapters.register_identity(
                    id_vulkan.unwrap(),
                    adapter,
                    &mut token,
                );
                return Some(id);
            }
            selected -= adapters_vk.len();
        }
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            if selected < adapters_mtl.len() {
                let adapter = Adapter {
                    raw: adapters_mtl.swap_remove(selected),
                };
                log::info!("Adapter Metal {:?}", adapter.raw.info);
                let id = backend::Metal::hub(self).adapters.register_identity(
                    id_metal.unwrap(),
                    adapter,
                    &mut token,
                );
                return Some(id);
            }
            selected -= adapters_mtl.len();
        }
        #[cfg(windows)]
        {
            if selected < adapters_dx12.len() {
                let adapter = Adapter {
                    raw: adapters_dx12.swap_remove(selected),
                };
                log::info!("Adapter Dx12 {:?}", adapter.raw.info);
                let id = backend::Dx12::hub(self).adapters.register_identity(
                    id_dx12.unwrap(),
                    adapter,
                    &mut token,
                );
                return Some(id);
            }
            selected -= adapters_dx12.len();
            if selected < adapters_dx11.len() {
                let adapter = Adapter {
                    raw: adapters_dx11.swap_remove(selected),
                };
                log::info!("Adapter Dx11 {:?}", adapter.raw.info);
                let id = backend::Dx11::hub(self).adapters.register_identity(
                    id_dx11.unwrap(),
                    adapter,
                    &mut token,
                );
                return Some(id);
            }
            selected -= adapters_dx11.len();
        }

        let _ = (selected, id_vulkan, id_metal, id_dx12, id_dx11);
        log::warn!("Some adapters are present, but enumerating them failed!");
        None
    }

    pub fn adapter_get_info<B: GfxBackend>(&self, adapter_id: AdapterId) -> AdapterInfo {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        let adapter = &adapter_guard[adapter_id];
        adapter.raw.info.clone()
    }

    pub fn adapter_destroy<B: GfxBackend>(&self, adapter_id: AdapterId) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (_adapter, _) = hub.adapters.unregister(adapter_id, &mut token);
    }
}

impl<F: IdentityFilter<DeviceId>> Global<F> {
    pub fn adapter_request_device<B: GfxBackend>(
        &self,
        adapter_id: AdapterId,
        desc: &DeviceDescriptor,
        id_in: F::Input,
    ) -> DeviceId {
        let hub = B::hub(self);
        let mut token = Token::root();
        let device = {
            let (adapter_guard, _) = hub.adapters.read(&mut token);
            let adapter = &adapter_guard[adapter_id].raw;
            let wishful_features =
                hal::Features::VERTEX_STORES_AND_ATOMICS |
                hal::Features::FRAGMENT_STORES_AND_ATOMICS;

            let family = adapter
                .queue_families
                .iter()
                .find(|family| family.queue_type().supports_graphics())
                .unwrap();
            let mut gpu = unsafe {
                adapter
                    .physical_device
                    .open(
                        &[(family, &[1.0])],
                        adapter.physical_device.features() & wishful_features,
                    )
                    .unwrap()
            };

            let limits = adapter.physical_device.limits();
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
            if limits.max_bound_descriptor_sets == 0 {
                log::warn!("max_bind_groups limit is missing");
            } else {
                assert!(
                    u32::from(limits.max_bound_descriptor_sets) >= desc.limits.max_bind_groups,
                    "Adapter does not support the requested max_bind_groups"
                );
            }

            let mem_props = adapter.physical_device.memory_properties();

            let supports_texture_d24_s8 = adapter
                .physical_device
                .format_properties(Some(hal::format::Format::D24UnormS8Uint))
                .optimal_tiling
                .contains(hal::format::ImageFeature::DEPTH_STENCIL_ATTACHMENT);

            Device::new(
                gpu.device,
                adapter_id,
                gpu.queue_groups.swap_remove(0),
                mem_props,
                supports_texture_d24_s8,
                desc.limits.max_bind_groups,
            )
        };

        hub.devices.register_identity(id_in, device, &mut token)
    }
}
