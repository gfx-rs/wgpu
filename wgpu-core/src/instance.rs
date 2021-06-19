use crate::{
    device::{Device, DeviceDescriptor},
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Input, Token},
    id::{AdapterId, DeviceId, SurfaceId, Valid},
    LabelHelpers, LifeGuard, Stored, DOWNLEVEL_WARNING_MESSAGE,
};

use wgt::{Backend, BackendBit, PowerPreference, BIND_BUFFER_ALIGNMENT};

use hal::{Adapter as _, Instance as _};
use thiserror::Error;

pub type RequestAdapterOptions = wgt::RequestAdapterOptions<SurfaceId>;
type HalInstance<A> = <A as hal::Api>::Instance;
type HalSurface<A> = <A as hal::Api>::Surface;

pub struct Instance {
    #[allow(dead_code)]
    name: String,
    #[cfg(vulkan)]
    pub vulkan: Option<HalInstance<hal::api::Vulkan>>,
    #[cfg(metal)]
    pub metal: Option<HalInstance<hal::api::Metal>>,
    #[cfg(dx12)]
    pub dx12: Option<HalInstance<hal::api::Dx12>>,
    #[cfg(dx11)]
    pub dx11: Option<HalInstance<hal::api::Dx11>>,
    #[cfg(gl)]
    pub gl: Option<HalInstance<hal::api::Gles>>,
}

impl Instance {
    pub fn new(name: &str, backends: BackendBit) -> Self {
        let mut flags = hal::InstanceFlag::empty();
        if cfg!(debug_assertions) {
            flags |= hal::InstanceFlag::VALIDATION;
            flags |= hal::InstanceFlag::DEBUG;
        }
        let hal_desc = hal::InstanceDescriptor {
            name: "wgpu",
            flags,
        };

        let map = |backend: Backend| unsafe {
            if backends.contains(backend.into()) {
                hal::Instance::init(&hal_desc).ok()
            } else {
                None
            }
        };
        Self {
            name: name.to_string(),
            #[cfg(vulkan)]
            vulkan: map(Backend::Vulkan),
            #[cfg(metal)]
            metal: map(Backend::Metal),
            #[cfg(dx12)]
            dx12: map(Backend::Dx12),
            #[cfg(dx11)]
            dx11: map(Backend::Dx11),
            #[cfg(gl)]
            gl: map(Backend::Gl),
        }
    }

    pub(crate) fn destroy_surface(&self, surface: Surface) {
        backends_map! {
            let map = |(surface_backend, self_backend)| {
                unsafe {
                    if let Some(suf) = surface_backend {
                        self_backend.as_ref().unwrap().destroy_surface(suf);
                    }
                }
            };

            #[cfg(vulkan)]
            map((surface.vulkan, &self.vulkan)),
            #[cfg(metal)]
            map((surface.metal, &self.metal)),
            #[cfg(dx12)]
            map((surface.dx12, &self.dx12)),
            #[cfg(dx11)]
            map((surface.dx11, &self.dx11)),
            #[cfg(gl)]
            map((surface.gl, &self.gl)),
        }
    }
}

pub struct Surface {
    #[cfg(vulkan)]
    pub vulkan: Option<HalSurface<hal::api::Vulkan>>,
    #[cfg(metal)]
    pub metal: Option<HalSurface<hal::api::Metal>>,
    #[cfg(dx12)]
    pub dx12: Option<HalSurface<hal::api::Dx12>>,
    #[cfg(dx11)]
    pub dx11: Option<HalSurface<hal::api::Dx11>>,
    #[cfg(gl)]
    pub gl: Option<HalSurface<hal::api::Gles>>,
}

impl crate::hub::Resource for Surface {
    const TYPE: &'static str = "Surface";

    fn life_guard(&self) -> &LifeGuard {
        unreachable!()
    }

    fn label(&self) -> &str {
        "<Surface>"
    }
}

pub struct Adapter<A: hal::Api> {
    pub(crate) raw: hal::ExposedAdapter<A>,
    life_guard: LifeGuard,
}

impl<A: HalApi> Adapter<A> {
    fn new(raw: hal::ExposedAdapter<A>) -> Self {
        Self {
            raw,
            life_guard: LifeGuard::new("<Adapter>"),
        }
    }

    pub fn get_swap_chain_preferred_format(
        &self,
        surface: &mut Surface,
    ) -> Result<wgt::TextureFormat, GetSwapChainPreferredFormatError> {
        // Check the four formats mentioned in the WebGPU spec.
        // Also, prefer sRGB over linear as it is better in
        // representing perceived colors.
        let preferred_formats = [
            wgt::TextureFormat::Bgra8UnormSrgb,
            wgt::TextureFormat::Rgba8UnormSrgb,
            wgt::TextureFormat::Bgra8Unorm,
            wgt::TextureFormat::Rgba8Unorm,
        ];

        let caps = unsafe {
            self.raw
                .adapter
                .surface_capabilities(A::get_surface_mut(surface))
                .ok_or(GetSwapChainPreferredFormatError::UnsupportedQueueFamily)?
        };

        preferred_formats
            .iter()
            .cloned()
            .find(|preferred| caps.formats.contains(preferred))
            .ok_or(GetSwapChainPreferredFormatError::NotFound)
    }

    pub(crate) fn get_texture_format_features(
        &self,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        let caps = unsafe { self.raw.adapter.texture_format_capabilities(format) };

        let mut allowed_usages = format.describe().guaranteed_format_features.allowed_usages;
        allowed_usages.set(
            wgt::TextureUsage::SAMPLED,
            caps.contains(hal::TextureFormatCapability::SAMPLED),
        );
        allowed_usages.set(
            wgt::TextureUsage::STORAGE,
            caps.contains(hal::TextureFormatCapability::STORAGE),
        );
        allowed_usages.set(
            wgt::TextureUsage::RENDER_ATTACHMENT,
            caps.intersects(
                hal::TextureFormatCapability::COLOR_ATTACHMENT
                    | hal::TextureFormatCapability::DEPTH_STENCIL_ATTACHMENT,
            ),
        );

        let mut flags = wgt::TextureFormatFeatureFlags::empty();
        flags.set(
            wgt::TextureFormatFeatureFlags::STORAGE_ATOMICS,
            caps.contains(hal::TextureFormatCapability::STORAGE_ATOMIC),
        );
        flags.set(
            wgt::TextureFormatFeatureFlags::STORAGE_READ_WRITE,
            caps.contains(hal::TextureFormatCapability::STORAGE_READ_WRITE),
        );

        let filterable = caps.contains(hal::TextureFormatCapability::SAMPLED_LINEAR);

        wgt::TextureFormatFeatures {
            allowed_usages,
            flags,
            filterable,
        }
    }

    fn create_device(
        &self,
        self_id: AdapterId,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
    ) -> Result<Device<A>, RequestDeviceError> {
        // Verify all features were exposed by the adapter
        if !self.raw.features.contains(desc.features) {
            return Err(RequestDeviceError::UnsupportedFeature(
                desc.features - self.raw.features,
            ));
        }

        let caps = &self.raw.capabilities;
        if !caps.downlevel.is_webgpu_compliant() {
            log::warn!("{}", DOWNLEVEL_WARNING_MESSAGE);
        }

        // Verify feature preconditions
        if desc
            .features
            .contains(wgt::Features::MAPPABLE_PRIMARY_BUFFERS)
            && self.raw.info.device_type == wgt::DeviceType::DiscreteGpu
        {
            log::warn!("Feature MAPPABLE_PRIMARY_BUFFERS enabled on a discrete gpu. This is a massive performance footgun and likely not what you wanted");
        }

        let gpu = unsafe { self.raw.adapter.open(desc.features) }.map_err(|err| match err {
            hal::DeviceError::Lost => RequestDeviceError::DeviceLost,
            hal::DeviceError::OutOfMemory => RequestDeviceError::OutOfMemory,
        })?;

        if let Some(_) = desc.label {
            //TODO
        }

        assert_eq!(
            0,
            BIND_BUFFER_ALIGNMENT % caps.alignments.storage_buffer_offset,
            "Adapter storage buffer offset alignment not compatible with WGPU"
        );
        assert_eq!(
            0,
            BIND_BUFFER_ALIGNMENT % caps.alignments.uniform_buffer_offset,
            "Adapter uniform buffer offset alignment not compatible with WGPU"
        );
        if caps.limits < desc.limits {
            return Err(RequestDeviceError::LimitsExceeded);
        }

        Device::new(
            gpu,
            Stored {
                value: Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            caps.alignments.clone(),
            caps.downlevel,
            desc,
            trace_path,
        )
        .or(Err(RequestDeviceError::OutOfMemory))
    }
}

impl<A: hal::Api> crate::hub::Resource for Adapter<A> {
    const TYPE: &'static str = "Adapter";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

#[derive(Clone, Debug, Error)]
pub enum GetSwapChainPreferredFormatError {
    #[error("no suitable format found")]
    NotFound,
    #[error("invalid adapter")]
    InvalidAdapter,
    #[error("invalid surface")]
    InvalidSurface,
    #[error("surface does not support the adapter's queue family")]
    UnsupportedQueueFamily,
}

#[derive(Clone, Debug, Error)]
/// Error when requesting a device from the adaptor
pub enum RequestDeviceError {
    #[error("parent adapter is invalid")]
    InvalidAdapter,
    #[error("connection to device was lost during initialization")]
    DeviceLost,
    #[error("device initialization failed due to implementation specific errors")]
    Internal,
    #[error("some of the requested device limits are not supported")]
    LimitsExceeded,
    #[error("device has no queue supporting graphics")]
    NoGraphicsQueue,
    #[error("not enough memory left")]
    OutOfMemory,
    #[error("unsupported features were requested: {0:?}")]
    UnsupportedFeature(wgt::Features),
}

pub enum AdapterInputs<'a, I> {
    IdSet(&'a [I], fn(&I) -> Backend),
    Mask(BackendBit, fn(Backend) -> I),
}

impl<I: Clone> AdapterInputs<'_, I> {
    fn find(&self, b: Backend) -> Option<I> {
        match *self {
            Self::IdSet(ids, ref fun) => ids.iter().find(|id| fun(id) == b).cloned(),
            Self::Mask(bits, ref fun) => {
                if bits.contains(b.into()) {
                    Some(fun(b))
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Clone, Debug, Error)]
#[error("adapter is invalid")]
pub struct InvalidAdapter;

#[derive(Clone, Debug, Error)]
pub enum RequestAdapterError {
    #[error("no suitable adapter found")]
    NotFound,
    #[error("surface {0:?} is invalid")]
    InvalidSurface(SurfaceId),
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    #[cfg(feature = "raw-window-handle")]
    pub fn instance_create_surface(
        &self,
        handle: &impl raw_window_handle::HasRawWindowHandle,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("create_surface", "Instance");

        let surface = unsafe {
            backends_map! {
                let map = |inst| {
                    inst
                    .as_ref()
                    .and_then(|inst| inst.create_surface(handle).map_err(|e| {
                        log::warn!("Error: {:?}", e);
                    }).ok())
                };

                Surface {
                    #[cfg(vulkan)]
                    vulkan: map(&self.instance.vulkan),
                    #[cfg(metal)]
                    metal: map(&self.instance.metal),
                    #[cfg(dx12)]
                    dx12: map(&self.instance.dx12),
                    #[cfg(dx11)]
                    dx11: map(&self.instance.dx11),
                    #[cfg(gl)]
                    gl: map(&self.instance.gl),
                }
            }
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }

    /*TODO: raw CALayer handling
    #[cfg(metal)]
    pub fn instance_create_surface_metal(
        &self,
        layer: *mut std::ffi::c_void,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("create_surface_metal", "Instance");

        let surface = Surface {
            metal: self.instance.metal.as_ref().map(|inst| {
                // we don't want to link to metal-rs for this
                #[allow(clippy::transmute_ptr_to_ref)]
                inst.create_surface_from_layer(unsafe { std::mem::transmute(layer) })
            }),
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }*/

    pub fn surface_drop(&self, id: SurfaceId) {
        profiling::scope!("drop", "Surface");
        let mut token = Token::root();
        let (surface, _) = self.surfaces.unregister(id, &mut token);
        self.instance.destroy_surface(surface.unwrap());
    }

    pub fn enumerate_adapters(&self, inputs: AdapterInputs<Input<G, AdapterId>>) -> Vec<AdapterId> {
        profiling::scope!("enumerate_adapters", "Instance");

        let instance = &self.instance;
        let mut token = Token::root();
        let mut adapters = Vec::new();

        backends_map! {
            let map = |(instance_field, backend, backend_info)| {
                if let Some(ref inst) = *instance_field {
                    let hub = HalApi::hub(self);
                    if let Some(id_backend) = inputs.find(backend) {
                        for raw in unsafe {inst.enumerate_adapters()} {
                            let adapter = Adapter::new(raw);
                            log::info!("Adapter {} {:?}", backend_info, adapter.raw.info);
                            let id = hub.adapters
                                .prepare(id_backend.clone())
                                .assign(adapter, &mut token);
                            adapters.push(id.0);
                        }
                    }
                }
            };

            #[cfg(vulkan)]
            map((&instance.vulkan, Backend::Vulkan, "Vulkan")),
            #[cfg(metal)]
            map((&instance.metal, Backend::Metal, "Metal")),
            #[cfg(dx12)]
            map((&instance.dx12, Backend::Dx12, "Dx12")),
            #[cfg(dx11)]
            map((&instance.dx11, Backend::Dx11, "Dx11")),
            #[cfg(gl)]
            map((&instance.gl, Backend::Gl, "GL")),
        }

        adapters
    }

    pub fn request_adapter(
        &self,
        desc: &RequestAdapterOptions,
        inputs: AdapterInputs<Input<G, AdapterId>>,
    ) -> Result<AdapterId, RequestAdapterError> {
        profiling::scope!("pick_adapter", "Instance");

        let instance = &self.instance;
        let mut token = Token::root();
        let (surface_guard, mut token) = self.surfaces.read(&mut token);
        let compatible_surface = desc
            .compatible_surface
            .map(|id| {
                surface_guard
                    .get(id)
                    .map_err(|_| RequestAdapterError::InvalidSurface(id))
            })
            .transpose()?;
        let mut device_types = Vec::new();

        let mut id_vulkan = inputs.find(Backend::Vulkan);
        let mut id_metal = inputs.find(Backend::Metal);
        let mut id_dx12 = inputs.find(Backend::Dx12);
        let mut id_dx11 = inputs.find(Backend::Dx11);
        let mut id_gl = inputs.find(Backend::Gl);

        backends_map! {
            let map = |(instance_backend, id_backend, surface_backend)| {
                match *instance_backend {
                    Some(ref inst) if id_backend.is_some() => {
                        let mut adapters = unsafe { inst.enumerate_adapters() };
                        if let Some(surface_backend) = compatible_surface.and_then(surface_backend) {
                            adapters.retain(|exposed| unsafe {
                                exposed.adapter.surface_capabilities(surface_backend).is_some()
                            });
                        }
                        device_types.extend(adapters.iter().map(|ad| ad.info.device_type));
                        adapters
                    }
                    _ => Vec::new(),
                }
            };

            // NB: The internal function definitions are a workaround for Rust
            // being weird with lifetimes for closure literals...
            #[cfg(vulkan)]
            let adapters_vk = map((&instance.vulkan, &id_vulkan, {
                fn surface_vulkan(surf: &Surface) -> Option<&HalSurface<hal::api::Vulkan>> {
                    surf.vulkan.as_ref()
                }
                surface_vulkan
            }));
            #[cfg(metal)]
            let adapters_mtl = map((&instance.metal, &id_metal, {
                fn surface_metal(surf: &Surface) -> Option<&HalSurface<hal::api::Metal>> {
                    surf.metal.as_ref()
                }
                surface_metal
            }));
            #[cfg(dx12)]
            let adapters_dx12 = map((&instance.dx12, &id_dx12, {
                fn surface_dx12(surf: &Surface) -> Option<&HalSurface<hal::api::Dx12>> {
                    surf.dx12.as_ref()
                }
                surface_dx12
            }));
            #[cfg(dx11)]
            let adapters_dx11 = map((&instance.dx11, &id_dx11, {
                fn surface_dx11(surf: &Surface) -> Option<&HalSurface<hal::api::Dx11>> {
                    surf.dx11.as_ref()
                }
                surface_dx11
            }));
            #[cfg(gl)]
            let adapters_gl = map((&instance.gl, &id_gl, {
                fn surface_gl(surf: &Surface) -> Option<&HalSurface<hal::api::Gles>> {
                    surf.gl.as_ref()
                }
                surface_gl
            }));
        }

        if device_types.is_empty() {
            return Err(RequestAdapterError::NotFound);
        }

        let (mut integrated, mut discrete, mut virt, mut cpu, mut other) =
            (None, None, None, None, None);

        for (i, ty) in device_types.into_iter().enumerate() {
            match ty {
                wgt::DeviceType::IntegratedGpu => {
                    integrated = integrated.or(Some(i));
                }
                wgt::DeviceType::DiscreteGpu => {
                    discrete = discrete.or(Some(i));
                }
                wgt::DeviceType::VirtualGpu => {
                    virt = virt.or(Some(i));
                }
                wgt::DeviceType::Cpu => {
                    cpu = cpu.or(Some(i));
                }
                wgt::DeviceType::Other => {
                    other = other.or(Some(i));
                }
            }
        }

        let preferred_gpu = match desc.power_preference {
            PowerPreference::LowPower => integrated.or(other).or(discrete).or(virt).or(cpu),
            PowerPreference::HighPerformance => discrete.or(other).or(integrated).or(virt).or(cpu),
        };

        let mut selected = preferred_gpu.unwrap_or(0);

        backends_map! {
            let map = |(info_adapter, id_backend, mut adapters_backend)| {
                if selected < adapters_backend.len() {
                    let adapter = Adapter::new(adapters_backend.swap_remove(selected));
                    log::info!("Adapter {} {:?}", info_adapter, adapter.raw.info);
                    let id = HalApi::hub(self).adapters
                        .prepare(id_backend.take().unwrap())
                        .assign(adapter, &mut token);
                    return Ok(id.0);
                }
                selected -= adapters_backend.len();
            };

            #[cfg(vulkan)]
            map(("Vulkan", &mut id_vulkan, adapters_vk)),
            #[cfg(metal)]
            map(("Metal", &mut id_metal, adapters_mtl)),
            #[cfg(dx12)]
            map(("Dx12", &mut id_dx12, adapters_dx12)),
            #[cfg(dx11)]
            map(("Dx11", &mut id_dx11, adapters_dx11)),
            #[cfg(gl)]
            map(("GL", &mut id_gl, adapters_gl)),
        }

        let _ = (
            selected,
            id_vulkan.take(),
            id_metal.take(),
            id_dx12.take(),
            id_dx11.take(),
            id_gl.take(),
        );
        log::warn!("Some adapters are present, but enumerating them failed!");
        Err(RequestAdapterError::NotFound)
    }

    pub fn adapter_get_info<A: HalApi>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::AdapterInfo, InvalidAdapter> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| adapter.raw.info.clone())
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_get_texture_format_features<A: HalApi>(
        &self,
        adapter_id: AdapterId,
        format: wgt::TextureFormat,
    ) -> Result<wgt::TextureFormatFeatures, InvalidAdapter> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| adapter.get_texture_format_features(format))
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_features<A: HalApi>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::Features, InvalidAdapter> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| adapter.raw.features)
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_limits<A: HalApi>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::Limits, InvalidAdapter> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| adapter.raw.capabilities.limits.clone())
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_downlevel_properties<A: HalApi>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::DownlevelCapabilities, InvalidAdapter> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| adapter.raw.capabilities.downlevel)
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_drop<A: HalApi>(&self, adapter_id: AdapterId) {
        profiling::scope!("drop", "Adapter");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (mut adapter_guard, _) = hub.adapters.write(&mut token);

        let free = match adapter_guard.get_mut(adapter_id) {
            Ok(adapter) => adapter.life_guard.ref_count.take().unwrap().load() == 1,
            Err(_) => true,
        };
        if free {
            hub.adapters
                .unregister_locked(adapter_id, &mut *adapter_guard);
        }
    }
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn adapter_request_device<A: HalApi>(
        &self,
        adapter_id: AdapterId,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
        id_in: Input<G, DeviceId>,
    ) -> (DeviceId, Option<RequestDeviceError>) {
        profiling::scope!("request_device", "Adapter");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.devices.prepare(id_in);

        let error = loop {
            let (adapter_guard, mut token) = hub.adapters.read(&mut token);
            let adapter = match adapter_guard.get(adapter_id) {
                Ok(adapter) => adapter,
                Err(_) => break RequestDeviceError::InvalidAdapter,
            };
            let device = match adapter.create_device(adapter_id, desc, trace_path) {
                Ok(device) => device,
                Err(e) => break e,
            };
            let id = fid.assign(device, &mut token);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }
}
