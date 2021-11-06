use crate::{
    device::{Device, DeviceDescriptor},
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Input, Token},
    id::{AdapterId, DeviceId, SurfaceId, Valid},
    present::Presentation,
    LabelHelpers, LifeGuard, Stored, DOWNLEVEL_WARNING_MESSAGE,
};

use wgt::{Backend, Backends, PowerPreference};

use hal::{Adapter as _, Instance as _};
use thiserror::Error;

pub type RequestAdapterOptions = wgt::RequestAdapterOptions<SurfaceId>;
type HalInstance<A> = <A as hal::Api>::Instance;
//TODO: remove this
pub struct HalSurface<A: hal::Api> {
    pub raw: A::Surface,
    //pub acquired_texture: Option<A::SurfaceTexture>,
}

#[derive(Clone, Debug, Error)]
#[error("Limit '{name}' value {requested} is better than allowed {allowed}")]
pub struct FailedLimit {
    name: &'static str,
    requested: u32,
    allowed: u32,
}

fn check_limits(requested: &wgt::Limits, allowed: &wgt::Limits) -> Vec<FailedLimit> {
    use std::cmp::Ordering;
    let mut failed = Vec::new();

    macro_rules! compare {
        ($name:ident, $ordering:ident) => {
            match requested.$name.cmp(&allowed.$name) {
                Ordering::$ordering | Ordering::Equal => (),
                _ => failed.push(FailedLimit {
                    name: stringify!($name),
                    requested: requested.$name,
                    allowed: allowed.$name,
                }),
            }
        };
    }

    compare!(max_texture_dimension_1d, Less);
    compare!(max_texture_dimension_2d, Less);
    compare!(max_texture_dimension_3d, Less);
    compare!(max_texture_array_layers, Less);
    compare!(max_bind_groups, Less);
    compare!(max_dynamic_uniform_buffers_per_pipeline_layout, Less);
    compare!(max_dynamic_storage_buffers_per_pipeline_layout, Less);
    compare!(max_sampled_textures_per_shader_stage, Less);
    compare!(max_samplers_per_shader_stage, Less);
    compare!(max_storage_buffers_per_shader_stage, Less);
    compare!(max_storage_textures_per_shader_stage, Less);
    compare!(max_uniform_buffers_per_shader_stage, Less);
    compare!(max_uniform_buffer_binding_size, Less);
    compare!(max_storage_buffer_binding_size, Less);
    compare!(max_vertex_buffers, Less);
    compare!(max_vertex_attributes, Less);
    compare!(max_vertex_buffer_array_stride, Less);
    compare!(max_push_constant_size, Less);
    compare!(min_uniform_buffer_offset_alignment, Greater);
    compare!(min_storage_buffer_offset_alignment, Greater);
    failed
}

#[test]
fn downlevel_default_limits_less_than_default_limits() {
    let res = check_limits(&wgt::Limits::downlevel_defaults(), &wgt::Limits::default());
    assert!(
        res.is_empty(),
        "Downlevel limits are greater than default limits",
    )
}

#[derive(Default)]
pub struct Instance {
    #[allow(dead_code)]
    pub name: String,
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
    pub fn new(name: &str, backends: Backends) -> Self {
        fn init<A: HalApi>(mask: Backends) -> Option<A::Instance> {
            if mask.contains(A::VARIANT.into()) {
                let mut flags = hal::InstanceFlags::empty();
                if cfg!(debug_assertions) {
                    flags |= hal::InstanceFlags::VALIDATION;
                    flags |= hal::InstanceFlags::DEBUG;
                }
                let hal_desc = hal::InstanceDescriptor {
                    name: "wgpu",
                    flags,
                };
                unsafe { hal::Instance::init(&hal_desc).ok() }
            } else {
                None
            }
        }

        Self {
            name: name.to_string(),
            #[cfg(vulkan)]
            vulkan: init::<hal::api::Vulkan>(backends),
            #[cfg(metal)]
            metal: init::<hal::api::Metal>(backends),
            #[cfg(dx12)]
            dx12: init::<hal::api::Dx12>(backends),
            #[cfg(dx11)]
            dx11: init::<hal::api::Dx11>(backends),
            #[cfg(gl)]
            gl: init::<hal::api::Gles>(backends),
        }
    }

    pub(crate) fn destroy_surface(&self, surface: Surface) {
        backends_map! {
            let map = |(surface_backend, self_backend)| {
                unsafe {
                    if let Some(suf) = surface_backend {
                        self_backend.as_ref().unwrap().destroy_surface(suf.raw);
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
    pub(crate) presentation: Option<Presentation>,
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

impl Surface {
    pub fn get_preferred_format<A: HalApi>(
        &self,
        adapter: &Adapter<A>,
    ) -> Result<wgt::TextureFormat, GetSurfacePreferredFormatError> {
        // Check the four formats mentioned in the WebGPU spec.
        // Also, prefer sRGB over linear as it is better in
        // representing perceived colors.
        let preferred_formats = [
            wgt::TextureFormat::Bgra8UnormSrgb,
            wgt::TextureFormat::Rgba8UnormSrgb,
            wgt::TextureFormat::Bgra8Unorm,
            wgt::TextureFormat::Rgba8Unorm,
        ];

        let suf = A::get_surface(self);
        let caps = unsafe {
            profiling::scope!("surface_capabilities");
            adapter
                .raw
                .adapter
                .surface_capabilities(&suf.raw)
                .ok_or(GetSurfacePreferredFormatError::UnsupportedQueueFamily)?
        };

        preferred_formats
            .iter()
            .cloned()
            .find(|preferred| caps.formats.contains(preferred))
            .ok_or(GetSurfacePreferredFormatError::NotFound)
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

    pub fn is_surface_supported(&self, surface: &Surface) -> bool {
        let suf = A::get_surface(surface);
        unsafe { self.raw.adapter.surface_capabilities(&suf.raw) }.is_some()
    }

    pub(crate) fn get_texture_format_features(
        &self,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        use hal::TextureFormatCapabilities as Tfc;

        let caps = unsafe { self.raw.adapter.texture_format_capabilities(format) };
        let mut allowed_usages = format.describe().guaranteed_format_features.allowed_usages;

        allowed_usages.set(
            wgt::TextureUsages::TEXTURE_BINDING,
            caps.contains(Tfc::SAMPLED),
        );
        allowed_usages.set(
            wgt::TextureUsages::STORAGE_BINDING,
            caps.contains(Tfc::STORAGE),
        );
        allowed_usages.set(
            wgt::TextureUsages::RENDER_ATTACHMENT,
            caps.intersects(Tfc::COLOR_ATTACHMENT | Tfc::DEPTH_STENCIL_ATTACHMENT),
        );

        let mut flags = wgt::TextureFormatFeatureFlags::empty();
        flags.set(
            wgt::TextureFormatFeatureFlags::STORAGE_ATOMICS,
            caps.contains(Tfc::STORAGE_ATOMIC),
        );
        flags.set(
            wgt::TextureFormatFeatureFlags::STORAGE_READ_WRITE,
            caps.contains(Tfc::STORAGE_READ_WRITE),
        );

        // We are currently taking the filtering and blending together,
        // but we may reconsider this in the future if there are formats
        // in the wild for which these two capabilities do not match.
        let filterable = caps.contains(Tfc::SAMPLED_LINEAR)
            && (!caps.contains(Tfc::COLOR_ATTACHMENT)
                || caps.contains(Tfc::COLOR_ATTACHMENT_BLEND));

        wgt::TextureFormatFeatures {
            allowed_usages,
            flags,
            filterable,
        }
    }

    fn create_device_from_hal(
        &self,
        self_id: AdapterId,
        open: hal::OpenDevice<A>,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
    ) -> Result<Device<A>, RequestDeviceError> {
        let caps = &self.raw.capabilities;
        Device::new(
            open,
            Stored {
                value: Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            caps.alignments.clone(),
            caps.downlevel.clone(),
            desc,
            trace_path,
        )
        .or(Err(RequestDeviceError::OutOfMemory))
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
        if wgt::Backends::PRIMARY.contains(wgt::Backends::from(A::VARIANT))
            && !caps.downlevel.is_webgpu_compliant()
        {
            let missing_flags = wgt::DownlevelFlags::compliant() - caps.downlevel.flags;
            log::warn!(
                "Missing downlevel flags: {:?}\n{}",
                missing_flags,
                DOWNLEVEL_WARNING_MESSAGE
            );
            log::info!("{:#?}", caps.downlevel);
        }

        // Verify feature preconditions
        if desc
            .features
            .contains(wgt::Features::MAPPABLE_PRIMARY_BUFFERS)
            && self.raw.info.device_type == wgt::DeviceType::DiscreteGpu
        {
            log::warn!("Feature MAPPABLE_PRIMARY_BUFFERS enabled on a discrete gpu. This is a massive performance footgun and likely not what you wanted");
        }

        if let Some(_) = desc.label {
            //TODO
        }

        if let Some(failed) = check_limits(&desc.limits, &caps.limits).pop() {
            return Err(RequestDeviceError::LimitsExceeded(failed));
        }

        let open = unsafe { self.raw.adapter.open(desc.features, &desc.limits) }.map_err(
            |err| match err {
                hal::DeviceError::Lost => RequestDeviceError::DeviceLost,
                hal::DeviceError::OutOfMemory => RequestDeviceError::OutOfMemory,
            },
        )?;

        self.create_device_from_hal(self_id, open, desc, trace_path)
    }
}

impl<A: hal::Api> crate::hub::Resource for Adapter<A> {
    const TYPE: &'static str = "Adapter";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

#[derive(Clone, Debug, Error)]
pub enum IsSurfaceSupportedError {
    #[error("invalid adapter")]
    InvalidAdapter,
    #[error("invalid surface")]
    InvalidSurface,
}

#[derive(Clone, Debug, Error)]
pub enum GetSurfacePreferredFormatError {
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
    #[error(transparent)]
    LimitsExceeded(#[from] FailedLimit),
    #[error("device has no queue supporting graphics")]
    NoGraphicsQueue,
    #[error("not enough memory left")]
    OutOfMemory,
    #[error("unsupported features were requested: {0:?}")]
    UnsupportedFeature(wgt::Features),
}

pub enum AdapterInputs<'a, I> {
    IdSet(&'a [I], fn(&I) -> Backend),
    Mask(Backends, fn(Backend) -> I),
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

        //Note: using adummy argument to work around the following error:
        //> cannot provide explicit generic arguments when `impl Trait` is used in argument position
        fn init<A: hal::Api>(
            _: A,
            inst: &Option<A::Instance>,
            handle: &impl raw_window_handle::HasRawWindowHandle,
        ) -> Option<HalSurface<A>> {
            inst.as_ref().and_then(|inst| unsafe {
                match inst.create_surface(handle) {
                    Ok(raw) => Some(HalSurface {
                        raw,
                        //acquired_texture: None,
                    }),
                    Err(e) => {
                        log::warn!("Error: {:?}", e);
                        None
                    }
                }
            })
        }

        let surface = Surface {
            presentation: None,
            #[cfg(vulkan)]
            vulkan: init(hal::api::Vulkan, &self.instance.vulkan, handle),
            #[cfg(metal)]
            metal: init(hal::api::Metal, &self.instance.metal, handle),
            #[cfg(dx12)]
            dx12: init(hal::api::Dx12, &self.instance.dx12, handle),
            #[cfg(dx11)]
            dx11: init(hal::api::Dx11, &self.instance.dx11, handle),
            #[cfg(gl)]
            gl: init(hal::api::Gles, &self.instance.gl, handle),
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }

    #[cfg(metal)]
    pub fn instance_create_surface_metal(
        &self,
        layer: *mut std::ffi::c_void,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("create_surface_metal", "Instance");

        let surface = Surface {
            presentation: None,
            metal: self.instance.metal.as_ref().map(|inst| HalSurface {
                raw: {
                    // we don't want to link to metal-rs for this
                    #[allow(clippy::transmute_ptr_to_ref)]
                    inst.create_surface_from_layer(unsafe { std::mem::transmute(layer) })
                },
                //acquired_texture: None,
            }),
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }

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
                        profiling::scope!("enumerating", backend_info);
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

        fn gather<A: HalApi, I: Clone>(
            _: A,
            instance: Option<&A::Instance>,
            inputs: &AdapterInputs<I>,
            compatible_surface: Option<&Surface>,
            force_software: bool,
            device_types: &mut Vec<wgt::DeviceType>,
        ) -> (Option<I>, Vec<hal::ExposedAdapter<A>>) {
            let id = inputs.find(A::VARIANT);
            match instance {
                Some(inst) if id.is_some() => {
                    let mut adapters = unsafe { inst.enumerate_adapters() };
                    if force_software {
                        adapters.retain(|exposed| exposed.info.device_type == wgt::DeviceType::Cpu);
                    }
                    if let Some(surface) = compatible_surface {
                        let suf_raw = &A::get_surface(surface).raw;
                        adapters.retain(|exposed| unsafe {
                            exposed.adapter.surface_capabilities(suf_raw).is_some()
                        });
                    }
                    device_types.extend(adapters.iter().map(|ad| ad.info.device_type));
                    (id, adapters)
                }
                _ => (id, Vec::new()),
            }
        }

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

        #[cfg(vulkan)]
        let (mut id_vulkan, adapters_vk) = gather(
            hal::api::Vulkan,
            self.instance.vulkan.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );
        #[cfg(metal)]
        let (mut id_metal, adapters_metal) = gather(
            hal::api::Metal,
            self.instance.metal.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );
        #[cfg(dx12)]
        let (mut id_dx12, adapters_dx12) = gather(
            hal::api::Dx12,
            self.instance.dx12.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );
        #[cfg(dx11)]
        let (mut id_dx11, adapters_dx11) = gather(
            hal::api::Dx11,
            self.instance.dx11.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );
        #[cfg(gl)]
        let (mut id_gl, adapters_gl) = gather(
            hal::api::Gles,
            self.instance.gl.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );

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
            map(("Metal", &mut id_metal, adapters_metal)),
            #[cfg(dx12)]
            map(("Dx12", &mut id_dx12, adapters_dx12)),
            #[cfg(dx11)]
            map(("Dx11", &mut id_dx11, adapters_dx11)),
            #[cfg(gl)]
            map(("GL", &mut id_gl, adapters_gl)),
        }

        let _ = selected;
        log::warn!("Some adapters are present, but enumerating them failed!");
        Err(RequestAdapterError::NotFound)
    }

    /// # Safety
    ///
    /// `hal_adapter` must be created from this global internal instance handle.
    pub unsafe fn create_adapter_from_hal<A: HalApi>(
        &self,
        hal_adapter: hal::ExposedAdapter<A>,
        input: Input<G, AdapterId>,
    ) -> AdapterId {
        profiling::scope!("create_adapter_from_hal", "Instance");

        let mut token = Token::root();
        let fid = A::hub(self).adapters.prepare(input);

        match A::VARIANT {
            #[cfg(vulkan)]
            Backend::Vulkan => fid.assign(Adapter::new(hal_adapter), &mut token).0,
            #[cfg(metal)]
            Backend::Metal => fid.assign(Adapter::new(hal_adapter), &mut token).0,
            #[cfg(dx12)]
            Backend::Dx12 => fid.assign(Adapter::new(hal_adapter), &mut token).0,
            #[cfg(dx11)]
            Backend::Dx11 => fid.assign(Adapter::new(hal_adapter), &mut token).0,
            #[cfg(gl)]
            Backend::Gl => fid.assign(Adapter::new(hal_adapter), &mut token).0,
            _ => unreachable!(),
        }
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
            .map(|adapter| adapter.raw.capabilities.downlevel.clone())
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

    /// # Safety
    ///
    /// - `hal_device` must be created from `adapter_id` or its internal handle.
    /// - `desc` must be a subset of `hal_device` features and limits.
    pub unsafe fn create_device_from_hal<A: HalApi>(
        &self,
        adapter_id: AdapterId,
        hal_device: hal::OpenDevice<A>,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
        id_in: Input<G, DeviceId>,
    ) -> (DeviceId, Option<RequestDeviceError>) {
        profiling::scope!("create_device_from_hal", "Adapter");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.devices.prepare(id_in);

        let error = loop {
            let (adapter_guard, mut token) = hub.adapters.read(&mut token);
            let adapter = match adapter_guard.get(adapter_id) {
                Ok(adapter) => adapter,
                Err(_) => break RequestDeviceError::InvalidAdapter,
            };
            let device =
                match adapter.create_device_from_hal(adapter_id, hal_device, desc, trace_path) {
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

/// Generates a set of backends from a comma separated list of case-insensitive backend names.
///
/// Whitespace is stripped, so both 'gl, dx12' and 'gl,dx12' are valid.
///
/// Always returns WEBGPU on wasm over webgpu.
///
/// Names:
/// - vulkan = "vulkan" or "vk"
/// - dx12   = "dx12" or "d3d12"
/// - dx11   = "dx11" or "d3d11"
/// - metal  = "metal" or "mtl"
/// - gles   = "opengl" or "gles" or "gl"
/// - webgpu = "webgpu"
pub fn parse_backends_from_comma_list(string: &str) -> Backends {
    let mut backends = Backends::empty();
    for backend in string.to_lowercase().split(',') {
        backends |= match backend.trim() {
            "vulkan" | "vk" => Backends::VULKAN,
            "dx12" | "d3d12" => Backends::DX12,
            "dx11" | "d3d11" => Backends::DX11,
            "metal" | "mtl" => Backends::METAL,
            "opengl" | "gles" | "gl" => Backends::GL,
            "webgpu" => Backends::BROWSER_WEBGPU,
            b => {
                log::warn!("unknown backend string '{}'", b);
                continue;
            }
        }
    }

    if backends.is_empty() {
        log::warn!("no valid backend strings found!");
    }

    backends
}
