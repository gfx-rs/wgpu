use crate::{
    device::{Device, DeviceDescriptor},
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Input, Token},
    id::{self, AdapterId, AnyBackend, DeviceId, Hkt, SurfaceId},
    present::Presentation,
    LifeGuard, DOWNLEVEL_WARNING_MESSAGE,
};
/* #[cfg(feature = "trace")]
use crate::hub::IdentityHandler; */

use wgt::{Backend, Backends, PowerPreference, BIND_BUFFER_ALIGNMENT};

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
            map((surface.raw.vulkan, &self.vulkan)),
            #[cfg(metal)]
            map((surface.raw.metal, &self.metal)),
            #[cfg(dx12)]
            map((surface.raw.dx12, &self.dx12)),
            #[cfg(dx11)]
            map((surface.raw.dx11, &self.dx11)),
            #[cfg(gl)]
            map((surface.raw.gl, &self.gl)),
        }
    }
}

pub struct Surface {
    pub(crate) presentation: Option<Presentation>,
    pub raw: SurfaceRaw,
}

pub struct SurfaceRaw {
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

    /* #[inline]
    fn trace_resources<'b, E, Trace: FnMut(id::Cached<<Self as AnyBackend>::Backend, id::IdGuardCon>) -> Result<(), E>>(
        _: <id::IdGuardCon<'b> as Hkt<Self>>::Output,
        _: Trace,
    ) -> Result<(), E>
        where
            <Self as AnyBackend>::Backend: HalApi,
    {
        // Nothing to trace.
        Ok(())
    } */

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

        let suf = A::get_surface(&self.raw);
        let caps = unsafe {
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
}

impl<A: HalApi> Adapter<A> {
    fn new(raw: hal::ExposedAdapter<A>) -> Self {
        Self {
            raw,
        }
    }

    pub fn is_surface_supported(&self, surface: &Surface) -> bool {
        let suf = A::get_surface(&surface.raw);
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
        self,
        /*self_id: AdapterId,*/
        open: hal::OpenDevice<A>,
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
        if let Some(failed) = check_limits(&desc.limits, &caps.limits).pop() {
            return Err(RequestDeviceError::LimitsExceeded(failed));
        }

        Device::new(
            open,
            /*Stored {
                value: Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            }*/self,
            desc,
            trace_path,
        )
        .or(Err(RequestDeviceError::OutOfMemory))
    }

    fn create_device(
        self,
        // self_id: AdapterId,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
    ) -> Result<Device<A>, RequestDeviceError> {
        let open = unsafe { self.raw.adapter.open(desc.features) }.map_err(|err| match err {
            hal::DeviceError::Lost => RequestDeviceError::DeviceLost,
            hal::DeviceError::OutOfMemory => RequestDeviceError::OutOfMemory,
        })?;

        self.create_device_from_hal(/*self_id, */open, desc, trace_path)
    }
}

impl<A: hal::Api> crate::hub::Resource for Adapter<A> {
    const TYPE: &'static str = "Adapter";

    #[inline]
    fn trace_resources<'b, E, Trace: FnMut(id::Cached<<Self as AnyBackend>::Backend, id::IdGuardCon>) -> Result<(), E>>(
        _: <id::IdGuardCon<'b> as Hkt<Self>>::Output,
        _: Trace,
    ) -> Result<(), E>
        where
            <Self as AnyBackend>::Backend: HalApi + 'b,
    {
        // Nothing to trace.
        Ok(())
    }

    fn life_guard(&self) -> &LifeGuard {
        unimplemented!("FIXME: This method needs to go away!")
        // &self.life_guard
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

/* pub enum AdapterInputs</* #[cfg(feature="replay")] *//*'a, I*/> {
    /* #[cfg(feature="replay")] */
    // IdSet(&'a [I], fn(&I) -> Backend),
    Mask(Backends/*, fn(Backend) -> I*/),
}

/* #[cfg(feature="replay")] */
pub trait AdapterFind : Clone {}
/* #[cfg(not(feature="replay"))]
pub trait AdapterFind {} */

/* #[cfg(feature="replay")] */
impl<I: Clone> AdapterFind for I {}
/* #[cfg(not(feature="replay"))]
impl<I> AdapterFind for I {} */

// #[cfg(feature="replay")]
pub type AdapterInputMap<G> = Input<G, id::Id<Adapter<hal::api::Empty>>>;
/* #[cfg(not(feature="replay"))]
pub type AdapterInputMap<G> = Input<G, id::Id<Adapter<hal::api::Empty>>>; */

/* #[cfg(feature="replay")] */
impl<I: AdapterFind> AdapterInputs<'_, I> {
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
} */

/* #[cfg(not(feature="replay"))]
impl<I: AdapterFind> AdapterInputs<I> {
    fn find(&self, b: Backend) -> Option<I> {
        let Self::Mask(bits, ref fun) = *self;
        if bits.contains(b.into()) {
            Some(fun(b))
        } else {
            None
        }
    }
} */

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

        let raw = SurfaceRaw {
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
        let surface = Surface {
            presentation: None,
            raw,
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

        let raw = SurfaceRaw {
            metal: self.instance.metal.as_ref().map(|inst| HalSurface {
                raw: {
                    // we don't want to link to metal-rs for this
                    #[allow(clippy::transmute_ptr_to_ref)]
                    inst.create_surface_from_layer(unsafe { std::mem::transmute(layer) })
                },
                //acquired_texture: None,
            }),
        };
        let surface = Surface {
            presentation: None,
            raw,
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

    pub fn enumerate_adapters(&self, inputs: /*AdapterInputs<AdapterInputMap<G>>*/Backends) -> Vec<AdapterId> {
        profiling::scope!("enumerate_adapters", "Instance");

        let instance = &self.instance;
        // let mut token = Token::root();
        let mut adapters = Vec::new();

        backends_map! {
            let map = |(instance_field, backend, backend_info)| {
                if let Some(ref inst) = *instance_field {
                    if /*let Some(_id_backend) = */inputs.contains(backend.into()) {
                        for raw in unsafe {inst.enumerate_adapters()} {
                            let adapter = Adapter::new(raw);
                            log::info!("Adapter {} {:?}", backend_info, adapter.raw.info);
                            /* #[cfg(feature="trace")]
                            {
                                fn get_hub<'a, A: HalApi, G: GlobalIdentityHandlerFactory>
                                    (_: &Adapter<A>, global: &'a Global<G>) -> &'a crate::hub::Hub<A, G> {
                                    HalApi::hub(global)
                                }
                                fn get_variant<A: HalApi>(_: &Adapter<A>) -> Backend { A::VARIANT }
                                let hub = get_hub(&adapter, self);
                                /*let id = */hub.adapters
                                    .process(_id_backend.clone(), get_variant(&adapter))
                                    /* .prepare(id_backend.clone())
                                    .assign(adapter, &mut token) */;
                            } */
                            adapters.push(/*id.0*/id::BoxId2::upcast_backend(Box::new(adapter)));
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

    #[allow(unused_assignments)] // Can error spuriously if only one backend is available.
    pub fn request_adapter(
        &self,
        desc: &RequestAdapterOptions,
        inputs: /*AdapterInputs<AdapterInputMap<G>>*/Backends,
    ) -> Result<AdapterId, RequestAdapterError> {
        profiling::scope!("pick_adapter", "Instance");

        let instance = &self.instance;
        let mut token = Token::root();
        let (surface_guard, _) = self.surfaces.read(&mut token);
        let compatible_surface = desc
            .compatible_surface
            .map(|id| {
                surface_guard
                    .get(id)
                    .map_err(|_| RequestAdapterError::InvalidSurface(id))
            })
            .transpose()?;
        let mut device_types = Vec::new();

        // let mut id_vulkan = inputs.contains(Backend::Vulkan);
        // let mut id_metal = inputs.contains(Backend::Metal);
        // let mut id_dx12 = inputs.contains(Backend::Dx12);
        // let mut id_dx11 = inputs.contains(Backend::Dx11);
        // let mut id_gl = inputs.contains(Backend::Gl);

        backends_map! {
            let map = |(instance_backend, backend, surface_backend)| {
                match *instance_backend {
                    Some(ref inst) if /*id_backend.is_some()*/inputs.contains(backend.into()) => {
                        let mut adapters = unsafe { inst.enumerate_adapters() };
                        if let Some(surface_backend) = compatible_surface.and_then(|suf| surface_backend(&suf.raw)) {
                            adapters.retain(|exposed| unsafe {
                                exposed.adapter.surface_capabilities(&surface_backend.raw).is_some()
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
            let adapters_vk = map((&instance.vulkan, /*&id_vulkan*/Backend::Vulkan, {
                fn surface_vulkan(surf: &SurfaceRaw) -> Option<&HalSurface<hal::api::Vulkan>> {
                    surf.vulkan.as_ref()
                }
                surface_vulkan
            }));
            #[cfg(metal)]
            let adapters_mtl = map((&instance.metal, /*&id_metal*/Backend::Metal, {
                fn surface_metal(surf: &SurfaceRaw) -> Option<&HalSurface<hal::api::Metal>> {
                    surf.metal.as_ref()
                }
                surface_metal
            }));
            #[cfg(dx12)]
            let adapters_dx12 = map((&instance.dx12, /*&id_dx12*/Backend::Dx12, {
                fn surface_dx12(surf: &SurfaceRaw) -> Option<&HalSurface<hal::api::Dx12>> {
                    surf.dx12.as_ref()
                }
                surface_dx12
            }));
            #[cfg(dx11)]
            let adapters_dx11 = map((&instance.dx11, /*&id_dx11*/Backend::Dx11, {
                fn surface_dx11(surf: &SurfaceRaw) -> Option<&HalSurface<hal::api::Dx11>> {
                    surf.dx11.as_ref()
                }
                surface_dx11
            }));
            #[cfg(gl)]
            let adapters_gl = map((&instance.gl, /*&id_gl*/Backend::Gles, {
                fn surface_gl(surf: &SurfaceRaw) -> Option<&HalSurface<hal::api::Gles>> {
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

        #[allow(unused_assignments)]
        let mut selected = preferred_gpu.unwrap_or(0);

        backends_map! {
            let map = |(info_adapter, _id_backend, mut adapters_backend)| {
                if selected < adapters_backend.len() {
                    let adapter = Adapter::new(adapters_backend.swap_remove(selected));
                    log::info!("Adapter {} {:?}", info_adapter, adapter.raw.info);
                    /* #[cfg(feature="trace")]
                    {
                        fn get_hub<'a, A: HalApi, G: GlobalIdentityHandlerFactory>
                            (_: &Adapter<A>, global: &'a Global<G>) -> &'a crate::hub::Hub<A, G> {
                            HalApi::hub(global)
                        }
                        fn get_variant<A: HalApi>(_: &Adapter<A>) -> Backend { A::VARIANT }
                        let hub = get_hub(&adapter, self);
                        /*let id = *//*HalApi::hub::<G>(self)*/hub.adapters
                            .process(_id_backend.take().unwrap(), get_variant(&adapter))
                            /* .prepare(id_backend.take().unwrap())
                            .assign(adapter, &mut token) */;
                    } */
                    return Ok(id::BoxId2::upcast_backend(Box::new(adapter)));
                }
                selected -= adapters_backend.len();
            };

            #[cfg(vulkan)]
            map(("Vulkan", /*&mut id_vulkan, */(), adapters_vk)),
            #[cfg(metal)]
            map(("Metal", /*&mut id_metal, */(), adapters_mtl)),
            #[cfg(dx12)]
            map(("Dx12", /*&mut id_dx12, */(), adapters_dx12)),
            #[cfg(dx11)]
            map(("Dx11", /*&mut id_dx11, */(), adapters_dx11)),
            #[cfg(gl)]
            map(("GL", /*&mut id_gl, */(), adapters_gl)),
        }

        /* let _ = (
            selected,
            id_vulkan.take(),
            id_metal.take(),
            id_dx12.take(),
            id_dx11.take(),
            id_gl.take(),
        ); */
        log::warn!("Some adapters are present, but enumerating them failed!");
        Err(RequestAdapterError::NotFound)
    }

    /// # Safety
    ///
    /// `hal_adapter` must be created from this global internal instance handle.
    pub unsafe fn create_adapter_from_hal<A: HalApi>(
        hal_adapter: hal::ExposedAdapter<A>,
    ) -> AdapterId {
        profiling::scope!("create_adapter_from_hal", "Instance");

        id::BoxId2::upcast_backend(Box::new(Adapter::new(hal_adapter)))
    }

    pub fn adapter_get_info<A: HalApi>(
        // &self,
        adapter: &Adapter<A>,
        // adapter_id: AdapterId,
    ) -> /*Result<*/wgt::AdapterInfo/*, InvalidAdapter>*/ {
        /* let hub = A::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| */adapter.raw.info.clone()/*)
            .map_err(|_| InvalidAdapter)*/
    }

    pub fn adapter_get_texture_format_features<A: HalApi>(
        // &self,
        adapter: &Adapter<A>,
        // adapter_id: AdapterId,
        format: wgt::TextureFormat,
    ) -> /*Result<*/wgt::TextureFormatFeatures/*, InvalidAdapter>*/ {
        /* let hub = A::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| */adapter.get_texture_format_features(format)/*)
            .map_err(|_| InvalidAdapter)*/
    }

    pub fn adapter_features<A: HalApi>(
        // &self,
        adapter: &Adapter<A>,
        // adapter_id: AdapterId,
    ) -> /*Result<*/wgt::Features/*, InvalidAdapter>*/ {
        /* let hub = A::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| */adapter.raw.features/*)
            .map_err(|_| InvalidAdapter)*/
    }

    pub fn adapter_limits<A: HalApi>(
        // &self,
        adapter: &Adapter<A>,
        // adapter_id: AdapterId,
    ) -> /*Result<*/wgt::Limits/*, InvalidAdapter>*/ {
        /*let hub = A::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| */adapter.raw.capabilities.limits.clone()/*)
            .map_err(|_| InvalidAdapter)*/
    }

    pub fn adapter_downlevel_properties<A: HalApi>(
        // &self,
        adapter: &Adapter<A>,
        // adapter_id: AdapterId,
    ) -> /*Result<*/wgt::DownlevelCapabilities/*, InvalidAdapter>*/ {
        /* let hub = A::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        adapter_guard
            .get(adapter_id)
            .map(|adapter| */adapter.raw.capabilities.downlevel.clone()/*)
            .map_err(|_| InvalidAdapter)*/
    }

    /* pub fn adapter_drop<A: HalApi>(&self, _adapter_id: /*AdapterId*/id::Id<Adapter<hal::api::Empty>>) {
        profiling::scope!("drop", "Adapter");

        /* let hub = A::hub(self);
        let mut token = Token::root();
        let (mut adapter_guard, _) = hub.adapters.write(&mut token);

        let free = match adapter_guard.get_mut(adapter_id) {
            Ok(adapter) => adapter.life_guard.ref_count.take().unwrap().load() == 1,
            Err(_) => true,
        };
        if free {
            hub.adapters
                .unregister_locked(adapter_id, &mut *adapter_guard);
        } */
        /* #[cfg(feature="trace")]
        let _ = A::hub(self)
            .adapters
            .free(_adapter_id); */
    } */
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn adapter_request_device<A: HalApi>(
        /* #[cfg(feature = "trace")]
        &self, */
        // adapter_id: AdapterId,
        adapter: Adapter<A>,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
        /* #[cfg(feature = "trace")]
        id_in: Input<G, id::Id<Device<hal::api::Empty>>>, */
    ) -> Result<DeviceId, RequestDeviceError> {
        profiling::scope!("request_device", "Adapter");

        /* #[cfg(feature = "trace")]
        let hub = A::hub(self); */
        // let mut token = Token::root();
        /* #[cfg(feature = "trace")]
        let fid;// = hub.devices.prepare(id_in); */

        let error = loop {
            /* let (adapter_guard, mut token) = hub.adapters.read(&mut token);
            let adapter = match adapter_guard.get(adapter_id) {
                Ok(adapter) => adapter,
                Err(_) => break RequestDeviceError::InvalidAdapter,
            }; */
            let device = match adapter.create_device(/*adapter_id, */desc, trace_path) {
                Ok(device) => device,
                Err(e) => break e,
            };
            /* #[cfg(feature = "trace")]
            { /*fid = */hub.devices./*prepare(id_in)*/process(id_in, A::VARIANT); } */
            let id = id::ValidId2::<Device<A>>::new(std::sync::Arc::new(device));
            let id = id::Id2::upcast_backend(id)/*fid.assign(device, &mut token)*/;
            return Ok(id);
        };

        // let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        // (id, Some(error))
        Err(error)
    }

    /// # Safety
    ///
    /// - `hal_device` must be created from `adapter_id` or its internal handle.
    /// - `desc` must be a subset of `hal_device` features and limits.
    pub unsafe fn create_device_from_hal<A: HalApi>(
        adapter: Adapter<A>,
        hal_device: hal::OpenDevice<A>,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
    ) -> Result<DeviceId, RequestDeviceError> {
        profiling::scope!("request_device", "Adapter");

        let device = adapter.create_device_from_hal(hal_device, desc, trace_path)?;
        let id = id::ValidId2::<Device<A>>::new(std::sync::Arc::new(device));
        Ok(id::Id2::upcast_backend(id))
    }
}
