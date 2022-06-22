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
    requested: u64,
    allowed: u64,
}

fn check_limits(requested: &wgt::Limits, allowed: &wgt::Limits) -> Vec<FailedLimit> {
    let mut failed = Vec::new();

    requested.check_limits_with_fail_fn(allowed, false, |name, requested, allowed| {
        failed.push(FailedLimit {
            name,
            requested,
            allowed,
        })
    });

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
        fn init<A: HalApi>(_: A, mask: Backends) -> Option<A::Instance> {
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
            vulkan: init(hal::api::Vulkan, backends),
            #[cfg(metal)]
            metal: init(hal::api::Metal, backends),
            #[cfg(dx12)]
            dx12: init(hal::api::Dx12, backends),
            #[cfg(dx11)]
            dx11: init(hal::api::Dx11, backends),
            #[cfg(gl)]
            gl: init(hal::api::Gles, backends),
        }
    }

    pub(crate) fn destroy_surface(&self, surface: Surface) {
        fn destroy<A: HalApi>(
            _: A,
            instance: &Option<A::Instance>,
            surface: Option<HalSurface<A>>,
        ) {
            unsafe {
                if let Some(suf) = surface {
                    instance.as_ref().unwrap().destroy_surface(suf.raw);
                }
            }
        }
        #[cfg(vulkan)]
        destroy(hal::api::Vulkan, &self.vulkan, surface.vulkan);
        #[cfg(metal)]
        destroy(hal::api::Metal, &self.metal, surface.metal);
        #[cfg(dx12)]
        destroy(hal::api::Dx12, &self.dx12, surface.dx12);
        #[cfg(dx11)]
        destroy(hal::api::Dx11, &self.dx11, surface.dx11);
        #[cfg(gl)]
        destroy(hal::api::Gles, &self.gl, surface.gl);
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
    pub fn get_supported_formats<A: HalApi>(
        &self,
        adapter: &Adapter<A>,
    ) -> Result<Vec<wgt::TextureFormat>, GetSurfacePreferredFormatError> {
        let suf = A::get_surface(self);
        let mut caps = unsafe {
            profiling::scope!("surface_capabilities");
            adapter
                .raw
                .adapter
                .surface_capabilities(&suf.raw)
                .ok_or(GetSurfacePreferredFormatError::UnsupportedQueueFamily)?
        };

        // TODO: maybe remove once we support texture view changing srgb-ness
        caps.formats.sort_by_key(|f| !f.describe().srgb);

        Ok(caps.formats)
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
        flags.set(
            wgt::TextureFormatFeatureFlags::FILTERABLE,
            caps.contains(Tfc::SAMPLED_LINEAR)
                && (!caps.contains(Tfc::COLOR_ATTACHMENT)
                    || caps.contains(Tfc::COLOR_ATTACHMENT_BLEND)),
        );

        flags.set(
            wgt::TextureFormatFeatureFlags::MULTISAMPLE,
            caps.contains(Tfc::MULTISAMPLE),
        );
        flags.set(
            wgt::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE,
            caps.contains(Tfc::MULTISAMPLE_RESOLVE),
        );

        wgt::TextureFormatFeatures {
            allowed_usages,
            flags,
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

        //Note: using a dummy argument to work around the following error:
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
            #[cfg(vulkan)]
            vulkan: None,
            #[cfg(gl)]
            gl: None,
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }

    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    pub fn create_surface_webgl_canvas(
        &self,
        canvas: &web_sys::HtmlCanvasElement,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("create_surface_webgl_canvas", "Instance");

        let surface = Surface {
            presentation: None,
            gl: self.instance.gl.as_ref().map(|inst| HalSurface {
                raw: {
                    inst.create_surface_from_canvas(canvas)
                        .expect("Create surface from canvas")
                },
            }),
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }

    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    pub fn create_surface_webgl_offscreen_canvas(
        &self,
        canvas: &web_sys::OffscreenCanvas,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("create_surface_webgl_offscreen_canvas", "Instance");

        let surface = Surface {
            presentation: None,
            gl: self.instance.gl.as_ref().map(|inst| HalSurface {
                raw: {
                    inst.create_surface_from_offscreen_canvas(canvas)
                        .expect("Create surface from offscreen canvas")
                },
            }),
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }

    #[cfg(dx12)]
    /// # Safety
    ///
    /// The visual must be valid and able to be used to make a swapchain with.
    pub unsafe fn instance_create_surface_from_visual(
        &self,
        visual: *mut std::ffi::c_void,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("instance_create_surface_from_visual", "Instance");

        let surface = Surface {
            presentation: None,
            #[cfg(vulkan)]
            vulkan: None,
            dx12: self.instance.dx12.as_ref().map(|inst| HalSurface {
                raw: { inst.create_surface_from_visual(visual as _) },
            }),
            dx11: None,
            #[cfg(gl)]
            gl: None,
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

    fn enumerate<A: HalApi>(
        &self,
        _: A,
        instance: &Option<A::Instance>,
        inputs: &AdapterInputs<Input<G, AdapterId>>,
        list: &mut Vec<AdapterId>,
    ) {
        let inst = match *instance {
            Some(ref inst) => inst,
            None => return,
        };
        let id_backend = match inputs.find(A::VARIANT) {
            Some(id) => id,
            None => return,
        };

        profiling::scope!("enumerating", &*format!("{:?}", A::VARIANT));
        let hub = HalApi::hub(self);
        let mut token = Token::root();

        let hal_adapters = unsafe { inst.enumerate_adapters() };
        for raw in hal_adapters {
            let adapter = Adapter::new(raw);
            log::info!("Adapter {:?} {:?}", A::VARIANT, adapter.raw.info);
            let id = hub
                .adapters
                .prepare(id_backend.clone())
                .assign(adapter, &mut token);
            list.push(id.0);
        }
    }

    pub fn enumerate_adapters(&self, inputs: AdapterInputs<Input<G, AdapterId>>) -> Vec<AdapterId> {
        profiling::scope!("enumerate_adapters", "Instance");

        let mut adapters = Vec::new();

        #[cfg(vulkan)]
        self.enumerate(
            hal::api::Vulkan,
            &self.instance.vulkan,
            &inputs,
            &mut adapters,
        );
        #[cfg(metal)]
        self.enumerate(
            hal::api::Metal,
            &self.instance.metal,
            &inputs,
            &mut adapters,
        );
        #[cfg(dx12)]
        self.enumerate(hal::api::Dx12, &self.instance.dx12, &inputs, &mut adapters);
        #[cfg(dx11)]
        self.enumerate(hal::api::Dx11, &self.instance.dx11, &inputs, &mut adapters);
        #[cfg(gl)]
        self.enumerate(hal::api::Gles, &self.instance.gl, &inputs, &mut adapters);

        adapters
    }

    fn select<A: HalApi>(
        &self,
        selected: &mut usize,
        new_id: Option<Input<G, AdapterId>>,
        mut list: Vec<hal::ExposedAdapter<A>>,
    ) -> Option<AdapterId> {
        match selected.checked_sub(list.len()) {
            Some(left) => {
                *selected = left;
                None
            }
            None => {
                let mut token = Token::root();
                let adapter = Adapter::new(list.swap_remove(*selected));
                log::info!("Adapter {:?} {:?}", A::VARIANT, adapter.raw.info);
                let id = HalApi::hub(self)
                    .adapters
                    .prepare(new_id.unwrap())
                    .assign(adapter, &mut token);
                Some(id.0)
            }
        }
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

        #[cfg(vulkan)]
        let (id_vulkan, adapters_vk) = gather(
            hal::api::Vulkan,
            self.instance.vulkan.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );
        #[cfg(metal)]
        let (id_metal, adapters_metal) = gather(
            hal::api::Metal,
            self.instance.metal.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );
        #[cfg(dx12)]
        let (id_dx12, adapters_dx12) = gather(
            hal::api::Dx12,
            self.instance.dx12.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );
        #[cfg(dx11)]
        let (id_dx11, adapters_dx11) = gather(
            hal::api::Dx11,
            self.instance.dx11.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );
        #[cfg(gl)]
        let (id_gl, adapters_gl) = gather(
            hal::api::Gles,
            self.instance.gl.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );

        // need to free the token to be used by `select`
        drop(surface_guard);
        drop(token);
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
            PowerPreference::HighPerformance => discrete.or(integrated).or(other).or(virt).or(cpu),
        };

        let mut selected = preferred_gpu.unwrap_or(0);
        #[cfg(vulkan)]
        if let Some(id) = self.select(&mut selected, id_vulkan, adapters_vk) {
            return Ok(id);
        }
        #[cfg(metal)]
        if let Some(id) = self.select(&mut selected, id_metal, adapters_metal) {
            return Ok(id);
        }
        #[cfg(dx12)]
        if let Some(id) = self.select(&mut selected, id_dx12, adapters_dx12) {
            return Ok(id);
        }
        #[cfg(dx11)]
        if let Some(id) = self.select(&mut selected, id_dx11, adapters_dx11) {
            return Ok(id);
        }
        #[cfg(gl)]
        if let Some(id) = self.select(&mut selected, id_gl, adapters_gl) {
            return Ok(id);
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

    pub fn adapter_downlevel_capabilities<A: HalApi>(
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
