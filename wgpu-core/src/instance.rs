use crate::{
    device::{Device, DeviceDescriptor},
    global::Global,
    hal_api::HalApi,
    hub::Token,
    id::{AdapterId, DeviceId, SurfaceId, Valid},
    identity::{GlobalIdentityHandlerFactory, Input},
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
    #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
    pub vulkan: Option<HalInstance<hal::api::Vulkan>>,
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    pub metal: Option<HalInstance<hal::api::Metal>>,
    #[cfg(all(feature = "dx12", windows))]
    pub dx12: Option<HalInstance<hal::api::Dx12>>,
    #[cfg(all(feature = "dx11", windows))]
    pub dx11: Option<HalInstance<hal::api::Dx11>>,
    #[cfg(feature = "gles")]
    pub gl: Option<HalInstance<hal::api::Gles>>,
    pub flags: wgt::InstanceFlags,
}

impl Instance {
    pub fn new(name: &str, instance_desc: wgt::InstanceDescriptor) -> Self {
        fn init<A: HalApi>(_: A, instance_desc: &wgt::InstanceDescriptor) -> Option<A::Instance> {
            if instance_desc.backends.contains(A::VARIANT.into()) {
                let hal_desc = hal::InstanceDescriptor {
                    name: "wgpu",
                    flags: instance_desc.flags,
                    dx12_shader_compiler: instance_desc.dx12_shader_compiler.clone(),
                    gles_minor_version: instance_desc.gles_minor_version,
                };
                match unsafe { hal::Instance::init(&hal_desc) } {
                    Ok(instance) => {
                        log::debug!("Instance::new: created {:?} backend", A::VARIANT);
                        Some(instance)
                    }
                    Err(err) => {
                        log::debug!(
                            "Instance::new: failed to create {:?} backend: {:?}",
                            A::VARIANT,
                            err
                        );
                        None
                    }
                }
            } else {
                log::trace!("Instance::new: backend {:?} not requested", A::VARIANT);
                None
            }
        }

        Self {
            name: name.to_string(),
            #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
            vulkan: init(hal::api::Vulkan, &instance_desc),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            metal: init(hal::api::Metal, &instance_desc),
            #[cfg(all(feature = "dx12", windows))]
            dx12: init(hal::api::Dx12, &instance_desc),
            #[cfg(all(feature = "dx11", windows))]
            dx11: init(hal::api::Dx11, &instance_desc),
            #[cfg(feature = "gles")]
            gl: init(hal::api::Gles, &instance_desc),
            flags: instance_desc.flags,
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
        #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
        destroy(hal::api::Vulkan, &self.vulkan, surface.vulkan);
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        destroy(hal::api::Metal, &self.metal, surface.metal);
        #[cfg(all(feature = "dx12", windows))]
        destroy(hal::api::Dx12, &self.dx12, surface.dx12);
        #[cfg(all(feature = "dx11", windows))]
        destroy(hal::api::Dx11, &self.dx11, surface.dx11);
        #[cfg(feature = "gles")]
        destroy(hal::api::Gles, &self.gl, surface.gl);
    }
}

pub struct Surface {
    pub(crate) presentation: Option<Presentation>,
    #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
    pub vulkan: Option<HalSurface<hal::api::Vulkan>>,
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    pub metal: Option<HalSurface<hal::api::Metal>>,
    #[cfg(all(feature = "dx12", windows))]
    pub dx12: Option<HalSurface<hal::api::Dx12>>,
    #[cfg(all(feature = "dx11", windows))]
    pub dx11: Option<HalSurface<hal::api::Dx11>>,
    #[cfg(feature = "gles")]
    pub gl: Option<HalSurface<hal::api::Gles>>,
}

impl crate::resource::Resource for Surface {
    const TYPE: &'static str = "Surface";

    fn life_guard(&self) -> &LifeGuard {
        unreachable!()
    }

    fn label(&self) -> &str {
        "<Surface>"
    }
}

impl Surface {
    pub fn get_capabilities<A: HalApi>(
        &self,
        adapter: &Adapter<A>,
    ) -> Result<hal::SurfaceCapabilities, GetSurfaceSupportError> {
        let suf = A::get_surface(self).ok_or(GetSurfaceSupportError::Unsupported)?;
        profiling::scope!("surface_capabilities");
        let caps = unsafe {
            adapter
                .raw
                .adapter
                .surface_capabilities(&suf.raw)
                .ok_or(GetSurfaceSupportError::Unsupported)?
        };

        Ok(caps)
    }
}

pub struct Adapter<A: hal::Api> {
    pub(crate) raw: hal::ExposedAdapter<A>,
    life_guard: LifeGuard,
}

impl<A: HalApi> Adapter<A> {
    fn new(mut raw: hal::ExposedAdapter<A>) -> Self {
        // WebGPU requires this offset alignment as lower bound on all adapters.
        const MIN_BUFFER_OFFSET_ALIGNMENT_LOWER_BOUND: u32 = 32;

        let limits = &mut raw.capabilities.limits;

        limits.min_uniform_buffer_offset_alignment = limits
            .min_uniform_buffer_offset_alignment
            .max(MIN_BUFFER_OFFSET_ALIGNMENT_LOWER_BOUND);
        limits.min_storage_buffer_offset_alignment = limits
            .min_storage_buffer_offset_alignment
            .max(MIN_BUFFER_OFFSET_ALIGNMENT_LOWER_BOUND);

        Self {
            raw,
            life_guard: LifeGuard::new("<Adapter>"),
        }
    }

    pub fn is_surface_supported(&self, surface: &Surface) -> bool {
        let suf = A::get_surface(surface);

        // If get_surface returns None, then the API does not advertise support for the surface.
        //
        // This could occur if the user is running their app on Wayland but Vulkan does not support
        // VK_KHR_wayland_surface.
        match suf {
            Some(suf) => unsafe { self.raw.adapter.surface_capabilities(&suf.raw) }.is_some(),
            None => false,
        }
    }

    pub(crate) fn get_texture_format_features(
        &self,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        use hal::TextureFormatCapabilities as Tfc;

        let caps = unsafe { self.raw.adapter.texture_format_capabilities(format) };
        let mut allowed_usages = wgt::TextureUsages::empty();

        allowed_usages.set(wgt::TextureUsages::COPY_SRC, caps.contains(Tfc::COPY_SRC));
        allowed_usages.set(wgt::TextureUsages::COPY_DST, caps.contains(Tfc::COPY_DST));
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
            wgt::TextureFormatFeatureFlags::STORAGE_READ_WRITE,
            caps.contains(Tfc::STORAGE_READ_WRITE),
        );

        flags.set(
            wgt::TextureFormatFeatureFlags::FILTERABLE,
            caps.contains(Tfc::SAMPLED_LINEAR),
        );

        flags.set(
            wgt::TextureFormatFeatureFlags::BLENDABLE,
            caps.contains(Tfc::COLOR_ATTACHMENT_BLEND),
        );

        flags.set(
            wgt::TextureFormatFeatureFlags::MULTISAMPLE_X2,
            caps.contains(Tfc::MULTISAMPLE_X2),
        );
        flags.set(
            wgt::TextureFormatFeatureFlags::MULTISAMPLE_X4,
            caps.contains(Tfc::MULTISAMPLE_X4),
        );
        flags.set(
            wgt::TextureFormatFeatureFlags::MULTISAMPLE_X8,
            caps.contains(Tfc::MULTISAMPLE_X8),
        );
        flags.set(
            wgt::TextureFormatFeatureFlags::MULTISAMPLE_X16,
            caps.contains(Tfc::MULTISAMPLE_X16),
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
        instance_flags: wgt::InstanceFlags,
        trace_path: Option<&std::path::Path>,
    ) -> Result<Device<A>, RequestDeviceError> {
        log::trace!("Adapter::create_device");

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
            instance_flags,
        )
        .or(Err(RequestDeviceError::OutOfMemory))
    }

    fn create_device(
        &self,
        self_id: AdapterId,
        desc: &DeviceDescriptor,
        instance_flags: wgt::InstanceFlags,
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
            log::warn!(
                "Feature MAPPABLE_PRIMARY_BUFFERS enabled on a discrete gpu. \
                        This is a massive performance footgun and likely not what you wanted"
            );
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
                hal::DeviceError::ResourceCreationFailed => RequestDeviceError::Internal,
            },
        )?;

        self.create_device_from_hal(self_id, open, desc, instance_flags, trace_path)
    }
}

impl<A: hal::Api> crate::resource::Resource for Adapter<A> {
    const TYPE: &'static str = "Adapter";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum IsSurfaceSupportedError {
    #[error("Invalid adapter")]
    InvalidAdapter,
    #[error("Invalid surface")]
    InvalidSurface,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum GetSurfaceSupportError {
    #[error("Invalid adapter")]
    InvalidAdapter,
    #[error("Invalid surface")]
    InvalidSurface,
    #[error("Surface is not supported by the adapter")]
    Unsupported,
}

#[derive(Clone, Debug, Error)]
/// Error when requesting a device from the adaptor
#[non_exhaustive]
pub enum RequestDeviceError {
    #[error("Parent adapter is invalid")]
    InvalidAdapter,
    #[error("Connection to device was lost during initialization")]
    DeviceLost,
    #[error("Device initialization failed due to implementation specific errors")]
    Internal,
    #[error(transparent)]
    LimitsExceeded(#[from] FailedLimit),
    #[error("Device has no queue supporting graphics")]
    NoGraphicsQueue,
    #[error("Not enough memory left")]
    OutOfMemory,
    #[error("Unsupported features were requested: {0:?}")]
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
#[error("Adapter is invalid")]
pub struct InvalidAdapter;

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum RequestAdapterError {
    #[error("No suitable adapter found")]
    NotFound,
    #[error("Surface {0:?} is invalid")]
    InvalidSurface(SurfaceId),
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    /// # Safety
    ///
    /// - `display_handle` must be a valid object to create a surface upon.
    /// - `window_handle` must remain valid as long as the returned
    ///   [`SurfaceId`] is being used.
    #[cfg(feature = "raw-window-handle")]
    pub unsafe fn instance_create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("Instance::create_surface");

        fn init<A: hal::Api>(
            inst: &Option<A::Instance>,
            display_handle: raw_window_handle::RawDisplayHandle,
            window_handle: raw_window_handle::RawWindowHandle,
        ) -> Option<HalSurface<A>> {
            inst.as_ref().and_then(|inst| unsafe {
                match inst.create_surface(display_handle, window_handle) {
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
            #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
            vulkan: init::<hal::api::Vulkan>(&self.instance.vulkan, display_handle, window_handle),
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            metal: init::<hal::api::Metal>(&self.instance.metal, display_handle, window_handle),
            #[cfg(all(feature = "dx12", windows))]
            dx12: init::<hal::api::Dx12>(&self.instance.dx12, display_handle, window_handle),
            #[cfg(all(feature = "dx11", windows))]
            dx11: init::<hal::api::Dx11>(&self.instance.dx11, display_handle, window_handle),
            #[cfg(feature = "gles")]
            gl: init::<hal::api::Gles>(&self.instance.gl, display_handle, window_handle),
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }

    /// # Safety
    ///
    /// `layer` must be a valid pointer.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    pub unsafe fn instance_create_surface_metal(
        &self,
        layer: *mut std::ffi::c_void,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("Instance::create_surface_metal");

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
            #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
            vulkan: None,
            #[cfg(feature = "gles")]
            gl: None,
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }

    #[cfg(all(
        target_arch = "wasm32",
        not(target_os = "emscripten"),
        feature = "gles"
    ))]
    pub fn create_surface_webgl_canvas(
        &self,
        canvas: web_sys::HtmlCanvasElement,
        id_in: Input<G, SurfaceId>,
    ) -> Result<SurfaceId, hal::InstanceError> {
        profiling::scope!("Instance::create_surface_webgl_canvas");

        let surface = Surface {
            presentation: None,
            gl: self
                .instance
                .gl
                .as_ref()
                .map(|inst| {
                    Ok(HalSurface {
                        raw: inst.create_surface_from_canvas(canvas)?,
                    })
                })
                .transpose()?,
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        Ok(id.0)
    }

    #[cfg(all(
        target_arch = "wasm32",
        not(target_os = "emscripten"),
        feature = "gles"
    ))]
    pub fn create_surface_webgl_offscreen_canvas(
        &self,
        canvas: web_sys::OffscreenCanvas,
        id_in: Input<G, SurfaceId>,
    ) -> Result<SurfaceId, hal::InstanceError> {
        profiling::scope!("Instance::create_surface_webgl_offscreen_canvas");

        let surface = Surface {
            presentation: None,
            gl: self
                .instance
                .gl
                .as_ref()
                .map(|inst| {
                    Ok(HalSurface {
                        raw: inst.create_surface_from_offscreen_canvas(canvas)?,
                    })
                })
                .transpose()?,
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        Ok(id.0)
    }

    #[cfg(all(feature = "dx12", windows))]
    /// # Safety
    ///
    /// The visual must be valid and able to be used to make a swapchain with.
    pub unsafe fn instance_create_surface_from_visual(
        &self,
        visual: *mut std::ffi::c_void,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("Instance::instance_create_surface_from_visual");

        let surface = Surface {
            presentation: None,
            #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
            vulkan: None,
            #[cfg(all(feature = "dx12", windows))]
            dx12: self.instance.dx12.as_ref().map(|inst| HalSurface {
                raw: unsafe { inst.create_surface_from_visual(visual as _) },
            }),
            #[cfg(all(feature = "dx11", windows))]
            dx11: None,
            #[cfg(feature = "gles")]
            gl: None,
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }

    #[cfg(all(feature = "dx12", windows))]
    /// # Safety
    ///
    /// The surface_handle must be valid and able to be used to make a swapchain with.
    pub unsafe fn instance_create_surface_from_surface_handle(
        &self,
        surface_handle: *mut std::ffi::c_void,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("Instance::instance_create_surface_from_surface_handle");

        let surface = Surface {
            presentation: None,
            #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
            vulkan: None,
            #[cfg(all(feature = "dx12", windows))]
            dx12: self.instance.dx12.as_ref().map(|inst| HalSurface {
                raw: unsafe { inst.create_surface_from_surface_handle(surface_handle) },
            }),
            #[cfg(all(feature = "dx11", windows))]
            dx11: None,
            #[cfg(feature = "gles")]
            gl: None,
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }

    #[cfg(all(feature = "dx12", windows))]
    /// # Safety
    ///
    /// The swap_chain_panel must be valid and able to be used to make a swapchain with.
    pub unsafe fn instance_create_surface_from_swap_chain_panel(
        &self,
        swap_chain_panel: *mut std::ffi::c_void,
        id_in: Input<G, SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("Instance::instance_create_surface_from_swap_chain_panel");

        let surface = Surface {
            presentation: None,
            #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
            vulkan: None,
            #[cfg(all(feature = "dx12", windows))]
            dx12: self.instance.dx12.as_ref().map(|inst| HalSurface {
                raw: unsafe { inst.create_surface_from_swap_chain_panel(swap_chain_panel as _) },
            }),
            #[cfg(all(feature = "dx11", windows))]
            dx11: None,
            #[cfg(feature = "gles")]
            gl: None,
        };

        let mut token = Token::root();
        let id = self.surfaces.prepare(id_in).assign(surface, &mut token);
        id.0
    }

    pub fn surface_drop(&self, id: SurfaceId) {
        profiling::scope!("Surface::drop");
        log::trace!("Surface::drop {id:?}");

        let mut token = Token::root();
        let (surface, _) = self.surfaces.unregister(id, &mut token);
        let mut surface = surface.unwrap();

        fn unconfigure<G: GlobalIdentityHandlerFactory, A: HalApi>(
            global: &Global<G>,
            surface: &mut HalSurface<A>,
            present: &Presentation,
        ) {
            let hub = HalApi::hub(global);
            hub.surface_unconfigure(present.device_id.value, surface);
        }

        if let Some(present) = surface.presentation.take() {
            match present.backend() {
                #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
                Backend::Vulkan => unconfigure(self, surface.vulkan.as_mut().unwrap(), &present),
                #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
                Backend::Metal => unconfigure(self, surface.metal.as_mut().unwrap(), &present),
                #[cfg(all(feature = "dx12", windows))]
                Backend::Dx12 => unconfigure(self, surface.dx12.as_mut().unwrap(), &present),
                #[cfg(all(feature = "dx11", windows))]
                Backend::Dx11 => unconfigure(self, surface.dx11.as_mut().unwrap(), &present),
                #[cfg(feature = "gles")]
                Backend::Gl => unconfigure(self, surface.gl.as_mut().unwrap(), &present),
                _ => unreachable!(),
            }
        }

        self.instance.destroy_surface(surface);
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
        profiling::scope!("Instance::enumerate_adapters");
        log::trace!("Instance::enumerate_adapters");

        let mut adapters = Vec::new();

        #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
        self.enumerate(
            hal::api::Vulkan,
            &self.instance.vulkan,
            &inputs,
            &mut adapters,
        );
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        self.enumerate(
            hal::api::Metal,
            &self.instance.metal,
            &inputs,
            &mut adapters,
        );
        #[cfg(all(feature = "dx12", windows))]
        self.enumerate(hal::api::Dx12, &self.instance.dx12, &inputs, &mut adapters);
        #[cfg(all(feature = "dx11", windows))]
        self.enumerate(hal::api::Dx11, &self.instance.dx11, &inputs, &mut adapters);
        #[cfg(feature = "gles")]
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
        profiling::scope!("Instance::pick_adapter");
        log::trace!("Instance::pick_adapter");

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
                        let surface = &A::get_surface(surface);
                        adapters.retain(|exposed| unsafe {
                            // If the surface does not exist for this backend,
                            // then the surface is not supported.
                            surface.is_some()
                                && exposed
                                    .adapter
                                    .surface_capabilities(&surface.unwrap().raw)
                                    .is_some()
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

        #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
        let (id_vulkan, adapters_vk) = gather(
            hal::api::Vulkan,
            self.instance.vulkan.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        let (id_metal, adapters_metal) = gather(
            hal::api::Metal,
            self.instance.metal.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );
        #[cfg(all(feature = "dx12", windows))]
        let (id_dx12, adapters_dx12) = gather(
            hal::api::Dx12,
            self.instance.dx12.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );
        #[cfg(all(feature = "dx11", windows))]
        let (id_dx11, adapters_dx11) = gather(
            hal::api::Dx11,
            self.instance.dx11.as_ref(),
            &inputs,
            compatible_surface,
            desc.force_fallback_adapter,
            &mut device_types,
        );
        #[cfg(feature = "gles")]
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
            // Since devices of type "Other" might really be "Unknown" and come
            // from APIs like OpenGL that don't specify device type, Prefer more
            // Specific types over Other.
            //
            // This means that backends which do provide accurate device types
            // will be preferred if their device type indicates an actual
            // hardware GPU (integrated or discrete).
            PowerPreference::LowPower => integrated.or(discrete).or(other).or(virt).or(cpu),
            PowerPreference::HighPerformance => discrete.or(integrated).or(other).or(virt).or(cpu),
            PowerPreference::None => {
                let option_min = |a: Option<usize>, b: Option<usize>| {
                    if let (Some(a), Some(b)) = (a, b) {
                        Some(a.min(b))
                    } else {
                        a.or(b)
                    }
                };
                // Pick the lowest id of these types
                option_min(option_min(discrete, integrated), other)
            }
        };

        let mut selected = preferred_gpu.unwrap_or(0);
        #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
        if let Some(id) = self.select(&mut selected, id_vulkan, adapters_vk) {
            return Ok(id);
        }
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        if let Some(id) = self.select(&mut selected, id_metal, adapters_metal) {
            return Ok(id);
        }
        #[cfg(all(feature = "dx12", windows))]
        if let Some(id) = self.select(&mut selected, id_dx12, adapters_dx12) {
            return Ok(id);
        }
        #[cfg(all(feature = "dx11", windows))]
        if let Some(id) = self.select(&mut selected, id_dx11, adapters_dx11) {
            return Ok(id);
        }
        #[cfg(feature = "gles")]
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
        profiling::scope!("Instance::create_adapter_from_hal");

        let mut token = Token::root();
        let fid = A::hub(self).adapters.prepare(input);

        match A::VARIANT {
            #[cfg(all(feature = "vulkan", not(target_arch = "wasm32")))]
            Backend::Vulkan => fid.assign(Adapter::new(hal_adapter), &mut token).0,
            #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
            Backend::Metal => fid.assign(Adapter::new(hal_adapter), &mut token).0,
            #[cfg(all(feature = "dx12", windows))]
            Backend::Dx12 => fid.assign(Adapter::new(hal_adapter), &mut token).0,
            #[cfg(all(feature = "dx11", windows))]
            Backend::Dx11 => fid.assign(Adapter::new(hal_adapter), &mut token).0,
            #[cfg(feature = "gles")]
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

    pub fn adapter_get_presentation_timestamp<A: HalApi>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::PresentationTimestamp, InvalidAdapter> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        let adapter = adapter_guard.get(adapter_id).map_err(|_| InvalidAdapter)?;

        Ok(unsafe { adapter.raw.adapter.get_presentation_timestamp() })
    }

    pub fn adapter_drop<A: HalApi>(&self, adapter_id: AdapterId) {
        profiling::scope!("Adapter::drop");
        log::trace!("Adapter::drop {adapter_id:?}");

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
        profiling::scope!("Adapter::request_device");
        log::trace!("Adapter::request_device");

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
                match adapter.create_device(adapter_id, desc, self.instance.flags, trace_path) {
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
        profiling::scope!("Adapter::create_device_from_hal");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.devices.prepare(id_in);

        let error = loop {
            let (adapter_guard, mut token) = hub.adapters.read(&mut token);
            let adapter = match adapter_guard.get(adapter_id) {
                Ok(adapter) => adapter,
                Err(_) => break RequestDeviceError::InvalidAdapter,
            };
            let device = match adapter.create_device_from_hal(
                adapter_id,
                hal_device,
                desc,
                self.instance.flags,
                trace_path,
            ) {
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
