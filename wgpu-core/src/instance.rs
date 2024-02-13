use std::sync::Arc;

use crate::{
    any_surface::AnySurface,
    api_log,
    device::{queue::Queue, resource::Device, DeviceDescriptor},
    global::Global,
    hal_api::HalApi,
    id::markers,
    id::{AdapterId, DeviceId, Id, Marker, QueueId, SurfaceId},
    present::Presentation,
    resource::{Resource, ResourceInfo, ResourceType},
    resource_log, LabelHelpers, DOWNLEVEL_WARNING_MESSAGE,
};

use parking_lot::Mutex;
use wgt::{Backend, Backends, PowerPreference};

use hal::{Adapter as _, Instance as _, OpenDevice};
use thiserror::Error;

pub type RequestAdapterOptions = wgt::RequestAdapterOptions<SurfaceId>;
type HalInstance<A> = <A as hal::Api>::Instance;

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
    #[cfg(gles)]
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
            #[cfg(vulkan)]
            vulkan: init(hal::api::Vulkan, &instance_desc),
            #[cfg(metal)]
            metal: init(hal::api::Metal, &instance_desc),
            #[cfg(dx12)]
            dx12: init(hal::api::Dx12, &instance_desc),
            #[cfg(gles)]
            gl: init(hal::api::Gles, &instance_desc),
            flags: instance_desc.flags,
        }
    }

    pub(crate) fn destroy_surface(&self, surface: Surface) {
        fn destroy<A: HalApi>(instance: &Option<A::Instance>, surface: AnySurface) {
            unsafe {
                if let Some(suf) = surface.take::<A>() {
                    instance.as_ref().unwrap().destroy_surface(suf);
                }
            }
        }
        match surface.raw.backend() {
            #[cfg(vulkan)]
            Backend::Vulkan => destroy::<hal::api::Vulkan>(&self.vulkan, surface.raw),
            #[cfg(metal)]
            Backend::Metal => destroy::<hal::api::Metal>(&self.metal, surface.raw),
            #[cfg(dx12)]
            Backend::Dx12 => destroy::<hal::api::Dx12>(&self.dx12, surface.raw),
            #[cfg(gles)]
            Backend::Gl => destroy::<hal::api::Gles>(&self.gl, surface.raw),
            _ => unreachable!(),
        }
    }
}

pub struct Surface {
    pub(crate) presentation: Mutex<Option<Presentation>>,
    pub(crate) info: ResourceInfo<Surface>,
    pub(crate) raw: AnySurface,
}

impl Resource for Surface {
    const TYPE: ResourceType = "Surface";

    type Marker = markers::Surface;

    fn as_info(&self) -> &ResourceInfo<Self> {
        &self.info
    }

    fn as_info_mut(&mut self) -> &mut ResourceInfo<Self> {
        &mut self.info
    }

    fn label(&self) -> String {
        String::from("<Surface>")
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
                .surface_capabilities(suf)
                .ok_or(GetSurfaceSupportError::Unsupported)?
        };

        Ok(caps)
    }
}

pub struct Adapter<A: HalApi> {
    pub(crate) raw: hal::ExposedAdapter<A>,
    pub(crate) info: ResourceInfo<Adapter<A>>,
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
            info: ResourceInfo::new("<Adapter>"),
        }
    }

    pub fn is_surface_supported(&self, surface: &Surface) -> bool {
        let suf = A::get_surface(surface);

        // If get_surface returns None, then the API does not advertise support for the surface.
        //
        // This could occur if the user is running their app on Wayland but Vulkan does not support
        // VK_KHR_wayland_surface.
        match suf {
            Some(suf) => unsafe { self.raw.adapter.surface_capabilities(suf) }.is_some(),
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

    fn create_device_and_queue_from_hal(
        self: &Arc<Self>,
        hal_device: OpenDevice<A>,
        desc: &DeviceDescriptor,
        instance_flags: wgt::InstanceFlags,
        trace_path: Option<&std::path::Path>,
    ) -> Result<(Device<A>, Queue<A>), RequestDeviceError> {
        api_log!("Adapter::create_device");

        if let Ok(device) = Device::new(
            hal_device.device,
            &hal_device.queue,
            self,
            desc,
            trace_path,
            instance_flags,
        ) {
            let queue = Queue {
                device: None,
                raw: Some(hal_device.queue),
                info: ResourceInfo::new("<Queue>"),
            };
            return Ok((device, queue));
        }
        Err(RequestDeviceError::OutOfMemory)
    }

    fn create_device_and_queue(
        self: &Arc<Self>,
        desc: &DeviceDescriptor,
        instance_flags: wgt::InstanceFlags,
        trace_path: Option<&std::path::Path>,
    ) -> Result<(Device<A>, Queue<A>), RequestDeviceError> {
        // Verify all features were exposed by the adapter
        if !self.raw.features.contains(desc.required_features) {
            return Err(RequestDeviceError::UnsupportedFeature(
                desc.required_features - self.raw.features,
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
            log::warn!("{:#?}", caps.downlevel);
        }

        // Verify feature preconditions
        if desc
            .required_features
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

        if let Some(failed) = check_limits(&desc.required_limits, &caps.limits).pop() {
            return Err(RequestDeviceError::LimitsExceeded(failed));
        }

        let open = unsafe {
            self.raw
                .adapter
                .open(desc.required_features, &desc.required_limits)
        }
        .map_err(|err| match err {
            hal::DeviceError::Lost => RequestDeviceError::DeviceLost,
            hal::DeviceError::OutOfMemory => RequestDeviceError::OutOfMemory,
            hal::DeviceError::ResourceCreationFailed => RequestDeviceError::Internal,
        })?;

        self.create_device_and_queue_from_hal(open, desc, instance_flags, trace_path)
    }
}

impl<A: HalApi> Resource for Adapter<A> {
    const TYPE: ResourceType = "Adapter";

    type Marker = markers::Adapter;

    fn as_info(&self) -> &ResourceInfo<Self> {
        &self.info
    }

    fn as_info_mut(&mut self) -> &mut ResourceInfo<Self> {
        &mut self.info
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
    #[error("Not enough memory left to request device")]
    OutOfMemory,
    #[error("Unsupported features were requested: {0:?}")]
    UnsupportedFeature(wgt::Features),
}

pub enum AdapterInputs<'a, M: Marker> {
    IdSet(&'a [Id<M>]),
    Mask(Backends, fn(Backend) -> Option<Id<M>>),
}

impl<M: Marker> AdapterInputs<'_, M> {
    fn find(&self, b: Backend) -> Option<Option<Id<M>>> {
        match *self {
            Self::IdSet(ids) => Some(Some(ids.iter().find(|id| id.backend() == b).copied()?)),
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

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateSurfaceError {
    #[error("No backend is available")]
    NoSupportedBackend,
    #[error(transparent)]
    InstanceError(#[from] hal::InstanceError),
}

impl Global {
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
        id_in: Option<SurfaceId>,
    ) -> Result<SurfaceId, CreateSurfaceError> {
        profiling::scope!("Instance::create_surface");

        fn init<A: HalApi>(
            inst: &Option<A::Instance>,
            display_handle: raw_window_handle::RawDisplayHandle,
            window_handle: raw_window_handle::RawWindowHandle,
        ) -> Option<Result<AnySurface, hal::InstanceError>> {
            inst.as_ref().map(|inst| unsafe {
                match inst.create_surface(display_handle, window_handle) {
                    Ok(raw) => Ok(AnySurface::new::<A>(raw)),
                    Err(e) => Err(e),
                }
            })
        }

        let mut hal_surface: Option<Result<AnySurface, hal::InstanceError>> = None;

        #[cfg(vulkan)]
        if hal_surface.is_none() {
            hal_surface =
                init::<hal::api::Vulkan>(&self.instance.vulkan, display_handle, window_handle);
        }
        #[cfg(metal)]
        if hal_surface.is_none() {
            hal_surface =
                init::<hal::api::Metal>(&self.instance.metal, display_handle, window_handle);
        }
        #[cfg(dx12)]
        if hal_surface.is_none() {
            hal_surface =
                init::<hal::api::Dx12>(&self.instance.dx12, display_handle, window_handle);
        }
        #[cfg(gles)]
        if hal_surface.is_none() {
            hal_surface = init::<hal::api::Gles>(&self.instance.gl, display_handle, window_handle);
        }

        let hal_surface = hal_surface.ok_or(CreateSurfaceError::NoSupportedBackend)??;

        let surface = Surface {
            presentation: Mutex::new(None),
            info: ResourceInfo::new("<Surface>"),
            raw: hal_surface,
        };

        let (id, _) = self.surfaces.prepare(id_in).assign(surface);
        Ok(id)
    }

    /// # Safety
    ///
    /// `layer` must be a valid pointer.
    #[cfg(metal)]
    pub unsafe fn instance_create_surface_metal(
        &self,
        layer: *mut std::ffi::c_void,
        id_in: Option<SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("Instance::create_surface_metal");

        let surface = Surface {
            presentation: Mutex::new(None),
            info: ResourceInfo::new("<Surface>"),
            raw: {
                let hal_surface = self
                    .instance
                    .metal
                    .as_ref()
                    .map(|inst| {
                        // we don't want to link to metal-rs for this
                        #[allow(clippy::transmute_ptr_to_ref)]
                        inst.create_surface_from_layer(unsafe { std::mem::transmute(layer) })
                    })
                    .unwrap();
                AnySurface::new::<hal::api::Metal>(hal_surface)
            },
        };

        let (id, _) = self.surfaces.prepare(id_in).assign(surface);
        id
    }

    #[cfg(dx12)]
    /// # Safety
    ///
    /// The visual must be valid and able to be used to make a swapchain with.
    pub unsafe fn instance_create_surface_from_visual(
        &self,
        visual: *mut std::ffi::c_void,
        id_in: Option<SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("Instance::instance_create_surface_from_visual");

        let surface = Surface {
            presentation: Mutex::new(None),
            info: ResourceInfo::new("<Surface>"),
            raw: {
                let hal_surface = self
                    .instance
                    .dx12
                    .as_ref()
                    .map(|inst| unsafe { inst.create_surface_from_visual(visual as _) })
                    .unwrap();
                AnySurface::new::<hal::api::Dx12>(hal_surface)
            },
        };

        let (id, _) = self.surfaces.prepare(id_in).assign(surface);
        id
    }

    #[cfg(dx12)]
    /// # Safety
    ///
    /// The surface_handle must be valid and able to be used to make a swapchain with.
    pub unsafe fn instance_create_surface_from_surface_handle(
        &self,
        surface_handle: *mut std::ffi::c_void,
        id_in: Option<SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("Instance::instance_create_surface_from_surface_handle");

        let surface = Surface {
            presentation: Mutex::new(None),
            info: ResourceInfo::new("<Surface>"),
            raw: {
                let hal_surface = self
                    .instance
                    .dx12
                    .as_ref()
                    .map(|inst| unsafe { inst.create_surface_from_surface_handle(surface_handle) })
                    .unwrap();
                AnySurface::new::<hal::api::Dx12>(hal_surface)
            },
        };

        let (id, _) = self.surfaces.prepare(id_in).assign(surface);
        id
    }

    #[cfg(dx12)]
    /// # Safety
    ///
    /// The swap_chain_panel must be valid and able to be used to make a swapchain with.
    pub unsafe fn instance_create_surface_from_swap_chain_panel(
        &self,
        swap_chain_panel: *mut std::ffi::c_void,
        id_in: Option<SurfaceId>,
    ) -> SurfaceId {
        profiling::scope!("Instance::instance_create_surface_from_swap_chain_panel");

        let surface = Surface {
            presentation: Mutex::new(None),
            info: ResourceInfo::new("<Surface>"),
            raw: {
                let hal_surface = self
                    .instance
                    .dx12
                    .as_ref()
                    .map(|inst| unsafe {
                        inst.create_surface_from_swap_chain_panel(swap_chain_panel as _)
                    })
                    .unwrap();
                AnySurface::new::<hal::api::Dx12>(hal_surface)
            },
        };

        let (id, _) = self.surfaces.prepare(id_in).assign(surface);
        id
    }

    pub fn surface_drop(&self, id: SurfaceId) {
        profiling::scope!("Surface::drop");

        api_log!("Surface::drop {id:?}");

        fn unconfigure<A: HalApi>(global: &Global, surface: &AnySurface, present: &Presentation) {
            let hub = HalApi::hub(global);
            if let Some(hal_surface) = surface.downcast_ref::<A>() {
                if let Some(device) = present.device.downcast_ref::<A>() {
                    hub.surface_unconfigure(device, hal_surface);
                }
            }
        }

        let surface = self.surfaces.unregister(id);
        if let Some(surface) = Arc::into_inner(surface.unwrap()) {
            if let Some(present) = surface.presentation.lock().take() {
                #[cfg(vulkan)]
                unconfigure::<hal::api::Vulkan>(self, &surface.raw, &present);
                #[cfg(metal)]
                unconfigure::<hal::api::Metal>(self, &surface.raw, &present);
                #[cfg(dx12)]
                unconfigure::<hal::api::Dx12>(self, &surface.raw, &present);
                #[cfg(gles)]
                unconfigure::<hal::api::Gles>(self, &surface.raw, &present);
            }

            self.instance.destroy_surface(surface);
        } else {
            panic!("Surface cannot be destroyed because is still in use");
        }
    }

    fn enumerate<A: HalApi>(
        &self,
        _: A,
        instance: &Option<A::Instance>,
        inputs: &AdapterInputs<markers::Adapter>,
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

        let hal_adapters = unsafe { inst.enumerate_adapters() };
        for raw in hal_adapters {
            let adapter = Adapter::new(raw);
            log::info!("Adapter {:?} {:?}", A::VARIANT, adapter.raw.info);
            let (id, _) = hub.adapters.prepare(id_backend).assign(adapter);
            list.push(id);
        }
    }

    pub fn enumerate_adapters(&self, inputs: AdapterInputs<markers::Adapter>) -> Vec<AdapterId> {
        profiling::scope!("Instance::enumerate_adapters");
        api_log!("Instance::enumerate_adapters");

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
        #[cfg(gles)]
        self.enumerate(hal::api::Gles, &self.instance.gl, &inputs, &mut adapters);

        adapters
    }

    fn select<A: HalApi>(
        &self,
        selected: &mut usize,
        new_id: Option<AdapterId>,
        mut list: Vec<hal::ExposedAdapter<A>>,
    ) -> Option<AdapterId> {
        match selected.checked_sub(list.len()) {
            Some(left) => {
                *selected = left;
                None
            }
            None => {
                let adapter = Adapter::new(list.swap_remove(*selected));
                log::info!("Adapter {:?} {:?}", A::VARIANT, adapter.raw.info);
                let (id, _) = HalApi::hub(self).adapters.prepare(new_id).assign(adapter);
                Some(id)
            }
        }
    }

    pub fn request_adapter(
        &self,
        desc: &RequestAdapterOptions,
        inputs: AdapterInputs<markers::Adapter>,
    ) -> Result<AdapterId, RequestAdapterError> {
        profiling::scope!("Instance::request_adapter");
        api_log!("Instance::request_adapter");

        fn gather<A: HalApi>(
            _: A,
            instance: Option<&A::Instance>,
            inputs: &AdapterInputs<markers::Adapter>,
            compatible_surface: Option<&Surface>,
            force_software: bool,
            device_types: &mut Vec<wgt::DeviceType>,
        ) -> (Option<Id<markers::Adapter>>, Vec<hal::ExposedAdapter<A>>) {
            let id = inputs.find(A::VARIANT);
            match (id, instance) {
                (Some(id), Some(inst)) => {
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
                                    .surface_capabilities(surface.unwrap())
                                    .is_some()
                        });
                    }
                    device_types.extend(adapters.iter().map(|ad| ad.info.device_type));
                    (id, adapters)
                }
                _ => (None, Vec::new()),
            }
        }

        let compatible_surface = desc
            .compatible_surface
            .map(|id| {
                self.surfaces
                    .get(id)
                    .map_err(|_| RequestAdapterError::InvalidSurface(id))
            })
            .transpose()?;
        let compatible_surface = compatible_surface.as_ref().map(|surface| surface.as_ref());
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
        #[cfg(gles)]
        let (id_gl, adapters_gl) = gather(
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
        #[cfg(gles)]
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
        input: Option<AdapterId>,
    ) -> AdapterId {
        profiling::scope!("Instance::create_adapter_from_hal");

        let fid = A::hub(self).adapters.prepare(input);

        let (id, _adapter): (_, Arc<Adapter<A>>) = match A::VARIANT {
            #[cfg(vulkan)]
            Backend::Vulkan => fid.assign(Adapter::new(hal_adapter)),
            #[cfg(metal)]
            Backend::Metal => fid.assign(Adapter::new(hal_adapter)),
            #[cfg(dx12)]
            Backend::Dx12 => fid.assign(Adapter::new(hal_adapter)),
            #[cfg(gles)]
            Backend::Gl => fid.assign(Adapter::new(hal_adapter)),
            _ => unreachable!(),
        };
        resource_log!("Created Adapter {:?}", id);
        id
    }

    pub fn adapter_get_info<A: HalApi>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::AdapterInfo, InvalidAdapter> {
        let hub = A::hub(self);

        hub.adapters
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

        hub.adapters
            .get(adapter_id)
            .map(|adapter| adapter.get_texture_format_features(format))
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_features<A: HalApi>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::Features, InvalidAdapter> {
        let hub = A::hub(self);

        hub.adapters
            .get(adapter_id)
            .map(|adapter| adapter.raw.features)
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_limits<A: HalApi>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::Limits, InvalidAdapter> {
        let hub = A::hub(self);

        hub.adapters
            .get(adapter_id)
            .map(|adapter| adapter.raw.capabilities.limits.clone())
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_downlevel_capabilities<A: HalApi>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::DownlevelCapabilities, InvalidAdapter> {
        let hub = A::hub(self);

        hub.adapters
            .get(adapter_id)
            .map(|adapter| adapter.raw.capabilities.downlevel.clone())
            .map_err(|_| InvalidAdapter)
    }

    pub fn adapter_get_presentation_timestamp<A: HalApi>(
        &self,
        adapter_id: AdapterId,
    ) -> Result<wgt::PresentationTimestamp, InvalidAdapter> {
        let hub = A::hub(self);

        let adapter = hub.adapters.get(adapter_id).map_err(|_| InvalidAdapter)?;

        Ok(unsafe { adapter.raw.adapter.get_presentation_timestamp() })
    }

    pub fn adapter_drop<A: HalApi>(&self, adapter_id: AdapterId) {
        profiling::scope!("Adapter::drop");
        api_log!("Adapter::drop {adapter_id:?}");

        let hub = A::hub(self);
        let mut adapters_locked = hub.adapters.write();

        let free = match adapters_locked.get(adapter_id) {
            Ok(adapter) => Arc::strong_count(adapter) == 1,
            Err(_) => true,
        };
        if free {
            hub.adapters
                .unregister_locked(adapter_id, &mut *adapters_locked);
        }
    }
}

impl Global {
    pub fn adapter_request_device<A: HalApi>(
        &self,
        adapter_id: AdapterId,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
        device_id_in: Option<DeviceId>,
        queue_id_in: Option<QueueId>,
    ) -> (DeviceId, QueueId, Option<RequestDeviceError>) {
        profiling::scope!("Adapter::request_device");
        api_log!("Adapter::request_device");

        let hub = A::hub(self);
        let device_fid = hub.devices.prepare(device_id_in);
        let queue_fid = hub.queues.prepare(queue_id_in);

        let error = loop {
            let adapter = match hub.adapters.get(adapter_id) {
                Ok(adapter) => adapter,
                Err(_) => break RequestDeviceError::InvalidAdapter,
            };
            let (device, mut queue) =
                match adapter.create_device_and_queue(desc, self.instance.flags, trace_path) {
                    Ok((device, queue)) => (device, queue),
                    Err(e) => break e,
                };
            let (device_id, _) = device_fid.assign(device);
            resource_log!("Created Device {:?}", device_id);

            let device = hub.devices.get(device_id).unwrap();
            queue.device = Some(device.clone());

            let (queue_id, _) = queue_fid.assign(queue);
            resource_log!("Created Queue {:?}", queue_id);

            device.queue_id.write().replace(queue_id);

            return (device_id, queue_id, None);
        };

        let device_id = device_fid.assign_error(desc.label.borrow_or_default());
        let queue_id = queue_fid.assign_error(desc.label.borrow_or_default());
        (device_id, queue_id, Some(error))
    }

    /// # Safety
    ///
    /// - `hal_device` must be created from `adapter_id` or its internal handle.
    /// - `desc` must be a subset of `hal_device` features and limits.
    pub unsafe fn create_device_from_hal<A: HalApi>(
        &self,
        adapter_id: AdapterId,
        hal_device: OpenDevice<A>,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
        device_id_in: Option<DeviceId>,
        queue_id_in: Option<QueueId>,
    ) -> (DeviceId, QueueId, Option<RequestDeviceError>) {
        profiling::scope!("Global::create_device_from_hal");

        let hub = A::hub(self);
        let devices_fid = hub.devices.prepare(device_id_in);
        let queues_fid = hub.queues.prepare(queue_id_in);

        let error = loop {
            let adapter = match hub.adapters.get(adapter_id) {
                Ok(adapter) => adapter,
                Err(_) => break RequestDeviceError::InvalidAdapter,
            };
            let (device, mut queue) = match adapter.create_device_and_queue_from_hal(
                hal_device,
                desc,
                self.instance.flags,
                trace_path,
            ) {
                Ok(device) => device,
                Err(e) => break e,
            };
            let (device_id, _) = devices_fid.assign(device);
            resource_log!("Created Device {:?}", device_id);

            let device = hub.devices.get(device_id).unwrap();
            queue.device = Some(device.clone());

            let (queue_id, _) = queues_fid.assign(queue);
            resource_log!("Created Queue {:?}", queue_id);

            device.queue_id.write().replace(queue_id);

            return (device_id, queue_id, None);
        };

        let device_id = devices_fid.assign_error(desc.label.borrow_or_default());
        let queue_id = queues_fid.assign_error(desc.label.borrow_or_default());
        (device_id, queue_id, Some(error))
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
/// - metal  = "metal" or "mtl"
/// - gles   = "opengl" or "gles" or "gl"
/// - webgpu = "webgpu"
pub fn parse_backends_from_comma_list(string: &str) -> Backends {
    let mut backends = Backends::empty();
    for backend in string.to_lowercase().split(',') {
        backends |= match backend.trim() {
            "vulkan" | "vk" => Backends::VULKAN,
            "dx12" | "d3d12" => Backends::DX12,
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
