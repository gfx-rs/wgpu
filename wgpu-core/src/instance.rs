use std::sync::Arc;
use std::{borrow::Cow, collections::HashMap};

use crate::{
    api_log,
    device::{queue::Queue, resource::Device, DeviceDescriptor, DeviceError},
    global::Global,
    hal_api::HalApi,
    id::{markers, AdapterId, DeviceId, QueueId, SurfaceId},
    lock::{rank, Mutex},
    present::Presentation,
    resource::ResourceType,
    resource_log, DOWNLEVEL_WARNING_MESSAGE,
};

use wgt::{Backend, Backends, PowerPreference};

use thiserror::Error;

pub type RequestAdapterOptions = wgt::RequestAdapterOptions<SurfaceId>;

#[derive(Clone, Debug, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[error("Limit '{name}' value {requested} is better than allowed {allowed}")]
pub struct FailedLimit {
    name: Cow<'static, str>,
    requested: u64,
    allowed: u64,
}

fn check_limits(requested: &wgt::Limits, allowed: &wgt::Limits) -> Vec<FailedLimit> {
    let mut failed = Vec::new();

    requested.check_limits_with_fail_fn(allowed, false, |name, requested, allowed| {
        failed.push(FailedLimit {
            name: Cow::Borrowed(name),
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
    /// List of instances per backend.
    ///
    /// The ordering in this list implies prioritization and needs to be preserved.
    pub instance_per_backend: Vec<(Backend, Box<dyn hal::DynInstance>)>,
    pub flags: wgt::InstanceFlags,
}

impl Instance {
    pub fn new(name: &str, instance_desc: wgt::InstanceDescriptor) -> Self {
        fn init<A: HalApi>(
            _: A,
            instance_desc: &wgt::InstanceDescriptor,
            instance_per_backend: &mut Vec<(Backend, Box<dyn hal::DynInstance>)>,
        ) {
            if instance_desc.backends.contains(A::VARIANT.into()) {
                let hal_desc = hal::InstanceDescriptor {
                    name: "wgpu",
                    flags: instance_desc.flags,
                    dx12_shader_compiler: instance_desc.dx12_shader_compiler.clone(),
                    gles_minor_version: instance_desc.gles_minor_version,
                };

                use hal::Instance as _;
                match unsafe { A::Instance::init(&hal_desc) } {
                    Ok(instance) => {
                        log::debug!("Instance::new: created {:?} backend", A::VARIANT);
                        instance_per_backend.push((A::VARIANT, Box::new(instance)));
                    }
                    Err(err) => {
                        log::debug!(
                            "Instance::new: failed to create {:?} backend: {:?}",
                            A::VARIANT,
                            err
                        );
                    }
                }
            } else {
                log::trace!("Instance::new: backend {:?} not requested", A::VARIANT);
            }
        }

        let mut instance_per_backend = Vec::new();

        #[cfg(vulkan)]
        init(hal::api::Vulkan, &instance_desc, &mut instance_per_backend);
        #[cfg(metal)]
        init(hal::api::Metal, &instance_desc, &mut instance_per_backend);
        #[cfg(dx12)]
        init(hal::api::Dx12, &instance_desc, &mut instance_per_backend);
        #[cfg(gles)]
        init(hal::api::Gles, &instance_desc, &mut instance_per_backend);

        Self {
            name: name.to_string(),
            instance_per_backend,
            flags: instance_desc.flags,
        }
    }

    pub fn raw(&self, backend: Backend) -> Option<&dyn hal::DynInstance> {
        self.instance_per_backend
            .iter()
            .find_map(|(instance_backend, instance)| {
                (*instance_backend == backend).then(|| instance.as_ref())
            })
    }
}

pub struct Surface {
    pub(crate) presentation: Mutex<Option<Presentation>>,
    pub surface_per_backend: HashMap<Backend, Box<dyn hal::DynSurface>>,
}

impl ResourceType for Surface {
    const TYPE: &'static str = "Surface";
}
impl crate::storage::StorageItem for Surface {
    type Marker = markers::Surface;
}

impl Surface {
    pub fn get_capabilities(
        &self,
        adapter: &Adapter,
    ) -> Result<hal::SurfaceCapabilities, GetSurfaceSupportError> {
        self.get_capabilities_with_raw(&adapter.raw)
    }

    pub fn get_capabilities_with_raw(
        &self,
        adapter: &hal::DynExposedAdapter,
    ) -> Result<hal::SurfaceCapabilities, GetSurfaceSupportError> {
        let suf = self
            .raw(adapter.backend())
            .ok_or(GetSurfaceSupportError::Unsupported)?;
        profiling::scope!("surface_capabilities");
        let caps = unsafe { adapter.adapter.surface_capabilities(suf) }
            .ok_or(GetSurfaceSupportError::Unsupported)?;

        Ok(caps)
    }

    pub fn raw(&self, backend: Backend) -> Option<&dyn hal::DynSurface> {
        self.surface_per_backend
            .get(&backend)
            .map(|surface| surface.as_ref())
    }
}

pub struct Adapter {
    pub(crate) raw: hal::DynExposedAdapter,
}

impl Adapter {
    fn new(mut raw: hal::DynExposedAdapter) -> Self {
        // WebGPU requires this offset alignment as lower bound on all adapters.
        const MIN_BUFFER_OFFSET_ALIGNMENT_LOWER_BOUND: u32 = 32;

        let limits = &mut raw.capabilities.limits;

        limits.min_uniform_buffer_offset_alignment = limits
            .min_uniform_buffer_offset_alignment
            .max(MIN_BUFFER_OFFSET_ALIGNMENT_LOWER_BOUND);
        limits.min_storage_buffer_offset_alignment = limits
            .min_storage_buffer_offset_alignment
            .max(MIN_BUFFER_OFFSET_ALIGNMENT_LOWER_BOUND);

        Self { raw }
    }

    pub fn is_surface_supported(&self, surface: &Surface) -> bool {
        // If get_capabilities returns Err, then the API does not advertise support for the surface.
        //
        // This could occur if the user is running their app on Wayland but Vulkan does not support
        // VK_KHR_wayland_surface.
        surface.get_capabilities(self).is_ok()
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

    #[allow(clippy::type_complexity)]
    fn create_device_and_queue_from_hal(
        self: &Arc<Self>,
        hal_device: hal::DynOpenDevice,
        desc: &DeviceDescriptor,
        instance_flags: wgt::InstanceFlags,
        trace_path: Option<&std::path::Path>,
    ) -> Result<(Arc<Device>, Arc<Queue>), RequestDeviceError> {
        api_log!("Adapter::create_device");

        let device = Device::new(
            hal_device.device,
            hal_device.queue.as_ref(),
            self,
            desc,
            trace_path,
            instance_flags,
        )?;

        let device = Arc::new(device);
        let queue = Arc::new(Queue::new(device.clone(), hal_device.queue));
        device.set_queue(&queue);
        Ok((device, queue))
    }

    #[allow(clippy::type_complexity)]
    fn create_device_and_queue(
        self: &Arc<Self>,
        desc: &DeviceDescriptor,
        instance_flags: wgt::InstanceFlags,
        trace_path: Option<&std::path::Path>,
    ) -> Result<(Arc<Device>, Arc<Queue>), RequestDeviceError> {
        // Verify all features were exposed by the adapter
        if !self.raw.features.contains(desc.required_features) {
            return Err(RequestDeviceError::UnsupportedFeature(
                desc.required_features - self.raw.features,
            ));
        }

        let caps = &self.raw.capabilities;
        if Backends::PRIMARY.contains(Backends::from(self.raw.backend()))
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

        if let Some(failed) = check_limits(&desc.required_limits, &caps.limits).pop() {
            return Err(RequestDeviceError::LimitsExceeded(failed));
        }

        let open = unsafe {
            self.raw.adapter.open(
                desc.required_features,
                &desc.required_limits,
                &desc.memory_hints,
            )
        }
        .map_err(DeviceError::from_hal)?;

        self.create_device_and_queue_from_hal(open, desc, instance_flags, trace_path)
    }
}

crate::impl_resource_type!(Adapter);
crate::impl_storage_item!(Adapter);

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum GetSurfaceSupportError {
    #[error("Surface is not supported by the adapter")]
    Unsupported,
}

#[derive(Clone, Debug, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
/// Error when requesting a device from the adaptor
#[non_exhaustive]
pub enum RequestDeviceError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    LimitsExceeded(#[from] FailedLimit),
    #[error("Device has no queue supporting graphics")]
    NoGraphicsQueue,
    #[error("Unsupported features were requested: {0:?}")]
    UnsupportedFeature(wgt::Features),
}

#[derive(Clone, Debug, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum RequestAdapterError {
    #[error("No suitable adapter found")]
    NotFound,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateSurfaceError {
    #[error("The backend {0} was not enabled on the instance.")]
    BackendNotEnabled(Backend),
    #[error("Failed to create surface for any enabled backend: {0:?}")]
    FailedToCreateSurfaceForAnyBackend(HashMap<Backend, hal::InstanceError>),
}

impl Global {
    /// Creates a new surface targeting the given display/window handles.
    ///
    /// Internally attempts to create hal surfaces for all enabled backends.
    ///
    /// Fails only if creation for surfaces for all enabled backends fails in which case
    /// the error for each enabled backend is listed.
    /// Vice versa, if creation for any backend succeeds, success is returned.
    /// Surface creation errors are logged to the debug log in any case.
    ///
    /// id_in:
    /// - If `Some`, the id to assign to the surface. A new one will be generated otherwise.
    ///
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

        let mut errors = HashMap::default();
        let mut surface_per_backend = HashMap::default();

        for (backend, instance) in &self.instance.instance_per_backend {
            match unsafe {
                instance
                    .as_ref()
                    .create_surface(display_handle, window_handle)
            } {
                Ok(raw) => {
                    surface_per_backend.insert(*backend, raw);
                }
                Err(err) => {
                    log::debug!(
                        "Instance::create_surface: failed to create surface for {:?}: {:?}",
                        backend,
                        err
                    );
                    errors.insert(*backend, err);
                }
            }
        }

        if surface_per_backend.is_empty() {
            Err(CreateSurfaceError::FailedToCreateSurfaceForAnyBackend(
                errors,
            ))
        } else {
            let surface = Surface {
                presentation: Mutex::new(rank::SURFACE_PRESENTATION, None),
                surface_per_backend,
            };

            let id = self
                .surfaces
                .prepare(id_in) // No specific backend for Surface, since it's not specific.
                .assign(Arc::new(surface));
            Ok(id)
        }
    }

    /// # Safety
    ///
    /// `layer` must be a valid pointer.
    #[cfg(metal)]
    pub unsafe fn instance_create_surface_metal(
        &self,
        layer: *mut std::ffi::c_void,
        id_in: Option<SurfaceId>,
    ) -> Result<SurfaceId, CreateSurfaceError> {
        profiling::scope!("Instance::create_surface_metal");

        let instance = self
            .instance
            .raw(Backend::Metal)
            .ok_or(CreateSurfaceError::BackendNotEnabled(Backend::Metal))?;
        let instance_metal: &hal::metal::Instance = instance.as_any().downcast_ref().unwrap();

        let layer = layer.cast();
        // SAFETY: We do this cast and deref. (rather than using `metal` to get the
        // object we want) to avoid direct coupling on the `metal` crate.
        //
        // To wit, this pointer…
        //
        // - …is properly aligned.
        // - …is dereferenceable to a `MetalLayerRef` as an invariant of the `metal`
        //   field.
        // - …points to an _initialized_ `MetalLayerRef`.
        // - …is only ever aliased via an immutable reference that lives within this
        //   lexical scope.
        let layer = unsafe { &*layer };
        let raw_surface: Box<dyn hal::DynSurface> =
            Box::new(instance_metal.create_surface_from_layer(layer));

        let surface = Surface {
            presentation: Mutex::new(rank::SURFACE_PRESENTATION, None),
            surface_per_backend: std::iter::once((Backend::Metal, raw_surface)).collect(),
        };

        let id = self.surfaces.prepare(id_in).assign(Arc::new(surface));
        Ok(id)
    }

    #[cfg(dx12)]
    fn instance_create_surface_dx12(
        &self,
        id_in: Option<SurfaceId>,
        create_surface_func: impl FnOnce(&hal::dx12::Instance) -> hal::dx12::Surface,
    ) -> Result<SurfaceId, CreateSurfaceError> {
        let instance = self
            .instance
            .raw(Backend::Dx12)
            .ok_or(CreateSurfaceError::BackendNotEnabled(Backend::Dx12))?;
        let instance_dx12 = instance.as_any().downcast_ref().unwrap();
        let surface: Box<dyn hal::DynSurface> = Box::new(create_surface_func(instance_dx12));

        let surface = Surface {
            presentation: Mutex::new(rank::SURFACE_PRESENTATION, None),
            surface_per_backend: std::iter::once((Backend::Dx12, surface)).collect(),
        };

        let id = self.surfaces.prepare(id_in).assign(Arc::new(surface));
        Ok(id)
    }

    #[cfg(dx12)]
    /// # Safety
    ///
    /// The visual must be valid and able to be used to make a swapchain with.
    pub unsafe fn instance_create_surface_from_visual(
        &self,
        visual: *mut std::ffi::c_void,
        id_in: Option<SurfaceId>,
    ) -> Result<SurfaceId, CreateSurfaceError> {
        profiling::scope!("Instance::instance_create_surface_from_visual");
        self.instance_create_surface_dx12(id_in, |inst| unsafe {
            inst.create_surface_from_visual(visual)
        })
    }

    #[cfg(dx12)]
    /// # Safety
    ///
    /// The surface_handle must be valid and able to be used to make a swapchain with.
    pub unsafe fn instance_create_surface_from_surface_handle(
        &self,
        surface_handle: *mut std::ffi::c_void,
        id_in: Option<SurfaceId>,
    ) -> Result<SurfaceId, CreateSurfaceError> {
        profiling::scope!("Instance::instance_create_surface_from_surface_handle");
        self.instance_create_surface_dx12(id_in, |inst| unsafe {
            inst.create_surface_from_surface_handle(surface_handle)
        })
    }

    #[cfg(dx12)]
    /// # Safety
    ///
    /// The swap_chain_panel must be valid and able to be used to make a swapchain with.
    pub unsafe fn instance_create_surface_from_swap_chain_panel(
        &self,
        swap_chain_panel: *mut std::ffi::c_void,
        id_in: Option<SurfaceId>,
    ) -> Result<SurfaceId, CreateSurfaceError> {
        profiling::scope!("Instance::instance_create_surface_from_swap_chain_panel");
        self.instance_create_surface_dx12(id_in, |inst| unsafe {
            inst.create_surface_from_swap_chain_panel(swap_chain_panel)
        })
    }

    pub fn surface_drop(&self, id: SurfaceId) {
        profiling::scope!("Surface::drop");

        api_log!("Surface::drop {id:?}");

        let surface = self.surfaces.remove(id);
        let surface =
            Arc::into_inner(surface).expect("Surface cannot be destroyed because is still in use");

        if let Some(present) = surface.presentation.lock().take() {
            for (&backend, surface) in &surface.surface_per_backend {
                if backend == present.device.backend() {
                    unsafe { surface.unconfigure(present.device.raw()) };
                }
            }
        }
        drop(surface)
    }

    pub fn enumerate_adapters(&self, backends: Backends) -> Vec<AdapterId> {
        profiling::scope!("Instance::enumerate_adapters");
        api_log!("Instance::enumerate_adapters");

        let mut adapters = Vec::new();
        for (_, instance) in self
            .instance
            .instance_per_backend
            .iter()
            .filter(|(backend, _)| backends.contains(Backends::from(*backend)))
        {
            profiling::scope!("enumerating", &*format!("{:?}", backend));

            let hal_adapters = unsafe { instance.enumerate_adapters(None) };
            for raw in hal_adapters {
                let adapter = Adapter::new(raw);
                log::info!("Adapter {:?}", adapter.raw.info);
                let id = self.hub.adapters.prepare(None).assign(Arc::new(adapter));
                adapters.push(id);
            }
        }
        adapters
    }

    pub fn request_adapter(
        &self,
        desc: &RequestAdapterOptions,
        backends: Backends,
        id_in: Option<AdapterId>,
    ) -> Result<AdapterId, RequestAdapterError> {
        profiling::scope!("Instance::request_adapter");
        api_log!("Instance::request_adapter");

        let compatible_surface = desc.compatible_surface.map(|id| self.surfaces.get(id));
        let compatible_surface = compatible_surface.as_ref().map(|surface| surface.as_ref());
        let mut adapters = Vec::new();

        for (backend, instance) in self
            .instance
            .instance_per_backend
            .iter()
            .filter(|(backend, _)| backends.contains(Backends::from(*backend)))
        {
            let compatible_hal_surface =
                compatible_surface.and_then(|surface| surface.raw(*backend));
            let mut backend_adapters =
                unsafe { instance.enumerate_adapters(compatible_hal_surface) };
            if desc.force_fallback_adapter {
                backend_adapters.retain(|exposed| exposed.info.device_type == wgt::DeviceType::Cpu);
            }
            if let Some(surface) = compatible_surface {
                backend_adapters
                    .retain(|exposed| surface.get_capabilities_with_raw(exposed).is_ok());
            }
            adapters.extend(backend_adapters);
        }

        match desc.power_preference {
            PowerPreference::LowPower => {
                sort(&mut adapters, true);
            }
            PowerPreference::HighPerformance => {
                sort(&mut adapters, false);
            }
            PowerPreference::None => {}
        };

        fn sort(adapters: &mut [hal::DynExposedAdapter], prefer_integrated_gpu: bool) {
            adapters.sort_by(|a, b| {
                get_order(a.info.device_type, prefer_integrated_gpu)
                    .cmp(&get_order(b.info.device_type, prefer_integrated_gpu))
            });
        }

        fn get_order(device_type: wgt::DeviceType, prefer_integrated_gpu: bool) -> u8 {
            // Since devices of type "Other" might really be "Unknown" and come
            // from APIs like OpenGL that don't specify device type, Prefer more
            // Specific types over Other.
            //
            // This means that backends which do provide accurate device types
            // will be preferred if their device type indicates an actual
            // hardware GPU (integrated or discrete).
            match device_type {
                wgt::DeviceType::DiscreteGpu if prefer_integrated_gpu => 2,
                wgt::DeviceType::IntegratedGpu if prefer_integrated_gpu => 1,
                wgt::DeviceType::DiscreteGpu => 1,
                wgt::DeviceType::IntegratedGpu => 2,
                wgt::DeviceType::Other => 3,
                wgt::DeviceType::VirtualGpu => 4,
                wgt::DeviceType::Cpu => 5,
            }
        }

        if let Some(adapter) = adapters.into_iter().next() {
            log::info!("Adapter {:?}", adapter.info);
            let id = self
                .hub
                .adapters
                .prepare(id_in)
                .assign(Arc::new(Adapter::new(adapter)));
            Ok(id)
        } else {
            Err(RequestAdapterError::NotFound)
        }
    }

    /// # Safety
    ///
    /// `hal_adapter` must be created from this global internal instance handle.
    pub unsafe fn create_adapter_from_hal(
        &self,
        hal_adapter: hal::DynExposedAdapter,
        input: Option<AdapterId>,
    ) -> AdapterId {
        profiling::scope!("Instance::create_adapter_from_hal");

        let fid = self.hub.adapters.prepare(input);
        let id = fid.assign(Arc::new(Adapter::new(hal_adapter)));

        resource_log!("Created Adapter {:?}", id);
        id
    }

    pub fn adapter_get_info(&self, adapter_id: AdapterId) -> wgt::AdapterInfo {
        let adapter = self.hub.adapters.get(adapter_id);
        adapter.raw.info.clone()
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
        adapter.raw.features
    }

    pub fn adapter_limits(&self, adapter_id: AdapterId) -> wgt::Limits {
        let adapter = self.hub.adapters.get(adapter_id);
        adapter.raw.capabilities.limits.clone()
    }

    pub fn adapter_downlevel_capabilities(
        &self,
        adapter_id: AdapterId,
    ) -> wgt::DownlevelCapabilities {
        let adapter = self.hub.adapters.get(adapter_id);
        adapter.raw.capabilities.downlevel.clone()
    }

    pub fn adapter_get_presentation_timestamp(
        &self,
        adapter_id: AdapterId,
    ) -> wgt::PresentationTimestamp {
        let adapter = self.hub.adapters.get(adapter_id);
        unsafe { adapter.raw.adapter.get_presentation_timestamp() }
    }

    pub fn adapter_drop(&self, adapter_id: AdapterId) {
        profiling::scope!("Adapter::drop");
        api_log!("Adapter::drop {adapter_id:?}");

        self.hub.adapters.remove(adapter_id);
    }
}

impl Global {
    pub fn adapter_request_device(
        &self,
        adapter_id: AdapterId,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
        device_id_in: Option<DeviceId>,
        queue_id_in: Option<QueueId>,
    ) -> Result<(DeviceId, QueueId), RequestDeviceError> {
        profiling::scope!("Adapter::request_device");
        api_log!("Adapter::request_device");

        let device_fid = self.hub.devices.prepare(device_id_in);
        let queue_fid = self.hub.queues.prepare(queue_id_in);

        let adapter = self.hub.adapters.get(adapter_id);
        let (device, queue) =
            adapter.create_device_and_queue(desc, self.instance.flags, trace_path)?;

        let device_id = device_fid.assign(device);
        resource_log!("Created Device {:?}", device_id);

        let queue_id = queue_fid.assign(queue);
        resource_log!("Created Queue {:?}", queue_id);

        Ok((device_id, queue_id))
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
        trace_path: Option<&std::path::Path>,
        device_id_in: Option<DeviceId>,
        queue_id_in: Option<QueueId>,
    ) -> Result<(DeviceId, QueueId), RequestDeviceError> {
        profiling::scope!("Global::create_device_from_hal");

        let devices_fid = self.hub.devices.prepare(device_id_in);
        let queues_fid = self.hub.queues.prepare(queue_id_in);

        let adapter = self.hub.adapters.get(adapter_id);
        let (device, queue) = adapter.create_device_and_queue_from_hal(
            hal_device,
            desc,
            self.instance.flags,
            trace_path,
        )?;

        let device_id = devices_fid.assign(device);
        resource_log!("Created Device {:?}", device_id);

        let queue_id = queues_fid.assign(queue);
        resource_log!("Created Queue {:?}", queue_id);

        Ok((device_id, queue_id))
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
