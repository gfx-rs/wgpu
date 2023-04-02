use crate::{
    context::{ObjectId, Unused},
    AdapterInfo, BindGroupDescriptor, BindGroupLayoutDescriptor, BindingResource, BufferBinding,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    DownlevelCapabilities, Features, Label, Limits, LoadOp, MapMode, Operations,
    PipelineLayoutDescriptor, RenderBundleEncoderDescriptor, RenderPipelineDescriptor,
    SamplerDescriptor, ShaderModuleDescriptor, ShaderModuleDescriptorSpirV, ShaderSource,
    SurfaceStatus, TextureDescriptor, TextureViewDescriptor, UncapturedErrorHandler,
};

use arrayvec::ArrayVec;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    any::Any,
    borrow::Cow::{Borrowed, Owned},
    error::Error,
    fmt,
    future::{ready, Ready},
    ops::Range,
    slice,
    sync::Arc,
};
use wgc::command::{bundle_ffi::*, compute_ffi::*, render_ffi::*};
use wgc::id::TypedId;

const LABEL: &str = "label";

pub struct Context(wgc::hub::Global<wgc::hub::IdentityManagerFactory>);

impl Drop for Context {
    fn drop(&mut self) {
        //nothing
    }
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context").field("type", &"Native").finish()
    }
}

impl Context {
    pub unsafe fn from_hal_instance<A: wgc::hub::HalApi>(hal_instance: A::Instance) -> Self {
        Self(unsafe {
            wgc::hub::Global::from_hal_instance::<A>(
                "wgpu",
                wgc::hub::IdentityManagerFactory,
                hal_instance,
            )
        })
    }

    /// # Safety
    ///
    /// - The raw instance handle returned must not be manually destroyed.
    pub unsafe fn instance_as_hal<A: wgc::hub::HalApi>(&self) -> Option<&A::Instance> {
        unsafe { self.0.instance_as_hal::<A>() }
    }

    pub unsafe fn from_core_instance(core_instance: wgc::instance::Instance) -> Self {
        Self(unsafe {
            wgc::hub::Global::from_instance(wgc::hub::IdentityManagerFactory, core_instance)
        })
    }

    pub(crate) fn global(&self) -> &wgc::hub::Global<wgc::hub::IdentityManagerFactory> {
        &self.0
    }

    pub fn enumerate_adapters(&self, backends: wgt::Backends) -> Vec<wgc::id::AdapterId> {
        self.0
            .enumerate_adapters(wgc::instance::AdapterInputs::Mask(backends, |_| ()))
    }

    pub unsafe fn create_adapter_from_hal<A: wgc::hub::HalApi>(
        &self,
        hal_adapter: hal::ExposedAdapter<A>,
    ) -> wgc::id::AdapterId {
        unsafe { self.0.create_adapter_from_hal(hal_adapter, ()) }
    }

    pub unsafe fn adapter_as_hal<A: wgc::hub::HalApi, F: FnOnce(Option<&A::Adapter>) -> R, R>(
        &self,
        adapter: wgc::id::AdapterId,
        hal_adapter_callback: F,
    ) -> R {
        unsafe {
            self.0
                .adapter_as_hal::<A, F, R>(adapter, hal_adapter_callback)
        }
    }

    pub unsafe fn create_device_from_hal<A: wgc::hub::HalApi>(
        &self,
        adapter: &wgc::id::AdapterId,
        hal_device: hal::OpenDevice<A>,
        desc: &crate::DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Result<(Device, Queue), crate::RequestDeviceError> {
        let global = &self.0;
        let (device_id, error) = unsafe {
            global.create_device_from_hal(
                *adapter,
                hal_device,
                &desc.map_label(|l| l.map(Borrowed)),
                trace_dir,
                (),
            )
        };
        if let Some(err) = error {
            self.handle_error_fatal(err, "Adapter::create_device_from_hal");
        }
        let error_sink = Arc::new(Mutex::new(ErrorSinkRaw::new()));
        let device = Device {
            id: device_id,
            error_sink: error_sink.clone(),
            features: desc.features,
        };
        let queue = Queue {
            id: device_id,
            error_sink,
        };
        Ok((device, queue))
    }

    pub unsafe fn create_texture_from_hal<A: wgc::hub::HalApi>(
        &self,
        hal_texture: A::Texture,
        device: &Device,
        desc: &TextureDescriptor,
    ) -> Texture {
        let descriptor = desc.map_label_and_view_formats(|l| l.map(Borrowed), |v| v.to_vec());
        let global = &self.0;
        let (id, error) =
            unsafe { global.create_texture_from_hal::<A>(hal_texture, device.id, &descriptor, ()) };
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_texture_from_hal",
            );
        }
        Texture {
            id,
            error_sink: Arc::clone(&device.error_sink),
        }
    }

    pub unsafe fn device_as_hal<A: wgc::hub::HalApi, F: FnOnce(Option<&A::Device>) -> R, R>(
        &self,
        device: &Device,
        hal_device_callback: F,
    ) -> R {
        unsafe {
            self.0
                .device_as_hal::<A, F, R>(device.id, hal_device_callback)
        }
    }

    pub unsafe fn surface_as_hal_mut<
        A: wgc::hub::HalApi,
        F: FnOnce(Option<&mut A::Surface>) -> R,
        R,
    >(
        &self,
        surface: &Surface,
        hal_surface_callback: F,
    ) -> R {
        unsafe {
            self.0
                .surface_as_hal_mut::<A, F, R>(surface.id, hal_surface_callback)
        }
    }

    pub unsafe fn texture_as_hal<A: wgc::hub::HalApi, F: FnOnce(Option<&A::Texture>)>(
        &self,
        texture: &Texture,
        hal_texture_callback: F,
    ) {
        unsafe {
            self.0
                .texture_as_hal::<A, F>(texture.id, hal_texture_callback)
        }
    }

    pub fn generate_report(&self) -> wgc::hub::GlobalReport {
        self.0.generate_report()
    }

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub unsafe fn create_surface_from_core_animation_layer(
        &self,
        layer: *mut std::ffi::c_void,
    ) -> Surface {
        let id = unsafe { self.0.instance_create_surface_metal(layer, ()) };
        Surface {
            id,
            configured_device: Mutex::default(),
        }
    }

    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    pub fn instance_create_surface_from_canvas(
        &self,
        canvas: web_sys::HtmlCanvasElement,
    ) -> Result<Surface, crate::CreateSurfaceError> {
        let id = self
            .0
            .create_surface_webgl_canvas(canvas, ())
            .map_err(|hal::InstanceError| crate::CreateSurfaceError {})?;
        Ok(Surface {
            id,
            configured_device: Mutex::default(),
        })
    }

    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    pub fn instance_create_surface_from_offscreen_canvas(
        &self,
        canvas: web_sys::OffscreenCanvas,
    ) -> Result<Surface, crate::CreateSurfaceError> {
        let id = self
            .0
            .create_surface_webgl_offscreen_canvas(canvas, ())
            .map_err(|hal::InstanceError| crate::CreateSurfaceError {})?;
        Ok(Surface {
            id,
            configured_device: Mutex::default(),
        })
    }

    #[cfg(target_os = "windows")]
    pub unsafe fn create_surface_from_visual(&self, visual: *mut std::ffi::c_void) -> Surface {
        let id = unsafe { self.0.instance_create_surface_from_visual(visual, ()) };
        Surface {
            id,
            configured_device: Mutex::default(),
        }
    }

    #[cfg(target_os = "windows")]
    pub unsafe fn create_surface_from_surface_handle(
        &self,
        surface_handle: *mut std::ffi::c_void,
    ) -> Surface {
        let id = unsafe {
            self.0
                .instance_create_surface_from_surface_handle(surface_handle, ())
        };
        Surface {
            id,
            configured_device: Mutex::default(),
        }
    }

    fn handle_error(
        &self,
        sink_mutex: &Mutex<ErrorSinkRaw>,
        cause: impl Error + Send + Sync + 'static,
        label_key: &'static str,
        label: Label,
        string: &'static str,
    ) {
        let error = wgc::error::ContextError {
            string,
            cause: Box::new(cause),
            label: label.unwrap_or_default().to_string(),
            label_key,
        };
        let mut sink = sink_mutex.lock();
        let mut source_opt: Option<&(dyn Error + 'static)> = Some(&error);
        while let Some(source) = source_opt {
            if let Some(wgc::device::DeviceError::OutOfMemory) =
                source.downcast_ref::<wgc::device::DeviceError>()
            {
                return sink.handle_error(crate::Error::OutOfMemory {
                    source: Box::new(error),
                });
            }
            source_opt = source.source();
        }

        // Otherwise, it is a validation error
        sink.handle_error(crate::Error::Validation {
            description: self.format_error(&error),
            source: Box::new(error),
        });
    }

    fn handle_error_nolabel(
        &self,
        sink_mutex: &Mutex<ErrorSinkRaw>,
        cause: impl Error + Send + Sync + 'static,
        string: &'static str,
    ) {
        self.handle_error(sink_mutex, cause, "", None, string)
    }

    #[track_caller]
    fn handle_error_fatal(
        &self,
        cause: impl Error + Send + Sync + 'static,
        operation: &'static str,
    ) -> ! {
        panic!("Error in {operation}: {f}", f = self.format_error(&cause));
    }

    fn format_error(&self, err: &(impl Error + 'static)) -> String {
        let global = self.global();
        let mut err_descs = vec![];

        let mut err_str = String::new();
        wgc::error::format_pretty_any(&mut err_str, global, err);
        err_descs.push(err_str);

        let mut source_opt = err.source();
        while let Some(source) = source_opt {
            let mut source_str = String::new();
            wgc::error::format_pretty_any(&mut source_str, global, source);
            err_descs.push(source_str);
            source_opt = source.source();
        }

        format!("Validation Error\n\nCaused by:\n{}", err_descs.join(""))
    }
}

fn map_buffer_copy_view(view: crate::ImageCopyBuffer) -> wgc::command::ImageCopyBuffer {
    wgc::command::ImageCopyBuffer {
        buffer: view.buffer.id.into(),
        layout: view.layout,
    }
}

fn map_texture_copy_view(view: crate::ImageCopyTexture) -> wgc::command::ImageCopyTexture {
    wgc::command::ImageCopyTexture {
        texture: view.texture.id.into(),
        mip_level: view.mip_level,
        origin: view.origin,
        aspect: view.aspect,
    }
}

#[cfg_attr(
    any(not(target_arch = "wasm32"), target_os = "emscripten"),
    allow(unused)
)]
fn map_texture_tagged_copy_view(
    view: crate::ImageCopyTextureTagged,
) -> wgc::command::ImageCopyTextureTagged {
    wgc::command::ImageCopyTextureTagged {
        texture: view.texture.id.into(),
        mip_level: view.mip_level,
        origin: view.origin,
        aspect: view.aspect,
        color_space: view.color_space,
        premultiplied_alpha: view.premultiplied_alpha,
    }
}

fn map_pass_channel<V: Copy + Default>(
    ops: Option<&Operations<V>>,
) -> wgc::command::PassChannel<V> {
    match ops {
        Some(&Operations {
            load: LoadOp::Clear(clear_value),
            store,
        }) => wgc::command::PassChannel {
            load_op: wgc::command::LoadOp::Clear,
            store_op: if store {
                wgc::command::StoreOp::Store
            } else {
                wgc::command::StoreOp::Discard
            },
            clear_value,
            read_only: false,
        },
        Some(&Operations {
            load: LoadOp::Load,
            store,
        }) => wgc::command::PassChannel {
            load_op: wgc::command::LoadOp::Load,
            store_op: if store {
                wgc::command::StoreOp::Store
            } else {
                wgc::command::StoreOp::Discard
            },
            clear_value: V::default(),
            read_only: false,
        },
        None => wgc::command::PassChannel {
            load_op: wgc::command::LoadOp::Load,
            store_op: wgc::command::StoreOp::Store,
            clear_value: V::default(),
            read_only: true,
        },
    }
}

#[derive(Debug)]
pub struct Surface {
    id: wgc::id::SurfaceId,
    /// Configured device is needed to know which backend
    /// code to execute when acquiring a new frame.
    configured_device: Mutex<Option<wgc::id::DeviceId>>,
}

impl Surface {
    // Not used on every platform
    #[allow(dead_code)]
    pub fn id(&self) -> wgc::id::SurfaceId {
        self.id
    }
}

#[derive(Debug)]
pub struct Device {
    id: wgc::id::DeviceId,
    error_sink: ErrorSink,
    features: Features,
}

impl Device {
    // Not used on every platform
    #[allow(dead_code)]
    pub fn id(&self) -> wgc::id::DeviceId {
        self.id
    }
}

#[derive(Debug)]
pub struct Buffer {
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct Texture {
    id: wgc::id::TextureId,
    error_sink: ErrorSink,
}

impl Texture {
    // Not used on every platform
    #[allow(dead_code)]
    pub fn id(&self) -> wgc::id::TextureId {
        self.id
    }
}

#[derive(Debug)]
pub struct Queue {
    id: wgc::id::QueueId,
    error_sink: ErrorSink,
}

impl Queue {
    // Not used on every platform
    #[allow(dead_code)]
    pub fn id(&self) -> wgc::id::QueueId {
        self.id
    }
}

#[derive(Debug)]
pub struct CommandEncoder {
    error_sink: ErrorSink,
    open: bool,
}

impl crate::Context for Context {
    type AdapterId = wgc::id::AdapterId;
    type AdapterData = ();
    type DeviceId = wgc::id::DeviceId;
    type DeviceData = Device;
    type QueueId = wgc::id::QueueId;
    type QueueData = Queue;
    type ShaderModuleId = wgc::id::ShaderModuleId;
    type ShaderModuleData = ();
    type BindGroupLayoutId = wgc::id::BindGroupLayoutId;
    type BindGroupLayoutData = ();
    type BindGroupId = wgc::id::BindGroupId;
    type BindGroupData = ();
    type TextureViewId = wgc::id::TextureViewId;
    type TextureViewData = ();
    type SamplerId = wgc::id::SamplerId;
    type SamplerData = ();
    type BufferId = wgc::id::BufferId;
    type BufferData = Buffer;
    type TextureId = wgc::id::TextureId;
    type TextureData = Texture;
    type QuerySetId = wgc::id::QuerySetId;
    type QuerySetData = ();
    type PipelineLayoutId = wgc::id::PipelineLayoutId;
    type PipelineLayoutData = ();
    type RenderPipelineId = wgc::id::RenderPipelineId;
    type RenderPipelineData = ();
    type ComputePipelineId = wgc::id::ComputePipelineId;
    type ComputePipelineData = ();
    type CommandEncoderId = wgc::id::CommandEncoderId;
    type CommandEncoderData = CommandEncoder;
    type ComputePassId = Unused;
    type ComputePassData = wgc::command::ComputePass;
    type RenderPassId = Unused;
    type RenderPassData = wgc::command::RenderPass;
    type CommandBufferId = wgc::id::CommandBufferId;
    type CommandBufferData = ();
    type RenderBundleEncoderId = Unused;
    type RenderBundleEncoderData = wgc::command::RenderBundleEncoder;
    type RenderBundleId = wgc::id::RenderBundleId;
    type RenderBundleData = ();

    type SurfaceId = wgc::id::SurfaceId;
    type SurfaceData = Surface;
    type SurfaceOutputDetail = SurfaceOutputDetail;
    type SubmissionIndex = Unused;
    type SubmissionIndexData = wgc::device::queue::WrappedSubmissionIndex;

    type RequestAdapterFuture = Ready<Option<(Self::AdapterId, Self::AdapterData)>>;

    #[allow(clippy::type_complexity)]
    type RequestDeviceFuture = Ready<
        Result<
            (
                Self::DeviceId,
                Self::DeviceData,
                Self::QueueId,
                Self::QueueData,
            ),
            crate::RequestDeviceError,
        >,
    >;

    type PopErrorScopeFuture = Ready<Option<crate::Error>>;

    fn init(instance_desc: wgt::InstanceDescriptor) -> Self {
        Self(wgc::hub::Global::new(
            "wgpu",
            wgc::hub::IdentityManagerFactory,
            instance_desc,
        ))
    }

    fn instance_create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<(Self::SurfaceId, Self::SurfaceData), crate::CreateSurfaceError> {
        let id = self
            .0
            .instance_create_surface(display_handle, window_handle, ());

        Ok((
            id,
            Surface {
                id,
                configured_device: Mutex::new(None),
            },
        ))
    }

    fn instance_request_adapter(
        &self,
        options: &crate::RequestAdapterOptions,
    ) -> Self::RequestAdapterFuture {
        let id = self.0.request_adapter(
            &wgc::instance::RequestAdapterOptions {
                power_preference: options.power_preference,
                force_fallback_adapter: options.force_fallback_adapter,
                compatible_surface: options.compatible_surface.map(|surface| surface.id.into()),
            },
            wgc::instance::AdapterInputs::Mask(wgt::Backends::all(), |_| ()),
        );
        ready(id.ok().map(|id| (id, ())))
    }

    fn adapter_request_device(
        &self,
        adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
        desc: &crate::DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture {
        let global = &self.0;
        let (device_id, error) = wgc::gfx_select!(*adapter => global.adapter_request_device(
            *adapter,
            &desc.map_label(|l| l.map(Borrowed)),
            trace_dir,
            ()
        ));
        if let Some(err) = error {
            log::error!("Error in Adapter::request_device: {}", err);
            return ready(Err(crate::RequestDeviceError));
        }
        let error_sink = Arc::new(Mutex::new(ErrorSinkRaw::new()));
        let device = Device {
            id: device_id,
            error_sink: error_sink.clone(),
            features: desc.features,
        };
        let queue = Queue {
            id: device_id,
            error_sink,
        };
        ready(Ok((device_id, device, device_id, queue)))
    }

    fn instance_poll_all_devices(&self, force_wait: bool) -> bool {
        let global = &self.0;
        match global.poll_all_devices(force_wait) {
            Ok(all_queue_empty) => all_queue_empty,
            Err(err) => self.handle_error_fatal(err, "Device::poll"),
        }
    }

    fn adapter_is_surface_supported(
        &self,
        adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
        surface: &Self::SurfaceId,
        _surface_data: &Self::SurfaceData,
    ) -> bool {
        let global = &self.0;
        match wgc::gfx_select!(adapter => global.adapter_is_surface_supported(*adapter, *surface)) {
            Ok(result) => result,
            Err(err) => self.handle_error_fatal(err, "Adapter::is_surface_supported"),
        }
    }

    fn adapter_features(
        &self,
        adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> Features {
        let global = &self.0;
        match wgc::gfx_select!(*adapter => global.adapter_features(*adapter)) {
            Ok(features) => features,
            Err(err) => self.handle_error_fatal(err, "Adapter::features"),
        }
    }

    fn adapter_limits(
        &self,
        adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> Limits {
        let global = &self.0;
        match wgc::gfx_select!(*adapter => global.adapter_limits(*adapter)) {
            Ok(limits) => limits,
            Err(err) => self.handle_error_fatal(err, "Adapter::limits"),
        }
    }

    fn adapter_downlevel_capabilities(
        &self,
        adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> DownlevelCapabilities {
        let global = &self.0;
        match wgc::gfx_select!(*adapter => global.adapter_downlevel_capabilities(*adapter)) {
            Ok(downlevel) => downlevel,
            Err(err) => self.handle_error_fatal(err, "Adapter::downlevel_properties"),
        }
    }

    fn adapter_get_info(
        &self,
        adapter: &wgc::id::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> AdapterInfo {
        let global = &self.0;
        match wgc::gfx_select!(*adapter => global.adapter_get_info(*adapter)) {
            Ok(info) => info,
            Err(err) => self.handle_error_fatal(err, "Adapter::get_info"),
        }
    }

    fn adapter_get_texture_format_features(
        &self,
        adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        let global = &self.0;
        match wgc::gfx_select!(*adapter => global.adapter_get_texture_format_features(*adapter, format))
        {
            Ok(info) => info,
            Err(err) => self.handle_error_fatal(err, "Adapter::get_texture_format_features"),
        }
    }

    fn adapter_get_presentation_timestamp(
        &self,
        adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::PresentationTimestamp {
        let global = &self.0;
        match wgc::gfx_select!(*adapter => global.adapter_get_presentation_timestamp(*adapter)) {
            Ok(timestamp) => timestamp,
            Err(err) => self.handle_error_fatal(err, "Adapter::correlate_presentation_timestamp"),
        }
    }

    fn surface_get_capabilities(
        &self,
        surface: &Self::SurfaceId,
        _surface_data: &Self::SurfaceData,
        adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> wgt::SurfaceCapabilities {
        let global = &self.0;
        match wgc::gfx_select!(adapter => global.surface_get_capabilities(*surface, *adapter)) {
            Ok(caps) => caps,
            Err(wgc::instance::GetSurfaceSupportError::Unsupported) => {
                wgt::SurfaceCapabilities::default()
            }
            Err(err) => self.handle_error_fatal(err, "Surface::get_supported_formats"),
        }
    }

    fn surface_configure(
        &self,
        surface: &Self::SurfaceId,
        surface_data: &Self::SurfaceData,
        device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        config: &crate::SurfaceConfiguration,
    ) {
        let global = &self.0;
        let error = wgc::gfx_select!(device => global.surface_configure(*surface, *device, config));
        if let Some(e) = error {
            self.handle_error_fatal(e, "Surface::configure");
        } else {
            *surface_data.configured_device.lock() = Some(*device);
        }
    }

    fn surface_get_current_texture(
        &self,
        surface: &Self::SurfaceId,
        surface_data: &Self::SurfaceData,
    ) -> (
        Option<Self::TextureId>,
        Option<Self::TextureData>,
        SurfaceStatus,
        Self::SurfaceOutputDetail,
    ) {
        let global = &self.0;
        let device_id = surface_data
            .configured_device
            .lock()
            .expect("Surface was not configured?");
        match wgc::gfx_select!(
            device_id => global.surface_get_current_texture(*surface, ())
        ) {
            Ok(wgc::present::SurfaceOutput { status, texture_id }) => {
                let (id, data) = {
                    (
                        texture_id,
                        texture_id.map(|id| Texture {
                            id,
                            error_sink: Arc::new(Mutex::new(ErrorSinkRaw::new())),
                        }),
                    )
                };

                (
                    id,
                    data,
                    status,
                    SurfaceOutputDetail {
                        surface_id: *surface,
                    },
                )
            }
            Err(err) => self.handle_error_fatal(err, "Surface::get_current_texture_view"),
        }
    }

    fn surface_present(&self, texture: &Self::TextureId, detail: &Self::SurfaceOutputDetail) {
        let global = &self.0;
        match wgc::gfx_select!(texture => global.surface_present(detail.surface_id)) {
            Ok(_status) => (),
            Err(err) => self.handle_error_fatal(err, "Surface::present"),
        }
    }

    fn surface_texture_discard(
        &self,
        texture: &Self::TextureId,
        detail: &Self::SurfaceOutputDetail,
    ) {
        let global = &self.0;
        match wgc::gfx_select!(texture => global.surface_texture_discard(detail.surface_id)) {
            Ok(_status) => (),
            Err(err) => self.handle_error_fatal(err, "Surface::discard_texture"),
        }
    }

    fn device_features(
        &self,
        device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> Features {
        let global = &self.0;
        match wgc::gfx_select!(device => global.device_features(*device)) {
            Ok(features) => features,
            Err(err) => self.handle_error_fatal(err, "Device::features"),
        }
    }

    fn device_limits(&self, device: &Self::DeviceId, _device_data: &Self::DeviceData) -> Limits {
        let global = &self.0;
        match wgc::gfx_select!(device => global.device_limits(*device)) {
            Ok(limits) => limits,
            Err(err) => self.handle_error_fatal(err, "Device::limits"),
        }
    }

    fn device_downlevel_properties(
        &self,
        device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> DownlevelCapabilities {
        let global = &self.0;
        match wgc::gfx_select!(device => global.device_downlevel_properties(*device)) {
            Ok(limits) => limits,
            Err(err) => self.handle_error_fatal(err, "Device::downlevel_properties"),
        }
    }

    #[cfg_attr(
        not(any(
            feature = "spirv",
            feature = "glsl",
            feature = "wgsl",
            feature = "naga"
        )),
        allow(unreachable_code, unused_variables)
    )]
    fn device_create_shader_module(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: ShaderModuleDescriptor,
        shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> (Self::ShaderModuleId, Self::ShaderModuleData) {
        let global = &self.0;
        let descriptor = wgc::pipeline::ShaderModuleDescriptor {
            label: desc.label.map(Borrowed),
            shader_bound_checks,
        };
        let source = match desc.source {
            #[cfg(feature = "spirv")]
            ShaderSource::SpirV(ref spv) => {
                // Parse the given shader code and store its representation.
                let options = naga::front::spv::Options {
                    adjust_coordinate_space: false, // we require NDC_Y_UP feature
                    strict_capabilities: true,
                    block_ctx_dump_prefix: None,
                };
                let parser = naga::front::spv::Frontend::new(spv.iter().cloned(), &options);
                let module = parser.parse().unwrap();
                wgc::pipeline::ShaderModuleSource::Naga(Owned(module))
            }
            #[cfg(feature = "glsl")]
            ShaderSource::Glsl {
                ref shader,
                stage,
                ref defines,
            } => {
                // Parse the given shader code and store its representation.
                let options = naga::front::glsl::Options {
                    stage,
                    defines: defines.clone(),
                };
                let mut parser = naga::front::glsl::Frontend::default();
                let module = parser.parse(&options, shader).unwrap();

                wgc::pipeline::ShaderModuleSource::Naga(Owned(module))
            }
            #[cfg(feature = "wgsl")]
            ShaderSource::Wgsl(ref code) => wgc::pipeline::ShaderModuleSource::Wgsl(Borrowed(code)),
            #[cfg(feature = "naga")]
            ShaderSource::Naga(module) => wgc::pipeline::ShaderModuleSource::Naga(module),
            ShaderSource::Dummy(_) => panic!("found `ShaderSource::Dummy`"),
        };
        let (id, error) = wgc::gfx_select!(
            device => global.device_create_shader_module(*device, &descriptor, source, ())
        );
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_shader_module",
            );
        }
        (id, ())
    }

    unsafe fn device_create_shader_module_spirv(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &ShaderModuleDescriptorSpirV,
    ) -> (Self::ShaderModuleId, Self::ShaderModuleData) {
        let global = &self.0;
        let descriptor = wgc::pipeline::ShaderModuleDescriptor {
            label: desc.label.map(Borrowed),
            // Doesn't matter the value since spirv shaders aren't mutated to include
            // runtime checks
            shader_bound_checks: unsafe { wgt::ShaderBoundChecks::unchecked() },
        };
        let (id, error) = wgc::gfx_select!(
            device => global.device_create_shader_module_spirv(*device, &descriptor, Borrowed(&desc.source), ())
        );
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_shader_module_spirv",
            );
        }
        (id, ())
    }

    fn device_create_bind_group_layout(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &BindGroupLayoutDescriptor,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        let global = &self.0;
        let descriptor = wgc::binding_model::BindGroupLayoutDescriptor {
            label: desc.label.map(Borrowed),
            entries: Borrowed(desc.entries),
        };
        let (id, error) = wgc::gfx_select!(
            device => global.device_create_bind_group_layout(*device, &descriptor, ())
        );
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_bind_group_layout",
            );
        }
        (id, ())
    }
    fn device_create_bind_group(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &BindGroupDescriptor,
    ) -> (Self::BindGroupId, Self::BindGroupData) {
        use wgc::binding_model as bm;

        let mut arrayed_texture_views = Vec::<ObjectId>::new();
        let mut arrayed_samplers = Vec::<ObjectId>::new();
        if device_data
            .features
            .contains(Features::TEXTURE_BINDING_ARRAY)
        {
            // gather all the array view IDs first
            for entry in desc.entries.iter() {
                if let BindingResource::TextureViewArray(array) = entry.resource {
                    arrayed_texture_views.extend(array.iter().map(|view| &view.id));
                }
                if let BindingResource::SamplerArray(array) = entry.resource {
                    arrayed_samplers.extend(array.iter().map(|sampler| &sampler.id));
                }
            }
        }
        let mut remaining_arrayed_texture_views = &arrayed_texture_views[..];
        let mut remaining_arrayed_samplers = &arrayed_samplers[..];

        let mut arrayed_buffer_bindings = Vec::new();
        if device_data
            .features
            .contains(Features::BUFFER_BINDING_ARRAY)
        {
            // gather all the buffers first
            for entry in desc.entries.iter() {
                if let BindingResource::BufferArray(array) = entry.resource {
                    arrayed_buffer_bindings.extend(array.iter().map(|binding| bm::BufferBinding {
                        buffer_id: binding.buffer.id.into(),
                        offset: binding.offset,
                        size: binding.size,
                    }));
                }
            }
        }
        let mut remaining_arrayed_buffer_bindings = &arrayed_buffer_bindings[..];

        let entries = desc
            .entries
            .iter()
            .map(|entry| bm::BindGroupEntry {
                binding: entry.binding,
                resource: match entry.resource {
                    BindingResource::Buffer(BufferBinding {
                        buffer,
                        offset,
                        size,
                    }) => bm::BindingResource::Buffer(bm::BufferBinding {
                        buffer_id: buffer.id.into(),
                        offset,
                        size,
                    }),
                    BindingResource::BufferArray(array) => {
                        let slice = &remaining_arrayed_buffer_bindings[..array.len()];
                        remaining_arrayed_buffer_bindings =
                            &remaining_arrayed_buffer_bindings[array.len()..];
                        bm::BindingResource::BufferArray(Borrowed(slice))
                    }
                    BindingResource::Sampler(sampler) => {
                        bm::BindingResource::Sampler(sampler.id.into())
                    }
                    BindingResource::SamplerArray(array) => {
                        let samplers = remaining_arrayed_samplers[..array.len()]
                            .iter()
                            .map(|id| <Self::SamplerId>::from(*id))
                            .collect::<Vec<_>>();
                        remaining_arrayed_samplers = &remaining_arrayed_samplers[array.len()..];
                        bm::BindingResource::SamplerArray(Owned(samplers))
                    }
                    BindingResource::TextureView(texture_view) => {
                        bm::BindingResource::TextureView(texture_view.id.into())
                    }
                    BindingResource::TextureViewArray(array) => {
                        let views = remaining_arrayed_texture_views[..array.len()]
                            .iter()
                            .map(|id| <Self::TextureViewId>::from(*id))
                            .collect::<Vec<_>>();
                        remaining_arrayed_texture_views =
                            &remaining_arrayed_texture_views[array.len()..];
                        bm::BindingResource::TextureViewArray(Owned(views))
                    }
                },
            })
            .collect::<Vec<_>>();
        let descriptor = bm::BindGroupDescriptor {
            label: desc.label.as_ref().map(|label| Borrowed(&label[..])),
            layout: desc.layout.id.into(),
            entries: Borrowed(&entries),
        };

        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device => global.device_create_bind_group(
            *device,
            &descriptor,
            ()
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_bind_group",
            );
        }
        (id, ())
    }
    fn device_create_pipeline_layout(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &PipelineLayoutDescriptor,
    ) -> (Self::PipelineLayoutId, Self::PipelineLayoutData) {
        // Limit is always less or equal to hal::MAX_BIND_GROUPS, so this is always right
        // Guards following ArrayVec
        assert!(
            desc.bind_group_layouts.len() <= wgc::MAX_BIND_GROUPS,
            "Bind group layout count {} exceeds device bind group limit {}",
            desc.bind_group_layouts.len(),
            wgc::MAX_BIND_GROUPS
        );

        let temp_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|bgl| bgl.id.into())
            .collect::<ArrayVec<_, { wgc::MAX_BIND_GROUPS }>>();
        let descriptor = wgc::binding_model::PipelineLayoutDescriptor {
            label: desc.label.map(Borrowed),
            bind_group_layouts: Borrowed(&temp_layouts),
            push_constant_ranges: Borrowed(desc.push_constant_ranges),
        };

        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device => global.device_create_pipeline_layout(
            *device,
            &descriptor,
            ()
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_pipeline_layout",
            );
        }
        (id, ())
    }
    fn device_create_render_pipeline(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &RenderPipelineDescriptor,
    ) -> (Self::RenderPipelineId, Self::RenderPipelineData) {
        use wgc::pipeline as pipe;

        let vertex_buffers: ArrayVec<_, { wgc::MAX_VERTEX_BUFFERS }> = desc
            .vertex
            .buffers
            .iter()
            .map(|vbuf| pipe::VertexBufferLayout {
                array_stride: vbuf.array_stride,
                step_mode: vbuf.step_mode,
                attributes: Borrowed(vbuf.attributes),
            })
            .collect();

        let implicit_pipeline_ids = match desc.layout {
            Some(_) => None,
            None => Some(wgc::device::ImplicitPipelineIds {
                root_id: (),
                group_ids: &[(); wgc::MAX_BIND_GROUPS],
            }),
        };
        let descriptor = pipe::RenderPipelineDescriptor {
            label: desc.label.map(Borrowed),
            layout: desc.layout.map(|l| l.id.into()),
            vertex: pipe::VertexState {
                stage: pipe::ProgrammableStageDescriptor {
                    module: desc.vertex.module.id.into(),
                    entry_point: Borrowed(desc.vertex.entry_point),
                },
                buffers: Borrowed(&vertex_buffers),
            },
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment: desc.fragment.as_ref().map(|frag| pipe::FragmentState {
                stage: pipe::ProgrammableStageDescriptor {
                    module: frag.module.id.into(),
                    entry_point: Borrowed(frag.entry_point),
                },
                targets: Borrowed(frag.targets),
            }),
            multiview: desc.multiview,
        };

        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device => global.device_create_render_pipeline(
            *device,
            &descriptor,
            (),
            implicit_pipeline_ids
        ));
        if let Some(cause) = error {
            if let wgc::pipeline::CreateRenderPipelineError::Internal { stage, ref error } = cause {
                log::error!("Shader translation error for stage {:?}: {}", stage, error);
                log::error!("Please report it to https://github.com/gfx-rs/naga");
            }
            self.handle_error(
                &device_data.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_render_pipeline",
            );
        }
        (id, ())
    }
    fn device_create_compute_pipeline(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &ComputePipelineDescriptor,
    ) -> (Self::ComputePipelineId, Self::ComputePipelineData) {
        use wgc::pipeline as pipe;

        let implicit_pipeline_ids = match desc.layout {
            Some(_) => None,
            None => Some(wgc::device::ImplicitPipelineIds {
                root_id: (),
                group_ids: &[(); wgc::MAX_BIND_GROUPS],
            }),
        };
        let descriptor = pipe::ComputePipelineDescriptor {
            label: desc.label.map(Borrowed),
            layout: desc.layout.map(|l| l.id.into()),
            stage: pipe::ProgrammableStageDescriptor {
                module: desc.module.id.into(),
                entry_point: Borrowed(desc.entry_point),
            },
        };

        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device => global.device_create_compute_pipeline(
            *device,
            &descriptor,
            (),
            implicit_pipeline_ids
        ));
        if let Some(cause) = error {
            if let wgc::pipeline::CreateComputePipelineError::Internal(ref error) = cause {
                log::warn!(
                    "Shader translation error for stage {:?}: {}",
                    wgt::ShaderStages::COMPUTE,
                    error
                );
                log::warn!("Please report it to https://github.com/gfx-rs/naga");
            }
            self.handle_error(
                &device_data.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_compute_pipeline",
            );
        }
        (id, ())
    }
    fn device_create_buffer(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &crate::BufferDescriptor<'_>,
    ) -> (Self::BufferId, Self::BufferData) {
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device => global.device_create_buffer(
            *device,
            &desc.map_label(|l| l.map(Borrowed)),
            ()
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_buffer",
            );
        }
        (
            id,
            Buffer {
                error_sink: Arc::clone(&device_data.error_sink),
            },
        )
    }
    fn device_create_texture(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &TextureDescriptor,
    ) -> (Self::TextureId, Self::TextureData) {
        let wgt_desc = desc.map_label_and_view_formats(|l| l.map(Borrowed), |v| v.to_vec());
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device => global.device_create_texture(
            *device,
            &wgt_desc,
            ()
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_texture",
            );
        }
        (
            id,
            Texture {
                id,
                error_sink: Arc::clone(&device_data.error_sink),
            },
        )
    }
    fn device_create_sampler(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &SamplerDescriptor,
    ) -> (Self::SamplerId, Self::SamplerData) {
        let descriptor = wgc::resource::SamplerDescriptor {
            label: desc.label.map(Borrowed),
            address_modes: [
                desc.address_mode_u,
                desc.address_mode_v,
                desc.address_mode_w,
            ],
            mag_filter: desc.mag_filter,
            min_filter: desc.min_filter,
            mipmap_filter: desc.mipmap_filter,
            lod_min_clamp: desc.lod_min_clamp,
            lod_max_clamp: desc.lod_max_clamp,
            compare: desc.compare,
            anisotropy_clamp: desc.anisotropy_clamp,
            border_color: desc.border_color,
        };

        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device => global.device_create_sampler(
            *device,
            &descriptor,
            ()
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_sampler",
            );
        }
        (id, ())
    }
    fn device_create_query_set(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &wgt::QuerySetDescriptor<Label>,
    ) -> (Self::QuerySetId, Self::QuerySetData) {
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device => global.device_create_query_set(
            *device,
            &desc.map_label(|l| l.map(Borrowed)),
            ()
        ));
        if let Some(cause) = error {
            self.handle_error_nolabel(&device_data.error_sink, cause, "Device::create_query_set");
        }
        (id, ())
    }
    fn device_create_command_encoder(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &CommandEncoderDescriptor,
    ) -> (Self::CommandEncoderId, Self::CommandEncoderData) {
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device => global.device_create_command_encoder(
            *device,
            &desc.map_label(|l| l.map(Borrowed)),
            ()
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_command_encoder",
            );
        }
        (
            id,
            CommandEncoder {
                error_sink: Arc::clone(&device_data.error_sink),
                open: true,
            },
        )
    }
    fn device_create_render_bundle_encoder(
        &self,
        device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        desc: &RenderBundleEncoderDescriptor,
    ) -> (Self::RenderBundleEncoderId, Self::RenderBundleEncoderData) {
        let descriptor = wgc::command::RenderBundleEncoderDescriptor {
            label: desc.label.map(Borrowed),
            color_formats: Borrowed(desc.color_formats),
            depth_stencil: desc.depth_stencil,
            sample_count: desc.sample_count,
            multiview: desc.multiview,
        };
        match wgc::command::RenderBundleEncoder::new(&descriptor, *device, None) {
            Ok(encoder) => (Unused, encoder),
            Err(e) => panic!("Error in Device::create_render_bundle_encoder: {e}"),
        }
    }
    #[cfg_attr(target_arch = "wasm32", allow(unused))]
    fn device_drop(&self, device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        let global = &self.0;

        #[cfg(any(not(target_arch = "wasm32"), target_os = "emscripten"))]
        {
            match wgc::gfx_select!(device => global.device_poll(*device, wgt::Maintain::Wait)) {
                Ok(_) => (),
                Err(err) => self.handle_error_fatal(err, "Device::drop"),
            }
        }

        wgc::gfx_select!(device => global.device_drop(*device));
    }
    fn device_poll(
        &self,
        device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        maintain: crate::Maintain,
    ) -> bool {
        let global = &self.0;
        let maintain_inner = maintain.map_index(|i| *i.1.as_ref().downcast_ref().unwrap());
        match wgc::gfx_select!(device => global.device_poll(
            *device,
            maintain_inner
        )) {
            Ok(queue_empty) => queue_empty,
            Err(err) => self.handle_error_fatal(err, "Device::poll"),
        }
    }
    fn device_on_uncaptured_error(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        handler: Box<dyn UncapturedErrorHandler>,
    ) {
        let mut error_sink = device_data.error_sink.lock();
        error_sink.uncaptured_handler = handler;
    }
    fn device_push_error_scope(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        filter: crate::ErrorFilter,
    ) {
        let mut error_sink = device_data.error_sink.lock();
        error_sink.scopes.push(ErrorScope {
            error: None,
            filter,
        });
    }
    fn device_pop_error_scope(
        &self,
        _device: &Self::DeviceId,
        device_data: &Self::DeviceData,
    ) -> Self::PopErrorScopeFuture {
        let mut error_sink = device_data.error_sink.lock();
        let scope = error_sink.scopes.pop().unwrap();
        ready(scope.error)
    }

    fn buffer_map_async(
        &self,
        buffer: &Self::BufferId,
        buffer_data: &Self::BufferData,
        mode: MapMode,
        range: Range<wgt::BufferAddress>,
        callback: Box<dyn FnOnce(Result<(), crate::BufferAsyncError>) + Send + 'static>,
    ) {
        let operation = wgc::resource::BufferMapOperation {
            host: match mode {
                MapMode::Read => wgc::device::HostMap::Read,
                MapMode::Write => wgc::device::HostMap::Write,
            },
            callback: wgc::resource::BufferMapCallback::from_rust(Box::new(|status| {
                let res = status.map_err(|_| crate::BufferAsyncError);
                callback(res);
            })),
        };

        let global = &self.0;
        match wgc::gfx_select!(buffer => global.buffer_map_async(*buffer, range, operation)) {
            Ok(()) => (),
            Err(cause) => {
                self.handle_error_nolabel(&buffer_data.error_sink, cause, "Buffer::map_async")
            }
        }
    }
    fn buffer_get_mapped_range(
        &self,
        buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        sub_range: Range<wgt::BufferAddress>,
    ) -> Box<dyn crate::context::BufferMappedRange> {
        let size = sub_range.end - sub_range.start;
        let global = &self.0;
        match wgc::gfx_select!(buffer => global.buffer_get_mapped_range(
            *buffer,
            sub_range.start,
            Some(size)
        )) {
            Ok((ptr, size)) => Box::new(BufferMappedRange {
                ptr,
                size: size as usize,
            }),
            Err(err) => self.handle_error_fatal(err, "Buffer::get_mapped_range"),
        }
    }

    fn buffer_unmap(&self, buffer: &Self::BufferId, buffer_data: &Self::BufferData) {
        let global = &self.0;
        match wgc::gfx_select!(buffer => global.buffer_unmap(*buffer)) {
            Ok(()) => (),
            Err(cause) => {
                self.handle_error_nolabel(&buffer_data.error_sink, cause, "Buffer::buffer_unmap")
            }
        }
    }

    fn texture_create_view(
        &self,
        texture: &Self::TextureId,
        texture_data: &Self::TextureData,
        desc: &TextureViewDescriptor,
    ) -> (Self::TextureViewId, Self::TextureViewData) {
        let descriptor = wgc::resource::TextureViewDescriptor {
            label: desc.label.map(Borrowed),
            format: desc.format,
            dimension: desc.dimension,
            range: wgt::ImageSubresourceRange {
                aspect: desc.aspect,
                base_mip_level: desc.base_mip_level,
                mip_level_count: desc.mip_level_count,
                base_array_layer: desc.base_array_layer,
                array_layer_count: desc.array_layer_count,
            },
        };
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(
            texture => global.texture_create_view(*texture, &descriptor, ())
        );
        if let Some(cause) = error {
            self.handle_error(
                &texture_data.error_sink,
                cause,
                LABEL,
                desc.label,
                "Texture::create_view",
            );
        }
        (id, ())
    }

    fn surface_drop(&self, surface: &Self::SurfaceId, _surface_data: &Self::SurfaceData) {
        self.0.surface_drop(*surface)
    }

    fn adapter_drop(&self, adapter: &Self::AdapterId, _adapter_data: &Self::AdapterData) {
        let global = &self.0;
        wgc::gfx_select!(*adapter => global.adapter_drop(*adapter))
    }

    fn buffer_destroy(&self, buffer: &Self::BufferId, _buffer_data: &Self::BufferData) {
        // Per spec, no error to report. Even calling destroy multiple times is valid.
        let global = &self.0;
        let _ = wgc::gfx_select!(buffer => global.buffer_destroy(*buffer));
    }

    fn buffer_drop(&self, buffer: &Self::BufferId, _buffer_data: &Self::BufferData) {
        let global = &self.0;
        wgc::gfx_select!(buffer => global.buffer_drop(*buffer, false))
    }

    fn texture_destroy(&self, texture: &Self::TextureId, _texture_data: &Self::TextureData) {
        // Per spec, no error to report. Even calling destroy multiple times is valid.
        let global = &self.0;
        let _ = wgc::gfx_select!(texture => global.texture_destroy(*texture));
    }

    fn texture_drop(&self, texture: &Self::TextureId, _texture_data: &Self::TextureData) {
        let global = &self.0;
        wgc::gfx_select!(texture => global.texture_drop(*texture, false))
    }

    fn texture_view_drop(
        &self,
        texture_view: &Self::TextureViewId,
        __texture_view_data: &Self::TextureViewData,
    ) {
        let global = &self.0;
        let _ = wgc::gfx_select!(*texture_view => global.texture_view_drop(*texture_view, false));
    }

    fn sampler_drop(&self, sampler: &Self::SamplerId, _sampler_data: &Self::SamplerData) {
        let global = &self.0;
        wgc::gfx_select!(*sampler => global.sampler_drop(*sampler))
    }

    fn query_set_drop(&self, query_set: &Self::QuerySetId, _query_set_data: &Self::QuerySetData) {
        let global = &self.0;
        wgc::gfx_select!(*query_set => global.query_set_drop(*query_set))
    }

    fn bind_group_drop(
        &self,
        bind_group: &Self::BindGroupId,
        _bind_group_data: &Self::BindGroupData,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*bind_group => global.bind_group_drop(*bind_group))
    }

    fn bind_group_layout_drop(
        &self,
        bind_group_layout: &Self::BindGroupLayoutId,
        _bind_group_layout_data: &Self::BindGroupLayoutData,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*bind_group_layout => global.bind_group_layout_drop(*bind_group_layout))
    }

    fn pipeline_layout_drop(
        &self,
        pipeline_layout: &Self::PipelineLayoutId,
        _pipeline_layout_data: &Self::PipelineLayoutData,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*pipeline_layout => global.pipeline_layout_drop(*pipeline_layout))
    }
    fn shader_module_drop(
        &self,
        shader_module: &Self::ShaderModuleId,
        _shader_module_data: &Self::ShaderModuleData,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*shader_module => global.shader_module_drop(*shader_module))
    }
    fn command_encoder_drop(
        &self,
        command_encoder: &Self::CommandEncoderId,
        command_encoder_data: &Self::CommandEncoderData,
    ) {
        if command_encoder_data.open {
            let global = &self.0;
            wgc::gfx_select!(command_encoder => global.command_encoder_drop(*command_encoder))
        }
    }

    fn command_buffer_drop(
        &self,
        command_buffer: &Self::CommandBufferId,
        _command_buffer_data: &Self::CommandBufferData,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*command_buffer => global.command_buffer_drop(*command_buffer))
    }

    fn render_bundle_drop(
        &self,
        render_bundle: &Self::RenderBundleId,
        _render_bundle_data: &Self::RenderBundleData,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*render_bundle => global.render_bundle_drop(*render_bundle))
    }

    fn compute_pipeline_drop(
        &self,
        pipeline: &Self::ComputePipelineId,
        _pipeline_data: &Self::ComputePipelineData,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*pipeline => global.compute_pipeline_drop(*pipeline))
    }

    fn render_pipeline_drop(
        &self,
        pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
    ) {
        let global = &self.0;
        wgc::gfx_select!(*pipeline => global.render_pipeline_drop(*pipeline))
    }

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline: &Self::ComputePipelineId,
        _pipeline_data: &Self::ComputePipelineData,
        index: u32,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(*pipeline => global.compute_pipeline_get_bind_group_layout(*pipeline, index, ()));
        if let Some(err) = error {
            panic!("Error reflecting bind group {index}: {err}");
        }
        (id, ())
    }

    fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
        index: u32,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(*pipeline => global.render_pipeline_get_bind_group_layout(*pipeline, index, ()));
        if let Some(err) = error {
            panic!("Error reflecting bind group {index}: {err}");
        }
        (id, ())
    }

    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        source: &Self::BufferId,
        _source_data: &Self::BufferData,
        source_offset: wgt::BufferAddress,
        destination: &Self::BufferId,
        _destination_data: &Self::BufferData,
        destination_offset: wgt::BufferAddress,
        copy_size: wgt::BufferAddress,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder => global.command_encoder_copy_buffer_to_buffer(
            *encoder,
            *source,
            source_offset,
            *destination,
            destination_offset,
            copy_size
        )) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::copy_buffer_to_buffer",
            );
        }
    }

    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        source: crate::ImageCopyBuffer,
        destination: crate::ImageCopyTexture,
        copy_size: wgt::Extent3d,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder => global.command_encoder_copy_buffer_to_texture(
            *encoder,
            &map_buffer_copy_view(source),
            &map_texture_copy_view(destination),
            &copy_size
        )) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::copy_buffer_to_texture",
            );
        }
    }

    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        source: crate::ImageCopyTexture,
        destination: crate::ImageCopyBuffer,
        copy_size: wgt::Extent3d,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder => global.command_encoder_copy_texture_to_buffer(
            *encoder,
            &map_texture_copy_view(source),
            &map_buffer_copy_view(destination),
            &copy_size
        )) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::copy_texture_to_buffer",
            );
        }
    }

    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        source: crate::ImageCopyTexture,
        destination: crate::ImageCopyTexture,
        copy_size: wgt::Extent3d,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder => global.command_encoder_copy_texture_to_texture(
            *encoder,
            &map_texture_copy_view(source),
            &map_texture_copy_view(destination),
            &copy_size
        )) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::copy_texture_to_texture",
            );
        }
    }

    fn command_encoder_begin_compute_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        desc: &ComputePassDescriptor,
    ) -> (Self::ComputePassId, Self::ComputePassData) {
        let timestamp_writes = desc
            .timestamp_writes
            .as_ref()
            .iter()
            .map(|t| wgc::command::ComputePassTimestampWrite {
                query_set: t.query_set.id.into(),
                query_index: t.query_index,
                location: match t.location {
                    crate::ComputePassTimestampLocation::Beginning => {
                        wgc::command::ComputePassTimestampLocation::Beginning
                    }
                    crate::ComputePassTimestampLocation::End => {
                        wgc::command::ComputePassTimestampLocation::End
                    }
                },
            })
            .collect::<Vec<_>>();
        (
            Unused,
            wgc::command::ComputePass::new(
                *encoder,
                &wgc::command::ComputePassDescriptor {
                    label: desc.label.map(Borrowed),
                    timestamp_writes: Borrowed(&timestamp_writes),
                },
            ),
        )
    }

    fn command_encoder_end_compute_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(
            encoder => global.command_encoder_run_compute_pass(*encoder, pass_data)
        ) {
            let name = wgc::gfx_select!(encoder => global.command_buffer_label(*encoder));
            self.handle_error(
                &encoder_data.error_sink,
                cause,
                "encoder",
                Some(&name),
                "a ComputePass",
            );
        }
    }

    fn command_encoder_begin_render_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        _encoder_data: &Self::CommandEncoderData,
        desc: &crate::RenderPassDescriptor<'_, '_>,
    ) -> (Self::RenderPassId, Self::RenderPassData) {
        if desc.color_attachments.len() > wgc::MAX_COLOR_ATTACHMENTS {
            self.handle_error_fatal(
                wgc::command::ColorAttachmentError::TooMany {
                    given: desc.color_attachments.len(),
                    limit: wgc::MAX_COLOR_ATTACHMENTS,
                },
                "CommandEncoder::begin_render_pass",
            );
        }
        let colors = desc
            .color_attachments
            .iter()
            .map(|ca| {
                ca.as_ref()
                    .map(|at| wgc::command::RenderPassColorAttachment {
                        view: at.view.id.into(),
                        resolve_target: at.resolve_target.map(|rt| rt.id.into()),
                        channel: map_pass_channel(Some(&at.ops)),
                    })
            })
            .collect::<ArrayVec<_, { wgc::MAX_COLOR_ATTACHMENTS }>>();

        let depth_stencil = desc.depth_stencil_attachment.as_ref().map(|dsa| {
            wgc::command::RenderPassDepthStencilAttachment {
                view: dsa.view.id.into(),
                depth: map_pass_channel(dsa.depth_ops.as_ref()),
                stencil: map_pass_channel(dsa.stencil_ops.as_ref()),
            }
        });

        let timestamp_writes = desc
            .timestamp_writes
            .as_ref()
            .iter()
            .map(|t| wgc::command::RenderPassTimestampWrite {
                query_set: t.query_set.id.into(),
                query_index: t.query_index,
                location: match t.location {
                    crate::RenderPassTimestampLocation::Beginning => {
                        wgc::command::RenderPassTimestampLocation::Beginning
                    }
                    crate::RenderPassTimestampLocation::End => {
                        wgc::command::RenderPassTimestampLocation::End
                    }
                },
            })
            .collect::<Vec<_>>();

        (
            Unused,
            wgc::command::RenderPass::new(
                *encoder,
                &wgc::command::RenderPassDescriptor {
                    label: desc.label.map(Borrowed),
                    color_attachments: Borrowed(&colors),
                    depth_stencil_attachment: depth_stencil.as_ref(),
                    timestamp_writes: Borrowed(&timestamp_writes),
                },
            ),
        )
    }

    fn command_encoder_end_render_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
    ) {
        let global = &self.0;
        if let Err(cause) =
            wgc::gfx_select!(encoder => global.command_encoder_run_render_pass(*encoder, pass_data))
        {
            let name = wgc::gfx_select!(encoder => global.command_buffer_label(*encoder));
            self.handle_error(
                &encoder_data.error_sink,
                cause,
                "encoder",
                Some(&name),
                "a RenderPass",
            );
        }
    }

    fn command_encoder_finish(
        &self,
        encoder: Self::CommandEncoderId,
        encoder_data: &mut Self::CommandEncoderData,
    ) -> (Self::CommandBufferId, Self::CommandBufferData) {
        let descriptor = wgt::CommandBufferDescriptor::default();
        encoder_data.open = false; // prevent the drop
        let global = &self.0;
        let (id, error) =
            wgc::gfx_select!(encoder => global.command_encoder_finish(encoder, &descriptor));
        if let Some(cause) = error {
            self.handle_error_nolabel(&encoder_data.error_sink, cause, "a CommandEncoder");
        }
        (id, ())
    }

    fn command_encoder_clear_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        texture: &crate::Texture,
        subresource_range: &wgt::ImageSubresourceRange,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder => global.command_encoder_clear_texture(
            *encoder,
            texture.id.into(),
            subresource_range
        )) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::clear_texture",
            );
        }
    }

    fn command_encoder_clear_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        buffer: &crate::Buffer,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder => global.command_encoder_clear_buffer(
            *encoder,
            buffer.id.into(),
            offset, size
        )) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::fill_buffer",
            );
        }
    }

    fn command_encoder_insert_debug_marker(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        label: &str,
    ) {
        let global = &self.0;
        if let Err(cause) =
            wgc::gfx_select!(encoder => global.command_encoder_insert_debug_marker(*encoder, label))
        {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::insert_debug_marker",
            );
        }
    }

    fn command_encoder_push_debug_group(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        label: &str,
    ) {
        let global = &self.0;
        if let Err(cause) =
            wgc::gfx_select!(encoder => global.command_encoder_push_debug_group(*encoder, label))
        {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::push_debug_group",
            );
        }
    }

    fn command_encoder_pop_debug_group(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
    ) {
        let global = &self.0;
        if let Err(cause) =
            wgc::gfx_select!(encoder => global.command_encoder_pop_debug_group(*encoder))
        {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::pop_debug_group",
            );
        }
    }

    fn command_encoder_write_timestamp(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder => global.command_encoder_write_timestamp(
            *encoder,
            *query_set,
            query_index
        )) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::write_timestamp",
            );
        }
    }

    fn command_encoder_resolve_query_set(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        first_query: u32,
        query_count: u32,
        destination: &Self::BufferId,
        _destination_data: &Self::BufferData,
        destination_offset: wgt::BufferAddress,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder => global.command_encoder_resolve_query_set(
            *encoder,
            *query_set,
            first_query,
            query_count,
            *destination,
            destination_offset
        )) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::resolve_query_set",
            );
        }
    }

    fn render_bundle_encoder_finish(
        &self,
        _encoder: Self::RenderBundleEncoderId,
        encoder_data: Self::RenderBundleEncoderData,
        desc: &crate::RenderBundleDescriptor,
    ) -> (Self::RenderBundleId, Self::RenderBundleData) {
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(encoder_data.parent() => global.render_bundle_encoder_finish(
            encoder_data,
            &desc.map_label(|l| l.map(Borrowed)),
            ()
        ));
        if let Some(err) = error {
            self.handle_error_fatal(err, "RenderBundleEncoder::finish");
        }
        (id, ())
    }

    fn queue_write_buffer(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        data: &[u8],
    ) {
        let global = &self.0;
        match wgc::gfx_select!(
            *queue => global.queue_write_buffer(*queue, *buffer, offset, data)
        ) {
            Ok(()) => (),
            Err(err) => {
                self.handle_error_nolabel(&queue_data.error_sink, err, "Queue::write_buffer")
            }
        }
    }

    fn queue_validate_write_buffer(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) -> Option<()> {
        let global = &self.0;
        match wgc::gfx_select!(
            *queue => global.queue_validate_write_buffer(*queue, *buffer, offset, size.get())
        ) {
            Ok(()) => Some(()),
            Err(err) => {
                self.handle_error_nolabel(&queue_data.error_sink, err, "Queue::write_buffer_with");
                None
            }
        }
    }

    fn queue_create_staging_buffer(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        size: wgt::BufferSize,
    ) -> Option<Box<dyn crate::context::QueueWriteBuffer>> {
        let global = &self.0;
        match wgc::gfx_select!(
            *queue => global.queue_create_staging_buffer(*queue, size, ())
        ) {
            Ok((buffer_id, ptr)) => Some(Box::new(QueueWriteBuffer {
                buffer_id,
                mapping: BufferMappedRange {
                    ptr,
                    size: size.get() as usize,
                },
            })),
            Err(err) => {
                self.handle_error_nolabel(&queue_data.error_sink, err, "Queue::write_buffer_with");
                None
            }
        }
    }

    fn queue_write_staging_buffer(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        staging_buffer: &dyn crate::context::QueueWriteBuffer,
    ) {
        let global = &self.0;
        let staging_buffer = staging_buffer
            .as_any()
            .downcast_ref::<QueueWriteBuffer>()
            .unwrap();
        match wgc::gfx_select!(
            *queue => global.queue_write_staging_buffer(*queue, *buffer, offset, staging_buffer.buffer_id)
        ) {
            Ok(()) => (),
            Err(err) => {
                self.handle_error_nolabel(&queue_data.error_sink, err, "Queue::write_buffer_with");
            }
        }
    }

    fn queue_write_texture(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        texture: crate::ImageCopyTexture,
        data: &[u8],
        data_layout: wgt::ImageDataLayout,
        size: wgt::Extent3d,
    ) {
        let global = &self.0;
        match wgc::gfx_select!(*queue => global.queue_write_texture(
            *queue,
            &map_texture_copy_view(texture),
            data,
            &data_layout,
            &size
        )) {
            Ok(()) => (),
            Err(err) => {
                self.handle_error_nolabel(&queue_data.error_sink, err, "Queue::write_texture")
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    fn queue_copy_external_image_to_texture(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        source: &wgt::ImageCopyExternalImage,
        dest: crate::ImageCopyTextureTagged,
        size: wgt::Extent3d,
    ) {
        let global = &self.0;
        match wgc::gfx_select!(*queue => global.queue_copy_external_image_to_texture(
            *queue,
            source,
            map_texture_tagged_copy_view(dest),
            size
        )) {
            Ok(()) => (),
            Err(err) => self.handle_error_nolabel(
                &queue_data.error_sink,
                err,
                "Queue::copy_external_image_to_texture",
            ),
        }
    }

    fn queue_submit<I: Iterator<Item = (Self::CommandBufferId, Self::CommandBufferData)>>(
        &self,
        queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        command_buffers: I,
    ) -> (Self::SubmissionIndex, Self::SubmissionIndexData) {
        let temp_command_buffers = command_buffers
            .map(|(i, _)| i)
            .collect::<SmallVec<[_; 4]>>();

        let global = &self.0;
        let index = match wgc::gfx_select!(*queue => global.queue_submit(*queue, &temp_command_buffers))
        {
            Ok(index) => index,
            Err(err) => self.handle_error_fatal(err, "Queue::submit"),
        };
        (Unused, index)
    }

    fn queue_get_timestamp_period(
        &self,
        queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
    ) -> f32 {
        let global = &self.0;
        let res = wgc::gfx_select!(queue => global.queue_get_timestamp_period(
            *queue
        ));
        match res {
            Ok(v) => v,
            Err(cause) => {
                self.handle_error_fatal(cause, "Queue::get_timestamp_period");
            }
        }
    }

    fn queue_on_submitted_work_done(
        &self,
        queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
        callback: Box<dyn FnOnce() + Send + 'static>,
    ) {
        let closure = wgc::device::queue::SubmittedWorkDoneClosure::from_rust(callback);

        let global = &self.0;
        let res = wgc::gfx_select!(queue => global.queue_on_submitted_work_done(*queue, closure));
        if let Err(cause) = res {
            self.handle_error_fatal(cause, "Queue::on_submitted_work_done");
        }
    }

    fn device_start_capture(&self, device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        let global = &self.0;
        wgc::gfx_select!(device => global.device_start_capture(*device));
    }

    fn device_stop_capture(&self, device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        let global = &self.0;
        wgc::gfx_select!(device => global.device_stop_capture(*device));
    }

    fn compute_pass_set_pipeline(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        pipeline: &Self::ComputePipelineId,
        _pipeline_data: &Self::ComputePipelineData,
    ) {
        wgpu_compute_pass_set_pipeline(pass_data, *pipeline)
    }

    fn compute_pass_set_bind_group(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        index: u32,
        bind_group: &Self::BindGroupId,
        _bind_group_data: &Self::BindGroupData,
        offsets: &[wgt::DynamicOffset],
    ) {
        unsafe {
            wgpu_compute_pass_set_bind_group(
                pass_data,
                index,
                *bind_group,
                offsets.as_ptr(),
                offsets.len(),
            )
        }
    }

    fn compute_pass_set_push_constants(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            wgpu_compute_pass_set_push_constant(
                pass_data,
                offset,
                data.len().try_into().unwrap(),
                data.as_ptr(),
            )
        }
    }

    fn compute_pass_insert_debug_marker(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        label: &str,
    ) {
        unsafe {
            let label = std::ffi::CString::new(label).unwrap();
            wgpu_compute_pass_insert_debug_marker(pass_data, label.as_ptr(), 0);
        }
    }

    fn compute_pass_push_debug_group(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        group_label: &str,
    ) {
        unsafe {
            let label = std::ffi::CString::new(group_label).unwrap();
            wgpu_compute_pass_push_debug_group(pass_data, label.as_ptr(), 0);
        }
    }

    fn compute_pass_pop_debug_group(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
    ) {
        wgpu_compute_pass_pop_debug_group(pass_data);
    }

    fn compute_pass_write_timestamp(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        wgpu_compute_pass_write_timestamp(pass_data, *query_set, query_index)
    }

    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        wgpu_compute_pass_begin_pipeline_statistics_query(pass_data, *query_set, query_index)
    }

    fn compute_pass_end_pipeline_statistics_query(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
    ) {
        wgpu_compute_pass_end_pipeline_statistics_query(pass_data)
    }

    fn compute_pass_dispatch_workgroups(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        x: u32,
        y: u32,
        z: u32,
    ) {
        wgpu_compute_pass_dispatch_workgroups(pass_data, x, y, z)
    }

    fn compute_pass_dispatch_workgroups_indirect(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        wgpu_compute_pass_dispatch_workgroups_indirect(pass_data, *indirect_buffer, indirect_offset)
    }

    fn render_bundle_encoder_set_pipeline(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
    ) {
        wgpu_render_bundle_set_pipeline(encoder_data, *pipeline)
    }

    fn render_bundle_encoder_set_bind_group(
        &self,
        __encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        index: u32,
        bind_group: &Self::BindGroupId,
        __bind_group_data: &Self::BindGroupData,
        offsets: &[wgt::DynamicOffset],
    ) {
        unsafe {
            wgpu_render_bundle_set_bind_group(
                encoder_data,
                index,
                *bind_group,
                offsets.as_ptr(),
                offsets.len(),
            )
        }
    }

    fn render_bundle_encoder_set_index_buffer(
        &self,
        __encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        buffer: &Self::BufferId,
        __buffer_data: &Self::BufferData,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        encoder_data.set_index_buffer(*buffer, index_format, offset, size)
    }

    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        __encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        slot: u32,
        buffer: &Self::BufferId,
        __buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        wgpu_render_bundle_set_vertex_buffer(encoder_data, slot, *buffer, offset, size)
    }

    fn render_bundle_encoder_set_push_constants(
        &self,
        __encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        stages: wgt::ShaderStages,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            wgpu_render_bundle_set_push_constants(
                encoder_data,
                stages,
                offset,
                data.len().try_into().unwrap(),
                data.as_ptr(),
            )
        }
    }

    fn render_bundle_encoder_draw(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) {
        wgpu_render_bundle_draw(
            encoder_data,
            vertices.end - vertices.start,
            instances.end - instances.start,
            vertices.start,
            instances.start,
        )
    }

    fn render_bundle_encoder_draw_indexed(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        wgpu_render_bundle_draw_indexed(
            encoder_data,
            indices.end - indices.start,
            instances.end - instances.start,
            indices.start,
            base_vertex,
            instances.start,
        )
    }

    fn render_bundle_encoder_draw_indirect(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        wgpu_render_bundle_draw_indirect(encoder_data, *indirect_buffer, indirect_offset)
    }

    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        wgpu_render_bundle_draw_indexed_indirect(encoder_data, *indirect_buffer, indirect_offset)
    }

    fn render_bundle_encoder_multi_draw_indirect(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count: u32,
    ) {
        unimplemented!()
    }

    fn render_bundle_encoder_multi_draw_indexed_indirect(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count: u32,
    ) {
        unimplemented!()
    }

    fn render_bundle_encoder_multi_draw_indirect_count(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count_buffer: &Self::BufferId,
        _count_buffer_data: &Self::BufferData,
        _count_buffer_offset: wgt::BufferAddress,
        _max_count: u32,
    ) {
        unimplemented!()
    }

    fn render_bundle_encoder_multi_draw_indexed_indirect_count(
        &self,
        _encoder: &mut Self::RenderBundleEncoderId,
        _encoder_data: &mut Self::RenderBundleEncoderData,
        _indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        _indirect_offset: wgt::BufferAddress,
        _count_buffer: &Self::BufferId,
        _count_buffer_data: &Self::BufferData,
        _count_buffer_offset: wgt::BufferAddress,
        _max_count: u32,
    ) {
        unimplemented!()
    }

    fn render_pass_set_pipeline(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
    ) {
        wgpu_render_pass_set_pipeline(pass_data, *pipeline)
    }

    fn render_pass_set_bind_group(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        index: u32,
        bind_group: &Self::BindGroupId,
        _bind_group_data: &Self::BindGroupData,
        offsets: &[wgt::DynamicOffset],
    ) {
        unsafe {
            wgpu_render_pass_set_bind_group(
                pass_data,
                index,
                *bind_group,
                offsets.as_ptr(),
                offsets.len(),
            )
        }
    }

    fn render_pass_set_index_buffer(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        pass_data.set_index_buffer(*buffer, index_format, offset, size)
    }

    fn render_pass_set_vertex_buffer(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        slot: u32,
        buffer: &Self::BufferId,
        _buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        wgpu_render_pass_set_vertex_buffer(pass_data, slot, *buffer, offset, size)
    }

    fn render_pass_set_push_constants(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        stages: wgt::ShaderStages,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            wgpu_render_pass_set_push_constants(
                pass_data,
                stages,
                offset,
                data.len().try_into().unwrap(),
                data.as_ptr(),
            )
        }
    }

    fn render_pass_draw(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) {
        wgpu_render_pass_draw(
            pass_data,
            vertices.end - vertices.start,
            instances.end - instances.start,
            vertices.start,
            instances.start,
        )
    }

    fn render_pass_draw_indexed(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        wgpu_render_pass_draw_indexed(
            pass_data,
            indices.end - indices.start,
            instances.end - instances.start,
            indices.start,
            base_vertex,
            instances.start,
        )
    }

    fn render_pass_draw_indirect(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        wgpu_render_pass_draw_indirect(pass_data, *indirect_buffer, indirect_offset)
    }

    fn render_pass_draw_indexed_indirect(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        wgpu_render_pass_draw_indexed_indirect(pass_data, *indirect_buffer, indirect_offset)
    }

    fn render_pass_multi_draw_indirect(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
        count: u32,
    ) {
        wgpu_render_pass_multi_draw_indirect(pass_data, *indirect_buffer, indirect_offset, count)
    }

    fn render_pass_multi_draw_indexed_indirect(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
        count: u32,
    ) {
        wgpu_render_pass_multi_draw_indexed_indirect(
            pass_data,
            *indirect_buffer,
            indirect_offset,
            count,
        )
    }

    fn render_pass_multi_draw_indirect_count(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
        count_buffer: &Self::BufferId,
        _count_buffer_data: &Self::BufferData,
        count_buffer_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        wgpu_render_pass_multi_draw_indirect_count(
            pass_data,
            *indirect_buffer,
            indirect_offset,
            *count_buffer,
            count_buffer_offset,
            max_count,
        )
    }

    fn render_pass_multi_draw_indexed_indirect_count(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
        count_buffer: &Self::BufferId,
        _count_buffer_data: &Self::BufferData,
        count_buffer_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        wgpu_render_pass_multi_draw_indexed_indirect_count(
            pass_data,
            *indirect_buffer,
            indirect_offset,
            *count_buffer,
            count_buffer_offset,
            max_count,
        )
    }

    fn render_pass_set_blend_constant(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        color: wgt::Color,
    ) {
        wgpu_render_pass_set_blend_constant(pass_data, &color)
    }

    fn render_pass_set_scissor_rect(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) {
        wgpu_render_pass_set_scissor_rect(pass_data, x, y, width, height)
    }

    fn render_pass_set_viewport(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        wgpu_render_pass_set_viewport(pass_data, x, y, width, height, min_depth, max_depth)
    }

    fn render_pass_set_stencil_reference(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        reference: u32,
    ) {
        wgpu_render_pass_set_stencil_reference(pass_data, reference)
    }

    fn render_pass_insert_debug_marker(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        label: &str,
    ) {
        unsafe {
            let label = std::ffi::CString::new(label).unwrap();
            wgpu_render_pass_insert_debug_marker(pass_data, label.as_ptr(), 0);
        }
    }

    fn render_pass_push_debug_group(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        group_label: &str,
    ) {
        unsafe {
            let label = std::ffi::CString::new(group_label).unwrap();
            wgpu_render_pass_push_debug_group(pass_data, label.as_ptr(), 0);
        }
    }

    fn render_pass_pop_debug_group(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
    ) {
        wgpu_render_pass_pop_debug_group(pass_data);
    }

    fn render_pass_write_timestamp(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        wgpu_render_pass_write_timestamp(pass_data, *query_set, query_index)
    }

    fn render_pass_begin_pipeline_statistics_query(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        wgpu_render_pass_begin_pipeline_statistics_query(pass_data, *query_set, query_index)
    }

    fn render_pass_end_pipeline_statistics_query(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
    ) {
        wgpu_render_pass_end_pipeline_statistics_query(pass_data)
    }

    fn render_pass_execute_bundles<'a>(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        render_bundles: Box<
            dyn Iterator<Item = (Self::RenderBundleId, &'a Self::RenderBundleData)> + 'a,
        >,
    ) {
        let temp_render_bundles = render_bundles.map(|(i, _)| i).collect::<SmallVec<[_; 4]>>();
        unsafe {
            wgpu_render_pass_execute_bundles(
                pass_data,
                temp_render_bundles.as_ptr(),
                temp_render_bundles.len(),
            )
        }
    }
}

impl<T> From<ObjectId> for wgc::id::Id<T> {
    fn from(id: ObjectId) -> Self {
        // If the id32 feature is enabled in wgpu-core, this will make sure that the id fits in a NonZeroU32.
        #[allow(clippy::useless_conversion)]
        let id = id.id().try_into().expect("Id exceeded 32-bits");
        // SAFETY: The id was created via the impl below
        unsafe { Self::from_raw(id) }
    }
}

impl<T> From<wgc::id::Id<T>> for ObjectId {
    fn from(id: wgc::id::Id<T>) -> Self {
        // If the id32 feature is enabled in wgpu-core, the conversion is not useless
        #[allow(clippy::useless_conversion)]
        let id = id.into_raw().into();
        Self::from_global_id(id)
    }
}

#[derive(Debug)]
pub struct SurfaceOutputDetail {
    surface_id: wgc::id::SurfaceId,
}

type ErrorSink = Arc<Mutex<ErrorSinkRaw>>;

struct ErrorScope {
    error: Option<crate::Error>,
    filter: crate::ErrorFilter,
}

struct ErrorSinkRaw {
    scopes: Vec<ErrorScope>,
    uncaptured_handler: Box<dyn crate::UncapturedErrorHandler>,
}

impl ErrorSinkRaw {
    fn new() -> ErrorSinkRaw {
        ErrorSinkRaw {
            scopes: Vec::new(),
            uncaptured_handler: Box::from(default_error_handler),
        }
    }

    fn handle_error(&mut self, err: crate::Error) {
        let filter = match err {
            crate::Error::OutOfMemory { .. } => crate::ErrorFilter::OutOfMemory,
            crate::Error::Validation { .. } => crate::ErrorFilter::Validation,
        };
        match self
            .scopes
            .iter_mut()
            .rev()
            .find(|scope| scope.filter == filter)
        {
            Some(scope) => {
                if scope.error.is_none() {
                    scope.error = Some(err);
                }
            }
            None => {
                (self.uncaptured_handler)(err);
            }
        }
    }
}

impl fmt::Debug for ErrorSinkRaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ErrorSink")
    }
}

fn default_error_handler(err: crate::Error) {
    log::error!("Handling wgpu errors as fatal by default");
    panic!("wgpu error: {err}\n");
}

#[derive(Debug)]
pub struct QueueWriteBuffer {
    buffer_id: wgc::id::StagingBufferId,
    mapping: BufferMappedRange,
}

impl crate::context::QueueWriteBuffer for QueueWriteBuffer {
    fn slice(&self) -> &[u8] {
        panic!()
    }

    #[inline]
    fn slice_mut(&mut self) -> &mut [u8] {
        use crate::context::BufferMappedRange;
        self.mapping.slice_mut()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
pub struct BufferMappedRange {
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for BufferMappedRange {}
unsafe impl Sync for BufferMappedRange {}

impl crate::context::BufferMappedRange for BufferMappedRange {
    #[inline]
    fn slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.size) }
    }

    #[inline]
    fn slice_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl Drop for BufferMappedRange {
    fn drop(&mut self) {
        // Intentionally left blank so that `BufferMappedRange` still
        // implements `Drop`, to match the web backend
    }
}
