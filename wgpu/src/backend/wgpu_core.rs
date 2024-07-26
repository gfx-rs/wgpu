use crate::{
    context::{ObjectId, Unused},
    AdapterInfo, BindGroupDescriptor, BindGroupLayoutDescriptor, BindingResource, BufferBinding,
    BufferDescriptor, CommandEncoderDescriptor, CompilationInfo, CompilationMessage,
    CompilationMessageType, ComputePassDescriptor, ComputePipelineDescriptor,
    DownlevelCapabilities, Features, Label, Limits, LoadOp, MapMode, Operations,
    PipelineCacheDescriptor, PipelineLayoutDescriptor, RenderBundleEncoderDescriptor,
    RenderPipelineDescriptor, SamplerDescriptor, ShaderModuleDescriptor,
    ShaderModuleDescriptorSpirV, ShaderSource, StoreOp, SurfaceStatus, SurfaceTargetUnsafe,
    TextureDescriptor, TextureViewDescriptor, UncapturedErrorHandler,
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
    ptr::NonNull,
    slice,
    sync::Arc,
};
use wgc::{
    command::bundle_ffi::*, device::DeviceLostClosure, gfx_select, id::CommandEncoderId,
    id::TextureViewId, pipeline::CreateShaderModuleError,
};
use wgt::WasmNotSendSync;

pub struct ContextWgpuCore(wgc::global::Global);

impl Drop for ContextWgpuCore {
    fn drop(&mut self) {
        //nothing
    }
}

impl fmt::Debug for ContextWgpuCore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextWgpuCore")
            .field("type", &"Native")
            .finish()
    }
}

impl ContextWgpuCore {
    pub unsafe fn from_hal_instance<A: wgc::hal_api::HalApi>(hal_instance: A::Instance) -> Self {
        Self(unsafe { wgc::global::Global::from_hal_instance::<A>("wgpu", hal_instance) })
    }

    /// # Safety
    ///
    /// - The raw instance handle returned must not be manually destroyed.
    pub unsafe fn instance_as_hal<A: wgc::hal_api::HalApi>(&self) -> Option<&A::Instance> {
        unsafe { self.0.instance_as_hal::<A>() }
    }

    pub unsafe fn from_core_instance(core_instance: wgc::instance::Instance) -> Self {
        Self(unsafe { wgc::global::Global::from_instance(core_instance) })
    }

    #[cfg(native)]
    pub fn enumerate_adapters(&self, backends: wgt::Backends) -> Vec<wgc::id::AdapterId> {
        self.0
            .enumerate_adapters(wgc::instance::AdapterInputs::Mask(backends, |_| None))
    }

    pub unsafe fn create_adapter_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        hal_adapter: hal::ExposedAdapter<A>,
    ) -> wgc::id::AdapterId {
        unsafe { self.0.create_adapter_from_hal(hal_adapter, None) }
    }

    pub unsafe fn adapter_as_hal<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&A::Adapter>) -> R,
        R,
    >(
        &self,
        adapter: wgc::id::AdapterId,
        hal_adapter_callback: F,
    ) -> R {
        unsafe {
            self.0
                .adapter_as_hal::<A, F, R>(adapter, hal_adapter_callback)
        }
    }

    pub unsafe fn buffer_as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Buffer>) -> R, R>(
        &self,
        id: wgc::id::BufferId,
        hal_buffer_callback: F,
    ) -> R {
        unsafe { self.0.buffer_as_hal::<A, F, R>(id, hal_buffer_callback) }
    }

    pub unsafe fn create_device_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        adapter: &wgc::id::AdapterId,
        hal_device: hal::OpenDevice<A>,
        desc: &crate::DeviceDescriptor<'_>,
        trace_dir: Option<&std::path::Path>,
    ) -> Result<(Device, Queue), crate::RequestDeviceError> {
        if trace_dir.is_some() {
            log::error!("Feature 'trace' has been removed temporarily, see https://github.com/gfx-rs/wgpu/issues/5974");
        }
        let (device_id, queue_id, error) = unsafe {
            self.0.create_device_from_hal(
                *adapter,
                hal_device,
                &desc.map_label(|l| l.map(Borrowed)),
                None,
                None,
                None,
            )
        };
        if let Some(err) = error {
            self.handle_error_fatal(err, "Adapter::create_device_from_hal");
        }
        let error_sink = Arc::new(Mutex::new(ErrorSinkRaw::new()));
        let device = Device {
            id: device_id,
            error_sink: error_sink.clone(),
            features: desc.required_features,
        };
        let queue = Queue {
            id: queue_id,
            error_sink,
        };
        Ok((device, queue))
    }

    pub unsafe fn create_texture_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        hal_texture: A::Texture,
        device: &Device,
        desc: &TextureDescriptor<'_>,
    ) -> Texture {
        let descriptor = desc.map_label_and_view_formats(|l| l.map(Borrowed), |v| v.to_vec());
        let (id, error) = unsafe {
            self.0
                .create_texture_from_hal::<A>(hal_texture, device.id, &descriptor, None)
        };
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                desc.label,
                "Device::create_texture_from_hal",
            );
        }
        Texture {
            id,
            error_sink: Arc::clone(&device.error_sink),
        }
    }

    pub unsafe fn create_buffer_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        hal_buffer: A::Buffer,
        device: &Device,
        desc: &BufferDescriptor<'_>,
    ) -> (wgc::id::BufferId, Buffer) {
        let (id, error) = unsafe {
            self.0.create_buffer_from_hal::<A>(
                hal_buffer,
                device.id,
                &desc.map_label(|l| l.map(Borrowed)),
                None,
            )
        };
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                desc.label,
                "Device::create_buffer_from_hal",
            );
        }
        (
            id,
            Buffer {
                error_sink: Arc::clone(&device.error_sink),
            },
        )
    }

    pub unsafe fn device_as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Device>) -> R, R>(
        &self,
        device: &Device,
        hal_device_callback: F,
    ) -> R {
        unsafe {
            self.0
                .device_as_hal::<A, F, R>(device.id, hal_device_callback)
        }
    }

    pub unsafe fn surface_as_hal<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&A::Surface>) -> R,
        R,
    >(
        &self,
        surface: &Surface,
        hal_surface_callback: F,
    ) -> R {
        unsafe {
            self.0
                .surface_as_hal::<A, F, R>(surface.id, hal_surface_callback)
        }
    }

    pub unsafe fn texture_as_hal<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&A::Texture>) -> R,
        R,
    >(
        &self,
        texture: &Texture,
        hal_texture_callback: F,
    ) -> R {
        unsafe {
            self.0
                .texture_as_hal::<A, F, R>(texture.id, hal_texture_callback)
        }
    }

    pub unsafe fn texture_view_as_hal<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&A::TextureView>) -> R,
        R,
    >(
        &self,
        texture_view_id: TextureViewId,
        hal_texture_view_callback: F,
    ) -> R {
        unsafe {
            self.0
                .texture_view_as_hal::<A, F, R>(texture_view_id, hal_texture_view_callback)
        }
    }

    /// This method will start the wgpu_core level command recording.
    pub unsafe fn command_encoder_as_hal_mut<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&mut A::CommandEncoder>) -> R,
        R,
    >(
        &self,
        command_encoder_id: CommandEncoderId,
        hal_command_encoder_callback: F,
    ) -> R {
        unsafe {
            self.0.command_encoder_as_hal_mut::<A, F, R>(
                command_encoder_id,
                hal_command_encoder_callback,
            )
        }
    }

    pub fn generate_report(&self) -> wgc::global::GlobalReport {
        self.0.generate_report()
    }

    fn handle_error(
        &self,
        sink_mutex: &Mutex<ErrorSinkRaw>,
        source: impl Error + WasmNotSendSync + 'static,
        label: Label<'_>,
        fn_ident: &'static str,
    ) {
        let error = wgc::error::ContextError {
            fn_ident,
            source: Box::new(source),
            label: label.unwrap_or_default().to_string(),
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
        source: impl Error + WasmNotSendSync + 'static,
        fn_ident: &'static str,
    ) {
        self.handle_error(sink_mutex, source, None, fn_ident)
    }

    #[track_caller]
    fn handle_error_fatal(
        &self,
        cause: impl Error + WasmNotSendSync + 'static,
        operation: &'static str,
    ) -> ! {
        panic!("Error in {operation}: {f}", f = self.format_error(&cause));
    }

    fn format_error(&self, err: &(impl Error + 'static)) -> String {
        let mut output = String::new();
        let mut level = 1;

        fn print_tree(output: &mut String, level: &mut usize, e: &(dyn Error + 'static)) {
            let mut print = |e: &(dyn Error + 'static)| {
                use std::fmt::Write;
                writeln!(output, "{}{}", " ".repeat(*level * 2), e).unwrap();

                if let Some(e) = e.source() {
                    *level += 1;
                    print_tree(output, level, e);
                    *level -= 1;
                }
            };
            if let Some(multi) = e.downcast_ref::<wgc::error::MultiError>() {
                for e in multi.errors() {
                    print(e);
                }
            } else {
                print(e);
            }
        }

        print_tree(&mut output, &mut level, err);

        format!("Validation Error\n\nCaused by:\n{}", output)
    }
}

fn map_buffer_copy_view(view: crate::ImageCopyBuffer<'_>) -> wgc::command::ImageCopyBuffer {
    wgc::command::ImageCopyBuffer {
        buffer: view.buffer.id.into(),
        layout: view.layout,
    }
}

fn map_texture_copy_view(view: crate::ImageCopyTexture<'_>) -> wgc::command::ImageCopyTexture {
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
    view: crate::ImageCopyTextureTagged<'_>,
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

fn map_store_op(op: StoreOp) -> wgc::command::StoreOp {
    match op {
        StoreOp::Store => wgc::command::StoreOp::Store,
        StoreOp::Discard => wgc::command::StoreOp::Discard,
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
            store_op: map_store_op(store),
            clear_value,
            read_only: false,
        },
        Some(&Operations {
            load: LoadOp::Load,
            store,
        }) => wgc::command::PassChannel {
            load_op: wgc::command::LoadOp::Load,
            store_op: map_store_op(store),
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
pub struct ShaderModule {
    compilation_info: CompilationInfo,
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
pub struct ComputePass {
    pass: Box<dyn wgc::command::DynComputePass>,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct RenderPass {
    pass: Box<dyn wgc::command::DynRenderPass>,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct CommandEncoder {
    error_sink: ErrorSink,
    open: bool,
}

impl crate::Context for ContextWgpuCore {
    type AdapterId = wgc::id::AdapterId;
    type AdapterData = ();
    type DeviceId = wgc::id::DeviceId;
    type DeviceData = Device;
    type QueueId = wgc::id::QueueId;
    type QueueData = Queue;
    type ShaderModuleId = wgc::id::ShaderModuleId;
    type ShaderModuleData = ShaderModule;
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
    type PipelineCacheId = wgc::id::PipelineCacheId;
    type PipelineCacheData = ();
    type CommandEncoderId = wgc::id::CommandEncoderId;
    type CommandEncoderData = CommandEncoder;
    type ComputePassId = Unused;
    type ComputePassData = ComputePass;
    type RenderPassId = Unused;
    type RenderPassData = RenderPass;
    type CommandBufferId = wgc::id::CommandBufferId;
    type CommandBufferData = ();
    type RenderBundleEncoderId = Unused;
    type RenderBundleEncoderData = wgc::command::RenderBundleEncoder;
    type RenderBundleId = wgc::id::RenderBundleId;
    type RenderBundleData = ();

    type SurfaceId = wgc::id::SurfaceId;
    type SurfaceData = Surface;
    type SurfaceOutputDetail = SurfaceOutputDetail;
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
    type CompilationInfoFuture = Ready<CompilationInfo>;

    fn init(instance_desc: wgt::InstanceDescriptor) -> Self {
        Self(wgc::global::Global::new("wgpu", instance_desc))
    }

    unsafe fn instance_create_surface(
        &self,
        target: SurfaceTargetUnsafe,
    ) -> Result<(Self::SurfaceId, Self::SurfaceData), crate::CreateSurfaceError> {
        let id = match target {
            SurfaceTargetUnsafe::RawHandle {
                raw_display_handle,
                raw_window_handle,
            } => unsafe {
                self.0
                    .instance_create_surface(raw_display_handle, raw_window_handle, None)
            },

            #[cfg(metal)]
            SurfaceTargetUnsafe::CoreAnimationLayer(layer) => unsafe {
                self.0.instance_create_surface_metal(layer, None)
            },

            #[cfg(dx12)]
            SurfaceTargetUnsafe::CompositionVisual(visual) => unsafe {
                self.0.instance_create_surface_from_visual(visual, None)
            },

            #[cfg(dx12)]
            SurfaceTargetUnsafe::SurfaceHandle(surface_handle) => unsafe {
                self.0
                    .instance_create_surface_from_surface_handle(surface_handle, None)
            },

            #[cfg(dx12)]
            SurfaceTargetUnsafe::SwapChainPanel(swap_chain_panel) => unsafe {
                self.0
                    .instance_create_surface_from_swap_chain_panel(swap_chain_panel, None)
            },
        }?;

        Ok((
            id,
            Surface {
                id,
                configured_device: Mutex::default(),
            },
        ))
    }

    fn instance_request_adapter(
        &self,
        options: &crate::RequestAdapterOptions<'_, '_>,
    ) -> Self::RequestAdapterFuture {
        let id = self.0.request_adapter(
            &wgc::instance::RequestAdapterOptions {
                power_preference: options.power_preference,
                force_fallback_adapter: options.force_fallback_adapter,
                compatible_surface: options.compatible_surface.map(|surface| surface.id.into()),
            },
            wgc::instance::AdapterInputs::Mask(wgt::Backends::all(), |_| None),
        );
        ready(id.ok().map(|id| (id, ())))
    }

    fn adapter_request_device(
        &self,
        adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
        desc: &crate::DeviceDescriptor<'_>,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture {
        if trace_dir.is_some() {
            log::error!("Feature 'trace' has been removed temporarily, see https://github.com/gfx-rs/wgpu/issues/5974");
        }
        let (device_id, queue_id, error) = wgc::gfx_select!(*adapter => self.0.adapter_request_device(
            *adapter,
            &desc.map_label(|l| l.map(Borrowed)),
            None,
            None,
            None
        ));
        if let Some(err) = error {
            return ready(Err(err.into()));
        }
        let error_sink = Arc::new(Mutex::new(ErrorSinkRaw::new()));
        let device = Device {
            id: device_id,
            error_sink: error_sink.clone(),
            features: desc.required_features,
        };
        let queue = Queue {
            id: queue_id,
            error_sink,
        };
        ready(Ok((device_id, device, device_id.into_queue_id(), queue)))
    }

    fn instance_poll_all_devices(&self, force_wait: bool) -> bool {
        match self.0.poll_all_devices(force_wait) {
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
        match wgc::gfx_select!(adapter => self.0.adapter_is_surface_supported(*adapter, *surface)) {
            Ok(result) => result,
            Err(err) => self.handle_error_fatal(err, "Adapter::is_surface_supported"),
        }
    }

    fn adapter_features(
        &self,
        adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> Features {
        match wgc::gfx_select!(*adapter => self.0.adapter_features(*adapter)) {
            Ok(features) => features,
            Err(err) => self.handle_error_fatal(err, "Adapter::features"),
        }
    }

    fn adapter_limits(
        &self,
        adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> Limits {
        match wgc::gfx_select!(*adapter => self.0.adapter_limits(*adapter)) {
            Ok(limits) => limits,
            Err(err) => self.handle_error_fatal(err, "Adapter::limits"),
        }
    }

    fn adapter_downlevel_capabilities(
        &self,
        adapter: &Self::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> DownlevelCapabilities {
        match wgc::gfx_select!(*adapter => self.0.adapter_downlevel_capabilities(*adapter)) {
            Ok(downlevel) => downlevel,
            Err(err) => self.handle_error_fatal(err, "Adapter::downlevel_properties"),
        }
    }

    fn adapter_get_info(
        &self,
        adapter: &wgc::id::AdapterId,
        _adapter_data: &Self::AdapterData,
    ) -> AdapterInfo {
        match wgc::gfx_select!(*adapter => self.0.adapter_get_info(*adapter)) {
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
        match wgc::gfx_select!(*adapter => self.0.adapter_get_texture_format_features(*adapter, format))
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
        match wgc::gfx_select!(*adapter => self.0.adapter_get_presentation_timestamp(*adapter)) {
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
        match wgc::gfx_select!(adapter => self.0.surface_get_capabilities(*surface, *adapter)) {
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
        let error = wgc::gfx_select!(device => self.0.surface_configure(*surface, *device, config));
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
        let device_id = surface_data
            .configured_device
            .lock()
            .expect("Surface was not configured?");
        match wgc::gfx_select!(
            device_id => self.0.surface_get_current_texture(*surface, None)
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
        match wgc::gfx_select!(texture => self.0.surface_present(detail.surface_id)) {
            Ok(_status) => (),
            Err(err) => self.handle_error_fatal(err, "Surface::present"),
        }
    }

    fn surface_texture_discard(
        &self,
        texture: &Self::TextureId,
        detail: &Self::SurfaceOutputDetail,
    ) {
        match wgc::gfx_select!(texture => self.0.surface_texture_discard(detail.surface_id)) {
            Ok(_status) => (),
            Err(err) => self.handle_error_fatal(err, "Surface::discard_texture"),
        }
    }

    fn device_features(
        &self,
        device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> Features {
        match wgc::gfx_select!(device => self.0.device_features(*device)) {
            Ok(features) => features,
            Err(err) => self.handle_error_fatal(err, "Device::features"),
        }
    }

    fn device_limits(&self, device: &Self::DeviceId, _device_data: &Self::DeviceData) -> Limits {
        match wgc::gfx_select!(device => self.0.device_limits(*device)) {
            Ok(limits) => limits,
            Err(err) => self.handle_error_fatal(err, "Device::limits"),
        }
    }

    fn device_downlevel_properties(
        &self,
        device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> DownlevelCapabilities {
        match wgc::gfx_select!(device => self.0.device_downlevel_properties(*device)) {
            Ok(limits) => limits,
            Err(err) => self.handle_error_fatal(err, "Device::downlevel_properties"),
        }
    }

    #[cfg_attr(
        not(any(
            feature = "spirv",
            feature = "glsl",
            feature = "wgsl",
            feature = "naga-ir"
        )),
        allow(unreachable_code, unused_variables)
    )]
    fn device_create_shader_module(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: ShaderModuleDescriptor<'_>,
        shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> (Self::ShaderModuleId, Self::ShaderModuleData) {
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
                wgc::pipeline::ShaderModuleSource::SpirV(Borrowed(spv), options)
            }
            #[cfg(feature = "glsl")]
            ShaderSource::Glsl {
                ref shader,
                stage,
                defines,
            } => {
                let options = naga::front::glsl::Options { stage, defines };
                wgc::pipeline::ShaderModuleSource::Glsl(Borrowed(shader), options)
            }
            #[cfg(feature = "wgsl")]
            ShaderSource::Wgsl(ref code) => wgc::pipeline::ShaderModuleSource::Wgsl(Borrowed(code)),
            #[cfg(feature = "naga-ir")]
            ShaderSource::Naga(module) => wgc::pipeline::ShaderModuleSource::Naga(module),
            ShaderSource::Dummy(_) => panic!("found `ShaderSource::Dummy`"),
        };
        let (id, error) = wgc::gfx_select!(
            device => self.0.device_create_shader_module(*device, &descriptor, source, None)
        );
        let compilation_info = match error {
            Some(cause) => {
                self.handle_error(
                    &device_data.error_sink,
                    cause.clone(),
                    desc.label,
                    "Device::create_shader_module",
                );
                CompilationInfo::from(cause)
            }
            None => CompilationInfo { messages: vec![] },
        };

        (id, ShaderModule { compilation_info })
    }

    unsafe fn device_create_shader_module_spirv(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &ShaderModuleDescriptorSpirV<'_>,
    ) -> (Self::ShaderModuleId, Self::ShaderModuleData) {
        let descriptor = wgc::pipeline::ShaderModuleDescriptor {
            label: desc.label.map(Borrowed),
            // Doesn't matter the value since spirv shaders aren't mutated to include
            // runtime checks
            shader_bound_checks: unsafe { wgt::ShaderBoundChecks::unchecked() },
        };
        let (id, error) = wgc::gfx_select!(
            device => self.0.device_create_shader_module_spirv(*device, &descriptor, Borrowed(&desc.source), None)
        );
        let compilation_info = match error {
            Some(cause) => {
                self.handle_error(
                    &device_data.error_sink,
                    cause.clone(),
                    desc.label,
                    "Device::create_shader_module_spirv",
                );
                CompilationInfo::from(cause)
            }
            None => CompilationInfo { messages: vec![] },
        };
        (id, ShaderModule { compilation_info })
    }

    fn device_create_bind_group_layout(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &BindGroupLayoutDescriptor<'_>,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        let descriptor = wgc::binding_model::BindGroupLayoutDescriptor {
            label: desc.label.map(Borrowed),
            entries: Borrowed(desc.entries),
        };
        let (id, error) = wgc::gfx_select!(
            device => self.0.device_create_bind_group_layout(*device, &descriptor, None)
        );
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
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
        desc: &BindGroupDescriptor<'_>,
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

        let (id, error) = wgc::gfx_select!(device => self.0.device_create_bind_group(
            *device,
            &descriptor,
            None
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
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
        desc: &PipelineLayoutDescriptor<'_>,
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

        let (id, error) = wgc::gfx_select!(device => self.0.device_create_pipeline_layout(
            *device,
            &descriptor,
            None
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
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
        desc: &RenderPipelineDescriptor<'_>,
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

        let descriptor = pipe::RenderPipelineDescriptor {
            label: desc.label.map(Borrowed),
            layout: desc.layout.map(|l| l.id.into()),
            vertex: pipe::VertexState {
                stage: pipe::ProgrammableStageDescriptor {
                    module: desc.vertex.module.id.into(),
                    entry_point: Some(Borrowed(desc.vertex.entry_point)),
                    constants: Borrowed(desc.vertex.compilation_options.constants),
                    zero_initialize_workgroup_memory: desc
                        .vertex
                        .compilation_options
                        .zero_initialize_workgroup_memory,
                    vertex_pulling_transform: desc
                        .vertex
                        .compilation_options
                        .vertex_pulling_transform,
                },
                buffers: Borrowed(&vertex_buffers),
            },
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment: desc.fragment.as_ref().map(|frag| pipe::FragmentState {
                stage: pipe::ProgrammableStageDescriptor {
                    module: frag.module.id.into(),
                    entry_point: Some(Borrowed(frag.entry_point)),
                    constants: Borrowed(frag.compilation_options.constants),
                    zero_initialize_workgroup_memory: frag
                        .compilation_options
                        .zero_initialize_workgroup_memory,
                    vertex_pulling_transform: false,
                },
                targets: Borrowed(frag.targets),
            }),
            multiview: desc.multiview,
            cache: desc.cache.map(|c| c.id.into()),
        };

        let (id, error) = wgc::gfx_select!(device => self.0.device_create_render_pipeline(
            *device,
            &descriptor,
            None,
            None,
        ));
        if let Some(cause) = error {
            if let wgc::pipeline::CreateRenderPipelineError::Internal { stage, ref error } = cause {
                log::error!("Shader translation error for stage {:?}: {}", stage, error);
                log::error!("Please report it to https://github.com/gfx-rs/wgpu");
            }
            self.handle_error(
                &device_data.error_sink,
                cause,
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
        desc: &ComputePipelineDescriptor<'_>,
    ) -> (Self::ComputePipelineId, Self::ComputePipelineData) {
        use wgc::pipeline as pipe;

        let descriptor = pipe::ComputePipelineDescriptor {
            label: desc.label.map(Borrowed),
            layout: desc.layout.map(|l| l.id.into()),
            stage: pipe::ProgrammableStageDescriptor {
                module: desc.module.id.into(),
                entry_point: Some(Borrowed(desc.entry_point)),
                constants: Borrowed(desc.compilation_options.constants),
                zero_initialize_workgroup_memory: desc
                    .compilation_options
                    .zero_initialize_workgroup_memory,
                vertex_pulling_transform: false,
            },
            cache: desc.cache.map(|c| c.id.into()),
        };

        let (id, error) = wgc::gfx_select!(device => self.0.device_create_compute_pipeline(
            *device,
            &descriptor,
            None,
            None,
        ));
        if let Some(cause) = error {
            if let wgc::pipeline::CreateComputePipelineError::Internal(ref error) = cause {
                log::error!(
                    "Shader translation error for stage {:?}: {}",
                    wgt::ShaderStages::COMPUTE,
                    error
                );
                log::error!("Please report it to https://github.com/gfx-rs/wgpu");
            }
            self.handle_error(
                &device_data.error_sink,
                cause,
                desc.label,
                "Device::create_compute_pipeline",
            );
        }
        (id, ())
    }

    unsafe fn device_create_pipeline_cache(
        &self,
        device: &Self::DeviceId,
        device_data: &Self::DeviceData,
        desc: &PipelineCacheDescriptor<'_>,
    ) -> (Self::PipelineCacheId, Self::PipelineCacheData) {
        use wgc::pipeline as pipe;

        let descriptor = pipe::PipelineCacheDescriptor {
            label: desc.label.map(Borrowed),
            data: desc.data.map(Borrowed),
            fallback: desc.fallback,
        };
        let (id, error) = wgc::gfx_select!(device => self.0.device_create_pipeline_cache(
            *device,
            &descriptor,
            None
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                desc.label,
                "Device::device_create_pipeline_cache_init",
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
        let (id, error) = wgc::gfx_select!(device => self.0.device_create_buffer(
            *device,
            &desc.map_label(|l| l.map(Borrowed)),
            None
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
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
        desc: &TextureDescriptor<'_>,
    ) -> (Self::TextureId, Self::TextureData) {
        let wgt_desc = desc.map_label_and_view_formats(|l| l.map(Borrowed), |v| v.to_vec());
        let (id, error) = wgc::gfx_select!(device => self.0.device_create_texture(
            *device,
            &wgt_desc,
            None
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
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
        desc: &SamplerDescriptor<'_>,
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

        let (id, error) = wgc::gfx_select!(device => self.0.device_create_sampler(
            *device,
            &descriptor,
            None
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
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
        desc: &wgt::QuerySetDescriptor<Label<'_>>,
    ) -> (Self::QuerySetId, Self::QuerySetData) {
        let (id, error) = wgc::gfx_select!(device => self.0.device_create_query_set(
            *device,
            &desc.map_label(|l| l.map(Borrowed)),
            None
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
        desc: &CommandEncoderDescriptor<'_>,
    ) -> (Self::CommandEncoderId, Self::CommandEncoderData) {
        let (id, error) = wgc::gfx_select!(device => self.0.device_create_command_encoder(
            *device,
            &desc.map_label(|l| l.map(Borrowed)),
            None
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
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
        desc: &RenderBundleEncoderDescriptor<'_>,
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
    #[doc(hidden)]
    fn device_make_invalid(&self, device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        wgc::gfx_select!(device => self.0.device_make_invalid(*device));
    }
    #[cfg_attr(not(any(native, Emscripten)), allow(unused))]
    fn device_drop(&self, device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        #[cfg(any(native, Emscripten))]
        {
            // Call device_poll, but don't check for errors. We have to use its
            // return value, but we just drop it.
            let _ = wgc::gfx_select!(device => self.0.device_poll(*device, wgt::Maintain::wait()));
            wgc::gfx_select!(device => self.0.device_drop(*device));
        }
    }
    #[cfg_attr(target_arch = "wasm32", allow(unused))]
    fn queue_drop(&self, queue: &Self::QueueId, _device_data: &Self::QueueData) {
        wgc::gfx_select!(queue => self.0.queue_drop(*queue));
    }
    fn device_set_device_lost_callback(
        &self,
        device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        device_lost_callback: crate::context::DeviceLostCallback,
    ) {
        let device_lost_closure = DeviceLostClosure::from_rust(device_lost_callback);
        wgc::gfx_select!(device => self.0.device_set_device_lost_closure(*device, device_lost_closure));
    }
    fn device_destroy(&self, device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        wgc::gfx_select!(device => self.0.device_destroy(*device));
    }
    fn device_mark_lost(
        &self,
        device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        message: &str,
    ) {
        // We do not provide a reason to device_lose, because all reasons other than
        // destroyed (which this is not) are "unknown".
        wgc::gfx_select!(device => self.0.device_mark_lost(*device, message));
    }
    fn device_poll(
        &self,
        device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
        maintain: crate::Maintain,
    ) -> wgt::MaintainResult {
        let maintain_inner = maintain.map_index(|i| *i.0.as_ref().downcast_ref().unwrap());
        match wgc::gfx_select!(device => self.0.device_poll(
            *device,
            maintain_inner
        )) {
            Ok(done) => match done {
                true => wgt::MaintainResult::SubmissionQueueEmpty,
                false => wgt::MaintainResult::Ok,
            },
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
        callback: crate::context::BufferMapCallback,
    ) {
        let operation = wgc::resource::BufferMapOperation {
            host: match mode {
                MapMode::Read => wgc::device::HostMap::Read,
                MapMode::Write => wgc::device::HostMap::Write,
            },
            callback: Some(wgc::resource::BufferMapCallback::from_rust(Box::new(
                |status| {
                    let res = status.map_err(|_| crate::BufferAsyncError);
                    callback(res);
                },
            ))),
        };

        match wgc::gfx_select!(buffer => self.0.buffer_map_async(*buffer, range.start, Some(range.end-range.start), operation))
        {
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
        match wgc::gfx_select!(buffer => self.0.buffer_get_mapped_range(
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
        match wgc::gfx_select!(buffer => self.0.buffer_unmap(*buffer)) {
            Ok(()) => (),
            Err(cause) => {
                self.handle_error_nolabel(&buffer_data.error_sink, cause, "Buffer::buffer_unmap")
            }
        }
    }

    fn shader_get_compilation_info(
        &self,
        _shader: &Self::ShaderModuleId,
        shader_data: &Self::ShaderModuleData,
    ) -> Self::CompilationInfoFuture {
        ready(shader_data.compilation_info.clone())
    }

    fn texture_create_view(
        &self,
        texture: &Self::TextureId,
        texture_data: &Self::TextureData,
        desc: &TextureViewDescriptor<'_>,
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
        let (id, error) = wgc::gfx_select!(
            texture => self.0.texture_create_view(*texture, &descriptor, None)
        );
        if let Some(cause) = error {
            self.handle_error(
                &texture_data.error_sink,
                cause,
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
        wgc::gfx_select!(*adapter => self.0.adapter_drop(*adapter))
    }

    fn buffer_destroy(&self, buffer: &Self::BufferId, _buffer_data: &Self::BufferData) {
        // Per spec, no error to report. Even calling destroy multiple times is valid.
        let _ = wgc::gfx_select!(buffer => self.0.buffer_destroy(*buffer));
    }

    fn buffer_drop(&self, buffer: &Self::BufferId, _buffer_data: &Self::BufferData) {
        wgc::gfx_select!(buffer => self.0.buffer_drop(*buffer, false))
    }

    fn texture_destroy(&self, texture: &Self::TextureId, _texture_data: &Self::TextureData) {
        // Per spec, no error to report. Even calling destroy multiple times is valid.
        let _ = wgc::gfx_select!(texture => self.0.texture_destroy(*texture));
    }

    fn texture_drop(&self, texture: &Self::TextureId, _texture_data: &Self::TextureData) {
        wgc::gfx_select!(texture => self.0.texture_drop(*texture, false))
    }

    fn texture_view_drop(
        &self,
        texture_view: &Self::TextureViewId,
        __texture_view_data: &Self::TextureViewData,
    ) {
        let _ = wgc::gfx_select!(*texture_view => self.0.texture_view_drop(*texture_view, false));
    }

    fn sampler_drop(&self, sampler: &Self::SamplerId, _sampler_data: &Self::SamplerData) {
        wgc::gfx_select!(*sampler => self.0.sampler_drop(*sampler))
    }

    fn query_set_drop(&self, query_set: &Self::QuerySetId, _query_set_data: &Self::QuerySetData) {
        wgc::gfx_select!(*query_set => self.0.query_set_drop(*query_set))
    }

    fn bind_group_drop(
        &self,
        bind_group: &Self::BindGroupId,
        _bind_group_data: &Self::BindGroupData,
    ) {
        wgc::gfx_select!(*bind_group => self.0.bind_group_drop(*bind_group))
    }

    fn bind_group_layout_drop(
        &self,
        bind_group_layout: &Self::BindGroupLayoutId,
        _bind_group_layout_data: &Self::BindGroupLayoutData,
    ) {
        wgc::gfx_select!(*bind_group_layout => self.0.bind_group_layout_drop(*bind_group_layout))
    }

    fn pipeline_layout_drop(
        &self,
        pipeline_layout: &Self::PipelineLayoutId,
        _pipeline_layout_data: &Self::PipelineLayoutData,
    ) {
        wgc::gfx_select!(*pipeline_layout => self.0.pipeline_layout_drop(*pipeline_layout))
    }
    fn shader_module_drop(
        &self,
        shader_module: &Self::ShaderModuleId,
        _shader_module_data: &Self::ShaderModuleData,
    ) {
        wgc::gfx_select!(*shader_module => self.0.shader_module_drop(*shader_module))
    }
    fn command_encoder_drop(
        &self,
        command_encoder: &Self::CommandEncoderId,
        command_encoder_data: &Self::CommandEncoderData,
    ) {
        if command_encoder_data.open {
            wgc::gfx_select!(command_encoder => self.0.command_encoder_drop(*command_encoder))
        }
    }

    fn command_buffer_drop(
        &self,
        command_buffer: &Self::CommandBufferId,
        _command_buffer_data: &Self::CommandBufferData,
    ) {
        wgc::gfx_select!(*command_buffer => self.0.command_buffer_drop(*command_buffer))
    }

    fn render_bundle_drop(
        &self,
        render_bundle: &Self::RenderBundleId,
        _render_bundle_data: &Self::RenderBundleData,
    ) {
        wgc::gfx_select!(*render_bundle => self.0.render_bundle_drop(*render_bundle))
    }

    fn compute_pipeline_drop(
        &self,
        pipeline: &Self::ComputePipelineId,
        _pipeline_data: &Self::ComputePipelineData,
    ) {
        wgc::gfx_select!(*pipeline => self.0.compute_pipeline_drop(*pipeline))
    }

    fn render_pipeline_drop(
        &self,
        pipeline: &Self::RenderPipelineId,
        _pipeline_data: &Self::RenderPipelineData,
    ) {
        wgc::gfx_select!(*pipeline => self.0.render_pipeline_drop(*pipeline))
    }

    fn pipeline_cache_drop(
        &self,
        cache: &Self::PipelineCacheId,
        _cache_data: &Self::PipelineCacheData,
    ) {
        wgc::gfx_select!(*cache => self.0.pipeline_cache_drop(*cache))
    }

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline: &Self::ComputePipelineId,
        _pipeline_data: &Self::ComputePipelineData,
        index: u32,
    ) -> (Self::BindGroupLayoutId, Self::BindGroupLayoutData) {
        let (id, error) = wgc::gfx_select!(*pipeline => self.0.compute_pipeline_get_bind_group_layout(*pipeline, index, None));
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
        let (id, error) = wgc::gfx_select!(*pipeline => self.0.render_pipeline_get_bind_group_layout(*pipeline, index, None));
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
        if let Err(cause) = wgc::gfx_select!(encoder => self.0.command_encoder_copy_buffer_to_buffer(
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
        source: crate::ImageCopyBuffer<'_>,
        destination: crate::ImageCopyTexture<'_>,
        copy_size: wgt::Extent3d,
    ) {
        if let Err(cause) = wgc::gfx_select!(encoder => self.0.command_encoder_copy_buffer_to_texture(
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
        source: crate::ImageCopyTexture<'_>,
        destination: crate::ImageCopyBuffer<'_>,
        copy_size: wgt::Extent3d,
    ) {
        if let Err(cause) = wgc::gfx_select!(encoder => self.0.command_encoder_copy_texture_to_buffer(
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
        source: crate::ImageCopyTexture<'_>,
        destination: crate::ImageCopyTexture<'_>,
        copy_size: wgt::Extent3d,
    ) {
        if let Err(cause) = wgc::gfx_select!(encoder => self.0.command_encoder_copy_texture_to_texture(
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
        encoder_data: &Self::CommandEncoderData,
        desc: &ComputePassDescriptor<'_>,
    ) -> (Self::ComputePassId, Self::ComputePassData) {
        let timestamp_writes =
            desc.timestamp_writes
                .as_ref()
                .map(|tw| wgc::command::PassTimestampWrites {
                    query_set: tw.query_set.id.into(),
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                });

        let (pass, err) = gfx_select!(encoder => self.0.command_encoder_create_compute_pass_dyn(*encoder, &wgc::command::ComputePassDescriptor {
            label: desc.label.map(Borrowed),
            timestamp_writes: timestamp_writes.as_ref(),
        }));

        if let Some(cause) = err {
            self.handle_error(
                &encoder_data.error_sink,
                cause,
                desc.label,
                "CommandEncoder::begin_compute_pass",
            );
        }

        (
            Unused,
            Self::ComputePassData {
                pass,
                error_sink: encoder_data.error_sink.clone(),
            },
        )
    }

    fn command_encoder_begin_render_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        encoder_data: &Self::CommandEncoderData,
        desc: &crate::RenderPassDescriptor<'_>,
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

        let timestamp_writes =
            desc.timestamp_writes
                .as_ref()
                .map(|tw| wgc::command::PassTimestampWrites {
                    query_set: tw.query_set.id.into(),
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                });

        let (pass, err) = gfx_select!(encoder => self.0.command_encoder_create_render_pass_dyn(*encoder, &wgc::command::RenderPassDescriptor {
            label: desc.label.map(Borrowed),
            timestamp_writes: timestamp_writes.as_ref(),
            color_attachments: std::borrow::Cow::Borrowed(&colors),
            depth_stencil_attachment: depth_stencil.as_ref(),
            occlusion_query_set: desc.occlusion_query_set.map(|query_set| query_set.id.into()),
        }));

        if let Some(cause) = err {
            self.handle_error(
                &encoder_data.error_sink,
                cause,
                desc.label,
                "CommandEncoder::begin_render_pass",
            );
        }

        (
            Unused,
            Self::RenderPassData {
                pass,
                error_sink: encoder_data.error_sink.clone(),
            },
        )
    }

    fn command_encoder_finish(
        &self,
        encoder: Self::CommandEncoderId,
        encoder_data: &mut Self::CommandEncoderData,
    ) -> (Self::CommandBufferId, Self::CommandBufferData) {
        let descriptor = wgt::CommandBufferDescriptor::default();
        encoder_data.open = false; // prevent the drop
        let (id, error) =
            wgc::gfx_select!(encoder => self.0.command_encoder_finish(encoder, &descriptor));
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
        if let Err(cause) = wgc::gfx_select!(encoder => self.0.command_encoder_clear_texture(
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
        size: Option<wgt::BufferAddress>,
    ) {
        if let Err(cause) = wgc::gfx_select!(encoder => self.0.command_encoder_clear_buffer(
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
        if let Err(cause) =
            wgc::gfx_select!(encoder => self.0.command_encoder_insert_debug_marker(*encoder, label))
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
        if let Err(cause) =
            wgc::gfx_select!(encoder => self.0.command_encoder_push_debug_group(*encoder, label))
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
        if let Err(cause) =
            wgc::gfx_select!(encoder => self.0.command_encoder_pop_debug_group(*encoder))
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
        if let Err(cause) = wgc::gfx_select!(encoder => self.0.command_encoder_write_timestamp(
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
        if let Err(cause) = wgc::gfx_select!(encoder => self.0.command_encoder_resolve_query_set(
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
        desc: &crate::RenderBundleDescriptor<'_>,
    ) -> (Self::RenderBundleId, Self::RenderBundleData) {
        let (id, error) = wgc::gfx_select!(encoder_data.parent() => self.0.render_bundle_encoder_finish(
            encoder_data,
            &desc.map_label(|l| l.map(Borrowed)),
            None
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
        match wgc::gfx_select!(
            *queue => self.0.queue_write_buffer(*queue, *buffer, offset, data)
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
        match wgc::gfx_select!(
            *queue => self.0.queue_validate_write_buffer(*queue, *buffer, offset, size)
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
        match wgc::gfx_select!(
            *queue => self.0.queue_create_staging_buffer(*queue, size, None)
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
        let staging_buffer = staging_buffer
            .as_any()
            .downcast_ref::<QueueWriteBuffer>()
            .unwrap();
        match wgc::gfx_select!(
            *queue => self.0.queue_write_staging_buffer(*queue, *buffer, offset, staging_buffer.buffer_id)
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
        texture: crate::ImageCopyTexture<'_>,
        data: &[u8],
        data_layout: wgt::ImageDataLayout,
        size: wgt::Extent3d,
    ) {
        match wgc::gfx_select!(*queue => self.0.queue_write_texture(
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

    #[cfg(any(webgpu, webgl))]
    fn queue_copy_external_image_to_texture(
        &self,
        queue: &Self::QueueId,
        queue_data: &Self::QueueData,
        source: &wgt::ImageCopyExternalImage,
        dest: crate::ImageCopyTextureTagged<'_>,
        size: wgt::Extent3d,
    ) {
        match wgc::gfx_select!(*queue => self.0.queue_copy_external_image_to_texture(
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
    ) -> Self::SubmissionIndexData {
        let temp_command_buffers = command_buffers
            .map(|(i, _)| i)
            .collect::<SmallVec<[_; 4]>>();

        let index = match wgc::gfx_select!(*queue => self.0.queue_submit(*queue, &temp_command_buffers))
        {
            Ok(index) => index,
            Err(err) => self.handle_error_fatal(err, "Queue::submit"),
        };

        for cmdbuf in &temp_command_buffers {
            wgc::gfx_select!(*queue => self.0.command_buffer_drop(*cmdbuf));
        }

        index
    }

    fn queue_get_timestamp_period(
        &self,
        queue: &Self::QueueId,
        _queue_data: &Self::QueueData,
    ) -> f32 {
        let res = wgc::gfx_select!(queue => self.0.queue_get_timestamp_period(
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
        callback: crate::context::SubmittedWorkDoneCallback,
    ) {
        let closure = wgc::device::queue::SubmittedWorkDoneClosure::from_rust(callback);

        let res = wgc::gfx_select!(queue => self.0.queue_on_submitted_work_done(*queue, closure));
        if let Err(cause) = res {
            self.handle_error_fatal(cause, "Queue::on_submitted_work_done");
        }
    }

    fn device_start_capture(&self, device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        wgc::gfx_select!(device => self.0.device_start_capture(*device));
    }

    fn device_stop_capture(&self, device: &Self::DeviceId, _device_data: &Self::DeviceData) {
        wgc::gfx_select!(device => self.0.device_stop_capture(*device));
    }

    fn device_get_internal_counters(
        &self,
        device: &Self::DeviceId,
        _device_data: &Self::DeviceData,
    ) -> wgt::InternalCounters {
        wgc::gfx_select!(device => self.0.device_get_internal_counters(*device))
    }

    fn pipeline_cache_get_data(
        &self,
        cache: &Self::PipelineCacheId,
        // TODO: Used for error handling?
        _cache_data: &Self::PipelineCacheData,
    ) -> Option<Vec<u8>> {
        wgc::gfx_select!(cache => self.0.pipeline_cache_get_data(*cache))
    }

    fn compute_pass_set_pipeline(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        pipeline: &Self::ComputePipelineId,
        _pipeline_data: &Self::ComputePipelineData,
    ) {
        if let Err(cause) = pass_data.pass.set_pipeline(&self.0, *pipeline) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::set_pipeline",
            );
        }
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
        if let Err(cause) = pass_data
            .pass
            .set_bind_group(&self.0, index, *bind_group, offsets)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::set_bind_group",
            );
        }
    }

    fn compute_pass_set_push_constants(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        offset: u32,
        data: &[u8],
    ) {
        if let Err(cause) = pass_data.pass.set_push_constants(&self.0, offset, data) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::set_push_constant",
            );
        }
    }

    fn compute_pass_insert_debug_marker(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        label: &str,
    ) {
        if let Err(cause) = pass_data.pass.insert_debug_marker(&self.0, label, 0) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::insert_debug_marker",
            );
        }
    }

    fn compute_pass_push_debug_group(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        group_label: &str,
    ) {
        if let Err(cause) = pass_data.pass.push_debug_group(&self.0, group_label, 0) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::push_debug_group",
            );
        }
    }

    fn compute_pass_pop_debug_group(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
    ) {
        if let Err(cause) = pass_data.pass.pop_debug_group(&self.0) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::pop_debug_group",
            );
        }
    }

    fn compute_pass_write_timestamp(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        if let Err(cause) = pass_data
            .pass
            .write_timestamp(&self.0, *query_set, query_index)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::write_timestamp",
            );
        }
    }

    fn compute_pass_begin_pipeline_statistics_query(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        if let Err(cause) =
            pass_data
                .pass
                .begin_pipeline_statistics_query(&self.0, *query_set, query_index)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::begin_pipeline_statistics_query",
            );
        }
    }

    fn compute_pass_end_pipeline_statistics_query(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
    ) {
        if let Err(cause) = pass_data.pass.end_pipeline_statistics_query(&self.0) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::end_pipeline_statistics_query",
            );
        }
    }

    fn compute_pass_dispatch_workgroups(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        x: u32,
        y: u32,
        z: u32,
    ) {
        if let Err(cause) = pass_data.pass.dispatch_workgroups(&self.0, x, y, z) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::dispatch_workgroups",
            );
        }
    }

    fn compute_pass_dispatch_workgroups_indirect(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
        indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        if let Err(cause) =
            pass_data
                .pass
                .dispatch_workgroups_indirect(&self.0, *indirect_buffer, indirect_offset)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::dispatch_workgroups_indirect",
            );
        }
    }

    fn compute_pass_end(
        &self,
        _pass: &mut Self::ComputePassId,
        pass_data: &mut Self::ComputePassData,
    ) {
        if let Err(cause) = pass_data.pass.end(&self.0) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::end",
            );
        }
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
        if let Err(cause) = pass_data.pass.set_pipeline(&self.0, *pipeline) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::set_pipeline",
            );
        }
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
        if let Err(cause) = pass_data
            .pass
            .set_bind_group(&self.0, index, *bind_group, offsets)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::set_bind_group",
            );
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
        if let Err(cause) =
            pass_data
                .pass
                .set_index_buffer(&self.0, *buffer, index_format, offset, size)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::set_index_buffer",
            );
        }
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
        if let Err(cause) = pass_data
            .pass
            .set_vertex_buffer(&self.0, slot, *buffer, offset, size)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::set_vertex_buffer",
            );
        }
    }

    fn render_pass_set_push_constants(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        stages: wgt::ShaderStages,
        offset: u32,
        data: &[u8],
    ) {
        if let Err(cause) = pass_data
            .pass
            .set_push_constants(&self.0, stages, offset, data)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::set_push_constants",
            );
        }
    }

    fn render_pass_draw(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) {
        if let Err(cause) = pass_data.pass.draw(
            &self.0,
            vertices.end - vertices.start,
            instances.end - instances.start,
            vertices.start,
            instances.start,
        ) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::draw",
            );
        }
    }

    fn render_pass_draw_indexed(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        if let Err(cause) = pass_data.pass.draw_indexed(
            &self.0,
            indices.end - indices.start,
            instances.end - instances.start,
            indices.start,
            base_vertex,
            instances.start,
        ) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::draw_indexed",
            );
        }
    }

    fn render_pass_draw_indirect(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        if let Err(cause) = pass_data
            .pass
            .draw_indirect(&self.0, *indirect_buffer, indirect_offset)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::draw_indirect",
            );
        }
    }

    fn render_pass_draw_indexed_indirect(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        indirect_buffer: &Self::BufferId,
        _indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        if let Err(cause) =
            pass_data
                .pass
                .draw_indexed_indirect(&self.0, *indirect_buffer, indirect_offset)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::draw_indexed_indirect",
            );
        }
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
        if let Err(cause) =
            pass_data
                .pass
                .multi_draw_indirect(&self.0, *indirect_buffer, indirect_offset, count)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::multi_draw_indirect",
            );
        }
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
        if let Err(cause) = pass_data.pass.multi_draw_indexed_indirect(
            &self.0,
            *indirect_buffer,
            indirect_offset,
            count,
        ) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::multi_draw_indexed_indirect",
            );
        }
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
        if let Err(cause) = pass_data.pass.multi_draw_indirect_count(
            &self.0,
            *indirect_buffer,
            indirect_offset,
            *count_buffer,
            count_buffer_offset,
            max_count,
        ) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::multi_draw_indirect_count",
            );
        }
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
        if let Err(cause) = pass_data.pass.multi_draw_indexed_indirect_count(
            &self.0,
            *indirect_buffer,
            indirect_offset,
            *count_buffer,
            count_buffer_offset,
            max_count,
        ) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::multi_draw_indexed_indirect_count",
            );
        }
    }

    fn render_pass_set_blend_constant(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        color: wgt::Color,
    ) {
        if let Err(cause) = pass_data.pass.set_blend_constant(&self.0, color) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::set_blend_constant",
            );
        }
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
        if let Err(cause) = pass_data
            .pass
            .set_scissor_rect(&self.0, x, y, width, height)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::set_scissor_rect",
            );
        }
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
        if let Err(cause) = pass_data
            .pass
            .set_viewport(&self.0, x, y, width, height, min_depth, max_depth)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::set_viewport",
            );
        }
    }

    fn render_pass_set_stencil_reference(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        reference: u32,
    ) {
        if let Err(cause) = pass_data.pass.set_stencil_reference(&self.0, reference) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::set_stencil_reference",
            );
        }
    }

    fn render_pass_insert_debug_marker(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        label: &str,
    ) {
        if let Err(cause) = pass_data.pass.insert_debug_marker(&self.0, label, 0) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::insert_debug_marker",
            );
        }
    }

    fn render_pass_push_debug_group(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        group_label: &str,
    ) {
        if let Err(cause) = pass_data.pass.push_debug_group(&self.0, group_label, 0) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::push_debug_group",
            );
        }
    }

    fn render_pass_pop_debug_group(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
    ) {
        if let Err(cause) = pass_data.pass.pop_debug_group(&self.0) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::pop_debug_group",
            );
        }
    }

    fn render_pass_write_timestamp(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        if let Err(cause) = pass_data
            .pass
            .write_timestamp(&self.0, *query_set, query_index)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::write_timestamp",
            );
        }
    }

    fn render_pass_begin_occlusion_query(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        query_index: u32,
    ) {
        if let Err(cause) = pass_data.pass.begin_occlusion_query(&self.0, query_index) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::begin_occlusion_query",
            );
        }
    }

    fn render_pass_end_occlusion_query(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
    ) {
        if let Err(cause) = pass_data.pass.end_occlusion_query(&self.0) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::end_occlusion_query",
            );
        }
    }

    fn render_pass_begin_pipeline_statistics_query(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        query_set: &Self::QuerySetId,
        _query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        if let Err(cause) =
            pass_data
                .pass
                .begin_pipeline_statistics_query(&self.0, *query_set, query_index)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::begin_pipeline_statistics_query",
            );
        }
    }

    fn render_pass_end_pipeline_statistics_query(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
    ) {
        if let Err(cause) = pass_data.pass.end_pipeline_statistics_query(&self.0) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::end_pipeline_statistics_query",
            );
        }
    }

    fn render_pass_execute_bundles(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
        render_bundles: &mut dyn Iterator<Item = (Self::RenderBundleId, &Self::RenderBundleData)>,
    ) {
        let temp_render_bundles = render_bundles.map(|(i, _)| i).collect::<SmallVec<[_; 4]>>();
        if let Err(cause) = pass_data
            .pass
            .execute_bundles(&self.0, &temp_render_bundles)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::execute_bundles",
            );
        }
    }

    fn render_pass_end(
        &self,
        _pass: &mut Self::RenderPassId,
        pass_data: &mut Self::RenderPassData,
    ) {
        if let Err(cause) = pass_data.pass.end(&self.0) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::end",
            );
        }
    }
}

impl<T> From<ObjectId> for wgc::id::Id<T>
where
    T: wgc::id::Marker,
{
    fn from(id: ObjectId) -> Self {
        let id = wgc::id::RawId::from_non_zero(id.id());
        // SAFETY: The id was created via the impl below
        unsafe { Self::from_raw(id) }
    }
}

impl<T> From<wgc::id::Id<T>> for ObjectId
where
    T: wgc::id::Marker,
{
    fn from(id: wgc::id::Id<T>) -> Self {
        let id = id.into_raw().into_non_zero();
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
            crate::Error::Internal { .. } => crate::ErrorFilter::Internal,
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

impl From<CreateShaderModuleError> for CompilationInfo {
    fn from(value: CreateShaderModuleError) -> Self {
        match value {
            #[cfg(feature = "wgsl")]
            CreateShaderModuleError::Parsing(v) => v.into(),
            #[cfg(feature = "glsl")]
            CreateShaderModuleError::ParsingGlsl(v) => v.into(),
            #[cfg(feature = "spirv")]
            CreateShaderModuleError::ParsingSpirV(v) => v.into(),
            CreateShaderModuleError::Validation(v) => v.into(),
            // Device errors are reported through the error sink, and are not compilation errors.
            // Same goes for native shader module generation errors.
            CreateShaderModuleError::Device(_) | CreateShaderModuleError::Generation => {
                CompilationInfo {
                    messages: Vec::new(),
                }
            }
            // Everything else is an error message without location information.
            _ => CompilationInfo {
                messages: vec![CompilationMessage {
                    message: value.to_string(),
                    message_type: CompilationMessageType::Error,
                    location: None,
                }],
            },
        }
    }
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
    ptr: NonNull<u8>,
    size: usize,
}

#[cfg(send_sync)]
unsafe impl Send for BufferMappedRange {}
#[cfg(send_sync)]
unsafe impl Sync for BufferMappedRange {}

impl crate::context::BufferMappedRange for BufferMappedRange {
    #[inline]
    fn slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    #[inline]
    fn slice_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }
}

impl Drop for BufferMappedRange {
    fn drop(&mut self) {
        // Intentionally left blank so that `BufferMappedRange` still
        // implements `Drop`, to match the web backend
    }
}
