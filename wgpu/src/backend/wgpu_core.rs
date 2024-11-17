use crate::{
    context::downcast_ref, AdapterInfo, BindGroupDescriptor, BindGroupLayoutDescriptor,
    BindingResource, BufferBinding, BufferDescriptor, CommandEncoderDescriptor, CompilationInfo,
    CompilationMessage, CompilationMessageType, ComputePassDescriptor, ComputePipelineDescriptor,
    DownlevelCapabilities, ErrorSource, Features, Label, Limits, LoadOp, MapMode, Operations,
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
    borrow::Cow::Borrowed,
    error::Error,
    fmt,
    future::{ready, Ready},
    ops::Range,
    ptr::NonNull,
    slice,
    sync::Arc,
};
use wgc::error::ContextErrorSource;
use wgc::{command::bundle_ffi::*, device::DeviceLostClosure, pipeline::CreateShaderModuleError};
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
        self.0.enumerate_adapters(backends)
    }

    pub unsafe fn create_adapter_from_hal<A: wgc::hal_api::HalApi>(
        &self,
        hal_adapter: hal::ExposedAdapter<A>,
    ) -> wgc::id::AdapterId {
        unsafe { self.0.create_adapter_from_hal(hal_adapter.into(), None) }
    }

    pub unsafe fn adapter_as_hal<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&A::Adapter>) -> R,
        R,
    >(
        &self,
        adapter: &wgc::id::AdapterId,
        hal_adapter_callback: F,
    ) -> R {
        unsafe {
            self.0
                .adapter_as_hal::<A, F, R>(*adapter, hal_adapter_callback)
        }
    }

    pub unsafe fn buffer_as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Buffer>) -> R, R>(
        &self,
        buffer: &Buffer,
        hal_buffer_callback: F,
    ) -> R {
        unsafe {
            self.0
                .buffer_as_hal::<A, F, R>(buffer.id, hal_buffer_callback)
        }
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
        let (device_id, queue_id) = unsafe {
            self.0.create_device_from_hal(
                *adapter,
                hal_device.into(),
                &desc.map_label(|l| l.map(Borrowed)),
                None,
                None,
                None,
            )
        }?;
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
                .create_texture_from_hal(Box::new(hal_texture), device.id, &descriptor, None)
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
    ) -> Buffer {
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
        Buffer {
            id,
            error_sink: Arc::clone(&device.error_sink),
        }
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
        texture_view_data: &wgc::id::TextureViewId,
        hal_texture_view_callback: F,
    ) -> R {
        unsafe {
            self.0
                .texture_view_as_hal::<A, F, R>(*texture_view_data, hal_texture_view_callback)
        }
    }

    /// This method will start the wgpu_core level command recording.
    pub unsafe fn command_encoder_as_hal_mut<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&mut A::CommandEncoder>) -> R,
        R,
    >(
        &self,
        command_encoder: &CommandEncoder,
        hal_command_encoder_callback: F,
    ) -> R {
        unsafe {
            self.0.command_encoder_as_hal_mut::<A, F, R>(
                command_encoder.id,
                hal_command_encoder_callback,
            )
        }
    }

    pub fn generate_report(&self) -> wgc::global::GlobalReport {
        self.0.generate_report()
    }

    #[cold]
    #[track_caller]
    #[inline(never)]
    fn handle_error_inner(
        &self,
        sink_mutex: &Mutex<ErrorSinkRaw>,
        source: ContextErrorSource,
        label: Label<'_>,
        fn_ident: &'static str,
    ) {
        let source_error: ErrorSource = Box::new(wgc::error::ContextError {
            fn_ident,
            source,
            label: label.unwrap_or_default().to_string(),
        });
        let mut sink = sink_mutex.lock();
        let mut source_opt: Option<&(dyn Error + 'static)> = Some(&*source_error);
        let error = loop {
            if let Some(source) = source_opt {
                if let Some(wgc::device::DeviceError::OutOfMemory) =
                    source.downcast_ref::<wgc::device::DeviceError>()
                {
                    break crate::Error::OutOfMemory {
                        source: source_error,
                    };
                }
                source_opt = source.source();
            } else {
                // Otherwise, it is a validation error
                break crate::Error::Validation {
                    description: self.format_error(&*source_error),
                    source: source_error,
                };
            }
        };
        sink.handle_error(error);
    }

    #[inline]
    #[track_caller]
    fn handle_error(
        &self,
        sink_mutex: &Mutex<ErrorSinkRaw>,
        source: impl Error + WasmNotSendSync + 'static,
        label: Label<'_>,
        fn_ident: &'static str,
    ) {
        self.handle_error_inner(sink_mutex, Box::new(source), label, fn_ident)
    }

    #[inline]
    #[track_caller]
    fn handle_error_nolabel(
        &self,
        sink_mutex: &Mutex<ErrorSinkRaw>,
        source: impl Error + WasmNotSendSync + 'static,
        fn_ident: &'static str,
    ) {
        self.handle_error_inner(sink_mutex, Box::new(source), None, fn_ident)
    }

    #[track_caller]
    #[cold]
    fn handle_error_fatal(
        &self,
        cause: impl Error + WasmNotSendSync + 'static,
        operation: &'static str,
    ) -> ! {
        panic!("Error in {operation}: {f}", f = self.format_error(&cause));
    }

    #[inline(never)]
    fn format_error(&self, err: &(dyn Error + 'static)) -> String {
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

        format!("Validation Error\n\nCaused by:\n{output}")
    }
}

fn map_buffer_copy_view(view: crate::ImageCopyBuffer<'_>) -> wgc::command::ImageCopyBuffer {
    wgc::command::ImageCopyBuffer {
        buffer: downcast_buffer(view.buffer).id,
        layout: view.layout,
    }
}

fn map_texture_copy_view(view: crate::ImageCopyTexture<'_>) -> wgc::command::ImageCopyTexture {
    wgc::command::ImageCopyTexture {
        texture: downcast_texture(view.texture).id,
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
        texture: downcast_texture(view.texture).id,
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

#[derive(Debug)]
pub struct Device {
    id: wgc::id::DeviceId,
    error_sink: ErrorSink,
    features: Features,
}

#[derive(Debug)]
pub struct Buffer {
    id: wgc::id::BufferId,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct ShaderModule {
    id: wgc::id::ShaderModuleId,
    compilation_info: CompilationInfo,
}

#[derive(Debug)]
pub struct Texture {
    id: wgc::id::TextureId,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct Queue {
    id: wgc::id::QueueId,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct ComputePipeline {
    id: wgc::id::ComputePipelineId,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct RenderPipeline {
    id: wgc::id::RenderPipelineId,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct ComputePass {
    pass: wgc::command::ComputePass,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct RenderPass {
    pass: wgc::command::RenderPass,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct CommandEncoder {
    id: wgc::id::CommandEncoderId,
    error_sink: ErrorSink,
    open: bool,
}

#[derive(Debug)]
pub struct Blas {
    id: wgc::id::BlasId,
    // error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct Tlas {
    id: wgc::id::TlasId,
    // error_sink: ErrorSink,
}

impl crate::Context for ContextWgpuCore {
    type AdapterData = wgc::id::AdapterId;
    type DeviceData = Device;
    type QueueData = Queue;
    type ShaderModuleData = ShaderModule;
    type BindGroupLayoutData = wgc::id::BindGroupLayoutId;
    type BindGroupData = wgc::id::BindGroupId;
    type TextureViewData = wgc::id::TextureViewId;
    type SamplerData = wgc::id::SamplerId;
    type BufferData = Buffer;
    type TextureData = Texture;
    type QuerySetData = wgc::id::QuerySetId;
    type PipelineLayoutData = wgc::id::PipelineLayoutId;
    type RenderPipelineData = RenderPipeline;
    type ComputePipelineData = ComputePipeline;
    type PipelineCacheData = wgc::id::PipelineCacheId;
    type CommandEncoderData = CommandEncoder;
    type ComputePassData = ComputePass;
    type RenderPassData = RenderPass;
    type CommandBufferData = wgc::id::CommandBufferId;
    type RenderBundleEncoderData = wgc::command::RenderBundleEncoder;
    type RenderBundleData = wgc::id::RenderBundleId;

    type SurfaceData = Surface;
    type SurfaceOutputDetail = SurfaceOutputDetail;
    type SubmissionIndexData = wgc::SubmissionIndex;

    type RequestAdapterFuture = Ready<Option<Self::AdapterData>>;
    type BlasData = Blas;
    type TlasData = Tlas;

    #[allow(clippy::type_complexity)]
    type RequestDeviceFuture =
        Ready<Result<(Self::DeviceData, Self::QueueData), crate::RequestDeviceError>>;

    type PopErrorScopeFuture = Ready<Option<crate::Error>>;
    type CompilationInfoFuture = Ready<CompilationInfo>;

    fn init(instance_desc: wgt::InstanceDescriptor) -> Self {
        Self(wgc::global::Global::new("wgpu", instance_desc))
    }

    unsafe fn instance_create_surface(
        &self,
        target: SurfaceTargetUnsafe,
    ) -> Result<Self::SurfaceData, crate::CreateSurfaceError> {
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

        Ok(Surface {
            id,
            configured_device: Mutex::default(),
        })
    }

    fn instance_request_adapter(
        &self,
        options: &crate::RequestAdapterOptions<'_, '_>,
    ) -> Self::RequestAdapterFuture {
        let id = self.0.request_adapter(
            &wgc::instance::RequestAdapterOptions {
                power_preference: options.power_preference,
                force_fallback_adapter: options.force_fallback_adapter,
                compatible_surface: options.compatible_surface.map(|surface| {
                    let surface: &<ContextWgpuCore as crate::Context>::SurfaceData =
                        downcast_ref(surface.surface_data.as_ref());
                    surface.id
                }),
            },
            wgt::Backends::all(),
            None,
        );
        ready(id.ok())
    }

    fn adapter_request_device(
        &self,
        adapter_data: &Self::AdapterData,
        desc: &crate::DeviceDescriptor<'_>,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture {
        if trace_dir.is_some() {
            log::error!("Feature 'trace' has been removed temporarily, see https://github.com/gfx-rs/wgpu/issues/5974");
        }
        let res = self.0.adapter_request_device(
            *adapter_data,
            &desc.map_label(|l| l.map(Borrowed)),
            None,
            None,
            None,
        );
        let (device_id, queue_id) = match res {
            Ok(ids) => ids,
            Err(err) => {
                return ready(Err(err.into()));
            }
        };
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
        ready(Ok((device, queue)))
    }

    fn instance_poll_all_devices(&self, force_wait: bool) -> bool {
        match self.0.poll_all_devices(force_wait) {
            Ok(all_queue_empty) => all_queue_empty,
            Err(err) => self.handle_error_fatal(err, "Device::poll"),
        }
    }

    fn adapter_is_surface_supported(
        &self,
        adapter_data: &Self::AdapterData,
        surface_data: &Self::SurfaceData,
    ) -> bool {
        self.0
            .adapter_is_surface_supported(*adapter_data, surface_data.id)
    }

    fn adapter_features(&self, adapter_data: &Self::AdapterData) -> Features {
        self.0.adapter_features(*adapter_data)
    }

    fn adapter_limits(&self, adapter_data: &Self::AdapterData) -> Limits {
        self.0.adapter_limits(*adapter_data)
    }

    fn adapter_downlevel_capabilities(
        &self,
        adapter_data: &Self::AdapterData,
    ) -> DownlevelCapabilities {
        self.0.adapter_downlevel_capabilities(*adapter_data)
    }

    fn adapter_get_info(&self, adapter_data: &Self::AdapterData) -> AdapterInfo {
        self.0.adapter_get_info(*adapter_data)
    }

    fn adapter_get_texture_format_features(
        &self,
        adapter_data: &Self::AdapterData,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        self.0
            .adapter_get_texture_format_features(*adapter_data, format)
    }

    fn adapter_get_presentation_timestamp(
        &self,
        adapter_data: &Self::AdapterData,
    ) -> wgt::PresentationTimestamp {
        self.0.adapter_get_presentation_timestamp(*adapter_data)
    }

    fn surface_get_capabilities(
        &self,
        surface_data: &Self::SurfaceData,
        adapter_data: &Self::AdapterData,
    ) -> wgt::SurfaceCapabilities {
        match self
            .0
            .surface_get_capabilities(surface_data.id, *adapter_data)
        {
            Ok(caps) => caps,
            Err(_) => wgt::SurfaceCapabilities::default(),
        }
    }

    fn surface_configure(
        &self,
        surface_data: &Self::SurfaceData,
        device_data: &Self::DeviceData,
        config: &crate::SurfaceConfiguration,
    ) {
        let error = self
            .0
            .surface_configure(surface_data.id, device_data.id, config);
        if let Some(e) = error {
            self.handle_error_fatal(e, "Surface::configure");
        } else {
            *surface_data.configured_device.lock() = Some(device_data.id);
        }
    }

    fn surface_get_current_texture(
        &self,
        surface_data: &Self::SurfaceData,
    ) -> (
        Option<Self::TextureData>,
        SurfaceStatus,
        Self::SurfaceOutputDetail,
    ) {
        match self.0.surface_get_current_texture(surface_data.id, None) {
            Ok(wgc::present::SurfaceOutput { status, texture_id }) => {
                let data = texture_id.map(|id| Texture {
                    id,
                    error_sink: Arc::new(Mutex::new(ErrorSinkRaw::new())),
                });

                (
                    data,
                    status,
                    SurfaceOutputDetail {
                        surface_id: surface_data.id,
                    },
                )
            }
            Err(err) => self.handle_error_fatal(err, "Surface::get_current_texture_view"),
        }
    }

    fn surface_present(&self, detail: &Self::SurfaceOutputDetail) {
        match self.0.surface_present(detail.surface_id) {
            Ok(_status) => (),
            Err(err) => self.handle_error_fatal(err, "Surface::present"),
        }
    }

    fn surface_texture_discard(&self, detail: &Self::SurfaceOutputDetail) {
        match self.0.surface_texture_discard(detail.surface_id) {
            Ok(_status) => (),
            Err(err) => self.handle_error_fatal(err, "Surface::discard_texture"),
        }
    }

    fn device_features(&self, device_data: &Self::DeviceData) -> Features {
        self.0.device_features(device_data.id)
    }

    fn device_limits(&self, device_data: &Self::DeviceData) -> Limits {
        self.0.device_limits(device_data.id)
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
        device_data: &Self::DeviceData,
        desc: ShaderModuleDescriptor<'_>,
        shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> Self::ShaderModuleData {
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
        let (id, error) =
            self.0
                .device_create_shader_module(device_data.id, &descriptor, source, None);
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

        ShaderModule {
            id,
            compilation_info,
        }
    }

    unsafe fn device_create_shader_module_spirv(
        &self,
        device_data: &Self::DeviceData,
        desc: &ShaderModuleDescriptorSpirV<'_>,
    ) -> Self::ShaderModuleData {
        let descriptor = wgc::pipeline::ShaderModuleDescriptor {
            label: desc.label.map(Borrowed),
            // Doesn't matter the value since spirv shaders aren't mutated to include
            // runtime checks
            shader_bound_checks: unsafe { wgt::ShaderBoundChecks::unchecked() },
        };
        let (id, error) = unsafe {
            self.0.device_create_shader_module_spirv(
                device_data.id,
                &descriptor,
                Borrowed(&desc.source),
                None,
            )
        };
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
        ShaderModule {
            id,
            compilation_info,
        }
    }

    fn device_create_bind_group_layout(
        &self,
        device_data: &Self::DeviceData,
        desc: &BindGroupLayoutDescriptor<'_>,
    ) -> Self::BindGroupLayoutData {
        let descriptor = wgc::binding_model::BindGroupLayoutDescriptor {
            label: desc.label.map(Borrowed),
            entries: Borrowed(desc.entries),
        };
        let (id, error) = self
            .0
            .device_create_bind_group_layout(device_data.id, &descriptor, None);
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                desc.label,
                "Device::create_bind_group_layout",
            );
        }
        id
    }
    fn device_create_bind_group(
        &self,
        device_data: &Self::DeviceData,
        desc: &BindGroupDescriptor<'_>,
    ) -> Self::BindGroupData {
        use wgc::binding_model as bm;

        let mut arrayed_texture_views = Vec::new();
        let mut arrayed_samplers = Vec::new();
        if device_data
            .features
            .contains(Features::TEXTURE_BINDING_ARRAY)
        {
            // gather all the array view IDs first
            for entry in desc.entries.iter() {
                if let BindingResource::TextureViewArray(array) = entry.resource {
                    arrayed_texture_views
                        .extend(array.iter().map(|view| *downcast_texture_view(view)));
                }
                if let BindingResource::SamplerArray(array) = entry.resource {
                    arrayed_samplers.extend(array.iter().map(|sampler| *downcast_sampler(sampler)));
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
                        buffer_id: downcast_buffer(binding.buffer).id,
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
                        buffer_id: downcast_buffer(buffer).id,
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
                        bm::BindingResource::Sampler(*downcast_sampler(sampler))
                    }
                    BindingResource::SamplerArray(array) => {
                        let slice = &remaining_arrayed_samplers[..array.len()];
                        remaining_arrayed_samplers = &remaining_arrayed_samplers[array.len()..];
                        bm::BindingResource::SamplerArray(Borrowed(slice))
                    }
                    BindingResource::TextureView(texture_view) => {
                        bm::BindingResource::TextureView(*downcast_texture_view(texture_view))
                    }
                    BindingResource::TextureViewArray(array) => {
                        let slice = &remaining_arrayed_texture_views[..array.len()];
                        remaining_arrayed_texture_views =
                            &remaining_arrayed_texture_views[array.len()..];
                        bm::BindingResource::TextureViewArray(Borrowed(slice))
                    }
                    BindingResource::AccelerationStructure(acceleration_structure) => {
                        bm::BindingResource::AccelerationStructure(
                            downcast_tlas(acceleration_structure).id,
                        )
                    }
                },
            })
            .collect::<Vec<_>>();
        let descriptor = bm::BindGroupDescriptor {
            label: desc.label.as_ref().map(|label| Borrowed(&label[..])),
            layout: *downcast_bind_group_layout(desc.layout),
            entries: Borrowed(&entries),
        };

        let (id, error) = self
            .0
            .device_create_bind_group(device_data.id, &descriptor, None);
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                desc.label,
                "Device::create_bind_group",
            );
        }
        id
    }
    fn device_create_pipeline_layout(
        &self,
        device_data: &Self::DeviceData,
        desc: &PipelineLayoutDescriptor<'_>,
    ) -> Self::PipelineLayoutData {
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
            .map(|bgl| *downcast_bind_group_layout(bgl))
            .collect::<ArrayVec<_, { wgc::MAX_BIND_GROUPS }>>();
        let descriptor = wgc::binding_model::PipelineLayoutDescriptor {
            label: desc.label.map(Borrowed),
            bind_group_layouts: Borrowed(&temp_layouts),
            push_constant_ranges: Borrowed(desc.push_constant_ranges),
        };

        let (id, error) = self
            .0
            .device_create_pipeline_layout(device_data.id, &descriptor, None);
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                desc.label,
                "Device::create_pipeline_layout",
            );
        }
        id
    }
    fn device_create_render_pipeline(
        &self,
        device_data: &Self::DeviceData,
        desc: &RenderPipelineDescriptor<'_>,
    ) -> Self::RenderPipelineData {
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
            layout: desc.layout.map(downcast_pipeline_layout).copied(),
            vertex: pipe::VertexState {
                stage: pipe::ProgrammableStageDescriptor {
                    module: downcast_shader_module(desc.vertex.module).id,
                    entry_point: desc.vertex.entry_point.map(Borrowed),
                    constants: Borrowed(desc.vertex.compilation_options.constants),
                    zero_initialize_workgroup_memory: desc
                        .vertex
                        .compilation_options
                        .zero_initialize_workgroup_memory,
                },
                buffers: Borrowed(&vertex_buffers),
            },
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment: desc.fragment.as_ref().map(|frag| pipe::FragmentState {
                stage: pipe::ProgrammableStageDescriptor {
                    module: downcast_shader_module(frag.module).id,
                    entry_point: frag.entry_point.map(Borrowed),
                    constants: Borrowed(frag.compilation_options.constants),
                    zero_initialize_workgroup_memory: frag
                        .compilation_options
                        .zero_initialize_workgroup_memory,
                },
                targets: Borrowed(frag.targets),
            }),
            multiview: desc.multiview,
            cache: desc.cache.map(downcast_pipeline_cache).copied(),
        };

        let (id, error) =
            self.0
                .device_create_render_pipeline(device_data.id, &descriptor, None, None);
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
        RenderPipeline {
            id,
            error_sink: Arc::clone(&device_data.error_sink),
        }
    }
    fn device_create_compute_pipeline(
        &self,
        device_data: &Self::DeviceData,
        desc: &ComputePipelineDescriptor<'_>,
    ) -> Self::ComputePipelineData {
        use wgc::pipeline as pipe;

        let descriptor = pipe::ComputePipelineDescriptor {
            label: desc.label.map(Borrowed),
            layout: desc.layout.map(downcast_pipeline_layout).copied(),
            stage: pipe::ProgrammableStageDescriptor {
                module: downcast_shader_module(desc.module).id,
                entry_point: desc.entry_point.map(Borrowed),
                constants: Borrowed(desc.compilation_options.constants),
                zero_initialize_workgroup_memory: desc
                    .compilation_options
                    .zero_initialize_workgroup_memory,
            },
            cache: desc.cache.map(downcast_pipeline_cache).copied(),
        };

        let (id, error) =
            self.0
                .device_create_compute_pipeline(device_data.id, &descriptor, None, None);
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
        ComputePipeline {
            id,
            error_sink: Arc::clone(&device_data.error_sink),
        }
    }

    unsafe fn device_create_pipeline_cache(
        &self,
        device_data: &Self::DeviceData,
        desc: &PipelineCacheDescriptor<'_>,
    ) -> Self::PipelineCacheData {
        use wgc::pipeline as pipe;

        let descriptor = pipe::PipelineCacheDescriptor {
            label: desc.label.map(Borrowed),
            data: desc.data.map(Borrowed),
            fallback: desc.fallback,
        };
        let (id, error) = unsafe {
            self.0
                .device_create_pipeline_cache(device_data.id, &descriptor, None)
        };
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                desc.label,
                "Device::device_create_pipeline_cache_init",
            );
        }
        id
    }

    fn device_create_buffer(
        &self,
        device_data: &Self::DeviceData,
        desc: &crate::BufferDescriptor<'_>,
    ) -> Self::BufferData {
        let (id, error) =
            self.0
                .device_create_buffer(device_data.id, &desc.map_label(|l| l.map(Borrowed)), None);
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                desc.label,
                "Device::create_buffer",
            );
        }

        Buffer {
            id,
            error_sink: Arc::clone(&device_data.error_sink),
        }
    }
    fn device_create_texture(
        &self,
        device_data: &Self::DeviceData,
        desc: &TextureDescriptor<'_>,
    ) -> Self::TextureData {
        let wgt_desc = desc.map_label_and_view_formats(|l| l.map(Borrowed), |v| v.to_vec());
        let (id, error) = self
            .0
            .device_create_texture(device_data.id, &wgt_desc, None);
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                desc.label,
                "Device::create_texture",
            );
        }

        Texture {
            id,
            error_sink: Arc::clone(&device_data.error_sink),
        }
    }
    fn device_create_sampler(
        &self,
        device_data: &Self::DeviceData,
        desc: &SamplerDescriptor<'_>,
    ) -> Self::SamplerData {
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

        let (id, error) = self
            .0
            .device_create_sampler(device_data.id, &descriptor, None);
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                desc.label,
                "Device::create_sampler",
            );
        }
        id
    }
    fn device_create_query_set(
        &self,
        device_data: &Self::DeviceData,
        desc: &wgt::QuerySetDescriptor<Label<'_>>,
    ) -> Self::QuerySetData {
        let (id, error) = self.0.device_create_query_set(
            device_data.id,
            &desc.map_label(|l| l.map(Borrowed)),
            None,
        );
        if let Some(cause) = error {
            self.handle_error_nolabel(&device_data.error_sink, cause, "Device::create_query_set");
        }
        id
    }
    fn device_create_command_encoder(
        &self,
        device_data: &Self::DeviceData,
        desc: &CommandEncoderDescriptor<'_>,
    ) -> Self::CommandEncoderData {
        let (id, error) = self.0.device_create_command_encoder(
            device_data.id,
            &desc.map_label(|l| l.map(Borrowed)),
            None,
        );
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                desc.label,
                "Device::create_command_encoder",
            );
        }

        CommandEncoder {
            id,
            error_sink: Arc::clone(&device_data.error_sink),
            open: true,
        }
    }
    fn device_create_render_bundle_encoder(
        &self,
        device_data: &Self::DeviceData,
        desc: &RenderBundleEncoderDescriptor<'_>,
    ) -> Self::RenderBundleEncoderData {
        let descriptor = wgc::command::RenderBundleEncoderDescriptor {
            label: desc.label.map(Borrowed),
            color_formats: Borrowed(desc.color_formats),
            depth_stencil: desc.depth_stencil,
            sample_count: desc.sample_count,
            multiview: desc.multiview,
        };
        match wgc::command::RenderBundleEncoder::new(&descriptor, device_data.id, None) {
            Ok(encoder) => encoder,
            Err(e) => panic!("Error in Device::create_render_bundle_encoder: {e}"),
        }
    }
    #[cfg_attr(not(any(native, Emscripten)), allow(unused))]
    fn device_drop(&self, device_data: &Self::DeviceData) {
        #[cfg(any(native, Emscripten))]
        {
            self.0.device_drop(device_data.id);
        }
    }
    #[cfg_attr(target_arch = "wasm32", allow(unused))]
    fn queue_drop(&self, queue_data: &Self::QueueData) {
        self.0.queue_drop(queue_data.id);
    }
    fn device_set_device_lost_callback(
        &self,
        device_data: &Self::DeviceData,
        device_lost_callback: crate::context::DeviceLostCallback,
    ) {
        let device_lost_closure = DeviceLostClosure::from_rust(device_lost_callback);
        self.0
            .device_set_device_lost_closure(device_data.id, device_lost_closure);
    }
    fn device_destroy(&self, device_data: &Self::DeviceData) {
        self.0.device_destroy(device_data.id);
    }
    fn device_poll(
        &self,
        device_data: &Self::DeviceData,
        maintain: crate::Maintain,
    ) -> wgt::MaintainResult {
        let maintain_inner = maintain.map_index(|i| *i.data.as_ref().downcast_ref().unwrap());
        match self.0.device_poll(device_data.id, maintain_inner) {
            Ok(done) => match done {
                true => wgt::MaintainResult::SubmissionQueueEmpty,
                false => wgt::MaintainResult::Ok,
            },
            Err(err) => self.handle_error_fatal(err, "Device::poll"),
        }
    }
    fn device_on_uncaptured_error(
        &self,
        device_data: &Self::DeviceData,
        handler: Box<dyn UncapturedErrorHandler>,
    ) {
        let mut error_sink = device_data.error_sink.lock();
        error_sink.uncaptured_handler = Some(handler);
    }
    fn device_push_error_scope(&self, device_data: &Self::DeviceData, filter: crate::ErrorFilter) {
        let mut error_sink = device_data.error_sink.lock();
        error_sink.scopes.push(ErrorScope {
            error: None,
            filter,
        });
    }
    fn device_pop_error_scope(&self, device_data: &Self::DeviceData) -> Self::PopErrorScopeFuture {
        let mut error_sink = device_data.error_sink.lock();
        let scope = error_sink.scopes.pop().unwrap();
        ready(scope.error)
    }

    fn buffer_map_async(
        &self,
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

        match self.0.buffer_map_async(
            buffer_data.id,
            range.start,
            Some(range.end - range.start),
            operation,
        ) {
            Ok(_) => (),
            Err(cause) => {
                self.handle_error_nolabel(&buffer_data.error_sink, cause, "Buffer::map_async")
            }
        }
    }
    fn buffer_get_mapped_range(
        &self,
        buffer_data: &Self::BufferData,
        sub_range: Range<wgt::BufferAddress>,
    ) -> Box<dyn crate::context::BufferMappedRange> {
        let size = sub_range.end - sub_range.start;
        match self
            .0
            .buffer_get_mapped_range(buffer_data.id, sub_range.start, Some(size))
        {
            Ok((ptr, size)) => Box::new(BufferMappedRange {
                ptr,
                size: size as usize,
            }),
            Err(err) => self.handle_error_fatal(err, "Buffer::get_mapped_range"),
        }
    }

    fn buffer_unmap(&self, buffer_data: &Self::BufferData) {
        match self.0.buffer_unmap(buffer_data.id) {
            Ok(()) => (),
            Err(cause) => {
                self.handle_error_nolabel(&buffer_data.error_sink, cause, "Buffer::buffer_unmap")
            }
        }
    }

    fn shader_get_compilation_info(
        &self,
        shader_data: &Self::ShaderModuleData,
    ) -> Self::CompilationInfoFuture {
        ready(shader_data.compilation_info.clone())
    }

    fn texture_create_view(
        &self,
        texture_data: &Self::TextureData,
        desc: &TextureViewDescriptor<'_>,
    ) -> Self::TextureViewData {
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
        let (id, error) = self
            .0
            .texture_create_view(texture_data.id, &descriptor, None);
        if let Some(cause) = error {
            self.handle_error(
                &texture_data.error_sink,
                cause,
                desc.label,
                "Texture::create_view",
            );
        }
        id
    }

    fn surface_drop(&self, surface_data: &Self::SurfaceData) {
        self.0.surface_drop(surface_data.id)
    }

    fn adapter_drop(&self, adapter_data: &Self::AdapterData) {
        self.0.adapter_drop(*adapter_data)
    }

    fn buffer_destroy(&self, buffer_data: &Self::BufferData) {
        // Per spec, no error to report. Even calling destroy multiple times is valid.
        let _ = self.0.buffer_destroy(buffer_data.id);
    }

    fn buffer_drop(&self, buffer_data: &Self::BufferData) {
        self.0.buffer_drop(buffer_data.id)
    }

    fn texture_destroy(&self, texture_data: &Self::TextureData) {
        // Per spec, no error to report. Even calling destroy multiple times is valid.
        let _ = self.0.texture_destroy(texture_data.id);
    }

    fn texture_drop(&self, texture_data: &Self::TextureData) {
        self.0.texture_drop(texture_data.id)
    }

    fn texture_view_drop(&self, texture_view_data: &Self::TextureViewData) {
        let _ = self.0.texture_view_drop(*texture_view_data);
    }

    fn sampler_drop(&self, sampler_data: &Self::SamplerData) {
        self.0.sampler_drop(*sampler_data)
    }

    fn query_set_drop(&self, query_set_data: &Self::QuerySetData) {
        self.0.query_set_drop(*query_set_data)
    }

    fn bind_group_drop(&self, bind_group_data: &Self::BindGroupData) {
        self.0.bind_group_drop(*bind_group_data)
    }

    fn bind_group_layout_drop(&self, bind_group_layout_data: &Self::BindGroupLayoutData) {
        self.0.bind_group_layout_drop(*bind_group_layout_data)
    }

    fn pipeline_layout_drop(&self, pipeline_layout_data: &Self::PipelineLayoutData) {
        self.0.pipeline_layout_drop(*pipeline_layout_data)
    }
    fn shader_module_drop(&self, shader_module_data: &Self::ShaderModuleData) {
        self.0.shader_module_drop(shader_module_data.id)
    }
    fn command_encoder_drop(&self, command_encoder_data: &Self::CommandEncoderData) {
        if command_encoder_data.open {
            self.0.command_encoder_drop(command_encoder_data.id)
        }
    }

    fn command_buffer_drop(&self, command_buffer_data: &Self::CommandBufferData) {
        self.0.command_buffer_drop(*command_buffer_data)
    }

    fn render_bundle_drop(&self, render_bundle_data: &Self::RenderBundleData) {
        self.0.render_bundle_drop(*render_bundle_data)
    }

    fn compute_pipeline_drop(&self, pipeline_data: &Self::ComputePipelineData) {
        self.0.compute_pipeline_drop(pipeline_data.id)
    }

    fn render_pipeline_drop(&self, pipeline_data: &Self::RenderPipelineData) {
        self.0.render_pipeline_drop(pipeline_data.id)
    }

    fn pipeline_cache_drop(&self, cache_data: &Self::PipelineCacheData) {
        self.0.pipeline_cache_drop(*cache_data)
    }

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline_data: &Self::ComputePipelineData,
        index: u32,
    ) -> Self::BindGroupLayoutData {
        let (id, error) =
            self.0
                .compute_pipeline_get_bind_group_layout(pipeline_data.id, index, None);
        if let Some(err) = error {
            self.handle_error_nolabel(
                &pipeline_data.error_sink,
                err,
                "ComputePipeline::get_bind_group_layout",
            )
        }
        id
    }

    fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline_data: &Self::RenderPipelineData,
        index: u32,
    ) -> Self::BindGroupLayoutData {
        let (id, error) =
            self.0
                .render_pipeline_get_bind_group_layout(pipeline_data.id, index, None);
        if let Some(err) = error {
            self.handle_error_nolabel(
                &pipeline_data.error_sink,
                err,
                "RenderPipeline::get_bind_group_layout",
            )
        }
        id
    }

    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder_data: &Self::CommandEncoderData,
        source_data: &Self::BufferData,
        source_offset: wgt::BufferAddress,
        destination_data: &Self::BufferData,
        destination_offset: wgt::BufferAddress,
        copy_size: wgt::BufferAddress,
    ) {
        if let Err(cause) = self.0.command_encoder_copy_buffer_to_buffer(
            encoder_data.id,
            source_data.id,
            source_offset,
            destination_data.id,
            destination_offset,
            copy_size,
        ) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::copy_buffer_to_buffer",
            );
        }
    }

    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder_data: &Self::CommandEncoderData,
        source: crate::ImageCopyBuffer<'_>,
        destination: crate::ImageCopyTexture<'_>,
        copy_size: wgt::Extent3d,
    ) {
        if let Err(cause) = self.0.command_encoder_copy_buffer_to_texture(
            encoder_data.id,
            &map_buffer_copy_view(source),
            &map_texture_copy_view(destination),
            &copy_size,
        ) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::copy_buffer_to_texture",
            );
        }
    }

    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder_data: &Self::CommandEncoderData,
        source: crate::ImageCopyTexture<'_>,
        destination: crate::ImageCopyBuffer<'_>,
        copy_size: wgt::Extent3d,
    ) {
        if let Err(cause) = self.0.command_encoder_copy_texture_to_buffer(
            encoder_data.id,
            &map_texture_copy_view(source),
            &map_buffer_copy_view(destination),
            &copy_size,
        ) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::copy_texture_to_buffer",
            );
        }
    }

    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder_data: &Self::CommandEncoderData,
        source: crate::ImageCopyTexture<'_>,
        destination: crate::ImageCopyTexture<'_>,
        copy_size: wgt::Extent3d,
    ) {
        if let Err(cause) = self.0.command_encoder_copy_texture_to_texture(
            encoder_data.id,
            &map_texture_copy_view(source),
            &map_texture_copy_view(destination),
            &copy_size,
        ) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::copy_texture_to_texture",
            );
        }
    }

    fn command_encoder_begin_compute_pass(
        &self,
        encoder_data: &Self::CommandEncoderData,
        desc: &ComputePassDescriptor<'_>,
    ) -> Self::ComputePassData {
        let timestamp_writes =
            desc.timestamp_writes
                .as_ref()
                .map(|tw| wgc::command::PassTimestampWrites {
                    query_set: *downcast_query_set(tw.query_set),
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                });

        let (pass, err) = self.0.command_encoder_create_compute_pass(
            encoder_data.id,
            &wgc::command::ComputePassDescriptor {
                label: desc.label.map(Borrowed),
                timestamp_writes: timestamp_writes.as_ref(),
            },
        );

        if let Some(cause) = err {
            self.handle_error(
                &encoder_data.error_sink,
                cause,
                desc.label,
                "CommandEncoder::begin_compute_pass",
            );
        }

        Self::ComputePassData {
            pass,
            error_sink: encoder_data.error_sink.clone(),
        }
    }

    fn command_encoder_begin_render_pass(
        &self,
        encoder_data: &Self::CommandEncoderData,
        desc: &crate::RenderPassDescriptor<'_>,
    ) -> Self::RenderPassData {
        let colors = desc
            .color_attachments
            .iter()
            .map(|ca| {
                ca.as_ref()
                    .map(|at| wgc::command::RenderPassColorAttachment {
                        view: *downcast_texture_view(at.view),
                        resolve_target: at.resolve_target.map(downcast_texture_view).copied(),
                        channel: map_pass_channel(Some(&at.ops)),
                    })
            })
            .collect::<Vec<_>>();

        let depth_stencil = desc.depth_stencil_attachment.as_ref().map(|dsa| {
            wgc::command::RenderPassDepthStencilAttachment {
                view: *downcast_texture_view(dsa.view),
                depth: map_pass_channel(dsa.depth_ops.as_ref()),
                stencil: map_pass_channel(dsa.stencil_ops.as_ref()),
            }
        });

        let timestamp_writes =
            desc.timestamp_writes
                .as_ref()
                .map(|tw| wgc::command::PassTimestampWrites {
                    query_set: *downcast_query_set(tw.query_set),
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                });

        let (pass, err) = self.0.command_encoder_create_render_pass(
            encoder_data.id,
            &wgc::command::RenderPassDescriptor {
                label: desc.label.map(Borrowed),
                timestamp_writes: timestamp_writes.as_ref(),
                color_attachments: std::borrow::Cow::Borrowed(&colors),
                depth_stencil_attachment: depth_stencil.as_ref(),
                occlusion_query_set: desc.occlusion_query_set.map(downcast_query_set).copied(),
            },
        );

        if let Some(cause) = err {
            self.handle_error(
                &encoder_data.error_sink,
                cause,
                desc.label,
                "CommandEncoder::begin_render_pass",
            );
        }

        Self::RenderPassData {
            pass,
            error_sink: encoder_data.error_sink.clone(),
        }
    }

    fn command_encoder_finish(
        &self,
        encoder_data: &mut Self::CommandEncoderData,
    ) -> Self::CommandBufferData {
        let descriptor = wgt::CommandBufferDescriptor::default();
        encoder_data.open = false; // prevent the drop
        let (id, error) = self.0.command_encoder_finish(encoder_data.id, &descriptor);
        if let Some(cause) = error {
            self.handle_error_nolabel(&encoder_data.error_sink, cause, "a CommandEncoder");
        }
        id
    }

    fn command_encoder_clear_texture(
        &self,
        encoder_data: &Self::CommandEncoderData,
        texture_data: &Self::TextureData,
        subresource_range: &wgt::ImageSubresourceRange,
    ) {
        if let Err(cause) = self.0.command_encoder_clear_texture(
            encoder_data.id,
            texture_data.id,
            subresource_range,
        ) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::clear_texture",
            );
        }
    }

    fn command_encoder_clear_buffer(
        &self,
        encoder_data: &Self::CommandEncoderData,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
    ) {
        if let Err(cause) =
            self.0
                .command_encoder_clear_buffer(encoder_data.id, buffer_data.id, offset, size)
        {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::fill_buffer",
            );
        }
    }

    fn command_encoder_insert_debug_marker(
        &self,
        encoder_data: &Self::CommandEncoderData,
        label: &str,
    ) {
        if let Err(cause) = self
            .0
            .command_encoder_insert_debug_marker(encoder_data.id, label)
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
        encoder_data: &Self::CommandEncoderData,
        label: &str,
    ) {
        if let Err(cause) = self
            .0
            .command_encoder_push_debug_group(encoder_data.id, label)
        {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::push_debug_group",
            );
        }
    }

    fn command_encoder_pop_debug_group(&self, encoder_data: &Self::CommandEncoderData) {
        if let Err(cause) = self.0.command_encoder_pop_debug_group(encoder_data.id) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::pop_debug_group",
            );
        }
    }

    fn command_encoder_write_timestamp(
        &self,
        encoder_data: &Self::CommandEncoderData,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        if let Err(cause) =
            self.0
                .command_encoder_write_timestamp(encoder_data.id, *query_set_data, query_index)
        {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::write_timestamp",
            );
        }
    }

    fn command_encoder_resolve_query_set(
        &self,
        encoder_data: &Self::CommandEncoderData,
        query_set_data: &Self::QuerySetData,
        first_query: u32,
        query_count: u32,
        destination_data: &Self::BufferData,
        destination_offset: wgt::BufferAddress,
    ) {
        if let Err(cause) = self.0.command_encoder_resolve_query_set(
            encoder_data.id,
            *query_set_data,
            first_query,
            query_count,
            destination_data.id,
            destination_offset,
        ) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::resolve_query_set",
            );
        }
    }

    fn render_bundle_encoder_finish(
        &self,
        encoder_data: Self::RenderBundleEncoderData,
        desc: &crate::RenderBundleDescriptor<'_>,
    ) -> Self::RenderBundleData {
        let (id, error) = self.0.render_bundle_encoder_finish(
            encoder_data,
            &desc.map_label(|l| l.map(Borrowed)),
            None,
        );
        if let Some(err) = error {
            self.handle_error_fatal(err, "RenderBundleEncoder::finish");
        }
        id
    }

    fn queue_write_buffer(
        &self,
        queue_data: &Self::QueueData,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        data: &[u8],
    ) {
        match self
            .0
            .queue_write_buffer(queue_data.id, buffer_data.id, offset, data)
        {
            Ok(()) => (),
            Err(err) => {
                self.handle_error_nolabel(&queue_data.error_sink, err, "Queue::write_buffer")
            }
        }
    }

    fn queue_validate_write_buffer(
        &self,
        queue_data: &Self::QueueData,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) -> Option<()> {
        match self
            .0
            .queue_validate_write_buffer(queue_data.id, buffer_data.id, offset, size)
        {
            Ok(()) => Some(()),
            Err(err) => {
                self.handle_error_nolabel(&queue_data.error_sink, err, "Queue::write_buffer_with");
                None
            }
        }
    }

    fn queue_create_staging_buffer(
        &self,
        queue_data: &Self::QueueData,
        size: wgt::BufferSize,
    ) -> Option<Box<dyn crate::context::QueueWriteBuffer>> {
        match self
            .0
            .queue_create_staging_buffer(queue_data.id, size, None)
        {
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
        queue_data: &Self::QueueData,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        staging_buffer: &dyn crate::context::QueueWriteBuffer,
    ) {
        let staging_buffer = staging_buffer
            .as_any()
            .downcast_ref::<QueueWriteBuffer>()
            .unwrap();
        match self.0.queue_write_staging_buffer(
            queue_data.id,
            buffer_data.id,
            offset,
            staging_buffer.buffer_id,
        ) {
            Ok(()) => (),
            Err(err) => {
                self.handle_error_nolabel(&queue_data.error_sink, err, "Queue::write_buffer_with");
            }
        }
    }

    fn queue_write_texture(
        &self,
        queue_data: &Self::QueueData,
        texture: crate::ImageCopyTexture<'_>,
        data: &[u8],
        data_layout: wgt::ImageDataLayout,
        size: wgt::Extent3d,
    ) {
        match self.0.queue_write_texture(
            queue_data.id,
            &map_texture_copy_view(texture),
            data,
            &data_layout,
            &size,
        ) {
            Ok(()) => (),
            Err(err) => {
                self.handle_error_nolabel(&queue_data.error_sink, err, "Queue::write_texture")
            }
        }
    }

    #[cfg(any(webgpu, webgl))]
    fn queue_copy_external_image_to_texture(
        &self,
        queue_data: &Self::QueueData,
        source: &wgt::ImageCopyExternalImage,
        dest: crate::ImageCopyTextureTagged<'_>,
        size: wgt::Extent3d,
    ) {
        match self.0.queue_copy_external_image_to_texture(
            queue_data.id,
            source,
            map_texture_tagged_copy_view(dest),
            size,
        ) {
            Ok(()) => (),
            Err(err) => self.handle_error_nolabel(
                &queue_data.error_sink,
                err,
                "Queue::copy_external_image_to_texture",
            ),
        }
    }

    fn queue_submit<I: Iterator<Item = Self::CommandBufferData>>(
        &self,
        queue_data: &Self::QueueData,
        command_buffers: I,
    ) -> Self::SubmissionIndexData {
        let temp_command_buffers = command_buffers.collect::<SmallVec<[_; 4]>>();

        let index = match self.0.queue_submit(queue_data.id, &temp_command_buffers) {
            Ok(index) => index,
            Err((index, err)) => {
                self.handle_error_nolabel(&queue_data.error_sink, err, "Queue::submit");
                index
            }
        };

        for cmdbuf in &temp_command_buffers {
            self.0.command_buffer_drop(*cmdbuf);
        }

        index
    }

    fn queue_get_timestamp_period(&self, queue_data: &Self::QueueData) -> f32 {
        self.0.queue_get_timestamp_period(queue_data.id)
    }

    fn queue_on_submitted_work_done(
        &self,
        queue_data: &Self::QueueData,
        callback: crate::context::SubmittedWorkDoneCallback,
    ) {
        let closure = wgc::device::queue::SubmittedWorkDoneClosure::from_rust(callback);
        self.0.queue_on_submitted_work_done(queue_data.id, closure);
    }

    fn device_start_capture(&self, device_data: &Self::DeviceData) {
        self.0.device_start_capture(device_data.id);
    }

    fn device_stop_capture(&self, device_data: &Self::DeviceData) {
        self.0.device_stop_capture(device_data.id);
    }

    fn device_get_internal_counters(
        &self,
        device_data: &Self::DeviceData,
    ) -> wgt::InternalCounters {
        self.0.device_get_internal_counters(device_data.id)
    }

    fn device_generate_allocator_report(
        &self,
        device_data: &Self::DeviceData,
    ) -> Option<wgt::AllocatorReport> {
        self.0.device_generate_allocator_report(device_data.id)
    }

    fn pipeline_cache_get_data(
        &self,
        // TODO: Used for error handling?
        cache_data: &Self::PipelineCacheData,
    ) -> Option<Vec<u8>> {
        self.0.pipeline_cache_get_data(*cache_data)
    }

    fn compute_pass_set_pipeline(
        &self,
        pass_data: &mut Self::ComputePassData,
        pipeline_data: &Self::ComputePipelineData,
    ) {
        if let Err(cause) = self
            .0
            .compute_pass_set_pipeline(&mut pass_data.pass, pipeline_data.id)
        {
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
        pass_data: &mut Self::ComputePassData,
        index: u32,
        bind_group_data: Option<&Self::BindGroupData>,
        offsets: &[wgt::DynamicOffset],
    ) {
        let bg = bind_group_data.cloned();
        if let Err(cause) =
            self.0
                .compute_pass_set_bind_group(&mut pass_data.pass, index, bg, offsets)
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
        pass_data: &mut Self::ComputePassData,
        offset: u32,
        data: &[u8],
    ) {
        if let Err(cause) =
            self.0
                .compute_pass_set_push_constants(&mut pass_data.pass, offset, data)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::set_push_constant",
            );
        }
    }

    fn compute_pass_insert_debug_marker(&self, pass_data: &mut Self::ComputePassData, label: &str) {
        if let Err(cause) = self
            .0
            .compute_pass_insert_debug_marker(&mut pass_data.pass, label, 0)
        {
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
        pass_data: &mut Self::ComputePassData,
        group_label: &str,
    ) {
        if let Err(cause) =
            self.0
                .compute_pass_push_debug_group(&mut pass_data.pass, group_label, 0)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::push_debug_group",
            );
        }
    }

    fn compute_pass_pop_debug_group(&self, pass_data: &mut Self::ComputePassData) {
        if let Err(cause) = self.0.compute_pass_pop_debug_group(&mut pass_data.pass) {
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
        pass_data: &mut Self::ComputePassData,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        if let Err(cause) =
            self.0
                .compute_pass_write_timestamp(&mut pass_data.pass, *query_set_data, query_index)
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
        pass_data: &mut Self::ComputePassData,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        if let Err(cause) = self.0.compute_pass_begin_pipeline_statistics_query(
            &mut pass_data.pass,
            *query_set_data,
            query_index,
        ) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::begin_pipeline_statistics_query",
            );
        }
    }

    fn compute_pass_end_pipeline_statistics_query(&self, pass_data: &mut Self::ComputePassData) {
        if let Err(cause) = self
            .0
            .compute_pass_end_pipeline_statistics_query(&mut pass_data.pass)
        {
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
        pass_data: &mut Self::ComputePassData,
        x: u32,
        y: u32,
        z: u32,
    ) {
        if let Err(cause) = self
            .0
            .compute_pass_dispatch_workgroups(&mut pass_data.pass, x, y, z)
        {
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
        pass_data: &mut Self::ComputePassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        if let Err(cause) = self.0.compute_pass_dispatch_workgroups_indirect(
            &mut pass_data.pass,
            indirect_buffer_data.id,
            indirect_offset,
        ) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "ComputePass::dispatch_workgroups_indirect",
            );
        }
    }

    fn compute_pass_end(&self, pass_data: &mut Self::ComputePassData) {
        if let Err(cause) = self.0.compute_pass_end(&mut pass_data.pass) {
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
        encoder_data: &mut Self::RenderBundleEncoderData,
        pipeline_data: &Self::RenderPipelineData,
    ) {
        wgpu_render_bundle_set_pipeline(encoder_data, pipeline_data.id)
    }

    fn render_bundle_encoder_set_bind_group(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        index: u32,
        bind_group_data: Option<&Self::BindGroupData>,
        offsets: &[wgt::DynamicOffset],
    ) {
        let bg = bind_group_data.cloned();
        unsafe {
            wgpu_render_bundle_set_bind_group(
                encoder_data,
                index,
                bg,
                offsets.as_ptr(),
                offsets.len(),
            )
        }
    }

    fn render_bundle_encoder_set_index_buffer(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        buffer_data: &Self::BufferData,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        encoder_data.set_index_buffer(buffer_data.id, index_format, offset, size)
    }

    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        slot: u32,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        wgpu_render_bundle_set_vertex_buffer(encoder_data, slot, buffer_data.id, offset, size)
    }

    fn render_bundle_encoder_set_push_constants(
        &self,
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
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        wgpu_render_bundle_draw_indirect(encoder_data, indirect_buffer_data.id, indirect_offset)
    }

    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        encoder_data: &mut Self::RenderBundleEncoderData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        wgpu_render_bundle_draw_indexed_indirect(
            encoder_data,
            indirect_buffer_data.id,
            indirect_offset,
        )
    }

    fn render_pass_set_pipeline(
        &self,
        pass_data: &mut Self::RenderPassData,
        pipeline_data: &Self::RenderPipelineData,
    ) {
        if let Err(cause) = self
            .0
            .render_pass_set_pipeline(&mut pass_data.pass, pipeline_data.id)
        {
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
        pass_data: &mut Self::RenderPassData,
        index: u32,
        bind_group_data: Option<&Self::BindGroupData>,
        offsets: &[wgt::DynamicOffset],
    ) {
        let bg = bind_group_data.cloned();
        if let Err(cause) =
            self.0
                .render_pass_set_bind_group(&mut pass_data.pass, index, bg, offsets)
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
        pass_data: &mut Self::RenderPassData,
        buffer_data: &Self::BufferData,
        index_format: wgt::IndexFormat,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        if let Err(cause) = self.0.render_pass_set_index_buffer(
            &mut pass_data.pass,
            buffer_data.id,
            index_format,
            offset,
            size,
        ) {
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
        pass_data: &mut Self::RenderPassData,
        slot: u32,
        buffer_data: &Self::BufferData,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        if let Err(cause) = self.0.render_pass_set_vertex_buffer(
            &mut pass_data.pass,
            slot,
            buffer_data.id,
            offset,
            size,
        ) {
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
        pass_data: &mut Self::RenderPassData,
        stages: wgt::ShaderStages,
        offset: u32,
        data: &[u8],
    ) {
        if let Err(cause) =
            self.0
                .render_pass_set_push_constants(&mut pass_data.pass, stages, offset, data)
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
        pass_data: &mut Self::RenderPassData,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) {
        if let Err(cause) = self.0.render_pass_draw(
            &mut pass_data.pass,
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
        pass_data: &mut Self::RenderPassData,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) {
        if let Err(cause) = self.0.render_pass_draw_indexed(
            &mut pass_data.pass,
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
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        if let Err(cause) = self.0.render_pass_draw_indirect(
            &mut pass_data.pass,
            indirect_buffer_data.id,
            indirect_offset,
        ) {
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
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
    ) {
        if let Err(cause) = self.0.render_pass_draw_indexed_indirect(
            &mut pass_data.pass,
            indirect_buffer_data.id,
            indirect_offset,
        ) {
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
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
        count: u32,
    ) {
        if let Err(cause) = self.0.render_pass_multi_draw_indirect(
            &mut pass_data.pass,
            indirect_buffer_data.id,
            indirect_offset,
            count,
        ) {
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
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
        count: u32,
    ) {
        if let Err(cause) = self.0.render_pass_multi_draw_indexed_indirect(
            &mut pass_data.pass,
            indirect_buffer_data.id,
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
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
        count_buffer_data: &Self::BufferData,
        count_buffer_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        if let Err(cause) = self.0.render_pass_multi_draw_indirect_count(
            &mut pass_data.pass,
            indirect_buffer_data.id,
            indirect_offset,
            count_buffer_data.id,
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
        pass_data: &mut Self::RenderPassData,
        indirect_buffer_data: &Self::BufferData,
        indirect_offset: wgt::BufferAddress,
        count_buffer_data: &Self::BufferData,
        count_buffer_offset: wgt::BufferAddress,
        max_count: u32,
    ) {
        if let Err(cause) = self.0.render_pass_multi_draw_indexed_indirect_count(
            &mut pass_data.pass,
            indirect_buffer_data.id,
            indirect_offset,
            count_buffer_data.id,
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
        pass_data: &mut Self::RenderPassData,
        color: wgt::Color,
    ) {
        if let Err(cause) = self
            .0
            .render_pass_set_blend_constant(&mut pass_data.pass, color)
        {
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
        pass_data: &mut Self::RenderPassData,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) {
        if let Err(cause) =
            self.0
                .render_pass_set_scissor_rect(&mut pass_data.pass, x, y, width, height)
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
        pass_data: &mut Self::RenderPassData,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        if let Err(cause) = self.0.render_pass_set_viewport(
            &mut pass_data.pass,
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        ) {
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
        pass_data: &mut Self::RenderPassData,
        reference: u32,
    ) {
        if let Err(cause) = self
            .0
            .render_pass_set_stencil_reference(&mut pass_data.pass, reference)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::set_stencil_reference",
            );
        }
    }

    fn render_pass_insert_debug_marker(&self, pass_data: &mut Self::RenderPassData, label: &str) {
        if let Err(cause) = self
            .0
            .render_pass_insert_debug_marker(&mut pass_data.pass, label, 0)
        {
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
        pass_data: &mut Self::RenderPassData,
        group_label: &str,
    ) {
        if let Err(cause) = self
            .0
            .render_pass_push_debug_group(&mut pass_data.pass, group_label, 0)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::push_debug_group",
            );
        }
    }

    fn render_pass_pop_debug_group(&self, pass_data: &mut Self::RenderPassData) {
        if let Err(cause) = self.0.render_pass_pop_debug_group(&mut pass_data.pass) {
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
        pass_data: &mut Self::RenderPassData,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        if let Err(cause) =
            self.0
                .render_pass_write_timestamp(&mut pass_data.pass, *query_set_data, query_index)
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
        pass_data: &mut Self::RenderPassData,
        query_index: u32,
    ) {
        if let Err(cause) = self
            .0
            .render_pass_begin_occlusion_query(&mut pass_data.pass, query_index)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::begin_occlusion_query",
            );
        }
    }

    fn render_pass_end_occlusion_query(&self, pass_data: &mut Self::RenderPassData) {
        if let Err(cause) = self.0.render_pass_end_occlusion_query(&mut pass_data.pass) {
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
        pass_data: &mut Self::RenderPassData,
        query_set_data: &Self::QuerySetData,
        query_index: u32,
    ) {
        if let Err(cause) = self.0.render_pass_begin_pipeline_statistics_query(
            &mut pass_data.pass,
            *query_set_data,
            query_index,
        ) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::begin_pipeline_statistics_query",
            );
        }
    }

    fn render_pass_end_pipeline_statistics_query(&self, pass_data: &mut Self::RenderPassData) {
        if let Err(cause) = self
            .0
            .render_pass_end_pipeline_statistics_query(&mut pass_data.pass)
        {
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
        pass_data: &mut Self::RenderPassData,
        render_bundles: &mut dyn Iterator<Item = &Self::RenderBundleData>,
    ) {
        let temp_render_bundles = render_bundles.copied().collect::<SmallVec<[_; 4]>>();
        if let Err(cause) = self
            .0
            .render_pass_execute_bundles(&mut pass_data.pass, &temp_render_bundles)
        {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::execute_bundles",
            );
        }
    }

    fn render_pass_end(&self, pass_data: &mut Self::RenderPassData) {
        if let Err(cause) = self.0.render_pass_end(&mut pass_data.pass) {
            self.handle_error(
                &pass_data.error_sink,
                cause,
                pass_data.pass.label(),
                "RenderPass::end",
            );
        }
    }

    fn device_create_blas(
        &self,
        device_data: &Self::DeviceData,
        desc: &crate::CreateBlasDescriptor<'_>,
        sizes: wgt::BlasGeometrySizeDescriptors,
    ) -> (Option<u64>, Self::BlasData) {
        let global = &self.0;
        let (id, handle, error) = global.device_create_blas(
            device_data.id,
            &desc.map_label(|l| l.map(Borrowed)),
            sizes,
            None,
        );
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                desc.label,
                "Device::create_blas",
            );
        }
        (
            handle,
            Blas {
                id,
                // error_sink: Arc::clone(&device_data.error_sink),
            },
        )
    }

    fn device_create_tlas(
        &self,
        device_data: &Self::DeviceData,
        desc: &crate::CreateTlasDescriptor<'_>,
    ) -> Self::TlasData {
        let global = &self.0;
        let (id, error) =
            global.device_create_tlas(device_data.id, &desc.map_label(|l| l.map(Borrowed)), None);
        if let Some(cause) = error {
            self.handle_error(
                &device_data.error_sink,
                cause,
                desc.label,
                "Device::create_blas",
            );
        }
        Tlas {
            id,
            // error_sink: Arc::clone(&device_data.error_sink),
        }
    }

    fn command_encoder_build_acceleration_structures_unsafe_tlas<'a>(
        &'a self,
        encoder_data: &Self::CommandEncoderData,
        blas: impl Iterator<Item = crate::ContextBlasBuildEntry<'a, Self>>,
        tlas: impl Iterator<Item = crate::ContextTlasBuildEntry<'a, Self>>,
    ) {
        let global = &self.0;

        let blas = blas.map(|e: crate::ContextBlasBuildEntry<'_, Self>| {
            let geometries = match e.geometries {
                crate::ContextBlasGeometries::TriangleGeometries(triangle_geometries) => {
                    let iter = triangle_geometries.into_iter().map(|tg| {
                        wgc::ray_tracing::BlasTriangleGeometry {
                            vertex_buffer: tg.vertex_buffer.id,
                            index_buffer: tg.index_buffer.map(|buf| buf.id),
                            transform_buffer: tg.transform_buffer.map(|buf| buf.id),
                            size: tg.size,
                            transform_buffer_offset: tg.transform_buffer_offset,
                            first_vertex: tg.first_vertex,
                            vertex_stride: tg.vertex_stride,
                            index_buffer_offset: tg.index_buffer_offset,
                        }
                    });
                    wgc::ray_tracing::BlasGeometries::TriangleGeometries(Box::new(iter))
                }
            };
            wgc::ray_tracing::BlasBuildEntry {
                blas_id: e.blas_data.id,
                geometries,
            }
        });

        let tlas = tlas
            .into_iter()
            .map(|e: crate::ContextTlasBuildEntry<'a, ContextWgpuCore>| {
                wgc::ray_tracing::TlasBuildEntry {
                    tlas_id: e.tlas_data.id,
                    instance_buffer_id: e.instance_buffer_data.id,
                    instance_count: e.instance_count,
                }
            });

        if let Err(cause) = global.command_encoder_build_acceleration_structures_unsafe_tlas(
            encoder_data.id,
            blas,
            tlas,
        ) {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::build_acceleration_structures_unsafe_tlas",
            );
        }
    }

    fn command_encoder_build_acceleration_structures<'a>(
        &'a self,
        encoder_data: &Self::CommandEncoderData,
        blas: impl Iterator<Item = crate::ContextBlasBuildEntry<'a, Self>>,
        tlas: impl Iterator<Item = crate::ContextTlasPackage<'a, Self>>,
    ) {
        let global = &self.0;

        let blas = blas.map(|e: crate::ContextBlasBuildEntry<'_, Self>| {
            let geometries = match e.geometries {
                crate::ContextBlasGeometries::TriangleGeometries(triangle_geometries) => {
                    let iter = triangle_geometries.into_iter().map(|tg| {
                        wgc::ray_tracing::BlasTriangleGeometry {
                            vertex_buffer: tg.vertex_buffer.id,
                            index_buffer: tg.index_buffer.map(|buf| buf.id),
                            transform_buffer: tg.transform_buffer.map(|buf| buf.id),
                            size: tg.size,
                            transform_buffer_offset: tg.transform_buffer_offset,
                            first_vertex: tg.first_vertex,
                            vertex_stride: tg.vertex_stride,
                            index_buffer_offset: tg.index_buffer_offset,
                        }
                    });
                    wgc::ray_tracing::BlasGeometries::TriangleGeometries(Box::new(iter))
                }
            };
            wgc::ray_tracing::BlasBuildEntry {
                blas_id: e.blas_data.id,
                geometries,
            }
        });

        let tlas = tlas.into_iter().map(|e| {
            let instances =
                e.instances
                    .map(|instance: Option<crate::ContextTlasInstance<'_, _>>| {
                        instance.map(|instance| wgc::ray_tracing::TlasInstance {
                            blas_id: instance.blas_data.id,
                            transform: instance.transform,
                            custom_index: instance.custom_index,
                            mask: instance.mask,
                        })
                    });
            wgc::ray_tracing::TlasPackage {
                tlas_id: e.tlas_data.id,
                instances: Box::new(instances),
                lowest_unmodified: e.lowest_unmodified,
            }
        });

        if let Err(cause) =
            global.command_encoder_build_acceleration_structures(encoder_data.id, blas, tlas)
        {
            self.handle_error_nolabel(
                &encoder_data.error_sink,
                cause,
                "CommandEncoder::build_acceleration_structures_unsafe_tlas",
            );
        }
    }

    fn blas_destroy(&self, blas_data: &Self::BlasData) {
        let global = &self.0;
        let _ = global.blas_destroy(blas_data.id);
    }

    fn blas_drop(&self, blas_data: &Self::BlasData) {
        let global = &self.0;
        global.blas_drop(blas_data.id)
    }

    fn tlas_destroy(&self, tlas_data: &Self::TlasData) {
        let global = &self.0;
        let _ = global.tlas_destroy(tlas_data.id);
    }

    fn tlas_drop(&self, tlas_data: &Self::TlasData) {
        let global = &self.0;
        global.tlas_drop(tlas_data.id)
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
    uncaptured_handler: Option<Box<dyn crate::UncapturedErrorHandler>>,
}

impl ErrorSinkRaw {
    fn new() -> ErrorSinkRaw {
        ErrorSinkRaw {
            scopes: Vec::new(),
            uncaptured_handler: None,
        }
    }

    #[track_caller]
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
                if let Some(custom_handler) = self.uncaptured_handler.as_ref() {
                    (custom_handler)(err);
                } else {
                    // direct call preserves #[track_caller] where dyn can't
                    default_error_handler(err);
                }
            }
        }
    }
}

impl fmt::Debug for ErrorSinkRaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ErrorSink")
    }
}

#[track_caller]
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

fn downcast_buffer(buffer: &crate::Buffer) -> &<ContextWgpuCore as crate::Context>::BufferData {
    downcast_ref(buffer.data.as_ref())
}
fn downcast_texture(texture: &crate::Texture) -> &<ContextWgpuCore as crate::Context>::TextureData {
    downcast_ref(texture.data.as_ref())
}
fn downcast_texture_view(
    texture_view: &crate::TextureView,
) -> &<ContextWgpuCore as crate::Context>::TextureViewData {
    downcast_ref(texture_view.data.as_ref())
}
fn downcast_tlas(tlas: &crate::Tlas) -> &<ContextWgpuCore as crate::Context>::TlasData {
    downcast_ref(tlas.data.as_ref())
}
fn downcast_sampler(sampler: &crate::Sampler) -> &<ContextWgpuCore as crate::Context>::SamplerData {
    downcast_ref(sampler.data.as_ref())
}
fn downcast_query_set(
    query_set: &crate::QuerySet,
) -> &<ContextWgpuCore as crate::Context>::QuerySetData {
    downcast_ref(query_set.data.as_ref())
}
fn downcast_bind_group_layout(
    bind_group_layout: &crate::BindGroupLayout,
) -> &<ContextWgpuCore as crate::Context>::BindGroupLayoutData {
    downcast_ref(bind_group_layout.data.as_ref())
}
fn downcast_pipeline_layout(
    pipeline_layout: &crate::PipelineLayout,
) -> &<ContextWgpuCore as crate::Context>::PipelineLayoutData {
    downcast_ref(pipeline_layout.data.as_ref())
}
fn downcast_shader_module(
    shader_module: &crate::ShaderModule,
) -> &<ContextWgpuCore as crate::Context>::ShaderModuleData {
    downcast_ref(shader_module.data.as_ref())
}
fn downcast_pipeline_cache(
    pipeline_cache: &crate::PipelineCache,
) -> &<ContextWgpuCore as crate::Context>::PipelineCacheData {
    downcast_ref(pipeline_cache.data.as_ref())
}
