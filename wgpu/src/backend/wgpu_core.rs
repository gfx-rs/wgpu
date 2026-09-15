use alloc::borrow::ToOwned;
use alloc::{
    borrow::Cow::{self, Borrowed},
    boxed::Box,
    format,
    string::{String, ToString as _},
    sync::Arc,
    vec,
    vec::Vec,
};
use core::{
    error::Error,
    fmt,
    future::ready,
    ops::{Deref, Range},
    pin::Pin,
    ptr::NonNull,
    slice,
};
use hashbrown::HashMap;

use arrayvec::ArrayVec;
use smallvec::SmallVec;
use wgc::{
    error::ContextErrorSource, pipeline::CreateShaderModuleError,
    resource::BlasPrepareCompactResult,
};
use wgt::{
    error::{ErrorType, WebGpuError},
    WasmNotSendSync,
};

use crate::{
    api,
    dispatch::{self, BlasCompactCallback, BufferMappedRangeInterface},
    BindingResource, Blas, BufferBinding, BufferDescriptor, CompilationInfo, CompilationMessage,
    CompilationMessageType, ErrorSource, Features, Label, LoadOp, MapMode, Operations,
    ShaderSource, SurfaceTargetUnsafe, TextureDescriptor, Tlas, WriteOnly,
};
use crate::{dispatch::DispatchAdapter, util::Mutex};

mod thread_id;

#[derive(Clone)]
pub struct ContextWgpuCore(Arc<wgc::instance::Instance>);

impl fmt::Debug for ContextWgpuCore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextWgpuCore")
            .field("type", &"Native")
            .finish()
    }
}

impl ContextWgpuCore {
    pub unsafe fn from_hal_instance<A: hal::Api>(hal_instance: A::Instance) -> Self {
        Self(wgc::instance::Instance::from_hal_instance::<A>(
            "wgpu".to_owned(),
            hal_instance,
        ))
    }

    /// # Safety
    ///
    /// - The raw instance handle returned must not be manually destroyed.
    pub unsafe fn instance_as_hal<A: hal::Api>(&self) -> Option<&A::Instance> {
        unsafe { self.0.as_hal::<A>() }
    }

    pub unsafe fn from_core_instance(core_instance: Arc<wgc::instance::Instance>) -> Self {
        Self(core_instance)
    }

    #[cfg(wgpu_core)]
    pub fn enumerate_adapters(&self, backends: wgt::Backends) -> Vec<Arc<wgc::instance::Adapter>> {
        self.0
            .enumerate_adapters(backends, false /* no limit bucketing */)
    }

    pub unsafe fn create_adapter_from_hal<A: hal::Api>(
        &self,
        hal_adapter: hal::ExposedAdapter<A>,
    ) -> Arc<wgc::instance::Adapter> {
        unsafe { self.0.create_adapter_from_hal(hal_adapter.into()) }
    }

    pub unsafe fn adapter_as_hal<A: hal::Api>(
        &self,
        adapter: &CoreAdapter,
    ) -> Option<impl Deref<Target = A::Adapter> + WasmNotSendSync> {
        unsafe { adapter.wgpu_adapter.clone().as_hal::<A>() }
    }

    pub unsafe fn buffer_as_hal<A: hal::Api>(
        &self,
        buffer: &CoreBuffer,
    ) -> Option<impl Deref<Target = A::Buffer>> {
        unsafe { buffer.wgpu_buffer.clone().as_hal::<A>() }
    }

    pub unsafe fn create_device_from_hal<A: hal::Api>(
        &self,
        adapter: &CoreAdapter,
        hal_device: hal::OpenDevice<A>,
        desc: &crate::DeviceDescriptor<'_>,
    ) -> Result<(CoreDevice, CoreQueue), crate::RequestDeviceError> {
        let (device, queue) = unsafe {
            adapter.wgpu_adapter.create_device_and_queue_from_hal(
                hal_device.into(),
                &desc.map_label(|l| l.map(Borrowed)),
            )
        }?;
        let error_sink = Arc::new(Mutex::new(ErrorSinkRaw::new()));
        let device = CoreDevice {
            context: self.clone(),
            wgpu_device: device,
            error_sink: error_sink.clone(),
            features: desc.required_features,
        };
        let queue = CoreQueue {
            context: self.clone(),
            wgpu_queue: queue,
            error_sink,
        };
        Ok((device, queue))
    }

    pub unsafe fn create_texture_from_hal<A: hal::Api>(
        &self,
        hal_texture: A::Texture,
        device: &CoreDevice,
        desc: &TextureDescriptor<'_>,
        initial_state: wgt::TextureUses,
    ) -> CoreTexture {
        let descriptor = desc.map_label_and_view_formats(|l| l.map(Borrowed), |v| v.to_vec());
        let (wgpu_texture, error) = unsafe {
            device.wgpu_device.create_texture_from_hal(
                Box::new(hal_texture),
                &descriptor,
                initial_state,
            )
        };
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                desc.label,
                "Device::create_texture_from_hal",
            );
        }
        CoreTexture {
            context: self.clone(),
            wgpu_texture,
            error_sink: Arc::clone(&device.error_sink),
        }
    }

    /// # Safety
    ///
    /// - `hal_buffer` must be created from `device`.
    /// - `hal_buffer` must be created respecting `desc`
    /// - `hal_buffer` must be initialized
    /// - `hal_buffer` must not have zero size.
    pub unsafe fn create_buffer_from_hal<A: hal::Api>(
        &self,
        hal_buffer: A::Buffer,
        device: &CoreDevice,
        desc: &BufferDescriptor<'_>,
    ) -> CoreBuffer {
        let (wgpu_buffer, error) = unsafe {
            device
                .wgpu_device
                .create_buffer_from_hal(Box::new(hal_buffer), &desc.map_label(|l| l.map(Borrowed)))
        };
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                desc.label,
                "Device::create_buffer_from_hal",
            );
        }
        CoreBuffer {
            context: self.clone(),
            wgpu_buffer,
            error_sink: Arc::clone(&device.error_sink),
        }
    }

    pub unsafe fn device_as_hal<A: hal::Api>(
        &self,
        device: &CoreDevice,
    ) -> Option<impl Deref<Target = A::Device>> {
        unsafe { device.wgpu_device.clone().as_hal::<A>() }
    }

    pub unsafe fn surface_as_hal<A: hal::Api>(
        &self,
        surface: &CoreSurface,
    ) -> Option<impl Deref<Target = A::Surface>> {
        unsafe { surface.wgpu_surface.clone().as_hal::<A>() }
    }

    pub unsafe fn texture_as_hal<A: hal::Api>(
        &self,
        texture: &CoreTexture,
    ) -> Option<impl Deref<Target = A::Texture>> {
        unsafe { texture.wgpu_texture.clone().as_hal::<A>() }
    }

    pub unsafe fn texture_view_as_hal<A: hal::Api>(
        &self,
        texture_view: &CoreTextureView,
    ) -> Option<impl Deref<Target = A::TextureView>> {
        unsafe { texture_view.wgpu_texture_view.clone().as_hal::<A>() }
    }

    /// This method will start the wgpu_core level command recording.
    pub unsafe fn command_encoder_as_hal_mut<
        A: hal::Api,
        F: FnOnce(Option<&mut A::CommandEncoder>) -> R,
        R,
    >(
        &self,
        command_encoder: &CoreCommandEncoder,
        hal_command_encoder_callback: F,
    ) -> R {
        unsafe {
            command_encoder
                .wgpu_command_encoder
                .as_hal_mut::<A, F, R>(hal_command_encoder_callback)
        }
    }

    pub unsafe fn blas_as_hal<A: hal::Api>(
        &self,
        blas: &CoreBlas,
    ) -> Option<impl Deref<Target = A::AccelerationStructure>> {
        unsafe { blas.wgpu_blas.clone().as_hal::<A>() }
    }

    pub unsafe fn tlas_as_hal<A: hal::Api>(
        &self,
        tlas: &CoreTlas,
    ) -> Option<impl Deref<Target = A::AccelerationStructure>> {
        unsafe { tlas.wgpu_tlas.clone().as_hal::<A>() }
    }

    #[cold]
    #[track_caller]
    #[inline(never)]
    fn handle_error_inner(
        &self,
        sink_mutex: &Mutex<ErrorSinkRaw>,
        error_type: ErrorType,
        source: ContextErrorSource,
        label: Label<'_>,
        fn_ident: &'static str,
    ) {
        let source: ErrorSource = Box::new(wgc::error::ContextError {
            fn_ident,
            source,
            label: label.unwrap_or_default().to_string(),
        });
        let final_error_handling = {
            let mut sink = sink_mutex.lock();
            let description = || self.format_error(&*source);
            let error = match error_type {
                ErrorType::Internal => {
                    let description = description();
                    crate::Error::Internal {
                        source,
                        description,
                    }
                }
                ErrorType::OutOfMemory => crate::Error::OutOfMemory { source },
                ErrorType::Validation => {
                    let description = description();
                    crate::Error::Validation {
                        source,
                        description,
                    }
                }
                ErrorType::DeviceLost => return, // will be surfaced via callback
            };
            sink.handle_error_or_return_handler(error)
        };

        if let Some(f) = final_error_handling {
            // If the user has provided their own `uncaptured_handler` callback, invoke it now,
            // having released our lock on `sink_mutex`. See the comments on
            // `handle_error_or_return_handler` for details.
            f();
        }
    }

    #[inline]
    #[track_caller]
    fn handle_error(
        &self,
        sink_mutex: &Mutex<ErrorSinkRaw>,
        source: impl WebGpuError + WasmNotSendSync + 'static,
        label: Label<'_>,
        fn_ident: &'static str,
    ) {
        let error_type = source.webgpu_error_type();
        self.handle_error_inner(sink_mutex, error_type, Box::new(source), label, fn_ident)
    }

    #[inline]
    #[track_caller]
    fn handle_error_nolabel(
        &self,
        sink_mutex: &Mutex<ErrorSinkRaw>,
        source: impl WebGpuError + WasmNotSendSync + 'static,
        fn_ident: &'static str,
    ) {
        let error_type = source.webgpu_error_type();
        self.handle_error_inner(sink_mutex, error_type, Box::new(source), None, fn_ident)
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
                use core::fmt::Write;
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

    pub unsafe fn queue_as_hal<A: hal::Api>(
        &self,
        queue: &CoreQueue,
    ) -> Option<impl Deref<Target = A::Queue> + WasmNotSendSync> {
        unsafe { queue.wgpu_queue.clone().as_hal::<A>() }
    }
}

fn map_buffer_copy_view(
    view: crate::TexelCopyBufferInfo<'_>,
) -> wgt::TexelCopyBufferInfo<Arc<wgc::resource::Buffer>> {
    wgt::TexelCopyBufferInfo {
        buffer: view.buffer.inner.as_core().wgpu_buffer.clone(),
        layout: view.layout,
    }
}

fn map_texture_copy_view(
    view: crate::TexelCopyTextureInfo<'_>,
) -> wgt::TexelCopyTextureInfo<Arc<wgc::resource::Texture>> {
    wgt::TexelCopyTextureInfo {
        texture: view.texture.inner.as_core().wgpu_texture.clone(),
        mip_level: view.mip_level,
        origin: view.origin,
        aspect: view.aspect,
    }
}

#[cfg_attr(not(webgl), expect(unused))]
fn map_texture_tagged_copy_view(
    view: crate::CopyExternalImageDestInfo<&api::Texture>,
) -> wgt::CopyExternalImageDestInfo<Arc<wgc::resource::Texture>> {
    wgt::CopyExternalImageDestInfo {
        texture: view.texture.inner.as_core().wgpu_texture.clone(),
        mip_level: view.mip_level,
        origin: view.origin,
        aspect: view.aspect,
        color_space: view.color_space,
        premultiplied_alpha: view.premultiplied_alpha,
    }
}

fn map_load_op<V: Copy>(load: &LoadOp<V>) -> LoadOp<Option<V>> {
    match *load {
        LoadOp::Clear(clear_value) => LoadOp::Clear(Some(clear_value)),
        LoadOp::DontCare(token) => LoadOp::DontCare(token),
        LoadOp::Load => LoadOp::Load,
    }
}

fn map_pass_channel<V: Copy>(ops: Option<&Operations<V>>) -> wgc::command::PassChannel<Option<V>> {
    match ops {
        Some(&Operations { load, store }) => wgc::command::PassChannel {
            load_op: Some(map_load_op(&load)),
            store_op: Some(store),
            read_only: false,
        },
        None => wgc::command::PassChannel {
            load_op: None,
            store_op: None,
            read_only: true,
        },
    }
}

pub struct CoreSurface {
    pub(crate) context: ContextWgpuCore,
    pub(crate) wgpu_surface: Arc<wgc::instance::Surface>,
    /// Configured device is needed to know which backend
    /// code to execute when acquiring a new frame.
    configured_device: Mutex<Option<Arc<wgc::device::Device>>>,
    /// The error sink with which to report errors.
    /// `None` if the surface has not been configured.
    error_sink: Mutex<Option<ErrorSink>>,
}

impl fmt::Debug for CoreSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoreSurface")
            .field("context", &self.context)
            .field("wgpu_surface", &Arc::as_ptr(&self.wgpu_surface))
            .field("configured_device", &self.configured_device)
            .field("error_sink", &self.error_sink)
            .finish()
    }
}

pub struct CoreAdapter {
    pub(crate) context: ContextWgpuCore,
    pub(crate) wgpu_adapter: Arc<wgc::instance::Adapter>,
}

impl fmt::Debug for CoreAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoreAdapter")
            .field("context", &self.context)
            .field("wgpu_adapter", &Arc::as_ptr(&self.wgpu_adapter))
            .finish()
    }
}

#[derive(Debug)]
pub struct CoreDevice {
    pub(crate) context: ContextWgpuCore,
    pub(crate) wgpu_device: Arc<wgc::device::Device>,
    error_sink: ErrorSink,
    features: Features,
}

#[derive(Debug)]
pub struct CoreBuffer {
    pub(crate) context: ContextWgpuCore,
    pub(crate) wgpu_buffer: Arc<wgc::resource::Buffer>,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct CoreShaderModule {
    pub(crate) wgpu_shader_module: Arc<wgc::pipeline::ShaderModule>,
    compilation_info: CompilationInfo,
}

#[derive(Debug)]
pub struct CoreBindGroupLayout {
    pub(crate) wgpu_bind_group_layout: Arc<wgc::binding_model::BindGroupLayout>,
}

#[derive(Debug)]
pub struct CoreBindGroup {
    pub(crate) wgpu_bind_group: Arc<wgc::binding_model::BindGroup>,
}

#[derive(Debug)]
pub struct CoreTexture {
    pub(crate) context: ContextWgpuCore,
    pub(crate) wgpu_texture: Arc<wgc::resource::Texture>,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct CoreTextureView {
    pub(crate) context: ContextWgpuCore,
    pub(crate) wgpu_texture_view: Arc<wgc::resource::TextureView>,
}

#[derive(Debug)]
pub struct CoreExternalTexture {
    pub(crate) wgpu_external_texture: Arc<wgc::resource::ExternalTexture>,
}

#[derive(Debug)]
pub struct CoreSampler {
    pub(crate) wgpu_sampler: Arc<wgc::resource::Sampler>,
}

#[derive(Debug)]
pub struct CoreQuerySet {
    pub(crate) wgpu_query_set: Arc<wgc::resource::QuerySet>,
}

#[derive(Debug)]
pub struct CorePipelineLayout {
    pub(crate) wgpu_pipeline_layout: Arc<wgc::binding_model::PipelineLayout>,
}

#[derive(Debug)]
pub struct CorePipelineCache {
    pub(crate) wgpu_pipeline_cache: Arc<wgc::pipeline::PipelineCache>,
}

pub struct CoreCommandBuffer {
    pub(crate) context: ContextWgpuCore,
    pub(crate) wgpu_command_buffer: Arc<wgc::command::CommandBuffer>,
}

impl fmt::Debug for CoreCommandBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoreCommandBuffer")
            .field("context", &self.context)
            .field(
                "wgpu_command_buffer",
                &Arc::as_ptr(&self.wgpu_command_buffer),
            )
            .finish()
    }
}

#[derive(Debug)]
pub struct CoreRenderBundleEncoder {
    pub(crate) context: ContextWgpuCore,
    error_sink: ErrorSink,
    encoder: Box<wgc::command::RenderBundleEncoder>,
}

#[derive(Debug)]
pub struct CoreRenderBundle {
    pub(crate) wgpu_render_bundle: Arc<wgc::command::RenderBundle>,
}

pub struct CoreQueue {
    pub(crate) context: ContextWgpuCore,
    pub(crate) wgpu_queue: Arc<wgc::device::queue::Queue>,
    error_sink: ErrorSink,
}

impl fmt::Debug for CoreQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoreQueue")
            .field("context", &self.context)
            .field("wgpu_queue", &Arc::as_ptr(&self.wgpu_queue))
            .field("error_sink", &self.error_sink)
            .finish()
    }
}

#[derive(Debug)]
pub struct CoreComputePipeline {
    pub(crate) context: ContextWgpuCore,
    pub(crate) wgpu_compute_pipeline: Arc<wgc::pipeline::ComputePipeline>,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct CoreRenderPipeline {
    pub(crate) context: ContextWgpuCore,
    pub(crate) wgpu_render_pipeline: Arc<wgc::pipeline::RenderPipeline>,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct CoreComputePass {
    pub(crate) context: ContextWgpuCore,
    pass: wgc::command::ComputePass,
    error_sink: ErrorSink,
    id: crate::cmp::Identifier,
}

#[derive(Debug)]
pub struct CoreRenderPass {
    pub(crate) context: ContextWgpuCore,
    pass: wgc::command::RenderPass,
    error_sink: ErrorSink,
    id: crate::cmp::Identifier,
}

pub struct CoreCommandEncoder {
    pub(crate) context: ContextWgpuCore,
    pub(crate) wgpu_command_encoder: Arc<wgc::command::CommandEncoder>,
    error_sink: ErrorSink,
}

impl fmt::Debug for CoreCommandEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoreCommandEncoder")
            .field("context", &self.context)
            .field(
                "wgpu_command_encoder",
                &Arc::as_ptr(&self.wgpu_command_encoder),
            )
            .field("error_sink", &self.error_sink)
            .finish()
    }
}

#[derive(Debug)]
pub struct CoreBlas {
    pub(crate) context: ContextWgpuCore,
    pub(crate) wgpu_blas: Arc<wgc::resource::Blas>,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct CoreTlas {
    pub(crate) context: ContextWgpuCore,
    pub(crate) wgpu_tlas: Arc<wgc::resource::Tlas>,
    // error_sink: ErrorSink,
}

pub struct CoreSurfaceOutputDetail {
    pub(crate) context: ContextWgpuCore,
    wgpu_surface: Arc<wgc::instance::Surface>,
    error_sink: ErrorSink,
}

impl fmt::Debug for CoreSurfaceOutputDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoreSurfaceOutputDetail")
            .field("context", &self.context)
            .field("wgpu_surface", &Arc::as_ptr(&self.wgpu_surface))
            .field("error_sink", &self.error_sink)
            .finish()
    }
}

type ErrorSink = Arc<Mutex<ErrorSinkRaw>>;

struct ErrorScope {
    error: Option<crate::Error>,
    filter: crate::ErrorFilter,
}

struct ErrorSinkRaw {
    scopes: HashMap<thread_id::ThreadId, Vec<ErrorScope>>,
    uncaptured_handler: Option<Arc<dyn crate::UncapturedErrorHandler>>,
}

impl ErrorSinkRaw {
    fn new() -> ErrorSinkRaw {
        ErrorSinkRaw {
            scopes: HashMap::new(),
            uncaptured_handler: None,
        }
    }

    /// Deliver the error to
    ///
    /// * the innermost error scope, if any, or
    /// * the uncaptured error handler, if there is one, or
    /// * [`default_error_handler()`].
    ///
    /// If a closure is returned, the caller should call it immediately after dropping the
    /// [`ErrorSink`] mutex guard. This makes sure that the user callback is not called with
    /// a wgpu mutex held.
    #[track_caller]
    #[must_use]
    fn handle_error_or_return_handler(&mut self, err: crate::Error) -> Option<impl FnOnce()> {
        let filter = match err {
            crate::Error::OutOfMemory { .. } => crate::ErrorFilter::OutOfMemory,
            crate::Error::Validation { .. } => crate::ErrorFilter::Validation,
            crate::Error::Internal { .. } => crate::ErrorFilter::Internal,
        };
        let thread_id = thread_id::ThreadId::current();
        let scopes = self.scopes.entry(thread_id).or_default();
        match scopes.iter_mut().rev().find(|scope| scope.filter == filter) {
            Some(scope) => {
                if scope.error.is_none() {
                    scope.error = Some(err);
                }
                None
            }
            None => {
                if let Some(custom_handler) = &self.uncaptured_handler {
                    let custom_handler = Arc::clone(custom_handler);
                    Some(move || (custom_handler)(err))
                } else {
                    // direct call preserves #[track_caller] where dyn can't
                    default_error_handler(err)
                }
            }
        }
    }
}

impl fmt::Debug for ErrorSinkRaw {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ErrorSink")
    }
}

#[track_caller]
fn default_error_handler(err: crate::Error) -> ! {
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
pub struct CoreQueueWriteBuffer {
    wgpu_staging_buffer: wgc::resource::StagingBuffer,
    mapping: CoreBufferMappedRange,
}

#[derive(Debug)]
pub struct CoreBufferMappedRange {
    ptr: NonNull<u8>,
    size: usize,
}

#[cfg(send_sync)]
unsafe impl Send for CoreBufferMappedRange {}
#[cfg(send_sync)]
unsafe impl Sync for CoreBufferMappedRange {}

impl Drop for CoreBufferMappedRange {
    fn drop(&mut self) {
        // Intentionally left blank so that `BufferMappedRange` still
        // implements `Drop`, to match the web backend
    }
}

crate::cmp::impl_eq_ord_hash_arc_address!(ContextWgpuCore => .0);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreAdapter => .wgpu_adapter);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreDevice => .wgpu_device);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreQueue => .wgpu_queue);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreShaderModule => .wgpu_shader_module);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreBindGroupLayout => .wgpu_bind_group_layout);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreBindGroup => .wgpu_bind_group);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreTextureView => .wgpu_texture_view);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreSampler => .wgpu_sampler);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreBuffer => .wgpu_buffer);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreTexture => .wgpu_texture);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreExternalTexture => .wgpu_external_texture);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreBlas => .wgpu_blas);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreTlas => .wgpu_tlas);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreQuerySet => .wgpu_query_set);
crate::cmp::impl_eq_ord_hash_arc_address!(CorePipelineLayout => .wgpu_pipeline_layout);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreRenderPipeline => .wgpu_render_pipeline);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreComputePipeline => .wgpu_compute_pipeline);
crate::cmp::impl_eq_ord_hash_arc_address!(CorePipelineCache => .wgpu_pipeline_cache);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreCommandEncoder => .wgpu_command_encoder);
crate::cmp::impl_eq_ord_hash_proxy!(CoreComputePass => .id);
crate::cmp::impl_eq_ord_hash_proxy!(CoreRenderPass => .id);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreCommandBuffer => .wgpu_command_buffer);
crate::cmp::impl_eq_ord_hash_box_address!(CoreRenderBundleEncoder => .encoder);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreRenderBundle => .wgpu_render_bundle);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreSurface => .wgpu_surface);
crate::cmp::impl_eq_ord_hash_arc_address!(CoreSurfaceOutputDetail => .wgpu_surface);
crate::cmp::impl_eq_ord_hash_proxy!(CoreQueueWriteBuffer => .mapping.ptr);
crate::cmp::impl_eq_ord_hash_proxy!(CoreBufferMappedRange => .ptr);

impl dispatch::InstanceInterface for ContextWgpuCore {
    fn new(desc: wgt::InstanceDescriptor) -> Self
    where
        Self: Sized,
    {
        Self(wgc::instance::Instance::new("wgpu", desc, None))
    }

    unsafe fn create_surface(
        &self,
        target: crate::api::SurfaceTargetUnsafe,
    ) -> Result<dispatch::DispatchSurface, crate::CreateSurfaceError> {
        let wgpu_surface = match target {
            SurfaceTargetUnsafe::RawHandle {
                raw_display_handle,
                raw_window_handle,
            } => unsafe { self.0.create_surface(raw_display_handle, raw_window_handle) },

            #[cfg(all(drm, not(target_os = "netbsd")))]
            SurfaceTargetUnsafe::Drm {
                fd,
                plane,
                connector_id,
                width,
                height,
                refresh_rate,
            } => unsafe {
                self.0
                    .create_surface_from_drm(fd, plane, connector_id, width, height, refresh_rate)
            },

            #[cfg(metal)]
            SurfaceTargetUnsafe::CoreAnimationLayer(layer) => unsafe {
                self.0.create_surface_metal(layer)
            },

            #[cfg(all(drm, target_os = "netbsd"))]
            SurfaceTargetUnsafe::Drm { .. } => Err(
                wgc::instance::CreateSurfaceError::BackendNotEnabled(wgt::Backend::Vulkan),
            ),

            #[cfg(dx12)]
            SurfaceTargetUnsafe::CompositionVisual(visual) => unsafe {
                self.0.create_surface_from_visual(visual)
            },

            #[cfg(dx12)]
            SurfaceTargetUnsafe::SurfaceHandle(surface_handle) => unsafe {
                self.0.create_surface_from_surface_handle(surface_handle)
            },

            #[cfg(dx12)]
            SurfaceTargetUnsafe::SwapChainPanel(swap_chain_panel) => unsafe {
                self.0
                    .create_surface_from_swap_chain_panel(swap_chain_panel)
            },
        }?;

        Ok(CoreSurface {
            context: self.clone(),
            wgpu_surface,
            configured_device: Mutex::default(),
            error_sink: Mutex::default(),
        }
        .into())
    }

    fn request_adapter(
        &self,
        options: &crate::api::RequestAdapterOptions<'_, '_>,
    ) -> Pin<Box<dyn dispatch::RequestAdapterFuture>> {
        let adapter = self.0.request_adapter(
            &wgt::RequestAdapterOptions {
                power_preference: options.power_preference,
                force_fallback_adapter: options.force_fallback_adapter,
                compatible_surface: options
                    .compatible_surface
                    .map(|surface| &*surface.inner.as_core().wgpu_surface),
                apply_limit_buckets: false,
            },
            wgt::Backends::all(),
        );
        let adapter = adapter.map(|wgpu_adapter| {
            let core = CoreAdapter {
                context: self.clone(),
                wgpu_adapter,
            };
            let generic: dispatch::DispatchAdapter = core.into();
            generic
        });
        Box::pin(ready(adapter))
    }

    fn poll_all_devices(&self, force_wait: bool) -> bool {
        match self.0.poll_all_devices(force_wait) {
            Ok(all_queue_empty) => all_queue_empty,
            Err(err) => self.handle_error_fatal(err, "Instance::poll_all_devices"),
        }
    }

    #[cfg(feature = "wgsl")]
    fn wgsl_language_features(&self) -> crate::WgslLanguageFeatures {
        use wgc::naga::front::wgsl::ImplementedLanguageExtension;
        ImplementedLanguageExtension::all().iter().copied().fold(
            crate::WgslLanguageFeatures::empty(),
            |acc, wle| {
                acc | match wle {
                    ImplementedLanguageExtension::ReadOnlyAndReadWriteStorageTextures => {
                        crate::WgslLanguageFeatures::ReadOnlyAndReadWriteStorageTextures
                    }
                    ImplementedLanguageExtension::Packed4x8IntegerDotProduct => {
                        crate::WgslLanguageFeatures::Packed4x8IntegerDotProduct
                    }
                    ImplementedLanguageExtension::PointerCompositeAccess => {
                        crate::WgslLanguageFeatures::PointerCompositeAccess
                    }
                    ImplementedLanguageExtension::ImmediateAddressSpace => {
                        crate::WgslLanguageFeatures::ImmediateAddressSpace
                    }
                }
            },
        )
    }

    fn enumerate_adapters(
        &self,
        backends: crate::Backends,
    ) -> Pin<Box<dyn dispatch::EnumerateAdapterFuture>> {
        let adapters: Vec<DispatchAdapter> = self
            .enumerate_adapters(backends)
            .into_iter()
            .map(|adapter| {
                let core = crate::backend::wgpu_core::CoreAdapter {
                    context: self.clone(),
                    wgpu_adapter: adapter,
                };
                core.into()
            })
            .collect();
        Box::pin(ready(adapters))
    }
}

impl dispatch::AdapterInterface for CoreAdapter {
    fn request_device(
        &self,
        desc: &crate::DeviceDescriptor<'_>,
    ) -> Pin<Box<dyn dispatch::RequestDeviceFuture>> {
        let res = self
            .wgpu_adapter
            .request_device(&desc.map_label(|l| l.map(Borrowed)));
        let (device, queue) = match res {
            Ok(ids) => ids,
            Err(err) => {
                return Box::pin(ready(Err(err.into())));
            }
        };
        let error_sink = Arc::new(Mutex::new(ErrorSinkRaw::new()));
        let device = CoreDevice {
            context: self.context.clone(),
            wgpu_device: device,
            error_sink: error_sink.clone(),
            features: desc.required_features,
        };
        let queue = CoreQueue {
            context: self.context.clone(),
            wgpu_queue: queue,
            error_sink,
        };
        Box::pin(ready(Ok((device.into(), queue.into()))))
    }

    fn is_surface_supported(&self, surface: &dispatch::DispatchSurface) -> bool {
        let surface = surface.as_core();

        self.wgpu_adapter
            .is_surface_supported(&surface.wgpu_surface)
    }

    fn features(&self) -> crate::Features {
        self.wgpu_adapter.features()
    }

    fn limits(&self) -> crate::Limits {
        self.wgpu_adapter.limits()
    }

    fn downlevel_capabilities(&self) -> crate::DownlevelCapabilities {
        self.wgpu_adapter.downlevel_capabilities()
    }

    fn get_info(&self) -> crate::AdapterInfo {
        self.wgpu_adapter.get_info()
    }

    fn get_texture_format_features(
        &self,
        format: crate::TextureFormat,
    ) -> crate::TextureFormatFeatures {
        self.wgpu_adapter.get_texture_format_features(format)
    }

    fn get_presentation_timestamp(&self) -> crate::PresentationTimestamp {
        self.wgpu_adapter.get_presentation_timestamp()
    }

    fn cooperative_matrix_properties(&self) -> Vec<crate::wgt::CooperativeMatrixProperties> {
        self.wgpu_adapter.cooperative_matrix_properties()
    }
}

impl Drop for CoreAdapter {
    fn drop(&mut self) {}
}

impl dispatch::DeviceInterface for CoreDevice {
    fn features(&self) -> crate::Features {
        *self.wgpu_device.features()
    }

    fn limits(&self) -> crate::Limits {
        self.wgpu_device.limits().clone()
    }

    fn adapter_info(&self) -> crate::AdapterInfo {
        self.wgpu_device.adapter_info()
    }

    // If we have no way to create a shader module, we can't return one, and so most of the function is unreachable.
    #[cfg_attr(
        not(any(
            feature = "spirv",
            feature = "glsl",
            feature = "wgsl",
            feature = "naga-ir"
        )),
        expect(unused)
    )]
    fn create_shader_module(
        &self,
        desc: crate::ShaderModuleDescriptor<'_>,
        shader_bound_checks: wgt::ShaderRuntimeChecks,
    ) -> dispatch::DispatchShaderModule {
        let descriptor = wgc::pipeline::ShaderModuleDescriptor {
            label: desc.label.map(Borrowed),
            runtime_checks: shader_bound_checks,
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
                let options = naga::front::glsl::Options {
                    stage,
                    defines: defines
                        .iter()
                        .map(|&(key, value)| (String::from(key), String::from(value)))
                        .collect(),
                };
                wgc::pipeline::ShaderModuleSource::Glsl(Borrowed(shader), options)
            }
            #[cfg(feature = "wgsl")]
            ShaderSource::Wgsl(ref code) => wgc::pipeline::ShaderModuleSource::Wgsl(Borrowed(code)),
            #[cfg(feature = "naga-ir")]
            ShaderSource::Naga(module) => wgc::pipeline::ShaderModuleSource::Naga(module),
            ShaderSource::Dummy(_) => panic!("found `ShaderSource::Dummy`"),
        };
        let (wgpu_shader_module, error) =
            self.wgpu_device.create_shader_module(&descriptor, source);
        let compilation_info = match error {
            Some(cause) => {
                self.context.handle_error(
                    &self.error_sink,
                    cause.clone(),
                    desc.label,
                    "Device::create_shader_module",
                );
                CompilationInfo::from(cause)
            }
            None => CompilationInfo { messages: vec![] },
        };

        CoreShaderModule {
            wgpu_shader_module,
            compilation_info,
        }
        .into()
    }

    unsafe fn create_shader_module_passthrough(
        &self,
        desc: &crate::ShaderModuleDescriptorPassthrough<'_>,
    ) -> dispatch::DispatchShaderModule {
        let desc = desc.map_label(|l| l.map(Cow::from));
        let (wgpu_shader_module, error) =
            unsafe { self.wgpu_device.create_shader_module_passthrough(&desc) };

        let compilation_info = match error {
            Some(cause) => {
                self.context.handle_error(
                    &self.error_sink,
                    cause.clone(),
                    desc.label.as_deref(),
                    "Device::create_shader_module_passthrough",
                );
                CompilationInfo::from(cause)
            }
            None => CompilationInfo { messages: vec![] },
        };

        CoreShaderModule {
            wgpu_shader_module,
            compilation_info,
        }
        .into()
    }

    fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor<'_>,
    ) -> dispatch::DispatchBindGroupLayout {
        let descriptor = wgc::binding_model::BindGroupLayoutDescriptor {
            label: desc.label.map(Borrowed),
            entries: Borrowed(desc.entries),
        };
        let (wgpu_bind_group_layout, error) =
            self.wgpu_device.create_bind_group_layout(&descriptor);
        if let Some(cause) = error {
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "Device::create_bind_group_layout",
            );
        }
        CoreBindGroupLayout {
            wgpu_bind_group_layout,
        }
        .into()
    }

    fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<'_>,
    ) -> dispatch::DispatchBindGroup {
        use wgc::binding_model as bm;

        let mut arrayed_texture_views = Vec::new();
        let mut arrayed_samplers = Vec::new();
        if self.features.contains(Features::TEXTURE_BINDING_ARRAY) {
            // gather all the array view first
            for entry in desc.entries.iter() {
                if let BindingResource::TextureViewArray(array) = entry.resource {
                    arrayed_texture_views.extend(
                        array
                            .iter()
                            .map(|view| view.inner.as_core().wgpu_texture_view.clone()),
                    );
                }
                if let BindingResource::SamplerArray(array) = entry.resource {
                    arrayed_samplers.extend(
                        array
                            .iter()
                            .map(|sampler| sampler.inner.as_core().wgpu_sampler.clone()),
                    );
                }
            }
        }
        let mut remaining_arrayed_texture_views = &arrayed_texture_views[..];
        let mut remaining_arrayed_samplers = &arrayed_samplers[..];

        let mut arrayed_buffer_bindings = Vec::new();
        if self.features.contains(Features::BUFFER_BINDING_ARRAY) {
            // gather all the buffers first
            for entry in desc.entries.iter() {
                if let BindingResource::BufferArray(array) = entry.resource {
                    arrayed_buffer_bindings.extend(array.iter().map(|binding| bm::BufferBinding {
                        buffer: binding.buffer.inner.as_core().wgpu_buffer.clone(),
                        offset: binding.offset,
                        size: binding.size.map(wgt::BufferSize::get),
                    }));
                }
            }
        }
        let mut remaining_arrayed_buffer_bindings = &arrayed_buffer_bindings[..];

        let mut arrayed_acceleration_structures = Vec::new();
        if self
            .features
            .contains(Features::ACCELERATION_STRUCTURE_BINDING_ARRAY)
        {
            // Gather all the TLAS IDs used by TLAS arrays first (same pattern as other arrayed resources).
            for entry in desc.entries.iter() {
                if let BindingResource::AccelerationStructureArray(array) = entry.resource {
                    arrayed_acceleration_structures.extend(
                        array
                            .iter()
                            .map(|tlas| tlas.inner.as_core().wgpu_tlas.clone()),
                    );
                }
            }
        }
        let mut remaining_arrayed_acceleration_structures = &arrayed_acceleration_structures[..];

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
                        buffer: buffer.inner.as_core().wgpu_buffer.clone(),
                        offset,
                        size: size.map(wgt::BufferSize::get),
                    }),
                    BindingResource::BufferArray(array) => {
                        let slice = &remaining_arrayed_buffer_bindings[..array.len()];
                        remaining_arrayed_buffer_bindings =
                            &remaining_arrayed_buffer_bindings[array.len()..];
                        bm::BindingResource::BufferArray(Borrowed(slice))
                    }
                    BindingResource::Sampler(sampler) => {
                        bm::BindingResource::Sampler(sampler.inner.as_core().wgpu_sampler.clone())
                    }
                    BindingResource::SamplerArray(array) => {
                        let slice = &remaining_arrayed_samplers[..array.len()];
                        remaining_arrayed_samplers = &remaining_arrayed_samplers[array.len()..];
                        bm::BindingResource::SamplerArray(Borrowed(slice))
                    }
                    BindingResource::TextureView(texture_view) => bm::BindingResource::TextureView(
                        texture_view.inner.as_core().wgpu_texture_view.clone(),
                    ),
                    BindingResource::TextureViewArray(array) => {
                        let slice = &remaining_arrayed_texture_views[..array.len()];
                        remaining_arrayed_texture_views =
                            &remaining_arrayed_texture_views[array.len()..];
                        bm::BindingResource::TextureViewArray(Borrowed(slice))
                    }
                    BindingResource::AccelerationStructure(acceleration_structure) => {
                        bm::BindingResource::AccelerationStructure(
                            acceleration_structure.inner.as_core().wgpu_tlas.clone(),
                        )
                    }
                    BindingResource::AccelerationStructureArray(array) => {
                        let slice = &remaining_arrayed_acceleration_structures[..array.len()];
                        remaining_arrayed_acceleration_structures =
                            &remaining_arrayed_acceleration_structures[array.len()..];
                        bm::BindingResource::AccelerationStructureArray(Borrowed(slice))
                    }
                    BindingResource::ExternalTexture(external_texture) => {
                        bm::BindingResource::ExternalTexture(
                            external_texture
                                .inner
                                .as_core()
                                .wgpu_external_texture
                                .clone(),
                        )
                    }
                },
            })
            .collect::<Vec<_>>();
        let descriptor = bm::BindGroupDescriptor {
            label: desc.label.as_ref().map(|label| Borrowed(&label[..])),
            layout: desc.layout.inner.as_core().wgpu_bind_group_layout.clone(),
            entries: Borrowed(&entries),
        };

        let (wgpu_bind_group, error) = self.wgpu_device.create_bind_group(&descriptor);
        if let Some(cause) = error {
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "Device::create_bind_group",
            );
        }
        CoreBindGroup { wgpu_bind_group }.into()
    }

    fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<'_>,
    ) -> dispatch::DispatchPipelineLayout {
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
            .map(|bgl| bgl.map(|bgl| bgl.inner.as_core().wgpu_bind_group_layout.clone()))
            .collect::<ArrayVec<_, { wgc::MAX_BIND_GROUPS }>>();
        let descriptor = wgc::binding_model::PipelineLayoutDescriptor {
            label: desc.label.map(Borrowed),
            bind_group_layouts: Borrowed(&temp_layouts),
            immediate_size: desc.immediate_size,
        };

        let (wgpu_pipeline_layout, error) = self.wgpu_device.create_pipeline_layout(&descriptor);
        if let Some(cause) = error {
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "Device::create_pipeline_layout",
            );
        }
        CorePipelineLayout {
            wgpu_pipeline_layout,
        }
        .into()
    }

    fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<'_>,
    ) -> dispatch::DispatchRenderPipeline {
        use wgc::pipeline as pipe;

        let vertex_buffers: ArrayVec<_, { wgc::MAX_VERTEX_BUFFERS }> = desc
            .vertex
            .buffers
            .iter()
            .map(|vbuf| {
                vbuf.as_ref().map(|vbuf| pipe::VertexBufferLayout {
                    array_stride: vbuf.array_stride,
                    step_mode: vbuf.step_mode,
                    attributes: Borrowed(vbuf.attributes),
                })
            })
            .collect();

        let vert_constants = desc
            .vertex
            .compilation_options
            .constants
            .iter()
            .map(|&(key, value)| (String::from(key), value))
            .collect();

        let descriptor = pipe::ResolvedGeneralRenderPipelineDescriptor {
            label: desc.label.map(Borrowed),
            layout: desc
                .layout
                .map(|layout| layout.inner.as_core().wgpu_pipeline_layout.clone()),
            vertex: wgc::pipeline::RenderPipelineVertexProcessor::Vertex(pipe::VertexState {
                stage: pipe::ProgrammableStageDescriptor {
                    module: desc
                        .vertex
                        .module
                        .inner
                        .as_core()
                        .wgpu_shader_module
                        .clone(),
                    entry_point: desc.vertex.entry_point.map(Borrowed),
                    constants: vert_constants,
                    zero_initialize_workgroup_memory: desc
                        .vertex
                        .compilation_options
                        .zero_initialize_workgroup_memory,
                },
                buffers: Borrowed(&vertex_buffers),
            }),
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment: desc.fragment.as_ref().map(|frag| {
                let frag_constants = frag
                    .compilation_options
                    .constants
                    .iter()
                    .map(|&(key, value)| (String::from(key), value))
                    .collect();
                pipe::FragmentState {
                    stage: pipe::ProgrammableStageDescriptor {
                        module: frag.module.inner.as_core().wgpu_shader_module.clone(),
                        entry_point: frag.entry_point.map(Borrowed),
                        constants: frag_constants,
                        zero_initialize_workgroup_memory: frag
                            .compilation_options
                            .zero_initialize_workgroup_memory,
                    },
                    targets: Borrowed(frag.targets),
                }
            }),
            multiview_mask: desc.multiview_mask,
            cache: desc
                .cache
                .map(|cache| cache.inner.as_core().wgpu_pipeline_cache.clone()),
        };

        let (wgpu_render_pipeline, error) = self.wgpu_device.create_render_pipeline(descriptor);
        if let Some(cause) = error {
            if let wgc::pipeline::CreateRenderPipelineError::Internal { stage, ref error } = cause {
                log::error!("Shader translation error for stage {stage:?}: {error}");
                log::error!("Please report it to https://github.com/gfx-rs/wgpu");
            }
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "Device::create_render_pipeline",
            );
        }
        CoreRenderPipeline {
            context: self.context.clone(),
            wgpu_render_pipeline,
            error_sink: Arc::clone(&self.error_sink),
        }
        .into()
    }

    fn create_mesh_pipeline(
        &self,
        desc: &crate::MeshPipelineDescriptor<'_>,
    ) -> dispatch::DispatchRenderPipeline {
        use wgc::pipeline as pipe;

        let mesh_constants = desc
            .mesh
            .compilation_options
            .constants
            .iter()
            .map(|&(key, value)| (String::from(key), value))
            .collect();
        let descriptor = pipe::MeshPipelineDescriptor {
            label: desc.label.map(Borrowed),
            task: desc.task.as_ref().map(|task| {
                let task_constants = task
                    .compilation_options
                    .constants
                    .iter()
                    .map(|&(key, value)| (String::from(key), value))
                    .collect();
                pipe::TaskState {
                    stage: pipe::ProgrammableStageDescriptor {
                        module: task.module.inner.as_core().wgpu_shader_module.clone(),
                        entry_point: task.entry_point.map(Borrowed),
                        constants: task_constants,
                        zero_initialize_workgroup_memory: desc
                            .mesh
                            .compilation_options
                            .zero_initialize_workgroup_memory,
                    },
                }
            }),
            mesh: pipe::MeshState {
                stage: pipe::ProgrammableStageDescriptor {
                    module: desc.mesh.module.inner.as_core().wgpu_shader_module.clone(),
                    entry_point: desc.mesh.entry_point.map(Borrowed),
                    constants: mesh_constants,
                    zero_initialize_workgroup_memory: desc
                        .mesh
                        .compilation_options
                        .zero_initialize_workgroup_memory,
                },
            },
            layout: desc
                .layout
                .map(|layout| layout.inner.as_core().wgpu_pipeline_layout.clone()),
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment: desc.fragment.as_ref().map(|frag| {
                let frag_constants = frag
                    .compilation_options
                    .constants
                    .iter()
                    .map(|&(key, value)| (String::from(key), value))
                    .collect();
                pipe::FragmentState {
                    stage: pipe::ProgrammableStageDescriptor {
                        module: frag.module.inner.as_core().wgpu_shader_module.clone(),
                        entry_point: frag.entry_point.map(Borrowed),
                        constants: frag_constants,
                        zero_initialize_workgroup_memory: frag
                            .compilation_options
                            .zero_initialize_workgroup_memory,
                    },
                    targets: Borrowed(frag.targets),
                }
            }),
            multiview: desc.multiview,
            cache: desc
                .cache
                .map(|cache| cache.inner.as_core().wgpu_pipeline_cache.clone()),
        };

        let (wgpu_render_pipeline, error) =
            self.wgpu_device.create_render_pipeline(descriptor.into());
        if let Some(cause) = error {
            if let wgc::pipeline::CreateRenderPipelineError::Internal { stage, ref error } = cause {
                log::error!("Shader translation error for stage {stage:?}: {error}");
                log::error!("Please report it to https://github.com/gfx-rs/wgpu");
            }
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "Device::create_render_pipeline",
            );
        }
        CoreRenderPipeline {
            context: self.context.clone(),
            wgpu_render_pipeline,
            error_sink: Arc::clone(&self.error_sink),
        }
        .into()
    }

    fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<'_>,
    ) -> dispatch::DispatchComputePipeline {
        use wgc::pipeline as pipe;

        let constants = desc
            .compilation_options
            .constants
            .iter()
            .map(|&(key, value)| (String::from(key), value))
            .collect();

        let descriptor = pipe::ComputePipelineDescriptor {
            label: desc.label.map(Borrowed),
            layout: desc
                .layout
                .map(|pll| pll.inner.as_core().wgpu_pipeline_layout.clone()),
            stage: pipe::ProgrammableStageDescriptor {
                module: desc.module.inner.as_core().wgpu_shader_module.clone(),
                entry_point: desc.entry_point.map(Borrowed),
                constants,
                zero_initialize_workgroup_memory: desc
                    .compilation_options
                    .zero_initialize_workgroup_memory,
            },
            cache: desc
                .cache
                .map(|cache| cache.inner.as_core().wgpu_pipeline_cache.clone()),
        };

        let (wgpu_compute_pipeline, error) = self.wgpu_device.create_compute_pipeline(descriptor);
        if let Some(cause) = error {
            if let wgc::pipeline::CreateComputePipelineError::Internal(ref error) = cause {
                log::error!(
                    "Shader translation error for stage {:?}: {}",
                    wgt::ShaderStages::COMPUTE,
                    error
                );
                log::error!("Please report it to https://github.com/gfx-rs/wgpu");
            }
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "Device::create_compute_pipeline",
            );
        }
        CoreComputePipeline {
            context: self.context.clone(),
            wgpu_compute_pipeline,
            error_sink: Arc::clone(&self.error_sink),
        }
        .into()
    }

    unsafe fn create_pipeline_cache(
        &self,
        desc: &crate::PipelineCacheDescriptor<'_>,
    ) -> dispatch::DispatchPipelineCache {
        use wgc::pipeline as pipe;

        let descriptor = pipe::PipelineCacheDescriptor {
            label: desc.label.map(Borrowed),
            data: desc.data.map(Borrowed),
            fallback: desc.fallback,
        };
        let (wgpu_pipeline_cache, error) =
            unsafe { self.wgpu_device.create_pipeline_cache(&descriptor) };
        if let Some(cause) = error {
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "Device::device_create_pipeline_cache_init",
            );
        }
        CorePipelineCache {
            wgpu_pipeline_cache,
        }
        .into()
    }

    fn create_buffer(&self, desc: &crate::BufferDescriptor<'_>) -> dispatch::DispatchBuffer {
        let (wgpu_buffer, error) = self
            .wgpu_device
            .create_buffer(&desc.map_label(|l| l.map(Borrowed)));
        if let Some(cause) = error {
            self.context
                .handle_error(&self.error_sink, cause, desc.label, "Device::create_buffer");
        }

        CoreBuffer {
            context: self.context.clone(),
            wgpu_buffer,
            error_sink: Arc::clone(&self.error_sink),
        }
        .into()
    }

    fn create_texture(&self, desc: &crate::TextureDescriptor<'_>) -> dispatch::DispatchTexture {
        let wgt_desc = desc.map_label_and_view_formats(|l| l.map(Borrowed), |v| v.to_vec());
        let (wgpu_texture, error) = self.wgpu_device.create_texture(&wgt_desc);
        if let Some(cause) = error {
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "Device::create_texture",
            );
        }

        CoreTexture {
            context: self.context.clone(),
            wgpu_texture,
            error_sink: Arc::clone(&self.error_sink),
        }
        .into()
    }

    fn create_external_texture(
        &self,
        desc: &crate::ExternalTextureDescriptor<'_>,
        planes: &[&crate::TextureView],
    ) -> dispatch::DispatchExternalTexture {
        let wgt_desc = desc.map_label(|l| l.map(Borrowed));
        let planes = planes
            .iter()
            .map(|plane| plane.inner.as_core().wgpu_texture_view.clone())
            .collect::<Vec<_>>();
        let (wgpu_external_texture, error) =
            self.wgpu_device.create_external_texture(&wgt_desc, &planes);
        if let Some(cause) = error {
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "Device::create_external_texture",
            );
        }

        CoreExternalTexture {
            wgpu_external_texture,
        }
        .into()
    }

    fn create_blas(
        &self,
        desc: &crate::CreateBlasDescriptor<'_>,
        sizes: crate::BlasGeometrySizeDescriptors,
    ) -> (Option<u64>, dispatch::DispatchBlas) {
        let (wgpu_blas, error) = self
            .wgpu_device
            .create_blas(&desc.map_label(|l| l.map(Borrowed)), sizes);
        if let Some(cause) = error {
            self.context
                .handle_error(&self.error_sink, cause, desc.label, "Device::create_blas");
        }
        (
            wgpu_blas.handle(),
            CoreBlas {
                context: self.context.clone(),
                wgpu_blas,
                error_sink: Arc::clone(&self.error_sink),
            }
            .into(),
        )
    }

    fn create_tlas(&self, desc: &crate::CreateTlasDescriptor<'_>) -> dispatch::DispatchTlas {
        let (wgpu_tlas, error) = self
            .wgpu_device
            .create_tlas(&desc.map_label(|l| l.map(Borrowed)));
        if let Some(cause) = error {
            self.context
                .handle_error(&self.error_sink, cause, desc.label, "Device::create_tlas");
        }
        CoreTlas {
            context: self.context.clone(),
            wgpu_tlas,
            // error_sink: Arc::clone(&self.error_sink),
        }
        .into()
    }

    fn create_sampler(&self, desc: &crate::SamplerDescriptor<'_>) -> dispatch::DispatchSampler {
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

        let (wgpu_sampler, error) = self.wgpu_device.create_sampler(&descriptor);
        if let Some(cause) = error {
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "Device::create_sampler",
            );
        }
        CoreSampler { wgpu_sampler }.into()
    }

    fn create_query_set(&self, desc: &crate::QuerySetDescriptor<'_>) -> dispatch::DispatchQuerySet {
        let (wgpu_query_set, error) = self
            .wgpu_device
            .create_query_set(&desc.map_label(|l| l.map(Borrowed)));
        if let Some(cause) = error {
            self.context
                .handle_error_nolabel(&self.error_sink, cause, "Device::create_query_set");
        }
        CoreQuerySet { wgpu_query_set }.into()
    }

    fn create_command_encoder(
        &self,
        desc: &crate::CommandEncoderDescriptor<'_>,
    ) -> dispatch::DispatchCommandEncoder {
        let (wgpu_command_encoder, error) = self
            .wgpu_device
            .create_command_encoder(&desc.map_label(|l| l.map(Borrowed)));
        if let Some(cause) = error {
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "Device::create_command_encoder",
            );
        }

        CoreCommandEncoder {
            context: self.context.clone(),
            wgpu_command_encoder,
            error_sink: Arc::clone(&self.error_sink),
        }
        .into()
    }

    fn create_render_bundle_encoder(
        &self,
        desc: &crate::RenderBundleEncoderDescriptor<'_>,
    ) -> dispatch::DispatchRenderBundleEncoder {
        let descriptor = wgc::command::RenderBundleEncoderDescriptor {
            label: desc.label.map(Borrowed),
            color_formats: Borrowed(desc.color_formats),
            depth_stencil: desc.depth_stencil,
            sample_count: desc.sample_count,
            multiview: desc.multiview,
        };
        let (encoder, error) = self.wgpu_device.create_render_bundle_encoder(&descriptor);
        if let Some(cause) = error {
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "Device::create_render_bundle_encoder",
            );
        }

        CoreRenderBundleEncoder {
            context: self.context.clone(),
            error_sink: Arc::clone(&self.error_sink),
            encoder,
        }
        .into()
    }

    fn set_device_lost_callback(&self, device_lost_callback: dispatch::BoxDeviceLostCallback) {
        self.wgpu_device
            .set_device_lost_closure(device_lost_callback);
    }

    fn on_uncaptured_error(&self, handler: Arc<dyn crate::UncapturedErrorHandler>) {
        let mut error_sink = self.error_sink.lock();
        error_sink.uncaptured_handler = Some(handler);
    }

    fn push_error_scope(&self, filter: crate::ErrorFilter) -> u32 {
        let mut error_sink = self.error_sink.lock();
        let thread_id = thread_id::ThreadId::current();
        let scopes = error_sink.scopes.entry(thread_id).or_default();
        let index = scopes
            .len()
            .try_into()
            .expect("Greater than 2^32 nested error scopes");
        scopes.push(ErrorScope {
            error: None,
            filter,
        });
        index
    }

    fn pop_error_scope(&self, index: u32) -> Pin<Box<dyn dispatch::PopErrorScopeFuture>> {
        let mut error_sink = self.error_sink.lock();

        // We go out of our way to avoid panicking while unwinding, because that would abort the process,
        // and we are supposed to just drop the error scope on the floor.
        let is_panicking = crate::util::is_panicking();
        let thread_id = thread_id::ThreadId::current();
        let err = "Mismatched pop_error_scope call: no error scope for this thread. Error scopes are thread-local.";
        let scopes = match error_sink.scopes.get_mut(&thread_id) {
            Some(s) => s,
            None => {
                if !is_panicking {
                    panic!("{err}");
                } else {
                    return Box::pin(ready(None));
                }
            }
        };
        if scopes.is_empty() && !is_panicking {
            panic!("{err}");
        }
        if index as usize != scopes.len() - 1 && !is_panicking {
            panic!(
                "Mismatched pop_error_scope call: error scopes must be popped in reverse order."
            );
        }

        // It would be more correct in this case to use `remove` here so that when unwinding is occurring
        // we would remove the correct error scope, but we don't have such a primitive on the web
        // and having consistent behavior here is more important. If you are unwinding and it unwinds
        // the guards in the wrong order, it's totally reasonable to have incorrect behavior.
        let scope = match scopes.pop() {
            Some(s) => s,
            None if !is_panicking => unreachable!(),
            None => return Box::pin(ready(None)),
        };

        Box::pin(ready(scope.error))
    }

    unsafe fn start_graphics_debugger_capture(&self) {
        unsafe { self.wgpu_device.start_graphics_debugger_capture() };
    }

    unsafe fn stop_graphics_debugger_capture(&self) {
        unsafe { self.wgpu_device.stop_graphics_debugger_capture() };
    }

    fn poll(&self, poll_type: wgt::PollType<u64>) -> Result<crate::PollStatus, crate::PollError> {
        match self.wgpu_device.poll(poll_type) {
            Ok(status) => Ok(status),
            Err(err) => {
                if let Some(poll_error) = err.to_poll_error() {
                    return Err(poll_error);
                }

                self.context.handle_error_fatal(err, "Device::poll")
            }
        }
    }

    fn get_internal_counters(&self) -> crate::InternalCounters {
        self.wgpu_device.get_internal_counters()
    }

    fn generate_allocator_report(&self) -> Option<wgt::AllocatorReport> {
        self.wgpu_device.generate_allocator_report()
    }

    fn destroy(&self) {
        self.wgpu_device.destroy();
    }
}

impl Drop for CoreDevice {
    fn drop(&mut self) {}
}

impl dispatch::QueueInterface for CoreQueue {
    fn write_buffer(
        &self,
        buffer: &dispatch::DispatchBuffer,
        offset: crate::BufferAddress,
        data: &[u8],
    ) {
        let buffer = buffer.as_core();

        match self
            .wgpu_queue
            .write_buffer(buffer.wgpu_buffer.clone(), offset, data)
        {
            Ok(()) => (),
            Err(err) => {
                self.context
                    .handle_error_nolabel(&self.error_sink, err, "Queue::write_buffer")
            }
        }
    }

    fn create_staging_buffer(
        &self,
        size: crate::BufferSize,
    ) -> Option<dispatch::DispatchQueueWriteBuffer> {
        match self.wgpu_queue.create_staging_buffer(size) {
            Ok((wgpu_staging_buffer, ptr)) => Some(
                CoreQueueWriteBuffer {
                    wgpu_staging_buffer,
                    mapping: CoreBufferMappedRange {
                        ptr,
                        size: size.get() as usize,
                    },
                }
                .into(),
            ),
            Err(err) => {
                self.context.handle_error_nolabel(
                    &self.error_sink,
                    err,
                    "Queue::write_buffer_with",
                );
                None
            }
        }
    }

    fn validate_write_buffer(
        &self,
        buffer: &dispatch::DispatchBuffer,
        offset: wgt::BufferAddress,
        size: wgt::BufferSize,
    ) -> Option<()> {
        let buffer = buffer.as_core();

        match self
            .wgpu_queue
            .validate_write_buffer(buffer.wgpu_buffer.clone(), offset, size)
        {
            Ok(()) => Some(()),
            Err(err) => {
                self.context.handle_error_nolabel(
                    &self.error_sink,
                    err,
                    "Queue::write_buffer_with",
                );
                None
            }
        }
    }

    fn write_staging_buffer(
        &self,
        buffer: &dispatch::DispatchBuffer,
        offset: crate::BufferAddress,
        staging_buffer: dispatch::DispatchQueueWriteBuffer,
    ) {
        let buffer = buffer.as_core();
        let staging_buffer = {
            #[allow(
                clippy::allow_attributes,
                unreachable_patterns,
                reason = "features may be disabled"
            )]
            match staging_buffer {
                dispatch::DispatchQueueWriteBuffer::Core(value) => value,
                _ => panic!(concat!(
                    stringify!(DispatchQueueWriteBuffer),
                    " is not core"
                )),
            }
        };

        match self.wgpu_queue.write_staging_buffer(
            buffer.wgpu_buffer.clone(),
            offset,
            staging_buffer.wgpu_staging_buffer,
        ) {
            Ok(()) => (),
            Err(err) => {
                self.context.handle_error_nolabel(
                    &self.error_sink,
                    err,
                    "Queue::write_buffer_with",
                );
            }
        }
    }

    fn write_texture(
        &self,
        texture: crate::TexelCopyTextureInfo<'_>,
        data: &[u8],
        data_layout: crate::TexelCopyBufferLayout,
        size: crate::Extent3d,
    ) {
        match self.wgpu_queue.write_texture(
            map_texture_copy_view(texture),
            data,
            &data_layout,
            &size,
        ) {
            Ok(()) => (),
            Err(err) => {
                self.context
                    .handle_error_nolabel(&self.error_sink, err, "Queue::write_texture")
            }
        }
    }

    // This method needs to exist if either webgpu or webgl is enabled,
    // but we only actually have an implementation if webgl is enabled.
    #[cfg(web)]
    #[cfg_attr(not(webgl), expect(unused_variables))]
    fn copy_external_image_to_texture(
        &self,
        source: &crate::CopyExternalImageSourceInfo,
        dest: crate::CopyExternalImageDestInfo<&crate::api::Texture>,
        size: crate::Extent3d,
    ) {
        #[cfg(webgl)]
        match self.wgpu_queue.copy_external_image_to_texture(
            source,
            map_texture_tagged_copy_view(dest),
            size,
        ) {
            Ok(()) => (),
            Err(err) => self.context.handle_error_nolabel(
                &self.error_sink,
                err,
                "Queue::copy_external_image_to_texture",
            ),
        }
    }

    fn submit(
        &self,
        command_buffers: &mut dyn Iterator<Item = dispatch::DispatchCommandBuffer>,
    ) -> u64 {
        let temp_command_buffers = command_buffers.collect::<SmallVec<[_; 4]>>();
        let command_buffers = temp_command_buffers
            .iter()
            .map(|cmdbuf| cmdbuf.as_core().wgpu_command_buffer.clone())
            .collect::<SmallVec<[_; 4]>>();

        let index = match self.wgpu_queue.submit(&command_buffers) {
            Ok(index) => index,
            Err((index, err)) => {
                self.context
                    .handle_error_nolabel(&self.error_sink, err, "Queue::submit");
                index
            }
        };

        drop(temp_command_buffers);

        index
    }

    fn get_timestamp_period(&self) -> f32 {
        self.wgpu_queue.get_timestamp_period()
    }

    fn on_submitted_work_done(&self, callback: dispatch::BoxSubmittedWorkDoneCallback) {
        self.wgpu_queue.on_submitted_work_done(callback);
    }

    fn compact_blas(&self, blas: &dispatch::DispatchBlas) -> (Option<u64>, dispatch::DispatchBlas) {
        let (wgpu_blas, error) = self.wgpu_queue.compact_blas(&blas.as_core().wgpu_blas);

        if let Some(cause) = error {
            self.context
                .handle_error_nolabel(&self.error_sink, cause, "Queue::compact_blas");
        }
        (
            wgpu_blas.handle(),
            CoreBlas {
                context: self.context.clone(),
                wgpu_blas,
                error_sink: Arc::clone(&self.error_sink),
            }
            .into(),
        )
    }

    fn present(&self, detail: &dispatch::DispatchSurfaceOutputDetail) {
        let detail = detail.as_core();
        match detail.wgpu_surface.present() {
            Ok(_status) => (),
            Err(err) => {
                self.context
                    .handle_error_nolabel(&self.error_sink, err, "Queue::present");
            }
        }
    }
}

impl dispatch::ShaderModuleInterface for CoreShaderModule {
    fn get_compilation_info(&self) -> Pin<Box<dyn dispatch::ShaderCompilationInfoFuture>> {
        Box::pin(ready(self.compilation_info.clone()))
    }
}

impl dispatch::BindGroupLayoutInterface for CoreBindGroupLayout {}

impl dispatch::BindGroupInterface for CoreBindGroup {}

impl dispatch::TextureViewInterface for CoreTextureView {}

impl dispatch::ExternalTextureInterface for CoreExternalTexture {
    fn destroy(&self) {
        self.wgpu_external_texture.destroy();
    }
}

impl dispatch::SamplerInterface for CoreSampler {}

impl dispatch::BufferInterface for CoreBuffer {
    fn map_async(
        &self,
        mode: crate::MapMode,
        range: Range<crate::BufferAddress>,
        callback: dispatch::BufferMapCallback,
    ) {
        let operation = wgc::resource::BufferMapOperation {
            host: match mode {
                MapMode::Read => wgc::device::HostMap::Read,
                MapMode::Write => wgc::device::HostMap::Write,
            },
            callback: Some(Box::new(|status| {
                let res = status.map_err(|_| crate::BufferAsyncError);
                callback(res);
            })),
        };

        match self
            .wgpu_buffer
            .map_async(range.start, Some(range.end - range.start), operation)
        {
            Ok(_) => (),
            Err(cause) => {
                self.context
                    .handle_error_nolabel(&self.error_sink, cause, "Buffer::map_async")
            }
        }
    }

    fn get_mapped_range(
        &self,
        sub_range: Range<crate::BufferAddress>,
    ) -> Result<dispatch::DispatchBufferMappedRange, crate::MapRangeError> {
        let size = sub_range.end - sub_range.start;
        self.wgpu_buffer
            .get_mapped_range(sub_range.start, Some(size))
            .map(|(ptr, size)| {
                CoreBufferMappedRange {
                    ptr,
                    size: size as usize,
                }
                .into()
            })
            .map_err(|err| crate::MapRangeError(self.context.format_error(&err)))
    }

    fn unmap(&self) {
        match self.wgpu_buffer.unmap() {
            Ok(()) => (),
            Err(cause) => {
                self.context
                    .handle_error_nolabel(&self.error_sink, cause, "Buffer::buffer_unmap")
            }
        }
    }

    fn destroy(&self) {
        self.wgpu_buffer.destroy();
    }
}

impl dispatch::TextureInterface for CoreTexture {
    fn create_view(
        &self,
        desc: &crate::TextureViewDescriptor<'_>,
    ) -> dispatch::DispatchTextureView {
        let descriptor = wgc::resource::TextureViewDescriptor {
            label: desc.label.map(Borrowed),
            format: desc.format,
            dimension: desc.dimension,
            usage: desc.usage,
            range: wgt::ImageSubresourceRange {
                aspect: desc.aspect,
                base_mip_level: desc.base_mip_level,
                mip_level_count: desc.mip_level_count,
                base_array_layer: desc.base_array_layer,
                array_layer_count: desc.array_layer_count,
            },
        };
        let (wgpu_texture_view, error) = self.wgpu_texture.create_view(&descriptor);
        if let Some(cause) = error {
            self.context
                .handle_error(&self.error_sink, cause, desc.label, "Texture::create_view");
        }
        CoreTextureView {
            context: self.context.clone(),
            wgpu_texture_view,
        }
        .into()
    }

    fn destroy(&self) {
        self.wgpu_texture.destroy();
    }
}

impl dispatch::BlasInterface for CoreBlas {
    fn prepare_compact_async(&self, callback: BlasCompactCallback) {
        let callback: Option<wgc::resource::BlasCompactCallback> =
            Some(Box::new(|status: BlasPrepareCompactResult| {
                let res = status.map_err(|_| crate::BlasAsyncError);
                callback(res);
            }));

        match self.wgpu_blas.prepare_compact_async(callback) {
            Ok(_) => (),
            Err(cause) => self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "Blas::prepare_compact_async",
            ),
        }
    }

    fn ready_for_compaction(&self) -> bool {
        match self.wgpu_blas.ready_for_compaction() {
            Ok(ready) => ready,
            Err(cause) => {
                self.context.handle_error_nolabel(
                    &self.error_sink,
                    cause,
                    "Blas::ready_for_compaction",
                );
                // A BLAS is definitely not ready for compaction if it's not valid
                false
            }
        }
    }
}

impl dispatch::TlasInterface for CoreTlas {}

impl dispatch::QuerySetInterface for CoreQuerySet {
    fn destroy(&self) {
        self.wgpu_query_set.destroy();
    }
}

impl dispatch::PipelineLayoutInterface for CorePipelineLayout {}

impl dispatch::RenderPipelineInterface for CoreRenderPipeline {
    fn get_bind_group_layout(&self, index: u32) -> dispatch::DispatchBindGroupLayout {
        let (wgpu_bind_group_layout, error) =
            self.wgpu_render_pipeline.get_bind_group_layout(index);
        if let Some(err) = error {
            self.context.handle_error_nolabel(
                &self.error_sink,
                err,
                "RenderPipeline::get_bind_group_layout",
            )
        }
        CoreBindGroupLayout {
            wgpu_bind_group_layout,
        }
        .into()
    }
}

impl dispatch::ComputePipelineInterface for CoreComputePipeline {
    fn get_bind_group_layout(&self, index: u32) -> dispatch::DispatchBindGroupLayout {
        let (wgpu_bind_group_layout, error) =
            self.wgpu_compute_pipeline.get_bind_group_layout(index);
        if let Some(err) = error {
            self.context.handle_error_nolabel(
                &self.error_sink,
                err,
                "ComputePipeline::get_bind_group_layout",
            )
        }
        CoreBindGroupLayout {
            wgpu_bind_group_layout,
        }
        .into()
    }
}

impl dispatch::PipelineCacheInterface for CorePipelineCache {
    fn get_data(&self) -> Option<Vec<u8>> {
        self.wgpu_pipeline_cache.get_data()
    }
}

impl dispatch::CommandEncoderInterface for CoreCommandEncoder {
    fn copy_buffer_to_buffer(
        &self,
        source: &dispatch::DispatchBuffer,
        source_offset: crate::BufferAddress,
        destination: &dispatch::DispatchBuffer,
        destination_offset: crate::BufferAddress,
        copy_size: Option<crate::BufferAddress>,
    ) {
        let source = source.as_core();
        let destination = destination.as_core();

        if let Err(cause) = self.wgpu_command_encoder.copy_buffer_to_buffer(
            source.wgpu_buffer.clone(),
            source_offset,
            destination.wgpu_buffer.clone(),
            destination_offset,
            copy_size,
        ) {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::copy_buffer_to_buffer",
            );
        }
    }

    fn copy_buffer_to_texture(
        &self,
        source: crate::TexelCopyBufferInfo<'_>,
        destination: crate::TexelCopyTextureInfo<'_>,
        copy_size: crate::Extent3d,
    ) {
        if let Err(cause) = self.wgpu_command_encoder.copy_buffer_to_texture(
            &map_buffer_copy_view(source),
            &map_texture_copy_view(destination),
            &copy_size,
        ) {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::copy_buffer_to_texture",
            );
        }
    }

    fn copy_texture_to_buffer(
        &self,
        source: crate::TexelCopyTextureInfo<'_>,
        destination: crate::TexelCopyBufferInfo<'_>,
        copy_size: crate::Extent3d,
    ) {
        if let Err(cause) = self.wgpu_command_encoder.copy_texture_to_buffer(
            &map_texture_copy_view(source),
            &map_buffer_copy_view(destination),
            &copy_size,
        ) {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::copy_texture_to_buffer",
            );
        }
    }

    fn copy_texture_to_texture(
        &self,
        source: crate::TexelCopyTextureInfo<'_>,
        destination: crate::TexelCopyTextureInfo<'_>,
        copy_size: crate::Extent3d,
    ) {
        if let Err(cause) = self.wgpu_command_encoder.copy_texture_to_texture(
            &map_texture_copy_view(source),
            &map_texture_copy_view(destination),
            &copy_size,
        ) {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::copy_texture_to_texture",
            );
        }
    }

    fn begin_compute_pass(
        &self,
        desc: &crate::ComputePassDescriptor<'_>,
    ) -> dispatch::DispatchComputePass {
        let timestamp_writes =
            desc.timestamp_writes
                .as_ref()
                .map(|tw| wgc::command::PassTimestampWrites {
                    query_set: tw.query_set.inner.as_core().wgpu_query_set.clone(),
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                });

        let (pass, err) =
            self.wgpu_command_encoder
                .begin_compute_pass(&wgc::command::ComputePassDescriptor {
                    label: desc.label.map(Borrowed),
                    timestamp_writes,
                });

        if let Some(cause) = err {
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "CommandEncoder::begin_compute_pass",
            );
        }

        CoreComputePass {
            context: self.context.clone(),
            pass,
            error_sink: self.error_sink.clone(),
            id: crate::cmp::Identifier::create(),
        }
        .into()
    }

    fn begin_render_pass(
        &self,
        desc: &crate::RenderPassDescriptor<'_>,
    ) -> dispatch::DispatchRenderPass {
        let colors = desc
            .color_attachments
            .iter()
            .map(|ca| {
                ca.as_ref()
                    .map(|at| wgc::command::RenderPassColorAttachment {
                        view: at.view.inner.as_core().wgpu_texture_view.clone(),
                        depth_slice: at.depth_slice,
                        resolve_target: at
                            .resolve_target
                            .map(|view| view.inner.as_core().wgpu_texture_view.clone()),
                        load_op: at.ops.load,
                        store_op: at.ops.store,
                    })
            })
            .collect::<Vec<_>>();

        let depth_stencil = desc.depth_stencil_attachment.as_ref().map(|dsa| {
            wgc::command::RenderPassDepthStencilAttachment {
                view: dsa.view.inner.as_core().wgpu_texture_view.clone(),
                depth: map_pass_channel(dsa.depth_ops.as_ref()),
                stencil: map_pass_channel(dsa.stencil_ops.as_ref()),
            }
        });

        let timestamp_writes =
            desc.timestamp_writes
                .as_ref()
                .map(|tw| wgc::command::PassTimestampWrites {
                    query_set: tw.query_set.inner.as_core().wgpu_query_set.clone(),
                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                    end_of_pass_write_index: tw.end_of_pass_write_index,
                });

        let (pass, err) = self.wgpu_command_encoder.begin_render_pass(
            wgc::command::ResolvedRenderPassDescriptor {
                label: desc.label.map(Borrowed),
                timestamp_writes,
                color_attachments: Borrowed(&colors),
                depth_stencil_attachment: depth_stencil,
                occlusion_query_set: desc
                    .occlusion_query_set
                    .map(|qs| qs.inner.as_core().wgpu_query_set.clone()),
                multiview_mask: desc.multiview_mask,
            },
        );

        if let Some(cause) = err {
            self.context.handle_error(
                &self.error_sink,
                cause,
                desc.label,
                "CommandEncoder::begin_render_pass",
            );
        }

        CoreRenderPass {
            context: self.context.clone(),
            pass,
            error_sink: self.error_sink.clone(),
            id: crate::cmp::Identifier::create(),
        }
        .into()
    }

    fn finish(&mut self) -> dispatch::DispatchCommandBuffer {
        let descriptor = wgt::CommandBufferDescriptor::default();
        let (wgpu_command_buffer, opt_label_and_error) =
            self.wgpu_command_encoder.finish(&descriptor);
        if let Some((label, cause)) = opt_label_and_error {
            self.context
                .handle_error(&self.error_sink, cause, Some(&label), "a CommandEncoder");
        }
        CoreCommandBuffer {
            context: self.context.clone(),
            wgpu_command_buffer,
        }
        .into()
    }

    fn clear_texture(
        &self,
        texture: &dispatch::DispatchTexture,
        subresource_range: &crate::ImageSubresourceRange,
    ) {
        let texture = texture.as_core();

        if let Err(cause) = self
            .wgpu_command_encoder
            .clear_texture(texture.wgpu_texture.clone(), subresource_range)
        {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::clear_texture",
            );
        }
    }

    fn clear_buffer(
        &self,
        buffer: &dispatch::DispatchBuffer,
        offset: crate::BufferAddress,
        size: Option<crate::BufferAddress>,
    ) {
        let buffer = buffer.as_core();

        if let Err(cause) =
            self.wgpu_command_encoder
                .clear_buffer(buffer.wgpu_buffer.clone(), offset, size)
        {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::fill_buffer",
            );
        }
    }

    fn insert_debug_marker(&self, label: &str) {
        if let Err(cause) = self.wgpu_command_encoder.insert_debug_marker(label) {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::insert_debug_marker",
            );
        }
    }

    fn push_debug_group(&self, label: &str) {
        if let Err(cause) = self.wgpu_command_encoder.push_debug_group(label) {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::push_debug_group",
            );
        }
    }

    fn pop_debug_group(&self) {
        if let Err(cause) = self.wgpu_command_encoder.pop_debug_group() {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::pop_debug_group",
            );
        }
    }

    fn write_timestamp(&self, query_set: &dispatch::DispatchQuerySet, query_index: u32) {
        let query_set = query_set.as_core();

        if let Err(cause) = self
            .wgpu_command_encoder
            .write_timestamp(query_set.wgpu_query_set.clone(), query_index)
        {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::write_timestamp",
            );
        }
    }

    fn resolve_query_set(
        &self,
        query_set: &dispatch::DispatchQuerySet,
        first_query: u32,
        query_count: u32,
        destination: &dispatch::DispatchBuffer,
        destination_offset: crate::BufferAddress,
    ) {
        let query_set = query_set.as_core();
        let destination = destination.as_core();

        if let Err(cause) = self.wgpu_command_encoder.resolve_query_set(
            query_set.wgpu_query_set.clone(),
            first_query,
            query_count,
            destination.wgpu_buffer.clone(),
            destination_offset,
        ) {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::resolve_query_set",
            );
        }
    }

    fn mark_acceleration_structures_built<'a>(
        &self,
        blas: &mut dyn Iterator<Item = &'a Blas>,
        tlas: &mut dyn Iterator<Item = &'a Tlas>,
    ) {
        let blas = blas
            .map(|b| b.inner.as_core().wgpu_blas.clone())
            .collect::<SmallVec<[_; 4]>>();
        let tlas = tlas
            .map(|t| t.inner.as_core().wgpu_tlas.clone())
            .collect::<SmallVec<[_; 4]>>();
        if let Err(cause) = self
            .wgpu_command_encoder
            .mark_acceleration_structures_built(&blas, &tlas)
        {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::build_acceleration_structures_unsafe_tlas",
            );
        }
    }

    fn build_acceleration_structures<'a>(
        &self,
        blas: &mut dyn Iterator<Item = &'a crate::BlasBuildEntry<'a>>,
        tlas: &mut dyn Iterator<Item = &'a crate::Tlas>,
    ) {
        let blas = blas.map(|e: &crate::BlasBuildEntry<'_>| {
            let geometries = match e.geometry {
                crate::BlasGeometries::TriangleGeometries(ref triangle_geometries) => {
                    let iter = triangle_geometries.iter().map(|tg| {
                        wgc::ray_tracing::BlasTriangleGeometry {
                            vertex_buffer: tg.vertex_buffer.inner.as_core().wgpu_buffer.clone(),
                            index_buffer: tg
                                .index_buffer
                                .map(|buf| buf.inner.as_core().wgpu_buffer.clone()),
                            transform_buffer: tg
                                .transform_buffer
                                .map(|buf| buf.inner.as_core().wgpu_buffer.clone()),
                            size: tg.size,
                            transform_buffer_offset: tg.transform_buffer_offset,
                            first_vertex: tg.first_vertex,
                            vertex_stride: tg.vertex_stride,
                            first_index: tg.first_index,
                        }
                    });
                    wgc::ray_tracing::BlasGeometries::TriangleGeometries(Box::new(iter))
                }
                crate::BlasGeometries::AabbGeometries(ref aabb_geometries) => {
                    let iter =
                        aabb_geometries
                            .iter()
                            .map(|ag| wgc::ray_tracing::BlasAabbGeometry {
                                aabb_buffer: ag.aabb_buffer.inner.as_core().wgpu_buffer.clone(),
                                stride: ag.stride,
                                size: ag.size,
                                primitive_offset: ag.primitive_offset,
                            });
                    wgc::ray_tracing::BlasGeometries::AabbGeometries(Box::new(iter))
                }
            };
            wgc::ray_tracing::BlasBuildEntry {
                blas: e.blas.inner.as_core().wgpu_blas.clone(),
                geometries,
            }
        });

        let tlas = tlas.into_iter().map(|e| {
            let instances = e
                .instances
                .iter()
                .map(|instance: &Option<crate::TlasInstance>| {
                    instance
                        .as_ref()
                        .map(|instance| wgc::ray_tracing::TlasInstance {
                            blas: instance.blas.as_core().wgpu_blas.clone(),
                            transform: &instance.transform,
                            custom_data: instance.custom_data,
                            mask: instance.mask,
                        })
                });
            wgc::ray_tracing::TlasPackage {
                tlas: e.inner.as_core().wgpu_tlas.clone(),
                instances: Box::new(instances),
                lowest_unmodified: e.lowest_unmodified,
            }
        });

        if let Err(cause) = self
            .wgpu_command_encoder
            .build_acceleration_structures(blas, tlas)
        {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::build_acceleration_structures_unsafe_tlas",
            );
        }
    }

    fn transition_resources<'a>(
        &mut self,
        buffer_transitions: &mut dyn Iterator<
            Item = wgt::BufferTransition<&'a dispatch::DispatchBuffer>,
        >,
        texture_transitions: &mut dyn Iterator<
            Item = wgt::TextureTransition<&'a dispatch::DispatchTexture>,
        >,
    ) {
        let result = self.wgpu_command_encoder.transition_resources(
            buffer_transitions.map(|t| wgt::BufferTransition {
                buffer: t.buffer.as_core().wgpu_buffer.clone(),
                state: t.state,
            }),
            texture_transitions.map(|t| wgt::TextureTransition {
                texture: t.texture.as_core().wgpu_texture.clone(),
                selector: t.selector.clone(),
                state: t.state,
            }),
        );

        if let Err(cause) = result {
            self.context.handle_error_nolabel(
                &self.error_sink,
                cause,
                "CommandEncoder::transition_resources",
            );
        }
    }
}

impl dispatch::CommandBufferInterface for CoreCommandBuffer {}

impl dispatch::ComputePassInterface for CoreComputePass {
    fn set_pipeline(&mut self, pipeline: &dispatch::DispatchComputePipeline) {
        let pipeline = pipeline.as_core();

        if let Err(cause) = self
            .pass
            .set_pipeline(pipeline.wgpu_compute_pipeline.clone())
        {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "ComputePass::set_pipeline",
            );
        }
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&dispatch::DispatchBindGroup>,
        offsets: &[crate::DynamicOffset],
    ) {
        let bg = bind_group.map(|bg| bg.as_core().wgpu_bind_group.clone());

        if let Err(cause) = self.pass.set_bind_group(index, bg, offsets) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "ComputePass::set_bind_group",
            );
        }
    }

    fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        if let Err(cause) = self.pass.set_immediates(offset, data) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "ComputePass::set_immediates",
            );
        }
    }

    fn insert_debug_marker(&mut self, label: &str) {
        if let Err(cause) = self.pass.insert_debug_marker(label, 0) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "ComputePass::insert_debug_marker",
            );
        }
    }

    fn push_debug_group(&mut self, group_label: &str) {
        if let Err(cause) = self.pass.push_debug_group(group_label, 0) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "ComputePass::push_debug_group",
            );
        }
    }

    fn pop_debug_group(&mut self) {
        if let Err(cause) = self.pass.pop_debug_group() {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "ComputePass::pop_debug_group",
            );
        }
    }

    fn write_timestamp(&mut self, query_set: &dispatch::DispatchQuerySet, query_index: u32) {
        let query_set = query_set.as_core();

        if let Err(cause) = self
            .pass
            .write_timestamp(query_set.wgpu_query_set.clone(), query_index)
        {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "ComputePass::write_timestamp",
            );
        }
    }

    fn begin_pipeline_statistics_query(
        &mut self,
        query_set: &dispatch::DispatchQuerySet,
        query_index: u32,
    ) {
        let query_set = query_set.as_core();

        if let Err(cause) = self
            .pass
            .begin_pipeline_statistics_query(query_set.wgpu_query_set.clone(), query_index)
        {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "ComputePass::begin_pipeline_statistics_query",
            );
        }
    }

    fn end_pipeline_statistics_query(&mut self) {
        if let Err(cause) = self.pass.end_pipeline_statistics_query() {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "ComputePass::end_pipeline_statistics_query",
            );
        }
    }

    fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
        if let Err(cause) = self.pass.dispatch_workgroups(x, y, z) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "ComputePass::dispatch_workgroups",
            );
        }
    }

    fn dispatch_workgroups_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) {
        let indirect_buffer = indirect_buffer.as_core();

        if let Err(cause) = self
            .pass
            .dispatch_workgroups_indirect(indirect_buffer.wgpu_buffer.clone(), indirect_offset)
        {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "ComputePass::dispatch_workgroups_indirect",
            );
        }
    }

    fn transition_resources<'a>(
        &mut self,
        buffer_transitions: &mut dyn Iterator<
            Item = wgt::BufferTransition<&'a dispatch::DispatchBuffer>,
        >,
        texture_transitions: &mut dyn Iterator<
            Item = wgt::TextureTransition<&'a dispatch::DispatchTextureView>,
        >,
    ) {
        let result = self.pass.transition_resources(
            buffer_transitions.map(|t| wgt::BufferTransition {
                buffer: t.buffer.as_core().wgpu_buffer.clone(),
                state: t.state,
            }),
            texture_transitions.map(|t| wgt::TextureTransition {
                texture: t.texture.as_core().wgpu_texture_view.clone(),
                selector: t.selector.clone(),
                state: t.state,
            }),
        );

        if let Err(cause) = result {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "ComputePass::transition_resources",
            );
        }
    }
}

impl Drop for CoreComputePass {
    fn drop(&mut self) {
        if let Err(cause) = self.pass.end() {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "ComputePass::end",
            );
        }
    }
}

impl dispatch::RenderPassInterface for CoreRenderPass {
    fn set_pipeline(&mut self, pipeline: &dispatch::DispatchRenderPipeline) {
        let pipeline = pipeline.as_core();

        if let Err(cause) = self
            .pass
            .set_pipeline(pipeline.wgpu_render_pipeline.clone())
        {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::set_pipeline",
            );
        }
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&dispatch::DispatchBindGroup>,
        offsets: &[crate::DynamicOffset],
    ) {
        let bg = bind_group.map(|bg| bg.as_core().wgpu_bind_group.clone());

        if let Err(cause) = self.pass.set_bind_group(index, bg, offsets) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::set_bind_group",
            );
        }
    }

    fn set_index_buffer(
        &mut self,
        buffer: &dispatch::DispatchBuffer,
        index_format: crate::IndexFormat,
        offset: crate::BufferAddress,
        size: Option<crate::BufferSize>,
    ) {
        let buffer = buffer.as_core();

        if let Err(cause) =
            self.pass
                .set_index_buffer(buffer.wgpu_buffer.clone(), index_format, offset, size)
        {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::set_index_buffer",
            );
        }
    }

    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: Option<&dispatch::DispatchBuffer>,
        offset: crate::BufferAddress,
        size: Option<crate::BufferSize>,
    ) {
        let buffer = buffer.map(|buffer| buffer.as_core().wgpu_buffer.clone());

        if let Err(cause) = self.pass.set_vertex_buffer(slot, buffer, offset, size) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::set_vertex_buffer",
            );
        }
    }

    fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        if let Err(cause) = self.pass.set_immediates(offset, data) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::set_immediates",
            );
        }
    }

    fn set_blend_constant(&mut self, color: crate::Color) {
        if let Err(cause) = self.pass.set_blend_constant(color) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::set_blend_constant",
            );
        }
    }

    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        if let Err(cause) = self.pass.set_scissor_rect(x, y, width, height) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::set_scissor_rect",
            );
        }
    }

    fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        if let Err(cause) = self
            .pass
            .set_viewport(x, y, width, height, min_depth, max_depth)
        {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::set_viewport",
            );
        }
    }

    fn set_stencil_reference(&mut self, reference: u32) {
        if let Err(cause) = self.pass.set_stencil_reference(reference) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::set_stencil_reference",
            );
        }
    }

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        if let Err(cause) = self.pass.draw(
            vertices.end - vertices.start,
            instances.end - instances.start,
            vertices.start,
            instances.start,
        ) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::draw",
            );
        }
    }

    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        if let Err(cause) = self.pass.draw_indexed(
            indices.end - indices.start,
            instances.end - instances.start,
            indices.start,
            base_vertex,
            instances.start,
        ) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::draw_indexed",
            );
        }
    }

    fn draw_mesh_tasks(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        if let Err(cause) = self
            .pass
            .draw_mesh_tasks(group_count_x, group_count_y, group_count_z)
        {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::draw_mesh_tasks",
            );
        }
    }

    fn draw_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) {
        let indirect_buffer = indirect_buffer.as_core();

        if let Err(cause) = self
            .pass
            .draw_indirect(indirect_buffer.wgpu_buffer.clone(), indirect_offset)
        {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::draw_indirect",
            );
        }
    }

    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) {
        let indirect_buffer = indirect_buffer.as_core();

        if let Err(cause) = self
            .pass
            .draw_indexed_indirect(indirect_buffer.wgpu_buffer.clone(), indirect_offset)
        {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::draw_indexed_indirect",
            );
        }
    }

    fn draw_mesh_tasks_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) {
        let indirect_buffer = indirect_buffer.as_core();

        if let Err(cause) = self
            .pass
            .draw_mesh_tasks_indirect(indirect_buffer.wgpu_buffer.clone(), indirect_offset)
        {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::draw_mesh_tasks_indirect",
            );
        }
    }

    fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count: u32,
    ) {
        let indirect_buffer = indirect_buffer.as_core();

        if let Err(cause) = self.pass.multi_draw_indirect(
            indirect_buffer.wgpu_buffer.clone(),
            indirect_offset,
            count,
        ) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::multi_draw_indirect",
            );
        }
    }

    fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count: u32,
    ) {
        let indirect_buffer = indirect_buffer.as_core();

        if let Err(cause) = self.pass.multi_draw_indexed_indirect(
            indirect_buffer.wgpu_buffer.clone(),
            indirect_offset,
            count,
        ) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::multi_draw_indexed_indirect",
            );
        }
    }

    fn multi_draw_mesh_tasks_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count: u32,
    ) {
        let indirect_buffer = indirect_buffer.as_core();

        if let Err(cause) = self.pass.multi_draw_mesh_tasks_indirect(
            indirect_buffer.wgpu_buffer.clone(),
            indirect_offset,
            count,
        ) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::multi_draw_mesh_tasks_indirect",
            );
        }
    }

    fn multi_draw_indirect_count(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count_buffer: &dispatch::DispatchBuffer,
        count_buffer_offset: crate::BufferAddress,
        max_count: u32,
    ) {
        let indirect_buffer = indirect_buffer.as_core();
        let count_buffer = count_buffer.as_core();

        if let Err(cause) = self.pass.multi_draw_indirect_count(
            indirect_buffer.wgpu_buffer.clone(),
            indirect_offset,
            count_buffer.wgpu_buffer.clone(),
            count_buffer_offset,
            max_count,
        ) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::multi_draw_indirect_count",
            );
        }
    }

    fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count_buffer: &dispatch::DispatchBuffer,
        count_buffer_offset: crate::BufferAddress,
        max_count: u32,
    ) {
        let indirect_buffer = indirect_buffer.as_core();
        let count_buffer = count_buffer.as_core();

        if let Err(cause) = self.pass.multi_draw_indexed_indirect_count(
            indirect_buffer.wgpu_buffer.clone(),
            indirect_offset,
            count_buffer.wgpu_buffer.clone(),
            count_buffer_offset,
            max_count,
        ) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::multi_draw_indexed_indirect_count",
            );
        }
    }

    fn multi_draw_mesh_tasks_indirect_count(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
        count_buffer: &dispatch::DispatchBuffer,
        count_buffer_offset: crate::BufferAddress,
        max_count: u32,
    ) {
        let indirect_buffer = indirect_buffer.as_core();
        let count_buffer = count_buffer.as_core();

        if let Err(cause) = self.pass.multi_draw_mesh_tasks_indirect_count(
            indirect_buffer.wgpu_buffer.clone(),
            indirect_offset,
            count_buffer.wgpu_buffer.clone(),
            count_buffer_offset,
            max_count,
        ) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::multi_draw_mesh_tasks_indirect_count",
            );
        }
    }

    fn insert_debug_marker(&mut self, label: &str) {
        if let Err(cause) = self.pass.insert_debug_marker(label, 0) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::insert_debug_marker",
            );
        }
    }

    fn push_debug_group(&mut self, group_label: &str) {
        if let Err(cause) = self.pass.push_debug_group(group_label, 0) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::push_debug_group",
            );
        }
    }

    fn pop_debug_group(&mut self) {
        if let Err(cause) = self.pass.pop_debug_group() {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::pop_debug_group",
            );
        }
    }

    fn write_timestamp(&mut self, query_set: &dispatch::DispatchQuerySet, query_index: u32) {
        let query_set = query_set.as_core();

        if let Err(cause) = self
            .pass
            .write_timestamp(query_set.wgpu_query_set.clone(), query_index)
        {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::write_timestamp",
            );
        }
    }

    fn begin_occlusion_query(&mut self, query_index: u32) {
        if let Err(cause) = self.pass.begin_occlusion_query(query_index) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::begin_occlusion_query",
            );
        }
    }

    fn end_occlusion_query(&mut self) {
        if let Err(cause) = self.pass.end_occlusion_query() {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::end_occlusion_query",
            );
        }
    }

    fn begin_pipeline_statistics_query(
        &mut self,
        query_set: &dispatch::DispatchQuerySet,
        query_index: u32,
    ) {
        let query_set = query_set.as_core();

        if let Err(cause) = self
            .pass
            .begin_pipeline_statistics_query(query_set.wgpu_query_set.clone(), query_index)
        {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::begin_pipeline_statistics_query",
            );
        }
    }

    fn end_pipeline_statistics_query(&mut self) {
        if let Err(cause) = self.pass.end_pipeline_statistics_query() {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::end_pipeline_statistics_query",
            );
        }
    }

    fn execute_bundles(
        &mut self,
        render_bundles: &mut dyn Iterator<Item = &dispatch::DispatchRenderBundle>,
    ) {
        let temp_render_bundles = render_bundles
            .map(|rb| rb.as_core().wgpu_render_bundle.clone())
            .collect::<SmallVec<[_; 4]>>();
        if let Err(cause) = self.pass.execute_bundles(&temp_render_bundles) {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::execute_bundles",
            );
        }
    }
}

impl Drop for CoreRenderPass {
    fn drop(&mut self) {
        if let Err(cause) = self.pass.end() {
            self.context.handle_error(
                &self.error_sink,
                cause,
                self.pass.label(),
                "RenderPass::end",
            );
        }
    }
}

impl dispatch::RenderBundleEncoderInterface for CoreRenderBundleEncoder {
    fn set_pipeline(&mut self, pipeline: &dispatch::DispatchRenderPipeline) {
        let pipeline = pipeline.as_core();

        self.encoder
            .set_pipeline(pipeline.wgpu_render_pipeline.clone())
            .expect("RenderBundleEncoder should not have ended")
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&dispatch::DispatchBindGroup>,
        offsets: &[crate::DynamicOffset],
    ) {
        let bg = bind_group.map(|bg| bg.as_core().wgpu_bind_group.clone());

        self.encoder
            .set_bind_group(index, bg, offsets)
            .expect("RenderBundleEncoder should not have ended");
    }

    fn set_index_buffer(
        &mut self,
        buffer: &dispatch::DispatchBuffer,
        index_format: crate::IndexFormat,
        offset: crate::BufferAddress,
        size: Option<crate::BufferSize>,
    ) {
        let buffer = buffer.as_core();

        self.encoder
            .set_index_buffer(buffer.wgpu_buffer.clone(), index_format, offset, size)
            .expect("RenderBundleEncoder should not have ended");
    }

    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: Option<&dispatch::DispatchBuffer>,
        offset: crate::BufferAddress,
        size: Option<crate::BufferSize>,
    ) {
        let buffer = buffer.map(|buffer| buffer.as_core().wgpu_buffer.clone());

        self.encoder
            .set_vertex_buffer(slot, buffer, offset, size)
            .expect("RenderBundleEncoder should not have ended");
    }

    fn set_immediates(&mut self, offset: u32, data: &[u8]) {
        if !data
            .len()
            .is_multiple_of(wgt::IMMEDIATE_DATA_ALIGNMENT as usize)
        {
            self.context.handle_error(
                &self.error_sink,
                wgc::binding_model::ImmediateUploadError::SizeUnaligned(data.len()),
                self.encoder.label(),
                "RenderBundleEncoder::set_immediates",
            );
            return;
        }

        self.encoder
            .set_immediates(offset, data)
            .expect("RenderBundleEncoder should not have ended");
    }

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.encoder
            .draw(
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            )
            .expect("RenderBundleEncoder should not have ended");
    }

    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.encoder
            .draw_indexed(
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            )
            .expect("RenderBundleEncoder should not have ended");
    }

    fn draw_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) {
        let indirect_buffer = indirect_buffer.as_core();

        self.encoder
            .draw_indirect(indirect_buffer.wgpu_buffer.clone(), indirect_offset)
            .expect("RenderBundleEncoder should not have ended");
    }

    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &dispatch::DispatchBuffer,
        indirect_offset: crate::BufferAddress,
    ) {
        let indirect_buffer = indirect_buffer.as_core();

        self.encoder
            .draw_indexed_indirect(indirect_buffer.wgpu_buffer.clone(), indirect_offset)
            .expect("RenderBundleEncoder should not have ended");
    }

    fn finish(mut self, desc: &crate::RenderBundleDescriptor<'_>) -> dispatch::DispatchRenderBundle
    where
        Self: Sized,
    {
        let label = self.encoder.label().map(alloc::string::ToString::to_string);
        let (wgpu_render_bundle, error) = self.encoder.finish(&desc.map_label(|l| l.map(Borrowed)));
        if let Some(err) = error {
            self.context.handle_error(
                &self.error_sink,
                err,
                label.as_deref(),
                "RenderBundleEncoder::finish",
            );
        }
        CoreRenderBundle { wgpu_render_bundle }.into()
    }

    #[cfg(custom)]
    fn finish_boxed(
        self: Box<Self>,
        desc: &crate::RenderBundleDescriptor<'_>,
    ) -> dispatch::DispatchRenderBundle {
        (*self).finish(desc)
    }
}

impl dispatch::RenderBundleInterface for CoreRenderBundle {}

impl dispatch::SurfaceInterface for CoreSurface {
    fn get_capabilities(&self, adapter: &dispatch::DispatchAdapter) -> wgt::SurfaceCapabilities {
        let adapter = adapter.as_core();

        self.wgpu_surface
            .get_capabilities(&adapter.wgpu_adapter)
            .unwrap_or_default()
    }

    fn display_hdr_info(&self, adapter: &dispatch::DispatchAdapter) -> wgt::DisplayHdrInfo {
        let adapter = adapter.as_core();

        self.wgpu_surface.display_hdr_info(&adapter.wgpu_adapter)
    }

    fn configure(&self, device: &dispatch::DispatchDevice, config: &crate::SurfaceConfiguration) {
        let device = device.as_core();

        let error = self.wgpu_surface.configure(&device.wgpu_device, config);
        if let Some(e) = error {
            self.context
                .handle_error_nolabel(&device.error_sink, e, "Surface::configure");
        } else {
            *self.configured_device.lock() = Some(device.wgpu_device.clone());
            *self.error_sink.lock() = Some(device.error_sink.clone());
        }
    }

    fn get_current_texture(
        &self,
    ) -> (
        Option<dispatch::DispatchTexture>,
        crate::SurfaceStatus,
        dispatch::DispatchSurfaceOutputDetail,
    ) {
        let error_sink = if let Some(error_sink) = self.error_sink.lock().as_ref() {
            error_sink.clone()
        } else {
            Arc::new(Mutex::new(ErrorSinkRaw::new()))
        };

        let output_detail = CoreSurfaceOutputDetail {
            context: self.context.clone(),
            wgpu_surface: self.wgpu_surface.clone(),
            error_sink: error_sink.clone(),
        }
        .into();

        match self.wgpu_surface.get_current_texture() {
            Ok(wgc::present::SurfaceOutput {
                status,
                texture: texture_id,
            }) => {
                let data = texture_id
                    .map(|wgpu_texture| CoreTexture {
                        context: self.context.clone(),
                        wgpu_texture,
                        error_sink,
                    })
                    .map(Into::into);

                (data, status, output_detail)
            }
            Err(err) => {
                let error_sink = self.error_sink.lock();
                match error_sink.as_ref() {
                    Some(error_sink) => {
                        self.context.handle_error_nolabel(
                            error_sink,
                            err,
                            "Surface::get_current_texture_view",
                        );
                        (None, crate::SurfaceStatus::Validation, output_detail)
                    }
                    None => self
                        .context
                        .handle_error_fatal(err, "Surface::get_current_texture_view"),
                }
            }
        }
    }
}

impl dispatch::SurfaceOutputDetailInterface for CoreSurfaceOutputDetail {
    fn texture_discard(&self) {
        match self.wgpu_surface.discard() {
            Ok(_status) => (),
            Err(err) => {
                self.context
                    .handle_error_nolabel(&self.error_sink, err, "Surface::discard_texture")
            }
        }
    }

    fn texture_release(&self) {
        match self.wgpu_surface.release() {
            Ok(_status) => (),
            Err(err) => {
                self.context
                    .handle_error_nolabel(&self.error_sink, err, "Surface::release_texture")
            }
        }
    }
}

impl dispatch::QueueWriteBufferInterface for CoreQueueWriteBuffer {
    #[inline]
    fn len(&self) -> usize {
        self.mapping.len()
    }

    #[inline]
    unsafe fn write_slice(&mut self) -> WriteOnly<'_, [u8]> {
        unsafe { self.mapping.write_slice() }
    }
}

impl dispatch::BufferMappedRangeInterface for CoreBufferMappedRange {
    #[inline]
    fn len(&self) -> usize {
        self.size
    }

    #[inline]
    unsafe fn read_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    #[inline]
    unsafe fn write_slice(&mut self) -> WriteOnly<'_, [u8]> {
        unsafe { WriteOnly::new(NonNull::slice_from_raw_parts(self.ptr, self.size)) }
    }

    #[cfg(webgpu)]
    fn as_uint8array(&self) -> &js_sys::Uint8Array {
        panic!("Only available on WebGPU")
    }
}
