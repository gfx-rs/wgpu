use crate::{
    AdapterInfo, BindGroupDescriptor, BindGroupLayoutDescriptor, BindingResource, BufferBinding,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    DownlevelCapabilities, Features, Label, Limits, LoadOp, MapMode, Operations,
    PipelineLayoutDescriptor, RenderBundleEncoderDescriptor, RenderPipelineDescriptor,
    SamplerDescriptor, ShaderModuleDescriptor, ShaderModuleDescriptorSpirV, ShaderSource,
    SurfaceStatus, TextureDescriptor, TextureFormat, TextureViewDescriptor,
};

use arrayvec::ArrayVec;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    borrow::Cow::Borrowed,
    error::Error,
    fmt,
    future::{ready, Ready},
    marker::PhantomData,
    ops::Range,
    slice,
    sync::Arc,
};

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
    #[cfg(any(not(target_arch = "wasm32"), feature = "emscripten"))]
    pub unsafe fn from_hal_instance<A: wgc::hub::HalApi>(hal_instance: A::Instance) -> Self {
        Self(wgc::hub::Global::from_hal_instance::<A>(
            "wgpu",
            wgc::hub::IdentityManagerFactory,
            hal_instance,
        ))
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "webgl2"))]
    pub unsafe fn instance_as_hal<A: wgc::hub::HalApi, F: FnOnce(Option<&A::Instance>) -> R, R>(
        &self,
        hal_instance_callback: F,
    ) -> R {
        self.0.instance_as_hal::<A, F, R>(hal_instance_callback)
    }

    pub(crate) fn global(&self) -> &wgc::hub::Global<wgc::hub::IdentityManagerFactory> {
        &self.0
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "emscripten"))]
    pub fn enumerate_adapters(&self, backends: wgt::Backends) -> Vec<wgc::id::AdapterId> {
        self.0
            .enumerate_adapters(wgc::instance::AdapterInputs::Mask(backends, |_| {
                PhantomData
            }))
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "emscripten"))]
    pub unsafe fn create_adapter_from_hal<A: wgc::hub::HalApi>(
        &self,
        hal_adapter: hal::ExposedAdapter<A>,
    ) -> wgc::id::AdapterId {
        self.0.create_adapter_from_hal(hal_adapter, PhantomData)
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "webgl2"))]
    pub unsafe fn adapter_as_hal<A: wgc::hub::HalApi, F: FnOnce(Option<&A::Adapter>) -> R, R>(
        &self,
        adapter: wgc::id::AdapterId,
        hal_adapter_callback: F,
    ) -> R {
        self.0
            .adapter_as_hal::<A, F, R>(adapter, hal_adapter_callback)
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "emscripten"))]
    pub unsafe fn create_device_from_hal<A: wgc::hub::HalApi>(
        &self,
        adapter: &wgc::id::AdapterId,
        hal_device: hal::OpenDevice<A>,
        desc: &crate::DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Result<(Device, wgc::id::QueueId), crate::RequestDeviceError> {
        let global = &self.0;
        let (device_id, error) = global.create_device_from_hal(
            *adapter,
            hal_device,
            &desc.map_label(|l| l.map(Borrowed)),
            trace_dir,
            PhantomData,
        );
        if let Some(err) = error {
            self.handle_error_fatal(err, "Adapter::create_device_from_hal");
        }
        let device = Device {
            id: device_id,
            error_sink: Arc::new(Mutex::new(ErrorSinkRaw::new())),
            features: desc.features,
        };
        Ok((device, device_id))
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "emscripten"))]
    pub unsafe fn create_texture_from_hal<A: wgc::hub::HalApi>(
        &self,
        hal_texture: A::Texture,
        device: &Device,
        desc: &TextureDescriptor,
    ) -> Texture {
        let global = &self.0;
        let (id, error) = global.create_texture_from_hal::<A>(
            hal_texture,
            device.id,
            &desc.map_label(|l| l.map(Borrowed)),
            PhantomData,
        );
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

    #[cfg(any(not(target_arch = "wasm32"), feature = "emscripten"))]
    pub unsafe fn device_as_hal<A: wgc::hub::HalApi, F: FnOnce(Option<&A::Device>) -> R, R>(
        &self,
        device: &Device,
        hal_device_callback: F,
    ) -> R {
        self.0
            .device_as_hal::<A, F, R>(device.id, hal_device_callback)
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "emscripten"))]
    pub unsafe fn texture_as_hal<A: wgc::hub::HalApi, F: FnOnce(Option<&A::Texture>)>(
        &self,
        texture: &Texture,
        hal_texture_callback: F,
    ) {
        self.0
            .texture_as_hal::<A, F>(texture.id, hal_texture_callback)
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "emscripten"))]
    pub fn generate_report(&self) -> wgc::hub::GlobalReport {
        self.0.generate_report()
    }

    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub unsafe fn create_surface_from_core_animation_layer(
        self: &Arc<Self>,
        layer: *mut std::ffi::c_void,
    ) -> crate::Surface {
        let id = self.0.instance_create_surface_metal(layer, PhantomData);
        crate::Surface {
            context: Arc::clone(self),
            id: Surface {
                id,
                configured_device: Mutex::default(),
            },
        }
    }

    #[cfg(all(target_arch = "wasm32", feature = "webgl", not(feature = "emscripten")))]
    pub fn instance_create_surface_from_canvas(
        self: &Arc<Self>,
        canvas: &web_sys::HtmlCanvasElement,
    ) -> Surface {
        let id = self.0.create_surface_webgl_canvas(canvas, PhantomData);
        Surface {
            id,
            configured_device: Mutex::default(),
        }
    }

    #[cfg(all(target_arch = "wasm32", feature = "webgl", not(feature = "emscripten")))]
    pub fn instance_create_surface_from_offscreen_canvas(
        self: &Arc<Self>,
        canvas: &web_sys::OffscreenCanvas,
    ) -> Surface {
        let id = self
            .0
            .create_surface_webgl_offscreen_canvas(canvas, PhantomData);
        Surface {
            id,
            configured_device: Mutex::default(),
        }
    }

    #[cfg(target_os = "windows")]
    pub unsafe fn create_surface_from_visual(
        self: &Arc<Self>,
        visual: *mut std::ffi::c_void,
    ) -> crate::Surface {
        let id = self
            .0
            .instance_create_surface_from_visual(visual, PhantomData);
        crate::Surface {
            context: Arc::clone(self),
            id: Surface {
                id,
                configured_device: Mutex::default(),
            },
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

    fn handle_error_fatal(
        &self,
        cause: impl Error + Send + Sync + 'static,
        string: &'static str,
    ) -> ! {
        panic!("Error in {}: {}", string, cause);
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

mod pass_impl {
    use super::Context;
    use smallvec::SmallVec;
    use std::convert::TryInto;
    use std::ops::Range;
    use wgc::command::{bundle_ffi::*, compute_ffi::*, render_ffi::*};

    impl crate::ComputePassInner<Context> for wgc::command::ComputePass {
        fn set_pipeline(&mut self, pipeline: &wgc::id::ComputePipelineId) {
            wgpu_compute_pass_set_pipeline(self, *pipeline)
        }
        fn set_bind_group(
            &mut self,
            index: u32,
            bind_group: &wgc::id::BindGroupId,
            offsets: &[wgt::DynamicOffset],
        ) {
            unsafe {
                wgpu_compute_pass_set_bind_group(
                    self,
                    index,
                    *bind_group,
                    offsets.as_ptr(),
                    offsets.len(),
                )
            }
        }
        fn set_push_constants(&mut self, offset: u32, data: &[u8]) {
            unsafe {
                wgpu_compute_pass_set_push_constant(
                    self,
                    offset,
                    data.len().try_into().unwrap(),
                    data.as_ptr(),
                )
            }
        }
        fn insert_debug_marker(&mut self, label: &str) {
            unsafe {
                let label = std::ffi::CString::new(label).unwrap();
                wgpu_compute_pass_insert_debug_marker(self, label.as_ptr(), 0);
            }
        }

        fn push_debug_group(&mut self, group_label: &str) {
            unsafe {
                let label = std::ffi::CString::new(group_label).unwrap();
                wgpu_compute_pass_push_debug_group(self, label.as_ptr(), 0);
            }
        }
        fn pop_debug_group(&mut self) {
            wgpu_compute_pass_pop_debug_group(self);
        }

        fn write_timestamp(&mut self, query_set: &wgc::id::QuerySetId, query_index: u32) {
            wgpu_compute_pass_write_timestamp(self, *query_set, query_index)
        }

        fn begin_pipeline_statistics_query(
            &mut self,
            query_set: &wgc::id::QuerySetId,
            query_index: u32,
        ) {
            wgpu_compute_pass_begin_pipeline_statistics_query(self, *query_set, query_index)
        }

        fn end_pipeline_statistics_query(&mut self) {
            wgpu_compute_pass_end_pipeline_statistics_query(self)
        }

        fn dispatch_workgroups(&mut self, x: u32, y: u32, z: u32) {
            wgpu_compute_pass_dispatch_workgroups(self, x, y, z)
        }
        fn dispatch_workgroups_indirect(
            &mut self,
            indirect_buffer: &super::Buffer,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_compute_pass_dispatch_workgroups_indirect(
                self,
                indirect_buffer.id,
                indirect_offset,
            )
        }
    }

    impl crate::RenderInner<Context> for wgc::command::RenderPass {
        fn set_pipeline(&mut self, pipeline: &wgc::id::RenderPipelineId) {
            wgpu_render_pass_set_pipeline(self, *pipeline)
        }
        fn set_bind_group(
            &mut self,
            index: u32,
            bind_group: &wgc::id::BindGroupId,
            offsets: &[wgt::DynamicOffset],
        ) {
            unsafe {
                wgpu_render_pass_set_bind_group(
                    self,
                    index,
                    *bind_group,
                    offsets.as_ptr(),
                    offsets.len(),
                )
            }
        }
        fn set_index_buffer(
            &mut self,
            buffer: &super::Buffer,
            index_format: wgt::IndexFormat,
            offset: wgt::BufferAddress,
            size: Option<wgt::BufferSize>,
        ) {
            self.set_index_buffer(buffer.id, index_format, offset, size)
        }
        fn set_vertex_buffer(
            &mut self,
            slot: u32,
            buffer: &super::Buffer,
            offset: wgt::BufferAddress,
            size: Option<wgt::BufferSize>,
        ) {
            wgpu_render_pass_set_vertex_buffer(self, slot, buffer.id, offset, size)
        }
        fn set_push_constants(&mut self, stages: wgt::ShaderStages, offset: u32, data: &[u8]) {
            unsafe {
                wgpu_render_pass_set_push_constants(
                    self,
                    stages,
                    offset,
                    data.len().try_into().unwrap(),
                    data.as_ptr(),
                )
            }
        }
        fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
            wgpu_render_pass_draw(
                self,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            )
        }
        fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
            wgpu_render_pass_draw_indexed(
                self,
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            )
        }
        fn draw_indirect(
            &mut self,
            indirect_buffer: &super::Buffer,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_render_pass_draw_indirect(self, indirect_buffer.id, indirect_offset)
        }
        fn draw_indexed_indirect(
            &mut self,
            indirect_buffer: &super::Buffer,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_render_pass_draw_indexed_indirect(self, indirect_buffer.id, indirect_offset)
        }
        fn multi_draw_indirect(
            &mut self,
            indirect_buffer: &super::Buffer,
            indirect_offset: wgt::BufferAddress,
            count: u32,
        ) {
            wgpu_render_pass_multi_draw_indirect(self, indirect_buffer.id, indirect_offset, count)
        }
        fn multi_draw_indexed_indirect(
            &mut self,
            indirect_buffer: &super::Buffer,
            indirect_offset: wgt::BufferAddress,
            count: u32,
        ) {
            wgpu_render_pass_multi_draw_indexed_indirect(
                self,
                indirect_buffer.id,
                indirect_offset,
                count,
            )
        }
        fn multi_draw_indirect_count(
            &mut self,
            indirect_buffer: &super::Buffer,
            indirect_offset: wgt::BufferAddress,
            count_buffer: &super::Buffer,
            count_buffer_offset: wgt::BufferAddress,
            max_count: u32,
        ) {
            wgpu_render_pass_multi_draw_indirect_count(
                self,
                indirect_buffer.id,
                indirect_offset,
                count_buffer.id,
                count_buffer_offset,
                max_count,
            )
        }
        fn multi_draw_indexed_indirect_count(
            &mut self,
            indirect_buffer: &super::Buffer,
            indirect_offset: wgt::BufferAddress,
            count_buffer: &super::Buffer,
            count_buffer_offset: wgt::BufferAddress,
            max_count: u32,
        ) {
            wgpu_render_pass_multi_draw_indexed_indirect_count(
                self,
                indirect_buffer.id,
                indirect_offset,
                count_buffer.id,
                count_buffer_offset,
                max_count,
            )
        }
    }

    impl crate::RenderPassInner<Context> for wgc::command::RenderPass {
        fn set_blend_constant(&mut self, color: wgt::Color) {
            wgpu_render_pass_set_blend_constant(self, &color)
        }
        fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
            wgpu_render_pass_set_scissor_rect(self, x, y, width, height)
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
            wgpu_render_pass_set_viewport(self, x, y, width, height, min_depth, max_depth)
        }
        fn set_stencil_reference(&mut self, reference: u32) {
            wgpu_render_pass_set_stencil_reference(self, reference)
        }

        fn insert_debug_marker(&mut self, label: &str) {
            unsafe {
                let label = std::ffi::CString::new(label).unwrap();
                wgpu_render_pass_insert_debug_marker(self, label.as_ptr(), 0);
            }
        }

        fn push_debug_group(&mut self, group_label: &str) {
            unsafe {
                let label = std::ffi::CString::new(group_label).unwrap();
                wgpu_render_pass_push_debug_group(self, label.as_ptr(), 0);
            }
        }

        fn pop_debug_group(&mut self) {
            wgpu_render_pass_pop_debug_group(self);
        }

        fn write_timestamp(&mut self, query_set: &wgc::id::QuerySetId, query_index: u32) {
            wgpu_render_pass_write_timestamp(self, *query_set, query_index)
        }

        fn begin_pipeline_statistics_query(
            &mut self,
            query_set: &wgc::id::QuerySetId,
            query_index: u32,
        ) {
            wgpu_render_pass_begin_pipeline_statistics_query(self, *query_set, query_index)
        }

        fn end_pipeline_statistics_query(&mut self) {
            wgpu_render_pass_end_pipeline_statistics_query(self)
        }

        fn execute_bundles<'a, I: Iterator<Item = &'a wgc::id::RenderBundleId>>(
            &mut self,
            render_bundles: I,
        ) {
            let temp_render_bundles = render_bundles.cloned().collect::<SmallVec<[_; 4]>>();
            unsafe {
                wgpu_render_pass_execute_bundles(
                    self,
                    temp_render_bundles.as_ptr(),
                    temp_render_bundles.len(),
                )
            }
        }
    }

    impl crate::RenderInner<Context> for wgc::command::RenderBundleEncoder {
        fn set_pipeline(&mut self, pipeline: &wgc::id::RenderPipelineId) {
            wgpu_render_bundle_set_pipeline(self, *pipeline)
        }
        fn set_bind_group(
            &mut self,
            index: u32,
            bind_group: &wgc::id::BindGroupId,
            offsets: &[wgt::DynamicOffset],
        ) {
            unsafe {
                wgpu_render_bundle_set_bind_group(
                    self,
                    index,
                    *bind_group,
                    offsets.as_ptr(),
                    offsets.len(),
                )
            }
        }
        fn set_index_buffer(
            &mut self,
            buffer: &super::Buffer,
            index_format: wgt::IndexFormat,
            offset: wgt::BufferAddress,
            size: Option<wgt::BufferSize>,
        ) {
            self.set_index_buffer(buffer.id, index_format, offset, size)
        }
        fn set_vertex_buffer(
            &mut self,
            slot: u32,
            buffer: &super::Buffer,
            offset: wgt::BufferAddress,
            size: Option<wgt::BufferSize>,
        ) {
            wgpu_render_bundle_set_vertex_buffer(self, slot, buffer.id, offset, size)
        }

        fn set_push_constants(&mut self, stages: wgt::ShaderStages, offset: u32, data: &[u8]) {
            unsafe {
                wgpu_render_bundle_set_push_constants(
                    self,
                    stages,
                    offset,
                    data.len().try_into().unwrap(),
                    data.as_ptr(),
                )
            }
        }
        fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
            wgpu_render_bundle_draw(
                self,
                vertices.end - vertices.start,
                instances.end - instances.start,
                vertices.start,
                instances.start,
            )
        }
        fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
            wgpu_render_bundle_draw_indexed(
                self,
                indices.end - indices.start,
                instances.end - instances.start,
                indices.start,
                base_vertex,
                instances.start,
            )
        }
        fn draw_indirect(
            &mut self,
            indirect_buffer: &super::Buffer,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_render_bundle_draw_indirect(self, indirect_buffer.id, indirect_offset)
        }
        fn draw_indexed_indirect(
            &mut self,
            indirect_buffer: &super::Buffer,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_render_bundle_draw_indexed_indirect(self, indirect_buffer.id, indirect_offset)
        }
        fn multi_draw_indirect(
            &mut self,
            _indirect_buffer: &super::Buffer,
            _indirect_offset: wgt::BufferAddress,
            _count: u32,
        ) {
            unimplemented!()
        }
        fn multi_draw_indexed_indirect(
            &mut self,
            _indirect_buffer: &super::Buffer,
            _indirect_offset: wgt::BufferAddress,
            _count: u32,
        ) {
            unimplemented!()
        }
        fn multi_draw_indirect_count(
            &mut self,
            _indirect_buffer: &super::Buffer,
            _indirect_offset: wgt::BufferAddress,
            _count_buffer: &super::Buffer,
            _count_buffer_offset: wgt::BufferAddress,
            _max_count: u32,
        ) {
            unimplemented!()
        }
        fn multi_draw_indexed_indirect_count(
            &mut self,
            _indirect_buffer: &super::Buffer,
            _indirect_offset: wgt::BufferAddress,
            _count_buffer: &super::Buffer,
            _count_buffer_offset: wgt::BufferAddress,
            _max_count: u32,
        ) {
            unimplemented!()
        }
    }
}

fn map_buffer_copy_view(view: crate::ImageCopyBuffer) -> wgc::command::ImageCopyBuffer {
    wgc::command::ImageCopyBuffer {
        buffer: view.buffer.id.id,
        layout: view.layout,
    }
}

fn map_texture_copy_view(view: crate::ImageCopyTexture) -> wgc::command::ImageCopyTexture {
    wgc::command::ImageCopyTexture {
        texture: view.texture.id.id,
        mip_level: view.mip_level,
        origin: view.origin,
        aspect: view.aspect,
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

#[derive(Debug)]
pub struct Device {
    id: wgc::id::DeviceId,
    error_sink: ErrorSink,
    features: Features,
}

#[derive(Debug)]
pub(crate) struct Buffer {
    id: wgc::id::BufferId,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub struct Texture {
    id: wgc::id::TextureId,
    error_sink: ErrorSink,
}

#[derive(Debug)]
pub(crate) struct CommandEncoder {
    id: wgc::id::CommandEncoderId,
    error_sink: ErrorSink,
    open: bool,
}

impl crate::Context for Context {
    type AdapterId = wgc::id::AdapterId;
    type DeviceId = Device;
    type QueueId = wgc::id::QueueId;
    type ShaderModuleId = wgc::id::ShaderModuleId;
    type BindGroupLayoutId = wgc::id::BindGroupLayoutId;
    type BindGroupId = wgc::id::BindGroupId;
    type TextureViewId = wgc::id::TextureViewId;
    type SamplerId = wgc::id::SamplerId;
    type QuerySetId = wgc::id::QuerySetId;
    type BufferId = Buffer;
    type TextureId = Texture;
    type PipelineLayoutId = wgc::id::PipelineLayoutId;
    type RenderPipelineId = wgc::id::RenderPipelineId;
    type ComputePipelineId = wgc::id::ComputePipelineId;
    type CommandEncoderId = CommandEncoder;
    type ComputePassId = wgc::command::ComputePass;
    type RenderPassId = wgc::command::RenderPass;
    type CommandBufferId = wgc::id::CommandBufferId;
    type RenderBundleEncoderId = wgc::command::RenderBundleEncoder;
    type RenderBundleId = wgc::id::RenderBundleId;
    type SurfaceId = Surface;

    type SurfaceOutputDetail = SurfaceOutputDetail;
    type SubmissionIndex = wgc::device::queue::WrappedSubmissionIndex;

    type RequestAdapterFuture = Ready<Option<Self::AdapterId>>;
    #[allow(clippy::type_complexity)]
    type RequestDeviceFuture =
        Ready<Result<(Self::DeviceId, Self::QueueId), crate::RequestDeviceError>>;
    type PopErrorScopeFuture = Ready<Option<crate::Error>>;

    fn init(backends: wgt::Backends) -> Self {
        Self(wgc::hub::Global::new(
            "wgpu",
            wgc::hub::IdentityManagerFactory,
            backends,
        ))
    }

    fn instance_create_surface(
        &self,
        handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Self::SurfaceId {
        Surface {
            id: self.0.instance_create_surface(handle, PhantomData),
            configured_device: Mutex::new(None),
        }
    }

    fn instance_request_adapter(
        &self,
        options: &crate::RequestAdapterOptions,
    ) -> Self::RequestAdapterFuture {
        let id = self.0.request_adapter(
            &wgc::instance::RequestAdapterOptions {
                power_preference: options.power_preference,
                force_fallback_adapter: options.force_fallback_adapter,
                compatible_surface: options.compatible_surface.map(|surface| surface.id.id),
            },
            wgc::instance::AdapterInputs::Mask(wgt::Backends::all(), |_| PhantomData),
        );
        ready(id.ok())
    }

    fn instance_poll_all_devices(&self, force_wait: bool) -> bool {
        let global = &self.0;
        match global.poll_all_devices(force_wait) {
            Ok(all_queue_empty) => all_queue_empty,
            Err(err) => self.handle_error_fatal(err, "Device::poll"),
        }
    }

    fn adapter_request_device(
        &self,
        adapter: &Self::AdapterId,
        desc: &crate::DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture {
        let global = &self.0;
        let (device_id, error) = wgc::gfx_select!(*adapter => global.adapter_request_device(
            *adapter,
            &desc.map_label(|l| l.map(Borrowed)),
            trace_dir,
            PhantomData
        ));
        if let Some(err) = error {
            log::error!("Error in Adapter::request_device: {}", err);
            return ready(Err(crate::RequestDeviceError));
        }
        let device = Device {
            id: device_id,
            error_sink: Arc::new(Mutex::new(ErrorSinkRaw::new())),
            features: desc.features,
        };
        ready(Ok((device, device_id)))
    }

    fn adapter_is_surface_supported(
        &self,
        adapter: &Self::AdapterId,
        surface: &Self::SurfaceId,
    ) -> bool {
        let global = &self.0;
        match wgc::gfx_select!(adapter => global.adapter_is_surface_supported(*adapter, surface.id))
        {
            Ok(result) => result,
            Err(err) => self.handle_error_fatal(err, "Adapter::is_surface_supported"),
        }
    }

    fn adapter_features(&self, adapter: &Self::AdapterId) -> Features {
        let global = &self.0;
        match wgc::gfx_select!(*adapter => global.adapter_features(*adapter)) {
            Ok(features) => features,
            Err(err) => self.handle_error_fatal(err, "Adapter::features"),
        }
    }

    fn adapter_limits(&self, adapter: &Self::AdapterId) -> Limits {
        let global = &self.0;
        match wgc::gfx_select!(*adapter => global.adapter_limits(*adapter)) {
            Ok(limits) => limits,
            Err(err) => self.handle_error_fatal(err, "Adapter::limits"),
        }
    }

    fn adapter_downlevel_capabilities(&self, adapter: &Self::AdapterId) -> DownlevelCapabilities {
        let global = &self.0;
        match wgc::gfx_select!(*adapter => global.adapter_downlevel_capabilities(*adapter)) {
            Ok(downlevel) => downlevel,
            Err(err) => self.handle_error_fatal(err, "Adapter::downlevel_properties"),
        }
    }

    fn adapter_get_info(&self, adapter: &wgc::id::AdapterId) -> AdapterInfo {
        let global = &self.0;
        match wgc::gfx_select!(*adapter => global.adapter_get_info(*adapter)) {
            Ok(info) => info,
            Err(err) => self.handle_error_fatal(err, "Adapter::get_info"),
        }
    }

    fn adapter_get_texture_format_features(
        &self,
        adapter: &Self::AdapterId,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        let global = &self.0;
        match wgc::gfx_select!(*adapter => global.adapter_get_texture_format_features(*adapter, format))
        {
            Ok(info) => info,
            Err(err) => self.handle_error_fatal(err, "Adapter::get_texture_format_features"),
        }
    }

    fn surface_get_supported_formats(
        &self,
        surface: &Self::SurfaceId,
        adapter: &Self::AdapterId,
    ) -> Vec<TextureFormat> {
        let global = &self.0;
        match wgc::gfx_select!(adapter => global.surface_get_supported_formats(surface.id, *adapter))
        {
            Ok(formats) => formats,
            Err(wgc::instance::GetSurfacePreferredFormatError::UnsupportedQueueFamily) => vec![],
            Err(err) => self.handle_error_fatal(err, "Surface::get_supported_formats"),
        }
    }

    fn surface_configure(
        &self,
        surface: &Self::SurfaceId,
        device: &Self::DeviceId,
        config: &wgt::SurfaceConfiguration,
    ) {
        let global = &self.0;
        let error =
            wgc::gfx_select!(device.id => global.surface_configure(surface.id, device.id, config));
        if let Some(e) = error {
            self.handle_error_fatal(e, "Surface::configure");
        } else {
            *surface.configured_device.lock() = Some(device.id);
        }
    }

    fn surface_get_current_texture(
        &self,
        surface: &Self::SurfaceId,
    ) -> (
        Option<Self::TextureId>,
        SurfaceStatus,
        Self::SurfaceOutputDetail,
    ) {
        let global = &self.0;
        let device_id = surface
            .configured_device
            .lock()
            .expect("Surface was not configured?");
        match wgc::gfx_select!(
            device_id => global.surface_get_current_texture(surface.id, PhantomData)
        ) {
            Ok(wgc::present::SurfaceOutput { status, texture_id }) => (
                texture_id.map(|id| Texture {
                    id,
                    error_sink: Arc::new(Mutex::new(ErrorSinkRaw::new())),
                }),
                status,
                SurfaceOutputDetail {
                    surface_id: surface.id,
                },
            ),
            Err(err) => self.handle_error_fatal(err, "Surface::get_current_texture_view"),
        }
    }

    fn surface_present(&self, texture: &Self::TextureId, detail: &Self::SurfaceOutputDetail) {
        let global = &self.0;
        match wgc::gfx_select!(texture.id => global.surface_present(detail.surface_id)) {
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
        match wgc::gfx_select!(texture.id => global.surface_texture_discard(detail.surface_id)) {
            Ok(_status) => (),
            Err(err) => self.handle_error_fatal(err, "Surface::discard_texture"),
        }
    }

    fn device_features(&self, device: &Self::DeviceId) -> Features {
        let global = &self.0;
        match wgc::gfx_select!(device.id => global.device_features(device.id)) {
            Ok(features) => features,
            Err(err) => self.handle_error_fatal(err, "Device::features"),
        }
    }

    fn device_limits(&self, device: &Self::DeviceId) -> Limits {
        let global = &self.0;
        match wgc::gfx_select!(device.id => global.device_limits(device.id)) {
            Ok(limits) => limits,
            Err(err) => self.handle_error_fatal(err, "Device::limits"),
        }
    }

    fn device_downlevel_properties(&self, device: &Self::DeviceId) -> DownlevelCapabilities {
        let global = &self.0;
        match wgc::gfx_select!(device.id => global.device_downlevel_properties(device.id)) {
            Ok(limits) => limits,
            Err(err) => self.handle_error_fatal(err, "Device::downlevel_properties"),
        }
    }

    fn device_create_shader_module(
        &self,
        device: &Self::DeviceId,
        desc: &ShaderModuleDescriptor,
        shader_bound_checks: wgt::ShaderBoundChecks,
    ) -> Self::ShaderModuleId {
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
                let parser = naga::front::spv::Parser::new(spv.iter().cloned(), &options);
                let module = parser.parse().unwrap();
                wgc::pipeline::ShaderModuleSource::Naga(module)
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
                let mut parser = naga::front::glsl::Parser::default();
                let module = parser.parse(&options, shader).unwrap();

                wgc::pipeline::ShaderModuleSource::Naga(module)
            }
            ShaderSource::Wgsl(ref code) => wgc::pipeline::ShaderModuleSource::Wgsl(Borrowed(code)),
        };
        let (id, error) = wgc::gfx_select!(
            device.id => global.device_create_shader_module(device.id, &descriptor, source, PhantomData)
        );
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_shader_module",
            );
        }
        id
    }

    unsafe fn device_create_shader_module_spirv(
        &self,
        device: &Self::DeviceId,
        desc: &ShaderModuleDescriptorSpirV,
    ) -> Self::ShaderModuleId {
        let global = &self.0;
        let descriptor = wgc::pipeline::ShaderModuleDescriptor {
            label: desc.label.map(Borrowed),
            // Doesn't matter the value since spirv shaders aren't mutated to include
            // runtime checks
            shader_bound_checks: wgt::ShaderBoundChecks::unchecked(),
        };
        let (id, error) = wgc::gfx_select!(
            device.id => global.device_create_shader_module_spirv(device.id, &descriptor, Borrowed(&desc.source), PhantomData)
        );
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_shader_module_spirv",
            );
        }
        id
    }

    fn device_create_bind_group_layout(
        &self,
        device: &Self::DeviceId,
        desc: &BindGroupLayoutDescriptor,
    ) -> Self::BindGroupLayoutId {
        let global = &self.0;
        let descriptor = wgc::binding_model::BindGroupLayoutDescriptor {
            label: desc.label.map(Borrowed),
            entries: Borrowed(desc.entries),
        };
        let (id, error) = wgc::gfx_select!(
            device.id => global.device_create_bind_group_layout(device.id, &descriptor, PhantomData)
        );
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_bind_group_layout",
            );
        }
        id
    }

    fn device_create_bind_group(
        &self,
        device: &Self::DeviceId,
        desc: &BindGroupDescriptor,
    ) -> Self::BindGroupId {
        use wgc::binding_model as bm;

        let mut arrayed_texture_views = Vec::new();
        let mut arrayed_samplers = Vec::new();
        if device.features.contains(Features::TEXTURE_BINDING_ARRAY) {
            // gather all the array view IDs first
            for entry in desc.entries.iter() {
                if let BindingResource::TextureViewArray(array) = entry.resource {
                    arrayed_texture_views.extend(array.iter().map(|view| view.id));
                }
                if let BindingResource::SamplerArray(array) = entry.resource {
                    arrayed_samplers.extend(array.iter().map(|sampler| sampler.id));
                }
            }
        }
        let mut remaining_arrayed_texture_views = &arrayed_texture_views[..];
        let mut remaining_arrayed_samplers = &arrayed_samplers[..];

        let mut arrayed_buffer_bindings = Vec::new();
        if device.features.contains(Features::BUFFER_BINDING_ARRAY) {
            // gather all the buffers first
            for entry in desc.entries.iter() {
                if let BindingResource::BufferArray(array) = entry.resource {
                    arrayed_buffer_bindings.extend(array.iter().map(|binding| bm::BufferBinding {
                        buffer_id: binding.buffer.id.id,
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
                        buffer_id: buffer.id.id,
                        offset,
                        size,
                    }),
                    BindingResource::BufferArray(array) => {
                        let slice = &remaining_arrayed_buffer_bindings[..array.len()];
                        remaining_arrayed_buffer_bindings =
                            &remaining_arrayed_buffer_bindings[array.len()..];
                        bm::BindingResource::BufferArray(Borrowed(slice))
                    }
                    BindingResource::Sampler(sampler) => bm::BindingResource::Sampler(sampler.id),
                    BindingResource::SamplerArray(array) => {
                        let slice = &remaining_arrayed_samplers[..array.len()];
                        remaining_arrayed_samplers = &remaining_arrayed_samplers[array.len()..];
                        bm::BindingResource::SamplerArray(Borrowed(slice))
                    }
                    BindingResource::TextureView(texture_view) => {
                        bm::BindingResource::TextureView(texture_view.id)
                    }
                    BindingResource::TextureViewArray(array) => {
                        let slice = &remaining_arrayed_texture_views[..array.len()];
                        remaining_arrayed_texture_views =
                            &remaining_arrayed_texture_views[array.len()..];
                        bm::BindingResource::TextureViewArray(Borrowed(slice))
                    }
                },
            })
            .collect::<Vec<_>>();
        let descriptor = bm::BindGroupDescriptor {
            label: desc.label.as_ref().map(|label| Borrowed(&label[..])),
            layout: desc.layout.id,
            entries: Borrowed(&entries),
        };

        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device.id => global.device_create_bind_group(
            device.id,
            &descriptor,
            PhantomData
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_bind_group",
            );
        }
        id
    }

    fn device_create_pipeline_layout(
        &self,
        device: &Self::DeviceId,
        desc: &PipelineLayoutDescriptor,
    ) -> Self::PipelineLayoutId {
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
            .map(|bgl| bgl.id)
            .collect::<ArrayVec<_, { wgc::MAX_BIND_GROUPS }>>();
        let descriptor = wgc::binding_model::PipelineLayoutDescriptor {
            label: desc.label.map(Borrowed),
            bind_group_layouts: Borrowed(&temp_layouts),
            push_constant_ranges: Borrowed(desc.push_constant_ranges),
        };

        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device.id => global.device_create_pipeline_layout(
            device.id,
            &descriptor,
            PhantomData
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_pipeline_layout",
            );
        }
        id
    }

    fn device_create_render_pipeline(
        &self,
        device: &Self::DeviceId,
        desc: &RenderPipelineDescriptor,
    ) -> Self::RenderPipelineId {
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
                root_id: PhantomData,
                group_ids: &[PhantomData; wgc::MAX_BIND_GROUPS],
            }),
        };
        let descriptor = pipe::RenderPipelineDescriptor {
            label: desc.label.map(Borrowed),
            layout: desc.layout.map(|l| l.id),
            vertex: pipe::VertexState {
                stage: pipe::ProgrammableStageDescriptor {
                    module: desc.vertex.module.id,
                    entry_point: Borrowed(desc.vertex.entry_point),
                },
                buffers: Borrowed(&vertex_buffers),
            },
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment: desc.fragment.as_ref().map(|frag| pipe::FragmentState {
                stage: pipe::ProgrammableStageDescriptor {
                    module: frag.module.id,
                    entry_point: Borrowed(frag.entry_point),
                },
                targets: Borrowed(frag.targets),
            }),
            multiview: desc.multiview,
        };

        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device.id => global.device_create_render_pipeline(
            device.id,
            &descriptor,
            PhantomData,
            implicit_pipeline_ids
        ));
        if let Some(cause) = error {
            if let wgc::pipeline::CreateRenderPipelineError::Internal { stage, ref error } = cause {
                log::error!("Shader translation error for stage {:?}: {}", stage, error);
                log::error!("Please report it to https://github.com/gfx-rs/naga");
            }
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_render_pipeline",
            );
        }
        id
    }

    fn device_create_compute_pipeline(
        &self,
        device: &Self::DeviceId,
        desc: &ComputePipelineDescriptor,
    ) -> Self::ComputePipelineId {
        use wgc::pipeline as pipe;

        let implicit_pipeline_ids = match desc.layout {
            Some(_) => None,
            None => Some(wgc::device::ImplicitPipelineIds {
                root_id: PhantomData,
                group_ids: &[PhantomData; wgc::MAX_BIND_GROUPS],
            }),
        };
        let descriptor = pipe::ComputePipelineDescriptor {
            label: desc.label.map(Borrowed),
            layout: desc.layout.map(|l| l.id),
            stage: pipe::ProgrammableStageDescriptor {
                module: desc.module.id,
                entry_point: Borrowed(desc.entry_point),
            },
        };

        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device.id => global.device_create_compute_pipeline(
            device.id,
            &descriptor,
            PhantomData,
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
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_compute_pipeline",
            );
        }
        id
    }

    fn device_create_buffer(
        &self,
        device: &Self::DeviceId,
        desc: &crate::BufferDescriptor<'_>,
    ) -> Self::BufferId {
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device.id => global.device_create_buffer(
            device.id,
            &desc.map_label(|l| l.map(Borrowed)),
            PhantomData
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_buffer",
            );
        }
        Buffer {
            id,
            error_sink: Arc::clone(&device.error_sink),
        }
    }

    fn device_create_texture(
        &self,
        device: &Self::DeviceId,
        desc: &TextureDescriptor,
    ) -> Self::TextureId {
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device.id => global.device_create_texture(
            device.id,
            &desc.map_label(|l| l.map(Borrowed)),
            PhantomData
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_texture",
            );
        }
        Texture {
            id,
            error_sink: Arc::clone(&device.error_sink),
        }
    }

    fn device_create_sampler(
        &self,
        device: &Self::DeviceId,
        desc: &SamplerDescriptor,
    ) -> Self::SamplerId {
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
        let (id, error) = wgc::gfx_select!(device.id => global.device_create_sampler(
            device.id,
            &descriptor,
            PhantomData
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_sampler",
            );
        }
        id
    }

    fn device_create_query_set(
        &self,
        device: &Self::DeviceId,
        desc: &wgt::QuerySetDescriptor<Label>,
    ) -> Self::QuerySetId {
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device.id => global.device_create_query_set(
            device.id,
            &desc.map_label(|l| l.map(Borrowed)),
            PhantomData
        ));
        if let Some(cause) = error {
            self.handle_error_nolabel(&device.error_sink, cause, "Device::create_query_set");
        }
        id
    }

    fn device_create_command_encoder(
        &self,
        device: &Self::DeviceId,
        desc: &CommandEncoderDescriptor,
    ) -> Self::CommandEncoderId {
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(device.id => global.device_create_command_encoder(
            device.id,
            &desc.map_label(|l| l.map(Borrowed)),
            PhantomData
        ));
        if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_command_encoder",
            );
        }
        CommandEncoder {
            id,
            error_sink: Arc::clone(&device.error_sink),
            open: true,
        }
    }

    fn device_create_render_bundle_encoder(
        &self,
        device: &Self::DeviceId,
        desc: &RenderBundleEncoderDescriptor,
    ) -> Self::RenderBundleEncoderId {
        let descriptor = wgc::command::RenderBundleEncoderDescriptor {
            label: desc.label.map(Borrowed),
            color_formats: Borrowed(desc.color_formats),
            depth_stencil: desc.depth_stencil,
            sample_count: desc.sample_count,
            multiview: desc.multiview,
        };
        match wgc::command::RenderBundleEncoder::new(&descriptor, device.id, None) {
            Ok(id) => id,
            Err(e) => panic!("Error in Device::create_render_bundle_encoder: {}", e),
        }
    }

    #[cfg_attr(target_arch = "wasm32", allow(unused))]
    fn device_drop(&self, device: &Self::DeviceId) {
        let global = &self.0;

        #[cfg(any(not(target_arch = "wasm32"), feature = "emscripten"))]
        {
            match wgc::gfx_select!(device.id => global.device_poll(device.id, wgt::Maintain::Wait))
            {
                Ok(_) => (),
                Err(err) => self.handle_error_fatal(err, "Device::drop"),
            }
        }

        wgc::gfx_select!(device.id => global.device_drop(device.id));
    }

    fn device_poll(&self, device: &Self::DeviceId, maintain: crate::Maintain) -> bool {
        let global = &self.0;
        let maintain_inner = maintain.map_index(|i| i.0);
        match wgc::gfx_select!(device.id => global.device_poll(
            device.id,
            maintain_inner
        )) {
            Ok(queue_empty) => queue_empty,
            Err(err) => self.handle_error_fatal(err, "Device::poll"),
        }
    }

    fn device_on_uncaptured_error(
        &self,
        device: &Self::DeviceId,
        handler: impl crate::UncapturedErrorHandler,
    ) {
        let mut error_sink = device.error_sink.lock();
        error_sink.uncaptured_handler = Box::new(handler);
    }

    fn device_push_error_scope(&self, device: &Self::DeviceId, filter: crate::ErrorFilter) {
        let mut error_sink = device.error_sink.lock();
        error_sink.scopes.push(ErrorScope {
            error: None,
            filter,
        });
    }

    fn device_pop_error_scope(&self, device: &Self::DeviceId) -> Self::PopErrorScopeFuture {
        let mut error_sink = device.error_sink.lock();
        let scope = error_sink.scopes.pop().unwrap();
        ready(scope.error)
    }

    fn buffer_map_async<F>(
        &self,
        buffer: &Self::BufferId,
        mode: MapMode,
        range: Range<wgt::BufferAddress>,
        callback: F,
    ) where
        F: FnOnce(Result<(), crate::BufferAsyncError>) + Send + 'static,
    {
        let operation = wgc::resource::BufferMapOperation {
            host: match mode {
                MapMode::Read => wgc::device::HostMap::Read,
                MapMode::Write => wgc::device::HostMap::Write,
            },
            callback: wgc::resource::BufferMapCallback::from_rust(Box::new(|status| {
                let res = match status {
                    wgc::resource::BufferMapAsyncStatus::Success => Ok(()),
                    _ => Err(crate::BufferAsyncError),
                };
                callback(res);
            })),
        };

        let global = &self.0;
        match wgc::gfx_select!(buffer.id => global.buffer_map_async(buffer.id, range, operation)) {
            Ok(()) => (),
            Err(cause) => self.handle_error_nolabel(&buffer.error_sink, cause, "Buffer::map_async"),
        }
    }

    fn buffer_get_mapped_range(
        &self,
        buffer: &Self::BufferId,
        sub_range: Range<wgt::BufferAddress>,
    ) -> BufferMappedRange {
        let size = sub_range.end - sub_range.start;
        let global = &self.0;
        match wgc::gfx_select!(buffer.id => global.buffer_get_mapped_range(
            buffer.id,
            sub_range.start,
            Some(size)
        )) {
            Ok((ptr, size)) => BufferMappedRange {
                ptr,
                size: size as usize,
            },
            Err(err) => self.handle_error_fatal(err, "Buffer::get_mapped_range"),
        }
    }

    fn buffer_unmap(&self, buffer: &Self::BufferId) {
        let global = &self.0;
        match wgc::gfx_select!(buffer.id => global.buffer_unmap(buffer.id)) {
            Ok(()) => (),
            Err(cause) => {
                self.handle_error_nolabel(&buffer.error_sink, cause, "Buffer::buffer_unmap")
            }
        }
    }

    fn texture_create_view(
        &self,
        texture: &Self::TextureId,
        desc: &TextureViewDescriptor,
    ) -> Self::TextureViewId {
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
            texture.id => global.texture_create_view(texture.id, &descriptor, PhantomData)
        );
        if let Some(cause) = error {
            self.handle_error(
                &texture.error_sink,
                cause,
                LABEL,
                desc.label,
                "Texture::create_view",
            );
        }
        id
    }

    fn surface_drop(&self, _surface: &Self::SurfaceId) {
        //TODO: swapchain needs to hold the surface alive
        //self.0.surface_drop(*surface)
    }

    fn adapter_drop(&self, adapter: &Self::AdapterId) {
        let global = &self.0;
        wgc::gfx_select!(*adapter => global.adapter_drop(*adapter))
    }

    fn buffer_destroy(&self, buffer: &Self::BufferId) {
        let global = &self.0;
        match wgc::gfx_select!(buffer.id => global.buffer_destroy(buffer.id)) {
            Ok(()) => (),
            Err(err) => self.handle_error_fatal(err, "Buffer::destroy"),
        }
    }
    fn buffer_drop(&self, buffer: &Self::BufferId) {
        let global = &self.0;
        wgc::gfx_select!(buffer.id => global.buffer_drop(buffer.id, false))
    }
    fn texture_destroy(&self, texture: &Self::TextureId) {
        let global = &self.0;
        match wgc::gfx_select!(texture.id => global.texture_destroy(texture.id)) {
            Ok(()) => (),
            Err(err) => self.handle_error_fatal(err, "Texture::destroy"),
        }
    }
    fn texture_drop(&self, texture: &Self::TextureId) {
        let global = &self.0;
        wgc::gfx_select!(texture.id => global.texture_drop(texture.id, false))
    }
    fn texture_view_drop(&self, texture_view: &Self::TextureViewId) {
        let global = &self.0;
        match wgc::gfx_select!(*texture_view => global.texture_view_drop(*texture_view, false)) {
            Ok(()) => (),
            Err(err) => self.handle_error_fatal(err, "TextureView::drop"),
        }
    }
    fn sampler_drop(&self, sampler: &Self::SamplerId) {
        let global = &self.0;
        wgc::gfx_select!(*sampler => global.sampler_drop(*sampler))
    }
    fn query_set_drop(&self, query_set: &Self::QuerySetId) {
        let global = &self.0;
        wgc::gfx_select!(*query_set => global.query_set_drop(*query_set))
    }
    fn bind_group_drop(&self, bind_group: &Self::BindGroupId) {
        let global = &self.0;
        wgc::gfx_select!(*bind_group => global.bind_group_drop(*bind_group))
    }
    fn bind_group_layout_drop(&self, bind_group_layout: &Self::BindGroupLayoutId) {
        let global = &self.0;
        wgc::gfx_select!(*bind_group_layout => global.bind_group_layout_drop(*bind_group_layout))
    }
    fn pipeline_layout_drop(&self, pipeline_layout: &Self::PipelineLayoutId) {
        let global = &self.0;
        wgc::gfx_select!(*pipeline_layout => global.pipeline_layout_drop(*pipeline_layout))
    }
    fn shader_module_drop(&self, shader_module: &Self::ShaderModuleId) {
        let global = &self.0;
        wgc::gfx_select!(*shader_module => global.shader_module_drop(*shader_module))
    }
    fn command_encoder_drop(&self, command_encoder: &Self::CommandEncoderId) {
        if command_encoder.open {
            let global = &self.0;
            wgc::gfx_select!(command_encoder.id => global.command_encoder_drop(command_encoder.id))
        }
    }
    fn command_buffer_drop(&self, command_buffer: &Self::CommandBufferId) {
        let global = &self.0;
        wgc::gfx_select!(*command_buffer => global.command_buffer_drop(*command_buffer))
    }
    fn render_bundle_drop(&self, render_bundle: &Self::RenderBundleId) {
        let global = &self.0;
        wgc::gfx_select!(*render_bundle => global.render_bundle_drop(*render_bundle))
    }
    fn compute_pipeline_drop(&self, pipeline: &Self::ComputePipelineId) {
        let global = &self.0;
        wgc::gfx_select!(*pipeline => global.compute_pipeline_drop(*pipeline))
    }
    fn render_pipeline_drop(&self, pipeline: &Self::RenderPipelineId) {
        let global = &self.0;
        wgc::gfx_select!(*pipeline => global.render_pipeline_drop(*pipeline))
    }

    fn compute_pipeline_get_bind_group_layout(
        &self,
        pipeline: &Self::ComputePipelineId,
        index: u32,
    ) -> Self::BindGroupLayoutId {
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(*pipeline => global.compute_pipeline_get_bind_group_layout(*pipeline, index, PhantomData));
        if let Some(err) = error {
            panic!("Error reflecting bind group {}: {}", index, err);
        }
        id
    }
    fn render_pipeline_get_bind_group_layout(
        &self,
        pipeline: &Self::RenderPipelineId,
        index: u32,
    ) -> Self::BindGroupLayoutId {
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(*pipeline => global.render_pipeline_get_bind_group_layout(*pipeline, index, PhantomData));
        if let Some(err) = error {
            panic!("Error reflecting bind group {}: {}", index, err);
        }
        id
    }

    fn command_encoder_copy_buffer_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        source: &Self::BufferId,
        source_offset: wgt::BufferAddress,
        destination: &Self::BufferId,
        destination_offset: wgt::BufferAddress,
        copy_size: wgt::BufferAddress,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_copy_buffer_to_buffer(
            encoder.id,
            source.id,
            source_offset,
            destination.id,
            destination_offset,
            copy_size
        )) {
            self.handle_error_nolabel(
                &encoder.error_sink,
                cause,
                "CommandEncoder::copy_buffer_to_buffer",
            );
        }
    }

    fn command_encoder_copy_buffer_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        source: crate::ImageCopyBuffer,
        destination: crate::ImageCopyTexture,
        copy_size: wgt::Extent3d,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_copy_buffer_to_texture(
            encoder.id,
            &map_buffer_copy_view(source),
            &map_texture_copy_view(destination),
            &copy_size
        )) {
            self.handle_error_nolabel(
                &encoder.error_sink,
                cause,
                "CommandEncoder::copy_buffer_to_texture",
            );
        }
    }

    fn command_encoder_copy_texture_to_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        source: crate::ImageCopyTexture,
        destination: crate::ImageCopyBuffer,
        copy_size: wgt::Extent3d,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_copy_texture_to_buffer(
            encoder.id,
            &map_texture_copy_view(source),
            &map_buffer_copy_view(destination),
            &copy_size
        )) {
            self.handle_error_nolabel(
                &encoder.error_sink,
                cause,
                "CommandEncoder::copy_texture_to_buffer",
            );
        }
    }

    fn command_encoder_copy_texture_to_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        source: crate::ImageCopyTexture,
        destination: crate::ImageCopyTexture,
        copy_size: wgt::Extent3d,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_copy_texture_to_texture(
            encoder.id,
            &map_texture_copy_view(source),
            &map_texture_copy_view(destination),
            &copy_size
        )) {
            self.handle_error_nolabel(
                &encoder.error_sink,
                cause,
                "CommandEncoder::copy_texture_to_texture",
            );
        }
    }

    fn command_encoder_write_timestamp(
        &self,
        encoder: &Self::CommandEncoderId,
        query_set: &Self::QuerySetId,
        query_index: u32,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_write_timestamp(
            encoder.id,
            *query_set,
            query_index
        )) {
            self.handle_error_nolabel(
                &encoder.error_sink,
                cause,
                "CommandEncoder::write_timestamp",
            );
        }
    }

    fn command_encoder_resolve_query_set(
        &self,
        encoder: &Self::CommandEncoderId,
        query_set: &Self::QuerySetId,
        first_query: u32,
        query_count: u32,
        destination: &Self::BufferId,
        destination_offset: wgt::BufferAddress,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_resolve_query_set(
            encoder.id,
            *query_set,
            first_query,
            query_count,
            destination.id,
            destination_offset
        )) {
            self.handle_error_nolabel(
                &encoder.error_sink,
                cause,
                "CommandEncoder::resolve_query_set",
            );
        }
    }

    fn command_encoder_begin_compute_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        desc: &ComputePassDescriptor,
    ) -> Self::ComputePassId {
        wgc::command::ComputePass::new(
            encoder.id,
            &wgc::command::ComputePassDescriptor {
                label: desc.label.map(Borrowed),
            },
        )
    }

    fn command_encoder_end_compute_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        pass: &mut Self::ComputePassId,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(
            encoder.id => global.command_encoder_run_compute_pass(encoder.id, pass)
        ) {
            let name = wgc::gfx_select!(encoder.id => global.command_buffer_label(encoder.id));
            self.handle_error(
                &encoder.error_sink,
                cause,
                "encoder",
                Some(&name),
                "a ComputePass",
            );
        }
    }

    fn command_encoder_begin_render_pass<'a>(
        &self,
        encoder: &Self::CommandEncoderId,
        desc: &crate::RenderPassDescriptor<'a, '_>,
    ) -> Self::RenderPassId {
        let colors = desc
            .color_attachments
            .iter()
            .map(|ca| wgc::command::RenderPassColorAttachment {
                view: ca.view.id,
                resolve_target: ca.resolve_target.map(|rt| rt.id),
                channel: map_pass_channel(Some(&ca.ops)),
            })
            .collect::<ArrayVec<_, { wgc::MAX_COLOR_ATTACHMENTS }>>();

        let depth_stencil = desc.depth_stencil_attachment.as_ref().map(|dsa| {
            wgc::command::RenderPassDepthStencilAttachment {
                view: dsa.view.id,
                depth: map_pass_channel(dsa.depth_ops.as_ref()),
                stencil: map_pass_channel(dsa.stencil_ops.as_ref()),
            }
        });

        wgc::command::RenderPass::new(
            encoder.id,
            &wgc::command::RenderPassDescriptor {
                label: desc.label.map(Borrowed),
                color_attachments: Borrowed(&colors),
                depth_stencil_attachment: depth_stencil.as_ref(),
            },
        )
    }

    fn command_encoder_end_render_pass(
        &self,
        encoder: &Self::CommandEncoderId,
        pass: &mut Self::RenderPassId,
    ) {
        let global = &self.0;
        if let Err(cause) =
            wgc::gfx_select!(encoder.id => global.command_encoder_run_render_pass(encoder.id, pass))
        {
            let name = wgc::gfx_select!(encoder.id => global.command_buffer_label(encoder.id));
            self.handle_error(
                &encoder.error_sink,
                cause,
                "encoder",
                Some(&name),
                "a RenderPass",
            );
        }
    }

    fn command_encoder_finish(&self, mut encoder: Self::CommandEncoderId) -> Self::CommandBufferId {
        let descriptor = wgt::CommandBufferDescriptor::default();
        encoder.open = false; // prevent the drop
        let global = &self.0;
        let (id, error) =
            wgc::gfx_select!(encoder.id => global.command_encoder_finish(encoder.id, &descriptor));
        if let Some(cause) = error {
            self.handle_error_nolabel(&encoder.error_sink, cause, "a CommandEncoder");
        }
        id
    }

    fn command_encoder_clear_texture(
        &self,
        encoder: &Self::CommandEncoderId,
        texture: &crate::Texture,
        subresource_range: &wgt::ImageSubresourceRange,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_clear_texture(
            encoder.id,
            texture.id.id,
            subresource_range
        )) {
            self.handle_error_nolabel(&encoder.error_sink, cause, "CommandEncoder::clear_texture");
        }
    }

    fn command_encoder_clear_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        buffer: &crate::Buffer,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_clear_buffer(
            encoder.id,
            buffer.id.id,
            offset, size
        )) {
            self.handle_error_nolabel(&encoder.error_sink, cause, "CommandEncoder::fill_buffer");
        }
    }

    fn command_encoder_insert_debug_marker(&self, encoder: &Self::CommandEncoderId, label: &str) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_insert_debug_marker(encoder.id, label))
        {
            self.handle_error_nolabel(
                &encoder.error_sink,
                cause,
                "CommandEncoder::insert_debug_marker",
            );
        }
    }
    fn command_encoder_push_debug_group(&self, encoder: &Self::CommandEncoderId, label: &str) {
        let global = &self.0;
        if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_push_debug_group(encoder.id, label))
        {
            self.handle_error_nolabel(
                &encoder.error_sink,
                cause,
                "CommandEncoder::push_debug_group",
            );
        }
    }
    fn command_encoder_pop_debug_group(&self, encoder: &Self::CommandEncoderId) {
        let global = &self.0;
        if let Err(cause) =
            wgc::gfx_select!(encoder.id => global.command_encoder_pop_debug_group(encoder.id))
        {
            self.handle_error_nolabel(
                &encoder.error_sink,
                cause,
                "CommandEncoder::pop_debug_group",
            );
        }
    }

    fn render_bundle_encoder_finish(
        &self,
        encoder: Self::RenderBundleEncoderId,
        desc: &crate::RenderBundleDescriptor,
    ) -> Self::RenderBundleId {
        let global = &self.0;
        let (id, error) = wgc::gfx_select!(encoder.parent() => global.render_bundle_encoder_finish(
            encoder,
            &desc.map_label(|l| l.map(Borrowed)),
            PhantomData
        ));
        if let Some(err) = error {
            self.handle_error_fatal(err, "RenderBundleEncoder::finish");
        }
        id
    }

    fn queue_write_buffer(
        &self,
        queue: &Self::QueueId,
        buffer: &Self::BufferId,
        offset: wgt::BufferAddress,
        data: &[u8],
    ) {
        let global = &self.0;
        match wgc::gfx_select!(
            *queue => global.queue_write_buffer(*queue, buffer.id, offset, data)
        ) {
            Ok(()) => (),
            Err(err) => self.handle_error_fatal(err, "Queue::write_buffer"),
        }
    }

    fn queue_write_texture(
        &self,
        queue: &Self::QueueId,
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
            Err(err) => self.handle_error_fatal(err, "Queue::write_texture"),
        }
    }

    fn queue_submit<I: Iterator<Item = Self::CommandBufferId>>(
        &self,
        queue: &Self::QueueId,
        command_buffers: I,
    ) -> Self::SubmissionIndex {
        let temp_command_buffers = command_buffers.collect::<SmallVec<[_; 4]>>();

        let global = &self.0;
        match wgc::gfx_select!(*queue => global.queue_submit(*queue, &temp_command_buffers)) {
            Ok(index) => index,
            Err(err) => self.handle_error_fatal(err, "Queue::submit"),
        }
    }

    fn queue_get_timestamp_period(&self, queue: &Self::QueueId) -> f32 {
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
        callback: Box<dyn FnOnce() + Send + 'static>,
    ) {
        let closure = wgc::device::queue::SubmittedWorkDoneClosure::from_rust(callback);

        let global = &self.0;
        let res = wgc::gfx_select!(queue => global.queue_on_submitted_work_done(*queue, closure));
        if let Err(cause) = res {
            self.handle_error_fatal(cause, "Queue::on_submitted_work_done");
        }
    }

    fn device_start_capture(&self, device: &Self::DeviceId) {
        let global = &self.0;
        wgc::gfx_select!(device.id => global.device_start_capture(device.id));
    }

    fn device_stop_capture(&self, device: &Self::DeviceId) {
        let global = &self.0;
        wgc::gfx_select!(device.id => global.device_stop_capture(device.id));
    }
}

#[derive(Debug)]
pub(crate) struct SurfaceOutputDetail {
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
    panic!("wgpu error: {}\n", err);
}

#[derive(Debug)]
pub struct BufferMappedRange {
    ptr: *mut u8,
    size: usize,
}

unsafe impl Send for BufferMappedRange {}
unsafe impl Sync for BufferMappedRange {}

impl crate::BufferMappedRangeSlice for BufferMappedRange {
    fn slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.size) }
    }

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
