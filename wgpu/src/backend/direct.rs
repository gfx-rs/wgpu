use crate::{
    backend::native_gpu_future, AdapterInfo, BindGroupDescriptor, BindGroupLayoutDescriptor,
    BindingResource, BufferBinding, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, DownlevelCapabilities, Features, Label, Limits, LoadOp, MapMode,
    Operations, PipelineLayoutDescriptor, RenderBundleEncoderDescriptor, RenderPipelineDescriptor,
    SamplerDescriptor, ShaderModuleDescriptor, ShaderModuleDescriptorSpirV, ShaderSource,
    SurfaceStatus, TextureDescriptor, TextureFormat, TextureViewDescriptor,
};

use arrayvec::ArrayVec;
use core::hash;
use hashbrown::{hash_map, HashMap};
use parking_lot::Mutex;
// use smallvec::SmallVec;
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

pub type Global = wgc::hub::Global<wgc::hub::IdentityManagerFactory>;

/// Wrapper over BindGroupLayoutId used for storing in the deduplication HashSet; it performs
/// hashing on the underlying list of [wgt::BindGroupLayoutEntry] s.
///
/// Note that for deduplication to work properly, the bind group layout entries must have unique
/// slot orders and be sorted in increasing binding index.
///
/// Does not perform label comparison for either equality or hashing; per discussion with @kvark
/// we are okay with BGLs returned from create_bind_group_layout using the label of "some
/// compatible BGL."
///
/// More subtly, the hash function does not include the device id; this is okay because we use the
/// raw hasher API, and therefore can use a custom equality comparison to make sure that we don't
/// return duplicates for identical entries on different devices.
struct BindGroupLayoutEntry(wgc::id::BindGroupLayoutId);

/* /// NOTE: Ignores device id for compatibility with the BGL hash.
impl hash::Hash for BindGroupLayoutEntry {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        let bgl = &self.0;
        wgc::gfx_select2!(&Arc bgl => bgl.entries().hash(state))
    }
} */

pub struct Context {
    global: Global,
    /// Bind group layouts set, used for deduplication.
    bind_group_layouts: Mutex<HashMap<BindGroupLayoutEntry, (), ()>>,
    /// Hasher state, duplicated outside the bind group layout set so we can hash without locking.
    hasher: hash::BuildHasherDefault<fxhash::FxHasher>,
}

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
        Self {
            global: wgc::hub::Global::from_hal_instance::<A>(
                "wgpu",
                wgc::hub::IdentityManagerFactory,
                hal_instance,
            ),
            // Since we never directly use the included hasher or Hash instance,
            // using the unit hasher () is fine.
            bind_group_layouts: Mutex::new(HashMap::with_hasher(())),
            // Instead, we cache the hasher outside the HashMap, allowing us to access it without
            // taking the HashMap lock.
            hasher: hash::BuildHasherDefault::default(),
        }
    }

    pub(crate) fn global(&self) -> &Global {
        &self.global
    }

    /// Clear dead BGLs out of the HashMap.
    fn clear_bind_group_layouts(&self) {
        // NOTE: In addition to memory safety, what we gain from taking this lock is the knowledge
        // that bind group layouts with count=1 are not accessible from other threads (since
        // getting at the reference requires taking the lock).  This allows us to safely remove
        // these entries without dealing with concurrent access, unlike "proper" weak pointers.
        self.bind_group_layouts.lock().retain(|key, _| !key.0.is_unique());
    }

    fn deduplicate_bind_group_layout<A: wgc::hub::HalApi>(
        &self,
        device_id: wgc::id::IdGuard<A, wgc::device::Device<wgc::id::Dummy>>,
        desc: BindGroupLayoutDescriptor,
    ) -> Result<wgc::id::BindGroupLayoutId, wgc::binding_model::CreateBindGroupLayoutError> {
        let hasher = &self.hasher;
        let make_hash = move |val: &[wgt::BindGroupLayoutEntry]| {
            use hash::{BuildHasher, Hash, Hasher};
            let mut state = hasher.build_hasher();
            val.hash(&mut state);
            state.finish()
        };
        let entries = desc.entries;
        // Sort entries by key to make sure we hash the same for equivalent bind group layouts
        // (note that we do *not* try to deduplicate here, since if there are duplicates it's an
        // error that will be caught locally).
        entries.sort_unstable_by_key(|entry| entry.binding);
        let hash = make_hash(entries);
        let desc = wgc::binding_model::BindGroupLayoutDescriptor {
            label: desc.label.map(Borrowed),
            entries: Borrowed(entries),
        };
        // NOTE: Theoretically, we could use an RwLock instead, and always start with read,
        // upgrading to a write lock only if the entry was not found; however, this is probably not
        // worthwhile since the only cases during which we'd be holding this lock for more than a
        // small amount of time are during hashmap resizes, which would occur with a write lock
        // taken anyway.  Therefore, if the blocking here is really a problem, we should probably
        // convert to a concurrent hash map (ideally one with concurrent resizing) or a different
        // data structure with better concurrency properties.
        let mut map_guard = self.bind_group_layouts.lock();
        let entry = map_guard.raw_entry_mut()
            .from_hash(hash, |other| {
                let other = &other.0;
                if other.backend() != A::VARIANT {
                    return false;
                }
                // NOTE: Must succeed due to backend equality check earlier.
                let other = wgc::id::expect_backend::<_, A>(other);
                device_id == other.device_id() && entries == other.entries()
            });
        Ok(match entry {
            hash_map::RawEntryMut::Occupied(ref occupied) => occupied.key(),
            hash_map::RawEntryMut::Vacant(vacant) => {
                // TODO: Consider caching hash directly with key to speed up resizes, if this
                // becomes a bottleneck.
                let key = BindGroupLayoutEntry(
                    Global::device_create_bind_group_layout(device_id.to_owned(), desc/*, PhantomData*/)?
                );
                vacant.insert_with_hasher(hash, key, (), |id| {
                    let bind_group_layout = &id.0;
                    wgc::gfx_select2!(&Arc bind_group_layout => make_hash((&*bind_group_layout).entries()))
                }).0
            },
        }.0.clone())
    }

    pub fn enumerate_adapters(&self, backends: wgt::Backends) -> Vec<wgc::id::AdapterId> {
        self.global
            .enumerate_adapters(/*wgc::instance::AdapterInputs::Mask(*/backends/*, |_| {
                PhantomData
            })*/)
    }

    pub unsafe fn create_adapter_from_hal<A: wgc::hub::HalApi>(
        hal_adapter: hal::ExposedAdapter<A>,
    ) -> wgc::id::AdapterId {
        Global::create_adapter_from_hal(hal_adapter)
    }

    pub unsafe fn create_device_from_hal<A: wgc::hub::HalApi>(
        &self,
        adapter: wgc::id::AdapterId,
        hal_device: hal::OpenDevice<A>,
        desc: &crate::DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Result<(Device, wgc::id::QueueId), crate::RequestDeviceError> {
        // FIXME: Check backend first.
        let adapter_id = wgc::id::expect_backend_box_owned(adapter);
        let device_id = match Global::create_device_from_hal::<A>(
            *adapter_id,
            hal_device,
            &desc.map_label(|l| l.map(Borrowed)),
            trace_dir
        )
        {
            Ok(device_id) => device_id,
            Err(err) => {
                // TODO: Consider not handling as fatal?
                self.handle_error_fatal(err, "Adapter::create_device_from_hal");
            },
        };
        let queue = device_id.clone();
        let device = Device {
            id: device_id,
            error_sink: Arc::new(Mutex::new(ErrorSinkRaw::new())),
            features: desc.features,
        };
        Ok((device, queue))
    }

    pub unsafe fn create_texture_from_hal<A: wgc::hub::HalApi>(
        &self,
        hal_texture: A::Texture,
        device: &Device,
        desc: &TextureDescriptor,
    ) -> Texture {
        let global = &self.global;
        // FIXME: Check backend first.
        let device_id = wgc::id::expect_backend_owned(device.id.clone());
        let (id, error) = global.create_texture_from_hal::<A>(
            hal_texture,
            device_id,
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

    pub fn generate_report(&self) -> wgc::hub::GlobalReport {
        self.global.generate_report()
    }

    /*TODO: raw surface
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub unsafe fn create_surface_from_core_animation_layer(
        self: &Arc<Self>,
        layer: *mut std::ffi::c_void,
    ) -> crate::Surface {
        let id = self.0.instance_create_surface_metal(layer, PhantomData);
        crate::Surface {
            context: Arc::clone(self),
            id,
        }
    }*/

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
        let sink = sink_mutex.lock();
        let mut source_opt: Option<&(dyn Error + 'static)> = Some(&error);
        while let Some(source) = source_opt {
            if let Some(wgc::device::DeviceError::OutOfMemory) =
                source.downcast_ref::<wgc::device::DeviceError>()
            {
                return sink.handle_error(crate::Error::OutOfMemoryError {
                    source: Box::new(error),
                });
            }
            source_opt = source.source();
        }

        // Otherwise, it is a validation error
        sink.handle_error(crate::Error::ValidationError {
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
    // use smallvec::SmallVec;
    // use std::convert::TryInto;
    use std::ops::Range;
    use wgc::command::{bundle_ffi::*, compute_ffi::*, render_ffi::*};

    impl<'a> crate::ComputePassInner<'a, Context> for wgc::command::ComputePass<'a> {
        fn set_pipeline(&mut self, pipeline: &'a <Context as crate::Context>::ComputePipelineId) {
            // FIXME: Make sure whole buffer becomes an error if the pipeline is an error.
            if let Some(pipeline) = pipeline {
                wgpu_compute_pass_set_pipeline(self, pipeline)
            }
        }
        fn set_bind_group(
            &mut self,
            index: u32,
            bind_group: &'a <Context as crate::Context>::BindGroupId,
            offsets: &[wgt::DynamicOffset],
        ) {
            // unsafe {
            // FIXME: Make sure whole buffer becomes an error if the bind group is an error.
            if let Some(bind_group) = bind_group {
                wgpu_compute_pass_set_bind_group(
                    self,
                    index,
                    bind_group,
                    /*offsets.as_ptr(),
                    offsets.len()*/offsets,
                )
            }
            // }
        }
        fn set_push_constants(&mut self, offset: u32, data: &[u8]) {
            // unsafe {
                wgpu_compute_pass_set_push_constant(
                    self,
                    offset,
                    /*data.len().try_into().unwrap(),
                    data.as_ptr()*/data,
                )
            // }
        }
        fn insert_debug_marker(&mut self, label: &str) {
            // unsafe {
                /* let label = std::ffi::CString::new(label).unwrap(); */
                wgpu_compute_pass_insert_debug_marker(self, label/*.as_ptr()*/, 0);
            // }
        }

        fn push_debug_group(&mut self, group_label: &str) {
            // unsafe {
                /* let label = std::ffi::CString::new(group_label).unwrap(); */
                wgpu_compute_pass_push_debug_group(self, group_label/*label.as_ptr()*/, 0);
            // }
        }
        fn pop_debug_group(&mut self) {
            wgpu_compute_pass_pop_debug_group(self);
        }

        fn write_timestamp(&mut self, query_set: &'a <Context as crate::Context>::QuerySetId, query_index: u32) {
            // FIXME: Make sure whole buffer becomes an error if the query set is an error.
            if let Some(query_set) = query_set {
                wgpu_compute_pass_write_timestamp(self, query_set, query_index)
            }
        }

        fn begin_pipeline_statistics_query(
            &mut self,
            query_set: &'a <Context as crate::Context>::QuerySetId,
            query_index: u32,
        ) {
            // FIXME: Make sure whole buffer becomes an error if the query set is an error.
            if let Some(query_set) = query_set {
                wgpu_compute_pass_begin_pipeline_statistics_query(self, query_set, query_index)
            }
        }

        fn end_pipeline_statistics_query(&mut self) {
            wgpu_compute_pass_end_pipeline_statistics_query(self)
        }

        fn dispatch(&mut self, x: u32, y: u32, z: u32) {
            wgpu_compute_pass_dispatch(self, x, y, z)
        }
        fn dispatch_indirect(
            &mut self,
            indirect_buffer: &'a super::Buffer,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_compute_pass_dispatch_indirect(self, &indirect_buffer.id, indirect_offset)
        }
    }

    impl<'a> crate::RenderInner<'a, Context> for wgc::command::RenderPass<'a> {
        fn set_pipeline(&mut self, pipeline: &'a <Context as crate::Context>::RenderPipelineId) {
            // FIXME: Make sure whole buffer becomes an error if the pipeline is an error.
            if let Some(pipeline) = pipeline {
                wgpu_render_pass_set_pipeline(self, pipeline)
            };
        }
        fn set_bind_group(
            &mut self,
            index: u32,
            bind_group: &'a <Context as crate::Context>::BindGroupId,
            offsets: &[wgt::DynamicOffset],
        ) {
            // unsafe {
            // FIXME: Make sure whole buffer becomes an error if the bind group is an error.
            if let Some(bind_group) = bind_group {
                wgpu_render_pass_set_bind_group(
                    self,
                    index,
                    bind_group,
                    /* offsets.as_ptr(),
                    offsets.len()*/offsets,
                )
            }
            // }
        }
        fn set_index_buffer(
            &mut self,
            buffer: &'a super::Buffer,
            index_format: wgt::IndexFormat,
            offset: wgt::BufferAddress,
            size: Option<wgt::BufferSize>,
        ) {
            self.set_index_buffer(&buffer.id, index_format, offset, size)
        }
        fn set_vertex_buffer(
            &mut self,
            slot: u32,
            buffer: &'a super::Buffer,
            offset: wgt::BufferAddress,
            size: Option<wgt::BufferSize>,
        ) {
            wgpu_render_pass_set_vertex_buffer(self, slot, &buffer.id, offset, size)
        }
        fn set_push_constants(&mut self, stages: wgt::ShaderStages, offset: u32, data: &[u8]) {
            // unsafe {
                wgpu_render_pass_set_push_constants(
                    self,
                    stages,
                    offset,
                    /*data.len().try_into().unwrap(),
                    data.as_ptr()*/data,
                )
            // }
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
            indirect_buffer: &'a super::Buffer,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_render_pass_draw_indirect(self, &indirect_buffer.id, indirect_offset)
        }
        fn draw_indexed_indirect(
            &mut self,
            indirect_buffer: &'a super::Buffer,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_render_pass_draw_indexed_indirect(self, &indirect_buffer.id, indirect_offset)
        }
        fn multi_draw_indirect(
            &mut self,
            indirect_buffer: &'a super::Buffer,
            indirect_offset: wgt::BufferAddress,
            count: u32,
        ) {
            wgpu_render_pass_multi_draw_indirect(self, &indirect_buffer.id, indirect_offset, count)
        }
        fn multi_draw_indexed_indirect(
            &mut self,
            indirect_buffer: &'a super::Buffer,
            indirect_offset: wgt::BufferAddress,
            count: u32,
        ) {
            wgpu_render_pass_multi_draw_indexed_indirect(
                self,
                &indirect_buffer.id,
                indirect_offset,
                count,
            )
        }
        fn multi_draw_indirect_count(
            &mut self,
            indirect_buffer: &'a super::Buffer,
            indirect_offset: wgt::BufferAddress,
            count_buffer: &'a super::Buffer,
            count_buffer_offset: wgt::BufferAddress,
            max_count: u32,
        ) {
            wgpu_render_pass_multi_draw_indirect_count(
                self,
                &indirect_buffer.id,
                indirect_offset,
                &count_buffer.id,
                count_buffer_offset,
                max_count,
            )
        }
        fn multi_draw_indexed_indirect_count(
            &mut self,
            indirect_buffer: &'a super::Buffer,
            indirect_offset: wgt::BufferAddress,
            count_buffer: &'a super::Buffer,
            count_buffer_offset: wgt::BufferAddress,
            max_count: u32,
        ) {
            wgpu_render_pass_multi_draw_indexed_indirect_count(
                self,
                &indirect_buffer.id,
                indirect_offset,
                &count_buffer.id,
                count_buffer_offset,
                max_count,
            )
        }
    }

    impl<'a> crate::RenderPassInner<'a, Context> for wgc::command::RenderPass<'a> {
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
            // unsafe {
                // let label = std::ffi::CString::new(label).unwrap();
                wgpu_render_pass_insert_debug_marker(self, label/*.as_ptr()*/, 0);
            // }
        }

        fn push_debug_group(&mut self, group_label: &str) {
            // unsafe {
                // let label = std::ffi::CString::new(group_label).unwrap();
                wgpu_render_pass_push_debug_group(self, group_label/*label.as_ptr()*/, 0);
            // }
        }

        fn pop_debug_group(&mut self) {
            wgpu_render_pass_pop_debug_group(self);
        }

        fn write_timestamp(&mut self, query_set: &'a <Context as crate::Context>::QuerySetId, query_index: u32) {
            // FIXME: Make sure whole buffer becomes an error if the query set is an error.
            if let Some(query_set) = query_set {
                wgpu_render_pass_write_timestamp(self, query_set, query_index)
            }
        }

        fn begin_pipeline_statistics_query(
            &mut self,
            query_set: &'a <Context as crate::Context>::QuerySetId,
            query_index: u32,
        ) {
            // FIXME: Make sure whole buffer becomes an error if the query set is an error.
            if let Some(query_set) = query_set {
                wgpu_render_pass_begin_pipeline_statistics_query(self, query_set, query_index)
            }
        }

        fn end_pipeline_statistics_query(&mut self) {
            wgpu_render_pass_end_pipeline_statistics_query(self)
        }

        fn execute_bundles<I: Iterator<Item = &'a wgc::id::RenderBundleId>>(
            &mut self,
            render_bundles: I,
        ) {
            // let temp_render_bundles = render_bundles.cloned().collect::<SmallVec<[_; 4]>>();
            // unsafe {
                wgpu_render_pass_execute_bundles(
                    self,
                    render_bundles/*temp_render_bundles*//*.as_ptr(),
                    temp_render_bundles.len()*/,
                )
            // }
        }
    }

    impl<'a> crate::RenderInner<'a, Context> for wgc::command::RenderBundleEncoder<'a> {
        fn set_pipeline(&mut self, pipeline: &'a <Context as crate::Context>::RenderPipelineId) {
            // FIXME: Make sure whole buffer becomes an error if the pipeline is an error.
            if let Some(pipeline) = pipeline {
                wgpu_render_bundle_set_pipeline(self, pipeline)
            };
        }
        fn set_bind_group(
            &mut self,
            index: u32,
            bind_group: &'a <Context as crate::Context>::BindGroupId,
            offsets: &[wgt::DynamicOffset],
        ) {
            // unsafe {
            // FIXME: Make sure whole buffer becomes an error if the bind group is an error.
            if let Some(bind_group) = bind_group {
                wgpu_render_bundle_set_bind_group(
                    self,
                    index,
                    bind_group,
                    offsets/*.as_ptr(),
                    offsets.len()*/,
                )
            }
            // }
        }
        fn set_index_buffer(
            &mut self,
            buffer: &'a super::Buffer,
            index_format: wgt::IndexFormat,
            offset: wgt::BufferAddress,
            size: Option<wgt::BufferSize>,
        ) {
            self.set_index_buffer(&buffer.id, index_format, offset, size)
        }
        fn set_vertex_buffer(
            &mut self,
            slot: u32,
            buffer: &'a super::Buffer,
            offset: wgt::BufferAddress,
            size: Option<wgt::BufferSize>,
        ) {
            wgpu_render_bundle_set_vertex_buffer(self, slot, &buffer.id, offset, size)
        }

        fn set_push_constants(&mut self, stages: wgt::ShaderStages, offset: u32, data: &[u8]) {
            //unsafe {
                wgpu_render_bundle_set_push_constants(
                    self,
                    stages,
                    offset,
                    data/*.len().try_into().unwrap(),
                    data.as_ptr()*/,
                )
            //}
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
            indirect_buffer: &'a super::Buffer,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_render_bundle_draw_indirect(self, &indirect_buffer.id, indirect_offset)
        }
        fn draw_indexed_indirect(
            &mut self,
            indirect_buffer: &'a super::Buffer,
            indirect_offset: wgt::BufferAddress,
        ) {
            wgpu_render_bundle_draw_indexed_indirect(self, &indirect_buffer.id, indirect_offset)
        }
        fn multi_draw_indirect(
            &mut self,
            _indirect_buffer: &'a super::Buffer,
            _indirect_offset: wgt::BufferAddress,
            _count: u32,
        ) {
            unimplemented!()
        }
        fn multi_draw_indexed_indirect(
            &mut self,
            _indirect_buffer: &'a super::Buffer,
            _indirect_offset: wgt::BufferAddress,
            _count: u32,
        ) {
            unimplemented!()
        }
        fn multi_draw_indirect_count(
            &mut self,
            _indirect_buffer: &'a super::Buffer,
            _indirect_offset: wgt::BufferAddress,
            _count_buffer: &'a super::Buffer,
            _count_buffer_offset: wgt::BufferAddress,
            _max_count: u32,
        ) {
            unimplemented!()
        }
        fn multi_draw_indexed_indirect_count(
            &mut self,
            _indirect_buffer: &'a super::Buffer,
            _indirect_offset: wgt::BufferAddress,
            _count_buffer: &'a super::Buffer,
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

impl<'a> crate::Context<'a> for Context {
    type AdapterId = wgc::id::AdapterId;
    type DeviceId = Device;
    type QueueId = wgc::id::QueueId;
    type ShaderModuleId = Option<wgc::id::ShaderModuleId>;
    type BindGroupLayoutId = Option<wgc::id::BindGroupLayoutId>;
    type BindGroupId = Option<wgc::id::BindGroupId>;
    type TextureViewId = wgc::id::TextureViewId;
    type SamplerId = Option<wgc::id::SamplerId>;
    type QuerySetId = Option<wgc::id::QuerySetId>;
    type BufferId = Buffer;
    type TextureId = Texture;
    type PipelineLayoutId = Option<wgc::id::PipelineLayoutId>;
    type RenderPipelineId = Option<wgc::id::RenderPipelineId>;
    type ComputePipelineId = Option<wgc::id::ComputePipelineId>;
    type CommandEncoderId = CommandEncoder;
    type ComputePassId = wgc::command::ComputePass<'a>;
    type RenderPassId = wgc::command::RenderPass<'a>;
    type CommandBufferId = wgc::id::CommandBufferId;
    type RenderBundleEncoderId = wgc::command::RenderBundleEncoder<'a>;
    type RenderBundleId = wgc::id::RenderBundleId;
    type SurfaceId = Surface;

    type SurfaceOutputDetail = SurfaceOutputDetail;

    type RequestAdapterFuture = Ready<Option<Self::AdapterId>>;
    #[allow(clippy::type_complexity)]
    type RequestDeviceFuture =
        Ready<Result<(Self::DeviceId, Self::QueueId), crate::RequestDeviceError>>;
    type MapAsyncFuture = native_gpu_future::GpuFuture<Result<(), crate::BufferAsyncError>>;
    type OnSubmittedWorkDoneFuture = native_gpu_future::GpuFuture<()>;

    fn init(backends: wgt::Backends) -> Self {
        Self {
            global: wgc::hub::Global::new(
                "wgpu",
                wgc::hub::IdentityManagerFactory,
                backends),
            // Since we never directly use the included hasher or Hash instance,
            // using the unit hasher () is fine.
            bind_group_layouts: Mutex::new(HashMap::with_hasher(())),
            // Instead, we cache the hasher outside the HashMap, allowing us to access it without
            // taking the HashMap lock.
            hasher: hash::BuildHasherDefault::default(),
        }
    }

    fn instance_create_surface(
        &self,
        handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Self::SurfaceId {
        Surface {
            id: self.global.instance_create_surface(handle, PhantomData),
            configured_device: Mutex::new(None),
        }
    }

    fn instance_request_adapter(
        &self,
        options: &crate::RequestAdapterOptions,
    ) -> Self::RequestAdapterFuture {
        let id = self.global.request_adapter(
            &wgc::instance::RequestAdapterOptions {
                power_preference: options.power_preference,
                compatible_surface: options.compatible_surface.map(|surface| surface.id.id),
            },
            /*wgc::instance::AdapterInputs::Mask(*/wgt::Backends::all()/*, |_| PhantomData)*/,
        );
        ready(id.ok())
    }

    /* fn instance_poll_all_devices(&self, force_wait: bool) {
        let global = &self.0;
        match global.poll_all_devices(force_wait) {
            Ok(()) => (),
            Err(err) => self.handle_error_fatal(err, "Device::poll"),
        }
    } */

    fn adapter_request_device(
        &self,
        adapter: Self::AdapterId,
        desc: &crate::DeviceDescriptor,
        trace_dir: Option<&std::path::Path>,
    ) -> Self::RequestDeviceFuture {
        // let global = &self.0;
        let /*(*/device_id/*, error)*/ = match wgc::gfx_select2!(
            Box adapter =>
            Global::adapter_request_device(
                *adapter,
                &desc.map_label(|l| l.map(Borrowed)),
                trace_dir
                // PhantomData
            ))
        {
            Ok(device_id) => device_id,
            Err(err) => {
                log::error!("Error in Adapter::request_device: {}", err);
                return ready(Err(crate::RequestDeviceError));
            },
        };
        let queue = device_id.clone();
        let device = Device {
            id: device_id,
            error_sink: Arc::new(Mutex::new(ErrorSinkRaw::new())),
            features: desc.features,
        };
        ready(Ok((device, queue)))
    }

    fn adapter_is_surface_supported(
        &self,
        adapter: &Self::AdapterId,
        surface: &Self::SurfaceId,
    ) -> bool {
        let global = &self.global;
        match wgc::gfx_select2!(
            &Box adapter =>
            global.adapter_is_surface_supported(&adapter, surface.id)) {
            Ok(result) => result,
            Err(err) => self.handle_error_fatal(err, "Adapter::is_surface_supported"),
        }
    }

    fn adapter_features(/*&self, */adapter: &Self::AdapterId) -> Features {
        // let global = &self.global;
        /*match */wgc::gfx_select2!(
            &Box adapter =>
            /*global.*/Global::adapter_features(&adapter))
        /*{
>>>>>>> 48b5ae04 (WIP)
            Ok(features) => features,
            Err(err) => self.handle_error_fatal(err, "Adapter::features"),
        }*/
    }

    fn adapter_limits(/*&self, */adapter: &Self::AdapterId) -> Limits {
        // let global = &self.global;
        /*match */wgc::gfx_select2!(
            &Box adapter =>
            /*global.*/Global::adapter_limits(&adapter))
        /*{
            Ok(limits) => limits,
            Err(err) => self.handle_error_fatal(err, "Adapter::limits"),
        }*/
    }

    fn adapter_downlevel_properties(/*&self, */adapter: &Self::AdapterId) -> DownlevelCapabilities {
        // let global = &self.global;
        /*match */wgc::gfx_select2!(
            &Box adapter =>
            /*global.*/Global::adapter_downlevel_properties(&adapter))
        /*{
            Ok(downlevel) => downlevel,
            Err(err) => self.handle_error_fatal(err, "Adapter::downlevel_properties"),
        }*/
    }

    fn adapter_get_info(/*&self, */adapter: &wgc::id::AdapterId) -> AdapterInfo {
        // let global = &self.global;
        /*match */wgc::gfx_select2!(
            &Box adapter =>
            /*global.*/Global::adapter_get_info(&adapter))
        /*{
            Ok(info) => info,
            Err(err) => self.handle_error_fatal(err, "Adapter::get_info"),
        }*/
    }

    fn adapter_get_texture_format_features(
        // &self,
        adapter: &Self::AdapterId,
        format: wgt::TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        // let global = &self.global;
        /*match */wgc::gfx_select2!(
            &Box adapter =>
            /*global.*/Global::adapter_get_texture_format_features(&adapter, format))
        /*{
            Ok(info) => info,
            Err(err) => self.handle_error_fatal(err, "Adapter::get_texture_format_features"),
        }*/
    }

    fn surface_get_preferred_format(
        &self,
        surface: &Self::SurfaceId,
        adapter: &Self::AdapterId,
    ) -> Option<TextureFormat> {
        let global = &self.global;
        match wgc::gfx_select2!(
            &Box adapter =>
            global.surface_get_preferred_format(surface.id, &adapter))
        {
            Ok(format) => Some(format),
            Err(wgc::instance::GetSurfacePreferredFormatError::UnsupportedQueueFamily) => None,
            Err(err) => self.handle_error_fatal(err, "Surface::get_preferred_format"),
        }
    }

    fn surface_configure(
        &self,
        surface: &Self::SurfaceId,
        device: &Self::DeviceId,
        config: &wgt::SurfaceConfiguration,
    ) {
        let global = &self.global;
        let device_id = device.id.clone();
        let error = wgc::gfx_select2!(Arc device_id => global.surface_configure(surface.id, device_id, config));
        if let Some(e) = error {
            self.handle_error_fatal(e, "Surface::configure");
        } else {
            // FIXME: We shouldn't need this.
            *surface.configured_device.lock() = Some(device.id.clone());
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
        let global = &self.global;
        let device_id = surface
            .configured_device
            .lock();
        let device_id = device_id.as_ref().expect("Surface was not configured?");
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
        let global = &self.global;
        match wgc::gfx_select!(texture.id => global.surface_present(detail.surface_id)) {
            Ok(_status) => (),
            Err(err) => self.handle_error_fatal(err, "Surface::present"),
        }
    }

    fn device_features(/*&self, */device: &Self::DeviceId) -> Features {
        // let global = &self.global;
        let device_id = &device.id;
        /*match */wgc::gfx_select2!(&Arc device_id => /*global.*/Global::device_features(device_id))/* {
            Ok(features) => features,
            Err(err) => self.handle_error_fatal(err, "Device::features"),
        }*/
    }

    fn device_limits(/*&self, */device: &Self::DeviceId) -> Limits {
        // let global = &self.global;
        let device_id = &device.id;
        /*match */wgc::gfx_select2!(&Arc device_id => /*global.*/Global::device_limits(device_id))/* {
            Ok(limits) => limits,
            Err(err) => self.handle_error_fatal(err, "Device::limits"),
        }*/
    }

    fn device_downlevel_properties(/*&self, */device: &Self::DeviceId) -> DownlevelCapabilities {
        // let global = &self.global;
        let device_id = &device.id;
        /*match */wgc::gfx_select2!(&Arc device_id => /*global.*/Global::device_downlevel_properties(device_id))/* {
            Ok(limits) => limits,
            Err(err) => self.handle_error_fatal(err, "Device::downlevel_properties"),
        }*/
    }

    fn device_create_shader_module(
        &self,
        device: &Self::DeviceId,
        desc: &ShaderModuleDescriptor,
    ) -> Self::ShaderModuleId {
        // let global = &self.0;
        let descriptor = wgc::pipeline::ShaderModuleDescriptor {
            label: desc.label.map(Borrowed),
        };
        let source = match desc.source {
            #[cfg(feature = "spirv")]
            ShaderSource::SpirV(ref spv) => {
                // Parse the given shader code and store its representation.
                let options = naga::front::spv::Options {
                    adjust_coordinate_space: false, // we require NDC_Y_UP feature
                    strict_capabilities: true,
                    flow_graph_dump_prefix: None,
                };
                let parser = naga::front::spv::Parser::new(spv.iter().cloned(), &options);
                let module = parser.parse().unwrap();
                wgc::pipeline::ShaderModuleSource::Naga(module)
            }
            ShaderSource::Wgsl(ref code) => wgc::pipeline::ShaderModuleSource::Wgsl(Borrowed(code)),
        };
        let device_id = device.id.clone();
        match wgc::gfx_select2!(
            Arc device_id => Global::device_create_shader_module(device_id, &descriptor, source)
        ) {
            Ok(id) => Some(id),
            Err(cause) => {
                self.handle_error(
                    &device.error_sink,
                    cause,
                    LABEL,
                    desc.label,
                    "Device::create_shader_module",
                );
                None
            },
        }
    }

    unsafe fn device_create_shader_module_spirv(
        &self,
        device: &Self::DeviceId,
        desc: &ShaderModuleDescriptorSpirV,
    ) -> Self::ShaderModuleId {
        // let global = &self.0;
        let descriptor = wgc::pipeline::ShaderModuleDescriptor {
            label: desc.label.as_deref().map(Borrowed),
        };
        let device_id = device.id.clone();
        match wgc::gfx_select2!(
            Arc device_id => Global::device_create_shader_module_spirv(device_id, &descriptor, Borrowed(&desc.source))
        ) {
            Ok(id) => Some(id),
            Err(cause) => {
                self.handle_error(
                    &device.error_sink,
                    cause,
                    LABEL,
                    desc.label,
                    "Device::create_shader_module_spirv",
                );
                None
            },
        }
    }

    fn device_create_bind_group_layout(
        &self,
        device: &Self::DeviceId,
        desc: BindGroupLayoutDescriptor,
    ) -> Self::BindGroupLayoutId {
        // let global = &self.global;
        // let device_id = device.id.clone();
        let device_id = &device.id;
        let bind_group_label = desc.label;
        /*let (id, error) = */match wgc::gfx_select2!(
            &Arc device_id =>
                // global.device_create_bind_group_layout(device_id, &descriptor, PhantomData)
                self.deduplicate_bind_group_layout(device_id, desc)
        ) {
            Ok(id) => Some(id),
            /*if let Some(cause) = error*/Err(cause) => {
                self.handle_error(
                    &device.error_sink,
                    cause,
                    LABEL,
                    bind_group_label,
                    "Device::create_bind_group_layout",
                );
                None
            },
        }
    }

    fn device_create_bind_group(
        &self,
        device: &Self::DeviceId,
        desc: &BindGroupDescriptor,
    ) -> Self::BindGroupId {
        use wgc::binding_model as bm;
        let device_id = device.id.clone();
        wgc::gfx_select2!(Arc device_id => {
        let mut arrayed_texture_views = Vec::new();
        if device.features.contains(Features::TEXTURE_BINDING_ARRAY) {
            // gather all the array view IDs first
            for entry in desc.entries.iter() {
                if let BindingResource::TextureViewArray(array) = entry.resource {
                    arrayed_texture_views.extend(array.iter().map(|view| view.id));
                }
            }
        }

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

        // let mut remaining_arrayed_texture_views = &arrayed_texture_views[..];
        // let mut remaining_arrayed_buffer_bindings = &arrayed_buffer_bindings[..];

        let error = core::cell::Cell::new(None::<bm::CreateBindGroupError>);
        let entries = desc
            .entries
            .iter()
            .scan((&arrayed_texture_views[..], &arrayed_buffer_bindings[..]), |(remaining_arrayed_texture_views, remaining_arrayed_buffer_bindings), entry| {
                Some(bm::BindGroupEntry {
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
                            *remaining_arrayed_buffer_bindings =
                                &remaining_arrayed_buffer_bindings[array.len()..];
                            bm::BindingResource::BufferArray(Borrowed(slice))
                        }
                        BindingResource::Sampler(sampler) => {
                            let sampler = if let Some(sampler) = &sampler.id {
                                sampler
                            } else {
                                error.set(Some(bm::CreateBindGroupError::InvalidSampler(entry.binding)));
                                return None;
                            };
                            bm::BindingResource::Sampler(wgc::id::expect_backend(&sampler))
                        },
                        BindingResource::TextureView(texture_view) => {
                            bm::BindingResource::TextureView(texture_view.id)
                        }
                        BindingResource::TextureViewArray(array) => {
                            let slice = &remaining_arrayed_texture_views[..array.len()];
                            *remaining_arrayed_texture_views =
                                &remaining_arrayed_texture_views[array.len()..];
                            bm::BindingResource::TextureViewArray(Borrowed(slice))
                        }
                    },
                })
            });
        // // NOTE: Kind of annoying that we need to allocate solely because of a combination of
        // // the mutable `remaining` counts, and the need to perform tracing.
        // .collect::<Vec<_>>();
        let global = &self.global;
        let result = /*let (id, error) = */(|| {
            let layout = desc.layout.id.clone().map(wgc::id::expect_backend_owned)
                .ok_or(bm::CreateBindGroupError::InvalidLayout)?;
            let descriptor = bm::BindGroupDescriptor {
                label: desc.label.as_ref().map(|label| Borrowed(&label[..])),
                layout,
                entries,
            };
            global.device_create_bind_group(
                device_id,
                descriptor,
                // PhantomData
            )
        })();
        match (result, error.into_inner()) {
            (Ok(id), None) => Some(id),
            (_, Some(cause)) | (Err(cause), None) => {
                self.handle_error(
                    &device.error_sink,
                    cause,
                    LABEL,
                    desc.label,
                    "Device::create_bind_group",
                );
                None
            },
        }
        /* if let Some(cause) = error {
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_bind_group",
            );
        }
        id*/
        })
    }

    fn device_create_pipeline_layout(
        &self,
        device: &Self::DeviceId,
        desc: &PipelineLayoutDescriptor,
    ) -> Self::PipelineLayoutId {
        // Limit is always less or equal to hal::MAX_BIND_GROUPS, so this is always right
        // Guards following ArrayVec
        assert!(
            desc.bind_group_layouts.len() <= hal::MAX_BIND_GROUPS,
            "Bind group layout count {} exceeds device bind group limit {}",
            desc.bind_group_layouts.len(),
            hal::MAX_BIND_GROUPS
        );

        // let global = &self.global;
        let device_id = device.id.clone();
        /*let (id, error) = */match wgc::gfx_select2!(Arc device_id => (move || {
            let bind_group_layouts = desc
                .bind_group_layouts
                .into_iter()
                .enumerate()
                .map(|(index, bgl)| bgl.id.clone().map(wgc::id::expect_backend_owned)
                     .ok_or(wgc::binding_model::CreatePipelineLayoutError::InvalidBindGroupLayout(index)))
                .collect::<Result<ArrayVec<_, { hal::MAX_BIND_GROUPS }>, _>>()?;
            let descriptor = wgc::binding_model::PipelineLayoutDescriptor {
                label: desc.label.map(Borrowed),
                bind_group_layouts,
                push_constant_ranges: Borrowed(&desc.push_constant_ranges),
            };
            Global::device_create_pipeline_layout(
                device_id,
                descriptor,
                //PhantomData
            )
        })()) {
            Ok(pipeline_layout) => Some(pipeline_layout),
            Err(cause) => {
                self.handle_error(
                    &device.error_sink,
                    cause,
                    LABEL,
                    desc.label,
                    "Device::create_pipeline_layout",
                );
                None
            },
        }
        // id
    }

    fn device_create_render_pipeline(
        &self,
        device: &Self::DeviceId,
        desc: RenderPipelineDescriptor,
    ) -> Self::RenderPipelineId {
        use wgc::pipeline as pipe;

        let vertex_buffers: ArrayVec<_, { hal::MAX_VERTEX_BUFFERS }> = desc
            .vertex
            .buffers
            .iter()
            .map(|vbuf| pipe::VertexBufferLayout {
                array_stride: vbuf.array_stride,
                step_mode: vbuf.step_mode,
                attributes: Borrowed(vbuf.attributes),
            })
            .collect();

        /* let implicit_pipeline_ids = match desc.layout {
            Some(_) => None,
            None => Some(wgc::device::ImplicitPipelineIds {
                // root_id: PhantomData,
                group_ids: &[PhantomData; hal::MAX_BIND_GROUPS],
            }),
        }; */
        // let global = &self.global;
        let device_id = device.id.clone();
        wgc::gfx_select2!(Arc device_id => {
            let /*(id, error)*/cause = loop {
                let vertex = if let Some(vertex_stage_module) = &desc.vertex.module.id {
                    pipe::VertexState {
                        stage: pipe::ProgrammableStageDescriptor {
                            module: wgc::id::expect_backend_box(vertex_stage_module),
                            entry_point: Borrowed(desc.vertex.entry_point),
                        },
                        buffers: Borrowed(&vertex_buffers),
                    }
                } else {
                    let error = wgc::pipeline::CreateRenderPipelineError::Stage {
                        stage: wgt::ShaderStages::VERTEX,
                        error: wgc::validation::StageError::InvalidModule
                    };
                    break error;
                };
                let fragment = if let Some(frag) = &desc.fragment {
                    Some(if let Some(fragment_stage_module) = &frag.module.id {
                        pipe::FragmentState {
                            stage: pipe::ProgrammableStageDescriptor {
                                module: wgc::id::expect_backend_box(fragment_stage_module),
                                entry_point: Borrowed(frag.entry_point),
                            },
                            targets: Borrowed(frag.targets),
                        }
                    } else {
                        let error = wgc::pipeline::CreateRenderPipelineError::Stage {
                            stage: wgt::ShaderStages::FRAGMENT,
                            error: wgc::validation::StageError::InvalidModule,
                        };
                        break error;
                    })
                } else {
                    None
                };
                let layout = match desc.layout.map(|layout| layout.id) {
                    Some(Some(layout)) => Some(layout),
                    None => None,
                    Some(None) => {
                        let error = wgc::pipeline::CreateRenderPipelineError::InvalidLayout;
                        break error;
                    },
                };
                let descriptor = pipe::RenderPipelineDescriptor {
                    label: desc.label.map(Borrowed),
                    layout: layout.map(wgc::id::expect_backend_owned),
                    vertex,
                    primitive: desc.primitive,
                    depth_stencil: desc.depth_stencil.clone(),
                    multisample: desc.multisample,
                    fragment,
                };
                match /*global.*/Global::device_create_render_pipeline(device_id, descriptor/*, implicit_pipeline_ids*/) {
                    Ok(id) => return Some(id),
                    Err(cause) => break cause,
                }
            };
            if let wgc::pipeline::CreateRenderPipelineError::Internal { stage, ref error } = cause {
                log::warn!("Shader translation error for stage {:?}: {}", stage, error);
                log::warn!("Please report it to https://github.com/gfx-rs/naga");
                log::warn!("Try enabling `wgpu/cross` feature as a workaround.");
            }
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_render_pipeline",
            );
            None
        })
    }

    fn device_create_compute_pipeline(
        &self,
        device: &Self::DeviceId,
        desc: ComputePipelineDescriptor,
    ) -> Self::ComputePipelineId {
        use wgc::pipeline as pipe;

        /* let implicit_pipeline_ids = match desc.layout {
            Some(_) => None,
            None => Some(wgc::device::ImplicitPipelineIds {
                // root_id: PhantomData,
                group_ids: &[PhantomData; hal::MAX_BIND_GROUPS],
            }),
        }; */

        // let global = &self.global;
        let device_id = device.id.clone();
        wgc::gfx_select2!(Arc device_id => {
            let cause = loop {
                let desc_module_id = if let Some(desc_module_id) = &desc.module.id {
                    desc_module_id
                } else {
                    let error = wgc::pipeline::CreateComputePipelineError::Stage(wgc::validation::StageError::InvalidModule);
                    break error;
                };
                let layout = match desc.layout.map(|layout| layout.id) {
                    Some(Some(layout)) => Some(layout),
                    None => None,
                    Some(None) => {
                        let error = wgc::pipeline::CreateComputePipelineError::InvalidLayout;
                        break error;
                    },
                };
                let descriptor = pipe::ComputePipelineDescriptor {
                    label: desc.label.map(Borrowed),
                    layout: layout.map(wgc::id::expect_backend_owned),
                    stage: pipe::ProgrammableStageDescriptor {
                        module: wgc::id::expect_backend_box(desc_module_id),
                        entry_point: Borrowed(desc.entry_point),
                    },
                };
                match /*global.*/Global::device_create_compute_pipeline(
                    device_id,
                    descriptor,
                    // PhantomData,
                    // implicit_pipeline_ids
                ) {
                    Ok(id) => return Some(id),
                    Err(cause) => break cause,
                }
            };
            if let wgc::pipeline::CreateComputePipelineError::Internal(ref error) = cause {
                log::warn!(
                    "Shader translation error for stage {:?}: {}",
                    wgt::ShaderStages::COMPUTE,
                    error
                );
                log::warn!("Please report it to https://github.com/gfx-rs/naga");
                log::warn!("Try enabling `wgpu/cross` feature as a workaround.");
            }
            self.handle_error(
                &device.error_sink,
                cause,
                LABEL,
                desc.label,
                "Device::create_compute_pipeline",
            );
            None
        })
    }

    fn device_create_buffer(
        &self,
        device: &Self::DeviceId,
        desc: &crate::BufferDescriptor<'_>,
    ) -> Self::BufferId {
        let global = &self.global;
        let device_id = device.id.clone();
        let (id, error) = wgc::gfx_select2!(Arc device_id => global.device_create_buffer(
            device_id,
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
        let global = &self.global;
        let device_id = device.id.clone();
        let (id, error) = wgc::gfx_select2!(Arc device_id => global.device_create_texture(
            device_id,
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

        // let global = &self.0;
        let device_id = device.id.clone();
        /*let (id, error) = */match wgc::gfx_select2!(Arc device_id => Global::device_create_sampler(
            device_id,
            &descriptor,
            // PhantomData
        )) {
            Ok(sampler) => Some(sampler),
            Err(cause) => {
                self.handle_error(
                    &device.error_sink,
                    cause,
                    LABEL,
                    desc.label,
                    "Device::create_sampler",
                );
                None
            },
        }
        // id
    }

    fn device_create_query_set(
        &self,
        device: &Self::DeviceId,
        desc: &wgt::QuerySetDescriptor<Label>,
    ) -> Self::QuerySetId {
        // let global = &self.0;
        let device_id = device.id.clone();
        let id = /*let (id, error) = */match wgc::gfx_select2!(Arc device_id => Global::device_create_query_set(
            device_id,
            &desc.map_label(|l| l.map(Borrowed)),
            // PhantomData
        )) {
            Ok(id) => Some(id),
            Err(cause) => {
                self.handle_error_nolabel(
                    &device.error_sink,
                    cause,
                    "Device::create_query_set",
                );
                None
            },
        };
        id
    }

    fn device_create_command_encoder(
        &self,
        device: &Self::DeviceId,
        desc: &CommandEncoderDescriptor,
    ) -> Self::CommandEncoderId {
        let global = &self.global;
        let device_id = device.id.clone();
        let (id, error) = wgc::gfx_select2!(Arc device_id => global.device_create_command_encoder(
            device_id,
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
        device: &'a Self::DeviceId,
        desc: &RenderBundleEncoderDescriptor,
    ) -> Self::RenderBundleEncoderId {
        let descriptor = wgc::command::RenderBundleEncoderDescriptor {
            label: desc.label.map(Borrowed),
            color_formats: Borrowed(desc.color_formats),
            depth_stencil: desc.depth_stencil,
            sample_count: desc.sample_count,
        };
        match wgc::command::RenderBundleEncoder::new(&descriptor, &device.id, None) {
            Ok(id) => id,
            Err(e) => panic!("Error in Device::create_render_bundle_encoder: {}", e),
        }
    }

    fn device_drop(&self, device: &Self::DeviceId) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let global = &self.global;
            let device_id = &device.id;
            match wgc::gfx_select2!(&Arc device_id => global.device_poll(device_id, true)) {
                Ok(()) => (),
                Err(err) => self.handle_error_fatal(err, "Device::drop"),
            }
        }
        /* //TODO: make this work in general
        #[cfg(not(target_arch = "wasm32"))]
        #[cfg(feature = "metal-auto-capture")]
        {
            let global = &self.global;
            let device_id = &device_id;
            wgc::gfx_select2!(&Arc device.id => global.device_drop(device_id));
        } */
    }

    fn device_poll(&self, device: &Self::DeviceId, maintain: crate::Maintain) {
        let global = &self.global;
        let device_id = &device.id;
        match wgc::gfx_select2!(&Arc device_id => global.device_poll(
            device_id,
            match maintain {
                crate::Maintain::Poll => false,
                crate::Maintain::Wait => true,
            }
        )) {
            Ok(()) => (),
            Err(err) => self.handle_error_fatal(err, "Device::poll"),
        }
        // Take this opportunity to clean up shared bind group layouts.
        self.clear_bind_group_layouts();
    }

    fn device_on_uncaptured_error(
        &self,
        device: &Self::DeviceId,
        handler: impl crate::UncapturedErrorHandler,
    ) {
        let mut error_sink = device.error_sink.lock();
        error_sink.uncaptured_handler = Box::new(handler);
    }

    fn buffer_map_async(
        &self,
        buffer: &Self::BufferId,
        mode: MapMode,
        range: Range<wgt::BufferAddress>,
    ) -> Self::MapAsyncFuture {
        let (future, completion) = native_gpu_future::new_gpu_future();

        extern "C" fn buffer_map_future_wrapper(
            status: wgc::resource::BufferMapAsyncStatus,
            user_data: *mut u8,
        ) {
            let completion =
                unsafe { native_gpu_future::GpuFutureCompletion::from_raw(user_data as _) };
            completion.complete(match status {
                wgc::resource::BufferMapAsyncStatus::Success => Ok(()),
                _ => Err(crate::BufferAsyncError),
            })
        }

        let operation = wgc::resource::BufferMapOperation {
            host: match mode {
                MapMode::Read => wgc::device::HostMap::Read,
                MapMode::Write => wgc::device::HostMap::Write,
            },
            callback: buffer_map_future_wrapper,
            user_data: completion.into_raw() as _,
        };

        let global = &self.global;
        match wgc::gfx_select!(buffer.id => global.buffer_map_async(buffer.id, range, operation)) {
            Ok(()) => (),
            Err(cause) => self.handle_error_nolabel(&buffer.error_sink, cause, "Buffer::map_async"),
        }
        future
    }

    fn buffer_get_mapped_range(
        &self,
        buffer: &Self::BufferId,
        sub_range: Range<wgt::BufferAddress>,
    ) -> BufferMappedRange {
        let size = sub_range.end - sub_range.start;
        let global = &self.global;
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
        let global = &self.global;
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
        let global = &self.global;
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

    /* fn adapter_drop(&self, adapter: &Self::AdapterId) {
        let global = &self.0;
        wgc::gfx_select!(adapter => global.adapter_drop(adapter))
    } */

    fn buffer_destroy(&self, buffer: &Self::BufferId) {
        let global = &self.global;
        match wgc::gfx_select!(buffer.id => global.buffer_destroy(buffer.id)) {
            Ok(()) => (),
            Err(err) => self.handle_error_fatal(err, "Buffer::destroy"),
        }
    }
    fn buffer_drop(&self, buffer: &Self::BufferId) {
        let global = &self.global;
        wgc::gfx_select!(buffer.id => global.buffer_drop(buffer.id, false))
    }
    fn texture_destroy(&self, texture: &Self::TextureId) {
        let global = &self.global;
        match wgc::gfx_select!(texture.id => global.texture_destroy(texture.id)) {
            Ok(()) => (),
            Err(err) => self.handle_error_fatal(err, "Texture::destroy"),
        }
    }
    fn texture_drop(&self, texture: &Self::TextureId) {
        let global = &self.global;
        wgc::gfx_select!(texture.id => global.texture_drop(texture.id, false))
    }
    fn texture_view_drop(&self, texture_view: &Self::TextureViewId) {
        let global = &self.global;
        match wgc::gfx_select!(*texture_view => global.texture_view_drop(*texture_view, false)) {
            Ok(()) => (),
            Err(err) => self.handle_error_fatal(err, "TextureView::drop"),
        }
    }
    /* fn sampler_drop(/*&self, */_sampler: &mut Self::SamplerId) {
        // NOTE: Dropping the query set should handle removal itself, so we only use this for
        // tracing.
        #[cfg(feature = "trace")]
        {
            // let global = &self.0;
            // FIXME: When these are not an Option, no need to call `take()`.
            if let Some(sampler) = _sampler.take() {
                wgc::gfx_select2!(Arc sampler => Global::sampler_drop(sampler))
            }
        }
    } */
    /* fn query_set_drop(/*&self, */_query_set: &mut Self::QuerySetId) {
        // NOTE: Dropping the query set should handle removal itself, so we only use this for
        // tracing.
        #[cfg(feature = "trace")]
        {
            // let global = &self.0;
            // FIXME: When these are not an Option, no need to call `take()`.
            if let Some(query_set) = _query_set.take() {
                wgc::gfx_select2!(Arc query_set => Global::query_set_drop(query_set))
            }
        }
    } */

    /* fn bind_group_drop(/*&self, */_bind_group: &mut Self::BindGroupId) {
        // NOTE: Dropping the bind group should handle removal itself, so we only use this for
        // tracing.
        #[cfg(feature = "trace")]
        {
            // let global = &self.0;
            // FIXME: When these are not an Option, no need to call `take()`.
            if let Some(bind_group) = _bind_group.take() {
                wgc::gfx_select2!(Arc bind_group => Global::bind_group_drop(bind_group))
            }
        }
    } */
    /* fn bind_group_layout_drop(&self, bind_group_layout: &Self::BindGroupLayoutId) {
        let global = &self.0;
        wgc::gfx_select!(*bind_group_layout => global.bind_group_layout_drop(*bind_group_layout))
    } */
    /* fn pipeline_layout_drop(/*&self, */_pipeline_layout: &mut Self::PipelineLayoutId) {
        // NOTE: Dropping the pipeline layout should handle removal itself, so we only use this for
        // tracing.
        #[cfg(feature = "trace")]
        {
            // let global = &self.0;
            // FIXME: When these are not an Option, no need to call `take()`.
            if let Some(pipeline_layout) = _pipeline_layout.take() {
                wgc::gfx_select2!(Arc pipeline_layout => Global::pipeline_layout_drop(pipeline_layout));
            }
        }
    } */
    /* fn shader_module_drop(/*&self, */_shader_module: &mut Self::ShaderModuleId) {
        // NOTE: Dropping the shader module should handle removal itself, so we only use this for
        // tracing.
        #[cfg(feature = "trace")]
        // FIXME: When these are not an Option, no need to call `take()`.
        if let Some(shader_module) = _shader_module.take() {
            // let global = &self.0;
            wgc::gfx_select2!(Box shader_module => Global::shader_module_drop(shader_module))
        }
    } */
    fn command_encoder_drop(&self, command_encoder: &Self::CommandEncoderId) {
        if command_encoder.open {
            let global = &self.global;
            wgc::gfx_select!(command_encoder.id => global.command_encoder_drop(command_encoder.id))
        }
    }
    fn command_buffer_drop(&self, command_buffer: &Self::CommandBufferId) {
        let global = &self.global;
        wgc::gfx_select!(*command_buffer => global.command_buffer_drop(*command_buffer))
    }
    /* fn render_bundle_drop(&self, render_bundle: &Self::RenderBundleId) {
        let global = &self.0;
        wgc::gfx_select!(*render_bundle => global.render_bundle_drop(*render_bundle))
    } */
    /* fn compute_pipeline_drop(&self, pipeline: &Self::ComputePipelineId) {
        let global = &self.0;
        wgc::gfx_select!(*pipeline => global.compute_pipeline_drop(*pipeline))
    } */
    /* fn render_pipeline_drop(&self, pipeline: &Self::RenderPipelineId) {
        let global = &self.0;
        wgc::gfx_select!(*pipeline => global.render_pipeline_drop(*pipeline))
    } */

    fn compute_pipeline_get_bind_group_layout(
        // &self,
        pipeline: &Self::ComputePipelineId,
        index: u32,
    ) -> Self::BindGroupLayoutId {
        // let global = &self.global;
        // FIXME: Instead of panicking, make sure bind group id becomes an error if the bind group
        // is an error.
        let pipeline = if let Some(pipeline) = pipeline {
            pipeline
        } else {
            let err = wgc::binding_model::GetBindGroupLayoutError::InvalidPipeline;
            panic!("Error reflecting bind group {}: {}", index, err);
        };
        /*let (id, error) = */match wgc::gfx_select2!(
            &Arc pipeline => /*global.*/Global::compute_pipeline_get_bind_group_layout(pipeline, index/*, PhantomData*/)
        ) {
            Ok(id) => Some(id),
            Err(err) => {
                panic!("Error reflecting bind group {}: {}", index, err);
            }
        }
    }

    fn render_pipeline_get_bind_group_layout(
        // &self,
        pipeline: &Self::RenderPipelineId,
        index: u32,
    ) -> Self::BindGroupLayoutId {
        // let global = &self.global;
        // FIXME: Instead of panicking, make sure bind group id becomes an error if the bind group
        // is an error.
        let pipeline = if let Some(pipeline) = pipeline {
            pipeline
        } else {
            let err = wgc::binding_model::GetBindGroupLayoutError::InvalidPipeline;
            panic!("Error reflecting bind group {}: {}", index, err);
        };
        /*let (id, error) = */match wgc::gfx_select2!(
            &Arc pipeline => /*global.*/Global::render_pipeline_get_bind_group_layout(pipeline, index/*, PhantomData*/)
        ) {
            Ok(id) => Some(id),
            Err(err) => {
                panic!("Error reflecting bind group {}: {}", index, err);
            },
        }
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
        let global = &self.global;
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
        let global = &self.global;
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
        let global = &self.global;
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
        let global = &self.global;
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
        // FIXME: Make sure whole encoder becomes an error if the query set is an error.
        if let Some(query_set) = query_set {
            let global = &self.global;
            if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_write_timestamp(
                encoder.id,
                query_set,
                query_index
            )) {
                self.handle_error_nolabel(
                    &encoder.error_sink,
                    cause,
                    "CommandEncoder::write_timestamp",
                );
            }
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
        // FIXME: Make sure whole encoder becomes an error if the query set is an error.
        if let Some(query_set) = query_set {
            let global = &self.global;
            if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_resolve_query_set(
                encoder.id,
                query_set,
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
        let global = &self.global;
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

    fn command_encoder_begin_render_pass<'b>(
        &self,
        encoder: &Self::CommandEncoderId,
        desc: &crate::RenderPassDescriptor<'b, '_>,
    ) -> Self::RenderPassId {
        let colors = desc
            .color_attachments
            .iter()
            .map(|ca| wgc::command::RenderPassColorAttachment {
                view: ca.view.id,
                resolve_target: ca.resolve_target.map(|rt| rt.id),
                channel: map_pass_channel(Some(&ca.ops)),
            })
            .collect::<ArrayVec<_, { hal::MAX_COLOR_TARGETS }>>();

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
        let global = &self.global;
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
        let global = &self.global;
        let (id, error) =
            wgc::gfx_select!(encoder.id => global.command_encoder_finish(encoder.id, &descriptor));
        if let Some(cause) = error {
            self.handle_error_nolabel(&encoder.error_sink, cause, "a CommandEncoder");
        }
        id
    }

    fn command_encoder_clear_image(
        &self,
        encoder: &Self::CommandEncoderId,
        texture: &crate::Texture,
        subresource_range: &wgt::ImageSubresourceRange,
    ) {
        let global = &self.global;
        if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_clear_image(
            encoder.id,
            texture.id.id,
            subresource_range
        )) {
            self.handle_error_nolabel(&encoder.error_sink, cause, "CommandEncoder::clear_image");
        }
    }

    fn command_encoder_clear_buffer(
        &self,
        encoder: &Self::CommandEncoderId,
        buffer: &crate::Buffer,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferSize>,
    ) {
        let global = &self.global;
        if let Err(cause) = wgc::gfx_select!(encoder.id => global.command_encoder_clear_buffer(
            encoder.id,
            buffer.id.id,
            offset, size
        )) {
            self.handle_error_nolabel(&encoder.error_sink, cause, "CommandEncoder::clear_buffer");
        }
    }

    fn command_encoder_insert_debug_marker(&self, encoder: &Self::CommandEncoderId, label: &str) {
        let global = &self.global;
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
        let global = &self.global;
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
        let global = &self.global;
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
        let global = &self.global;
        /* let (id, error) = */match wgc::gfx_select!(encoder.parent() => global.render_bundle_encoder_finish(
            encoder,
            &desc.map_label(|l| l.map(Borrowed))
            // PhantomData
        )) {
            Ok(id) => id,
            /*if let Some*/Err(err)/* = error*/=> {
                self.handle_error_fatal(err, "RenderBundleEncoder::finish");
            },
        }
    }

    fn queue_write_buffer(
        &self,
        queue: &Self::QueueId,
        buffer: &Self::BufferId,
        offset: wgt::BufferAddress,
        data: &[u8],
    ) {
        let global = &self.global;
        match wgc::gfx_select2!(
            &Arc queue => global.queue_write_buffer(queue, buffer.id, offset, data)
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
        let global = &self.global;
        match wgc::gfx_select2!(&Arc queue => global.queue_write_texture(
            queue,
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
    ) {
        // let temp_command_buffers = command_buffers.collect::<SmallVec<[_; 4]>>();

        let global = &self.global;
        match wgc::gfx_select2!(&Arc queue => global.queue_submit(queue, /*&temp_command_buffers*/command_buffers)) {
            Ok(()) => (),
            Err(err) => self.handle_error_fatal(err, "Queue::submit"),
        }
    }

    fn queue_get_timestamp_period(&self, queue: &Self::QueueId) -> f32 {
        let global = &self.global;
        /*let res = */wgc::gfx_select2!(&Arc queue => global.queue_get_timestamp_period(
            queue
        ))/*;
        match res {
            Ok(v) => v,
            Err(cause) => {
                self.handle_error_fatal(cause, "Queue::get_timestamp_period");
            }
        }*/
    }

    fn queue_on_submitted_work_done(
        &self,
        queue: &Self::QueueId,
    ) -> Self::OnSubmittedWorkDoneFuture {
        let (future, completion) = native_gpu_future::new_gpu_future();

        extern "C" fn submitted_work_done_future_wrapper(user_data: *mut u8) {
            let completion =
                unsafe { native_gpu_future::GpuFutureCompletion::from_raw(user_data as _) };
            completion.complete(())
        }

        let closure = wgc::device::queue::SubmittedWorkDoneClosure {
            callback: submitted_work_done_future_wrapper,
            user_data: completion.into_raw() as _,
        };

        let res = wgc::gfx_select2!(&Arc queue => Global::queue_on_submitted_work_done(queue, closure));
        if let Err(cause) = res {
            self.handle_error_fatal(cause, "Queue::on_submitted_work_done");
        }
        future
    }

    fn device_start_capture(&self, device: &Self::DeviceId) {
        let global = &self.global;
        let device_id = &device.id;
        wgc::gfx_select2!(&Arc device_id => global.device_start_capture(device_id));
    }

    fn device_stop_capture(&self, device: &Self::DeviceId) {
        let global = &self.global;
        let device_id = &device.id;
        wgc::gfx_select2!(&Arc device_id => global.device_stop_capture(device_id));
    }
}

#[derive(Debug)]
pub(crate) struct SurfaceOutputDetail {
    surface_id: wgc::id::SurfaceId,
}

type ErrorSink = Arc<Mutex<ErrorSinkRaw>>;

struct ErrorSinkRaw {
    uncaptured_handler: Box<dyn crate::UncapturedErrorHandler>,
}

impl ErrorSinkRaw {
    fn new() -> ErrorSinkRaw {
        ErrorSinkRaw {
            uncaptured_handler: Box::from(default_error_handler),
        }
    }

    fn handle_error(&self, err: crate::Error) {
        (self.uncaptured_handler)(err);
    }
}

impl fmt::Debug for ErrorSinkRaw {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ErrorSink")
    }
}

fn default_error_handler(err: crate::Error) {
    log::error!("wgpu error: {}\n", err);

    panic!("Handling wgpu errors as fatal by default");
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
