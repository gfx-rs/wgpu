/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    binding_model,
    command,
    conv,
    hub::{AllIdentityFilter, GfxBackend, Global, IdentityFilter, Token},
    id,
    pipeline,
    resource,
    swap_chain,
    track::{BufferState, TextureState, TrackerSet},
    BufferAddress,
    FastHashMap,
    Features,
    LifeGuard,
    Stored,
};

use arrayvec::ArrayVec;
use copyless::VecHelper as _;
use hal::{
    self,
    command::CommandBuffer as _,
    device::Device as _,
    queue::CommandQueue as _,
    window::{PresentationSurface as _, Surface as _},
};
use parking_lot::{Mutex, MutexGuard};
use rendy_descriptor::{DescriptorAllocator, DescriptorRanges};
use rendy_memory::{Block, Heaps};
use smallvec::SmallVec;

use std::{
    collections::hash_map::Entry,
    ffi,
    iter,
    marker::PhantomData,
    ops,
    ptr,
    slice,
    sync::atomic::Ordering,
};

mod life;

pub const MAX_COLOR_TARGETS: usize = 4;
pub const MAX_MIP_LEVELS: usize = 16;
pub const MAX_VERTEX_BUFFERS: usize = 8;

/// Bound uniform/storage buffer offsets must be aligned to this number.
pub const BIND_BUFFER_ALIGNMENT: hal::buffer::Offset = 256;

pub fn all_buffer_stages() -> hal::pso::PipelineStage {
    use hal::pso::PipelineStage as Ps;
    Ps::DRAW_INDIRECT
        | Ps::VERTEX_INPUT
        | Ps::VERTEX_SHADER
        | Ps::FRAGMENT_SHADER
        | Ps::COMPUTE_SHADER
        | Ps::TRANSFER
        | Ps::HOST
}
pub fn all_image_stages() -> hal::pso::PipelineStage {
    use hal::pso::PipelineStage as Ps;
    Ps::EARLY_FRAGMENT_TESTS
        | Ps::LATE_FRAGMENT_TESTS
        | Ps::COLOR_ATTACHMENT_OUTPUT
        | Ps::VERTEX_SHADER
        | Ps::FRAGMENT_SHADER
        | Ps::COMPUTE_SHADER
        | Ps::TRANSFER
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum HostMap {
    Read,
    Write,
}

#[derive(Clone, Debug, Hash, PartialEq)]
pub(crate) struct AttachmentData<T> {
    pub colors: ArrayVec<[T; MAX_COLOR_TARGETS]>,
    pub resolves: ArrayVec<[T; MAX_COLOR_TARGETS]>,
    pub depth_stencil: Option<T>,
}
impl<T: PartialEq> Eq for AttachmentData<T> {}
impl<T> AttachmentData<T> {
    pub(crate) fn all(&self) -> impl Iterator<Item = &T> {
        self.colors
            .iter()
            .chain(&self.resolves)
            .chain(&self.depth_stencil)
    }
}

impl RenderPassContext {
    // Assumed the renderpass only contains one subpass
    pub(crate) fn compatible(&self, other: &RenderPassContext) -> bool {
        self.colors == other.colors && self.depth_stencil == other.depth_stencil
    }
}

pub(crate) type RenderPassKey = AttachmentData<hal::pass::Attachment>;
pub(crate) type FramebufferKey = AttachmentData<id::TextureViewId>;
pub(crate) type RenderPassContext = AttachmentData<resource::TextureFormat>;

type BufferMapResult = Result<*mut u8, hal::device::MapError>;
type BufferMapPendingCallback = (resource::BufferMapOperation, BufferMapResult);

pub type BufferMapReadCallback =
    unsafe extern "C" fn(status: resource::BufferMapAsyncStatus, data: *const u8, userdata: *mut u8);
pub type BufferMapWriteCallback =
    unsafe extern "C" fn(status: resource::BufferMapAsyncStatus, data: *mut u8, userdata: *mut u8);

fn map_buffer<B: hal::Backend>(
    raw: &B::Device,
    buffer: &mut resource::Buffer<B>,
    buffer_range: ops::Range<BufferAddress>,
    kind: HostMap,
) -> BufferMapResult {
    let is_coherent = buffer
        .memory
        .properties()
        .contains(hal::memory::Properties::COHERENT);
    let (ptr, mapped_range) = {
        let mapped = buffer.memory.map(raw, buffer_range)?;
        (mapped.ptr(), mapped.range())
    };

    if !is_coherent {
        match kind {
            HostMap::Read => unsafe {
                raw.invalidate_mapped_memory_ranges(iter::once((
                    buffer.memory.memory(),
                    mapped_range,
                )))
                .unwrap();
            },
            HostMap::Write => {
                buffer.mapped_write_ranges.push(mapped_range);
            }
        }
    }

    Ok(ptr.as_ptr())
}

fn unmap_buffer<B: hal::Backend>(
    raw: &B::Device,
    buffer: &mut resource::Buffer<B>,
) {
    if !buffer.mapped_write_ranges.is_empty() {
        unsafe {
            raw
                .flush_mapped_memory_ranges(
                    buffer
                        .mapped_write_ranges
                        .iter()
                        .map(|r| (buffer.memory.memory(), r.clone())),
                )
                .unwrap()
        };
        buffer.mapped_write_ranges.clear();
    }

    buffer.memory.unmap(raw);
}

//Note: this logic is specifically moved out of `handle_mapping()` in order to
// have nothing locked by the time we execute users callback code.
fn fire_map_callbacks<I: IntoIterator<Item = BufferMapPendingCallback>>(callbacks: I) {
    for (operation, result) in callbacks {
        let (status, ptr) = match result {
            Ok(ptr) => (resource::BufferMapAsyncStatus::Success, ptr),
            Err(e) => {
                log::error!("failed to map buffer: {:?}", e);
                (resource::BufferMapAsyncStatus::Error, ptr::null_mut())
            }
        };
        match operation {
            resource::BufferMapOperation::Read(on_read) => {
                on_read(status, ptr)
            }
            resource::BufferMapOperation::Write(on_write) => {
                on_write(status, ptr)
            }
        }
    }
}


#[derive(Debug)]
pub struct Device<B: hal::Backend> {
    pub(crate) raw: B::Device,
    pub(crate) adapter_id: id::AdapterId,
    pub(crate) queue_group: hal::queue::QueueGroup<B>,
    pub(crate) com_allocator: command::CommandAllocator<B>,
    mem_allocator: Mutex<Heaps<B>>,
    desc_allocator: Mutex<DescriptorAllocator<B>>,
    life_guard: LifeGuard,
    pub(crate) trackers: Mutex<TrackerSet>,
    pub(crate) render_passes: Mutex<FastHashMap<RenderPassKey, B::RenderPass>>,
    pub(crate) framebuffers: Mutex<FastHashMap<FramebufferKey, B::Framebuffer>>,
    // Life tracker should be locked right after the device and before anything else.
    life_tracker: Mutex<life::LifetimeTracker<B>>,
    temp_suspected: life::SuspectedResources,
    pub(crate) features: Features,
}

impl<B: GfxBackend> Device<B> {
    pub(crate) fn new(
        raw: B::Device,
        adapter_id: id::AdapterId,
        queue_group: hal::queue::QueueGroup<B>,
        mem_props: hal::adapter::MemoryProperties,
        supports_texture_d24_s8: bool,
        max_bind_groups: u32,
    ) -> Self {
        // don't start submission index at zero
        let life_guard = LifeGuard::new();
        life_guard.submission_index.fetch_add(1, Ordering::Relaxed);

        let heaps = {
            let types = mem_props.memory_types.iter().map(|mt| {
                use rendy_memory::{DynamicConfig, HeapsConfig, LinearConfig};
                let config = HeapsConfig {
                    linear: if mt.properties.contains(hal::memory::Properties::CPU_VISIBLE) {
                        Some(LinearConfig {
                            linear_size: 0x10_00_00,
                        })
                    } else {
                        None
                    },
                    dynamic: Some(DynamicConfig {
                        block_size_granularity: 0x1_00,
                        max_chunk_size: 0x1_00_00_00,
                        min_device_allocation: 0x1_00_00,
                    }),
                };
                (mt.properties, mt.heap_index as u32, config)
            });
            unsafe { Heaps::new(types, mem_props.memory_heaps.iter().cloned()) }
        };

        Device {
            raw,
            adapter_id,
            com_allocator: command::CommandAllocator::new(queue_group.family),
            mem_allocator: Mutex::new(heaps),
            desc_allocator: Mutex::new(DescriptorAllocator::new()),
            queue_group,
            life_guard,
            trackers: Mutex::new(TrackerSet::new(B::VARIANT)),
            render_passes: Mutex::new(FastHashMap::default()),
            framebuffers: Mutex::new(FastHashMap::default()),
            life_tracker: Mutex::new(life::LifetimeTracker::new()),
            temp_suspected: life::SuspectedResources::default(),
            features: Features {
                max_bind_groups,
                supports_texture_d24_s8,
            },
        }
    }

    fn lock_life<'a>(
        &'a self, _token: &mut Token<'a, Self>
    ) -> MutexGuard<'a, life::LifetimeTracker<B>> {
        self.life_tracker.lock()
    }

    fn maintain<'a, F: AllIdentityFilter>(
        &'a self,
        global: &Global<F>,
        force_wait: bool,
        token: &mut Token<'a, Self>,
    ) -> Vec<BufferMapPendingCallback> {
        let mut life_tracker = self.lock_life(token);

        life_tracker.triage_suspected(global, &self.trackers, token);
        life_tracker.triage_mapped(global, token);
        life_tracker.triage_framebuffers(global, &mut *self.framebuffers.lock(), token);
        life_tracker.cleanup(
            &self.raw,
            force_wait,
            &self.mem_allocator,
            &self.desc_allocator,
        );
        let callbacks = life_tracker.handle_mapping(global, &self.raw, token);

        unsafe {
            self.desc_allocator.lock().cleanup(&self.raw);
        }

        callbacks
    }

    fn create_buffer(
        &self,
        self_id: id::DeviceId,
        desc: &resource::BufferDescriptor,
    ) -> resource::Buffer<B> {
        debug_assert_eq!(self_id.backend(), B::VARIANT);
        let (usage, _memory_properties) = conv::map_buffer_usage(desc.usage);

        let rendy_usage = {
            use rendy_memory::MemoryUsageValue as Muv;
            use resource::BufferUsage as Bu;

            if !desc.usage.intersects(Bu::MAP_READ | Bu::MAP_WRITE) {
                Muv::Data
            } else if (Bu::MAP_WRITE | Bu::COPY_SRC).contains(desc.usage) {
                Muv::Upload
            } else if (Bu::MAP_READ | Bu::COPY_DST).contains(desc.usage) {
                Muv::Download
            } else {
                Muv::Dynamic
            }
        };

        let mut buffer = unsafe { self.raw.create_buffer(desc.size, usage).unwrap() };
        let requirements = unsafe { self.raw.get_buffer_requirements(&buffer) };
        let memory = self
            .mem_allocator
            .lock()
            .allocate(
                &self.raw,
                requirements.type_mask as u32,
                rendy_usage,
                requirements.size,
                requirements.alignment,
            )
            .unwrap();

        unsafe {
            self.raw
                .bind_buffer_memory(memory.memory(), memory.range().start, &mut buffer)
                .unwrap()
        };

        resource::Buffer {
            raw: buffer,
            device_id: Stored {
                value: self_id,
                ref_count: self.life_guard.add_ref(),
            },
            usage: desc.usage,
            memory,
            size: desc.size,
            full_range: (),
            mapped_write_ranges: Vec::new(),
            pending_mapping: None,
            life_guard: LifeGuard::new(),
        }
    }

    fn create_texture(
        &self,
        self_id: id::DeviceId,
        desc: &resource::TextureDescriptor,
    ) -> resource::Texture<B> {
        debug_assert_eq!(self_id.backend(), B::VARIANT);

        // Ensure `D24Plus` textures cannot be copied
        match desc.format {
            resource::TextureFormat::Depth24Plus | resource::TextureFormat::Depth24PlusStencil8 => {
                assert!(!desc.usage.intersects(
                    resource::TextureUsage::COPY_SRC | resource::TextureUsage::COPY_DST
                ));
            }
            _ => {}
        }

        let kind = conv::map_texture_dimension_size(
            desc.dimension,
            desc.size,
            desc.array_layer_count,
            desc.sample_count,
        );
        let format = conv::map_texture_format(desc.format, self.features);
        let aspects = format.surface_desc().aspects;
        let usage = conv::map_texture_usage(desc.usage, aspects);

        assert!((desc.mip_level_count as usize) < MAX_MIP_LEVELS);
        let mut view_capabilities = hal::image::ViewCapabilities::empty();

        // 2D textures with array layer counts that are multiples of 6 could be cubemaps
        // Following gpuweb/gpuweb#68 always add the hint in that case
        if desc.dimension == resource::TextureDimension::D2 && desc.array_layer_count % 6 == 0 {
            view_capabilities |= hal::image::ViewCapabilities::KIND_CUBE;
        };

        // TODO: 2D arrays, cubemap arrays

        let mut image = unsafe {
            self.raw.create_image(
                kind,
                desc.mip_level_count as hal::image::Level,
                format,
                hal::image::Tiling::Optimal,
                usage,
                view_capabilities,
            )
        }
        .unwrap();
        let requirements = unsafe { self.raw.get_image_requirements(&image) };

        let memory = self
            .mem_allocator
            .lock()
            .allocate(
                &self.raw,
                requirements.type_mask as u32,
                rendy_memory::Data,
                requirements.size,
                requirements.alignment,
            )
            .unwrap();

        unsafe {
            self.raw
                .bind_image_memory(memory.memory(), memory.range().start, &mut image)
                .unwrap()
        };

        resource::Texture {
            raw: image,
            device_id: Stored {
                value: self_id,
                ref_count: self.life_guard.add_ref(),
            },
            usage: desc.usage,
            kind,
            format: desc.format,
            full_range: hal::image::SubresourceRange {
                aspects,
                levels: 0 .. desc.mip_level_count as hal::image::Level,
                layers: 0 .. desc.array_layer_count as hal::image::Layer,
            },
            memory,
            life_guard: LifeGuard::new(),
        }
    }
}

impl<B: hal::Backend> Device<B> {
    pub(crate) fn destroy_bind_group(&self, bind_group: binding_model::BindGroup<B>) {
        unsafe {
            self.desc_allocator.lock().free(iter::once(bind_group.raw));
        }
    }

    pub(crate) fn destroy_buffer(&self, buffer: resource::Buffer<B>) {
        unsafe {
            self.mem_allocator.lock().free(&self.raw, buffer.memory);
            self.raw.destroy_buffer(buffer.raw);
        }
    }

    pub(crate) fn destroy_texture(&self, texture: resource::Texture<B>) {
        unsafe {
            self.mem_allocator.lock().free(&self.raw, texture.memory);
            self.raw.destroy_image(texture.raw);
        }
    }

    pub(crate) fn dispose(self) {
        self.life_tracker.lock().cleanup(
            &self.raw,
            true,
            &self.mem_allocator,
            &self.desc_allocator,
        );
        self.com_allocator.destroy(&self.raw);
        let desc_alloc = self.desc_allocator.into_inner();
        let mem_alloc = self.mem_allocator.into_inner();
        unsafe {
            desc_alloc.dispose(&self.raw);
            mem_alloc.dispose(&self.raw);
        }
    }
}

#[derive(Debug)]
pub struct ShaderModule<B: hal::Backend> {
    pub(crate) raw: B::ShaderModule,
}


impl<F: IdentityFilter<id::BufferId>> Global<F> {
    pub fn device_create_buffer<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::BufferDescriptor,
        id_in: F::Input,
    ) -> id::BufferId {
        let hub = B::hub(self);
        let mut token = Token::root();

        log::info!("Create buffer {:?} with ID {:?}", desc, id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        let buffer = device.create_buffer(device_id, desc);
        let ref_count = buffer.life_guard.add_ref();

        let id = hub.buffers.register_identity(id_in, buffer, &mut token);
        device
            .trackers
            .lock()
            .buffers
            .init(
                id,
                ref_count,
                BufferState::with_usage(resource::BufferUsage::empty()),
            )
            .unwrap();
        id
    }

    pub fn device_create_buffer_mapped<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::BufferDescriptor,
        id_in: F::Input,
    ) -> (id::BufferId, *mut u8) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let mut desc = desc.clone();
        desc.usage |= resource::BufferUsage::MAP_WRITE;

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        let mut buffer = device.create_buffer(device_id, &desc);
        let ref_count = buffer.life_guard.add_ref();

        let pointer = match map_buffer(&device.raw, &mut buffer, 0 .. desc.size, HostMap::Write) {
            Ok(ptr) => ptr,
            Err(e) => {
                log::error!("failed to create buffer in a mapped state: {:?}", e);
                ptr::null_mut()
            }
        };

        let id = hub.buffers.register_identity(id_in, buffer, &mut token);
        device.trackers
            .lock()
            .buffers.init(
                id,
                ref_count,
                BufferState::with_usage(resource::BufferUsage::MAP_WRITE),
            )
            .unwrap();

        (id, pointer)
    }

    pub fn device_set_buffer_sub_data<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        data: &[u8],
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut buffer_guard, _) = hub.buffers.write(&mut token);
        let device = &device_guard[device_id];
        let mut buffer = &mut buffer_guard[buffer_id];
        assert!(buffer.usage.contains(resource::BufferUsage::MAP_WRITE));
        //assert!(buffer isn't used by the GPU);

        match map_buffer(
            &device.raw,
            &mut buffer,
            offset .. offset + data.len() as BufferAddress,
            HostMap::Write,
        ) {
            Ok(ptr) => unsafe {
                ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            },
            Err(e) => {
                log::error!("failed to map a buffer: {:?}", e);
                return;
            }
        }

        unmap_buffer(&device.raw, buffer);
    }

    pub fn device_get_buffer_sub_data<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        data: &mut [u8],
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut buffer_guard, _) = hub.buffers.write(&mut token);
        let device = &device_guard[device_id];
        let mut buffer = &mut buffer_guard[buffer_id];
        assert!(buffer.usage.contains(resource::BufferUsage::MAP_READ));
        //assert!(buffer isn't used by the GPU);

        match map_buffer(
            &device.raw,
            &mut buffer,
            offset .. offset + data.len() as BufferAddress,
            HostMap::Read,
        ) {
            Ok(ptr) => unsafe {
                ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), data.len());
            },
            Err(e) => {
                log::error!("failed to map a buffer: {:?}", e);
                return;
            }
        }

        unmap_buffer(&device.raw, buffer);
    }

    pub fn buffer_destroy<B: GfxBackend>(&self, buffer_id: id::BufferId) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let device_id = {
            let (mut buffer_guard, _) = hub.buffers.write(&mut token);
            let buffer = &mut buffer_guard[buffer_id];
            buffer.life_guard.ref_count.take();
            buffer.device_id.value
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources.buffers.push(buffer_id);
    }
}

impl<F: IdentityFilter<id::TextureId>> Global<F> {
    pub fn device_create_texture<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::TextureDescriptor,
        id_in: F::Input,
    ) -> id::TextureId {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        let texture = device.create_texture(device_id, desc);
        let range = texture.full_range.clone();
        let ref_count = texture.life_guard.add_ref();

        let id = hub.textures.register_identity(id_in, texture, &mut token);
        device.trackers
            .lock()
            .textures.init(
                id,
                ref_count,
                TextureState::with_range(&range),
            )
            .unwrap();
        id
    }

    pub fn texture_destroy<B: GfxBackend>(&self, texture_id: id::TextureId) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let device_id = {
            let (mut texture_guard, _) = hub.textures.write(&mut token);
            let texture = &mut texture_guard[texture_id];
            texture.life_guard.ref_count.take();
            texture.device_id.value
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources.textures.push(texture_id);
    }
}

impl<F: IdentityFilter<id::TextureViewId>> Global<F> {
    pub fn texture_create_view<B: GfxBackend>(
        &self,
        texture_id: id::TextureId,
        desc: Option<&resource::TextureViewDescriptor>,
        id_in: F::Input,
    ) -> id::TextureViewId {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (texture_guard, mut token) = hub.textures.read(&mut token);
        let texture = &texture_guard[texture_id];
        let device = &device_guard[texture.device_id.value];

        let (format, view_kind, range) = match desc {
            Some(desc) => {
                let kind = conv::map_texture_view_dimension(desc.dimension);
                let end_level = if desc.level_count == 0 {
                    texture.full_range.levels.end
                } else {
                    (desc.base_mip_level + desc.level_count) as u8
                };
                let end_layer = if desc.array_layer_count == 0 {
                    texture.full_range.layers.end
                } else {
                    (desc.base_array_layer + desc.array_layer_count) as u16
                };
                let range = hal::image::SubresourceRange {
                    aspects: texture.full_range.aspects,
                    levels: desc.base_mip_level as u8 .. end_level,
                    layers: desc.base_array_layer as u16 .. end_layer,
                };
                (desc.format, kind, range)
            }
            None => {
                let kind = match texture.kind {
                    hal::image::Kind::D1(_, 1) => hal::image::ViewKind::D1,
                    hal::image::Kind::D1(..) => hal::image::ViewKind::D1Array,
                    hal::image::Kind::D2(_, _, 1, _) => hal::image::ViewKind::D2,
                    hal::image::Kind::D2(..) => hal::image::ViewKind::D2Array,
                    hal::image::Kind::D3(..) => hal::image::ViewKind::D3,
                };
                (texture.format, kind, texture.full_range.clone())
            }
        };

        let raw = unsafe {
            device
                .raw
                .create_image_view(
                    &texture.raw,
                    view_kind,
                    conv::map_texture_format(format, device.features),
                    hal::format::Swizzle::NO,
                    range.clone(),
                )
                .unwrap()
        };

        let view = resource::TextureView {
            inner: resource::TextureViewInner::Native {
                raw,
                source_id: Stored {
                    value: texture_id,
                    ref_count: texture.life_guard.add_ref(),
                },
            },
            format: texture.format,
            extent: texture.kind.extent().at_level(range.levels.start),
            samples: texture.kind.num_samples(),
            range,
            life_guard: LifeGuard::new(),
        };
        let ref_count = view.life_guard.add_ref();

        let id = hub.texture_views.register_identity(id_in, view, &mut token);
        device.trackers
            .lock()
            .views.init(id, ref_count, PhantomData)
            .unwrap();
        id
    }

    pub fn texture_view_destroy<B: GfxBackend>(&self, texture_view_id: id::TextureViewId) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let device_id = {
            let (texture_guard, mut token) = hub.textures.read(&mut token);
            let (mut texture_view_guard, _) = hub.texture_views.write(&mut token);

            let view = &mut texture_view_guard[texture_view_id];
            view.life_guard.ref_count.take();
            match view.inner {
                resource::TextureViewInner::Native { ref source_id, .. } => {
                    texture_guard[source_id.value].device_id.value
                }
                resource::TextureViewInner::SwapChain { .. } => {
                    panic!("Can't destroy a swap chain image")
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources.texture_views.push(texture_view_id);
    }
}

impl<F: IdentityFilter<id::SamplerId>> Global<F> {
    pub fn device_create_sampler<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::SamplerDescriptor,
        id_in: F::Input,
    ) -> id::SamplerId {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];

        let info = hal::image::SamplerDesc {
            min_filter: conv::map_filter(desc.min_filter),
            mag_filter: conv::map_filter(desc.mag_filter),
            mip_filter: conv::map_filter(desc.mipmap_filter),
            wrap_mode: (
                conv::map_wrap(desc.address_mode_u),
                conv::map_wrap(desc.address_mode_v),
                conv::map_wrap(desc.address_mode_w),
            ),
            lod_bias: hal::image::Lod(0.0),
            lod_range: hal::image::Lod(desc.lod_min_clamp) .. hal::image::Lod(desc.lod_max_clamp),
            comparison: if desc.compare_function == resource::CompareFunction::Always {
                None
            } else {
                Some(conv::map_compare_function(desc.compare_function))
            },
            border: hal::image::PackedColor(0),
            normalized: true,
            anisotropic: hal::image::Anisotropic::Off, //TODO
        };

        let sampler = resource::Sampler {
            raw: unsafe { device.raw.create_sampler(&info).unwrap() },
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.add_ref(),
            },
            life_guard: LifeGuard::new(),
        };
        let ref_count = sampler.life_guard.add_ref();

        let id = hub.samplers.register_identity(id_in, sampler, &mut token);
        device.trackers
            .lock()
            .samplers.init(id, ref_count, PhantomData)
            .unwrap();
        id
    }

    pub fn sampler_destroy<B: GfxBackend>(&self, sampler_id: id::SamplerId) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let device_id = {
            let (mut sampler_guard, _) = hub.samplers.write(&mut token);
            let sampler = &mut sampler_guard[sampler_id];
            sampler.life_guard.ref_count.take();
            sampler.device_id.value
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources.samplers.push(sampler_id);
    }
}

impl<F: IdentityFilter<id::BindGroupLayoutId>> Global<F> {
    pub fn device_create_bind_group_layout<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &binding_model::BindGroupLayoutDescriptor,
        id_in: F::Input,
    ) -> id::BindGroupLayoutId {
        let mut token = Token::root();
        let hub = B::hub(self);
        let bindings = unsafe { slice::from_raw_parts(desc.bindings, desc.bindings_length) };
        let bindings_map: FastHashMap<_, _> = bindings
            .iter()
            .cloned()
            .map(|b| (b.binding, b))
            .collect();

        // TODO: deduplicate the bind group layouts at some level.
        // We can't do it right here, because in the remote scenario
        // the client need to know if the same ID can be used, or not.
        if false {
            let (bgl_guard, _) = hub.bind_group_layouts.read(&mut token);
            let bind_group_layout_id = bgl_guard
                .iter(device_id.backend())
                .find(|(_, bgl)| bgl.bindings == bindings_map);

            if let Some((id, _)) = bind_group_layout_id {
                return id;
            }
        }

        let raw_bindings = bindings
            .iter()
            .map(|binding| hal::pso::DescriptorSetLayoutBinding {
                binding: binding.binding,
                ty: conv::map_binding_type(binding),
                count: 1, //TODO: consolidate
                stage_flags: conv::map_shader_stage_flags(binding.visibility),
                immutable_samplers: false, // TODO
            })
            .collect::<Vec<_>>(); //TODO: avoid heap allocation

        let raw = unsafe {
            let (device_guard, _) = hub.devices.read(&mut token);
            device_guard[device_id]
                .raw
                .create_descriptor_set_layout(&raw_bindings, &[])
                .unwrap()
        };

        let layout = binding_model::BindGroupLayout {
            raw,
            bindings: bindings_map,
            desc_ranges: DescriptorRanges::from_bindings(&raw_bindings),
            dynamic_count: bindings.iter().filter(|b| b.dynamic).count(),
        };

        hub.bind_group_layouts
            .register_identity(id_in, layout, &mut token)
    }

    pub fn bind_group_layout_destroy<B: GfxBackend>(&self, bind_group_layout_id: id::BindGroupLayoutId) {
        let hub = B::hub(self);
        let mut token = Token::root();
        //TODO: track usage by GPU
        hub.bind_group_layouts.unregister(bind_group_layout_id, &mut token);
    }
}

impl<F: IdentityFilter<id::PipelineLayoutId>> Global<F> {
    pub fn device_create_pipeline_layout<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &binding_model::PipelineLayoutDescriptor,
        id_in: F::Input,
    ) -> id::PipelineLayoutId {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        let bind_group_layout_ids = unsafe {
            slice::from_raw_parts(desc.bind_group_layouts, desc.bind_group_layouts_length)
        };

        assert!(desc.bind_group_layouts_length <= (device.features.max_bind_groups as usize),
            "Cannot set a bind group which is beyond the `max_bind_groups` limit requested on device creation");

        // TODO: push constants
        let pipeline_layout = {
            let (bind_group_layout_guard, _) = hub.bind_group_layouts.read(&mut token);
            let descriptor_set_layouts = bind_group_layout_ids
                .iter()
                .map(|&id| &bind_group_layout_guard[id].raw);
            unsafe {
                device
                    .raw
                    .create_pipeline_layout(descriptor_set_layouts, &[])
            }
            .unwrap()
        };

        let layout = binding_model::PipelineLayout {
            raw: pipeline_layout,
            bind_group_layout_ids: bind_group_layout_ids.iter().cloned().collect(),
        };
        hub.pipeline_layouts
            .register_identity(id_in, layout, &mut token)
    }

    pub fn pipeline_layout_destroy<B: GfxBackend>(&self, pipeline_layout_id: id::PipelineLayoutId) {
        let hub = B::hub(self);
        let mut token = Token::root();
        //TODO: track usage by GPU
        hub.pipeline_layouts.unregister(pipeline_layout_id, &mut token);
    }
}

impl<F: IdentityFilter<id::BindGroupId>> Global<F> {
    pub fn device_create_bind_group<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &binding_model::BindGroupDescriptor,
        id_in: F::Input,
    ) -> id::BindGroupId {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        let (bind_group_layout_guard, mut token) = hub.bind_group_layouts.read(&mut token);
        let bind_group_layout = &bind_group_layout_guard[desc.layout];
        let bindings =
            unsafe { slice::from_raw_parts(desc.bindings, desc.bindings_length as usize) };
        assert_eq!(bindings.len(), bind_group_layout.bindings.len());

        let desc_set = unsafe {
            let mut desc_sets = ArrayVec::<[_; 1]>::new();
            device
                .desc_allocator
                .lock()
                .allocate(
                    &device.raw,
                    &bind_group_layout.raw,
                    bind_group_layout.desc_ranges,
                    1,
                    &mut desc_sets,
                )
                .unwrap();
            desc_sets.pop().unwrap()
        };

        // fill out the descriptors
        let mut used = TrackerSet::new(B::VARIANT);
        {
            let (buffer_guard, mut token) = hub.buffers.read(&mut token);
            let (texture_guard, mut token) = hub.textures.read(&mut token); //skip token
            let (texture_view_guard, mut token) = hub.texture_views.read(&mut token);
            let (sampler_guard, _) = hub.samplers.read(&mut token);

            //TODO: group writes into contiguous sections
            let mut writes = Vec::new();
            for b in bindings.iter() {
                let decl = bind_group_layout.bindings.get(&b.binding)
                    .expect("Failed to find binding declaration for binding");
                let descriptor = match b.resource {
                    binding_model::BindingResource::Buffer(ref bb) => {
                        let (alignment, usage) = match decl.ty {
                            binding_model::BindingType::UniformBuffer => {
                                (BIND_BUFFER_ALIGNMENT, resource::BufferUsage::UNIFORM)
                            }
                            binding_model::BindingType::StorageBuffer => {
                                (BIND_BUFFER_ALIGNMENT, resource::BufferUsage::STORAGE)
                            }
                            binding_model::BindingType::ReadonlyStorageBuffer => {
                                (BIND_BUFFER_ALIGNMENT, resource::BufferUsage::STORAGE_READ)
                            }
                            binding_model::BindingType::Sampler
                            | binding_model::BindingType::SampledTexture
                            | binding_model::BindingType::StorageTexture => {
                                panic!("Mismatched buffer binding for {:?}", decl)
                            }
                        };
                        assert_eq!(
                            bb.offset as hal::buffer::Offset % alignment,
                            0,
                            "Misaligned buffer offset {}",
                            bb.offset
                        );
                        let buffer = used
                            .buffers
                            .use_extend(&*buffer_guard, bb.buffer, (), usage)
                            .unwrap();
                        assert!(
                            buffer.usage.contains(usage),
                            "Expected buffer usage {:?}",
                            usage
                        );

                        let end = if bb.size == 0 {
                            None
                        } else {
                            let end = bb.offset + bb.size;
                            assert!(
                                end <= buffer.size,
                                "Bound buffer range {:?} does not fit in buffer size {}",
                                bb.offset .. end,
                                buffer.size
                            );
                            Some(end)
                        };

                        let range = Some(bb.offset) .. end;
                        hal::pso::Descriptor::Buffer(&buffer.raw, range)
                    }
                    binding_model::BindingResource::Sampler(id) => {
                        assert_eq!(decl.ty, binding_model::BindingType::Sampler);
                        let sampler = used
                            .samplers
                            .use_extend(&*sampler_guard, id, (), ())
                            .unwrap();
                        hal::pso::Descriptor::Sampler(&sampler.raw)
                    }
                    binding_model::BindingResource::TextureView(id) => {
                        let (usage, image_layout) = match decl.ty {
                            binding_model::BindingType::SampledTexture => (
                                resource::TextureUsage::SAMPLED,
                                hal::image::Layout::ShaderReadOnlyOptimal,
                            ),
                            binding_model::BindingType::StorageTexture => {
                                (resource::TextureUsage::STORAGE, hal::image::Layout::General)
                            }
                            _ => panic!("Mismatched texture binding for {:?}", decl),
                        };
                        let view = used
                            .views
                            .use_extend(&*texture_view_guard, id, (), ())
                            .unwrap();
                        match view.inner {
                            resource::TextureViewInner::Native {
                                ref raw,
                                ref source_id,
                            } => {
                                // Careful here: the texture may no longer have its own ref count,
                                // if it was deleted by the user.
                                let texture = &texture_guard[source_id.value];
                                used.textures
                                    .change_extend(
                                        source_id.value,
                                        &source_id.ref_count,
                                        view.range.clone(),
                                        usage,
                                    )
                                    .unwrap();
                                assert!(texture.usage.contains(usage));

                                hal::pso::Descriptor::Image(raw, image_layout)
                            }
                            resource::TextureViewInner::SwapChain { .. } => {
                                panic!("Unable to create a bind group with a swap chain image")
                            }
                        }
                    }
                };
                writes.alloc().init(hal::pso::DescriptorSetWrite {
                    set: desc_set.raw(),
                    binding: b.binding,
                    array_offset: 0, //TODO
                    descriptors: iter::once(descriptor),
                });
            }

            unsafe {
                device.raw.write_descriptor_sets(writes);
            }
        }

        let bind_group = binding_model::BindGroup {
            raw: desc_set,
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.add_ref(),
            },
            layout_id: desc.layout,
            life_guard: LifeGuard::new(),
            used,
            dynamic_count: bind_group_layout.dynamic_count,
        };
        let ref_count = bind_group.life_guard.add_ref();


        let id = hub
            .bind_groups
            .register_identity(id_in, bind_group, &mut token);
        log::debug!("Bind group {:?} {:#?}",
            id, hub.bind_groups.read(&mut token).0[id].used);

        device
            .trackers
            .lock()
            .bind_groups
            .init(id, ref_count, PhantomData)
            .unwrap();
        id
    }

    pub fn bind_group_destroy<B: GfxBackend>(&self, bind_group_id: id::BindGroupId) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let device_id = {
            let (mut bind_group_guard, _) = hub.bind_groups.write(&mut token);
            let bind_group = &mut bind_group_guard[bind_group_id];
            bind_group.life_guard.ref_count.take();
            bind_group.device_id.value
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources.bind_groups.push(bind_group_id);
    }
}

impl<F: IdentityFilter<id::ShaderModuleId>> Global<F> {
    pub fn device_create_shader_module<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &pipeline::ShaderModuleDescriptor,
        id_in: F::Input,
    ) -> id::ShaderModuleId {
        let hub = B::hub(self);
        let mut token = Token::root();

        let spv = unsafe { slice::from_raw_parts(desc.code.bytes, desc.code.length) };
        let shader = {
            let (device_guard, _) = hub.devices.read(&mut token);
            ShaderModule {
                raw: unsafe {
                    device_guard[device_id]
                        .raw
                        .create_shader_module(spv)
                        .unwrap()
                },
            }
        };
        hub.shader_modules
            .register_identity(id_in, shader, &mut token)
    }

    pub fn shader_module_destroy<B: GfxBackend>(&self, shader_module_id: id::ShaderModuleId) {
        let hub = B::hub(self);
        let mut token = Token::root();
        //TODO: track usage by GPU
        hub.shader_modules.unregister(shader_module_id, &mut token);
    }
}

impl<F: IdentityFilter<id::CommandEncoderId>> Global<F> {
    pub fn device_create_command_encoder<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        _desc: &command::CommandEncoderDescriptor,
        id_in: F::Input,
    ) -> id::CommandEncoderId {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];

        let dev_stored = Stored {
            value: device_id,
            ref_count: device.life_guard.add_ref(),
        };

        let lowest_active_index = device
            .lock_life(&mut token)
            .lowest_active_submission();

        let mut comb = device
            .com_allocator
            .allocate(dev_stored, &device.raw, device.features, lowest_active_index);
        unsafe {
            comb.raw.last_mut().unwrap().begin_primary(
                hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
            );
        }

        hub.command_buffers
            .register_identity(id_in, comb, &mut token)
    }

    pub fn command_encoder_destroy<B: GfxBackend>(
        &self, command_encoder_id: id::CommandEncoderId
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let comb = {
            let (mut command_buffer_guard, _) = hub.command_buffers.write(&mut token);
            command_buffer_guard.remove(command_encoder_id).unwrap()
        };

        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let device = &mut device_guard[comb.device_id.value];
        device.temp_suspected.clear();
        // As the tracker is cleared/dropped, we need to consider all the resources
        // that it references for destruction in the next GC pass.
        {
            let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
            let (buffer_guard, mut token) = hub.buffers.read(&mut token);
            let (texture_guard, mut token) = hub.textures.read(&mut token);
            let (texture_view_guard, mut token) = hub.texture_views.read(&mut token);
            let (sampler_guard, _) = hub.samplers.read(&mut token);

            for id in comb.trackers.buffers.used() {
                if buffer_guard[id].life_guard.ref_count.is_none() {
                    device.temp_suspected.buffers.push(id);
                }
            }
            for id in comb.trackers.textures.used() {
                if texture_guard[id].life_guard.ref_count.is_none() {
                    device.temp_suspected.textures.push(id);
                }
            }
            for id in comb.trackers.views.used() {
                if texture_view_guard[id].life_guard.ref_count.is_none() {
                    device.temp_suspected.texture_views.push(id);
                }
            }
            for id in comb.trackers.bind_groups.used() {
                if bind_group_guard[id].life_guard.ref_count.is_none() {
                    device.temp_suspected.bind_groups.push(id);
                }
            }
            for id in comb.trackers.samplers.used() {
                if sampler_guard[id].life_guard.ref_count.is_none() {
                    device.temp_suspected.samplers.push(id);
                }
            }
        }

        device
            .lock_life(&mut token)
            .suspected_resources.extend(&device.temp_suspected);
        device.com_allocator.discard(comb);
    }
}

impl<F: IdentityFilter<id::CommandBufferId>> Global<F> {
    pub fn command_buffer_destroy<B: GfxBackend>(
        &self, command_buffer_id: id::CommandBufferId
    ) {
        self.command_encoder_destroy::<B>(command_buffer_id)
    }
}

impl<F: AllIdentityFilter + IdentityFilter<id::CommandBufferId>> Global<F> {
    pub fn queue_submit<B: GfxBackend>(
        &self,
        queue_id: id::QueueId,
        command_buffer_ids: &[id::CommandBufferId],
    ) {
        let hub = B::hub(self);

        let (submit_index, fence) = {
            let mut token = Token::root();
            let (mut device_guard, mut token) = hub.devices.write(&mut token);
            let device = &mut device_guard[queue_id];
            device.temp_suspected.clear();

            let submit_index = 1 + device
                .life_guard
                .submission_index
                .fetch_add(1, Ordering::Relaxed);

            let (mut swap_chain_guard, mut token) = hub.swap_chains.write(&mut token);
            let (mut command_buffer_guard, mut token) = hub.command_buffers.write(&mut token);
            let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
            let (buffer_guard, mut token) = hub.buffers.read(&mut token);
            let (texture_guard, mut token) = hub.textures.read(&mut token);
            let (texture_view_guard, mut token) = hub.texture_views.read(&mut token);
            let (sampler_guard, _) = hub.samplers.read(&mut token);

            //Note: locking the trackers has to be done after the storages
            let mut signal_swapchain_semaphores = SmallVec::<[_; 1]>::new();
            let mut trackers = device.trackers.lock();

            //TODO: if multiple command buffers are submitted, we can re-use the last
            // native command buffer of the previous chain instead of always creating
            // a temporary one, since the chains are not finished.

            // finish all the command buffers first
            for &cmb_id in command_buffer_ids {
                let comb = &mut command_buffer_guard[cmb_id];

                if let Some((sc_id, fbo)) = comb.used_swap_chain.take() {
                    let sc = &mut swap_chain_guard[sc_id.value];
                    if sc.acquired_framebuffers.is_empty() {
                        signal_swapchain_semaphores.push(sc_id.value);
                    }
                    sc.acquired_framebuffers.push(fbo);
                }

                // optimize the tracked states
                comb.trackers.optimize();

                // update submission IDs
                for id in comb.trackers.buffers.used() {
                    assert!(buffer_guard[id].pending_mapping.is_none());
                    if !buffer_guard[id].life_guard.use_at(submit_index) {
                        device.temp_suspected.buffers.push(id);
                    }
                }
                for id in comb.trackers.textures.used() {
                    if !texture_guard[id].life_guard.use_at(submit_index) {
                        device.temp_suspected.textures.push(id);
                    }
                }
                for id in comb.trackers.views.used() {
                    if !texture_view_guard[id].life_guard.use_at(submit_index) {
                        device.temp_suspected.texture_views.push(id);
                    }
                }
                for id in comb.trackers.bind_groups.used() {
                    if !bind_group_guard[id].life_guard.use_at(submit_index) {
                        device.temp_suspected.bind_groups.push(id);
                    }
                }
                for id in comb.trackers.samplers.used() {
                    if !sampler_guard[id].life_guard.use_at(submit_index) {
                        device.temp_suspected.samplers.push(id);
                    }
                }

                // execute resource transitions
                let mut transit = device.com_allocator.extend(comb);
                unsafe {
                    transit.begin_primary(
                        hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
                    );
                }
                log::trace!("Stitching command buffer {:?} before submission", cmb_id);
                command::CommandBuffer::insert_barriers(
                    &mut transit,
                    &mut *trackers,
                    &comb.trackers,
                    &*buffer_guard,
                    &*texture_guard,
                );
                unsafe {
                    transit.finish();
                }
                comb.raw.insert(0, transit);
                unsafe {
                    comb.raw.last_mut().unwrap().finish();
                }
            }

            log::debug!("Device after submission {}: {:#?}", submit_index, trackers);

            // now prepare the GPU submission
            let fence = device.raw.create_fence(false).unwrap();
            let submission = hal::queue::Submission {
                command_buffers: command_buffer_ids
                    .iter()
                    .flat_map(|&cmb_id| &command_buffer_guard[cmb_id].raw),
                wait_semaphores: Vec::new(),
                signal_semaphores: signal_swapchain_semaphores
                    .into_iter()
                    .map(|sc_id| &swap_chain_guard[sc_id].semaphore),

            };

            unsafe {
                device.queue_group.queues[0].submit(submission, Some(&fence));
            }

            (submit_index, fence)
        };

        // No need for write access to the device from here on out
        let callbacks = {
            let mut token = Token::root();
            let (device_guard, mut token) = hub.devices.read(&mut token);
            let device = &device_guard[queue_id];

            let callbacks = device.maintain(self, false, &mut token);
            device
                .lock_life(&mut token)
                .track_submission(submit_index, fence, &device.temp_suspected);

            // finally, return the command buffers to the allocator
            for &cmb_id in command_buffer_ids {
                let (cmd_buf, _) = hub.command_buffers.unregister(cmb_id, &mut token);
                device.com_allocator.after_submit(cmd_buf, submit_index);
            }

            callbacks
        };

        fire_map_callbacks(callbacks);
    }
}

impl<F: IdentityFilter<id::RenderPipelineId>> Global<F> {
    pub fn device_create_render_pipeline<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &pipeline::RenderPipelineDescriptor,
        id_in: F::Input,
    ) -> id::RenderPipelineId {
        let hub = B::hub(self);
        let mut token = Token::root();

        let sc = desc.sample_count;
        assert!(
            sc == 1 || sc == 2 || sc == 4 || sc == 8 || sc == 16 || sc == 32,
            "Invalid sample_count of {}",
            sc
        );
        let sc = sc as u8;

        let color_states =
            unsafe { slice::from_raw_parts(desc.color_states, desc.color_states_length) };
        let depth_stencil_state = unsafe { desc.depth_stencil_state.as_ref() };

        let rasterizer = conv::map_rasterization_state_descriptor(
            &unsafe { desc.rasterization_state.as_ref() }
                .cloned()
                .unwrap_or_default(),
        );

        let desc_vbs = unsafe {
            slice::from_raw_parts(
                desc.vertex_input.vertex_buffers,
                desc.vertex_input.vertex_buffers_length,
            )
        };
        let mut vertex_strides = Vec::with_capacity(desc_vbs.len());
        let mut vertex_buffers = Vec::with_capacity(desc_vbs.len());
        let mut attributes = Vec::new();
        for (i, vb_state) in desc_vbs.iter().enumerate() {
            vertex_strides
                .alloc()
                .init((vb_state.stride, vb_state.step_mode));
            if vb_state.attributes_length == 0 {
                continue;
            }
            vertex_buffers.alloc().init(hal::pso::VertexBufferDesc {
                binding: i as u32,
                stride: vb_state.stride as u32,
                rate: match vb_state.step_mode {
                    pipeline::InputStepMode::Vertex => hal::pso::VertexInputRate::Vertex,
                    pipeline::InputStepMode::Instance => hal::pso::VertexInputRate::Instance(1),
                },
            });
            let desc_atts =
                unsafe { slice::from_raw_parts(vb_state.attributes, vb_state.attributes_length) };
            for attribute in desc_atts {
                assert_eq!(0, attribute.offset >> 32);
                attributes.alloc().init(hal::pso::AttributeDesc {
                    location: attribute.shader_location,
                    binding: i as u32,
                    element: hal::pso::Element {
                        format: conv::map_vertex_format(attribute.format),
                        offset: attribute.offset as u32,
                    },
                });
            }
        }

        let input_assembler = hal::pso::InputAssemblerDesc {
            primitive: conv::map_primitive_topology(desc.primitive_topology),
            with_adjacency: false,
            restart_index: None, //TODO
        };

        let blender = hal::pso::BlendDesc {
            logic_op: None, // TODO
            targets: color_states
                .iter()
                .map(conv::map_color_state_descriptor)
                .collect(),
        };
        let depth_stencil = depth_stencil_state
            .map(conv::map_depth_stencil_state_descriptor)
            .unwrap_or_default();

        let multisampling: Option<hal::pso::Multisampling> = if sc == 1 {
            None
        } else {
            Some(hal::pso::Multisampling {
                rasterization_samples: sc,
                sample_shading: None,
                sample_mask: desc.sample_mask as u64,
                alpha_coverage: desc.alpha_to_coverage_enabled,
                alpha_to_one: false,
            })
        };

        // TODO
        let baked_states = hal::pso::BakedStates {
            viewport: None,
            scissor: None,
            blend_color: None,
            depth_bounds: None,
        };

        let raw_pipeline = {
            let (device_guard, mut token) = hub.devices.read(&mut token);
            let device = &device_guard[device_id];
            let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);
            let layout = &pipeline_layout_guard[desc.layout].raw;
            let (shader_module_guard, _) = hub.shader_modules.read(&mut token);

            let rp_key = RenderPassKey {
                colors: color_states
                    .iter()
                    .map(|at| hal::pass::Attachment {
                        format: Some(conv::map_texture_format(at.format, device.features)),
                        samples: sc,
                        ops: hal::pass::AttachmentOps::PRESERVE,
                        stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                        layouts: hal::image::Layout::General .. hal::image::Layout::General,
                    })
                    .collect(),
                // We can ignore the resolves as the vulkan specs says:
                // As an additional special case, if two render passes have a single subpass,
                // they are compatible even if they have different resolve attachment references
                // or depth/stencil resolve modes but satisfy the other compatibility conditions.
                resolves: ArrayVec::new(),
                depth_stencil: depth_stencil_state.map(|at| hal::pass::Attachment {
                    format: Some(conv::map_texture_format(at.format, device.features)),
                    samples: sc,
                    ops: hal::pass::AttachmentOps::PRESERVE,
                    stencil_ops: hal::pass::AttachmentOps::PRESERVE,
                    layouts: hal::image::Layout::General .. hal::image::Layout::General,
                }),
            };

            let mut render_pass_cache = device.render_passes.lock();
            let main_pass = match render_pass_cache.entry(rp_key) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => {
                    let color_ids = [
                        (0, hal::image::Layout::ColorAttachmentOptimal),
                        (1, hal::image::Layout::ColorAttachmentOptimal),
                        (2, hal::image::Layout::ColorAttachmentOptimal),
                        (3, hal::image::Layout::ColorAttachmentOptimal),
                    ];

                    let depth_id = (
                        desc.color_states_length,
                        hal::image::Layout::DepthStencilAttachmentOptimal,
                    );

                    let subpass = hal::pass::SubpassDesc {
                        colors: &color_ids[.. desc.color_states_length],
                        depth_stencil: depth_stencil_state.map(|_| &depth_id),
                        inputs: &[],
                        resolves: &[],
                        preserves: &[],
                    };

                    let pass = unsafe {
                        device
                            .raw
                            .create_render_pass(e.key().all(), &[subpass], &[])
                    }
                    .unwrap();
                    e.insert(pass)
                }
            };

            let vertex = hal::pso::EntryPoint::<B> {
                entry: unsafe { ffi::CStr::from_ptr(desc.vertex_stage.entry_point) }
                    .to_str()
                    .to_owned()
                    .unwrap(), // TODO
                module: &shader_module_guard[desc.vertex_stage.module].raw,
                specialization: hal::pso::Specialization::EMPTY,
            };
            let fragment =
                unsafe { desc.fragment_stage.as_ref() }.map(|stage| hal::pso::EntryPoint::<B> {
                    entry: unsafe { ffi::CStr::from_ptr(stage.entry_point) }
                        .to_str()
                        .to_owned()
                        .unwrap(), // TODO
                    module: &shader_module_guard[stage.module].raw,
                    specialization: hal::pso::Specialization::EMPTY,
                });

            let shaders = hal::pso::GraphicsShaderSet {
                vertex,
                hull: None,
                domain: None,
                geometry: None,
                fragment,
            };

            let subpass = hal::pass::Subpass {
                index: 0,
                main_pass,
            };

            // TODO
            let flags = hal::pso::PipelineCreationFlags::empty();
            // TODO
            let parent = hal::pso::BasePipeline::None;

            let pipeline_desc = hal::pso::GraphicsPipelineDesc {
                shaders,
                rasterizer,
                vertex_buffers,
                attributes,
                input_assembler,
                blender,
                depth_stencil,
                multisampling,
                baked_states,
                layout,
                subpass,
                flags,
                parent,
            };

            // TODO: cache
            unsafe {
                device
                    .raw
                    .create_graphics_pipeline(&pipeline_desc, None)
                    .unwrap()
            }
        };

        let pass_context = RenderPassContext {
            colors: color_states.iter().map(|state| state.format).collect(),
            resolves: ArrayVec::new(),
            depth_stencil: depth_stencil_state.map(|state| state.format),
        };

        let mut flags = pipeline::PipelineFlags::empty();
        for state in color_states {
            if state.color_blend.uses_color() | state.alpha_blend.uses_color() {
                flags |= pipeline::PipelineFlags::BLEND_COLOR;
            }
        }
        if let Some(ds) = depth_stencil_state {
            if ds.needs_stencil_reference() {
                flags |= pipeline::PipelineFlags::STENCIL_REFERENCE;
            }
        }

        let pipeline = pipeline::RenderPipeline {
            raw: raw_pipeline,
            layout_id: desc.layout,
            pass_context,
            flags,
            index_format: desc.vertex_input.index_format,
            vertex_strides,
            sample_count: sc,
        };

        hub.render_pipelines
            .register_identity(id_in, pipeline, &mut token)
    }

    pub fn render_pipeline_destroy<B: GfxBackend>(&self, render_pipeline_id: id::RenderPipelineId) {
        let hub = B::hub(self);
        let mut token = Token::root();
        //TODO: track usage by GPU
        hub.render_pipelines.unregister(render_pipeline_id, &mut token);
    }
}

impl<F: IdentityFilter<id::ComputePipelineId>> Global<F> {
    pub fn device_create_compute_pipeline<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &pipeline::ComputePipelineDescriptor,
        id_in: F::Input,
    ) -> id::ComputePipelineId {
        let hub = B::hub(self);
        let mut token = Token::root();

        let raw_pipeline = {
            let (device_guard, mut token) = hub.devices.read(&mut token);
            let device = &device_guard[device_id].raw;
            let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);
            let layout = &pipeline_layout_guard[desc.layout].raw;
            let pipeline_stage = &desc.compute_stage;
            let (shader_module_guard, _) = hub.shader_modules.read(&mut token);

            let shader = hal::pso::EntryPoint::<B> {
                entry: unsafe { ffi::CStr::from_ptr(pipeline_stage.entry_point) }
                    .to_str()
                    .to_owned()
                    .unwrap(), // TODO
                module: &shader_module_guard[pipeline_stage.module].raw,
                specialization: hal::pso::Specialization::EMPTY,
            };

            // TODO
            let flags = hal::pso::PipelineCreationFlags::empty();
            // TODO
            let parent = hal::pso::BasePipeline::None;

            let pipeline_desc = hal::pso::ComputePipelineDesc {
                shader,
                layout,
                flags,
                parent,
            };

            unsafe {
                device
                    .create_compute_pipeline(&pipeline_desc, None)
                    .unwrap()
            }
        };

        let pipeline = pipeline::ComputePipeline {
            raw: raw_pipeline,
            layout_id: desc.layout,
        };
        hub.compute_pipelines
            .register_identity(id_in, pipeline, &mut token)
    }

    pub fn compute_pipeline_destroy<B: GfxBackend>(&self, compute_pipeline_id: id::ComputePipelineId) {
        let hub = B::hub(self);
        let mut token = Token::root();
        //TODO: track usage by GPU
        hub.compute_pipelines.unregister(compute_pipeline_id, &mut token);
    }
}

fn validate_swap_chain_descriptor(
    config: &mut hal::window::SwapchainConfig,
    caps: &hal::window::SurfaceCapabilities,
) {
    let width = config.extent.width;
    let height = config.extent.height;
    if width < caps.extents.start().width
        || width > caps.extents.end().width
        || height < caps.extents.start().height
        || height > caps.extents.end().height
    {
        log::warn!(
            "Requested size {}x{} is outside of the supported range: {:?}",
            width,
            height,
            caps.extents
        );
    }
    if !caps.present_modes.contains(config.present_mode) {
        log::warn!(
            "Surface does not support present mode: {:?}, falling back to {:?}",
            config.present_mode,
            hal::window::PresentMode::FIFO
        );
        config.present_mode = hal::window::PresentMode::FIFO;
    }
}

impl<F: IdentityFilter<id::SwapChainId>> Global<F> {
    pub fn device_create_swap_chain<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        surface_id: id::SurfaceId,
        desc: &swap_chain::SwapChainDescriptor,
    ) -> id::SwapChainId {
        log::info!("creating swap chain {:?}", desc);
        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
        let (adapter_guard, mut token) = hub.adapters.read(&mut token);
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut swap_chain_guard, _) = hub.swap_chains.write(&mut token);
        let device = &device_guard[device_id];
        let surface = &mut surface_guard[surface_id];

        let (caps, formats) = {
            let suf = B::get_surface_mut(surface);
            let adapter = &adapter_guard[device.adapter_id];
            assert!(suf.supports_queue_family(&adapter.raw.queue_families[0]));
            let formats = suf.supported_formats(&adapter.raw.physical_device);
            let caps = suf.capabilities(&adapter.raw.physical_device);
            (caps, formats)
        };
        let num_frames = swap_chain::DESIRED_NUM_FRAMES
            .max(*caps.image_count.start())
            .min(*caps.image_count.end());
        let mut config = desc.to_hal(num_frames, device.features);
        if let Some(formats) = formats {
            assert!(
                formats.contains(&config.format),
                "Requested format {:?} is not in supported list: {:?}",
                config.format,
                formats
            );
        }
        validate_swap_chain_descriptor(&mut config, &caps);

        unsafe {
            B::get_surface_mut(surface)
                .configure_swapchain(&device.raw, config)
                .unwrap();
        }

        let sc_id = surface_id.to_swap_chain_id(B::VARIANT);
        if let Some(sc) = swap_chain_guard.remove(sc_id) {
            unsafe {
                device.raw.destroy_semaphore(sc.semaphore);
            }
        }
        let swap_chain = swap_chain::SwapChain {
            life_guard: LifeGuard::new(),
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.add_ref(),
            },
            desc: desc.clone(),
            num_frames,
            semaphore: device.raw.create_semaphore().unwrap(),
            acquired_view_id: None,
            acquired_framebuffers: Vec::new(),
        };
        swap_chain_guard.insert(sc_id, swap_chain);
        sc_id
    }
}

impl<F: AllIdentityFilter> Global<F> {
    pub fn device_poll<B: GfxBackend>(&self, device_id: id::DeviceId, force_wait: bool) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let callbacks = {
            let (device_guard, mut token) = hub.devices.read(&mut token);
            device_guard[device_id].maintain(self, force_wait, &mut token)
        };
        fire_map_callbacks(callbacks);
    }

    fn poll_devices<B: GfxBackend>(
        &self,
        force_wait: bool,
        callbacks: &mut Vec<BufferMapPendingCallback>,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        for (_, device) in device_guard.iter(B::VARIANT) {
            let cbs = device.maintain(self, force_wait, &mut token);
            callbacks.extend(cbs);
        }
    }

    pub fn poll_all_devices(&self, force_wait: bool) {
        use crate::backend;
        let mut callbacks = Vec::new();

        #[cfg(any(
            not(any(target_os = "ios", target_os = "macos")),
            feature = "gfx-backend-vulkan"
        ))]
        self.poll_devices::<backend::Vulkan>(force_wait, &mut callbacks);
        #[cfg(windows)]
        self.poll_devices::<backend::Dx11>(force_wait, &mut callbacks);
        #[cfg(windows)]
        self.poll_devices::<backend::Dx12>(force_wait, &mut callbacks);
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        self.poll_devices::<backend::Metal>(force_wait, &mut callbacks);

        fire_map_callbacks(callbacks);
    }
}

impl<F: AllIdentityFilter + IdentityFilter<id::DeviceId>> Global<F> {
    pub fn device_destroy<B: GfxBackend>(&self, device_id: id::DeviceId) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device, mut token) = hub.devices.unregister(device_id, &mut token);
        device.maintain(self, true, &mut token);
        drop(token);
        device.com_allocator.destroy(&device.raw);
    }
}

impl<F> Global<F> {
    pub fn buffer_map_async<B: GfxBackend>(
        &self,
        buffer_id: id::BufferId,
        usage: resource::BufferUsage,
        range: std::ops::Range<BufferAddress>,
        operation: resource::BufferMapOperation,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);

        let (device_id, ref_count) = {
            let (mut buffer_guard, _) = hub.buffers.write(&mut token);
            let buffer = &mut buffer_guard[buffer_id];

            if usage.contains(resource::BufferUsage::MAP_READ) {
                assert!(buffer.usage.contains(resource::BufferUsage::MAP_READ));
            }

            if usage.contains(resource::BufferUsage::MAP_WRITE) {
                assert!(buffer.usage.contains(resource::BufferUsage::MAP_WRITE));
            }

            if buffer.pending_mapping.is_some() {
                operation.call_error();
                return;
            }

            buffer.pending_mapping = Some(resource::BufferPendingMapping {
                range,
                op: operation,
                parent_ref_count: buffer.life_guard.add_ref(),
            });
            (buffer.device_id.value, buffer.life_guard.add_ref())
        };

        let device = &device_guard[device_id];

        device
            .trackers
            .lock()
            .buffers
            .change_replace(buffer_id, &ref_count, (), usage);

        device
            .lock_life(&mut token)
            .map(buffer_id, ref_count);
    }

    pub fn buffer_unmap<B: GfxBackend>(&self, buffer_id: id::BufferId) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut buffer_guard, _) = hub.buffers.write(&mut token);
        let buffer = &mut buffer_guard[buffer_id];

        unmap_buffer(
            &device_guard[buffer.device_id.value].raw,
            buffer,
        );
    }
}
