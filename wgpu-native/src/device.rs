use crate::{binding_model, command, conv, pipeline, resource, swap_chain};
use crate::hub::HUB;
use crate::track::{DummyUsage, Stitch, TrackerSet, TrackPermit};
use crate::{
    LifeGuard, RefCount, Stored, SubmissionIndex,
    BufferMapAsyncStatus, BufferMapOperation,
};
use crate::{
    BufferId, CommandBufferId, AdapterId, DeviceId, QueueId,
    BindGroupId, TextureId, TextureViewId, SurfaceId,
};
#[cfg(feature = "local")]
use crate::{
    BindGroupLayoutId, PipelineLayoutId, SamplerId, SwapChainId,
    ShaderModuleId, CommandEncoderId, RenderPipelineId, ComputePipelineId,
};

use arrayvec::ArrayVec;
use back;
use hal::backend::FastHashMap;
use hal::command::RawCommandBuffer;
use hal::queue::RawCommandQueue;
use hal::{
    self,
    DescriptorPool as _DescriptorPool,
    Device as _Device,
    Surface as _Surface,
};
use log::{info, trace};
//use rendy_memory::{allocator, Config, Heaps};
use parking_lot::{Mutex};

use std::{ffi, iter, slice};
use std::collections::hash_map::Entry;
use std::sync::atomic::Ordering;


pub fn all_buffer_stages() -> hal::pso::PipelineStage {
    use hal::pso::PipelineStage as Ps;
    Ps::DRAW_INDIRECT |
    Ps::VERTEX_INPUT |
    Ps::VERTEX_SHADER |
    Ps::FRAGMENT_SHADER |
    Ps::COMPUTE_SHADER |
    Ps::TRANSFER |
    Ps::HOST
}
pub fn all_image_stages() -> hal::pso::PipelineStage {
    use hal::pso::PipelineStage as Ps;
    Ps::EARLY_FRAGMENT_TESTS |
    Ps::LATE_FRAGMENT_TESTS |
    Ps::COLOR_ATTACHMENT_OUTPUT |
    Ps::VERTEX_SHADER |
    Ps::FRAGMENT_SHADER |
    Ps::COMPUTE_SHADER |
    Ps::TRANSFER
}

#[derive(Clone, Debug, Hash, PartialEq)]
pub(crate) struct AttachmentData<T> {
    pub colors: ArrayVec<[T; 4]>,
    pub depth_stencil: Option<T>,
}
impl<T: PartialEq> Eq for AttachmentData<T> {}
impl<T> AttachmentData<T> {
    pub(crate) fn all(&self) -> impl Iterator<Item = &T> {
        self.colors.iter().chain(&self.depth_stencil)
    }
}

pub(crate) type RenderPassKey = AttachmentData<hal::pass::Attachment>;
pub(crate) type FramebufferKey = AttachmentData<TextureViewId>;
pub(crate) type RenderPassContext = AttachmentData<resource::TextureFormat>;

#[derive(Debug, PartialEq)]
enum ResourceId {
    Buffer(BufferId),
    Texture(TextureId),
    TextureView(TextureViewId),
}

enum NativeResource<B: hal::Backend> {
    Buffer(B::Buffer, B::Memory),
    Image(B::Image, B::Memory),
    ImageView(B::ImageView),
    Framebuffer(B::Framebuffer),
}

struct ActiveSubmission<B: hal::Backend> {
    index: SubmissionIndex,
    fence: B::Fence,
    // Note: we keep the associated ID here in order to be able to check
    // at any point what resources are used in a submission.
    resources: Vec<(Option<ResourceId>, NativeResource<B>)>,
    mapped: Vec<BufferId>,
}

struct DestroyedResources<B: hal::Backend> {
    /// Resources that the user has requested be mapped, but are still in use.
    mapped: Vec<Stored<BufferId>>,
    /// Resources that are destroyed by the user but still referenced by
    /// other objects or command buffers.
    referenced: Vec<(ResourceId, RefCount)>,
    /// Resources that are not referenced any more but still used by GPU.
    /// Grouped by submissions associated with a fence and a submission index.
    active: Vec<ActiveSubmission<B>>,
    /// Resources that are neither referenced or used, just pending
    /// actual deletion.
    free: Vec<NativeResource<B>>,
    ready_to_map: Vec<BufferId>,
}

unsafe impl<B: hal::Backend> Send for DestroyedResources<B> {}
unsafe impl<B: hal::Backend> Sync for DestroyedResources<B> {}

impl<B: hal::Backend> DestroyedResources<B> {
    fn destroy(&mut self, resource_id: ResourceId, ref_count: RefCount) {
        debug_assert!(!self.referenced.iter().any(|r| r.0 == resource_id));
        self.referenced.push((resource_id, ref_count));
    }

    fn map(&mut self, buffer: BufferId, ref_count: RefCount) {
        self.mapped.push(Stored{value: buffer, ref_count});
    }

    /// Returns the last submission index that is done.
    fn cleanup(&mut self, device: &B::Device) -> SubmissionIndex {
        let mut last_done = 0;

        for i in (0..self.active.len()).rev() {
            if unsafe {
                device.get_fence_status(&self.active[i].fence).unwrap()
            } {
                let a = self.active.swap_remove(i);
                last_done = last_done.max(a.index);
                self.free.extend(a.resources.into_iter().map(|(_, r)| r));
                unsafe {
                    device.destroy_fence(a.fence);
                }
            }
        }

        for resource in self.free.drain(..) {
            match resource {
                NativeResource::Buffer(raw, memory) => unsafe {
                    device.destroy_buffer(raw);
                    device.free_memory(memory);
                },
                NativeResource::Image(raw, memory) => unsafe {
                    device.destroy_image(raw);
                    device.free_memory(memory);
                },
                NativeResource::ImageView(raw) => unsafe {
                    device.destroy_image_view(raw);
                },
                NativeResource::Framebuffer(raw) => unsafe {
                    device.destroy_framebuffer(raw);
                },
            }
        }

        last_done
    }
}

impl DestroyedResources<back::Backend> {
    fn triage_referenced(
        &mut self,
        trackers: &mut TrackerSet,
    ) {
        for i in (0..self.referenced.len()).rev() {
            let num_refs = self.referenced[i].1.load();
            // Before destruction, a resource is expected to have the following strong refs:
            //  1. in resource itself
            //  2. in the device tracker
            //  3. in this list
            if num_refs <= 3 {
                let resource_id = self.referenced.swap_remove(i).0;
                assert_eq!(num_refs, 3, "Resource {:?} misses some references", resource_id);
                let (life_guard, resource) = match resource_id {
                    ResourceId::Buffer(id) => {
                        trackers.buffers.remove(id);
                        let buf = HUB.buffers.unregister(id);
                        (buf.life_guard, NativeResource::Buffer(buf.raw, buf.memory))
                    }
                    ResourceId::Texture(id) => {
                        trackers.textures.remove(id);
                        let tex = HUB.textures.unregister(id);
                        let memory = match tex.placement {
                            // swapchain-owned images don't need explicit destruction
                            resource::TexturePlacement::SwapChain(_) => continue,
                            resource::TexturePlacement::Memory(mem) => mem,
                            resource::TexturePlacement::Void => unreachable!(),
                        };
                        (tex.life_guard, NativeResource::Image(tex.raw, memory))
                    }
                    ResourceId::TextureView(id) => {
                        trackers.views.remove(id);
                        let view = HUB.texture_views.unregister(id);
                        (view.life_guard, NativeResource::ImageView(view.raw))
                    }
                };

                let submit_index = life_guard.submission_index.load(Ordering::Acquire);
                match self.active
                    .iter_mut()
                    .find(|a| a.index == submit_index)
                {
                    Some(a) => a.resources.push((Some(resource_id), resource)),
                    None => self.free.push(resource),
                }
            }
        }

        let buffer_guard = HUB.buffers.read();

        for i in (0..self.mapped.len()).rev() {
            let resource_id = self.mapped.swap_remove(i).value;
            let buf = &buffer_guard[resource_id];

            let usage = match buf.pending_map_operation {
                Some(BufferMapOperation::Read(..)) => resource::BufferUsageFlags::MAP_READ,
                Some(BufferMapOperation::Write(..)) => resource::BufferUsageFlags::MAP_WRITE,
                _ => unreachable!(),
            };
            trackers.buffers.get_with_replaced_usage(&buffer_guard, resource_id, usage).unwrap();
            
            let submit_index = buf.life_guard.submission_index.load(Ordering::Acquire);
            self.active
                .iter_mut()
                .find(|a| a.index == submit_index)
                .map_or(&mut self.ready_to_map, |a| &mut a.mapped)
                .push(resource_id);
        }
    }

    fn triage_framebuffers(
        &mut self,
        framebuffers: &mut FastHashMap<FramebufferKey, <back::Backend as hal::Backend>::Framebuffer>,
    ) {
        let texture_view_guard = HUB.texture_views.read();
        let remove_list = framebuffers
            .keys()
            .filter_map(|key| {
                let mut last_submit: SubmissionIndex = 0;
                for &at in key.all() {
                    if texture_view_guard.contains(at) {
                        return None
                    }
                    // This attachment is no longer registered.
                    // Let's see if it's used by any of the active submissions.
                    let res_id = &Some(ResourceId::TextureView(at));
                    for a in &self.active {
                        if a.resources.iter().any(|&(ref id, _)| id == res_id) {
                            last_submit = last_submit.max(a.index);
                        }
                    }
                }
                Some((key.clone(), last_submit))
            })
            .collect::<FastHashMap<_,_>>();

        for (ref key, submit_index) in remove_list {
            let resource = NativeResource::Framebuffer(framebuffers.remove(key).unwrap());
            match self.active
                .iter_mut()
                .find(|a| a.index == submit_index)
            {
                Some(a) => a.resources.push((None, resource)),
                None => self.free.push(resource),
            }
        }
    }

    fn handle_mapping(&mut self, raw: &<back::Backend as hal::Backend>::Device) {
        for buffer_id in self.ready_to_map.drain(..) {
            let mut operation = None;
            let (result, ptr) = {
                let mut buffer_guard = HUB.buffers.write();
                let mut buffer = &mut buffer_guard[buffer_id];
                std::mem::swap(&mut operation, &mut buffer.pending_map_operation);
                match operation.clone().unwrap() {
                    BufferMapOperation::Read(range, ..) => {
                        match map_buffer(raw, &mut buffer, range.clone(), true, false) {
                            Ok(ptr) => (BufferMapAsyncStatus::Success, Some(ptr)),
                            Err(e) => {
                                log::error!("failed to map buffer for reading: {}", e);
                                (BufferMapAsyncStatus::Error, None)
                            }
                        }
                    },
                    BufferMapOperation::Write(range, ..) => {
                        match map_buffer(raw, &mut buffer, range.clone(), false, true) {
                            Ok(ptr) => (BufferMapAsyncStatus::Success, Some(ptr)),
                            Err(e) => {
                                log::error!("failed to map buffer for writing: {}", e);
                                (BufferMapAsyncStatus::Error, None)
                            }
                        }
                    },
                }
            };

            match operation.unwrap() {
                BufferMapOperation::Read(_, callback, userdata) => callback(result, ptr.unwrap_or(std::ptr::null_mut()), userdata),
                BufferMapOperation::Write(_, callback, userdata) => callback(result, ptr.unwrap_or(std::ptr::null_mut()), userdata),
            };
        }
    }
}

fn map_buffer(
    raw: &<back::Backend as hal::Backend>::Device,
    buffer: &mut resource::Buffer<back::Backend>,
    range: std::ops::Range<u64>,
    read: bool,
    write: bool,
) -> Result<*mut u8, hal::mapping::Error> {
    let pointer = unsafe { raw.map_memory(&buffer.memory, range.clone()) }?;
    if read && !buffer.memory_properties.contains(hal::memory::Properties::COHERENT) {
        unsafe { raw.invalidate_mapped_memory_ranges(iter::once((&buffer.memory, range.clone()))).unwrap()  }; // TODO
    }
    if write && !buffer.memory_properties.contains(hal::memory::Properties::COHERENT) {
        buffer.mapped_write_ranges.push(range.clone());
    }
    Ok(pointer)
}

pub struct Device<B: hal::Backend> {
    pub(crate) raw: B::Device,
    adapter_id: AdapterId,
    pub(crate) queue_group: hal::QueueGroup<B, hal::General>,
    //mem_allocator: Heaps<B::Memory>,
    pub(crate) com_allocator: command::CommandAllocator<B>,
    life_guard: LifeGuard,
    pub(crate) trackers: Mutex<TrackerSet>,
    mem_props: hal::MemoryProperties,
    pub(crate) render_passes: Mutex<FastHashMap<RenderPassKey, B::RenderPass>>,
    pub(crate) framebuffers: Mutex<FastHashMap<FramebufferKey, B::Framebuffer>>,
    desc_pool: Mutex<B::DescriptorPool>,
    destroyed: Mutex<DestroyedResources<B>>,
}

impl<B: hal::Backend> Device<B> {
    pub(crate) fn new(
        raw: B::Device,
        adapter_id: AdapterId,
        queue_group: hal::QueueGroup<B, hal::General>,
        mem_props: hal::MemoryProperties,
    ) -> Self {
        // TODO: These values are just taken from rendy's test
        // Need to set reasonable values per memory type instead
        /*let arena = Some(allocator::ArenaConfig {
            arena_size: 32 * 1024,
        });
        let dynamic = Some(allocator::DynamicConfig {
            blocks_per_chunk: 64,
            block_size_granularity: 256,
            max_block_size: 32 * 1024,
        });
        let config = Config { arena, dynamic };
        let mem_allocator = unsafe {
            Heaps::new(
                mem_props
                    .memory_types
                    .iter()
                    .map(|mt| (mt.properties.into(), mt.heap_index as u32, config)),
                mem_props.memory_heaps.clone(),
            )
        };*/

        //TODO: generic allocator for descriptors
        let desc_pool = Mutex::new(
            unsafe {
                raw.create_descriptor_pool(
                    100,
                    &[
                        hal::pso::DescriptorRangeDesc {
                            ty: hal::pso::DescriptorType::Sampler,
                            count: 100,
                        },
                        hal::pso::DescriptorRangeDesc {
                            ty: hal::pso::DescriptorType::SampledImage,
                            count: 100,
                        },
                        hal::pso::DescriptorRangeDesc {
                            ty: hal::pso::DescriptorType::UniformBuffer,
                            count: 100,
                        },
                        hal::pso::DescriptorRangeDesc {
                            ty: hal::pso::DescriptorType::StorageBuffer,
                            count: 100,
                        },
                    ],
                )
            }.unwrap()
        );

        // don't start submission index at zero
        let life_guard = LifeGuard::new();
        life_guard.submission_index.fetch_add(1, Ordering::Relaxed);

        Device {
            raw,
            adapter_id,
            //mem_allocator,
            com_allocator: command::CommandAllocator::new(queue_group.family()),
            queue_group,
            life_guard,
            trackers: Mutex::new(TrackerSet::new()),
            mem_props,
            render_passes: Mutex::new(FastHashMap::default()),
            framebuffers: Mutex::new(FastHashMap::default()),
            desc_pool,
            destroyed: Mutex::new(DestroyedResources {
                mapped: Vec::new(),
                referenced: Vec::new(),
                active: Vec::new(),
                free: Vec::new(),
                ready_to_map: Vec::new(),
            }),
        }
    }
}


pub struct ShaderModule<B: hal::Backend> {
    pub(crate) raw: B::ShaderModule,
}


pub fn device_create_buffer(
    device_id: DeviceId,
    desc: &resource::BufferDescriptor,
) -> resource::Buffer<back::Backend> {
    let device_guard = HUB.devices.read();
    let device = &device_guard[device_id];
    let (usage, memory_properties) = conv::map_buffer_usage(desc.usage);

    let mut buffer = unsafe {
        device.raw.create_buffer(desc.size as u64, usage).unwrap()
    };
    let requirements = unsafe {
        device.raw.get_buffer_requirements(&buffer)
    };
    let device_type = device
        .mem_props
        .memory_types
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            requirements.type_mask & (1 << id) != 0
                && memory_type
                    .properties
                    .contains(memory_properties)
        })
        .unwrap()
        .into();
    // TODO: allocate with rendy
    let memory = unsafe {
        device.raw
            .allocate_memory(device_type, requirements.size)
            .unwrap()
    };
    unsafe {
        device.raw
            .bind_buffer_memory(&memory, 0, &mut buffer)
            .unwrap()
    };

    resource::Buffer {
        raw: buffer,
        device_id: Stored {
            value: device_id,
            ref_count: device.life_guard.ref_count.clone(),
        },
        memory_properties,
        memory,
        mapped_write_ranges: Vec::new(),
        pending_map_operation: None,
        life_guard: LifeGuard::new(),
    }
}

pub fn device_track_buffer(
    device_id: DeviceId,
    buffer_id: BufferId,
    ref_count: RefCount,
    flags: resource::BufferUsageFlags,
) {
    let query = HUB.devices
        .read()
        [device_id].trackers
        .lock()
        .buffers
        .query(buffer_id, &ref_count, flags);
    assert!(query.initialized);
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_buffer(
    device_id: DeviceId,
    desc: &resource::BufferDescriptor,
) -> BufferId {
    let buffer = device_create_buffer(device_id, desc);
    let ref_count = buffer.life_guard.ref_count.clone();
    let id = HUB.buffers.register_local(buffer);
    device_track_buffer(device_id, id, ref_count, resource::BufferUsageFlags::empty());
    id
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_buffer_mapped(
    device_id: DeviceId,
    desc: &resource::BufferDescriptor,
    mapped_ptr_out: *mut *mut u8
) -> BufferId {
    let mut desc = desc.clone();
    desc.usage |= resource::BufferUsageFlags::MAP_WRITE;
    let mut buffer = device_create_buffer(device_id, &desc);

    let device_guard = HUB.devices.read();
    let device = &device_guard[device_id];

    match map_buffer(&device.raw, &mut buffer, 0..(desc.size as u64), false, true) {
        Ok(ptr) => unsafe { *mapped_ptr_out = ptr; }
        Err(e) => {
            log::error!("failed to create buffer in a mapped state: {}", e);
            unsafe { *mapped_ptr_out = std::ptr::null_mut(); }
        }
    }

    let ref_count = buffer.life_guard.ref_count.clone();
    let id = HUB.buffers.register_local(buffer);
    device_track_buffer(device_id, id, ref_count, resource::BufferUsageFlags::MAP_WRITE);
    id
}

#[no_mangle]
pub extern "C" fn wgpu_buffer_destroy(buffer_id: BufferId) {
    let buffer_guard = HUB.buffers.read();
    let buffer = &buffer_guard[buffer_id];
    HUB.devices
        .read()
        [buffer.device_id.value].destroyed
        .lock()
        .destroy(
            ResourceId::Buffer(buffer_id),
            buffer.life_guard.ref_count.clone(),
        );
}

pub fn device_create_texture(
    device_id: DeviceId,
    desc: &resource::TextureDescriptor,
) -> resource::Texture<back::Backend> {
    let kind = conv::map_texture_dimension_size(desc.dimension, desc.size, desc.array_size);
    let format = conv::map_texture_format(desc.format);
    let aspects = format.surface_desc().aspects;
    let usage = conv::map_texture_usage(desc.usage, aspects);
    let device_guard = HUB.devices.read();
    let device = &device_guard[device_id];

    let mut image = unsafe {
        device.raw.create_image(
            kind,
            1, // TODO: mips
            format,
            hal::image::Tiling::Optimal, // TODO: linear
            usage,
            hal::image::ViewCapabilities::empty(), // TODO: format, 2d array, cube
        )
    }
    .unwrap();
    let requirements = unsafe {
        device.raw.get_image_requirements(&image)
    };
    let device_type = device
        .mem_props
        .memory_types
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            // TODO
            requirements.type_mask & (1 << id) != 0
                && memory_type
                    .properties
                    .contains(hal::memory::Properties::DEVICE_LOCAL)
        })
        .unwrap()
        .into();
    // TODO: allocate with rendy
    let memory = unsafe {
        device.raw
            .allocate_memory(device_type, requirements.size)
            .unwrap()
    };
    unsafe {
        device.raw
            .bind_image_memory(&memory, 0, &mut image)
            .unwrap()
    };

    resource::Texture {
        raw: image,
        device_id: Stored {
            value: device_id,
            ref_count: device.life_guard.ref_count.clone(),
        },
        kind,
        format: desc.format,
        full_range: hal::image::SubresourceRange {
            aspects,
            levels: 0 .. 1, //TODO: mips
            layers: 0 .. desc.array_size as u16,
        },
        placement: resource::TexturePlacement::Memory(memory),
        life_guard: LifeGuard::new(),
    }
}

pub fn device_track_texture(
    device_id: DeviceId,
    texture_id: TextureId,
    ref_count: RefCount,
) {
    let query = HUB.devices
        .read()
        [device_id].trackers
        .lock()
        .textures
        .query(texture_id, &ref_count, resource::TextureUsageFlags::UNINITIALIZED);
    assert!(query.initialized);
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_texture(
    device_id: DeviceId,
    desc: &resource::TextureDescriptor,
) -> TextureId {
    let texture = device_create_texture(device_id, desc);
    let ref_count = texture.life_guard.ref_count.clone();
    let id = HUB.textures.register_local(texture);
    device_track_texture(device_id, id, ref_count);
    id
}

pub fn texture_create_view(
    texture_id: TextureId,
    format: resource::TextureFormat,
    view_kind: hal::image::ViewKind,
    range: hal::image::SubresourceRange,
) -> resource::TextureView<back::Backend> {
    let texture_guard = HUB.textures.read();
    let texture = &texture_guard[texture_id];

    let raw = unsafe {
        HUB.devices
            .read()
            [texture.device_id.value].raw
            .create_image_view(
                &texture.raw,
                view_kind,
                conv::map_texture_format(format),
                hal::format::Swizzle::NO,
                range,
            )
            .unwrap()
    };

    resource::TextureView {
        raw,
        texture_id: Stored {
            value: texture_id,
            ref_count: texture.life_guard.ref_count.clone(),
        },
        format: texture.format,
        extent: texture.kind.extent(),
        samples: texture.kind.num_samples(),
        is_owned_by_swap_chain: false,
        life_guard: LifeGuard::new(),
    }
}

pub fn device_track_view(
    texture_id: TextureId,
    view_id: BufferId,
    ref_count: RefCount,
) {
    let device_id = HUB.textures
        .read()
        [texture_id].device_id.value;
    let query = HUB.devices
        .read()
        [device_id].trackers
        .lock()
        .views
        .query(view_id, &ref_count, DummyUsage);
    assert!(query.initialized);
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_texture_create_view(
    texture_id: TextureId,
    desc: &resource::TextureViewDescriptor,
) -> TextureViewId {
    let view = texture_create_view(
        texture_id,
        desc.format,
        conv::map_texture_view_dimension(desc.dimension),
        hal::image::SubresourceRange {
            aspects: conv::map_texture_aspect_flags(desc.aspect),
            levels: desc.base_mip_level as u8 .. (desc.base_mip_level + desc.level_count) as u8,
            layers: desc.base_array_layer as u16 .. (desc.base_array_layer + desc.array_count) as u16,
        },
    );
    let ref_count = view.life_guard.ref_count.clone();
    let id = HUB.texture_views.register_local(view);
    device_track_view(texture_id, id, ref_count);
    id
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_texture_create_default_view(texture_id: TextureId) -> TextureViewId {
    let (format, view_kind, range) = {
        let texture_guard = HUB.textures.read();
        let texture = &texture_guard[texture_id];
        let view_kind = match texture.kind {
            hal::image::Kind::D1(_, 1) => hal::image::ViewKind::D1,
            hal::image::Kind::D1(..) => hal::image::ViewKind::D1Array,
            hal::image::Kind::D2(_, _, 1, _) => hal::image::ViewKind::D2,
            hal::image::Kind::D2(..) => hal::image::ViewKind::D2Array,
            hal::image::Kind::D3(..) => hal::image::ViewKind::D3,
        };
        (texture.format, view_kind, texture.full_range.clone())
    };
    let view = texture_create_view(
        texture_id,
        format,
        view_kind,
        range,
    );
    let ref_count = view.life_guard.ref_count.clone();
    let id = HUB.texture_views.register_local(view);
    device_track_view(texture_id, id, ref_count);
    id
}

#[no_mangle]
pub extern "C" fn wgpu_texture_destroy(texture_id: TextureId) {
    let texture_guard = HUB.textures.read();
    let texture = &texture_guard[texture_id];
    HUB.devices
        .read()
        [texture.device_id.value].destroyed
        .lock()
        .destroy(
            ResourceId::Texture(texture_id),
            texture.life_guard.ref_count.clone(),
        );
}

#[no_mangle]
pub extern "C" fn wgpu_texture_view_destroy(texture_view_id: TextureViewId) {
    let texture_view_guard = HUB.texture_views.read();
    let view = &texture_view_guard[texture_view_id];
    let device_id = HUB.textures
        .read()
        [view.texture_id.value].device_id.value;
    HUB.devices
        .read()
        [device_id].destroyed
        .lock()
        .destroy(
            ResourceId::TextureView(texture_view_id),
            view.life_guard.ref_count.clone(),
        );
}


pub fn device_create_sampler(
    device_id: DeviceId, desc: &resource::SamplerDescriptor
) -> resource::Sampler<back::Backend> {
    let device_guard = HUB.devices.read();
    let device = &device_guard[device_id];

    let info = hal::image::SamplerInfo {
        min_filter: conv::map_filter(desc.min_filter),
        mag_filter: conv::map_filter(desc.mag_filter),
        mip_filter: conv::map_filter(desc.mipmap_filter),
        wrap_mode: (
            conv::map_wrap(desc.r_address_mode),
            conv::map_wrap(desc.s_address_mode),
            conv::map_wrap(desc.t_address_mode),
        ),
        lod_bias: 0.0.into(),
        lod_range: desc.lod_min_clamp.into() .. desc.lod_max_clamp.into(),
        comparison: if desc.compare_function == resource::CompareFunction::Always {
            None
        } else {
            Some(conv::map_compare_function(desc.compare_function))
        },
        border: hal::image::PackedColor(match desc.border_color {
            resource::BorderColor::TransparentBlack => 0x00000000,
            resource::BorderColor::OpaqueBlack => 0x000000FF,
            resource::BorderColor::OpaqueWhite => 0xFFFFFFFF,
        }),
        anisotropic: hal::image::Anisotropic::Off, //TODO
    };
    let raw = unsafe {
        device.raw
            .create_sampler(info)
            .unwrap()
    };

    resource::Sampler {
        raw,
    }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_sampler(
    device_id: DeviceId, desc: &resource::SamplerDescriptor
) -> SamplerId {
    let sampler = device_create_sampler(device_id, desc);
    HUB.samplers.register_local(sampler)
}


pub fn device_create_bind_group_layout(
    device_id: DeviceId,
    desc: &binding_model::BindGroupLayoutDescriptor,
) -> binding_model::BindGroupLayout<back::Backend> {
    let bindings = unsafe { slice::from_raw_parts(desc.bindings, desc.bindings_length) };

    let raw = unsafe {
        HUB.devices
            .read()
            [device_id].raw
            .create_descriptor_set_layout(
                bindings.iter().map(|binding| {
                    hal::pso::DescriptorSetLayoutBinding {
                        binding: binding.binding,
                        ty: conv::map_binding_type(binding.ty),
                        count: 1, //TODO: consolidate
                        stage_flags: conv::map_shader_stage_flags(binding.visibility),
                        immutable_samplers: false, // TODO
                    }
                }),
                &[],
            )
            .unwrap()
    };

    binding_model::BindGroupLayout {
        raw,
    }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_bind_group_layout(
    device_id: DeviceId,
    desc: &binding_model::BindGroupLayoutDescriptor,
) -> BindGroupLayoutId {
    let layout = device_create_bind_group_layout(device_id, desc);
    HUB.bind_group_layouts.register_local(layout)
}

pub fn device_create_pipeline_layout(
    device_id: DeviceId,
    desc: &binding_model::PipelineLayoutDescriptor,
) -> binding_model::PipelineLayout<back::Backend> {
    let bind_group_layout_ids = unsafe {
        slice::from_raw_parts(desc.bind_group_layouts, desc.bind_group_layouts_length)
    };
    let bind_group_layout_guard = HUB.bind_group_layouts.read();
    let descriptor_set_layouts = bind_group_layout_ids
        .iter()
        .map(|&id| &bind_group_layout_guard[id].raw);

    // TODO: push constants
    let pipeline_layout = unsafe {
        HUB.devices
            .read()
            [device_id].raw
            .create_pipeline_layout(descriptor_set_layouts, &[])
    }
    .unwrap();

    binding_model::PipelineLayout {
        raw: pipeline_layout,
        bind_group_layout_ids: bind_group_layout_ids
            .iter()
            .cloned()
            .collect(),
    }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_pipeline_layout(
    device_id: DeviceId,
    desc: &binding_model::PipelineLayoutDescriptor,
) -> PipelineLayoutId {
    let layout = device_create_pipeline_layout(device_id, desc);
    HUB.pipeline_layouts.register_local(layout)
}

pub fn device_create_bind_group(
    device_id: DeviceId,
    desc: &binding_model::BindGroupDescriptor,
) -> binding_model::BindGroup<back::Backend> {
    let device_guard = HUB.devices.read();
    let device = &device_guard[device_id];
    let bind_group_layout_guard = HUB.bind_group_layouts.read();
    let bind_group_layout = &bind_group_layout_guard[desc.layout];
    let bindings = unsafe {
        slice::from_raw_parts(desc.bindings, desc.bindings_length as usize)
    };

    let mut desc_pool = device.desc_pool.lock();
    let desc_set = unsafe {
        desc_pool
            .allocate_set(&bind_group_layout.raw)
            .unwrap()
    };

    let buffer_guard = HUB.buffers.read();
    let sampler_guard = HUB.samplers.read();
    let texture_view_guard = HUB.texture_views.read();

    //TODO: group writes into contiguous sections
    let mut writes = Vec::new();
    let mut used = TrackerSet::new();
    for b in bindings {
        let descriptor = match b.resource {
            binding_model::BindingResource::Buffer(ref bb) => {
                let buffer = used.buffers
                    .get_with_extended_usage(
                        &*buffer_guard,
                        bb.buffer,
                        resource::BufferUsageFlags::UNIFORM,
                    )
                    .unwrap();
                let range = Some(bb.offset as u64) .. Some((bb.offset + bb.size) as u64);
                hal::pso::Descriptor::Buffer(&buffer.raw, range)
            }
            binding_model::BindingResource::Sampler(id) => {
                let sampler = &sampler_guard[id];
                hal::pso::Descriptor::Sampler(&sampler.raw)
            }
            binding_model::BindingResource::TextureView(id) => {
                let view = &texture_view_guard[id];
                used.views.query(id, &view.life_guard.ref_count, DummyUsage);
                used.textures
                    .transit(
                        view.texture_id.value,
                        &view.texture_id.ref_count,
                        resource::TextureUsageFlags::SAMPLED,
                        TrackPermit::EXTEND,
                    )
                    .unwrap();
                hal::pso::Descriptor::Image(&view.raw, hal::image::Layout::ShaderReadOnlyOptimal)
            }
        };
        let write = hal::pso::DescriptorSetWrite {
            set: &desc_set,
            binding: b.binding,
            array_offset: 0, //TODO
            descriptors: iter::once(descriptor),
        };
        writes.push(write);
    }

    unsafe {
        device.raw.write_descriptor_sets(writes);
    }

    binding_model::BindGroup {
        raw: desc_set,
        layout_id: desc.layout,
        life_guard: LifeGuard::new(),
        used,
    }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_bind_group(
    device_id: DeviceId,
    desc: &binding_model::BindGroupDescriptor,
) -> BindGroupId {
    let bind_group = device_create_bind_group(device_id, desc);
    HUB.bind_groups.register_local(bind_group)
}

#[no_mangle]
pub extern "C" fn wgpu_bind_group_destroy(bind_group_id: BindGroupId) {
    HUB.bind_groups.unregister(bind_group_id);
}


pub fn device_create_shader_module(
    device_id: DeviceId,
    desc: &pipeline::ShaderModuleDescriptor,
) -> ShaderModule<back::Backend> {
    let spv = unsafe { slice::from_raw_parts(desc.code.bytes, desc.code.length) };
    let shader = unsafe {
        HUB.devices
            .read()
            [device_id].raw
            .create_shader_module(spv)
            .unwrap()
    };

    ShaderModule { raw: shader }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_shader_module(
    device_id: DeviceId,
    desc: &pipeline::ShaderModuleDescriptor,
) -> ShaderModuleId {
    let module = device_create_shader_module(device_id, desc);
    HUB.shader_modules.register_local(module)
}

pub fn device_create_command_encoder(
    device_id: DeviceId,
    _desc: &command::CommandEncoderDescriptor,
) -> command::CommandBuffer<back::Backend> {
    let device_guard = HUB.devices.read();
    let device = &device_guard[device_id];

    let dev_stored = Stored {
        value: device_id,
        ref_count: device.life_guard.ref_count.clone(),
    };
    let mut cmb = device.com_allocator.allocate(dev_stored, &device.raw);
    unsafe {
        cmb.raw.last_mut().unwrap().begin(
            hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
            hal::command::CommandBufferInheritanceInfo::default(),
        );
    }
    cmb
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_command_encoder(
    device_id: DeviceId,
    desc: &command::CommandEncoderDescriptor,
) -> CommandEncoderId {
    let cmb = device_create_command_encoder(device_id, desc);
    HUB.command_buffers.register_local(cmb)
}

#[no_mangle]
pub extern "C" fn wgpu_device_get_queue(device_id: DeviceId) -> QueueId {
    device_id
}

#[no_mangle]
pub extern "C" fn wgpu_queue_submit(
    queue_id: QueueId,
    command_buffer_ptr: *const CommandBufferId,
    command_buffer_count: usize,
) {
    let command_buffer_ids =
        unsafe { slice::from_raw_parts(command_buffer_ptr, command_buffer_count) };

    let (old_submit_index, fence) = {
        let mut device_guard = HUB.devices.write();
        let device = &mut device_guard[queue_id];

        let mut swap_chain_links = Vec::new();

        let old_submit_index = device
            .life_guard
            .submission_index
            .fetch_add(1, Ordering::Relaxed);
        let mut trackers = device.trackers.lock();

        //TODO: if multiple command buffers are submitted, we can re-use the last
        // native command buffer of the previous chain instead of always creating
        // a temporary one, since the chains are not finished.
        {
            let mut command_buffer_guard = HUB.command_buffers.write();
            let buffer_guard = HUB.buffers.read();
            let texture_guard = HUB.textures.read();
            let texture_view_guard = HUB.texture_views.read();

            // finish all the command buffers first
            for &cmb_id in command_buffer_ids {
                let comb = &mut command_buffer_guard[cmb_id];
                swap_chain_links.extend(comb.swap_chain_links.drain(..));
                // update submission IDs
                comb.life_guard.submission_index
                    .store(old_submit_index, Ordering::Release);
                for id in comb.trackers.buffers.used() {
                    buffer_guard[id].life_guard.submission_index
                        .store(old_submit_index, Ordering::Release);
                }
                for id in comb.trackers.textures.used() {
                    texture_guard[id].life_guard.submission_index
                        .store(old_submit_index, Ordering::Release);
                }
                for id in comb.trackers.views.used() {
                    texture_view_guard[id].life_guard.submission_index
                        .store(old_submit_index, Ordering::Release);
                }

                // execute resource transitions
                let mut transit = device.com_allocator.extend(comb);
                unsafe {
                    transit.begin(
                        hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
                        hal::command::CommandBufferInheritanceInfo::default(),
                    );
                }
                command::CommandBuffer::insert_barriers(
                    &mut transit,
                    &mut *trackers,
                    &comb.trackers,
                    Stitch::Init,
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
        }

        // now prepare the GPU submission
        let fence = device.raw.create_fence(false).unwrap();
        {
            let command_buffer_guard = HUB.command_buffers.read();
            let surface_guard = HUB.surfaces.read();

            let wait_semaphores = swap_chain_links
                .into_iter()
                .flat_map(|link| {
                    //TODO: check the epoch
                    surface_guard[link.swap_chain_id].swap_chain
                        .as_ref()
                        .map(|swap_chain| (
                            &swap_chain.frames[link.image_index as usize].sem_available,
                            hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                        ))
                });

            let submission =
                hal::queue::Submission::<_, _, &[<back::Backend as hal::Backend>::Semaphore]> {
                    //TODO: may `OneShot` be enough?
                    command_buffers: command_buffer_ids
                        .iter()
                        .flat_map(|&cmb_id| &command_buffer_guard[cmb_id].raw),
                    wait_semaphores,
                    signal_semaphores: &[], //TODO: signal `sem_present`?
                };

            unsafe {
                device.queue_group.queues[0]
                    .as_raw_mut()
                    .submit(submission, Some(&fence));
            }
        }

        (old_submit_index, fence)
    };

    // No need for write access to the device from here on out
    let device_guard = HUB.devices.read();
    let device = &device_guard[queue_id];

    let mut trackers = device.trackers.lock();

    let last_done = {
        let mut destroyed = device.destroyed.lock();
        destroyed.triage_referenced(&mut *trackers);
        destroyed.triage_framebuffers(&mut *device.framebuffers.lock());
        let last_done = destroyed.cleanup(&device.raw);
        destroyed.handle_mapping(&device.raw);

        destroyed.active.push(ActiveSubmission {
            index: old_submit_index + 1,
            fence,
            resources: Vec::new(),
            mapped: Vec::new(),
        });

        last_done
    };

    if last_done != 0 {
        device.com_allocator.maintain(last_done);
    }

    // finally, return the command buffers to the allocator
    for &cmb_id in command_buffer_ids {
        let cmd_buf = HUB.command_buffers.unregister(cmb_id);
        device.com_allocator.after_submit(cmd_buf);
    }
}


pub fn device_create_render_pipeline(
    device_id: DeviceId,
    desc: &pipeline::RenderPipelineDescriptor,
) -> pipeline::RenderPipeline<back::Backend> {
    let device_guard = HUB.devices.read();
    let device = &device_guard[device_id];
    let pipeline_layout_guard = HUB.pipeline_layouts.read();
    let layout = &pipeline_layout_guard[desc.layout].raw;
    let shader_module_guard = HUB.shader_modules.read();

    let color_states = unsafe {
        slice::from_raw_parts(
            desc.color_states,
            desc.color_states_length,
        )
    };
    let depth_stencil_state = unsafe {
        desc.depth_stencil_state.as_ref()
    };

    let rp_key = RenderPassKey {
        colors: color_states
            .iter()
            .map(|at| hal::pass::Attachment {
                format: Some(conv::map_texture_format(at.format)),
                samples: desc.sample_count as u8,
                ops: hal::pass::AttachmentOps::PRESERVE,
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::Layout::General..hal::image::Layout::General,
            })
            .collect(),
        depth_stencil: depth_stencil_state
            .map(|at| hal::pass::Attachment {
                format: Some(conv::map_texture_format(at.format)),
                samples: desc.sample_count as u8,
                ops: hal::pass::AttachmentOps::PRESERVE,
                stencil_ops: hal::pass::AttachmentOps::PRESERVE,
                layouts: hal::image::Layout::General..hal::image::Layout::General,
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
                colors: &color_ids[..desc.color_states_length],
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

    let vertex = hal::pso::EntryPoint::<back::Backend> {
        entry: unsafe { ffi::CStr::from_ptr(desc.vertex_stage.entry_point) }
            .to_str()
            .to_owned()
            .unwrap(), // TODO
        module: &shader_module_guard[desc.vertex_stage.module].raw,
        specialization: hal::pso::Specialization {
            // TODO
            constants: &[],
            data: &[],
        },
    };
    let fragment = hal::pso::EntryPoint::<back::Backend> {
        entry: unsafe { ffi::CStr::from_ptr(desc.fragment_stage.entry_point) }
            .to_str()
            .to_owned()
            .unwrap(), // TODO
        module: &shader_module_guard[desc.fragment_stage.module].raw,
        specialization: hal::pso::Specialization {
            // TODO
            constants: &[],
            data: &[],
        },
    };

    let shaders = hal::pso::GraphicsShaderSet {
        vertex,
        hull: None,
        domain: None,
        geometry: None,
        fragment: Some(fragment),
    };
    let rasterizer = conv::map_rasterization_state_descriptor(&desc.rasterization_state);

    let desc_vbs = unsafe {
        slice::from_raw_parts(desc.vertex_buffer_state.vertex_buffers, desc.vertex_buffer_state.vertex_buffers_count)
    };
    let mut vertex_buffers = Vec::with_capacity(desc_vbs.len());
    let mut attributes = Vec::new();
    for (i, vb_state) in desc_vbs.iter().enumerate() {
        if vb_state.attributes_count == 0 {
            continue
        }
        vertex_buffers.push(hal::pso::VertexBufferDesc {
            binding: i as u32,
            stride: vb_state.stride,
            rate: match vb_state.step_mode {
                pipeline::InputStepMode::Vertex => 0,
                pipeline::InputStepMode::Instance => 1,
            },
        });
        let desc_atts = unsafe {
            slice::from_raw_parts(vb_state.attributes, vb_state.attributes_count)
        };
        for attribute in desc_atts {
            attributes.push(hal::pso::AttributeDesc {
                location: attribute.attribute_index,
                binding: i as u32,
                element: hal::pso::Element {
                    format: conv::map_vertex_format(attribute.format),
                    offset: attribute.offset,
                },
            });
        }
    }

    let input_assembler = hal::pso::InputAssemblerDesc {
        primitive: conv::map_primitive_topology(desc.primitive_topology),
        primitive_restart: hal::pso::PrimitiveRestart::Disabled, // TODO
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

    // TODO
    let multisampling: Option<hal::pso::Multisampling> = None;

    // TODO
    let baked_states = hal::pso::BakedStates {
        viewport: None,
        scissor: None,
        blend_color: None,
        depth_bounds: None,
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
    let pipeline = unsafe {
        device.raw
            .create_graphics_pipeline(&pipeline_desc, None)
            .unwrap()
    };

    let pass_context = RenderPassContext {
        colors: color_states
            .iter()
            .map(|state| state.format)
            .collect(),
        depth_stencil: depth_stencil_state
            .map(|state| state.format),
    }; 

    pipeline::RenderPipeline {
        raw: pipeline,
        layout_id: desc.layout,
        pass_context,
    }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_render_pipeline(
    device_id: DeviceId,
    desc: &pipeline::RenderPipelineDescriptor,
) -> RenderPipelineId {
    let pipeline = device_create_render_pipeline(device_id, desc);
    HUB.render_pipelines.register_local(pipeline)
}

pub fn device_create_compute_pipeline(
    device_id: DeviceId,
    desc: &pipeline::ComputePipelineDescriptor,
) -> pipeline::ComputePipeline<back::Backend> {
    let device_guard = HUB.devices.read();
    let device = &device_guard[device_id].raw;
    let pipeline_layout_guard = HUB.pipeline_layouts.read();
    let layout = &pipeline_layout_guard[desc.layout].raw;
    let pipeline_stage = &desc.compute_stage;
    let shader_module_guard = HUB.shader_modules.read();

    let shader = hal::pso::EntryPoint::<back::Backend> {
        entry: unsafe { ffi::CStr::from_ptr(pipeline_stage.entry_point) }
            .to_str()
            .to_owned()
            .unwrap(), // TODO
        module: &shader_module_guard[pipeline_stage.module].raw,
        specialization: hal::pso::Specialization {
            // TODO
            constants: &[],
            data: &[],
        },
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

    let pipeline = unsafe {
        device
            .create_compute_pipeline(&pipeline_desc, None)
            .unwrap()
    };

    pipeline::ComputePipeline {
        raw: pipeline,
        layout_id: desc.layout,
    }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_compute_pipeline(
    device_id: DeviceId,
    desc: &pipeline::ComputePipelineDescriptor,
) -> ComputePipelineId {
    let pipeline = device_create_compute_pipeline(device_id, desc);
    HUB.compute_pipelines.register_local(pipeline)
}

pub fn device_create_swap_chain(
    device_id: DeviceId,
    surface_id: SurfaceId,
    desc: &swap_chain::SwapChainDescriptor,
) -> Vec<resource::Texture<back::Backend>> {
    info!("creating swap chain {:?}", desc);

    let device_guard = HUB.devices.read();
    let device = &device_guard[device_id];
    let mut surface_guard = HUB.surfaces.write();
    let surface = &mut surface_guard[surface_id];

    let (caps, formats, _present_modes, _composite_alphas) = {
        let adapter_guard = HUB.adapters.read();
        let adapter = &adapter_guard[device.adapter_id];
        assert!(surface.raw.supports_queue_family(&adapter.queue_families[0]));
        surface.raw.compatibility(&adapter.physical_device)
    };
    let num_frames = caps.image_count.start; //TODO: configure?
    let usage = conv::map_texture_usage(desc.usage, hal::format::Aspects::COLOR);
    let config = hal::SwapchainConfig::new(
        desc.width,
        desc.height,
        conv::map_texture_format(desc.format),
        num_frames, //TODO: configure?
    );

    if let Some(formats) = formats {
        assert!(formats.contains(&config.format),
            "Requested format {:?} is not in supported list: {:?}",
            config.format, formats);
    }
    //TODO: properly exclusive range
    /* TODO: this is way too restrictive
    assert!(desc.width >= caps.extents.start.width && desc.width <= caps.extents.end.width &&
        desc.height >= caps.extents.start.height && desc.height <= caps.extents.end.height,
        "Requested size {}x{} is outside of the supported range: {:?}",
        desc.width, desc.height, caps.extents);
    */

    let (old_raw, sem_available, command_pool) = match surface.swap_chain.take() {
        Some(mut old) => {
            //TODO: remove this once gfx-rs stops destroying the old swapchain
            device.raw.wait_idle().unwrap();
            let mut destroyed = device.destroyed.lock();
            assert_eq!(old.device_id.value, device_id);
            for frame in old.frames {
                destroyed.destroy(ResourceId::Texture(frame.texture_id.value), frame.texture_id.ref_count);
                destroyed.destroy(ResourceId::TextureView(frame.view_id.value), frame.view_id.ref_count);
            }
            unsafe {
                old.command_pool.reset()
            };
            (Some(old.raw), old.sem_available, old.command_pool)
        }
        _ => unsafe {
            let sem_available = device.raw
                .create_semaphore()
                .unwrap();
            let command_pool = device.raw
                .create_command_pool_typed(
                    &device.queue_group,
                    hal::pool::CommandPoolCreateFlags::RESET_INDIVIDUAL,
                )
                .unwrap();
            (None, sem_available, command_pool)
        }
    };

    let (raw, backbuffer) = unsafe {
        device.raw
            .create_swapchain(
                &mut surface.raw,
                config.with_image_usage(usage),
                old_raw,
            )
            .unwrap()
    };
    surface.swap_chain = Some(swap_chain::SwapChain {
        raw,
        device_id: Stored {
            value: device_id,
            ref_count: device.life_guard.ref_count.clone(),
        },
        desc: desc.clone(),
        frames: Vec::with_capacity(num_frames as usize),
        acquired: Vec::with_capacity(1), //TODO: get it from gfx-hal?
        sem_available,
        command_pool,
    });

    let images = match backbuffer {
        hal::Backbuffer::Images(images) => images,
        hal::Backbuffer::Framebuffer(_) => panic!("Deprecated API detected!"),
    };

    images
        .into_iter()
        .map(|raw| resource::Texture {
            raw,
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.ref_count.clone(),
            },
            kind: hal::image::Kind::D2(desc.width, desc.height, 1, 1),
            format: desc.format,
            full_range: hal::image::SubresourceRange {
                aspects: hal::format::Aspects::COLOR,
                levels: 0 .. 1,
                layers: 0 .. 1,
            },
            placement: resource::TexturePlacement::Void,
            life_guard: LifeGuard::new(),
        })
        .collect()
}

#[cfg(feature = "local")]
pub fn swap_chain_populate_textures(
    swap_chain_id: SwapChainId,
    textures: Vec<resource::Texture<back::Backend>>,
) {
    let mut surface_guard = HUB.surfaces.write();
    let swap_chain = surface_guard[swap_chain_id].swap_chain
        .as_mut()
        .unwrap();
    let device_guard = HUB.devices.read();
    let device = &device_guard[swap_chain.device_id.value];
    let mut trackers = device.trackers.lock();

    for (i, mut texture) in textures.into_iter().enumerate() {
        let format = texture.format;
        let kind = texture.kind;

        let view_raw = unsafe {
            device.raw
                .create_image_view(
                    &texture.raw,
                    hal::image::ViewKind::D2,
                    conv::map_texture_format(format),
                    hal::format::Swizzle::NO,
                    texture.full_range.clone(),
                )
                .unwrap()
            };
        texture.placement = resource::TexturePlacement::SwapChain(swap_chain::SwapChainLink {
            swap_chain_id, //TODO: strongly
            epoch: Mutex::new(0),
            image_index: i as hal::SwapImageIndex,
        });
        let texture_id = Stored {
            ref_count: texture.life_guard.ref_count.clone(),
            value: HUB.textures.register_local(texture),
        };
        trackers.textures.query(
            texture_id.value,
            &texture_id.ref_count,
            resource::TextureUsageFlags::UNINITIALIZED,
        );

        let view = resource::TextureView {
            raw: view_raw,
            texture_id: texture_id.clone(),
            format,
            extent: kind.extent(),
            samples: kind.num_samples(),
            is_owned_by_swap_chain: true,
            life_guard: LifeGuard::new(),
        };
        let view_id = Stored {
             ref_count: view.life_guard.ref_count.clone(),
             value: HUB.texture_views.register_local(view),
        };
        trackers.views.query(
            view_id.value,
            &view_id.ref_count,
            DummyUsage,
        );

        swap_chain.frames.push(swap_chain::Frame {
            texture_id,
            view_id,
            fence: device.raw.create_fence(true).unwrap(),
            sem_available: device.raw.create_semaphore().unwrap(),
            sem_present: device.raw.create_semaphore().unwrap(),
            comb: swap_chain.command_pool.acquire_command_buffer(),
        });
    }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_swap_chain(
    device_id: DeviceId,
    surface_id: SurfaceId,
    desc: &swap_chain::SwapChainDescriptor,
) -> SwapChainId {
    let textures = device_create_swap_chain(device_id, surface_id, desc);
    swap_chain_populate_textures(surface_id, textures);
    surface_id
}


#[no_mangle]
pub extern "C" fn wgpu_buffer_set_sub_data(
    buffer_id: BufferId,
    start: u32, count: u32, data: *const u8,
) {
    let buffer_guard = HUB.buffers.read();
    let buffer = &buffer_guard[buffer_id];
    let mut device_guard = HUB.devices.write();
    let device = &mut device_guard[buffer.device_id.value];

    //Note: this is just doing `update_buffer`, which is limited to 64KB

    trace!("transit {:?} to transfer dst", buffer_id);
    let barrier = device.trackers
        .lock()
        .buffers
        .transit(
            buffer_id,
            &buffer.life_guard.ref_count,
            resource::BufferUsageFlags::TRANSFER_DST,
            TrackPermit::REPLACE,
        )
        .unwrap()
        .into_source()
        .map(|old| hal::memory::Barrier::Buffer {
            states: conv::map_buffer_state(old) ..
                hal::buffer::State::TRANSFER_WRITE,
            target: &buffer.raw,
            families: None,
            range: None .. None, //TODO: could be partial
        });

    // Note: this is not pretty. If we need one-time service command buffers,
    // we'll need to have some internal abstractions for them to be safe.
    let mut comb = device.com_allocator.allocate(buffer.device_id.clone(), &device.raw);
    // mark as used by the next submission, conservatively
    let last_submit_index = device.life_guard.submission_index.load(Ordering::Acquire);
    comb.life_guard.submission_index.store(last_submit_index + 1, Ordering::Release);
    unsafe {
        let raw = comb.raw.last_mut().unwrap();
        raw.begin(
            hal::command::CommandBufferFlags::ONE_TIME_SUBMIT,
            hal::command::CommandBufferInheritanceInfo::default(),
        );
        raw.pipeline_barrier(
            all_buffer_stages() .. hal::pso::PipelineStage::TRANSFER,
            hal::memory::Dependencies::empty(),
            barrier,
        );
        raw.update_buffer(
            &buffer.raw,
            start as hal::buffer::Offset,
            slice::from_raw_parts(data, count as usize),
        );
        raw.finish();

        let submission = hal::queue::Submission {
            command_buffers: iter::once(&*raw),
            wait_semaphores: None,
            signal_semaphores: None,
        };
        device.queue_group.queues[0]
            .as_raw_mut()
            .submit::<_, _, <back::Backend as hal::Backend>::Semaphore, _, _>(submission, None);
    }

    device.com_allocator.after_submit(comb);
}

#[no_mangle]
pub extern "C" fn wgpu_device_destroy(device_id: BufferId) {
    HUB.devices.unregister(device_id);
}

pub type BufferMapReadCallback = extern "C" fn(status: BufferMapAsyncStatus, data: *const u8, userdata: *mut u8);
pub type BufferMapWriteCallback = extern "C" fn(status: BufferMapAsyncStatus, data: *mut u8, userdata: *mut u8);

#[no_mangle]
pub extern "C" fn wgpu_buffer_map_read_async(
    buffer_id: BufferId,
    start: u32, size: u32, callback: BufferMapReadCallback, userdata: *mut u8,
) {
    let mut buffer_guard = HUB.buffers.write();
    let buffer = &mut buffer_guard[buffer_id];

    let range = start as u64..(start + size) as u64;
    buffer.pending_map_operation = Some(BufferMapOperation::Read(range, callback, userdata));

    HUB.devices
        .read()
        [buffer.device_id.value].destroyed
        .lock()
        .map(buffer_id, buffer.life_guard.ref_count.clone());
}

#[no_mangle]
pub extern "C" fn wgpu_buffer_map_write_async(
    buffer_id: BufferId,
    start: u32, size: u32, callback: BufferMapWriteCallback, userdata: *mut u8,
) {
    let mut buffer_guard = HUB.buffers.write();
    let buffer = &mut buffer_guard[buffer_id];

    let range = start as u64..(start + size) as u64;
    buffer.pending_map_operation = Some(BufferMapOperation::Write(range, callback, userdata));

    HUB.devices
        .read()
        [buffer.device_id.value].destroyed
        .lock()
        .map(buffer_id, buffer.life_guard.ref_count.clone());
}

#[no_mangle]
pub extern "C" fn wgpu_buffer_unmap(
    buffer_id: BufferId,
) {
    let mut buffer_guard = HUB.buffers.write();
    let buffer = &mut buffer_guard[buffer_id];
    let device_guard = HUB.devices.read();
    let device = &device_guard[buffer.device_id.value];

    if !buffer.mapped_write_ranges.is_empty() {
        unsafe { device.raw.flush_mapped_memory_ranges( buffer.mapped_write_ranges.iter().map(|r| {(&buffer.memory, r.clone())}) ).unwrap() }; // TODO
        buffer.mapped_write_ranges.clear();
    }

    unsafe { device.raw.unmap_memory(&buffer.memory) };
}
