use crate::{
    binding_model,
    command,
    conv,
    hub::{HUB, Root, Token},
    pipeline,
    resource,
    swap_chain,
    track::{Stitch, TrackerSet},
    AdapterId,
    BindGroupId,
    BufferAddress,
    BufferId,
    BufferMapAsyncStatus,
    BufferMapOperation,
    CommandBufferId,
    DeviceId,
    LifeGuard,
    QueueId,
    RefCount,
    Stored,
    SubmissionIndex,
    SurfaceId,
    TextureId,
    TextureViewId,
};
#[cfg(feature = "local")]
use crate::{
    BindGroupLayoutId,
    CommandEncoderId,
    ComputePipelineId,
    PipelineLayoutId,
    RenderPipelineId,
    SamplerId,
    ShaderModuleId,
    SwapChainId,
};

use arrayvec::ArrayVec;
use back;
use copyless::VecHelper as _;
use hal::{
    self,
    backend::FastHashMap,
    command::RawCommandBuffer,
    queue::RawCommandQueue,
    Device as _,
    Surface as _,
};
use log::{info, trace};
use parking_lot::Mutex;
use rendy_descriptor::{DescriptorSet, DescriptorAllocator, DescriptorRanges};
use rendy_memory::{Block, Heaps, MemoryBlock};

#[cfg(feature = "local")]
use std::sync::atomic::AtomicBool;
use std::{collections::hash_map::Entry, ffi, iter, ptr, ops::Range, slice, sync::atomic::Ordering};

const CLEANUP_WAIT_MS: u64 = 5000;
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
        self.colors.iter().chain(&self.resolves).chain(&self.depth_stencil)
    }
}

impl RenderPassContext {
    // Assumed the renderpass only contains one subpass
    pub(crate) fn compatible(&self, other: &RenderPassContext) -> bool {
        self.colors == other.colors && self.depth_stencil == other.depth_stencil
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
    BindGroup(BindGroupId),
}

#[derive(Debug)]
enum NativeResource<B: hal::Backend> {
    Buffer(B::Buffer, MemoryBlock<B>),
    Image(B::Image, MemoryBlock<B>),
    ImageView(B::ImageView),
    Framebuffer(B::Framebuffer),
    DescriptorSet(DescriptorSet<B>),
}

#[derive(Debug)]
struct ActiveSubmission<B: hal::Backend> {
    index: SubmissionIndex,
    fence: B::Fence,
    // Note: we keep the associated ID here in order to be able to check
    // at any point what resources are used in a submission.
    resources: Vec<(Option<ResourceId>, NativeResource<B>)>,
    mapped: Vec<BufferId>,
}

/// A struct responsible for tracking resource lifetimes.
///
/// Here is how host mapping is handled:
///   1. When mapping is requested we add the buffer to the pending list of `mapped` buffers.
///   2. When `triage_referenced` is called, it checks the last submission index associated with each of the mapped buffer,
/// and register the buffer with either a submission in flight, or straight into `ready_to_map` vector.
///   3. When `ActiveSubmission` is retired, the mapped buffers associated with it are moved to `ready_to_map` vector.
///   4. Finally, `handle_mapping` issues all the callbacks.

#[derive(Debug)]
struct PendingResources<B: hal::Backend> {
    /// Resources that the user has requested be mapped, but are still in use.
    mapped: Vec<Stored<BufferId>>,
    /// Resources that are destroyed by the user but still referenced by
    /// other objects or command buffers.
    referenced: Vec<(ResourceId, RefCount)>,
    /// Resources that are not referenced any more but still used by GPU.
    /// Grouped by submissions associated with a fence and a submission index.
    /// The active submissions have to be stored in FIFO order: oldest come first.
    active: Vec<ActiveSubmission<B>>,
    /// Resources that are neither referenced or used, just pending
    /// actual deletion.
    free: Vec<NativeResource<B>>,
    ready_to_map: Vec<BufferId>,
}

unsafe impl<B: hal::Backend> Send for PendingResources<B> {}
unsafe impl<B: hal::Backend> Sync for PendingResources<B> {}

impl<B: hal::Backend> PendingResources<B> {
    fn destroy(&mut self, resource_id: ResourceId, ref_count: RefCount) {
        debug_assert!(!self.referenced.iter().any(|r| r.0 == resource_id));
        self.referenced.push((resource_id, ref_count));
    }

    fn map(&mut self, buffer: BufferId, ref_count: RefCount) {
        self.mapped.push(Stored {
            value: buffer,
            ref_count,
        });
    }

    /// Returns the last submission index that is done.
    fn cleanup(
        &mut self,
        device: &B::Device,
        heaps_mutex: &Mutex<Heaps<B>>,
        descriptor_allocator_mutex: &Mutex<DescriptorAllocator<B>>,
        force_wait: bool,
    ) -> SubmissionIndex {
        if force_wait && !self.active.is_empty() {
            let status = unsafe {
                device.wait_for_fences(
                    self.active.iter().map(|a| &a.fence),
                    hal::device::WaitFor::All,
                    CLEANUP_WAIT_MS * 1_000_000,
                )
            };
            assert_eq!(status, Ok(true), "GPU got stuck :(");
        }

        //TODO: enable when `is_sorted_by_key` is stable
        //debug_assert!(self.active.is_sorted_by_key(|a| a.index));
        let done_count = self
            .active
            .iter()
            .position(|a| unsafe { !device.get_fence_status(&a.fence).unwrap() })
            .unwrap_or(self.active.len());
        let last_done = if done_count != 0 {
            self.active[done_count - 1].index
        } else {
            return 0
        };

        for a in self.active.drain(.. done_count) {
            trace!("Active submission {} is done", a.index);
            self.free.extend(a.resources.into_iter().map(|(_, r)| r));
            self.ready_to_map.extend(a.mapped);
            unsafe {
                device.destroy_fence(a.fence);
            }
        }

        let mut heaps = heaps_mutex.lock();
        let mut descriptor_allocator = descriptor_allocator_mutex.lock();
        for resource in self.free.drain(..) {
            match resource {
                NativeResource::Buffer(raw, memory) => unsafe {
                    device.destroy_buffer(raw);
                    heaps.free(device, memory);
                },
                NativeResource::Image(raw, memory) => unsafe {
                    device.destroy_image(raw);
                    heaps.free(device, memory);
                },
                NativeResource::ImageView(raw) => unsafe {
                    device.destroy_image_view(raw);
                },
                NativeResource::Framebuffer(raw) => unsafe {
                    device.destroy_framebuffer(raw);
                },
                NativeResource::DescriptorSet(raw) => unsafe {
                    descriptor_allocator.free(iter::once(raw));
                },
            }
        }

        last_done
    }
}

type DeviceToken<'a> = Token<'a, Device<back::Backend>>;

impl PendingResources<back::Backend> {
    fn triage_referenced(&mut self, trackers: &mut TrackerSet, mut token: &mut DeviceToken) {
        // Before destruction, a resource is expected to have the following strong refs:
        //  1. in resource itself
        //  2. in the device tracker
        //  3. in this list
        const MIN_REFS: usize = 4;

        if self.referenced.iter().all(|r| r.1.load() >= MIN_REFS) {
            return;
        }
        //TODO: lock less, if possible
        let (mut bind_group_guard, mut token) = HUB.bind_groups.write(&mut token);
        let (mut buffer_guard, mut token) = HUB.buffers.write(&mut token);
        let (mut texture_guard, mut token) = HUB.textures.write(&mut token);
        let (mut teview_view_guard, _) = HUB.texture_views.write(&mut token);

        for i in (0 .. self.referenced.len()).rev() {
            let num_refs = self.referenced[i].1.load();
            if num_refs <= 3 {
                let resource_id = self.referenced.swap_remove(i).0;
                assert_eq!(
                    num_refs, 3,
                    "Resource {:?} misses some references",
                    resource_id
                );
                let (life_guard, resource) = match resource_id {
                    ResourceId::Buffer(id) => {
                        if buffer_guard[id].pending_map_operation.is_some() {
                            continue
                        }
                        trackers.buffers.remove(id);
                        let buf = buffer_guard.remove(id);
                        #[cfg(feature = "local")]
                        HUB.buffers.identity.lock().free(id);
                        (buf.life_guard, NativeResource::Buffer(buf.raw, buf.memory))
                    }
                    ResourceId::Texture(id) => {
                        trackers.textures.remove(id);
                        let tex = texture_guard.remove(id);
                        #[cfg(feature = "local")]
                        HUB.textures.identity.lock().free(id);
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
                        let view = teview_view_guard.remove(id);
                        #[cfg(feature = "local")]
                        HUB.texture_views.identity.lock().free(id);
                        (view.life_guard, NativeResource::ImageView(view.raw))
                    }
                    ResourceId::BindGroup(id) => {
                        trackers.bind_groups.remove(id);
                        let bind_group = bind_group_guard.remove(id);
                        #[cfg(feature = "local")]
                        HUB.bind_groups.identity.lock().free(id);
                        (bind_group.life_guard, NativeResource::DescriptorSet(bind_group.raw))
                    }
                };

                let submit_index = life_guard.submission_index.load(Ordering::Acquire);
                match self.active.iter_mut().find(|a| a.index == submit_index) {
                    Some(a) => {
                        a.resources.alloc().init((Some(resource_id), resource));
                    }
                    None => self.free.push(resource),
                }
            }
        }
    }

    fn triage_mapped(&mut self, mut token: &mut DeviceToken) {
        if self.mapped.is_empty() {
            return;
        }
        let (buffer_guard, _) = HUB.buffers.read(&mut token);

        for stored in self.mapped.drain(..) {
            let resource_id = stored.value;
            let buf = &buffer_guard[resource_id];

            let submit_index = buf.life_guard.submission_index.load(Ordering::Acquire);
            trace!(
                "Mapping of {:?} at submission {:?} gets assigned to active {:?}",
                resource_id,
                submit_index,
                self.active.iter().position(|a| a.index == submit_index)
            );

            self.active
                .iter_mut()
                .find(|a| a.index == submit_index)
                .map_or(&mut self.ready_to_map, |a| &mut a.mapped)
                .push(resource_id);
        }
    }

    fn triage_framebuffers(
        &mut self,
        framebuffers: &mut FastHashMap<
            FramebufferKey,
            <back::Backend as hal::Backend>::Framebuffer,
        >,
        mut token: &mut DeviceToken,
    ) {
        let (texture_view_guard, _) = HUB.texture_views.read(&mut token);
        let remove_list = framebuffers
            .keys()
            .filter_map(|key| {
                let mut last_submit: SubmissionIndex = 0;
                for &at in key.all() {
                    if texture_view_guard.contains(at) {
                        return None;
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
            .collect::<FastHashMap<_, _>>();

        for (ref key, submit_index) in remove_list {
            let resource = NativeResource::Framebuffer(framebuffers.remove(key).unwrap());
            match self.active.iter_mut().find(|a| a.index == submit_index) {
                Some(a) => {
                    a.resources.alloc().init((None, resource));
                }
                None => self.free.push(resource),
            }
        }
    }

    fn handle_mapping(
        &mut self,
        raw: &<back::Backend as hal::Backend>::Device,
        mut token: &mut DeviceToken,
    ) -> Vec<BufferMapPendingCallback> {
        if self.ready_to_map.is_empty() {
            return Vec::new()
        }
        let (mut buffer_guard, _) = HUB.buffers.write(&mut token);
        self.ready_to_map
            .drain(..)
            .map(|buffer_id| {
                let buffer = &mut buffer_guard[buffer_id];
                let operation = buffer.pending_map_operation.take().unwrap();
                let result = match operation {
                    BufferMapOperation::Read(ref range, ..) => {
                        map_buffer(raw, buffer, range.clone(), HostMap::Read)
                    }
                    BufferMapOperation::Write(ref range, ..) => {
                        map_buffer(raw, buffer, range.clone(), HostMap::Write)
                    }
                };
                (operation, result)
            })
            .collect()
    }
}

type BufferMapResult = Result<*mut u8, hal::mapping::Error>;
type BufferMapPendingCallback = (BufferMapOperation, BufferMapResult);

fn map_buffer(
    raw: &<back::Backend as hal::Backend>::Device,
    buffer: &mut resource::Buffer<back::Backend>,
    buffer_range: Range<BufferAddress>,
    kind: HostMap,
) -> BufferMapResult {
    let is_coherent = buffer.memory
        .properties()
        .contains(hal::memory::Properties::COHERENT);
    let (ptr, mapped_range) = {
        let mapped = buffer.memory.map(raw, buffer_range)?;
        (mapped.ptr(), mapped.range())
    };

    if !is_coherent {
        match kind {
            HostMap::Read => unsafe {
                raw.invalidate_mapped_memory_ranges(
                    iter::once((buffer.memory.memory(), mapped_range)),
                    )
                    .unwrap();
            },
            HostMap::Write => {
                buffer.mapped_write_ranges.push(mapped_range);
            }
        }
    }

    Ok(ptr.as_ptr())
}

pub struct Device<B: hal::Backend> {
    pub(crate) raw: B::Device,
    pub(crate) adapter_id: AdapterId,
    pub(crate) queue_group: hal::QueueGroup<B, hal::General>,
    pub(crate) com_allocator: command::CommandAllocator<B>,
    mem_allocator: Mutex<Heaps<B>>,
    desc_allocator: Mutex<DescriptorAllocator<B>>,
    life_guard: LifeGuard,
    pub(crate) trackers: Mutex<TrackerSet>,
    pub(crate) render_passes: Mutex<FastHashMap<RenderPassKey, B::RenderPass>>,
    pub(crate) framebuffers: Mutex<FastHashMap<FramebufferKey, B::Framebuffer>>,
    pending: Mutex<PendingResources<B>>,
}

impl<B: hal::Backend> Device<B> {
    pub(crate) fn new(
        raw: B::Device,
        adapter_id: AdapterId,
        queue_group: hal::QueueGroup<B, hal::General>,
        mem_props: hal::MemoryProperties,
    ) -> Self {
        // don't start submission index at zero
        let life_guard = LifeGuard::new();
        life_guard.submission_index.fetch_add(1, Ordering::Relaxed);

        let heaps = {
            let types = mem_props.memory_types
                .iter()
                .map(|mt| {
                    use rendy_memory::{HeapsConfig, LinearConfig, DynamicConfig};
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
                    (mt.properties.into(), mt.heap_index as u32, config)
                });
            unsafe {
                Heaps::new(types, mem_props.memory_heaps.iter().cloned())
            }

        };

        Device {
            raw,
            adapter_id,
            com_allocator: command::CommandAllocator::new(queue_group.family()),
            mem_allocator: Mutex::new(heaps),
            desc_allocator: Mutex::new(DescriptorAllocator::new()),
            queue_group,
            life_guard,
            trackers: Mutex::new(TrackerSet::new()),
            render_passes: Mutex::new(FastHashMap::default()),
            framebuffers: Mutex::new(FastHashMap::default()),
            pending: Mutex::new(PendingResources {
                mapped: Vec::new(),
                referenced: Vec::new(),
                active: Vec::new(),
                free: Vec::new(),
                ready_to_map: Vec::new(),
            }),
        }
    }
}

impl Device<back::Backend> {
    fn maintain(
        &self, force_wait: bool, token: &mut DeviceToken,
    ) -> Vec<BufferMapPendingCallback> {
        let mut pending = self.pending.lock();
        let mut trackers = self.trackers.lock();

        pending.triage_referenced(&mut *trackers, token);
        pending.triage_mapped(token);
        pending.triage_framebuffers(&mut *self.framebuffers.lock(), token);
        let last_done = pending.cleanup(&self.raw, &self.mem_allocator, &self.desc_allocator, force_wait);
        let callbacks = pending.handle_mapping(&self.raw, token);

        unsafe {
            self.desc_allocator
                .lock()
                .cleanup(&self.raw);
        }

        if last_done != 0 {
            self.com_allocator.maintain(last_done);
        }

        callbacks
    }

    //Note: this logic is specifically moved out of `handle_mapping()` in order to
    // have nothing locked by the time we execute users callback code.
    fn fire_map_callbacks<I: IntoIterator<Item = BufferMapPendingCallback>>(callbacks: I) {
        for (operation, result) in callbacks {
            let (status, ptr) = match result {
                Ok(ptr) => (BufferMapAsyncStatus::Success, ptr),
                Err(e) => {
                    log::error!("failed to map buffer: {}", e);
                    (BufferMapAsyncStatus::Error, ptr::null_mut())
                }
            };
            match operation {
                BufferMapOperation::Read(_, on_read, userdata) => on_read(status, ptr, userdata),
                BufferMapOperation::Write(_, on_write, userdata) => on_write(status, ptr, userdata),
            }
        }
    }
}

#[derive(Debug)]
pub struct ShaderModule<B: hal::Backend> {
    pub(crate) raw: B::ShaderModule,
}

pub fn device_create_buffer(
    device_id: DeviceId,
    desc: &resource::BufferDescriptor,
) -> resource::Buffer<back::Backend> {
    let (device_guard, _) = HUB.devices.read(&mut Token::root());
    let device = &device_guard[device_id];
    let (usage, _memory_properties) = conv::map_buffer_usage(desc.usage);

    let rendy_usage = {
        use resource::BufferUsage as Bu;
        use rendy_memory::MemoryUsageValue as Muv;

        if !desc.usage.intersects(Bu::MAP_READ | Bu::MAP_WRITE) {
            Muv::Data
        } else if (Bu::MAP_WRITE | Bu::TRANSFER_SRC).contains(desc.usage) {
            Muv::Upload
        } else if (Bu::MAP_READ | Bu::TRANSFER_DST).contains(desc.usage) {
            Muv::Download
        } else {
            Muv::Dynamic
        }
    };

    let mut buffer = unsafe { device.raw.create_buffer(desc.size, usage).unwrap() };
    let requirements = unsafe { device.raw.get_buffer_requirements(&buffer) };
    let memory = device.mem_allocator
        .lock()
        .allocate(
            &device.raw,
            requirements.type_mask as u32,
            rendy_usage,
            requirements.size,
            requirements.alignment,
        )
        .unwrap();

    unsafe {
        device.raw
            .bind_buffer_memory(memory.memory(), memory.range().start, &mut buffer)
            .unwrap()
    };

    resource::Buffer {
        raw: buffer,
        device_id: Stored {
            value: device_id,
            ref_count: device.life_guard.ref_count.clone(),
        },
        memory,
        size: desc.size,
        mapped_write_ranges: Vec::new(),
        pending_map_operation: None,
        life_guard: LifeGuard::new(),
    }
}

pub fn device_track_buffer(
    device_id: DeviceId,
    buffer_id: BufferId,
    ref_count: RefCount,
    flags: resource::BufferUsage,
    token: &mut Token<Root>,
) {
    let (device_guard, _) = HUB.devices.read(token);
    let ok = device_guard[device_id]
        .trackers
        .lock()
        .buffers
        .init(buffer_id, &ref_count, (), flags);
    assert!(ok);
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_buffer(
    device_id: DeviceId,
    desc: &resource::BufferDescriptor,
) -> BufferId {
    let buffer = device_create_buffer(device_id, desc);
    let ref_count = buffer.life_guard.ref_count.clone();
    let mut token = Token::root();
    let id = HUB.buffers.register_local(buffer, &mut token);
    device_track_buffer(device_id, id, ref_count, resource::BufferUsage::empty(), &mut token);
    id
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_buffer_mapped(
    device_id: DeviceId,
    desc: &resource::BufferDescriptor,
    mapped_ptr_out: *mut *mut u8,
) -> BufferId {
    let mut desc = desc.clone();
    desc.usage |= resource::BufferUsage::MAP_WRITE;
    let mut buffer = device_create_buffer(device_id, &desc);
    let mut token = Token::root();

    {
        let (device_guard, _) = HUB.devices.read(&mut token);
        let device = &device_guard[device_id];

        match map_buffer(
            &device.raw,
            &mut buffer,
            0 .. desc.size,
            HostMap::Write,
        ) {
            Ok(ptr) => unsafe {
                *mapped_ptr_out = ptr;
            },
            Err(e) => {
                log::error!("failed to create buffer in a mapped state: {}", e);
                unsafe {
                    *mapped_ptr_out = ptr::null_mut();
                }
            }
        }
    }

    let ref_count = buffer.life_guard.ref_count.clone();
    let id = HUB.buffers.register_local(buffer, &mut token);
    device_track_buffer(device_id, id, ref_count, resource::BufferUsage::MAP_WRITE,  &mut token);
    id
}

#[no_mangle]
pub extern "C" fn wgpu_buffer_destroy(buffer_id: BufferId) {
    let mut token = Token::root();
    let (device_guard, mut token) = HUB.devices.read(&mut token);
    let (buffer_guard, _) = HUB.buffers.read(&mut token);
    let buffer = &buffer_guard[buffer_id];
    device_guard[buffer.device_id.value].pending.lock().destroy(
        ResourceId::Buffer(buffer_id),
        buffer.life_guard.ref_count.clone(),
    );
}

pub fn device_create_texture(
    device_id: DeviceId,
    desc: &resource::TextureDescriptor,
) -> resource::Texture<back::Backend> {
    let kind = conv::map_texture_dimension_size(desc.dimension, desc.size, desc.array_layer_count, desc.sample_count);
    let format = conv::map_texture_format(desc.format);
    let aspects = format.surface_desc().aspects;
    let usage = conv::map_texture_usage(desc.usage, aspects);

    let (device_guard, _) = HUB.devices.read(&mut Token::root());
    let device = &device_guard[device_id];

    assert!((desc.mip_level_count as usize) < MAX_MIP_LEVELS);

    let mut image = unsafe {
        device.raw.create_image(
            kind,
            desc.mip_level_count as hal::image::Level,
            format,
            hal::image::Tiling::Optimal,
            usage,
            hal::image::ViewCapabilities::empty(), // TODO: format, 2d array, cube
        )
    }
    .unwrap();
    let requirements = unsafe { device.raw.get_image_requirements(&image) };

    let memory = device.mem_allocator
        .lock()
        .allocate(
            &device.raw,
            requirements.type_mask as u32,
            rendy_memory::Data,
            requirements.size,
            requirements.alignment,
        )
        .unwrap();

    unsafe {
        device.raw
            .bind_image_memory(memory.memory(), memory.range().start, &mut image)
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
            levels: 0 .. desc.mip_level_count as hal::image::Level,
            layers: 0 .. desc.array_layer_count as hal::image::Layer,
        },
        placement: resource::TexturePlacement::Memory(memory),
        life_guard: LifeGuard::new(),
    }
}

pub fn device_track_texture(
    device_id: DeviceId,
    texture_id: TextureId,
    ref_count: RefCount,
    full_range: hal::image::SubresourceRange,
    token: &mut Token<Root>,
) {
    let (device_guard, _) = HUB.devices.read(token);
    let ok = device_guard[device_id]
        .trackers
        .lock()
        .textures
        .init(
            texture_id,
            &ref_count,
            full_range,
            resource::TextureUsage::UNINITIALIZED,
        );
    assert!(ok);
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_texture(
    device_id: DeviceId,
    desc: &resource::TextureDescriptor,
) -> TextureId {
    let texture = device_create_texture(device_id, desc);
    let ref_count = texture.life_guard.ref_count.clone();
    let range = texture.full_range.clone();
    let mut token = Token::root();
    let id = HUB.textures.register_local(texture, &mut token);
    device_track_texture(device_id, id, ref_count, range, &mut token);
    id
}

pub fn texture_create_view(
    texture_id: TextureId,
    format: resource::TextureFormat,
    view_kind: hal::image::ViewKind,
    range: hal::image::SubresourceRange,
    token: &mut Token<Root>,
) -> resource::TextureView<back::Backend> {
    let (device_guard, mut token) = HUB.devices.read(token);
    let (texture_guard, _) = HUB.textures.read(&mut token);
    let texture = &texture_guard[texture_id];

    let raw = unsafe {
        device_guard[texture.device_id.value]
            .raw
            .create_image_view(
                &texture.raw,
                view_kind,
                conv::map_texture_format(format),
                hal::format::Swizzle::NO,
                range.clone(),
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
        extent: texture.kind.extent().at_level(range.levels.start),
        samples: texture.kind.num_samples(),
        range,
        is_owned_by_swap_chain: false,
        life_guard: LifeGuard::new(),
    }
}

pub fn device_track_view(
    texture_id: TextureId,
    view_id: TextureViewId,
    ref_count: RefCount,
    token: &mut Token<Root>,
) {
    let (device_guard, mut token) = HUB.devices.read(token);
    let (texture_guard, _) = HUB.textures.read(&mut token);

    let device_id = texture_guard[texture_id].device_id.value;
    let ok = device_guard[device_id]
        .trackers
        .lock()
        .views
        .init(view_id, &ref_count, (), ());
    assert!(ok);
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_texture_create_view(
    texture_id: TextureId,
    desc: &resource::TextureViewDescriptor,
) -> TextureViewId {
    let mut token = Token::root();
    let view = texture_create_view(
        texture_id,
        desc.format,
        conv::map_texture_view_dimension(desc.dimension),
        hal::image::SubresourceRange {
            aspects: conv::map_texture_aspect_flags(desc.aspect),
            levels: desc.base_mip_level as u8 .. (desc.base_mip_level + desc.level_count) as u8,
            layers: desc.base_array_layer as u16
                .. (desc.base_array_layer + desc.array_count) as u16,
        },
        &mut token,
    );
    let ref_count = view.life_guard.ref_count.clone();
    let id = HUB.texture_views.register_local(view, &mut token);
    device_track_view(texture_id, id, ref_count, &mut token);
    id
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_texture_create_default_view(texture_id: TextureId) -> TextureViewId {
    let mut token = Token::root();
    let (format, view_kind, range) = {
        let (texture_guard, _) = HUB.textures.read(&mut token);
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
    let view = texture_create_view(texture_id, format, view_kind, range, &mut token);
    let ref_count = view.life_guard.ref_count.clone();
    let id = HUB.texture_views.register_local(view, &mut token);
    device_track_view(texture_id, id, ref_count, &mut token);
    id
}

#[no_mangle]
pub extern "C" fn wgpu_texture_destroy(texture_id: TextureId) {
    let mut token = Token::root();
    let (device_guard, mut token) = HUB.devices.read(&mut token);
    let (texture_guard, _) = HUB.textures.read(&mut token);
    let texture = &texture_guard[texture_id];
    device_guard[texture.device_id.value]
        .pending
        .lock()
        .destroy(
            ResourceId::Texture(texture_id),
            texture.life_guard.ref_count.clone(),
        );
}

#[no_mangle]
pub extern "C" fn wgpu_texture_view_destroy(texture_view_id: TextureViewId) {
    let mut token = Token::root();
    let (device_guard, mut token) = HUB.devices.read(&mut token);
    let (texture_guard, mut token) = HUB.textures.read(&mut token);
    let (texture_view_guard, _) = HUB.texture_views.read(&mut token);
    let view = &texture_view_guard[texture_view_id];
    let device_id = texture_guard[view.texture_id.value].device_id.value;
    device_guard[device_id].pending.lock().destroy(
        ResourceId::TextureView(texture_view_id),
        view.life_guard.ref_count.clone(),
    );
}

pub fn device_create_sampler(
    device_id: DeviceId,
    desc: &resource::SamplerDescriptor,
    token: &mut Token<Root>,
) -> resource::Sampler<back::Backend> {
    let (device_guard, _) = HUB.devices.read(token);
    let device = &device_guard[device_id];

    let info = hal::image::SamplerInfo {
        min_filter: conv::map_filter(desc.min_filter),
        mag_filter: conv::map_filter(desc.mag_filter),
        mip_filter: conv::map_filter(desc.mipmap_filter),
        wrap_mode: (
            conv::map_wrap(desc.address_mode_u),
            conv::map_wrap(desc.address_mode_v),
            conv::map_wrap(desc.address_mode_w),
        ),
        lod_bias: 0.0.into(),
        lod_range: desc.lod_min_clamp.into() .. desc.lod_max_clamp.into(),
        comparison: if desc.compare_function == resource::CompareFunction::Always {
            None
        } else {
            Some(conv::map_compare_function(desc.compare_function))
        },
        border: hal::image::PackedColor(0),
        anisotropic: hal::image::Anisotropic::Off, //TODO
    };
    let raw = unsafe { device.raw.create_sampler(info).unwrap() };

    resource::Sampler { raw }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_sampler(
    device_id: DeviceId,
    desc: &resource::SamplerDescriptor,
) -> SamplerId {
    let mut token = Token::root();
    let sampler = device_create_sampler(device_id, desc, &mut token);
    HUB.samplers.register_local(sampler, &mut token)
}

pub fn device_create_bind_group_layout(
    device_id: DeviceId,
    desc: &binding_model::BindGroupLayoutDescriptor,
    token: &mut Token<Root>,
) -> binding_model::BindGroupLayout<back::Backend> {
    let bindings = unsafe { slice::from_raw_parts(desc.bindings, desc.bindings_length) };

    let raw_bindings = bindings
        .iter()
        .map(|binding| hal::pso::DescriptorSetLayoutBinding {
            binding: binding.binding,
            ty: conv::map_binding_type(binding.ty),
            count: 1, //TODO: consolidate
            stage_flags: conv::map_shader_stage_flags(binding.visibility),
            immutable_samplers: false, // TODO
        })
        .collect::<Vec<_>>(); //TODO: avoid heap allocation

    let raw = unsafe {
        let (device_guard, _) = HUB.devices.read(token);
        device_guard[device_id]
            .raw
            .create_descriptor_set_layout(&raw_bindings, &[])
            .unwrap()
    };

    binding_model::BindGroupLayout {
        raw,
        bindings: bindings.to_vec(),
        desc_ranges: DescriptorRanges::from_bindings(&raw_bindings),
        dynamic_count: bindings
            .iter()
            .filter(|b| match b.ty {
                binding_model::BindingType::UniformBufferDynamic
                | binding_model::BindingType::StorageBufferDynamic => true,
                _ => false,
            })
            .count(),
    }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_bind_group_layout(
    device_id: DeviceId,
    desc: &binding_model::BindGroupLayoutDescriptor,
) -> BindGroupLayoutId {
    let mut token = Token::root();
    let layout = device_create_bind_group_layout(device_id, desc, &mut token);
    HUB.bind_group_layouts.register_local(layout, &mut token)
}

pub fn device_create_pipeline_layout(
    device_id: DeviceId,
    desc: &binding_model::PipelineLayoutDescriptor,
    token: &mut Token<Root>,
) -> binding_model::PipelineLayout<back::Backend> {
    let (device_guard, mut token) = HUB.devices.read(token);
    let bind_group_layout_ids =
        unsafe { slice::from_raw_parts(desc.bind_group_layouts, desc.bind_group_layouts_length) };
    let (bind_group_layout_guard, _) = HUB.bind_group_layouts.read(&mut token);
    let descriptor_set_layouts = bind_group_layout_ids
        .iter()
        .map(|&id| &bind_group_layout_guard[id].raw);

    // TODO: push constants
    let pipeline_layout = unsafe {
        device_guard[device_id]
            .raw
            .create_pipeline_layout(descriptor_set_layouts, &[])
    }
    .unwrap();

    binding_model::PipelineLayout {
        raw: pipeline_layout,
        bind_group_layout_ids: bind_group_layout_ids.iter().cloned().collect(),
    }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_pipeline_layout(
    device_id: DeviceId,
    desc: &binding_model::PipelineLayoutDescriptor,
) -> PipelineLayoutId {
    let mut token = Token::root();
    let layout = device_create_pipeline_layout(device_id, desc, &mut token);
    HUB.pipeline_layouts.register_local(layout, &mut token)
}

pub fn device_create_bind_group(
    device_id: DeviceId,
    desc: &binding_model::BindGroupDescriptor,
    token: &mut Token<Root>,
) -> binding_model::BindGroup<back::Backend> {
    let (device_guard, mut token) = HUB.devices.read(token);
    let device = &device_guard[device_id];
    let (bind_group_layout_guard, _) = HUB.bind_group_layouts.read(&mut token);
    let bind_group_layout = &bind_group_layout_guard[desc.layout];
    let bindings = unsafe { slice::from_raw_parts(desc.bindings, desc.bindings_length as usize) };
    assert_eq!(bindings.len(), bind_group_layout.bindings.len());

    let desc_set = unsafe {
        let mut desc_sets = ArrayVec::<[_; 1]>::new();
        device.desc_allocator
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

    let (buffer_guard, mut token) = HUB.buffers.read(&mut token);
    let (_, mut token) = HUB.textures.read(&mut token); //skip token
    let (texture_view_guard, mut token) = HUB.texture_views.read(&mut token);
    let (sampler_guard, _) = HUB.samplers.read(&mut token);

    //TODO: group writes into contiguous sections
    let mut writes = Vec::new();
    let mut used = TrackerSet::new();
    for (b, decl) in bindings.iter().zip(&bind_group_layout.bindings) {
        let descriptor = match b.resource {
            binding_model::BindingResource::Buffer(ref bb) => {
                let (alignment, usage) = match decl.ty {
                    binding_model::BindingType::UniformBuffer
                    | binding_model::BindingType::UniformBufferDynamic => {
                        (BIND_BUFFER_ALIGNMENT, resource::BufferUsage::UNIFORM)
                    }
                    binding_model::BindingType::StorageBuffer
                    | binding_model::BindingType::StorageBufferDynamic => {
                        (BIND_BUFFER_ALIGNMENT, resource::BufferUsage::STORAGE)
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
                let range = Some(bb.offset) .. Some(bb.offset + bb.size);
                hal::pso::Descriptor::Buffer(&buffer.raw, range)
            }
            binding_model::BindingResource::Sampler(id) => {
                assert_eq!(decl.ty, binding_model::BindingType::Sampler);
                let sampler = &sampler_guard[id];
                hal::pso::Descriptor::Sampler(&sampler.raw)
            }
            binding_model::BindingResource::TextureView(id) => {
                let (usage, image_layout) = match decl.ty {
                    binding_model::BindingType::SampledTexture => (
                        resource::TextureUsage::SAMPLED,
                        hal::image::Layout::ShaderReadOnlyOptimal,
                    ),
                    binding_model::BindingType::StorageTexture => (
                        resource::TextureUsage::STORAGE,
                        hal::image::Layout::General,
                    ),
                    _ => panic!("Mismatched texture binding for {:?}", decl),
                };
                let view = used.views
                    .use_extend(&*texture_view_guard, id, (), ())
                    .unwrap();
                used.textures
                    .change_extend(
                        view.texture_id.value,
                        &view.texture_id.ref_count,
                        view.range.clone(),
                        usage,
                    )
                    .unwrap();
                hal::pso::Descriptor::Image(&view.raw, image_layout)
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

    binding_model::BindGroup {
        raw: desc_set,
        device_id: Stored {
            value: device_id,
            ref_count: device.life_guard.ref_count.clone(),
        },
        layout_id: desc.layout,
        life_guard: LifeGuard::new(),
        used,
        dynamic_count: bind_group_layout.dynamic_count,
    }
}

pub fn device_track_bind_group(
    device_id: DeviceId,
    bind_group_id: BindGroupId,
    ref_count: RefCount,
    token: &mut Token<Root>,
) {
    let (device_guard, _) = HUB.devices.read(token);
    let ok = device_guard[device_id]
        .trackers
        .lock()
        .bind_groups
        .init(bind_group_id, &ref_count, (), ());
    assert!(ok);
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_bind_group(
    device_id: DeviceId,
    desc: &binding_model::BindGroupDescriptor,
) -> BindGroupId {
    let mut token = Token::root();
    let bind_group = device_create_bind_group(device_id, desc, &mut token);
    let ref_count = bind_group.life_guard.ref_count.clone();
    let id = HUB.bind_groups.register_local(bind_group, &mut token);
    device_track_bind_group(device_id, id, ref_count, &mut token);
    id
}

#[no_mangle]
pub extern "C" fn wgpu_bind_group_destroy(bind_group_id: BindGroupId) {
    let mut token = Token::root();
    let (device_guard, mut token) = HUB.devices.read(&mut token);
    let (bind_group_guard, _) = HUB.bind_groups.read(&mut token);
    let bind_group = &bind_group_guard[bind_group_id];
    device_guard[bind_group.device_id.value].pending.lock().destroy(
        ResourceId::BindGroup(bind_group_id),
        bind_group.life_guard.ref_count.clone(),
    );
}

pub fn device_create_shader_module(
    device_id: DeviceId,
    desc: &pipeline::ShaderModuleDescriptor,
    token: &mut Token<Root>,
) -> ShaderModule<back::Backend> {
    let spv = unsafe { slice::from_raw_parts(desc.code.bytes, desc.code.length) };
    let (device_guard, _) = HUB.devices.read(token);
    let shader = unsafe {
        device_guard[device_id]
            .raw
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
    let mut token = Token::root();
    let module = device_create_shader_module(device_id, desc, &mut token);
    HUB.shader_modules.register_local(module, &mut token)
}

pub fn device_create_command_encoder(
    device_id: DeviceId,
    _desc: &command::CommandEncoderDescriptor,
    token: &mut Token<Root>,
) -> command::CommandBuffer<back::Backend> {
    let (device_guard, _) = HUB.devices.read(token);
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
    let mut token = Token::root();
    let cmb = device_create_command_encoder(device_id, desc, &mut token);
    HUB.command_buffers.register_local(cmb, &mut token)
}

#[no_mangle]
pub extern "C" fn wgpu_device_get_queue(device_id: DeviceId) -> QueueId {
    device_id
}

#[no_mangle]
pub extern "C" fn wgpu_queue_submit(
    queue_id: QueueId,
    command_buffers: *const CommandBufferId,
    command_buffers_length: usize,
) {
    let command_buffer_ids =
        unsafe { slice::from_raw_parts(command_buffers, command_buffers_length) };

    let (submit_index, fence) = {
        let mut token = Token::root();
        let (surface_guard, mut token) = HUB.surfaces.read(&mut token);
        let (mut device_guard, mut token) = HUB.devices.write(&mut token);
        let device = &mut device_guard[queue_id];
        let mut trackers = device.trackers.lock();
        let mut wait_semaphores = Vec::new();

        let submit_index = 1 + device
            .life_guard
            .submission_index
            .fetch_add(1, Ordering::Relaxed);

        //TODO: if multiple command buffers are submitted, we can re-use the last
        // native command buffer of the previous chain instead of always creating
        // a temporary one, since the chains are not finished.
        {
            let (mut command_buffer_guard, mut token) = HUB.command_buffers.write(&mut token);
            let (bind_group_guard, mut token) = HUB.bind_groups.read(&mut token);
            let (buffer_guard, mut token) = HUB.buffers.read(&mut token);
            let (texture_guard, mut token) = HUB.textures.read(&mut token);
            let (texture_view_guard, _) = HUB.texture_views.read(&mut token);

            // finish all the command buffers first
            for &cmb_id in command_buffer_ids {
                let comb = &mut command_buffer_guard[cmb_id];
                for link in comb.swap_chain_links.drain(..) {
                    let swap_chain = surface_guard[link.swap_chain_id].swap_chain.as_ref().unwrap();
                    let frame = &swap_chain.frames[link.image_index as usize];
                    if frame.need_waiting.swap(false, Ordering::AcqRel) {
                        assert_eq!(frame.acquired_epoch, Some(link.epoch),
                            "{}. Image index {} with epoch {} != current epoch {:?}",
                            "Attempting to render to a swapchain output that has already been presented",
                            link.image_index, link.epoch, frame.acquired_epoch);
                        wait_semaphores.push((
                            &frame.sem_available,
                            hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                        ));
                    }
                }

                // optimize the tracked states
                comb.trackers.optimize();

                // update submission IDs
                for id in comb.trackers.buffers.used() {
                    let buffer = &buffer_guard[id];
                    assert!(buffer.pending_map_operation.is_none());
                    buffer
                        .life_guard
                        .submission_index
                        .store(submit_index, Ordering::Release);
                }
                for id in comb.trackers.textures.used() {
                    texture_guard[id]
                        .life_guard
                        .submission_index
                        .store(submit_index, Ordering::Release);
                }
                for id in comb.trackers.views.used() {
                    texture_view_guard[id]
                        .life_guard
                        .submission_index
                        .store(submit_index, Ordering::Release);
                }
                for id in comb.trackers.bind_groups.used() {
                    bind_group_guard[id]
                        .life_guard
                        .submission_index
                        .store(submit_index, Ordering::Release);
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
            let (command_buffer_guard, _) = HUB.command_buffers.read(&mut token);
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

        (submit_index, fence)
    };

    // No need for write access to the device from here on out
    let callbacks = {
        let mut token = Token::root();
        let (device_guard, mut token) = HUB.devices.read(&mut token);
        let device = &device_guard[queue_id];

        let callbacks = device.maintain(false, &mut token);
        device.pending.lock().active.alloc().init(ActiveSubmission {
            index: submit_index,
            fence,
            resources: Vec::new(),
            mapped: Vec::new(),
        });

        // finally, return the command buffers to the allocator
        for &cmb_id in command_buffer_ids {
            let (cmd_buf, _) = HUB.command_buffers.unregister(cmb_id, &mut token);
            device.com_allocator.after_submit(cmd_buf, submit_index);
        }

        callbacks
    };

    Device::fire_map_callbacks(callbacks);
}

pub fn device_create_render_pipeline(
    device_id: DeviceId,
    desc: &pipeline::RenderPipelineDescriptor,
    token: &mut Token<Root>,
) -> pipeline::RenderPipeline<back::Backend> {
    let sc = desc.sample_count;
    assert!(sc == 1 || sc == 2 || sc == 4 || sc == 8 || sc == 16 || sc == 32,
        "Invalid sample_count of {}", sc);
    let sc = sc as u8;

    let (device_guard, mut token) = HUB.devices.read(token);
    let device = &device_guard[device_id];
    let (pipeline_layout_guard, mut token) = HUB.pipeline_layouts.read(&mut token);
    let layout = &pipeline_layout_guard[desc.layout].raw;
    let (shader_module_guard, _) = HUB.shader_modules.read(&mut token);

    let color_states =
        unsafe { slice::from_raw_parts(desc.color_states, desc.color_states_length) };
    let depth_stencil_state = unsafe { desc.depth_stencil_state.as_ref() };

    let rp_key = RenderPassKey {
        colors: color_states
            .iter()
            .map(|at| hal::pass::Attachment {
                format: Some(conv::map_texture_format(at.format)),
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
            format: Some(conv::map_texture_format(at.format)),
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

    let vertex = hal::pso::EntryPoint::<back::Backend> {
        entry: unsafe { ffi::CStr::from_ptr(desc.vertex_stage.entry_point) }
            .to_str()
            .to_owned()
            .unwrap(), // TODO
        module: &shader_module_guard[desc.vertex_stage.module].raw,
        specialization: hal::pso::Specialization::EMPTY,
    };
    let fragment = unsafe { desc.fragment_stage.as_ref() }.map(|stage| hal::pso::EntryPoint::<
        back::Backend,
    > {
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
    let rasterizer = conv::map_rasterization_state_descriptor(&desc.rasterization_state);

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
        vertex_strides.alloc().init((vb_state.stride, vb_state.step_mode));
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

    let multisampling: Option<hal::pso::Multisampling> = if sc == 1 {
        None
    } else {
        Some(hal::pso::Multisampling {
            rasterization_samples: sc,
            sample_shading: None,
            sample_mask: !0,
            alpha_coverage: false,
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
        device
            .raw
            .create_graphics_pipeline(&pipeline_desc, None)
            .unwrap()
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

    pipeline::RenderPipeline {
        raw: pipeline,
        layout_id: desc.layout,
        pass_context,
        flags,
        index_format: desc.vertex_input.index_format,
        vertex_strides,
        sample_count: sc,
    }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_device_create_render_pipeline(
    device_id: DeviceId,
    desc: &pipeline::RenderPipelineDescriptor,
) -> RenderPipelineId {
    let mut token = Token::root();
    let pipeline = device_create_render_pipeline(device_id, desc, &mut token);
    HUB.render_pipelines.register_local(pipeline, &mut token)
}

pub fn device_create_compute_pipeline(
    device_id: DeviceId,
    desc: &pipeline::ComputePipelineDescriptor,
    token: &mut Token<Root>,
) -> pipeline::ComputePipeline<back::Backend> {
    let (device_guard, mut token) = HUB.devices.read(token);
    let device = &device_guard[device_id].raw;
    let (pipeline_layout_guard, mut token) = HUB.pipeline_layouts.read(&mut token);
    let layout = &pipeline_layout_guard[desc.layout].raw;
    let pipeline_stage = &desc.compute_stage;
    let (shader_module_guard, _) = HUB.shader_modules.read(&mut token);

    let shader = hal::pso::EntryPoint::<back::Backend> {
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
    let mut token = Token::root();
    let pipeline = device_create_compute_pipeline(device_id, desc, &mut token);
    HUB.compute_pipelines.register_local(pipeline, &mut token)
}

pub fn device_create_swap_chain(
    device_id: DeviceId,
    surface_id: SurfaceId,
    desc: &swap_chain::SwapChainDescriptor,
    token: &mut Token<Root>,
) -> Vec<resource::Texture<back::Backend>> {
    info!("creating swap chain {:?}", desc);

    let (mut surface_guard, mut token) = HUB.surfaces.write(token);
    let (adapter_guard, mut token) = HUB.adapters.read(&mut token);
    let (device_guard, _) = HUB.devices.read(&mut token);
    let device = &device_guard[device_id];
    let surface = &mut surface_guard[surface_id];

    let (caps, formats, _present_modes) = {
        let adapter = &adapter_guard[device.adapter_id];
        assert!(surface
            .raw
            .supports_queue_family(&adapter.queue_families[0]));
        surface.raw.compatibility(&adapter.physical_device)
    };
    let num_frames = caps.image_count.start; //TODO: configure?
    let usage = conv::map_texture_usage(desc.usage, hal::format::Aspects::COLOR);
    let mut config = hal::SwapchainConfig::new(
        desc.width,
        desc.height,
        conv::map_texture_format(desc.format),
        num_frames, //TODO: configure?
    );
    //TODO: check for supported
    config.composite_alpha = hal::window::CompositeAlpha::OPAQUE;

    if let Some(formats) = formats {
        assert!(
            formats.contains(&config.format),
            "Requested format {:?} is not in supported list: {:?}",
            config.format,
            formats
        );
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
            let mut pending = device.pending.lock();
            assert_eq!(old.device_id.value, device_id);
            for frame in old.frames {
                pending.destroy(
                    ResourceId::Texture(frame.texture_id.value),
                    frame.texture_id.ref_count,
                );
                pending.destroy(
                    ResourceId::TextureView(frame.view_id.value),
                    frame.view_id.ref_count,
                );
            }
            unsafe { old.command_pool.reset() };
            (Some(old.raw), old.sem_available, old.command_pool)
        }
        _ => unsafe {
            let sem_available = device.raw.create_semaphore().unwrap();
            let command_pool = device
                .raw
                .create_command_pool_typed(
                    &device.queue_group,
                    hal::pool::CommandPoolCreateFlags::RESET_INDIVIDUAL,
                )
                .unwrap();
            (None, sem_available, command_pool)
        },
    };

    let (raw, images) = unsafe {
        device
            .raw
            .create_swapchain(&mut surface.raw, config.with_image_usage(usage), old_raw)
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
    token: &mut Token<Root>,
) {
    let (mut surface_guard, mut token) = HUB.surfaces.write(token);
    let swap_chain = surface_guard[swap_chain_id].swap_chain.as_mut().unwrap();
    let (device_guard, mut token) = HUB.devices.read(&mut token);
    let device = &device_guard[swap_chain.device_id.value];
    let mut trackers = device.trackers.lock();

    for (i, mut texture) in textures.into_iter().enumerate() {
        let format = texture.format;
        let kind = texture.kind;
        let range = texture.full_range.clone();

        let view_raw = unsafe {
            device
                .raw
                .create_image_view(
                    &texture.raw,
                    hal::image::ViewKind::D2,
                    conv::map_texture_format(format),
                    hal::format::Swizzle::NO,
                    range.clone(),
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
            value: HUB.textures.register_local(texture, &mut token),
        };
        trackers.textures.init(
            texture_id.value,
            &texture_id.ref_count,
            range.clone(),
            resource::TextureUsage::UNINITIALIZED,
        );

        let view = resource::TextureView {
            raw: view_raw,
            texture_id: texture_id.clone(),
            format,
            extent: kind.extent(),
            samples: kind.num_samples(),
            range,
            is_owned_by_swap_chain: true,
            life_guard: LifeGuard::new(),
        };
        let view_id = Stored {
            ref_count: view.life_guard.ref_count.clone(),
            value: HUB.texture_views.register_local(view, &mut token),
        };
        trackers
            .views
            .init(view_id.value, &view_id.ref_count, (), ());

        swap_chain.frames.alloc().init(swap_chain::Frame {
            texture_id,
            view_id,
            fence: device.raw.create_fence(true).unwrap(),
            sem_available: device.raw.create_semaphore().unwrap(),
            sem_present: device.raw.create_semaphore().unwrap(),
            acquired_epoch: None,
            need_waiting: AtomicBool::new(false),
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
    let mut token = Token::root();
    let textures = device_create_swap_chain(device_id, surface_id, desc, &mut token);
    swap_chain_populate_textures(surface_id, textures, &mut token);
    surface_id
}

#[no_mangle]
pub extern "C" fn wgpu_device_poll(device_id: DeviceId, force_wait: bool) {
    let (device_guard, mut token) = HUB.devices.read(&mut Token::root());
    let callbacks = device_guard[device_id].maintain(force_wait, &mut token);
    Device::fire_map_callbacks(callbacks);
}

#[no_mangle]
pub extern "C" fn wgpu_device_destroy(device_id: DeviceId) {
    let (device, mut token) = HUB.devices.unregister(device_id, &mut Token::root());
    device.maintain(true, &mut token);
    device.com_allocator.destroy(&device.raw);
}

pub type BufferMapReadCallback =
    extern "C" fn(status: BufferMapAsyncStatus, data: *const u8, userdata: *mut u8);
pub type BufferMapWriteCallback =
    extern "C" fn(status: BufferMapAsyncStatus, data: *mut u8, userdata: *mut u8);

#[no_mangle]
pub extern "C" fn wgpu_buffer_map_read_async(
    buffer_id: BufferId,
    start: BufferAddress,
    size: BufferAddress,
    callback: BufferMapReadCallback,
    userdata: *mut u8,
) {
    let mut token = Token::root();
    let (device_guard, mut token) = HUB.devices.read(&mut token);

    let (device_id, ref_count) = {
        let (mut buffer_guard, _) = HUB.buffers.write(&mut token);
        let buffer = &mut buffer_guard[buffer_id];

        if buffer.pending_map_operation.is_some() {
            log::error!("wgpu_buffer_map_read_async failed: buffer mapping is pending");
            callback(BufferMapAsyncStatus::Error, std::ptr::null_mut(), userdata);
            return;
        }

        let range = start .. start + size;
        buffer.pending_map_operation = Some(BufferMapOperation::Read(range, callback, userdata));
        (buffer.device_id.value, buffer.life_guard.ref_count.clone())
    };

    let device = &device_guard[device_id];

    device
        .trackers
        .lock()
        .buffers
        .change_replace(buffer_id, &ref_count, (), resource::BufferUsage::MAP_READ);

    device.pending.lock().map(buffer_id, ref_count);
}

#[no_mangle]
pub extern "C" fn wgpu_buffer_map_write_async(
    buffer_id: BufferId,
    start: BufferAddress,
    size: BufferAddress,
    callback: BufferMapWriteCallback,
    userdata: *mut u8,
) {
    let mut token = Token::root();
    let (device_guard, mut token) = HUB.devices.read(&mut token);

    let (device_id, ref_count) = {
        let (mut buffer_guard, _) = HUB.buffers.write(&mut token);
        let buffer = &mut buffer_guard[buffer_id];

        if buffer.pending_map_operation.is_some() {
            log::error!("wgpu_buffer_map_write_async failed: buffer mapping is pending");
            callback(BufferMapAsyncStatus::Error, std::ptr::null_mut(), userdata);
            return;
        }

        let range = start .. start + size;
        buffer.pending_map_operation = Some(BufferMapOperation::Write(range, callback, userdata));
        (buffer.device_id.value, buffer.life_guard.ref_count.clone())
    };

   let device = &device_guard[device_id];

    device
        .trackers
        .lock()
        .buffers
        .change_replace(buffer_id, &ref_count, (), resource::BufferUsage::MAP_WRITE);

    device.pending.lock().map(buffer_id, ref_count);
}

#[no_mangle]
pub extern "C" fn wgpu_buffer_unmap(buffer_id: BufferId) {
    let mut token = Token::root();
    let (device_guard, mut token) = HUB.devices.read(&mut token);
    let (mut buffer_guard, _) = HUB.buffers.write(&mut token);

    let buffer = &mut buffer_guard[buffer_id];
    let device_raw = &device_guard[buffer.device_id.value].raw;

    if !buffer.mapped_write_ranges.is_empty() {
        unsafe {
            device_raw
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

    buffer.memory.unmap(device_raw);
}
