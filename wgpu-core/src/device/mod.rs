/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    binding_model, command, conv,
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Input, Token},
    id, pipeline, resource, swap_chain,
    track::{BufferState, TextureState, TrackerSet},
    validation, FastHashMap, LifeGuard, PrivateFeatures, Stored, MAX_BIND_GROUPS,
};

use arrayvec::ArrayVec;
use copyless::VecHelper as _;
use gfx_descriptor::DescriptorAllocator;
use gfx_memory::{Block, Heaps};
use hal::{
    command::CommandBuffer as _,
    device::Device as _,
    window::{PresentationSurface as _, Surface as _},
};
use parking_lot::{Mutex, MutexGuard};
use wgt::{
    BufferAddress, BufferSize, InputStepMode, TextureDimension, TextureFormat,
    BIND_BUFFER_ALIGNMENT,
};

use std::{
    collections::hash_map::Entry, ffi, iter, marker::PhantomData, mem, ptr, slice,
    sync::atomic::Ordering,
};

use spirv_headers::ExecutionModel;

mod life;
mod queue;
#[cfg(any(feature = "trace", feature = "replay"))]
pub mod trace;

use smallvec::SmallVec;
#[cfg(feature = "trace")]
use trace::{Action, Trace};

pub type Label = *const std::os::raw::c_char;
#[cfg(feature = "trace")]
fn own_label(label: &Label) -> String {
    if label.is_null() {
        String::new()
    } else {
        unsafe { ffi::CStr::from_ptr(*label) }
            .to_string_lossy()
            .to_string()
    }
}

pub const MAX_COLOR_TARGETS: usize = 4;
pub const MAX_MIP_LEVELS: usize = 16;
pub const MAX_VERTEX_BUFFERS: usize = 16;
pub const MAX_ANISOTROPY: u8 = 16;

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

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HostMap {
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

pub(crate) type RenderPassKey = AttachmentData<(hal::pass::Attachment, hal::image::Layout)>;
pub(crate) type FramebufferKey = AttachmentData<id::TextureViewId>;
pub(crate) type RenderPassContext = AttachmentData<TextureFormat>;

type BufferMapResult = Result<ptr::NonNull<u8>, hal::device::MapError>;
type BufferMapPendingCallback = (resource::BufferMapOperation, resource::BufferMapAsyncStatus);

fn map_buffer<B: hal::Backend>(
    raw: &B::Device,
    buffer: &mut resource::Buffer<B>,
    sub_range: hal::buffer::SubRange,
    kind: HostMap,
) -> BufferMapResult {
    let (ptr, segment, needs_sync) = {
        let segment = hal::memory::Segment {
            offset: sub_range.offset,
            size: sub_range.size,
        };
        let mapped = buffer.memory.map(raw, segment)?;
        let mr = mapped.range();
        let segment = hal::memory::Segment {
            offset: mr.start,
            size: Some(mr.end - mr.start),
        };
        (mapped.ptr(), segment, !mapped.is_coherent())
    };

    buffer.sync_mapped_writes = match kind {
        HostMap::Read if needs_sync => unsafe {
            raw.invalidate_mapped_memory_ranges(iter::once((buffer.memory.memory(), segment)))
                .unwrap();
            None
        },
        HostMap::Write if needs_sync => Some(segment),
        _ => None,
    };
    Ok(ptr)
}

fn unmap_buffer<B: hal::Backend>(raw: &B::Device, buffer: &mut resource::Buffer<B>) {
    if let Some(segment) = buffer.sync_mapped_writes.take() {
        unsafe {
            raw.flush_mapped_memory_ranges(iter::once((buffer.memory.memory(), segment)))
                .unwrap()
        };
    }
}

//Note: this logic is specifically moved out of `handle_mapping()` in order to
// have nothing locked by the time we execute users callback code.
fn fire_map_callbacks<I: IntoIterator<Item = BufferMapPendingCallback>>(callbacks: I) {
    for (operation, status) in callbacks {
        unsafe { (operation.callback)(status, operation.user_data) }
    }
}

#[derive(Debug)]
pub struct Device<B: hal::Backend> {
    pub(crate) raw: B::Device,
    pub(crate) adapter_id: Stored<id::AdapterId>,
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
    pub(crate) hal_limits: hal::Limits,
    pub(crate) private_features: PrivateFeatures,
    limits: wgt::Limits,
    extensions: wgt::Extensions,
    //TODO: move this behind another mutex. This would allow several methods to switch
    // to borrow Device immutably, such as `write_buffer`, `write_texture`, and `buffer_unmap`.
    pending_writes: queue::PendingWrites<B>,
    #[cfg(feature = "trace")]
    pub(crate) trace: Option<Mutex<Trace>>,
}

impl<B: GfxBackend> Device<B> {
    pub(crate) fn new(
        raw: B::Device,
        adapter_id: Stored<id::AdapterId>,
        queue_group: hal::queue::QueueGroup<B>,
        mem_props: hal::adapter::MemoryProperties,
        hal_limits: hal::Limits,
        private_features: PrivateFeatures,
        desc: &wgt::DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
    ) -> Self {
        // don't start submission index at zero
        let life_guard = LifeGuard::new();
        life_guard.submission_index.fetch_add(1, Ordering::Relaxed);

        let com_allocator = command::CommandAllocator::new(queue_group.family, &raw);
        let heaps = unsafe {
            Heaps::new(
                &mem_props,
                gfx_memory::GeneralConfig {
                    block_size_granularity: 0x100,
                    max_chunk_size: 0x100_0000,
                    min_device_allocation: 0x1_0000,
                },
                gfx_memory::LinearConfig {
                    linear_size: 0x100_0000,
                },
                hal_limits.non_coherent_atom_size as u64,
            )
        };
        #[cfg(not(feature = "trace"))]
        match trace_path {
            Some(_) => log::error!("Feature 'trace' is not enabled"),
            None => (),
        }

        Device {
            raw,
            adapter_id,
            com_allocator,
            mem_allocator: Mutex::new(heaps),
            desc_allocator: Mutex::new(DescriptorAllocator::new()),
            queue_group,
            life_guard,
            trackers: Mutex::new(TrackerSet::new(B::VARIANT)),
            render_passes: Mutex::new(FastHashMap::default()),
            framebuffers: Mutex::new(FastHashMap::default()),
            life_tracker: Mutex::new(life::LifetimeTracker::new()),
            temp_suspected: life::SuspectedResources::default(),
            #[cfg(feature = "trace")]
            trace: trace_path.and_then(|path| match Trace::new(path) {
                Ok(mut trace) => {
                    trace.add(Action::Init {
                        desc: desc.clone(),
                        backend: B::VARIANT,
                    });
                    Some(Mutex::new(trace))
                }
                Err(e) => {
                    log::error!("Unable to start a trace in '{:?}': {:?}", path, e);
                    None
                }
            }),
            hal_limits,
            private_features,
            limits: desc.limits.clone(),
            extensions: desc.extensions.clone(),
            pending_writes: queue::PendingWrites::new(),
        }
    }

    fn lock_life_internal<'this, 'token: 'this>(
        tracker: &'this Mutex<life::LifetimeTracker<B>>,
        _token: &mut Token<'token, Self>,
    ) -> MutexGuard<'this, life::LifetimeTracker<B>> {
        tracker.lock()
    }

    fn lock_life<'this, 'token: 'this>(
        &'this self,
        token: &mut Token<'token, Self>,
    ) -> MutexGuard<'this, life::LifetimeTracker<B>> {
        Self::lock_life_internal(&self.life_tracker, token)
    }

    fn maintain<'this, 'token: 'this, G: GlobalIdentityHandlerFactory>(
        &'this self,
        global: &Global<G>,
        force_wait: bool,
        token: &mut Token<'token, Self>,
    ) -> Vec<BufferMapPendingCallback> {
        let mut life_tracker = self.lock_life(token);

        life_tracker.triage_suspected(
            global,
            &self.trackers,
            #[cfg(feature = "trace")]
            self.trace.as_ref(),
            token,
        );
        life_tracker.triage_mapped(global, token);
        life_tracker.triage_framebuffers(global, &mut *self.framebuffers.lock(), token);
        let last_done = life_tracker.triage_submissions(&self.raw, force_wait);
        let callbacks = life_tracker.handle_mapping(global, &self.raw, &self.trackers, token);
        life_tracker.cleanup(&self.raw, &self.mem_allocator, &self.desc_allocator);

        self.com_allocator.maintain(&self.raw, last_done);
        callbacks
    }

    fn create_buffer(
        &self,
        self_id: id::DeviceId,
        desc: &wgt::BufferDescriptor<Label>,
        memory_kind: gfx_memory::Kind,
    ) -> resource::Buffer<B> {
        debug_assert_eq!(self_id.backend(), B::VARIANT);
        let (mut usage, _memory_properties) = conv::map_buffer_usage(desc.usage);
        if desc.mapped_at_creation && !desc.usage.contains(wgt::BufferUsage::MAP_WRITE) {
            // we are going to be copying into it, internally
            usage |= hal::buffer::Usage::TRANSFER_DST;
        }

        let mem_usage = {
            use gfx_memory::MemoryUsage;
            use wgt::BufferUsage as Bu;

            //TODO: use linear allocation when we can ensure the freeing is linear
            if !desc.usage.intersects(Bu::MAP_READ | Bu::MAP_WRITE) {
                MemoryUsage::Private
            } else if (Bu::MAP_WRITE | Bu::COPY_SRC).contains(desc.usage) {
                MemoryUsage::Staging { read_back: false }
            } else if (Bu::MAP_READ | Bu::COPY_DST).contains(desc.usage) {
                MemoryUsage::Staging { read_back: true }
            } else {
                let is_native_only = self
                    .extensions
                    .contains(wgt::Extensions::MAPPABLE_PRIMARY_BUFFERS);
                assert!(
                    is_native_only,
                    "MAP usage can only be combined with the opposite COPY, requested {:?}",
                    desc.usage
                );
                MemoryUsage::Dynamic {
                    sparse_updates: false,
                }
            }
        };

        let mut buffer = unsafe { self.raw.create_buffer(desc.size.max(1), usage).unwrap() };
        if !desc.label.is_null() {
            unsafe {
                let label = ffi::CStr::from_ptr(desc.label).to_string_lossy();
                self.raw.set_buffer_name(&mut buffer, &label)
            };
        }
        let requirements = unsafe { self.raw.get_buffer_requirements(&buffer) };
        let memory = self
            .mem_allocator
            .lock()
            .allocate(
                &self.raw,
                requirements.type_mask as u32,
                mem_usage,
                memory_kind,
                requirements.size,
                requirements.alignment,
            )
            .unwrap();

        unsafe {
            self.raw
                .bind_buffer_memory(memory.memory(), memory.segment().offset, &mut buffer)
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
            sync_mapped_writes: None,
            map_state: resource::BufferMapState::Idle,
            life_guard: LifeGuard::new(),
        }
    }

    fn create_texture(
        &self,
        self_id: id::DeviceId,
        desc: &wgt::TextureDescriptor<Label>,
    ) -> resource::Texture<B> {
        debug_assert_eq!(self_id.backend(), B::VARIANT);

        // Ensure `D24Plus` textures cannot be copied
        match desc.format {
            TextureFormat::Depth24Plus | TextureFormat::Depth24PlusStencil8 => {
                assert!(
                    !desc
                        .usage
                        .intersects(wgt::TextureUsage::COPY_SRC | wgt::TextureUsage::COPY_DST),
                    "D24Plus textures cannot be copied"
                );
            }
            _ => {}
        }

        let kind = conv::map_texture_dimension_size(desc.dimension, desc.size, desc.sample_count);
        let format = conv::map_texture_format(desc.format, self.private_features);
        let aspects = format.surface_desc().aspects;
        let usage = conv::map_texture_usage(desc.usage, aspects);

        assert!(
            (desc.mip_level_count as usize) < MAX_MIP_LEVELS,
            "Texture descriptor mip level count ({}) must be less than device max mip levels ({})",
            desc.mip_level_count,
            MAX_MIP_LEVELS
        );
        let mut view_capabilities = hal::image::ViewCapabilities::empty();

        // 2D textures with array layer counts that are multiples of 6 could be cubemaps
        // Following gpuweb/gpuweb#68 always add the hint in that case
        if desc.dimension == TextureDimension::D2 && desc.size.depth % 6 == 0 {
            view_capabilities |= hal::image::ViewCapabilities::KIND_CUBE;
        };

        // TODO: 2D arrays, cubemap arrays

        let mut image = unsafe {
            let mut image = self
                .raw
                .create_image(
                    kind,
                    desc.mip_level_count as hal::image::Level,
                    format,
                    hal::image::Tiling::Optimal,
                    usage,
                    view_capabilities,
                )
                .unwrap();
            if !desc.label.is_null() {
                let label = ffi::CStr::from_ptr(desc.label).to_string_lossy();
                self.raw.set_image_name(&mut image, &label);
            }
            image
        };
        let requirements = unsafe { self.raw.get_image_requirements(&image) };

        let memory = self
            .mem_allocator
            .lock()
            .allocate(
                &self.raw,
                requirements.type_mask as u32,
                gfx_memory::MemoryUsage::Private,
                gfx_memory::Kind::General,
                requirements.size,
                requirements.alignment,
            )
            .unwrap();

        unsafe {
            self.raw
                .bind_image_memory(memory.memory(), memory.segment().offset, &mut image)
                .unwrap()
        };

        resource::Texture {
            raw: image,
            device_id: Stored {
                value: self_id,
                ref_count: self.life_guard.add_ref(),
            },
            usage: desc.usage,
            dimension: desc.dimension,
            kind,
            format: desc.format,
            full_range: hal::image::SubresourceRange {
                aspects,
                levels: 0..desc.mip_level_count as hal::image::Level,
                layers: 0..kind.num_layers(),
            },
            memory,
            life_guard: LifeGuard::new(),
        }
    }

    /// Create a compatible render pass with a given key.
    ///
    /// This functions doesn't consider the following aspects for compatibility:
    ///  - image layouts
    ///  - resolve attachments
    fn create_compatible_render_pass(&self, key: &RenderPassKey) -> B::RenderPass {
        let mut color_ids = [(0, hal::image::Layout::ColorAttachmentOptimal); MAX_COLOR_TARGETS];
        for i in 0..key.colors.len() {
            color_ids[i].0 = i;
        }
        let depth_id = key.depth_stencil.as_ref().map(|_| {
            (
                key.colors.len(),
                hal::image::Layout::DepthStencilAttachmentOptimal,
            )
        });

        let subpass = hal::pass::SubpassDesc {
            colors: &color_ids[..key.colors.len()],
            depth_stencil: depth_id.as_ref(),
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };
        let all = key.all().map(|(at, _)| at);

        unsafe {
            self.raw
                .create_render_pass(all, iter::once(subpass), &[])
                .unwrap()
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

    /// Wait for idle and remove resources that we can, before we die.
    pub(crate) fn prepare_to_die(&mut self) {
        let mut life_tracker = self.life_tracker.lock();
        life_tracker.triage_submissions(&self.raw, true);
        life_tracker.cleanup(&self.raw, &self.mem_allocator, &self.desc_allocator);
    }

    pub(crate) fn dispose(self) {
        let mut desc_alloc = self.desc_allocator.into_inner();
        let mut mem_alloc = self.mem_allocator.into_inner();
        self.pending_writes
            .dispose(&self.raw, &self.com_allocator, &mut mem_alloc);
        self.com_allocator.destroy(&self.raw);
        unsafe {
            desc_alloc.clear(&self.raw);
            mem_alloc.clear(&self.raw);
            for (_, rp) in self.render_passes.lock().drain() {
                self.raw.destroy_render_pass(rp);
            }
            for (_, fbo) in self.framebuffers.lock().drain() {
                self.raw.destroy_framebuffer(fbo);
            }
        }
    }
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn device_extensions<B: GfxBackend>(&self, device_id: id::DeviceId) -> wgt::Extensions {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];

        device.extensions.clone()
    }

    pub fn device_limits<B: GfxBackend>(&self, device_id: id::DeviceId) -> wgt::Limits {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];

        device.limits.clone()
    }

    pub fn device_create_buffer<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &wgt::BufferDescriptor<Label>,
        id_in: Input<G, id::BufferId>,
    ) -> id::BufferId {
        let hub = B::hub(self);
        let mut token = Token::root();

        log::info!("Create buffer {:?} with ID {:?}", desc, id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        let mut buffer = device.create_buffer(device_id, desc, gfx_memory::Kind::General);
        let ref_count = buffer.life_guard.add_ref();

        let buffer_use = if !desc.mapped_at_creation {
            resource::BufferUse::EMPTY
        } else if desc.usage.contains(wgt::BufferUsage::MAP_WRITE) {
            // buffer is mappable, so we are just doing that at start
            match map_buffer(
                &device.raw,
                &mut buffer,
                hal::buffer::SubRange::WHOLE,
                HostMap::Write,
            ) {
                Ok(ptr) => {
                    buffer.map_state = resource::BufferMapState::Active {
                        ptr,
                        sub_range: hal::buffer::SubRange::WHOLE,
                        host: HostMap::Write,
                    };
                }
                Err(e) => {
                    log::error!("failed to create buffer in a mapped state: {:?}", e);
                }
            };
            resource::BufferUse::MAP_WRITE
        } else {
            // buffer needs staging area for initialization only
            let mut stage = device.create_buffer(
                device_id,
                &wgt::BufferDescriptor {
                    #[allow(trivial_casts)]
                    label: b"<init_buffer>\0".as_ptr() as *const _,
                    size: desc.size,
                    usage: wgt::BufferUsage::MAP_WRITE | wgt::BufferUsage::COPY_SRC,
                    mapped_at_creation: false,
                },
                gfx_memory::Kind::Linear,
            );
            let ptr = stage
                .memory
                .map(&device.raw, hal::memory::Segment::ALL)
                .unwrap()
                .ptr();
            buffer.map_state = resource::BufferMapState::Init {
                ptr,
                stage_buffer: stage.raw,
                stage_memory: stage.memory,
            };
            resource::BufferUse::COPY_DST
        };

        let id = hub.buffers.register_identity(id_in, buffer, &mut token);
        log::info!("Created buffer {:?} with {:?}", id, desc);
        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => {
                let mut desc = desc.map_label(own_label);
                let mapped_at_creation = mem::replace(&mut desc.mapped_at_creation, false);
                if mapped_at_creation && !desc.usage.contains(wgt::BufferUsage::MAP_WRITE) {
                    desc.usage |= wgt::BufferUsage::COPY_DST;
                }
                trace.lock().add(trace::Action::CreateBuffer { id, desc })
            }
            None => (),
        };

        device
            .trackers
            .lock()
            .buffers
            .init(id, ref_count, BufferState::with_usage(buffer_use))
            .unwrap();
        id
    }

    #[cfg(feature = "replay")]
    pub fn device_wait_for_buffer<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        buffer_id: id::BufferId,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let last_submission = {
            let (buffer_guard, _) = hub.buffers.write(&mut token);
            buffer_guard[buffer_id]
                .life_guard
                .submission_index
                .load(Ordering::Acquire)
        };

        let device = &device_guard[device_id];
        let mut life_lock = device.lock_life(&mut token);
        if life_lock.lowest_active_submission() <= last_submission {
            log::info!(
                "Waiting for submission {:?} before accessing buffer {:?}",
                last_submission,
                buffer_id
            );
            life_lock.triage_submissions(&device.raw, true);
        }
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
        assert!(
            buffer.usage.contains(wgt::BufferUsage::MAP_WRITE),
            "Buffer usage {:?} must contain usage flag MAP_WRITE",
            buffer.usage
        );
        //assert!(buffer isn't used by the GPU);

        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => {
                let mut trace = trace.lock();
                let data_path = trace.make_binary("bin", data);
                trace.add(trace::Action::WriteBuffer {
                    id: buffer_id,
                    data: data_path,
                    range: offset..offset + data.len() as BufferAddress,
                    queued: false,
                });
            }
            None => (),
        };

        match map_buffer(
            &device.raw,
            &mut buffer,
            hal::buffer::SubRange {
                offset,
                size: Some(data.len() as BufferAddress),
            },
            HostMap::Write,
        ) {
            Ok(ptr) => unsafe {
                ptr::copy_nonoverlapping(data.as_ptr(), ptr.as_ptr(), data.len());
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
        assert!(
            buffer.usage.contains(wgt::BufferUsage::MAP_READ),
            "Buffer usage {:?} must contain usage flag MAP_READ",
            buffer.usage
        );
        //assert!(buffer isn't used by the GPU);

        match map_buffer(
            &device.raw,
            &mut buffer,
            hal::buffer::SubRange {
                offset,
                size: Some(data.len() as BufferAddress),
            },
            HostMap::Read,
        ) {
            Ok(ptr) => unsafe {
                ptr::copy_nonoverlapping(ptr.as_ptr(), data.as_mut_ptr(), data.len());
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

        log::info!("Buffer {:?} is dropped", buffer_id);
        let device_id = {
            let (mut buffer_guard, _) = hub.buffers.write(&mut token);
            let buffer = &mut buffer_guard[buffer_id];
            buffer.life_guard.ref_count.take();
            buffer.device_id.value
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources
            .buffers
            .push(buffer_id);
    }

    pub fn device_create_texture<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &wgt::TextureDescriptor<Label>,
        id_in: Input<G, id::TextureId>,
    ) -> id::TextureId {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        let texture = device.create_texture(device_id, desc);
        let range = texture.full_range.clone();
        let ref_count = texture.life_guard.add_ref();

        let id = hub.textures.register_identity(id_in, texture, &mut token);
        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => trace.lock().add(trace::Action::CreateTexture {
                id,
                desc: desc.map_label(own_label),
            }),
            None => (),
        };

        device
            .trackers
            .lock()
            .textures
            .init(id, ref_count, TextureState::with_range(&range))
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
            .suspected_resources
            .textures
            .push(texture_id);
    }

    pub fn texture_create_view<B: GfxBackend>(
        &self,
        texture_id: id::TextureId,
        desc: Option<&wgt::TextureViewDescriptor<Label>>,
        id_in: Input<G, id::TextureViewId>,
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
                    levels: desc.base_mip_level as u8..end_level,
                    layers: desc.base_array_layer as u16..end_layer,
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
                    conv::map_texture_format(format, device.private_features),
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
        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => trace.lock().add(trace::Action::CreateTextureView {
                id,
                parent_id: texture_id,
                desc: desc.map(|d| d.map_label(own_label)),
            }),
            None => (),
        };

        device
            .trackers
            .lock()
            .views
            .init(id, ref_count, PhantomData)
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
            .suspected_resources
            .texture_views
            .push(texture_view_id);
    }

    pub fn device_create_sampler<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &wgt::SamplerDescriptor<Label>,
        id_in: Input<G, id::SamplerId>,
    ) -> id::SamplerId {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];

        if let Some(clamp) = desc.anisotropy_clamp {
            assert!(
                device.extensions.contains(wgt::Extensions::ANISOTROPIC_FILTERING),
                "Anisotropic clamp may only be used when the anisotropic filtering extension is enabled"
            );
            let valid_clamp = clamp <= MAX_ANISOTROPY && conv::is_power_of_two(clamp as u32);
            assert!(
                valid_clamp,
                "Anisotropic clamp must be one of the values: 1, 2, 4, 8, or 16"
            );
        }

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
            lod_range: hal::image::Lod(desc.lod_min_clamp)..hal::image::Lod(desc.lod_max_clamp),
            comparison: desc.compare.and_then(conv::map_compare_function),
            border: hal::image::PackedColor(0),
            normalized: true,
            anisotropy_clamp: desc.anisotropy_clamp,
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
        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => trace.lock().add(trace::Action::CreateSampler {
                id,
                desc: desc.map_label(own_label),
            }),
            None => (),
        };

        device
            .trackers
            .lock()
            .samplers
            .init(id, ref_count, PhantomData)
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
            .suspected_resources
            .samplers
            .push(sampler_id);
    }

    pub fn device_create_bind_group_layout<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &wgt::BindGroupLayoutDescriptor,
        id_in: Input<G, id::BindGroupLayoutId>,
    ) -> Result<id::BindGroupLayoutId, binding_model::BindGroupLayoutError> {
        let mut token = Token::root();
        let hub = B::hub(self);
        let mut entry_map = FastHashMap::default();
        for entry in desc.bindings {
            if entry_map.insert(entry.binding, entry.clone()).is_some() {
                return Err(binding_model::BindGroupLayoutError::ConflictBinding(
                    entry.binding,
                ));
            }
        }

        // TODO: deduplicate the bind group layouts at some level.
        // We can't do it right here, because in the remote scenario
        // the client need to know if the same ID can be used, or not.
        if false {
            let (bgl_guard, _) = hub.bind_group_layouts.read(&mut token);
            let bind_group_layout_id = bgl_guard
                .iter(device_id.backend())
                .find(|(_, bgl)| bgl.entries == entry_map);

            if let Some((id, _)) = bind_group_layout_id {
                return Ok(id);
            }
        }

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];

        // Validate the count parameter
        for binding in desc
            .bindings
            .iter()
            .filter(|binding| binding.count.is_some())
        {
            if let Some(count) = binding.count {
                if count == 0 {
                    return Err(binding_model::BindGroupLayoutError::ZeroCount);
                }
                match binding.ty {
                    wgt::BindingType::SampledTexture { .. } => {
                        if !device
                            .extensions
                            .contains(wgt::Extensions::TEXTURE_BINDING_ARRAY)
                        {
                            return Err(binding_model::BindGroupLayoutError::MissingExtension(
                                wgt::Extensions::TEXTURE_BINDING_ARRAY,
                            ));
                        }
                    }
                    _ => return Err(binding_model::BindGroupLayoutError::ArrayUnsupported),
                }
            } else {
                unreachable!() // programming bug
            }
        }

        let raw_bindings = desc
            .bindings
            .iter()
            .map(|binding| hal::pso::DescriptorSetLayoutBinding {
                binding: binding.binding,
                ty: conv::map_binding_type(binding),
                count: binding
                    .count
                    .map_or(1, |v| v as hal::pso::DescriptorArrayIndex), //TODO: consolidate
                stage_flags: conv::map_shader_stage_flags(binding.visibility),
                immutable_samplers: false, // TODO
            })
            .collect::<Vec<_>>(); //TODO: avoid heap allocation

        let raw = unsafe {
            let mut raw_layout = device
                .raw
                .create_descriptor_set_layout(&raw_bindings, &[])
                .unwrap();
            if let Some(label) = desc.label {
                device
                    .raw
                    .set_descriptor_set_layout_name(&mut raw_layout, label);
            }
            raw_layout
        };

        let layout = binding_model::BindGroupLayout {
            raw,
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.add_ref(),
            },
            life_guard: LifeGuard::new(),
            entries: entry_map,
            desc_counts: raw_bindings.iter().cloned().collect(),
            dynamic_count: desc
                .bindings
                .iter()
                .filter(|b| b.has_dynamic_offset())
                .count(),
        };

        let id = hub
            .bind_group_layouts
            .register_identity(id_in, layout, &mut token);
        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => trace.lock().add(trace::Action::CreateBindGroupLayout {
                id,
                label: desc.label.map_or_else(String::new, str::to_string),
                entries: desc.bindings.to_owned(),
            }),
            None => (),
        };
        Ok(id)
    }

    pub fn bind_group_layout_destroy<B: GfxBackend>(
        &self,
        bind_group_layout_id: id::BindGroupLayoutId,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_id, ref_count) = {
            let (mut bind_group_layout_guard, _) = hub.bind_group_layouts.write(&mut token);
            let layout = &mut bind_group_layout_guard[bind_group_layout_id];
            (
                layout.device_id.value,
                layout.life_guard.ref_count.take().unwrap(),
            )
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources
            .bind_group_layouts
            .push(Stored {
                value: bind_group_layout_id,
                ref_count,
            });
    }

    pub fn device_create_pipeline_layout<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &binding_model::PipelineLayoutDescriptor,
        id_in: Input<G, id::PipelineLayoutId>,
    ) -> Result<id::PipelineLayoutId, binding_model::PipelineLayoutError> {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        let bind_group_layout_ids = unsafe {
            slice::from_raw_parts(desc.bind_group_layouts, desc.bind_group_layouts_length)
        };

        if desc.bind_group_layouts_length > (device.limits.max_bind_groups as usize) {
            return Err(binding_model::PipelineLayoutError::TooManyGroups(
                desc.bind_group_layouts_length,
            ));
        }

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
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.add_ref(),
            },
            life_guard: LifeGuard::new(),
            bind_group_layout_ids: {
                let (bind_group_layout_guard, _) = hub.bind_group_layouts.read(&mut token);
                bind_group_layout_ids
                    .iter()
                    .map(|&id| Stored {
                        value: id,
                        ref_count: bind_group_layout_guard[id].life_guard.add_ref(),
                    })
                    .collect()
            },
        };

        let id = hub
            .pipeline_layouts
            .register_identity(id_in, layout, &mut token);
        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => trace.lock().add(trace::Action::CreatePipelineLayout {
                id,
                bind_group_layouts: bind_group_layout_ids.to_owned(),
            }),
            None => (),
        };
        Ok(id)
    }

    pub fn pipeline_layout_destroy<B: GfxBackend>(&self, pipeline_layout_id: id::PipelineLayoutId) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_id, ref_count) = {
            let (mut pipeline_layout_guard, _) = hub.pipeline_layouts.write(&mut token);
            let layout = &mut pipeline_layout_guard[pipeline_layout_id];
            (
                layout.device_id.value,
                layout.life_guard.ref_count.take().unwrap(),
            )
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources
            .pipeline_layouts
            .push(Stored {
                value: pipeline_layout_id,
                ref_count,
            });
    }

    pub fn device_create_bind_group<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &binding_model::BindGroupDescriptor,
        id_in: Input<G, id::BindGroupId>,
    ) -> id::BindGroupId {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        let (bind_group_layout_guard, mut token) = hub.bind_group_layouts.read(&mut token);
        let bind_group_layout = &bind_group_layout_guard[desc.layout];
        assert_eq!(desc.bindings.len(), bind_group_layout.entries.len(), "Bind group has {} entries and bind group layout has {} entries, they should be the same.", desc.bindings.len(), bind_group_layout.entries.len());

        let desc_set = unsafe {
            let mut desc_sets = ArrayVec::<[_; 1]>::new();
            device
                .desc_allocator
                .lock()
                .allocate(
                    &device.raw,
                    &bind_group_layout.raw,
                    &bind_group_layout.desc_counts,
                    1,
                    &mut desc_sets,
                )
                .unwrap();
            desc_sets.pop().unwrap()
        };

        if let Some(..) = desc.label {
            //TODO: https://github.com/gfx-rs/gfx-extras/pull/5
            //unsafe {
            //    let label = ffi::CStr::from_ptr(desc.label).to_string_lossy();
            //    device.raw.set_descriptor_set_name(desc_set.raw_mut(), &label);
            //}
        }

        // fill out the descriptors
        let mut used = TrackerSet::new(B::VARIANT);
        {
            let (buffer_guard, mut token) = hub.buffers.read(&mut token);
            let (texture_guard, mut token) = hub.textures.read(&mut token); //skip token
            let (texture_view_guard, mut token) = hub.texture_views.read(&mut token);
            let (sampler_guard, _) = hub.samplers.read(&mut token);

            //TODO: group writes into contiguous sections
            let mut writes = Vec::new();
            for b in desc.bindings {
                let decl = bind_group_layout
                    .entries
                    .get(&b.binding)
                    .expect("Failed to find binding declaration for binding");
                let descriptors: SmallVec<[_; 1]> = match b.resource {
                    binding_model::BindingResource::Buffer(ref bb) => {
                        let (alignment, pub_usage, internal_use) = match decl.ty {
                            wgt::BindingType::UniformBuffer { .. } => (
                                BIND_BUFFER_ALIGNMENT,
                                wgt::BufferUsage::UNIFORM,
                                resource::BufferUse::UNIFORM,
                            ),
                            wgt::BindingType::StorageBuffer { readonly, .. } => (
                                BIND_BUFFER_ALIGNMENT,
                                wgt::BufferUsage::STORAGE,
                                if readonly {
                                    resource::BufferUse::STORAGE_STORE
                                } else {
                                    resource::BufferUse::STORAGE_LOAD
                                },
                            ),
                            wgt::BindingType::Sampler { .. }
                            | wgt::BindingType::StorageTexture { .. }
                            | wgt::BindingType::SampledTexture { .. } | _ => {
                                panic!("Mismatched buffer binding type for {:?}. Expected a type of UniformBuffer, StorageBuffer or ReadonlyStorageBuffer", decl)
                            }
                        };

                        assert_eq!(
                            bb.offset % alignment,
                            0,
                            "Buffer offset {} must be a multiple of alignment {}",
                            bb.offset,
                            alignment
                        );

                        let buffer = used
                            .buffers
                            .use_extend(&*buffer_guard, bb.buffer, (), internal_use)
                            .unwrap();
                        assert!(
                            buffer.usage.contains(pub_usage),
                            "Buffer usage {:?} must contain usage flag(s) {:?}",
                            buffer.usage,
                            pub_usage
                        );

                        let sub_range = hal::buffer::SubRange {
                            offset: bb.offset,
                            size: if bb.size == BufferSize::WHOLE {
                                None
                            } else {
                                let end = bb.offset + bb.size.0;
                                assert!(
                                    end <= buffer.size,
                                    "Bound buffer range {:?} does not fit in buffer size {}",
                                    bb.offset..end,
                                    buffer.size
                                );
                                Some(bb.size.0)
                            },
                        };
                        SmallVec::from([hal::pso::Descriptor::Buffer(&buffer.raw, sub_range)])
                    }
                    binding_model::BindingResource::Sampler(id) => {
                        match decl.ty {
                            wgt::BindingType::Sampler{ .. } => {}
                            _ => panic!("Mismatched sampler binding type in {:?}. Expected a type of Sampler", decl.ty),
                        }

                        let sampler = used
                            .samplers
                            .use_extend(&*sampler_guard, id, (), ())
                            .unwrap();
                        SmallVec::from([hal::pso::Descriptor::Sampler(&sampler.raw)])
                    }
                    binding_model::BindingResource::TextureView(id) => {
                        let (pub_usage, internal_use, image_layout) = match decl.ty {
                            wgt::BindingType::SampledTexture { .. } => (
                                wgt::TextureUsage::SAMPLED,
                                resource::TextureUse::SAMPLED,
                                hal::image::Layout::ShaderReadOnlyOptimal,
                            ),
                            wgt::BindingType::StorageTexture { readonly, .. } => (
                                wgt::TextureUsage::STORAGE,
                                if readonly {
                                    resource::TextureUse::STORAGE_LOAD
                                } else {
                                    resource::TextureUse::STORAGE_STORE
                                },
                                hal::image::Layout::General,
                            ),
                            _ => panic!("Mismatched texture binding type in {:?}. Expected a type of SampledTexture, ReadonlyStorageTexture or WriteonlyStorageTexture", decl),
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
                                        internal_use,
                                    )
                                    .unwrap();
                                assert!(
                                    texture.usage.contains(pub_usage),
                                    "Texture usage {:?} must contain usage flag(s) {:?}",
                                    texture.usage,
                                    pub_usage
                                );

                                SmallVec::from([hal::pso::Descriptor::Image(raw, image_layout)])
                            }
                            resource::TextureViewInner::SwapChain { .. } => {
                                panic!("Unable to create a bind group with a swap chain image")
                            }
                        }
                    }
                    binding_model::BindingResource::TextureViewArray(ref bindings_array) => {
                        assert!(device.extensions.contains(wgt::Extensions::TEXTURE_BINDING_ARRAY), "Extension TEXTURE_BINDING_ARRAY must be enabled to use TextureViewArrays in a bind group");

                        if let Some(count) = decl.count {
                            assert_eq!(
                                count as usize,
                                bindings_array.len(),
                                "Binding count declared with {} items, but {} items were provided",
                                count,
                                bindings_array.len()
                            );
                        } else {
                            panic!(
                                "Binding declared as a single item, but bind group is using it as an array",
                            );
                        }

                        let (pub_usage, internal_use, image_layout) = match decl.ty {
                            wgt::BindingType::SampledTexture { .. } => (
                                wgt::TextureUsage::SAMPLED,
                                resource::TextureUse::SAMPLED,
                                hal::image::Layout::ShaderReadOnlyOptimal,
                            ),
                            _ => panic!("Mismatched texture binding type in {:?}. Expected a type of SampledTextureArray", decl),
                        };
                        bindings_array
                            .iter()
                            .map(|id| {
                                let view = used
                                    .views
                                    .use_extend(&*texture_view_guard, *id, (), ())
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
                                                internal_use,
                                            )
                                            .unwrap();
                                        assert!(
                                            texture.usage.contains(pub_usage),
                                            "Texture usage {:?} must contain usage flag(s) {:?}",
                                            texture.usage,
                                            pub_usage
                                        );

                                        hal::pso::Descriptor::Image(raw, image_layout)
                                    }
                                    resource::TextureViewInner::SwapChain { .. } => panic!(
                                        "Unable to create a bind group with a swap chain image"
                                    ),
                                }
                            })
                            .collect()
                    }
                };
                writes.alloc().init(hal::pso::DescriptorSetWrite {
                    set: desc_set.raw(),
                    binding: b.binding,
                    array_offset: 0, //TODO
                    descriptors,
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
        log::debug!(
            "Bind group {:?} {:#?}",
            id,
            hub.bind_groups.read(&mut token).0[id].used
        );
        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => trace.lock().add(trace::Action::CreateBindGroup {
                id,
                label: desc.label.map_or_else(String::new, str::to_string),
                layout_id: desc.layout,
                entries: desc
                    .bindings
                    .iter()
                    .map(|entry| {
                        let res = match entry.resource {
                            binding_model::BindingResource::Buffer(ref b) => {
                                trace::BindingResource::Buffer {
                                    id: b.buffer,
                                    offset: b.offset,
                                    size: b.size,
                                }
                            }
                            binding_model::BindingResource::TextureView(id) => {
                                trace::BindingResource::TextureView(id)
                            }
                            binding_model::BindingResource::Sampler(id) => {
                                trace::BindingResource::Sampler(id)
                            }
                            binding_model::BindingResource::TextureViewArray(ref id_array) => {
                                trace::BindingResource::TextureViewArray(id_array.to_vec())
                            }
                        };
                        (entry.binding, res)
                    })
                    .collect(),
            }),
            None => (),
        };

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
            .suspected_resources
            .bind_groups
            .push(bind_group_id);
    }

    pub fn device_create_shader_module<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &pipeline::ShaderModuleDescriptor,
        id_in: Input<G, id::ShaderModuleId>,
    ) -> id::ShaderModuleId {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];

        let spv = unsafe { slice::from_raw_parts(desc.code.bytes, desc.code.length) };
        let raw = unsafe { device.raw.create_shader_module(spv).unwrap() };

        let module = if device.private_features.shader_validation {
            // Parse the given shader code and store its representation.
            let spv_iter = spv.into_iter().cloned();
            let mut parser = naga::front::spirv::Parser::new(spv_iter);
            parser
                .parse()
                .map_err(|err| {
                    log::warn!("Failed to parse shader SPIR-V code: {:?}", err);
                    log::warn!("Shader module will not be validated");
                })
                .ok()
        } else {
            None
        };
        let shader = pipeline::ShaderModule {
            raw,
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.add_ref(),
            },
            module,
        };

        let id = hub
            .shader_modules
            .register_identity(id_in, shader, &mut token);
        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => {
                let mut trace = trace.lock();
                let data = trace.make_binary("spv", unsafe {
                    slice::from_raw_parts(desc.code.bytes as *const u8, desc.code.length * 4)
                });
                trace.add(trace::Action::CreateShaderModule { id, data });
            }
            None => {}
        };
        id
    }

    pub fn shader_module_destroy<B: GfxBackend>(&self, shader_module_id: id::ShaderModuleId) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (module, _) = hub.shader_modules.unregister(shader_module_id, &mut token);

        let device = &device_guard[module.device_id.value];
        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => trace
                .lock()
                .add(trace::Action::DestroyShaderModule(shader_module_id)),
            None => (),
        };
        unsafe {
            device.raw.destroy_shader_module(module.raw);
        }
    }

    pub fn device_create_command_encoder<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &wgt::CommandEncoderDescriptor,
        id_in: Input<G, id::CommandEncoderId>,
    ) -> id::CommandEncoderId {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];

        let dev_stored = Stored {
            value: device_id,
            ref_count: device.life_guard.add_ref(),
        };

        let mut command_buffer = device.com_allocator.allocate(
            dev_stored,
            &device.raw,
            device.limits.clone(),
            device.private_features,
            #[cfg(feature = "trace")]
            device.trace.is_some(),
        );

        unsafe {
            let raw_command_buffer = command_buffer.raw.last_mut().unwrap();
            if !desc.label.is_null() {
                let label = ffi::CStr::from_ptr(desc.label).to_string_lossy();
                device
                    .raw
                    .set_command_buffer_name(raw_command_buffer, &label);
            }
            raw_command_buffer.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
        }

        hub.command_buffers
            .register_identity(id_in, command_buffer, &mut token)
    }

    pub fn command_encoder_destroy<B: GfxBackend>(&self, command_encoder_id: id::CommandEncoderId) {
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
            let (compute_pipe_guard, mut token) = hub.compute_pipelines.read(&mut token);
            let (render_pipe_guard, mut token) = hub.render_pipelines.read(&mut token);
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
            for id in comb.trackers.compute_pipes.used() {
                if compute_pipe_guard[id].life_guard.ref_count.is_none() {
                    device.temp_suspected.compute_pipelines.push(id);
                }
            }
            for id in comb.trackers.render_pipes.used() {
                if render_pipe_guard[id].life_guard.ref_count.is_none() {
                    device.temp_suspected.render_pipelines.push(id);
                }
            }
        }

        device
            .lock_life(&mut token)
            .suspected_resources
            .extend(&device.temp_suspected);
        device.com_allocator.discard(comb);
    }

    pub fn command_buffer_destroy<B: GfxBackend>(&self, command_buffer_id: id::CommandBufferId) {
        self.command_encoder_destroy::<B>(command_buffer_id)
    }

    pub fn device_create_render_pipeline<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &pipeline::RenderPipelineDescriptor,
        id_in: Input<G, id::RenderPipelineId>,
    ) -> Result<id::RenderPipelineId, pipeline::RenderPipelineError> {
        let hub = B::hub(self);
        let mut token = Token::root();

        let samples = {
            let sc = desc.sample_count;
            if sc == 0 || sc > 32 || !conv::is_power_of_two(sc) {
                return Err(pipeline::RenderPipelineError::InvalidSampleCount(sc));
            }
            sc as u8
        };

        let color_states =
            unsafe { slice::from_raw_parts(desc.color_states, desc.color_states_length) };
        let depth_stencil_state = unsafe { desc.depth_stencil_state.as_ref() };

        let rasterization_state = unsafe { desc.rasterization_state.as_ref() }.cloned();
        let rasterizer = conv::map_rasterization_state_descriptor(
            &rasterization_state.clone().unwrap_or_default(),
        );

        let mut interface = validation::StageInterface::default();
        let mut validated_stages = wgt::ShaderStage::empty();

        let desc_vbs = unsafe {
            slice::from_raw_parts(
                desc.vertex_state.vertex_buffers,
                desc.vertex_state.vertex_buffers_length,
            )
        };
        let mut vertex_strides = Vec::with_capacity(desc_vbs.len());
        let mut vertex_buffers = Vec::with_capacity(desc_vbs.len());
        let mut attributes = Vec::new();
        for (i, vb_state) in desc_vbs.iter().enumerate() {
            vertex_strides
                .alloc()
                .init((vb_state.array_stride, vb_state.step_mode));
            if vb_state.attributes_length == 0 {
                continue;
            }
            vertex_buffers.alloc().init(hal::pso::VertexBufferDesc {
                binding: i as u32,
                stride: vb_state.array_stride as u32,
                rate: match vb_state.step_mode {
                    InputStepMode::Vertex => hal::pso::VertexInputRate::Vertex,
                    InputStepMode::Instance => hal::pso::VertexInputRate::Instance(1),
                },
            });
            let desc_atts =
                unsafe { slice::from_raw_parts(vb_state.attributes, vb_state.attributes_length) };
            for attribute in desc_atts {
                if attribute.offset >= 0x10000000 {
                    return Err(
                        pipeline::RenderPipelineError::InvalidVertexAttributeOffset {
                            location: attribute.shader_location,
                            offset: attribute.offset,
                        },
                    );
                }
                attributes.alloc().init(hal::pso::AttributeDesc {
                    location: attribute.shader_location,
                    binding: i as u32,
                    element: hal::pso::Element {
                        format: conv::map_vertex_format(attribute.format),
                        offset: attribute.offset as u32,
                    },
                });
                interface.insert(
                    attribute.shader_location,
                    validation::MaybeOwned::Owned(validation::map_vertex_format(attribute.format)),
                );
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

        let multisampling: Option<hal::pso::Multisampling> = if samples == 1 {
            None
        } else {
            Some(hal::pso::Multisampling {
                rasterization_samples: samples,
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

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        let (raw_pipeline, layout_ref_count) = {
            let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);
            let (bgl_guard, mut token) = hub.bind_group_layouts.read(&mut token);
            let layout = &pipeline_layout_guard[desc.layout];
            let group_layouts = layout
                .bind_group_layout_ids
                .iter()
                .map(|id| &bgl_guard[id.value].entries)
                .collect::<ArrayVec<[&binding_model::BindEntryMap; MAX_BIND_GROUPS]>>();

            let (shader_module_guard, _) = hub.shader_modules.read(&mut token);

            let rp_key = RenderPassKey {
                colors: color_states
                    .iter()
                    .map(|state| {
                        let at = hal::pass::Attachment {
                            format: Some(conv::map_texture_format(
                                state.format,
                                device.private_features,
                            )),
                            samples,
                            ops: hal::pass::AttachmentOps::PRESERVE,
                            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                            layouts: hal::image::Layout::General..hal::image::Layout::General,
                        };
                        (at, hal::image::Layout::ColorAttachmentOptimal)
                    })
                    .collect(),
                // We can ignore the resolves as the vulkan specs says:
                // As an additional special case, if two render passes have a single subpass,
                // they are compatible even if they have different resolve attachment references
                // or depth/stencil resolve modes but satisfy the other compatibility conditions.
                resolves: ArrayVec::new(),
                depth_stencil: depth_stencil_state.map(|state| {
                    let at = hal::pass::Attachment {
                        format: Some(conv::map_texture_format(
                            state.format,
                            device.private_features,
                        )),
                        samples,
                        ops: hal::pass::AttachmentOps::PRESERVE,
                        stencil_ops: hal::pass::AttachmentOps::PRESERVE,
                        layouts: hal::image::Layout::General..hal::image::Layout::General,
                    };
                    (at, hal::image::Layout::DepthStencilAttachmentOptimal)
                }),
            };

            let vertex = {
                let entry_point_name =
                    unsafe { ffi::CStr::from_ptr(desc.vertex_stage.entry_point) }
                        .to_str()
                        .to_owned()
                        .unwrap();

                let shader_module = &shader_module_guard[desc.vertex_stage.module];

                if let Some(ref module) = shader_module.module {
                    let flag = wgt::ShaderStage::VERTEX;
                    interface = validation::check_stage(
                        module,
                        &group_layouts,
                        entry_point_name,
                        ExecutionModel::Vertex,
                        interface,
                    )
                    .map_err(|error| pipeline::RenderPipelineError::Stage { flag, error })?;
                    validated_stages |= flag;
                }

                hal::pso::EntryPoint::<B> {
                    entry: entry_point_name, // TODO
                    module: &shader_module.raw,
                    specialization: hal::pso::Specialization::EMPTY,
                }
            };

            let fragment = match unsafe { desc.fragment_stage.as_ref() } {
                Some(stage) => {
                    let entry_point_name = unsafe { ffi::CStr::from_ptr(stage.entry_point) }
                        .to_str()
                        .to_owned()
                        .unwrap();

                    let shader_module = &shader_module_guard[stage.module];

                    if validated_stages == wgt::ShaderStage::VERTEX {
                        if let Some(ref module) = shader_module.module {
                            let flag = wgt::ShaderStage::FRAGMENT;
                            interface = validation::check_stage(
                                module,
                                &group_layouts,
                                entry_point_name,
                                ExecutionModel::Fragment,
                                interface,
                            )
                            .map_err(|error| {
                                pipeline::RenderPipelineError::Stage { flag, error }
                            })?;
                            validated_stages |= flag;
                        }
                    }

                    Some(hal::pso::EntryPoint::<B> {
                        entry: entry_point_name,
                        module: &shader_module.raw,
                        specialization: hal::pso::Specialization::EMPTY,
                    })
                }
                None => None,
            };

            if validated_stages.contains(wgt::ShaderStage::FRAGMENT) {
                for (i, state) in color_states.iter().enumerate() {
                    let output = &interface[&(i as wgt::ShaderLocation)];
                    if !validation::check_texture_format(state.format, output) {
                        log::warn!(
                            "Incompatible fragment output[{}]. Shader: {:?}. Expected: {:?}",
                            i,
                            state.format,
                            &**output
                        );
                        return Err(pipeline::RenderPipelineError::IncompatibleOutputFormat {
                            index: i as u8,
                        });
                    }
                }
            }

            let shaders = hal::pso::GraphicsShaderSet {
                vertex,
                hull: None,
                domain: None,
                geometry: None,
                fragment,
            };

            // TODO
            let flags = hal::pso::PipelineCreationFlags::empty();

            let mut render_pass_cache = device.render_passes.lock();
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
                layout: &layout.raw,
                subpass: hal::pass::Subpass {
                    index: 0,
                    main_pass: match render_pass_cache.entry(rp_key) {
                        Entry::Occupied(e) => e.into_mut(),
                        Entry::Vacant(e) => {
                            let pass = device.create_compatible_render_pass(e.key());
                            e.insert(pass)
                        }
                    },
                },
                flags,
                parent: hal::pso::BasePipeline::None,
            };
            // TODO: cache
            let pipeline = unsafe {
                device
                    .raw
                    .create_graphics_pipeline(&pipeline_desc, None)
                    .unwrap()
            };

            (pipeline, layout.life_guard.add_ref())
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
            if ds.is_read_only() {
                flags |= pipeline::PipelineFlags::DEPTH_STENCIL_READ_ONLY;
            }
        }

        let pipeline = pipeline::RenderPipeline {
            raw: raw_pipeline,
            layout_id: Stored {
                value: desc.layout,
                ref_count: layout_ref_count,
            },
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.add_ref(),
            },
            pass_context,
            flags,
            index_format: desc.vertex_state.index_format,
            vertex_strides,
            sample_count: samples,
            life_guard: LifeGuard::new(),
        };

        let id = hub
            .render_pipelines
            .register_identity(id_in, pipeline, &mut token);

        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => trace.lock().add(trace::Action::CreateRenderPipeline {
                id,
                desc: trace::RenderPipelineDescriptor {
                    layout: desc.layout,
                    vertex_stage: trace::ProgrammableStageDescriptor::new(&desc.vertex_stage),
                    fragment_stage: unsafe { desc.fragment_stage.as_ref() }
                        .map(trace::ProgrammableStageDescriptor::new),
                    primitive_topology: desc.primitive_topology,
                    rasterization_state,
                    color_states: color_states.to_vec(),
                    depth_stencil_state: depth_stencil_state.cloned(),
                    vertex_state: trace::VertexStateDescriptor {
                        index_format: desc.vertex_state.index_format,
                        vertex_buffers: desc_vbs
                            .iter()
                            .map(|vbl| trace::VertexBufferLayoutDescriptor {
                                array_stride: vbl.array_stride,
                                step_mode: vbl.step_mode,
                                attributes: unsafe {
                                    slice::from_raw_parts(vbl.attributes, vbl.attributes_length)
                                }
                                .iter()
                                .cloned()
                                .collect(),
                            })
                            .collect(),
                    },
                    sample_count: desc.sample_count,
                    sample_mask: desc.sample_mask,
                    alpha_to_coverage_enabled: desc.alpha_to_coverage_enabled,
                },
            }),
            None => (),
        };
        Ok(id)
    }

    pub fn render_pipeline_destroy<B: GfxBackend>(&self, render_pipeline_id: id::RenderPipelineId) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);

        let (device_id, layout_id) = {
            let (mut pipeline_guard, _) = hub.render_pipelines.write(&mut token);
            let pipeline = &mut pipeline_guard[render_pipeline_id];
            pipeline.life_guard.ref_count.take();
            (pipeline.device_id.value, pipeline.layout_id.clone())
        };

        let mut life_lock = device_guard[device_id].lock_life(&mut token);
        life_lock
            .suspected_resources
            .render_pipelines
            .push(render_pipeline_id);
        life_lock
            .suspected_resources
            .pipeline_layouts
            .push(layout_id);
    }

    pub fn device_create_compute_pipeline<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &pipeline::ComputePipelineDescriptor,
        id_in: Input<G, id::ComputePipelineId>,
    ) -> Result<id::ComputePipelineId, pipeline::ComputePipelineError> {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        let (raw_pipeline, layout_ref_count) = {
            let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);
            let (bgl_guard, mut token) = hub.bind_group_layouts.read(&mut token);
            let layout = &pipeline_layout_guard[desc.layout];
            let group_layouts = layout
                .bind_group_layout_ids
                .iter()
                .map(|id| &bgl_guard[id.value].entries)
                .collect::<ArrayVec<[&binding_model::BindEntryMap; MAX_BIND_GROUPS]>>();

            let interface = validation::StageInterface::default();
            let pipeline_stage = &desc.compute_stage;
            let (shader_module_guard, _) = hub.shader_modules.read(&mut token);

            let entry_point_name = unsafe { ffi::CStr::from_ptr(pipeline_stage.entry_point) }
                .to_str()
                .to_owned()
                .unwrap();

            let shader_module = &shader_module_guard[pipeline_stage.module];

            if let Some(ref module) = shader_module.module {
                let _ = validation::check_stage(
                    module,
                    &group_layouts,
                    entry_point_name,
                    ExecutionModel::GLCompute,
                    interface,
                )
                .map_err(pipeline::ComputePipelineError::Stage)?;
            }

            let shader = hal::pso::EntryPoint::<B> {
                entry: entry_point_name, // TODO
                module: &shader_module.raw,
                specialization: hal::pso::Specialization::EMPTY,
            };

            // TODO
            let flags = hal::pso::PipelineCreationFlags::empty();
            // TODO
            let parent = hal::pso::BasePipeline::None;

            let pipeline_desc = hal::pso::ComputePipelineDesc {
                shader,
                layout: &layout.raw,
                flags,
                parent,
            };

            let pipeline = unsafe {
                device
                    .raw
                    .create_compute_pipeline(&pipeline_desc, None)
                    .unwrap()
            };
            (pipeline, layout.life_guard.add_ref())
        };

        let pipeline = pipeline::ComputePipeline {
            raw: raw_pipeline,
            layout_id: Stored {
                value: desc.layout,
                ref_count: layout_ref_count,
            },
            device_id: Stored {
                value: device_id,
                ref_count: device.life_guard.add_ref(),
            },
            life_guard: LifeGuard::new(),
        };
        let id = hub
            .compute_pipelines
            .register_identity(id_in, pipeline, &mut token);

        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => trace.lock().add(trace::Action::CreateComputePipeline {
                id,
                desc: trace::ComputePipelineDescriptor {
                    layout: desc.layout,
                    compute_stage: trace::ProgrammableStageDescriptor::new(&desc.compute_stage),
                },
            }),
            None => (),
        };
        Ok(id)
    }

    pub fn compute_pipeline_destroy<B: GfxBackend>(
        &self,
        compute_pipeline_id: id::ComputePipelineId,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);

        let (device_id, layout_id) = {
            let (mut pipeline_guard, _) = hub.compute_pipelines.write(&mut token);
            let pipeline = &mut pipeline_guard[compute_pipeline_id];
            pipeline.life_guard.ref_count.take();
            (pipeline.device_id.value, pipeline.layout_id.clone())
        };

        let mut life_lock = device_guard[device_id].lock_life(&mut token);
        life_lock
            .suspected_resources
            .compute_pipelines
            .push(compute_pipeline_id);
        life_lock
            .suspected_resources
            .pipeline_layouts
            .push(layout_id);
    }

    pub fn device_create_swap_chain<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        surface_id: id::SurfaceId,
        desc: &wgt::SwapChainDescriptor,
    ) -> id::SwapChainId {
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
            let adapter = &adapter_guard[device.adapter_id.value];
            assert!(
                suf.supports_queue_family(&adapter.raw.queue_families[0]),
                "Surface {:?} doesn't support queue family {:?}",
                suf,
                &adapter.raw.queue_families[0]
            );
            let formats = suf.supported_formats(&adapter.raw.physical_device);
            let caps = suf.capabilities(&adapter.raw.physical_device);
            (caps, formats)
        };
        let num_frames = swap_chain::DESIRED_NUM_FRAMES
            .max(*caps.image_count.start())
            .min(*caps.image_count.end());
        let mut config =
            swap_chain::swap_chain_descriptor_to_hal(&desc, num_frames, device.private_features);
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
            assert!(
                sc.acquired_view_id.is_none(),
                "SwapChainOutput must be dropped before a new SwapChain is made."
            );
            unsafe {
                device.raw.destroy_semaphore(sc.semaphore);
            }
        }
        #[cfg(feature = "trace")]
        match device.trace {
            Some(ref trace) => trace.lock().add(Action::CreateSwapChain {
                id: sc_id,
                desc: desc.clone(),
            }),
            None => (),
        };

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

    #[cfg(feature = "replay")]
    /// Only triange suspected resource IDs. This helps us to avoid ID collisions
    /// upon creating new resources when re-playing a trace.
    pub fn device_maintain_ids<B: GfxBackend>(&self, device_id: id::DeviceId) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        device.lock_life(&mut token).triage_suspected(
            self,
            &device.trackers,
            #[cfg(feature = "trace")]
            None,
            &mut token,
        );
    }

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

    pub fn device_destroy<B: GfxBackend>(&self, device_id: id::DeviceId) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let device = {
            let (mut device, _) = hub.devices.unregister(device_id, &mut token);
            device.prepare_to_die();
            device
        };

        // Adapter is only referenced by the device and itself.
        // This isn't a robust way to destroy them, we should find a better one.
        if device.adapter_id.ref_count.load() == 1 {
            let (_adapter, _) = hub.adapters.unregister(device.adapter_id.value, &mut token);
        }

        device.dispose();
    }

    pub fn buffer_map_async<B: GfxBackend>(
        &self,
        buffer_id: id::BufferId,
        range: std::ops::Range<BufferAddress>,
        op: resource::BufferMapOperation,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (pub_usage, internal_use) = match op.host {
            HostMap::Read => (wgt::BufferUsage::MAP_READ, resource::BufferUse::MAP_READ),
            HostMap::Write => (wgt::BufferUsage::MAP_WRITE, resource::BufferUse::MAP_WRITE),
        };

        let (device_id, ref_count) = {
            let (mut buffer_guard, _) = hub.buffers.write(&mut token);
            let buffer = &mut buffer_guard[buffer_id];

            assert!(
                buffer.usage.contains(pub_usage),
                "Buffer usage {:?} must contain usage flag(s) {:?}",
                buffer.usage,
                pub_usage
            );
            buffer.map_state = match buffer.map_state {
                resource::BufferMapState::Init { .. } | resource::BufferMapState::Active { .. } => {
                    panic!("Buffer already mapped")
                }
                resource::BufferMapState::Waiting(_) => {
                    op.call_error();
                    return;
                }
                resource::BufferMapState::Idle => {
                    resource::BufferMapState::Waiting(resource::BufferPendingMapping {
                        sub_range: hal::buffer::SubRange {
                            offset: range.start,
                            size: Some(range.end - range.start),
                        },
                        op,
                        parent_ref_count: buffer.life_guard.add_ref(),
                    })
                }
            };
            log::debug!("Buffer {:?} map state -> Waiting", buffer_id);

            (buffer.device_id.value, buffer.life_guard.add_ref())
        };

        let device = &device_guard[device_id];
        device
            .trackers
            .lock()
            .buffers
            .change_replace(buffer_id, &ref_count, (), internal_use);

        device.lock_life(&mut token).map(buffer_id, ref_count);
    }

    pub fn buffer_get_mapped_range<B: GfxBackend>(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        _size: BufferSize,
    ) -> *mut u8 {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (buffer_guard, _) = hub.buffers.read(&mut token);
        let buffer = &buffer_guard[buffer_id];

        match buffer.map_state {
            resource::BufferMapState::Init { ptr, .. }
            | resource::BufferMapState::Active { ptr, .. } => unsafe {
                ptr.as_ptr().offset(offset as isize)
            },
            resource::BufferMapState::Idle | resource::BufferMapState::Waiting(_) => {
                log::error!("Buffer is not mapped");
                ptr::null_mut()
            }
        }
    }

    pub fn buffer_unmap<B: GfxBackend>(&self, buffer_id: id::BufferId) {
        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let (mut buffer_guard, _) = hub.buffers.write(&mut token);
        let buffer = &mut buffer_guard[buffer_id];
        let device = &mut device_guard[buffer.device_id.value];

        log::debug!("Buffer {:?} map state -> Idle", buffer_id);
        match mem::replace(&mut buffer.map_state, resource::BufferMapState::Idle) {
            resource::BufferMapState::Init {
                ptr,
                stage_buffer,
                stage_memory,
            } => {
                #[cfg(feature = "trace")]
                match device.trace {
                    Some(ref trace) => {
                        let mut trace = trace.lock();
                        let data = trace.make_binary("bin", unsafe {
                            slice::from_raw_parts(ptr.as_ptr(), buffer.size as usize)
                        });
                        trace.add(trace::Action::WriteBuffer {
                            id: buffer_id,
                            data,
                            range: 0..buffer.size,
                            queued: true,
                        });
                    }
                    None => (),
                };
                let _ = ptr;

                let last_submit_index = device.life_guard.submission_index.load(Ordering::Relaxed);
                buffer.life_guard.use_at(last_submit_index + 1);
                let region = hal::command::BufferCopy {
                    src: 0,
                    dst: 0,
                    size: buffer.size,
                };
                let transition_src = hal::memory::Barrier::Buffer {
                    states: hal::buffer::Access::HOST_WRITE..hal::buffer::Access::TRANSFER_READ,
                    target: &stage_buffer,
                    range: hal::buffer::SubRange::WHOLE,
                    families: None,
                };
                let transition_dst = hal::memory::Barrier::Buffer {
                    states: hal::buffer::Access::empty()..hal::buffer::Access::TRANSFER_WRITE,
                    target: &buffer.raw,
                    range: hal::buffer::SubRange::WHOLE,
                    families: None,
                };
                unsafe {
                    let comb = device.borrow_pending_writes();
                    comb.pipeline_barrier(
                        hal::pso::PipelineStage::HOST..hal::pso::PipelineStage::TRANSFER,
                        hal::memory::Dependencies::empty(),
                        iter::once(transition_src).chain(iter::once(transition_dst)),
                    );
                    comb.copy_buffer(&stage_buffer, &buffer.raw, iter::once(region));
                }
                device
                    .pending_writes
                    .consume_temp(stage_buffer, stage_memory);
            }
            resource::BufferMapState::Idle => {
                log::error!("Buffer is not mapped");
            }
            resource::BufferMapState::Waiting(_) => {}
            resource::BufferMapState::Active {
                ptr,
                sub_range,
                host,
            } => {
                if host == HostMap::Write {
                    #[cfg(feature = "trace")]
                    match device.trace {
                        Some(ref trace) => {
                            let mut trace = trace.lock();
                            let size = sub_range.size_to(buffer.size);
                            let data = trace.make_binary("bin", unsafe {
                                slice::from_raw_parts(ptr.as_ptr(), size as usize)
                            });
                            trace.add(trace::Action::WriteBuffer {
                                id: buffer_id,
                                data,
                                range: sub_range.offset..sub_range.offset + size,
                                queued: false,
                            });
                        }
                        None => (),
                    };
                    let _ = (ptr, sub_range);
                }
                unmap_buffer(&device.raw, buffer);
            }
        }
    }
}
