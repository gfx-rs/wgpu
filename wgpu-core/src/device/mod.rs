/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{
    binding_model, command, conv,
    device::life::WaitIdleError,
    hub::{
        GfxBackend, Global, GlobalIdentityHandlerFactory, Hub, Input, InvalidId, Storage, Token,
    },
    id, instance,
    memory_init_tracker::{MemoryInitKind, MemoryInitTracker, MemoryInitTrackerAction},
    pipeline, resource, swap_chain,
    track::{BufferState, TextureSelector, TextureState, TrackerSet},
    validation::{self, check_buffer_usage, check_texture_usage},
    FastHashMap, Label, LabelHelpers, LifeGuard, MultiRefCount, PrivateFeatures, Stored,
    SubmissionIndex, MAX_BIND_GROUPS,
};

use arrayvec::ArrayVec;
use copyless::VecHelper as _;
use hal::{
    command::CommandBuffer as _,
    device::Device as _,
    window::{PresentationSurface as _, Surface as _},
};
use parking_lot::{Mutex, MutexGuard};
use thiserror::Error;
use wgt::{
    BufferAddress, BufferSize, InputStepMode, TextureDimension, TextureFormat, TextureViewDimension,
};

use std::{
    borrow::Cow,
    collections::{hash_map::Entry, BTreeMap},
    iter,
    marker::PhantomData,
    mem,
    ops::Range,
    ptr,
    sync::atomic::Ordering,
};

pub mod alloc;
pub mod descriptor;
mod life;
pub mod queue;
#[cfg(any(feature = "trace", feature = "replay"))]
pub mod trace;

use smallvec::SmallVec;

pub const MAX_COLOR_TARGETS: usize = 4;
pub const MAX_MIP_LEVELS: u32 = 16;
pub const MAX_VERTEX_BUFFERS: usize = 16;
pub const MAX_ANISOTROPY: u8 = 16;
pub const SHADER_STAGE_COUNT: usize = 3;

pub type DeviceDescriptor<'a> = wgt::DeviceDescriptor<Label<'a>>;

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
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub enum HostMap {
    Read,
    Write,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serial-pass", derive(serde::Deserialize, serde::Serialize))]
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

    pub(crate) fn map<U, F: Fn(&T) -> U>(&self, fun: F) -> AttachmentData<U> {
        AttachmentData {
            colors: self.colors.iter().map(&fun).collect(),
            resolves: self.resolves.iter().map(&fun).collect(),
            depth_stencil: self.depth_stencil.as_ref().map(&fun),
        }
    }
}

pub(crate) type AttachmentDataVec<T> = ArrayVec<[T; MAX_COLOR_TARGETS + MAX_COLOR_TARGETS + 1]>;
pub(crate) type RenderPassKey = AttachmentData<(hal::pass::Attachment, hal::image::Layout)>;
#[derive(Debug, Eq, Hash, PartialEq)]
pub(crate) struct FramebufferKey {
    pub(crate) attachments: AttachmentData<hal::image::FramebufferAttachment>,
    pub(crate) extent: wgt::Extent3d,
    pub(crate) samples: hal::image::NumSamples,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serial-pass", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct RenderPassContext {
    pub attachments: AttachmentData<TextureFormat>,
    pub sample_count: u8,
}
#[derive(Clone, Debug, Error)]
pub enum RenderPassCompatibilityError {
    #[error("Incompatible color attachment: {0:?} != {1:?}")]
    IncompatibleColorAttachment(
        ArrayVec<[TextureFormat; MAX_COLOR_TARGETS]>,
        ArrayVec<[TextureFormat; MAX_COLOR_TARGETS]>,
    ),
    #[error("Incompatible depth-stencil attachment: {0:?} != {1:?}")]
    IncompatibleDepthStencilAttachment(Option<TextureFormat>, Option<TextureFormat>),
    #[error("Incompatible sample count: {0:?} != {1:?}")]
    IncompatibleSampleCount(u8, u8),
}

impl RenderPassContext {
    // Assumed the renderpass only contains one subpass
    pub(crate) fn check_compatible(
        &self,
        other: &RenderPassContext,
    ) -> Result<(), RenderPassCompatibilityError> {
        if self.attachments.colors != other.attachments.colors {
            return Err(RenderPassCompatibilityError::IncompatibleColorAttachment(
                self.attachments.colors.clone(),
                other.attachments.colors.clone(),
            ));
        }
        if self.attachments.depth_stencil != other.attachments.depth_stencil {
            return Err(
                RenderPassCompatibilityError::IncompatibleDepthStencilAttachment(
                    self.attachments.depth_stencil,
                    other.attachments.depth_stencil,
                ),
            );
        }
        if self.sample_count != other.sample_count {
            return Err(RenderPassCompatibilityError::IncompatibleSampleCount(
                self.sample_count,
                other.sample_count,
            ));
        }
        Ok(())
    }
}

type BufferMapPendingCallback = (resource::BufferMapOperation, resource::BufferMapAsyncStatus);

fn map_buffer<B: hal::Backend>(
    raw: &B::Device,
    buffer: &mut resource::Buffer<B>,
    offset: hal::buffer::Offset,
    size: BufferAddress,
    kind: HostMap,
) -> Result<ptr::NonNull<u8>, resource::BufferAccessError> {
    let &mut (_, ref mut block) = buffer
        .raw
        .as_mut()
        .ok_or(resource::BufferAccessError::Destroyed)?;
    let ptr = block.map(raw, offset, size).map_err(DeviceError::from)?;

    buffer.sync_mapped_writes = match kind {
        HostMap::Read if !block.is_coherent() => {
            block.invalidate_range(raw, offset, Some(size))?;
            None
        }
        HostMap::Write if !block.is_coherent() => Some(hal::memory::Segment {
            offset,
            size: Some(size),
        }),
        _ => None,
    };

    // Zero out uninitialized parts of the mapping. (Spec dictates all resources behave as if they were initialized with zero)
    //
    // If this is a read mapping, ideally we would use a `fill_buffer` command before reading the data from GPU (i.e. `invalidate_range`).
    // However, this would require us to kick off and wait for a command buffer or piggy back on an existing one (the later is likely the only worthwhile option).
    // As reading uninitialized memory isn't a particular important path to support,
    // we instead just initialize the memory here and make sure it is GPU visible, so this happens at max only once for every buffer region.
    //
    // If this is a write mapping zeroing out the memory here is the only reasonable way as all data is pushed to GPU anyways.
    let zero_init_needs_flush_now = !block.is_coherent() && buffer.sync_mapped_writes.is_none(); // No need to flush if it is flushed later anyways.
    for uninitialized_range in buffer.initialization_status.drain(offset..(size + offset)) {
        let num_bytes = uninitialized_range.end - uninitialized_range.start;
        unsafe {
            ptr::write_bytes(
                ptr.as_ptr().offset(uninitialized_range.start as isize),
                0,
                num_bytes as usize,
            )
        };
        if zero_init_needs_flush_now {
            block.flush_range(raw, uninitialized_range.start, Some(num_bytes))?;
        }
    }

    Ok(ptr)
}

fn unmap_buffer<B: hal::Backend>(
    raw: &B::Device,
    buffer: &mut resource::Buffer<B>,
) -> Result<(), resource::BufferAccessError> {
    let &mut (_, ref mut block) = buffer
        .raw
        .as_mut()
        .ok_or(resource::BufferAccessError::Destroyed)?;
    if let Some(segment) = buffer.sync_mapped_writes.take() {
        block.flush_range(raw, segment.offset, segment.size)?;
    }
    block.unmap(raw);
    Ok(())
}

//Note: this logic is specifically moved out of `handle_mapping()` in order to
// have nothing locked by the time we execute users callback code.
fn fire_map_callbacks<I: IntoIterator<Item = BufferMapPendingCallback>>(callbacks: I) {
    for (operation, status) in callbacks {
        unsafe { (operation.callback)(status, operation.user_data) }
    }
}

#[derive(Debug)]
pub(crate) struct RenderPassLock<B: hal::Backend> {
    pub(crate) render_passes: FastHashMap<RenderPassKey, B::RenderPass>,
    pub(crate) framebuffers: FastHashMap<FramebufferKey, B::Framebuffer>,
}

/// Structure describing a logical device. Some members are internally mutable,
/// stored behind mutexes.
/// TODO: establish clear order of locking for these:
/// `mem_allocator`, `desc_allocator`, `life_tracke`, `trackers`,
/// `render_passes`, `pending_writes`, `trace`.
///
/// Currently, the rules are:
/// 1. `life_tracker` is locked after `hub.devices`, enforced by the type system
/// 1. `self.trackers` is locked last (unenforced)
/// 1. `self.trace` is locked last (unenforced)
#[derive(Debug)]
pub struct Device<B: hal::Backend> {
    pub(crate) raw: B::Device,
    pub(crate) adapter_id: Stored<id::AdapterId>,
    pub(crate) queue_group: hal::queue::QueueGroup<B>,
    pub(crate) cmd_allocator: command::CommandAllocator<B>,
    mem_allocator: Mutex<alloc::MemoryAllocator<B>>,
    desc_allocator: Mutex<descriptor::DescriptorAllocator<B>>,
    //Note: The submission index here corresponds to the last submission that is done.
    pub(crate) life_guard: LifeGuard,
    pub(crate) active_submission_index: SubmissionIndex,
    /// Has to be locked temporarily only (locked last)
    pub(crate) trackers: Mutex<TrackerSet>,
    pub(crate) render_passes: Mutex<RenderPassLock<B>>,
    // Life tracker should be locked right after the device and before anything else.
    life_tracker: Mutex<life::LifetimeTracker<B>>,
    temp_suspected: life::SuspectedResources,
    pub(crate) hal_limits: hal::Limits,
    pub(crate) private_features: PrivateFeatures,
    pub(crate) limits: wgt::Limits,
    pub(crate) features: wgt::Features,
    pub(crate) downlevel: wgt::DownlevelProperties,
    spv_options: naga::back::spv::Options,
    //TODO: move this behind another mutex. This would allow several methods to switch
    // to borrow Device immutably, such as `write_buffer`, `write_texture`, and `buffer_unmap`.
    pending_writes: queue::PendingWrites<B>,
    #[cfg(feature = "trace")]
    pub(crate) trace: Option<Mutex<trace::Trace>>,
}

#[derive(Clone, Debug, Error)]
pub enum CreateDeviceError {
    #[error("not enough memory left")]
    OutOfMemory,
}

impl<B: GfxBackend> Device<B> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        raw: B::Device,
        adapter_id: Stored<id::AdapterId>,
        queue_group: hal::queue::QueueGroup<B>,
        mem_props: hal::adapter::MemoryProperties,
        hal_limits: hal::Limits,
        private_features: PrivateFeatures,
        downlevel: wgt::DownlevelProperties,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
    ) -> Result<Self, CreateDeviceError> {
        let cmd_allocator = command::CommandAllocator::new(queue_group.family, &raw)
            .or(Err(CreateDeviceError::OutOfMemory))?;

        let mem_allocator = alloc::MemoryAllocator::new(mem_props, hal_limits);
        let descriptors = descriptor::DescriptorAllocator::new();
        #[cfg(not(feature = "trace"))]
        match trace_path {
            Some(_) => log::error!("Feature 'trace' is not enabled"),
            None => (),
        }

        let spv_options = {
            use naga::back::spv;
            let mut flags = spv::WriterFlags::empty();
            flags.set(spv::WriterFlags::DEBUG, cfg!(debug_assertions));
            //Note: we don't adjust the coordinate space, because `NDC_Y_UP` is required.
            spv::Options {
                lang_version: (1, 0),
                capabilities: [
                    spv::Capability::Shader,
                    spv::Capability::Matrix,
                    spv::Capability::Sampled1D,
                    spv::Capability::Image1D,
                ]
                .iter()
                .cloned()
                .collect(),
                flags,
            }
        };

        Ok(Self {
            raw,
            adapter_id,
            cmd_allocator,
            mem_allocator: Mutex::new(mem_allocator),
            desc_allocator: Mutex::new(descriptors),
            queue_group,
            life_guard: LifeGuard::new("<device>"),
            active_submission_index: 0,
            trackers: Mutex::new(TrackerSet::new(B::VARIANT)),
            render_passes: Mutex::new(RenderPassLock {
                render_passes: FastHashMap::default(),
                framebuffers: FastHashMap::default(),
            }),
            life_tracker: Mutex::new(life::LifetimeTracker::new()),
            temp_suspected: life::SuspectedResources::default(),
            #[cfg(feature = "trace")]
            trace: trace_path.and_then(|path| match trace::Trace::new(path) {
                Ok(mut trace) => {
                    trace.add(trace::Action::Init {
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
            features: desc.features,
            downlevel,
            spv_options,
            pending_writes: queue::PendingWrites::new(),
        })
    }

    pub(crate) fn last_completed_submission_index(&self) -> SubmissionIndex {
        self.life_guard.submission_index.load(Ordering::Acquire)
    }

    fn lock_life_internal<'this, 'token: 'this>(
        tracker: &'this Mutex<life::LifetimeTracker<B>>,
        _token: &mut Token<'token, Self>,
    ) -> MutexGuard<'this, life::LifetimeTracker<B>> {
        tracker.lock()
    }

    fn lock_life<'this, 'token: 'this>(
        &'this self,
        //TODO: fix this - the token has to be borrowed for the lock
        token: &mut Token<'token, Self>,
    ) -> MutexGuard<'this, life::LifetimeTracker<B>> {
        Self::lock_life_internal(&self.life_tracker, token)
    }

    fn maintain<'this, 'token: 'this, G: GlobalIdentityHandlerFactory>(
        &'this self,
        hub: &Hub<B, G>,
        force_wait: bool,
        token: &mut Token<'token, Self>,
    ) -> Result<Vec<BufferMapPendingCallback>, WaitIdleError> {
        let mut life_tracker = self.lock_life(token);

        life_tracker.triage_suspected(
            hub,
            &self.trackers,
            #[cfg(feature = "trace")]
            self.trace.as_ref(),
            token,
        );
        life_tracker.triage_mapped(hub, token);
        let last_done = life_tracker.triage_submissions(&self.raw, force_wait)?;
        let callbacks = life_tracker.handle_mapping(hub, &self.raw, &self.trackers, token);
        life_tracker.cleanup(&self.raw, &self.mem_allocator, &self.desc_allocator);

        self.life_guard
            .submission_index
            .store(last_done, Ordering::Release);
        self.cmd_allocator.maintain(&self.raw, last_done);
        Ok(callbacks)
    }

    fn untrack<'this, 'token: 'this, G: GlobalIdentityHandlerFactory>(
        &'this mut self,
        hub: &Hub<B, G>,
        trackers: &TrackerSet,
        mut token: &mut Token<'token, Self>,
    ) {
        self.temp_suspected.clear();
        // As the tracker is cleared/dropped, we need to consider all the resources
        // that it references for destruction in the next GC pass.
        {
            let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
            let (compute_pipe_guard, mut token) = hub.compute_pipelines.read(&mut token);
            let (render_pipe_guard, mut token) = hub.render_pipelines.read(&mut token);
            let (query_set_guard, mut token) = hub.query_sets.read(&mut token);
            let (buffer_guard, mut token) = hub.buffers.read(&mut token);
            let (texture_guard, mut token) = hub.textures.read(&mut token);
            let (texture_view_guard, mut token) = hub.texture_views.read(&mut token);
            let (sampler_guard, _) = hub.samplers.read(&mut token);

            for id in trackers.buffers.used() {
                if buffer_guard[id].life_guard.ref_count.is_none() {
                    self.temp_suspected.buffers.push(id);
                }
            }
            for id in trackers.textures.used() {
                if texture_guard[id].life_guard.ref_count.is_none() {
                    self.temp_suspected.textures.push(id);
                }
            }
            for id in trackers.views.used() {
                if texture_view_guard[id].life_guard.ref_count.is_none() {
                    self.temp_suspected.texture_views.push(id);
                }
            }
            for id in trackers.bind_groups.used() {
                if bind_group_guard[id].life_guard.ref_count.is_none() {
                    self.temp_suspected.bind_groups.push(id);
                }
            }
            for id in trackers.samplers.used() {
                if sampler_guard[id].life_guard.ref_count.is_none() {
                    self.temp_suspected.samplers.push(id);
                }
            }
            for id in trackers.compute_pipes.used() {
                if compute_pipe_guard[id].life_guard.ref_count.is_none() {
                    self.temp_suspected.compute_pipelines.push(id);
                }
            }
            for id in trackers.render_pipes.used() {
                if render_pipe_guard[id].life_guard.ref_count.is_none() {
                    self.temp_suspected.render_pipelines.push(id);
                }
            }
            for id in trackers.query_sets.used() {
                if query_set_guard[id].life_guard.ref_count.is_none() {
                    self.temp_suspected.query_sets.push(id);
                }
            }
        }

        self.lock_life(&mut token)
            .suspected_resources
            .extend(&self.temp_suspected);
    }

    fn create_buffer(
        &self,
        self_id: id::DeviceId,
        desc: &resource::BufferDescriptor,
        transient: bool,
    ) -> Result<resource::Buffer<B>, resource::CreateBufferError> {
        debug_assert_eq!(self_id.backend(), B::VARIANT);
        let (mut usage, _memory_properties) = conv::map_buffer_usage(desc.usage);
        if desc.mapped_at_creation {
            if desc.size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
                return Err(resource::CreateBufferError::UnalignedSize);
            }
            if !desc.usage.contains(wgt::BufferUsage::MAP_WRITE) {
                // we are going to be copying into it, internally
                usage |= hal::buffer::Usage::TRANSFER_DST;
            }
        } else {
            // We are required to zero out (initialize) all memory.
            // This is done on demand using fill_buffer which requires write transfer usage!
            usage |= hal::buffer::Usage::TRANSFER_DST;
        }

        if desc.usage.is_empty() {
            return Err(resource::CreateBufferError::EmptyUsage);
        }

        let mem_usage = {
            use gpu_alloc::UsageFlags as Uf;
            use wgt::BufferUsage as Bu;

            let mut flags = Uf::empty();
            let map_flags = desc.usage & (Bu::MAP_READ | Bu::MAP_WRITE);
            let map_copy_flags =
                desc.usage & (Bu::MAP_READ | Bu::MAP_WRITE | Bu::COPY_SRC | Bu::COPY_DST);
            if map_flags.is_empty() || !(desc.usage - map_copy_flags).is_empty() {
                flags |= Uf::FAST_DEVICE_ACCESS;
            }
            if transient {
                flags |= Uf::TRANSIENT;
            }

            if !map_flags.is_empty() {
                let upload_usage = Bu::MAP_WRITE | Bu::COPY_SRC;
                let download_usage = Bu::MAP_READ | Bu::COPY_DST;

                flags |= Uf::HOST_ACCESS;
                if desc.usage.contains(upload_usage) {
                    flags |= Uf::UPLOAD;
                }
                if desc.usage.contains(download_usage) {
                    flags |= Uf::DOWNLOAD;
                }

                let is_native_only = self
                    .features
                    .contains(wgt::Features::MAPPABLE_PRIMARY_BUFFERS);
                if !is_native_only
                    && !upload_usage.contains(desc.usage)
                    && !download_usage.contains(desc.usage)
                {
                    return Err(resource::CreateBufferError::UsageMismatch(desc.usage));
                }
            }

            flags
        };

        let mut buffer = unsafe {
            self.raw
                .create_buffer(desc.size.max(1), usage, hal::memory::SparseFlags::empty())
        }
        .map_err(|err| match err {
            hal::buffer::CreationError::OutOfMemory(_) => DeviceError::OutOfMemory,
            _ => panic!("failed to create buffer: {}", err),
        })?;
        if let Some(ref label) = desc.label {
            unsafe { self.raw.set_buffer_name(&mut buffer, label) };
        }

        let requirements = unsafe { self.raw.get_buffer_requirements(&buffer) };
        let block = self
            .mem_allocator
            .lock()
            .allocate(&self.raw, requirements, mem_usage)?;
        block.bind_buffer(&self.raw, &mut buffer)?;

        Ok(resource::Buffer {
            raw: Some((buffer, block)),
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            usage: desc.usage,
            size: desc.size,
            initialization_status: MemoryInitTracker::new(desc.size),
            sync_mapped_writes: None,
            map_state: resource::BufferMapState::Idle,
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
        })
    }

    fn create_texture(
        &self,
        self_id: id::DeviceId,
        adapter: &crate::instance::Adapter<B>,
        desc: &resource::TextureDescriptor,
    ) -> Result<resource::Texture<B>, resource::CreateTextureError> {
        debug_assert_eq!(self_id.backend(), B::VARIANT);

        let format_desc = desc.format.describe();
        let required_features = format_desc.required_features;
        if !self.features.contains(required_features) {
            return Err(resource::CreateTextureError::MissingFeature(
                required_features,
                desc.format,
            ));
        }

        // Ensure `D24Plus` textures cannot be copied
        match desc.format {
            TextureFormat::Depth24Plus | TextureFormat::Depth24PlusStencil8 => {
                if desc
                    .usage
                    .intersects(wgt::TextureUsage::COPY_SRC | wgt::TextureUsage::COPY_DST)
                {
                    return Err(resource::CreateTextureError::CannotCopyD24Plus);
                }
            }
            _ => {}
        }

        if desc.usage.is_empty() {
            return Err(resource::CreateTextureError::EmptyUsage);
        }

        let format_features = if self
            .features
            .contains(wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            adapter.get_texture_format_features(desc.format)
        } else {
            format_desc.guaranteed_format_features
        };

        let missing_allowed_usages = desc.usage - format_features.allowed_usages;
        if !missing_allowed_usages.is_empty() {
            return Err(resource::CreateTextureError::InvalidUsages(
                missing_allowed_usages,
                desc.format,
            ));
        }

        let kind = conv::map_texture_dimension_size(
            desc.dimension,
            desc.size,
            desc.sample_count,
            &self.limits,
        )?;
        let format = conv::map_texture_format(desc.format, self.private_features);
        let aspects = format.surface_desc().aspects;
        let usage = conv::map_texture_usage(desc.usage, aspects);

        let mip_level_count = desc.mip_level_count;
        if mip_level_count == 0
            || mip_level_count > MAX_MIP_LEVELS
            || mip_level_count > kind.compute_num_levels() as u32
        {
            return Err(resource::CreateTextureError::InvalidMipLevelCount(
                mip_level_count,
            ));
        }
        let mut view_caps = hal::image::ViewCapabilities::empty();
        // 2D textures with array layer counts that are multiples of 6 could be cubemaps
        // Following gpuweb/gpuweb#68 always add the hint in that case
        if desc.dimension == TextureDimension::D2 && desc.size.depth_or_array_layers % 6 == 0 {
            view_caps |= hal::image::ViewCapabilities::KIND_CUBE;
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
                    hal::memory::SparseFlags::empty(),
                    view_caps,
                )
                .map_err(|err| match err {
                    hal::image::CreationError::OutOfMemory(_) => DeviceError::OutOfMemory,
                    _ => panic!("failed to create texture: {}", err),
                })?;
            if let Some(ref label) = desc.label {
                self.raw.set_image_name(&mut image, label);
            }
            image
        };

        let requirements = unsafe { self.raw.get_image_requirements(&image) };
        let block = self.mem_allocator.lock().allocate(
            &self.raw,
            requirements,
            gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
        )?;
        block.bind_image(&self.raw, &mut image)?;

        Ok(resource::Texture {
            raw: Some((image, block)),
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            usage: desc.usage,
            aspects,
            dimension: desc.dimension,
            kind,
            format: desc.format,
            format_features,
            framebuffer_attachment: hal::image::FramebufferAttachment {
                usage,
                format,
                view_caps,
            },
            full_range: TextureSelector {
                levels: 0..desc.mip_level_count as hal::image::Level,
                layers: 0..kind.num_layers(),
            },
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
        })
    }

    fn create_texture_view(
        &self,
        texture: &resource::Texture<B>,
        texture_id: id::TextureId,
        desc: &resource::TextureViewDescriptor,
    ) -> Result<resource::TextureView<B>, resource::CreateTextureViewError> {
        let &(ref texture_raw, _) = texture
            .raw
            .as_ref()
            .ok_or(resource::CreateTextureViewError::InvalidTexture)?;

        let view_dim =
            match desc.dimension {
                Some(dim) => {
                    use hal::image::Kind;

                    let required_tex_dim = dim.compatible_texture_dimension();

                    if required_tex_dim != texture.dimension {
                        return Err(
                            resource::CreateTextureViewError::InvalidTextureViewDimension {
                                view: dim,
                                image: texture.dimension,
                            },
                        );
                    }

                    if let Kind::D2(_, _, depth, _) = texture.kind {
                        match dim {
                            TextureViewDimension::Cube if depth != 6 => {
                                return Err(
                                    resource::CreateTextureViewError::InvalidCubemapTextureDepth {
                                        depth,
                                    },
                                )
                            }
                            TextureViewDimension::CubeArray if depth % 6 != 0 => return Err(
                                resource::CreateTextureViewError::InvalidCubemapArrayTextureDepth {
                                    depth,
                                },
                            ),
                            _ => {}
                        }
                    }

                    dim
                }
                None => match texture.kind {
                    hal::image::Kind::D1(..) => wgt::TextureViewDimension::D1,
                    hal::image::Kind::D2(_, _, depth, _)
                        if depth > 1 && desc.array_layer_count.is_none() =>
                    {
                        wgt::TextureViewDimension::D2Array
                    }
                    hal::image::Kind::D2(..) => wgt::TextureViewDimension::D2,
                    hal::image::Kind::D3(..) => wgt::TextureViewDimension::D3,
                },
            };

        let required_level_count =
            desc.base_mip_level + desc.mip_level_count.map_or(1, |count| count.get());
        let required_layer_count =
            desc.base_array_layer + desc.array_layer_count.map_or(1, |count| count.get());
        let level_end = texture.full_range.levels.end;
        let layer_end = texture.full_range.layers.end;
        if required_level_count > level_end as u32 {
            return Err(resource::CreateTextureViewError::TooManyMipLevels {
                requested: required_level_count,
                total: level_end,
            });
        }
        if required_layer_count > layer_end as u32 {
            return Err(resource::CreateTextureViewError::TooManyArrayLayers {
                requested: required_layer_count,
                total: layer_end,
            });
        };

        let aspects = match desc.aspect {
            wgt::TextureAspect::All => texture.aspects,
            wgt::TextureAspect::DepthOnly => hal::format::Aspects::DEPTH,
            wgt::TextureAspect::StencilOnly => hal::format::Aspects::STENCIL,
        };
        if !texture.aspects.contains(aspects) {
            return Err(resource::CreateTextureViewError::InvalidAspect {
                requested: aspects,
                total: texture.aspects,
            });
        }

        let end_level = desc
            .mip_level_count
            .map_or(level_end, |_| required_level_count as u8);
        let end_layer = desc
            .array_layer_count
            .map_or(layer_end, |_| required_layer_count as u16);
        let selector = TextureSelector {
            levels: desc.base_mip_level as u8..end_level,
            layers: desc.base_array_layer as u16..end_layer,
        };

        let view_layer_count = (selector.layers.end - selector.layers.start) as u32;
        let layer_check_ok = match view_dim {
            wgt::TextureViewDimension::D1
            | wgt::TextureViewDimension::D2
            | wgt::TextureViewDimension::D3 => view_layer_count == 1,
            wgt::TextureViewDimension::D2Array => true,
            wgt::TextureViewDimension::Cube => view_layer_count == 6,
            wgt::TextureViewDimension::CubeArray => view_layer_count % 6 == 0,
        };
        if !layer_check_ok {
            return Err(resource::CreateTextureViewError::InvalidArrayLayerCount {
                requested: view_layer_count,
                dim: view_dim,
            });
        }

        let format = desc.format.unwrap_or(texture.format);
        let range = hal::image::SubresourceRange {
            aspects,
            level_start: desc.base_mip_level as _,
            level_count: desc.mip_level_count.map(|v| v.get() as _),
            layer_start: desc.base_array_layer as _,
            layer_count: desc.array_layer_count.map(|v| v.get() as _),
        };
        let hal_extent = texture.kind.extent().at_level(desc.base_mip_level as _);

        let raw = unsafe {
            self.raw
                .create_image_view(
                    texture_raw,
                    conv::map_texture_view_dimension(view_dim),
                    conv::map_texture_format(format, self.private_features),
                    hal::format::Swizzle::NO,
                    range,
                )
                .or(Err(resource::CreateTextureViewError::OutOfMemory))?
        };

        Ok(resource::TextureView {
            inner: resource::TextureViewInner::Native {
                raw,
                source_id: Stored {
                    value: id::Valid(texture_id),
                    ref_count: texture.life_guard.add_ref(),
                },
            },
            aspects,
            format: texture.format,
            format_features: texture.format_features,
            dimension: view_dim,
            extent: wgt::Extent3d {
                width: hal_extent.width,
                height: hal_extent.height,
                depth_or_array_layers: view_layer_count,
            },
            samples: texture.kind.num_samples(),
            framebuffer_attachment: texture.framebuffer_attachment.clone(),
            // once a storage - forever a storage
            sampled_internal_use: if texture.usage.contains(wgt::TextureUsage::STORAGE) {
                resource::TextureUse::SAMPLED | resource::TextureUse::STORAGE_LOAD
            } else {
                resource::TextureUse::SAMPLED
            },
            selector,
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
        })
    }

    fn create_sampler(
        &self,
        self_id: id::DeviceId,
        desc: &resource::SamplerDescriptor,
    ) -> Result<resource::Sampler<B>, resource::CreateSamplerError> {
        let clamp_to_border_enabled = self
            .features
            .contains(wgt::Features::ADDRESS_MODE_CLAMP_TO_BORDER);
        let clamp_to_border_found = desc
            .address_modes
            .iter()
            .any(|am| am == &wgt::AddressMode::ClampToBorder);
        if clamp_to_border_found && !clamp_to_border_enabled {
            return Err(resource::CreateSamplerError::MissingFeature(
                wgt::Features::ADDRESS_MODE_CLAMP_TO_BORDER,
            ));
        }

        let actual_clamp = if let Some(clamp) = desc.anisotropy_clamp {
            let clamp = clamp.get();
            let valid_clamp = clamp <= MAX_ANISOTROPY && conv::is_power_of_two(clamp as u32);
            if !valid_clamp {
                return Err(resource::CreateSamplerError::InvalidClamp(clamp));
            }
            if self.private_features.anisotropic_filtering {
                Some(clamp)
            } else {
                None
            }
        } else {
            None
        };

        let border = match desc.border_color {
            None | Some(wgt::SamplerBorderColor::TransparentBlack) => {
                hal::image::BorderColor::TransparentBlack
            }
            Some(wgt::SamplerBorderColor::OpaqueBlack) => hal::image::BorderColor::OpaqueBlack,
            Some(wgt::SamplerBorderColor::OpaqueWhite) => hal::image::BorderColor::OpaqueWhite,
        };

        let info = hal::image::SamplerDesc {
            min_filter: conv::map_filter(desc.min_filter),
            mag_filter: conv::map_filter(desc.mag_filter),
            mip_filter: conv::map_filter(desc.mipmap_filter),
            wrap_mode: (
                conv::map_wrap(desc.address_modes[0]),
                conv::map_wrap(desc.address_modes[1]),
                conv::map_wrap(desc.address_modes[2]),
            ),
            lod_bias: hal::image::Lod(0.0),
            lod_range: hal::image::Lod(desc.lod_min_clamp)..hal::image::Lod(desc.lod_max_clamp),
            comparison: desc.compare.map(conv::map_compare_function),
            border,
            normalized: true,
            anisotropy_clamp: actual_clamp,
        };

        let raw = unsafe {
            self.raw.create_sampler(&info).map_err(|err| match err {
                hal::device::AllocationError::OutOfMemory(_) => {
                    resource::CreateSamplerError::Device(DeviceError::OutOfMemory)
                }
                hal::device::AllocationError::TooManyObjects => {
                    resource::CreateSamplerError::TooManyObjects
                }
            })?
        };
        Ok(resource::Sampler {
            raw,
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
            comparison: info.comparison.is_some(),
        })
    }

    fn create_shader_module<'a>(
        &self,
        self_id: id::DeviceId,
        desc: &pipeline::ShaderModuleDescriptor<'a>,
        source: pipeline::ShaderModuleSource<'a>,
    ) -> Result<pipeline::ShaderModule<B>, pipeline::CreateShaderModuleError> {
        // First, try to produce a Naga module.
        let (spv, module) = match source {
            pipeline::ShaderModuleSource::SpirV(spv) => {
                // Parse the given shader code and store its representation.
                let options = naga::front::spv::Options {
                    adjust_coordinate_space: false, // we require NDC_Y_UP feature
                    flow_graph_dump_prefix: None,
                };
                let parser = naga::front::spv::Parser::new(spv.iter().cloned(), &options);
                let module = match parser.parse() {
                    Ok(module) => Some(module),
                    Err(err) => {
                        // TODO: eventually, when Naga gets support for all features,
                        // we want to convert these to a hard error,
                        log::warn!("Failed to parse shader SPIR-V code: {:?}", err);
                        log::warn!("Shader module will not be validated or reflected");
                        None
                    }
                };
                (Some(spv), module)
            }
            pipeline::ShaderModuleSource::Wgsl(code) => {
                // TODO: refactor the corresponding Naga error to be owned, and then
                // display it instead of unwrapping
                match naga::front::wgsl::parse_str(&code) {
                    Ok(module) => (None, Some(module)),
                    Err(err) => {
                        log::error!("Failed to parse WGSL code: {}", err);
                        return Err(pipeline::CreateShaderModuleError::Parsing);
                    }
                }
            }
            pipeline::ShaderModuleSource::Naga(module) => (None, Some(module)),
        };

        let (naga_result, interface) = match module {
            // If succeeded, then validate it and attempt to give it to gfx-hal directly.
            Some(module) if desc.flags.contains(wgt::ShaderFlags::VALIDATION) || spv.is_none() => {
                let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all())
                    .validate(&module)?;
                if !self.features.contains(wgt::Features::PUSH_CONSTANTS)
                    && module
                        .global_variables
                        .iter()
                        .any(|(_, var)| var.class == naga::StorageClass::PushConstant)
                {
                    return Err(pipeline::CreateShaderModuleError::MissingFeature(
                        wgt::Features::PUSH_CONSTANTS,
                    ));
                }
                let interface = validation::Interface::new(&module, &info);
                let shader = hal::device::NagaShader { module, info };
                let naga_result = if desc
                    .flags
                    .contains(wgt::ShaderFlags::EXPERIMENTAL_TRANSLATION)
                    || !cfg!(feature = "cross")
                {
                    match unsafe { self.raw.create_shader_module_from_naga(shader) } {
                        Ok(raw) => Ok(raw),
                        Err((hal::device::ShaderError::CompilationFailed(msg), shader)) => {
                            log::warn!("Shader module compilation failed: {}", msg);
                            Err(Some(shader))
                        }
                        Err((_, shader)) => Err(Some(shader)),
                    }
                } else {
                    Err(Some(shader))
                };
                (naga_result, Some(interface))
            }
            _ => (Err(None), None),
        };

        // Otherwise, fall back to SPIR-V.
        let spv_result = match naga_result {
            Ok(raw) => Ok(raw),
            Err(maybe_shader) => {
                let spv = match spv {
                    Some(data) => Ok(data),
                    None => {
                        // Produce a SPIR-V from the Naga module
                        let shader = maybe_shader.unwrap();
                        naga::back::spv::write_vec(&shader.module, &shader.info, &self.spv_options)
                            .map(Cow::Owned)
                    }
                };
                match spv {
                    Ok(data) => unsafe { self.raw.create_shader_module(&data) },
                    Err(e) => Err(hal::device::ShaderError::CompilationFailed(format!(
                        "{}",
                        e
                    ))),
                }
            }
        };

        Ok(pipeline::ShaderModule {
            raw: match spv_result {
                Ok(raw) => raw,
                Err(hal::device::ShaderError::OutOfMemory(_)) => {
                    return Err(DeviceError::OutOfMemory.into());
                }
                Err(error) => {
                    log::error!("Shader error: {}", error);
                    return Err(pipeline::CreateShaderModuleError::Parsing);
                }
            },
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            interface,
            #[cfg(debug_assertions)]
            label: desc.label.to_string_or_default(),
        })
    }

    /// Create a compatible render pass with a given key.
    ///
    /// This functions doesn't consider the following aspects for compatibility:
    ///  - image layouts
    ///  - resolve attachments
    fn create_compatible_render_pass(
        &self,
        key: &RenderPassKey,
    ) -> Result<B::RenderPass, hal::device::OutOfMemory> {
        let mut color_ids = [(0, hal::image::Layout::ColorAttachmentOptimal); MAX_COLOR_TARGETS];
        for (index, color) in color_ids[..key.colors.len()].iter_mut().enumerate() {
            color.0 = index;
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
        let all = key.all().map(|&(ref at, _)| at.clone());

        unsafe {
            self.raw
                .create_render_pass(all, iter::once(subpass), iter::empty())
        }
    }

    fn deduplicate_bind_group_layout(
        self_id: id::DeviceId,
        entry_map: &binding_model::BindEntryMap,
        guard: &Storage<binding_model::BindGroupLayout<B>, id::BindGroupLayoutId>,
    ) -> Option<id::BindGroupLayoutId> {
        guard
            .iter(self_id.backend())
            .find(|&(_, ref bgl)| bgl.device_id.value.0 == self_id && bgl.entries == *entry_map)
            .map(|(id, value)| {
                value.multi_ref_count.inc();
                id
            })
    }

    fn get_introspection_bind_group_layouts<'a>(
        pipeline_layout: &binding_model::PipelineLayout<B>,
        bgl_guard: &'a Storage<binding_model::BindGroupLayout<B>, id::BindGroupLayoutId>,
    ) -> ArrayVec<[&'a binding_model::BindEntryMap; MAX_BIND_GROUPS]> {
        pipeline_layout
            .bind_group_layout_ids
            .iter()
            .map(|&id| &bgl_guard[id].entries)
            .collect()
    }

    fn create_bind_group_layout(
        &self,
        self_id: id::DeviceId,
        label: Option<&str>,
        entry_map: binding_model::BindEntryMap,
    ) -> Result<binding_model::BindGroupLayout<B>, binding_model::CreateBindGroupLayoutError> {
        let mut desc_count = descriptor::DescriptorTotalCount::default();
        for binding in entry_map.values() {
            use wgt::BindingType as Bt;
            let (counter, array_feature) = match binding.ty {
                Bt::Buffer {
                    ty: wgt::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: _,
                } => (&mut desc_count.uniform_buffer, None),
                Bt::Buffer {
                    ty: wgt::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: _,
                } => (&mut desc_count.uniform_buffer_dynamic, None),
                Bt::Buffer {
                    ty: wgt::BufferBindingType::Storage { .. },
                    has_dynamic_offset: false,
                    min_binding_size: _,
                } => (&mut desc_count.storage_buffer, None),
                Bt::Buffer {
                    ty: wgt::BufferBindingType::Storage { .. },
                    has_dynamic_offset: true,
                    min_binding_size: _,
                } => (&mut desc_count.storage_buffer_dynamic, None),
                Bt::Sampler { .. } => (&mut desc_count.sampler, None),
                Bt::Texture { .. } => (
                    &mut desc_count.sampled_image,
                    Some(wgt::Features::SAMPLED_TEXTURE_BINDING_ARRAY),
                ),
                Bt::StorageTexture { .. } => (&mut desc_count.storage_image, None),
            };
            *counter += match binding.count {
                // Validate the count parameter
                Some(count) => {
                    let feature = array_feature
                        .ok_or(binding_model::CreateBindGroupLayoutError::ArrayUnsupported)?;
                    if !self.features.contains(feature) {
                        return Err(binding_model::CreateBindGroupLayoutError::MissingFeature(
                            feature,
                        ));
                    }
                    count.get()
                }
                None => 1,
            };
        }

        let raw_bindings = entry_map
            .values()
            .map(|entry| hal::pso::DescriptorSetLayoutBinding {
                binding: entry.binding,
                ty: conv::map_binding_type(entry),
                count: entry
                    .count
                    .map_or(1, |v| v.get() as hal::pso::DescriptorArrayIndex), //TODO: consolidate
                stage_flags: conv::map_shader_stage_flags(entry.visibility),
                immutable_samplers: false, // TODO
            });
        let raw = unsafe {
            let mut raw_layout = self
                .raw
                .create_descriptor_set_layout(raw_bindings, iter::empty())
                .or(Err(DeviceError::OutOfMemory))?;
            if let Some(label) = label {
                self.raw
                    .set_descriptor_set_layout_name(&mut raw_layout, label);
            }
            raw_layout
        };

        let mut count_validator = binding_model::BindingTypeMaxCountValidator::default();
        for entry in entry_map.values() {
            count_validator.add_binding(entry);
        }
        // If a single bind group layout violates limits, the pipeline layout is definitely
        // going to violate limits too, lets catch it now.
        count_validator
            .validate(&self.limits)
            .map_err(binding_model::CreateBindGroupLayoutError::TooManyBindings)?;

        Ok(binding_model::BindGroupLayout {
            raw,
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            multi_ref_count: MultiRefCount::new(),
            desc_count,
            dynamic_count: entry_map
                .values()
                .filter(|b| b.ty.has_dynamic_offset())
                .count(),
            count_validator,
            entries: entry_map,
            #[cfg(debug_assertions)]
            label: label.unwrap_or("").to_string(),
        })
    }

    fn create_bind_group<G: GlobalIdentityHandlerFactory>(
        &self,
        self_id: id::DeviceId,
        layout: &binding_model::BindGroupLayout<B>,
        desc: &binding_model::BindGroupDescriptor,
        hub: &Hub<B, G>,
        token: &mut Token<binding_model::BindGroupLayout<B>>,
    ) -> Result<binding_model::BindGroup<B>, binding_model::CreateBindGroupError> {
        use crate::binding_model::{BindingResource as Br, CreateBindGroupError as Error};
        {
            // Check that the number of entries in the descriptor matches
            // the number of entries in the layout.
            let actual = desc.entries.len();
            let expected = layout.entries.len();
            if actual != expected {
                return Err(Error::BindingsNumMismatch { expected, actual });
            }
        }

        // TODO: arrayvec/smallvec
        // Record binding info for dynamic offset validation
        let mut dynamic_binding_info = Vec::new();
        // fill out the descriptors
        let mut used = TrackerSet::new(B::VARIANT);

        let (buffer_guard, mut token) = hub.buffers.read(token);
        let (texture_guard, mut token) = hub.textures.read(&mut token); //skip token
        let (texture_view_guard, mut token) = hub.texture_views.read(&mut token);
        let (sampler_guard, _) = hub.samplers.read(&mut token);

        // `BTreeMap` has ordered bindings as keys, which allows us to coalesce
        // the descriptor writes into a single transaction.
        let mut write_map = BTreeMap::new();
        let mut used_buffer_ranges = Vec::new();
        for entry in desc.entries.iter() {
            let binding = entry.binding;
            // Find the corresponding declaration in the layout
            let decl = layout
                .entries
                .get(&binding)
                .ok_or(Error::MissingBindingDeclaration(binding))?;
            let descriptors: SmallVec<[_; 1]> = match entry.resource {
                Br::Buffer(ref bb) => {
                    let (binding_ty, dynamic, min_size) = match decl.ty {
                        wgt::BindingType::Buffer {
                            ty,
                            has_dynamic_offset,
                            min_binding_size,
                        } => (ty, has_dynamic_offset, min_binding_size),
                        _ => {
                            return Err(Error::WrongBindingType {
                                binding,
                                actual: decl.ty,
                                expected: "UniformBuffer, StorageBuffer or ReadonlyStorageBuffer",
                            })
                        }
                    };
                    let (pub_usage, internal_use, range_limit) = match binding_ty {
                        wgt::BufferBindingType::Uniform => (
                            wgt::BufferUsage::UNIFORM,
                            resource::BufferUse::UNIFORM,
                            self.limits.max_uniform_buffer_binding_size,
                        ),
                        wgt::BufferBindingType::Storage { read_only } => (
                            wgt::BufferUsage::STORAGE,
                            if read_only {
                                resource::BufferUse::STORAGE_LOAD
                            } else {
                                resource::BufferUse::STORAGE_STORE
                            },
                            self.limits.max_storage_buffer_binding_size,
                        ),
                    };

                    if bb.offset % wgt::BIND_BUFFER_ALIGNMENT != 0 {
                        return Err(Error::UnalignedBufferOffset(bb.offset));
                    }

                    let buffer = used
                        .buffers
                        .use_extend(&*buffer_guard, bb.buffer_id, (), internal_use)
                        .map_err(|_| Error::InvalidBuffer(bb.buffer_id))?;
                    check_buffer_usage(buffer.usage, pub_usage)?;
                    let &(ref buffer_raw, _) = buffer
                        .raw
                        .as_ref()
                        .ok_or(Error::InvalidBuffer(bb.buffer_id))?;

                    let (bind_size, bind_end) = match bb.size {
                        Some(size) => {
                            let end = bb.offset + size.get();
                            if end > buffer.size {
                                return Err(Error::BindingRangeTooLarge {
                                    buffer: bb.buffer_id,
                                    range: bb.offset..end,
                                    size: buffer.size,
                                });
                            }
                            (size.get(), end)
                        }
                        None => (buffer.size - bb.offset, buffer.size),
                    };

                    if bind_size > range_limit as u64 {
                        return Err(Error::BufferRangeTooLarge {
                            binding,
                            given: bind_size as u32,
                            limit: range_limit,
                        });
                    }

                    // Record binding info for validating dynamic offsets
                    if dynamic {
                        dynamic_binding_info.push(binding_model::BindGroupDynamicBindingData {
                            maximum_dynamic_offset: buffer.size - bind_end,
                        });
                    }

                    if let Some(non_zero) = min_size {
                        let min_size = non_zero.get();
                        if min_size > bind_size {
                            return Err(Error::BindingSizeTooSmall {
                                buffer: bb.buffer_id,
                                actual: bind_size,
                                min: min_size,
                            });
                        }
                    } else if bind_size == 0 {
                        return Err(Error::BindingZeroSize(bb.buffer_id));
                    }

                    used_buffer_ranges.push(MemoryInitTrackerAction {
                        id: bb.buffer_id,
                        range: bb.offset..(bb.offset + bind_size),
                        kind: MemoryInitKind::NeedsInitializedMemory,
                    });

                    let sub_range = hal::buffer::SubRange {
                        offset: bb.offset,
                        size: Some(bind_size),
                    };
                    SmallVec::from([hal::pso::Descriptor::Buffer(buffer_raw, sub_range)])
                }
                Br::Sampler(id) => {
                    match decl.ty {
                        wgt::BindingType::Sampler {
                            filtering: _,
                            comparison,
                        } => {
                            let sampler = used
                                .samplers
                                .use_extend(&*sampler_guard, id, (), ())
                                .map_err(|_| Error::InvalidSampler(id))?;

                            // Check the actual sampler to also (not) be a comparison sampler
                            if sampler.comparison != comparison {
                                return Err(Error::WrongSamplerComparison);
                            }

                            SmallVec::from([hal::pso::Descriptor::Sampler(&sampler.raw)])
                        }
                        _ => {
                            return Err(Error::WrongBindingType {
                                binding,
                                actual: decl.ty,
                                expected: "Sampler",
                            })
                        }
                    }
                }
                Br::TextureView(id) => {
                    let view = used
                        .views
                        .use_extend(&*texture_view_guard, id, (), ())
                        .map_err(|_| Error::InvalidTextureView(id))?;
                    let format_info = view.format.describe();
                    let (pub_usage, internal_use) = match decl.ty {
                        wgt::BindingType::Texture {
                            sample_type,
                            view_dimension,
                            multisampled,
                        } => {
                            use wgt::TextureSampleType as Tst;
                            if multisampled != (view.samples != 1) {
                                return Err(Error::InvalidTextureMultisample {
                                    binding,
                                    layout_multisampled: multisampled,
                                    view_samples: view.samples as u32,
                                });
                            }
                            match (sample_type, format_info.sample_type, view.format_features.filterable ) {
                                (Tst::Uint, Tst::Uint, ..) |
                                (Tst::Sint, Tst::Sint, ..) |
                                (Tst::Depth, Tst::Depth, ..) |
                                // if we expect non-filterable, accept anything float
                                (Tst::Float { filterable: false }, Tst::Float { .. }, ..) |
                                // if we expect filterable, require it
                                (Tst::Float { filterable: true }, Tst::Float { filterable: true }, ..) |
                                // if we expect filterable, also accept Float that is defined as unfilterable if filterable feature is explicitly enabled
                                // (only hit if wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES is enabled)
                                (Tst::Float { filterable: true }, Tst::Float { .. }, true) |
                                // if we expect float, also accept depth
                                (Tst::Float { .. }, Tst::Depth, ..) => {}
                                _ => {
                                    return Err(Error::InvalidTextureSampleType {
                                    binding,
                                    layout_sample_type: sample_type,
                                    view_format: view.format,
                                })
                            },
                            }
                            if view_dimension != view.dimension {
                                return Err(Error::InvalidTextureDimension {
                                    binding,
                                    layout_dimension: view_dimension,
                                    view_dimension: view.dimension,
                                });
                            }
                            (wgt::TextureUsage::SAMPLED, view.sampled_internal_use)
                        }
                        wgt::BindingType::StorageTexture {
                            access,
                            format,
                            view_dimension,
                        } => {
                            if format != view.format {
                                return Err(Error::InvalidStorageTextureFormat {
                                    binding,
                                    layout_format: format,
                                    view_format: view.format,
                                });
                            }
                            if view_dimension != view.dimension {
                                return Err(Error::InvalidTextureDimension {
                                    binding,
                                    layout_dimension: view_dimension,
                                    view_dimension: view.dimension,
                                });
                            }
                            let internal_use = match access {
                                wgt::StorageTextureAccess::ReadOnly => {
                                    resource::TextureUse::STORAGE_LOAD
                                }
                                wgt::StorageTextureAccess::WriteOnly => {
                                    resource::TextureUse::STORAGE_STORE
                                }
                                wgt::StorageTextureAccess::ReadWrite => {
                                    if !view.format_features.flags.contains(
                                        wgt::TextureFormatFeatureFlags::STORAGE_READ_WRITE,
                                    ) {
                                        return Err(if self.features.contains(
                                            wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                                        ) {
                                            Error::StorageReadWriteNotSupported(view.format)
                                        } else {
                                            Error::MissingFeatures(wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
                                        });
                                    }

                                    resource::TextureUse::STORAGE_STORE
                                        | resource::TextureUse::STORAGE_LOAD
                                }
                            };
                            (wgt::TextureUsage::STORAGE, internal_use)
                        }
                        _ => return Err(Error::WrongBindingType {
                            binding,
                            actual: decl.ty,
                            expected:
                                "SampledTexture, ReadonlyStorageTexture or WriteonlyStorageTexture",
                        }),
                    };
                    if view
                        .aspects
                        .contains(hal::format::Aspects::DEPTH | hal::format::Aspects::STENCIL)
                    {
                        return Err(Error::DepthStencilAspect);
                    }
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
                                    view.selector.clone(),
                                    internal_use,
                                )
                                .unwrap();
                            check_texture_usage(texture.usage, pub_usage)?;
                            let image_layout =
                                conv::map_texture_state(internal_use, view.aspects).1;
                            SmallVec::from([hal::pso::Descriptor::Image(raw, image_layout)])
                        }
                        resource::TextureViewInner::SwapChain { .. } => {
                            return Err(Error::SwapChainImage);
                        }
                    }
                }
                Br::TextureViewArray(ref bindings_array) => {
                    let required_feats = wgt::Features::SAMPLED_TEXTURE_BINDING_ARRAY;
                    if !self.features.contains(required_feats) {
                        return Err(Error::MissingFeatures(required_feats));
                    }

                    if let Some(count) = decl.count {
                        let count = count.get() as usize;
                        let num_bindings = bindings_array.len();
                        if count != num_bindings {
                            return Err(Error::BindingArrayLengthMismatch {
                                actual: num_bindings,
                                expected: count,
                            });
                        }
                    } else {
                        return Err(Error::SingleBindingExpected);
                    }

                    bindings_array
                        .iter()
                        .map(|&id| {
                            let view = used
                                .views
                                .use_extend(&*texture_view_guard, id, (), ())
                                .map_err(|_| Error::InvalidTextureView(id))?;
                            let (pub_usage, internal_use) = match decl.ty {
                                wgt::BindingType::Texture { .. } => {
                                    (wgt::TextureUsage::SAMPLED, view.sampled_internal_use)
                                }
                                _ => {
                                    return Err(Error::WrongBindingType {
                                        binding,
                                        actual: decl.ty,
                                        expected: "SampledTextureArray",
                                    })
                                }
                            };
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
                                            view.selector.clone(),
                                            internal_use,
                                        )
                                        .unwrap();
                                    check_texture_usage(texture.usage, pub_usage)?;
                                    let image_layout =
                                        conv::map_texture_state(internal_use, view.aspects).1;
                                    Ok(hal::pso::Descriptor::Image(raw, image_layout))
                                }
                                resource::TextureViewInner::SwapChain { .. } => {
                                    Err(Error::SwapChainImage)
                                }
                            }
                        })
                        .collect::<Result<_, _>>()?
                }
            };
            if write_map.insert(binding, descriptors).is_some() {
                return Err(Error::DuplicateBinding(binding));
            }
        }

        let mut desc_sets =
            self.desc_allocator
                .lock()
                .allocate(&self.raw, &layout.raw, &layout.desc_count, 1)?;
        let mut desc_set = desc_sets.pop().unwrap();

        // Set the descriptor set's label for easier debugging.
        if let Some(label) = desc.label.as_ref() {
            unsafe {
                self.raw.set_descriptor_set_name(desc_set.raw_mut(), &label);
            }
        }

        if let Some(start_binding) = write_map.keys().next().cloned() {
            let descriptors = write_map.into_iter().flat_map(|(_, list)| list);
            unsafe {
                let write = hal::pso::DescriptorSetWrite {
                    set: desc_set.raw_mut(),
                    binding: start_binding,
                    array_offset: 0,
                    descriptors,
                };
                self.raw.write_descriptor_set(write);
            }
        }

        Ok(binding_model::BindGroup {
            raw: desc_set,
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            layout_id: id::Valid(desc.layout),
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
            used,
            used_buffer_ranges,
            dynamic_binding_info,
        })
    }

    fn create_pipeline_layout(
        &self,
        self_id: id::DeviceId,
        desc: &binding_model::PipelineLayoutDescriptor,
        bgl_guard: &Storage<binding_model::BindGroupLayout<B>, id::BindGroupLayoutId>,
    ) -> Result<binding_model::PipelineLayout<B>, binding_model::CreatePipelineLayoutError> {
        use crate::binding_model::CreatePipelineLayoutError as Error;

        let bind_group_layouts_count = desc.bind_group_layouts.len();
        let device_max_bind_groups = self.limits.max_bind_groups as usize;
        if bind_group_layouts_count > device_max_bind_groups {
            return Err(Error::TooManyGroups {
                actual: bind_group_layouts_count,
                max: device_max_bind_groups,
            });
        }

        if !desc.push_constant_ranges.is_empty()
            && !self.features.contains(wgt::Features::PUSH_CONSTANTS)
        {
            return Err(Error::MissingFeature(wgt::Features::PUSH_CONSTANTS));
        }
        let mut used_stages = wgt::ShaderStage::empty();
        for (index, pc) in desc.push_constant_ranges.iter().enumerate() {
            if pc.stages.intersects(used_stages) {
                return Err(Error::MoreThanOnePushConstantRangePerStage {
                    index,
                    provided: pc.stages,
                    intersected: pc.stages & used_stages,
                });
            }
            used_stages |= pc.stages;

            let device_max_pc_size = self.limits.max_push_constant_size;
            if device_max_pc_size < pc.range.end {
                return Err(Error::PushConstantRangeTooLarge {
                    index,
                    range: pc.range.clone(),
                    max: device_max_pc_size,
                });
            }

            if pc.range.start % wgt::PUSH_CONSTANT_ALIGNMENT != 0 {
                return Err(Error::MisalignedPushConstantRange {
                    index,
                    bound: pc.range.start,
                });
            }
            if pc.range.end % wgt::PUSH_CONSTANT_ALIGNMENT != 0 {
                return Err(Error::MisalignedPushConstantRange {
                    index,
                    bound: pc.range.end,
                });
            }
        }

        let mut count_validator = binding_model::BindingTypeMaxCountValidator::default();

        // validate total resource counts
        for &id in desc.bind_group_layouts.iter() {
            let bind_group_layout = bgl_guard
                .get(id)
                .map_err(|_| Error::InvalidBindGroupLayout(id))?;
            count_validator.merge(&bind_group_layout.count_validator);
        }
        count_validator
            .validate(&self.limits)
            .map_err(Error::TooManyBindings)?;

        let descriptor_set_layouts = desc
            .bind_group_layouts
            .iter()
            .map(|&id| &bgl_guard.get(id).unwrap().raw);
        let push_constants = desc
            .push_constant_ranges
            .iter()
            .map(|pc| (conv::map_shader_stage_flags(pc.stages), pc.range.clone()));

        let raw = unsafe {
            let raw_layout = self
                .raw
                .create_pipeline_layout(descriptor_set_layouts, push_constants)
                .or(Err(DeviceError::OutOfMemory))?;
            if let Some(_) = desc.label {
                //TODO-0.6: needs gfx changes published
                //self.raw.set_pipeline_layout_name(&mut raw_layout, label);
            }
            raw_layout
        };

        Ok(binding_model::PipelineLayout {
            raw,
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
            bind_group_layout_ids: desc
                .bind_group_layouts
                .iter()
                .map(|&id| {
                    bgl_guard.get(id).unwrap().multi_ref_count.inc();
                    id::Valid(id)
                })
                .collect(),
            push_constant_ranges: desc.push_constant_ranges.iter().cloned().collect(),
        })
    }

    //TODO: refactor this. It's the only method of `Device` that registers new objects
    // (the pipeline layout).
    fn derive_pipeline_layout(
        &self,
        self_id: id::DeviceId,
        implicit_context: Option<ImplicitPipelineContext>,
        mut derived_group_layouts: ArrayVec<[binding_model::BindEntryMap; MAX_BIND_GROUPS]>,
        bgl_guard: &mut Storage<binding_model::BindGroupLayout<B>, id::BindGroupLayoutId>,
        pipeline_layout_guard: &mut Storage<binding_model::PipelineLayout<B>, id::PipelineLayoutId>,
    ) -> Result<
        (id::PipelineLayoutId, pipeline::ImplicitBindGroupCount),
        pipeline::ImplicitLayoutError,
    > {
        let derived_bind_group_count =
            derived_group_layouts.len() as pipeline::ImplicitBindGroupCount;

        while derived_group_layouts
            .last()
            .map_or(false, |map| map.is_empty())
        {
            derived_group_layouts.pop();
        }
        let mut ids = implicit_context.ok_or(pipeline::ImplicitLayoutError::MissingIds(0))?;
        let group_count = derived_group_layouts.len();
        if ids.group_ids.len() < group_count {
            log::error!(
                "Not enough bind group IDs ({}) specified for the implicit layout ({})",
                ids.group_ids.len(),
                derived_group_layouts.len()
            );
            return Err(pipeline::ImplicitLayoutError::MissingIds(
                derived_bind_group_count,
            ));
        }

        for (bgl_id, map) in ids.group_ids.iter_mut().zip(derived_group_layouts) {
            match Device::deduplicate_bind_group_layout(self_id, &map, bgl_guard) {
                Some(dedup_id) => {
                    *bgl_id = dedup_id;
                }
                None => {
                    let bgl = self.create_bind_group_layout(self_id, None, map)?;
                    bgl_guard.insert(*bgl_id, bgl);
                }
            };
        }

        let layout_desc = binding_model::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: Cow::Borrowed(&ids.group_ids[..group_count]),
            push_constant_ranges: Cow::Borrowed(&[]), //TODO?
        };
        let layout = self.create_pipeline_layout(self_id, &layout_desc, bgl_guard)?;
        pipeline_layout_guard.insert(ids.root_id, layout);
        Ok((ids.root_id, derived_bind_group_count))
    }

    fn create_compute_pipeline<G: GlobalIdentityHandlerFactory>(
        &self,
        self_id: id::DeviceId,
        desc: &pipeline::ComputePipelineDescriptor,
        implicit_context: Option<ImplicitPipelineContext>,
        hub: &Hub<B, G>,
        token: &mut Token<Self>,
    ) -> Result<
        (
            pipeline::ComputePipeline<B>,
            pipeline::ImplicitBindGroupCount,
            id::PipelineLayoutId,
        ),
        pipeline::CreateComputePipelineError,
    > {
        if !self
            .downlevel
            .flags
            .contains(wgt::DownlevelFlags::COMPUTE_SHADERS)
        {
            return Err(pipeline::CreateComputePipelineError::ComputeShadersUnsupported);
        }

        //TODO: only lock mutable if the layout is derived
        let (mut pipeline_layout_guard, mut token) = hub.pipeline_layouts.write(token);
        let (mut bgl_guard, mut token) = hub.bind_group_layouts.write(&mut token);

        let mut derived_group_layouts =
            ArrayVec::<[binding_model::BindEntryMap; MAX_BIND_GROUPS]>::new();

        let io = validation::StageIo::default();
        let (shader_module_guard, _) = hub.shader_modules.read(&mut token);

        let entry_point_name = &desc.stage.entry_point;
        let shader_module = shader_module_guard.get(desc.stage.module).map_err(|_| {
            pipeline::CreateComputePipelineError::Stage(validation::StageError::InvalidModule)
        })?;

        let flag = wgt::ShaderStage::COMPUTE;
        if let Some(ref interface) = shader_module.interface {
            let provided_layouts = match desc.layout {
                Some(pipeline_layout_id) => Some(Device::get_introspection_bind_group_layouts(
                    pipeline_layout_guard
                        .get(pipeline_layout_id)
                        .map_err(|_| pipeline::CreateComputePipelineError::InvalidLayout)?,
                    &*bgl_guard,
                )),
                None => {
                    for _ in 0..self.limits.max_bind_groups {
                        derived_group_layouts.push(binding_model::BindEntryMap::default());
                    }
                    None
                }
            };
            let _ = interface
                .check_stage(
                    provided_layouts.as_ref().map(|p| p.as_slice()),
                    &mut derived_group_layouts,
                    &entry_point_name,
                    flag,
                    io,
                )
                .map_err(pipeline::CreateComputePipelineError::Stage)?;
        } else if desc.layout.is_none() {
            return Err(pipeline::ImplicitLayoutError::ReflectionError(flag).into());
        }

        let shader = hal::pso::EntryPoint::<B> {
            entry: &entry_point_name, // TODO
            module: &shader_module.raw,
            specialization: hal::pso::Specialization::EMPTY,
        };

        // TODO
        let flags = hal::pso::PipelineCreationFlags::empty();
        // TODO
        let parent = hal::pso::BasePipeline::None;

        let (pipeline_layout_id, derived_bind_group_count) = match desc.layout {
            Some(id) => (id, 0),
            None => self.derive_pipeline_layout(
                self_id,
                implicit_context,
                derived_group_layouts,
                &mut *bgl_guard,
                &mut *pipeline_layout_guard,
            )?,
        };
        let layout = pipeline_layout_guard
            .get(pipeline_layout_id)
            .map_err(|_| pipeline::CreateComputePipelineError::InvalidLayout)?;

        let pipeline_desc = hal::pso::ComputePipelineDesc {
            label: desc.label.as_ref().map(AsRef::as_ref),
            shader,
            layout: &layout.raw,
            flags,
            parent,
        };

        let raw =
            unsafe { self.raw.create_compute_pipeline(&pipeline_desc, None) }.map_err(|err| {
                match err {
                    hal::pso::CreationError::OutOfMemory(_) => {
                        pipeline::CreateComputePipelineError::Device(DeviceError::OutOfMemory)
                    }
                    hal::pso::CreationError::ShaderCreationError(_, error) => {
                        pipeline::CreateComputePipelineError::Internal(error)
                    }
                    _ => {
                        log::error!("failed to create compute pipeline: {}", err);
                        pipeline::CreateComputePipelineError::Device(DeviceError::OutOfMemory)
                    }
                }
            })?;

        let pipeline = pipeline::ComputePipeline {
            raw,
            layout_id: Stored {
                value: id::Valid(pipeline_layout_id),
                ref_count: layout.life_guard.add_ref(),
            },
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
        };
        Ok((pipeline, derived_bind_group_count, pipeline_layout_id))
    }

    fn create_render_pipeline<G: GlobalIdentityHandlerFactory>(
        &self,
        self_id: id::DeviceId,
        desc: &pipeline::RenderPipelineDescriptor,
        implicit_context: Option<ImplicitPipelineContext>,
        hub: &Hub<B, G>,
        token: &mut Token<Self>,
    ) -> Result<
        (
            pipeline::RenderPipeline<B>,
            pipeline::ImplicitBindGroupCount,
            id::PipelineLayoutId,
        ),
        pipeline::CreateRenderPipelineError,
    > {
        //TODO: only lock mutable if the layout is derived
        let (mut pipeline_layout_guard, mut token) = hub.pipeline_layouts.write(token);
        let (mut bgl_guard, mut token) = hub.bind_group_layouts.write(&mut token);

        let mut derived_group_layouts =
            ArrayVec::<[binding_model::BindEntryMap; MAX_BIND_GROUPS]>::new();

        let color_states = desc
            .fragment
            .as_ref()
            .map_or(&[][..], |fragment| &fragment.targets);
        let depth_stencil_state = desc.depth_stencil.as_ref();
        let rasterizer =
            conv::map_primitive_state_to_rasterizer(&desc.primitive, depth_stencil_state);

        let mut io = validation::StageIo::default();
        let mut validated_stages = wgt::ShaderStage::empty();

        let desc_vbs = &desc.vertex.buffers;
        let mut vertex_strides = Vec::with_capacity(desc_vbs.len());
        let mut vertex_buffers = Vec::with_capacity(desc_vbs.len());
        let mut attributes = Vec::new();
        for (i, vb_state) in desc_vbs.iter().enumerate() {
            vertex_strides
                .alloc()
                .init((vb_state.array_stride, vb_state.step_mode));
            if vb_state.attributes.is_empty() {
                continue;
            }
            if vb_state.array_stride > self.limits.max_vertex_buffer_array_stride as u64 {
                return Err(pipeline::CreateRenderPipelineError::VertexStrideTooLarge {
                    index: i as u32,
                    given: vb_state.array_stride as u32,
                    limit: self.limits.max_vertex_buffer_array_stride,
                });
            }
            if vb_state.array_stride % wgt::VERTEX_STRIDE_ALIGNMENT != 0 {
                return Err(pipeline::CreateRenderPipelineError::UnalignedVertexStride {
                    index: i as u32,
                    stride: vb_state.array_stride,
                });
            }
            vertex_buffers.alloc().init(hal::pso::VertexBufferDesc {
                binding: i as u32,
                stride: vb_state.array_stride as u32,
                rate: match vb_state.step_mode {
                    InputStepMode::Vertex => hal::pso::VertexInputRate::Vertex,
                    InputStepMode::Instance => hal::pso::VertexInputRate::Instance(1),
                },
            });
            let desc_atts = &vb_state.attributes;
            for attribute in desc_atts.iter() {
                if attribute.offset >= 0x10000000 {
                    return Err(
                        pipeline::CreateRenderPipelineError::InvalidVertexAttributeOffset {
                            location: attribute.shader_location,
                            offset: attribute.offset,
                        },
                    );
                }

                if let wgt::VertexFormat::Float64
                | wgt::VertexFormat::Float64x2
                | wgt::VertexFormat::Float64x3
                | wgt::VertexFormat::Float64x4 = attribute.format
                {
                    if !self
                        .features
                        .contains(wgt::Features::VERTEX_ATTRIBUTE_64BIT)
                    {
                        return Err(pipeline::CreateRenderPipelineError::MissingFeature(
                            wgt::Features::VERTEX_ATTRIBUTE_64BIT,
                        ));
                    }
                }

                attributes.alloc().init(hal::pso::AttributeDesc {
                    location: attribute.shader_location,
                    binding: i as u32,
                    element: hal::pso::Element {
                        format: conv::map_vertex_format(attribute.format),
                        offset: attribute.offset as u32,
                    },
                });
                io.insert(
                    attribute.shader_location,
                    validation::NumericType::from_vertex_format(attribute.format),
                );
            }
        }

        if vertex_buffers.len() > self.limits.max_vertex_buffers as usize {
            return Err(pipeline::CreateRenderPipelineError::TooManyVertexBuffers {
                given: vertex_buffers.len() as u32,
                limit: self.limits.max_vertex_buffers,
            });
        }
        if attributes.len() > self.limits.max_vertex_attributes as usize {
            return Err(
                pipeline::CreateRenderPipelineError::TooManyVertexAttributes {
                    given: attributes.len() as u32,
                    limit: self.limits.max_vertex_attributes,
                },
            );
        }

        if desc.primitive.strip_index_format.is_some()
            && desc.primitive.topology != wgt::PrimitiveTopology::LineStrip
            && desc.primitive.topology != wgt::PrimitiveTopology::TriangleStrip
        {
            return Err(
                pipeline::CreateRenderPipelineError::StripIndexFormatForNonStripTopology {
                    strip_index_format: desc.primitive.strip_index_format,
                    topology: desc.primitive.topology,
                },
            );
        }

        let input_assembler = conv::map_primitive_state_to_input_assembler(&desc.primitive);

        let blender = hal::pso::BlendDesc {
            logic_op: None, // TODO
            targets: color_states
                .iter()
                .map(conv::map_color_target_state)
                .collect(),
        };
        let depth_stencil = depth_stencil_state
            .map(conv::map_depth_stencil_state)
            .unwrap_or_default();

        // TODO
        let baked_states = hal::pso::BakedStates {
            viewport: None,
            scissor: None,
            blend_color: None,
            depth_bounds: None,
        };

        if desc.primitive.clamp_depth && !self.features.contains(wgt::Features::DEPTH_CLAMPING) {
            return Err(pipeline::CreateRenderPipelineError::MissingFeature(
                wgt::Features::DEPTH_CLAMPING,
            ));
        }
        if desc.primitive.polygon_mode != wgt::PolygonMode::Fill
            && !self.features.contains(wgt::Features::NON_FILL_POLYGON_MODE)
        {
            return Err(pipeline::CreateRenderPipelineError::MissingFeature(
                wgt::Features::NON_FILL_POLYGON_MODE,
            ));
        }

        if desc.primitive.conservative
            && !self
                .features
                .contains(wgt::Features::CONSERVATIVE_RASTERIZATION)
        {
            return Err(pipeline::CreateRenderPipelineError::MissingFeature(
                wgt::Features::CONSERVATIVE_RASTERIZATION,
            ));
        }

        if desc.primitive.conservative && desc.primitive.polygon_mode != wgt::PolygonMode::Fill {
            return Err(
                pipeline::CreateRenderPipelineError::ConservativeRasterizationNonFillPolygonMode,
            );
        }

        if desc.layout.is_none() {
            for _ in 0..self.limits.max_bind_groups {
                derived_group_layouts.push(binding_model::BindEntryMap::default());
            }
        }

        let samples = {
            let sc = desc.multisample.count;
            if sc == 0 || sc > 32 || !conv::is_power_of_two(sc) {
                return Err(pipeline::CreateRenderPipelineError::InvalidSampleCount(sc));
            }
            sc as u8
        };
        let multisampling = if samples == 1 {
            None
        } else {
            Some(conv::map_multisample_state(&desc.multisample))
        };

        let rp_key = RenderPassKey {
            colors: color_states
                .iter()
                .map(|state| {
                    let at = hal::pass::Attachment {
                        format: Some(conv::map_texture_format(
                            state.format,
                            self.private_features,
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
                        self.private_features,
                    )),
                    samples,
                    ops: hal::pass::AttachmentOps::PRESERVE,
                    stencil_ops: hal::pass::AttachmentOps::PRESERVE,
                    layouts: hal::image::Layout::General..hal::image::Layout::General,
                };
                (at, hal::image::Layout::DepthStencilAttachmentOptimal)
            }),
        };

        let (shader_module_guard, _) = hub.shader_modules.read(&mut token);

        let vertex = {
            let stage = &desc.vertex.stage;
            let flag = wgt::ShaderStage::VERTEX;

            let shader_module = shader_module_guard.get(stage.module).map_err(|_| {
                pipeline::CreateRenderPipelineError::Stage {
                    flag,
                    error: validation::StageError::InvalidModule,
                }
            })?;

            if let Some(ref interface) = shader_module.interface {
                let provided_layouts = match desc.layout {
                    Some(pipeline_layout_id) => Some(Device::get_introspection_bind_group_layouts(
                        pipeline_layout_guard
                            .get(pipeline_layout_id)
                            .map_err(|_| pipeline::CreateRenderPipelineError::InvalidLayout)?,
                        &*bgl_guard,
                    )),
                    None => None,
                };

                io = interface
                    .check_stage(
                        provided_layouts.as_ref().map(|p| p.as_slice()),
                        &mut derived_group_layouts,
                        &stage.entry_point,
                        flag,
                        io,
                    )
                    .map_err(|error| pipeline::CreateRenderPipelineError::Stage { flag, error })?;
                validated_stages |= flag;
            }

            hal::pso::EntryPoint::<B> {
                entry: &stage.entry_point,
                module: &shader_module.raw,
                specialization: hal::pso::Specialization::EMPTY,
            }
        };

        let fragment = match desc.fragment {
            Some(ref fragment) => {
                let entry_point_name = &fragment.stage.entry_point;
                let flag = wgt::ShaderStage::FRAGMENT;

                let shader_module =
                    shader_module_guard
                        .get(fragment.stage.module)
                        .map_err(|_| pipeline::CreateRenderPipelineError::Stage {
                            flag,
                            error: validation::StageError::InvalidModule,
                        })?;

                let provided_layouts = match desc.layout {
                    Some(pipeline_layout_id) => Some(Device::get_introspection_bind_group_layouts(
                        pipeline_layout_guard
                            .get(pipeline_layout_id)
                            .map_err(|_| pipeline::CreateRenderPipelineError::InvalidLayout)?,
                        &*bgl_guard,
                    )),
                    None => None,
                };

                if validated_stages == wgt::ShaderStage::VERTEX {
                    if let Some(ref interface) = shader_module.interface {
                        io = interface
                            .check_stage(
                                provided_layouts.as_ref().map(|p| p.as_slice()),
                                &mut derived_group_layouts,
                                &entry_point_name,
                                flag,
                                io,
                            )
                            .map_err(|error| pipeline::CreateRenderPipelineError::Stage {
                                flag,
                                error,
                            })?;
                        validated_stages |= flag;
                    }
                }

                Some(hal::pso::EntryPoint::<B> {
                    entry: &entry_point_name,
                    module: &shader_module.raw,
                    specialization: hal::pso::Specialization::EMPTY,
                })
            }
            None => None,
        };

        if validated_stages.contains(wgt::ShaderStage::FRAGMENT) {
            for (i, state) in color_states.iter().enumerate() {
                match io.get(&(i as wgt::ShaderLocation)) {
                    Some(output) if validation::check_texture_format(state.format, output) => {}
                    Some(output) => {
                        log::warn!(
                            "Incompatible fragment output[{}] from shader: {:?}, expected {:?}",
                            i,
                            output,
                            state.format,
                        );
                        return Err(
                            pipeline::CreateRenderPipelineError::IncompatibleOutputFormat {
                                index: i as u8,
                            },
                        );
                    }
                    None if state.write_mask.is_empty() => {}
                    None => {
                        log::warn!("Missing fragment output[{}], expected {:?}", i, state,);
                        return Err(pipeline::CreateRenderPipelineError::MissingOutput {
                            index: i as u8,
                        });
                    }
                }
            }
        }
        let last_stage = match desc.fragment {
            Some(_) => wgt::ShaderStage::FRAGMENT,
            None => wgt::ShaderStage::VERTEX,
        };
        if desc.layout.is_none() && !validated_stages.contains(last_stage) {
            return Err(pipeline::ImplicitLayoutError::ReflectionError(last_stage).into());
        }

        let primitive_assembler = hal::pso::PrimitiveAssemblerDesc::Vertex {
            buffers: &vertex_buffers,
            attributes: &attributes,
            input_assembler,
            vertex,
            tessellation: None,
            geometry: None,
        };

        // TODO
        let flags = hal::pso::PipelineCreationFlags::empty();
        // TODO
        let parent = hal::pso::BasePipeline::None;

        let (pipeline_layout_id, derived_bind_group_count) = match desc.layout {
            Some(id) => (id, 0),
            None => self.derive_pipeline_layout(
                self_id,
                implicit_context,
                derived_group_layouts,
                &mut *bgl_guard,
                &mut *pipeline_layout_guard,
            )?,
        };
        let layout = pipeline_layout_guard
            .get(pipeline_layout_id)
            .map_err(|_| pipeline::CreateRenderPipelineError::InvalidLayout)?;

        let mut rp_lock = self.render_passes.lock();
        let pipeline_desc = hal::pso::GraphicsPipelineDesc {
            label: desc.label.as_ref().map(AsRef::as_ref),
            primitive_assembler,
            rasterizer,
            fragment,
            blender,
            depth_stencil,
            multisampling,
            baked_states,
            layout: &layout.raw,
            subpass: hal::pass::Subpass {
                index: 0,
                main_pass: match rp_lock.render_passes.entry(rp_key) {
                    Entry::Occupied(e) => e.into_mut(),
                    Entry::Vacant(e) => {
                        let pass = self
                            .create_compatible_render_pass(e.key())
                            .or(Err(DeviceError::OutOfMemory))?;
                        e.insert(pass)
                    }
                },
            },
            flags,
            parent,
        };
        // TODO: cache
        let raw =
            unsafe { self.raw.create_graphics_pipeline(&pipeline_desc, None) }.map_err(|err| {
                match err {
                    hal::pso::CreationError::OutOfMemory(_) => {
                        pipeline::CreateRenderPipelineError::Device(DeviceError::OutOfMemory)
                    }
                    hal::pso::CreationError::ShaderCreationError(stage, error) => {
                        pipeline::CreateRenderPipelineError::Internal {
                            stage: conv::map_hal_flags_to_shader_stage(stage),
                            error,
                        }
                    }
                    _ => {
                        log::error!("failed to create graphics pipeline: {}", err);
                        pipeline::CreateRenderPipelineError::Device(DeviceError::OutOfMemory)
                    }
                }
            })?;

        let pass_context = RenderPassContext {
            attachments: AttachmentData {
                colors: color_states.iter().map(|state| state.format).collect(),
                resolves: ArrayVec::new(),
                depth_stencil: depth_stencil_state.as_ref().map(|state| state.format),
            },
            sample_count: samples,
        };

        let mut flags = pipeline::PipelineFlags::empty();
        for state in color_states.iter() {
            if let Some(ref bs) = state.blend {
                if bs.color.uses_color() | bs.alpha.uses_color() {
                    flags |= pipeline::PipelineFlags::BLEND_COLOR;
                }
            }
        }
        if let Some(ds) = depth_stencil_state.as_ref() {
            if ds.stencil.is_enabled() && ds.stencil.needs_ref_value() {
                flags |= pipeline::PipelineFlags::STENCIL_REFERENCE;
            }
            if !ds.is_read_only() {
                flags |= pipeline::PipelineFlags::WRITES_DEPTH_STENCIL;
            }
        }

        let pipeline = pipeline::RenderPipeline {
            raw,
            layout_id: Stored {
                value: id::Valid(pipeline_layout_id),
                ref_count: layout.life_guard.add_ref(),
            },
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            pass_context,
            flags,
            strip_index_format: desc.primitive.strip_index_format,
            vertex_strides,
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
        };
        Ok((pipeline, derived_bind_group_count, pipeline_layout_id))
    }

    fn wait_for_submit(
        &self,
        submission_index: SubmissionIndex,
        token: &mut Token<Self>,
    ) -> Result<(), WaitIdleError> {
        if self.last_completed_submission_index() <= submission_index {
            log::info!("Waiting for submission {:?}", submission_index);
            self.lock_life(token)
                .triage_submissions(&self.raw, true)
                .map(|_| ())
        } else {
            Ok(())
        }
    }
}

impl<B: hal::Backend> Device<B> {
    pub(crate) fn destroy_bind_group(&self, bind_group: binding_model::BindGroup<B>) {
        self.desc_allocator
            .lock()
            .free(&self.raw, iter::once(bind_group.raw));
    }

    pub(crate) fn destroy_buffer(&self, buffer: resource::Buffer<B>) {
        if let Some((raw, memory)) = buffer.raw {
            unsafe {
                self.mem_allocator.lock().free(&self.raw, memory);
                self.raw.destroy_buffer(raw);
            }
        }
    }

    pub(crate) fn destroy_texture(&self, texture: resource::Texture<B>) {
        if let Some((raw, memory)) = texture.raw {
            unsafe {
                self.mem_allocator.lock().free(&self.raw, memory);
                self.raw.destroy_image(raw);
            }
        }
    }

    /// Wait for idle and remove resources that we can, before we die.
    pub(crate) fn prepare_to_die(&mut self) {
        let mut life_tracker = self.life_tracker.lock();
        if let Err(error) = life_tracker.triage_submissions(&self.raw, true) {
            log::error!("failed to triage submissions: {}", error);
        }
        life_tracker.cleanup(&self.raw, &self.mem_allocator, &self.desc_allocator);
    }

    pub(crate) fn dispose(self) {
        let mut desc_alloc = self.desc_allocator.into_inner();
        let mut mem_alloc = self.mem_allocator.into_inner();
        self.pending_writes
            .dispose(&self.raw, &self.cmd_allocator, &mut mem_alloc);
        self.cmd_allocator.destroy(&self.raw);
        unsafe {
            desc_alloc.cleanup(&self.raw);
            mem_alloc.clear(&self.raw);
            let rps = self.render_passes.into_inner();
            for (_, rp) in rps.render_passes {
                self.raw.destroy_render_pass(rp);
            }
            for (_, fbo) in rps.framebuffers {
                self.raw.destroy_framebuffer(fbo);
            }
        }
    }
}

impl<B: hal::Backend> crate::hub::Resource for Device<B> {
    const TYPE: &'static str = "Device";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}

#[derive(Clone, Debug, Error)]
#[error("device is invalid")]
pub struct InvalidDevice;

#[derive(Clone, Debug, Error)]
pub enum DeviceError {
    #[error("parent device is invalid")]
    Invalid,
    #[error("parent device is lost")]
    Lost,
    #[error("not enough memory left")]
    OutOfMemory,
}

impl From<hal::device::WaitError> for DeviceError {
    fn from(err: hal::device::WaitError) -> Self {
        match err {
            hal::device::WaitError::OutOfMemory(_) => Self::OutOfMemory,
            hal::device::WaitError::DeviceLost(_) => Self::Lost,
        }
    }
}

impl From<gpu_alloc::MapError> for DeviceError {
    fn from(err: gpu_alloc::MapError) -> Self {
        match err {
            gpu_alloc::MapError::OutOfDeviceMemory | gpu_alloc::MapError::OutOfHostMemory => {
                DeviceError::OutOfMemory
            }
            _ => panic!("failed to map buffer: {}", err),
        }
    }
}

impl DeviceError {
    fn from_bind(err: hal::device::BindError) -> Self {
        match err {
            hal::device::BindError::OutOfMemory(_) => Self::OutOfMemory,
            _ => panic!("failed to bind memory: {}", err),
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ImplicitPipelineContext {
    pub root_id: id::PipelineLayoutId,
    pub group_ids: ArrayVec<[id::BindGroupLayoutId; MAX_BIND_GROUPS]>,
}

pub struct ImplicitPipelineIds<'a, G: GlobalIdentityHandlerFactory> {
    pub root_id: Input<G, id::PipelineLayoutId>,
    pub group_ids: &'a [Input<G, id::BindGroupLayoutId>],
}

impl<G: GlobalIdentityHandlerFactory> ImplicitPipelineIds<'_, G> {
    fn prepare<B: hal::Backend>(self, hub: &Hub<B, G>) -> ImplicitPipelineContext {
        ImplicitPipelineContext {
            root_id: hub.pipeline_layouts.prepare(self.root_id).into_id(),
            group_ids: self
                .group_ids
                .iter()
                .map(|id_in| hub.bind_group_layouts.prepare(id_in.clone()).into_id())
                .collect(),
        }
    }
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn adapter_get_swap_chain_preferred_format<B: GfxBackend>(
        &self,
        adapter_id: id::AdapterId,
        surface_id: id::SurfaceId,
    ) -> Result<TextureFormat, instance::GetSwapChainPreferredFormatError> {
        profiling::scope!("Adapter::get_swap_chain_preferred_format");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
        let (adapter_guard, mut _token) = hub.adapters.read(&mut token);
        let adapter = adapter_guard
            .get(adapter_id)
            .map_err(|_| instance::GetSwapChainPreferredFormatError::InvalidAdapter)?;
        let surface = surface_guard
            .get_mut(surface_id)
            .map_err(|_| instance::GetSwapChainPreferredFormatError::InvalidSurface)?;

        adapter.get_swap_chain_preferred_format(surface)
    }

    pub fn device_features<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
    ) -> Result<wgt::Features, InvalidDevice> {
        profiling::scope!("Device::features");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;

        Ok(device.features)
    }

    pub fn device_limits<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
    ) -> Result<wgt::Limits, InvalidDevice> {
        profiling::scope!("Device::limits");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;

        Ok(device.limits.clone())
    }

    pub fn device_downlevel_properties<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
    ) -> Result<wgt::DownlevelProperties, InvalidDevice> {
        profiling::scope!("Device::downlevel_properties");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;

        Ok(device.downlevel)
    }

    pub fn device_create_buffer<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::BufferDescriptor,
        id_in: Input<G, id::BufferId>,
    ) -> (id::BufferId, Option<resource::CreateBufferError>) {
        profiling::scope!("Device::create_buffer");

        let hub = B::hub(self);
        let mut token = Token::root();
        let fid = hub.buffers.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                let mut desc = desc.clone();
                let mapped_at_creation = mem::replace(&mut desc.mapped_at_creation, false);
                if mapped_at_creation && !desc.usage.contains(wgt::BufferUsage::MAP_WRITE) {
                    desc.usage |= wgt::BufferUsage::COPY_DST;
                }
                trace
                    .lock()
                    .add(trace::Action::CreateBuffer(fid.id(), desc));
            }

            let mut buffer = match device.create_buffer(device_id, desc, false) {
                Ok(buffer) => buffer,
                Err(e) => break e,
            };
            let ref_count = buffer.life_guard.add_ref();

            let buffer_use = if !desc.mapped_at_creation {
                resource::BufferUse::EMPTY
            } else if desc.usage.contains(wgt::BufferUsage::MAP_WRITE) {
                // buffer is mappable, so we are just doing that at start
                let map_size = buffer.size;
                let ptr = match map_buffer(&device.raw, &mut buffer, 0, map_size, HostMap::Write) {
                    Ok(ptr) => ptr,
                    Err(e) => {
                        let (raw, memory) = buffer.raw.unwrap();
                        device.lock_life(&mut token).schedule_resource_destruction(
                            queue::TempResource::Buffer(raw),
                            memory,
                            !0,
                        );
                        break e.into();
                    }
                };
                buffer.map_state = resource::BufferMapState::Active {
                    ptr,
                    sub_range: hal::buffer::SubRange::WHOLE,
                    host: HostMap::Write,
                };
                resource::BufferUse::MAP_WRITE
            } else {
                // buffer needs staging area for initialization only
                let stage_desc = wgt::BufferDescriptor {
                    label: Some(Cow::Borrowed("<init_buffer>")),
                    size: desc.size,
                    usage: wgt::BufferUsage::MAP_WRITE | wgt::BufferUsage::COPY_SRC,
                    mapped_at_creation: false,
                };
                let mut stage = match device.create_buffer(device_id, &stage_desc, true) {
                    Ok(stage) => stage,
                    Err(e) => {
                        let (raw, memory) = buffer.raw.unwrap();
                        device.lock_life(&mut token).schedule_resource_destruction(
                            queue::TempResource::Buffer(raw),
                            memory,
                            !0,
                        );
                        break e;
                    }
                };
                let (stage_buffer, mut stage_memory) = stage.raw.unwrap();
                let ptr = match stage_memory.map(&device.raw, 0, stage.size) {
                    Ok(ptr) => ptr,
                    Err(e) => {
                        let (raw, memory) = buffer.raw.unwrap();
                        let mut life_lock = device.lock_life(&mut token);
                        life_lock.schedule_resource_destruction(
                            queue::TempResource::Buffer(raw),
                            memory,
                            !0,
                        );
                        life_lock.schedule_resource_destruction(
                            queue::TempResource::Buffer(stage_buffer),
                            stage_memory,
                            !0,
                        );
                        break e.into();
                    }
                };

                // Zero initialize memory and then mark both staging and buffer as initialized
                // (it's guaranteed that this is the case by the time the buffer is usable)
                unsafe { ptr::write_bytes(ptr.as_ptr(), 0, buffer.size as usize) };
                buffer.initialization_status.clear(0..buffer.size);
                stage.initialization_status.clear(0..buffer.size);

                buffer.map_state = resource::BufferMapState::Init {
                    ptr,
                    needs_flush: !stage_memory.is_coherent(),
                    stage_buffer,
                    stage_memory,
                };
                resource::BufferUse::COPY_DST
            };

            let id = fid.assign(buffer, &mut token);
            log::info!("Created buffer {:?} with {:?}", id, desc);

            device
                .trackers
                .lock()
                .buffers
                .init(id, ref_count, BufferState::with_usage(buffer_use))
                .unwrap();
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    #[cfg(feature = "replay")]
    pub fn device_wait_for_buffer<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        buffer_id: id::BufferId,
    ) -> Result<(), WaitIdleError> {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let last_submission = {
            let (buffer_guard, _) = hub.buffers.write(&mut token);
            match buffer_guard.get(buffer_id) {
                Ok(buffer) => buffer.life_guard.submission_index.load(Ordering::Acquire),
                Err(_) => return Ok(()),
            }
        };

        device_guard
            .get(device_id)
            .map_err(|_| DeviceError::Invalid)?
            .wait_for_submit(last_submission, &mut token)
    }

    pub fn device_set_buffer_sub_data<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        data: &[u8],
    ) -> Result<(), resource::BufferAccessError> {
        profiling::scope!("Device::set_buffer_sub_data");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut buffer_guard, _) = hub.buffers.write(&mut token);
        let device = device_guard
            .get(device_id)
            .map_err(|_| DeviceError::Invalid)?;
        let buffer = buffer_guard
            .get_mut(buffer_id)
            .map_err(|_| resource::BufferAccessError::Invalid)?;
        check_buffer_usage(buffer.usage, wgt::BufferUsage::MAP_WRITE)?;
        //assert!(buffer isn't used by the GPU);

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            let mut trace = trace.lock();
            let data_path = trace.make_binary("bin", data);
            trace.add(trace::Action::WriteBuffer {
                id: buffer_id,
                data: data_path,
                range: offset..offset + data.len() as BufferAddress,
                queued: false,
            });
        }

        buffer
            .raw
            .as_mut()
            .unwrap()
            .1
            .write_bytes(&device.raw, offset, data)?;

        Ok(())
    }

    pub fn device_get_buffer_sub_data<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        data: &mut [u8],
    ) -> Result<(), resource::BufferAccessError> {
        profiling::scope!("Device::get_buffer_sub_data");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut buffer_guard, _) = hub.buffers.write(&mut token);
        let device = device_guard
            .get(device_id)
            .map_err(|_| DeviceError::Invalid)?;
        let buffer = buffer_guard
            .get_mut(buffer_id)
            .map_err(|_| resource::BufferAccessError::Invalid)?;
        check_buffer_usage(buffer.usage, wgt::BufferUsage::MAP_READ)?;
        //assert!(buffer isn't used by the GPU);

        buffer
            .raw
            .as_mut()
            .unwrap()
            .1
            .read_bytes(&device.raw, offset, data)?;

        Ok(())
    }

    pub fn buffer_label<B: GfxBackend>(&self, id: id::BufferId) -> String {
        B::hub(self).buffers.label_for_resource(id)
    }

    pub fn buffer_destroy<B: GfxBackend>(
        &self,
        buffer_id: id::BufferId,
    ) -> Result<(), resource::DestroyError> {
        profiling::scope!("Buffer::destroy");

        let hub = B::hub(self);
        let mut token = Token::root();

        //TODO: lock pending writes separately, keep the device read-only
        let (mut device_guard, mut token) = hub.devices.write(&mut token);

        log::info!("Buffer {:?} is destroyed", buffer_id);
        let (mut buffer_guard, _) = hub.buffers.write(&mut token);
        let buffer = buffer_guard
            .get_mut(buffer_id)
            .map_err(|_| resource::DestroyError::Invalid)?;

        let device = &mut device_guard[buffer.device_id.value];

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace.lock().add(trace::Action::FreeBuffer(buffer_id));
        }

        let (raw, memory) = buffer
            .raw
            .take()
            .ok_or(resource::DestroyError::AlreadyDestroyed)?;
        let temp = queue::TempResource::Buffer(raw);

        if device.pending_writes.dst_buffers.contains(&buffer_id) {
            device.pending_writes.temp_resources.push((temp, memory));
        } else {
            let last_submit_index = buffer.life_guard.submission_index.load(Ordering::Acquire);
            drop(buffer_guard);
            device.lock_life(&mut token).schedule_resource_destruction(
                temp,
                memory,
                last_submit_index,
            );
        }

        Ok(())
    }

    pub fn buffer_drop<B: GfxBackend>(&self, buffer_id: id::BufferId, wait: bool) {
        profiling::scope!("Buffer::drop");

        let hub = B::hub(self);
        let mut token = Token::root();

        log::info!("Buffer {:?} is dropped", buffer_id);
        let (ref_count, last_submit_index, device_id) = {
            let (mut buffer_guard, _) = hub.buffers.write(&mut token);
            match buffer_guard.get_mut(buffer_id) {
                Ok(buffer) => {
                    let ref_count = buffer.life_guard.ref_count.take().unwrap();
                    let last_submit_index =
                        buffer.life_guard.submission_index.load(Ordering::Acquire);
                    (ref_count, last_submit_index, buffer.device_id.value)
                }
                Err(InvalidId) => {
                    hub.buffers.unregister_locked(buffer_id, &mut *buffer_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        {
            let mut life_lock = device.lock_life(&mut token);
            if device.pending_writes.dst_buffers.contains(&buffer_id) {
                life_lock.future_suspected_buffers.push(Stored {
                    value: id::Valid(buffer_id),
                    ref_count,
                });
            } else {
                drop(ref_count);
                life_lock
                    .suspected_resources
                    .buffers
                    .push(id::Valid(buffer_id));
            }
        }

        if wait {
            match device.wait_for_submit(last_submit_index, &mut token) {
                Ok(()) => (),
                Err(e) => log::error!("Failed to wait for buffer {:?}: {:?}", buffer_id, e),
            }
        }
    }

    pub fn device_create_texture<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::TextureDescriptor,
        id_in: Input<G, id::TextureId>,
    ) -> (id::TextureId, Option<resource::CreateTextureError>) {
        profiling::scope!("Device::create_texture");

        let hub = B::hub(self);
        let mut token = Token::root();
        let fid = hub.textures.prepare(id_in);

        let (adapter_guard, mut token) = hub.adapters.read(&mut token);
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreateTexture(fid.id(), desc.clone()));
            }

            let adapter = &adapter_guard[device.adapter_id.value];
            let texture = match device.create_texture(device_id, adapter, desc) {
                Ok(texture) => texture,
                Err(error) => break error,
            };
            let num_levels = texture.full_range.levels.end;
            let num_layers = texture.full_range.layers.end;
            let ref_count = texture.life_guard.add_ref();

            let id = fid.assign(texture, &mut token);
            log::info!("Created texture {:?} with {:?}", id, desc);

            device
                .trackers
                .lock()
                .textures
                .init(id, ref_count, TextureState::new(num_levels, num_layers))
                .unwrap();
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn texture_label<B: GfxBackend>(&self, id: id::TextureId) -> String {
        B::hub(self).textures.label_for_resource(id)
    }

    pub fn texture_destroy<B: GfxBackend>(
        &self,
        texture_id: id::TextureId,
    ) -> Result<(), resource::DestroyError> {
        profiling::scope!("Texture::destroy");

        let hub = B::hub(self);
        let mut token = Token::root();

        //TODO: lock pending writes separately, keep the device read-only
        let (mut device_guard, mut token) = hub.devices.write(&mut token);

        log::info!("Buffer {:?} is destroyed", texture_id);
        let (mut texture_guard, _) = hub.textures.write(&mut token);
        let texture = texture_guard
            .get_mut(texture_id)
            .map_err(|_| resource::DestroyError::Invalid)?;

        let device = &mut device_guard[texture.device_id.value];

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace.lock().add(trace::Action::FreeTexture(texture_id));
        }

        let (raw, memory) = texture
            .raw
            .take()
            .ok_or(resource::DestroyError::AlreadyDestroyed)?;
        let temp = queue::TempResource::Image(raw);

        if device.pending_writes.dst_textures.contains(&texture_id) {
            device.pending_writes.temp_resources.push((temp, memory));
        } else {
            let last_submit_index = texture.life_guard.submission_index.load(Ordering::Acquire);
            drop(texture_guard);
            device.lock_life(&mut token).schedule_resource_destruction(
                temp,
                memory,
                last_submit_index,
            );
        }

        Ok(())
    }

    pub fn texture_drop<B: GfxBackend>(&self, texture_id: id::TextureId, wait: bool) {
        profiling::scope!("Texture::drop");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (ref_count, last_submit_index, device_id) = {
            let (mut texture_guard, _) = hub.textures.write(&mut token);
            match texture_guard.get_mut(texture_id) {
                Ok(texture) => {
                    let ref_count = texture.life_guard.ref_count.take().unwrap();
                    let last_submit_index =
                        texture.life_guard.submission_index.load(Ordering::Acquire);
                    (ref_count, last_submit_index, texture.device_id.value)
                }
                Err(InvalidId) => {
                    hub.textures
                        .unregister_locked(texture_id, &mut *texture_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        {
            let mut life_lock = device.lock_life(&mut token);
            if device.pending_writes.dst_textures.contains(&texture_id) {
                life_lock.future_suspected_textures.push(Stored {
                    value: id::Valid(texture_id),
                    ref_count,
                });
            } else {
                drop(ref_count);
                life_lock
                    .suspected_resources
                    .textures
                    .push(id::Valid(texture_id));
            }
        }

        if wait {
            match device.wait_for_submit(last_submit_index, &mut token) {
                Ok(()) => (),
                Err(e) => log::error!("Failed to wait for texture {:?}: {:?}", texture_id, e),
            }
        }
    }

    pub fn texture_create_view<B: GfxBackend>(
        &self,
        texture_id: id::TextureId,
        desc: &resource::TextureViewDescriptor,
        id_in: Input<G, id::TextureViewId>,
    ) -> (id::TextureViewId, Option<resource::CreateTextureViewError>) {
        profiling::scope!("Texture::create_view");

        let hub = B::hub(self);
        let mut token = Token::root();
        let fid = hub.texture_views.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (texture_guard, mut token) = hub.textures.read(&mut token);
        let error = loop {
            let texture = match texture_guard.get(texture_id) {
                Ok(texture) => texture,
                Err(_) => break resource::CreateTextureViewError::InvalidTexture,
            };
            let device = &device_guard[texture.device_id.value];
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::CreateTextureView {
                    id: fid.id(),
                    parent_id: texture_id,
                    desc: desc.clone(),
                });
            }

            let view = match device.create_texture_view(texture, texture_id, desc) {
                Ok(view) => view,
                Err(e) => break e,
            };
            let ref_count = view.life_guard.add_ref();
            let id = fid.assign(view, &mut token);

            device
                .trackers
                .lock()
                .views
                .init(id, ref_count, PhantomData)
                .unwrap();
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn texture_view_label<B: GfxBackend>(&self, id: id::TextureViewId) -> String {
        B::hub(self).texture_views.label_for_resource(id)
    }

    pub fn texture_view_drop<B: GfxBackend>(
        &self,
        texture_view_id: id::TextureViewId,
        wait: bool,
    ) -> Result<(), resource::TextureViewDestroyError> {
        profiling::scope!("TextureView::drop");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (last_submit_index, device_id) = {
            let (texture_guard, mut token) = hub.textures.read(&mut token);
            let (mut texture_view_guard, _) = hub.texture_views.write(&mut token);

            match texture_view_guard.get_mut(texture_view_id) {
                Ok(view) => {
                    let _ref_count = view.life_guard.ref_count.take();
                    let last_submit_index =
                        view.life_guard.submission_index.load(Ordering::Acquire);
                    let device_id = match view.inner {
                        resource::TextureViewInner::Native { ref source_id, .. } => {
                            texture_guard[source_id.value].device_id.value
                        }
                        resource::TextureViewInner::SwapChain { .. } => {
                            return Err(resource::TextureViewDestroyError::SwapChainImage)
                        }
                    };
                    (last_submit_index, device_id)
                }
                Err(InvalidId) => {
                    hub.texture_views
                        .unregister_locked(texture_view_id, &mut *texture_view_guard);
                    return Ok(());
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];
        device
            .lock_life(&mut token)
            .suspected_resources
            .texture_views
            .push(id::Valid(texture_view_id));

        if wait {
            match device.wait_for_submit(last_submit_index, &mut token) {
                Ok(()) => (),
                Err(e) => log::error!(
                    "Failed to wait for texture view {:?}: {:?}",
                    texture_view_id,
                    e
                ),
            }
        }
        Ok(())
    }

    pub fn device_create_sampler<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::SamplerDescriptor,
        id_in: Input<G, id::SamplerId>,
    ) -> (id::SamplerId, Option<resource::CreateSamplerError>) {
        profiling::scope!("Device::create_sampler");

        let hub = B::hub(self);
        let mut token = Token::root();
        let fid = hub.samplers.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreateSampler(fid.id(), desc.clone()));
            }

            let sampler = match device.create_sampler(device_id, desc) {
                Ok(sampler) => sampler,
                Err(e) => break e,
            };
            let ref_count = sampler.life_guard.add_ref();
            let id = fid.assign(sampler, &mut token);

            device
                .trackers
                .lock()
                .samplers
                .init(id, ref_count, PhantomData)
                .unwrap();
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn sampler_label<B: GfxBackend>(&self, id: id::SamplerId) -> String {
        B::hub(self).samplers.label_for_resource(id)
    }

    pub fn sampler_drop<B: GfxBackend>(&self, sampler_id: id::SamplerId) {
        profiling::scope!("Sampler::drop");

        let hub = B::hub(self);
        let mut token = Token::root();

        let device_id = {
            let (mut sampler_guard, _) = hub.samplers.write(&mut token);
            match sampler_guard.get_mut(sampler_id) {
                Ok(sampler) => {
                    sampler.life_guard.ref_count.take();
                    sampler.device_id.value
                }
                Err(InvalidId) => {
                    hub.samplers
                        .unregister_locked(sampler_id, &mut *sampler_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources
            .samplers
            .push(id::Valid(sampler_id));
    }

    pub fn device_create_bind_group_layout<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &binding_model::BindGroupLayoutDescriptor,
        id_in: Input<G, id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::CreateBindGroupLayoutError>,
    ) {
        profiling::scope!("Device::create_bind_group_layout");

        let mut token = Token::root();
        let hub = B::hub(self);
        let fid = hub.bind_group_layouts.prepare(id_in);

        let error = 'outer: loop {
            let (device_guard, mut token) = hub.devices.read(&mut token);
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreateBindGroupLayout(fid.id(), desc.clone()));
            }

            let mut entry_map = FastHashMap::default();
            for entry in desc.entries.iter() {
                if entry_map.insert(entry.binding, entry.clone()).is_some() {
                    break 'outer binding_model::CreateBindGroupLayoutError::ConflictBinding(
                        entry.binding,
                    );
                }
            }

            // If there is an equivalent BGL, just bump the refcount and return it.
            // This is only applicable for identity filters that are generating new IDs,
            // so their inputs are `PhantomData` of size 0.
            if mem::size_of::<Input<G, id::BindGroupLayoutId>>() == 0 {
                let (bgl_guard, _) = hub.bind_group_layouts.read(&mut token);
                if let Some(id) =
                    Device::deduplicate_bind_group_layout(device_id, &entry_map, &*bgl_guard)
                {
                    return (id, None);
                }
            }

            let layout = match device.create_bind_group_layout(
                device_id,
                desc.label.as_ref().map(|cow| cow.as_ref()),
                entry_map,
            ) {
                Ok(layout) => layout,
                Err(e) => break e,
            };

            let id = fid.assign(layout, &mut token);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn bind_group_layout_label<B: GfxBackend>(&self, id: id::BindGroupLayoutId) -> String {
        B::hub(self).bind_group_layouts.label_for_resource(id)
    }

    pub fn bind_group_layout_drop<B: GfxBackend>(
        &self,
        bind_group_layout_id: id::BindGroupLayoutId,
    ) {
        profiling::scope!("BindGroupLayout::drop");

        let hub = B::hub(self);
        let mut token = Token::root();
        let device_id = {
            let (mut bind_group_layout_guard, _) = hub.bind_group_layouts.write(&mut token);
            match bind_group_layout_guard.get_mut(bind_group_layout_id) {
                Ok(layout) => layout.device_id.value,
                Err(InvalidId) => {
                    hub.bind_group_layouts
                        .unregister_locked(bind_group_layout_id, &mut *bind_group_layout_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources
            .bind_group_layouts
            .push(id::Valid(bind_group_layout_id));
    }

    pub fn device_create_pipeline_layout<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &binding_model::PipelineLayoutDescriptor,
        id_in: Input<G, id::PipelineLayoutId>,
    ) -> (
        id::PipelineLayoutId,
        Option<binding_model::CreatePipelineLayoutError>,
    ) {
        profiling::scope!("Device::create_pipeline_layout");

        let hub = B::hub(self);
        let mut token = Token::root();
        let fid = hub.pipeline_layouts.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreatePipelineLayout(fid.id(), desc.clone()));
            }

            let layout = {
                let (bgl_guard, _) = hub.bind_group_layouts.read(&mut token);
                match device.create_pipeline_layout(device_id, desc, &*bgl_guard) {
                    Ok(layout) => layout,
                    Err(e) => break e,
                }
            };

            let id = fid.assign(layout, &mut token);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn pipeline_layout_label<B: GfxBackend>(&self, id: id::PipelineLayoutId) -> String {
        B::hub(self).pipeline_layouts.label_for_resource(id)
    }

    pub fn pipeline_layout_drop<B: GfxBackend>(&self, pipeline_layout_id: id::PipelineLayoutId) {
        profiling::scope!("PipelineLayout::drop");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_id, ref_count) = {
            let (mut pipeline_layout_guard, _) = hub.pipeline_layouts.write(&mut token);
            match pipeline_layout_guard.get_mut(pipeline_layout_id) {
                Ok(layout) => (
                    layout.device_id.value,
                    layout.life_guard.ref_count.take().unwrap(),
                ),
                Err(InvalidId) => {
                    hub.pipeline_layouts
                        .unregister_locked(pipeline_layout_id, &mut *pipeline_layout_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources
            .pipeline_layouts
            .push(Stored {
                value: id::Valid(pipeline_layout_id),
                ref_count,
            });
    }

    pub fn device_create_bind_group<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &binding_model::BindGroupDescriptor,
        id_in: Input<G, id::BindGroupId>,
    ) -> (id::BindGroupId, Option<binding_model::CreateBindGroupError>) {
        profiling::scope!("Device::create_bind_group");

        let hub = B::hub(self);
        let mut token = Token::root();
        let fid = hub.bind_groups.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (bind_group_layout_guard, mut token) = hub.bind_group_layouts.read(&mut token);

        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreateBindGroup(fid.id(), desc.clone()));
            }

            let bind_group_layout = match bind_group_layout_guard.get(desc.layout) {
                Ok(layout) => layout,
                Err(_) => break binding_model::CreateBindGroupError::InvalidLayout,
            };
            let bind_group = match device.create_bind_group(
                device_id,
                bind_group_layout,
                desc,
                &hub,
                &mut token,
            ) {
                Ok(bind_group) => bind_group,
                Err(e) => break e,
            };
            let ref_count = bind_group.life_guard.add_ref();

            let id = fid.assign(bind_group, &mut token);
            log::debug!(
                "Bind group {:?} {:#?}",
                id,
                hub.bind_groups.read(&mut token).0[id].used
            );

            device
                .trackers
                .lock()
                .bind_groups
                .init(id, ref_count, PhantomData)
                .unwrap();
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn bind_group_label<B: GfxBackend>(&self, id: id::BindGroupId) -> String {
        B::hub(self).bind_groups.label_for_resource(id)
    }

    pub fn bind_group_drop<B: GfxBackend>(&self, bind_group_id: id::BindGroupId) {
        profiling::scope!("BindGroup::drop");

        let hub = B::hub(self);
        let mut token = Token::root();

        let device_id = {
            let (mut bind_group_guard, _) = hub.bind_groups.write(&mut token);
            match bind_group_guard.get_mut(bind_group_id) {
                Ok(bind_group) => {
                    bind_group.life_guard.ref_count.take();
                    bind_group.device_id.value
                }
                Err(InvalidId) => {
                    hub.bind_groups
                        .unregister_locked(bind_group_id, &mut *bind_group_guard);
                    return;
                }
            }
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources
            .bind_groups
            .push(id::Valid(bind_group_id));
    }

    pub fn device_create_shader_module<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &pipeline::ShaderModuleDescriptor,
        source: pipeline::ShaderModuleSource,
        id_in: Input<G, id::ShaderModuleId>,
    ) -> (
        id::ShaderModuleId,
        Option<pipeline::CreateShaderModuleError>,
    ) {
        profiling::scope!("Device::create_shader_module");

        let hub = B::hub(self);
        let mut token = Token::root();
        let fid = hub.shader_modules.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                let mut trace = trace.lock();
                let data = match source {
                    pipeline::ShaderModuleSource::SpirV(ref spv) => {
                        trace.make_binary("spv", unsafe {
                            std::slice::from_raw_parts(spv.as_ptr() as *const u8, spv.len() * 4)
                        })
                    }
                    pipeline::ShaderModuleSource::Wgsl(ref code) => {
                        trace.make_binary("wgsl", code.as_bytes())
                    }
                    pipeline::ShaderModuleSource::Naga(_) => {
                        // we don't want to enable Naga serialization just for this alone
                        trace.make_binary("ron", &[])
                    }
                };
                trace.add(trace::Action::CreateShaderModule {
                    id: fid.id(),
                    desc: desc.clone(),
                    data,
                });
            };

            let shader = match device.create_shader_module(device_id, desc, source) {
                Ok(shader) => shader,
                Err(e) => break e,
            };
            let id = fid.assign(shader, &mut token);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn shader_module_label<B: GfxBackend>(&self, id: id::ShaderModuleId) -> String {
        B::hub(self).shader_modules.label_for_resource(id)
    }

    pub fn shader_module_drop<B: GfxBackend>(&self, shader_module_id: id::ShaderModuleId) {
        profiling::scope!("ShaderModule::drop");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (module, _) = hub.shader_modules.unregister(shader_module_id, &mut token);
        if let Some(module) = module {
            let device = &device_guard[module.device_id.value];
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::DestroyShaderModule(shader_module_id));
            }
            unsafe {
                device.raw.destroy_shader_module(module.raw);
            }
        }
    }

    pub fn device_create_command_encoder<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &wgt::CommandEncoderDescriptor<Label>,
        id_in: Input<G, id::CommandEncoderId>,
    ) -> (id::CommandEncoderId, Option<command::CommandAllocatorError>) {
        profiling::scope!("Device::create_command_encoder");

        let hub = B::hub(self);
        let mut token = Token::root();
        let fid = hub.command_buffers.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };

            let dev_stored = Stored {
                value: id::Valid(device_id),
                ref_count: device.life_guard.add_ref(),
            };

            let mut command_buffer = match device.cmd_allocator.allocate(
                dev_stored,
                &device.raw,
                device.limits.clone(),
                device.downlevel,
                device.private_features,
                &desc.label,
                #[cfg(feature = "trace")]
                device.trace.is_some(),
            ) {
                Ok(cmd_buf) => cmd_buf,
                Err(e) => break e,
            };

            let mut raw = command_buffer.raw.first_mut().unwrap();
            unsafe {
                if let Some(ref label) = desc.label {
                    device.raw.set_command_buffer_name(&mut raw, label);
                }
                raw.begin_primary(hal::command::CommandBufferFlags::ONE_TIME_SUBMIT);
            }

            let id = fid.assign(command_buffer, &mut token);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn command_buffer_label<B: GfxBackend>(&self, id: id::CommandBufferId) -> String {
        B::hub(self).command_buffers.label_for_resource(id)
    }

    pub fn command_encoder_drop<B: GfxBackend>(&self, command_encoder_id: id::CommandEncoderId) {
        profiling::scope!("CommandEncoder::drop");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let (cmdbuf, _) = hub
            .command_buffers
            .unregister(command_encoder_id, &mut token);
        if let Some(cmdbuf) = cmdbuf {
            let device = &mut device_guard[cmdbuf.device_id.value];
            device.untrack::<G>(&hub, &cmdbuf.trackers, &mut token);
            device.cmd_allocator.discard(cmdbuf);
        }
    }

    pub fn command_buffer_drop<B: GfxBackend>(&self, command_buffer_id: id::CommandBufferId) {
        profiling::scope!("CommandBuffer::drop");
        self.command_encoder_drop::<B>(command_buffer_id)
    }

    pub fn device_create_render_bundle_encoder(
        &self,
        device_id: id::DeviceId,
        desc: &command::RenderBundleEncoderDescriptor,
    ) -> (
        id::RenderBundleEncoderId,
        Option<command::CreateRenderBundleError>,
    ) {
        profiling::scope!("Device::create_render_bundle_encoder");
        let (encoder, error) = match command::RenderBundleEncoder::new(desc, device_id, None) {
            Ok(encoder) => (encoder, None),
            Err(e) => (command::RenderBundleEncoder::dummy(device_id), Some(e)),
        };
        (Box::into_raw(Box::new(encoder)), error)
    }

    pub fn render_bundle_encoder_finish<B: GfxBackend>(
        &self,
        bundle_encoder: command::RenderBundleEncoder,
        desc: &command::RenderBundleDescriptor,
        id_in: Input<G, id::RenderBundleId>,
    ) -> (id::RenderBundleId, Option<command::RenderBundleError>) {
        profiling::scope!("RenderBundleEncoder::finish");

        let hub = B::hub(self);
        let mut token = Token::root();
        let fid = hub.render_bundles.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(bundle_encoder.parent()) {
                Ok(device) => device,
                Err(_) => break command::RenderBundleError::INVALID_DEVICE,
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::CreateRenderBundle {
                    id: fid.id(),
                    desc: trace::new_render_bundle_encoder_descriptor(
                        desc.label.clone(),
                        &bundle_encoder.context,
                    ),
                    base: bundle_encoder.to_base_pass(),
                });
            }

            let render_bundle = match bundle_encoder.finish(desc, device, &hub, &mut token) {
                Ok(bundle) => bundle,
                Err(e) => break e,
            };

            log::debug!("Render bundle {:#?}", render_bundle.used);
            let ref_count = render_bundle.life_guard.add_ref();
            let id = fid.assign(render_bundle, &mut token);

            device
                .trackers
                .lock()
                .bundles
                .init(id, ref_count, PhantomData)
                .unwrap();
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn render_bundle_label<B: GfxBackend>(&self, id: id::RenderBundleId) -> String {
        B::hub(self).render_bundles.label_for_resource(id)
    }

    pub fn render_bundle_drop<B: GfxBackend>(&self, render_bundle_id: id::RenderBundleId) {
        profiling::scope!("RenderBundle::drop");
        let hub = B::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device_id = {
            let (mut bundle_guard, _) = hub.render_bundles.write(&mut token);
            match bundle_guard.get_mut(render_bundle_id) {
                Ok(bundle) => {
                    bundle.life_guard.ref_count.take();
                    bundle.device_id.value
                }
                Err(InvalidId) => {
                    hub.render_bundles
                        .unregister_locked(render_bundle_id, &mut *bundle_guard);
                    return;
                }
            }
        };

        device_guard[device_id]
            .lock_life(&mut token)
            .suspected_resources
            .render_bundles
            .push(id::Valid(render_bundle_id));
    }

    pub fn device_create_query_set<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &wgt::QuerySetDescriptor,
        id_in: Input<G, id::QuerySetId>,
    ) -> (id::QuerySetId, Option<resource::CreateQuerySetError>) {
        profiling::scope!("Device::create_query_set");

        let hub = B::hub(self);
        let mut token = Token::root();
        let fid = hub.query_sets.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::CreateQuerySet {
                    id: fid.id(),
                    desc: desc.clone(),
                });
            }

            match desc.ty {
                wgt::QueryType::Timestamp => {
                    if !device.features.contains(wgt::Features::TIMESTAMP_QUERY) {
                        break resource::CreateQuerySetError::MissingFeature(
                            wgt::Features::TIMESTAMP_QUERY,
                        );
                    }
                }
                wgt::QueryType::PipelineStatistics(..) => {
                    if !device
                        .features
                        .contains(wgt::Features::PIPELINE_STATISTICS_QUERY)
                    {
                        break resource::CreateQuerySetError::MissingFeature(
                            wgt::Features::PIPELINE_STATISTICS_QUERY,
                        );
                    }
                }
            }

            if desc.count == 0 {
                break resource::CreateQuerySetError::ZeroCount;
            }

            if desc.count >= wgt::QUERY_SET_MAX_QUERIES {
                break resource::CreateQuerySetError::TooManyQueries {
                    count: desc.count,
                    maximum: wgt::QUERY_SET_MAX_QUERIES,
                };
            }

            let query_set = {
                let (hal_type, elements) = conv::map_query_type(&desc.ty);

                resource::QuerySet {
                    raw: unsafe { device.raw.create_query_pool(hal_type, desc.count).unwrap() },
                    device_id: Stored {
                        value: id::Valid(device_id),
                        ref_count: device.life_guard.add_ref(),
                    },
                    life_guard: LifeGuard::new(""),
                    desc: desc.clone(),
                    elements,
                }
            };

            let ref_count = query_set.life_guard.add_ref();
            let id = fid.assign(query_set, &mut token);

            device
                .trackers
                .lock()
                .query_sets
                .init(id, ref_count, PhantomData)
                .unwrap();

            return (id.0, None);
        };

        let id = fid.assign_error("", &mut token);
        (id, Some(error))
    }

    pub fn query_set_drop<B: GfxBackend>(&self, query_set_id: id::QuerySetId) {
        profiling::scope!("QuerySet::drop");

        let hub = B::hub(self);
        let mut token = Token::root();

        let device_id = {
            let (mut query_set_guard, _) = hub.query_sets.write(&mut token);
            let query_set = query_set_guard.get_mut(query_set_id).unwrap();
            query_set.life_guard.ref_count.take();
            query_set.device_id.value
        };

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = &device_guard[device_id];

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            trace
                .lock()
                .add(trace::Action::DestroyQuerySet(query_set_id));
        }

        device
            .lock_life(&mut token)
            .suspected_resources
            .query_sets
            .push(id::Valid(query_set_id));
    }

    pub fn device_create_render_pipeline<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        desc: &pipeline::RenderPipelineDescriptor,
        id_in: Input<G, id::RenderPipelineId>,
        implicit_pipeline_ids: Option<ImplicitPipelineIds<G>>,
    ) -> (
        id::RenderPipelineId,
        pipeline::ImplicitBindGroupCount,
        Option<pipeline::CreateRenderPipelineError>,
    ) {
        profiling::scope!("Device::create_render_pipeline");

        let hub = B::hub(self);
        let mut token = Token::root();

        let fid = hub.render_pipelines.prepare(id_in);
        let implicit_context = implicit_pipeline_ids.map(|ipi| ipi.prepare(&hub));

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::CreateRenderPipeline {
                    id: fid.id(),
                    desc: desc.clone(),
                    implicit_context: implicit_context.clone(),
                });
            }

            let (pipeline, derived_bind_group_count, _layout_id) = match device
                .create_render_pipeline(device_id, desc, implicit_context, &hub, &mut token)
            {
                Ok(pair) => pair,
                Err(e) => break e,
            };

            let id = fid.assign(pipeline, &mut token);
            return (id.0, derived_bind_group_count, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, 0, Some(error))
    }

    /// Get an ID of one of the bind group layouts. The ID adds a refcount,
    /// which needs to be released by calling `bind_group_layout_drop`.
    pub fn render_pipeline_get_bind_group_layout<B: GfxBackend>(
        &self,
        pipeline_id: id::RenderPipelineId,
        index: u32,
        id_in: Input<G, id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::GetBindGroupLayoutError>,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);

        let error = loop {
            let (bgl_guard, mut token) = hub.bind_group_layouts.read(&mut token);
            let (_, mut token) = hub.bind_groups.read(&mut token);
            let (pipeline_guard, _) = hub.render_pipelines.read(&mut token);

            let pipeline = match pipeline_guard.get(pipeline_id) {
                Ok(pipeline) => pipeline,
                Err(_) => break binding_model::GetBindGroupLayoutError::InvalidPipeline,
            };
            let id = match pipeline_layout_guard[pipeline.layout_id.value]
                .bind_group_layout_ids
                .get(index as usize)
            {
                Some(id) => id,
                None => break binding_model::GetBindGroupLayoutError::InvalidGroupIndex(index),
            };

            bgl_guard[*id].multi_ref_count.inc();
            return (id.0, None);
        };

        let id = hub
            .bind_group_layouts
            .prepare(id_in)
            .assign_error("<derived>", &mut token);
        (id, Some(error))
    }

    pub fn render_pipeline_label<B: GfxBackend>(&self, id: id::RenderPipelineId) -> String {
        B::hub(self).render_pipelines.label_for_resource(id)
    }

    pub fn render_pipeline_drop<B: GfxBackend>(&self, render_pipeline_id: id::RenderPipelineId) {
        profiling::scope!("RenderPipeline::drop");
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);

        let (device_id, layout_id) = {
            let (mut pipeline_guard, _) = hub.render_pipelines.write(&mut token);
            match pipeline_guard.get_mut(render_pipeline_id) {
                Ok(pipeline) => {
                    pipeline.life_guard.ref_count.take();
                    (pipeline.device_id.value, pipeline.layout_id.clone())
                }
                Err(InvalidId) => {
                    hub.render_pipelines
                        .unregister_locked(render_pipeline_id, &mut *pipeline_guard);
                    return;
                }
            }
        };

        let mut life_lock = device_guard[device_id].lock_life(&mut token);
        life_lock
            .suspected_resources
            .render_pipelines
            .push(id::Valid(render_pipeline_id));
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
        implicit_pipeline_ids: Option<ImplicitPipelineIds<G>>,
    ) -> (
        id::ComputePipelineId,
        pipeline::ImplicitBindGroupCount,
        Option<pipeline::CreateComputePipelineError>,
    ) {
        profiling::scope!("Device::create_compute_pipeline");

        let hub = B::hub(self);
        let mut token = Token::root();

        let fid = hub.compute_pipelines.prepare(id_in);
        let implicit_context = implicit_pipeline_ids.map(|ipi| ipi.prepare(&hub));

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::CreateComputePipeline {
                    id: fid.id(),
                    desc: desc.clone(),
                    implicit_context: implicit_context.clone(),
                });
            }

            let (pipeline, derived_bind_group_count, _layout_id) = match device
                .create_compute_pipeline(device_id, desc, implicit_context, &hub, &mut token)
            {
                Ok(pair) => pair,
                Err(e) => break e,
            };

            let id = fid.assign(pipeline, &mut token);
            return (id.0, derived_bind_group_count, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, 0, Some(error))
    }

    /// Get an ID of one of the bind group layouts. The ID adds a refcount,
    /// which needs to be released by calling `bind_group_layout_drop`.
    pub fn compute_pipeline_get_bind_group_layout<B: GfxBackend>(
        &self,
        pipeline_id: id::ComputePipelineId,
        index: u32,
        id_in: Input<G, id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::GetBindGroupLayoutError>,
    ) {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (pipeline_layout_guard, mut token) = hub.pipeline_layouts.read(&mut token);

        let error = loop {
            let (bgl_guard, mut token) = hub.bind_group_layouts.read(&mut token);
            let (_, mut token) = hub.bind_groups.read(&mut token);
            let (pipeline_guard, _) = hub.compute_pipelines.read(&mut token);

            let pipeline = match pipeline_guard.get(pipeline_id) {
                Ok(pipeline) => pipeline,
                Err(_) => break binding_model::GetBindGroupLayoutError::InvalidPipeline,
            };
            let id = match pipeline_layout_guard[pipeline.layout_id.value]
                .bind_group_layout_ids
                .get(index as usize)
            {
                Some(id) => id,
                None => break binding_model::GetBindGroupLayoutError::InvalidGroupIndex(index),
            };

            bgl_guard[*id].multi_ref_count.inc();
            return (id.0, None);
        };

        let id = hub
            .bind_group_layouts
            .prepare(id_in)
            .assign_error("<derived>", &mut token);
        (id, Some(error))
    }

    pub fn compute_pipeline_label<B: GfxBackend>(&self, id: id::ComputePipelineId) -> String {
        B::hub(self).compute_pipelines.label_for_resource(id)
    }

    pub fn compute_pipeline_drop<B: GfxBackend>(&self, compute_pipeline_id: id::ComputePipelineId) {
        profiling::scope!("ComputePipeline::drop");
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);

        let (device_id, layout_id) = {
            let (mut pipeline_guard, _) = hub.compute_pipelines.write(&mut token);
            match pipeline_guard.get_mut(compute_pipeline_id) {
                Ok(pipeline) => {
                    pipeline.life_guard.ref_count.take();
                    (pipeline.device_id.value, pipeline.layout_id.clone())
                }
                Err(InvalidId) => {
                    hub.compute_pipelines
                        .unregister_locked(compute_pipeline_id, &mut *pipeline_guard);
                    return;
                }
            }
        };

        let mut life_lock = device_guard[device_id].lock_life(&mut token);
        life_lock
            .suspected_resources
            .compute_pipelines
            .push(id::Valid(compute_pipeline_id));
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
    ) -> (id::SwapChainId, Option<swap_chain::CreateSwapChainError>) {
        profiling::scope!("Device::create_swap_chain");

        fn validate_swap_chain_descriptor(
            config: &mut hal::window::SwapchainConfig,
            caps: &hal::window::SurfaceCapabilities,
        ) -> Result<(), swap_chain::CreateSwapChainError> {
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
            if width == 0 || height == 0 {
                return Err(swap_chain::CreateSwapChainError::ZeroArea);
            }
            Ok(())
        }

        log::info!("creating swap chain {:?}", desc);
        let sc_id = surface_id.to_swap_chain_id(B::VARIANT);
        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
        let (adapter_guard, mut token) = hub.adapters.read(&mut token);
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut swap_chain_guard, _) = hub.swap_chains.write(&mut token);

        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreateSwapChain(sc_id, desc.clone()));
            }

            let surface = match surface_guard.get_mut(surface_id) {
                Ok(surface) => surface,
                Err(_) => break swap_chain::CreateSwapChainError::InvalidSurface,
            };

            let (caps, formats) = {
                let surface = B::get_surface_mut(surface);
                let adapter = &adapter_guard[device.adapter_id.value];
                let queue_family = &adapter.raw.queue_families[0];
                if !surface.supports_queue_family(queue_family) {
                    break swap_chain::CreateSwapChainError::UnsupportedQueueFamily;
                }
                let formats = surface.supported_formats(&adapter.raw.physical_device);
                let caps = surface.capabilities(&adapter.raw.physical_device);
                (caps, formats)
            };

            let num_frames = swap_chain::DESIRED_NUM_FRAMES
                .max(*caps.image_count.start())
                .min(*caps.image_count.end());
            let mut config = swap_chain::swap_chain_descriptor_to_hal(
                &desc,
                num_frames,
                device.private_features,
            );
            if let Some(formats) = formats {
                if !formats.contains(&config.format) {
                    break swap_chain::CreateSwapChainError::UnsupportedFormat {
                        requested: config.format,
                        available: formats,
                    };
                }
            }
            if let Err(error) = validate_swap_chain_descriptor(&mut config, &caps) {
                break error;
            }
            let framebuffer_attachment = config.framebuffer_attachment();

            match unsafe { B::get_surface_mut(surface).configure_swapchain(&device.raw, config) } {
                Ok(()) => (),
                Err(hal::window::SwapchainError::OutOfMemory(_)) => {
                    break DeviceError::OutOfMemory.into()
                }
                Err(hal::window::SwapchainError::DeviceLost(_)) => break DeviceError::Lost.into(),
                Err(err) => panic!("failed to configure swap chain on creation: {}", err),
            }

            if let Some(sc) = swap_chain_guard.try_remove(sc_id) {
                if sc.acquired_view_id.is_some() {
                    break swap_chain::CreateSwapChainError::SwapChainOutputExists;
                }
                unsafe {
                    device.raw.destroy_semaphore(sc.semaphore);
                }
            }

            let swap_chain = swap_chain::SwapChain {
                life_guard: LifeGuard::new("<SwapChain>"),
                device_id: Stored {
                    value: id::Valid(device_id),
                    ref_count: device.life_guard.add_ref(),
                },
                desc: desc.clone(),
                num_frames,
                semaphore: match device.raw.create_semaphore() {
                    Ok(sem) => sem,
                    Err(_) => break DeviceError::OutOfMemory.into(),
                },
                acquired_view_id: None,
                active_submission_index: 0,
                framebuffer_attachment,
            };
            swap_chain_guard.insert(sc_id, swap_chain);

            return (sc_id, None);
        };

        swap_chain_guard.insert_error(sc_id, "");
        (sc_id, Some(error))
    }

    #[cfg(feature = "replay")]
    /// Only triange suspected resource IDs. This helps us to avoid ID collisions
    /// upon creating new resources when re-playing a trace.
    pub fn device_maintain_ids<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
    ) -> Result<(), InvalidDevice> {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;
        device.lock_life(&mut token).triage_suspected(
            &hub,
            &device.trackers,
            #[cfg(feature = "trace")]
            None,
            &mut token,
        );
        Ok(())
    }

    pub fn device_poll<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
        force_wait: bool,
    ) -> Result<(), WaitIdleError> {
        profiling::scope!("Device::poll");

        let hub = B::hub(self);
        let mut token = Token::root();
        let callbacks = {
            let (device_guard, mut token) = hub.devices.read(&mut token);
            device_guard
                .get(device_id)
                .map_err(|_| DeviceError::Invalid)?
                .maintain(&hub, force_wait, &mut token)?
        };
        fire_map_callbacks(callbacks);
        Ok(())
    }

    fn poll_devices<B: GfxBackend>(
        &self,
        force_wait: bool,
        callbacks: &mut Vec<BufferMapPendingCallback>,
    ) -> Result<(), WaitIdleError> {
        profiling::scope!("Device::poll_devices");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        for (_, device) in device_guard.iter(B::VARIANT) {
            let cbs = device.maintain(&hub, force_wait, &mut token)?;
            callbacks.extend(cbs);
        }
        Ok(())
    }

    pub fn poll_all_devices(&self, force_wait: bool) -> Result<(), WaitIdleError> {
        use crate::backend;
        let mut callbacks = Vec::new();

        #[cfg(vulkan)]
        {
            self.poll_devices::<backend::Vulkan>(force_wait, &mut callbacks)?;
        }
        #[cfg(metal)]
        {
            self.poll_devices::<backend::Metal>(force_wait, &mut callbacks)?;
        }
        #[cfg(dx12)]
        {
            self.poll_devices::<backend::Dx12>(force_wait, &mut callbacks)?;
        }
        #[cfg(dx11)]
        {
            self.poll_devices::<backend::Dx11>(force_wait, &mut callbacks)?;
        }

        fire_map_callbacks(callbacks);

        Ok(())
    }

    pub fn device_label<B: GfxBackend>(&self, id: id::DeviceId) -> String {
        B::hub(self).devices.label_for_resource(id)
    }

    pub fn device_drop<B: GfxBackend>(&self, device_id: id::DeviceId) {
        profiling::scope!("Device::drop");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (device, _) = hub.devices.unregister(device_id, &mut token);
        if let Some(mut device) = device {
            device.prepare_to_die();

            // Adapter is only referenced by the device and itself.
            // This isn't a robust way to destroy them, we should find a better one.
            if device.adapter_id.ref_count.load() == 1 {
                let (_adapter, _) = hub
                    .adapters
                    .unregister(device.adapter_id.value.0, &mut token);
            }

            device.dispose();
        }
    }

    pub fn buffer_map_async<B: GfxBackend>(
        &self,
        buffer_id: id::BufferId,
        range: Range<BufferAddress>,
        op: resource::BufferMapOperation,
    ) -> Result<(), resource::BufferAccessError> {
        profiling::scope!("Device::buffer_map_async");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (pub_usage, internal_use) = match op.host {
            HostMap::Read => (wgt::BufferUsage::MAP_READ, resource::BufferUse::MAP_READ),
            HostMap::Write => (wgt::BufferUsage::MAP_WRITE, resource::BufferUse::MAP_WRITE),
        };

        if range.start % wgt::COPY_BUFFER_ALIGNMENT != 0
            || range.end % wgt::COPY_BUFFER_ALIGNMENT != 0
        {
            return Err(resource::BufferAccessError::UnalignedRange);
        }

        let (device_id, ref_count) = {
            let (mut buffer_guard, _) = hub.buffers.write(&mut token);
            let buffer = buffer_guard
                .get_mut(buffer_id)
                .map_err(|_| resource::BufferAccessError::Invalid)?;

            check_buffer_usage(buffer.usage, pub_usage)?;
            buffer.map_state = match buffer.map_state {
                resource::BufferMapState::Init { .. } | resource::BufferMapState::Active { .. } => {
                    return Err(resource::BufferAccessError::AlreadyMapped);
                }
                resource::BufferMapState::Waiting(_) => {
                    op.call_error();
                    return Ok(());
                }
                resource::BufferMapState::Idle => {
                    resource::BufferMapState::Waiting(resource::BufferPendingMapping {
                        range,
                        op,
                        parent_ref_count: buffer.life_guard.add_ref(),
                    })
                }
            };
            log::debug!("Buffer {:?} map state -> Waiting", buffer_id);

            (buffer.device_id.value, buffer.life_guard.add_ref())
        };

        let device = &device_guard[device_id];
        device.trackers.lock().buffers.change_replace(
            id::Valid(buffer_id),
            &ref_count,
            (),
            internal_use,
        );

        device
            .lock_life(&mut token)
            .map(id::Valid(buffer_id), ref_count);

        Ok(())
    }

    pub fn buffer_get_mapped_range<B: GfxBackend>(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferSize>,
    ) -> Result<(*mut u8, u64), resource::BufferAccessError> {
        profiling::scope!("Device::buffer_get_mapped_range");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (buffer_guard, _) = hub.buffers.read(&mut token);
        let buffer = buffer_guard
            .get(buffer_id)
            .map_err(|_| resource::BufferAccessError::Invalid)?;

        let range_size = if let Some(size) = size {
            size.into()
        } else if offset > buffer.size {
            0
        } else {
            buffer.size - offset
        };

        if offset % 8 != 0 {
            return Err(resource::BufferAccessError::UnalignedOffset { offset });
        }
        if range_size % 4 != 0 {
            return Err(resource::BufferAccessError::UnalignedRangeSize { range_size });
        }

        match buffer.map_state {
            resource::BufferMapState::Init { ptr, .. } => {
                // offset (u64) can not be < 0, so no need to validate the lower bound
                if offset + range_size > buffer.size {
                    return Err(resource::BufferAccessError::OutOfBoundsOverrun {
                        index: offset + range_size - 1,
                        max: buffer.size,
                    });
                }
                unsafe { Ok((ptr.as_ptr().offset(offset as isize), range_size)) }
            }
            resource::BufferMapState::Active {
                ptr, ref sub_range, ..
            } => {
                if offset < sub_range.offset {
                    return Err(resource::BufferAccessError::OutOfBoundsUnderrun {
                        index: offset,
                        min: sub_range.offset,
                    });
                }
                let range_end_offset = sub_range
                    .size
                    .map(|size| size + sub_range.offset)
                    .unwrap_or(buffer.size);
                if offset + range_size > range_end_offset {
                    return Err(resource::BufferAccessError::OutOfBoundsOverrun {
                        index: offset + range_size - 1,
                        max: range_end_offset,
                    });
                }
                unsafe { Ok((ptr.as_ptr().offset(offset as isize), range_size)) }
            }
            resource::BufferMapState::Idle | resource::BufferMapState::Waiting(_) => {
                Err(resource::BufferAccessError::NotMapped)
            }
        }
    }

    fn buffer_unmap_inner<B: GfxBackend>(
        &self,
        buffer_id: id::BufferId,
    ) -> Result<Option<BufferMapPendingCallback>, resource::BufferAccessError> {
        profiling::scope!("Device::buffer_unmap");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let (mut buffer_guard, _) = hub.buffers.write(&mut token);
        let buffer = buffer_guard
            .get_mut(buffer_id)
            .map_err(|_| resource::BufferAccessError::Invalid)?;
        let device = &mut device_guard[buffer.device_id.value];

        log::debug!("Buffer {:?} map state -> Idle", buffer_id);
        match mem::replace(&mut buffer.map_state, resource::BufferMapState::Idle) {
            resource::BufferMapState::Init {
                ptr,
                stage_buffer,
                stage_memory,
                needs_flush,
            } => {
                #[cfg(feature = "trace")]
                if let Some(ref trace) = device.trace {
                    let mut trace = trace.lock();
                    let data = trace.make_binary("bin", unsafe {
                        std::slice::from_raw_parts(ptr.as_ptr(), buffer.size as usize)
                    });
                    trace.add(trace::Action::WriteBuffer {
                        id: buffer_id,
                        data,
                        range: 0..buffer.size,
                        queued: true,
                    });
                }
                let _ = ptr;

                if needs_flush {
                    stage_memory.flush_range(&device.raw, 0, None)?;
                }

                let &(ref buf_raw, _) = buffer
                    .raw
                    .as_ref()
                    .ok_or(resource::BufferAccessError::Destroyed)?;

                buffer.life_guard.use_at(device.active_submission_index + 1);
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
                    target: buf_raw,
                    range: hal::buffer::SubRange::WHOLE,
                    families: None,
                };
                unsafe {
                    let cmdbuf = device.borrow_pending_writes();
                    cmdbuf.pipeline_barrier(
                        hal::pso::PipelineStage::HOST..hal::pso::PipelineStage::TRANSFER,
                        hal::memory::Dependencies::empty(),
                        iter::once(transition_src).chain(iter::once(transition_dst)),
                    );
                    if buffer.size > 0 {
                        cmdbuf.copy_buffer(&stage_buffer, buf_raw, iter::once(region));
                    }
                }
                device
                    .pending_writes
                    .consume_temp(queue::TempResource::Buffer(stage_buffer), stage_memory);
                device.pending_writes.dst_buffers.insert(buffer_id);
            }
            resource::BufferMapState::Idle => {
                return Err(resource::BufferAccessError::NotMapped);
            }
            resource::BufferMapState::Waiting(pending) => {
                return Ok(Some((pending.op, resource::BufferMapAsyncStatus::Aborted)));
            }
            resource::BufferMapState::Active {
                ptr,
                sub_range,
                host,
            } => {
                if host == HostMap::Write {
                    #[cfg(feature = "trace")]
                    if let Some(ref trace) = device.trace {
                        let mut trace = trace.lock();
                        let size = sub_range.size_to(buffer.size);
                        let data = trace.make_binary("bin", unsafe {
                            std::slice::from_raw_parts(ptr.as_ptr(), size as usize)
                        });
                        trace.add(trace::Action::WriteBuffer {
                            id: buffer_id,
                            data,
                            range: sub_range.offset..sub_range.offset + size,
                            queued: false,
                        });
                    }
                    let _ = (ptr, sub_range);
                }
                unmap_buffer(&device.raw, buffer)?;
            }
        }
        Ok(None)
    }

    pub fn buffer_unmap<B: GfxBackend>(
        &self,
        buffer_id: id::BufferId,
    ) -> Result<(), resource::BufferAccessError> {
        self.buffer_unmap_inner::<B>(buffer_id)
            //Note: outside inner function so no locks are held when calling the callback
            .map(|pending_callback| fire_map_callbacks(pending_callback.into_iter()))
    }

    pub fn start_capture<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
    ) -> Result<(), InvalidDevice> {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;
        device.raw.start_capture();
        Ok(())
    }

    pub fn stop_capture<B: GfxBackend>(
        &self,
        device_id: id::DeviceId,
    ) -> Result<(), InvalidDevice> {
        let hub = B::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;
        device.raw.stop_capture();
        Ok(())
    }
}
