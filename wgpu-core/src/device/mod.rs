use crate::{
    binding_model, command, conv,
    device::life::WaitIdleError,
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Hub, Input, InvalidId, Storage, Token},
    id,
    init_tracker::{BufferInitTracker, BufferInitTrackerAction, MemoryInitKind},
    instance, pipeline, present, resource,
    track::{BufferState, TextureSelector, TextureState, TrackerSet, UsageConflict},
    validation::{self, check_buffer_usage, check_texture_usage},
    FastHashMap, Label, LabelHelpers as _, LifeGuard, MultiRefCount, Stored, SubmissionIndex,
    DOWNLEVEL_ERROR_MESSAGE,
};

use arrayvec::ArrayVec;
use copyless::VecHelper as _;
use hal::{CommandEncoder as _, Device as _};
use parking_lot::{Mutex, MutexGuard};
use smallvec::SmallVec;
use thiserror::Error;
use wgt::{BufferAddress, TextureFormat, TextureViewDimension};

use std::{borrow::Cow, iter, marker::PhantomData, mem, ops::Range, ptr, sync::atomic::Ordering};

mod life;
pub mod queue;
#[cfg(any(feature = "trace", feature = "replay"))]
pub mod trace;

pub const SHADER_STAGE_COUNT: usize = 3;
const CLEANUP_WAIT_MS: u32 = 5000;

const IMPLICIT_FAILURE: &str = "failed implicit";
const EP_FAILURE: &str = "EP is invalid";

pub type DeviceDescriptor<'a> = wgt::DeviceDescriptor<Label<'a>>;

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
    pub colors: ArrayVec<T, { hal::MAX_COLOR_TARGETS }>,
    pub resolves: ArrayVec<T, { hal::MAX_COLOR_TARGETS }>,
    pub depth_stencil: Option<T>,
}
impl<T: PartialEq> Eq for AttachmentData<T> {}
impl<T> AttachmentData<T> {
    pub(crate) fn map<U, F: Fn(&T) -> U>(&self, fun: F) -> AttachmentData<U> {
        AttachmentData {
            colors: self.colors.iter().map(&fun).collect(),
            resolves: self.resolves.iter().map(&fun).collect(),
            depth_stencil: self.depth_stencil.as_ref().map(&fun),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "serial-pass", derive(serde::Deserialize, serde::Serialize))]
pub(crate) struct RenderPassContext {
    pub attachments: AttachmentData<TextureFormat>,
    pub sample_count: u32,
}
#[derive(Clone, Debug, Error)]
pub enum RenderPassCompatibilityError {
    #[error("Incompatible color attachment: {0:?} != {1:?}")]
    IncompatibleColorAttachment(
        ArrayVec<TextureFormat, { hal::MAX_COLOR_TARGETS }>,
        ArrayVec<TextureFormat, { hal::MAX_COLOR_TARGETS }>,
    ),
    #[error("Incompatible depth-stencil attachment: {0:?} != {1:?}")]
    IncompatibleDepthStencilAttachment(Option<TextureFormat>, Option<TextureFormat>),
    #[error("Incompatible sample count: {0:?} != {1:?}")]
    IncompatibleSampleCount(u32, u32),
}

impl RenderPassContext {
    // Assumed the renderpass only contains one subpass
    pub(crate) fn check_compatible(
        &self,
        other: &Self,
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

pub type BufferMapPendingClosure = (resource::BufferMapOperation, resource::BufferMapAsyncStatus);

#[derive(Default)]
pub struct UserClosures {
    pub mappings: Vec<BufferMapPendingClosure>,
    pub submissions: SmallVec<[queue::SubmittedWorkDoneClosure; 1]>,
}

impl UserClosures {
    fn extend(&mut self, other: Self) {
        self.mappings.extend(other.mappings);
        self.submissions.extend(other.submissions);
    }

    unsafe fn fire(self) {
        //Note: this logic is specifically moved out of `handle_mapping()` in order to
        // have nothing locked by the time we execute users callback code.
        for (operation, status) in self.mappings {
            (operation.callback)(status, operation.user_data);
        }
        for closure in self.submissions {
            (closure.callback)(closure.user_data);
        }
    }
}

fn map_buffer<A: hal::Api>(
    raw: &A::Device,
    buffer: &mut resource::Buffer<A>,
    offset: BufferAddress,
    size: BufferAddress,
    kind: HostMap,
) -> Result<ptr::NonNull<u8>, resource::BufferAccessError> {
    let mapping = unsafe {
        raw.map_buffer(buffer.raw.as_ref().unwrap(), offset..offset + size)
            .map_err(DeviceError::from)?
    };

    buffer.sync_mapped_writes = match kind {
        HostMap::Read if !mapping.is_coherent => unsafe {
            raw.invalidate_mapped_ranges(
                buffer.raw.as_ref().unwrap(),
                iter::once(offset..offset + size),
            );
            None
        },
        HostMap::Write if !mapping.is_coherent => Some(offset..offset + size),
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
    let zero_init_needs_flush_now = mapping.is_coherent && buffer.sync_mapped_writes.is_none(); // No need to flush if it is flushed later anyways.
    for uninitialized_range in buffer.initialization_status.drain(offset..(size + offset)) {
        let num_bytes = uninitialized_range.end - uninitialized_range.start;
        unsafe {
            ptr::write_bytes(
                mapping
                    .ptr
                    .as_ptr()
                    .offset(uninitialized_range.start as isize),
                0,
                num_bytes as usize,
            )
        };
        if zero_init_needs_flush_now {
            unsafe {
                raw.flush_mapped_ranges(
                    buffer.raw.as_ref().unwrap(),
                    iter::once(uninitialized_range.start..uninitialized_range.start + num_bytes),
                )
            };
        }
    }

    Ok(mapping.ptr)
}

struct CommandAllocator<A: hal::Api> {
    free_encoders: Vec<A::CommandEncoder>,
}

impl<A: hal::Api> CommandAllocator<A> {
    fn acquire_encoder(
        &mut self,
        device: &A::Device,
        queue: &A::Queue,
    ) -> Result<A::CommandEncoder, hal::DeviceError> {
        match self.free_encoders.pop() {
            Some(encoder) => Ok(encoder),
            None => unsafe {
                let hal_desc = hal::CommandEncoderDescriptor { label: None, queue };
                device.create_command_encoder(&hal_desc)
            },
        }
    }

    fn release_encoder(&mut self, encoder: A::CommandEncoder) {
        self.free_encoders.push(encoder);
    }

    fn dispose(self, device: &A::Device) {
        log::info!("Destroying {} command encoders", self.free_encoders.len());
        for cmd_encoder in self.free_encoders {
            unsafe {
                device.destroy_command_encoder(cmd_encoder);
            }
        }
    }
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
pub struct Device<A: hal::Api> {
    pub(crate) raw: A::Device,
    pub(crate) adapter_id: Stored<id::AdapterId>,
    pub(crate) queue: A::Queue,
    //pub(crate) cmd_allocator: command::CommandAllocator<A>,
    //mem_allocator: Mutex<alloc::MemoryAllocator<A>>,
    //desc_allocator: Mutex<descriptor::DescriptorAllocator<A>>,
    //Note: The submission index here corresponds to the last submission that is done.
    pub(crate) life_guard: LifeGuard,
    command_allocator: Mutex<CommandAllocator<A>>,
    pub(crate) active_submission_index: SubmissionIndex,
    fence: A::Fence,
    /// Has to be locked temporarily only (locked last)
    pub(crate) trackers: Mutex<TrackerSet>,
    // Life tracker should be locked right after the device and before anything else.
    life_tracker: Mutex<life::LifetimeTracker<A>>,
    temp_suspected: life::SuspectedResources,
    pub(crate) alignments: hal::Alignments,
    pub(crate) limits: wgt::Limits,
    pub(crate) features: wgt::Features,
    pub(crate) downlevel: wgt::DownlevelCapabilities,
    //TODO: move this behind another mutex. This would allow several methods to switch
    // to borrow Device immutably, such as `write_buffer`, `write_texture`, and `buffer_unmap`.
    pending_writes: queue::PendingWrites<A>,
    #[cfg(feature = "trace")]
    pub(crate) trace: Option<Mutex<trace::Trace>>,
}

#[derive(Clone, Debug, Error)]
pub enum CreateDeviceError {
    #[error("not enough memory left")]
    OutOfMemory,
}

impl<A: hal::Api> Device<A> {
    pub(crate) fn require_features(&self, feature: wgt::Features) -> Result<(), MissingFeatures> {
        if self.features.contains(feature) {
            Ok(())
        } else {
            Err(MissingFeatures(feature))
        }
    }

    pub(crate) fn require_downlevel_flags(
        &self,
        flags: wgt::DownlevelFlags,
    ) -> Result<(), MissingDownlevelFlags> {
        if self.downlevel.flags.contains(flags) {
            Ok(())
        } else {
            Err(MissingDownlevelFlags(flags))
        }
    }
}

impl<A: HalApi> Device<A> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        open: hal::OpenDevice<A>,
        adapter_id: Stored<id::AdapterId>,
        alignments: hal::Alignments,
        downlevel: wgt::DownlevelCapabilities,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
    ) -> Result<Self, CreateDeviceError> {
        #[cfg(not(feature = "trace"))]
        if let Some(_) = trace_path {
            log::error!("Feature 'trace' is not enabled");
        }
        let fence =
            unsafe { open.device.create_fence() }.map_err(|_| CreateDeviceError::OutOfMemory)?;

        let mut com_alloc = CommandAllocator {
            free_encoders: Vec::new(),
        };
        let pending_encoder = com_alloc
            .acquire_encoder(&open.device, &open.queue)
            .map_err(|_| CreateDeviceError::OutOfMemory)?;
        let pending_writes = queue::PendingWrites::new(pending_encoder);

        Ok(Self {
            raw: open.device,
            adapter_id,
            queue: open.queue,
            life_guard: LifeGuard::new("<device>"),
            command_allocator: Mutex::new(com_alloc),
            active_submission_index: 0,
            fence,
            trackers: Mutex::new(TrackerSet::new(A::VARIANT)),
            life_tracker: Mutex::new(life::LifetimeTracker::new()),
            temp_suspected: life::SuspectedResources::default(),
            #[cfg(feature = "trace")]
            trace: trace_path.and_then(|path| match trace::Trace::new(path) {
                Ok(mut trace) => {
                    trace.add(trace::Action::Init {
                        desc: desc.clone(),
                        backend: A::VARIANT,
                    });
                    Some(Mutex::new(trace))
                }
                Err(e) => {
                    log::error!("Unable to start a trace in '{:?}': {:?}", path, e);
                    None
                }
            }),
            alignments,
            limits: desc.limits.clone(),
            features: desc.features,
            downlevel,
            pending_writes,
        })
    }

    fn lock_life<'this, 'token: 'this>(
        &'this self,
        //TODO: fix this - the token has to be borrowed for the lock
        _token: &mut Token<'token, Self>,
    ) -> MutexGuard<'this, life::LifetimeTracker<A>> {
        self.life_tracker.lock()
    }

    fn maintain<'this, 'token: 'this, G: GlobalIdentityHandlerFactory>(
        &'this self,
        hub: &Hub<A, G>,
        force_wait: bool,
        token: &mut Token<'token, Self>,
    ) -> Result<UserClosures, WaitIdleError> {
        profiling::scope!("maintain", "Device");
        let mut life_tracker = self.lock_life(token);

        life_tracker
            .suspected_resources
            .extend(&self.temp_suspected);

        life_tracker.triage_suspected(
            hub,
            &self.trackers,
            #[cfg(feature = "trace")]
            self.trace.as_ref(),
            token,
        );
        life_tracker.triage_mapped(hub, token);

        let last_done_index = if force_wait {
            let current_index = self.active_submission_index;
            unsafe {
                self.raw
                    .wait(&self.fence, current_index, CLEANUP_WAIT_MS)
                    .map_err(DeviceError::from)?
            };
            current_index
        } else {
            unsafe {
                self.raw
                    .get_fence_value(&self.fence)
                    .map_err(DeviceError::from)?
            }
        };

        let submission_closures =
            life_tracker.triage_submissions(last_done_index, &self.command_allocator);
        let mapping_closures = life_tracker.handle_mapping(hub, &self.raw, &self.trackers, token);
        life_tracker.cleanup(&self.raw);

        Ok(UserClosures {
            mappings: mapping_closures,
            submissions: submission_closures,
        })
    }

    fn untrack<'this, 'token: 'this, G: GlobalIdentityHandlerFactory>(
        &'this mut self,
        hub: &Hub<A, G>,
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

        self.temp_suspected.clear();
    }

    fn create_buffer(
        &self,
        self_id: id::DeviceId,
        desc: &resource::BufferDescriptor,
        transient: bool,
    ) -> Result<resource::Buffer<A>, resource::CreateBufferError> {
        debug_assert_eq!(self_id.backend(), A::VARIANT);
        let mut usage = conv::map_buffer_usage(desc.usage);

        if desc.usage.is_empty() {
            return Err(resource::CreateBufferError::EmptyUsage);
        }

        if desc.mapped_at_creation {
            if desc.size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
                return Err(resource::CreateBufferError::UnalignedSize);
            }
            if !desc.usage.contains(wgt::BufferUsages::MAP_WRITE) {
                // we are going to be copying into it, internally
                usage |= hal::BufferUses::COPY_DST;
            }
        } else {
            // We are required to zero out (initialize) all memory.
            // This is done on demand using fill_buffer which requires write transfer usage!
            usage |= hal::BufferUses::COPY_DST;
        }

        let actual_size = if desc.size == 0 {
            wgt::COPY_BUFFER_ALIGNMENT
        } else if desc.usage.contains(wgt::BufferUsages::VERTEX) {
            // Bumping the size by 1 so that we can bind an empty range at the end of the buffer.
            desc.size + 1
        } else {
            desc.size
        };

        let mut memory_flags = hal::MemoryFlags::empty();
        memory_flags.set(hal::MemoryFlags::TRANSIENT, transient);

        let hal_desc = hal::BufferDescriptor {
            label: desc.label.borrow_option(),
            size: actual_size,
            usage,
            memory_flags,
        };
        let buffer = unsafe { self.raw.create_buffer(&hal_desc) }.map_err(DeviceError::from)?;

        Ok(resource::Buffer {
            raw: Some(buffer),
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            usage: desc.usage,
            size: desc.size,
            initialization_status: BufferInitTracker::new(desc.size),
            sync_mapped_writes: None,
            map_state: resource::BufferMapState::Idle,
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
        })
    }

    fn create_texture_from_hal(
        &self,
        hal_texture: A::Texture,
        self_id: id::DeviceId,
        adapter: &crate::instance::Adapter<A>,
        desc: &resource::TextureDescriptor,
    ) -> Result<resource::Texture<A>, resource::CreateTextureError> {
        debug_assert_eq!(self_id.backend(), A::VARIANT);

        let hal_usage = conv::map_texture_usage(desc.usage, desc.format.into());

        let format_features = self
            .describe_format_features(adapter, desc.format)
            .map_err(|error| resource::CreateTextureError::MissingFeatures(desc.format, error))?;

        // Ensure `D24Plus` textures cannot be copied
        match desc.format {
            TextureFormat::Depth24Plus | TextureFormat::Depth24PlusStencil8 => {
                if desc
                    .usage
                    .intersects(wgt::TextureUsages::COPY_SRC | wgt::TextureUsages::COPY_DST)
                {
                    return Err(resource::CreateTextureError::CannotCopyD24Plus);
                }
            }
            _ => {}
        }

        if desc.usage.is_empty() {
            return Err(resource::CreateTextureError::EmptyUsage);
        }

        let missing_allowed_usages = desc.usage - format_features.allowed_usages;
        if !missing_allowed_usages.is_empty() {
            return Err(resource::CreateTextureError::InvalidUsages(
                missing_allowed_usages,
                desc.format,
            ));
        }

        conv::check_texture_dimension_size(
            desc.dimension,
            desc.size,
            desc.sample_count,
            &self.limits,
        )?;

        let mips = desc.mip_level_count;
        if mips == 0 || mips > hal::MAX_MIP_LEVELS || mips > desc.size.max_mips() {
            return Err(resource::CreateTextureError::InvalidMipLevelCount(mips));
        }

        Ok(resource::Texture {
            inner: resource::TextureInner::Native {
                raw: Some(hal_texture),
            },
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            desc: desc.map_label(|_| ()),
            hal_usage,
            format_features,
            full_range: TextureSelector {
                levels: 0..desc.mip_level_count,
                layers: 0..desc.array_layer_count(),
            },
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
        })
    }

    fn create_texture(
        &self,
        self_id: id::DeviceId,
        adapter: &crate::instance::Adapter<A>,
        desc: &resource::TextureDescriptor,
    ) -> Result<resource::Texture<A>, resource::CreateTextureError> {
        let hal_usage = conv::map_texture_usage(desc.usage, desc.format.into());
        let hal_desc = hal::TextureDescriptor {
            label: desc.label.borrow_option(),
            size: desc.size,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            dimension: desc.dimension,
            format: desc.format,
            usage: hal_usage,
            memory_flags: hal::MemoryFlags::empty(),
        };
        let raw = unsafe {
            self.raw
                .create_texture(&hal_desc)
                .map_err(DeviceError::from)?
        };

        self.create_texture_from_hal(raw, self_id, adapter, desc)
    }

    fn create_texture_view(
        &self,
        texture: &resource::Texture<A>,
        texture_id: id::TextureId,
        desc: &resource::TextureViewDescriptor,
    ) -> Result<resource::TextureView<A>, resource::CreateTextureViewError> {
        let texture_raw = texture
            .inner
            .as_raw()
            .ok_or(resource::CreateTextureViewError::InvalidTexture)?;

        let view_dim = match desc.dimension {
            Some(dim) => {
                if texture.desc.dimension != dim.compatible_texture_dimension() {
                    return Err(
                        resource::CreateTextureViewError::InvalidTextureViewDimension {
                            view: dim,
                            texture: texture.desc.dimension,
                        },
                    );
                }
                dim
            }
            None => match texture.desc.dimension {
                wgt::TextureDimension::D1 => wgt::TextureViewDimension::D1,
                wgt::TextureDimension::D2
                    if texture.desc.size.depth_or_array_layers > 1
                        && desc.range.array_layer_count.is_none() =>
                {
                    wgt::TextureViewDimension::D2Array
                }
                wgt::TextureDimension::D2 => wgt::TextureViewDimension::D2,
                wgt::TextureDimension::D3 => wgt::TextureViewDimension::D3,
            },
        };

        let required_level_count =
            desc.range.base_mip_level + desc.range.mip_level_count.map_or(1, |count| count.get());
        let required_layer_count = match desc.range.array_layer_count {
            Some(count) => desc.range.base_array_layer + count.get(),
            None => texture.desc.array_layer_count(),
        };
        let level_end = texture.full_range.levels.end;
        let layer_end = texture.full_range.layers.end;
        if required_level_count > level_end {
            return Err(resource::CreateTextureViewError::TooManyMipLevels {
                requested: required_level_count,
                total: level_end,
            });
        }
        if required_layer_count > layer_end {
            return Err(resource::CreateTextureViewError::TooManyArrayLayers {
                requested: required_layer_count,
                total: layer_end,
            });
        };

        match view_dim {
            TextureViewDimension::Cube if required_layer_count != 6 => {
                return Err(
                    resource::CreateTextureViewError::InvalidCubemapTextureDepth {
                        depth: required_layer_count,
                    },
                )
            }
            TextureViewDimension::CubeArray if required_layer_count % 6 != 0 => {
                return Err(
                    resource::CreateTextureViewError::InvalidCubemapArrayTextureDepth {
                        depth: required_layer_count,
                    },
                )
            }
            _ => {}
        }

        let full_aspect = hal::FormatAspects::from(texture.desc.format);
        let select_aspect = hal::FormatAspects::from(desc.range.aspect);
        if (full_aspect & select_aspect).is_empty() {
            return Err(resource::CreateTextureViewError::InvalidAspect {
                texture_format: texture.desc.format,
                requested_aspect: desc.range.aspect,
            });
        }

        let end_level = desc
            .range
            .mip_level_count
            .map_or(level_end, |_| required_level_count);
        let end_layer = desc
            .range
            .array_layer_count
            .map_or(layer_end, |_| required_layer_count);
        let selector = TextureSelector {
            levels: desc.range.base_mip_level..end_level,
            layers: desc.range.base_array_layer..end_layer,
        };

        let view_layer_count = selector.layers.end - selector.layers.start;
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

        let mut extent = texture
            .desc
            .mip_level_size(desc.range.base_mip_level)
            .unwrap();
        if view_dim != wgt::TextureViewDimension::D3 {
            extent.depth_or_array_layers = view_layer_count;
        }
        let format = desc.format.unwrap_or(texture.desc.format);
        if format != texture.desc.format {
            return Err(resource::CreateTextureViewError::FormatReinterpretation {
                texture: texture.desc.format,
                view: format,
            });
        }

        // filter the usages based on the other criteria
        let usage = {
            let mask_copy = !(hal::TextureUses::COPY_SRC | hal::TextureUses::COPY_DST);
            let mask_dimension = match view_dim {
                wgt::TextureViewDimension::Cube | wgt::TextureViewDimension::CubeArray => {
                    hal::TextureUses::RESOURCE
                }
                wgt::TextureViewDimension::D3 => {
                    hal::TextureUses::RESOURCE
                        | hal::TextureUses::STORAGE_READ
                        | hal::TextureUses::STORAGE_WRITE
                }
                _ => hal::TextureUses::all(),
            };
            let mask_mip_level = if selector.levels.end - selector.levels.start != 1 {
                hal::TextureUses::RESOURCE
            } else {
                hal::TextureUses::all()
            };
            texture.hal_usage & mask_copy & mask_dimension & mask_mip_level
        };

        log::debug!(
            "Create view for texture {:?} filters usages to {:?}",
            texture_id,
            usage
        );
        let hal_desc = hal::TextureViewDescriptor {
            label: desc.label.borrow_option(),
            format,
            dimension: view_dim,
            usage,
            range: desc.range.clone(),
        };

        let raw = unsafe {
            self.raw
                .create_texture_view(texture_raw, &hal_desc)
                .map_err(|_| resource::CreateTextureViewError::OutOfMemory)?
        };

        Ok(resource::TextureView {
            raw,
            parent_id: Stored {
                value: id::Valid(texture_id),
                ref_count: texture.life_guard.add_ref(),
            },
            desc: resource::HalTextureViewDescriptor {
                format: hal_desc.format,
                dimension: hal_desc.dimension,
                range: hal_desc.range,
            },
            format_features: texture.format_features,
            extent,
            samples: texture.desc.sample_count,
            // once a storage - forever a storage
            sampled_internal_use: if texture
                .desc
                .usage
                .contains(wgt::TextureUsages::STORAGE_BINDING)
            {
                hal::TextureUses::RESOURCE | hal::TextureUses::STORAGE_READ
            } else {
                hal::TextureUses::RESOURCE
            },
            selector,
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
        })
    }

    fn create_sampler(
        &self,
        self_id: id::DeviceId,
        desc: &resource::SamplerDescriptor,
    ) -> Result<resource::Sampler<A>, resource::CreateSamplerError> {
        if desc
            .address_modes
            .iter()
            .any(|am| am == &wgt::AddressMode::ClampToBorder)
        {
            self.require_features(wgt::Features::ADDRESS_MODE_CLAMP_TO_BORDER)?;
        }

        let lod_clamp = if desc.lod_min_clamp > 0.0 || desc.lod_max_clamp < 32.0 {
            Some(desc.lod_min_clamp..desc.lod_max_clamp)
        } else {
            None
        };

        let anisotropy_clamp = if let Some(clamp) = desc.anisotropy_clamp {
            let clamp = clamp.get();
            let valid_clamp = clamp <= hal::MAX_ANISOTROPY && conv::is_power_of_two(clamp as u32);
            if !valid_clamp {
                return Err(resource::CreateSamplerError::InvalidClamp(clamp));
            }
            if self
                .downlevel
                .flags
                .contains(wgt::DownlevelFlags::ANISOTROPIC_FILTERING)
            {
                std::num::NonZeroU8::new(clamp)
            } else {
                None
            }
        } else {
            None
        };

        //TODO: check for wgt::DownlevelFlags::COMPARISON_SAMPLERS

        let hal_desc = hal::SamplerDescriptor {
            label: desc.label.borrow_option(),
            address_modes: desc.address_modes,
            mag_filter: desc.mag_filter,
            min_filter: desc.min_filter,
            mipmap_filter: desc.mipmap_filter,
            lod_clamp,
            compare: desc.compare,
            anisotropy_clamp,
            border_color: desc.border_color,
        };

        let raw = unsafe {
            self.raw
                .create_sampler(&hal_desc)
                .map_err(DeviceError::from)?
        };
        Ok(resource::Sampler {
            raw,
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
            comparison: desc.compare.is_some(),
            filtering: desc.min_filter == wgt::FilterMode::Linear
                || desc.mag_filter == wgt::FilterMode::Linear,
        })
    }

    fn create_shader_module<'a>(
        &self,
        self_id: id::DeviceId,
        desc: &pipeline::ShaderModuleDescriptor<'a>,
        source: pipeline::ShaderModuleSource<'a>,
    ) -> Result<pipeline::ShaderModule<A>, pipeline::CreateShaderModuleError> {
        let module = match source {
            pipeline::ShaderModuleSource::Wgsl(code) => {
                profiling::scope!("naga::wgsl::parse_str");
                // TODO: refactor the corresponding Naga error to be owned, and then
                // display it instead of unwrapping
                match naga::front::wgsl::parse_str(&code) {
                    Ok(module) => module,
                    Err(err) => {
                        log::error!("Failed to parse WGSL code for {:?}: {}", desc.label, err);
                        return Err(pipeline::CreateShaderModuleError::Parsing(
                            pipeline::NagaParseError {
                                shader_source: code.to_string(),
                                error: err,
                            },
                        ));
                    }
                }
            }
            pipeline::ShaderModuleSource::Naga(module) => module,
        };

        use naga::valid::Capabilities as Caps;
        profiling::scope!("naga::validate");

        let mut caps = Caps::empty();
        caps.set(
            Caps::PUSH_CONSTANT,
            self.features.contains(wgt::Features::PUSH_CONSTANTS),
        );
        caps.set(
            Caps::FLOAT64,
            self.features.contains(wgt::Features::SHADER_FLOAT64),
        );
        caps.set(
            Caps::PRIMITIVE_INDEX,
            self.features
                .contains(wgt::Features::SHADER_PRIMITIVE_INDEX),
        );
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
            .validate(&module)?;
        let interface = validation::Interface::new(&module, &info, self.features);
        let hal_shader = hal::ShaderInput::Naga(hal::NagaShader { module, info });

        let hal_desc = hal::ShaderModuleDescriptor {
            label: desc.label.borrow_option(),
        };
        let raw = match unsafe { self.raw.create_shader_module(&hal_desc, hal_shader) } {
            Ok(raw) => raw,
            Err(error) => {
                return Err(match error {
                    hal::ShaderError::Device(error) => {
                        pipeline::CreateShaderModuleError::Device(error.into())
                    }
                    hal::ShaderError::Compilation(ref msg) => {
                        log::error!("Shader error: {}", msg);
                        pipeline::CreateShaderModuleError::Generation
                    }
                })
            }
        };

        Ok(pipeline::ShaderModule {
            raw,
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            interface: Some(interface),
            #[cfg(debug_assertions)]
            label: desc.label.borrow_or_default().to_string(),
        })
    }

    #[allow(unused_unsafe)]
    unsafe fn create_shader_module_spirv<'a>(
        &self,
        self_id: id::DeviceId,
        desc: &pipeline::ShaderModuleDescriptor<'a>,
        source: &'a [u32],
    ) -> Result<pipeline::ShaderModule<A>, pipeline::CreateShaderModuleError> {
        self.require_features(wgt::Features::SPIRV_SHADER_PASSTHROUGH)?;
        let hal_desc = hal::ShaderModuleDescriptor {
            label: desc.label.borrow_option(),
        };
        let hal_shader = hal::ShaderInput::SpirV(source);
        let raw = match unsafe { self.raw.create_shader_module(&hal_desc, hal_shader) } {
            Ok(raw) => raw,
            Err(error) => {
                return Err(match error {
                    hal::ShaderError::Device(error) => {
                        pipeline::CreateShaderModuleError::Device(error.into())
                    }
                    hal::ShaderError::Compilation(ref msg) => {
                        log::error!("Shader error: {}", msg);
                        pipeline::CreateShaderModuleError::Generation
                    }
                })
            }
        };

        Ok(pipeline::ShaderModule {
            raw,
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            interface: None,
            #[cfg(debug_assertions)]
            label: desc.label.borrow_or_default().to_string(),
        })
    }

    fn deduplicate_bind_group_layout(
        self_id: id::DeviceId,
        entry_map: &binding_model::BindEntryMap,
        guard: &Storage<binding_model::BindGroupLayout<A>, id::BindGroupLayoutId>,
    ) -> Option<id::BindGroupLayoutId> {
        guard
            .iter(self_id.backend())
            .find(|&(_, bgl)| bgl.device_id.value.0 == self_id && bgl.entries == *entry_map)
            .map(|(id, value)| {
                value.multi_ref_count.inc();
                id
            })
    }

    fn get_introspection_bind_group_layouts<'a>(
        pipeline_layout: &binding_model::PipelineLayout<A>,
        bgl_guard: &'a Storage<binding_model::BindGroupLayout<A>, id::BindGroupLayoutId>,
    ) -> ArrayVec<&'a binding_model::BindEntryMap, { hal::MAX_BIND_GROUPS }> {
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
    ) -> Result<binding_model::BindGroupLayout<A>, binding_model::CreateBindGroupLayoutError> {
        #[derive(PartialEq)]
        enum WritableStorage {
            Yes,
            No,
        }

        for entry in entry_map.values() {
            use wgt::BindingType as Bt;

            let mut required_features = wgt::Features::empty();
            let mut required_downlevel_flags = wgt::DownlevelFlags::empty();
            let (array_feature, writable_storage) = match entry.ty {
                Bt::Buffer {
                    ty: wgt::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: _,
                } => (
                    Some(wgt::Features::BUFFER_BINDING_ARRAY),
                    WritableStorage::No,
                ),
                Bt::Buffer {
                    ty: wgt::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: _,
                } => (
                    Some(wgt::Features::BUFFER_BINDING_ARRAY),
                    WritableStorage::No,
                ),
                Bt::Buffer {
                    ty: wgt::BufferBindingType::Storage { read_only },
                    ..
                } => (
                    Some(
                        wgt::Features::BUFFER_BINDING_ARRAY
                            | wgt::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                    ),
                    match read_only {
                        true => WritableStorage::No,
                        false => WritableStorage::Yes,
                    },
                ),
                Bt::Sampler { .. } => (None, WritableStorage::No),
                Bt::Texture { .. } => (
                    Some(wgt::Features::TEXTURE_BINDING_ARRAY),
                    WritableStorage::No,
                ),
                Bt::StorageTexture {
                    access,
                    view_dimension,
                    format: _,
                } => {
                    match view_dimension {
                        wgt::TextureViewDimension::Cube | wgt::TextureViewDimension::CubeArray => {
                            return Err(binding_model::CreateBindGroupLayoutError::Entry {
                                binding: entry.binding,
                                error: binding_model::BindGroupLayoutEntryError::StorageTextureCube,
                            })
                        }
                        _ => (),
                    }
                    (
                        Some(
                            wgt::Features::TEXTURE_BINDING_ARRAY
                                | wgt::Features::STORAGE_RESOURCE_BINDING_ARRAY,
                        ),
                        match access {
                            wgt::StorageTextureAccess::WriteOnly => WritableStorage::Yes,
                            wgt::StorageTextureAccess::ReadOnly => {
                                required_features |=
                                    wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
                                WritableStorage::No
                            }
                            wgt::StorageTextureAccess::ReadWrite => {
                                required_features |=
                                    wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
                                WritableStorage::Yes
                            }
                        },
                    )
                }
            };

            // Validate the count parameter
            if entry.count.is_some() {
                required_features |= array_feature
                    .ok_or(binding_model::BindGroupLayoutEntryError::ArrayUnsupported)
                    .map_err(|error| binding_model::CreateBindGroupLayoutError::Entry {
                        binding: entry.binding,
                        error,
                    })?;
            }
            if entry.visibility.contains(wgt::ShaderStages::VERTEX) {
                if writable_storage == WritableStorage::Yes {
                    required_features |= wgt::Features::VERTEX_WRITABLE_STORAGE;
                }
                if let Bt::Buffer {
                    ty: wgt::BufferBindingType::Storage { .. },
                    ..
                } = entry.ty
                {
                    required_downlevel_flags |= wgt::DownlevelFlags::VERTEX_STORAGE;
                }
            }
            if writable_storage == WritableStorage::Yes
                && entry.visibility.contains(wgt::ShaderStages::FRAGMENT)
            {
                required_downlevel_flags |= wgt::DownlevelFlags::FRAGMENT_WRITABLE_STORAGE;
            }

            self.require_features(required_features)
                .map_err(binding_model::BindGroupLayoutEntryError::MissingFeatures)
                .map_err(|error| binding_model::CreateBindGroupLayoutError::Entry {
                    binding: entry.binding,
                    error,
                })?;
            self.require_downlevel_flags(required_downlevel_flags)
                .map_err(binding_model::BindGroupLayoutEntryError::MissingDownlevelFlags)
                .map_err(|error| binding_model::CreateBindGroupLayoutError::Entry {
                    binding: entry.binding,
                    error,
                })?;
        }

        let mut hal_bindings = entry_map.values().cloned().collect::<Vec<_>>();
        hal_bindings.sort_by_key(|b| b.binding);
        let hal_desc = hal::BindGroupLayoutDescriptor {
            label,
            entries: &hal_bindings,
        };
        let raw = unsafe {
            self.raw
                .create_bind_group_layout(&hal_desc)
                .map_err(DeviceError::from)?
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

    #[allow(clippy::too_many_arguments)]
    fn create_buffer_binding<'a>(
        bb: &binding_model::BufferBinding,
        binding: u32,
        decl: &wgt::BindGroupLayoutEntry,
        used_buffer_ranges: &mut Vec<BufferInitTrackerAction>,
        dynamic_binding_info: &mut Vec<binding_model::BindGroupDynamicBindingData>,
        used: &mut TrackerSet,
        storage: &'a Storage<resource::Buffer<A>, id::BufferId>,
        limits: &wgt::Limits,
    ) -> Result<hal::BufferBinding<'a, A>, binding_model::CreateBindGroupError> {
        use crate::binding_model::CreateBindGroupError as Error;

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
                wgt::BufferUsages::UNIFORM,
                hal::BufferUses::UNIFORM,
                limits.max_uniform_buffer_binding_size,
            ),
            wgt::BufferBindingType::Storage { read_only } => (
                wgt::BufferUsages::STORAGE,
                if read_only {
                    hal::BufferUses::STORAGE_READ
                } else {
                    hal::BufferUses::STORAGE_WRITE
                },
                limits.max_storage_buffer_binding_size,
            ),
        };

        if bb.offset % wgt::BIND_BUFFER_ALIGNMENT != 0 {
            return Err(Error::UnalignedBufferOffset(bb.offset));
        }

        let buffer = used
            .buffers
            .use_extend(storage, bb.buffer_id, (), internal_use)
            .map_err(|_| Error::InvalidBuffer(bb.buffer_id))?;
        check_buffer_usage(buffer.usage, pub_usage)?;
        let raw_buffer = buffer
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

        used_buffer_ranges.extend(buffer.initialization_status.create_action(
            bb.buffer_id,
            bb.offset..(bb.offset + bind_size),
            MemoryInitKind::NeedsInitializedMemory,
        ));

        Ok(hal::BufferBinding {
            buffer: raw_buffer,
            offset: bb.offset,
            size: bb.size,
        })
    }

    fn create_bind_group<G: GlobalIdentityHandlerFactory>(
        &self,
        self_id: id::DeviceId,
        layout: &binding_model::BindGroupLayout<A>,
        desc: &binding_model::BindGroupDescriptor,
        hub: &Hub<A, G>,
        token: &mut Token<binding_model::BindGroupLayout<A>>,
    ) -> Result<binding_model::BindGroup<A>, binding_model::CreateBindGroupError> {
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
        let mut used = TrackerSet::new(A::VARIANT);

        let (buffer_guard, mut token) = hub.buffers.read(token);
        let (texture_guard, mut token) = hub.textures.read(&mut token); //skip token
        let (texture_view_guard, mut token) = hub.texture_views.read(&mut token);
        let (sampler_guard, _) = hub.samplers.read(&mut token);

        let mut used_buffer_ranges = Vec::new();
        let mut hal_entries = Vec::with_capacity(desc.entries.len());
        let mut hal_buffers = Vec::new();
        let mut hal_samplers = Vec::new();
        let mut hal_textures = Vec::new();
        for entry in desc.entries.iter() {
            let binding = entry.binding;
            // Find the corresponding declaration in the layout
            let decl = layout
                .entries
                .get(&binding)
                .ok_or(Error::MissingBindingDeclaration(binding))?;
            let res_index = match entry.resource {
                Br::Buffer(ref bb) => {
                    let bb = Self::create_buffer_binding(
                        bb,
                        binding,
                        decl,
                        &mut used_buffer_ranges,
                        &mut dynamic_binding_info,
                        &mut used,
                        &*buffer_guard,
                        &self.limits,
                    )?;

                    let res_index = hal_buffers.len();
                    hal_buffers.push(bb);
                    res_index
                }
                Br::BufferArray(ref bindings_array) => {
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

                    let res_index = hal_buffers.len();
                    for bb in bindings_array.iter() {
                        let bb = Self::create_buffer_binding(
                            bb,
                            binding,
                            decl,
                            &mut used_buffer_ranges,
                            &mut dynamic_binding_info,
                            &mut used,
                            &*buffer_guard,
                            &self.limits,
                        )?;
                        hal_buffers.push(bb);
                    }
                    res_index
                }
                Br::Sampler(id) => {
                    match decl.ty {
                        wgt::BindingType::Sampler {
                            filtering,
                            comparison,
                        } => {
                            let sampler = used
                                .samplers
                                .use_extend(&*sampler_guard, id, (), ())
                                .map_err(|_| Error::InvalidSampler(id))?;

                            // Check the actual sampler to also (not) be a comparison sampler
                            if sampler.comparison != comparison {
                                return Err(Error::WrongSamplerComparison {
                                    binding,
                                    layout_cmp: comparison,
                                    sampler_cmp: sampler.comparison,
                                });
                            }
                            // Check the actual sampler to be non-filtering, if required
                            if sampler.filtering && !filtering {
                                return Err(Error::WrongSamplerFiltering {
                                    binding,
                                    layout_flt: filtering,
                                    sampler_flt: sampler.filtering,
                                });
                            }

                            let res_index = hal_samplers.len();
                            hal_samplers.push(&sampler.raw);
                            res_index
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
                    let (pub_usage, internal_use) = Self::texture_use_parameters(
                        binding,
                        decl,
                        view,
                        "SampledTexture, ReadonlyStorageTexture or WriteonlyStorageTexture",
                    )?;

                    // Careful here: the texture may no longer have its own ref count,
                    // if it was deleted by the user.
                    used.textures
                        .change_extend(
                            view.parent_id.value,
                            &view.parent_id.ref_count,
                            view.selector.clone(),
                            internal_use,
                        )
                        .map_err(UsageConflict::from)?;
                    let texture = &texture_guard[view.parent_id.value];
                    check_texture_usage(texture.desc.usage, pub_usage)?;

                    let res_index = hal_textures.len();
                    hal_textures.push(hal::TextureBinding {
                        view: &view.raw,
                        usage: internal_use,
                    });
                    res_index
                }
                Br::TextureViewArray(ref bindings_array) => {
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

                    let res_index = hal_textures.len();
                    for &id in bindings_array.iter() {
                        let view = used
                            .views
                            .use_extend(&*texture_view_guard, id, (), ())
                            .map_err(|_| Error::InvalidTextureView(id))?;
                        let (pub_usage, internal_use) = Self::texture_use_parameters(
                            binding, decl, view,
                            "SampledTextureArray, ReadonlyStorageTextureArray or WriteonlyStorageTextureArray"
                        )?;

                        // Careful here: the texture may no longer have its own ref count,
                        // if it was deleted by the user.
                        used.textures
                            .change_extend(
                                view.parent_id.value,
                                &view.parent_id.ref_count,
                                view.selector.clone(),
                                internal_use,
                            )
                            .map_err(UsageConflict::from)?;
                        let texture = &texture_guard[view.parent_id.value];
                        check_texture_usage(texture.desc.usage, pub_usage)?;

                        hal_textures.push(hal::TextureBinding {
                            view: &view.raw,
                            usage: internal_use,
                        });
                    }

                    res_index
                }
            };

            hal_entries.push(hal::BindGroupEntry {
                binding,
                resource_index: res_index as u32,
            });
        }

        hal_entries.sort_by_key(|entry| entry.binding);
        for (a, b) in hal_entries.iter().zip(hal_entries.iter().skip(1)) {
            if a.binding == b.binding {
                return Err(Error::DuplicateBinding(a.binding));
            }
        }

        let hal_desc = hal::BindGroupDescriptor {
            label: desc.label.borrow_option(),
            layout: &layout.raw,
            entries: &hal_entries,
            buffers: &hal_buffers,
            samplers: &hal_samplers,
            textures: &hal_textures,
        };
        let raw = unsafe {
            self.raw
                .create_bind_group(&hal_desc)
                .map_err(DeviceError::from)?
        };

        Ok(binding_model::BindGroup {
            raw,
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

    fn texture_use_parameters(
        binding: u32,
        decl: &wgt::BindGroupLayoutEntry,
        view: &crate::resource::TextureView<A>,
        expected: &'static str,
    ) -> Result<(wgt::TextureUsages, hal::TextureUses), binding_model::CreateBindGroupError> {
        use crate::binding_model::CreateBindGroupError as Error;
        if view
            .desc
            .aspects()
            .contains(hal::FormatAspects::DEPTH | hal::FormatAspects::STENCIL)
        {
            return Err(Error::DepthStencilAspect);
        }
        let format_info = view.desc.format.describe();
        match decl.ty {
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
                        view_samples: view.samples,
                    });
                }
                match (sample_type, format_info.sample_type, view.format_features.filterable) {
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
                            view_format: view.desc.format,
                        })
                    }
                }
                if view_dimension != view.desc.dimension {
                    return Err(Error::InvalidTextureDimension {
                        binding,
                        layout_dimension: view_dimension,
                        view_dimension: view.desc.dimension,
                    });
                }
                Ok((
                    wgt::TextureUsages::TEXTURE_BINDING,
                    view.sampled_internal_use,
                ))
            }
            wgt::BindingType::StorageTexture {
                access,
                format,
                view_dimension,
            } => {
                if format != view.desc.format {
                    return Err(Error::InvalidStorageTextureFormat {
                        binding,
                        layout_format: format,
                        view_format: view.desc.format,
                    });
                }
                if view_dimension != view.desc.dimension {
                    return Err(Error::InvalidTextureDimension {
                        binding,
                        layout_dimension: view_dimension,
                        view_dimension: view.desc.dimension,
                    });
                }

                let mip_level_count = view.selector.levels.end - view.selector.levels.start;
                if mip_level_count != 1 {
                    return Err(Error::InvalidStorageTextureMipLevelCount {
                        binding,
                        mip_level_count,
                    });
                }

                let internal_use = match access {
                    wgt::StorageTextureAccess::WriteOnly => hal::TextureUses::STORAGE_WRITE,
                    wgt::StorageTextureAccess::ReadOnly => {
                        if !view
                            .format_features
                            .flags
                            .contains(wgt::TextureFormatFeatureFlags::STORAGE_READ_WRITE)
                        {
                            return Err(Error::StorageReadNotSupported(view.desc.format));
                        }
                        hal::TextureUses::STORAGE_READ
                    }
                    wgt::StorageTextureAccess::ReadWrite => {
                        if !view
                            .format_features
                            .flags
                            .contains(wgt::TextureFormatFeatureFlags::STORAGE_READ_WRITE)
                        {
                            return Err(Error::StorageReadNotSupported(view.desc.format));
                        }

                        hal::TextureUses::STORAGE_WRITE | hal::TextureUses::STORAGE_READ
                    }
                };
                Ok((wgt::TextureUsages::STORAGE_BINDING, internal_use))
            }
            _ => Err(Error::WrongBindingType {
                binding,
                actual: decl.ty,
                expected,
            }),
        }
    }

    fn create_pipeline_layout(
        &self,
        self_id: id::DeviceId,
        desc: &binding_model::PipelineLayoutDescriptor,
        bgl_guard: &Storage<binding_model::BindGroupLayout<A>, id::BindGroupLayoutId>,
    ) -> Result<binding_model::PipelineLayout<A>, binding_model::CreatePipelineLayoutError> {
        use crate::binding_model::CreatePipelineLayoutError as Error;

        let bind_group_layouts_count = desc.bind_group_layouts.len();
        let device_max_bind_groups = self.limits.max_bind_groups as usize;
        if bind_group_layouts_count > device_max_bind_groups {
            return Err(Error::TooManyGroups {
                actual: bind_group_layouts_count,
                max: device_max_bind_groups,
            });
        }

        if !desc.push_constant_ranges.is_empty() {
            self.require_features(wgt::Features::PUSH_CONSTANTS)?;
        }

        let mut used_stages = wgt::ShaderStages::empty();
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

        let bgl_vec = desc
            .bind_group_layouts
            .iter()
            .map(|&id| &bgl_guard.get(id).unwrap().raw)
            .collect::<Vec<_>>();
        let hal_desc = hal::PipelineLayoutDescriptor {
            label: desc.label.borrow_option(),
            flags: hal::PipelineLayoutFlags::BASE_VERTEX_INSTANCE,
            bind_group_layouts: &bgl_vec,
            push_constant_ranges: desc.push_constant_ranges.as_ref(),
        };

        let raw = unsafe {
            self.raw
                .create_pipeline_layout(&hal_desc)
                .map_err(DeviceError::from)?
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
        mut derived_group_layouts: ArrayVec<binding_model::BindEntryMap, { hal::MAX_BIND_GROUPS }>,
        bgl_guard: &mut Storage<binding_model::BindGroupLayout<A>, id::BindGroupLayoutId>,
        pipeline_layout_guard: &mut Storage<binding_model::PipelineLayout<A>, id::PipelineLayoutId>,
    ) -> Result<id::PipelineLayoutId, pipeline::ImplicitLayoutError> {
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
            return Err(pipeline::ImplicitLayoutError::MissingIds(group_count as _));
        }

        for (bgl_id, map) in ids.group_ids.iter_mut().zip(derived_group_layouts) {
            match Device::deduplicate_bind_group_layout(self_id, &map, bgl_guard) {
                Some(dedup_id) => {
                    *bgl_id = dedup_id;
                }
                None => {
                    let bgl = self.create_bind_group_layout(self_id, None, map)?;
                    bgl_guard.force_replace(*bgl_id, bgl);
                }
            };
        }

        let layout_desc = binding_model::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: Cow::Borrowed(&ids.group_ids[..group_count]),
            push_constant_ranges: Cow::Borrowed(&[]), //TODO?
        };
        let layout = self.create_pipeline_layout(self_id, &layout_desc, bgl_guard)?;
        pipeline_layout_guard.force_replace(ids.root_id, layout);
        Ok(ids.root_id)
    }

    fn create_compute_pipeline<G: GlobalIdentityHandlerFactory>(
        &self,
        self_id: id::DeviceId,
        desc: &pipeline::ComputePipelineDescriptor,
        implicit_context: Option<ImplicitPipelineContext>,
        hub: &Hub<A, G>,
        token: &mut Token<Self>,
    ) -> Result<pipeline::ComputePipeline<A>, pipeline::CreateComputePipelineError> {
        //TODO: only lock mutable if the layout is derived
        let (mut pipeline_layout_guard, mut token) = hub.pipeline_layouts.write(token);
        let (mut bgl_guard, mut token) = hub.bind_group_layouts.write(&mut token);

        // This has to be done first, or otherwise the IDs may be pointing to entries
        // that are not even in the storage.
        if let Some(ref ids) = implicit_context {
            pipeline_layout_guard.insert_error(ids.root_id, IMPLICIT_FAILURE);
            for &bgl_id in ids.group_ids.iter() {
                bgl_guard.insert_error(bgl_id, IMPLICIT_FAILURE);
            }
        }

        self.require_downlevel_flags(wgt::DownlevelFlags::COMPUTE_SHADERS)?;

        let mut derived_group_layouts =
            ArrayVec::<binding_model::BindEntryMap, { hal::MAX_BIND_GROUPS }>::new();

        let io = validation::StageIo::default();
        let (shader_module_guard, _) = hub.shader_modules.read(&mut token);

        let shader_module = shader_module_guard
            .get(desc.stage.module)
            .map_err(|_| validation::StageError::InvalidModule)?;

        {
            let flag = wgt::ShaderStages::COMPUTE;
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
            if let Some(ref interface) = shader_module.interface {
                let _ = interface.check_stage(
                    provided_layouts.as_ref().map(|p| p.as_slice()),
                    &mut derived_group_layouts,
                    &desc.stage.entry_point,
                    flag,
                    io,
                )?;
            }
        }

        let pipeline_layout_id = match desc.layout {
            Some(id) => id,
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

        let pipeline_desc = hal::ComputePipelineDescriptor {
            label: desc.label.borrow_option(),
            layout: &layout.raw,
            stage: hal::ProgrammableStage {
                entry_point: desc.stage.entry_point.as_ref(),
                module: &shader_module.raw,
            },
        };

        let raw =
            unsafe { self.raw.create_compute_pipeline(&pipeline_desc) }.map_err(
                |err| match err {
                    hal::PipelineError::Device(error) => {
                        pipeline::CreateComputePipelineError::Device(error.into())
                    }
                    hal::PipelineError::Linkage(_stages, msg) => {
                        pipeline::CreateComputePipelineError::Internal(msg)
                    }
                    hal::PipelineError::EntryPoint(_stage) => {
                        pipeline::CreateComputePipelineError::Internal(EP_FAILURE.to_string())
                    }
                },
            )?;

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
        Ok(pipeline)
    }

    fn create_render_pipeline<G: GlobalIdentityHandlerFactory>(
        &self,
        self_id: id::DeviceId,
        adapter: &crate::instance::Adapter<A>,
        desc: &pipeline::RenderPipelineDescriptor,
        implicit_context: Option<ImplicitPipelineContext>,
        hub: &Hub<A, G>,
        token: &mut Token<Self>,
    ) -> Result<pipeline::RenderPipeline<A>, pipeline::CreateRenderPipelineError> {
        //TODO: only lock mutable if the layout is derived
        let (mut pipeline_layout_guard, mut token) = hub.pipeline_layouts.write(token);
        let (mut bgl_guard, mut token) = hub.bind_group_layouts.write(&mut token);

        // This has to be done first, or otherwise the IDs may be pointing to entries
        // that are not even in the storage.
        if let Some(ref ids) = implicit_context {
            pipeline_layout_guard.insert_error(ids.root_id, IMPLICIT_FAILURE);
            for &bgl_id in ids.group_ids.iter() {
                bgl_guard.insert_error(bgl_id, IMPLICIT_FAILURE);
            }
        }

        let mut derived_group_layouts =
            ArrayVec::<binding_model::BindEntryMap, { hal::MAX_BIND_GROUPS }>::new();

        let color_targets = desc
            .fragment
            .as_ref()
            .map_or(&[][..], |fragment| &fragment.targets);
        let depth_stencil_state = desc.depth_stencil.as_ref();

        if !color_targets.is_empty() && {
            let first = &color_targets[0];
            color_targets[1..]
                .iter()
                .any(|ct| ct.write_mask != first.write_mask || ct.blend != first.blend)
        } {
            log::info!("Color targets: {:?}", color_targets);
            self.require_downlevel_flags(wgt::DownlevelFlags::INDEPENDENT_BLENDING)?;
        }

        let mut io = validation::StageIo::default();
        let mut validated_stages = wgt::ShaderStages::empty();

        let mut vertex_strides = Vec::with_capacity(desc.vertex.buffers.len());
        let mut vertex_buffers = Vec::with_capacity(desc.vertex.buffers.len());
        let mut total_attributes = 0;
        for (i, vb_state) in desc.vertex.buffers.iter().enumerate() {
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
            vertex_buffers.alloc().init(hal::VertexBufferLayout {
                array_stride: vb_state.array_stride,
                step_mode: vb_state.step_mode,
                attributes: vb_state.attributes.as_ref(),
            });

            for attribute in vb_state.attributes.iter() {
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
                    self.require_features(wgt::Features::VERTEX_ATTRIBUTE_64BIT)?;
                }

                io.insert(
                    attribute.shader_location,
                    validation::InterfaceVar::vertex_attribute(attribute.format),
                );
            }
            total_attributes += vb_state.attributes.len();
        }

        if vertex_buffers.len() > self.limits.max_vertex_buffers as usize {
            return Err(pipeline::CreateRenderPipelineError::TooManyVertexBuffers {
                given: vertex_buffers.len() as u32,
                limit: self.limits.max_vertex_buffers,
            });
        }
        if total_attributes > self.limits.max_vertex_attributes as usize {
            return Err(
                pipeline::CreateRenderPipelineError::TooManyVertexAttributes {
                    given: total_attributes as u32,
                    limit: self.limits.max_vertex_attributes,
                },
            );
        }

        if desc.primitive.strip_index_format.is_some() && !desc.primitive.topology.is_strip() {
            return Err(
                pipeline::CreateRenderPipelineError::StripIndexFormatForNonStripTopology {
                    strip_index_format: desc.primitive.strip_index_format,
                    topology: desc.primitive.topology,
                },
            );
        }

        if desc.primitive.clamp_depth {
            self.require_features(wgt::Features::DEPTH_CLAMPING)?;
        }
        if desc.primitive.polygon_mode != wgt::PolygonMode::Fill {
            self.require_features(wgt::Features::NON_FILL_POLYGON_MODE)?;
        }

        if desc.primitive.conservative {
            self.require_features(wgt::Features::CONSERVATIVE_RASTERIZATION)?;
        }

        if desc.primitive.conservative && desc.primitive.polygon_mode != wgt::PolygonMode::Fill {
            return Err(
                pipeline::CreateRenderPipelineError::ConservativeRasterizationNonFillPolygonMode,
            );
        }

        for (i, cs) in color_targets.iter().enumerate() {
            let error = loop {
                let format_features = self.describe_format_features(adapter, cs.format)?;
                if !format_features
                    .allowed_usages
                    .contains(wgt::TextureUsages::RENDER_ATTACHMENT)
                {
                    break Some(pipeline::ColorStateError::FormatNotRenderable(cs.format));
                }
                if cs.blend.is_some() && !format_features.filterable {
                    break Some(pipeline::ColorStateError::FormatNotBlendable(cs.format));
                }
                if !hal::FormatAspects::from(cs.format).contains(hal::FormatAspects::COLOR) {
                    break Some(pipeline::ColorStateError::FormatNotColor(cs.format));
                }

                break None;
            };
            if let Some(e) = error {
                return Err(pipeline::CreateRenderPipelineError::ColorState(i as u8, e));
            }
        }

        if let Some(ds) = depth_stencil_state {
            let error = loop {
                if !self
                    .describe_format_features(adapter, ds.format)?
                    .allowed_usages
                    .contains(wgt::TextureUsages::RENDER_ATTACHMENT)
                {
                    break Some(pipeline::DepthStencilStateError::FormatNotRenderable(
                        ds.format,
                    ));
                }
                let aspect = hal::FormatAspects::from(ds.format);
                if ds.is_depth_enabled() && !aspect.contains(hal::FormatAspects::DEPTH) {
                    break Some(pipeline::DepthStencilStateError::FormatNotDepth(ds.format));
                }
                if ds.stencil.is_enabled() && !aspect.contains(hal::FormatAspects::STENCIL) {
                    break Some(pipeline::DepthStencilStateError::FormatNotStencil(
                        ds.format,
                    ));
                }
                break None;
            };
            if let Some(e) = error {
                return Err(pipeline::CreateRenderPipelineError::DepthStencilState(e));
            }
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
            sc
        };

        let (shader_module_guard, _) = hub.shader_modules.read(&mut token);

        let vertex_stage = {
            let stage = &desc.vertex.stage;
            let flag = wgt::ShaderStages::VERTEX;

            let shader_module = shader_module_guard.get(stage.module).map_err(|_| {
                pipeline::CreateRenderPipelineError::Stage {
                    stage: flag,
                    error: validation::StageError::InvalidModule,
                }
            })?;

            let provided_layouts = match desc.layout {
                Some(pipeline_layout_id) => {
                    let pipeline_layout = pipeline_layout_guard
                        .get(pipeline_layout_id)
                        .map_err(|_| pipeline::CreateRenderPipelineError::InvalidLayout)?;
                    Some(Device::get_introspection_bind_group_layouts(
                        pipeline_layout,
                        &*bgl_guard,
                    ))
                }
                None => None,
            };

            if let Some(ref interface) = shader_module.interface {
                io = interface
                    .check_stage(
                        provided_layouts.as_ref().map(|p| p.as_slice()),
                        &mut derived_group_layouts,
                        &stage.entry_point,
                        flag,
                        io,
                    )
                    .map_err(|error| pipeline::CreateRenderPipelineError::Stage {
                        stage: flag,
                        error,
                    })?;
                validated_stages |= flag;
            }

            hal::ProgrammableStage {
                module: &shader_module.raw,
                entry_point: stage.entry_point.as_ref(),
            }
        };

        let fragment_stage = match desc.fragment {
            Some(ref fragment) => {
                let flag = wgt::ShaderStages::FRAGMENT;

                let shader_module =
                    shader_module_guard
                        .get(fragment.stage.module)
                        .map_err(|_| pipeline::CreateRenderPipelineError::Stage {
                            stage: flag,
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

                if validated_stages == wgt::ShaderStages::VERTEX {
                    if let Some(ref interface) = shader_module.interface {
                        io = interface
                            .check_stage(
                                provided_layouts.as_ref().map(|p| p.as_slice()),
                                &mut derived_group_layouts,
                                &fragment.stage.entry_point,
                                flag,
                                io,
                            )
                            .map_err(|error| pipeline::CreateRenderPipelineError::Stage {
                                stage: flag,
                                error,
                            })?;
                        validated_stages |= flag;
                    }
                }

                Some(hal::ProgrammableStage {
                    module: &shader_module.raw,
                    entry_point: fragment.stage.entry_point.as_ref(),
                })
            }
            None => None,
        };

        if validated_stages.contains(wgt::ShaderStages::FRAGMENT) {
            for (i, state) in color_targets.iter().enumerate() {
                match io.get(&(i as wgt::ShaderLocation)) {
                    Some(output) => {
                        validation::check_texture_format(state.format, &output.ty).map_err(
                            |pipeline| {
                                pipeline::CreateRenderPipelineError::ColorState(
                                    i as u8,
                                    pipeline::ColorStateError::IncompatibleFormat {
                                        pipeline,
                                        shader: output.ty,
                                    },
                                )
                            },
                        )?;
                    }
                    None if state.write_mask.is_empty() => {}
                    None => {
                        log::warn!("Missing fragment output[{}], expected {:?}", i, state,);
                        return Err(pipeline::CreateRenderPipelineError::ColorState(
                            i as u8,
                            pipeline::ColorStateError::Missing,
                        ));
                    }
                }
            }
        }
        let last_stage = match desc.fragment {
            Some(_) => wgt::ShaderStages::FRAGMENT,
            None => wgt::ShaderStages::VERTEX,
        };
        if desc.layout.is_none() && !validated_stages.contains(last_stage) {
            return Err(pipeline::ImplicitLayoutError::ReflectionError(last_stage).into());
        }

        let pipeline_layout_id = match desc.layout {
            Some(id) => id,
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

        let pipeline_desc = hal::RenderPipelineDescriptor {
            label: desc.label.borrow_option(),
            layout: &layout.raw,
            vertex_buffers: &vertex_buffers,
            vertex_stage,
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment_stage,
            color_targets,
        };
        let raw =
            unsafe { self.raw.create_render_pipeline(&pipeline_desc) }.map_err(
                |err| match err {
                    hal::PipelineError::Device(error) => {
                        pipeline::CreateRenderPipelineError::Device(error.into())
                    }
                    hal::PipelineError::Linkage(stage, msg) => {
                        pipeline::CreateRenderPipelineError::Internal { stage, error: msg }
                    }
                    hal::PipelineError::EntryPoint(stage) => {
                        pipeline::CreateRenderPipelineError::Internal {
                            stage: hal::auxil::map_naga_stage(stage),
                            error: EP_FAILURE.to_string(),
                        }
                    }
                },
            )?;

        let pass_context = RenderPassContext {
            attachments: AttachmentData {
                colors: color_targets.iter().map(|state| state.format).collect(),
                resolves: ArrayVec::new(),
                depth_stencil: depth_stencil_state.as_ref().map(|state| state.format),
            },
            sample_count: samples,
        };

        let mut flags = pipeline::PipelineFlags::empty();
        for state in color_targets.iter() {
            if let Some(ref bs) = state.blend {
                if bs.color.uses_constant() | bs.alpha.uses_constant() {
                    flags |= pipeline::PipelineFlags::BLEND_CONSTANT;
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
        Ok(pipeline)
    }

    fn describe_format_features(
        &self,
        adapter: &crate::instance::Adapter<A>,
        format: TextureFormat,
    ) -> Result<wgt::TextureFormatFeatures, MissingFeatures> {
        let format_desc = format.describe();
        self.require_features(format_desc.required_features)?;

        if self
            .features
            .contains(wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            Ok(adapter.get_texture_format_features(format))
        } else {
            Ok(format_desc.guaranteed_format_features)
        }
    }

    fn wait_for_submit(
        &self,
        submission_index: SubmissionIndex,
        token: &mut Token<Self>,
    ) -> Result<(), WaitIdleError> {
        let last_done_index = unsafe {
            self.raw
                .get_fence_value(&self.fence)
                .map_err(DeviceError::from)?
        };
        if last_done_index < submission_index {
            log::info!("Waiting for submission {:?}", submission_index);
            unsafe {
                self.raw
                    .wait(&self.fence, submission_index, !0)
                    .map_err(DeviceError::from)?
            };
            let closures = self
                .lock_life(token)
                .triage_submissions(submission_index, &self.command_allocator);
            assert!(
                closures.is_empty(),
                "wait_for_submit is not expected to work with closures"
            );
        }
        Ok(())
    }

    fn create_query_set(
        &self,
        self_id: id::DeviceId,
        desc: &resource::QuerySetDescriptor,
    ) -> Result<resource::QuerySet<A>, resource::CreateQuerySetError> {
        use resource::CreateQuerySetError as Error;

        match desc.ty {
            wgt::QueryType::Occlusion => {}
            wgt::QueryType::Timestamp => {
                self.require_features(wgt::Features::TIMESTAMP_QUERY)?;
            }
            wgt::QueryType::PipelineStatistics(..) => {
                self.require_features(wgt::Features::PIPELINE_STATISTICS_QUERY)?;
            }
        }

        if desc.count == 0 {
            return Err(Error::ZeroCount);
        }

        if desc.count > wgt::QUERY_SET_MAX_QUERIES {
            return Err(Error::TooManyQueries {
                count: desc.count,
                maximum: wgt::QUERY_SET_MAX_QUERIES,
            });
        }

        let hal_desc = desc.map_label(super::LabelHelpers::borrow_option);
        Ok(resource::QuerySet {
            raw: unsafe { self.raw.create_query_set(&hal_desc).unwrap() },
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            life_guard: LifeGuard::new(""),
            desc: desc.map_label(|_| ()),
        })
    }
}

impl<A: hal::Api> Device<A> {
    pub(crate) fn destroy_buffer(&self, buffer: resource::Buffer<A>) {
        if let Some(raw) = buffer.raw {
            unsafe {
                self.raw.destroy_buffer(raw);
            }
        }
    }

    pub(crate) fn destroy_command_buffer(&self, cmd_buf: command::CommandBuffer<A>) {
        let mut baked = cmd_buf.into_baked();
        unsafe {
            baked.encoder.reset_all(baked.list.into_iter());
        }
        unsafe {
            self.raw.destroy_command_encoder(baked.encoder);
        }
    }

    /// Wait for idle and remove resources that we can, before we die.
    pub(crate) fn prepare_to_die(&mut self) {
        self.pending_writes.deactivate();
        let mut life_tracker = self.life_tracker.lock();
        let current_index = self.active_submission_index;
        if let Err(error) = unsafe { self.raw.wait(&self.fence, current_index, CLEANUP_WAIT_MS) } {
            log::error!("failed to wait for the device: {:?}", error);
        }
        let _ = life_tracker.triage_submissions(current_index, &self.command_allocator);
        life_tracker.cleanup(&self.raw);
        #[cfg(feature = "trace")]
        {
            self.trace = None;
        }
    }

    pub(crate) fn dispose(self) {
        self.pending_writes.dispose(&self.raw);
        self.command_allocator.into_inner().dispose(&self.raw);
        unsafe {
            self.raw.destroy_fence(self.fence);
            self.raw.exit(self.queue);
        }
    }
}

impl<A: hal::Api> crate::hub::Resource for Device<A> {
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

impl From<hal::DeviceError> for DeviceError {
    fn from(error: hal::DeviceError) -> Self {
        match error {
            hal::DeviceError::Lost => DeviceError::Lost,
            hal::DeviceError::OutOfMemory => DeviceError::OutOfMemory,
        }
    }
}

#[derive(Clone, Debug, Error)]
#[error("Features {0:?} are required but not enabled on the device")]
pub struct MissingFeatures(pub wgt::Features);

#[derive(Clone, Debug, Error)]
#[error(
    "Downlevel flags {0:?} are required but not supported on the device.\n{}",
    DOWNLEVEL_ERROR_MESSAGE
)]
pub struct MissingDownlevelFlags(pub wgt::DownlevelFlags);

#[derive(Clone, Debug)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub struct ImplicitPipelineContext {
    pub root_id: id::PipelineLayoutId,
    pub group_ids: ArrayVec<id::BindGroupLayoutId, { hal::MAX_BIND_GROUPS }>,
}

pub struct ImplicitPipelineIds<'a, G: GlobalIdentityHandlerFactory> {
    pub root_id: Input<G, id::PipelineLayoutId>,
    pub group_ids: &'a [Input<G, id::BindGroupLayoutId>],
}

impl<G: GlobalIdentityHandlerFactory> ImplicitPipelineIds<'_, G> {
    fn prepare<A: hal::Api>(self, hub: &Hub<A, G>) -> ImplicitPipelineContext {
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
    pub fn adapter_is_surface_supported<A: HalApi>(
        &self,
        adapter_id: id::AdapterId,
        surface_id: id::SurfaceId,
    ) -> Result<bool, instance::IsSurfaceSupportedError> {
        let hub = A::hub(self);
        let mut token = Token::root();

        let (surface_guard, mut token) = self.surfaces.read(&mut token);
        let (adapter_guard, mut _token) = hub.adapters.read(&mut token);
        let adapter = adapter_guard
            .get(adapter_id)
            .map_err(|_| instance::IsSurfaceSupportedError::InvalidAdapter)?;
        let surface = surface_guard
            .get(surface_id)
            .map_err(|_| instance::IsSurfaceSupportedError::InvalidSurface)?;
        Ok(adapter.is_surface_supported(surface))
    }
    pub fn surface_get_preferred_format<A: HalApi>(
        &self,
        surface_id: id::SurfaceId,
        adapter_id: id::AdapterId,
    ) -> Result<TextureFormat, instance::GetSurfacePreferredFormatError> {
        let hub = A::hub(self);
        let mut token = Token::root();

        let (surface_guard, mut token) = self.surfaces.read(&mut token);
        let (adapter_guard, mut _token) = hub.adapters.read(&mut token);
        let adapter = adapter_guard
            .get(adapter_id)
            .map_err(|_| instance::GetSurfacePreferredFormatError::InvalidAdapter)?;
        let surface = surface_guard
            .get(surface_id)
            .map_err(|_| instance::GetSurfacePreferredFormatError::InvalidSurface)?;

        surface.get_preferred_format(adapter)
    }

    pub fn device_features<A: HalApi>(
        &self,
        device_id: id::DeviceId,
    ) -> Result<wgt::Features, InvalidDevice> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;

        Ok(device.features)
    }

    pub fn device_limits<A: HalApi>(
        &self,
        device_id: id::DeviceId,
    ) -> Result<wgt::Limits, InvalidDevice> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;

        Ok(device.limits.clone())
    }

    pub fn device_downlevel_properties<A: HalApi>(
        &self,
        device_id: id::DeviceId,
    ) -> Result<wgt::DownlevelCapabilities, InvalidDevice> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;

        Ok(device.downlevel.clone())
    }

    pub fn device_create_buffer<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::BufferDescriptor,
        id_in: Input<G, id::BufferId>,
    ) -> (id::BufferId, Option<resource::CreateBufferError>) {
        profiling::scope!("create_buffer", "Device");

        let hub = A::hub(self);
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
                if mapped_at_creation && !desc.usage.contains(wgt::BufferUsages::MAP_WRITE) {
                    desc.usage |= wgt::BufferUsages::COPY_DST;
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
                hal::BufferUses::empty()
            } else if desc.usage.contains(wgt::BufferUsages::MAP_WRITE) {
                // buffer is mappable, so we are just doing that at start
                let map_size = buffer.size;
                let ptr = match map_buffer(&device.raw, &mut buffer, 0, map_size, HostMap::Write) {
                    Ok(ptr) => ptr,
                    Err(e) => {
                        let raw = buffer.raw.unwrap();
                        device
                            .lock_life(&mut token)
                            .schedule_resource_destruction(queue::TempResource::Buffer(raw), !0);
                        break e.into();
                    }
                };
                buffer.map_state = resource::BufferMapState::Active {
                    ptr,
                    range: 0..map_size,
                    host: HostMap::Write,
                };
                hal::BufferUses::MAP_WRITE
            } else {
                // buffer needs staging area for initialization only
                let stage_desc = wgt::BufferDescriptor {
                    label: Some(Cow::Borrowed("<init_buffer>")),
                    size: desc.size,
                    usage: wgt::BufferUsages::MAP_WRITE | wgt::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                };
                let mut stage = match device.create_buffer(device_id, &stage_desc, true) {
                    Ok(stage) => stage,
                    Err(e) => {
                        let raw = buffer.raw.unwrap();
                        device
                            .lock_life(&mut token)
                            .schedule_resource_destruction(queue::TempResource::Buffer(raw), !0);
                        break e;
                    }
                };
                let stage_buffer = stage.raw.unwrap();
                let mapping = match unsafe { device.raw.map_buffer(&stage_buffer, 0..stage.size) } {
                    Ok(mapping) => mapping,
                    Err(e) => {
                        let raw = buffer.raw.unwrap();
                        let mut life_lock = device.lock_life(&mut token);
                        life_lock
                            .schedule_resource_destruction(queue::TempResource::Buffer(raw), !0);
                        life_lock.schedule_resource_destruction(
                            queue::TempResource::Buffer(stage_buffer),
                            !0,
                        );
                        break DeviceError::from(e).into();
                    }
                };

                // Zero initialize memory and then mark both staging and buffer as initialized
                // (it's guaranteed that this is the case by the time the buffer is usable)
                unsafe { ptr::write_bytes(mapping.ptr.as_ptr(), 0, buffer.size as usize) };
                buffer.initialization_status.clear(0..buffer.size);
                stage.initialization_status.clear(0..buffer.size);

                buffer.map_state = resource::BufferMapState::Init {
                    ptr: mapping.ptr,
                    needs_flush: !mapping.is_coherent,
                    stage_buffer,
                };
                hal::BufferUses::COPY_DST
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
    pub fn device_wait_for_buffer<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        buffer_id: id::BufferId,
    ) -> Result<(), WaitIdleError> {
        let hub = A::hub(self);
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

    #[doc(hidden)]
    pub fn device_set_buffer_sub_data<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        data: &[u8],
    ) -> Result<(), resource::BufferAccessError> {
        profiling::scope!("set_buffer_sub_data", "Device");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut buffer_guard, _) = hub.buffers.write(&mut token);
        let device = device_guard
            .get(device_id)
            .map_err(|_| DeviceError::Invalid)?;
        let buffer = buffer_guard
            .get_mut(buffer_id)
            .map_err(|_| resource::BufferAccessError::Invalid)?;
        check_buffer_usage(buffer.usage, wgt::BufferUsages::MAP_WRITE)?;
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

        let raw_buf = buffer.raw.as_ref().unwrap();
        unsafe {
            let mapping = device
                .raw
                .map_buffer(raw_buf, offset..offset + data.len() as u64)
                .map_err(DeviceError::from)?;
            ptr::copy_nonoverlapping(data.as_ptr(), mapping.ptr.as_ptr(), data.len());
            if !mapping.is_coherent {
                device
                    .raw
                    .flush_mapped_ranges(raw_buf, iter::once(offset..offset + data.len() as u64));
            }
            device
                .raw
                .unmap_buffer(raw_buf)
                .map_err(DeviceError::from)?;
        }

        Ok(())
    }

    #[doc(hidden)]
    pub fn device_get_buffer_sub_data<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        data: &mut [u8],
    ) -> Result<(), resource::BufferAccessError> {
        profiling::scope!("get_buffer_sub_data", "Device");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (mut buffer_guard, _) = hub.buffers.write(&mut token);
        let device = device_guard
            .get(device_id)
            .map_err(|_| DeviceError::Invalid)?;
        let buffer = buffer_guard
            .get_mut(buffer_id)
            .map_err(|_| resource::BufferAccessError::Invalid)?;
        check_buffer_usage(buffer.usage, wgt::BufferUsages::MAP_READ)?;
        //assert!(buffer isn't used by the GPU);

        let raw_buf = buffer.raw.as_ref().unwrap();
        unsafe {
            let mapping = device
                .raw
                .map_buffer(raw_buf, offset..offset + data.len() as u64)
                .map_err(DeviceError::from)?;
            if !mapping.is_coherent {
                device.raw.invalidate_mapped_ranges(
                    raw_buf,
                    iter::once(offset..offset + data.len() as u64),
                );
            }
            ptr::copy_nonoverlapping(mapping.ptr.as_ptr(), data.as_mut_ptr(), data.len());
            device
                .raw
                .unmap_buffer(raw_buf)
                .map_err(DeviceError::from)?;
        }

        Ok(())
    }

    pub fn buffer_label<A: HalApi>(&self, id: id::BufferId) -> String {
        A::hub(self).buffers.label_for_resource(id)
    }

    pub fn buffer_destroy<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
    ) -> Result<(), resource::DestroyError> {
        profiling::scope!("destroy", "Buffer");

        let hub = A::hub(self);
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

        let raw = buffer
            .raw
            .take()
            .ok_or(resource::DestroyError::AlreadyDestroyed)?;
        let temp = queue::TempResource::Buffer(raw);

        if device.pending_writes.dst_buffers.contains(&buffer_id) {
            device.pending_writes.temp_resources.push(temp);
        } else {
            let last_submit_index = buffer.life_guard.submission_index.load(Ordering::Acquire);
            drop(buffer_guard);
            device
                .lock_life(&mut token)
                .schedule_resource_destruction(temp, last_submit_index);
        }

        Ok(())
    }

    pub fn buffer_drop<A: HalApi>(&self, buffer_id: id::BufferId, wait: bool) {
        profiling::scope!("drop", "Buffer");

        let hub = A::hub(self);
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

    pub fn device_create_texture<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::TextureDescriptor,
        id_in: Input<G, id::TextureId>,
    ) -> (id::TextureId, Option<resource::CreateTextureError>) {
        profiling::scope!("create_texture", "Device");

        let hub = A::hub(self);
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

    /// # Safety
    ///
    /// - `hal_texture` must be created from `device_id` corresponding raw handle.
    /// - `hal_texture` must be created respecting `desc`
    pub unsafe fn create_texture_from_hal<A: HalApi>(
        &self,
        hal_texture: A::Texture,
        device_id: id::DeviceId,
        desc: &resource::TextureDescriptor,
        id_in: Input<G, id::TextureId>,
    ) -> (id::TextureId, Option<resource::CreateTextureError>) {
        profiling::scope!("create_texture", "Device");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.textures.prepare(id_in);

        let (adapter_guard, mut token) = hub.adapters.read(&mut token);
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };

            // NB: Any change done through the raw texture handle will not be recorded in the replay
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::CreateTexture(fid.id(), desc.clone()));
            }

            let adapter = &adapter_guard[device.adapter_id.value];
            let texture =
                match device.create_texture_from_hal(hal_texture, device_id, adapter, desc) {
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

    pub fn texture_label<A: HalApi>(&self, id: id::TextureId) -> String {
        A::hub(self).textures.label_for_resource(id)
    }

    pub fn texture_destroy<A: HalApi>(
        &self,
        texture_id: id::TextureId,
    ) -> Result<(), resource::DestroyError> {
        profiling::scope!("destroy", "Texture");

        let hub = A::hub(self);
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

        match texture.inner {
            resource::TextureInner::Native { ref mut raw } => {
                let raw = raw.take().ok_or(resource::DestroyError::AlreadyDestroyed)?;
                let temp = queue::TempResource::Texture(raw);

                if device.pending_writes.dst_textures.contains(&texture_id) {
                    device.pending_writes.temp_resources.push(temp);
                } else {
                    let last_submit_index =
                        texture.life_guard.submission_index.load(Ordering::Acquire);
                    drop(texture_guard);
                    device
                        .lock_life(&mut token)
                        .schedule_resource_destruction(temp, last_submit_index);
                }
            }
            resource::TextureInner::Surface { .. } => {} //TODO
        }

        Ok(())
    }

    pub fn texture_drop<A: HalApi>(&self, texture_id: id::TextureId, wait: bool) {
        profiling::scope!("drop", "Texture");

        let hub = A::hub(self);
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

    pub fn texture_create_view<A: HalApi>(
        &self,
        texture_id: id::TextureId,
        desc: &resource::TextureViewDescriptor,
        id_in: Input<G, id::TextureViewId>,
    ) -> (id::TextureViewId, Option<resource::CreateTextureViewError>) {
        profiling::scope!("create_view", "Texture");

        let hub = A::hub(self);
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

    pub fn texture_view_label<A: HalApi>(&self, id: id::TextureViewId) -> String {
        A::hub(self).texture_views.label_for_resource(id)
    }

    pub fn texture_view_drop<A: HalApi>(
        &self,
        texture_view_id: id::TextureViewId,
        wait: bool,
    ) -> Result<(), resource::TextureViewDestroyError> {
        profiling::scope!("drop", "TextureView");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (last_submit_index, device_id) = {
            let (texture_guard, mut token) = hub.textures.read(&mut token);
            let (mut texture_view_guard, _) = hub.texture_views.write(&mut token);

            match texture_view_guard.get_mut(texture_view_id) {
                Ok(view) => {
                    let _ref_count = view.life_guard.ref_count.take();
                    let last_submit_index =
                        view.life_guard.submission_index.load(Ordering::Acquire);
                    let device_id = texture_guard[view.parent_id.value].device_id.value;
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

    pub fn device_create_sampler<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::SamplerDescriptor,
        id_in: Input<G, id::SamplerId>,
    ) -> (id::SamplerId, Option<resource::CreateSamplerError>) {
        profiling::scope!("create_sampler", "Device");

        let hub = A::hub(self);
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

    pub fn sampler_label<A: HalApi>(&self, id: id::SamplerId) -> String {
        A::hub(self).samplers.label_for_resource(id)
    }

    pub fn sampler_drop<A: HalApi>(&self, sampler_id: id::SamplerId) {
        profiling::scope!("drop", "Sampler");

        let hub = A::hub(self);
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

    pub fn device_create_bind_group_layout<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &binding_model::BindGroupLayoutDescriptor,
        id_in: Input<G, id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::CreateBindGroupLayoutError>,
    ) {
        profiling::scope!("create_bind_group_layout", "Device");

        let mut token = Token::root();
        let hub = A::hub(self);
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
                if entry_map.insert(entry.binding, *entry).is_some() {
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
                desc.label.borrow_option(),
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

    pub fn bind_group_layout_label<A: HalApi>(&self, id: id::BindGroupLayoutId) -> String {
        A::hub(self).bind_group_layouts.label_for_resource(id)
    }

    pub fn bind_group_layout_drop<A: HalApi>(&self, bind_group_layout_id: id::BindGroupLayoutId) {
        profiling::scope!("drop", "BindGroupLayout");

        let hub = A::hub(self);
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

    pub fn device_create_pipeline_layout<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &binding_model::PipelineLayoutDescriptor,
        id_in: Input<G, id::PipelineLayoutId>,
    ) -> (
        id::PipelineLayoutId,
        Option<binding_model::CreatePipelineLayoutError>,
    ) {
        profiling::scope!("create_pipeline_layout", "Device");

        let hub = A::hub(self);
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

    pub fn pipeline_layout_label<A: HalApi>(&self, id: id::PipelineLayoutId) -> String {
        A::hub(self).pipeline_layouts.label_for_resource(id)
    }

    pub fn pipeline_layout_drop<A: HalApi>(&self, pipeline_layout_id: id::PipelineLayoutId) {
        profiling::scope!("drop", "PipelineLayout");

        let hub = A::hub(self);
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

    pub fn device_create_bind_group<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &binding_model::BindGroupDescriptor,
        id_in: Input<G, id::BindGroupId>,
    ) -> (id::BindGroupId, Option<binding_model::CreateBindGroupError>) {
        profiling::scope!("create_bind_group", "Device");

        let hub = A::hub(self);
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
            let bind_group =
                match device.create_bind_group(device_id, bind_group_layout, desc, hub, &mut token)
                {
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

    pub fn bind_group_label<A: HalApi>(&self, id: id::BindGroupId) -> String {
        A::hub(self).bind_groups.label_for_resource(id)
    }

    pub fn bind_group_drop<A: HalApi>(&self, bind_group_id: id::BindGroupId) {
        profiling::scope!("drop", "BindGroup");

        let hub = A::hub(self);
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

    pub fn device_create_shader_module<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &pipeline::ShaderModuleDescriptor,
        source: pipeline::ShaderModuleSource,
        id_in: Input<G, id::ShaderModuleId>,
    ) -> (
        id::ShaderModuleId,
        Option<pipeline::CreateShaderModuleError>,
    ) {
        profiling::scope!("create_shader_module", "Device");

        let hub = A::hub(self);
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
                    pipeline::ShaderModuleSource::Wgsl(ref code) => {
                        trace.make_binary("wgsl", code.as_bytes())
                    }
                    pipeline::ShaderModuleSource::Naga(ref module) => {
                        let string =
                            ron::ser::to_string_pretty(module, ron::ser::PrettyConfig::default())
                                .unwrap();
                        trace.make_binary("ron", string.as_bytes())
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

    #[allow(unused_unsafe)] // Unsafe-ness of internal calls has little to do with unsafe-ness of this.
    /// # Safety
    ///
    /// This function passes SPIR-V binary to the backend as-is and can potentially result in a
    /// driver crash.
    pub unsafe fn device_create_shader_module_spirv<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &pipeline::ShaderModuleDescriptor,
        source: Cow<[u32]>,
        id_in: Input<G, id::ShaderModuleId>,
    ) -> (
        id::ShaderModuleId,
        Option<pipeline::CreateShaderModuleError>,
    ) {
        profiling::scope!("create_shader_module", "Device");

        let hub = A::hub(self);
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
                let data = trace.make_binary("spv", unsafe {
                    std::slice::from_raw_parts(source.as_ptr() as *const u8, source.len() * 4)
                });
                trace.add(trace::Action::CreateShaderModule {
                    id: fid.id(),
                    desc: desc.clone(),
                    data,
                });
            };

            let shader = match device.create_shader_module_spirv(device_id, desc, &source) {
                Ok(shader) => shader,
                Err(e) => break e,
            };
            let id = fid.assign(shader, &mut token);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn shader_module_label<A: HalApi>(&self, id: id::ShaderModuleId) -> String {
        A::hub(self).shader_modules.label_for_resource(id)
    }

    pub fn shader_module_drop<A: HalApi>(&self, shader_module_id: id::ShaderModuleId) {
        profiling::scope!("drop", "ShaderModule");

        let hub = A::hub(self);
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

    pub fn device_create_command_encoder<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &wgt::CommandEncoderDescriptor<Label>,
        id_in: Input<G, id::CommandEncoderId>,
    ) -> (id::CommandEncoderId, Option<DeviceError>) {
        profiling::scope!("create_command_encoder", "Device");

        let hub = A::hub(self);
        let mut token = Token::root();
        let fid = hub.command_buffers.prepare(id_in);

        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid,
            };
            let dev_stored = Stored {
                value: id::Valid(device_id),
                ref_count: device.life_guard.add_ref(),
            };
            let encoder = match device
                .command_allocator
                .lock()
                .acquire_encoder(&device.raw, &device.queue)
            {
                Ok(raw) => raw,
                Err(_) => break DeviceError::OutOfMemory,
            };
            let command_buffer = command::CommandBuffer::new(
                encoder,
                dev_stored,
                device.limits.clone(),
                device.downlevel.clone(),
                device.features,
                #[cfg(feature = "trace")]
                device.trace.is_some(),
                &desc.label,
            );

            let id = fid.assign(command_buffer, &mut token);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    pub fn command_buffer_label<A: HalApi>(&self, id: id::CommandBufferId) -> String {
        A::hub(self).command_buffers.label_for_resource(id)
    }

    pub fn command_encoder_drop<A: HalApi>(&self, command_encoder_id: id::CommandEncoderId) {
        profiling::scope!("drop", "CommandEncoder");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let (cmdbuf, _) = hub
            .command_buffers
            .unregister(command_encoder_id, &mut token);
        if let Some(cmdbuf) = cmdbuf {
            let device = &mut device_guard[cmdbuf.device_id.value];
            device.untrack::<G>(hub, &cmdbuf.trackers, &mut token);
        }
    }

    pub fn command_buffer_drop<A: HalApi>(&self, command_buffer_id: id::CommandBufferId) {
        profiling::scope!("drop", "CommandBuffer");
        self.command_encoder_drop::<A>(command_buffer_id)
    }

    pub fn device_create_render_bundle_encoder(
        &self,
        device_id: id::DeviceId,
        desc: &command::RenderBundleEncoderDescriptor,
    ) -> (
        id::RenderBundleEncoderId,
        Option<command::CreateRenderBundleError>,
    ) {
        profiling::scope!("create_render_bundle_encoder", "Device");
        let (encoder, error) = match command::RenderBundleEncoder::new(desc, device_id, None) {
            Ok(encoder) => (encoder, None),
            Err(e) => (command::RenderBundleEncoder::dummy(device_id), Some(e)),
        };
        (Box::into_raw(Box::new(encoder)), error)
    }

    pub fn render_bundle_encoder_finish<A: HalApi>(
        &self,
        bundle_encoder: command::RenderBundleEncoder,
        desc: &command::RenderBundleDescriptor,
        id_in: Input<G, id::RenderBundleId>,
    ) -> (id::RenderBundleId, Option<command::RenderBundleError>) {
        profiling::scope!("finish", "RenderBundleEncoder");

        let hub = A::hub(self);
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
                        bundle_encoder.is_ds_read_only,
                    ),
                    base: bundle_encoder.to_base_pass(),
                });
            }

            let render_bundle = match bundle_encoder.finish(desc, device, hub, &mut token) {
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

    pub fn render_bundle_label<A: HalApi>(&self, id: id::RenderBundleId) -> String {
        A::hub(self).render_bundles.label_for_resource(id)
    }

    pub fn render_bundle_drop<A: HalApi>(&self, render_bundle_id: id::RenderBundleId) {
        profiling::scope!("drop", "RenderBundle");
        let hub = A::hub(self);
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

    pub fn device_create_query_set<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &resource::QuerySetDescriptor,
        id_in: Input<G, id::QuerySetId>,
    ) -> (id::QuerySetId, Option<resource::CreateQuerySetError>) {
        profiling::scope!("create_query_set", "Device");

        let hub = A::hub(self);
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

            let query_set = match device.create_query_set(device_id, desc) {
                Ok(query_set) => query_set,
                Err(err) => break err,
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

    pub fn query_set_drop<A: HalApi>(&self, query_set_id: id::QuerySetId) {
        profiling::scope!("drop", "QuerySet");

        let hub = A::hub(self);
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

    pub fn device_create_render_pipeline<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &pipeline::RenderPipelineDescriptor,
        id_in: Input<G, id::RenderPipelineId>,
        implicit_pipeline_ids: Option<ImplicitPipelineIds<G>>,
    ) -> (
        id::RenderPipelineId,
        Option<pipeline::CreateRenderPipelineError>,
    ) {
        profiling::scope!("create_render_pipeline", "Device");

        let hub = A::hub(self);
        let mut token = Token::root();

        let fid = hub.render_pipelines.prepare(id_in);
        let implicit_context = implicit_pipeline_ids.map(|ipi| ipi.prepare(hub));

        let (adapter_guard, mut token) = hub.adapters.read(&mut token);
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            let adapter = &adapter_guard[device.adapter_id.value];
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace.lock().add(trace::Action::CreateRenderPipeline {
                    id: fid.id(),
                    desc: desc.clone(),
                    implicit_context: implicit_context.clone(),
                });
            }

            let pipeline = match device.create_render_pipeline(
                device_id,
                adapter,
                desc,
                implicit_context,
                hub,
                &mut token,
            ) {
                Ok(pair) => pair,
                Err(e) => break e,
            };

            let id = fid.assign(pipeline, &mut token);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    /// Get an ID of one of the bind group layouts. The ID adds a refcount,
    /// which needs to be released by calling `bind_group_layout_drop`.
    pub fn render_pipeline_get_bind_group_layout<A: HalApi>(
        &self,
        pipeline_id: id::RenderPipelineId,
        index: u32,
        id_in: Input<G, id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::GetBindGroupLayoutError>,
    ) {
        let hub = A::hub(self);
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

    pub fn render_pipeline_label<A: HalApi>(&self, id: id::RenderPipelineId) -> String {
        A::hub(self).render_pipelines.label_for_resource(id)
    }

    pub fn render_pipeline_drop<A: HalApi>(&self, render_pipeline_id: id::RenderPipelineId) {
        profiling::scope!("drop", "RenderPipeline");
        let hub = A::hub(self);
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

    pub fn device_create_compute_pipeline<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        desc: &pipeline::ComputePipelineDescriptor,
        id_in: Input<G, id::ComputePipelineId>,
        implicit_pipeline_ids: Option<ImplicitPipelineIds<G>>,
    ) -> (
        id::ComputePipelineId,
        Option<pipeline::CreateComputePipelineError>,
    ) {
        profiling::scope!("create_compute_pipeline", "Device");

        let hub = A::hub(self);
        let mut token = Token::root();

        let fid = hub.compute_pipelines.prepare(id_in);
        let implicit_context = implicit_pipeline_ids.map(|ipi| ipi.prepare(hub));

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

            let pipeline = match device.create_compute_pipeline(
                device_id,
                desc,
                implicit_context,
                hub,
                &mut token,
            ) {
                Ok(pair) => pair,
                Err(e) => break e,
            };

            let id = fid.assign(pipeline, &mut token);
            return (id.0, None);
        };

        let id = fid.assign_error(desc.label.borrow_or_default(), &mut token);
        (id, Some(error))
    }

    /// Get an ID of one of the bind group layouts. The ID adds a refcount,
    /// which needs to be released by calling `bind_group_layout_drop`.
    pub fn compute_pipeline_get_bind_group_layout<A: HalApi>(
        &self,
        pipeline_id: id::ComputePipelineId,
        index: u32,
        id_in: Input<G, id::BindGroupLayoutId>,
    ) -> (
        id::BindGroupLayoutId,
        Option<binding_model::GetBindGroupLayoutError>,
    ) {
        let hub = A::hub(self);
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

    pub fn compute_pipeline_label<A: HalApi>(&self, id: id::ComputePipelineId) -> String {
        A::hub(self).compute_pipelines.label_for_resource(id)
    }

    pub fn compute_pipeline_drop<A: HalApi>(&self, compute_pipeline_id: id::ComputePipelineId) {
        profiling::scope!("drop", "ComputePipeline");
        let hub = A::hub(self);
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

    pub fn surface_configure<A: HalApi>(
        &self,
        surface_id: id::SurfaceId,
        device_id: id::DeviceId,
        config: &wgt::SurfaceConfiguration,
    ) -> Option<present::ConfigureSurfaceError> {
        use hal::{Adapter as _, Surface as _};
        use present::ConfigureSurfaceError as E;
        profiling::scope!("surface_configure");

        fn validate_surface_configuraiton(
            config: &mut hal::SurfaceConfiguration,
            caps: &hal::SurfaceCapabilities,
        ) -> Result<(), E> {
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
            if !caps.present_modes.contains(&config.present_mode) {
                log::warn!(
                    "Surface does not support present mode: {:?}, falling back to FIFO",
                    config.present_mode,
                );
                config.present_mode = wgt::PresentMode::Fifo;
            }
            if !caps.formats.contains(&config.format) {
                return Err(E::UnsupportedFormat {
                    requested: config.format,
                    available: caps.formats.clone(),
                });
            }
            if !caps.usage.contains(config.usage) {
                return Err(E::UnsupportedUsage);
            }
            if width == 0 || height == 0 {
                return Err(E::ZeroArea);
            }
            Ok(())
        }

        log::info!("configuring surface with {:?}", config);
        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut surface_guard, mut token) = self.surfaces.write(&mut token);
        let (adapter_guard, mut token) = hub.adapters.read(&mut token);
        let (device_guard, _token) = hub.devices.read(&mut token);

        let error = loop {
            let device = match device_guard.get(device_id) {
                Ok(device) => device,
                Err(_) => break DeviceError::Invalid.into(),
            };
            #[cfg(feature = "trace")]
            if let Some(ref trace) = device.trace {
                trace
                    .lock()
                    .add(trace::Action::ConfigureSurface(surface_id, config.clone()));
            }

            let surface = match surface_guard.get_mut(surface_id) {
                Ok(surface) => surface,
                Err(_) => break E::InvalidSurface,
            };

            let caps = unsafe {
                let suf = A::get_surface(surface);
                let adapter = &adapter_guard[device.adapter_id.value];
                match adapter.raw.adapter.surface_capabilities(&suf.raw) {
                    Some(caps) => caps,
                    None => break E::UnsupportedQueueFamily,
                }
            };

            let num_frames = present::DESIRED_NUM_FRAMES
                .max(*caps.swap_chain_sizes.start())
                .min(*caps.swap_chain_sizes.end());
            let mut hal_config = hal::SurfaceConfiguration {
                swap_chain_size: num_frames,
                present_mode: config.present_mode,
                composite_alpha_mode: hal::CompositeAlphaMode::Opaque,
                format: config.format,
                extent: wgt::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                usage: conv::map_texture_usage(config.usage, hal::FormatAspects::COLOR),
            };

            if let Err(error) = validate_surface_configuraiton(&mut hal_config, &caps) {
                break error;
            }

            match unsafe {
                A::get_surface_mut(surface)
                    .raw
                    .configure(&device.raw, &hal_config)
            } {
                Ok(()) => (),
                Err(error) => {
                    break match error {
                        hal::SurfaceError::Outdated | hal::SurfaceError::Lost => E::InvalidSurface,
                        hal::SurfaceError::Device(error) => E::Device(error.into()),
                        hal::SurfaceError::Other(message) => {
                            log::error!("surface configuration failed: {}", message);
                            E::InvalidSurface
                        }
                    }
                }
            }

            if let Some(present) = surface.presentation.take() {
                if present.acquired_texture.is_some() {
                    break E::PreviousOutputExists;
                }
            }

            surface.presentation = Some(present::Presentation {
                device_id: Stored {
                    value: id::Valid(device_id),
                    ref_count: device.life_guard.add_ref(),
                },
                config: config.clone(),
                num_frames,
                acquired_texture: None,
            });

            return None;
        };

        Some(error)
    }

    #[cfg(feature = "replay")]
    /// Only triange suspected resource IDs. This helps us to avoid ID collisions
    /// upon creating new resources when re-playing a trace.
    pub fn device_maintain_ids<A: HalApi>(
        &self,
        device_id: id::DeviceId,
    ) -> Result<(), InvalidDevice> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let device = device_guard.get(device_id).map_err(|_| InvalidDevice)?;
        device.lock_life(&mut token).triage_suspected(
            hub,
            &device.trackers,
            #[cfg(feature = "trace")]
            None,
            &mut token,
        );
        Ok(())
    }

    pub fn device_poll<A: HalApi>(
        &self,
        device_id: id::DeviceId,
        force_wait: bool,
    ) -> Result<(), WaitIdleError> {
        let closures = {
            let hub = A::hub(self);
            let mut token = Token::root();
            let (device_guard, mut token) = hub.devices.read(&mut token);
            device_guard
                .get(device_id)
                .map_err(|_| DeviceError::Invalid)?
                .maintain(hub, force_wait, &mut token)?
        };
        unsafe {
            closures.fire();
        }
        Ok(())
    }

    fn poll_devices<A: HalApi>(
        &self,
        force_wait: bool,
        closures: &mut UserClosures,
    ) -> Result<(), WaitIdleError> {
        profiling::scope!("poll_devices");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        for (_, device) in device_guard.iter(A::VARIANT) {
            let cbs = device.maintain(hub, force_wait, &mut token)?;
            closures.extend(cbs);
        }
        Ok(())
    }

    pub fn poll_all_devices(&self, force_wait: bool) -> Result<(), WaitIdleError> {
        let mut closures = UserClosures::default();

        #[cfg(vulkan)]
        {
            self.poll_devices::<hal::api::Vulkan>(force_wait, &mut closures)?;
        }
        #[cfg(metal)]
        {
            self.poll_devices::<hal::api::Metal>(force_wait, &mut closures)?;
        }
        #[cfg(dx12)]
        {
            self.poll_devices::<hal::api::Dx12>(force_wait, &mut closures)?;
        }
        #[cfg(dx11)]
        {
            self.poll_devices::<hal::api::Dx11>(force_wait, &mut closures)?;
        }

        unsafe {
            closures.fire();
        }

        Ok(())
    }

    pub fn device_label<A: HalApi>(&self, id: id::DeviceId) -> String {
        A::hub(self).devices.label_for_resource(id)
    }

    pub fn device_start_capture<A: HalApi>(&self, id: id::DeviceId) {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        if let Ok(device) = device_guard.get(id) {
            unsafe { device.raw.start_capture() };
        }
    }

    pub fn device_stop_capture<A: HalApi>(&self, id: id::DeviceId) {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        if let Ok(device) = device_guard.get(id) {
            unsafe { device.raw.stop_capture() };
        }
    }

    pub fn device_drop<A: HalApi>(&self, device_id: id::DeviceId) {
        profiling::scope!("drop", "Device");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (device, _) = hub.devices.unregister(device_id, &mut token);
        if let Some(mut device) = device {
            device.prepare_to_die();

            // Adapter is only referenced by the device and itself.
            // This isn't a robust way to destroy them, we should find a better one.
            if device.adapter_id.ref_count.load() == 1 {
                let _ = hub
                    .adapters
                    .unregister(device.adapter_id.value.0, &mut token);
            }

            device.dispose();
        }
    }

    pub fn buffer_map_async<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
        range: Range<BufferAddress>,
        op: resource::BufferMapOperation,
    ) -> Result<(), resource::BufferAccessError> {
        profiling::scope!("map_async", "Buffer");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, mut token) = hub.devices.read(&mut token);
        let (pub_usage, internal_use) = match op.host {
            HostMap::Read => (wgt::BufferUsages::MAP_READ, hal::BufferUses::MAP_READ),
            HostMap::Write => (wgt::BufferUsages::MAP_WRITE, hal::BufferUses::MAP_WRITE),
        };

        if range.start % wgt::MAP_ALIGNMENT != 0 || range.end % wgt::COPY_BUFFER_ALIGNMENT != 0 {
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

    pub fn buffer_get_mapped_range<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) -> Result<(*mut u8, u64), resource::BufferAccessError> {
        profiling::scope!("get_mapped_range", "Buffer");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (buffer_guard, _) = hub.buffers.read(&mut token);
        let buffer = buffer_guard
            .get(buffer_id)
            .map_err(|_| resource::BufferAccessError::Invalid)?;

        let range_size = if let Some(size) = size {
            size
        } else if offset > buffer.size {
            0
        } else {
            buffer.size - offset
        };

        if offset % wgt::MAP_ALIGNMENT != 0 {
            return Err(resource::BufferAccessError::UnalignedOffset { offset });
        }
        if range_size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
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
            resource::BufferMapState::Active { ptr, ref range, .. } => {
                if offset < range.start {
                    return Err(resource::BufferAccessError::OutOfBoundsUnderrun {
                        index: offset,
                        min: range.start,
                    });
                }
                if offset + range_size > range.end {
                    return Err(resource::BufferAccessError::OutOfBoundsOverrun {
                        index: offset + range_size - 1,
                        max: range.end,
                    });
                }
                unsafe { Ok((ptr.as_ptr().offset(offset as isize), range_size)) }
            }
            resource::BufferMapState::Idle | resource::BufferMapState::Waiting(_) => {
                Err(resource::BufferAccessError::NotMapped)
            }
        }
    }

    fn buffer_unmap_inner<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
    ) -> Result<Option<BufferMapPendingClosure>, resource::BufferAccessError> {
        profiling::scope!("unmap", "Buffer");

        let hub = A::hub(self);
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
                    unsafe {
                        device
                            .raw
                            .flush_mapped_ranges(&stage_buffer, iter::once(0..buffer.size));
                    }
                }

                let raw_buf = buffer
                    .raw
                    .as_ref()
                    .ok_or(resource::BufferAccessError::Destroyed)?;

                buffer.life_guard.use_at(device.active_submission_index + 1);
                let region = wgt::BufferSize::new(buffer.size).map(|size| hal::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                });
                let transition_src = hal::BufferBarrier {
                    buffer: &stage_buffer,
                    usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
                };
                let transition_dst = hal::BufferBarrier {
                    buffer: raw_buf,
                    usage: hal::BufferUses::empty()..hal::BufferUses::COPY_DST,
                };
                let encoder = device.pending_writes.activate();
                unsafe {
                    encoder.transition_buffers(
                        iter::once(transition_src).chain(iter::once(transition_dst)),
                    );
                    if buffer.size > 0 {
                        encoder.copy_buffer_to_buffer(&stage_buffer, raw_buf, region.into_iter());
                    }
                }
                device
                    .pending_writes
                    .consume_temp(queue::TempResource::Buffer(stage_buffer));
                device.pending_writes.dst_buffers.insert(buffer_id);
            }
            resource::BufferMapState::Idle => {
                return Err(resource::BufferAccessError::NotMapped);
            }
            resource::BufferMapState::Waiting(pending) => {
                return Ok(Some((pending.op, resource::BufferMapAsyncStatus::Aborted)));
            }
            resource::BufferMapState::Active { ptr, range, host } => {
                if host == HostMap::Write {
                    #[cfg(feature = "trace")]
                    if let Some(ref trace) = device.trace {
                        let mut trace = trace.lock();
                        let size = range.end - range.start;
                        let data = trace.make_binary("bin", unsafe {
                            std::slice::from_raw_parts(ptr.as_ptr(), size as usize)
                        });
                        trace.add(trace::Action::WriteBuffer {
                            id: buffer_id,
                            data,
                            range: range.clone(),
                            queued: false,
                        });
                    }
                    let _ = (ptr, range);
                }
                unsafe {
                    device
                        .raw
                        .unmap_buffer(buffer.raw.as_ref().unwrap())
                        .map_err(DeviceError::from)?
                };
            }
        }
        Ok(None)
    }

    pub fn buffer_unmap<A: HalApi>(
        &self,
        buffer_id: id::BufferId,
    ) -> Result<(), resource::BufferAccessError> {
        //Note: outside inner function so no locks are held when calling the callback
        let closure = self.buffer_unmap_inner::<A>(buffer_id)?;
        if let Some((operation, status)) = closure {
            unsafe {
                (operation.callback)(status, operation.user_data);
            }
        }
        Ok(())
    }
}
