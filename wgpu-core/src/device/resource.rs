#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    binding_model::{
        self, get_bind_group_layout, try_get_bind_group_layout, BglOrDuplicate,
        BindGroupLayoutInner,
    },
    command, conv,
    device::life::WaitIdleError,
    device::{
        AttachmentData, CommandAllocator, DeviceLostInvocation, MissingDownlevelFlags,
        MissingFeatures, RenderPassContext, CLEANUP_WAIT_MS,
    },
    hal_api::HalApi,
    hal_label,
    hub::{Hub, Token},
    id,
    identity::GlobalIdentityHandlerFactory,
    init_tracker::{
        BufferInitTracker, BufferInitTrackerAction, MemoryInitKind, TextureInitRange,
        TextureInitTracker, TextureInitTrackerAction,
    },
    instance::Adapter,
    pipeline,
    resource::{self, Buffer, TextureViewNotRenderableReason},
    storage::Storage,
    track::{BindGroupStates, TextureSelector, Tracker},
    validation::{self, check_buffer_usage, check_texture_usage},
    FastHashMap, LabelHelpers as _, LifeGuard, MultiRefCount, RefCount, Stored, SubmissionIndex,
};

use arrayvec::ArrayVec;
use hal::{CommandEncoder as _, Device as _};
use parking_lot::{Mutex, MutexGuard};
use smallvec::SmallVec;
use thiserror::Error;
use wgt::{DeviceLostReason, TextureFormat, TextureSampleType, TextureViewDimension};

use std::{borrow::Cow, iter, num::NonZeroU32};

use super::{
    life, queue, DeviceDescriptor, DeviceError, ImplicitPipelineContext, UserClosures, EP_FAILURE,
    IMPLICIT_FAILURE, ZERO_BUFFER_SIZE,
};

/// Structure describing a logical device. Some members are internally mutable,
/// stored behind mutexes.
///
/// TODO: establish clear order of locking for these:
/// `mem_allocator`, `desc_allocator`, `life_tracker`, `trackers`,
/// `render_passes`, `pending_writes`, `trace`.
///
/// Currently, the rules are:
/// 1. `life_tracker` is locked after `hub.devices`, enforced by the type system
/// 1. `self.trackers` is locked last (unenforced)
/// 1. `self.trace` is locked last (unenforced)
pub struct Device<A: HalApi> {
    pub(crate) raw: A::Device,
    pub(crate) adapter_id: Stored<id::AdapterId>,
    pub(crate) queue: A::Queue,
    pub(crate) zero_buffer: A::Buffer,
    //pub(crate) cmd_allocator: command::CommandAllocator<A>,
    //mem_allocator: Mutex<alloc::MemoryAllocator<A>>,
    //desc_allocator: Mutex<descriptor::DescriptorAllocator<A>>,
    //Note: The submission index here corresponds to the last submission that is done.
    pub(crate) life_guard: LifeGuard,

    /// A clone of `life_guard.ref_count`.
    ///
    /// Holding a separate clone of the `RefCount` here lets us tell whether the
    /// device is referenced by other resources, even if `life_guard.ref_count`
    /// was set to `None` by a call to `device_drop`.
    pub(super) ref_count: RefCount,

    pub(super) command_allocator: Mutex<CommandAllocator<A>>,
    pub(crate) active_submission_index: SubmissionIndex,
    pub(super) fence: A::Fence,

    /// Is this device valid? Valid is closely associated with "lose the device",
    /// which can be triggered by various methods, including at the end of device
    /// destroy, and by any GPU errors that cause us to no longer trust the state
    /// of the device. Ideally we would like to fold valid into the storage of
    /// the device itself (for example as an Error enum), but unfortunately we
    /// need to continue to be able to retrieve the device in poll_devices to
    /// determine if it can be dropped. If our internal accesses of devices were
    /// done through ref-counted references and external accesses checked for
    /// Error enums, we wouldn't need this. For now, we need it. All the call
    /// sites where we check it are areas that should be revisited if we start
    /// using ref-counted references for internal access.
    pub(crate) valid: bool,

    /// All live resources allocated with this [`Device`].
    ///
    /// Has to be locked temporarily only (locked last)
    pub(crate) trackers: Mutex<Tracker<A>>,
    // Life tracker should be locked right after the device and before anything else.
    life_tracker: Mutex<life::LifetimeTracker<A>>,
    /// Temporary storage for resource management functions. Cleared at the end
    /// of every call (unless an error occurs).
    pub(super) temp_suspected: life::SuspectedResources,
    pub(crate) alignments: hal::Alignments,
    pub(crate) limits: wgt::Limits,
    pub(crate) features: wgt::Features,
    pub(crate) downlevel: wgt::DownlevelCapabilities,
    pub(crate) instance_flags: wgt::InstanceFlags,
    // TODO: move this behind another mutex. This would allow several methods to
    // switch to borrow Device immutably, such as `write_buffer`, `write_texture`,
    // and `buffer_unmap`.
    pub(super) pending_writes: queue::PendingWrites<A>,
    #[cfg(feature = "trace")]
    pub(crate) trace: Option<Mutex<trace::Trace>>,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateDeviceError {
    #[error("Not enough memory left")]
    OutOfMemory,
    #[error("Failed to create internal buffer for initializing textures")]
    FailedToCreateZeroBuffer(#[from] DeviceError),
}

impl<A: HalApi> Device<A> {
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
    pub(crate) fn new(
        open: hal::OpenDevice<A>,
        adapter_id: Stored<id::AdapterId>,
        alignments: hal::Alignments,
        downlevel: wgt::DownlevelCapabilities,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
        instance_flags: wgt::InstanceFlags,
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
        let mut pending_writes = queue::PendingWrites::<A>::new(pending_encoder);

        // Create zeroed buffer used for texture clears.
        let zero_buffer = unsafe {
            open.device
                .create_buffer(&hal::BufferDescriptor {
                    label: hal_label(Some("(wgpu internal) zero init buffer"), instance_flags),
                    size: ZERO_BUFFER_SIZE,
                    usage: hal::BufferUses::COPY_SRC | hal::BufferUses::COPY_DST,
                    memory_flags: hal::MemoryFlags::empty(),
                })
                .map_err(DeviceError::from)?
        };
        pending_writes.activate();
        unsafe {
            pending_writes
                .command_encoder
                .transition_buffers(iter::once(hal::BufferBarrier {
                    buffer: &zero_buffer,
                    usage: hal::BufferUses::empty()..hal::BufferUses::COPY_DST,
                }));
            pending_writes
                .command_encoder
                .clear_buffer(&zero_buffer, 0..ZERO_BUFFER_SIZE);
            pending_writes
                .command_encoder
                .transition_buffers(iter::once(hal::BufferBarrier {
                    buffer: &zero_buffer,
                    usage: hal::BufferUses::COPY_DST..hal::BufferUses::COPY_SRC,
                }));
        }

        let life_guard = LifeGuard::new("<device>");
        let ref_count = life_guard.add_ref();
        Ok(Self {
            raw: open.device,
            adapter_id,
            queue: open.queue,
            zero_buffer,
            life_guard,
            ref_count,
            command_allocator: Mutex::new(com_alloc),
            active_submission_index: 0,
            fence,
            valid: true,
            trackers: Mutex::new(Tracker::new()),
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
            instance_flags,
            pending_writes,
        })
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    pub(super) fn lock_life<'this, 'token: 'this>(
        &'this self,
        //TODO: fix this - the token has to be borrowed for the lock
        _token: &mut Token<'token, Self>,
    ) -> MutexGuard<'this, life::LifetimeTracker<A>> {
        self.life_tracker.lock()
    }

    /// Check this device for completed commands.
    ///
    /// The `maintain` argument tells how the maintence function should behave, either
    /// blocking or just polling the current state of the gpu.
    ///
    /// Return a pair `(closures, queue_empty)`, where:
    ///
    /// - `closures` is a list of actions to take: mapping buffers, notifying the user
    ///
    /// - `queue_empty` is a boolean indicating whether there are more queue
    ///   submissions still in flight. (We have to take the locks needed to
    ///   produce this information for other reasons, so we might as well just
    ///   return it to our callers.)
    pub(super) fn maintain<'this, 'token: 'this, G: GlobalIdentityHandlerFactory>(
        &'this self,
        hub: &Hub<A, G>,
        maintain: wgt::Maintain<queue::WrappedSubmissionIndex>,
        token: &mut Token<'token, Self>,
    ) -> Result<(UserClosures, bool), WaitIdleError> {
        profiling::scope!("Device::maintain");
        let mut life_tracker = self.lock_life(token);

        // Normally, `temp_suspected` exists only to save heap
        // allocations: it's cleared at the start of the function
        // call, and cleared by the end. But `Global::queue_submit` is
        // fallible; if it exits early, it may leave some resources in
        // `temp_suspected`.
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

        let last_done_index = if maintain.is_wait() {
            let index_to_wait_for = match maintain {
                wgt::Maintain::WaitForSubmissionIndex(submission_index) => {
                    // We don't need to check to see if the queue id matches
                    // as we already checked this from inside the poll call.
                    submission_index.index
                }
                _ => self.active_submission_index,
            };
            unsafe {
                self.raw
                    .wait(&self.fence, index_to_wait_for, CLEANUP_WAIT_MS)
                    .map_err(DeviceError::from)?
            };
            index_to_wait_for
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

        // Detect if we have been destroyed and now need to lose the device.
        // If we are invalid (set at start of destroy) and our queue is empty,
        // and we have a DeviceLostClosure, return the closure to be called by
        // our caller. This will complete the steps for both destroy and for
        // "lose the device".
        let mut device_lost_invocations = SmallVec::new();
        if !self.valid && life_tracker.queue_empty() && life_tracker.device_lost_closure.is_some() {
            device_lost_invocations.push(DeviceLostInvocation {
                closure: life_tracker.device_lost_closure.take().unwrap(),
                reason: DeviceLostReason::Destroyed,
                message: String::new(),
            });
        }

        let closures = UserClosures {
            mappings: mapping_closures,
            submissions: submission_closures,
            device_lost_invocations,
        };
        Ok((closures, life_tracker.queue_empty()))
    }

    pub(super) fn untrack<'this, 'token: 'this, G: GlobalIdentityHandlerFactory>(
        &'this mut self,
        hub: &Hub<A, G>,
        trackers: &Tracker<A>,
        token: &mut Token<'token, Self>,
    ) {
        self.temp_suspected.clear();
        // As the tracker is cleared/dropped, we need to consider all the resources
        // that it references for destruction in the next GC pass.
        {
            let (bind_group_guard, mut token) = hub.bind_groups.read(token);
            let (compute_pipe_guard, mut token) = hub.compute_pipelines.read(&mut token);
            let (render_pipe_guard, mut token) = hub.render_pipelines.read(&mut token);
            let (query_set_guard, mut token) = hub.query_sets.read(&mut token);
            let (buffer_guard, mut token) = hub.buffers.read(&mut token);
            let (texture_guard, mut token) = hub.textures.read(&mut token);
            let (texture_view_guard, mut token) = hub.texture_views.read(&mut token);
            let (sampler_guard, _) = hub.samplers.read(&mut token);

            for id in trackers.buffers.used() {
                if buffer_guard
                    .get_occupied_or_destroyed(id.0)
                    .unwrap()
                    .life_guard
                    .ref_count
                    .is_none()
                {
                    self.temp_suspected.buffers.push(id);
                }
            }
            for id in trackers.textures.used() {
                if texture_guard
                    .get_occupied_or_destroyed(id.0)
                    .unwrap()
                    .life_guard
                    .ref_count
                    .is_none()
                {
                    self.temp_suspected.textures.push(id);
                }
            }
            for id in trackers.views.used() {
                if texture_view_guard
                    .get_occupied_or_destroyed(id.0)
                    .unwrap()
                    .life_guard
                    .ref_count
                    .is_none()
                {
                    self.temp_suspected.texture_views.push(id);
                }
            }
            for id in trackers.bind_groups.used() {
                if bind_group_guard
                    .get_occupied_or_destroyed(id.0)
                    .unwrap()
                    .life_guard
                    .ref_count
                    .is_none()
                {
                    self.temp_suspected.bind_groups.push(id);
                }
            }
            for id in trackers.samplers.used() {
                if sampler_guard
                    .get_occupied_or_destroyed(id.0)
                    .unwrap()
                    .life_guard
                    .ref_count
                    .is_none()
                {
                    self.temp_suspected.samplers.push(id);
                }
            }
            for id in trackers.compute_pipelines.used() {
                if compute_pipe_guard
                    .get_occupied_or_destroyed(id.0)
                    .unwrap()
                    .life_guard
                    .ref_count
                    .is_none()
                {
                    self.temp_suspected.compute_pipelines.push(id);
                }
            }
            for id in trackers.render_pipelines.used() {
                if render_pipe_guard
                    .get_occupied_or_destroyed(id.0)
                    .unwrap()
                    .life_guard
                    .ref_count
                    .is_none()
                {
                    self.temp_suspected.render_pipelines.push(id);
                }
            }
            for id in trackers.query_sets.used() {
                if query_set_guard
                    .get_occupied_or_destroyed(id.0)
                    .unwrap()
                    .life_guard
                    .ref_count
                    .is_none()
                {
                    self.temp_suspected.query_sets.push(id);
                }
            }
        }

        self.lock_life(token)
            .suspected_resources
            .extend(&self.temp_suspected);

        self.temp_suspected.clear();
    }

    pub(super) fn create_buffer(
        &self,
        self_id: id::DeviceId,
        desc: &resource::BufferDescriptor,
        transient: bool,
    ) -> Result<Buffer<A>, resource::CreateBufferError> {
        debug_assert_eq!(self_id.backend(), A::VARIANT);

        if desc.size > self.limits.max_buffer_size {
            return Err(resource::CreateBufferError::MaxBufferSize {
                requested: desc.size,
                maximum: self.limits.max_buffer_size,
            });
        }

        if desc.usage.contains(wgt::BufferUsages::INDEX)
            && desc.usage.contains(
                wgt::BufferUsages::VERTEX
                    | wgt::BufferUsages::UNIFORM
                    | wgt::BufferUsages::INDIRECT
                    | wgt::BufferUsages::STORAGE,
            )
        {
            self.require_downlevel_flags(wgt::DownlevelFlags::UNRESTRICTED_INDEX_BUFFER)?;
        }

        let mut usage = conv::map_buffer_usage(desc.usage);

        if desc.usage.is_empty() || desc.usage.contains_invalid_bits() {
            return Err(resource::CreateBufferError::InvalidUsage(desc.usage));
        }

        if !self
            .features
            .contains(wgt::Features::MAPPABLE_PRIMARY_BUFFERS)
        {
            use wgt::BufferUsages as Bu;
            let write_mismatch = desc.usage.contains(Bu::MAP_WRITE)
                && !(Bu::MAP_WRITE | Bu::COPY_SRC).contains(desc.usage);
            let read_mismatch = desc.usage.contains(Bu::MAP_READ)
                && !(Bu::MAP_READ | Bu::COPY_DST).contains(desc.usage);
            if write_mismatch || read_mismatch {
                return Err(resource::CreateBufferError::UsageMismatch(desc.usage));
            }
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
            // We are required to zero out (initialize) all memory. This is done
            // on demand using clear_buffer which requires write transfer usage!
            usage |= hal::BufferUses::COPY_DST;
        }

        let actual_size = if desc.size == 0 {
            wgt::COPY_BUFFER_ALIGNMENT
        } else if desc.usage.contains(wgt::BufferUsages::VERTEX) {
            // Bumping the size by 1 so that we can bind an empty range at the
            // end of the buffer.
            desc.size + 1
        } else {
            desc.size
        };
        let clear_remainder = actual_size % wgt::COPY_BUFFER_ALIGNMENT;
        let aligned_size = if clear_remainder != 0 {
            actual_size + wgt::COPY_BUFFER_ALIGNMENT - clear_remainder
        } else {
            actual_size
        };

        let mut memory_flags = hal::MemoryFlags::empty();
        memory_flags.set(hal::MemoryFlags::TRANSIENT, transient);

        let hal_desc = hal::BufferDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            size: aligned_size,
            usage,
            memory_flags,
        };
        let buffer = unsafe { self.raw.create_buffer(&hal_desc) }.map_err(DeviceError::from)?;

        Ok(Buffer {
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

    pub(super) fn create_texture_from_hal(
        &self,
        hal_texture: A::Texture,
        hal_usage: hal::TextureUses,
        self_id: id::DeviceId,
        desc: &resource::TextureDescriptor,
        format_features: wgt::TextureFormatFeatures,
        clear_mode: resource::TextureClearMode<A>,
    ) -> resource::Texture<A> {
        debug_assert_eq!(self_id.backend(), A::VARIANT);

        resource::Texture {
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
            initialization_status: TextureInitTracker::new(
                desc.mip_level_count,
                desc.array_layer_count(),
            ),
            full_range: TextureSelector {
                mips: 0..desc.mip_level_count,
                layers: 0..desc.array_layer_count(),
            },
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
            clear_mode,
        }
    }

    pub fn create_buffer_from_hal(
        &self,
        hal_buffer: A::Buffer,
        self_id: id::DeviceId,
        desc: &resource::BufferDescriptor,
    ) -> Buffer<A> {
        debug_assert_eq!(self_id.backend(), A::VARIANT);

        Buffer {
            raw: Some(hal_buffer),
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            usage: desc.usage,
            size: desc.size,
            initialization_status: BufferInitTracker::new(0),
            sync_mapped_writes: None,
            map_state: resource::BufferMapState::Idle,
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
        }
    }

    pub(super) fn create_texture(
        &self,
        self_id: id::DeviceId,
        adapter: &Adapter<A>,
        desc: &resource::TextureDescriptor,
    ) -> Result<resource::Texture<A>, resource::CreateTextureError> {
        use resource::{CreateTextureError, TextureDimensionError};

        if desc.usage.is_empty() || desc.usage.contains_invalid_bits() {
            return Err(CreateTextureError::InvalidUsage(desc.usage));
        }

        conv::check_texture_dimension_size(
            desc.dimension,
            desc.size,
            desc.sample_count,
            &self.limits,
        )?;

        if desc.dimension != wgt::TextureDimension::D2 {
            // Depth textures can only be 2D
            if desc.format.is_depth_stencil_format() {
                return Err(CreateTextureError::InvalidDepthDimension(
                    desc.dimension,
                    desc.format,
                ));
            }
            // Renderable textures can only be 2D
            if desc.usage.contains(wgt::TextureUsages::RENDER_ATTACHMENT) {
                return Err(CreateTextureError::InvalidDimensionUsages(
                    wgt::TextureUsages::RENDER_ATTACHMENT,
                    desc.dimension,
                ));
            }

            // Compressed textures can only be 2D
            if desc.format.is_compressed() {
                return Err(CreateTextureError::InvalidCompressedDimension(
                    desc.dimension,
                    desc.format,
                ));
            }
        }

        if desc.format.is_compressed() {
            let (block_width, block_height) = desc.format.block_dimensions();

            if desc.size.width % block_width != 0 {
                return Err(CreateTextureError::InvalidDimension(
                    TextureDimensionError::NotMultipleOfBlockWidth {
                        width: desc.size.width,
                        block_width,
                        format: desc.format,
                    },
                ));
            }

            if desc.size.height % block_height != 0 {
                return Err(CreateTextureError::InvalidDimension(
                    TextureDimensionError::NotMultipleOfBlockHeight {
                        height: desc.size.height,
                        block_height,
                        format: desc.format,
                    },
                ));
            }
        }

        let format_features = self
            .describe_format_features(adapter, desc.format)
            .map_err(|error| CreateTextureError::MissingFeatures(desc.format, error))?;

        if desc.sample_count > 1 {
            if desc.mip_level_count != 1 {
                return Err(CreateTextureError::InvalidMipLevelCount {
                    requested: desc.mip_level_count,
                    maximum: 1,
                });
            }

            if desc.size.depth_or_array_layers != 1 {
                return Err(CreateTextureError::InvalidDimension(
                    TextureDimensionError::MultisampledDepthOrArrayLayer(
                        desc.size.depth_or_array_layers,
                    ),
                ));
            }

            if desc.usage.contains(wgt::TextureUsages::STORAGE_BINDING) {
                return Err(CreateTextureError::InvalidMultisampledStorageBinding);
            }

            if !desc.usage.contains(wgt::TextureUsages::RENDER_ATTACHMENT) {
                return Err(CreateTextureError::MultisampledNotRenderAttachment);
            }

            if !format_features.flags.intersects(
                wgt::TextureFormatFeatureFlags::MULTISAMPLE_X4
                    | wgt::TextureFormatFeatureFlags::MULTISAMPLE_X2
                    | wgt::TextureFormatFeatureFlags::MULTISAMPLE_X8
                    | wgt::TextureFormatFeatureFlags::MULTISAMPLE_X16,
            ) {
                return Err(CreateTextureError::InvalidMultisampledFormat(desc.format));
            }

            if !format_features
                .flags
                .sample_count_supported(desc.sample_count)
            {
                return Err(CreateTextureError::InvalidSampleCount(
                    desc.sample_count,
                    desc.format,
                    desc.format
                        .guaranteed_format_features(self.features)
                        .flags
                        .supported_sample_counts(),
                    adapter
                        .get_texture_format_features(desc.format)
                        .flags
                        .supported_sample_counts(),
                ));
            };
        }

        let mips = desc.mip_level_count;
        let max_levels_allowed = desc.size.max_mips(desc.dimension).min(hal::MAX_MIP_LEVELS);
        if mips == 0 || mips > max_levels_allowed {
            return Err(CreateTextureError::InvalidMipLevelCount {
                requested: mips,
                maximum: max_levels_allowed,
            });
        }

        let missing_allowed_usages = desc.usage - format_features.allowed_usages;
        if !missing_allowed_usages.is_empty() {
            // detect downlevel incompatibilities
            let wgpu_allowed_usages = desc
                .format
                .guaranteed_format_features(self.features)
                .allowed_usages;
            let wgpu_missing_usages = desc.usage - wgpu_allowed_usages;
            return Err(CreateTextureError::InvalidFormatUsages(
                missing_allowed_usages,
                desc.format,
                wgpu_missing_usages.is_empty(),
            ));
        }

        let mut hal_view_formats = vec![];
        for format in desc.view_formats.iter() {
            if desc.format == *format {
                continue;
            }
            if desc.format.remove_srgb_suffix() != format.remove_srgb_suffix() {
                return Err(CreateTextureError::InvalidViewFormat(*format, desc.format));
            }
            hal_view_formats.push(*format);
        }
        if !hal_view_formats.is_empty() {
            self.require_downlevel_flags(wgt::DownlevelFlags::VIEW_FORMATS)?;
        }

        // Enforce having COPY_DST/DEPTH_STENCIL_WRITE/COLOR_TARGET otherwise we
        // wouldn't be able to initialize the texture.
        let hal_usage = conv::map_texture_usage(desc.usage, desc.format.into())
            | if desc.format.is_depth_stencil_format() {
                hal::TextureUses::DEPTH_STENCIL_WRITE
            } else if desc.usage.contains(wgt::TextureUsages::COPY_DST) {
                hal::TextureUses::COPY_DST // (set already)
            } else {
                // Use COPY_DST only if we can't use COLOR_TARGET
                if format_features
                    .allowed_usages
                    .contains(wgt::TextureUsages::RENDER_ATTACHMENT)
                    && desc.dimension == wgt::TextureDimension::D2
                // Render targets dimension must be 2d
                {
                    hal::TextureUses::COLOR_TARGET
                } else {
                    hal::TextureUses::COPY_DST
                }
            };

        let hal_desc = hal::TextureDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            size: desc.size,
            mip_level_count: desc.mip_level_count,
            sample_count: desc.sample_count,
            dimension: desc.dimension,
            format: desc.format,
            usage: hal_usage,
            memory_flags: hal::MemoryFlags::empty(),
            view_formats: hal_view_formats,
        };

        let raw_texture = unsafe {
            self.raw
                .create_texture(&hal_desc)
                .map_err(DeviceError::from)?
        };

        let clear_mode = if hal_usage
            .intersects(hal::TextureUses::DEPTH_STENCIL_WRITE | hal::TextureUses::COLOR_TARGET)
        {
            let (is_color, usage) = if desc.format.is_depth_stencil_format() {
                (false, hal::TextureUses::DEPTH_STENCIL_WRITE)
            } else {
                (true, hal::TextureUses::COLOR_TARGET)
            };
            let dimension = match desc.dimension {
                wgt::TextureDimension::D1 => wgt::TextureViewDimension::D1,
                wgt::TextureDimension::D2 => wgt::TextureViewDimension::D2,
                wgt::TextureDimension::D3 => unreachable!(),
            };

            let clear_label = hal_label(
                Some("(wgpu internal) clear texture view"),
                self.instance_flags,
            );

            let mut clear_views = SmallVec::new();
            for mip_level in 0..desc.mip_level_count {
                for array_layer in 0..desc.size.depth_or_array_layers {
                    let desc = hal::TextureViewDescriptor {
                        label: clear_label,
                        format: desc.format,
                        dimension,
                        usage,
                        range: wgt::ImageSubresourceRange {
                            aspect: wgt::TextureAspect::All,
                            base_mip_level: mip_level,
                            mip_level_count: Some(1),
                            base_array_layer: array_layer,
                            array_layer_count: Some(1),
                        },
                    };
                    clear_views.push(
                        unsafe { self.raw.create_texture_view(&raw_texture, &desc) }
                            .map_err(DeviceError::from)?,
                    );
                }
            }
            resource::TextureClearMode::RenderPass {
                clear_views,
                is_color,
            }
        } else {
            resource::TextureClearMode::BufferCopy
        };

        let mut texture = self.create_texture_from_hal(
            raw_texture,
            hal_usage,
            self_id,
            desc,
            format_features,
            clear_mode,
        );
        texture.hal_usage = hal_usage;
        Ok(texture)
    }

    pub(super) fn create_texture_view(
        &self,
        texture: &resource::Texture<A>,
        texture_id: id::TextureId,
        desc: &resource::TextureViewDescriptor,
    ) -> Result<resource::TextureView<A>, resource::CreateTextureViewError> {
        let texture_raw = texture
            .inner
            .as_raw()
            .ok_or(resource::CreateTextureViewError::InvalidTexture)?;

        // resolve TextureViewDescriptor defaults
        // https://gpuweb.github.io/gpuweb/#abstract-opdef-resolving-gputextureviewdescriptor-defaults

        let resolved_format = desc.format.unwrap_or_else(|| {
            texture
                .desc
                .format
                .aspect_specific_format(desc.range.aspect)
                .unwrap_or(texture.desc.format)
        });

        let resolved_dimension = desc
            .dimension
            .unwrap_or_else(|| match texture.desc.dimension {
                wgt::TextureDimension::D1 => wgt::TextureViewDimension::D1,
                wgt::TextureDimension::D2 => {
                    if texture.desc.array_layer_count() == 1 {
                        wgt::TextureViewDimension::D2
                    } else {
                        wgt::TextureViewDimension::D2Array
                    }
                }
                wgt::TextureDimension::D3 => wgt::TextureViewDimension::D3,
            });

        let resolved_mip_level_count = desc.range.mip_level_count.unwrap_or_else(|| {
            texture
                .desc
                .mip_level_count
                .saturating_sub(desc.range.base_mip_level)
        });

        let resolved_array_layer_count =
            desc.range
                .array_layer_count
                .unwrap_or_else(|| match resolved_dimension {
                    wgt::TextureViewDimension::D1
                    | wgt::TextureViewDimension::D2
                    | wgt::TextureViewDimension::D3 => 1,
                    wgt::TextureViewDimension::Cube => 6,
                    wgt::TextureViewDimension::D2Array | wgt::TextureViewDimension::CubeArray => {
                        texture
                            .desc
                            .array_layer_count()
                            .saturating_sub(desc.range.base_array_layer)
                    }
                });

        // validate TextureViewDescriptor

        let aspects = hal::FormatAspects::new(texture.desc.format, desc.range.aspect);
        if aspects.is_empty() {
            return Err(resource::CreateTextureViewError::InvalidAspect {
                texture_format: texture.desc.format,
                requested_aspect: desc.range.aspect,
            });
        }

        let format_is_good = if desc.range.aspect == wgt::TextureAspect::All {
            resolved_format == texture.desc.format
                || texture.desc.view_formats.contains(&resolved_format)
        } else {
            Some(resolved_format)
                == texture
                    .desc
                    .format
                    .aspect_specific_format(desc.range.aspect)
        };
        if !format_is_good {
            return Err(resource::CreateTextureViewError::FormatReinterpretation {
                texture: texture.desc.format,
                view: resolved_format,
            });
        }

        // check if multisampled texture is seen as anything but 2D
        if texture.desc.sample_count > 1 && resolved_dimension != wgt::TextureViewDimension::D2 {
            return Err(
                resource::CreateTextureViewError::InvalidMultisampledTextureViewDimension(
                    resolved_dimension,
                ),
            );
        }

        // check if the dimension is compatible with the texture
        if texture.desc.dimension != resolved_dimension.compatible_texture_dimension() {
            return Err(
                resource::CreateTextureViewError::InvalidTextureViewDimension {
                    view: resolved_dimension,
                    texture: texture.desc.dimension,
                },
            );
        }

        match resolved_dimension {
            TextureViewDimension::D1 | TextureViewDimension::D2 | TextureViewDimension::D3 => {
                if resolved_array_layer_count != 1 {
                    return Err(resource::CreateTextureViewError::InvalidArrayLayerCount {
                        requested: resolved_array_layer_count,
                        dim: resolved_dimension,
                    });
                }
            }
            TextureViewDimension::Cube => {
                if resolved_array_layer_count != 6 {
                    return Err(
                        resource::CreateTextureViewError::InvalidCubemapTextureDepth {
                            depth: resolved_array_layer_count,
                        },
                    );
                }
            }
            TextureViewDimension::CubeArray => {
                if resolved_array_layer_count % 6 != 0 {
                    return Err(
                        resource::CreateTextureViewError::InvalidCubemapArrayTextureDepth {
                            depth: resolved_array_layer_count,
                        },
                    );
                }
            }
            _ => {}
        }

        match resolved_dimension {
            TextureViewDimension::Cube | TextureViewDimension::CubeArray => {
                if texture.desc.size.width != texture.desc.size.height {
                    return Err(resource::CreateTextureViewError::InvalidCubeTextureViewSize);
                }
            }
            _ => {}
        }

        if resolved_mip_level_count == 0 {
            return Err(resource::CreateTextureViewError::ZeroMipLevelCount);
        }

        let mip_level_end = desc
            .range
            .base_mip_level
            .saturating_add(resolved_mip_level_count);

        let level_end = texture.desc.mip_level_count;
        if mip_level_end > level_end {
            return Err(resource::CreateTextureViewError::TooManyMipLevels {
                requested: mip_level_end,
                total: level_end,
            });
        }

        if resolved_array_layer_count == 0 {
            return Err(resource::CreateTextureViewError::ZeroArrayLayerCount);
        }

        let array_layer_end = desc
            .range
            .base_array_layer
            .saturating_add(resolved_array_layer_count);

        let layer_end = texture.desc.array_layer_count();
        if array_layer_end > layer_end {
            return Err(resource::CreateTextureViewError::TooManyArrayLayers {
                requested: array_layer_end,
                total: layer_end,
            });
        };

        // https://gpuweb.github.io/gpuweb/#abstract-opdef-renderable-texture-view
        let render_extent = 'b: loop {
            if !texture
                .desc
                .usage
                .contains(wgt::TextureUsages::RENDER_ATTACHMENT)
            {
                break 'b Err(TextureViewNotRenderableReason::Usage(texture.desc.usage));
            }

            if !(resolved_dimension == TextureViewDimension::D2
                || (self.features.contains(wgt::Features::MULTIVIEW)
                    && resolved_dimension == TextureViewDimension::D2Array))
            {
                break 'b Err(TextureViewNotRenderableReason::Dimension(
                    resolved_dimension,
                ));
            }

            if resolved_mip_level_count != 1 {
                break 'b Err(TextureViewNotRenderableReason::MipLevelCount(
                    resolved_mip_level_count,
                ));
            }

            if resolved_array_layer_count != 1
                && !(self.features.contains(wgt::Features::MULTIVIEW))
            {
                break 'b Err(TextureViewNotRenderableReason::ArrayLayerCount(
                    resolved_array_layer_count,
                ));
            }

            if aspects != hal::FormatAspects::from(texture.desc.format) {
                break 'b Err(TextureViewNotRenderableReason::Aspects(aspects));
            }

            break 'b Ok(texture
                .desc
                .compute_render_extent(desc.range.base_mip_level));
        };

        // filter the usages based on the other criteria
        let usage = {
            let mask_copy = !(hal::TextureUses::COPY_SRC | hal::TextureUses::COPY_DST);
            let mask_dimension = match resolved_dimension {
                wgt::TextureViewDimension::Cube | wgt::TextureViewDimension::CubeArray => {
                    hal::TextureUses::RESOURCE
                }
                wgt::TextureViewDimension::D3 => {
                    hal::TextureUses::RESOURCE
                        | hal::TextureUses::STORAGE_READ
                        | hal::TextureUses::STORAGE_READ_WRITE
                }
                _ => hal::TextureUses::all(),
            };
            let mask_mip_level = if resolved_mip_level_count == 1 {
                hal::TextureUses::all()
            } else {
                hal::TextureUses::RESOURCE
            };
            texture.hal_usage & mask_copy & mask_dimension & mask_mip_level
        };

        log::debug!(
            "Create view for texture {:?} filters usages to {:?}",
            texture_id,
            usage
        );

        // use the combined depth-stencil format for the view
        let format = if resolved_format.is_depth_stencil_component(texture.desc.format) {
            texture.desc.format
        } else {
            resolved_format
        };

        let resolved_range = wgt::ImageSubresourceRange {
            aspect: desc.range.aspect,
            base_mip_level: desc.range.base_mip_level,
            mip_level_count: Some(resolved_mip_level_count),
            base_array_layer: desc.range.base_array_layer,
            array_layer_count: Some(resolved_array_layer_count),
        };

        let hal_desc = hal::TextureViewDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            format,
            dimension: resolved_dimension,
            usage,
            range: resolved_range,
        };

        let raw = unsafe {
            self.raw
                .create_texture_view(texture_raw, &hal_desc)
                .map_err(|_| resource::CreateTextureViewError::OutOfMemory)?
        };

        let selector = TextureSelector {
            mips: desc.range.base_mip_level..mip_level_end,
            layers: desc.range.base_array_layer..array_layer_end,
        };

        Ok(resource::TextureView {
            raw,
            parent_id: Stored {
                value: id::Valid(texture_id),
                ref_count: texture.life_guard.add_ref(),
            },
            device_id: texture.device_id.clone(),
            desc: resource::HalTextureViewDescriptor {
                format: resolved_format,
                dimension: resolved_dimension,
                range: resolved_range,
            },
            format_features: texture.format_features,
            render_extent,
            samples: texture.desc.sample_count,
            selector,
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
        })
    }

    pub(super) fn create_sampler(
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

        if desc.border_color == Some(wgt::SamplerBorderColor::Zero) {
            self.require_features(wgt::Features::ADDRESS_MODE_CLAMP_TO_ZERO)?;
        }

        if desc.lod_min_clamp < 0.0 {
            return Err(resource::CreateSamplerError::InvalidLodMinClamp(
                desc.lod_min_clamp,
            ));
        }
        if desc.lod_max_clamp < desc.lod_min_clamp {
            return Err(resource::CreateSamplerError::InvalidLodMaxClamp {
                lod_min_clamp: desc.lod_min_clamp,
                lod_max_clamp: desc.lod_max_clamp,
            });
        }

        if desc.anisotropy_clamp < 1 {
            return Err(resource::CreateSamplerError::InvalidAnisotropy(
                desc.anisotropy_clamp,
            ));
        }

        if desc.anisotropy_clamp != 1 {
            if !matches!(desc.min_filter, wgt::FilterMode::Linear) {
                return Err(
                    resource::CreateSamplerError::InvalidFilterModeWithAnisotropy {
                        filter_type: resource::SamplerFilterErrorType::MinFilter,
                        filter_mode: desc.min_filter,
                        anisotropic_clamp: desc.anisotropy_clamp,
                    },
                );
            }
            if !matches!(desc.mag_filter, wgt::FilterMode::Linear) {
                return Err(
                    resource::CreateSamplerError::InvalidFilterModeWithAnisotropy {
                        filter_type: resource::SamplerFilterErrorType::MagFilter,
                        filter_mode: desc.mag_filter,
                        anisotropic_clamp: desc.anisotropy_clamp,
                    },
                );
            }
            if !matches!(desc.mipmap_filter, wgt::FilterMode::Linear) {
                return Err(
                    resource::CreateSamplerError::InvalidFilterModeWithAnisotropy {
                        filter_type: resource::SamplerFilterErrorType::MipmapFilter,
                        filter_mode: desc.mipmap_filter,
                        anisotropic_clamp: desc.anisotropy_clamp,
                    },
                );
            }
        }

        let anisotropy_clamp = if self
            .downlevel
            .flags
            .contains(wgt::DownlevelFlags::ANISOTROPIC_FILTERING)
        {
            // Clamp anisotropy clamp to [1, 16] per the wgpu-hal interface
            desc.anisotropy_clamp.min(16)
        } else {
            // If it isn't supported, set this unconditionally to 1
            1
        };

        //TODO: check for wgt::DownlevelFlags::COMPARISON_SAMPLERS

        let hal_desc = hal::SamplerDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            address_modes: desc.address_modes,
            mag_filter: desc.mag_filter,
            min_filter: desc.min_filter,
            mipmap_filter: desc.mipmap_filter,
            lod_clamp: desc.lod_min_clamp..desc.lod_max_clamp,
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

    pub(super) fn create_shader_module<'a>(
        &self,
        self_id: id::DeviceId,
        desc: &pipeline::ShaderModuleDescriptor<'a>,
        source: pipeline::ShaderModuleSource<'a>,
    ) -> Result<pipeline::ShaderModule<A>, pipeline::CreateShaderModuleError> {
        let (module, source) = match source {
            #[cfg(feature = "wgsl")]
            pipeline::ShaderModuleSource::Wgsl(code) => {
                profiling::scope!("naga::wgsl::parse_str");
                let module = naga::front::wgsl::parse_str(&code).map_err(|inner| {
                    pipeline::CreateShaderModuleError::Parsing(pipeline::ShaderError {
                        source: code.to_string(),
                        label: desc.label.as_ref().map(|l| l.to_string()),
                        inner: Box::new(inner),
                    })
                })?;
                (Cow::Owned(module), code.into_owned())
            }
            pipeline::ShaderModuleSource::Naga(module) => (module, String::new()),
            pipeline::ShaderModuleSource::Dummy(_) => panic!("found `ShaderModuleSource::Dummy`"),
        };
        for (_, var) in module.global_variables.iter() {
            match var.binding {
                Some(ref br) if br.group >= self.limits.max_bind_groups => {
                    return Err(pipeline::CreateShaderModuleError::InvalidGroupIndex {
                        bind: br.clone(),
                        group: br.group,
                        limit: self.limits.max_bind_groups,
                    });
                }
                _ => continue,
            };
        }

        use naga::valid::Capabilities as Caps;
        profiling::scope!("naga::validate");

        let mut caps = Caps::empty();
        caps.set(
            Caps::PUSH_CONSTANT,
            self.features.contains(wgt::Features::PUSH_CONSTANTS),
        );
        caps.set(
            Caps::FLOAT64,
            self.features.contains(wgt::Features::SHADER_F64),
        );
        caps.set(
            Caps::PRIMITIVE_INDEX,
            self.features
                .contains(wgt::Features::SHADER_PRIMITIVE_INDEX),
        );
        caps.set(
            Caps::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            self.features.contains(
                wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            ),
        );
        caps.set(
            Caps::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
            self.features.contains(
                wgt::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
            ),
        );
        // TODO: This needs a proper wgpu feature
        caps.set(
            Caps::SAMPLER_NON_UNIFORM_INDEXING,
            self.features.contains(
                wgt::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
            ),
        );
        caps.set(
            Caps::STORAGE_TEXTURE_16BIT_NORM_FORMATS,
            self.features
                .contains(wgt::Features::TEXTURE_FORMAT_16BIT_NORM),
        );
        caps.set(
            Caps::MULTIVIEW,
            self.features.contains(wgt::Features::MULTIVIEW),
        );
        caps.set(
            Caps::EARLY_DEPTH_TEST,
            self.features
                .contains(wgt::Features::SHADER_EARLY_DEPTH_TEST),
        );
        caps.set(
            Caps::MULTISAMPLED_SHADING,
            self.downlevel
                .flags
                .contains(wgt::DownlevelFlags::MULTISAMPLED_SHADING),
        );
        caps.set(
            Caps::DUAL_SOURCE_BLENDING,
            self.features.contains(wgt::Features::DUAL_SOURCE_BLENDING),
        );
        caps.set(
            Caps::CUBE_ARRAY_TEXTURES,
            self.downlevel
                .flags
                .contains(wgt::DownlevelFlags::CUBE_ARRAY_TEXTURES),
        );

        let debug_source =
            if self.instance_flags.contains(wgt::InstanceFlags::DEBUG) && !source.is_empty() {
                Some(hal::DebugSource {
                    file_name: Cow::Owned(
                        desc.label
                            .as_ref()
                            .map_or("shader".to_string(), |l| l.to_string()),
                    ),
                    source_code: Cow::Owned(source.clone()),
                })
            } else {
                None
            };

        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
            .validate(&module)
            .map_err(|inner| {
                pipeline::CreateShaderModuleError::Validation(pipeline::ShaderError {
                    source,
                    label: desc.label.as_ref().map(|l| l.to_string()),
                    inner: Box::new(inner),
                })
            })?;

        let interface =
            validation::Interface::new(&module, &info, self.limits.clone(), self.features);
        let hal_shader = hal::ShaderInput::Naga(hal::NagaShader {
            module,
            info,
            debug_source,
        });
        let hal_desc = hal::ShaderModuleDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            runtime_checks: desc.shader_bound_checks.runtime_checks(),
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
    pub(super) unsafe fn create_shader_module_spirv<'a>(
        &self,
        self_id: id::DeviceId,
        desc: &pipeline::ShaderModuleDescriptor<'a>,
        source: &'a [u32],
    ) -> Result<pipeline::ShaderModule<A>, pipeline::CreateShaderModuleError> {
        self.require_features(wgt::Features::SPIRV_SHADER_PASSTHROUGH)?;
        let hal_desc = hal::ShaderModuleDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            runtime_checks: desc.shader_bound_checks.runtime_checks(),
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

    pub(super) fn deduplicate_bind_group_layout(
        self_id: id::DeviceId,
        entry_map: &binding_model::BindEntryMap,
        guard: &Storage<binding_model::BindGroupLayout<A>, id::BindGroupLayoutId>,
    ) -> Option<id::BindGroupLayoutId> {
        guard
            .iter(self_id.backend())
            .find(|&(_, bgl)| {
                bgl.device_id.value.0 == self_id
                    && bgl
                        .as_inner()
                        .map_or(false, |inner| inner.entries == *entry_map)
            })
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
            .map(|&id| {
                &get_bind_group_layout(bgl_guard, id)
                    .1
                    .assume_deduplicated()
                    .entries
            })
            .collect()
    }

    /// Generate information about late-validated buffer bindings for pipelines.
    //TODO: should this be combined with `get_introspection_bind_group_layouts` in some way?
    fn make_late_sized_buffer_groups<'a>(
        shader_binding_sizes: &FastHashMap<naga::ResourceBinding, wgt::BufferSize>,
        layout: &binding_model::PipelineLayout<A>,
        bgl_guard: &'a Storage<binding_model::BindGroupLayout<A>, id::BindGroupLayoutId>,
    ) -> ArrayVec<pipeline::LateSizedBufferGroup, { hal::MAX_BIND_GROUPS }> {
        // Given the shader-required binding sizes and the pipeline layout,
        // return the filtered list of them in the layout order,
        // removing those with given `min_binding_size`.
        layout
            .bind_group_layout_ids
            .iter()
            .enumerate()
            .map(|(group_index, &bgl_id)| pipeline::LateSizedBufferGroup {
                shader_sizes: get_bind_group_layout(bgl_guard, bgl_id)
                    .1
                    .assume_deduplicated()
                    .entries
                    .values()
                    .filter_map(|entry| match entry.ty {
                        wgt::BindingType::Buffer {
                            min_binding_size: None,
                            ..
                        } => {
                            let rb = naga::ResourceBinding {
                                group: group_index as u32,
                                binding: entry.binding,
                            };
                            let shader_size =
                                shader_binding_sizes.get(&rb).map_or(0, |nz| nz.get());
                            Some(shader_size)
                        }
                        _ => None,
                    })
                    .collect(),
            })
            .collect()
    }

    pub(super) fn create_bind_group_layout(
        &self,
        self_id: id::DeviceId,
        label: &crate::Label,
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
                Bt::Sampler { .. } => (
                    Some(wgt::Features::TEXTURE_BINDING_ARRAY),
                    WritableStorage::No,
                ),
                Bt::Texture {
                    multisampled: true,
                    sample_type: TextureSampleType::Float { filterable: true },
                    ..
                } => {
                    return Err(binding_model::CreateBindGroupLayoutError::Entry {
                        binding: entry.binding,
                        error: binding_model::BindGroupLayoutEntryError::SampleTypeFloatFilterableBindingMultisampled,
                    });
                }
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
                    match access {
                        wgt::StorageTextureAccess::ReadOnly
                        | wgt::StorageTextureAccess::ReadWrite
                            if !self.features.contains(
                                wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                            ) =>
                        {
                            return Err(binding_model::CreateBindGroupLayoutError::Entry {
                                binding: entry.binding,
                                error: binding_model::BindGroupLayoutEntryError::StorageTextureReadWrite,
                            });
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

            if entry.visibility.contains_invalid_bits() {
                return Err(
                    binding_model::CreateBindGroupLayoutError::InvalidVisibility(entry.visibility),
                );
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

        let bgl_flags = conv::bind_group_layout_flags(self.features);

        let mut hal_bindings = entry_map.values().cloned().collect::<Vec<_>>();
        hal_bindings.sort_by_key(|b| b.binding);
        let hal_desc = hal::BindGroupLayoutDescriptor {
            label: label.to_hal(self.instance_flags),
            flags: bgl_flags,
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
        // If a single bind group layout violates limits, the pipeline layout is
        // definitely going to violate limits too, lets catch it now.
        count_validator
            .validate(&self.limits)
            .map_err(binding_model::CreateBindGroupLayoutError::TooManyBindings)?;

        Ok(binding_model::BindGroupLayout {
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            multi_ref_count: MultiRefCount::new(),
            inner: BglOrDuplicate::Inner(BindGroupLayoutInner {
                raw,
                dynamic_count: entry_map
                    .values()
                    .filter(|b| b.ty.has_dynamic_offset())
                    .count(),
                count_validator,
                entries: entry_map,
                #[cfg(debug_assertions)]
                label: label.borrow_or_default().to_string(),
            }),
        })
    }

    fn create_buffer_binding<'a>(
        device_id: id::DeviceId,
        bb: &binding_model::BufferBinding,
        binding: u32,
        decl: &wgt::BindGroupLayoutEntry,
        used_buffer_ranges: &mut Vec<BufferInitTrackerAction>,
        dynamic_binding_info: &mut Vec<binding_model::BindGroupDynamicBindingData>,
        late_buffer_binding_sizes: &mut FastHashMap<u32, wgt::BufferSize>,
        used: &mut BindGroupStates<A>,
        storage: &'a Storage<Buffer<A>, id::BufferId>,
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
                    hal::BufferUses::STORAGE_READ_WRITE
                },
                limits.max_storage_buffer_binding_size,
            ),
        };

        let (align, align_limit_name) =
            binding_model::buffer_binding_type_alignment(limits, binding_ty);
        if bb.offset % align as u64 != 0 {
            return Err(Error::UnalignedBufferOffset(
                bb.offset,
                align_limit_name,
                align,
            ));
        }

        let buffer = used
            .buffers
            .add_single(storage, bb.buffer_id, internal_use)
            .ok_or(Error::InvalidBuffer(bb.buffer_id))?;

        if buffer.device_id.value.0 != device_id {
            return Err(DeviceError::WrongDevice.into());
        }

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
                binding_idx: binding,
                buffer_size: buffer.size,
                binding_range: bb.offset..bind_end,
                maximum_dynamic_offset: buffer.size - bind_end,
                binding_type: binding_ty,
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
        } else {
            let late_size =
                wgt::BufferSize::new(bind_size).ok_or(Error::BindingZeroSize(bb.buffer_id))?;
            late_buffer_binding_sizes.insert(binding, late_size);
        }

        assert_eq!(bb.offset % wgt::COPY_BUFFER_ALIGNMENT, 0);
        used_buffer_ranges.extend(buffer.initialization_status.create_action(
            bb.buffer_id,
            bb.offset..bb.offset + bind_size,
            MemoryInitKind::NeedsInitializedMemory,
        ));

        Ok(hal::BufferBinding {
            buffer: raw_buffer,
            offset: bb.offset,
            size: bb.size,
        })
    }

    fn create_texture_binding(
        device_id: id::DeviceId,
        view: &resource::TextureView<A>,
        texture_guard: &Storage<resource::Texture<A>, id::TextureId>,
        internal_use: hal::TextureUses,
        pub_usage: wgt::TextureUsages,
        used: &mut BindGroupStates<A>,
        used_texture_ranges: &mut Vec<TextureInitTrackerAction>,
    ) -> Result<(), binding_model::CreateBindGroupError> {
        // Careful here: the texture may no longer have its own ref count,
        // if it was deleted by the user.
        let texture = used
            .textures
            .add_single(
                texture_guard,
                view.parent_id.value.0,
                view.parent_id.ref_count.clone(),
                Some(view.selector.clone()),
                internal_use,
            )
            .ok_or(binding_model::CreateBindGroupError::InvalidTexture(
                view.parent_id.value.0,
            ))?;

        if texture.device_id.value.0 != device_id {
            return Err(DeviceError::WrongDevice.into());
        }

        check_texture_usage(texture.desc.usage, pub_usage)?;

        used_texture_ranges.push(TextureInitTrackerAction {
            id: view.parent_id.value.0,
            range: TextureInitRange {
                mip_range: view.desc.range.mip_range(texture.desc.mip_level_count),
                layer_range: view
                    .desc
                    .range
                    .layer_range(texture.desc.array_layer_count()),
            },
            kind: MemoryInitKind::NeedsInitializedMemory,
        });

        Ok(())
    }

    // This function expects the provided bind group layout to be resolved
    // (not passing a duplicate) beforehand.
    pub(super) fn create_bind_group<G: GlobalIdentityHandlerFactory>(
        &self,
        self_id: id::DeviceId,
        layout: &binding_model::BindGroupLayout<A>,
        layout_id: id::Valid<id::BindGroupLayoutId>,
        desc: &binding_model::BindGroupDescriptor,
        hub: &Hub<A, G>,
        token: &mut Token<binding_model::BindGroupLayout<A>>,
    ) -> Result<binding_model::BindGroup<A>, binding_model::CreateBindGroupError> {
        use crate::binding_model::{BindingResource as Br, CreateBindGroupError as Error};
        {
            // Check that the number of entries in the descriptor matches
            // the number of entries in the layout.
            let actual = desc.entries.len();
            let expected = layout.assume_deduplicated().entries.len();
            if actual != expected {
                return Err(Error::BindingsNumMismatch { expected, actual });
            }
        }

        // TODO: arrayvec/smallvec, or re-use allocations
        // Record binding info for dynamic offset validation
        let mut dynamic_binding_info = Vec::new();
        // Map of binding -> shader reflected size
        //Note: we can't collect into a vector right away because
        // it needs to be in BGL iteration order, not BG entry order.
        let mut late_buffer_binding_sizes = FastHashMap::default();
        // fill out the descriptors
        let mut used = BindGroupStates::new();

        let (buffer_guard, mut token) = hub.buffers.read(token);
        let (texture_guard, mut token) = hub.textures.read(&mut token); //skip token
        let (texture_view_guard, mut token) = hub.texture_views.read(&mut token);
        let (sampler_guard, _) = hub.samplers.read(&mut token);

        let mut used_buffer_ranges = Vec::new();
        let mut used_texture_ranges = Vec::new();
        let mut hal_entries = Vec::with_capacity(desc.entries.len());
        let mut hal_buffers = Vec::new();
        let mut hal_samplers = Vec::new();
        let mut hal_textures = Vec::new();
        for entry in desc.entries.iter() {
            let binding = entry.binding;
            // Find the corresponding declaration in the layout
            let decl = layout
                .assume_deduplicated()
                .entries
                .get(&binding)
                .ok_or(Error::MissingBindingDeclaration(binding))?;
            let (res_index, count) = match entry.resource {
                Br::Buffer(ref bb) => {
                    let bb = Self::create_buffer_binding(
                        self_id,
                        bb,
                        binding,
                        decl,
                        &mut used_buffer_ranges,
                        &mut dynamic_binding_info,
                        &mut late_buffer_binding_sizes,
                        &mut used,
                        &*buffer_guard,
                        &self.limits,
                    )?;

                    let res_index = hal_buffers.len();
                    hal_buffers.push(bb);
                    (res_index, 1)
                }
                Br::BufferArray(ref bindings_array) => {
                    let num_bindings = bindings_array.len();
                    Self::check_array_binding(self.features, decl.count, num_bindings)?;

                    let res_index = hal_buffers.len();
                    for bb in bindings_array.iter() {
                        let bb = Self::create_buffer_binding(
                            self_id,
                            bb,
                            binding,
                            decl,
                            &mut used_buffer_ranges,
                            &mut dynamic_binding_info,
                            &mut late_buffer_binding_sizes,
                            &mut used,
                            &*buffer_guard,
                            &self.limits,
                        )?;
                        hal_buffers.push(bb);
                    }
                    (res_index, num_bindings)
                }
                Br::Sampler(id) => {
                    match decl.ty {
                        wgt::BindingType::Sampler(ty) => {
                            let sampler = used
                                .samplers
                                .add_single(&*sampler_guard, id)
                                .ok_or(Error::InvalidSampler(id))?;

                            if sampler.device_id.value.0 != self_id {
                                return Err(DeviceError::WrongDevice.into());
                            }

                            // Allowed sampler values for filtering and comparison
                            let (allowed_filtering, allowed_comparison) = match ty {
                                wgt::SamplerBindingType::Filtering => (None, false),
                                wgt::SamplerBindingType::NonFiltering => (Some(false), false),
                                wgt::SamplerBindingType::Comparison => (None, true),
                            };

                            if let Some(allowed_filtering) = allowed_filtering {
                                if allowed_filtering != sampler.filtering {
                                    return Err(Error::WrongSamplerFiltering {
                                        binding,
                                        layout_flt: allowed_filtering,
                                        sampler_flt: sampler.filtering,
                                    });
                                }
                            }

                            if allowed_comparison != sampler.comparison {
                                return Err(Error::WrongSamplerComparison {
                                    binding,
                                    layout_cmp: allowed_comparison,
                                    sampler_cmp: sampler.comparison,
                                });
                            }

                            let res_index = hal_samplers.len();
                            hal_samplers.push(&sampler.raw);
                            (res_index, 1)
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
                Br::SamplerArray(ref bindings_array) => {
                    let num_bindings = bindings_array.len();
                    Self::check_array_binding(self.features, decl.count, num_bindings)?;

                    let res_index = hal_samplers.len();
                    for &id in bindings_array.iter() {
                        let sampler = used
                            .samplers
                            .add_single(&*sampler_guard, id)
                            .ok_or(Error::InvalidSampler(id))?;

                        if sampler.device_id.value.0 != self_id {
                            return Err(DeviceError::WrongDevice.into());
                        }

                        hal_samplers.push(&sampler.raw);
                    }

                    (res_index, num_bindings)
                }
                Br::TextureView(id) => {
                    let view = used
                        .views
                        .add_single(&*texture_view_guard, id)
                        .ok_or(Error::InvalidTextureView(id))?;
                    let (pub_usage, internal_use) = Self::texture_use_parameters(
                        binding,
                        decl,
                        view,
                        "SampledTexture, ReadonlyStorageTexture or WriteonlyStorageTexture",
                    )?;
                    Self::create_texture_binding(
                        self_id,
                        view,
                        &texture_guard,
                        internal_use,
                        pub_usage,
                        &mut used,
                        &mut used_texture_ranges,
                    )?;
                    let res_index = hal_textures.len();
                    hal_textures.push(hal::TextureBinding {
                        view: &view.raw,
                        usage: internal_use,
                    });
                    (res_index, 1)
                }
                Br::TextureViewArray(ref bindings_array) => {
                    let num_bindings = bindings_array.len();
                    Self::check_array_binding(self.features, decl.count, num_bindings)?;

                    let res_index = hal_textures.len();
                    for &id in bindings_array.iter() {
                        let view = used
                            .views
                            .add_single(&*texture_view_guard, id)
                            .ok_or(Error::InvalidTextureView(id))?;
                        let (pub_usage, internal_use) =
                            Self::texture_use_parameters(binding, decl, view,
                                                         "SampledTextureArray, ReadonlyStorageTextureArray or WriteonlyStorageTextureArray")?;
                        Self::create_texture_binding(
                            self_id,
                            view,
                            &texture_guard,
                            internal_use,
                            pub_usage,
                            &mut used,
                            &mut used_texture_ranges,
                        )?;
                        hal_textures.push(hal::TextureBinding {
                            view: &view.raw,
                            usage: internal_use,
                        });
                    }

                    (res_index, num_bindings)
                }
            };

            hal_entries.push(hal::BindGroupEntry {
                binding,
                resource_index: res_index as u32,
                count: count as u32,
            });
        }

        used.optimize();

        hal_entries.sort_by_key(|entry| entry.binding);
        for (a, b) in hal_entries.iter().zip(hal_entries.iter().skip(1)) {
            if a.binding == b.binding {
                return Err(Error::DuplicateBinding(a.binding));
            }
        }

        let layout_inner = layout.assume_deduplicated();

        let hal_desc = hal::BindGroupDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            layout: &layout_inner.raw,
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

        // manually add a dependency on BGL
        layout.multi_ref_count.inc();

        Ok(binding_model::BindGroup {
            raw,
            device_id: Stored {
                value: id::Valid(self_id),
                ref_count: self.life_guard.add_ref(),
            },
            layout_id,
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
            used,
            used_buffer_ranges,
            used_texture_ranges,
            dynamic_binding_info,
            // collect in the order of BGL iteration
            late_buffer_binding_sizes: layout_inner
                .entries
                .keys()
                .flat_map(|binding| late_buffer_binding_sizes.get(binding).cloned())
                .collect(),
        })
    }

    fn check_array_binding(
        features: wgt::Features,
        count: Option<NonZeroU32>,
        num_bindings: usize,
    ) -> Result<(), super::binding_model::CreateBindGroupError> {
        use super::binding_model::CreateBindGroupError as Error;

        if let Some(count) = count {
            let count = count.get() as usize;
            if count < num_bindings {
                return Err(Error::BindingArrayPartialLengthMismatch {
                    actual: num_bindings,
                    expected: count,
                });
            }
            if count != num_bindings
                && !features.contains(wgt::Features::PARTIALLY_BOUND_BINDING_ARRAY)
            {
                return Err(Error::BindingArrayLengthMismatch {
                    actual: num_bindings,
                    expected: count,
                });
            }
            if num_bindings == 0 {
                return Err(Error::BindingArrayZeroLength);
            }
        } else {
            return Err(Error::SingleBindingExpected);
        };

        Ok(())
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
                let compat_sample_type = view
                    .desc
                    .format
                    .sample_type(Some(view.desc.range.aspect))
                    .unwrap();
                match (sample_type, compat_sample_type) {
                    (Tst::Uint, Tst::Uint) |
                    (Tst::Sint, Tst::Sint) |
                    (Tst::Depth, Tst::Depth) |
                    // if we expect non-filterable, accept anything float
                    (Tst::Float { filterable: false }, Tst::Float { .. }) |
                    // if we expect filterable, require it
                    (Tst::Float { filterable: true }, Tst::Float { filterable: true }) |
                    // if we expect non-filterable, also accept depth
                    (Tst::Float { filterable: false }, Tst::Depth) => {}
                    // if we expect filterable, also accept Float that is defined as
                    // unfilterable if filterable feature is explicitly enabled (only hit
                    // if wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES is
                    // enabled)
                    (Tst::Float { filterable: true }, Tst::Float { .. }) if view.format_features.flags.contains(wgt::TextureFormatFeatureFlags::FILTERABLE) => {}
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
                    hal::TextureUses::RESOURCE,
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

                let mip_level_count = view.selector.mips.end - view.selector.mips.start;
                if mip_level_count != 1 {
                    return Err(Error::InvalidStorageTextureMipLevelCount {
                        binding,
                        mip_level_count,
                    });
                }

                let internal_use = match access {
                    wgt::StorageTextureAccess::WriteOnly => hal::TextureUses::STORAGE_READ_WRITE,
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

                        hal::TextureUses::STORAGE_READ_WRITE
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

    pub(super) fn create_pipeline_layout(
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
            let Some(bind_group_layout) = try_get_bind_group_layout(bgl_guard, id) else {
                return Err(Error::InvalidBindGroupLayout(id));
            };

            if bind_group_layout.device_id.value.0 != self_id {
                return Err(DeviceError::WrongDevice.into());
            }

            count_validator.merge(&bind_group_layout.assume_deduplicated().count_validator);
        }
        count_validator
            .validate(&self.limits)
            .map_err(Error::TooManyBindings)?;

        let bgl_vec = desc
            .bind_group_layouts
            .iter()
            .map(|&id| {
                &try_get_bind_group_layout(bgl_guard, id)
                    .unwrap()
                    .assume_deduplicated()
                    .raw
            })
            .collect::<Vec<_>>();
        let hal_desc = hal::PipelineLayoutDescriptor {
            label: desc.label.to_hal(self.instance_flags),
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
                    // manually add a dependency to BGL
                    let (id, layout) = get_bind_group_layout(bgl_guard, id::Valid(id));
                    layout.multi_ref_count.inc();
                    id
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
                    let bgl = self.create_bind_group_layout(self_id, &None, map)?;
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

    pub(super) fn create_compute_pipeline<G: GlobalIdentityHandlerFactory>(
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
        let mut shader_binding_sizes = FastHashMap::default();

        let io = validation::StageIo::default();
        let (shader_module_guard, _) = hub.shader_modules.read(&mut token);

        let shader_module = shader_module_guard
            .get(desc.stage.module)
            .map_err(|_| validation::StageError::InvalidModule)?;

        if shader_module.device_id.value.0 != self_id {
            return Err(DeviceError::WrongDevice.into());
        }

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
                    &mut shader_binding_sizes,
                    &desc.stage.entry_point,
                    flag,
                    io,
                    None,
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

        if layout.device_id.value.0 != self_id {
            return Err(DeviceError::WrongDevice.into());
        }

        let late_sized_buffer_groups =
            Device::make_late_sized_buffer_groups(&shader_binding_sizes, layout, &*bgl_guard);

        let pipeline_desc = hal::ComputePipelineDescriptor {
            label: desc.label.to_hal(self.instance_flags),
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
            late_sized_buffer_groups,
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
        };
        Ok(pipeline)
    }

    pub(super) fn create_render_pipeline<G: GlobalIdentityHandlerFactory>(
        &self,
        self_id: id::DeviceId,
        adapter: &Adapter<A>,
        desc: &pipeline::RenderPipelineDescriptor,
        implicit_context: Option<ImplicitPipelineContext>,
        hub: &Hub<A, G>,
        token: &mut Token<Self>,
    ) -> Result<pipeline::RenderPipeline<A>, pipeline::CreateRenderPipelineError> {
        use wgt::TextureFormatFeatureFlags as Tfff;

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
        let mut shader_binding_sizes = FastHashMap::default();

        let num_attachments = desc.fragment.as_ref().map(|f| f.targets.len()).unwrap_or(0);
        if num_attachments > hal::MAX_COLOR_ATTACHMENTS {
            return Err(pipeline::CreateRenderPipelineError::ColorAttachment(
                command::ColorAttachmentError::TooMany {
                    given: num_attachments,
                    limit: hal::MAX_COLOR_ATTACHMENTS,
                },
            ));
        }

        let color_targets = desc
            .fragment
            .as_ref()
            .map_or(&[][..], |fragment| &fragment.targets);
        let depth_stencil_state = desc.depth_stencil.as_ref();

        let cts: ArrayVec<_, { hal::MAX_COLOR_ATTACHMENTS }> =
            color_targets.iter().filter_map(|x| x.as_ref()).collect();
        if !cts.is_empty() && {
            let first = &cts[0];
            cts[1..]
                .iter()
                .any(|ct| ct.write_mask != first.write_mask || ct.blend != first.blend)
        } {
            log::info!("Color targets: {:?}", color_targets);
            self.require_downlevel_flags(wgt::DownlevelFlags::INDEPENDENT_BLEND)?;
        }

        let mut io = validation::StageIo::default();
        let mut validated_stages = wgt::ShaderStages::empty();

        let mut vertex_steps = Vec::with_capacity(desc.vertex.buffers.len());
        let mut vertex_buffers = Vec::with_capacity(desc.vertex.buffers.len());
        let mut total_attributes = 0;
        let mut shader_expects_dual_source_blending = false;
        let mut pipeline_expects_dual_source_blending = false;
        for (i, vb_state) in desc.vertex.buffers.iter().enumerate() {
            vertex_steps.push(pipeline::VertexStep {
                stride: vb_state.array_stride,
                mode: vb_state.step_mode,
            });
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
            vertex_buffers.push(hal::VertexBufferLayout {
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

                let previous = io.insert(
                    attribute.shader_location,
                    validation::InterfaceVar::vertex_attribute(attribute.format),
                );

                if previous.is_some() {
                    return Err(pipeline::CreateRenderPipelineError::ShaderLocationClash(
                        attribute.shader_location,
                    ));
                }
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

        if desc.primitive.unclipped_depth {
            self.require_features(wgt::Features::DEPTH_CLIP_CONTROL)?;
        }

        if desc.primitive.polygon_mode == wgt::PolygonMode::Line {
            self.require_features(wgt::Features::POLYGON_MODE_LINE)?;
        }
        if desc.primitive.polygon_mode == wgt::PolygonMode::Point {
            self.require_features(wgt::Features::POLYGON_MODE_POINT)?;
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
            if let Some(cs) = cs.as_ref() {
                let error = loop {
                    if cs.write_mask.contains_invalid_bits() {
                        break Some(pipeline::ColorStateError::InvalidWriteMask(cs.write_mask));
                    }

                    let format_features = self.describe_format_features(adapter, cs.format)?;
                    if !format_features
                        .allowed_usages
                        .contains(wgt::TextureUsages::RENDER_ATTACHMENT)
                    {
                        break Some(pipeline::ColorStateError::FormatNotRenderable(cs.format));
                    }
                    let blendable = format_features.flags.contains(Tfff::BLENDABLE);
                    let filterable = format_features.flags.contains(Tfff::FILTERABLE);
                    let adapter_specific = self
                        .features
                        .contains(wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);
                    // according to WebGPU specifications the texture needs to be
                    // [`TextureFormatFeatureFlags::FILTERABLE`] if blending is set - use
                    // [`Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`] to elude
                    // this limitation
                    if cs.blend.is_some() && (!blendable || (!filterable && !adapter_specific)) {
                        break Some(pipeline::ColorStateError::FormatNotBlendable(cs.format));
                    }
                    if !hal::FormatAspects::from(cs.format).contains(hal::FormatAspects::COLOR) {
                        break Some(pipeline::ColorStateError::FormatNotColor(cs.format));
                    }
                    if desc.multisample.count > 1
                        && !format_features
                            .flags
                            .sample_count_supported(desc.multisample.count)
                    {
                        break Some(pipeline::ColorStateError::InvalidSampleCount(
                            desc.multisample.count,
                            cs.format,
                            cs.format
                                .guaranteed_format_features(self.features)
                                .flags
                                .supported_sample_counts(),
                            adapter
                                .get_texture_format_features(cs.format)
                                .flags
                                .supported_sample_counts(),
                        ));
                    }
                    if let Some(blend_mode) = cs.blend {
                        for factor in [
                            blend_mode.color.src_factor,
                            blend_mode.color.dst_factor,
                            blend_mode.alpha.src_factor,
                            blend_mode.alpha.dst_factor,
                        ] {
                            if factor.ref_second_blend_source() {
                                self.require_features(wgt::Features::DUAL_SOURCE_BLENDING)?;
                                if i == 0 {
                                    pipeline_expects_dual_source_blending = true;
                                    break;
                                } else {
                                    return Err(crate::pipeline::CreateRenderPipelineError
                            ::BlendFactorOnUnsupportedTarget { factor, target: i as u32 });
                                }
                            }
                        }
                    }
                    break None;
                };
                if let Some(e) = error {
                    return Err(pipeline::CreateRenderPipelineError::ColorState(i as u8, e));
                }
            }
        }

        if let Some(ds) = depth_stencil_state {
            let error = loop {
                let format_features = self.describe_format_features(adapter, ds.format)?;
                if !format_features
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
                if desc.multisample.count > 1
                    && !format_features
                        .flags
                        .sample_count_supported(desc.multisample.count)
                {
                    break Some(pipeline::DepthStencilStateError::InvalidSampleCount(
                        desc.multisample.count,
                        ds.format,
                        ds.format
                            .guaranteed_format_features(self.features)
                            .flags
                            .supported_sample_counts(),
                        adapter
                            .get_texture_format_features(ds.format)
                            .flags
                            .supported_sample_counts(),
                    ));
                }

                break None;
            };
            if let Some(e) = error {
                return Err(pipeline::CreateRenderPipelineError::DepthStencilState(e));
            }

            if ds.bias.clamp != 0.0 {
                self.require_downlevel_flags(wgt::DownlevelFlags::DEPTH_BIAS_CLAMP)?;
            }
        }

        if desc.layout.is_none() {
            for _ in 0..self.limits.max_bind_groups {
                derived_group_layouts.push(binding_model::BindEntryMap::default());
            }
        }

        let samples = {
            let sc = desc.multisample.count;
            if sc == 0 || sc > 32 || !conv::is_power_of_two_u32(sc) {
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

            if shader_module.device_id.value.0 != self_id {
                return Err(DeviceError::WrongDevice.into());
            }

            let provided_layouts = match desc.layout {
                Some(pipeline_layout_id) => {
                    let pipeline_layout = pipeline_layout_guard
                        .get(pipeline_layout_id)
                        .map_err(|_| pipeline::CreateRenderPipelineError::InvalidLayout)?;

                    if pipeline_layout.device_id.value.0 != self_id {
                        return Err(DeviceError::WrongDevice.into());
                    }

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
                        &mut shader_binding_sizes,
                        &stage.entry_point,
                        flag,
                        io,
                        desc.depth_stencil.as_ref().map(|d| d.depth_compare),
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
                                &mut shader_binding_sizes,
                                &fragment.stage.entry_point,
                                flag,
                                io,
                                desc.depth_stencil.as_ref().map(|d| d.depth_compare),
                            )
                            .map_err(|error| pipeline::CreateRenderPipelineError::Stage {
                                stage: flag,
                                error,
                            })?;
                        validated_stages |= flag;
                    }
                }

                if let Some(ref interface) = shader_module.interface {
                    shader_expects_dual_source_blending = interface
                        .fragment_uses_dual_source_blending(&fragment.stage.entry_point)
                        .map_err(|error| pipeline::CreateRenderPipelineError::Stage {
                            stage: flag,
                            error,
                        })?;
                }

                Some(hal::ProgrammableStage {
                    module: &shader_module.raw,
                    entry_point: fragment.stage.entry_point.as_ref(),
                })
            }
            None => None,
        };

        if !pipeline_expects_dual_source_blending && shader_expects_dual_source_blending {
            return Err(
                pipeline::CreateRenderPipelineError::ShaderExpectsPipelineToUseDualSourceBlending,
            );
        }
        if pipeline_expects_dual_source_blending && !shader_expects_dual_source_blending {
            return Err(
                pipeline::CreateRenderPipelineError::PipelineExpectsShaderToUseDualSourceBlending,
            );
        }

        if validated_stages.contains(wgt::ShaderStages::FRAGMENT) {
            for (i, output) in io.iter() {
                match color_targets.get(*i as usize) {
                    Some(&Some(ref state)) => {
                        validation::check_texture_format(state.format, &output.ty).map_err(
                            |pipeline| {
                                pipeline::CreateRenderPipelineError::ColorState(
                                    *i as u8,
                                    pipeline::ColorStateError::IncompatibleFormat {
                                        pipeline,
                                        shader: output.ty,
                                    },
                                )
                            },
                        )?;
                    }
                    _ => {
                        log::info!(
                            "The fragment stage {:?} output @location({}) values are ignored",
                            fragment_stage
                                .as_ref()
                                .map_or("", |stage| stage.entry_point),
                            i
                        );
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

        // Multiview is only supported if the feature is enabled
        if desc.multiview.is_some() {
            self.require_features(wgt::Features::MULTIVIEW)?;
        }

        if !self
            .downlevel
            .flags
            .contains(wgt::DownlevelFlags::BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED)
        {
            for (binding, size) in shader_binding_sizes.iter() {
                if size.get() % 16 != 0 {
                    return Err(pipeline::CreateRenderPipelineError::UnalignedShader {
                        binding: binding.binding,
                        group: binding.group,
                        size: size.get(),
                    });
                }
            }
        }

        let late_sized_buffer_groups =
            Device::make_late_sized_buffer_groups(&shader_binding_sizes, layout, &*bgl_guard);

        let pipeline_desc = hal::RenderPipelineDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            layout: &layout.raw,
            vertex_buffers: &vertex_buffers,
            vertex_stage,
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment_stage,
            color_targets,
            multiview: desc.multiview,
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
                colors: color_targets
                    .iter()
                    .map(|state| state.as_ref().map(|s| s.format))
                    .collect(),
                resolves: ArrayVec::new(),
                depth_stencil: depth_stencil_state.as_ref().map(|state| state.format),
            },
            sample_count: samples,
            multiview: desc.multiview,
        };

        let mut flags = pipeline::PipelineFlags::empty();
        for state in color_targets.iter().filter_map(|s| s.as_ref()) {
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
            if !ds.is_depth_read_only() {
                flags |= pipeline::PipelineFlags::WRITES_DEPTH;
            }
            if !ds.is_stencil_read_only(desc.primitive.cull_mode) {
                flags |= pipeline::PipelineFlags::WRITES_STENCIL;
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
            vertex_steps,
            late_sized_buffer_groups,
            life_guard: LifeGuard::new(desc.label.borrow_or_default()),
        };
        Ok(pipeline)
    }

    pub(super) fn describe_format_features(
        &self,
        adapter: &Adapter<A>,
        format: TextureFormat,
    ) -> Result<wgt::TextureFormatFeatures, MissingFeatures> {
        self.require_features(format.required_features())?;

        let using_device_features = self
            .features
            .contains(wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);
        // If we're running downlevel, we need to manually ask the backend what
        // we can use as we can't trust WebGPU.
        let downlevel = !self.downlevel.is_webgpu_compliant();

        if using_device_features || downlevel {
            Ok(adapter.get_texture_format_features(format))
        } else {
            Ok(format.guaranteed_format_features(self.features))
        }
    }

    pub(super) fn wait_for_submit(
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

    pub(super) fn create_query_set(
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

        let hal_desc = desc.map_label(|label| label.to_hal(self.instance_flags));
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

    pub(crate) fn lose<'this, 'token: 'this>(
        &'this mut self,
        token: &mut Token<'token, Self>,
        message: &str,
    ) {
        // Follow the steps at https://gpuweb.github.io/gpuweb/#lose-the-device.

        // Mark the device explicitly as invalid. This is checked in various
        // places to prevent new work from being submitted.
        self.valid = false;

        // 1) Resolve the GPUDevice device.lost promise.
        let mut life_tracker = self.lock_life(token);
        if life_tracker.device_lost_closure.is_some() {
            let device_lost_closure = life_tracker.device_lost_closure.take().unwrap();
            device_lost_closure.call(DeviceLostReason::Unknown, message.to_string());
        }

        // 2) Complete any outstanding mapAsync() steps.
        // 3) Complete any outstanding onSubmittedWorkDone() steps.

        // These parts are passively accomplished by setting valid to false,
        // since that will prevent any new work from being added to the queues.
        // Future calls to poll_devices will continue to check the work queues
        // until they are cleared, and then drop the device.
    }
}

impl<A: HalApi> Device<A> {
    pub(crate) fn destroy_buffer(&self, buffer: Buffer<A>) {
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
            self.raw.destroy_buffer(self.zero_buffer);
            self.raw.destroy_fence(self.fence);
            self.raw.exit(self.queue);
        }
    }
}

impl<A: HalApi> crate::resource::Resource for Device<A> {
    const TYPE: &'static str = "Device";

    fn life_guard(&self) -> &LifeGuard {
        &self.life_guard
    }
}
