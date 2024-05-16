#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    binding_model::{self, BindGroup, BindGroupLayout, BindGroupLayoutEntryError},
    command, conv,
    device::{
        bgl, create_validator,
        life::{LifetimeTracker, WaitIdleError},
        queue::PendingWrites,
        AttachmentData, DeviceLostInvocation, MissingDownlevelFlags, MissingFeatures,
        RenderPassContext, CLEANUP_WAIT_MS,
    },
    hal_api::HalApi,
    hal_label,
    hub::Hub,
    id,
    init_tracker::{
        BufferInitTracker, BufferInitTrackerAction, MemoryInitKind, TextureInitRange,
        TextureInitTracker, TextureInitTrackerAction,
    },
    instance::Adapter,
    lock::{rank, Mutex, MutexGuard, RwLock},
    pipeline,
    pool::ResourcePool,
    registry::Registry,
    resource::{
        self, Buffer, QuerySet, Resource, ResourceInfo, ResourceType, Sampler, Texture,
        TextureView, TextureViewNotRenderableReason,
    },
    resource_log,
    snatch::{SnatchGuard, SnatchLock, Snatchable},
    storage::Storage,
    track::{
        BindGroupStates, TextureSelector, Tracker, TrackerIndexAllocators, UsageScope,
        UsageScopePool,
    },
    validation::{
        self, check_buffer_usage, check_texture_usage, validate_color_attachment_bytes_per_sample,
    },
    FastHashMap, LabelHelpers as _, SubmissionIndex,
};

use arrayvec::ArrayVec;
use hal::{CommandEncoder as _, Device as _};
use once_cell::sync::OnceCell;

use smallvec::SmallVec;
use thiserror::Error;
use wgt::{DeviceLostReason, TextureFormat, TextureSampleType, TextureViewDimension};

use std::{
    borrow::Cow,
    iter,
    num::NonZeroU32,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Weak,
    },
};

use super::{
    life::ResourceMaps,
    queue::{self, Queue},
    DeviceDescriptor, DeviceError, ImplicitPipelineContext, UserClosures, ENTRYPOINT_FAILURE_ERROR,
    IMPLICIT_BIND_GROUP_LAYOUT_ERROR_LABEL, ZERO_BUFFER_SIZE,
};

/// Structure describing a logical device. Some members are internally mutable,
/// stored behind mutexes.
///
/// TODO: establish clear order of locking for these:
/// `life_tracker`, `trackers`, `render_passes`, `pending_writes`, `trace`.
///
/// Currently, the rules are:
/// 1. `life_tracker` is locked after `hub.devices`, enforced by the type system
/// 1. `self.trackers` is locked last (unenforced)
/// 1. `self.trace` is locked last (unenforced)
///
/// Right now avoid locking twice same resource or registry in a call execution
/// and minimize the locking to the minimum scope possible
/// Unless otherwise specified, no lock may be acquired while holding another lock.
/// This means that you must inspect function calls made while a lock is held
/// to see what locks the callee may try to acquire.
///
/// As far as this point:
/// device_maintain_ids locks Device::lifetime_tracker, and calls...
/// triage_suspected locks Device::trackers, and calls...
/// Registry::unregister locks Registry::storage
///
/// Important:
/// When locking pending_writes please check that trackers is not locked
/// trackers should be locked only when needed for the shortest time possible
pub struct Device<A: HalApi> {
    raw: Option<A::Device>,
    pub(crate) adapter: Arc<Adapter<A>>,
    pub(crate) queue: OnceCell<Weak<Queue<A>>>,
    queue_to_drop: OnceCell<A::Queue>,
    pub(crate) zero_buffer: Option<A::Buffer>,
    pub(crate) info: ResourceInfo<Device<A>>,

    pub(crate) command_allocator: command::CommandAllocator<A>,
    //Note: The submission index here corresponds to the last submission that is done.
    pub(crate) active_submission_index: AtomicU64, //SubmissionIndex,
    // NOTE: if both are needed, the `snatchable_lock` must be consistently acquired before the
    // `fence` lock to avoid deadlocks.
    pub(crate) fence: RwLock<Option<A::Fence>>,
    pub(crate) snatchable_lock: SnatchLock,

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
    pub(crate) valid: AtomicBool,

    /// All live resources allocated with this [`Device`].
    ///
    /// Has to be locked temporarily only (locked last)
    /// and never before pending_writes
    pub(crate) trackers: Mutex<Tracker<A>>,
    pub(crate) tracker_indices: TrackerIndexAllocators,
    // Life tracker should be locked right after the device and before anything else.
    life_tracker: Mutex<LifetimeTracker<A>>,
    /// Pool of bind group layouts, allowing deduplication.
    pub(crate) bgl_pool: ResourcePool<bgl::EntryMap, BindGroupLayout<A>>,
    pub(crate) alignments: hal::Alignments,
    pub(crate) limits: wgt::Limits,
    pub(crate) features: wgt::Features,
    pub(crate) downlevel: wgt::DownlevelCapabilities,
    pub(crate) instance_flags: wgt::InstanceFlags,
    pub(crate) pending_writes: Mutex<Option<PendingWrites<A>>>,
    pub(crate) deferred_destroy: Mutex<Vec<DeferredDestroy<A>>>,
    #[cfg(feature = "trace")]
    pub(crate) trace: Mutex<Option<trace::Trace>>,
    pub(crate) usage_scopes: UsageScopePool<A>,

    /// Temporary storage, cleared at the start of every call,
    /// retained only to save allocations.
    temp_suspected: Mutex<Option<ResourceMaps<A>>>,
}

pub(crate) enum DeferredDestroy<A: HalApi> {
    TextureView(Weak<TextureView<A>>),
    BindGroup(Weak<BindGroup<A>>),
}

impl<A: HalApi> std::fmt::Debug for Device<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("adapter", &self.adapter.info.label())
            .field("limits", &self.limits)
            .field("features", &self.features)
            .field("downlevel", &self.downlevel)
            .finish()
    }
}

impl<A: HalApi> Drop for Device<A> {
    fn drop(&mut self) {
        resource_log!("Destroy raw Device {:?}", self.info.label());
        let raw = self.raw.take().unwrap();
        let pending_writes = self.pending_writes.lock().take().unwrap();
        pending_writes.dispose(&raw);
        self.command_allocator.dispose(&raw);
        unsafe {
            raw.destroy_buffer(self.zero_buffer.take().unwrap());
            raw.destroy_fence(self.fence.write().take().unwrap());
            let queue = self.queue_to_drop.take().unwrap();
            raw.exit(queue);
        }
    }
}

#[derive(Clone, Debug, Error)]
pub enum CreateDeviceError {
    #[error("Not enough memory left to create device")]
    OutOfMemory,
    #[error("Failed to create internal buffer for initializing textures")]
    FailedToCreateZeroBuffer(#[from] DeviceError),
}

impl<A: HalApi> Device<A> {
    pub(crate) fn raw(&self) -> &A::Device {
        self.raw.as_ref().unwrap()
    }
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
        raw_device: A::Device,
        raw_queue: &A::Queue,
        adapter: &Arc<Adapter<A>>,
        desc: &DeviceDescriptor,
        trace_path: Option<&std::path::Path>,
        instance_flags: wgt::InstanceFlags,
    ) -> Result<Self, CreateDeviceError> {
        #[cfg(not(feature = "trace"))]
        if let Some(_) = trace_path {
            log::error!("Feature 'trace' is not enabled");
        }
        let fence =
            unsafe { raw_device.create_fence() }.map_err(|_| CreateDeviceError::OutOfMemory)?;

        let command_allocator = command::CommandAllocator::new();
        let pending_encoder = command_allocator
            .acquire_encoder(&raw_device, raw_queue)
            .map_err(|_| CreateDeviceError::OutOfMemory)?;
        let mut pending_writes = PendingWrites::<A>::new(pending_encoder);

        // Create zeroed buffer used for texture clears.
        let zero_buffer = unsafe {
            raw_device
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

        let alignments = adapter.raw.capabilities.alignments.clone();
        let downlevel = adapter.raw.capabilities.downlevel.clone();

        Ok(Self {
            raw: Some(raw_device),
            adapter: adapter.clone(),
            queue: OnceCell::new(),
            queue_to_drop: OnceCell::new(),
            zero_buffer: Some(zero_buffer),
            info: ResourceInfo::new("<device>", None),
            command_allocator,
            active_submission_index: AtomicU64::new(0),
            fence: RwLock::new(rank::DEVICE_FENCE, Some(fence)),
            snatchable_lock: unsafe { SnatchLock::new(rank::DEVICE_SNATCHABLE_LOCK) },
            valid: AtomicBool::new(true),
            trackers: Mutex::new(rank::DEVICE_TRACKERS, Tracker::new()),
            tracker_indices: TrackerIndexAllocators::new(),
            life_tracker: Mutex::new(rank::DEVICE_LIFE_TRACKER, LifetimeTracker::new()),
            temp_suspected: Mutex::new(rank::DEVICE_TEMP_SUSPECTED, Some(ResourceMaps::new())),
            bgl_pool: ResourcePool::new(),
            #[cfg(feature = "trace")]
            trace: Mutex::new(
                rank::DEVICE_TRACE,
                trace_path.and_then(|path| match trace::Trace::new(path) {
                    Ok(mut trace) => {
                        trace.add(trace::Action::Init {
                            desc: desc.clone(),
                            backend: A::VARIANT,
                        });
                        Some(trace)
                    }
                    Err(e) => {
                        log::error!("Unable to start a trace in '{path:?}': {e}");
                        None
                    }
                }),
            ),
            alignments,
            limits: desc.required_limits.clone(),
            features: desc.required_features,
            downlevel,
            instance_flags,
            pending_writes: Mutex::new(rank::DEVICE_PENDING_WRITES, Some(pending_writes)),
            deferred_destroy: Mutex::new(rank::DEVICE_DEFERRED_DESTROY, Vec::new()),
            usage_scopes: Mutex::new(rank::DEVICE_USAGE_SCOPES, Default::default()),
        })
    }

    pub fn is_valid(&self) -> bool {
        self.valid.load(Ordering::Acquire)
    }

    pub(crate) fn release_queue(&self, queue: A::Queue) {
        assert!(self.queue_to_drop.set(queue).is_ok());
    }

    pub(crate) fn lock_life<'a>(&'a self) -> MutexGuard<'a, LifetimeTracker<A>> {
        self.life_tracker.lock()
    }

    /// Run some destroy operations that were deferred.
    ///
    /// Destroying the resources requires taking a write lock on the device's snatch lock,
    /// so a good reason for deferring resource destruction is when we don't know for sure
    /// how risky it is to take the lock (typically, it shouldn't be taken from the drop
    /// implementation of a reference-counted structure).
    /// The snatch lock must not be held while this function is called.
    pub(crate) fn deferred_resource_destruction(&self) {
        while let Some(item) = self.deferred_destroy.lock().pop() {
            match item {
                DeferredDestroy::TextureView(view) => {
                    let Some(view) = view.upgrade() else {
                        continue;
                    };
                    let Some(raw_view) = view.raw.snatch(self.snatchable_lock.write()) else {
                        continue;
                    };

                    resource_log!("Destroy raw TextureView (destroyed) {:?}", view.label());
                    #[cfg(feature = "trace")]
                    if let Some(t) = self.trace.lock().as_mut() {
                        t.add(trace::Action::DestroyTextureView(view.info.id()));
                    }
                    unsafe {
                        use hal::Device;
                        self.raw().destroy_texture_view(raw_view);
                    }
                }
                DeferredDestroy::BindGroup(bind_group) => {
                    let Some(bind_group) = bind_group.upgrade() else {
                        continue;
                    };
                    let Some(raw_bind_group) = bind_group.raw.snatch(self.snatchable_lock.write())
                    else {
                        continue;
                    };

                    resource_log!("Destroy raw BindGroup (destroyed) {:?}", bind_group.label());
                    #[cfg(feature = "trace")]
                    if let Some(t) = self.trace.lock().as_mut() {
                        t.add(trace::Action::DestroyBindGroup(bind_group.info.id()));
                    }
                    unsafe {
                        use hal::Device;
                        self.raw().destroy_bind_group(raw_bind_group);
                    }
                }
            }
        }
    }

    pub fn get_queue(&self) -> Option<Arc<Queue<A>>> {
        self.queue.get().as_ref()?.upgrade()
    }

    pub fn set_queue(&self, queue: Arc<Queue<A>>) {
        assert!(self.queue.set(Arc::downgrade(&queue)).is_ok());
    }

    /// Check this device for completed commands.
    ///
    /// The `maintain` argument tells how the maintenance function should behave, either
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
    pub(crate) fn maintain<'this>(
        &'this self,
        fence_guard: crate::lock::RwLockReadGuard<Option<A::Fence>>,
        maintain: wgt::Maintain<queue::WrappedSubmissionIndex>,
        snatch_guard: SnatchGuard,
    ) -> Result<(UserClosures, bool), WaitIdleError> {
        profiling::scope!("Device::maintain");
        let fence = fence_guard.as_ref().unwrap();
        let last_done_index = if maintain.is_wait() {
            let index_to_wait_for = match maintain {
                wgt::Maintain::WaitForSubmissionIndex(submission_index) => {
                    // We don't need to check to see if the queue id matches
                    // as we already checked this from inside the poll call.
                    submission_index.index
                }
                _ => self.active_submission_index.load(Ordering::Relaxed),
            };
            unsafe {
                self.raw
                    .as_ref()
                    .unwrap()
                    .wait(fence, index_to_wait_for, CLEANUP_WAIT_MS)
                    .map_err(DeviceError::from)?
            };
            index_to_wait_for
        } else {
            unsafe {
                self.raw
                    .as_ref()
                    .unwrap()
                    .get_fence_value(fence)
                    .map_err(DeviceError::from)?
            }
        };

        let mut life_tracker = self.lock_life();
        let submission_closures =
            life_tracker.triage_submissions(last_done_index, &self.command_allocator);

        life_tracker.triage_suspected(&self.trackers);

        life_tracker.triage_mapped();

        let mapping_closures =
            life_tracker.handle_mapping(self.raw(), &self.trackers, &snatch_guard);

        let queue_empty = life_tracker.queue_empty();

        // Detect if we have been destroyed and now need to lose the device.
        // If we are invalid (set at start of destroy) and our queue is empty,
        // and we have a DeviceLostClosure, return the closure to be called by
        // our caller. This will complete the steps for both destroy and for
        // "lose the device".
        let mut device_lost_invocations = SmallVec::new();
        let mut should_release_gpu_resource = false;
        if !self.is_valid() && queue_empty {
            // We can release gpu resources associated with this device (but not
            // while holding the life_tracker lock).
            should_release_gpu_resource = true;

            // If we have a DeviceLostClosure, build an invocation with the
            // reason DeviceLostReason::Destroyed and no message.
            if life_tracker.device_lost_closure.is_some() {
                device_lost_invocations.push(DeviceLostInvocation {
                    closure: life_tracker.device_lost_closure.take().unwrap(),
                    reason: DeviceLostReason::Destroyed,
                    message: String::new(),
                });
            }
        }

        // Don't hold the locks while calling release_gpu_resources.
        drop(life_tracker);
        drop(fence_guard);
        drop(snatch_guard);

        if should_release_gpu_resource {
            self.release_gpu_resources();
        }

        let closures = UserClosures {
            mappings: mapping_closures,
            submissions: submission_closures,
            device_lost_invocations,
        };
        Ok((closures, queue_empty))
    }

    pub(crate) fn untrack(&self, trackers: &Tracker<A>) {
        // If we have a previously allocated `ResourceMap`, just use that.
        let mut temp_suspected = self
            .temp_suspected
            .lock()
            .take()
            .unwrap_or_else(|| ResourceMaps::new());
        temp_suspected.clear();

        // As the tracker is cleared/dropped, we need to consider all the resources
        // that it references for destruction in the next GC pass.
        {
            for resource in trackers.buffers.used_resources() {
                if resource.is_unique() {
                    temp_suspected
                        .buffers
                        .insert(resource.as_info().tracker_index(), resource.clone());
                }
            }
            for resource in trackers.textures.used_resources() {
                if resource.is_unique() {
                    temp_suspected
                        .textures
                        .insert(resource.as_info().tracker_index(), resource.clone());
                }
            }
            for resource in trackers.views.used_resources() {
                if resource.is_unique() {
                    temp_suspected
                        .texture_views
                        .insert(resource.as_info().tracker_index(), resource.clone());
                }
            }
            for resource in trackers.bind_groups.used_resources() {
                if resource.is_unique() {
                    temp_suspected
                        .bind_groups
                        .insert(resource.as_info().tracker_index(), resource.clone());
                }
            }
            for resource in trackers.samplers.used_resources() {
                if resource.is_unique() {
                    temp_suspected
                        .samplers
                        .insert(resource.as_info().tracker_index(), resource.clone());
                }
            }
            for resource in trackers.compute_pipelines.used_resources() {
                if resource.is_unique() {
                    temp_suspected
                        .compute_pipelines
                        .insert(resource.as_info().tracker_index(), resource.clone());
                }
            }
            for resource in trackers.render_pipelines.used_resources() {
                if resource.is_unique() {
                    temp_suspected
                        .render_pipelines
                        .insert(resource.as_info().tracker_index(), resource.clone());
                }
            }
            for resource in trackers.query_sets.used_resources() {
                if resource.is_unique() {
                    temp_suspected
                        .query_sets
                        .insert(resource.as_info().tracker_index(), resource.clone());
                }
            }
        }
        self.lock_life()
            .suspected_resources
            .extend(&mut temp_suspected);
        // Save this resource map for later reuse.
        *self.temp_suspected.lock() = Some(temp_suspected);
    }

    pub(crate) fn create_buffer(
        self: &Arc<Self>,
        desc: &resource::BufferDescriptor,
        transient: bool,
    ) -> Result<Buffer<A>, resource::CreateBufferError> {
        debug_assert_eq!(self.as_info().id().backend(), A::VARIANT);

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
        let buffer = unsafe { self.raw().create_buffer(&hal_desc) }.map_err(DeviceError::from)?;

        Ok(Buffer {
            raw: Snatchable::new(buffer),
            device: self.clone(),
            usage: desc.usage,
            size: desc.size,
            initialization_status: RwLock::new(
                rank::BUFFER_INITIALIZATION_STATUS,
                BufferInitTracker::new(aligned_size),
            ),
            sync_mapped_writes: Mutex::new(rank::BUFFER_SYNC_MAPPED_WRITES, None),
            map_state: Mutex::new(rank::BUFFER_MAP_STATE, resource::BufferMapState::Idle),
            info: ResourceInfo::new(
                desc.label.borrow_or_default(),
                Some(self.tracker_indices.buffers.clone()),
            ),
            bind_groups: Mutex::new(rank::BUFFER_BIND_GROUPS, Vec::new()),
        })
    }

    pub(crate) fn create_texture_from_hal(
        self: &Arc<Self>,
        hal_texture: A::Texture,
        hal_usage: hal::TextureUses,
        desc: &resource::TextureDescriptor,
        format_features: wgt::TextureFormatFeatures,
        clear_mode: resource::TextureClearMode<A>,
    ) -> Texture<A> {
        debug_assert_eq!(self.as_info().id().backend(), A::VARIANT);

        Texture {
            inner: Snatchable::new(resource::TextureInner::Native { raw: hal_texture }),
            device: self.clone(),
            desc: desc.map_label(|_| ()),
            hal_usage,
            format_features,
            initialization_status: RwLock::new(
                rank::TEXTURE_INITIALIZATION_STATUS,
                TextureInitTracker::new(desc.mip_level_count, desc.array_layer_count()),
            ),
            full_range: TextureSelector {
                mips: 0..desc.mip_level_count,
                layers: 0..desc.array_layer_count(),
            },
            info: ResourceInfo::new(
                desc.label.borrow_or_default(),
                Some(self.tracker_indices.textures.clone()),
            ),
            clear_mode: RwLock::new(rank::TEXTURE_CLEAR_MODE, clear_mode),
            views: Mutex::new(rank::TEXTURE_VIEWS, Vec::new()),
            bind_groups: Mutex::new(rank::TEXTURE_BIND_GROUPS, Vec::new()),
        }
    }

    pub fn create_buffer_from_hal(
        self: &Arc<Self>,
        hal_buffer: A::Buffer,
        desc: &resource::BufferDescriptor,
    ) -> Buffer<A> {
        debug_assert_eq!(self.as_info().id().backend(), A::VARIANT);

        Buffer {
            raw: Snatchable::new(hal_buffer),
            device: self.clone(),
            usage: desc.usage,
            size: desc.size,
            initialization_status: RwLock::new(
                rank::BUFFER_INITIALIZATION_STATUS,
                BufferInitTracker::new(0),
            ),
            sync_mapped_writes: Mutex::new(rank::BUFFER_SYNC_MAPPED_WRITES, None),
            map_state: Mutex::new(rank::BUFFER_MAP_STATE, resource::BufferMapState::Idle),
            info: ResourceInfo::new(
                desc.label.borrow_or_default(),
                Some(self.tracker_indices.buffers.clone()),
            ),
            bind_groups: Mutex::new(rank::BUFFER_BIND_GROUPS, Vec::new()),
        }
    }

    pub(crate) fn create_texture(
        self: &Arc<Self>,
        adapter: &Adapter<A>,
        desc: &resource::TextureDescriptor,
    ) -> Result<Texture<A>, resource::CreateTextureError> {
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

        {
            let (width_multiple, height_multiple) = desc.format.size_multiple_requirement();

            if desc.size.width % width_multiple != 0 {
                return Err(CreateTextureError::InvalidDimension(
                    TextureDimensionError::WidthNotMultipleOf {
                        width: desc.size.width,
                        multiple: width_multiple,
                        format: desc.format,
                    },
                ));
            }

            if desc.size.height % height_multiple != 0 {
                return Err(CreateTextureError::InvalidDimension(
                    TextureDimensionError::HeightNotMultipleOf {
                        height: desc.size.height,
                        multiple: height_multiple,
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

        let hal_usage = conv::map_texture_usage_for_texture(desc, &format_features);

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
                .as_ref()
                .unwrap()
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
                wgt::TextureDimension::D1 => TextureViewDimension::D1,
                wgt::TextureDimension::D2 => TextureViewDimension::D2,
                wgt::TextureDimension::D3 => unreachable!(),
            };

            let clear_label = hal_label(
                Some("(wgpu internal) clear texture view"),
                self.instance_flags,
            );

            let mut clear_views = SmallVec::new();
            for mip_level in 0..desc.mip_level_count {
                for array_layer in 0..desc.size.depth_or_array_layers {
                    macro_rules! push_clear_view {
                        ($format:expr, $aspect:expr) => {
                            let desc = hal::TextureViewDescriptor {
                                label: clear_label,
                                format: $format,
                                dimension,
                                usage,
                                range: wgt::ImageSubresourceRange {
                                    aspect: $aspect,
                                    base_mip_level: mip_level,
                                    mip_level_count: Some(1),
                                    base_array_layer: array_layer,
                                    array_layer_count: Some(1),
                                },
                            };
                            clear_views.push(Some(
                                unsafe { self.raw().create_texture_view(&raw_texture, &desc) }
                                    .map_err(DeviceError::from)?,
                            ));
                        };
                    }

                    if let Some(planes) = desc.format.planes() {
                        for plane in 0..planes {
                            let aspect = wgt::TextureAspect::from_plane(plane).unwrap();
                            let format = desc.format.aspect_specific_format(aspect).unwrap();
                            push_clear_view!(format, aspect);
                        }
                    } else {
                        push_clear_view!(desc.format, wgt::TextureAspect::All);
                    }
                }
            }
            resource::TextureClearMode::RenderPass {
                clear_views,
                is_color,
            }
        } else {
            resource::TextureClearMode::BufferCopy
        };

        let mut texture =
            self.create_texture_from_hal(raw_texture, hal_usage, desc, format_features, clear_mode);
        texture.hal_usage = hal_usage;
        Ok(texture)
    }

    pub(crate) fn create_texture_view(
        self: &Arc<Self>,
        texture: &Arc<Texture<A>>,
        desc: &resource::TextureViewDescriptor,
    ) -> Result<TextureView<A>, resource::CreateTextureViewError> {
        let snatch_guard = texture.device.snatchable_lock.read();

        let texture_raw = texture
            .raw(&snatch_guard)
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
                wgt::TextureDimension::D1 => TextureViewDimension::D1,
                wgt::TextureDimension::D2 => {
                    if texture.desc.array_layer_count() == 1 {
                        TextureViewDimension::D2
                    } else {
                        TextureViewDimension::D2Array
                    }
                }
                wgt::TextureDimension::D3 => TextureViewDimension::D3,
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
                    TextureViewDimension::D1
                    | TextureViewDimension::D2
                    | TextureViewDimension::D3 => 1,
                    TextureViewDimension::Cube => 6,
                    TextureViewDimension::D2Array | TextureViewDimension::CubeArray => texture
                        .desc
                        .array_layer_count()
                        .saturating_sub(desc.range.base_array_layer),
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
        if texture.desc.sample_count > 1 && resolved_dimension != TextureViewDimension::D2 {
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
        let render_extent = 'error: {
            if !texture
                .desc
                .usage
                .contains(wgt::TextureUsages::RENDER_ATTACHMENT)
            {
                break 'error Err(TextureViewNotRenderableReason::Usage(texture.desc.usage));
            }

            if !(resolved_dimension == TextureViewDimension::D2
                || (self.features.contains(wgt::Features::MULTIVIEW)
                    && resolved_dimension == TextureViewDimension::D2Array))
            {
                break 'error Err(TextureViewNotRenderableReason::Dimension(
                    resolved_dimension,
                ));
            }

            if resolved_mip_level_count != 1 {
                break 'error Err(TextureViewNotRenderableReason::MipLevelCount(
                    resolved_mip_level_count,
                ));
            }

            if resolved_array_layer_count != 1
                && !(self.features.contains(wgt::Features::MULTIVIEW))
            {
                break 'error Err(TextureViewNotRenderableReason::ArrayLayerCount(
                    resolved_array_layer_count,
                ));
            }

            if aspects != hal::FormatAspects::from(texture.desc.format) {
                break 'error Err(TextureViewNotRenderableReason::Aspects(aspects));
            }

            Ok(texture
                .desc
                .compute_render_extent(desc.range.base_mip_level))
        };

        // filter the usages based on the other criteria
        let usage = {
            let mask_copy = !(hal::TextureUses::COPY_SRC | hal::TextureUses::COPY_DST);
            let mask_dimension = match resolved_dimension {
                TextureViewDimension::Cube | TextureViewDimension::CubeArray => {
                    hal::TextureUses::RESOURCE
                }
                TextureViewDimension::D3 => {
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
            texture.as_info().id(),
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
                .as_ref()
                .unwrap()
                .create_texture_view(texture_raw, &hal_desc)
                .map_err(|_| resource::CreateTextureViewError::OutOfMemory)?
        };

        let selector = TextureSelector {
            mips: desc.range.base_mip_level..mip_level_end,
            layers: desc.range.base_array_layer..array_layer_end,
        };

        Ok(TextureView {
            raw: Snatchable::new(raw),
            parent: texture.clone(),
            device: self.clone(),
            desc: resource::HalTextureViewDescriptor {
                texture_format: texture.desc.format,
                format: resolved_format,
                dimension: resolved_dimension,
                range: resolved_range,
            },
            format_features: texture.format_features,
            render_extent,
            samples: texture.desc.sample_count,
            selector,
            info: ResourceInfo::new(
                desc.label.borrow_or_default(),
                Some(self.tracker_indices.texture_views.clone()),
            ),
        })
    }

    pub(crate) fn create_sampler(
        self: &Arc<Self>,
        desc: &resource::SamplerDescriptor,
    ) -> Result<Sampler<A>, resource::CreateSamplerError> {
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
                .as_ref()
                .unwrap()
                .create_sampler(&hal_desc)
                .map_err(DeviceError::from)?
        };
        Ok(Sampler {
            raw: Some(raw),
            device: self.clone(),
            info: ResourceInfo::new(
                desc.label.borrow_or_default(),
                Some(self.tracker_indices.samplers.clone()),
            ),
            comparison: desc.compare.is_some(),
            filtering: desc.min_filter == wgt::FilterMode::Linear
                || desc.mag_filter == wgt::FilterMode::Linear,
        })
    }

    pub(crate) fn create_shader_module<'a>(
        self: &Arc<Self>,
        desc: &pipeline::ShaderModuleDescriptor<'a>,
        source: pipeline::ShaderModuleSource<'a>,
    ) -> Result<pipeline::ShaderModule<A>, pipeline::CreateShaderModuleError> {
        let (module, source) = match source {
            #[cfg(feature = "wgsl")]
            pipeline::ShaderModuleSource::Wgsl(code) => {
                profiling::scope!("naga::front::wgsl::parse_str");
                let module = naga::front::wgsl::parse_str(&code).map_err(|inner| {
                    pipeline::CreateShaderModuleError::Parsing(naga::error::ShaderError {
                        source: code.to_string(),
                        label: desc.label.as_ref().map(|l| l.to_string()),
                        inner: Box::new(inner),
                    })
                })?;
                (Cow::Owned(module), code.into_owned())
            }
            #[cfg(feature = "spirv")]
            pipeline::ShaderModuleSource::SpirV(spv, options) => {
                let parser = naga::front::spv::Frontend::new(spv.iter().cloned(), &options);
                profiling::scope!("naga::front::spv::Frontend");
                let module = parser.parse().map_err(|inner| {
                    pipeline::CreateShaderModuleError::ParsingSpirV(naga::error::ShaderError {
                        source: String::new(),
                        label: desc.label.as_ref().map(|l| l.to_string()),
                        inner: Box::new(inner),
                    })
                })?;
                (Cow::Owned(module), String::new())
            }
            #[cfg(feature = "glsl")]
            pipeline::ShaderModuleSource::Glsl(code, options) => {
                let mut parser = naga::front::glsl::Frontend::default();
                profiling::scope!("naga::front::glsl::Frontend.parse");
                let module = parser.parse(&options, &code).map_err(|inner| {
                    pipeline::CreateShaderModuleError::ParsingGlsl(naga::error::ShaderError {
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

        profiling::scope!("naga::validate");
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

        let info = create_validator(
            self.features,
            self.downlevel.flags,
            naga::valid::ValidationFlags::all(),
        )
        .validate(&module)
        .map_err(|inner| {
            pipeline::CreateShaderModuleError::Validation(naga::error::ShaderError {
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
        let raw = match unsafe {
            self.raw
                .as_ref()
                .unwrap()
                .create_shader_module(&hal_desc, hal_shader)
        } {
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
            raw: Some(raw),
            device: self.clone(),
            interface: Some(interface),
            info: ResourceInfo::new(desc.label.borrow_or_default(), None),
            label: desc.label.borrow_or_default().to_string(),
        })
    }

    #[allow(unused_unsafe)]
    pub(crate) unsafe fn create_shader_module_spirv<'a>(
        self: &Arc<Self>,
        desc: &pipeline::ShaderModuleDescriptor<'a>,
        source: &'a [u32],
    ) -> Result<pipeline::ShaderModule<A>, pipeline::CreateShaderModuleError> {
        self.require_features(wgt::Features::SPIRV_SHADER_PASSTHROUGH)?;
        let hal_desc = hal::ShaderModuleDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            runtime_checks: desc.shader_bound_checks.runtime_checks(),
        };
        let hal_shader = hal::ShaderInput::SpirV(source);
        let raw = match unsafe {
            self.raw
                .as_ref()
                .unwrap()
                .create_shader_module(&hal_desc, hal_shader)
        } {
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
            raw: Some(raw),
            device: self.clone(),
            interface: None,
            info: ResourceInfo::new(desc.label.borrow_or_default(), None),
            label: desc.label.borrow_or_default().to_string(),
        })
    }

    /// Generate information about late-validated buffer bindings for pipelines.
    //TODO: should this be combined with `get_introspection_bind_group_layouts` in some way?
    pub(crate) fn make_late_sized_buffer_groups(
        shader_binding_sizes: &FastHashMap<naga::ResourceBinding, wgt::BufferSize>,
        layout: &binding_model::PipelineLayout<A>,
    ) -> ArrayVec<pipeline::LateSizedBufferGroup, { hal::MAX_BIND_GROUPS }> {
        // Given the shader-required binding sizes and the pipeline layout,
        // return the filtered list of them in the layout order,
        // removing those with given `min_binding_size`.
        layout
            .bind_group_layouts
            .iter()
            .enumerate()
            .map(|(group_index, bgl)| pipeline::LateSizedBufferGroup {
                shader_sizes: bgl
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

    pub(crate) fn create_bind_group_layout(
        self: &Arc<Self>,
        label: &crate::Label,
        entry_map: bgl::EntryMap,
        origin: bgl::Origin,
    ) -> Result<BindGroupLayout<A>, binding_model::CreateBindGroupLayoutError> {
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
                        error:
                            BindGroupLayoutEntryError::SampleTypeFloatFilterableBindingMultisampled,
                    });
                }
                Bt::Texture {
                    multisampled,
                    view_dimension,
                    ..
                } => {
                    if multisampled && view_dimension != TextureViewDimension::D2 {
                        return Err(binding_model::CreateBindGroupLayoutError::Entry {
                            binding: entry.binding,
                            error: BindGroupLayoutEntryError::Non2DMultisampled(view_dimension),
                        });
                    }

                    (
                        Some(wgt::Features::TEXTURE_BINDING_ARRAY),
                        WritableStorage::No,
                    )
                }
                Bt::StorageTexture {
                    access,
                    view_dimension,
                    format: _,
                } => {
                    match view_dimension {
                        TextureViewDimension::Cube | TextureViewDimension::CubeArray => {
                            return Err(binding_model::CreateBindGroupLayoutError::Entry {
                                binding: entry.binding,
                                error: BindGroupLayoutEntryError::StorageTextureCube,
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
                                error: BindGroupLayoutEntryError::StorageTextureReadWrite,
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
                Bt::AccelerationStructure => todo!(),
            };

            // Validate the count parameter
            if entry.count.is_some() {
                required_features |= array_feature
                    .ok_or(BindGroupLayoutEntryError::ArrayUnsupported)
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
                .map_err(BindGroupLayoutEntryError::MissingFeatures)
                .map_err(|error| binding_model::CreateBindGroupLayoutError::Entry {
                    binding: entry.binding,
                    error,
                })?;
            self.require_downlevel_flags(required_downlevel_flags)
                .map_err(BindGroupLayoutEntryError::MissingDownlevelFlags)
                .map_err(|error| binding_model::CreateBindGroupLayoutError::Entry {
                    binding: entry.binding,
                    error,
                })?;
        }

        let bgl_flags = conv::bind_group_layout_flags(self.features);

        let hal_bindings = entry_map.values().copied().collect::<Vec<_>>();
        let label = label.to_hal(self.instance_flags);
        let hal_desc = hal::BindGroupLayoutDescriptor {
            label,
            flags: bgl_flags,
            entries: &hal_bindings,
        };
        let raw = unsafe {
            self.raw
                .as_ref()
                .unwrap()
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

        Ok(BindGroupLayout {
            raw: Some(raw),
            device: self.clone(),
            entries: entry_map,
            origin,
            binding_count_validator: count_validator,
            info: ResourceInfo::new(
                label.unwrap_or("<BindGroupLayout>"),
                Some(self.tracker_indices.bind_group_layouts.clone()),
            ),
            label: label.unwrap_or_default().to_string(),
        })
    }

    pub(crate) fn create_buffer_binding<'a>(
        bb: &binding_model::BufferBinding,
        binding: u32,
        decl: &wgt::BindGroupLayoutEntry,
        used_buffer_ranges: &mut Vec<BufferInitTrackerAction<A>>,
        dynamic_binding_info: &mut Vec<binding_model::BindGroupDynamicBindingData>,
        late_buffer_binding_sizes: &mut FastHashMap<u32, wgt::BufferSize>,
        used: &mut BindGroupStates<A>,
        storage: &'a Storage<Buffer<A>>,
        limits: &wgt::Limits,
        device_id: id::Id<id::markers::Device>,
        snatch_guard: &'a SnatchGuard<'a>,
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

        if buffer.device.as_info().id() != device_id {
            return Err(DeviceError::WrongDevice.into());
        }

        check_buffer_usage(bb.buffer_id, buffer.usage, pub_usage)?;
        let raw_buffer = buffer
            .raw
            .get(snatch_guard)
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
            None => {
                if buffer.size < bb.offset {
                    return Err(Error::BindingRangeTooLarge {
                        buffer: bb.buffer_id,
                        range: bb.offset..bb.offset,
                        size: buffer.size,
                    });
                }
                (buffer.size - bb.offset, buffer.size)
            }
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
        used_buffer_ranges.extend(buffer.initialization_status.read().create_action(
            buffer,
            bb.offset..bb.offset + bind_size,
            MemoryInitKind::NeedsInitializedMemory,
        ));

        Ok(hal::BufferBinding {
            buffer: raw_buffer,
            offset: bb.offset,
            size: bb.size,
        })
    }

    fn create_sampler_binding<'a>(
        used: &BindGroupStates<A>,
        storage: &'a Storage<Sampler<A>>,
        id: id::Id<id::markers::Sampler>,
        device_id: id::Id<id::markers::Device>,
    ) -> Result<&'a Sampler<A>, binding_model::CreateBindGroupError> {
        use crate::binding_model::CreateBindGroupError as Error;

        let sampler = used
            .samplers
            .add_single(storage, id)
            .ok_or(Error::InvalidSampler(id))?;

        if sampler.device.as_info().id() != device_id {
            return Err(DeviceError::WrongDevice.into());
        }

        Ok(sampler)
    }

    pub(crate) fn create_texture_binding<'a>(
        self: &Arc<Self>,
        binding: u32,
        decl: &wgt::BindGroupLayoutEntry,
        storage: &'a Storage<TextureView<A>>,
        id: id::Id<id::markers::TextureView>,
        used: &mut BindGroupStates<A>,
        used_texture_ranges: &mut Vec<TextureInitTrackerAction<A>>,
        snatch_guard: &'a SnatchGuard<'a>,
    ) -> Result<hal::TextureBinding<'a, A>, binding_model::CreateBindGroupError> {
        use crate::binding_model::CreateBindGroupError as Error;

        let view = used
            .views
            .add_single(storage, id)
            .ok_or(Error::InvalidTextureView(id))?;

        if view.device.as_info().id() != self.as_info().id() {
            return Err(DeviceError::WrongDevice.into());
        }

        let (pub_usage, internal_use) = self.texture_use_parameters(
            binding,
            decl,
            view,
            "SampledTexture, ReadonlyStorageTexture or WriteonlyStorageTexture",
        )?;
        let texture = &view.parent;
        let texture_id = texture.as_info().id();
        // Careful here: the texture may no longer have its own ref count,
        // if it was deleted by the user.
        let texture = used
            .textures
            .add_single(texture, Some(view.selector.clone()), internal_use)
            .ok_or(binding_model::CreateBindGroupError::InvalidTexture(
                texture_id,
            ))?;

        if texture.device.as_info().id() != view.device.as_info().id() {
            return Err(DeviceError::WrongDevice.into());
        }

        check_texture_usage(texture.desc.usage, pub_usage)?;

        used_texture_ranges.push(TextureInitTrackerAction {
            texture: texture.clone(),
            range: TextureInitRange {
                mip_range: view.desc.range.mip_range(texture.desc.mip_level_count),
                layer_range: view
                    .desc
                    .range
                    .layer_range(texture.desc.array_layer_count()),
            },
            kind: MemoryInitKind::NeedsInitializedMemory,
        });

        Ok(hal::TextureBinding {
            view: view
                .raw(snatch_guard)
                .ok_or(Error::InvalidTextureView(id))?,
            usage: internal_use,
        })
    }

    // This function expects the provided bind group layout to be resolved
    // (not passing a duplicate) beforehand.
    pub(crate) fn create_bind_group(
        self: &Arc<Self>,
        layout: &Arc<BindGroupLayout<A>>,
        desc: &binding_model::BindGroupDescriptor,
        hub: &Hub<A>,
    ) -> Result<BindGroup<A>, binding_model::CreateBindGroupError> {
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

        // TODO: arrayvec/smallvec, or re-use allocations
        // Record binding info for dynamic offset validation
        let mut dynamic_binding_info = Vec::new();
        // Map of binding -> shader reflected size
        //Note: we can't collect into a vector right away because
        // it needs to be in BGL iteration order, not BG entry order.
        let mut late_buffer_binding_sizes = FastHashMap::default();
        // fill out the descriptors
        let mut used = BindGroupStates::new();

        let buffer_guard = hub.buffers.read();
        let texture_view_guard = hub.texture_views.read();
        let sampler_guard = hub.samplers.read();

        let mut used_buffer_ranges = Vec::new();
        let mut used_texture_ranges = Vec::new();
        let mut hal_entries = Vec::with_capacity(desc.entries.len());
        let mut hal_buffers = Vec::new();
        let mut hal_samplers = Vec::new();
        let mut hal_textures = Vec::new();
        let snatch_guard = self.snatchable_lock.read();
        for entry in desc.entries.iter() {
            let binding = entry.binding;
            // Find the corresponding declaration in the layout
            let decl = layout
                .entries
                .get(binding)
                .ok_or(Error::MissingBindingDeclaration(binding))?;
            let (res_index, count) = match entry.resource {
                Br::Buffer(ref bb) => {
                    let bb = Self::create_buffer_binding(
                        bb,
                        binding,
                        decl,
                        &mut used_buffer_ranges,
                        &mut dynamic_binding_info,
                        &mut late_buffer_binding_sizes,
                        &mut used,
                        &*buffer_guard,
                        &self.limits,
                        self.as_info().id(),
                        &snatch_guard,
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
                            bb,
                            binding,
                            decl,
                            &mut used_buffer_ranges,
                            &mut dynamic_binding_info,
                            &mut late_buffer_binding_sizes,
                            &mut used,
                            &*buffer_guard,
                            &self.limits,
                            self.as_info().id(),
                            &snatch_guard,
                        )?;
                        hal_buffers.push(bb);
                    }
                    (res_index, num_bindings)
                }
                Br::Sampler(id) => match decl.ty {
                    wgt::BindingType::Sampler(ty) => {
                        let sampler = Self::create_sampler_binding(
                            &used,
                            &sampler_guard,
                            id,
                            self.as_info().id(),
                        )?;

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
                        hal_samplers.push(sampler.raw());
                        (res_index, 1)
                    }
                    _ => {
                        return Err(Error::WrongBindingType {
                            binding,
                            actual: decl.ty,
                            expected: "Sampler",
                        })
                    }
                },
                Br::SamplerArray(ref bindings_array) => {
                    let num_bindings = bindings_array.len();
                    Self::check_array_binding(self.features, decl.count, num_bindings)?;

                    let res_index = hal_samplers.len();
                    for &id in bindings_array.iter() {
                        let sampler = Self::create_sampler_binding(
                            &used,
                            &sampler_guard,
                            id,
                            self.as_info().id(),
                        )?;

                        hal_samplers.push(sampler.raw());
                    }

                    (res_index, num_bindings)
                }
                Br::TextureView(id) => {
                    let tb = self.create_texture_binding(
                        binding,
                        decl,
                        &texture_view_guard,
                        id,
                        &mut used,
                        &mut used_texture_ranges,
                        &snatch_guard,
                    )?;
                    let res_index = hal_textures.len();
                    hal_textures.push(tb);
                    (res_index, 1)
                }
                Br::TextureViewArray(ref bindings_array) => {
                    let num_bindings = bindings_array.len();
                    Self::check_array_binding(self.features, decl.count, num_bindings)?;

                    let res_index = hal_textures.len();
                    for &id in bindings_array.iter() {
                        let tb = self.create_texture_binding(
                            binding,
                            decl,
                            &texture_view_guard,
                            id,
                            &mut used,
                            &mut used_texture_ranges,
                            &snatch_guard,
                        )?;

                        hal_textures.push(tb);
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
        let hal_desc = hal::BindGroupDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            layout: layout.raw(),
            entries: &hal_entries,
            buffers: &hal_buffers,
            samplers: &hal_samplers,
            textures: &hal_textures,
            acceleration_structures: &[],
        };
        let raw = unsafe {
            self.raw
                .as_ref()
                .unwrap()
                .create_bind_group(&hal_desc)
                .map_err(DeviceError::from)?
        };

        Ok(BindGroup {
            raw: Snatchable::new(raw),
            device: self.clone(),
            layout: layout.clone(),
            info: ResourceInfo::new(
                desc.label.borrow_or_default(),
                Some(self.tracker_indices.bind_groups.clone()),
            ),
            used,
            used_buffer_ranges,
            used_texture_ranges,
            dynamic_binding_info,
            // collect in the order of BGL iteration
            late_buffer_binding_sizes: layout
                .entries
                .indices()
                .flat_map(|binding| late_buffer_binding_sizes.get(&binding).cloned())
                .collect(),
        })
    }

    pub(crate) fn check_array_binding(
        features: wgt::Features,
        count: Option<NonZeroU32>,
        num_bindings: usize,
    ) -> Result<(), binding_model::CreateBindGroupError> {
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

    pub(crate) fn texture_use_parameters(
        self: &Arc<Self>,
        binding: u32,
        decl: &wgt::BindGroupLayoutEntry,
        view: &TextureView<A>,
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
                    .sample_type(Some(view.desc.range.aspect), Some(self.features))
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

    pub(crate) fn create_pipeline_layout(
        self: &Arc<Self>,
        desc: &binding_model::PipelineLayoutDescriptor,
        bgl_registry: &Registry<BindGroupLayout<A>>,
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

        // Collect references to the BGLs
        let mut bind_group_layouts = ArrayVec::new();
        for &id in desc.bind_group_layouts.iter() {
            let Ok(bgl) = bgl_registry.get(id) else {
                return Err(Error::InvalidBindGroupLayout(id));
            };

            bind_group_layouts.push(bgl);
        }

        // Validate total resource counts and check for a matching device
        for bgl in &bind_group_layouts {
            if bgl.device.as_info().id() != self.as_info().id() {
                return Err(DeviceError::WrongDevice.into());
            }

            count_validator.merge(&bgl.binding_count_validator);
        }

        count_validator
            .validate(&self.limits)
            .map_err(Error::TooManyBindings)?;

        let raw_bind_group_layouts = bind_group_layouts
            .iter()
            .map(|bgl| bgl.raw())
            .collect::<ArrayVec<_, { hal::MAX_BIND_GROUPS }>>();

        let hal_desc = hal::PipelineLayoutDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            flags: hal::PipelineLayoutFlags::FIRST_VERTEX_INSTANCE,
            bind_group_layouts: &raw_bind_group_layouts,
            push_constant_ranges: desc.push_constant_ranges.as_ref(),
        };

        let raw = unsafe {
            self.raw
                .as_ref()
                .unwrap()
                .create_pipeline_layout(&hal_desc)
                .map_err(DeviceError::from)?
        };

        drop(raw_bind_group_layouts);

        Ok(binding_model::PipelineLayout {
            raw: Some(raw),
            device: self.clone(),
            info: ResourceInfo::new(
                desc.label.borrow_or_default(),
                Some(self.tracker_indices.pipeline_layouts.clone()),
            ),
            bind_group_layouts,
            push_constant_ranges: desc.push_constant_ranges.iter().cloned().collect(),
        })
    }

    //TODO: refactor this. It's the only method of `Device` that registers new objects
    // (the pipeline layout).
    pub(crate) fn derive_pipeline_layout(
        self: &Arc<Self>,
        implicit_context: Option<ImplicitPipelineContext>,
        mut derived_group_layouts: ArrayVec<bgl::EntryMap, { hal::MAX_BIND_GROUPS }>,
        bgl_registry: &Registry<BindGroupLayout<A>>,
        pipeline_layout_registry: &Registry<binding_model::PipelineLayout<A>>,
    ) -> Result<Arc<binding_model::PipelineLayout<A>>, pipeline::ImplicitLayoutError> {
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
            let bgl = self.create_bind_group_layout(&None, map, bgl::Origin::Derived)?;
            bgl_registry.force_replace(*bgl_id, bgl);
        }

        let layout_desc = binding_model::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: Cow::Borrowed(&ids.group_ids[..group_count]),
            push_constant_ranges: Cow::Borrowed(&[]), //TODO?
        };
        let layout = self.create_pipeline_layout(&layout_desc, bgl_registry)?;
        pipeline_layout_registry.force_replace(ids.root_id, layout);
        Ok(pipeline_layout_registry.get(ids.root_id).unwrap())
    }

    pub(crate) fn create_compute_pipeline(
        self: &Arc<Self>,
        desc: &pipeline::ComputePipelineDescriptor,
        implicit_context: Option<ImplicitPipelineContext>,
        hub: &Hub<A>,
    ) -> Result<pipeline::ComputePipeline<A>, pipeline::CreateComputePipelineError> {
        // This has to be done first, or otherwise the IDs may be pointing to entries
        // that are not even in the storage.
        if let Some(ref ids) = implicit_context {
            let mut pipeline_layout_guard = hub.pipeline_layouts.write();
            pipeline_layout_guard.insert_error(ids.root_id, IMPLICIT_BIND_GROUP_LAYOUT_ERROR_LABEL);
            let mut bgl_guard = hub.bind_group_layouts.write();
            for &bgl_id in ids.group_ids.iter() {
                bgl_guard.insert_error(bgl_id, IMPLICIT_BIND_GROUP_LAYOUT_ERROR_LABEL);
            }
        }

        self.require_downlevel_flags(wgt::DownlevelFlags::COMPUTE_SHADERS)?;

        let shader_module = hub
            .shader_modules
            .get(desc.stage.module)
            .map_err(|_| validation::StageError::InvalidModule)?;

        if shader_module.device.as_info().id() != self.as_info().id() {
            return Err(DeviceError::WrongDevice.into());
        }

        // Get the pipeline layout from the desc if it is provided.
        let pipeline_layout = match desc.layout {
            Some(pipeline_layout_id) => {
                let pipeline_layout = hub
                    .pipeline_layouts
                    .get(pipeline_layout_id)
                    .map_err(|_| pipeline::CreateComputePipelineError::InvalidLayout)?;

                if pipeline_layout.device.as_info().id() != self.as_info().id() {
                    return Err(DeviceError::WrongDevice.into());
                }

                Some(pipeline_layout)
            }
            None => None,
        };

        let mut binding_layout_source = match pipeline_layout {
            Some(ref pipeline_layout) => {
                validation::BindingLayoutSource::Provided(pipeline_layout.get_binding_maps())
            }
            None => validation::BindingLayoutSource::new_derived(&self.limits),
        };
        let mut shader_binding_sizes = FastHashMap::default();
        let io = validation::StageIo::default();

        let final_entry_point_name;

        {
            let stage = wgt::ShaderStages::COMPUTE;

            final_entry_point_name = shader_module.finalize_entry_point_name(
                stage,
                desc.stage.entry_point.as_ref().map(|ep| ep.as_ref()),
            )?;

            if let Some(ref interface) = shader_module.interface {
                let _ = interface.check_stage(
                    &mut binding_layout_source,
                    &mut shader_binding_sizes,
                    &final_entry_point_name,
                    stage,
                    io,
                    None,
                )?;
            }
        }

        let pipeline_layout = match binding_layout_source {
            validation::BindingLayoutSource::Provided(_) => {
                drop(binding_layout_source);
                pipeline_layout.unwrap()
            }
            validation::BindingLayoutSource::Derived(entries) => self.derive_pipeline_layout(
                implicit_context,
                entries,
                &hub.bind_group_layouts,
                &hub.pipeline_layouts,
            )?,
        };

        let late_sized_buffer_groups =
            Device::make_late_sized_buffer_groups(&shader_binding_sizes, &pipeline_layout);

        let cache = 'cache: {
            let Some(cache) = desc.cache else {
                break 'cache None;
            };
            let Ok(cache) = hub.pipeline_caches.get(cache) else {
                break 'cache None;
            };

            if cache.device.as_info().id() != self.as_info().id() {
                return Err(DeviceError::WrongDevice.into());
            }
            Some(cache)
        };

        let pipeline_desc = hal::ComputePipelineDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            layout: pipeline_layout.raw(),
            stage: hal::ProgrammableStage {
                module: shader_module.raw(),
                entry_point: final_entry_point_name.as_ref(),
                constants: desc.stage.constants.as_ref(),
                zero_initialize_workgroup_memory: desc.stage.zero_initialize_workgroup_memory,
                vertex_pulling_transform: false,
            },
            cache: cache.as_ref().and_then(|it| it.raw.as_ref()),
        };

        let raw = unsafe {
            self.raw
                .as_ref()
                .unwrap()
                .create_compute_pipeline(&pipeline_desc)
        }
        .map_err(|err| match err {
            hal::PipelineError::Device(error) => {
                pipeline::CreateComputePipelineError::Device(error.into())
            }
            hal::PipelineError::Linkage(_stages, msg) => {
                pipeline::CreateComputePipelineError::Internal(msg)
            }
            hal::PipelineError::EntryPoint(_stage) => {
                pipeline::CreateComputePipelineError::Internal(ENTRYPOINT_FAILURE_ERROR.to_string())
            }
        })?;

        let pipeline = pipeline::ComputePipeline {
            raw: Some(raw),
            layout: pipeline_layout,
            device: self.clone(),
            _shader_module: shader_module,
            late_sized_buffer_groups,
            info: ResourceInfo::new(
                desc.label.borrow_or_default(),
                Some(self.tracker_indices.compute_pipelines.clone()),
            ),
        };
        Ok(pipeline)
    }

    pub(crate) fn create_render_pipeline(
        self: &Arc<Self>,
        adapter: &Adapter<A>,
        desc: &pipeline::RenderPipelineDescriptor,
        implicit_context: Option<ImplicitPipelineContext>,
        hub: &Hub<A>,
    ) -> Result<pipeline::RenderPipeline<A>, pipeline::CreateRenderPipelineError> {
        use wgt::TextureFormatFeatureFlags as Tfff;

        // This has to be done first, or otherwise the IDs may be pointing to entries
        // that are not even in the storage.
        if let Some(ref ids) = implicit_context {
            //TODO: only lock mutable if the layout is derived
            let mut pipeline_layout_guard = hub.pipeline_layouts.write();
            let mut bgl_guard = hub.bind_group_layouts.write();
            pipeline_layout_guard.insert_error(ids.root_id, IMPLICIT_BIND_GROUP_LAYOUT_ERROR_LABEL);
            for &bgl_id in ids.group_ids.iter() {
                bgl_guard.insert_error(bgl_id, IMPLICIT_BIND_GROUP_LAYOUT_ERROR_LABEL);
            }
        }

        let mut shader_binding_sizes = FastHashMap::default();

        let num_attachments = desc.fragment.as_ref().map(|f| f.targets.len()).unwrap_or(0);
        let max_attachments = self.limits.max_color_attachments as usize;
        if num_attachments > max_attachments {
            return Err(pipeline::CreateRenderPipelineError::ColorAttachment(
                command::ColorAttachmentError::TooMany {
                    given: num_attachments,
                    limit: max_attachments,
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
            log::debug!("Color targets: {:?}", color_targets);
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
            let mut last_stride = 0;
            for attribute in vb_state.attributes.iter() {
                last_stride = last_stride.max(attribute.offset + attribute.format.size());
            }
            vertex_steps.push(pipeline::VertexStep {
                stride: vb_state.array_stride,
                last_stride,
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

        let mut target_specified = false;

        for (i, cs) in color_targets.iter().enumerate() {
            if let Some(cs) = cs.as_ref() {
                target_specified = true;
                let error = 'error: {
                    if cs.write_mask.contains_invalid_bits() {
                        break 'error Some(pipeline::ColorStateError::InvalidWriteMask(
                            cs.write_mask,
                        ));
                    }

                    let format_features = self.describe_format_features(adapter, cs.format)?;
                    if !format_features
                        .allowed_usages
                        .contains(wgt::TextureUsages::RENDER_ATTACHMENT)
                    {
                        break 'error Some(pipeline::ColorStateError::FormatNotRenderable(
                            cs.format,
                        ));
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
                        break 'error Some(pipeline::ColorStateError::FormatNotBlendable(
                            cs.format,
                        ));
                    }
                    if !hal::FormatAspects::from(cs.format).contains(hal::FormatAspects::COLOR) {
                        break 'error Some(pipeline::ColorStateError::FormatNotColor(cs.format));
                    }

                    if desc.multisample.count > 1
                        && !format_features
                            .flags
                            .sample_count_supported(desc.multisample.count)
                    {
                        break 'error Some(pipeline::ColorStateError::InvalidSampleCount(
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
                                    return Err(pipeline::CreateRenderPipelineError
                            ::BlendFactorOnUnsupportedTarget { factor, target: i as u32 });
                                }
                            }
                        }
                    }

                    break 'error None;
                };
                if let Some(e) = error {
                    return Err(pipeline::CreateRenderPipelineError::ColorState(i as u8, e));
                }
            }
        }

        let limit = self.limits.max_color_attachment_bytes_per_sample;
        let formats = color_targets
            .iter()
            .map(|cs| cs.as_ref().map(|cs| cs.format));
        if let Err(total) = validate_color_attachment_bytes_per_sample(formats, limit) {
            return Err(pipeline::CreateRenderPipelineError::ColorAttachment(
                command::ColorAttachmentError::TooManyBytesPerSample { total, limit },
            ));
        }

        if let Some(ds) = depth_stencil_state {
            target_specified = true;
            let error = 'error: {
                let format_features = self.describe_format_features(adapter, ds.format)?;
                if !format_features
                    .allowed_usages
                    .contains(wgt::TextureUsages::RENDER_ATTACHMENT)
                {
                    break 'error Some(pipeline::DepthStencilStateError::FormatNotRenderable(
                        ds.format,
                    ));
                }

                let aspect = hal::FormatAspects::from(ds.format);
                if ds.is_depth_enabled() && !aspect.contains(hal::FormatAspects::DEPTH) {
                    break 'error Some(pipeline::DepthStencilStateError::FormatNotDepth(ds.format));
                }
                if ds.stencil.is_enabled() && !aspect.contains(hal::FormatAspects::STENCIL) {
                    break 'error Some(pipeline::DepthStencilStateError::FormatNotStencil(
                        ds.format,
                    ));
                }
                if desc.multisample.count > 1
                    && !format_features
                        .flags
                        .sample_count_supported(desc.multisample.count)
                {
                    break 'error Some(pipeline::DepthStencilStateError::InvalidSampleCount(
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

                break 'error None;
            };
            if let Some(e) = error {
                return Err(pipeline::CreateRenderPipelineError::DepthStencilState(e));
            }

            if ds.bias.clamp != 0.0 {
                self.require_downlevel_flags(wgt::DownlevelFlags::DEPTH_BIAS_CLAMP)?;
            }
        }

        if !target_specified {
            return Err(pipeline::CreateRenderPipelineError::NoTargetSpecified);
        }

        // Get the pipeline layout from the desc if it is provided.
        let pipeline_layout = match desc.layout {
            Some(pipeline_layout_id) => {
                let pipeline_layout = hub
                    .pipeline_layouts
                    .get(pipeline_layout_id)
                    .map_err(|_| pipeline::CreateRenderPipelineError::InvalidLayout)?;

                if pipeline_layout.device.as_info().id() != self.as_info().id() {
                    return Err(DeviceError::WrongDevice.into());
                }

                Some(pipeline_layout)
            }
            None => None,
        };

        let mut binding_layout_source = match pipeline_layout {
            Some(ref pipeline_layout) => {
                validation::BindingLayoutSource::Provided(pipeline_layout.get_binding_maps())
            }
            None => validation::BindingLayoutSource::new_derived(&self.limits),
        };

        let samples = {
            let sc = desc.multisample.count;
            if sc == 0 || sc > 32 || !conv::is_power_of_two_u32(sc) {
                return Err(pipeline::CreateRenderPipelineError::InvalidSampleCount(sc));
            }
            sc
        };

        let vertex_shader_module;
        let vertex_entry_point_name;

        let vertex_stage = {
            let stage_desc = &desc.vertex.stage;
            let stage = wgt::ShaderStages::VERTEX;

            vertex_shader_module = hub.shader_modules.get(stage_desc.module).map_err(|_| {
                pipeline::CreateRenderPipelineError::Stage {
                    stage,
                    error: validation::StageError::InvalidModule,
                }
            })?;
            if vertex_shader_module.device.as_info().id() != self.as_info().id() {
                return Err(DeviceError::WrongDevice.into());
            }

            let stage_err = |error| pipeline::CreateRenderPipelineError::Stage { stage, error };

            vertex_entry_point_name = vertex_shader_module
                .finalize_entry_point_name(
                    stage,
                    stage_desc.entry_point.as_ref().map(|ep| ep.as_ref()),
                )
                .map_err(stage_err)?;

            if let Some(ref interface) = vertex_shader_module.interface {
                io = interface
                    .check_stage(
                        &mut binding_layout_source,
                        &mut shader_binding_sizes,
                        &vertex_entry_point_name,
                        stage,
                        io,
                        desc.depth_stencil.as_ref().map(|d| d.depth_compare),
                    )
                    .map_err(stage_err)?;
                validated_stages |= stage;
            }

            hal::ProgrammableStage {
                module: vertex_shader_module.raw(),
                entry_point: &vertex_entry_point_name,
                constants: stage_desc.constants.as_ref(),
                zero_initialize_workgroup_memory: stage_desc.zero_initialize_workgroup_memory,
                vertex_pulling_transform: stage_desc.vertex_pulling_transform,
            }
        };

        let mut fragment_shader_module = None;
        let fragment_entry_point_name;
        let fragment_stage = match desc.fragment {
            Some(ref fragment_state) => {
                let stage = wgt::ShaderStages::FRAGMENT;

                let shader_module = fragment_shader_module.insert(
                    hub.shader_modules
                        .get(fragment_state.stage.module)
                        .map_err(|_| pipeline::CreateRenderPipelineError::Stage {
                            stage,
                            error: validation::StageError::InvalidModule,
                        })?,
                );

                let stage_err = |error| pipeline::CreateRenderPipelineError::Stage { stage, error };

                fragment_entry_point_name = shader_module
                    .finalize_entry_point_name(
                        stage,
                        fragment_state
                            .stage
                            .entry_point
                            .as_ref()
                            .map(|ep| ep.as_ref()),
                    )
                    .map_err(stage_err)?;

                if validated_stages == wgt::ShaderStages::VERTEX {
                    if let Some(ref interface) = shader_module.interface {
                        io = interface
                            .check_stage(
                                &mut binding_layout_source,
                                &mut shader_binding_sizes,
                                &fragment_entry_point_name,
                                stage,
                                io,
                                desc.depth_stencil.as_ref().map(|d| d.depth_compare),
                            )
                            .map_err(stage_err)?;
                        validated_stages |= stage;
                    }
                }

                if let Some(ref interface) = shader_module.interface {
                    shader_expects_dual_source_blending = interface
                        .fragment_uses_dual_source_blending(&fragment_entry_point_name)
                        .map_err(|error| pipeline::CreateRenderPipelineError::Stage {
                            stage,
                            error,
                        })?;
                }

                Some(hal::ProgrammableStage {
                    module: shader_module.raw(),
                    entry_point: &fragment_entry_point_name,
                    constants: fragment_state.stage.constants.as_ref(),
                    zero_initialize_workgroup_memory: fragment_state
                        .stage
                        .zero_initialize_workgroup_memory,
                    vertex_pulling_transform: false,
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
                    Some(Some(state)) => {
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
                        log::warn!(
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

        let pipeline_layout = match binding_layout_source {
            validation::BindingLayoutSource::Provided(_) => {
                drop(binding_layout_source);
                pipeline_layout.unwrap()
            }
            validation::BindingLayoutSource::Derived(entries) => self.derive_pipeline_layout(
                implicit_context,
                entries,
                &hub.bind_group_layouts,
                &hub.pipeline_layouts,
            )?,
        };

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
            Device::make_late_sized_buffer_groups(&shader_binding_sizes, &pipeline_layout);

        let pipeline_cache = 'cache: {
            let Some(cache) = desc.cache else {
                break 'cache None;
            };
            let Ok(cache) = hub.pipeline_caches.get(cache) else {
                break 'cache None;
            };

            if cache.device.as_info().id() != self.as_info().id() {
                return Err(DeviceError::WrongDevice.into());
            }
            Some(cache)
        };

        let pipeline_desc = hal::RenderPipelineDescriptor {
            label: desc.label.to_hal(self.instance_flags),
            layout: pipeline_layout.raw(),
            vertex_buffers: &vertex_buffers,
            vertex_stage,
            primitive: desc.primitive,
            depth_stencil: desc.depth_stencil.clone(),
            multisample: desc.multisample,
            fragment_stage,
            color_targets,
            multiview: desc.multiview,
            cache: pipeline_cache.as_ref().and_then(|it| it.raw.as_ref()),
        };
        let raw = unsafe {
            self.raw
                .as_ref()
                .unwrap()
                .create_render_pipeline(&pipeline_desc)
        }
        .map_err(|err| match err {
            hal::PipelineError::Device(error) => {
                pipeline::CreateRenderPipelineError::Device(error.into())
            }
            hal::PipelineError::Linkage(stage, msg) => {
                pipeline::CreateRenderPipelineError::Internal { stage, error: msg }
            }
            hal::PipelineError::EntryPoint(stage) => {
                pipeline::CreateRenderPipelineError::Internal {
                    stage: hal::auxil::map_naga_stage(stage),
                    error: ENTRYPOINT_FAILURE_ERROR.to_string(),
                }
            }
        })?;

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

        let shader_modules = {
            let mut shader_modules = ArrayVec::new();
            shader_modules.push(vertex_shader_module);
            shader_modules.extend(fragment_shader_module);
            shader_modules
        };

        let pipeline = pipeline::RenderPipeline {
            raw: Some(raw),
            layout: pipeline_layout,
            device: self.clone(),
            pass_context,
            _shader_modules: shader_modules,
            flags,
            strip_index_format: desc.primitive.strip_index_format,
            vertex_steps,
            late_sized_buffer_groups,
            info: ResourceInfo::new(
                desc.label.borrow_or_default(),
                Some(self.tracker_indices.render_pipelines.clone()),
            ),
        };
        Ok(pipeline)
    }

    /// # Safety
    /// The `data` field on `desc` must have previously been returned from [`crate::global::Global::pipeline_cache_get_data`]
    pub unsafe fn create_pipeline_cache(
        self: &Arc<Self>,
        desc: &pipeline::PipelineCacheDescriptor,
    ) -> Result<pipeline::PipelineCache<A>, pipeline::CreatePipelineCacheError> {
        use crate::pipeline_cache;
        self.require_features(wgt::Features::PIPELINE_CACHE)?;
        let data = if let Some((data, validation_key)) = desc
            .data
            .as_ref()
            .zip(self.raw().pipeline_cache_validation_key())
        {
            let data = pipeline_cache::validate_pipeline_cache(
                data,
                &self.adapter.raw.info,
                validation_key,
            );
            match data {
                Ok(data) => Some(data),
                Err(e) if e.was_avoidable() || !desc.fallback => return Err(e.into()),
                // If the error was unavoidable and we are asked to fallback, do so
                Err(_) => None,
            }
        } else {
            None
        };
        let cache_desc = hal::PipelineCacheDescriptor {
            data,
            label: desc.label.to_hal(self.instance_flags),
        };
        let raw = match unsafe { self.raw().create_pipeline_cache(&cache_desc) } {
            Ok(raw) => raw,
            Err(e) => return Err(e.into()),
        };
        let cache = pipeline::PipelineCache {
            device: self.clone(),
            info: ResourceInfo::new(
                desc.label.borrow_or_default(),
                Some(self.tracker_indices.pipeline_caches.clone()),
            ),
            // This would be none in the error condition, which we don't implement yet
            raw: Some(raw),
        };
        Ok(cache)
    }

    pub(crate) fn get_texture_format_features(
        &self,
        adapter: &Adapter<A>,
        format: TextureFormat,
    ) -> wgt::TextureFormatFeatures {
        // Variant of adapter.get_texture_format_features that takes device features into account
        use wgt::TextureFormatFeatureFlags as tfsc;
        let mut format_features = adapter.get_texture_format_features(format);
        if (format == TextureFormat::R32Float
            || format == TextureFormat::Rg32Float
            || format == TextureFormat::Rgba32Float)
            && !self.features.contains(wgt::Features::FLOAT32_FILTERABLE)
        {
            format_features.flags.set(tfsc::FILTERABLE, false);
        }
        format_features
    }

    pub(crate) fn describe_format_features(
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
        let downlevel = !self
            .downlevel
            .flags
            .contains(wgt::DownlevelFlags::WEBGPU_TEXTURE_FORMAT_SUPPORT);

        if using_device_features || downlevel {
            Ok(self.get_texture_format_features(adapter, format))
        } else {
            Ok(format.guaranteed_format_features(self.features))
        }
    }

    pub(crate) fn wait_for_submit(
        &self,
        submission_index: SubmissionIndex,
    ) -> Result<(), WaitIdleError> {
        let guard = self.fence.read();
        let fence = guard.as_ref().unwrap();
        let last_done_index = unsafe {
            self.raw
                .as_ref()
                .unwrap()
                .get_fence_value(fence)
                .map_err(DeviceError::from)?
        };
        if last_done_index < submission_index {
            log::info!("Waiting for submission {:?}", submission_index);
            unsafe {
                self.raw
                    .as_ref()
                    .unwrap()
                    .wait(fence, submission_index, !0)
                    .map_err(DeviceError::from)?
            };
            drop(guard);
            let closures = self
                .lock_life()
                .triage_submissions(submission_index, &self.command_allocator);
            assert!(
                closures.is_empty(),
                "wait_for_submit is not expected to work with closures"
            );
        }
        Ok(())
    }

    pub(crate) fn create_query_set(
        self: &Arc<Self>,
        desc: &resource::QuerySetDescriptor,
    ) -> Result<QuerySet<A>, resource::CreateQuerySetError> {
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
        Ok(QuerySet {
            raw: Some(unsafe { self.raw().create_query_set(&hal_desc).unwrap() }),
            device: self.clone(),
            info: ResourceInfo::new("", Some(self.tracker_indices.query_sets.clone())),
            desc: desc.map_label(|_| ()),
        })
    }

    pub(crate) fn lose(&self, message: &str) {
        // Follow the steps at https://gpuweb.github.io/gpuweb/#lose-the-device.

        // Mark the device explicitly as invalid. This is checked in various
        // places to prevent new work from being submitted.
        self.valid.store(false, Ordering::Release);

        // 1) Resolve the GPUDevice device.lost promise.
        let mut life_lock = self.lock_life();
        let closure = life_lock.device_lost_closure.take();
        // It's important to not hold the lock while calling the closure and while calling
        // release_gpu_resources which may take the lock again.
        drop(life_lock);

        if let Some(device_lost_closure) = closure {
            device_lost_closure.call(DeviceLostReason::Unknown, message.to_string());
        }

        // 2) Complete any outstanding mapAsync() steps.
        // 3) Complete any outstanding onSubmittedWorkDone() steps.

        // These parts are passively accomplished by setting valid to false,
        // since that will prevent any new work from being added to the queues.
        // Future calls to poll_devices will continue to check the work queues
        // until they are cleared, and then drop the device.

        // Eagerly release GPU resources.
        self.release_gpu_resources();
    }

    pub(crate) fn release_gpu_resources(&self) {
        // This is called when the device is lost, which makes every associated
        // resource invalid and unusable. This is an opportunity to release all of
        // the underlying gpu resources, even though the objects remain visible to
        // the user agent. We purge this memory naturally when resources have been
        // moved into the appropriate buckets, so this function just needs to
        // initiate movement into those buckets, and it can do that by calling
        // "destroy" on all the resources we know about.

        // During these iterations, we discard all errors. We don't care!
        let trackers = self.trackers.lock();
        for buffer in trackers.buffers.used_resources() {
            let _ = buffer.destroy();
        }
        for texture in trackers.textures.used_resources() {
            let _ = texture.destroy();
        }
    }

    pub(crate) fn new_usage_scope(&self) -> UsageScope<'_, A> {
        UsageScope::new_pooled(&self.usage_scopes, &self.tracker_indices)
    }
}

impl<A: HalApi> Device<A> {
    pub(crate) fn destroy_command_buffer(&self, mut cmd_buf: command::CommandBuffer<A>) {
        let mut baked = cmd_buf.extract_baked_commands();
        unsafe {
            baked.encoder.reset_all(baked.list.into_iter());
        }
        unsafe {
            self.raw
                .as_ref()
                .unwrap()
                .destroy_command_encoder(baked.encoder);
        }
    }

    /// Wait for idle and remove resources that we can, before we die.
    pub(crate) fn prepare_to_die(&self) {
        self.pending_writes.lock().as_mut().unwrap().deactivate();
        let current_index = self.active_submission_index.load(Ordering::Relaxed);
        if let Err(error) = unsafe {
            let fence = self.fence.read();
            let fence = fence.as_ref().unwrap();
            self.raw
                .as_ref()
                .unwrap()
                .wait(fence, current_index, CLEANUP_WAIT_MS)
        } {
            log::error!("failed to wait for the device: {error}");
        }
        let mut life_tracker = self.lock_life();
        let _ = life_tracker.triage_submissions(current_index, &self.command_allocator);
        if let Some(device_lost_closure) = life_tracker.device_lost_closure.take() {
            // It's important to not hold the lock while calling the closure.
            drop(life_tracker);
            device_lost_closure.call(DeviceLostReason::Dropped, "Device is dying.".to_string());
        }
        #[cfg(feature = "trace")]
        {
            *self.trace.lock() = None;
        }
    }
}

impl<A: HalApi> Resource for Device<A> {
    const TYPE: ResourceType = "Device";

    type Marker = id::markers::Device;

    fn as_info(&self) -> &ResourceInfo<Self> {
        &self.info
    }

    fn as_info_mut(&mut self) -> &mut ResourceInfo<Self> {
        &mut self.info
    }
}
