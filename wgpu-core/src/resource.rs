#[cfg(feature = "trace")]
use crate::device::trace;
use crate::{
    binding_model::BindGroup,
    device::{
        queue, resource::DeferredDestroy, BufferMapPendingClosure, Device, DeviceError,
        DeviceMismatch, HostMap, MissingDownlevelFlags, MissingFeatures,
    },
    global::Global,
    hal_api::HalApi,
    id::{
        AdapterId, BufferId, CommandEncoderId, DeviceId, Id, Marker, SurfaceId, TextureId,
        TextureViewId,
    },
    init_tracker::{BufferInitTracker, TextureInitTracker},
    lock::{Mutex, RwLock},
    resource_log,
    snatch::{ExclusiveSnatchGuard, SnatchGuard, Snatchable},
    track::{SharedTrackerIndexAllocator, TextureSelector, TrackerIndex},
    Label, SubmissionIndex,
};

use hal::CommandEncoder;
use smallvec::SmallVec;
use thiserror::Error;
use wgt::WasmNotSendSync;

use std::{
    borrow::Borrow,
    fmt::Debug,
    iter, mem,
    ops::Range,
    ptr::NonNull,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Weak,
    },
};

/// Information about the wgpu-core resource.
///
/// Each type representing a `wgpu-core` resource, like [`Device`],
/// [`Buffer`], etc., contains a `ResourceInfo` which contains
/// its latest submission index and label.
///
/// A resource may need to be retained for any of several reasons:
/// and any lifetime logic will be handled by `Arc<Resource>` refcount
///
/// - The user may hold a reference to it (via a `wgpu::Buffer`, say).
///
/// - Other resources may depend on it (a texture view's backing
///   texture, for example).
///
/// - It may be used by commands sent to the GPU that have not yet
///   finished execution.
///
/// [`Device`]: crate::device::resource::Device
/// [`Buffer`]: crate::resource::Buffer
#[derive(Debug)]
pub(crate) struct ResourceInfo<T: Resource> {
    id: Option<Id<T::Marker>>,
    tracker_index: TrackerIndex,
    tracker_indices: Option<Arc<SharedTrackerIndexAllocator>>,
    /// The index of the last queue submission in which the resource
    /// was used.
    ///
    /// Each queue submission is fenced and assigned an index number
    /// sequentially. Thus, when a queue submission completes, we know any
    /// resources used in that submission and any lower-numbered submissions are
    /// no longer in use by the GPU.
    submission_index: AtomicUsize,

    /// The `label` from the descriptor used to create the resource.
    label: String,
}

impl<T: Resource> Drop for ResourceInfo<T> {
    fn drop(&mut self) {
        if let Some(indices) = &self.tracker_indices {
            indices.free(self.tracker_index);
        }
    }
}

impl<T: Resource> ResourceInfo<T> {
    pub(crate) fn new(
        label: &Label,
        tracker_indices: Option<Arc<SharedTrackerIndexAllocator>>,
    ) -> Self {
        let tracker_index = tracker_indices
            .as_ref()
            .map(|indices| indices.alloc())
            .unwrap_or(TrackerIndex::INVALID);
        Self {
            id: None,
            tracker_index,
            tracker_indices,
            submission_index: AtomicUsize::new(0),
            label: label
                .as_ref()
                .map(|label| label.to_string())
                .unwrap_or_default(),
        }
    }

    pub(crate) fn id(&self) -> Id<T::Marker> {
        self.id.unwrap()
    }

    pub(crate) fn tracker_index(&self) -> TrackerIndex {
        debug_assert!(self.tracker_index != TrackerIndex::INVALID);
        self.tracker_index
    }

    pub(crate) fn set_id(&mut self, id: Id<T::Marker>) {
        self.id = Some(id);
    }

    /// Record that this resource will be used by the queue submission with the
    /// given index.
    pub(crate) fn use_at(&self, submit_index: SubmissionIndex) {
        self.submission_index
            .store(submit_index as _, Ordering::Release);
    }

    pub(crate) fn submission_index(&self) -> SubmissionIndex {
        self.submission_index.load(Ordering::Acquire) as _
    }
}

#[derive(Clone, Debug)]
pub struct ResourceErrorIdent {
    r#type: ResourceType,
    label: String,
}

impl std::fmt::Display for ResourceErrorIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{} with '{}' label", self.r#type, self.label)
    }
}

pub(crate) trait ParentDevice<A: HalApi>: Resource {
    fn device(&self) -> &Arc<Device<A>>;

    fn same_device_as<O: ParentDevice<A>>(&self, other: &O) -> Result<(), DeviceError> {
        if self.device().is_equal(other.device()) {
            Ok(())
        } else {
            Err(DeviceError::DeviceMismatch(Box::new(DeviceMismatch {
                res: self.error_ident(),
                res_device: self.device().error_ident(),
                target: Some(other.error_ident()),
                target_device: other.device().error_ident(),
            })))
        }
    }

    fn same_device(&self, device: &Arc<Device<A>>) -> Result<(), DeviceError> {
        if self.device().is_equal(device) {
            Ok(())
        } else {
            Err(DeviceError::DeviceMismatch(Box::new(DeviceMismatch {
                res: self.error_ident(),
                res_device: self.device().error_ident(),
                target: None,
                target_device: device.error_ident(),
            })))
        }
    }
}

pub(crate) type ResourceType = &'static str;

pub(crate) trait Resource: 'static + Sized + WasmNotSendSync {
    type Marker: Marker;
    const TYPE: ResourceType;
    fn as_info(&self) -> &ResourceInfo<Self>;
    fn as_info_mut(&mut self) -> &mut ResourceInfo<Self>;

    /// Returns a string identifying this resource for logging and errors.
    ///
    /// It may be a user-provided string or it may be a placeholder from wgpu.
    ///
    /// It is non-empty unless the user-provided string was empty.
    fn label(&self) -> &str {
        &self.as_info().label
    }

    fn ref_count(self: &Arc<Self>) -> usize {
        Arc::strong_count(self)
    }
    fn is_unique(self: &Arc<Self>) -> bool {
        self.ref_count() == 1
    }
    fn is_equal(self: &Arc<Self>, other: &Arc<Self>) -> bool {
        Arc::ptr_eq(self, other)
    }
    fn error_ident(&self) -> ResourceErrorIdent {
        ResourceErrorIdent {
            r#type: Self::TYPE,
            label: self.label().to_owned(),
        }
    }
}

/// The status code provided to the buffer mapping callback.
///
/// This is very similar to `BufferAccessResult`, except that this is FFI-friendly.
#[repr(C)]
#[derive(Debug)]
pub enum BufferMapAsyncStatus {
    /// The Buffer is successfully mapped, `get_mapped_range` can be called.
    ///
    /// All other variants of this enum represent failures to map the buffer.
    Success,
    /// The buffer is already mapped.
    ///
    /// While this is treated as an error, it does not prevent mapped range from being accessed.
    AlreadyMapped,
    /// Mapping was already requested.
    MapAlreadyPending,
    /// An unknown error.
    Error,
    /// Mapping was aborted (by unmapping or destroying the buffer before mapping
    /// happened).
    Aborted,
    /// The context is Lost.
    ContextLost,
    /// The buffer is in an invalid state.
    Invalid,
    /// The range isn't fully contained in the buffer.
    InvalidRange,
    /// The range isn't properly aligned.
    InvalidAlignment,
    /// Incompatible usage flags.
    InvalidUsageFlags,
}

#[derive(Debug)]
pub(crate) enum BufferMapState<A: HalApi> {
    /// Mapped at creation.
    Init {
        ptr: NonNull<u8>,
        stage_buffer: Arc<Buffer<A>>,
        needs_flush: bool,
    },
    /// Waiting for GPU to be done before mapping
    Waiting(BufferPendingMapping<A>),
    /// Mapped
    Active {
        ptr: NonNull<u8>,
        range: hal::MemoryRange,
        host: HostMap,
    },
    /// Not mapped
    Idle,
}

#[cfg(send_sync)]
unsafe impl<A: HalApi> Send for BufferMapState<A> {}
#[cfg(send_sync)]
unsafe impl<A: HalApi> Sync for BufferMapState<A> {}

#[repr(C)]
pub struct BufferMapCallbackC {
    pub callback: unsafe extern "C" fn(status: BufferMapAsyncStatus, user_data: *mut u8),
    pub user_data: *mut u8,
}

#[cfg(send_sync)]
unsafe impl Send for BufferMapCallbackC {}

#[derive(Debug)]
pub struct BufferMapCallback {
    // We wrap this so creating the enum in the C variant can be unsafe,
    // allowing our call function to be safe.
    inner: BufferMapCallbackInner,
}

#[cfg(send_sync)]
type BufferMapCallbackCallback = Box<dyn FnOnce(BufferAccessResult) + Send + 'static>;
#[cfg(not(send_sync))]
type BufferMapCallbackCallback = Box<dyn FnOnce(BufferAccessResult) + 'static>;

enum BufferMapCallbackInner {
    Rust { callback: BufferMapCallbackCallback },
    C { inner: BufferMapCallbackC },
}

impl Debug for BufferMapCallbackInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            BufferMapCallbackInner::Rust { callback: _ } => f.debug_struct("Rust").finish(),
            BufferMapCallbackInner::C { inner: _ } => f.debug_struct("C").finish(),
        }
    }
}

impl BufferMapCallback {
    pub fn from_rust(callback: BufferMapCallbackCallback) -> Self {
        Self {
            inner: BufferMapCallbackInner::Rust { callback },
        }
    }

    /// # Safety
    ///
    /// - The callback pointer must be valid to call with the provided user_data
    ///   pointer.
    ///
    /// - Both pointers must point to valid memory until the callback is
    ///   invoked, which may happen at an unspecified time.
    pub unsafe fn from_c(inner: BufferMapCallbackC) -> Self {
        Self {
            inner: BufferMapCallbackInner::C { inner },
        }
    }

    pub(crate) fn call(self, result: BufferAccessResult) {
        match self.inner {
            BufferMapCallbackInner::Rust { callback } => {
                callback(result);
            }
            // SAFETY: the contract of the call to from_c says that this unsafe is sound.
            BufferMapCallbackInner::C { inner } => unsafe {
                let status = match result {
                    Ok(()) => BufferMapAsyncStatus::Success,
                    Err(BufferAccessError::Device(_)) => BufferMapAsyncStatus::ContextLost,
                    Err(BufferAccessError::InvalidBufferId(_))
                    | Err(BufferAccessError::DestroyedResource(_)) => BufferMapAsyncStatus::Invalid,
                    Err(BufferAccessError::AlreadyMapped) => BufferMapAsyncStatus::AlreadyMapped,
                    Err(BufferAccessError::MapAlreadyPending) => {
                        BufferMapAsyncStatus::MapAlreadyPending
                    }
                    Err(BufferAccessError::MissingBufferUsage(_)) => {
                        BufferMapAsyncStatus::InvalidUsageFlags
                    }
                    Err(BufferAccessError::UnalignedRange)
                    | Err(BufferAccessError::UnalignedRangeSize { .. })
                    | Err(BufferAccessError::UnalignedOffset { .. }) => {
                        BufferMapAsyncStatus::InvalidAlignment
                    }
                    Err(BufferAccessError::OutOfBoundsUnderrun { .. })
                    | Err(BufferAccessError::OutOfBoundsOverrun { .. })
                    | Err(BufferAccessError::NegativeRange { .. }) => {
                        BufferMapAsyncStatus::InvalidRange
                    }
                    Err(_) => BufferMapAsyncStatus::Error,
                };

                (inner.callback)(status, inner.user_data);
            },
        }
    }
}

#[derive(Debug)]
pub struct BufferMapOperation {
    pub host: HostMap,
    pub callback: Option<BufferMapCallback>,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum BufferAccessError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Buffer map failed")]
    Failed,
    #[error("BufferId {0:?} is invalid")]
    InvalidBufferId(BufferId),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error("Buffer is already mapped")]
    AlreadyMapped,
    #[error("Buffer map is pending")]
    MapAlreadyPending,
    #[error(transparent)]
    MissingBufferUsage(#[from] MissingBufferUsageError),
    #[error("Buffer is not mapped")]
    NotMapped,
    #[error(
        "Buffer map range must start aligned to `MAP_ALIGNMENT` and end to `COPY_BUFFER_ALIGNMENT`"
    )]
    UnalignedRange,
    #[error("Buffer offset invalid: offset {offset} must be multiple of 8")]
    UnalignedOffset { offset: wgt::BufferAddress },
    #[error("Buffer range size invalid: range_size {range_size} must be multiple of 4")]
    UnalignedRangeSize { range_size: wgt::BufferAddress },
    #[error("Buffer access out of bounds: index {index} would underrun the buffer (limit: {min})")]
    OutOfBoundsUnderrun {
        index: wgt::BufferAddress,
        min: wgt::BufferAddress,
    },
    #[error(
        "Buffer access out of bounds: last index {index} would overrun the buffer (limit: {max})"
    )]
    OutOfBoundsOverrun {
        index: wgt::BufferAddress,
        max: wgt::BufferAddress,
    },
    #[error("Buffer map range start {start} is greater than end {end}")]
    NegativeRange {
        start: wgt::BufferAddress,
        end: wgt::BufferAddress,
    },
    #[error("Buffer map aborted")]
    MapAborted,
}

#[derive(Clone, Debug, Error)]
#[error("Usage flags {actual:?} of {res} do not contain required usage flags {expected:?}")]
pub struct MissingBufferUsageError {
    pub(crate) res: ResourceErrorIdent,
    pub(crate) actual: wgt::BufferUsages,
    pub(crate) expected: wgt::BufferUsages,
}

#[derive(Clone, Debug, Error)]
#[error("Usage flags {actual:?} of {res} do not contain required usage flags {expected:?}")]
pub struct MissingTextureUsageError {
    pub(crate) res: ResourceErrorIdent,
    pub(crate) actual: wgt::TextureUsages,
    pub(crate) expected: wgt::TextureUsages,
}

#[derive(Clone, Debug, Error)]
#[error("{0} has been destroyed")]
pub struct DestroyedResourceError(pub ResourceErrorIdent);

pub type BufferAccessResult = Result<(), BufferAccessError>;

#[derive(Debug)]
pub(crate) struct BufferPendingMapping<A: HalApi> {
    pub(crate) range: Range<wgt::BufferAddress>,
    pub(crate) op: BufferMapOperation,
    // hold the parent alive while the mapping is active
    pub(crate) _parent_buffer: Arc<Buffer<A>>,
}

pub type BufferDescriptor<'a> = wgt::BufferDescriptor<Label<'a>>;

#[derive(Debug)]
pub struct Buffer<A: HalApi> {
    pub(crate) raw: Snatchable<A::Buffer>,
    pub(crate) device: Arc<Device<A>>,
    pub(crate) usage: wgt::BufferUsages,
    pub(crate) size: wgt::BufferAddress,
    pub(crate) initialization_status: RwLock<BufferInitTracker>,
    pub(crate) sync_mapped_writes: Mutex<Option<hal::MemoryRange>>,
    pub(crate) info: ResourceInfo<Buffer<A>>,
    pub(crate) map_state: Mutex<BufferMapState<A>>,
    pub(crate) bind_groups: Mutex<Vec<Weak<BindGroup<A>>>>,
}

impl<A: HalApi> Drop for Buffer<A> {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            resource_log!("Destroy raw {}", self.error_ident());
            unsafe {
                use hal::Device;
                self.device.raw().destroy_buffer(raw);
            }
        }
    }
}

impl<A: HalApi> Buffer<A> {
    pub(crate) fn raw(&self, guard: &SnatchGuard) -> Option<&A::Buffer> {
        self.raw.get(guard)
    }

    pub(crate) fn try_raw<'a>(
        &'a self,
        guard: &'a SnatchGuard,
    ) -> Result<&A::Buffer, DestroyedResourceError> {
        self.raw
            .get(guard)
            .ok_or_else(|| DestroyedResourceError(self.error_ident()))
    }

    pub(crate) fn check_destroyed<'a>(
        &'a self,
        guard: &'a SnatchGuard,
    ) -> Result<(), DestroyedResourceError> {
        self.raw
            .get(guard)
            .map(|_| ())
            .ok_or_else(|| DestroyedResourceError(self.error_ident()))
    }

    /// Checks that the given buffer usage contains the required buffer usage,
    /// returns an error otherwise.
    pub(crate) fn check_usage(
        self: &Arc<Self>,
        expected: wgt::BufferUsages,
    ) -> Result<(), MissingBufferUsageError> {
        if self.usage.contains(expected) {
            Ok(())
        } else {
            Err(MissingBufferUsageError {
                res: self.error_ident(),
                actual: self.usage,
                expected,
            })
        }
    }

    /// Returns the mapping callback in case of error so that the callback can be fired outside
    /// of the locks that are held in this function.
    pub(crate) fn map_async(
        self: &Arc<Self>,
        offset: wgt::BufferAddress,
        size: Option<wgt::BufferAddress>,
        op: BufferMapOperation,
    ) -> Result<(), (BufferMapOperation, BufferAccessError)> {
        let range_size = if let Some(size) = size {
            size
        } else if offset > self.size {
            0
        } else {
            self.size - offset
        };

        if offset % wgt::MAP_ALIGNMENT != 0 {
            return Err((op, BufferAccessError::UnalignedOffset { offset }));
        }
        if range_size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err((op, BufferAccessError::UnalignedRangeSize { range_size }));
        }

        let range = offset..(offset + range_size);

        if range.start % wgt::MAP_ALIGNMENT != 0 || range.end % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err((op, BufferAccessError::UnalignedRange));
        }

        let (pub_usage, internal_use) = match op.host {
            HostMap::Read => (wgt::BufferUsages::MAP_READ, hal::BufferUses::MAP_READ),
            HostMap::Write => (wgt::BufferUsages::MAP_WRITE, hal::BufferUses::MAP_WRITE),
        };

        if let Err(e) = self.check_usage(pub_usage) {
            return Err((op, e.into()));
        }

        if range.start > range.end {
            return Err((
                op,
                BufferAccessError::NegativeRange {
                    start: range.start,
                    end: range.end,
                },
            ));
        }
        if range.end > self.size {
            return Err((
                op,
                BufferAccessError::OutOfBoundsOverrun {
                    index: range.end,
                    max: self.size,
                },
            ));
        }

        let device = &self.device;
        if let Err(e) = device.check_is_valid() {
            return Err((op, e.into()));
        }

        {
            let snatch_guard = device.snatchable_lock.read();
            if let Err(e) = self.check_destroyed(&snatch_guard) {
                return Err((op, e.into()));
            }
        }

        {
            let map_state = &mut *self.map_state.lock();
            *map_state = match *map_state {
                BufferMapState::Init { .. } | BufferMapState::Active { .. } => {
                    return Err((op, BufferAccessError::AlreadyMapped));
                }
                BufferMapState::Waiting(_) => {
                    return Err((op, BufferAccessError::MapAlreadyPending));
                }
                BufferMapState::Idle => BufferMapState::Waiting(BufferPendingMapping {
                    range,
                    op,
                    _parent_buffer: self.clone(),
                }),
            };
        }

        let snatch_guard = device.snatchable_lock.read();
        {
            let mut trackers = device.as_ref().trackers.lock();
            trackers.buffers.set_single(self, internal_use);
            //TODO: Check if draining ALL buffers is correct!
            let _ = trackers.buffers.drain_transitions(&snatch_guard);
        }
        drop(snatch_guard);

        device.lock_life().map(self);

        Ok(())
    }

    // Note: This must not be called while holding a lock.
    pub(crate) fn unmap(
        self: &Arc<Self>,
        #[cfg(feature = "trace")] buffer_id: BufferId,
    ) -> Result<(), BufferAccessError> {
        if let Some((mut operation, status)) = self.unmap_inner(
            #[cfg(feature = "trace")]
            buffer_id,
        )? {
            if let Some(callback) = operation.callback.take() {
                callback.call(status);
            }
        }

        Ok(())
    }

    fn unmap_inner(
        self: &Arc<Self>,
        #[cfg(feature = "trace")] buffer_id: BufferId,
    ) -> Result<Option<BufferMapPendingClosure>, BufferAccessError> {
        use hal::Device;

        let device = &self.device;
        let snatch_guard = device.snatchable_lock.read();
        let raw_buf = self.try_raw(&snatch_guard)?;
        log::debug!("{} map state -> Idle", self.error_ident());
        match mem::replace(&mut *self.map_state.lock(), BufferMapState::Idle) {
            BufferMapState::Init {
                ptr,
                stage_buffer,
                needs_flush,
            } => {
                #[cfg(feature = "trace")]
                if let Some(ref mut trace) = *device.trace.lock() {
                    let data = trace.make_binary("bin", unsafe {
                        std::slice::from_raw_parts(ptr.as_ptr(), self.size as usize)
                    });
                    trace.add(trace::Action::WriteBuffer {
                        id: buffer_id,
                        data,
                        range: 0..self.size,
                        queued: true,
                    });
                }
                let _ = ptr;
                if needs_flush {
                    unsafe {
                        device.raw().flush_mapped_ranges(
                            stage_buffer.raw(&snatch_guard).unwrap(),
                            iter::once(0..self.size),
                        );
                    }
                }

                self.info
                    .use_at(device.active_submission_index.load(Ordering::Relaxed) + 1);
                let region = wgt::BufferSize::new(self.size).map(|size| hal::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size,
                });
                let transition_src = hal::BufferBarrier {
                    buffer: stage_buffer.raw(&snatch_guard).unwrap(),
                    usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
                };
                let transition_dst = hal::BufferBarrier {
                    buffer: raw_buf,
                    usage: hal::BufferUses::empty()..hal::BufferUses::COPY_DST,
                };
                let mut pending_writes = device.pending_writes.lock();
                let pending_writes = pending_writes.as_mut().unwrap();
                let encoder = pending_writes.activate();
                unsafe {
                    encoder.transition_buffers(
                        iter::once(transition_src).chain(iter::once(transition_dst)),
                    );
                    if self.size > 0 {
                        encoder.copy_buffer_to_buffer(
                            stage_buffer.raw(&snatch_guard).unwrap(),
                            raw_buf,
                            region.into_iter(),
                        );
                    }
                }
                pending_writes.consume_temp(queue::TempResource::Buffer(stage_buffer));
                pending_writes.insert_buffer(self);
            }
            BufferMapState::Idle => {
                return Err(BufferAccessError::NotMapped);
            }
            BufferMapState::Waiting(pending) => {
                return Ok(Some((pending.op, Err(BufferAccessError::MapAborted))));
            }
            BufferMapState::Active { ptr, range, host } => {
                if host == HostMap::Write {
                    #[cfg(feature = "trace")]
                    if let Some(ref mut trace) = *device.trace.lock() {
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
                        .raw()
                        .unmap_buffer(raw_buf)
                        .map_err(DeviceError::from)?
                };
            }
        }
        Ok(None)
    }

    pub(crate) fn destroy(self: &Arc<Self>) -> Result<(), DestroyError> {
        let device = &self.device;

        let temp = {
            let snatch_guard = device.snatchable_lock.write();
            let raw = match self.raw.snatch(snatch_guard) {
                Some(raw) => raw,
                None => {
                    return Err(DestroyError::AlreadyDestroyed);
                }
            };

            let bind_groups = {
                let mut guard = self.bind_groups.lock();
                mem::take(&mut *guard)
            };

            queue::TempResource::DestroyedBuffer(Arc::new(DestroyedBuffer {
                raw: Some(raw),
                device: Arc::clone(&self.device),
                submission_index: self.info.submission_index(),
                tracker_index: self.info.tracker_index(),
                label: self.label().to_owned(),
                bind_groups,
            }))
        };

        let mut pending_writes = device.pending_writes.lock();
        let pending_writes = pending_writes.as_mut().unwrap();
        if pending_writes.contains_buffer(self) {
            pending_writes.consume_temp(temp);
        } else {
            let last_submit_index = self.info.submission_index();
            device
                .lock_life()
                .schedule_resource_destruction(temp, last_submit_index);
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateBufferError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Failed to map buffer while creating: {0}")]
    AccessError(#[from] BufferAccessError),
    #[error("Buffers that are mapped at creation have to be aligned to `COPY_BUFFER_ALIGNMENT`")]
    UnalignedSize,
    #[error("Invalid usage flags {0:?}")]
    InvalidUsage(wgt::BufferUsages),
    #[error("`MAP` usage can only be combined with the opposite `COPY`, requested {0:?}")]
    UsageMismatch(wgt::BufferUsages),
    #[error("Buffer size {requested} is greater than the maximum buffer size ({maximum})")]
    MaxBufferSize { requested: u64, maximum: u64 },
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
}

impl<A: HalApi> Resource for Buffer<A> {
    const TYPE: ResourceType = "Buffer";

    type Marker = crate::id::markers::Buffer;

    fn as_info(&self) -> &ResourceInfo<Self> {
        &self.info
    }

    fn as_info_mut(&mut self) -> &mut ResourceInfo<Self> {
        &mut self.info
    }
}

impl<A: HalApi> ParentDevice<A> for Buffer<A> {
    fn device(&self) -> &Arc<Device<A>> {
        &self.device
    }
}

/// A buffer that has been marked as destroyed and is staged for actual deletion soon.
#[derive(Debug)]
pub struct DestroyedBuffer<A: HalApi> {
    raw: Option<A::Buffer>,
    device: Arc<Device<A>>,
    label: String,
    pub(crate) tracker_index: TrackerIndex,
    pub(crate) submission_index: u64,
    bind_groups: Vec<Weak<BindGroup<A>>>,
}

impl<A: HalApi> DestroyedBuffer<A> {
    pub fn label(&self) -> &dyn Debug {
        &self.label
    }
}

impl<A: HalApi> Drop for DestroyedBuffer<A> {
    fn drop(&mut self) {
        let mut deferred = self.device.deferred_destroy.lock();
        for bind_group in self.bind_groups.drain(..) {
            deferred.push(DeferredDestroy::BindGroup(bind_group));
        }
        drop(deferred);

        if let Some(raw) = self.raw.take() {
            resource_log!("Destroy raw Buffer (destroyed) {:?}", self.label());

            unsafe {
                use hal::Device;
                self.device.raw().destroy_buffer(raw);
            }
        }
    }
}

/// A temporary buffer, consumed by the command that uses it.
///
/// A [`StagingBuffer`] is designed for one-shot uploads of data to the GPU. It
/// is always created mapped, and the command that uses it destroys the buffer
/// when it is done.
///
/// [`StagingBuffer`]s can be created with [`queue_create_staging_buffer`] and
/// used with [`queue_write_staging_buffer`]. They are also used internally by
/// operations like [`queue_write_texture`] that need to upload data to the GPU,
/// but that don't belong to any particular wgpu command buffer.
///
/// Used `StagingBuffer`s are accumulated in [`Device::pending_writes`], to be
/// freed once their associated operation's queue submission has finished
/// execution.
///
/// [`queue_create_staging_buffer`]: Global::queue_create_staging_buffer
/// [`queue_write_staging_buffer`]: Global::queue_write_staging_buffer
/// [`queue_write_texture`]: Global::queue_write_texture
/// [`Device::pending_writes`]: crate::device::Device
#[derive(Debug)]
pub struct StagingBuffer<A: HalApi> {
    pub(crate) raw: Mutex<Option<A::Buffer>>,
    pub(crate) device: Arc<Device<A>>,
    pub(crate) size: wgt::BufferAddress,
    pub(crate) is_coherent: bool,
    pub(crate) info: ResourceInfo<StagingBuffer<A>>,
}

impl<A: HalApi> Drop for StagingBuffer<A> {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.lock().take() {
            resource_log!("Destroy raw {}", self.error_ident());
            unsafe {
                use hal::Device;
                self.device.raw().destroy_buffer(raw);
            }
        }
    }
}

impl<A: HalApi> Resource for StagingBuffer<A> {
    const TYPE: ResourceType = "StagingBuffer";

    type Marker = crate::id::markers::StagingBuffer;

    fn as_info(&self) -> &ResourceInfo<Self> {
        &self.info
    }

    fn as_info_mut(&mut self) -> &mut ResourceInfo<Self> {
        &mut self.info
    }
}

impl<A: HalApi> ParentDevice<A> for StagingBuffer<A> {
    fn device(&self) -> &Arc<Device<A>> {
        &self.device
    }
}

pub type TextureDescriptor<'a> = wgt::TextureDescriptor<Label<'a>, Vec<wgt::TextureFormat>>;

#[derive(Debug)]
pub(crate) enum TextureInner<A: HalApi> {
    Native {
        raw: A::Texture,
    },
    Surface {
        raw: Option<A::SurfaceTexture>,
        parent_id: SurfaceId,
    },
}

impl<A: HalApi> TextureInner<A> {
    pub(crate) fn raw(&self) -> Option<&A::Texture> {
        match self {
            Self::Native { raw } => Some(raw),
            Self::Surface { raw: Some(tex), .. } => Some(tex.borrow()),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum TextureClearMode<A: HalApi> {
    BufferCopy,
    // View for clear via RenderPass for every subsurface (mip/layer/slice)
    RenderPass {
        clear_views: SmallVec<[Option<A::TextureView>; 1]>,
        is_color: bool,
    },
    Surface {
        clear_view: Option<A::TextureView>,
    },
    // Texture can't be cleared, attempting to do so will cause panic.
    // (either because it is impossible for the type of texture or it is being destroyed)
    None,
}

#[derive(Debug)]
pub struct Texture<A: HalApi> {
    pub(crate) inner: Snatchable<TextureInner<A>>,
    pub(crate) device: Arc<Device<A>>,
    pub(crate) desc: wgt::TextureDescriptor<(), Vec<wgt::TextureFormat>>,
    pub(crate) hal_usage: hal::TextureUses,
    pub(crate) format_features: wgt::TextureFormatFeatures,
    pub(crate) initialization_status: RwLock<TextureInitTracker>,
    pub(crate) full_range: TextureSelector,
    pub(crate) info: ResourceInfo<Texture<A>>,
    pub(crate) clear_mode: RwLock<TextureClearMode<A>>,
    pub(crate) views: Mutex<Vec<Weak<TextureView<A>>>>,
    pub(crate) bind_groups: Mutex<Vec<Weak<BindGroup<A>>>>,
}

impl<A: HalApi> Texture<A> {
    /// Checks that the given texture usage contains the required texture usage,
    /// returns an error otherwise.
    pub(crate) fn check_usage(
        &self,
        expected: wgt::TextureUsages,
    ) -> Result<(), MissingTextureUsageError> {
        if self.desc.usage.contains(expected) {
            Ok(())
        } else {
            Err(MissingTextureUsageError {
                res: self.error_ident(),
                actual: self.desc.usage,
                expected,
            })
        }
    }
}

impl<A: HalApi> Drop for Texture<A> {
    fn drop(&mut self) {
        use hal::Device;
        let mut clear_mode = self.clear_mode.write();
        let clear_mode = &mut *clear_mode;
        match *clear_mode {
            TextureClearMode::Surface {
                ref mut clear_view, ..
            } => {
                if let Some(view) = clear_view.take() {
                    unsafe {
                        self.device.raw().destroy_texture_view(view);
                    }
                }
            }
            TextureClearMode::RenderPass {
                ref mut clear_views,
                ..
            } => {
                clear_views.iter_mut().for_each(|clear_view| {
                    if let Some(view) = clear_view.take() {
                        unsafe {
                            self.device.raw().destroy_texture_view(view);
                        }
                    }
                });
            }
            _ => {}
        };

        if let Some(TextureInner::Native { raw }) = self.inner.take() {
            resource_log!("Destroy raw {}", self.error_ident());
            unsafe {
                self.device.raw().destroy_texture(raw);
            }
        }
    }
}

impl<A: HalApi> Texture<A> {
    pub(crate) fn try_inner<'a>(
        &'a self,
        guard: &'a SnatchGuard,
    ) -> Result<&'a TextureInner<A>, DestroyedResourceError> {
        self.inner
            .get(guard)
            .ok_or_else(|| DestroyedResourceError(self.error_ident()))
    }

    pub(crate) fn raw<'a>(&'a self, snatch_guard: &'a SnatchGuard) -> Option<&'a A::Texture> {
        self.inner.get(snatch_guard)?.raw()
    }

    pub(crate) fn try_raw<'a>(
        &'a self,
        guard: &'a SnatchGuard,
    ) -> Result<&'a A::Texture, DestroyedResourceError> {
        self.inner
            .get(guard)
            .and_then(|t| t.raw())
            .ok_or_else(|| DestroyedResourceError(self.error_ident()))
    }

    pub(crate) fn check_destroyed<'a>(
        &'a self,
        guard: &'a SnatchGuard,
    ) -> Result<(), DestroyedResourceError> {
        self.inner
            .get(guard)
            .map(|_| ())
            .ok_or_else(|| DestroyedResourceError(self.error_ident()))
    }

    pub(crate) fn inner_mut<'a>(
        &'a self,
        guard: &mut ExclusiveSnatchGuard,
    ) -> Option<&'a mut TextureInner<A>> {
        self.inner.get_mut(guard)
    }
    pub(crate) fn get_clear_view<'a>(
        clear_mode: &'a TextureClearMode<A>,
        desc: &'a wgt::TextureDescriptor<(), Vec<wgt::TextureFormat>>,
        mip_level: u32,
        depth_or_layer: u32,
    ) -> &'a A::TextureView {
        match *clear_mode {
            TextureClearMode::BufferCopy => {
                panic!("Given texture is cleared with buffer copies, not render passes")
            }
            TextureClearMode::None => {
                panic!("Given texture can't be cleared")
            }
            TextureClearMode::Surface { ref clear_view, .. } => clear_view.as_ref().unwrap(),
            TextureClearMode::RenderPass {
                ref clear_views, ..
            } => {
                let index = if desc.dimension == wgt::TextureDimension::D3 {
                    (0..mip_level).fold(0, |acc, mip| {
                        acc + (desc.size.depth_or_array_layers >> mip).max(1)
                    })
                } else {
                    mip_level * desc.size.depth_or_array_layers
                } + depth_or_layer;
                clear_views[index as usize].as_ref().unwrap()
            }
        }
    }

    pub(crate) fn destroy(self: &Arc<Self>) -> Result<(), DestroyError> {
        let device = &self.device;

        let temp = {
            let snatch_guard = device.snatchable_lock.write();
            let raw = match self.inner.snatch(snatch_guard) {
                Some(TextureInner::Native { raw }) => raw,
                Some(TextureInner::Surface { .. }) => {
                    return Ok(());
                }
                None => {
                    return Err(DestroyError::AlreadyDestroyed);
                }
            };

            let views = {
                let mut guard = self.views.lock();
                mem::take(&mut *guard)
            };

            let bind_groups = {
                let mut guard = self.bind_groups.lock();
                mem::take(&mut *guard)
            };

            queue::TempResource::DestroyedTexture(Arc::new(DestroyedTexture {
                raw: Some(raw),
                views,
                bind_groups,
                device: Arc::clone(&self.device),
                tracker_index: self.info.tracker_index(),
                submission_index: self.info.submission_index(),
                label: self.label().to_owned(),
            }))
        };

        let mut pending_writes = device.pending_writes.lock();
        let pending_writes = pending_writes.as_mut().unwrap();
        if pending_writes.contains_texture(self) {
            pending_writes.consume_temp(temp);
        } else {
            let last_submit_index = self.info.submission_index();
            device
                .lock_life()
                .schedule_resource_destruction(temp, last_submit_index);
        }

        Ok(())
    }
}

impl Global {
    /// # Safety
    ///
    /// - The raw buffer handle must not be manually destroyed
    pub unsafe fn buffer_as_hal<A: HalApi, F: FnOnce(Option<&A::Buffer>) -> R, R>(
        &self,
        id: BufferId,
        hal_buffer_callback: F,
    ) -> R {
        profiling::scope!("Buffer::as_hal");

        let hub = A::hub(self);
        let buffer_opt = { hub.buffers.try_get(id).ok().flatten() };
        let buffer = buffer_opt.as_ref().unwrap();

        let hal_buffer = {
            let snatch_guard = buffer.device.snatchable_lock.read();
            buffer.raw(&snatch_guard)
        };

        hal_buffer_callback(hal_buffer)
    }

    /// # Safety
    ///
    /// - The raw texture handle must not be manually destroyed
    pub unsafe fn texture_as_hal<A: HalApi, F: FnOnce(Option<&A::Texture>) -> R, R>(
        &self,
        id: TextureId,
        hal_texture_callback: F,
    ) -> R {
        profiling::scope!("Texture::as_hal");

        let hub = A::hub(self);
        let texture_opt = { hub.textures.try_get(id).ok().flatten() };
        let texture = texture_opt.as_ref().unwrap();
        let snatch_guard = texture.device.snatchable_lock.read();
        let hal_texture = texture.raw(&snatch_guard);

        hal_texture_callback(hal_texture)
    }

    /// # Safety
    ///
    /// - The raw texture view handle must not be manually destroyed
    pub unsafe fn texture_view_as_hal<A: HalApi, F: FnOnce(Option<&A::TextureView>) -> R, R>(
        &self,
        id: TextureViewId,
        hal_texture_view_callback: F,
    ) -> R {
        profiling::scope!("TextureView::as_hal");

        let hub = A::hub(self);
        let texture_view_opt = { hub.texture_views.try_get(id).ok().flatten() };
        let texture_view = texture_view_opt.as_ref().unwrap();
        let snatch_guard = texture_view.device.snatchable_lock.read();
        let hal_texture_view = texture_view.raw(&snatch_guard);

        hal_texture_view_callback(hal_texture_view)
    }

    /// # Safety
    ///
    /// - The raw adapter handle must not be manually destroyed
    pub unsafe fn adapter_as_hal<A: HalApi, F: FnOnce(Option<&A::Adapter>) -> R, R>(
        &self,
        id: AdapterId,
        hal_adapter_callback: F,
    ) -> R {
        profiling::scope!("Adapter::as_hal");

        let hub = A::hub(self);
        let adapter = hub.adapters.try_get(id).ok().flatten();
        let hal_adapter = adapter.as_ref().map(|adapter| &adapter.raw.adapter);

        hal_adapter_callback(hal_adapter)
    }

    /// # Safety
    ///
    /// - The raw device handle must not be manually destroyed
    pub unsafe fn device_as_hal<A: HalApi, F: FnOnce(Option<&A::Device>) -> R, R>(
        &self,
        id: DeviceId,
        hal_device_callback: F,
    ) -> R {
        profiling::scope!("Device::as_hal");

        let hub = A::hub(self);
        let device = hub.devices.try_get(id).ok().flatten();
        let hal_device = device.as_ref().map(|device| device.raw());

        hal_device_callback(hal_device)
    }

    /// # Safety
    ///
    /// - The raw fence handle must not be manually destroyed
    pub unsafe fn device_fence_as_hal<A: HalApi, F: FnOnce(Option<&A::Fence>) -> R, R>(
        &self,
        id: DeviceId,
        hal_fence_callback: F,
    ) -> R {
        profiling::scope!("Device::fence_as_hal");

        let hub = A::hub(self);
        let device = hub.devices.try_get(id).ok().flatten();
        let hal_fence = device.as_ref().map(|device| device.fence.read());

        hal_fence_callback(hal_fence.as_deref().unwrap().as_ref())
    }

    /// # Safety
    /// - The raw surface handle must not be manually destroyed
    pub unsafe fn surface_as_hal<A: HalApi, F: FnOnce(Option<&A::Surface>) -> R, R>(
        &self,
        id: SurfaceId,
        hal_surface_callback: F,
    ) -> R {
        profiling::scope!("Surface::as_hal");

        let surface = self.surfaces.get(id).ok();
        let hal_surface = surface
            .as_ref()
            .and_then(|surface| A::surface_as_hal(surface));

        hal_surface_callback(hal_surface)
    }

    /// # Safety
    ///
    /// - The raw command encoder handle must not be manually destroyed
    pub unsafe fn command_encoder_as_hal_mut<
        A: HalApi,
        F: FnOnce(Option<&mut A::CommandEncoder>) -> R,
        R,
    >(
        &self,
        id: CommandEncoderId,
        hal_command_encoder_callback: F,
    ) -> R {
        profiling::scope!("CommandEncoder::as_hal");

        let hub = A::hub(self);
        let cmd_buf = hub
            .command_buffers
            .get(id.into_command_buffer_id())
            .unwrap();
        let mut cmd_buf_data = cmd_buf.data.lock();
        let cmd_buf_data = cmd_buf_data.as_mut().unwrap();
        let cmd_buf_raw = cmd_buf_data.encoder.open().ok();

        hal_command_encoder_callback(cmd_buf_raw)
    }
}

/// A texture that has been marked as destroyed and is staged for actual deletion soon.
#[derive(Debug)]
pub struct DestroyedTexture<A: HalApi> {
    raw: Option<A::Texture>,
    views: Vec<Weak<TextureView<A>>>,
    bind_groups: Vec<Weak<BindGroup<A>>>,
    device: Arc<Device<A>>,
    label: String,
    pub(crate) tracker_index: TrackerIndex,
    pub(crate) submission_index: u64,
}

impl<A: HalApi> DestroyedTexture<A> {
    pub fn label(&self) -> &dyn Debug {
        &self.label
    }
}

impl<A: HalApi> Drop for DestroyedTexture<A> {
    fn drop(&mut self) {
        let device = &self.device;

        let mut deferred = device.deferred_destroy.lock();
        for view in self.views.drain(..) {
            deferred.push(DeferredDestroy::TextureView(view));
        }
        for bind_group in self.bind_groups.drain(..) {
            deferred.push(DeferredDestroy::BindGroup(bind_group));
        }
        drop(deferred);

        if let Some(raw) = self.raw.take() {
            resource_log!("Destroy raw Texture (destroyed) {:?}", self.label());

            unsafe {
                use hal::Device;
                self.device.raw().destroy_texture(raw);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum TextureErrorDimension {
    X,
    Y,
    Z,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TextureDimensionError {
    #[error("Dimension {0:?} is zero")]
    Zero(TextureErrorDimension),
    #[error("Dimension {dim:?} value {given} exceeds the limit of {limit}")]
    LimitExceeded {
        dim: TextureErrorDimension,
        given: u32,
        limit: u32,
    },
    #[error("Sample count {0} is invalid")]
    InvalidSampleCount(u32),
    #[error("Width {width} is not a multiple of {format:?}'s block width ({block_width})")]
    NotMultipleOfBlockWidth {
        width: u32,
        block_width: u32,
        format: wgt::TextureFormat,
    },
    #[error("Height {height} is not a multiple of {format:?}'s block height ({block_height})")]
    NotMultipleOfBlockHeight {
        height: u32,
        block_height: u32,
        format: wgt::TextureFormat,
    },
    #[error(
        "Width {width} is not a multiple of {format:?}'s width multiple requirement ({multiple})"
    )]
    WidthNotMultipleOf {
        width: u32,
        multiple: u32,
        format: wgt::TextureFormat,
    },
    #[error("Height {height} is not a multiple of {format:?}'s height multiple requirement ({multiple})")]
    HeightNotMultipleOf {
        height: u32,
        multiple: u32,
        format: wgt::TextureFormat,
    },
    #[error("Multisampled texture depth or array layers must be 1, got {0}")]
    MultisampledDepthOrArrayLayer(u32),
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateTextureError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    CreateTextureView(#[from] CreateTextureViewError),
    #[error("Invalid usage flags {0:?}")]
    InvalidUsage(wgt::TextureUsages),
    #[error(transparent)]
    InvalidDimension(#[from] TextureDimensionError),
    #[error("Depth texture ({1:?}) can't be created as {0:?}")]
    InvalidDepthDimension(wgt::TextureDimension, wgt::TextureFormat),
    #[error("Compressed texture ({1:?}) can't be created as {0:?}")]
    InvalidCompressedDimension(wgt::TextureDimension, wgt::TextureFormat),
    #[error(
        "Texture descriptor mip level count {requested} is invalid, maximum allowed is {maximum}"
    )]
    InvalidMipLevelCount { requested: u32, maximum: u32 },
    #[error(
        "Texture usages {0:?} are not allowed on a texture of type {1:?}{}",
        if *.2 { " due to downlevel restrictions" } else { "" }
    )]
    InvalidFormatUsages(wgt::TextureUsages, wgt::TextureFormat, bool),
    #[error("The view format {0:?} is not compatible with texture format {1:?}, only changing srgb-ness is allowed.")]
    InvalidViewFormat(wgt::TextureFormat, wgt::TextureFormat),
    #[error("Texture usages {0:?} are not allowed on a texture of dimensions {1:?}")]
    InvalidDimensionUsages(wgt::TextureUsages, wgt::TextureDimension),
    #[error("Texture usage STORAGE_BINDING is not allowed for multisampled textures")]
    InvalidMultisampledStorageBinding,
    #[error("Format {0:?} does not support multisampling")]
    InvalidMultisampledFormat(wgt::TextureFormat),
    #[error("Sample count {0} is not supported by format {1:?} on this device. The WebGPU spec guarantees {2:?} samples are supported by this format. With the TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES feature your device supports {3:?}.")]
    InvalidSampleCount(u32, wgt::TextureFormat, Vec<u32>, Vec<u32>),
    #[error("Multisampled textures must have RENDER_ATTACHMENT usage")]
    MultisampledNotRenderAttachment,
    #[error("Texture format {0:?} can't be used due to missing features")]
    MissingFeatures(wgt::TextureFormat, #[source] MissingFeatures),
    #[error(transparent)]
    MissingDownlevelFlags(#[from] MissingDownlevelFlags),
}

impl<A: HalApi> Resource for Texture<A> {
    const TYPE: ResourceType = "Texture";

    type Marker = crate::id::markers::Texture;

    fn as_info(&self) -> &ResourceInfo<Self> {
        &self.info
    }

    fn as_info_mut(&mut self) -> &mut ResourceInfo<Self> {
        &mut self.info
    }
}

impl<A: HalApi> ParentDevice<A> for Texture<A> {
    fn device(&self) -> &Arc<Device<A>> {
        &self.device
    }
}

impl<A: HalApi> Borrow<TextureSelector> for Texture<A> {
    fn borrow(&self) -> &TextureSelector {
        &self.full_range
    }
}

/// Describes a [`TextureView`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct TextureViewDescriptor<'a> {
    /// Debug label of the texture view.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Format of the texture view, or `None` for the same format as the texture
    /// itself.
    ///
    /// At this time, it must be the same the underlying format of the texture.
    pub format: Option<wgt::TextureFormat>,
    /// The dimension of the texture view.
    ///
    /// - For 1D textures, this must be `D1`.
    /// - For 2D textures it must be one of `D2`, `D2Array`, `Cube`, or `CubeArray`.
    /// - For 3D textures it must be `D3`.
    pub dimension: Option<wgt::TextureViewDimension>,
    /// Range within the texture that is accessible via this view.
    pub range: wgt::ImageSubresourceRange,
}

#[derive(Debug)]
pub(crate) struct HalTextureViewDescriptor {
    pub texture_format: wgt::TextureFormat,
    pub format: wgt::TextureFormat,
    pub dimension: wgt::TextureViewDimension,
    pub range: wgt::ImageSubresourceRange,
}

impl HalTextureViewDescriptor {
    pub fn aspects(&self) -> hal::FormatAspects {
        hal::FormatAspects::new(self.texture_format, self.range.aspect)
    }
}

#[derive(Debug, Copy, Clone, Error)]
pub enum TextureViewNotRenderableReason {
    #[error("The texture this view references doesn't include the RENDER_ATTACHMENT usage. Provided usages: {0:?}")]
    Usage(wgt::TextureUsages),
    #[error("The dimension of this texture view is not 2D. View dimension: {0:?}")]
    Dimension(wgt::TextureViewDimension),
    #[error("This texture view has more than one mipmap level. View mipmap levels: {0:?}")]
    MipLevelCount(u32),
    #[error("This texture view has more than one array layer. View array layers: {0:?}")]
    ArrayLayerCount(u32),
    #[error(
        "The aspects of this texture view are a subset of the aspects in the original texture. Aspects: {0:?}"
    )]
    Aspects(hal::FormatAspects),
}

#[derive(Debug)]
pub struct TextureView<A: HalApi> {
    pub(crate) raw: Snatchable<A::TextureView>,
    // if it's a surface texture - it's none
    pub(crate) parent: Arc<Texture<A>>,
    pub(crate) device: Arc<Device<A>>,
    pub(crate) desc: HalTextureViewDescriptor,
    pub(crate) format_features: wgt::TextureFormatFeatures,
    /// This is `Err` only if the texture view is not renderable
    pub(crate) render_extent: Result<wgt::Extent3d, TextureViewNotRenderableReason>,
    pub(crate) samples: u32,
    pub(crate) selector: TextureSelector,
    pub(crate) info: ResourceInfo<TextureView<A>>,
}

impl<A: HalApi> Drop for TextureView<A> {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            resource_log!("Destroy raw {}", self.error_ident());
            unsafe {
                use hal::Device;
                self.device.raw().destroy_texture_view(raw);
            }
        }
    }
}

impl<A: HalApi> TextureView<A> {
    pub(crate) fn raw<'a>(&'a self, snatch_guard: &'a SnatchGuard) -> Option<&'a A::TextureView> {
        self.raw.get(snatch_guard)
    }

    pub(crate) fn try_raw<'a>(
        &'a self,
        guard: &'a SnatchGuard,
    ) -> Result<&A::TextureView, DestroyedResourceError> {
        self.raw
            .get(guard)
            .ok_or_else(|| DestroyedResourceError(self.error_ident()))
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateTextureViewError {
    #[error("TextureId {0:?} is invalid")]
    InvalidTextureId(TextureId),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error("Not enough memory left to create texture view")]
    OutOfMemory,
    #[error("Invalid texture view dimension `{view:?}` with texture of dimension `{texture:?}`")]
    InvalidTextureViewDimension {
        view: wgt::TextureViewDimension,
        texture: wgt::TextureDimension,
    },
    #[error("Invalid texture view dimension `{0:?}` of a multisampled texture")]
    InvalidMultisampledTextureViewDimension(wgt::TextureViewDimension),
    #[error("Invalid texture depth `{depth}` for texture view of dimension `Cubemap`. Cubemap views must use images of size 6.")]
    InvalidCubemapTextureDepth { depth: u32 },
    #[error("Invalid texture depth `{depth}` for texture view of dimension `CubemapArray`. Cubemap views must use images with sizes which are a multiple of 6.")]
    InvalidCubemapArrayTextureDepth { depth: u32 },
    #[error("Source texture width and height must be equal for a texture view of dimension `Cube`/`CubeArray`")]
    InvalidCubeTextureViewSize,
    #[error("Mip level count is 0")]
    ZeroMipLevelCount,
    #[error("Array layer count is 0")]
    ZeroArrayLayerCount,
    #[error(
        "TextureView mip level count + base mip level {requested} must be <= Texture mip level count {total}"
    )]
    TooManyMipLevels { requested: u32, total: u32 },
    #[error("TextureView array layer count + base array layer {requested} must be <= Texture depth/array layer count {total}")]
    TooManyArrayLayers { requested: u32, total: u32 },
    #[error("Requested array layer count {requested} is not valid for the target view dimension {dim:?}")]
    InvalidArrayLayerCount {
        requested: u32,
        dim: wgt::TextureViewDimension,
    },
    #[error("Aspect {requested_aspect:?} is not in the source texture format {texture_format:?}")]
    InvalidAspect {
        texture_format: wgt::TextureFormat,
        requested_aspect: wgt::TextureAspect,
    },
    #[error("Unable to view texture {texture:?} as {view:?}")]
    FormatReinterpretation {
        texture: wgt::TextureFormat,
        view: wgt::TextureFormat,
    },
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TextureViewDestroyError {}

impl<A: HalApi> Resource for TextureView<A> {
    const TYPE: ResourceType = "TextureView";

    type Marker = crate::id::markers::TextureView;

    fn as_info(&self) -> &ResourceInfo<Self> {
        &self.info
    }

    fn as_info_mut(&mut self) -> &mut ResourceInfo<Self> {
        &mut self.info
    }
}

impl<A: HalApi> ParentDevice<A> for TextureView<A> {
    fn device(&self) -> &Arc<Device<A>> {
        &self.device
    }
}

/// Describes a [`Sampler`]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SamplerDescriptor<'a> {
    /// Debug label of the sampler.
    ///
    /// This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// How to deal with out of bounds accesses in the u (i.e. x) direction
    pub address_modes: [wgt::AddressMode; 3],
    /// How to filter the texture when it needs to be magnified (made larger)
    pub mag_filter: wgt::FilterMode,
    /// How to filter the texture when it needs to be minified (made smaller)
    pub min_filter: wgt::FilterMode,
    /// How to filter between mip map levels
    pub mipmap_filter: wgt::FilterMode,
    /// Minimum level of detail (i.e. mip level) to use
    pub lod_min_clamp: f32,
    /// Maximum level of detail (i.e. mip level) to use
    pub lod_max_clamp: f32,
    /// If this is enabled, this is a comparison sampler using the given comparison function.
    pub compare: Option<wgt::CompareFunction>,
    /// Must be at least 1. If this is not 1, all filter modes must be linear.
    pub anisotropy_clamp: u16,
    /// Border color to use when address_mode is
    /// [`AddressMode::ClampToBorder`](wgt::AddressMode::ClampToBorder)
    pub border_color: Option<wgt::SamplerBorderColor>,
}

#[derive(Debug)]
pub struct Sampler<A: HalApi> {
    pub(crate) raw: Option<A::Sampler>,
    pub(crate) device: Arc<Device<A>>,
    pub(crate) info: ResourceInfo<Self>,
    /// `true` if this is a comparison sampler
    pub(crate) comparison: bool,
    /// `true` if this is a filtering sampler
    pub(crate) filtering: bool,
}

impl<A: HalApi> Drop for Sampler<A> {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            resource_log!("Destroy raw {}", self.error_ident());
            unsafe {
                use hal::Device;
                self.device.raw().destroy_sampler(raw);
            }
        }
    }
}

impl<A: HalApi> Sampler<A> {
    pub(crate) fn raw(&self) -> &A::Sampler {
        self.raw.as_ref().unwrap()
    }
}

#[derive(Copy, Clone)]
pub enum SamplerFilterErrorType {
    MagFilter,
    MinFilter,
    MipmapFilter,
}

impl Debug for SamplerFilterErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            SamplerFilterErrorType::MagFilter => write!(f, "magFilter"),
            SamplerFilterErrorType::MinFilter => write!(f, "minFilter"),
            SamplerFilterErrorType::MipmapFilter => write!(f, "mipmapFilter"),
        }
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateSamplerError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Invalid lodMinClamp: {0}. Must be greater or equal to 0.0")]
    InvalidLodMinClamp(f32),
    #[error("Invalid lodMaxClamp: {lod_max_clamp}. Must be greater or equal to lodMinClamp (which is {lod_min_clamp}).")]
    InvalidLodMaxClamp {
        lod_min_clamp: f32,
        lod_max_clamp: f32,
    },
    #[error("Invalid anisotropic clamp: {0}. Must be at least 1.")]
    InvalidAnisotropy(u16),
    #[error("Invalid filter mode for {filter_type:?}: {filter_mode:?}. When anistropic clamp is not 1 (it is {anisotropic_clamp}), all filter modes must be linear.")]
    InvalidFilterModeWithAnisotropy {
        filter_type: SamplerFilterErrorType,
        filter_mode: wgt::FilterMode,
        anisotropic_clamp: u16,
    },
    #[error("Cannot create any more samplers")]
    TooManyObjects,
    /// AddressMode::ClampToBorder requires feature ADDRESS_MODE_CLAMP_TO_BORDER.
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
}

impl<A: HalApi> Resource for Sampler<A> {
    const TYPE: ResourceType = "Sampler";

    type Marker = crate::id::markers::Sampler;

    fn as_info(&self) -> &ResourceInfo<Self> {
        &self.info
    }

    fn as_info_mut(&mut self) -> &mut ResourceInfo<Self> {
        &mut self.info
    }
}

impl<A: HalApi> ParentDevice<A> for Sampler<A> {
    fn device(&self) -> &Arc<Device<A>> {
        &self.device
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateQuerySetError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("QuerySets cannot be made with zero queries")]
    ZeroCount,
    #[error("{count} is too many queries for a single QuerySet. QuerySets cannot be made more than {maximum} queries.")]
    TooManyQueries { count: u32, maximum: u32 },
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
}

pub type QuerySetDescriptor<'a> = wgt::QuerySetDescriptor<Label<'a>>;

#[derive(Debug)]
pub struct QuerySet<A: HalApi> {
    pub(crate) raw: Option<A::QuerySet>,
    pub(crate) device: Arc<Device<A>>,
    pub(crate) info: ResourceInfo<Self>,
    pub(crate) desc: wgt::QuerySetDescriptor<()>,
}

impl<A: HalApi> Drop for QuerySet<A> {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            resource_log!("Destroy raw {}", self.error_ident());
            unsafe {
                use hal::Device;
                self.device.raw().destroy_query_set(raw);
            }
        }
    }
}

impl<A: HalApi> ParentDevice<A> for QuerySet<A> {
    fn device(&self) -> &Arc<Device<A>> {
        &self.device
    }
}

impl<A: HalApi> Resource for QuerySet<A> {
    const TYPE: ResourceType = "QuerySet";

    type Marker = crate::id::markers::QuerySet;

    fn as_info(&self) -> &ResourceInfo<Self> {
        &self.info
    }

    fn as_info_mut(&mut self) -> &mut ResourceInfo<Self> {
        &mut self.info
    }
}

impl<A: HalApi> QuerySet<A> {
    pub(crate) fn raw(&self) -> &A::QuerySet {
        self.raw.as_ref().unwrap()
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum DestroyError {
    #[error("Resource is invalid")]
    Invalid,
    #[error("Resource is already destroyed")]
    AlreadyDestroyed,
}
