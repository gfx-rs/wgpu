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
    id::{AdapterId, BufferId, CommandEncoderId, DeviceId, SurfaceId, TextureId, TextureViewId},
    init_tracker::{BufferInitTracker, TextureInitTracker},
    lock::{rank, Mutex, RwLock},
    resource_log,
    snatch::{SnatchGuard, Snatchable},
    track::{SharedTrackerIndexAllocator, TextureSelector, TrackerIndex},
    weak_vec::WeakVec,
    Label, LabelHelpers, SubmissionIndex,
};

use smallvec::SmallVec;
use thiserror::Error;

use std::num::NonZeroU64;
use std::{
    borrow::{Borrow, Cow},
    fmt::Debug,
    mem::{self, ManuallyDrop},
    ops::Range,
    ptr::NonNull,
    sync::Arc,
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
pub(crate) struct TrackingData {
    tracker_index: TrackerIndex,
    tracker_indices: Arc<SharedTrackerIndexAllocator>,
}

impl Drop for TrackingData {
    fn drop(&mut self) {
        self.tracker_indices.free(self.tracker_index);
    }
}

impl TrackingData {
    pub(crate) fn new(tracker_indices: Arc<SharedTrackerIndexAllocator>) -> Self {
        Self {
            tracker_index: tracker_indices.alloc(),
            tracker_indices,
        }
    }

    pub(crate) fn tracker_index(&self) -> TrackerIndex {
        self.tracker_index
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResourceErrorIdent {
    r#type: Cow<'static, str>,
    label: String,
}

impl std::fmt::Display for ResourceErrorIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{} with '{}' label", self.r#type, self.label)
    }
}

pub trait ParentDevice: Labeled {
    fn device(&self) -> &Arc<Device>;

    fn is_equal(self: &Arc<Self>, other: &Arc<Self>) -> bool {
        Arc::ptr_eq(self, other)
    }

    fn same_device_as<O: ParentDevice>(&self, other: &O) -> Result<(), DeviceError> {
        if Arc::ptr_eq(self.device(), other.device()) {
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

    fn same_device(&self, device: &Device) -> Result<(), DeviceError> {
        if std::ptr::eq(&**self.device(), device) {
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

#[macro_export]
macro_rules! impl_parent_device {
    ($ty:ident) => {
        impl $crate::resource::ParentDevice for $ty {
            fn device(&self) -> &Arc<Device> {
                &self.device
            }
        }
    };
}

pub trait ResourceType {
    const TYPE: &'static str;
}

#[macro_export]
macro_rules! impl_resource_type {
    ($ty:ident) => {
        impl $crate::resource::ResourceType for $ty {
            const TYPE: &'static str = stringify!($ty);
        }
    };
}

pub trait Labeled: ResourceType {
    /// Returns a string identifying this resource for logging and errors.
    ///
    /// It may be a user-provided string or it may be a placeholder from wgpu.
    ///
    /// It is non-empty unless the user-provided string was empty.
    fn label(&self) -> &str;

    fn error_ident(&self) -> ResourceErrorIdent {
        ResourceErrorIdent {
            r#type: Cow::Borrowed(Self::TYPE),
            label: self.label().to_owned(),
        }
    }
}

#[macro_export]
macro_rules! impl_labeled {
    ($ty:ident) => {
        impl $crate::resource::Labeled for $ty {
            fn label(&self) -> &str {
                &self.label
            }
        }
    };
}

pub(crate) trait Trackable {
    fn tracker_index(&self) -> TrackerIndex;
}

#[macro_export]
macro_rules! impl_trackable {
    ($ty:ident) => {
        impl $crate::resource::Trackable for $ty {
            fn tracker_index(&self) -> $crate::track::TrackerIndex {
                self.tracking_data.tracker_index()
            }
        }
    };
}

#[derive(Debug)]
pub(crate) enum BufferMapState {
    /// Mapped at creation.
    Init { staging_buffer: StagingBuffer },
    /// Waiting for GPU to be done before mapping
    Waiting(BufferPendingMapping),
    /// Mapped
    Active {
        mapping: hal::BufferMapping,
        range: hal::MemoryRange,
        host: HostMap,
    },
    /// Not mapped
    Idle,
}

#[cfg(send_sync)]
unsafe impl Send for BufferMapState {}
#[cfg(send_sync)]
unsafe impl Sync for BufferMapState {}

#[cfg(send_sync)]
pub type BufferMapCallback = Box<dyn FnOnce(BufferAccessResult) + Send + 'static>;
#[cfg(not(send_sync))]
pub type BufferMapCallback = Box<dyn FnOnce(BufferAccessResult) + 'static>;

pub struct BufferMapOperation {
    pub host: HostMap,
    pub callback: Option<BufferMapCallback>,
}

impl Debug for BufferMapOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferMapOperation")
            .field("host", &self.host)
            .field("callback", &self.callback.as_ref().map(|_| "?"))
            .finish()
    }
}

#[derive(Clone, Debug, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum BufferAccessError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Buffer map failed")]
    Failed,
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
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
}

#[derive(Clone, Debug, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[error("{0} has been destroyed")]
pub struct DestroyedResourceError(pub ResourceErrorIdent);

#[derive(Clone, Debug, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[error("{0} is invalid")]
pub struct InvalidResourceError(pub ResourceErrorIdent);

pub enum Fallible<T: ParentDevice> {
    Valid(Arc<T>),
    Invalid(Arc<String>),
}

impl<T: ParentDevice> Fallible<T> {
    pub fn get(self) -> Result<Arc<T>, InvalidResourceError> {
        match self {
            Fallible::Valid(v) => Ok(v),
            Fallible::Invalid(label) => Err(InvalidResourceError(ResourceErrorIdent {
                r#type: Cow::Borrowed(T::TYPE),
                label: (*label).clone(),
            })),
        }
    }
}

impl<T: ParentDevice> Clone for Fallible<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Valid(v) => Self::Valid(v.clone()),
            Self::Invalid(l) => Self::Invalid(l.clone()),
        }
    }
}

impl<T: ParentDevice> ResourceType for Fallible<T> {
    const TYPE: &'static str = T::TYPE;
}

impl<T: ParentDevice + crate::storage::StorageItem> crate::storage::StorageItem for Fallible<T> {
    type Marker = T::Marker;
}

pub type BufferAccessResult = Result<(), BufferAccessError>;

#[derive(Debug)]
pub(crate) struct BufferPendingMapping {
    pub(crate) range: Range<wgt::BufferAddress>,
    pub(crate) op: BufferMapOperation,
    // hold the parent alive while the mapping is active
    pub(crate) _parent_buffer: Arc<Buffer>,
}

pub type BufferDescriptor<'a> = wgt::BufferDescriptor<Label<'a>>;

#[derive(Debug)]
pub struct Buffer {
    pub(crate) raw: Snatchable<Box<dyn hal::DynBuffer>>,
    pub(crate) device: Arc<Device>,
    pub(crate) usage: wgt::BufferUsages,
    pub(crate) size: wgt::BufferAddress,
    pub(crate) initialization_status: RwLock<BufferInitTracker>,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
    pub(crate) tracking_data: TrackingData,
    pub(crate) map_state: Mutex<BufferMapState>,
    pub(crate) bind_groups: Mutex<WeakVec<BindGroup>>,
    #[cfg(feature = "indirect-validation")]
    pub(crate) raw_indirect_validation_bind_group: Snatchable<Box<dyn hal::DynBindGroup>>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        #[cfg(feature = "indirect-validation")]
        if let Some(raw) = self.raw_indirect_validation_bind_group.take() {
            unsafe {
                self.device.raw().destroy_bind_group(raw);
            }
        }
        if let Some(raw) = self.raw.take() {
            resource_log!("Destroy raw {}", self.error_ident());
            unsafe {
                self.device.raw().destroy_buffer(raw);
            }
        }
    }
}

impl Buffer {
    pub(crate) fn raw<'a>(&'a self, guard: &'a SnatchGuard) -> Option<&'a dyn hal::DynBuffer> {
        self.raw.get(guard).map(|b| b.as_ref())
    }

    pub(crate) fn try_raw<'a>(
        &'a self,
        guard: &'a SnatchGuard,
    ) -> Result<&'a dyn hal::DynBuffer, DestroyedResourceError> {
        self.raw
            .get(guard)
            .map(|raw| raw.as_ref())
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
        &self,
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
    ) -> Result<SubmissionIndex, (BufferMapOperation, BufferAccessError)> {
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

        // TODO: we are ignoring the transition here, I think we need to add a barrier
        // at the end of the submission
        device
            .trackers
            .lock()
            .buffers
            .set_single(self, internal_use);

        let submit_index = if let Some(queue) = device.get_queue() {
            queue.lock_life().map(self).unwrap_or(0) // '0' means no wait is necessary
        } else {
            // We can safely unwrap below since we just set the `map_state` to `BufferMapState::Waiting`.
            let (mut operation, status) = self.map(&device.snatchable_lock.read()).unwrap();
            if let Some(callback) = operation.callback.take() {
                callback(status);
            }
            0
        };

        Ok(submit_index)
    }

    /// This function returns [`None`] only if [`Self::map_state`] is not [`BufferMapState::Waiting`].
    #[must_use]
    pub(crate) fn map(&self, snatch_guard: &SnatchGuard) -> Option<BufferMapPendingClosure> {
        // This _cannot_ be inlined into the match. If it is, the lock will be held
        // open through the whole match, resulting in a deadlock when we try to re-lock
        // the buffer back to active.
        let mapping = mem::replace(&mut *self.map_state.lock(), BufferMapState::Idle);
        let pending_mapping = match mapping {
            BufferMapState::Waiting(pending_mapping) => pending_mapping,
            // Mapping cancelled
            BufferMapState::Idle => return None,
            // Mapping queued at least twice by map -> unmap -> map
            // and was already successfully mapped below
            BufferMapState::Active { .. } => {
                *self.map_state.lock() = mapping;
                return None;
            }
            _ => panic!("No pending mapping."),
        };
        let status = if pending_mapping.range.start != pending_mapping.range.end {
            let host = pending_mapping.op.host;
            let size = pending_mapping.range.end - pending_mapping.range.start;
            match crate::device::map_buffer(
                self,
                pending_mapping.range.start,
                size,
                host,
                snatch_guard,
            ) {
                Ok(mapping) => {
                    *self.map_state.lock() = BufferMapState::Active {
                        mapping,
                        range: pending_mapping.range.clone(),
                        host,
                    };
                    Ok(())
                }
                Err(e) => {
                    log::error!("Mapping failed: {e}");
                    Err(e)
                }
            }
        } else {
            *self.map_state.lock() = BufferMapState::Active {
                mapping: hal::BufferMapping {
                    ptr: NonNull::dangling(),
                    is_coherent: true,
                },
                range: pending_mapping.range,
                host: pending_mapping.op.host,
            };
            Ok(())
        };
        Some((pending_mapping.op, status))
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
                callback(status);
            }
        }

        Ok(())
    }

    fn unmap_inner(
        self: &Arc<Self>,
        #[cfg(feature = "trace")] buffer_id: BufferId,
    ) -> Result<Option<BufferMapPendingClosure>, BufferAccessError> {
        let device = &self.device;
        let snatch_guard = device.snatchable_lock.read();
        let raw_buf = self.try_raw(&snatch_guard)?;
        match mem::replace(&mut *self.map_state.lock(), BufferMapState::Idle) {
            BufferMapState::Init { staging_buffer } => {
                #[cfg(feature = "trace")]
                if let Some(ref mut trace) = *device.trace.lock() {
                    let data = trace.make_binary("bin", staging_buffer.get_data());
                    trace.add(trace::Action::WriteBuffer {
                        id: buffer_id,
                        data,
                        range: 0..self.size,
                        queued: true,
                    });
                }

                let staging_buffer = staging_buffer.flush();

                if let Some(queue) = device.get_queue() {
                    let region = wgt::BufferSize::new(self.size).map(|size| hal::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size,
                    });
                    let transition_src = hal::BufferBarrier {
                        buffer: staging_buffer.raw(),
                        usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
                    };
                    let transition_dst = hal::BufferBarrier::<dyn hal::DynBuffer> {
                        buffer: raw_buf,
                        usage: hal::BufferUses::empty()..hal::BufferUses::COPY_DST,
                    };
                    let mut pending_writes = queue.pending_writes.lock();
                    let encoder = pending_writes.activate();
                    unsafe {
                        encoder.transition_buffers(&[transition_src, transition_dst]);
                        if self.size > 0 {
                            encoder.copy_buffer_to_buffer(
                                staging_buffer.raw(),
                                raw_buf,
                                region.as_slice(),
                            );
                        }
                    }
                    pending_writes.consume(staging_buffer);
                    pending_writes.insert_buffer(self);
                }
            }
            BufferMapState::Idle => {
                return Err(BufferAccessError::NotMapped);
            }
            BufferMapState::Waiting(pending) => {
                return Ok(Some((pending.op, Err(BufferAccessError::MapAborted))));
            }
            BufferMapState::Active {
                mapping,
                range,
                host,
            } => {
                #[allow(clippy::collapsible_if)]
                if host == HostMap::Write {
                    #[cfg(feature = "trace")]
                    if let Some(ref mut trace) = *device.trace.lock() {
                        let size = range.end - range.start;
                        let data = trace.make_binary("bin", unsafe {
                            std::slice::from_raw_parts(mapping.ptr.as_ptr(), size as usize)
                        });
                        trace.add(trace::Action::WriteBuffer {
                            id: buffer_id,
                            data,
                            range: range.clone(),
                            queued: false,
                        });
                    }
                    if !mapping.is_coherent {
                        unsafe { device.raw().flush_mapped_ranges(raw_buf, &[range]) };
                    }
                }
                unsafe { device.raw().unmap_buffer(raw_buf) };
            }
        }
        Ok(None)
    }

    pub(crate) fn destroy(self: &Arc<Self>) -> Result<(), DestroyError> {
        let device = &self.device;

        let temp = {
            let mut snatch_guard = device.snatchable_lock.write();

            let raw = match self.raw.snatch(&mut snatch_guard) {
                Some(raw) => raw,
                None => {
                    return Err(DestroyError::AlreadyDestroyed);
                }
            };

            #[cfg(feature = "indirect-validation")]
            let raw_indirect_validation_bind_group = self
                .raw_indirect_validation_bind_group
                .snatch(&mut snatch_guard);

            drop(snatch_guard);

            let bind_groups = {
                let mut guard = self.bind_groups.lock();
                mem::take(&mut *guard)
            };

            queue::TempResource::DestroyedBuffer(DestroyedBuffer {
                raw: ManuallyDrop::new(raw),
                device: Arc::clone(&self.device),
                label: self.label().to_owned(),
                bind_groups,
                #[cfg(feature = "indirect-validation")]
                raw_indirect_validation_bind_group,
            })
        };

        if let Some(queue) = device.get_queue() {
            let mut pending_writes = queue.pending_writes.lock();
            if pending_writes.contains_buffer(self) {
                pending_writes.consume_temp(temp);
            } else {
                let mut life_lock = queue.lock_life();
                let last_submit_index = life_lock.get_buffer_latest_submission_index(self);
                if let Some(last_submit_index) = last_submit_index {
                    life_lock.schedule_resource_destruction(temp, last_submit_index);
                }
            }
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
    #[error("Failed to create bind group for indirect buffer validation: {0}")]
    IndirectValidationBindGroup(DeviceError),
}

crate::impl_resource_type!(Buffer);
crate::impl_labeled!(Buffer);
crate::impl_parent_device!(Buffer);
crate::impl_storage_item!(Buffer);
crate::impl_trackable!(Buffer);

/// A buffer that has been marked as destroyed and is staged for actual deletion soon.
#[derive(Debug)]
pub struct DestroyedBuffer {
    raw: ManuallyDrop<Box<dyn hal::DynBuffer>>,
    device: Arc<Device>,
    label: String,
    bind_groups: WeakVec<BindGroup>,
    #[cfg(feature = "indirect-validation")]
    raw_indirect_validation_bind_group: Option<Box<dyn hal::DynBindGroup>>,
}

impl DestroyedBuffer {
    pub fn label(&self) -> &dyn Debug {
        &self.label
    }
}

impl Drop for DestroyedBuffer {
    fn drop(&mut self) {
        let mut deferred = self.device.deferred_destroy.lock();
        deferred.push(DeferredDestroy::BindGroups(mem::take(
            &mut self.bind_groups,
        )));
        drop(deferred);

        #[cfg(feature = "indirect-validation")]
        if let Some(raw) = self.raw_indirect_validation_bind_group.take() {
            unsafe {
                self.device.raw().destroy_bind_group(raw);
            }
        }

        resource_log!("Destroy raw Buffer (destroyed) {:?}", self.label());
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            hal::DynDevice::destroy_buffer(self.device.raw(), raw);
        }
    }
}

#[cfg(send_sync)]
unsafe impl Send for StagingBuffer {}
#[cfg(send_sync)]
unsafe impl Sync for StagingBuffer {}

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
pub struct StagingBuffer {
    raw: Box<dyn hal::DynBuffer>,
    device: Arc<Device>,
    pub(crate) size: wgt::BufferSize,
    is_coherent: bool,
    ptr: NonNull<u8>,
}

impl StagingBuffer {
    pub(crate) fn new(device: &Arc<Device>, size: wgt::BufferSize) -> Result<Self, DeviceError> {
        profiling::scope!("StagingBuffer::new");
        let stage_desc = hal::BufferDescriptor {
            label: crate::hal_label(Some("(wgpu internal) Staging"), device.instance_flags),
            size: size.get(),
            usage: hal::BufferUses::MAP_WRITE | hal::BufferUses::COPY_SRC,
            memory_flags: hal::MemoryFlags::TRANSIENT,
        };

        let raw = unsafe { device.raw().create_buffer(&stage_desc) }
            .map_err(|e| device.handle_hal_error(e))?;
        let mapping = unsafe { device.raw().map_buffer(raw.as_ref(), 0..size.get()) }
            .map_err(|e| device.handle_hal_error(e))?;

        let staging_buffer = StagingBuffer {
            raw,
            device: device.clone(),
            size,
            is_coherent: mapping.is_coherent,
            ptr: mapping.ptr,
        };

        Ok(staging_buffer)
    }

    /// SAFETY: You must not call any functions of `self`
    /// until you stopped using the returned pointer.
    pub(crate) unsafe fn ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    #[cfg(feature = "trace")]
    pub(crate) fn get_data(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size.get() as usize) }
    }

    pub(crate) fn write_zeros(&mut self) {
        unsafe { core::ptr::write_bytes(self.ptr.as_ptr(), 0, self.size.get() as usize) };
    }

    pub(crate) fn write(&mut self, data: &[u8]) {
        assert!(data.len() >= self.size.get() as usize);
        // SAFETY: With the assert above, all of `copy_nonoverlapping`'s
        // requirements are satisfied.
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.ptr.as_ptr(),
                self.size.get() as usize,
            );
        }
    }

    /// SAFETY: The offsets and size must be in-bounds.
    pub(crate) unsafe fn write_with_offset(
        &mut self,
        data: &[u8],
        src_offset: isize,
        dst_offset: isize,
        size: usize,
    ) {
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr().offset(src_offset),
                self.ptr.as_ptr().offset(dst_offset),
                size,
            );
        }
    }

    pub(crate) fn flush(self) -> FlushedStagingBuffer {
        let device = self.device.raw();
        if !self.is_coherent {
            #[allow(clippy::single_range_in_vec_init)]
            unsafe {
                device.flush_mapped_ranges(self.raw.as_ref(), &[0..self.size.get()])
            };
        }
        unsafe { device.unmap_buffer(self.raw.as_ref()) };

        let StagingBuffer {
            raw, device, size, ..
        } = self;

        FlushedStagingBuffer {
            raw: ManuallyDrop::new(raw),
            device,
            size,
        }
    }
}

crate::impl_resource_type!(StagingBuffer);
crate::impl_storage_item!(StagingBuffer);

#[derive(Debug)]
pub struct FlushedStagingBuffer {
    raw: ManuallyDrop<Box<dyn hal::DynBuffer>>,
    device: Arc<Device>,
    pub(crate) size: wgt::BufferSize,
}

impl FlushedStagingBuffer {
    pub(crate) fn raw(&self) -> &dyn hal::DynBuffer {
        self.raw.as_ref()
    }
}

impl Drop for FlushedStagingBuffer {
    fn drop(&mut self) {
        resource_log!("Destroy raw StagingBuffer");
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe { self.device.raw().destroy_buffer(raw) };
    }
}

pub type TextureDescriptor<'a> = wgt::TextureDescriptor<Label<'a>, Vec<wgt::TextureFormat>>;

#[derive(Debug)]
pub(crate) enum TextureInner {
    Native {
        raw: Box<dyn hal::DynTexture>,
    },
    Surface {
        raw: Box<dyn hal::DynSurfaceTexture>,
    },
}

impl TextureInner {
    pub(crate) fn raw(&self) -> &dyn hal::DynTexture {
        match self {
            Self::Native { raw } => raw.as_ref(),
            Self::Surface { raw, .. } => raw.as_ref().borrow(),
        }
    }
}

#[derive(Debug)]
pub enum TextureClearMode {
    BufferCopy,
    // View for clear via RenderPass for every subsurface (mip/layer/slice)
    RenderPass {
        clear_views: SmallVec<[ManuallyDrop<Box<dyn hal::DynTextureView>>; 1]>,
        is_color: bool,
    },
    Surface {
        clear_view: ManuallyDrop<Box<dyn hal::DynTextureView>>,
    },
    // Texture can't be cleared, attempting to do so will cause panic.
    // (either because it is impossible for the type of texture or it is being destroyed)
    None,
}

#[derive(Debug)]
pub struct Texture {
    pub(crate) inner: Snatchable<TextureInner>,
    pub(crate) device: Arc<Device>,
    pub(crate) desc: wgt::TextureDescriptor<(), Vec<wgt::TextureFormat>>,
    pub(crate) hal_usage: hal::TextureUses,
    pub(crate) format_features: wgt::TextureFormatFeatures,
    pub(crate) initialization_status: RwLock<TextureInitTracker>,
    pub(crate) full_range: TextureSelector,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
    pub(crate) tracking_data: TrackingData,
    pub(crate) clear_mode: TextureClearMode,
    pub(crate) views: Mutex<WeakVec<TextureView>>,
    pub(crate) bind_groups: Mutex<WeakVec<BindGroup>>,
}

impl Texture {
    pub(crate) fn new(
        device: &Arc<Device>,
        inner: TextureInner,
        hal_usage: hal::TextureUses,
        desc: &TextureDescriptor,
        format_features: wgt::TextureFormatFeatures,
        clear_mode: TextureClearMode,
        init: bool,
    ) -> Self {
        Texture {
            inner: Snatchable::new(inner),
            device: device.clone(),
            desc: desc.map_label(|_| ()),
            hal_usage,
            format_features,
            initialization_status: RwLock::new(
                rank::TEXTURE_INITIALIZATION_STATUS,
                if init {
                    TextureInitTracker::new(desc.mip_level_count, desc.array_layer_count())
                } else {
                    TextureInitTracker::new(desc.mip_level_count, 0)
                },
            ),
            full_range: TextureSelector {
                mips: 0..desc.mip_level_count,
                layers: 0..desc.array_layer_count(),
            },
            label: desc.label.to_string(),
            tracking_data: TrackingData::new(device.tracker_indices.textures.clone()),
            clear_mode,
            views: Mutex::new(rank::TEXTURE_VIEWS, WeakVec::new()),
            bind_groups: Mutex::new(rank::TEXTURE_BIND_GROUPS, WeakVec::new()),
        }
    }
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

impl Drop for Texture {
    fn drop(&mut self) {
        match self.clear_mode {
            TextureClearMode::Surface {
                ref mut clear_view, ..
            } => {
                // SAFETY: We are in the Drop impl and we don't use clear_view anymore after this point.
                let raw = unsafe { ManuallyDrop::take(clear_view) };
                unsafe {
                    self.device.raw().destroy_texture_view(raw);
                }
            }
            TextureClearMode::RenderPass {
                ref mut clear_views,
                ..
            } => {
                clear_views.iter_mut().for_each(|clear_view| {
                    // SAFETY: We are in the Drop impl and we don't use clear_view anymore after this point.
                    let raw = unsafe { ManuallyDrop::take(clear_view) };
                    unsafe {
                        self.device.raw().destroy_texture_view(raw);
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

impl Texture {
    pub(crate) fn try_inner<'a>(
        &'a self,
        guard: &'a SnatchGuard,
    ) -> Result<&'a TextureInner, DestroyedResourceError> {
        self.inner
            .get(guard)
            .ok_or_else(|| DestroyedResourceError(self.error_ident()))
    }

    pub(crate) fn raw<'a>(
        &'a self,
        snatch_guard: &'a SnatchGuard,
    ) -> Option<&'a dyn hal::DynTexture> {
        Some(self.inner.get(snatch_guard)?.raw())
    }

    pub(crate) fn try_raw<'a>(
        &'a self,
        guard: &'a SnatchGuard,
    ) -> Result<&'a dyn hal::DynTexture, DestroyedResourceError> {
        self.inner
            .get(guard)
            .map(|t| t.raw())
            .ok_or_else(|| DestroyedResourceError(self.error_ident()))
    }

    pub(crate) fn get_clear_view<'a>(
        clear_mode: &'a TextureClearMode,
        desc: &'a wgt::TextureDescriptor<(), Vec<wgt::TextureFormat>>,
        mip_level: u32,
        depth_or_layer: u32,
    ) -> &'a dyn hal::DynTextureView {
        match *clear_mode {
            TextureClearMode::BufferCopy => {
                panic!("Given texture is cleared with buffer copies, not render passes")
            }
            TextureClearMode::None => {
                panic!("Given texture can't be cleared")
            }
            TextureClearMode::Surface { ref clear_view, .. } => clear_view.as_ref(),
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
                clear_views[index as usize].as_ref()
            }
        }
    }

    pub(crate) fn destroy(self: &Arc<Self>) -> Result<(), DestroyError> {
        let device = &self.device;

        let temp = {
            let raw = match self.inner.snatch(&mut device.snatchable_lock.write()) {
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

            queue::TempResource::DestroyedTexture(DestroyedTexture {
                raw: ManuallyDrop::new(raw),
                views,
                bind_groups,
                device: Arc::clone(&self.device),
                label: self.label().to_owned(),
            })
        };

        if let Some(queue) = device.get_queue() {
            let mut pending_writes = queue.pending_writes.lock();
            if pending_writes.contains_texture(self) {
                pending_writes.consume_temp(temp);
            } else {
                let mut life_lock = queue.lock_life();
                let last_submit_index = life_lock.get_texture_latest_submission_index(self);
                if let Some(last_submit_index) = last_submit_index {
                    life_lock.schedule_resource_destruction(temp, last_submit_index);
                }
            }
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

        let hub = &self.hub;

        if let Ok(buffer) = hub.buffers.get(id).get() {
            let snatch_guard = buffer.device.snatchable_lock.read();
            let hal_buffer = buffer
                .raw(&snatch_guard)
                .and_then(|b| b.as_any().downcast_ref());
            hal_buffer_callback(hal_buffer)
        } else {
            hal_buffer_callback(None)
        }
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

        let hub = &self.hub;

        if let Ok(texture) = hub.textures.get(id).get() {
            let snatch_guard = texture.device.snatchable_lock.read();
            let hal_texture = texture.raw(&snatch_guard);
            let hal_texture = hal_texture
                .as_ref()
                .and_then(|it| it.as_any().downcast_ref());
            hal_texture_callback(hal_texture)
        } else {
            hal_texture_callback(None)
        }
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

        let hub = &self.hub;

        if let Ok(texture_view) = hub.texture_views.get(id).get() {
            let snatch_guard = texture_view.device.snatchable_lock.read();
            let hal_texture_view = texture_view.raw(&snatch_guard);
            let hal_texture_view = hal_texture_view
                .as_ref()
                .and_then(|it| it.as_any().downcast_ref());
            hal_texture_view_callback(hal_texture_view)
        } else {
            hal_texture_view_callback(None)
        }
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

        let hub = &self.hub;
        let adapter = hub.adapters.get(id);
        let hal_adapter = adapter.raw.adapter.as_any().downcast_ref();

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

        let device = self.hub.devices.get(id);
        let hal_device = device.raw().as_any().downcast_ref();

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

        let device = self.hub.devices.get(id);
        let fence = device.fence.read();
        hal_fence_callback(fence.as_any().downcast_ref())
    }

    /// # Safety
    /// - The raw surface handle must not be manually destroyed
    pub unsafe fn surface_as_hal<A: HalApi, F: FnOnce(Option<&A::Surface>) -> R, R>(
        &self,
        id: SurfaceId,
        hal_surface_callback: F,
    ) -> R {
        profiling::scope!("Surface::as_hal");

        let surface = self.surfaces.get(id);
        let hal_surface = surface
            .raw(A::VARIANT)
            .and_then(|surface| surface.as_any().downcast_ref());

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

        let hub = &self.hub;

        let cmd_buf = hub.command_buffers.get(id.into_command_buffer_id());
        let cmd_buf_data = cmd_buf.try_get();

        if let Ok(mut cmd_buf_data) = cmd_buf_data {
            let cmd_buf_raw = cmd_buf_data
                .encoder
                .open(&cmd_buf.device)
                .ok()
                .and_then(|encoder| encoder.as_any_mut().downcast_mut());
            hal_command_encoder_callback(cmd_buf_raw)
        } else {
            hal_command_encoder_callback(None)
        }
    }
}

/// A texture that has been marked as destroyed and is staged for actual deletion soon.
#[derive(Debug)]
pub struct DestroyedTexture {
    raw: ManuallyDrop<Box<dyn hal::DynTexture>>,
    views: WeakVec<TextureView>,
    bind_groups: WeakVec<BindGroup>,
    device: Arc<Device>,
    label: String,
}

impl DestroyedTexture {
    pub fn label(&self) -> &dyn Debug {
        &self.label
    }
}

impl Drop for DestroyedTexture {
    fn drop(&mut self) {
        let device = &self.device;

        let mut deferred = device.deferred_destroy.lock();
        deferred.push(DeferredDestroy::TextureViews(mem::take(&mut self.views)));
        deferred.push(DeferredDestroy::BindGroups(mem::take(
            &mut self.bind_groups,
        )));
        drop(deferred);

        resource_log!("Destroy raw Texture (destroyed) {:?}", self.label());
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            self.device.raw().destroy_texture(raw);
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

crate::impl_resource_type!(Texture);
crate::impl_labeled!(Texture);
crate::impl_parent_device!(Texture);
crate::impl_storage_item!(Texture);
crate::impl_trackable!(Texture);

impl Borrow<TextureSelector> for Texture {
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
pub struct TextureView {
    pub(crate) raw: Snatchable<Box<dyn hal::DynTextureView>>,
    // if it's a surface texture - it's none
    pub(crate) parent: Arc<Texture>,
    pub(crate) device: Arc<Device>,
    pub(crate) desc: HalTextureViewDescriptor,
    pub(crate) format_features: wgt::TextureFormatFeatures,
    /// This is `Err` only if the texture view is not renderable
    pub(crate) render_extent: Result<wgt::Extent3d, TextureViewNotRenderableReason>,
    pub(crate) samples: u32,
    pub(crate) selector: TextureSelector,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
    pub(crate) tracking_data: TrackingData,
}

impl Drop for TextureView {
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            resource_log!("Destroy raw {}", self.error_ident());
            unsafe {
                self.device.raw().destroy_texture_view(raw);
            }
        }
    }
}

impl TextureView {
    pub(crate) fn raw<'a>(
        &'a self,
        snatch_guard: &'a SnatchGuard,
    ) -> Option<&'a dyn hal::DynTextureView> {
        self.raw.get(snatch_guard).map(|it| it.as_ref())
    }

    pub(crate) fn try_raw<'a>(
        &'a self,
        guard: &'a SnatchGuard,
    ) -> Result<&'a dyn hal::DynTextureView, DestroyedResourceError> {
        self.raw
            .get(guard)
            .map(|it| it.as_ref())
            .ok_or_else(|| DestroyedResourceError(self.error_ident()))
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CreateTextureViewError {
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
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
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TextureViewDestroyError {}

crate::impl_resource_type!(TextureView);
crate::impl_labeled!(TextureView);
crate::impl_parent_device!(TextureView);
crate::impl_storage_item!(TextureView);
crate::impl_trackable!(TextureView);

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
pub struct Sampler {
    pub(crate) raw: ManuallyDrop<Box<dyn hal::DynSampler>>,
    pub(crate) device: Arc<Device>,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
    pub(crate) tracking_data: TrackingData,
    /// `true` if this is a comparison sampler
    pub(crate) comparison: bool,
    /// `true` if this is a filtering sampler
    pub(crate) filtering: bool,
}

impl Drop for Sampler {
    fn drop(&mut self) {
        resource_log!("Destroy raw {}", self.error_ident());
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            self.device.raw().destroy_sampler(raw);
        }
    }
}

impl Sampler {
    pub(crate) fn raw(&self) -> &dyn hal::DynSampler {
        self.raw.as_ref()
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
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
}

crate::impl_resource_type!(Sampler);
crate::impl_labeled!(Sampler);
crate::impl_parent_device!(Sampler);
crate::impl_storage_item!(Sampler);
crate::impl_trackable!(Sampler);

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
pub struct QuerySet {
    pub(crate) raw: ManuallyDrop<Box<dyn hal::DynQuerySet>>,
    pub(crate) device: Arc<Device>,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
    pub(crate) tracking_data: TrackingData,
    pub(crate) desc: wgt::QuerySetDescriptor<()>,
}

impl Drop for QuerySet {
    fn drop(&mut self) {
        resource_log!("Destroy raw {}", self.error_ident());
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            self.device.raw().destroy_query_set(raw);
        }
    }
}

crate::impl_resource_type!(QuerySet);
crate::impl_labeled!(QuerySet);
crate::impl_parent_device!(QuerySet);
crate::impl_storage_item!(QuerySet);
crate::impl_trackable!(QuerySet);

impl QuerySet {
    pub(crate) fn raw(&self) -> &dyn hal::DynQuerySet {
        self.raw.as_ref()
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum DestroyError {
    #[error("Resource is already destroyed")]
    AlreadyDestroyed,
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
}

pub type BlasDescriptor<'a> = wgt::CreateBlasDescriptor<Label<'a>>;
pub type TlasDescriptor<'a> = wgt::CreateTlasDescriptor<Label<'a>>;

pub(crate) trait AccelerationStructure: Trackable {
    fn raw<'a>(&'a self, guard: &'a SnatchGuard) -> Option<&'a dyn hal::DynAccelerationStructure>;
}

#[derive(Debug)]
pub struct Blas {
    pub(crate) raw: Snatchable<Box<dyn hal::DynAccelerationStructure>>,
    pub(crate) device: Arc<Device>,
    pub(crate) size_info: hal::AccelerationStructureBuildSizes,
    pub(crate) sizes: wgt::BlasGeometrySizeDescriptors,
    pub(crate) flags: wgt::AccelerationStructureFlags,
    pub(crate) update_mode: wgt::AccelerationStructureUpdateMode,
    pub(crate) built_index: RwLock<Option<NonZeroU64>>,
    pub(crate) handle: u64,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
    pub(crate) tracking_data: TrackingData,
}

impl Drop for Blas {
    fn drop(&mut self) {
        resource_log!("Destroy raw {}", self.error_ident());
        // SAFETY: We are in the Drop impl, and we don't use self.raw anymore after this point.
        if let Some(raw) = self.raw.take() {
            unsafe {
                self.device.raw().destroy_acceleration_structure(raw);
            }
        }
    }
}

impl AccelerationStructure for Blas {
    fn raw<'a>(&'a self, guard: &'a SnatchGuard) -> Option<&'a dyn hal::DynAccelerationStructure> {
        Some(self.raw.get(guard)?.as_ref())
    }
}

impl Blas {
    pub(crate) fn destroy(self: &Arc<Self>) -> Result<(), DestroyError> {
        let device = &self.device;

        let temp = {
            let mut snatch_guard = device.snatchable_lock.write();

            let raw = match self.raw.snatch(&mut snatch_guard) {
                Some(raw) => raw,
                None => {
                    return Err(DestroyError::AlreadyDestroyed);
                }
            };

            drop(snatch_guard);

            queue::TempResource::DestroyedAccelerationStructure(DestroyedAccelerationStructure {
                raw: ManuallyDrop::new(raw),
                device: Arc::clone(&self.device),
                label: self.label().to_owned(),
                bind_groups: WeakVec::new(),
            })
        };

        if let Some(queue) = device.get_queue() {
            let mut pending_writes = queue.pending_writes.lock();
            if pending_writes.contains_blas(self) {
                pending_writes.consume_temp(temp);
            } else {
                let mut life_lock = queue.lock_life();
                let last_submit_index = life_lock.get_blas_latest_submission_index(self);
                if let Some(last_submit_index) = last_submit_index {
                    life_lock.schedule_resource_destruction(temp, last_submit_index);
                }
            }
        }

        Ok(())
    }
}

crate::impl_resource_type!(Blas);
crate::impl_labeled!(Blas);
crate::impl_parent_device!(Blas);
crate::impl_storage_item!(Blas);
crate::impl_trackable!(Blas);

#[derive(Debug)]
pub struct Tlas {
    pub(crate) raw: Snatchable<Box<dyn hal::DynAccelerationStructure>>,
    pub(crate) device: Arc<Device>,
    pub(crate) size_info: hal::AccelerationStructureBuildSizes,
    pub(crate) max_instance_count: u32,
    pub(crate) flags: wgt::AccelerationStructureFlags,
    pub(crate) update_mode: wgt::AccelerationStructureUpdateMode,
    pub(crate) built_index: RwLock<Option<NonZeroU64>>,
    pub(crate) dependencies: RwLock<Vec<Arc<Blas>>>,
    pub(crate) instance_buffer: ManuallyDrop<Box<dyn hal::DynBuffer>>,
    /// The `label` from the descriptor used to create the resource.
    pub(crate) label: String,
    pub(crate) tracking_data: TrackingData,
    pub(crate) bind_groups: Mutex<WeakVec<BindGroup>>,
}

impl Drop for Tlas {
    fn drop(&mut self) {
        unsafe {
            resource_log!("Destroy raw {}", self.error_ident());
            if let Some(structure) = self.raw.take() {
                self.device.raw().destroy_acceleration_structure(structure);
            }
            let buffer = ManuallyDrop::take(&mut self.instance_buffer);
            self.device.raw().destroy_buffer(buffer);
        }
    }
}

impl AccelerationStructure for Tlas {
    fn raw<'a>(&'a self, guard: &'a SnatchGuard) -> Option<&'a dyn hal::DynAccelerationStructure> {
        Some(self.raw.get(guard)?.as_ref())
    }
}

crate::impl_resource_type!(Tlas);
crate::impl_labeled!(Tlas);
crate::impl_parent_device!(Tlas);
crate::impl_storage_item!(Tlas);
crate::impl_trackable!(Tlas);

impl Tlas {
    pub(crate) fn destroy(self: &Arc<Self>) -> Result<(), DestroyError> {
        let device = &self.device;

        let temp = {
            let mut snatch_guard = device.snatchable_lock.write();

            let raw = match self.raw.snatch(&mut snatch_guard) {
                Some(raw) => raw,
                None => {
                    return Err(DestroyError::AlreadyDestroyed);
                }
            };

            drop(snatch_guard);

            queue::TempResource::DestroyedAccelerationStructure(DestroyedAccelerationStructure {
                raw: ManuallyDrop::new(raw),
                device: Arc::clone(&self.device),
                label: self.label().to_owned(),
                bind_groups: mem::take(&mut self.bind_groups.lock()),
            })
        };

        if let Some(queue) = device.get_queue() {
            let mut pending_writes = queue.pending_writes.lock();
            if pending_writes.contains_tlas(self) {
                pending_writes.consume_temp(temp);
            } else {
                let mut life_lock = queue.lock_life();
                let last_submit_index = life_lock.get_tlas_latest_submission_index(self);
                if let Some(last_submit_index) = last_submit_index {
                    life_lock.schedule_resource_destruction(temp, last_submit_index);
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct DestroyedAccelerationStructure {
    raw: ManuallyDrop<Box<dyn hal::DynAccelerationStructure>>,
    device: Arc<Device>,
    label: String,
    // only filled if the acceleration structure is a TLAS
    bind_groups: WeakVec<BindGroup>,
}

impl DestroyedAccelerationStructure {
    pub fn label(&self) -> &dyn Debug {
        &self.label
    }
}

impl Drop for DestroyedAccelerationStructure {
    fn drop(&mut self) {
        let mut deferred = self.device.deferred_destroy.lock();
        deferred.push(DeferredDestroy::BindGroups(mem::take(
            &mut self.bind_groups,
        )));
        drop(deferred);

        resource_log!("Destroy raw Buffer (destroyed) {:?}", self.label());
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        unsafe {
            hal::DynDevice::destroy_acceleration_structure(self.device.raw(), raw);
        }
    }
}
