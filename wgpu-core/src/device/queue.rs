#[cfg(feature = "trace")]
use crate::device::trace::Action;
use crate::{
    api_log,
    command::{
        extract_texture_selector, validate_linear_texture_data, validate_texture_copy_range,
        ClearError, CommandAllocator, CommandBuffer, CommandEncoderError, CopySide,
        TexelCopyTextureInfo, TransferError,
    },
    conv,
    device::{DeviceError, WaitIdleError},
    get_lowest_common_denom,
    global::Global,
    id::{self, QueueId},
    init_tracker::{has_copy_partial_init_tracker_coverage, TextureInitRange},
    lock::{rank, Mutex, MutexGuard, RwLockWriteGuard},
    resource::{
        Buffer, BufferAccessError, BufferMapState, DestroyedBuffer, DestroyedResourceError,
        DestroyedTexture, Fallible, FlushedStagingBuffer, InvalidResourceError, Labeled,
        ParentDevice, ResourceErrorIdent, StagingBuffer, Texture, TextureInner, Trackable,
    },
    resource_log,
    snatch::SnatchGuard,
    track::{self, Tracker, TrackerIndex},
    FastHashMap, SubmissionIndex,
};

use smallvec::SmallVec;

use crate::resource::{Blas, DestroyedAccelerationStructure, Tlas};
use crate::scratch::ScratchBuffer;
use std::{
    iter,
    mem::{self, ManuallyDrop},
    ptr::NonNull,
    sync::{atomic::Ordering, Arc},
};
use thiserror::Error;

use super::{life::LifetimeTracker, Device};

pub struct Queue {
    raw: Box<dyn hal::DynQueue>,
    pub(crate) pending_writes: Mutex<PendingWrites>,
    life_tracker: Mutex<LifetimeTracker>,
    // The device needs to be dropped last (`Device.zero_buffer` might be referenced by the encoder in pending writes).
    pub(crate) device: Arc<Device>,
}

impl Queue {
    pub(crate) fn new(
        device: Arc<Device>,
        raw: Box<dyn hal::DynQueue>,
    ) -> Result<Self, DeviceError> {
        let pending_encoder = device
            .command_allocator
            .acquire_encoder(device.raw(), raw.as_ref())
            .map_err(DeviceError::from_hal);

        let pending_encoder = match pending_encoder {
            Ok(pending_encoder) => pending_encoder,
            Err(e) => {
                return Err(e);
            }
        };

        let mut pending_writes = PendingWrites::new(pending_encoder);

        let zero_buffer = device.zero_buffer.as_ref();
        pending_writes.activate();
        unsafe {
            pending_writes
                .command_encoder
                .transition_buffers(&[hal::BufferBarrier {
                    buffer: zero_buffer,
                    usage: hal::StateTransition {
                        from: hal::BufferUses::empty(),
                        to: hal::BufferUses::COPY_DST,
                    },
                }]);
            pending_writes
                .command_encoder
                .clear_buffer(zero_buffer, 0..super::ZERO_BUFFER_SIZE);
            pending_writes
                .command_encoder
                .transition_buffers(&[hal::BufferBarrier {
                    buffer: zero_buffer,
                    usage: hal::StateTransition {
                        from: hal::BufferUses::COPY_DST,
                        to: hal::BufferUses::COPY_SRC,
                    },
                }]);
        }

        Ok(Queue {
            raw,
            device,
            pending_writes: Mutex::new(rank::QUEUE_PENDING_WRITES, pending_writes),
            life_tracker: Mutex::new(rank::QUEUE_LIFE_TRACKER, LifetimeTracker::new()),
        })
    }

    pub(crate) fn raw(&self) -> &dyn hal::DynQueue {
        self.raw.as_ref()
    }

    #[track_caller]
    pub(crate) fn lock_life<'a>(&'a self) -> MutexGuard<'a, LifetimeTracker> {
        self.life_tracker.lock()
    }

    pub(crate) fn maintain(
        &self,
        submission_index: u64,
        snatch_guard: &SnatchGuard,
    ) -> (
        SmallVec<[SubmittedWorkDoneClosure; 1]>,
        Vec<super::BufferMapPendingClosure>,
        bool,
    ) {
        let mut life_tracker = self.lock_life();
        let submission_closures = life_tracker.triage_submissions(submission_index);

        let mapping_closures = life_tracker.handle_mapping(snatch_guard);

        let queue_empty = life_tracker.queue_empty();

        (submission_closures, mapping_closures, queue_empty)
    }
}

crate::impl_resource_type!(Queue);
// TODO: https://github.com/gfx-rs/wgpu/issues/4014
impl Labeled for Queue {
    fn label(&self) -> &str {
        ""
    }
}
crate::impl_parent_device!(Queue);
crate::impl_storage_item!(Queue);

impl Drop for Queue {
    fn drop(&mut self) {
        resource_log!("Drop {}", self.error_ident());

        let last_successful_submission_index = self
            .device
            .last_successful_submission_index
            .load(Ordering::Acquire);

        let fence = self.device.fence.read();

        // Try waiting on the last submission using the following sequence of timeouts
        let timeouts_in_ms = [100, 200, 400, 800, 1600, 3200];

        for (i, timeout_ms) in timeouts_in_ms.into_iter().enumerate() {
            let is_last_iter = i == timeouts_in_ms.len() - 1;

            api_log!(
                "Waiting on last submission. try: {}/{}. timeout: {}ms",
                i + 1,
                timeouts_in_ms.len(),
                timeout_ms
            );

            let wait_res = unsafe {
                self.device.raw().wait(
                    fence.as_ref(),
                    last_successful_submission_index,
                    #[cfg(not(target_arch = "wasm32"))]
                    timeout_ms,
                    #[cfg(target_arch = "wasm32")]
                    0, // WebKit and Chromium don't support a non-0 timeout
                )
            };
            // Note: If we don't panic below we are in UB land (destroying resources while they are still in use by the GPU).
            match wait_res {
                Ok(true) => break,
                Ok(false) => {
                    // It's fine that we timed out on WebGL; GL objects can be deleted early as they
                    // will be kept around by the driver if GPU work hasn't finished.
                    // Moreover, the way we emulate read mappings on WebGL allows us to execute map_buffer earlier than on other
                    // backends since getBufferSubData is synchronous with respect to the other previously enqueued GL commands.
                    // Relying on this behavior breaks the clean abstraction wgpu-hal tries to maintain and
                    // we should find ways to improve this. See https://github.com/gfx-rs/wgpu/issues/6538.
                    #[cfg(target_arch = "wasm32")]
                    {
                        break;
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        if is_last_iter {
                            panic!(
                                "We timed out while waiting on the last successful submission to complete!"
                            );
                        }
                    }
                }
                Err(e) => match e {
                    hal::DeviceError::OutOfMemory => {
                        if is_last_iter {
                            panic!(
                                "We ran into an OOM error while waiting on the last successful submission to complete!"
                            );
                        }
                    }
                    hal::DeviceError::Lost => {
                        self.device.handle_hal_error(e); // will lose the device
                        break;
                    }
                    hal::DeviceError::ResourceCreationFailed => unreachable!(),
                    hal::DeviceError::Unexpected => {
                        panic!(
                            "We ran into an unexpected error while waiting on the last successful submission to complete!"
                        );
                    }
                },
            }
        }
        drop(fence);

        let snatch_guard = self.device.snatchable_lock.read();
        let (submission_closures, mapping_closures, queue_empty) =
            self.maintain(last_successful_submission_index, &snatch_guard);
        drop(snatch_guard);

        assert!(queue_empty);

        let closures = crate::device::UserClosures {
            mappings: mapping_closures,
            submissions: submission_closures,
            device_lost_invocations: SmallVec::new(),
        };

        closures.fire();
    }
}

#[cfg(send_sync)]
pub type SubmittedWorkDoneClosure = Box<dyn FnOnce() + Send + 'static>;
#[cfg(not(send_sync))]
pub type SubmittedWorkDoneClosure = Box<dyn FnOnce() + 'static>;

/// A texture or buffer to be freed soon.
///
/// This is just a tagged raw texture or buffer, generally about to be added to
/// some other more specific container like:
///
/// - `PendingWrites::temp_resources`: resources used by queue writes and
///   unmaps, waiting to be folded in with the next queue submission
///
/// - `ActiveSubmission::temp_resources`: temporary resources used by a queue
///   submission, to be freed when it completes
#[derive(Debug)]
pub enum TempResource {
    StagingBuffer(FlushedStagingBuffer),
    ScratchBuffer(ScratchBuffer),
    DestroyedBuffer(DestroyedBuffer),
    DestroyedTexture(DestroyedTexture),
    DestroyedAccelerationStructure(DestroyedAccelerationStructure),
}

/// A series of raw [`CommandBuffer`]s that have been submitted to a
/// queue, and the [`wgpu_hal::CommandEncoder`] that built them.
///
/// [`CommandBuffer`]: hal::Api::CommandBuffer
/// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
pub(crate) struct EncoderInFlight {
    inner: crate::command::CommandEncoder,
    pub(crate) trackers: Tracker,

    /// These are the buffers that have been tracked by `PendingWrites`.
    pub(crate) pending_buffers: FastHashMap<TrackerIndex, Arc<Buffer>>,
    /// These are the textures that have been tracked by `PendingWrites`.
    pub(crate) pending_textures: FastHashMap<TrackerIndex, Arc<Texture>>,
    /// These are the BLASes that have been tracked by `PendingWrites`.
    pub(crate) pending_blas_s: FastHashMap<TrackerIndex, Arc<Blas>>,
    /// These are the TLASes that have been tracked by `PendingWrites`.
    pub(crate) pending_tlas_s: FastHashMap<TrackerIndex, Arc<Tlas>>,
}

/// A private command encoder for writes made directly on the device
/// or queue.
///
/// Operations like `buffer_unmap`, `queue_write_buffer`, and
/// `queue_write_texture` need to copy data to the GPU. At the hal
/// level, this must be done by encoding and submitting commands, but
/// these operations are not associated with any specific wgpu command
/// buffer.
///
/// Instead, `Device::pending_writes` owns one of these values, which
/// has its own hal command encoder and resource lists. The commands
/// accumulated here are automatically submitted to the queue the next
/// time the user submits a wgpu command buffer, ahead of the user's
/// commands.
///
/// Important:
/// When locking pending_writes be sure that tracker is not locked
/// and try to lock trackers for the minimum timespan possible
///
/// All uses of [`StagingBuffer`]s end up here.
#[derive(Debug)]
pub(crate) struct PendingWrites {
    // The command encoder needs to be destroyed before any other resource in pending writes.
    pub command_encoder: Box<dyn hal::DynCommandEncoder>,

    /// True if `command_encoder` is in the "recording" state, as
    /// described in the docs for the [`wgpu_hal::CommandEncoder`]
    /// trait.
    ///
    /// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
    pub is_recording: bool,

    temp_resources: Vec<TempResource>,
    dst_buffers: FastHashMap<TrackerIndex, Arc<Buffer>>,
    dst_textures: FastHashMap<TrackerIndex, Arc<Texture>>,
    dst_blas_s: FastHashMap<TrackerIndex, Arc<Blas>>,
    dst_tlas_s: FastHashMap<TrackerIndex, Arc<Tlas>>,
}

impl PendingWrites {
    pub fn new(command_encoder: Box<dyn hal::DynCommandEncoder>) -> Self {
        Self {
            command_encoder,
            is_recording: false,
            temp_resources: Vec::new(),
            dst_buffers: FastHashMap::default(),
            dst_textures: FastHashMap::default(),
            dst_blas_s: FastHashMap::default(),
            dst_tlas_s: FastHashMap::default(),
        }
    }

    pub fn insert_buffer(&mut self, buffer: &Arc<Buffer>) {
        self.dst_buffers
            .insert(buffer.tracker_index(), buffer.clone());
    }

    pub fn insert_texture(&mut self, texture: &Arc<Texture>) {
        self.dst_textures
            .insert(texture.tracker_index(), texture.clone());
    }

    pub fn contains_buffer(&self, buffer: &Arc<Buffer>) -> bool {
        self.dst_buffers.contains_key(&buffer.tracker_index())
    }

    pub fn contains_texture(&self, texture: &Arc<Texture>) -> bool {
        self.dst_textures.contains_key(&texture.tracker_index())
    }

    pub fn insert_blas(&mut self, blas: &Arc<Blas>) {
        self.dst_blas_s.insert(blas.tracker_index(), blas.clone());
    }

    pub fn insert_tlas(&mut self, tlas: &Arc<Tlas>) {
        self.dst_tlas_s.insert(tlas.tracker_index(), tlas.clone());
    }

    pub fn contains_blas(&mut self, blas: &Arc<Blas>) -> bool {
        self.dst_blas_s.contains_key(&blas.tracker_index())
    }

    pub fn contains_tlas(&mut self, tlas: &Arc<Tlas>) -> bool {
        self.dst_tlas_s.contains_key(&tlas.tracker_index())
    }

    pub fn consume_temp(&mut self, resource: TempResource) {
        self.temp_resources.push(resource);
    }

    pub fn consume(&mut self, buffer: FlushedStagingBuffer) {
        self.temp_resources
            .push(TempResource::StagingBuffer(buffer));
    }

    fn pre_submit(
        &mut self,
        command_allocator: &CommandAllocator,
        device: &Arc<Device>,
        queue: &Queue,
    ) -> Result<Option<EncoderInFlight>, DeviceError> {
        if self.is_recording {
            let pending_buffers = mem::take(&mut self.dst_buffers);
            let pending_textures = mem::take(&mut self.dst_textures);
            let pending_blas_s = mem::take(&mut self.dst_blas_s);
            let pending_tlas_s = mem::take(&mut self.dst_tlas_s);

            let cmd_buf = unsafe { self.command_encoder.end_encoding() }
                .map_err(|e| device.handle_hal_error(e))?;
            self.is_recording = false;

            let new_encoder = command_allocator
                .acquire_encoder(device.raw(), queue.raw())
                .map_err(|e| device.handle_hal_error(e))?;

            let encoder = EncoderInFlight {
                inner: crate::command::CommandEncoder {
                    raw: ManuallyDrop::new(mem::replace(&mut self.command_encoder, new_encoder)),
                    list: vec![cmd_buf],
                    device: device.clone(),
                    is_open: false,
                    hal_label: None,
                },
                trackers: Tracker::new(),
                pending_buffers,
                pending_textures,
                pending_blas_s,
                pending_tlas_s,
            };
            Ok(Some(encoder))
        } else {
            self.dst_buffers.clear();
            self.dst_textures.clear();
            Ok(None)
        }
    }

    pub fn activate(&mut self) -> &mut dyn hal::DynCommandEncoder {
        if !self.is_recording {
            unsafe {
                self.command_encoder
                    .begin_encoding(Some("(wgpu internal) PendingWrites"))
                    .unwrap();
            }
            self.is_recording = true;
        }
        self.command_encoder.as_mut()
    }
}

impl Drop for PendingWrites {
    fn drop(&mut self) {
        unsafe {
            if self.is_recording {
                self.command_encoder.discard_encoding();
            }
        }
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum QueueWriteError {
    #[error(transparent)]
    Queue(#[from] DeviceError),
    #[error(transparent)]
    Transfer(#[from] TransferError),
    #[error(transparent)]
    MemoryInitFailure(#[from] ClearError),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum QueueSubmitError {
    #[error(transparent)]
    Queue(#[from] DeviceError),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error(transparent)]
    Unmap(#[from] BufferAccessError),
    #[error("{0} is still mapped")]
    BufferStillMapped(ResourceErrorIdent),
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
    #[error(transparent)]
    CommandEncoder(#[from] CommandEncoderError),
    #[error(transparent)]
    ValidateBlasActionsError(#[from] crate::ray_tracing::ValidateBlasActionsError),
    #[error(transparent)]
    ValidateTlasActionsError(#[from] crate::ray_tracing::ValidateTlasActionsError),
}

//TODO: move out common parts of write_xxx.

impl Queue {
    pub fn write_buffer(
        &self,
        buffer: Fallible<Buffer>,
        buffer_offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::write_buffer");
        api_log!("Queue::write_buffer");

        let buffer = buffer.get()?;

        let data_size = data.len() as wgt::BufferAddress;

        self.same_device_as(buffer.as_ref())?;

        let data_size = if let Some(data_size) = wgt::BufferSize::new(data_size) {
            data_size
        } else {
            log::trace!("Ignoring write_buffer of size 0");
            return Ok(());
        };

        // Platform validation requires that the staging buffer always be
        // freed, even if an error occurs. All paths from here must call
        // `device.pending_writes.consume`.
        let mut staging_buffer = StagingBuffer::new(&self.device, data_size)?;
        let mut pending_writes = self.pending_writes.lock();

        let staging_buffer = {
            profiling::scope!("copy");
            staging_buffer.write(data);
            staging_buffer.flush()
        };

        let result = self.write_staging_buffer_impl(
            &mut pending_writes,
            &staging_buffer,
            buffer,
            buffer_offset,
        );

        pending_writes.consume(staging_buffer);
        result
    }

    pub fn create_staging_buffer(
        &self,
        buffer_size: wgt::BufferSize,
    ) -> Result<(StagingBuffer, NonNull<u8>), QueueWriteError> {
        profiling::scope!("Queue::create_staging_buffer");
        resource_log!("Queue::create_staging_buffer");

        let staging_buffer = StagingBuffer::new(&self.device, buffer_size)?;
        let ptr = unsafe { staging_buffer.ptr() };

        Ok((staging_buffer, ptr))
    }

    pub fn write_staging_buffer(
        &self,
        buffer: Fallible<Buffer>,
        buffer_offset: wgt::BufferAddress,
        staging_buffer: StagingBuffer,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::write_staging_buffer");

        let buffer = buffer.get()?;

        let mut pending_writes = self.pending_writes.lock();

        // At this point, we have taken ownership of the staging_buffer from the
        // user. Platform validation requires that the staging buffer always
        // be freed, even if an error occurs. All paths from here must call
        // `device.pending_writes.consume`.
        let staging_buffer = staging_buffer.flush();

        let result = self.write_staging_buffer_impl(
            &mut pending_writes,
            &staging_buffer,
            buffer,
            buffer_offset,
        );

        pending_writes.consume(staging_buffer);
        result
    }

    pub fn validate_write_buffer(
        &self,
        buffer: Fallible<Buffer>,
        buffer_offset: u64,
        buffer_size: wgt::BufferSize,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::validate_write_buffer");

        let buffer = buffer.get()?;

        self.validate_write_buffer_impl(&buffer, buffer_offset, buffer_size)?;

        Ok(())
    }

    fn validate_write_buffer_impl(
        &self,
        buffer: &Buffer,
        buffer_offset: u64,
        buffer_size: wgt::BufferSize,
    ) -> Result<(), TransferError> {
        buffer.check_usage(wgt::BufferUsages::COPY_DST)?;
        if buffer_size.get() % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(TransferError::UnalignedCopySize(buffer_size.get()));
        }
        if buffer_offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(TransferError::UnalignedBufferOffset(buffer_offset));
        }
        if buffer_offset + buffer_size.get() > buffer.size {
            return Err(TransferError::BufferOverrun {
                start_offset: buffer_offset,
                end_offset: buffer_offset + buffer_size.get(),
                buffer_size: buffer.size,
                side: CopySide::Destination,
            });
        }

        Ok(())
    }

    fn write_staging_buffer_impl(
        &self,
        pending_writes: &mut PendingWrites,
        staging_buffer: &FlushedStagingBuffer,
        buffer: Arc<Buffer>,
        buffer_offset: u64,
    ) -> Result<(), QueueWriteError> {
        let transition = {
            let mut trackers = self.device.trackers.lock();
            trackers
                .buffers
                .set_single(&buffer, hal::BufferUses::COPY_DST)
        };

        let snatch_guard = self.device.snatchable_lock.read();
        let dst_raw = buffer.try_raw(&snatch_guard)?;

        self.same_device_as(buffer.as_ref())?;

        self.validate_write_buffer_impl(&buffer, buffer_offset, staging_buffer.size)?;

        let region = hal::BufferCopy {
            src_offset: 0,
            dst_offset: buffer_offset,
            size: staging_buffer.size,
        };
        let barriers = iter::once(hal::BufferBarrier {
            buffer: staging_buffer.raw(),
            usage: hal::StateTransition {
                from: hal::BufferUses::MAP_WRITE,
                to: hal::BufferUses::COPY_SRC,
            },
        })
        .chain(transition.map(|pending| pending.into_hal(&buffer, &snatch_guard)))
        .collect::<Vec<_>>();
        let encoder = pending_writes.activate();
        unsafe {
            encoder.transition_buffers(&barriers);
            encoder.copy_buffer_to_buffer(staging_buffer.raw(), dst_raw, &[region]);
        }

        pending_writes.insert_buffer(&buffer);

        // Ensure the overwritten bytes are marked as initialized so
        // they don't need to be nulled prior to mapping or binding.
        {
            buffer
                .initialization_status
                .write()
                .drain(buffer_offset..(buffer_offset + staging_buffer.size.get()));
        }

        Ok(())
    }

    pub fn write_texture(
        &self,
        destination: wgt::TexelCopyTextureInfo<Fallible<Texture>>,
        data: &[u8],
        data_layout: &wgt::TexelCopyBufferLayout,
        size: &wgt::Extent3d,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::write_texture");
        api_log!("Queue::write_texture");

        if size.width == 0 || size.height == 0 || size.depth_or_array_layers == 0 {
            log::trace!("Ignoring write_texture of size 0");
            return Ok(());
        }

        let dst = destination.texture.get()?;
        let destination = wgt::TexelCopyTextureInfo {
            texture: (),
            mip_level: destination.mip_level,
            origin: destination.origin,
            aspect: destination.aspect,
        };

        self.same_device_as(dst.as_ref())?;

        dst.check_usage(wgt::TextureUsages::COPY_DST)
            .map_err(TransferError::MissingTextureUsage)?;

        // Note: Doing the copy range validation early is important because ensures that the
        // dimensions are not going to cause overflow in other parts of the validation.
        let (hal_copy_size, array_layer_count) =
            validate_texture_copy_range(&destination, &dst.desc, CopySide::Destination, size)?;

        let (selector, dst_base) = extract_texture_selector(&destination, size, &dst)?;

        if !dst_base.aspect.is_one() {
            return Err(TransferError::CopyAspectNotOne.into());
        }

        if !conv::is_valid_copy_dst_texture_format(dst.desc.format, destination.aspect) {
            return Err(TransferError::CopyToForbiddenTextureFormat {
                format: dst.desc.format,
                aspect: destination.aspect,
            }
            .into());
        }

        // Note: `_source_bytes_per_array_layer` is ignored since we
        // have a staging copy, and it can have a different value.
        let (required_bytes_in_copy, _source_bytes_per_array_layer) = validate_linear_texture_data(
            data_layout,
            dst.desc.format,
            destination.aspect,
            data.len() as wgt::BufferAddress,
            CopySide::Source,
            size,
            false,
        )?;

        if dst.desc.format.is_depth_stencil_format() {
            self.device
                .require_downlevel_flags(wgt::DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES)
                .map_err(TransferError::from)?;
        }

        let mut pending_writes = self.pending_writes.lock();
        let encoder = pending_writes.activate();

        // If the copy does not fully cover the layers, we need to initialize to
        // zero *first* as we don't keep track of partial texture layer inits.
        //
        // Strictly speaking we only need to clear the areas of a layer
        // untouched, but this would get increasingly messy.
        let init_layer_range = if dst.desc.dimension == wgt::TextureDimension::D3 {
            // volume textures don't have a layer range as array volumes aren't supported
            0..1
        } else {
            destination.origin.z..destination.origin.z + size.depth_or_array_layers
        };
        let mut dst_initialization_status = dst.initialization_status.write();
        if dst_initialization_status.mips[destination.mip_level as usize]
            .check(init_layer_range.clone())
            .is_some()
        {
            if has_copy_partial_init_tracker_coverage(size, destination.mip_level, &dst.desc) {
                for layer_range in dst_initialization_status.mips[destination.mip_level as usize]
                    .drain(init_layer_range)
                    .collect::<Vec<std::ops::Range<u32>>>()
                {
                    let mut trackers = self.device.trackers.lock();
                    crate::command::clear_texture(
                        &dst,
                        TextureInitRange {
                            mip_range: destination.mip_level..(destination.mip_level + 1),
                            layer_range,
                        },
                        encoder,
                        &mut trackers.textures,
                        &self.device.alignments,
                        self.device.zero_buffer.as_ref(),
                        &self.device.snatchable_lock.read(),
                    )
                    .map_err(QueueWriteError::from)?;
                }
            } else {
                dst_initialization_status.mips[destination.mip_level as usize]
                    .drain(init_layer_range);
            }
        }

        let snatch_guard = self.device.snatchable_lock.read();

        let dst_raw = dst.try_raw(&snatch_guard)?;

        let (block_width, block_height) = dst.desc.format.block_dimensions();
        let width_in_blocks = size.width / block_width;
        let height_in_blocks = size.height / block_height;

        let block_size = dst
            .desc
            .format
            .block_copy_size(Some(destination.aspect))
            .unwrap();
        let bytes_in_last_row = width_in_blocks * block_size;

        let bytes_per_row = data_layout.bytes_per_row.unwrap_or(bytes_in_last_row);
        let rows_per_image = data_layout.rows_per_image.unwrap_or(height_in_blocks);

        let bytes_per_row_alignment = get_lowest_common_denom(
            self.device.alignments.buffer_copy_pitch.get() as u32,
            block_size,
        );
        let stage_bytes_per_row = wgt::math::align_to(bytes_in_last_row, bytes_per_row_alignment);

        // Platform validation requires that the staging buffer always be
        // freed, even if an error occurs. All paths from here must call
        // `device.pending_writes.consume`.
        let staging_buffer = if stage_bytes_per_row == bytes_per_row {
            profiling::scope!("copy aligned");
            // Fast path if the data is already being aligned optimally.
            let stage_size = wgt::BufferSize::new(required_bytes_in_copy).unwrap();
            let mut staging_buffer = StagingBuffer::new(&self.device, stage_size)?;
            staging_buffer.write(&data[data_layout.offset as usize..]);
            staging_buffer
        } else {
            profiling::scope!("copy chunked");
            // Copy row by row into the optimal alignment.
            let block_rows_in_copy =
                (size.depth_or_array_layers - 1) * rows_per_image + height_in_blocks;
            let stage_size =
                wgt::BufferSize::new(stage_bytes_per_row as u64 * block_rows_in_copy as u64)
                    .unwrap();
            let mut staging_buffer = StagingBuffer::new(&self.device, stage_size)?;
            let copy_bytes_per_row = stage_bytes_per_row.min(bytes_per_row) as usize;
            for layer in 0..size.depth_or_array_layers {
                let rows_offset = layer * rows_per_image;
                for row in rows_offset..rows_offset + height_in_blocks {
                    let src_offset = data_layout.offset as u32 + row * bytes_per_row;
                    let dst_offset = row * stage_bytes_per_row;
                    unsafe {
                        staging_buffer.write_with_offset(
                            data,
                            src_offset as isize,
                            dst_offset as isize,
                            copy_bytes_per_row,
                        )
                    }
                }
            }
            staging_buffer
        };

        let staging_buffer = staging_buffer.flush();

        let regions = (0..array_layer_count)
            .map(|array_layer_offset| {
                let mut texture_base = dst_base.clone();
                texture_base.array_layer += array_layer_offset;
                hal::BufferTextureCopy {
                    buffer_layout: wgt::TexelCopyBufferLayout {
                        offset: array_layer_offset as u64
                            * rows_per_image as u64
                            * stage_bytes_per_row as u64,
                        bytes_per_row: Some(stage_bytes_per_row),
                        rows_per_image: Some(rows_per_image),
                    },
                    texture_base,
                    size: hal_copy_size,
                }
            })
            .collect::<Vec<_>>();

        {
            let buffer_barrier = hal::BufferBarrier {
                buffer: staging_buffer.raw(),
                usage: hal::StateTransition {
                    from: hal::BufferUses::MAP_WRITE,
                    to: hal::BufferUses::COPY_SRC,
                },
            };

            let mut trackers = self.device.trackers.lock();
            let transition =
                trackers
                    .textures
                    .set_single(&dst, selector, hal::TextureUses::COPY_DST);
            let texture_barriers = transition
                .map(|pending| pending.into_hal(dst_raw))
                .collect::<Vec<_>>();

            unsafe {
                encoder.transition_textures(&texture_barriers);
                encoder.transition_buffers(&[buffer_barrier]);
                encoder.copy_buffer_to_texture(staging_buffer.raw(), dst_raw, &regions);
            }
        }

        pending_writes.consume(staging_buffer);
        pending_writes.insert_texture(&dst);

        Ok(())
    }

    #[cfg(webgl)]
    pub fn copy_external_image_to_texture(
        &self,
        source: &wgt::CopyExternalImageSourceInfo,
        destination: wgt::CopyExternalImageDestInfo<Fallible<Texture>>,
        size: wgt::Extent3d,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::copy_external_image_to_texture");

        if size.width == 0 || size.height == 0 || size.depth_or_array_layers == 0 {
            log::trace!("Ignoring write_texture of size 0");
            return Ok(());
        }

        let mut needs_flag = false;
        needs_flag |= matches!(source.source, wgt::ExternalImageSource::OffscreenCanvas(_));
        needs_flag |= source.origin != wgt::Origin2d::ZERO;
        needs_flag |= destination.color_space != wgt::PredefinedColorSpace::Srgb;
        #[allow(clippy::bool_comparison)]
        if matches!(source.source, wgt::ExternalImageSource::ImageBitmap(_)) {
            needs_flag |= source.flip_y != false;
            needs_flag |= destination.premultiplied_alpha != false;
        }

        if needs_flag {
            self.device
                .require_downlevel_flags(wgt::DownlevelFlags::UNRESTRICTED_EXTERNAL_TEXTURE_COPIES)
                .map_err(TransferError::from)?;
        }

        let src_width = source.source.width();
        let src_height = source.source.height();

        let dst = destination.texture.get()?;
        let premultiplied_alpha = destination.premultiplied_alpha;
        let destination = wgt::TexelCopyTextureInfo {
            texture: (),
            mip_level: destination.mip_level,
            origin: destination.origin,
            aspect: destination.aspect,
        };

        if !conv::is_valid_external_image_copy_dst_texture_format(dst.desc.format) {
            return Err(
                TransferError::ExternalCopyToForbiddenTextureFormat(dst.desc.format).into(),
            );
        }
        if dst.desc.dimension != wgt::TextureDimension::D2 {
            return Err(TransferError::InvalidDimensionExternal.into());
        }
        dst.check_usage(wgt::TextureUsages::COPY_DST | wgt::TextureUsages::RENDER_ATTACHMENT)
            .map_err(TransferError::MissingTextureUsage)?;
        if dst.desc.sample_count != 1 {
            return Err(TransferError::InvalidSampleCount {
                sample_count: dst.desc.sample_count,
            }
            .into());
        }

        if source.origin.x + size.width > src_width {
            return Err(TransferError::TextureOverrun {
                start_offset: source.origin.x,
                end_offset: source.origin.x + size.width,
                texture_size: src_width,
                dimension: crate::resource::TextureErrorDimension::X,
                side: CopySide::Source,
            }
            .into());
        }
        if source.origin.y + size.height > src_height {
            return Err(TransferError::TextureOverrun {
                start_offset: source.origin.y,
                end_offset: source.origin.y + size.height,
                texture_size: src_height,
                dimension: crate::resource::TextureErrorDimension::Y,
                side: CopySide::Source,
            }
            .into());
        }
        if size.depth_or_array_layers != 1 {
            return Err(TransferError::TextureOverrun {
                start_offset: 0,
                end_offset: size.depth_or_array_layers,
                texture_size: 1,
                dimension: crate::resource::TextureErrorDimension::Z,
                side: CopySide::Source,
            }
            .into());
        }

        // Note: Doing the copy range validation early is important because ensures that the
        // dimensions are not going to cause overflow in other parts of the validation.
        let (hal_copy_size, _) =
            validate_texture_copy_range(&destination, &dst.desc, CopySide::Destination, &size)?;

        let (selector, dst_base) = extract_texture_selector(&destination, &size, &dst)?;

        let mut pending_writes = self.pending_writes.lock();
        let encoder = pending_writes.activate();

        // If the copy does not fully cover the layers, we need to initialize to
        // zero *first* as we don't keep track of partial texture layer inits.
        //
        // Strictly speaking we only need to clear the areas of a layer
        // untouched, but this would get increasingly messy.
        let init_layer_range = if dst.desc.dimension == wgt::TextureDimension::D3 {
            // volume textures don't have a layer range as array volumes aren't supported
            0..1
        } else {
            destination.origin.z..destination.origin.z + size.depth_or_array_layers
        };
        let mut dst_initialization_status = dst.initialization_status.write();
        if dst_initialization_status.mips[destination.mip_level as usize]
            .check(init_layer_range.clone())
            .is_some()
        {
            if has_copy_partial_init_tracker_coverage(&size, destination.mip_level, &dst.desc) {
                for layer_range in dst_initialization_status.mips[destination.mip_level as usize]
                    .drain(init_layer_range)
                    .collect::<Vec<std::ops::Range<u32>>>()
                {
                    let mut trackers = self.device.trackers.lock();
                    crate::command::clear_texture(
                        &dst,
                        TextureInitRange {
                            mip_range: destination.mip_level..(destination.mip_level + 1),
                            layer_range,
                        },
                        encoder,
                        &mut trackers.textures,
                        &self.device.alignments,
                        self.device.zero_buffer.as_ref(),
                        &self.device.snatchable_lock.read(),
                    )
                    .map_err(QueueWriteError::from)?;
                }
            } else {
                dst_initialization_status.mips[destination.mip_level as usize]
                    .drain(init_layer_range);
            }
        }

        let snatch_guard = self.device.snatchable_lock.read();
        let dst_raw = dst.try_raw(&snatch_guard)?;

        let regions = hal::TextureCopy {
            src_base: hal::TextureCopyBase {
                mip_level: 0,
                array_layer: 0,
                origin: source.origin.to_3d(0),
                aspect: hal::FormatAspects::COLOR,
            },
            dst_base,
            size: hal_copy_size,
        };

        let mut trackers = self.device.trackers.lock();
        let transitions = trackers
            .textures
            .set_single(&dst, selector, hal::TextureUses::COPY_DST);

        // `copy_external_image_to_texture` is exclusive to the WebGL backend.
        // Don't go through the `DynCommandEncoder` abstraction and directly to the WebGL backend.
        let encoder_webgl = encoder
            .as_any_mut()
            .downcast_mut::<hal::gles::CommandEncoder>()
            .unwrap();
        let dst_raw_webgl = dst_raw
            .as_any()
            .downcast_ref::<hal::gles::Texture>()
            .unwrap();
        let transitions_webgl = transitions.map(|pending| {
            let dyn_transition = pending.into_hal(dst_raw);
            hal::TextureBarrier {
                texture: dst_raw_webgl,
                range: dyn_transition.range,
                usage: dyn_transition.usage,
            }
        });

        use hal::CommandEncoder as _;
        unsafe {
            encoder_webgl.transition_textures(transitions_webgl);
            encoder_webgl.copy_external_image_to_texture(
                source,
                dst_raw_webgl,
                premultiplied_alpha,
                iter::once(regions),
            );
        }

        Ok(())
    }

    pub fn submit(
        &self,
        command_buffers: &[Arc<CommandBuffer>],
    ) -> Result<SubmissionIndex, (SubmissionIndex, QueueSubmitError)> {
        profiling::scope!("Queue::submit");
        api_log!("Queue::submit");

        let submit_index;

        let res = 'error: {
            let snatch_guard = self.device.snatchable_lock.read();

            // Fence lock must be acquired after the snatch lock everywhere to avoid deadlocks.
            let mut fence = self.device.fence.write();
            submit_index = self
                .device
                .active_submission_index
                .fetch_add(1, Ordering::SeqCst)
                + 1;
            let mut active_executions = Vec::new();

            let mut used_surface_textures = track::TextureUsageScope::default();

            // Use a hashmap here to deduplicate the surface textures that are used in the command buffers.
            // This avoids vulkan deadlocking from the same surface texture being submitted multiple times.
            let mut submit_surface_textures_owned = FastHashMap::default();

            {
                if !command_buffers.is_empty() {
                    profiling::scope!("prepare");

                    let mut first_error = None;

                    //TODO: if multiple command buffers are submitted, we can re-use the last
                    // native command buffer of the previous chain instead of always creating
                    // a temporary one, since the chains are not finished.

                    // finish all the command buffers first
                    for command_buffer in command_buffers {
                        profiling::scope!("process command buffer");

                        // we reset the used surface textures every time we use
                        // it, so make sure to set_size on it.
                        used_surface_textures.set_size(self.device.tracker_indices.textures.size());

                        // Note that we are required to invalidate all command buffers in both the success and failure paths.
                        // This is why we `continue` and don't early return via `?`.
                        #[allow(unused_mut)]
                        let mut cmd_buf_data = command_buffer.take_finished();

                        #[cfg(feature = "trace")]
                        if let Some(ref mut trace) = *self.device.trace.lock() {
                            if let Ok(ref mut cmd_buf_data) = cmd_buf_data {
                                trace.add(Action::Submit(
                                    submit_index,
                                    cmd_buf_data.commands.take().unwrap(),
                                ));
                            }
                        }

                        if first_error.is_some() {
                            continue;
                        }

                        let mut baked = match cmd_buf_data {
                            Ok(cmd_buf_data) => {
                                let res = validate_command_buffer(
                                    command_buffer,
                                    self,
                                    &cmd_buf_data,
                                    &snatch_guard,
                                    &mut submit_surface_textures_owned,
                                    &mut used_surface_textures,
                                );
                                if let Err(err) = res {
                                    first_error.get_or_insert(err);
                                    continue;
                                }
                                cmd_buf_data.into_baked_commands()
                            }
                            Err(err) => {
                                first_error.get_or_insert(err.into());
                                continue;
                            }
                        };

                        // execute resource transitions
                        if let Err(e) = baked.encoder.open_pass(Some("(wgpu internal) Transit")) {
                            break 'error Err(e.into());
                        }

                        //Note: locking the trackers has to be done after the storages
                        let mut trackers = self.device.trackers.lock();
                        if let Err(e) = baked.initialize_buffer_memory(&mut trackers, &snatch_guard)
                        {
                            break 'error Err(e.into());
                        }
                        if let Err(e) = baked.initialize_texture_memory(
                            &mut trackers,
                            &self.device,
                            &snatch_guard,
                        ) {
                            break 'error Err(e.into());
                        }

                        //Note: stateless trackers are not merged:
                        // device already knows these resources exist.
                        CommandBuffer::insert_barriers_from_device_tracker(
                            baked.encoder.raw.as_mut(),
                            &mut trackers,
                            &baked.trackers,
                            &snatch_guard,
                        );

                        if let Err(e) = baked.encoder.close_and_push_front() {
                            break 'error Err(e.into());
                        }

                        // Transition surface textures into `Present` state.
                        // Note: we could technically do it after all of the command buffers,
                        // but here we have a command encoder by hand, so it's easier to use it.
                        if !used_surface_textures.is_empty() {
                            if let Err(e) = baked.encoder.open_pass(Some("(wgpu internal) Present"))
                            {
                                break 'error Err(e.into());
                            }
                            let texture_barriers = trackers
                                .textures
                                .set_from_usage_scope_and_drain_transitions(
                                    &used_surface_textures,
                                    &snatch_guard,
                                )
                                .collect::<Vec<_>>();
                            unsafe {
                                baked.encoder.raw.transition_textures(&texture_barriers);
                            };
                            if let Err(e) = baked.encoder.close() {
                                break 'error Err(e.into());
                            }
                            used_surface_textures = track::TextureUsageScope::default();
                        }

                        // done
                        active_executions.push(EncoderInFlight {
                            inner: baked.encoder,
                            trackers: baked.trackers,
                            pending_buffers: FastHashMap::default(),
                            pending_textures: FastHashMap::default(),
                            pending_blas_s: FastHashMap::default(),
                            pending_tlas_s: FastHashMap::default(),
                        });
                    }

                    if let Some(first_error) = first_error {
                        break 'error Err(first_error);
                    }
                }
            }

            let mut pending_writes = self.pending_writes.lock();

            {
                used_surface_textures.set_size(self.device.tracker_indices.textures.size());
                for texture in pending_writes.dst_textures.values() {
                    match texture.try_inner(&snatch_guard) {
                        Ok(TextureInner::Native { .. }) => {}
                        Ok(TextureInner::Surface { .. }) => {
                            // Compare the Arcs by pointer as Textures don't implement Eq
                            submit_surface_textures_owned
                                .insert(Arc::as_ptr(texture), texture.clone());

                            unsafe {
                                used_surface_textures
                                    .merge_single(texture, None, hal::TextureUses::PRESENT)
                                    .unwrap()
                            };
                        }
                        Err(e) => break 'error Err(e.into()),
                    }
                }

                if !used_surface_textures.is_empty() {
                    let mut trackers = self.device.trackers.lock();

                    let texture_barriers = trackers
                        .textures
                        .set_from_usage_scope_and_drain_transitions(
                            &used_surface_textures,
                            &snatch_guard,
                        )
                        .collect::<Vec<_>>();
                    unsafe {
                        pending_writes
                            .command_encoder
                            .transition_textures(&texture_barriers);
                    };
                }
            }

            match pending_writes.pre_submit(&self.device.command_allocator, &self.device, self) {
                Ok(Some(pending_execution)) => {
                    active_executions.insert(0, pending_execution);
                }
                Ok(None) => {}
                Err(e) => break 'error Err(e.into()),
            }
            let hal_command_buffers = active_executions
                .iter()
                .flat_map(|e| e.inner.list.iter().map(|b| b.as_ref()))
                .collect::<Vec<_>>();

            {
                let mut submit_surface_textures =
                    SmallVec::<[&dyn hal::DynSurfaceTexture; 2]>::with_capacity(
                        submit_surface_textures_owned.len(),
                    );

                for texture in submit_surface_textures_owned.values() {
                    let raw = match texture.inner.get(&snatch_guard) {
                        Some(TextureInner::Surface { raw, .. }) => raw.as_ref(),
                        _ => unreachable!(),
                    };
                    submit_surface_textures.push(raw);
                }

                if let Err(e) = unsafe {
                    self.raw().submit(
                        &hal_command_buffers,
                        &submit_surface_textures,
                        (fence.as_mut(), submit_index),
                    )
                }
                .map_err(|e| self.device.handle_hal_error(e))
                {
                    break 'error Err(e.into());
                }

                // Advance the successful submission index.
                self.device
                    .last_successful_submission_index
                    .fetch_max(submit_index, Ordering::SeqCst);
            }

            profiling::scope!("cleanup");

            // this will register the new submission to the life time tracker
            self.lock_life().track_submission(
                submit_index,
                pending_writes.temp_resources.drain(..),
                active_executions,
            );
            drop(pending_writes);

            // This will schedule destruction of all resources that are no longer needed
            // by the user but used in the command stream, among other things.
            let fence_guard = RwLockWriteGuard::downgrade(fence);
            let (closures, _) =
                match self
                    .device
                    .maintain(fence_guard, wgt::Maintain::Poll, snatch_guard)
                {
                    Ok(closures) => closures,
                    Err(WaitIdleError::Device(err)) => {
                        break 'error Err(QueueSubmitError::Queue(err))
                    }
                    Err(WaitIdleError::WrongSubmissionIndex(..)) => unreachable!(),
                };

            Ok(closures)
        };

        let callbacks = match res {
            Ok(ok) => ok,
            Err(e) => return Err((submit_index, e)),
        };

        // the closures should execute with nothing locked!
        callbacks.fire();

        api_log!("Queue::submit returned submit index {submit_index}");

        Ok(submit_index)
    }

    pub fn get_timestamp_period(&self) -> f32 {
        unsafe { self.raw().get_timestamp_period() }
    }

    /// `closure` is guaranteed to be called.
    pub fn on_submitted_work_done(
        &self,
        closure: SubmittedWorkDoneClosure,
    ) -> Option<SubmissionIndex> {
        api_log!("Queue::on_submitted_work_done");
        //TODO: flush pending writes
        self.lock_life().add_work_done_closure(closure)
    }
}

impl Global {
    pub fn queue_write_buffer(
        &self,
        queue_id: QueueId,
        buffer_id: id::BufferId,
        buffer_offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), QueueWriteError> {
        let queue = self.hub.queues.get(queue_id);

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *queue.device.trace.lock() {
            let data_path = trace.make_binary("bin", data);
            trace.add(Action::WriteBuffer {
                id: buffer_id,
                data: data_path,
                range: buffer_offset..buffer_offset + data.len() as u64,
                queued: true,
            });
        }

        let buffer = self.hub.buffers.get(buffer_id);
        queue.write_buffer(buffer, buffer_offset, data)
    }

    pub fn queue_create_staging_buffer(
        &self,
        queue_id: QueueId,
        buffer_size: wgt::BufferSize,
        id_in: Option<id::StagingBufferId>,
    ) -> Result<(id::StagingBufferId, NonNull<u8>), QueueWriteError> {
        let queue = self.hub.queues.get(queue_id);
        let (staging_buffer, ptr) = queue.create_staging_buffer(buffer_size)?;

        let fid = self.hub.staging_buffers.prepare(id_in);
        let id = fid.assign(staging_buffer);

        Ok((id, ptr))
    }

    pub fn queue_write_staging_buffer(
        &self,
        queue_id: QueueId,
        buffer_id: id::BufferId,
        buffer_offset: wgt::BufferAddress,
        staging_buffer_id: id::StagingBufferId,
    ) -> Result<(), QueueWriteError> {
        let queue = self.hub.queues.get(queue_id);
        let buffer = self.hub.buffers.get(buffer_id);
        let staging_buffer = self.hub.staging_buffers.remove(staging_buffer_id);
        queue.write_staging_buffer(buffer, buffer_offset, staging_buffer)
    }

    pub fn queue_validate_write_buffer(
        &self,
        queue_id: QueueId,
        buffer_id: id::BufferId,
        buffer_offset: u64,
        buffer_size: wgt::BufferSize,
    ) -> Result<(), QueueWriteError> {
        let queue = self.hub.queues.get(queue_id);
        let buffer = self.hub.buffers.get(buffer_id);
        queue.validate_write_buffer(buffer, buffer_offset, buffer_size)
    }

    pub fn queue_write_texture(
        &self,
        queue_id: QueueId,
        destination: &TexelCopyTextureInfo,
        data: &[u8],
        data_layout: &wgt::TexelCopyBufferLayout,
        size: &wgt::Extent3d,
    ) -> Result<(), QueueWriteError> {
        let queue = self.hub.queues.get(queue_id);

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *queue.device.trace.lock() {
            let data_path = trace.make_binary("bin", data);
            trace.add(Action::WriteTexture {
                to: *destination,
                data: data_path,
                layout: *data_layout,
                size: *size,
            });
        }

        let destination = wgt::TexelCopyTextureInfo {
            texture: self.hub.textures.get(destination.texture),
            mip_level: destination.mip_level,
            origin: destination.origin,
            aspect: destination.aspect,
        };
        queue.write_texture(destination, data, data_layout, size)
    }

    #[cfg(webgl)]
    pub fn queue_copy_external_image_to_texture(
        &self,
        queue_id: QueueId,
        source: &wgt::CopyExternalImageSourceInfo,
        destination: crate::command::CopyExternalImageDestInfo,
        size: wgt::Extent3d,
    ) -> Result<(), QueueWriteError> {
        let queue = self.hub.queues.get(queue_id);
        let destination = wgt::CopyExternalImageDestInfo {
            texture: self.hub.textures.get(destination.texture),
            mip_level: destination.mip_level,
            origin: destination.origin,
            aspect: destination.aspect,
            color_space: destination.color_space,
            premultiplied_alpha: destination.premultiplied_alpha,
        };
        queue.copy_external_image_to_texture(source, destination, size)
    }

    pub fn queue_submit(
        &self,
        queue_id: QueueId,
        command_buffer_ids: &[id::CommandBufferId],
    ) -> Result<SubmissionIndex, (SubmissionIndex, QueueSubmitError)> {
        let queue = self.hub.queues.get(queue_id);
        let command_buffer_guard = self.hub.command_buffers.read();
        let command_buffers = command_buffer_ids
            .iter()
            .map(|id| command_buffer_guard.get(*id))
            .collect::<Vec<_>>();
        drop(command_buffer_guard);
        queue.submit(&command_buffers)
    }

    pub fn queue_get_timestamp_period(&self, queue_id: QueueId) -> f32 {
        let queue = self.hub.queues.get(queue_id);
        queue.get_timestamp_period()
    }

    pub fn queue_on_submitted_work_done(
        &self,
        queue_id: QueueId,
        closure: SubmittedWorkDoneClosure,
    ) -> SubmissionIndex {
        api_log!("Queue::on_submitted_work_done {queue_id:?}");

        //TODO: flush pending writes
        let queue = self.hub.queues.get(queue_id);
        let result = queue.on_submitted_work_done(closure);
        result.unwrap_or(0) // '0' means no wait is necessary
    }
}

fn validate_command_buffer(
    command_buffer: &CommandBuffer,
    queue: &Queue,
    cmd_buf_data: &crate::command::CommandBufferMutable,
    snatch_guard: &SnatchGuard,
    submit_surface_textures_owned: &mut FastHashMap<*const Texture, Arc<Texture>>,
    used_surface_textures: &mut track::TextureUsageScope,
) -> Result<(), QueueSubmitError> {
    command_buffer.same_device_as(queue)?;

    {
        profiling::scope!("check resource state");

        {
            profiling::scope!("buffers");
            for buffer in cmd_buf_data.trackers.buffers.used_resources() {
                buffer.check_destroyed(snatch_guard)?;

                match *buffer.map_state.lock() {
                    BufferMapState::Idle => (),
                    _ => return Err(QueueSubmitError::BufferStillMapped(buffer.error_ident())),
                }
            }
        }
        {
            profiling::scope!("textures");
            for texture in cmd_buf_data.trackers.textures.used_resources() {
                let should_extend = match texture.try_inner(snatch_guard)? {
                    TextureInner::Native { .. } => false,
                    TextureInner::Surface { .. } => {
                        // Compare the Arcs by pointer as Textures don't implement Eq.
                        submit_surface_textures_owned.insert(Arc::as_ptr(texture), texture.clone());

                        true
                    }
                };
                if should_extend {
                    unsafe {
                        used_surface_textures
                            .merge_single(texture, None, hal::TextureUses::PRESENT)
                            .unwrap();
                    };
                }
            }
        }

        if let Err(e) = cmd_buf_data.validate_blas_actions() {
            return Err(e.into());
        }
        if let Err(e) = cmd_buf_data.validate_tlas_actions(snatch_guard) {
            return Err(e.into());
        }
    }
    Ok(())
}
