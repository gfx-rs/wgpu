#[cfg(feature = "trace")]
use crate::device::trace::Action;
use crate::{
    api_log,
    command::{
        extract_texture_selector, validate_linear_texture_data, validate_texture_copy_range,
        ClearError, CommandAllocator, CommandBuffer, CopySide, ImageCopyTexture, TransferError,
    },
    conv,
    device::{DeviceError, WaitIdleError},
    get_lowest_common_denom,
    global::Global,
    hal_label,
    id::{self, QueueId},
    init_tracker::{has_copy_partial_init_tracker_coverage, TextureInitRange},
    lock::RwLockWriteGuard,
    resource::{
        Buffer, BufferAccessError, BufferMapState, DestroyedBuffer, DestroyedResourceError,
        DestroyedTexture, FlushedStagingBuffer, Labeled, ParentDevice, ResourceErrorIdent,
        StagingBuffer, Texture, TextureInner, Trackable,
    },
    resource_log,
    track::{self, Tracker, TrackerIndex},
    FastHashMap, SubmissionIndex,
};

use smallvec::SmallVec;

use std::{
    iter,
    mem::{self, ManuallyDrop},
    ptr::NonNull,
    sync::{atomic::Ordering, Arc},
};
use thiserror::Error;

use super::Device;

pub struct Queue {
    raw: ManuallyDrop<Box<dyn hal::DynQueue>>,
    pub(crate) device: Arc<Device>,
}

impl Queue {
    pub(crate) fn new(device: Arc<Device>, raw: Box<dyn hal::DynQueue>) -> Self {
        Queue {
            raw: ManuallyDrop::new(raw),
            device,
        }
    }

    pub(crate) fn raw(&self) -> &dyn hal::DynQueue {
        self.raw.as_ref()
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
        // SAFETY: we never access `self.raw` beyond this point.
        let queue = unsafe { ManuallyDrop::take(&mut self.raw) };
        self.device.release_queue(queue);
    }
}

#[repr(C)]
pub struct SubmittedWorkDoneClosureC {
    pub callback: unsafe extern "C" fn(user_data: *mut u8),
    pub user_data: *mut u8,
}

#[cfg(send_sync)]
unsafe impl Send for SubmittedWorkDoneClosureC {}

pub struct SubmittedWorkDoneClosure {
    // We wrap this so creating the enum in the C variant can be unsafe,
    // allowing our call function to be safe.
    inner: SubmittedWorkDoneClosureInner,
}

#[cfg(send_sync)]
type SubmittedWorkDoneCallback = Box<dyn FnOnce() + Send + 'static>;
#[cfg(not(send_sync))]
type SubmittedWorkDoneCallback = Box<dyn FnOnce() + 'static>;

enum SubmittedWorkDoneClosureInner {
    Rust { callback: SubmittedWorkDoneCallback },
    C { inner: SubmittedWorkDoneClosureC },
}

impl SubmittedWorkDoneClosure {
    pub fn from_rust(callback: SubmittedWorkDoneCallback) -> Self {
        Self {
            inner: SubmittedWorkDoneClosureInner::Rust { callback },
        }
    }

    /// # Safety
    ///
    /// - The callback pointer must be valid to call with the provided `user_data`
    ///   pointer.
    ///
    /// - Both pointers must point to `'static` data, as the callback may happen at
    ///   an unspecified time.
    pub unsafe fn from_c(inner: SubmittedWorkDoneClosureC) -> Self {
        Self {
            inner: SubmittedWorkDoneClosureInner::C { inner },
        }
    }

    pub(crate) fn call(self) {
        match self.inner {
            SubmittedWorkDoneClosureInner::Rust { callback } => callback(),
            // SAFETY: the contract of the call to from_c says that this unsafe is sound.
            SubmittedWorkDoneClosureInner::C { inner } => unsafe {
                (inner.callback)(inner.user_data)
            },
        }
    }
}

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
    DestroyedBuffer(DestroyedBuffer),
    DestroyedTexture(DestroyedTexture),
}

/// A series of raw [`CommandBuffer`]s that have been submitted to a
/// queue, and the [`wgpu_hal::CommandEncoder`] that built them.
///
/// [`CommandBuffer`]: hal::Api::CommandBuffer
/// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
pub(crate) struct EncoderInFlight {
    raw: Box<dyn hal::DynCommandEncoder>,
    cmd_buffers: Vec<Box<dyn hal::DynCommandBuffer>>,
    pub(crate) trackers: Tracker,

    /// These are the buffers that have been tracked by `PendingWrites`.
    pub(crate) pending_buffers: FastHashMap<TrackerIndex, Arc<Buffer>>,
    /// These are the textures that have been tracked by `PendingWrites`.
    pub(crate) pending_textures: FastHashMap<TrackerIndex, Arc<Texture>>,
}

impl EncoderInFlight {
    /// Free all of our command buffers.
    ///
    /// Return the command encoder, fully reset and ready to be
    /// reused.
    pub(crate) unsafe fn land(mut self) -> Box<dyn hal::DynCommandEncoder> {
        unsafe { self.raw.reset_all(self.cmd_buffers) };
        {
            // This involves actually decrementing the ref count of all command buffer
            // resources, so can be _very_ expensive.
            profiling::scope!("drop command buffer trackers");
            drop(self.trackers);
            drop(self.pending_buffers);
            drop(self.pending_textures);
        }
        self.raw
    }
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
}

impl PendingWrites {
    pub fn new(command_encoder: Box<dyn hal::DynCommandEncoder>) -> Self {
        Self {
            command_encoder,
            is_recording: false,
            temp_resources: Vec::new(),
            dst_buffers: FastHashMap::default(),
            dst_textures: FastHashMap::default(),
        }
    }

    pub fn dispose(mut self, device: &dyn hal::DynDevice) {
        unsafe {
            if self.is_recording {
                self.command_encoder.discard_encoding();
            }
            device.destroy_command_encoder(self.command_encoder);
        }

        self.temp_resources.clear();
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
        device: &dyn hal::DynDevice,
        queue: &dyn hal::DynQueue,
    ) -> Result<Option<EncoderInFlight>, DeviceError> {
        if self.is_recording {
            let pending_buffers = mem::take(&mut self.dst_buffers);
            let pending_textures = mem::take(&mut self.dst_textures);

            let cmd_buf = unsafe { self.command_encoder.end_encoding()? };
            self.is_recording = false;

            let new_encoder = command_allocator.acquire_encoder(device, queue)?;

            let encoder = EncoderInFlight {
                raw: mem::replace(&mut self.command_encoder, new_encoder),
                cmd_buffers: vec![cmd_buf],
                trackers: Tracker::new(),
                pending_buffers,
                pending_textures,
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

    pub fn deactivate(&mut self) {
        if self.is_recording {
            unsafe {
                self.command_encoder.discard_encoding();
            }
            self.is_recording = false;
        }
    }
}

#[derive(Clone, Debug, Error)]
#[error("Queue is invalid")]
pub struct InvalidQueue;

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum QueueWriteError {
    #[error("QueueId is invalid")]
    InvalidQueueId,
    #[error(transparent)]
    Queue(#[from] DeviceError),
    #[error(transparent)]
    Transfer(#[from] TransferError),
    #[error(transparent)]
    MemoryInitFailure(#[from] ClearError),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum QueueSubmitError {
    #[error("QueueId is invalid")]
    InvalidQueueId,
    #[error(transparent)]
    Queue(#[from] DeviceError),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error(transparent)]
    Unmap(#[from] BufferAccessError),
    #[error("{0} is still mapped")]
    BufferStillMapped(ResourceErrorIdent),
    #[error("Surface output was dropped before the command buffer got submitted")]
    SurfaceOutputDropped,
    #[error("Surface was unconfigured before the command buffer got submitted")]
    SurfaceUnconfigured,
    #[error("GPU got stuck :(")]
    StuckGpu,
}

//TODO: move out common parts of write_xxx.

impl Global {
    pub fn queue_write_buffer(
        &self,
        queue_id: QueueId,
        buffer_id: id::BufferId,
        buffer_offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::write_buffer");
        api_log!("Queue::write_buffer {buffer_id:?} {}bytes", data.len());

        let hub = &self.hub;

        let buffer = hub
            .buffers
            .get(buffer_id)
            .map_err(|_| TransferError::InvalidBufferId(buffer_id))?;

        let queue = hub
            .queues
            .get(queue_id)
            .map_err(|_| QueueWriteError::InvalidQueueId)?;

        let device = &queue.device;

        let data_size = data.len() as wgt::BufferAddress;

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *device.trace.lock() {
            let data_path = trace.make_binary("bin", data);
            trace.add(Action::WriteBuffer {
                id: buffer_id,
                data: data_path,
                range: buffer_offset..buffer_offset + data_size,
                queued: true,
            });
        }

        buffer.same_device_as(queue.as_ref())?;

        let data_size = if let Some(data_size) = wgt::BufferSize::new(data_size) {
            data_size
        } else {
            log::trace!("Ignoring write_buffer of size 0");
            return Ok(());
        };

        // Platform validation requires that the staging buffer always be
        // freed, even if an error occurs. All paths from here must call
        // `device.pending_writes.consume`.
        let mut staging_buffer = StagingBuffer::new(device, data_size)?;
        let mut pending_writes = device.pending_writes.lock();

        let staging_buffer = {
            profiling::scope!("copy");
            staging_buffer.write(data);
            staging_buffer.flush()
        };

        let result = self.queue_write_staging_buffer_impl(
            &queue,
            device,
            &mut pending_writes,
            &staging_buffer,
            buffer_id,
            buffer_offset,
        );

        pending_writes.consume(staging_buffer);
        result
    }

    pub fn queue_create_staging_buffer(
        &self,
        queue_id: QueueId,
        buffer_size: wgt::BufferSize,
        id_in: Option<id::StagingBufferId>,
    ) -> Result<(id::StagingBufferId, NonNull<u8>), QueueWriteError> {
        profiling::scope!("Queue::create_staging_buffer");
        let hub = &self.hub;

        let queue = hub
            .queues
            .get(queue_id)
            .map_err(|_| QueueWriteError::InvalidQueueId)?;

        let device = &queue.device;

        let staging_buffer = StagingBuffer::new(device, buffer_size)?;
        let ptr = unsafe { staging_buffer.ptr() };

        let fid = hub.staging_buffers.prepare(queue_id.backend(), id_in);
        let id = fid.assign(Arc::new(staging_buffer));
        resource_log!("Queue::create_staging_buffer {id:?}");

        Ok((id, ptr))
    }

    pub fn queue_write_staging_buffer(
        &self,
        queue_id: QueueId,
        buffer_id: id::BufferId,
        buffer_offset: wgt::BufferAddress,
        staging_buffer_id: id::StagingBufferId,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::write_staging_buffer");
        let hub = &self.hub;

        let queue = hub
            .queues
            .get(queue_id)
            .map_err(|_| QueueWriteError::InvalidQueueId)?;

        let device = &queue.device;

        let staging_buffer = hub
            .staging_buffers
            .unregister(staging_buffer_id)
            .and_then(Arc::into_inner)
            .ok_or_else(|| QueueWriteError::Transfer(TransferError::InvalidBufferId(buffer_id)))?;

        let mut pending_writes = device.pending_writes.lock();

        // At this point, we have taken ownership of the staging_buffer from the
        // user. Platform validation requires that the staging buffer always
        // be freed, even if an error occurs. All paths from here must call
        // `device.pending_writes.consume`.
        let staging_buffer = staging_buffer.flush();

        let result = self.queue_write_staging_buffer_impl(
            &queue,
            device,
            &mut pending_writes,
            &staging_buffer,
            buffer_id,
            buffer_offset,
        );

        pending_writes.consume(staging_buffer);
        result
    }

    pub fn queue_validate_write_buffer(
        &self,
        _queue_id: QueueId,
        buffer_id: id::BufferId,
        buffer_offset: u64,
        buffer_size: wgt::BufferSize,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::validate_write_buffer");
        let hub = &self.hub;

        let buffer = hub
            .buffers
            .get(buffer_id)
            .map_err(|_| TransferError::InvalidBufferId(buffer_id))?;

        self.queue_validate_write_buffer_impl(&buffer, buffer_offset, buffer_size)?;

        Ok(())
    }

    fn queue_validate_write_buffer_impl(
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

    fn queue_write_staging_buffer_impl(
        &self,
        queue: &Arc<Queue>,
        device: &Arc<Device>,
        pending_writes: &mut PendingWrites,
        staging_buffer: &FlushedStagingBuffer,
        buffer_id: id::BufferId,
        buffer_offset: u64,
    ) -> Result<(), QueueWriteError> {
        let hub = &self.hub;

        let dst = hub
            .buffers
            .get(buffer_id)
            .map_err(|_| TransferError::InvalidBufferId(buffer_id))?;

        let transition = {
            let mut trackers = device.trackers.lock();
            trackers.buffers.set_single(&dst, hal::BufferUses::COPY_DST)
        };

        let snatch_guard = device.snatchable_lock.read();
        let dst_raw = dst.try_raw(&snatch_guard)?;

        dst.same_device_as(queue.as_ref())?;

        self.queue_validate_write_buffer_impl(&dst, buffer_offset, staging_buffer.size)?;

        let region = hal::BufferCopy {
            src_offset: 0,
            dst_offset: buffer_offset,
            size: staging_buffer.size,
        };
        let barriers = iter::once(hal::BufferBarrier {
            buffer: staging_buffer.raw(),
            usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
        })
        .chain(transition.map(|pending| pending.into_hal(&dst, &snatch_guard)))
        .collect::<Vec<_>>();
        let encoder = pending_writes.activate();
        unsafe {
            encoder.transition_buffers(&barriers);
            encoder.copy_buffer_to_buffer(staging_buffer.raw(), dst_raw, &[region]);
        }

        pending_writes.insert_buffer(&dst);

        // Ensure the overwritten bytes are marked as initialized so
        // they don't need to be nulled prior to mapping or binding.
        {
            dst.initialization_status
                .write()
                .drain(buffer_offset..(buffer_offset + staging_buffer.size.get()));
        }

        Ok(())
    }

    pub fn queue_write_texture(
        &self,
        queue_id: QueueId,
        destination: &ImageCopyTexture,
        data: &[u8],
        data_layout: &wgt::ImageDataLayout,
        size: &wgt::Extent3d,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::write_texture");
        api_log!("Queue::write_texture {:?} {size:?}", destination.texture);

        let hub = &self.hub;

        let queue = hub
            .queues
            .get(queue_id)
            .map_err(|_| QueueWriteError::InvalidQueueId)?;

        let device = &queue.device;

        #[cfg(feature = "trace")]
        if let Some(ref mut trace) = *device.trace.lock() {
            let data_path = trace.make_binary("bin", data);
            trace.add(Action::WriteTexture {
                to: *destination,
                data: data_path,
                layout: *data_layout,
                size: *size,
            });
        }

        if size.width == 0 || size.height == 0 || size.depth_or_array_layers == 0 {
            log::trace!("Ignoring write_texture of size 0");
            return Ok(());
        }

        let dst = hub
            .textures
            .get(destination.texture)
            .map_err(|_| TransferError::InvalidTextureId(destination.texture))?;

        dst.same_device_as(queue.as_ref())?;

        dst.check_usage(wgt::TextureUsages::COPY_DST)
            .map_err(TransferError::MissingTextureUsage)?;

        // Note: Doing the copy range validation early is important because ensures that the
        // dimensions are not going to cause overflow in other parts of the validation.
        let (hal_copy_size, array_layer_count) =
            validate_texture_copy_range(destination, &dst.desc, CopySide::Destination, size)?;

        let (selector, dst_base) = extract_texture_selector(destination, size, &dst)?;

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
            device
                .require_downlevel_flags(wgt::DownlevelFlags::DEPTH_TEXTURE_AND_BUFFER_COPIES)
                .map_err(TransferError::from)?;
        }

        let mut pending_writes = device.pending_writes.lock();
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
                    let mut trackers = device.trackers.lock();
                    crate::command::clear_texture(
                        &dst,
                        TextureInitRange {
                            mip_range: destination.mip_level..(destination.mip_level + 1),
                            layer_range,
                        },
                        encoder,
                        &mut trackers.textures,
                        &device.alignments,
                        device.zero_buffer.as_ref(),
                        &device.snatchable_lock.read(),
                    )
                    .map_err(QueueWriteError::from)?;
                }
            } else {
                dst_initialization_status.mips[destination.mip_level as usize]
                    .drain(init_layer_range);
            }
        }

        let snatch_guard = device.snatchable_lock.read();

        // Re-get `dst` immutably here, so that the mutable borrow of the
        // `texture_guard.get` above ends in time for the `clear_texture`
        // call above. Since we've held `texture_guard` the whole time, we know
        // the texture hasn't gone away in the mean time, so we can unwrap.
        let dst = hub.textures.get(destination.texture).unwrap();

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

        let bytes_per_row_alignment =
            get_lowest_common_denom(device.alignments.buffer_copy_pitch.get() as u32, block_size);
        let stage_bytes_per_row = wgt::math::align_to(bytes_in_last_row, bytes_per_row_alignment);

        // Platform validation requires that the staging buffer always be
        // freed, even if an error occurs. All paths from here must call
        // `device.pending_writes.consume`.
        let staging_buffer = if stage_bytes_per_row == bytes_per_row {
            profiling::scope!("copy aligned");
            // Fast path if the data is already being aligned optimally.
            let stage_size = wgt::BufferSize::new(required_bytes_in_copy).unwrap();
            let mut staging_buffer = StagingBuffer::new(device, stage_size)?;
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
            let mut staging_buffer = StagingBuffer::new(device, stage_size)?;
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
                    buffer_layout: wgt::ImageDataLayout {
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
                usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
            };

            let mut trackers = device.trackers.lock();
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
    pub fn queue_copy_external_image_to_texture(
        &self,
        queue_id: QueueId,
        source: &wgt::ImageCopyExternalImage,
        destination: crate::command::ImageCopyTextureTagged,
        size: wgt::Extent3d,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::copy_external_image_to_texture");

        let hub = &self.hub;

        let queue = hub
            .queues
            .get(queue_id)
            .map_err(|_| QueueWriteError::InvalidQueueId)?;

        let device = &queue.device;

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
            device
                .require_downlevel_flags(wgt::DownlevelFlags::UNRESTRICTED_EXTERNAL_TEXTURE_COPIES)
                .map_err(TransferError::from)?;
        }

        let src_width = source.source.width();
        let src_height = source.source.height();

        let dst = hub.textures.get(destination.texture).unwrap();

        if !conv::is_valid_external_image_copy_dst_texture_format(dst.desc.format) {
            return Err(
                TransferError::ExternalCopyToForbiddenTextureFormat(dst.desc.format).into(),
            );
        }
        if dst.desc.dimension != wgt::TextureDimension::D2 {
            return Err(TransferError::InvalidDimensionExternal(destination.texture).into());
        }
        dst.check_usage(wgt::TextureUsages::COPY_DST)
            .map_err(TransferError::MissingTextureUsage)?;
        if !dst
            .desc
            .usage
            .contains(wgt::TextureUsages::RENDER_ATTACHMENT)
        {
            return Err(
                TransferError::MissingRenderAttachmentUsageFlag(destination.texture).into(),
            );
        }
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
        let (hal_copy_size, _) = validate_texture_copy_range(
            &destination.to_untagged(),
            &dst.desc,
            CopySide::Destination,
            &size,
        )?;

        let (selector, dst_base) =
            extract_texture_selector(&destination.to_untagged(), &size, &dst)?;

        let mut pending_writes = device.pending_writes.lock();
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
                    let mut trackers = device.trackers.lock();
                    crate::command::clear_texture(
                        &dst,
                        TextureInitRange {
                            mip_range: destination.mip_level..(destination.mip_level + 1),
                            layer_range,
                        },
                        encoder,
                        &mut trackers.textures,
                        &device.alignments,
                        device.zero_buffer.as_ref(),
                        &device.snatchable_lock.read(),
                    )
                    .map_err(QueueWriteError::from)?;
                }
            } else {
                dst_initialization_status.mips[destination.mip_level as usize]
                    .drain(init_layer_range);
            }
        }

        let snatch_guard = device.snatchable_lock.read();
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

        let mut trackers = device.trackers.lock();
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
                destination.premultiplied_alpha,
                iter::once(regions),
            );
        }

        Ok(())
    }

    pub fn queue_submit(
        &self,
        queue_id: QueueId,
        command_buffer_ids: &[id::CommandBufferId],
    ) -> Result<SubmissionIndex, QueueSubmitError> {
        profiling::scope!("Queue::submit");
        api_log!("Queue::submit {queue_id:?}");

        let (submit_index, callbacks) = {
            let hub = &self.hub;

            let queue = hub
                .queues
                .get(queue_id)
                .map_err(|_| QueueSubmitError::InvalidQueueId)?;

            let device = &queue.device;

            let snatch_guard = device.snatchable_lock.read();

            // Fence lock must be acquired after the snatch lock everywhere to avoid deadlocks.
            let mut fence = device.fence.write();
            let submit_index = device
                .active_submission_index
                .fetch_add(1, Ordering::SeqCst)
                + 1;
            let mut active_executions = Vec::new();

            let mut used_surface_textures = track::TextureUsageScope::default();

            // Use a hashmap here to deduplicate the surface textures that are used in the command buffers.
            // This avoids vulkan deadlocking from the same surface texture being submitted multiple times.
            let mut submit_surface_textures_owned = FastHashMap::default();

            {
                let mut command_buffer_guard = hub.command_buffers.write();

                if !command_buffer_ids.is_empty() {
                    profiling::scope!("prepare");

                    //TODO: if multiple command buffers are submitted, we can re-use the last
                    // native command buffer of the previous chain instead of always creating
                    // a temporary one, since the chains are not finished.

                    // finish all the command buffers first
                    for &cmb_id in command_buffer_ids {
                        profiling::scope!("process command buffer");

                        // we reset the used surface textures every time we use
                        // it, so make sure to set_size on it.
                        used_surface_textures.set_size(device.tracker_indices.textures.size());

                        #[allow(unused_mut)]
                        let mut cmdbuf = match command_buffer_guard.replace_with_error(cmb_id) {
                            Ok(cmdbuf) => cmdbuf,
                            Err(_) => continue,
                        };

                        #[cfg(feature = "trace")]
                        if let Some(ref mut trace) = *device.trace.lock() {
                            trace.add(Action::Submit(
                                submit_index,
                                cmdbuf
                                    .data
                                    .lock()
                                    .as_mut()
                                    .unwrap()
                                    .commands
                                    .take()
                                    .unwrap(),
                            ));
                        }

                        cmdbuf.same_device_as(queue.as_ref())?;

                        if !cmdbuf.is_finished() {
                            let cmdbuf = Arc::into_inner(cmdbuf).expect(
                                "Command buffer cannot be destroyed because is still in use",
                            );
                            device.destroy_command_buffer(cmdbuf);
                            continue;
                        }

                        {
                            profiling::scope!("check resource state");

                            let cmd_buf_data = cmdbuf.data.lock();
                            let cmd_buf_trackers = &cmd_buf_data.as_ref().unwrap().trackers;

                            // update submission IDs
                            {
                                profiling::scope!("buffers");
                                for buffer in cmd_buf_trackers.buffers.used_resources() {
                                    buffer.check_destroyed(&snatch_guard)?;

                                    match *buffer.map_state.lock() {
                                        BufferMapState::Idle => (),
                                        _ => {
                                            return Err(QueueSubmitError::BufferStillMapped(
                                                buffer.error_ident(),
                                            ))
                                        }
                                    }
                                }
                            }
                            {
                                profiling::scope!("textures");
                                for texture in cmd_buf_trackers.textures.used_resources() {
                                    let should_extend = match texture.try_inner(&snatch_guard)? {
                                        TextureInner::Native { .. } => false,
                                        TextureInner::Surface { .. } => {
                                            // Compare the Arcs by pointer as Textures don't implement Eq.
                                            submit_surface_textures_owned
                                                .insert(Arc::as_ptr(&texture), texture.clone());

                                            true
                                        }
                                    };
                                    if should_extend {
                                        unsafe {
                                            used_surface_textures
                                                .merge_single(
                                                    &texture,
                                                    None,
                                                    hal::TextureUses::PRESENT,
                                                )
                                                .unwrap();
                                        };
                                    }
                                }
                            }
                        }
                        let mut baked = cmdbuf.from_arc_into_baked();

                        // execute resource transitions
                        unsafe {
                            baked
                                .encoder
                                .begin_encoding(hal_label(
                                    Some("(wgpu internal) Transit"),
                                    device.instance_flags,
                                ))
                                .map_err(DeviceError::from)?
                        };

                        //Note: locking the trackers has to be done after the storages
                        let mut trackers = device.trackers.lock();
                        baked.initialize_buffer_memory(&mut trackers, &snatch_guard)?;
                        baked.initialize_texture_memory(&mut trackers, device, &snatch_guard)?;
                        //Note: stateless trackers are not merged:
                        // device already knows these resources exist.
                        CommandBuffer::insert_barriers_from_device_tracker(
                            baked.encoder.as_mut(),
                            &mut trackers,
                            &baked.trackers,
                            &snatch_guard,
                        );

                        let transit = unsafe { baked.encoder.end_encoding().unwrap() };
                        baked.list.insert(0, transit);

                        // Transition surface textures into `Present` state.
                        // Note: we could technically do it after all of the command buffers,
                        // but here we have a command encoder by hand, so it's easier to use it.
                        if !used_surface_textures.is_empty() {
                            unsafe {
                                baked
                                    .encoder
                                    .begin_encoding(hal_label(
                                        Some("(wgpu internal) Present"),
                                        device.instance_flags,
                                    ))
                                    .map_err(DeviceError::from)?
                            };
                            let texture_barriers = trackers
                                .textures
                                .set_from_usage_scope_and_drain_transitions(
                                    &used_surface_textures,
                                    &snatch_guard,
                                )
                                .collect::<Vec<_>>();
                            let present = unsafe {
                                baked.encoder.transition_textures(&texture_barriers);
                                baked.encoder.end_encoding().unwrap()
                            };
                            baked.list.push(present);
                            used_surface_textures = track::TextureUsageScope::default();
                        }

                        // done
                        active_executions.push(EncoderInFlight {
                            raw: baked.encoder,
                            cmd_buffers: baked.list,
                            trackers: baked.trackers,
                            pending_buffers: FastHashMap::default(),
                            pending_textures: FastHashMap::default(),
                        });
                    }
                }
            }

            let mut pending_writes = device.pending_writes.lock();

            {
                used_surface_textures.set_size(hub.textures.read().len());
                for texture in pending_writes.dst_textures.values() {
                    match texture.try_inner(&snatch_guard)? {
                        TextureInner::Native { .. } => {}
                        TextureInner::Surface { .. } => {
                            // Compare the Arcs by pointer as Textures don't implement Eq
                            submit_surface_textures_owned
                                .insert(Arc::as_ptr(texture), texture.clone());

                            unsafe {
                                used_surface_textures
                                    .merge_single(texture, None, hal::TextureUses::PRESENT)
                                    .unwrap()
                            };
                        }
                    }
                }

                if !used_surface_textures.is_empty() {
                    let mut trackers = device.trackers.lock();

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

            if let Some(pending_execution) =
                pending_writes.pre_submit(&device.command_allocator, device.raw(), queue.raw())?
            {
                active_executions.insert(0, pending_execution);
            }

            let hal_command_buffers = active_executions
                .iter()
                .flat_map(|e| e.cmd_buffers.iter().map(|b| b.as_ref()))
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

                unsafe {
                    queue
                        .raw()
                        .submit(
                            &hal_command_buffers,
                            &submit_surface_textures,
                            (fence.as_mut(), submit_index),
                        )
                        .map_err(DeviceError::from)?;
                }

                // Advance the successful submission index.
                device
                    .last_successful_submission_index
                    .fetch_max(submit_index, Ordering::SeqCst);
            }

            profiling::scope!("cleanup");

            // this will register the new submission to the life time tracker
            device.lock_life().track_submission(
                submit_index,
                pending_writes.temp_resources.drain(..),
                active_executions,
            );
            drop(pending_writes);

            // This will schedule destruction of all resources that are no longer needed
            // by the user but used in the command stream, among other things.
            let fence_guard = RwLockWriteGuard::downgrade(fence);
            let (closures, _) =
                match device.maintain(fence_guard, wgt::Maintain::Poll, snatch_guard) {
                    Ok(closures) => closures,
                    Err(WaitIdleError::Device(err)) => return Err(QueueSubmitError::Queue(err)),
                    Err(WaitIdleError::StuckGpu) => return Err(QueueSubmitError::StuckGpu),
                    Err(WaitIdleError::WrongSubmissionIndex(..)) => unreachable!(),
                };

            (submit_index, closures)
        };

        // the closures should execute with nothing locked!
        callbacks.fire();

        api_log!("Queue::submit to {queue_id:?} returned submit index {submit_index}");

        Ok(submit_index)
    }

    pub fn queue_get_timestamp_period(&self, queue_id: QueueId) -> Result<f32, InvalidQueue> {
        let hub = &self.hub;
        match hub.queues.get(queue_id) {
            Ok(queue) => Ok(unsafe { queue.raw().get_timestamp_period() }),
            Err(_) => Err(InvalidQueue),
        }
    }

    pub fn queue_on_submitted_work_done(
        &self,
        queue_id: QueueId,
        closure: SubmittedWorkDoneClosure,
    ) -> Result<(), InvalidQueue> {
        api_log!("Queue::on_submitted_work_done {queue_id:?}");

        //TODO: flush pending writes
        let hub = &self.hub;
        match hub.queues.get(queue_id) {
            Ok(queue) => queue.device.lock_life().add_work_done_closure(closure),
            Err(_) => return Err(InvalidQueue),
        }
        Ok(())
    }
}
