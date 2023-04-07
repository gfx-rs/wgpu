#[cfg(feature = "trace")]
use crate::device::trace::Action;
use crate::{
    command::{
        extract_texture_selector, validate_linear_texture_data, validate_texture_copy_range,
        ClearError, CommandBuffer, CopySide, ImageCopyTexture, TransferError,
    },
    conv,
    device::{DeviceError, WaitIdleError},
    get_lowest_common_denom,
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Input, Token},
    id,
    init_tracker::{has_copy_partial_init_tracker_coverage, TextureInitRange},
    resource::{BufferAccessError, BufferMapState, StagingBuffer, TextureInner},
    track, FastHashSet, SubmissionIndex,
};

use hal::{CommandEncoder as _, Device as _, Queue as _};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{iter, mem, ptr};
use thiserror::Error;

/// Number of command buffers that we generate from the same pool
/// for the write_xxx commands, before the pool is recycled.
///
/// If we don't stop at some point, the pool will grow forever,
/// without a concrete moment of when it can be cleared.
const WRITE_COMMAND_BUFFERS_PER_POOL: usize = 64;

#[repr(C)]
pub struct SubmittedWorkDoneClosureC {
    pub callback: unsafe extern "C" fn(user_data: *mut u8),
    pub user_data: *mut u8,
}

unsafe impl Send for SubmittedWorkDoneClosureC {}

pub struct SubmittedWorkDoneClosure {
    // We wrap this so creating the enum in the C variant can be unsafe,
    // allowing our call function to be safe.
    inner: SubmittedWorkDoneClosureInner,
}

enum SubmittedWorkDoneClosureInner {
    Rust {
        callback: Box<dyn FnOnce() + Send + 'static>,
    },
    C {
        inner: SubmittedWorkDoneClosureC,
    },
}

impl SubmittedWorkDoneClosure {
    pub fn from_rust(callback: Box<dyn FnOnce() + Send + 'static>) -> Self {
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

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct WrappedSubmissionIndex {
    pub queue_id: id::QueueId,
    pub index: SubmissionIndex,
}

/// A texture or buffer to be freed soon.
///
/// This is just a tagged raw texture or buffer, generally about to be added to
/// some other more specific container like:
///
/// - `PendingWrites::temp_resources`: resources used by queue writes and
///   unmaps, waiting to be folded in with the next queue submission
///
/// - `ActiveSubmission::last_resources`: temporary resources used by a queue
///   submission, to be freed when it completes
///
/// - `LifetimeTracker::free_resources`: resources to be freed in the next
///   `maintain` call, no longer used anywhere
#[derive(Debug)]
pub enum TempResource<A: hal::Api> {
    Buffer(A::Buffer),
    Texture(A::Texture, SmallVec<[A::TextureView; 1]>),
}

/// A queue execution for a particular command encoder.
pub(super) struct EncoderInFlight<A: hal::Api> {
    raw: A::CommandEncoder,
    cmd_buffers: Vec<A::CommandBuffer>,
}

impl<A: hal::Api> EncoderInFlight<A> {
    pub(super) unsafe fn land(mut self) -> A::CommandEncoder {
        unsafe { self.raw.reset_all(self.cmd_buffers.into_iter()) };
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
/// All uses of [`StagingBuffer`]s end up here.
#[derive(Debug)]
pub(crate) struct PendingWrites<A: hal::Api> {
    pub command_encoder: A::CommandEncoder,
    pub is_active: bool,
    pub temp_resources: Vec<TempResource<A>>,
    pub dst_buffers: FastHashSet<id::BufferId>,
    pub dst_textures: FastHashSet<id::TextureId>,
    pub executing_command_buffers: Vec<A::CommandBuffer>,
}

impl<A: hal::Api> PendingWrites<A> {
    pub fn new(command_encoder: A::CommandEncoder) -> Self {
        Self {
            command_encoder,
            is_active: false,
            temp_resources: Vec::new(),
            dst_buffers: FastHashSet::default(),
            dst_textures: FastHashSet::default(),
            executing_command_buffers: Vec::new(),
        }
    }

    pub fn dispose(mut self, device: &A::Device) {
        unsafe {
            if self.is_active {
                self.command_encoder.discard_encoding();
            }
            self.command_encoder
                .reset_all(self.executing_command_buffers.into_iter());
            device.destroy_command_encoder(self.command_encoder);
        }

        for resource in self.temp_resources {
            match resource {
                TempResource::Buffer(buffer) => unsafe {
                    device.destroy_buffer(buffer);
                },
                TempResource::Texture(texture, views) => unsafe {
                    for view in views.into_iter() {
                        device.destroy_texture_view(view);
                    }
                    device.destroy_texture(texture);
                },
            }
        }
    }

    pub fn consume_temp(&mut self, resource: TempResource<A>) {
        self.temp_resources.push(resource);
    }

    fn consume(&mut self, buffer: StagingBuffer<A>) {
        self.temp_resources.push(TempResource::Buffer(buffer.raw));
    }

    #[must_use]
    fn pre_submit(&mut self) -> Option<&A::CommandBuffer> {
        self.dst_buffers.clear();
        self.dst_textures.clear();
        if self.is_active {
            let cmd_buf = unsafe { self.command_encoder.end_encoding().unwrap() };
            self.is_active = false;
            self.executing_command_buffers.push(cmd_buf);
            self.executing_command_buffers.last()
        } else {
            None
        }
    }

    #[must_use]
    fn post_submit(
        &mut self,
        command_allocator: &Mutex<super::CommandAllocator<A>>,
        device: &A::Device,
        queue: &A::Queue,
    ) -> Option<EncoderInFlight<A>> {
        if self.executing_command_buffers.len() >= WRITE_COMMAND_BUFFERS_PER_POOL {
            let new_encoder = command_allocator
                .lock()
                .acquire_encoder(device, queue)
                .unwrap();
            Some(EncoderInFlight {
                raw: mem::replace(&mut self.command_encoder, new_encoder),
                cmd_buffers: mem::take(&mut self.executing_command_buffers),
            })
        } else {
            None
        }
    }

    pub fn activate(&mut self) -> &mut A::CommandEncoder {
        if !self.is_active {
            unsafe {
                self.command_encoder
                    .begin_encoding(Some("(wgpu internal) PendingWrites"))
                    .unwrap();
            }
            self.is_active = true;
        }
        &mut self.command_encoder
    }

    pub fn deactivate(&mut self) {
        if self.is_active {
            unsafe {
                self.command_encoder.discard_encoding();
            }
            self.is_active = false;
        }
    }
}

fn prepare_staging_buffer<A: HalApi>(
    device: &mut A::Device,
    size: wgt::BufferAddress,
) -> Result<(StagingBuffer<A>, *mut u8), DeviceError> {
    profiling::scope!("prepare_staging_buffer");
    let stage_desc = hal::BufferDescriptor {
        label: Some("(wgpu internal) Staging"),
        size,
        usage: hal::BufferUses::MAP_WRITE | hal::BufferUses::COPY_SRC,
        memory_flags: hal::MemoryFlags::TRANSIENT,
    };

    let buffer = unsafe { device.create_buffer(&stage_desc)? };
    let mapping = unsafe { device.map_buffer(&buffer, 0..size) }?;

    let staging_buffer = StagingBuffer {
        raw: buffer,
        size,
        is_coherent: mapping.is_coherent,
    };

    Ok((staging_buffer, mapping.ptr.as_ptr()))
}

impl<A: hal::Api> StagingBuffer<A> {
    unsafe fn flush(&self, device: &A::Device) -> Result<(), DeviceError> {
        if !self.is_coherent {
            unsafe { device.flush_mapped_ranges(&self.raw, iter::once(0..self.size)) };
        }
        unsafe { device.unmap_buffer(&self.raw)? };
        Ok(())
    }
}

#[derive(Clone, Debug, Error)]
#[error("Queue is invalid")]
pub struct InvalidQueue;

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum QueueWriteError {
    #[error(transparent)]
    Queue(#[from] DeviceError),
    #[error(transparent)]
    Transfer(#[from] TransferError),
    #[error(transparent)]
    MemoryInitFailure(#[from] ClearError),
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum QueueSubmitError {
    #[error(transparent)]
    Queue(#[from] DeviceError),
    #[error("Buffer {0:?} is destroyed")]
    DestroyedBuffer(id::BufferId),
    #[error("Texture {0:?} is destroyed")]
    DestroyedTexture(id::TextureId),
    #[error(transparent)]
    Unmap(#[from] BufferAccessError),
    #[error("Buffer {0:?} is still mapped")]
    BufferStillMapped(id::BufferId),
    #[error("Surface output was dropped before the command buffer got submitted")]
    SurfaceOutputDropped,
    #[error("Surface was unconfigured before the command buffer got submitted")]
    SurfaceUnconfigured,
    #[error("GPU got stuck :(")]
    StuckGpu,
}

//TODO: move out common parts of write_xxx.

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn queue_write_buffer<A: HalApi>(
        &self,
        queue_id: id::QueueId,
        buffer_id: id::BufferId,
        buffer_offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::write_buffer");

        let hub = A::hub(self);
        let root_token = &mut Token::root();

        let (mut device_guard, ref mut device_token) = hub.devices.write(root_token);
        let device = device_guard
            .get_mut(queue_id)
            .map_err(|_| DeviceError::Invalid)?;

        let data_size = data.len() as wgt::BufferAddress;

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            let mut trace = trace.lock();
            let data_path = trace.make_binary("bin", data);
            trace.add(Action::WriteBuffer {
                id: buffer_id,
                data: data_path,
                range: buffer_offset..buffer_offset + data_size,
                queued: true,
            });
        }

        if data_size == 0 {
            log::trace!("Ignoring write_buffer of size 0");
            return Ok(());
        }

        // Platform validation requires that the staging buffer always be
        // freed, even if an error occurs. All paths from here must call
        // `device.pending_writes.consume`.
        let (staging_buffer, staging_buffer_ptr) =
            prepare_staging_buffer(&mut device.raw, data_size)?;

        if let Err(flush_error) = unsafe {
            profiling::scope!("copy");
            ptr::copy_nonoverlapping(data.as_ptr(), staging_buffer_ptr, data.len());
            staging_buffer.flush(&device.raw)
        } {
            device.pending_writes.consume(staging_buffer);
            return Err(flush_error.into());
        }

        let result = self.queue_write_staging_buffer_impl(
            device,
            device_token,
            &staging_buffer,
            buffer_id,
            buffer_offset,
        );

        device.pending_writes.consume(staging_buffer);
        result
    }

    pub fn queue_create_staging_buffer<A: HalApi>(
        &self,
        queue_id: id::QueueId,
        buffer_size: wgt::BufferSize,
        id_in: Input<G, id::StagingBufferId>,
    ) -> Result<(id::StagingBufferId, *mut u8), QueueWriteError> {
        profiling::scope!("Queue::create_staging_buffer");
        let hub = A::hub(self);
        let root_token = &mut Token::root();

        let (mut device_guard, ref mut device_token) = hub.devices.write(root_token);
        let device = device_guard
            .get_mut(queue_id)
            .map_err(|_| DeviceError::Invalid)?;

        let (staging_buffer, staging_buffer_ptr) =
            prepare_staging_buffer(&mut device.raw, buffer_size.get())?;

        let fid = hub.staging_buffers.prepare(id_in);
        let id = fid.assign(staging_buffer, device_token);

        Ok((id.0, staging_buffer_ptr))
    }

    pub fn queue_write_staging_buffer<A: HalApi>(
        &self,
        queue_id: id::QueueId,
        buffer_id: id::BufferId,
        buffer_offset: wgt::BufferAddress,
        staging_buffer_id: id::StagingBufferId,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::write_staging_buffer");
        let hub = A::hub(self);
        let root_token = &mut Token::root();

        let (mut device_guard, ref mut device_token) = hub.devices.write(root_token);
        let device = device_guard
            .get_mut(queue_id)
            .map_err(|_| DeviceError::Invalid)?;

        let staging_buffer = hub
            .staging_buffers
            .unregister(staging_buffer_id, device_token)
            .0
            .ok_or(TransferError::InvalidBuffer(buffer_id))?;

        // At this point, we have taken ownership of the staging_buffer from the
        // user. Platform validation requires that the staging buffer always
        // be freed, even if an error occurs. All paths from here must call
        // `device.pending_writes.consume`.
        if let Err(flush_error) = unsafe { staging_buffer.flush(&device.raw) } {
            device.pending_writes.consume(staging_buffer);
            return Err(flush_error.into());
        }

        let result = self.queue_write_staging_buffer_impl(
            device,
            device_token,
            &staging_buffer,
            buffer_id,
            buffer_offset,
        );

        device.pending_writes.consume(staging_buffer);
        result
    }

    pub fn queue_validate_write_buffer<A: HalApi>(
        &self,
        _queue_id: id::QueueId,
        buffer_id: id::BufferId,
        buffer_offset: u64,
        buffer_size: u64,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::validate_write_buffer");
        let hub = A::hub(self);
        let root_token = &mut Token::root();

        let (_, ref mut device_token) = hub.devices.read(root_token);

        let buffer_guard = hub.buffers.read(device_token).0;
        let buffer = buffer_guard
            .get(buffer_id)
            .map_err(|_| TransferError::InvalidBuffer(buffer_id))?;

        self.queue_validate_write_buffer_impl(buffer, buffer_id, buffer_offset, buffer_size)?;

        Ok(())
    }

    fn queue_validate_write_buffer_impl<A: HalApi>(
        &self,
        buffer: &super::resource::Buffer<A>,
        buffer_id: id::BufferId,
        buffer_offset: u64,
        buffer_size: u64,
    ) -> Result<(), TransferError> {
        if !buffer.usage.contains(wgt::BufferUsages::COPY_DST) {
            return Err(TransferError::MissingCopyDstUsageFlag(
                Some(buffer_id),
                None,
            ));
        }
        if buffer_size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(TransferError::UnalignedCopySize(buffer_size));
        }
        if buffer_offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(TransferError::UnalignedBufferOffset(buffer_offset));
        }
        if buffer_offset + buffer_size > buffer.size {
            return Err(TransferError::BufferOverrun {
                start_offset: buffer_offset,
                end_offset: buffer_offset + buffer_size,
                buffer_size: buffer.size,
                side: CopySide::Destination,
            });
        }

        Ok(())
    }

    fn queue_write_staging_buffer_impl<A: HalApi>(
        &self,
        device: &mut super::Device<A>,
        device_token: &mut Token<super::Device<A>>,
        staging_buffer: &StagingBuffer<A>,
        buffer_id: id::BufferId,
        buffer_offset: u64,
    ) -> Result<(), QueueWriteError> {
        let hub = A::hub(self);

        let buffer_guard = hub.buffers.read(device_token).0;

        let mut trackers = device.trackers.lock();
        let (dst, transition) = trackers
            .buffers
            .set_single(&buffer_guard, buffer_id, hal::BufferUses::COPY_DST)
            .ok_or(TransferError::InvalidBuffer(buffer_id))?;
        let dst_raw = dst
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidBuffer(buffer_id))?;

        let src_buffer_size = staging_buffer.size;
        self.queue_validate_write_buffer_impl(dst, buffer_id, buffer_offset, src_buffer_size)?;

        dst.life_guard.use_at(device.active_submission_index + 1);

        let region = wgt::BufferSize::new(src_buffer_size).map(|size| hal::BufferCopy {
            src_offset: 0,
            dst_offset: buffer_offset,
            size,
        });
        let barriers = iter::once(hal::BufferBarrier {
            buffer: &staging_buffer.raw,
            usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
        })
        .chain(transition.map(|pending| pending.into_hal(dst)));
        let encoder = device.pending_writes.activate();
        unsafe {
            encoder.transition_buffers(barriers);
            encoder.copy_buffer_to_buffer(&staging_buffer.raw, dst_raw, region.into_iter());
        }

        device.pending_writes.dst_buffers.insert(buffer_id);

        // Ensure the overwritten bytes are marked as initialized so
        // they don't need to be nulled prior to mapping or binding.
        {
            drop(buffer_guard);
            let mut buffer_guard = hub.buffers.write(device_token).0;

            let dst = buffer_guard.get_mut(buffer_id).unwrap();
            dst.initialization_status
                .drain(buffer_offset..(buffer_offset + src_buffer_size));
        }

        Ok(())
    }

    pub fn queue_write_texture<A: HalApi>(
        &self,
        queue_id: id::QueueId,
        destination: &ImageCopyTexture,
        data: &[u8],
        data_layout: &wgt::ImageDataLayout,
        size: &wgt::Extent3d,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::write_texture");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let device = device_guard
            .get_mut(queue_id)
            .map_err(|_| DeviceError::Invalid)?;

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            let mut trace = trace.lock();
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

        let (mut texture_guard, _) = hub.textures.write(&mut token); // For clear we need write access to the texture. TODO: Can we acquire write lock later?
        let dst = texture_guard
            .get_mut(destination.texture)
            .map_err(|_| TransferError::InvalidTexture(destination.texture))?;

        if !dst.desc.usage.contains(wgt::TextureUsages::COPY_DST) {
            return Err(
                TransferError::MissingCopyDstUsageFlag(None, Some(destination.texture)).into(),
            );
        }

        // Note: Doing the copy range validation early is important because ensures that the
        // dimensions are not going to cause overflow in other parts of the validation.
        let (hal_copy_size, array_layer_count) =
            validate_texture_copy_range(destination, &dst.desc, CopySide::Destination, size)?;

        let (selector, dst_base) = extract_texture_selector(destination, size, dst)?;

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
        let (_, _source_bytes_per_array_layer) = validate_linear_texture_data(
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

        let (block_width, block_height) = dst.desc.format.block_dimensions();
        let width_blocks = size.width / block_width;
        let height_blocks = size.height / block_height;

        let block_rows_per_image = data_layout.rows_per_image.unwrap_or(
            // doesn't really matter because we need this only if we copy
            // more than one layer, and then we validate for this being not
            // None
            size.height,
        );

        let block_size = dst
            .desc
            .format
            .block_size(Some(destination.aspect))
            .unwrap();
        let bytes_per_row_alignment =
            get_lowest_common_denom(device.alignments.buffer_copy_pitch.get() as u32, block_size);
        let stage_bytes_per_row =
            wgt::math::align_to(block_size * width_blocks, bytes_per_row_alignment);

        let block_rows_in_copy =
            (size.depth_or_array_layers - 1) * block_rows_per_image + height_blocks;
        let stage_size = stage_bytes_per_row as u64 * block_rows_in_copy as u64;

        let mut trackers = device.trackers.lock();
        let encoder = device.pending_writes.activate();

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
        if dst.initialization_status.mips[destination.mip_level as usize]
            .check(init_layer_range.clone())
            .is_some()
        {
            if has_copy_partial_init_tracker_coverage(size, destination.mip_level, &dst.desc) {
                for layer_range in dst.initialization_status.mips[destination.mip_level as usize]
                    .drain(init_layer_range)
                    .collect::<Vec<std::ops::Range<u32>>>()
                {
                    crate::command::clear_texture(
                        &*texture_guard,
                        id::Valid(destination.texture),
                        TextureInitRange {
                            mip_range: destination.mip_level..(destination.mip_level + 1),
                            layer_range,
                        },
                        encoder,
                        &mut trackers.textures,
                        &device.alignments,
                        &device.zero_buffer,
                    )
                    .map_err(QueueWriteError::from)?;
                }
            } else {
                dst.initialization_status.mips[destination.mip_level as usize]
                    .drain(init_layer_range);
            }
        }

        // Re-get `dst` immutably here, so that the mutable borrow of the
        // `texture_guard.get_mut` above ends in time for the `clear_texture`
        // call above. Since we've held `texture_guard` the whole time, we know
        // the texture hasn't gone away in the mean time, so we can unwrap.
        let dst = texture_guard.get(destination.texture).unwrap();
        let transition = trackers
            .textures
            .set_single(
                dst,
                destination.texture,
                selector,
                hal::TextureUses::COPY_DST,
            )
            .ok_or(TransferError::InvalidTexture(destination.texture))?;

        dst.life_guard.use_at(device.active_submission_index + 1);

        let dst_raw = dst
            .inner
            .as_raw()
            .ok_or(TransferError::InvalidTexture(destination.texture))?;

        let bytes_per_row = data_layout
            .bytes_per_row
            .unwrap_or(width_blocks * block_size);

        // Platform validation requires that the staging buffer always be
        // freed, even if an error occurs. All paths from here must call
        // `device.pending_writes.consume`.
        let (staging_buffer, staging_buffer_ptr) =
            prepare_staging_buffer(&mut device.raw, stage_size)?;

        if stage_bytes_per_row == bytes_per_row {
            profiling::scope!("copy aligned");
            // Fast path if the data is already being aligned optimally.
            unsafe {
                ptr::copy_nonoverlapping(
                    data.as_ptr().offset(data_layout.offset as isize),
                    staging_buffer_ptr,
                    stage_size as usize,
                );
            }
        } else {
            profiling::scope!("copy chunked");
            // Copy row by row into the optimal alignment.
            let copy_bytes_per_row = stage_bytes_per_row.min(bytes_per_row) as usize;
            for layer in 0..size.depth_or_array_layers {
                let rows_offset = layer * block_rows_per_image;
                for row in 0..height_blocks {
                    unsafe {
                        ptr::copy_nonoverlapping(
                            data.as_ptr().offset(
                                data_layout.offset as isize
                                    + (rows_offset + row) as isize * bytes_per_row as isize,
                            ),
                            staging_buffer_ptr.offset(
                                (rows_offset + row) as isize * stage_bytes_per_row as isize,
                            ),
                            copy_bytes_per_row,
                        );
                    }
                }
            }
        }

        if let Err(e) = unsafe { staging_buffer.flush(&device.raw) } {
            device.pending_writes.consume(staging_buffer);
            return Err(e.into());
        }

        let regions = (0..array_layer_count).map(|rel_array_layer| {
            let mut texture_base = dst_base.clone();
            texture_base.array_layer += rel_array_layer;
            hal::BufferTextureCopy {
                buffer_layout: wgt::ImageDataLayout {
                    offset: rel_array_layer as u64
                        * block_rows_per_image as u64
                        * stage_bytes_per_row as u64,
                    bytes_per_row: Some(stage_bytes_per_row),
                    rows_per_image: Some(block_rows_per_image),
                },
                texture_base,
                size: hal_copy_size,
            }
        });
        let barrier = hal::BufferBarrier {
            buffer: &staging_buffer.raw,
            usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
        };

        unsafe {
            encoder.transition_textures(transition.map(|pending| pending.into_hal(dst)));
            encoder.transition_buffers(iter::once(barrier));
            encoder.copy_buffer_to_texture(&staging_buffer.raw, dst_raw, regions);
        }

        device.pending_writes.consume(staging_buffer);
        device
            .pending_writes
            .dst_textures
            .insert(destination.texture);

        Ok(())
    }

    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    pub fn queue_copy_external_image_to_texture<A: HalApi>(
        &self,
        queue_id: id::QueueId,
        source: &wgt::ImageCopyExternalImage,
        destination: crate::command::ImageCopyTextureTagged,
        size: wgt::Extent3d,
    ) -> Result<(), QueueWriteError> {
        profiling::scope!("Queue::copy_external_image_to_texture");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let device = device_guard
            .get_mut(queue_id)
            .map_err(|_| DeviceError::Invalid)?;

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

        let (mut texture_guard, _) = hub.textures.write(&mut token); // For clear we need write access to the texture. TODO: Can we acquire write lock later?
        let dst = texture_guard.get_mut(destination.texture).unwrap();

        if !conv::is_valid_external_image_copy_dst_texture_format(dst.desc.format) {
            return Err(
                TransferError::ExternalCopyToForbiddenTextureFormat(dst.desc.format).into(),
            );
        }
        if dst.desc.dimension != wgt::TextureDimension::D2 {
            return Err(TransferError::InvalidDimensionExternal(destination.texture).into());
        }
        if !dst.desc.usage.contains(wgt::TextureUsages::COPY_DST) {
            return Err(
                TransferError::MissingCopyDstUsageFlag(None, Some(destination.texture)).into(),
            );
        }
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
            extract_texture_selector(&destination.to_untagged(), &size, dst)?;

        let mut trackers = device.trackers.lock();
        let encoder = device.pending_writes.activate();

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
        if dst.initialization_status.mips[destination.mip_level as usize]
            .check(init_layer_range.clone())
            .is_some()
        {
            if has_copy_partial_init_tracker_coverage(&size, destination.mip_level, &dst.desc) {
                for layer_range in dst.initialization_status.mips[destination.mip_level as usize]
                    .drain(init_layer_range)
                    .collect::<Vec<std::ops::Range<u32>>>()
                {
                    crate::command::clear_texture(
                        &*texture_guard,
                        id::Valid(destination.texture),
                        TextureInitRange {
                            mip_range: destination.mip_level..(destination.mip_level + 1),
                            layer_range,
                        },
                        encoder,
                        &mut trackers.textures,
                        &device.alignments,
                        &device.zero_buffer,
                    )
                    .map_err(QueueWriteError::from)?;
                }
            } else {
                dst.initialization_status.mips[destination.mip_level as usize]
                    .drain(init_layer_range);
            }
        }

        let dst = texture_guard.get(destination.texture).unwrap();

        let transitions = trackers
            .textures
            .set_single(
                dst,
                destination.texture,
                selector,
                hal::TextureUses::COPY_DST,
            )
            .ok_or(TransferError::InvalidTexture(destination.texture))?;

        dst.life_guard.use_at(device.active_submission_index + 1);

        let dst_raw = dst
            .inner
            .as_raw()
            .ok_or(TransferError::InvalidTexture(destination.texture))?;

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

        unsafe {
            encoder.transition_textures(transitions.map(|pending| pending.into_hal(dst)));
            encoder.copy_external_image_to_texture(
                source,
                dst_raw,
                destination.premultiplied_alpha,
                iter::once(regions),
            );
        }

        Ok(())
    }

    pub fn queue_submit<A: HalApi>(
        &self,
        queue_id: id::QueueId,
        command_buffer_ids: &[id::CommandBufferId],
    ) -> Result<WrappedSubmissionIndex, QueueSubmitError> {
        profiling::scope!("Queue::submit");

        let (submit_index, callbacks) = {
            let hub = A::hub(self);
            let mut token = Token::root();

            let (mut device_guard, mut token) = hub.devices.write(&mut token);
            let device = device_guard
                .get_mut(queue_id)
                .map_err(|_| DeviceError::Invalid)?;
            device.temp_suspected.clear();
            device.active_submission_index += 1;
            let submit_index = device.active_submission_index;
            let mut active_executions = Vec::new();
            let mut used_surface_textures = track::TextureUsageScope::new();

            {
                let (mut command_buffer_guard, mut token) = hub.command_buffers.write(&mut token);

                if !command_buffer_ids.is_empty() {
                    profiling::scope!("prepare");

                    let (render_bundle_guard, mut token) = hub.render_bundles.read(&mut token);
                    let (_, mut token) = hub.pipeline_layouts.read(&mut token);
                    let (bind_group_guard, mut token) = hub.bind_groups.read(&mut token);
                    let (compute_pipe_guard, mut token) = hub.compute_pipelines.read(&mut token);
                    let (render_pipe_guard, mut token) = hub.render_pipelines.read(&mut token);
                    let (mut buffer_guard, mut token) = hub.buffers.write(&mut token);
                    let (mut texture_guard, mut token) = hub.textures.write(&mut token);
                    let (texture_view_guard, mut token) = hub.texture_views.read(&mut token);
                    let (sampler_guard, mut token) = hub.samplers.read(&mut token);
                    let (query_set_guard, _) = hub.query_sets.read(&mut token);

                    //Note: locking the trackers has to be done after the storages
                    let mut trackers = device.trackers.lock();

                    //TODO: if multiple command buffers are submitted, we can re-use the last
                    // native command buffer of the previous chain instead of always creating
                    // a temporary one, since the chains are not finished.

                    // finish all the command buffers first
                    for &cmb_id in command_buffer_ids {
                        // we reset the used surface textures every time we use
                        // it, so make sure to set_size on it.
                        used_surface_textures.set_size(texture_guard.len());

                        #[allow(unused_mut)]
                        let mut cmdbuf = match hub
                            .command_buffers
                            .unregister_locked(cmb_id, &mut *command_buffer_guard)
                        {
                            Some(cmdbuf) => cmdbuf,
                            None => continue,
                        };
                        #[cfg(feature = "trace")]
                        if let Some(ref trace) = device.trace {
                            trace.lock().add(Action::Submit(
                                submit_index,
                                cmdbuf.commands.take().unwrap(),
                            ));
                        }
                        if !cmdbuf.is_finished() {
                            device.destroy_command_buffer(cmdbuf);
                            continue;
                        }

                        // optimize the tracked states
                        // cmdbuf.trackers.optimize();

                        // update submission IDs
                        for id in cmdbuf.trackers.buffers.used() {
                            let buffer = &mut buffer_guard[id];
                            let raw_buf = match buffer.raw {
                                Some(ref raw) => raw,
                                None => {
                                    return Err(QueueSubmitError::DestroyedBuffer(id.0));
                                }
                            };
                            if !buffer.life_guard.use_at(submit_index) {
                                if let BufferMapState::Active { .. } = buffer.map_state {
                                    log::warn!("Dropped buffer has a pending mapping.");
                                    unsafe { device.raw.unmap_buffer(raw_buf) }
                                        .map_err(DeviceError::from)?;
                                }
                                device.temp_suspected.buffers.push(id);
                            } else {
                                match buffer.map_state {
                                    BufferMapState::Idle => (),
                                    _ => return Err(QueueSubmitError::BufferStillMapped(id.0)),
                                }
                            }
                        }
                        for id in cmdbuf.trackers.textures.used() {
                            let texture = &mut texture_guard[id];
                            let should_extend = match texture.inner {
                                TextureInner::Native { raw: None } => {
                                    return Err(QueueSubmitError::DestroyedTexture(id.0));
                                }
                                TextureInner::Native { raw: Some(_) } => false,
                                TextureInner::Surface {
                                    ref mut has_work, ..
                                } => {
                                    *has_work = true;
                                    true
                                }
                            };
                            if !texture.life_guard.use_at(submit_index) {
                                device.temp_suspected.textures.push(id);
                            }
                            if should_extend {
                                unsafe {
                                    let ref_count = cmdbuf.trackers.textures.get_ref_count(id);
                                    used_surface_textures
                                        .merge_single(
                                            &*texture_guard,
                                            id,
                                            None,
                                            ref_count,
                                            hal::TextureUses::PRESENT,
                                        )
                                        .unwrap();
                                };
                            }
                        }
                        for id in cmdbuf.trackers.views.used() {
                            if !texture_view_guard[id].life_guard.use_at(submit_index) {
                                device.temp_suspected.texture_views.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.bind_groups.used() {
                            let bg = &bind_group_guard[id];
                            if !bg.life_guard.use_at(submit_index) {
                                device.temp_suspected.bind_groups.push(id);
                            }
                            // We need to update the submission indices for the contained
                            // state-less (!) resources as well, so that they don't get
                            // deleted too early if the parent bind group goes out of scope.
                            for sub_id in bg.used.views.used() {
                                texture_view_guard[sub_id].life_guard.use_at(submit_index);
                            }
                            for sub_id in bg.used.samplers.used() {
                                sampler_guard[sub_id].life_guard.use_at(submit_index);
                            }
                        }
                        // assert!(cmdbuf.trackers.samplers.is_empty());
                        for id in cmdbuf.trackers.compute_pipelines.used() {
                            if !compute_pipe_guard[id].life_guard.use_at(submit_index) {
                                device.temp_suspected.compute_pipelines.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.render_pipelines.used() {
                            if !render_pipe_guard[id].life_guard.use_at(submit_index) {
                                device.temp_suspected.render_pipelines.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.query_sets.used() {
                            if !query_set_guard[id].life_guard.use_at(submit_index) {
                                device.temp_suspected.query_sets.push(id);
                            }
                        }
                        for id in cmdbuf.trackers.bundles.used() {
                            let bundle = &render_bundle_guard[id];
                            if !bundle.life_guard.use_at(submit_index) {
                                device.temp_suspected.render_bundles.push(id);
                            }
                            // We need to update the submission indices for the contained
                            // state-less (!) resources as well, excluding the bind groups.
                            // They don't get deleted too early if the bundle goes out of scope.
                            for sub_id in bundle.used.render_pipelines.used() {
                                render_pipe_guard[sub_id].life_guard.use_at(submit_index);
                            }
                            for sub_id in bundle.used.query_sets.used() {
                                query_set_guard[sub_id].life_guard.use_at(submit_index);
                            }
                        }

                        let mut baked = cmdbuf.into_baked();
                        // execute resource transitions
                        unsafe {
                            baked
                                .encoder
                                .begin_encoding(Some("(wgpu internal) Transit"))
                                .map_err(DeviceError::from)?
                        };
                        log::trace!("Stitching command buffer {:?} before submission", cmb_id);
                        baked
                            .initialize_buffer_memory(&mut *trackers, &mut *buffer_guard)
                            .map_err(|err| QueueSubmitError::DestroyedBuffer(err.0))?;
                        baked
                            .initialize_texture_memory(&mut *trackers, &mut *texture_guard, device)
                            .map_err(|err| QueueSubmitError::DestroyedTexture(err.0))?;
                        //Note: stateless trackers are not merged:
                        // device already knows these resources exist.
                        CommandBuffer::insert_barriers_from_tracker(
                            &mut baked.encoder,
                            &mut *trackers,
                            &baked.trackers,
                            &*buffer_guard,
                            &*texture_guard,
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
                                    .begin_encoding(Some("(wgpu internal) Present"))
                                    .map_err(DeviceError::from)?
                            };
                            trackers
                                .textures
                                .set_from_usage_scope(&*texture_guard, &used_surface_textures);
                            let texture_barriers = trackers.textures.drain().map(|pending| {
                                let tex = unsafe { texture_guard.get_unchecked(pending.id) };
                                pending.into_hal(tex)
                            });
                            let present = unsafe {
                                baked.encoder.transition_textures(texture_barriers);
                                baked.encoder.end_encoding().unwrap()
                            };
                            baked.list.push(present);
                            used_surface_textures = track::TextureUsageScope::new();
                        }

                        // done
                        active_executions.push(EncoderInFlight {
                            raw: baked.encoder,
                            cmd_buffers: baked.list,
                        });
                    }

                    log::trace!("Device after submission {}", submit_index);
                }

                let super::Device {
                    ref mut pending_writes,
                    ref mut queue,
                    ref mut fence,
                    ..
                } = *device;

                {
                    // TODO: These blocks have a few organizational issues, and
                    // should be refactored.
                    //
                    // 1) It's similar to the code we have per-command-buffer
                    //    (at the begin and end) Maybe we can merge some?
                    //
                    // 2) It's doing the extra locking unconditionally. Maybe we
                    //    can only do so if any surfaces are being written to?
                    let (_, mut token) = hub.buffers.read(&mut token); // skip token
                    let (mut texture_guard, _) = hub.textures.write(&mut token);

                    used_surface_textures.set_size(texture_guard.len());

                    for &id in pending_writes.dst_textures.iter() {
                        let texture = texture_guard.get_mut(id).unwrap();
                        match texture.inner {
                            TextureInner::Native { raw: None } => {
                                return Err(QueueSubmitError::DestroyedTexture(id));
                            }
                            TextureInner::Native { raw: Some(_) } => {}
                            TextureInner::Surface {
                                ref mut has_work, ..
                            } => {
                                *has_work = true;
                                let ref_count = texture.life_guard.add_ref();
                                unsafe {
                                    used_surface_textures
                                        .merge_single(
                                            &*texture_guard,
                                            id::Valid(id),
                                            None,
                                            &ref_count,
                                            hal::TextureUses::PRESENT,
                                        )
                                        .unwrap()
                                };
                            }
                        }
                    }

                    if !used_surface_textures.is_empty() {
                        let mut trackers = device.trackers.lock();

                        trackers
                            .textures
                            .set_from_usage_scope(&*texture_guard, &used_surface_textures);
                        let texture_barriers = trackers.textures.drain().map(|pending| {
                            let tex = unsafe { texture_guard.get_unchecked(pending.id) };
                            pending.into_hal(tex)
                        });

                        unsafe {
                            pending_writes
                                .command_encoder
                                .transition_textures(texture_barriers);
                        };
                    }
                }

                let refs = pending_writes
                    .pre_submit()
                    .into_iter()
                    .chain(
                        active_executions
                            .iter()
                            .flat_map(|pool_execution| pool_execution.cmd_buffers.iter()),
                    )
                    .collect::<Vec<_>>();
                unsafe {
                    queue
                        .submit(&refs, Some((fence, submit_index)))
                        .map_err(DeviceError::from)?;
                }
            }

            profiling::scope!("cleanup");
            if let Some(pending_execution) = device.pending_writes.post_submit(
                &device.command_allocator,
                &device.raw,
                &device.queue,
            ) {
                active_executions.push(pending_execution);
            }

            // this will register the new submission to the life time tracker
            let mut pending_write_resources = mem::take(&mut device.pending_writes.temp_resources);
            device.lock_life(&mut token).track_submission(
                submit_index,
                pending_write_resources.drain(..),
                active_executions,
            );

            // This will schedule destruction of all resources that are no longer needed
            // by the user but used in the command stream, among other things.
            let (closures, _) = match device.maintain(hub, wgt::Maintain::Poll, &mut token) {
                Ok(closures) => closures,
                Err(WaitIdleError::Device(err)) => return Err(QueueSubmitError::Queue(err)),
                Err(WaitIdleError::StuckGpu) => return Err(QueueSubmitError::StuckGpu),
                Err(WaitIdleError::WrongSubmissionIndex(..)) => unreachable!(),
            };

            // pending_write_resources has been drained, so it's empty, but we
            // want to retain its heap allocation.
            device.pending_writes.temp_resources = pending_write_resources;
            device.temp_suspected.clear();
            device.lock_life(&mut token).post_submit();

            (submit_index, closures)
        };

        // the closures should execute with nothing locked!
        callbacks.fire();

        Ok(WrappedSubmissionIndex {
            queue_id,
            index: submit_index,
        })
    }

    pub fn queue_get_timestamp_period<A: HalApi>(
        &self,
        queue_id: id::QueueId,
    ) -> Result<f32, InvalidQueue> {
        let hub = A::hub(self);
        let mut token = Token::root();
        let (device_guard, _) = hub.devices.read(&mut token);
        match device_guard.get(queue_id) {
            Ok(device) => Ok(unsafe { device.queue.get_timestamp_period() }),
            Err(_) => Err(InvalidQueue),
        }
    }

    pub fn queue_on_submitted_work_done<A: HalApi>(
        &self,
        queue_id: id::QueueId,
        closure: SubmittedWorkDoneClosure,
    ) -> Result<(), InvalidQueue> {
        //TODO: flush pending writes
        let closure_opt = {
            let hub = A::hub(self);
            let mut token = Token::root();
            let (device_guard, mut token) = hub.devices.read(&mut token);
            match device_guard.get(queue_id) {
                Ok(device) => device.lock_life(&mut token).add_work_done_closure(closure),
                Err(_) => return Err(InvalidQueue),
            }
        };
        if let Some(closure) = closure_opt {
            closure.call();
        }
        Ok(())
    }
}
