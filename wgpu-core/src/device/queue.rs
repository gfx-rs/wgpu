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
    hub::{Global, GlobalIdentityHandlerFactory, HalApi, Token},
    id,
    init_tracker::{has_copy_partial_init_tracker_coverage, TextureInitRange},
    resource::{BufferAccessError, BufferMapState, TextureInner},
    track, FastHashSet, SubmissionIndex,
};

use hal::{CommandEncoder as _, Device as _, Queue as _};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{iter, mem, num::NonZeroU32, ptr};
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
    /// - The callback pointer must be valid to call with the provided user_data pointer.
    /// - Both pointers must point to 'static data as the callback may happen at an unspecified time.
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

struct StagingData<A: hal::Api> {
    buffer: A::Buffer,
}

impl<A: hal::Api> StagingData<A> {
    unsafe fn write(
        &self,
        device: &A::Device,
        offset: wgt::BufferAddress,
        data: &[u8],
    ) -> Result<(), hal::DeviceError> {
        let mapping = device.map_buffer(&self.buffer, offset..offset + data.len() as u64)?;
        ptr::copy_nonoverlapping(data.as_ptr(), mapping.ptr.as_ptr(), data.len());
        if !mapping.is_coherent {
            device
                .flush_mapped_ranges(&self.buffer, iter::once(offset..offset + data.len() as u64));
        }
        device.unmap_buffer(&self.buffer)?;
        Ok(())
    }
}

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
        self.raw.reset_all(self.cmd_buffers.into_iter());
        self.raw
    }
}

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

    fn consume(&mut self, stage: StagingData<A>) {
        self.temp_resources.push(TempResource::Buffer(stage.buffer));
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

impl<A: HalApi> super::Device<A> {
    fn prepare_stage(&mut self, size: wgt::BufferAddress) -> Result<StagingData<A>, DeviceError> {
        profiling::scope!("prepare_stage");
        let stage_desc = hal::BufferDescriptor {
            label: Some("(wgpu internal) Staging"),
            size,
            usage: hal::BufferUses::MAP_WRITE | hal::BufferUses::COPY_SRC,
            memory_flags: hal::MemoryFlags::TRANSIENT,
        };
        let buffer = unsafe { self.raw.create_buffer(&stage_desc)? };
        Ok(StagingData { buffer })
    }
}

#[derive(Clone, Debug, Error)]
#[error("queue is invalid")]
pub struct InvalidQueue;

#[derive(Clone, Debug, Error)]
pub enum QueueWriteError {
    #[error(transparent)]
    Queue(#[from] DeviceError),
    #[error(transparent)]
    Transfer(#[from] TransferError),
    #[error(transparent)]
    MemoryInitFailure(#[from] ClearError),
}

#[derive(Clone, Debug, Error)]
pub enum QueueSubmitError {
    #[error(transparent)]
    Queue(#[from] DeviceError),
    #[error("buffer {0:?} is destroyed")]
    DestroyedBuffer(id::BufferId),
    #[error("texture {0:?} is destroyed")]
    DestroyedTexture(id::TextureId),
    #[error(transparent)]
    Unmap(#[from] BufferAccessError),
    #[error("surface output was dropped before the command buffer got submitted")]
    SurfaceOutputDropped,
    #[error("surface was unconfigured before the command buffer got submitted")]
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
        profiling::scope!("write_buffer", "Queue");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (mut device_guard, mut token) = hub.devices.write(&mut token);
        let device = device_guard
            .get_mut(queue_id)
            .map_err(|_| DeviceError::Invalid)?;
        let (buffer_guard, _) = hub.buffers.read(&mut token);

        #[cfg(feature = "trace")]
        if let Some(ref trace) = device.trace {
            let mut trace = trace.lock();
            let data_path = trace.make_binary("bin", data);
            trace.add(Action::WriteBuffer {
                id: buffer_id,
                data: data_path,
                range: buffer_offset..buffer_offset + data.len() as wgt::BufferAddress,
                queued: true,
            });
        }

        let data_size = data.len() as wgt::BufferAddress;
        if data_size == 0 {
            log::trace!("Ignoring write_buffer of size 0");
            return Ok(());
        }

        let stage = device.prepare_stage(data_size)?;
        unsafe {
            profiling::scope!("copy");
            stage.write(&device.raw, 0, data)
        }
        .map_err(DeviceError::from)?;

        let mut trackers = device.trackers.lock();
        let (dst, transition) = trackers
            .buffers
            .set_single(&*buffer_guard, buffer_id, hal::BufferUses::COPY_DST)
            .ok_or(TransferError::InvalidBuffer(buffer_id))?;
        let dst_raw = dst
            .raw
            .as_ref()
            .ok_or(TransferError::InvalidBuffer(buffer_id))?;
        if !dst.usage.contains(wgt::BufferUsages::COPY_DST) {
            return Err(TransferError::MissingCopyDstUsageFlag(Some(buffer_id), None).into());
        }
        dst.life_guard.use_at(device.active_submission_index + 1);

        if data_size % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(TransferError::UnalignedCopySize(data_size).into());
        }
        if buffer_offset % wgt::COPY_BUFFER_ALIGNMENT != 0 {
            return Err(TransferError::UnalignedBufferOffset(buffer_offset).into());
        }
        if buffer_offset + data_size > dst.size {
            return Err(TransferError::BufferOverrun {
                start_offset: buffer_offset,
                end_offset: buffer_offset + data_size,
                buffer_size: dst.size,
                side: CopySide::Destination,
            }
            .into());
        }

        let region = wgt::BufferSize::new(data.len() as u64).map(|size| hal::BufferCopy {
            src_offset: 0,
            dst_offset: buffer_offset,
            size,
        });
        let barriers = iter::once(hal::BufferBarrier {
            buffer: &stage.buffer,
            usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
        })
        .chain(transition.map(|pending| pending.into_hal(dst)));
        let encoder = device.pending_writes.activate();
        unsafe {
            encoder.transition_buffers(barriers);
            encoder.copy_buffer_to_buffer(&stage.buffer, dst_raw, region.into_iter());
        }

        device.pending_writes.consume(stage);
        device.pending_writes.dst_buffers.insert(buffer_id);

        // Ensure the overwritten bytes are marked as initialized so they don't need to be nulled prior to mapping or binding.
        {
            drop(buffer_guard);
            let (mut buffer_guard, _) = hub.buffers.write(&mut token);

            let dst = buffer_guard.get_mut(buffer_id).unwrap();
            dst.initialization_status
                .drain(buffer_offset..(buffer_offset + data_size));
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
        profiling::scope!("write_texture", "Queue");

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
                to: destination.clone(),
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
        let (selector, dst_base, texture_format) =
            extract_texture_selector(destination, size, &*texture_guard)?;
        let format_desc = texture_format.describe();
        //Note: `_source_bytes_per_array_layer` is ignored since we have a staging copy,
        // and it can have a different value.
        let (_, _source_bytes_per_array_layer) = validate_linear_texture_data(
            data_layout,
            texture_format,
            data.len() as wgt::BufferAddress,
            CopySide::Source,
            format_desc.block_size as wgt::BufferAddress,
            size,
            false,
        )?;

        if !conv::is_valid_copy_dst_texture_format(texture_format) {
            return Err(TransferError::CopyToForbiddenTextureFormat(texture_format).into());
        }
        let (block_width, block_height) = format_desc.block_dimensions;
        let width_blocks = size.width / block_width as u32;
        let height_blocks = size.height / block_height as u32;

        let block_rows_per_image = match data_layout.rows_per_image {
            Some(rows_per_image) => rows_per_image.get(),
            None => {
                // doesn't really matter because we need this only if we copy more than one layer, and then we validate for this being not None
                size.height
            }
        };

        let bytes_per_row_alignment = get_lowest_common_denom(
            device.alignments.buffer_copy_pitch.get() as u32,
            format_desc.block_size as u32,
        );
        let stage_bytes_per_row = hal::auxil::align_to(
            format_desc.block_size as u32 * width_blocks,
            bytes_per_row_alignment,
        );

        let block_rows_in_copy =
            (size.depth_or_array_layers - 1) * block_rows_per_image + height_blocks;
        let stage_size = stage_bytes_per_row as u64 * block_rows_in_copy as u64;
        let stage = device.prepare_stage(stage_size)?;

        let dst = texture_guard.get_mut(destination.texture).unwrap();
        if !dst.desc.usage.contains(wgt::TextureUsages::COPY_DST) {
            return Err(
                TransferError::MissingCopyDstUsageFlag(None, Some(destination.texture)).into(),
            );
        }

        let mut trackers = device.trackers.lock();
        let encoder = device.pending_writes.activate();

        // If the copy does not fully cover the layers, we need to initialize to zero *first* as we don't keep track of partial texture layer inits.
        // Strictly speaking we only need to clear the areas of a layer untouched, but this would get increasingly messy.

        let init_layer_range = if dst.desc.dimension == wgt::TextureDimension::D3 {
            0..1 // volume textures don't have a layer range as array volumes aren't supported
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

        let (dst, transition) = trackers
            .textures
            .set_single(
                &*texture_guard,
                destination.texture,
                selector,
                hal::TextureUses::COPY_DST,
            )
            .ok_or(TransferError::InvalidTexture(destination.texture))?;

        let (hal_copy_size, array_layer_count) =
            validate_texture_copy_range(destination, &dst.desc, CopySide::Destination, size)?;
        dst.life_guard.use_at(device.active_submission_index + 1);

        let bytes_per_row = if let Some(bytes_per_row) = data_layout.bytes_per_row {
            bytes_per_row.get()
        } else {
            width_blocks * format_desc.block_size as u32
        };

        let mapping = unsafe { device.raw.map_buffer(&stage.buffer, 0..stage_size) }
            .map_err(DeviceError::from)?;
        unsafe {
            if stage_bytes_per_row == bytes_per_row {
                profiling::scope!("copy aligned");
                // Fast path if the data is already being aligned optimally.
                ptr::copy_nonoverlapping(
                    data.as_ptr().offset(data_layout.offset as isize),
                    mapping.ptr.as_ptr(),
                    stage_size as usize,
                );
            } else {
                profiling::scope!("copy chunked");
                // Copy row by row into the optimal alignment.
                let copy_bytes_per_row = stage_bytes_per_row.min(bytes_per_row) as usize;
                for layer in 0..size.depth_or_array_layers {
                    let rows_offset = layer * block_rows_per_image;
                    for row in 0..height_blocks {
                        ptr::copy_nonoverlapping(
                            data.as_ptr().offset(
                                data_layout.offset as isize
                                    + (rows_offset + row) as isize * bytes_per_row as isize,
                            ),
                            mapping.ptr.as_ptr().offset(
                                (rows_offset + row) as isize * stage_bytes_per_row as isize,
                            ),
                            copy_bytes_per_row,
                        );
                    }
                }
            }
        }
        unsafe {
            if !mapping.is_coherent {
                device
                    .raw
                    .flush_mapped_ranges(&stage.buffer, iter::once(0..stage_size));
            }
            device
                .raw
                .unmap_buffer(&stage.buffer)
                .map_err(DeviceError::from)?;
        }

        let regions = (0..array_layer_count).map(|rel_array_layer| {
            let mut texture_base = dst_base.clone();
            texture_base.array_layer += rel_array_layer;
            hal::BufferTextureCopy {
                buffer_layout: wgt::ImageDataLayout {
                    offset: rel_array_layer as u64
                        * block_rows_per_image as u64
                        * stage_bytes_per_row as u64,
                    bytes_per_row: NonZeroU32::new(stage_bytes_per_row),
                    rows_per_image: NonZeroU32::new(block_rows_per_image),
                },
                texture_base,
                size: hal_copy_size,
            }
        });
        let barrier = hal::BufferBarrier {
            buffer: &stage.buffer,
            usage: hal::BufferUses::MAP_WRITE..hal::BufferUses::COPY_SRC,
        };

        let dst_raw = dst
            .inner
            .as_raw()
            .ok_or(TransferError::InvalidTexture(destination.texture))?;

        unsafe {
            encoder
                .transition_textures(transition.map(|pending| pending.into_hal(dst)).into_iter());
            encoder.transition_buffers(iter::once(barrier));
            encoder.copy_buffer_to_texture(&stage.buffer, dst_raw, regions);
        }

        device.pending_writes.consume(stage);
        device
            .pending_writes
            .dst_textures
            .insert(destination.texture);

        Ok(())
    }

    pub fn queue_submit<A: HalApi>(
        &self,
        queue_id: id::QueueId,
        command_buffer_ids: &[id::CommandBufferId],
    ) -> Result<WrappedSubmissionIndex, QueueSubmitError> {
        profiling::scope!("submit", "Queue");

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

                    used_surface_textures.set_size(texture_guard.len());

                    //TODO: if multiple command buffers are submitted, we can re-use the last
                    // native command buffer of the previous chain instead of always creating
                    // a temporary one, since the chains are not finished.

                    // finish all the command buffers first
                    for &cmb_id in command_buffer_ids {
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
                                    _ => panic!("Buffer {:?} is still mapped", id),
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
                    //TODO: these blocks have a few organizational issues and should be refactored
                    // (1) it's similar to the code we have per-command-buffer (at the begin and end)
                    // Maybe we an merge some?
                    // (2) it's doing the extra locking unconditionally
                    // Maybe we can only do so if any surfaces are being written to?

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
            let (closures, _) = match device.maintain(hub, wgt::Maintain::Wait, &mut token) {
                Ok(closures) => closures,
                Err(WaitIdleError::Device(err)) => return Err(QueueSubmitError::Queue(err)),
                Err(WaitIdleError::StuckGpu) => return Err(QueueSubmitError::StuckGpu),
                Err(WaitIdleError::WrongSubmissionIndex(..)) => unreachable!(),
            };

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
